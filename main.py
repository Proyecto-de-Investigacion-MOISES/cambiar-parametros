import pandas as pd
import numpy as np
import os

from pandarallel import pandarallel

# Inicializar pandarallel
pandarallel.initialize(progress_bar=True)

# Create the output folders if they don't exist
os.makedirs("output", exist_ok=True)
os.makedirs("output/static", exist_ok=True)
os.makedirs("output/dynamic/soil_moisture/04-2023/30-04-2023", exist_ok=True)


def rellenar_con_valor_cercano(serie, df, ncols, nrows):
    for idx in serie[serie.isna()].index:
        if idx not in df.index:
            continue  # Saltar índices que no existen en el DataFrame original

        fila, columna = divmod(idx, ncols)
        rango_busqueda = 1
        valor_encontrado = False

        while not valor_encontrado and rango_busqueda < max(ncols, nrows):
            for i in range(-rango_busqueda, rango_busqueda + 1):
                for j in range(-rango_busqueda, rango_busqueda + 1):
                    fila_vecino = fila + i
                    columna_vecino = columna + j
                    if 0 <= fila_vecino < nrows and 0 <= columna_vecino < ncols:
                        indice_vecino = fila_vecino * ncols + columna_vecino
                        if (
                            indice_vecino in df.index
                            and not np.isnan(serie[indice_vecino])
                            and idx != indice_vecino
                        ):
                            serie.at[idx] = serie[indice_vecino]
                            valor_encontrado = True
                            break
                if valor_encontrado:
                    break
            rango_busqueda += 1

    return serie


def crear_serie_de_archivo(ruta_archivo, nombre_columna, df_index, ncols, nrows):
    with open(ruta_archivo, "r") as file:
        lines = file.readlines()

    matrix_values = [line.strip().replace(",", ".").split() for line in lines[6:]]
    flattened_values = [
        float(val) if val != "-9999" else np.nan
        for sublist in matrix_values
        for val in sublist
    ]

    value_serie = pd.Series(flattened_values, name=nombre_columna)
    value_serie = value_serie.reindex(df_index)

    # Rellenar NaNs con valores cercanos
    value_serie = rellenar_con_valor_cercano(value_serie, df_parquet, ncols, nrows)

    return value_serie


# Función para realizar el cálculo de suma ponderada
@staticmethod
def calculate(row, df, column_name):
    """Calculate the weighted sum of the values"""
    est_values = [
        df.loc[est, column_name] if est in df.index else 0
        for est in [row["est_1"], row["est_2"], row["est_3"]]
    ]
    weights = [row["peso_1"], row["peso_2"], row["peso_3"]]
    calculation = sum(w * v for w, v in zip(weights, est_values))
    return calculation


# Configuración de la geometría de la matriz
ncols = 13901
nrows = 6715

# Cargar DataFrame
df_parquet = pd.read_parquet("referencia/idw_referencia.parquet")

# Procesamiento de archivos y creación de series
rutas = [
    "input/punto_marchitez.txt",
    "input/capacidad_campo.txt",
    "input/capacidad_total.txt",
    "input/umbral_seco.txt",
    "input/umbral_intermedio.txt",
    "input/umbral_humedo.txt",
]
nombres_columnas = [
    "punto_marchitez",
    "capacidad_campo",
    "cap_total",
    "umbral_seco",
    "umbral_intermedio",
    "umbral_humedo",
]

series = []

for ruta, nombre_columna in zip(rutas, nombres_columnas):
    serie_resultante = crear_serie_de_archivo(
        ruta, nombre_columna, df_parquet.index, ncols, nrows
    )
    series.append(serie_resultante)

# Crear un DataFrame con las series como columnas
serie_propiedades_hidricas = [series[3], series[4], series[5], series[2]]
serie_tasas_secado = [series[0], series[1], series[2]]

df_propiedades_hidricas = pd.concat(serie_propiedades_hidricas, axis=1)

df_propiedades_hidricas.to_parquet("output/static/propiedades_hidricas_downsampled.parquet")

print(df_propiedades_hidricas)

# Cargar el archivo Excel
df_excel = pd.read_excel("input/parametros_moises.xlsx")
df_excel.drop(columns=["Nombre"], inplace=True)
df_excel.set_index("Cod", inplace=True)


# Aplicar la función `calculate` a cada fila
df_parquet["ts_hasta_cc"] = df_parquet.parallel_apply(
    calculate, axis=1, args=(df_excel, "Hasta CC")
)

df_parquet["ts_tras_cc"] = df_parquet.parallel_apply(
    calculate, axis=1, args=(df_excel, "Tras CC")
)
df_parquet["ts_tras_pm"] = df_parquet.parallel_apply(
    calculate, axis=1, args=(df_excel, "Por debajo PtoM")
)

df_parquet.drop(
    columns=["est_1", "peso_1", "est_2", "peso_2", "est_3", "peso_3"], inplace=True
)

df_parquet = pd.concat(
    [df_parquet, series[0], series[1], series[2]],
    axis=1,
)

df_parquet.to_parquet("output/static/tasas_secado_downsampled.parquet")

df_marchitez = pd.DataFrame()
df_marchitez["soil_moisture"] = df_parquet[["punto_marchitez"]]

df_marchitez.to_parquet("output/dynamic/soil_moisture/04-2023/30-04-2023/23.parquet")

df = pd.read_parquet("output/static/propiedades_hidricas_downsampled.parquet")
df2 = pd.read_parquet("output/static/tasas_secado_downsampled.parquet")
print(df)
print(df2)
print(df_marchitez)
