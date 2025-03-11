#!/usr/bin/env python3
import argparse
import pandas as pd
import pmdarima as pm
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Determina los mejores parámetros de ARIMA para una serie de tiempo en un archivo CSV."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Ruta al archivo CSV que contiene la serie de tiempo."
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Nombre de la columna que contiene la serie de tiempo. Si no se especifica, se usará la primera columna."
    )
    parser.add_argument(
        "--seasonal",
        action="store_true",
        help="Indica si se debe buscar un modelo SARIMA (con componente estacional)."
    )
    parser.add_argument(
        "--m",
        type=int,
        default=1,
        help="Período de la estacionalidad (solo para modelos SARIMA). Por defecto es 1."
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Muestra la traza del proceso de búsqueda de auto_arima."
    )

    args = parser.parse_args()

    # Leer el archivo CSV
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        sys.exit(f"Error al leer el archivo CSV: {e}")

    # Seleccionar la columna de la serie de tiempo
    if args.column:
        if args.column not in df.columns:
            sys.exit(f"La columna '{args.column}' no se encuentra en el archivo CSV. Columnas disponibles: {list(df.columns)}")
        series = df[args.column]
    else:
        # Usar la primera columna
        series = df.iloc[:, 0]
    
    # Validación de datos: comprobación de valores nulos y conversión a numérico
    if series.isnull().any():
        sys.exit("La serie contiene valores nulos. Por favor, limpia los datos antes de proceder.")
    
    try:
        series = pd.to_numeric(series)
    except Exception as e:
        sys.exit(f"Error al convertir la serie a valores numéricos: {e}")

    print("Buscando los mejores parámetros de ARIMA...")
    try:
        stepwise_model = pm.auto_arima(
            series,
            seasonal=args.seasonal,
            m=args.m,
            trace=args.trace,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True
        )
    except Exception as e:
        sys.exit(f"Error durante auto_arima: {e}")

    print("Mejores parámetros encontrados:")
    print(stepwise_model.summary())

if __name__ == "__main__":
    main()
