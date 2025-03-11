#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import warnings
import sys
from statsmodels.tsa.arima.model import ARIMA

def main():
    parser = argparse.ArgumentParser(
        description="Determina los mejores parámetros de ARIMA para una serie de tiempo usando statsmodels con búsqueda en malla."
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
        help="Nombre de la columna que contiene la serie de tiempo. Si no se especifica, se usará la primera columna numérica encontrada (por ejemplo, 'CLOSE')."
    )
    parser.add_argument(
        "--p_max",
        type=int,
        default=3,
        help="Valor máximo para p (orden autorregresivo)."
    )
    parser.add_argument(
        "--d_max",
        type=int,
        default=2,
        help="Valor máximo para d (diferenciación)."
    )
    parser.add_argument(
        "--q_max",
        type=int,
        default=3,
        help="Valor máximo para q (orden de media móvil)."
    )
    args = parser.parse_args()

    # Leer el archivo CSV
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        sys.exit(f"Error al leer el archivo CSV: {e}")

    # Si no se especifica la columna, buscamos la primera columna numérica
    if args.column is None:
        for col in df.columns:
            try:
                pd.to_numeric(df[col])
                args.column = col
                print(f"Se usará la columna '{col}' como serie de tiempo.")
                break
            except Exception:
                continue
        if args.column is None:
            sys.exit("No se encontró ninguna columna numérica en el CSV.")
    else:
        if args.column not in df.columns:
            sys.exit(f"La columna '{args.column}' no se encuentra en el archivo CSV. Columnas disponibles: {list(df.columns)}")

    series = df[args.column]

    # Validación de datos: comprobación de valores nulos y conversión a numérico
    if series.isnull().any():
        sys.exit("La serie contiene valores nulos. Por favor, limpia los datos antes de proceder.")

    try:
        series = pd.to_numeric(series)
    except Exception as e:
        sys.exit(f"Error al convertir la serie a valores numéricos: {e}")

    best_aic = np.inf
    best_order = None
    best_model = None
    warnings.filterwarnings("ignore")
    
    print("Buscando los mejores parámetros ARIMA...")
    for p in range(0, args.p_max + 1):
        for d in range(0, args.d_max + 1):
            for q in range(0, args.q_max + 1):
                try:
                    model = ARIMA(series, order=(p, d, q)).fit()
                    current_aic = model.aic
                    print(f"Probando ARIMA({p},{d},{q}) AIC = {current_aic:.2f}")
                    if current_aic < best_aic:
                        best_aic = current_aic
                        best_order = (p, d, q)
                        best_model = model
                except Exception as e:
                    print(f"ARIMA({p},{d},{q}) falló: {e}")
                    continue

    if best_order is None:
        sys.exit("No se encontró ningún modelo ARIMA válido.")

    print("\nMejores parámetros encontrados:")
    print(f"ARIMA{best_order} con AIC = {best_aic:.2f}")
    print(best_model.summary())

if __name__ == "__main__":
    main()
