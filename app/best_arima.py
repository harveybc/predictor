#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import sys
import warnings
import random
from statsmodels.tsa.arima.model import ARIMA
from deap import base, creator, tools, algorithms

def eval_arima(individual, series):
    p, d, q = individual
    # Si se pasan parámetros negativos, asigna un alto valor de penalización.
    if p < 0 or d < 0 or q < 0:
        return (1e6,),
    try:
        model = ARIMA(series, order=(int(p), int(d), int(q))).fit()
        aic = model.aic
    except Exception as e:
        # Si el modelo falla, asigna una penalización alta.
        aic = 1e6
    return (aic,)

def main():
    parser = argparse.ArgumentParser(
        description="Optimiza los parámetros ARIMA usando DEAP (algoritmos evolutivos) para minimizar el AIC."
    )
    parser.add_argument("csv_file", type=str, help="Ruta al archivo CSV con la serie de tiempo.")
    parser.add_argument("--column", type=str, default=None, help="Nombre de la columna que contiene la serie de tiempo (por defecto, la primera).")
    parser.add_argument("--p_max", type=int, default=5, help="Valor máximo para p (orden autorregresivo).")
    parser.add_argument("--d_max", type=int, default=2, help="Valor máximo para d (diferenciación).")
    parser.add_argument("--q_max", type=int, default=5, help="Valor máximo para q (orden de media móvil).")
    parser.add_argument("--pop_size", type=int, default=20, help="Tamaño de la población.")
    parser.add_argument("--ngen", type=int, default=10, help="Número de generaciones.")
    args = parser.parse_args()

    # Leer el CSV y seleccionar la columna de la serie
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        sys.exit(f"Error al leer el CSV: {e}")

    if args.column:
        if args.column not in df.columns:
            sys.exit(f"La columna '{args.column}' no se encuentra. Columnas disponibles: {list(df.columns)}")
        series = df[args.column]
    else:
        series = df.iloc[:, 0]

    if series.isnull().any():
        sys.exit("La serie contiene valores nulos. Limpia los datos y vuelve a intentarlo.")

    try:
        series = pd.to_numeric(series)
    except Exception as e:
        sys.exit(f"Error al convertir la serie a valores numéricos: {e}")

    warnings.filterwarnings("ignore")

    # Configuración de DEAP para minimizar el AIC.
    # Se define la aptitud con un único objetivo a minimizar.
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # Generadores de atributos: enteros aleatorios dentro de los rangos definidos.
    toolbox.register("attr_p", random.randint, 0, args.p_max)
    toolbox.register("attr_d", random.randint, 0, args.d_max)
    toolbox.register("attr_q", random.randint, 0, args.q_max)
    # Inicialización del individuo: lista [p, d, q].
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_p, toolbox.attr_d, toolbox.attr_q), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Función de evaluación.
    def eval_func(individual):
        return eval_arima(individual, series)

    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=[0, 0, 0], up=[args.p_max, args.d_max, args.q_max], indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=args.pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    print("Iniciando la optimización evolutiva de los parámetros ARIMA...")
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=args.ngen,
                        stats=stats, halloffame=hof, verbose=True)

    best = hof[0]
    best_aic = eval_arima(best, series)[0]
    print("\nMejores parámetros encontrados:")
    print(f"ARIMA({int(best[0])},{int(best[1])},{int(best[2])}) con AIC = {best_aic:.2f}")

if __name__ == "__main__":
    main()
