#!/usr/bin/env python
"""
Default Optimizer Plugin

Este plugin utiliza algoritmos genéticos (DEAP) para optimizar los hiperparámetros
del Predictor Plugin. La función optimize realiza una búsqueda en el espacio de hiperparámetros
definidos y retorna un diccionario con los parámetros óptimos encontrados.

Se asume que:
  - Los hiperparámetros a optimizar están definidos en "hyperparameter_bounds".
  - Algunos parámetros deben ser tratados como enteros (por ejemplo, 'num_layers' o 'early_patience').

Nota: Se utiliza un número reducido de epochs para la evaluación en el proceso de optimización.
"""

import random
import numpy as np
import time
from deap import base, creator, tools, algorithms

class Plugin:
    # Parámetros por defecto del optimizador.
    plugin_params = {
        "population_size": 20,
        "n_generations": 10,
        "cxpb": 0.5,      # Probabilidad de cruce.
        "mutpb": 0.2,     # Probabilidad de mutación.
        # Espacio de hiperparámetros a optimizar, con sus límites.
        "hyperparameter_bounds": {
            "learning_rate": (1e-5, 1e-2),
            "num_layers": (1, 5),
            "layer_size": (16, 256),
            "l2_reg": (1e-7, 1e-3),
            "mmd_lambda": (1e-5, 1e-2),
            "early_patience": (10, 100)
        }
    }
    plugin_debug_vars = ["population_size", "n_generations", "cxpb", "mutpb"]

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        """
        Actualiza los parámetros del optimizador combinando los parámetros específicos con la configuración global.
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Devuelve información de debug de los parámetros relevantes del optimizador.
        """
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Agrega la información de debug del optimizador al diccionario proporcionado.
        """
        debug_info.update(self.get_debug_info())

    def optimize(self, predictor_plugin, preprocessor_plugin, config):
        """
        Realiza la optimización de hiperparámetros utilizando algoritmos genéticos (DEAP).

        Args:
            predictor_plugin: Plugin encargado del predictor, que se evaluará con los hiperparámetros.
            preprocessor_plugin: Plugin encargado del preprocesamiento de datos.
            config (dict): Configuración global.

        Returns:
            dict: Diccionario con los hiperparámetros óptimos.
        """
        # Extraer el espacio de búsqueda de hiperparámetros.
        bounds = self.params["hyperparameter_bounds"]
        hyper_keys = list(bounds.keys())
        lower_bounds = [bounds[key][0] for key in hyper_keys]
        upper_bounds = [bounds[key][1] for key in hyper_keys]

        # Especificar qué parámetros se deben tratar como enteros.
        int_params = {"num_layers", "early_patience"}

        # Configuración de DEAP: se define el individuo y la función objetivo.
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Generador de atributo: un número aleatorio entre el límite inferior y superior.
        def random_param(low, up):
            return random.uniform(low, up)

        # Registrar atributos para cada hiperparámetro.
        for i, key in enumerate(hyper_keys):
            low = lower_bounds[i]
            up = upper_bounds[i]
            toolbox.register(f"attr_{key}", random_param, low, up)

        # Definir el individuo como una lista de hiperparámetros.
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         [toolbox.__getattribute__(f"attr_{key}") for key in hyper_keys], n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Función de evaluación: construye y entrena el modelo con los hiperparámetros dados,
        # devolviendo la pérdida de validación del último epoch.
        def eval_individual(individual):
            # Mapear el individuo a un diccionario de hiperparámetros.
            hyper_dict = {}
            for i, key in enumerate(hyper_keys):
                value = individual[i]
                if key in int_params:
                    value = int(round(value))
                hyper_dict[key] = value

            print(f"Evaluating individual: {hyper_dict}")

            # Combinar los hiperparámetros con la configuración actual.
            new_config = config.copy()
            new_config.update(hyper_dict)

            # Obtener los datos de entrenamiento y validación usando el Preprocessor Plugin.
            datasets = preprocessor_plugin.run_preprocessing(new_config)
            x_train, y_train = datasets["x_train"], datasets["y_train"]
            x_val, y_val = datasets["x_val"], datasets["y_val"]

            # Construir y entrenar el modelo utilizando el Predictor Plugin.
            window_size = new_config.get("window_size")
            if new_config["plugin"] in ["lstm", "cnn", "transformer", "ann"]:
                predictor_plugin.build_model(input_shape=(window_size, x_train.shape[2]), x_train=x_train, config=new_config)
            else:
                predictor_plugin.build_model(input_shape=x_train.shape[1], x_train=x_train, config=new_config)
            try:
                # Para optimización, usar menos epochs.
                history, _, _, val_preds, _ = predictor_plugin.train(
                    x_train, y_train,
                    epochs=new_config.get("epochs", 10),
                    batch_size=new_config.get("batch_size", 32),
                    threshold_error=new_config.get("threshold_error", 0.001),
                    x_val=x_val, y_val=y_val, config=new_config
                )
                fitness = history.history["val_loss"][-1]
            except Exception as e:
                print(f"Training failed for individual {hyper_dict}: {e}")
                fitness = float("inf")
            return (fitness,)

        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutUniformInt, low=lower_bounds, up=upper_bounds, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population_size = self.params.get("population_size", 20)
        n_generations = self.params.get("n_generations", 10)
        population = toolbox.population(n=population_size)

        print("Starting hyperparameter optimization...")
        start_opt = time.time()
        # Ejecutar el algoritmo genético (utilizando eaSimple).
        population, logbook = algorithms.eaSimple(
            population, toolbox,
            cxpb=self.params.get("cxpb", 0.5),
            mutpb=self.params.get("mutpb", 0.2),
            ngen=n_generations, verbose=True
        )
        end_opt = time.time()
        print(f"Optimization completed in {end_opt - start_opt:.2f} seconds.")

        # Seleccionar el mejor individuo.
        best_ind = tools.selBest(population, k=1)[0]
        best_hyper = {}
        for i, key in enumerate(hyper_keys):
            value = best_ind[i]
            if key in int_params:
                value = int(round(value))
            best_hyper[key] = value
        print(f"Best hyperparameters found: {best_hyper}")
        return best_hyper

# Debugging usage example (cuando se ejecuta el plugin directamente)
if __name__ == "__main__":
    optimizer_plugin = Plugin()
    test_config = {
        "plugin": "ann",
        "x_train_file": "data/train.csv",
        "x_validation_file": "data/val.csv",
        "x_test_file": "data/test.csv",
        "window_size": 24,
        "time_horizon": 1,
        "batch_size": 32,
        "epochs": 10,
        "threshold_error": 0.001
    }
    # Cargar instancias de Predictor y Preprocessor Plugins.
    from app.plugin_loader import load_plugin
    predictor_class, _ = load_plugin('predictor.plugins', test_config.get('plugin', 'default_predictor'))
    predictor_plugin = predictor_class()
    predictor_plugin.set_params(**test_config)
    from app.plugin_loader import load_plugin as load_preprocessor_plugin
    preprocessor_class, _ = load_preprocessor_plugin('preprocessor.plugins', 'default_preprocessor')
    preprocessor_plugin = preprocessor_class()
    preprocessor_plugin.set_params(**test_config)
    best_params = optimizer_plugin.optimize(predictor_plugin, preprocessor_plugin, test_config)
    print("Optimized parameters:", best_params)
