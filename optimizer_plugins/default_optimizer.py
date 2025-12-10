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
        "optimization_patience": 3, # Paciencia para early stopping
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
    plugin_debug_vars = ["population_size", "n_generations", "cxpb", "mutpb", "optimization_patience"]

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
        # Fix: Ensure 'plugin' key exists for predictor building
        if "plugin" not in config:
            config["plugin"] = config.get("predictor_plugin", "default_predictor")

        # Extraer el espacio de búsqueda de hiperparámetros.
        bounds = self.params["hyperparameter_bounds"]
        hyper_keys = list(bounds.keys())
        
        # Determine parameter types based on bounds (int vs float)
        param_types = {}
        lower_bounds = []
        upper_bounds = []
        
        for key in hyper_keys:
            low, up = bounds[key]
            lower_bounds.append(low)
            upper_bounds.append(up)
            # Heuristic: if both bounds are int, treat as int
            if isinstance(low, int) and isinstance(up, int):
                param_types[key] = 'int'
            else:
                param_types[key] = 'float'

        # Configuración de DEAP: se define el individuo y la función objetivo.
        # Fix: Check if classes already exist to avoid RuntimeError on re-run
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Generador de atributo: maneja int y float
        def make_attr(low, up, ptype):
            if ptype == 'int':
                return random.randint(low, up)
            else:
                return random.uniform(low, up)

        # Registrar atributos para cada hiperparámetro.
        for i, key in enumerate(hyper_keys):
            low = lower_bounds[i]
            up = upper_bounds[i]
            ptype = param_types[key]
            toolbox.register(f"attr_{key}", make_attr, low, up, ptype)

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
                ptype = param_types[key]
                
                # Specific handling for requested parameters
                if key == "use_log1p_features":
                    # 0 -> None, 1 -> ["typical_price"]
                    val_int = int(round(value))
                    hyper_dict[key] = ["typical_price"] if val_int == 1 else None
                elif key == "positional_encoding":
                    val_int = int(round(value))
                    hyper_dict[key] = bool(val_int)
                elif ptype == 'int':
                    hyper_dict[key] = int(round(value))
                else:
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
            if new_config["plugin"] in ["lstm", "cnn", "transformer", "ann", "mimo"]:
                # Handle 3D input for sequence models
                input_shape = (window_size, x_train.shape[2]) if len(x_train.shape) == 3 else (x_train.shape[1],)
                predictor_plugin.build_model(input_shape=input_shape, x_train=x_train, config=new_config)
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

        # Custom mutation function to handle mixed types
        def mutate(individual, indpb):
            for i, key in enumerate(hyper_keys):
                if random.random() < indpb:
                    low = lower_bounds[i]
                    up = upper_bounds[i]
                    ptype = param_types[key]
                    
                    if ptype == 'int':
                        individual[i] = random.randint(low, up)
                    else:
                        # Gaussian mutation for floats
                        sigma = (up - low) * 0.1
                        val = individual[i] + random.gauss(0, sigma)
                        individual[i] = max(low, min(up, val))
            return individual,

        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", mutate, indpb=self.params.get("mutpb", 0.2))
        toolbox.register("select", tools.selTournament, tournsize=3)

        population_size = self.params.get("population_size", 20)
        n_generations = self.params.get("n_generations", 10)
        patience = self.params.get("optimization_patience", 3)
        
        population = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)

        print("Starting hyperparameter optimization with early stopping...")
        start_opt = time.time()
        
        # Custom optimization loop with early stopping
        # Evaluate the entire population
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        hof.update(population)
        
        best_fitness = float("inf")
        if hof:
            best_fitness = hof[0].fitness.values[0]
            
        no_improve_counter = 0
        
        for gen in range(n_generations):
            print(f"-- Generation {gen + 1}/{n_generations} --")
            
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.params.get("cxpb", 0.5):
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.params.get("mutpb", 0.2):
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace the old population by the offspring
            population[:] = offspring
            
            # Update HallOfFame
            hof.update(population)
            
            # Check for improvement
            current_best = hof[0].fitness.values[0]
            print(f"  Best Val Loss: {current_best}")
            
            if current_best < best_fitness:
                best_fitness = current_best
                no_improve_counter = 0
                print(f"  New best found!")
            else:
                no_improve_counter += 1
                print(f"  No improvement for {no_improve_counter} generations.")
                
            if no_improve_counter >= patience:
                print(f"Early stopping triggered after {gen + 1} generations.")
                break

        end_opt = time.time()
        print(f"Optimization completed in {end_opt - start_opt:.2f} seconds.")

        # Seleccionar el mejor individuo.
        best_ind = hof[0]
        best_hyper = {}
        for i, key in enumerate(hyper_keys):
            value = best_ind[i]
            ptype = param_types[key]
            
            if key == "use_log1p_features":
                val_int = int(round(value))
                best_hyper[key] = ["typical_price"] if val_int == 1 else None
            elif key == "positional_encoding":
                val_int = int(round(value))
                best_hyper[key] = bool(val_int)
            elif ptype == 'int':
                best_hyper[key] = int(round(value))
            else:
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
