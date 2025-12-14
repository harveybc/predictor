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
import json
import gc
import tensorflow as tf
import os
from deap import base, creator, tools, algorithms
from app.plugin_loader import load_plugin

from predictor_plugins.common.callbacks import capture_resource_snapshot

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

        # Load Target Plugin to pass to preprocessor
        target_plugin_name = config.get('target_plugin', 'default_target')
        try:
            target_class, _ = load_plugin('target.plugins', target_plugin_name)
            target_plugin = target_class()
            target_plugin.set_params(**config)
        except Exception as e:
            print(f"Failed to load Target Plugin inside optimizer: {e}")
            raise e

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
        
        # Global counters for progress tracking
        self.eval_counter = 0
        self.current_gen = 0
        self.best_fitness_so_far = float("inf")
        self.patience_counter = 0
        self.best_naive_mae_so_far = None

        def _json_sanitize(obj):
            """Recursively convert numpy/tensor types into JSON-serializable Python types."""
            # TensorFlow tensors
            try:
                if isinstance(obj, tf.Tensor):
                    obj = obj.numpy()
            except Exception:
                pass

            # Numpy scalars
            if isinstance(obj, np.generic):
                return obj.item()

            # Numpy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()

            if isinstance(obj, dict):
                return {str(k): _json_sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_json_sanitize(v) for v in obj]

            # Plain Python types are fine (including None)
            return obj

        def _atomic_json_dump(path: str, payload: dict) -> None:
            """Write JSON atomically to avoid corrupt/truncated files on crashes/OOM."""
            tmp_path = f"{path}.tmp"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump(_json_sanitize(payload), f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)

        def _append_resource_row(stage: str, gen: int | None = None, cand: int | None = None, extra: dict | None = None) -> None:
            log_path = config.get("optimizer_resource_log_file")
            if not log_path:
                return
            try:
                snap = capture_resource_snapshot(include_gpu=bool(config.get("memory_log_gpu", True)), include_gc=bool(config.get("memory_log_gc", False)))
                os.makedirs(os.path.dirname(str(log_path)) or ".", exist_ok=True)
                new_file = not os.path.exists(str(log_path))
                with open(str(log_path), "a", encoding="utf-8") as f:
                    if new_file:
                        f.write(
                            "ts,stage,generation,candidate,VmRSS_kB,VmHWM_kB,gpu_current_B,gpu_peak_B,extra\n"
                        )
                    extra_json = ""
                    if extra:
                        try:
                            extra_json = json.dumps(_json_sanitize(extra), separators=(",", ":"))
                        except Exception:
                            extra_json = ""
                    f.write(
                        f"{snap.ts:.3f},{stage},{gen if gen is not None else ''},{cand if cand is not None else ''},"
                        f"{snap.rss_kb if snap.rss_kb is not None else ''},{snap.hwm_kb if snap.hwm_kb is not None else ''},"
                        f"{snap.gpu_current_bytes if snap.gpu_current_bytes is not None else ''},{snap.gpu_peak_bytes if snap.gpu_peak_bytes is not None else ''},"
                        f"{extra_json}\n"
                    )
                    f.flush()
            except Exception as e:
                print(f"WARN: optimizer resource logging failed: {e}")
        
        def eval_individual(individual):
            _append_resource_row("candidate_start", gen=int(self.current_gen or 0), cand=int(self.eval_counter + 1))

            # CRITICAL FIX: Clear Keras session and garbage collect to prevent OOM
            if hasattr(predictor_plugin, 'model'):
                del predictor_plugin.model
            tf.keras.backend.clear_session()
            gc.collect()

            # Optional: recreate predictor plugin per candidate to avoid hidden state accumulation
            predictor_for_eval = predictor_plugin
            if bool(config.get("optimizer_recreate_predictor_each_eval", False)):
                try:
                    predictor_for_eval = predictor_plugin.__class__(config)
                    predictor_for_eval.set_params(**config)
                except Exception as e:
                    print(f"WARN: could not recreate predictor plugin; using shared instance: {e}")
                    predictor_for_eval = predictor_plugin

            self.eval_counter += 1
            
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

            print(f"\n--- Evaluating Candidate {self.eval_counter}/{population_size} (Gen {self.current_gen + 1}/{n_generations}) ---")
            print(f"Params: {hyper_dict}")

            # Combinar los hiperparámetros con la configuración actual.
            new_config = config.copy()
            new_config.update(hyper_dict)

            # Tag epoch-level resource logs so we can correlate with GA generation/candidate.
            new_config.setdefault("memory_log_tag", f"ga_gen{int(self.current_gen or 0)}_cand{int(self.eval_counter)}")

            # CRITICAL: During GA optimization, avoid post-fit MC uncertainty passes.
            # The default BaseBayesianKerasPredictor does MC prediction over full train+val
            # after every fit, which can explode memory/time on large datasets.
            new_config.setdefault("disable_postfit_uncertainty", True)
            new_config.setdefault("mc_samples", 1)
            new_config.setdefault("predict_batch_size", new_config.get("batch_size", 32))

            # Obtener los datos de entrenamiento y validación usando el Preprocessor Plugin.
            datasets = preprocessor_plugin.run_preprocessing(target_plugin, new_config)
            
            # Handle tuple return from stl_preprocessor (datasets, params)
            if isinstance(datasets, tuple):
                datasets = datasets[0]
                
            x_train, y_train = datasets["x_train"], datasets["y_train"]
            x_val, y_val = datasets["x_val"], datasets["y_val"]

            # Construir y entrenar el modelo utilizando el Predictor Plugin.
            window_size = new_config.get("window_size")
            if new_config["plugin"] in ["lstm", "cnn", "transformer", "ann", "mimo"]:
                # Handle 3D input for sequence models
                input_shape = (window_size, x_train.shape[2]) if len(x_train.shape) == 3 else (x_train.shape[1],)
                predictor_for_eval.build_model(input_shape=input_shape, x_train=x_train, config=new_config)
            else:
                predictor_for_eval.build_model(input_shape=x_train.shape[1], x_train=x_train, config=new_config)
            
            # Calculate Naive MAE for the highest horizon
            # Naive prediction: use the last known value (t-1) as prediction for all future steps
            # For normalized data, this is often just 0 if using returns, or the last value if using prices.
            # Assuming standard time series where x_val contains history.
            # We need to know the target column index or assume it's the last one or specific one.
            # However, a simpler robust way for "Naive" in this context (often used in this repo) 
            # is Mean Absolute Error of (y_true[1:] - y_true[:-1]) for 1-step, but for N-step it varies.
            # The user defined Naive MAE as "using the current value as prediction".
            # We will calculate it using the last value of the input window vs the target.
            
            naive_mae = float("inf")
            try:
                # Get the highest horizon index
                predicted_horizons = new_config.get("predicted_horizons", [1])
                max_horizon = max(predicted_horizons)
                max_h_idx = predicted_horizons.index(max_horizon)
                
                # y_val is a dict for MIMO or list/array. Preprocessor usually returns dict for MIMO.
                # If it's a dict: {'output_horizon_1': ..., 'output_horizon_6': ...}
                # We need the target for the max horizon.
                
                y_true_max_h = None
                if isinstance(y_val, dict):
                    key = f"output_horizon_{max_horizon}"
                    if key in y_val:
                        y_true_max_h = y_val[key]
                elif isinstance(y_val, list):
                     if len(y_val) > max_h_idx:
                        y_true_max_h = y_val[max_h_idx]
                elif isinstance(y_val, np.ndarray):
                    # If single output or stacked
                    y_true_max_h = y_val
                
                if y_true_max_h is not None:
                    # Naive prediction: The last value in the input sequence x_val.
                    # x_val shape: (samples, window_size, features)
                    # We assume the target feature is at index 0 or we need to know which one.
                    # Config has "target_column". But x_val is numpy array.
                    # We'll assume the target is the first column (index 0) of the input if not specified,
                    # OR better, we use the "baseline" if available from preprocessor, but preprocessor returns x/y.
                    # Let's use the last value of the input window (index -1) of the first feature (index 0).
                    # This is a standard heuristic if we don't have column mapping here.
                    
                    # x_val[:, -1, 0] is the most recent value for each sample.
                    # We compare this against y_true_max_h.
                    
                    # Ensure shapes match
                    # preds_naive = x_val[:, -1, 0].reshape(-1, 1)
                    
                    # Calculate MAE
                    # naive_errors = np.abs(y_true_max_h - preds_naive)
                    # naive_mae = np.mean(naive_errors)
                    
                    # Use baseline_val if available (as per pipeline logic)
                    baseline_val = datasets.get("baseline_val")
                    if baseline_val is not None:
                        # Pipeline logic: denormalize(baseline) - denormalize(target)
                        # But here we are in normalized space for optimization fitness (val_loss).
                        # If we want to compare apples to apples with val_loss (which is usually MSE or MAE on normalized data),
                        # we should compute Naive MAE on normalized data too.
                        # Pipeline computes metrics on DENORMALIZED data.
                        # The user said: "Validation Naive MAE and for the validation MAE... verify that the MAE is claulated in the same exact way"
                        # The optimizer uses `history.history["val_loss"][-1]` as fitness.
                        # If the model loss is MSE/MAE on normalized data, then we should compute Naive MAE on normalized data.
                        # BUT, the pipeline prints Denormalized MAE.
                        # If the user sees "Validation MAE" in the pipeline output, that is Denormalized.
                        # If the user sees "Val Loss" in the optimizer output, that is Normalized (usually).
                        # The user complained "values are way too diferent".
                        # This confirms we are comparing Normalized Loss vs Denormalized Naive (or vice versa).
                        
                        # To fix this, we must calculate the Validation MAE exactly as the pipeline does:
                        # 1. Predict on x_val
                        # 2. Denormalize predictions and targets
                        # 3. Compute MAE
                        
                        # We cannot rely on history['val_loss'] if we want to match the pipeline's "Validation MAE".
                        # We must run a prediction step here.
                        pass # Logic moved below to "After training" block
            except Exception as e:
                print(f"Warning: Could not calculate Naive MAE pre-check: {e}")
                naive_mae = float("inf")

            try:
                # Para optimización, usar menos epochs.
                _append_resource_row("before_fit", gen=int(self.current_gen or 0), cand=int(self.eval_counter), extra={"params": hyper_dict})
                history, _, _, val_preds, _ = predictor_for_eval.train(
                    x_train, y_train,
                    epochs=new_config.get("epochs", 10),
                    batch_size=new_config.get("batch_size", 32),
                    threshold_error=new_config.get("threshold_error", 0.001),
                    x_val=x_val, y_val=y_val, config=new_config
                )
                _append_resource_row("after_fit", gen=int(self.current_gen or 0), cand=int(self.eval_counter))
                
                # --- Compute Metrics Exactly like Pipeline ---
                # We need to replicate compute_train_val_metrics logic for the max horizon
                
                # 1. Get predictions for max horizon
                # val_preds is a list of arrays (one per horizon)
                predicted_horizons = new_config.get("predicted_horizons", [1])
                max_horizon = max(predicted_horizons)
                max_h_idx = predicted_horizons.index(max_horizon)
                
                val_preds_h = val_preds[max_h_idx].flatten()
                
                # 2. Get targets for max horizon
                # y_val is dict or list. We need to extract the array.
                if isinstance(y_val, dict):
                    y_true_max_h = y_val[f"output_horizon_{max_horizon}"].flatten()
                elif isinstance(y_val, list):
                    y_true_max_h = y_val[max_h_idx].flatten()
                else:
                    y_true_max_h = y_val.flatten()
                
                # 3. Get baseline for Naive
                baseline_val = datasets.get("baseline_val")
                
                # 4. Align lengths
                num_val_pts = min(len(val_preds_h), len(y_true_max_h))
                if baseline_val is not None:
                    num_val_pts = min(num_val_pts, len(baseline_val))
                
                val_preds_h = val_preds_h[:num_val_pts]
                y_true_max_h = y_true_max_h[:num_val_pts]
                
                # 5. Denormalize (Crucial step from pipeline)
                # We need the 'denormalize' function. We can import it or implement simple version if params available.
                # Pipeline uses: from .stl_norm import denormalize
                # We should import it at top of file.
                from pipeline_plugins.stl_norm import denormalize, denormalize_returns
                
                val_target_price = denormalize(y_true_max_h, new_config)
                val_pred_price = denormalize(val_preds_h, new_config)
                
                # 6. Compute MAE (Pipeline: denormalize_returns(pred - target))
                # Wait, pipeline does: np.mean(np.abs(denormalize_returns(val_preds_h - val_target_h, params)))
                # This implies the model predicts returns/normalized values, and we denormalize the DIFFERENCE?
                # Or denormalize the values then subtract?
                # Pipeline code: val_mae_h = np.mean(np.abs(denormalize_returns(val_preds_h - val_target_h, params)))
                # If use_returns=False (which is enforced in pipeline), denormalize_returns might just be identity or simple scaling.
                # Let's look at stl_metrics.py again.
                # "train_mae_h = np.mean(np.abs(denormalize_returns(train_preds_h - train_target_h, params)))"
                # AND "train_naive_mae_h = np.mean(np.abs(denormalize(baseline_train_h, params) - train_target_price))"
                
                # So for Model MAE: denormalize_returns(pred - target)
                # For Naive MAE: denormalize(baseline) - denormalize(target)
                
                # Let's follow this exactly.
                
                # Model MAE
                val_mae = np.mean(np.abs(denormalize_returns(val_preds_h - y_true_max_h, new_config)))
                
                # Naive MAE
                naive_mae = float("inf")
                if baseline_val is not None:
                    baseline_val_h = baseline_val[:num_val_pts].flatten()
                    # Note: pipeline uses denormalize(baseline) - val_target_price
                    # val_target_price was computed as denormalize(y_true_max_h)
                    naive_mae = np.mean(np.abs(denormalize(baseline_val_h, new_config) - val_target_price))
                
                # Use this calculated MAE as fitness instead of val_loss?
                # The user wants to compare them. If we optimize for val_loss (normalized MSE usually), 
                # but print Denormalized MAE, that's fine, as long as we print both correctly.
                # BUT, if the user wants to optimize for the metric they see, we should return val_mae as fitness.
                # "Validation MAE of the champion has not been incremented" -> implies we should use this MAE for early stopping/fitness.
                # Let's use val_mae as the fitness metric to be consistent.
                
                fitness = val_mae
                train_loss = history.history["loss"][-1] # Keep this for info
                
                print(f"Candidate Result -> Val MAE (Max H): {fitness:.6f} | Naive MAE (Max H): {naive_mae:.6f} | Train Loss: {train_loss:.6f}")
            except Exception as e:
                print(f"Training/Metrics failed for individual {hyper_dict}: {e}")
                fitness = float("inf")

            # Hard cleanup after each candidate as well (prevents long-run accumulation)
            try:
                if hasattr(predictor_for_eval, 'model'):
                    del predictor_for_eval.model
            except Exception:
                pass
            tf.keras.backend.clear_session()
            gc.collect()

            _append_resource_row("candidate_end", gen=int(self.current_gen or 0), cand=int(self.eval_counter))
            
            # Print optimization state
            print(f"Optimization State:")
            print(f"  Current Best Val MAE to Beat: {self.best_fitness_so_far:.6f}")
            print(f"  Patience Counter: {self.patience_counter}/{patience}")
            print(f"------------------------------------------------------------")
            
            # Store naive_mae in individual for stats later (hacky but effective)
            individual.naive_mae = naive_mae
            
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
        self.current_gen = 0
        self.eval_counter = 0 # Reset for initial population
        
        # Enforce bounds on initial population (just in case)
        for ind in population:
            for i, val in enumerate(ind):
                low = lower_bounds[i]
                up = upper_bounds[i]
                ind[i] = max(low, min(up, val))

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        hof.update(population)
        
        if hof:
            self.best_fitness_so_far = hof[0].fitness.values[0]
            
        no_improve_counter = 0
        self.patience_counter = 0 # Sync with local var
        
        # Statistics tracking
        stats_history = []
        
        for gen in range(n_generations):
            gen_start_time = time.time()
            self.current_gen = gen + 1 # Update for logging
            self.eval_counter = 0 # Reset per generation
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
            
            # Enforce bounds on all offspring (fix for cxBlend producing out-of-bounds values)
            for child in offspring:
                for i, val in enumerate(child):
                    low = lower_bounds[i]
                    up = upper_bounds[i]
                    child[i] = max(low, min(up, val))
            
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
            
            if current_best < self.best_fitness_so_far:
                self.best_fitness_so_far = current_best
                no_improve_counter = 0
                self.patience_counter = 0
                print(f"  New best found!")

                # Capture champion naive MAE at the moment the global champion improves
                try:
                    self.best_naive_mae_so_far = getattr(hof[0], "naive_mae", self.best_naive_mae_so_far)
                except Exception:
                    pass
            else:
                no_improve_counter += 1
                self.patience_counter = no_improve_counter
                print(f"  No improvement for {no_improve_counter} generations.")
            
            # --- Save Statistics and Champion Parameters ---
            gen_end_time = time.time()
            elapsed_time = gen_end_time - start_opt
            gen_duration = gen_end_time - gen_start_time
            
            # Calculate average MAE for this generation
            valid_fitnesses = [ind.fitness.values[0] for ind in population if ind.fitness.valid and ind.fitness.values[0] != float("inf")]
            avg_mae = sum(valid_fitnesses) / len(valid_fitnesses) if valid_fitnesses else float("inf")
            
            # Get Naive MAE from the best individual (or average, but usually we want the benchmark)
            # We stored it in individual.naive_mae during evaluation
            best_ind_gen = tools.selBest(population, 1)[0]
            champion_naive_mae_gen = getattr(best_ind_gen, "naive_mae", None)

            # Global champion naive MAE (HallOfFame) for top-level stats
            champion_naive_mae_global = getattr(hof[0], "naive_mae", None) if hof else None
            if champion_naive_mae_global is None:
                champion_naive_mae_global = self.best_naive_mae_so_far
            
            stats_history.append({
                "generation": self.current_gen,
                "duration": gen_duration,
                "avg_mae": avg_mae,
                "champion_naive_mae": champion_naive_mae_gen
            })
            
            avg_time_per_epoch = sum(s["duration"] for s in stats_history) / len(stats_history)
            total_candidates = int(self.eval_counter)
            
            stats_data = {
                "total_time_elapsed": elapsed_time,
                "average_time_per_epoch": avg_time_per_epoch,
                "candidates_evaluated_so_far": total_candidates, # Or track exactly with self.eval_counter accumulator
                "champion_validation_mae": float(self.best_fitness_so_far) if self.best_fitness_so_far is not None else None,
                "champion_naive_mae_max_horizon": champion_naive_mae_global,
                "average_mae_per_epoch": [s["avg_mae"] for s in stats_history],
                "history": stats_history
            }
            
            # Save Statistics
            stats_file = config.get("optimization_statistics", "optimization_stats.json")
            try:
                _atomic_json_dump(stats_file, stats_data)
                print(f"  Statistics saved to {stats_file}")
            except Exception as e:
                print(f"  Failed to save statistics: {e}")

            # Save Champion Parameters
            best_ind_so_far = hof[0]
            best_hyper_so_far = {}
            for i, key in enumerate(hyper_keys):
                value = best_ind_so_far[i]
                ptype = param_types[key]
                if key == "use_log1p_features":
                    val_int = int(round(value))
                    best_hyper_so_far[key] = ["typical_price"] if val_int == 1 else None
                elif key == "positional_encoding":
                    val_int = int(round(value))
                    best_hyper_so_far[key] = bool(val_int)
                elif ptype == 'int':
                    best_hyper_so_far[key] = int(round(value))
                else:
                    best_hyper_so_far[key] = value
            
            params_file = config.get("optimization_parameters_file", "optimization_parameters.json")
            try:
                _atomic_json_dump(params_file, best_hyper_so_far)
                print(f"  Champion parameters saved to {params_file}")
            except Exception as e:
                print(f"  Failed to save champion parameters: {e}")

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
