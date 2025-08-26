import numpy as np
import pandas as pd
from .helpers import load_normalization_json, denormalize_all_datasets, load_normalized_csv, exclude_columns_from_datasets
from .sliding_windows import create_sliding_windows, extract_baselines_from_sliding_windows
from .target_calculation import calculate_targets_from_baselines
from .anti_naive_lock import apply_log_returns_to_series, apply_feature_normalization


class STLPreprocessorZScore:
    """
    1. Load already normalized CSV data ✅
    2. Denormalize all input datasets using JSON parameters
    3. Create sliding windows from denormalized data
    4. Extract baselines (last elements of each window for target column)
    5. Calculate log return targets with those baselines (train, validation, test)
     6. Create SECOND sliding windows matrix from the ORIGINAL normalized datasets transformed with per-column log-returns
         (applies to all numeric features). Dates preserved; no change to target pipeline.
     7. Keep baselines and targets unchanged (they're already calculated correctly)
    """

    # Plugin-specific parameters they get overwritten if declared in the config
    plugin_params = {
        "window_size": 48,
        "predicted_horizons": [1, 6],
        "target_column": "CLOSE",
        "use_returns": True,
        "anti_naive_lock_enabled": True,
    "feature_preprocessing_strategy": "selective",
    "add_window_stats": True,
    "window_stats_periods": [12, 48],
    "reverse_time_axis": False,
    # New: optional multi-scale returns augmentation (causal, within-window)
    "add_multi_scale_returns": False,
    "multi_scale_return_periods": [6, 24, 72]
    }
    
    plugin_debug_vars = ["window_size", "predicted_horizons", "target_column"]

    # Start of plugin interface methods    
    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value
    
    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}
    
    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())
    # End of plugin interface methods

    def process_data(self, config):
        # Main process orchestration
        try:
            self.set_params(**config)
            config = self.params
            
            predicted_horizons = config['predicted_horizons']
            if not isinstance(predicted_horizons, list) or not predicted_horizons:
                raise ValueError("predicted_horizons must be a non-empty list")
            
            # 1. Load already normalized CSV data
            print("Step 1: Load normalized CSV data")
            normalized_data, dates = load_normalized_csv(config)
            if not normalized_data:
                raise ValueError("No data loaded - check file paths in config")
            
            # 2. Denormalize all input datasets using JSON parameters
            print("Step 2: Denormalize all input datasets")
            denormalized_data = denormalize_all_datasets(normalized_data, config)
            
            # 3. Create FIRST sliding windows from denormalized data used only and only for baseline extraction
            print("Step 3: Create first sliding windows from denormalized data")
            denorm_sliding_windows = create_sliding_windows(denormalized_data, config, dates)

            # 4. Extract baselines from the sliding windows (last elements of each window for target column)
            print("Step 4: Extract baselines from sliding windows")
            baselines = extract_baselines_from_sliding_windows(denorm_sliding_windows, config)

            # 5. Calculate targets directly from baselines
            print("Step 5: Calculate targets from baselines")
            #TODO: verify this method is correct
            targets = calculate_targets_from_baselines(baselines, config)

            # 6. Create SECOND sliding windows from DENORMALIZED datasets after applying price log-returns
            #    Apply log returns to raw price columns only (OPEN/HIGH/LOW/CLOSE). Preserve other columns.
            print("Step 6: Create second sliding windows from denormalized data")
            #denorm_returns_x = self._build_denorm_price_returns_x(denormalized_data, config)
            final_sliding_windows = create_sliding_windows(denormalized_data, config, dates)

            # 7. Align final sliding windows with target data length
            print("Step 7: Align sliding windows with target data")
            final_sliding_windows = self._align_sliding_windows_with_targets(final_sliding_windows, targets, config)

            # 7b. Optionally add simple window statistics features to strengthen signal
            if config.get('add_window_stats', True):
                try:
                    final_sliding_windows = self._augment_with_window_stats(final_sliding_windows, config)
                except Exception as aug_e:
                    print(f"WARN: Failed to add window stats: {aug_e}")

            # 7b.2. Optionally add multi-scale cumulative returns for price columns (causal, tail-only)
            if config.get('add_multi_scale_returns', False):
                try:
                    final_sliding_windows = self._augment_with_multiscale_returns(final_sliding_windows, config)
                except Exception as aug2_e:
                    print(f"WARN: Failed to add multi-scale returns: {aug2_e}")

            # 7c. Optionally reverse time axis (test model expectation of temporal ordering)
            if config.get('reverse_time_axis', False):
                try:
                    for split in ['train','val','test']:
                        Xk = f'X_{split}'
                        if Xk in final_sliding_windows and hasattr(final_sliding_windows[Xk], 'shape'):
                            final_sliding_windows[Xk] = final_sliding_windows[Xk][:, ::-1, :]
                    print("  Applied reverse_time_axis: windows reversed along time dimension (axis=1)")
                except Exception as rev_e:
                    print(f"WARN: reverse_time_axis failed: {rev_e}")

            # 7c.1 Optionally normalize features post-augmentation using train-only stats
            if config.get('normalize_after_preprocessing', False):
                try:
                    fn_train = final_sliding_windows.get('feature_names_train', final_sliding_windows.get('feature_names', []))
                    Xtr = final_sliding_windows.get('X_train')
                    Xva = final_sliding_windows.get('X_val')
                    Xte = final_sliding_windows.get('X_test')
                    if Xtr is not None and Xva is not None and Xte is not None and isinstance(fn_train, list):
                        Xtr_n, Xva_n, Xte_n, norm_stats = apply_feature_normalization(Xtr.copy(), Xva.copy(), Xte.copy(), fn_train)
                        final_sliding_windows['X_train'] = Xtr_n
                        final_sliding_windows['X_val'] = Xva_n
                        final_sliding_windows['X_test'] = Xte_n
                        # Keep feature names unchanged; store stats into params for reference
                        self.params['post_feature_norm_stats'] = norm_stats
                        print(f"  Applied post-augmentation z-score normalization using training stats (features={len(fn_train)})")
                    else:
                        print("  Skipped post-augmentation normalization (missing X arrays or feature names)")
                except Exception as norm_e:
                    print(f"WARN: post-augmentation normalization failed: {norm_e}")

            # 7d. Optional: residualize targets by subtracting naive last-step CLOSE return
            naive_info = {}
            if config.get('residualize_targets_with_last_close', True):
                try:
                    target_col = config.get('target_column', 'CLOSE')
                    tfac = float(config.get('target_factor', 1000.0))
                    for split in ['train','val','test']:
                        Xk = f'X_{split}'
                        fn_sp = final_sliding_windows.get(f'feature_names_{split}', final_sliding_windows.get('feature_names', []))
                        if Xk not in final_sliding_windows or not isinstance(fn_sp, list) or target_col not in fn_sp:
                            continue
                        ci = fn_sp.index(target_col)
                        Xsp = final_sliding_windows[Xk]
                        # last-step CLOSE log-return (unscaled)
                        naive_unscaled = Xsp[:, -1, ci].astype(np.float64)
                        naive_scaled = (tfac * naive_unscaled).astype(np.float32)
                        naive_info[f'naive_last_close_scaled_{split}'] = naive_scaled
                        # Adjust all horizon targets by subtracting naive_scaled
                        y_split = targets.get(f'y_{split}', {})
                        for hk, yarr in list(y_split.items()):
                            if not isinstance(yarr, np.ndarray):
                                yarr = np.asarray(yarr)
                            m = min(len(yarr), len(naive_scaled))
                            y_split[hk] = (yarr[:m].astype(np.float32) - naive_scaled[:m].astype(np.float32))
                        targets[f'y_{split}'] = y_split
                    if naive_info:
                        print("  Residualized targets using last-step CLOSE return; stored naive arrays for pipeline add-back")
                except Exception as res_e:
                    print(f"WARN: residualize_targets_with_last_close failed: {res_e}")
            
            # === Debugging & Invariants: Verify targets vs baselines and X statistics ===
            try:
                tgt_factor = float(config.get('target_factor', 1000.0))
                target_col = config.get('target_column', 'CLOSE')
                # Per-split feature names
                fn_train = final_sliding_windows.get('feature_names_train', final_sliding_windows.get('feature_names', []))
                fn_val = final_sliding_windows.get('feature_names_val', final_sliding_windows.get('feature_names', []))
                fn_test = final_sliding_windows.get('feature_names_test', final_sliding_windows.get('feature_names', []))

                # 1) Re-derive H1 targets from baselines and compare
                for split in ['train', 'val', 'test']:
                    base = baselines.get(f'baseline_{split}')
                    if base is None or len(base) < 2:
                        continue
                    # Recompute H1 log-returns scaled by target_factor
                    base = np.asarray(base, dtype=np.float64)
                    valid = (base[:-1] > 0) & (base[1:] > 0)
                    recomputed = np.zeros(len(base)-1, dtype=np.float64)
                    recomputed[valid] = tgt_factor * np.log(base[1:][valid] / base[:-1][valid])
                    # Truncate to max_samples used in target calc
                    max_h = max(config['predicted_horizons'])
                    max_samples = len(base) - max_h
                    if max_samples > 0:
                        recomputed = recomputed[:max_samples]
                        y_split = targets.get(f'y_{split}', {})
                        y_h1 = y_split.get('output_horizon_1')
                        if y_h1 is not None and len(y_h1) > 0:
                            y_h1 = np.asarray(y_h1, dtype=np.float64)
                            m = min(len(y_h1), len(recomputed))
                            diff = np.abs(y_h1[:m] - recomputed[:m])
                            print(f"DEBUG[{split.upper()}]: H1 targets check — mean|diff|={diff.mean():.6f}, max|diff|={diff.max():.6f}, samples={m}")

                # 2) Feature distributions for X (train)
                Xtr = final_sliding_windows.get('X_train')
                if Xtr is not None and hasattr(Xtr, 'shape'):
                    # global stats
                    print(f"DEBUG[X_train]: shape={Xtr.shape}, global mean={np.mean(Xtr):.6f}, std={np.std(Xtr):.6f}")
                    # CLOSE last-timestep stats and correlation with H1 target
                    if isinstance(fn_train, list) and target_col in fn_train:
                        ci = fn_train.index(target_col)
                        x_last_close = Xtr[:, -1, ci].astype(np.float64)
                        y_h1_tr = targets.get('y_train', {}).get('output_horizon_1')
                        if y_h1_tr is not None and len(y_h1_tr) == len(x_last_close):
                            y_arr = np.asarray(y_h1_tr, dtype=np.float64)
                            # Corr can be nan if std is zero; guard
                            corr = np.nan
                            if np.std(x_last_close) > 0 and np.std(y_arr) > 0:
                                corr = np.corrcoef(x_last_close, y_arr)[0,1]
                            print(f"DEBUG[CLOSE vs H1]: mean(x)={x_last_close.mean():.6f}, std(x)={x_last_close.std():.6f}, mean(y)={y_arr.mean():.6f}, std(y)={y_arr.std():.6f}, corr={corr}")
                        else:
                            print("DEBUG: Skipping CLOSE/H1 correlation — length mismatch")
                    else:
                        print("DEBUG: CLOSE not found in feature_names_train for X stats")

                # 3) Dates alignment check between FIRST (denorm) and SECOND (final) windows
                for split in ['train','val','test']:
                    d1 = denorm_sliding_windows.get(f'x_dates_{split}')
                    d2 = final_sliding_windows.get(f'x_dates_{split}')
                    if d1 is None or d2 is None:
                        print(f"DEBUG: Missing dates for {split}, skip alignment check")
                        continue
                    m = min(len(d1), len(d2))
                    eq = np.array(d1[:m]) == np.array(d2[:m])
                    mism = int((~eq).sum()) if hasattr(eq, 'sum') else 'NA'
                    print(f"DEBUG[DATES {split.upper()}]: first {m} dates equal? mismatches={mism}")

                # 4) Sample rows dump: date, baseline[t], baseline[t+1], y1[t], x_last_close_ret[t]
                for split in ['train','val','test']:
                    base = baselines.get(f'baseline_{split}')
                    d2 = final_sliding_windows.get(f'x_dates_{split}')
                    Xsp = final_sliding_windows.get(f'X_{split}')
                    fn_sp = final_sliding_windows.get(f'feature_names_{split}', final_sliding_windows.get('feature_names', []))
                    if base is None or d2 is None or Xsp is None or not isinstance(fn_sp, list) or 'CLOSE' not in fn_sp:
                        continue
                    ci = fn_sp.index('CLOSE')
                    y_h1 = targets.get(f'y_{split}',{}).get('output_horizon_1')
                    if y_h1 is None:
                        continue
                    n = min(5, len(y_h1), len(base)-1, len(d2), len(Xsp))
                    print(f"DEBUG[SAMPLE {split.upper()}] t, date, base[t], base[t+1], y1[t], x_last_close_ret[t]")
                    for t in range(n):
                        x_last = float(Xsp[t,-1,ci]) if Xsp.shape[2] > ci else float('nan')
                        print(f"  {t}: {d2[t]} | {float(base[t]):.6f} | {float(base[t+1]):.6f} | {float(y_h1[t]):.6f} | {x_last:.6f}")
            except Exception as dbg_e:
                print(f"WARN: Debug invariant checks failed: {dbg_e}")

            # Return final results
            #TODO: verify this method is correct and required
            output, preprocessor_params = self._prepare_final_output(final_sliding_windows, targets, baselines, config)
            # attach naive info if present
            for k, v in naive_info.items():
                output[k] = v
            
            # Store baselines for access in output preparation
            self.extracted_baselines = baselines
            
            self.params.update(preprocessor_params)
            return output

        except Exception as e:
            print(f"ERROR in process_data: {e}")
            raise

    def _build_denorm_price_returns_x(self, denormalized_data, config):
        """Build a dict with x_*_df only, applying log-returns to price features on DENORMALIZED data.

        - Applies ln(p_t/p_{t-1}) to columns in config['price_features'] that exist in the DataFrame.
        - Preserves other columns unchanged.
        - Keeps index/length the same; first row per column becomes 0.0 by design.
        - Optionally standardizes features post-transform if normalize_after_preprocessing is True.
        """
        import pandas as pd
        price_features = set([c for c in config.get('price_features', ['OPEN','HIGH','LOW','CLOSE'])])

        out = {}
        for split in ['train', 'val', 'test']:
            key = f'x_{split}_df'
            if key not in denormalized_data:
                continue
            df = denormalized_data[key]
            if df is None or len(df) == 0:
                out[key] = df
                continue

            df_tx = df.copy()
            for col in df_tx.columns:
                series = df_tx[col]
                if col in price_features and pd.api.types.is_numeric_dtype(series):
                    try:
                        df_tx[col] = apply_log_returns_to_series(series)
                    except Exception as e:
                        print(f"        WARN: price log-returns failed for '{col}' in {key}: {e}; preserving original")
                        df_tx[col] = series
                else:
                    df_tx[col] = series

            out[key] = df_tx

        return out

    def _align_sliding_windows_with_targets(self, sliding_windows, targets, config):
        """Align sliding windows with target data to ensure same number of samples."""
        print("  Aligning sliding windows with target data...")
        
        # Get the first target to determine the target length
        predicted_horizons = config['predicted_horizons']
        first_horizon = predicted_horizons[0]
        
        # Find target lengths for each split
        target_lengths = {}
        for split in ['train', 'val', 'test']:
            target_key = f'y_{split}'
            if target_key in targets and f'output_horizon_{first_horizon}' in targets[target_key]:
                target_length = len(targets[target_key][f'output_horizon_{first_horizon}'])
                target_lengths[split] = target_length
                print(f"    {split} target length: {target_length}")
            else:
                target_lengths[split] = 0
        
        # Trim sliding windows to match target lengths
        aligned_windows = {}

        for key, windows in sliding_windows.items():
            if key.startswith('X_'):
                # Extract split name (train, val, test)
                split = key.split('_')[1]
                if split in target_lengths and target_lengths[split] > 0:
                    target_length = target_lengths[split]
                    if hasattr(windows, 'shape') and len(windows) > target_length:
                        aligned_windows[key] = windows[:target_length]
                        print(f"    Trimmed {key} from {len(windows)} to {target_length} samples")
                    else:
                        aligned_windows[key] = windows
                        
                else:
                    aligned_windows[key] = windows
                    
            else:
                # Keep non-window data as is
                aligned_windows[key] = windows
                

        return aligned_windows

    def _augment_with_window_stats(self, sliding_windows, config):
        """Append simple window statistics for CLOSE as additional channels.

        For each split X_{split} with shape (N, T, F), add for CLOSE:
          - mean over last k timesteps
          - std over last k timesteps
          - momentum = last - mean over last k timesteps (difference; bounded and stable for returns)
        for each k in window_stats_periods intersecting [2, T].
        """
        periods = config.get('window_stats_periods', [12, 48])
        target_col = config.get('target_column', 'CLOSE')
        for split in ['train', 'val', 'test']:
            X_key = f'X_{split}'
            fn_key = f'feature_names_{split}'
            if X_key not in sliding_windows:
                continue
            X = sliding_windows[X_key]
            fn = sliding_windows.get(fn_key, sliding_windows.get('feature_names', []))
            if not isinstance(fn, list) or target_col not in fn:
                continue
            ci = fn.index(target_col)
            N, T, F = X.shape
            add_feats = []
            add_names = []
            for k in periods:
                if k < 2 or k > T:
                    continue
                # Compute over the trailing k timesteps for each sample
                window_slice = X[:, -k:, ci]
                mean_k = np.mean(window_slice, axis=1, keepdims=True)
                std_k = np.std(window_slice, axis=1, keepdims=True) + 1e-9
                last = X[:, -1:, ci]
                # Use difference instead of ratio to avoid exploding values when mean is near zero (returns)
                mom_k = last - mean_k
                # Tile along time dimension to match (N, T, 1) using last value repeated
                mean_feat = np.repeat(mean_k, T, axis=1)
                std_feat = np.repeat(std_k, T, axis=1)
                mom_feat = np.repeat(mom_k, T, axis=1)
                add_feats.extend([mean_feat[..., None], std_feat[..., None], mom_feat[..., None]])
                add_names.extend([f'{target_col}_mean_{k}', f'{target_col}_std_{k}', f'{target_col}_mom_{k}'])
            if add_feats:
                extra = np.concatenate(add_feats, axis=2)  # (N, T, 3*len(periods_kept))
                sliding_windows[X_key] = np.concatenate([X, extra], axis=2)
                # Update feature names
                updated_names = fn + add_names
                sliding_windows[fn_key] = updated_names
                if split == 'train':
                    sliding_windows['feature_names'] = updated_names
                print(f"  Added window stats for {split}: +{len(add_names)} features; new shape {sliding_windows[X_key].shape}")
        return sliding_windows

    def _augment_with_multiscale_returns(self, sliding_windows, config):
        """Append multi-scale cumulative returns for price columns as additional channels.

        For each split X_{split} with shape (N, T, F), and for each price column present in
        config['price_features'], compute cumulative log-returns over the last k timesteps
        (sum of returns over window tail) for k in multi_scale_return_periods intersecting [2, T].

        The resulting (N, 1) per (col, k) is repeated across time to (N, T, 1) and concatenated
        as causal features (no future leakage).
        """
        periods = config.get('multi_scale_return_periods', [6, 24, 72])
        price_features = list(config.get('price_features', ['OPEN', 'HIGH', 'LOW', 'CLOSE']))
        for split in ['train', 'val', 'test']:
            X_key = f'X_{split}'
            fn_key = f'feature_names_{split}'
            if X_key not in sliding_windows:
                continue
            X = sliding_windows[X_key]
            fn = sliding_windows.get(fn_key, sliding_windows.get('feature_names', []))
            if not isinstance(fn, list) or X is None or not hasattr(X, 'shape'):
                continue
            # Identify indices for price columns available in this split
            price_indices = [(c, fn.index(c)) for c in price_features if c in fn]
            if not price_indices:
                continue
            N, T, F = X.shape
            add_feats = []
            add_names = []
            for col, ci in price_indices:
                for k in periods:
                    if k < 2 or k > T:
                        continue
                    tail = X[:, -k:, ci]  # (N, k)
                    cumret = np.sum(tail, axis=1, keepdims=True)  # (N, 1)
                    cumret_feat = np.repeat(cumret, T, axis=1)[..., None]  # (N, T, 1)
                    add_feats.append(cumret_feat)
                    add_names.append(f"{col}_cumret_{k}")
            if add_feats:
                extra = np.concatenate(add_feats, axis=2)
                sliding_windows[X_key] = np.concatenate([X, extra], axis=2)
                # Update feature names
                updated_names = fn + add_names
                sliding_windows[fn_key] = updated_names
                if split == 'train':
                    sliding_windows['feature_names'] = updated_names
                print(f"  Added multi-scale returns for {split}: +{len(add_names)} features; new shape {sliding_windows[X_key].shape}")
        return sliding_windows

    def _prepare_final_output(self, sliding_windows, targets, baselines, config):
        """Prepare final output structure."""
        # Use the baselines passed as parameter (extracted from denormalized data)
        baseline_data = {}
        if isinstance(baselines, dict):
            # baselines is already in the correct format
            baseline_data = baselines
        else:
            # Handle legacy format
            for split in ['train', 'val', 'test']:
                baseline_key = f'baseline_{split}'
                baseline_data[baseline_key] = np.array([])
        
        # Validate that we have the required data structures
        required_sliding_window_keys = ['X_train', 'X_val', 'X_test']
        required_target_keys = ['y_train', 'y_val', 'y_test']
        
        for key in required_sliding_window_keys:
            if key not in sliding_windows:
                print(f"WARNING: Missing sliding window data: {key}")
                sliding_windows[key] = np.array([])
        
        for key in required_target_keys:
            if key not in targets:
                print(f"WARNING: Missing target data: {key}")
                targets[key] = {}
        
        output = {
            # Final sliding windows for model (SECOND sliding windows after anti-naive-lock)
            "x_train": sliding_windows['X_train'],
            "x_val": sliding_windows['X_val'],
            "x_test": sliding_windows['X_test'],
            
            # Targets by horizon (calculated from FIRST sliding windows)
            "y_train": targets['y_train'],
            "y_val": targets['y_val'],
            "y_test": targets['y_test'],
            
            # Dates
            "x_train_dates": sliding_windows.get('x_dates_train'),
            "y_train_dates": sliding_windows.get('x_dates_train'),
            "x_val_dates": sliding_windows.get('x_dates_val'),
            "y_val_dates": sliding_windows.get('x_dates_val'),
            "x_test_dates": sliding_windows.get('x_dates_test'),
            "y_test_dates": sliding_windows.get('x_dates_test'),
            
            # Baselines for prediction reconstruction
            "baseline_train": baseline_data.get('baseline_train', np.array([])),
            "baseline_val": baseline_data.get('baseline_val', np.array([])),
            "baseline_test": baseline_data.get('baseline_test', np.array([])),
            
            # Metadata
            "feature_names": sliding_windows.get('feature_names', []),
            "feature_names_train": sliding_windows.get('feature_names_train', []),
            "feature_names_val": sliding_windows.get('feature_names_val', []),
            "feature_names_test": sliding_windows.get('feature_names_test', []),
            "target_returns_means": targets.get('target_returns_means', []),
            "target_returns_stds": targets.get('target_returns_stds', []),
            "predicted_horizons": config['predicted_horizons'],
            "normalization_json": load_normalization_json(config),
        }
        
        # Print summary statistics
        print("\nPreprocessing Summary:")
        print(f"  X_train shape: {output['x_train'].shape if hasattr(output['x_train'], 'shape') else 'N/A'}")
        print(f"  X_val shape: {output['x_val'].shape if hasattr(output['x_val'], 'shape') else 'N/A'}")
        print(f"  X_test shape: {output['x_test'].shape if hasattr(output['x_test'], 'shape') else 'N/A'}")
        print(f"  Feature names: {len(output['feature_names'])}")
        print(f"  Predicted horizons: {output['predicted_horizons']}")
        print(f"  Target normalization parameters available: {len(output['target_returns_means'])}")
        print(f"  Baseline train length: {len(output['baseline_train'])}")
        print(f"  Baseline val length: {len(output['baseline_val'])}")
        print(f"  Baseline test length: {len(output['baseline_test'])}")

        output, preprocessor_params = exclude_columns_from_datasets(output, self.params, config)

        return output, preprocessor_params
    
    def run_preprocessing(self, config):
        """Run preprocessing with configuration."""
        run_config = self.params.copy()
        run_config.update(config)
        self.set_params(**run_config)
        processed_data = self.process_data(self.params)
        
        params_with_targets = self.params.copy()
        params_with_targets.update({
            "target_returns_means": processed_data.get("target_returns_means", []),
            "target_returns_stds": processed_data.get("target_returns_stds", []),
            "normalization_json": processed_data.get("normalization_json", {})
        })
        
        return processed_data, params_with_targets


# Plugin interface alias for the system
PreprocessorPlugin = STLPreprocessorZScore
