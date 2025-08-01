import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

def load_normalization_json(config):
    """Loads normalization parameters from JSON file."""
    if config.get("use_normalization_json"):
        norm_json = config["use_normalization_json"]
        if isinstance(norm_json, str):
            try:
                with open(norm_json, 'r') as f:
                    norm_json = json.load(f)
                return norm_json
            except Exception as e:
                print(f"WARN: Failed to load norm JSON {norm_json}: {e}")
                return {}
        return norm_json
    return {}

def denormalize(data, norm_json, column_name="CLOSE"):
    """Denormalizes data using JSON normalization parameters."""
    data = np.asarray(data)
    if isinstance(norm_json, dict) and column_name in norm_json:
        try:
            if "mean" in norm_json[column_name] and "std" in norm_json[column_name]:
                mean = norm_json[column_name]["mean"]
                std = norm_json[column_name]["std"]
                return (data * std) + mean
            else:
                print(f"WARN: Missing 'mean' or 'std' in norm JSON for {column_name}")
                return data
        except Exception as e:
            print(f"WARN: Error during denormalize for {column_name}: {e}")
            return data
    return data

def verify_date_consistency(date_lists, dataset_name):
    """
    Verifies that all date arrays in date_lists have the same first and last elements.
    Prints a warning if any array does not match.
    """
    if not date_lists: 
        return
    valid_date_lists = [d for d in date_lists if d is not None and len(d) > 0]
    if not valid_date_lists: 
        return
    first_len = len(valid_date_lists[0])
    if not all(len(d) == first_len for d in valid_date_lists):
         print(f"WARN: Length mismatch in {dataset_name} dates: {[len(d) for d in valid_date_lists]}")
         return
    try: 
        first = valid_date_lists[0][0]
        last = valid_date_lists[0][-1]
    except IndexError: 
        print(f"WARN: Could not access first/last element in {dataset_name} dates.")
        return
    consistent = True
    for i, d in enumerate(valid_date_lists):
        try:
            if len(d) > 0:
                if d[0] != first or d[-1] != last:
                    print(f"Warning: Date array {i} in {dataset_name} range ({d[0]} to {d[-1]}) does not match first array ({first} to {last}).")
                    consistent = False
            else: 
                print(f"Warning: Date array {i} in {dataset_name} is empty.")
                consistent = False
        except Exception as e: 
            print(f"Warning: Could not compare dates in {dataset_name} for array {i}. Error: {e}.")
            consistent = False

def normalize_series(series, name, scalers, fit=False, normalize_features=True):
    """Normalizes a series using StandardScaler."""
    if not normalize_features: 
        return series.astype(np.float32)
    
    series = series.astype(np.float32)
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        print(f"WARN: NaNs/Infs in '{name}' pre-norm. Filling...", end="")
        series = pd.Series(series).fillna(method='ffill').fillna(method='bfill').values
        if np.any(np.isnan(series)) or np.any(np.isinf(series)):
            print(f" FAILED. Filling with 0.", end="")
            series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
        print(" OK.")
    
    data_reshaped = series.reshape(-1, 1)
    if fit:
        scaler = StandardScaler()
        if np.std(data_reshaped) < 1e-9:
            print(f"WARN: '{name}' constant. Dummy scaler.")
            class DummyScaler:
                def fit(self, X): pass
                def transform(self, X): return X.astype(np.float32)
                def inverse_transform(self, X): return X.astype(np.float32)
            scaler = DummyScaler()
        else: 
            scaler.fit(data_reshaped)
        scalers[name] = scaler
    else:
        if name not in scalers: 
            raise RuntimeError(f"Scaler '{name}' not fitted.")
        scaler = scalers[name]
    
    normalized_data = scaler.transform(data_reshaped)
    return normalized_data.flatten()

def align_features(feature_dict, base_length):
    """Aligns features to a common base length."""
    aligned_features = {}
    min_len = base_length
    feature_lengths = {'base': base_length}
    valid_keys = [k for k, v in feature_dict.items() if v is not None]
    if not valid_keys: 
        return {}, 0
    
    for name in valid_keys: 
        feature_lengths[name] = len(feature_dict[name])
        min_len = min(min_len, feature_lengths[name])
    
    needs_alignment = any(l != min_len for l in feature_lengths.values() if l > 0)
    if needs_alignment:
        for name in valid_keys:
            series = feature_dict[name]
            current_len = len(series)
            if current_len > min_len: 
                aligned_features[name] = series[current_len - min_len:]
            elif current_len == min_len: 
                aligned_features[name] = series
            else: 
                print(f"WARN: Feature '{name}' len({current_len})<target({min_len}).")
                aligned_features[name] = None
    else: 
        aligned_features = {k: feature_dict[k] for k in valid_keys}
    
    final_lengths = {name: len(s) for name, s in aligned_features.items() if s is not None}
    unique_lengths = set(final_lengths.values())
    if len(unique_lengths) > 1: 
        raise RuntimeError(f"Alignment FAILED! Inconsistent lengths: {final_lengths}")
    return aligned_features, min_len