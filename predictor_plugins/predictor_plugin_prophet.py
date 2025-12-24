#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predictor_plugin_prophet.py

Predictor plugin using Meta Prophet.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet

from .common.base import BasePredictorPlugin

class MockHistory:
    def __init__(self):
        self.history = {'loss': [0.0], 'val_loss': [0.0]}

class Plugin(BasePredictorPlugin):
    """
    Predictor plugin wrapping Meta Prophet.
    Trains one Prophet model per horizon.
    """

    plugin_params: Dict[str, Any] = {
        "predicted_horizons": [1],
        "prophet_params": {},  # Params passed to Prophet constructor
        "interval_width": 0.6827, # Approx 1 sigma
    }

    plugin_debug_vars: List[str] = [
        "predicted_horizons",
        "prophet_params",
        "interval_width",
    ]

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.models: Dict[int, Prophet] = {}
        self.output_names: List[str] = []

    def build_model(
        self,
        input_shape: Tuple[int, ...],
        x_train: Any,
        config: Dict[str, Any],
    ) -> None:
        """
        Setup models dict. Actual creation happens in train because we need data.
        """
        if config:
            self.params.update(config)
        
        horizons = self.params.get("predicted_horizons", [1])
        self.output_names = [f"output_horizon_{h}" for h in horizons]
        # Models will be created in train

    def train(
        self,
        x_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        epochs: int,
        batch_size: int,
        threshold_error: float,
        x_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[Any, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        
        if config:
            self.params.update(config)

        train_dates = self.params.get("train_dates")
        val_dates = self.params.get("val_dates")

        if train_dates is None:
            raise ValueError("Prophet plugin requires 'train_dates' in params. Ensure pipeline passes them.")

        horizons = self.params.get("predicted_horizons", [1])
        prophet_params = self.params.get("prophet_params", {})
        interval_width = self.params.get("interval_width", 0.6827)

        train_preds_list = []
        train_unc_list = []
        val_preds_list = []
        val_unc_list = []

        for h in horizons:
            key = f"output_horizon_{h}"
            y = y_train.get(key)
            if y is None:
                # Should not happen if y_train is correct
                continue
            
            # Prepare DataFrame for Prophet
            # y is (samples, 1)
            df = pd.DataFrame({
                'ds': pd.to_datetime(train_dates),
                'y': y.flatten()
            })

            # Initialize and train Prophet
            m = Prophet(interval_width=interval_width, **prophet_params)
            # If we had regressors, we would add them here
            # m.add_regressor(...)
            
            m.fit(df)
            self.models[h] = m

            # Predict on train
            forecast_train = m.predict(df)
            yhat_train = forecast_train['yhat'].values.reshape(-1, 1)
            unc_train = ((forecast_train['yhat_upper'] - forecast_train['yhat_lower']) / 2).values.reshape(-1, 1)
            
            train_preds_list.append(yhat_train)
            train_unc_list.append(unc_train)

            # Predict on val
            if val_dates is not None and key in y_val:
                df_val = pd.DataFrame({
                    'ds': pd.to_datetime(val_dates),
                    'y': y_val[key].flatten() # Prophet ignores y in predict, but good for structure
                })
                forecast_val = m.predict(df_val)
                yhat_val = forecast_val['yhat'].values.reshape(-1, 1)
                unc_val = ((forecast_val['yhat_upper'] - forecast_val['yhat_lower']) / 2).values.reshape(-1, 1)
                
                val_preds_list.append(yhat_val)
                val_unc_list.append(unc_val)
            else:
                val_preds_list.append(np.zeros((len(x_val), 1)))
                val_unc_list.append(np.zeros((len(x_val), 1)))

        return MockHistory(), train_preds_list, train_unc_list, val_preds_list, val_unc_list

    def predict_with_uncertainty(
        self,
        x_test: np.ndarray,
        mc_samples: int = 50
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        
        test_dates = self.params.get("test_dates")
        if test_dates is None:
             raise ValueError("Prophet plugin requires 'test_dates' in params.")

        horizons = self.params.get("predicted_horizons", [1])
        preds_list = []
        unc_list = []

        for h in horizons:
            m = self.models.get(h)
            if m is None:
                preds_list.append(np.zeros((len(x_test), 1)))
                unc_list.append(np.zeros((len(x_test), 1)))
                continue

            df_test = pd.DataFrame({'ds': pd.to_datetime(test_dates)})
            forecast = m.predict(df_test)
            
            yhat = forecast['yhat'].values.reshape(-1, 1)
            unc = ((forecast['yhat_upper'] - forecast['yhat_lower']) / 2).values.reshape(-1, 1)
            
            preds_list.append(yhat)
            unc_list.append(unc)

        return preds_list, unc_list

    def save(self, file_path: str) -> None:
        # Save all models in a pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"Prophet models saved to {file_path}")

    def load(self, file_path: str) -> None:
        with open(file_path, 'rb') as f:
            self.models = pickle.load(f)
        print(f"Prophet models loaded from {file_path}")
