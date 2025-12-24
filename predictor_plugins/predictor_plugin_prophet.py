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
        "add_country_holidays": None, # e.g. "US"
        "daily_seasonality": "auto",
        "weekly_seasonality": "auto",
        "yearly_seasonality": "auto",
        "use_regressors": False,
    }

    plugin_debug_vars: List[str] = [
        "predicted_horizons",
        "prophet_params",
        "interval_width",
        "add_country_holidays",
        "daily_seasonality",
        "weekly_seasonality",
        "yearly_seasonality",
        "use_regressors",
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
        country_holidays = self.params.get("add_country_holidays")
        daily_seasonality = self.params.get("daily_seasonality", "auto")
        weekly_seasonality = self.params.get("weekly_seasonality", "auto")
        yearly_seasonality = self.params.get("yearly_seasonality", "auto")
        use_regressors = self.params.get("use_regressors", False)
        feature_names = self.params.get("feature_names", [])

        train_preds_list = []
        train_unc_list = []
        val_preds_list = []
        val_unc_list = []

        # Prepare regressors if enabled
        regressor_cols = []
        if use_regressors and len(feature_names) > 0:
            # x_train is (N, W, F). We take the last step of the window: (N, F)
            # This assumes the last step contains the most recent known values for the regressors
            # corresponding to the prediction point.
            # Note: Prophet regressors must be known in the future for prediction.
            # If x_test contains future values (e.g. known external variables), this is valid.
            # If x_test contains only past values, this is technically data leakage or invalid for forecasting
            # unless we are doing 1-step ahead with known next-step inputs.
            # Given the pipeline structure, x_test usually contains the window leading up to the target.
            # So we are using the most recent past to predict the future.
            # Prophet treats regressors as concurrent with 'y'.
            # So we should align x_train[i, -1] with y_train[i].
            x_train_reg = x_train[:, -1, :]
            regressor_cols = feature_names
            
            if x_val is not None:
                x_val_reg = x_val[:, -1, :]

        for h in horizons:
            key = f"output_horizon_{h}"
            y = y_train.get(key)
            if y is None:
                continue
            
            # Prepare DataFrame for Prophet
            data = {
                'ds': pd.to_datetime(train_dates),
                'y': y.flatten()
            }
            
            if use_regressors and regressor_cols:
                for idx, col in enumerate(regressor_cols):
                    if idx < x_train_reg.shape[1]:
                        data[col] = x_train_reg[:, idx]

            df = pd.DataFrame(data)

            # Initialize and train Prophet
            m = Prophet(
                interval_width=interval_width,
    def predict_with_uncertainty(
        self,
        x_test: np.ndarray,
        mc_samples: int = 50
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        
        test_dates = self.params.get("test_dates")
        if test_dates is None:
             raise ValueError("Prophet plugin requires 'test_dates' in params.")

        horizons = self.params.get("predicted_horizons", [1])
        use_regressors = self.params.get("use_regressors", False)
        feature_names = self.params.get("feature_names", [])
        
        preds_list = []
        unc_list = []

        # Prepare regressors if enabled
        regressor_cols = []
        if use_regressors and len(feature_names) > 0:
            x_test_reg = x_test[:, -1, :]
            regressor_cols = feature_names

        for h in horizons:
            m = self.models.get(h)
            if m is None:
                preds_list.append(np.zeros((len(x_test), 1)))
                unc_list.append(np.zeros((len(x_test), 1)))
                continue

            test_data = {'ds': pd.to_datetime(test_dates)}
            
            if use_regressors and regressor_cols:
                for idx, col in enumerate(regressor_cols):
                    if idx < x_test_reg.shape[1]:
                        test_data[col] = x_test_reg[:, idx]

            df_test = pd.DataFrame(test_data)
            forecast = m.predict(df_test)
            
            yhat = forecast['yhat'].values.reshape(-1, 1)
            unc = ((forecast['yhat_upper'] - forecast['yhat_lower']) / 2).values.reshape(-1, 1)
            
            preds_list.append(yhat)
            unc_list.append(unc)

        return preds_list, unc_listnd regressor_cols:
                    for idx, col in enumerate(regressor_cols):
                        if idx < x_val_reg.shape[1]:
                            val_data[col] = x_val_reg[:, idx]
                            
                df_val = pd.DataFrame(val_data)
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
