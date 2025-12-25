#!/usr/bin/env python
"""Pure N-BEATS Predictor (Interpretable & Generic).

Implements the N-BEATS architecture (Oreshkin et al., 2020) with support for:
- Generic Blocks (Fully Connected)
- Trend Blocks (Polynomial Basis)
- Seasonality Blocks (Fourier Basis)

Architecture:
- Input: Flattened window.
- Stacks of Blocks: Configurable sequence of block types.
- Doubly Residual Topology.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Lambda, Add, Subtract, Flatten, Activation, Dropout, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from .common.losses import mae_magnitude
from .common.base import BaseBayesianKerasPredictor
from .common.positional_encoding import positional_encoding

class Plugin(BaseBayesianKerasPredictor):
    plugin_params = {
        "stack_types": ["trend", "seasonality", "generic"],
        "nbeats_blocks": 3, # Blocks per stack
        "nbeats_layers": 4, # Layers per block (for Generic)
        "nbeats_units": 256, # Hidden units
        "trend_polynomial_degree": 3,
        "seasonality_harmonics": 10,
        "activation": "swish",
        "l2_reg": 1e-5,
        "dropout_rate": 0.0,
        "learning_rate": 1e-3,
        "early_patience": 20,
        "batch_size": 64,
        "predicted_horizons": [1],
        "positional_encoding": False,
        # Legacy
        "kl_weight": 0.0, "kl_anneal_epochs": 0, "mc_samples": 1, "mmd_lambda": 0.0, "sigma_mmd": 1.0,
    }
    
    plugin_debug_vars = [
        "stack_types", "nbeats_blocks", "nbeats_layers", "nbeats_units",
        "trend_polynomial_degree", "seasonality_harmonics",
        "activation", "l2_reg", "learning_rate", "predicted_horizons"
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        time_steps, channels = input_shape
        input_dim = time_steps * channels
        ph = self.params["predicted_horizons"]
        forecast_length = max(ph)
        
        stack_types = self.params.get("stack_types", ["generic"])
        blocks_per_stack = self.params.get("nbeats_blocks", 3)
        layers = self.params.get("nbeats_layers", 4)
        units = self.params.get("nbeats_units", 256)
        act = self.params.get("activation", "swish")
        l2_reg_v = self.params.get("l2_reg", 1e-5)
        dropout_rate = self.params.get("dropout_rate", 0.0)
        
        trend_degree = self.params.get("trend_polynomial_degree", 3)
        harmonics = self.params.get("seasonality_harmonics", 10)

        # --- Basis Functions ---
        
        def linspace(length):
            return tf.cast(tf.linspace(-0.5, 0.5, length), dtype=tf.float32)

        # Trend Basis
        t_backcast = linspace(input_dim)
        t_forecast = linspace(forecast_length)
        
        # Polynomial matrix: [time_steps, degree+1]
        T_b = tf.stack([t_backcast ** p for p in range(trend_degree + 1)], axis=1)
        T_f = tf.stack([t_forecast ** p for p in range(trend_degree + 1)], axis=1)
        T_b = tf.constant(T_b)
        T_f = tf.constant(T_f)

        # Seasonality Basis
        # Fourier series: cos(2pi i t), sin(2pi i t)
        S_b_list = [tf.ones_like(t_backcast)]
        S_f_list = [tf.ones_like(t_forecast)]
        for i in range(1, harmonics + 1):
            S_b_list.append(tf.cos(2 * np.pi * i * t_backcast))
            S_b_list.append(tf.sin(2 * np.pi * i * t_backcast))
            S_f_list.append(tf.cos(2 * np.pi * i * t_forecast))
            S_f_list.append(tf.sin(2 * np.pi * i * t_forecast))
        
        S_b = tf.stack(S_b_list, axis=1)
        S_f = tf.stack(S_f_list, axis=1)
        S_b = tf.constant(S_b)
        S_f = tf.constant(S_f)

        # --- Model Construction ---

        inputs = Input(shape=(time_steps, channels), name="input_layer")
        
        if self.params.get("positional_encoding", False):
            pe = positional_encoding(time_steps, channels)
            seq_in = Lambda(lambda t, pe=pe: t + pe, name="add_positional_encoding")(inputs)
        else:
            seq_in = inputs

        flat_in = Flatten(name="flatten_input")(seq_in)
        residual = flat_in
        forecast_accum = None

        # Stack Loop
        for stack_idx, stack_type in enumerate(stack_types):
            for b in range(blocks_per_stack):
                block_name = f"stack{stack_idx}_{stack_type}_b{b}"
                
                # FC Stack
                x = residual
                for l in range(layers):
                    x = Dense(
                        units, 
                        activation=act,
                        kernel_regularizer=l2(l2_reg_v),
                        name=f"{block_name}_dense{l}"
                    )(x)
                    if dropout_rate > 0:
                        x = Dropout(dropout_rate, name=f"{block_name}_drop{l}")(x)

                # Basis Projection
                if stack_type == "trend":
                    # Predict coefficients theta
                    theta_dim = trend_degree + 1
                    theta_b = Dense(theta_dim, activation="linear", name=f"{block_name}_theta_b")(x)
                    theta_f = Dense(theta_dim, activation="linear", name=f"{block_name}_theta_f")(x)
                    
                    # Project to time domain
                    backcast = Lambda(lambda t: tf.matmul(t, T_b, transpose_b=True), name=f"{block_name}_backcast")(theta_b)
                    forecast = Lambda(lambda t: tf.matmul(t, T_f, transpose_b=True), name=f"{block_name}_forecast")(theta_f)

                elif stack_type == "seasonality":
                    theta_dim = 2 * harmonics + 1
                    theta_b = Dense(theta_dim, activation="linear", name=f"{block_name}_theta_b")(x)
                    theta_f = Dense(theta_dim, activation="linear", name=f"{block_name}_theta_f")(x)
                    
                    backcast = Lambda(lambda t: tf.matmul(t, S_b, transpose_b=True), name=f"{block_name}_backcast")(theta_b)
                    forecast = Lambda(lambda t: tf.matmul(t, S_f, transpose_b=True), name=f"{block_name}_forecast")(theta_f)

                else: # Generic
                    backcast = Dense(input_dim, activation="linear", name=f"{block_name}_backcast")(x)
                    forecast = Dense(forecast_length, activation="linear", name=f"{block_name}_forecast")(x)

                # Update Residuals
                residual = Subtract(name=f"{block_name}_resid")([residual, backcast])
                
                if forecast_accum is None:
                    forecast_accum = forecast
                else:
                    forecast_accum = Add(name=f"{block_name}_accum")([forecast_accum, forecast])

        # --- Output Mapping ---
        # forecast_accum is [batch, forecast_length]
        # We need to map this to the specific requested horizons
        
        outputs = []
        self.output_names = []

        for horizon in ph:
            # If horizon is within forecast_length, we can slice it?
            # But N-BEATS predicts a contiguous vector 1..H.
            # If predicted_horizons are indices (1-based), we map to 0-based index.
            
            idx = horizon - 1
            if idx < forecast_length:
                # Slice the specific step
                out = Lambda(lambda t, i=idx: t[:, i:i+1], name=f"slice_h{horizon}")(forecast_accum)
            else:
                # Fallback if horizon > forecast_length (shouldn't happen if configured right)
                out = Dense(1, activation="linear", name=f"dense_h{horizon}")(forecast_accum)
                
            outputs.append(out)
            self.output_names.append(f"output_horizon_{horizon}")

        self.model = Model(inputs=inputs, outputs=outputs, name=f"NBEATS_Interpretable")
        
        optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))
        loss_dict = {nm: Huber() for nm in self.output_names}
        metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}
        
        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        self.model.summary(line_length=140)

if __name__ == '__main__':
    plug = Plugin({"predicted_horizons": [1,3], "positional_encoding": True})
    plug.build_model((96, 8), None, {})
    print('Outputs:', plug.output_names)
