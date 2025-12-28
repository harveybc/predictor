#!/usr/bin/env python
"""Temporal Fusion Transformer (TFT) Predictor Plugin.

Implements a TFT-inspired architecture adapted for the current pipeline.
Key components:
- Gated Residual Networks (GRN)
- LSTM Encoder
- Multi-Head Attention
- Skip connections and Gating
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Lambda, Add, Activation, Dropout, LSTM, 
    LayerNormalization, MultiHeadAttention, Concatenate, 
    GlobalAveragePooling1D, TimeDistributed, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from .common.losses import mae_magnitude
from .common.base import BaseKerasPredictor

class Plugin(BaseKerasPredictor):
    plugin_params = {
        "tft_hidden_units": 64,
        "tft_num_heads": 4,
        "tft_dropout": 0.1,
        "tft_lstm_layers": 2,
        "learning_rate": 1e-3,
        "early_patience": 20,
        "batch_size": 64,
        "predicted_horizons": [1],
    }
    
    plugin_debug_vars = [
        "tft_hidden_units", "tft_num_heads", "tft_dropout", 
        "tft_lstm_layers", "learning_rate", "predicted_horizons"
    ]

    def _glu(self, x, units):
        # Gated Linear Unit
        # GLU(x) = sigma(W1 x + b1) * (W2 x + b2)
        # Implementation:
        # val = Dense(units)(x)
        # gate = Dense(units, activation='sigmoid')(x)
        # return val * gate
        
        val = Dense(units, activation=None)(x)
        gate = Dense(units, activation='sigmoid')(x)
        return Multiply()([val, gate])

    def _grn(self, x, units, dropout_rate, context=None):
        # Gated Residual Network
        # x -> Dense -> ELU -> Dense -> Dropout -> GLU -> Add(x) -> Norm
        
        skip = x
        if x.shape[-1] != units:
            skip = Dense(units)(x) # Projection for skip connection if dims mismatch

        h = Dense(units, activation='elu')(x)
        h = Dense(units)(h)
        h = Dropout(dropout_rate)(h)
        h = self._glu(h, units)
        
        h = Add()([skip, h])
        h = LayerNormalization()(h)
        return h

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)

        time_steps, channels = input_shape
        ph = self.params["predicted_horizons"]
        
        units = self.params["tft_hidden_units"]
        num_heads = self.params["tft_num_heads"]
        dropout = self.params["tft_dropout"]
        lstm_layers = self.params["tft_lstm_layers"]
        
        inputs = Input(shape=(time_steps, channels), name="input_layer")
        
        # 1. Variable Selection / Embedding
        # Project input to hidden units
        x = self._grn(inputs, units, dropout)
        
        # 2. LSTM Encoder (Locality)
        # Processes the sequence to capture local patterns
        for _ in range(lstm_layers):
            x = LSTM(units, return_sequences=True, dropout=dropout)(x)
            x = self._grn(x, units, dropout) # Apply GRN after LSTM
            
        # 3. Temporal Fusion Decoder (Attention)
        # Multi-Head Attention to capture long-range dependencies
        # Self-attention on the LSTM output
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=units)(x, x)
        
        # Post-attention gating
        h = self._grn(attn_out, units, dropout)
        h = Add()([x, h]) # Skip connection over attention
        h = LayerNormalization()(h)
        
        # 4. Output Generation
        # Use the output at the last timestep as the context for prediction
        context = Lambda(lambda t: t[:, -1, :])(h)
        
        outputs = []
        self.output_names = []

        for horizon in ph:
            # Each horizon gets its own head
            head = self._grn(context, units, dropout)
            out = Dense(1, activation="linear", name=f"output_horizon_{horizon}")(head)
            outputs.append(out)
            self.output_names.append(f"output_horizon_{horizon}")

        self.model = Model(inputs=inputs, outputs=outputs, name=f"TFT_Simple_{len(ph)}H")
        
        optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))
        loss_dict = {nm: Huber() for nm in self.output_names}
        metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}
        
        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        self.model.summary(line_length=140)

    def predict_with_uncertainty(self, x_test, mc_samples: int = 1):
        """Deterministic prediction (uncertainty = 0)."""
        preds = self.model.predict(x_test, verbose=0)
        if not isinstance(preds, list):
            preds = [preds]
        
        # Return predictions and zero uncertainty
        unc = [np.zeros_like(p) for p in preds]
        return preds, unc
