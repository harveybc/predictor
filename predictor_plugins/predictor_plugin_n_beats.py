#!/usr/bin/env python
"""Hybrid N-BEATS + Bayesian multi-head predictor (bi-LSTM + DenseFlipout per horizon).

Replaces the shared CNN conv trunk with an N-BEATS style fully-connected block stack
that preserves temporal resolution through learned projections, then feeds a
shared temporal feature map into per-horizon BiLSTM + Bayesian heads.

Improvements:
- Added LayerNormalization and Dropout for better stability and regularization.
- Support for 'swish' activation.
- Configurable trunk projection channels.
"""
from __future__ import annotations
import tensorflow as tf, tensorflow_probability as tfp
from tensorflow.keras.layers import (
    Input, Dense, Lambda, Add, Concatenate, Bidirectional, LSTM, Conv1D, 
    LayerNormalization, Dropout, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from .common.losses import mae_magnitude, composite_loss_multihead as composite_loss, composite_loss_basic, random_normal_initializer_44, composite_loss_noreturns, r2_metric
from .common.bayesian import posterior_mean_field, prior_fn
from .common.base import BaseBayesianKerasPredictor
from .common.positional_encoding import positional_encoding

class Plugin(BaseBayesianKerasPredictor):
    plugin_params = {
        # Trunk (N-BEATS style)
        "nbeats_replace": "shared",      # only 'shared' implemented
        "nbeats_blocks": 3,
        "nbeats_layers": 4,
        "nbeats_units": 256,
        "trunk_projection_channels": 64,  # if None -> input channels
        # Head sequence processing
        "head_reduction_channels": None,   # optional Conv1D 1x1 after trunk
        "bilstm_units": 128,
        # General / regularization
        "activation": "swish",
        "l2_reg": 1e-5,
        "dropout_rate": 0.1,
        # Training / optimization
        "learning_rate": 5e-4,
        "early_patience": 20,
        "batch_size": 64,
        # Bayesian / uncertainty
        "kl_weight": 1e-3,
        "kl_anneal_epochs": 25,
        "mc_samples": 50,
        "mmd_lambda": 0.1,
        "sigma_mmd": 1.0,
        # Horizons
        "predicted_horizons": [1],
        # Positional encoding
        "positional_encoding": False,
    }
    plugin_debug_vars = [
        "nbeats_replace","nbeats_blocks","nbeats_layers","nbeats_units",
        "trunk_projection_channels","head_reduction_channels","bilstm_units",
        "activation","l2_reg","learning_rate","kl_weight","kl_anneal_epochs",
        "mc_samples","mmd_lambda","sigma_mmd","predicted_horizons","positional_encoding",
        "dropout_rate"
    ]

    # Backwards compatibility mapping (old keys -> new)
    _legacy_key_map = {
        "stack_blocks": "nbeats_blocks",
        "block_layers": "nbeats_layers",
        "block_units": "nbeats_units",
    }

    def _apply_legacy_keys(self):
        for old, new in self._legacy_key_map.items():
            if old in self.params and new not in self.params:
                self.params[new] = self.params[old]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)
        self._apply_legacy_keys()

        time_steps, channels = input_shape
        ph = self.params["predicted_horizons"]
        act = self.params.get("activation", "swish")
        l2_reg_v = self.params.get("l2_reg", 1e-5)
        dropout_rate = self.params.get("dropout_rate", 0.1)
        
        blocks = self.params["nbeats_blocks"]
        layers = self.params["nbeats_layers"]
        units = self.params["nbeats_units"]
        proj_ch = self.params["trunk_projection_channels"] or channels
        head_red = self.params.get("head_reduction_channels")
        bilstm_units = self.params.get("bilstm_units", 128)
        mmd_lambda = self.params.get("mmd_lambda", 0.0)
        sigma_mmd = self.params.get("sigma_mmd", 1.0)

        inputs = Input(shape=(time_steps, channels), name="input_layer")
        if self.params.get("positional_encoding", False):
            pe = positional_encoding(time_steps, channels)
            seq_in = Lambda(lambda t, pe=pe: t + pe, name="add_positional_encoding")(inputs)
        else:
            seq_in = inputs

        mode = self.params.get("nbeats_replace", "shared")
        if mode != "shared":
            raise ValueError("Only nbeats_replace='shared' is implemented in this version.")

        # Flatten sequence for N-BEATS fully connected processing
        flat_in = Lambda(lambda t: tf.reshape(t, (-1, time_steps * channels)), name="flatten_input")(seq_in)
        residual = flat_in
        block_feature_seqs = []  # (batch, time_steps, proj_ch) per block

        for b in range(blocks):
            x_block = residual
            for l in range(layers):
                x_block = Dense(
                    units, 
                    kernel_regularizer=l2(l2_reg_v),
                    name=f"b{b}_dense{l}"
                )(x_block)
                x_block = LayerNormalization(name=f"b{b}_ln{l}")(x_block)
                x_block = Activation(act, name=f"b{b}_act{l}")(x_block)
                if dropout_rate > 0:
                    x_block = Dropout(dropout_rate, name=f"b{b}_drop{l}")(x_block)

            # Backcast (reconstruction of input)
            backcast = Dense(
                time_steps * channels, activation="linear",
                kernel_regularizer=l2(l2_reg_v), name=f"b{b}_backcast"
            )(x_block)
            
            # Update residual (remove what we've explained)
            residual = Lambda(lambda t: t[0] - t[1], name=f"b{b}_residual")([residual, backcast])

            # Forecast/Projection (features for the next stage)
            proj = Dense(
                time_steps * proj_ch, activation="linear",
                kernel_regularizer=l2(l2_reg_v), name=f"b{b}_proj"
            )(x_block)
            
            # Reshape back to sequence
            proj_seq = Lambda(
                lambda t, ts=time_steps, pc=proj_ch: tf.reshape(t, (-1, ts, pc)),
                name=f"b{b}_proj_reshape"
            )(proj)
            block_feature_seqs.append(proj_seq)

        # Combine block outputs
        if len(block_feature_seqs) == 1:
            shared_seq = block_feature_seqs[0]
        else:
            # Concatenate features from all blocks
            shared_seq = Concatenate(axis=-1, name="trunk_concat")(block_feature_seqs)

        # Optional reduction
        if head_red:
            shared_seq = Conv1D(
                filters=head_red, kernel_size=1, padding="same", activation=act,
                kernel_regularizer=l2(l2_reg_v), name="head_channel_reduction"
            )(shared_seq)

        DenseFlipout = tfp.layers.DenseFlipout
        KL_W = self.kl_weight_var
        outputs = []
        self.output_names = []

        for horizon in ph:
            suf = f"_h{horizon}"
            # BiLSTM for temporal dependencies in the extracted features
            lstm_out = Bidirectional(
                LSTM(bilstm_units, return_sequences=False, dropout=dropout_rate),
                name=f"bilstm{suf}"
            )(shared_seq)
            
            flip_name = f"flipout{suf}"
            flip_layer = DenseFlipout(
                units=1,
                activation="linear",
                kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flip_name: posterior_mean_field(dt, sh, bs, tr, nm),
                kernel_prior_fn=lambda dt, sh, bs, tr, nm=flip_name: prior_fn(dt, sh, bs, tr, nm),
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_W,
                name=flip_name,
            )
            bayes = Lambda(lambda t, fl=flip_layer: fl(t), name=f"bayes_out{suf}")(lstm_out)
            bias = Dense(1, activation="linear", kernel_initializer=random_normal_initializer_44, name=f"bias{suf}")(lstm_out)
            out = Add(name=f"output_horizon_{horizon}")([bayes, bias])
            outputs.append(out)
            self.output_names.append(f"output_horizon_{horizon}")

        self.model = Model(inputs=inputs, outputs=outputs, name=f"NBEATSHybrid_{len(ph)}H")
        optimizer = AdamW(learning_rate=self.params.get("learning_rate", 5e-4))
        loss_dict = {}
        use_returns = self.params.get("use_returns", False)
        
        if use_returns:
            def _loss_fn(y_true, y_pred):
                return composite_loss_basic(y_true, y_pred, mmd_lambda=mmd_lambda, sigma=sigma_mmd)
            for nm in self.output_names:
                loss_dict[nm] = _loss_fn
        else:
            huber = Huber()
            for nm in self.output_names:
                loss_dict[nm] = huber
                
        metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}
        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        self.model.summary(line_length=140)

if __name__ == '__main__':
    plug = Plugin({"predicted_horizons": [1,3], "positional_encoding": True})
    plug.build_model((96, 8), None, {})
    print('Outputs:', plug.output_names)
