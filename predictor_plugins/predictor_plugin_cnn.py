#!/usr/bin/env python
"""CNN multi-horizon predictor using shared BaseBayesianKerasPredictor.

Concrete plugin now only implements build_model & parameter lists; all training,
metrics, persistence, and MC uncertainty logic are inherited.
"""
from __future__ import annotations
import tensorflow as tf, tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Lambda, Bidirectional, LSTM, Add, Conv1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from .common.losses import mae_magnitude, composite_loss_multihead as composite_loss, random_normal_initializer_44
from .common.bayesian import posterior_mean_field, prior_fn
from .common.base import BaseBayesianKerasPredictor
from .common.positional_encoding import positional_encoding


class Plugin(BaseBayesianKerasPredictor):
    plugin_params = {
        "batch_size": 32,
        "branch_units": 64,
        "merged_units": 128,
        "learning_rate": 0.001,
        "activation": "relu",
        "l2_reg": 1e-4,
        "mmd_lambda": 0.1,
        "sigma_mmd": 1.0,
        "predicted_horizons": [1],
        "kl_weight": 1e-3,
        "kl_anneal_epochs": 10,
        "early_patience": 10,
        "mc_samples": 50,
    "positional_encoding": False,
    }
    plugin_debug_vars = [
        "batch_size","branch_units","merged_units","learning_rate","l2_reg","mmd_lambda","sigma_mmd","predicted_horizons","kl_weight","early_patience","mc_samples","positional_encoding"
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)
        w, c = input_shape
        ph = self.params["predicted_horizons"]
        act = self.params.get("activation", "relu")
        l2_reg_v = self.params.get("l2_reg", 1e-4)
        merged_units = self.params.get("merged_units", 128)
        branch_units = self.params.get("branch_units", 64)
        lstm_units = max(8, branch_units // 2)
        inputs = Input(shape=(w, c), name="input_layer")
        # Optional positional encoding
        if self.params.get("positional_encoding", False):
            pe = positional_encoding(w, c)
            x_in = Lambda(lambda t, pe=pe: t + pe, name="add_positional_encoding")(inputs)
        else:
            x_in = inputs
        x = Conv1D(filters=merged_units, kernel_size=3, strides=2, padding="same", activation=act, kernel_regularizer=l2(l2_reg_v), name="conv_1")(x_in)
        x = Conv1D(filters=branch_units, kernel_size=3, strides=2, padding="same", activation=act, kernel_regularizer=l2(l2_reg_v), name="conv_2")(x)
        merged = x
        outputs = []
        self.output_names = []
        DenseFlipout = tfp.layers.DenseFlipout
        KL_WEIGHT = self.kl_weight_var
        mmd_lambda = self.params.get("mmd_lambda", 0.0)
        sigma_mmd = self.params.get("sigma_mmd", 1.0)
        for horizon in ph:
            suf = f"_h{horizon}"
            h_in = Conv1D(filters=branch_units, kernel_size=3, strides=2, padding="valid", activation=act, kernel_regularizer=l2(l2_reg_v), name=f"head_conv1{suf}")(merged)
            h_in = Conv1D(filters=lstm_units, kernel_size=3, strides=2, padding="valid", activation=act, kernel_regularizer=l2(l2_reg_v), name=f"head_conv2{suf}")(h_in)
            lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=False), name=f"bilstm{suf}")(h_in)
            flip_name = f"flipout{suf}"
            flip_layer = DenseFlipout(
                units=1,
                activation="linear",
                kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flip_name: posterior_mean_field(dt, sh, bs, tr, nm),
                kernel_prior_fn=lambda dt, sh, bs, tr, nm=flip_name: prior_fn(dt, sh, bs, tr, nm),
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT,
                name=flip_name,
            )
            bayes = Lambda(lambda t, fl=flip_layer: fl(t), name=f"bayes_out{suf}")(lstm_out)
            bias = Dense(1, activation="linear", kernel_initializer=random_normal_initializer_44, name=f"bias{suf}")(lstm_out)
            out = Add(name=f"output_horizon_{horizon}")([bayes, bias])
            outputs.append(out)
            self.output_names.append(f"output_horizon_{horizon}")
        self.model = Model(inputs=inputs, outputs=outputs, name=f"CNNPredictor_{len(ph)}H")
        optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))
        loss_dict = {}
        for i, nm in enumerate(self.output_names):
            loss_dict[nm] = (lambda idx=i: (lambda yt, yp: composite_loss(yt, yp, head_index=idx, mmd_lambda=mmd_lambda, sigma=sigma_mmd, p=0, i=0, d=0, list_last_signed_error=[], list_last_stddev=[], list_last_mmd=[], list_local_feedback=[])))()
        metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}
        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        self.model.summary(line_length=140)

if __name__ == "__main__":  # Minimal sanity check
    plugin = Plugin()
    plugin.build_model((24, 3), None, {"predicted_horizons": [1]})
    print(plugin.output_names)