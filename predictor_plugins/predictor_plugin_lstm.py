#!/usr/bin/env python
"""LSTM multi-horizon predictor using BaseBayesianKerasPredictor."""
from __future__ import annotations
import tensorflow as tf, tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Lambda, Bidirectional, LSTM, Add, Conv1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from .common.losses import mae_magnitude, composite_loss_multihead as composite_loss, random_normal_initializer_44
from .common.bayesian import posterior_mean_field, prior_fn
from .common.base import BaseBayesianKerasPredictor


class Plugin(BaseBayesianKerasPredictor):
    plugin_params = {
        "batch_size": 32,
        "trunk_lstm_units": 64,
        "trunk_layers": 2,
        "head_lstm_units": 32,
        "learning_rate": 1e-3,
        "activation": "relu",
        "l2_reg": 1e-4,
        "mmd_lambda": 0.1,
        "sigma_mmd": 1.0,
        "predicted_horizons": [1],
        "kl_weight": 1e-3,
        "kl_anneal_epochs": 10,
        "conv_filters": 64,
        "mc_samples": 50,
        "early_patience": 10,
    }
    plugin_debug_vars = [
        "batch_size","trunk_lstm_units","trunk_layers","head_lstm_units","learning_rate","l2_reg","mmd_lambda","sigma_mmd","predicted_horizons","conv_filters","mc_samples"
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)
        w, c = input_shape
        ph = self.params["predicted_horizons"]
        trunk_units = self.params.get("trunk_lstm_units", 64)
        trunk_layers = self.params.get("trunk_layers", 2)
        head_units = self.params.get("head_lstm_units", 32)
        l2_reg_v = self.params.get("l2_reg", 1e-4)
        act = self.params.get("activation", "relu")
        conv_filters = self.params.get("conv_filters", 64)
        inputs = Input(shape=(w, c), name="input_layer")
        x = Conv1D(filters=conv_filters, kernel_size=3, padding='same', activation=act, kernel_regularizer=l2(l2_reg_v), name='trunk_conv')(inputs)
        for i in range(trunk_layers):
            x = Bidirectional(LSTM(trunk_units, return_sequences=(i < trunk_layers - 1), kernel_regularizer=l2(l2_reg_v)), name=f"trunk_bilstm_{i+1}")(x)
        trunk_out = x
        outputs = []
        self.output_names = []
        DenseFlipout = tfp.layers.DenseFlipout
        KLW = self.kl_weight_var
        mmd_lambda = self.params.get("mmd_lambda", 0.0)
        sigma_mmd = self.params.get("sigma_mmd", 1.0)
        for horizon in ph:
            suf = f"_h{horizon}"
            h = trunk_out
            if len(h.shape) == 3:
                h = Conv1D(filters=head_units, kernel_size=3, padding='same', activation=act, kernel_regularizer=l2(l2_reg_v), name=f"head_conv{suf}")(h)
                h = Bidirectional(LSTM(head_units, return_sequences=False, kernel_regularizer=l2(l2_reg_v)), name=f"head_bilstm{suf}")(h)
            flip_name = f"flipout{suf}"
            flip_layer = DenseFlipout(
                units=1,
                activation='linear',
                kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flip_name: posterior_mean_field(dt, sh, bs, tr, nm),
                kernel_prior_fn=lambda dt, sh, bs, tr, nm=flip_name: prior_fn(dt, sh, bs, tr, nm),
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KLW,
                name=flip_name,
            )
            bayes = Lambda(lambda t, fl=flip_layer: fl(t), name=f"bayes_out{suf}")(h)
            bias = Dense(1, activation='linear', kernel_initializer=random_normal_initializer_44, name=f"bias{suf}")(h)
            final = Add(name=f"output_horizon_{horizon}")([bayes, bias])
            outputs.append(final)
            self.output_names.append(f"output_horizon_{horizon}")
        self.model = Model(inputs=inputs, outputs=outputs, name=f"LSTMPredictor_{len(ph)}H")
        optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))
        loss_dict = {}
        for i, nm in enumerate(self.output_names):
            loss_dict[nm] = (lambda idx=i: (lambda yt, yp: composite_loss(yt, yp, head_index=idx, mmd_lambda=mmd_lambda, sigma=sigma_mmd, p=0, i=0, d=0, list_last_signed_error=[], list_last_stddev=[], list_last_mmd=[], list_local_feedback=[])))()
        metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}
        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        self.model.summary(line_length=140)

if __name__ == '__main__':
    pl = Plugin({"predicted_horizons": [1,3], "plotted_horizon": 1})
    pl.build_model((24,3), None, {})
    print(pl.output_names)