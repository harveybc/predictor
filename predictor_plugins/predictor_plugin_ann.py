#!/usr/bin/env python
"""ANN multi-horizon predictor plugin (clean, unified interface).

Single authoritative implementation. Multi-head outputs named
``output_horizon_<H>`` each shape (batch,1). Bayesian Flipout + bias per head.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import LambdaCallback
import tensorflow.keras.backend as K
from tqdm import tqdm

from .common.losses import (
    mae_magnitude, r2_metric, composite_loss_multihead as composite_loss,
    random_normal_initializer_44, compute_mmd,
)
from .common.callbacks import (
    ReduceLROnPlateauWithCounter, EarlyStoppingWithPatienceCounter,
)
from .common.bayesian import posterior_mean_field, prior_fn, build_kl_anneal_callback, predict_mc_welford


"""Posterior/prior factories now imported from common.bayesian (posterior_mean_field, prior_fn)."""


class Plugin:
    plugin_params = {
        "batch_size": 32,
        "hidden_units": 256,
        "num_hidden_layers": 2,
        "dropout_rate": 0.1,
        "activation": "relu",
        "learning_rate": 0.001,
        "mmd_lambda": 0.1,
        "sigma_mmd": 1.0,
        "predicted_horizons": [1],
        "early_patience": 10,
        "kl_weight": 1e-3,
        "kl_anneal_epochs": 10,
    "mc_samples": 50,
    }
    plugin_debug_vars = [
        "batch_size",
        "hidden_units",
        "num_hidden_layers",
        "dropout_rate",
        "learning_rate",
        "mmd_lambda",
        "sigma_mmd",
        "predicted_horizons",
        "early_patience",
        "kl_weight",
    "mc_samples",
    ]

    def __init__(self, config=None):
        self.params = self.plugin_params.copy()
        if config:
            self.params.update(config)
        ph = self.params.get("predicted_horizons", [1])
        n_out = len(ph)
        self.kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="kl_weight_var")
        self.local_p_control = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.local_i_control = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.local_d_control = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.last_signed_error = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.last_stddev = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.last_mmd = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.local_feedback = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.model = None
        if not hasattr(tfp.layers.DenseFlipout, "_already_patched_add_variable"):
            def _patched_add_variable(layer_instance, name, shape, dtype, initializer, trainable, **kwargs):
                return layer_instance.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, **kwargs)
            tfp.layers.DenseFlipout.add_variable = _patched_add_variable
            tfp.layers.DenseFlipout._already_patched_add_variable = True

    def set_params(self, **kwargs):
        """Update runtime hyper-parameters (in-place)."""
        for k, v in kwargs.items():
            self.params[k] = v
    def get_debug_info(self):
        return {k: self.params.get(k) for k in self.plugin_debug_vars}
    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    def build_model(self, input_shape, x_train, config):
        """Build (or rebuild) the Keras model.

        Args:
            input_shape: Tuple (window, channels).
            x_train: (unused placeholder for interface symmetry).
            config: Optional dict overriding parameters.
        """
        if config:
            self.params.update(config)
        window, channels = input_shape
        ph = self.params["predicted_horizons"]
        act = self.params.get("activation", "relu")
        hidden = self.params.get("hidden_units", 256)
        n_layers = self.params.get("num_hidden_layers", 2)
        dr = self.params.get("dropout_rate", 0.0)
        inputs = Input(shape=(window, channels), name="input_layer")
        x = tf.reshape(inputs, (-1, window * channels))
        for i in range(n_layers):
            x = Dense(hidden, activation=act, name=f"shared_dense_{i}")(x)
            if dr > 0:
                x = Dropout(dr, name=f"shared_dropout_{i}")(x)
        trunk = x
        DenseFlipout = tfp.layers.DenseFlipout
        KLW = self.kl_weight_var
        outputs = []
        self.output_names = []
        for horizon in ph:
            suf = f"_h{horizon}"
            head = Dense(hidden // 2, activation=act, name=f"head_dense1{suf}")(trunk)
            if dr > 0:
                head = Dropout(dr, name=f"head_dropout1{suf}")(head)
            flip_name = f"flipout{suf}"
            flip_layer = DenseFlipout(
                units=1,
                activation="linear",
                kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flip_name: posterior_mean_field(dt, sh, bs, tr, nm),
                kernel_prior_fn=lambda dt, sh, bs, tr, nm=flip_name: prior_fn(dt, sh, bs, tr, nm),
                kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KLW,
                name=flip_name,
            )
            bayes = Lambda(lambda t, fl=flip_layer: fl(t), name=f"bayes_out{suf}")(head)
            bias = Dense(1, activation="linear", kernel_initializer=random_normal_initializer_44, name=f"bias{suf}")(head)
            out = Add(name=f"output_horizon_{horizon}")([bayes, bias])
            outputs.append(out)
            self.output_names.append(f"output_horizon_{horizon}")
        self.model = Model(inputs=inputs, outputs=outputs, name=f"ANNPredictor_{len(ph)}H")
        optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))
        mmd_lambda = self.params.get("mmd_lambda", 0.0)
        sigma_mmd = self.params.get("sigma_mmd", 1.0)
        loss_dict = {}
        for i, nm in enumerate(self.output_names):
            loss_dict[nm] = (lambda idx=i: (lambda yt, yp: composite_loss(yt, yp, head_index=idx, mmd_lambda=mmd_lambda, sigma=sigma_mmd, p=0, i=0, d=0, list_last_signed_error=[], list_last_stddev=[], list_last_mmd=[], list_local_feedback=[])))()
        metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}
        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        self.model.summary(line_length=140)

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        """Train the model.

        Expects y_* as dict mapping output layer names to arrays matching horizon order.
        Returns (history, train_means, train_stds, val_means, val_stds).
        """
        if config:
            self.params.update(config)
        if 'predicted_horizons' not in self.params or 'plotted_horizon' not in self.params:
            raise ValueError("Config must contain 'predicted_horizons' and 'plotted_horizon'.")
        ph = self.params['predicted_horizons']
        plotted = self.params['plotted_horizon']
        if plotted not in ph:
            raise ValueError('plotted_horizon must be one of predicted_horizons')
        plotted_index = ph.index(plotted)
        # KL annealing via shared callback util
        callbacks = [
            EarlyStoppingWithPatienceCounter(monitor='val_loss', patience=self.params.get('early_patience', 10), restore_best_weights=True, verbose=1),
            ReduceLROnPlateauWithCounter(monitor='val_loss', factor=0.5, patience=max(1, self.params.get('early_patience', 10)//4), verbose=1),
            LambdaCallback(on_epoch_end=lambda e, l: print(f"Epoch {e+1}: LR={K.get_value(self.model.optimizer.learning_rate):.6f}")),
            build_kl_anneal_callback(self, self.params.get('kl_weight', 1e-3), self.params.get('kl_anneal_epochs', 10))
        ]
        #!/usr/bin/env python
        """ANN multi-horizon predictor using shared base + bayesian utilities."""
        from __future__ import annotations
        import tensorflow as tf, tensorflow_probability as tfp
        from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Add
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import AdamW
        from .common.losses import mae_magnitude, composite_loss_multihead as composite_loss, random_normal_initializer_44
        from .common.bayesian import posterior_mean_field, prior_fn
        from .common.base import BaseBayesianKerasPredictor


        class Plugin(BaseBayesianKerasPredictor):
            plugin_params = {
                "batch_size": 32,
                "hidden_units": 256,
                "num_hidden_layers": 2,
                "dropout_rate": 0.1,
                "activation": "relu",
                "learning_rate": 0.001,
                "mmd_lambda": 0.1,
                "sigma_mmd": 1.0,
                "predicted_horizons": [1],
                "early_patience": 10,
                "kl_weight": 1e-3,
                "kl_anneal_epochs": 10,
                "mc_samples": 50,
            }
            plugin_debug_vars = [
                "batch_size","hidden_units","num_hidden_layers","dropout_rate","learning_rate","mmd_lambda","sigma_mmd","predicted_horizons","early_patience","kl_weight","mc_samples"
            ]

            def build_model(self, input_shape, x_train, config):
                if config:
                    self.params.update(config)
                window, channels = input_shape
                ph = self.params["predicted_horizons"]
                act = self.params.get("activation", "relu")
                hidden = self.params.get("hidden_units", 256)
                n_layers = self.params.get("num_hidden_layers", 2)
                dr = self.params.get("dropout_rate", 0.0)
                inputs = Input(shape=(window, channels), name="input_layer")
                x = tf.reshape(inputs, (-1, window * channels))
                for i in range(n_layers):
                    x = Dense(hidden, activation=act, name=f"shared_dense_{i}")(x)
                    if dr > 0:
                        x = Dropout(dr, name=f"shared_dropout_{i}")(x)
                trunk = x
                DenseFlipout = tfp.layers.DenseFlipout
                KLW = self.kl_weight_var
                outputs = []
                self.output_names = []
                mmd_lambda = self.params.get("mmd_lambda", 0.0)
                sigma_mmd = self.params.get("sigma_mmd", 1.0)
                for horizon in ph:
                    suf = f"_h{horizon}"
                    head = Dense(hidden // 2, activation=act, name=f"head_dense1{suf}")(trunk)
                    if dr > 0:
                        head = Dropout(dr, name=f"head_dropout1{suf}")(head)
                    flip_name = f"flipout{suf}"
                    flip_layer = DenseFlipout(
                        units=1,
                        activation="linear",
                        kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flip_name: posterior_mean_field(dt, sh, bs, tr, nm),
                        kernel_prior_fn=lambda dt, sh, bs, tr, nm=flip_name: prior_fn(dt, sh, bs, tr, nm),
                        kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KLW,
                        name=flip_name,
                    )
                    bayes = Lambda(lambda t, fl=flip_layer: fl(t), name=f"bayes_out{suf}")(head)
                    bias = Dense(1, activation="linear", kernel_initializer=random_normal_initializer_44, name=f"bias{suf}")(head)
                    out = Add(name=f"output_horizon_{horizon}")([bayes, bias])
                    outputs.append(out)
                    self.output_names.append(f"output_horizon_{horizon}")
                self.model = Model(inputs=inputs, outputs=outputs, name=f"ANNPredictor_{len(ph)}H")
                optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))
                loss_dict = {}
                for i, nm in enumerate(self.output_names):
                    loss_dict[nm] = (lambda idx=i: (lambda yt, yp: composite_loss(yt, yp, head_index=idx, mmd_lambda=mmd_lambda, sigma=sigma_mmd, p=0, i=0, d=0, list_last_signed_error=[], list_last_stddev=[], list_last_mmd=[], list_local_feedback=[])))()
                metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}
                self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
                self.model.summary(line_length=140)

        if __name__ == '__main__':
            plug = Plugin({"predicted_horizons": [1, 3], "plotted_horizon": 1})
            plug.build_model((24, 3), None, {})
            print('Outputs:', plug.output_names)