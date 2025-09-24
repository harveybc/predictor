#!/usr/bin/env python
"""Clean ANN predictor plugin (multi-horizon) aligned with CNN plugin.

Single authoritative implementation. Legacy duplicates removed.
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


def _posterior_mean_field(dtype, kernel_shape, bias_size, trainable, name):
    if not isinstance(name, str):
        name = None
    bias_size = 0
    n = int(np.prod(kernel_shape)) + bias_size
    c = np.log(np.expm1(1.0))
    loc = tf.Variable(tf.random.normal([n], stddev=0.05, seed=11), dtype=dtype, trainable=trainable, name=(f"{name}_loc" if name else "posterior_loc"))
    scale = tf.Variable(tf.random.normal([n], stddev=0.05, seed=12), dtype=dtype, trainable=trainable, name=(f"{name}_scale" if name else "posterior_scale"))
    scale = 1e-3 + tf.nn.softplus(scale + c)
    scale = tf.clip_by_value(scale, 1e-3, 1.0)
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=tf.reshape(loc, kernel_shape), scale=tf.reshape(scale, kernel_shape)),
        reinterpreted_batch_ndims=len(kernel_shape),
    )


def _prior_fn(dtype, kernel_shape, bias_size, trainable, name):
    if not isinstance(name, str):
        name = None
    bias_size = 0
    n = int(np.prod(kernel_shape)) + bias_size
    loc = tf.zeros([n], dtype=dtype)
    scale = tf.ones([n], dtype=dtype)
    return tfp.distributions.Independent(
        tfp.distributions.Normal(loc=tf.reshape(loc, kernel_shape), scale=tf.reshape(scale, kernel_shape)),
        reinterpreted_batch_ndims=len(kernel_shape),
    )


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
    ]

    def __init__(self, config=None):
        self.params = self.plugin_params.copy()
        if config:
            self.params.update(config)
        ph = self.params.get("predicted_horizons", [1])
        n_out = len(ph)
        self.kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="kl_weight_var")
        # Control placeholders
        self.local_p_control = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.local_i_control = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.local_d_control = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.last_signed_error = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.last_stddev = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.last_mmd = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.local_feedback = [tf.Variable(0.0, trainable=False) for _ in range(n_out)]
        self.model: Model | None = None
        if not hasattr(tfp.layers.DenseFlipout, "_already_patched_add_variable"):
            def _patched_add_variable(layer_instance, name, shape, dtype, initializer, trainable, **kwargs):
                return layer_instance.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, **kwargs)
            tfp.layers.DenseFlipout.add_variable = _patched_add_variable
            tfp.layers.DenseFlipout._already_patched_add_variable = True

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            self.params[k] = v
    def get_debug_info(self):
        return {k: self.params.get(k) for k in self.plugin_debug_vars}
    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

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
        for horizon in ph:
            suf = f"_h{horizon}"
            head = Dense(hidden // 2, activation=act, name=f"head_dense1{suf}")(trunk)
            if dr > 0:
                head = Dropout(dr, name=f"head_dropout1{suf}")(head)
            flip_name = f"flipout{suf}"
            flip_layer = DenseFlipout(
                units=1,
                activation="linear",
                kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flip_name: _posterior_mean_field(dt, sh, bs, tr, nm),
                kernel_prior_fn=lambda dt, sh, bs, tr, nm=flip_name: _prior_fn(dt, sh, bs, tr, nm),
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
        if config:
            self.params.update(config)
        if 'predicted_horizons' not in self.params or 'plotted_horizon' not in self.params:
            raise ValueError("Config must contain 'predicted_horizons' and 'plotted_horizon'.")
        ph = self.params['predicted_horizons']
        plotted = self.params['plotted_horizon']
        if plotted not in ph:
            raise ValueError('plotted_horizon must be one of predicted_horizons')
        plotted_index = ph.index(plotted)
        class KLAnneal(tf.keras.callbacks.Callback):
            def __init__(self, plugin, target, epochs_anneal): super().__init__(); self.plugin=plugin; self.target=target; self.epochs_anneal=epochs_anneal
            def on_epoch_begin(self, epoch, logs=None): self.plugin.kl_weight_var.assign(self.target*min(1.0, (epoch+1)/self.epochs_anneal))
        callbacks = [
            EarlyStoppingWithPatienceCounter(monitor='val_loss', patience=self.params.get('early_patience', 10), restore_best_weights=True, verbose=1),
            ReduceLROnPlateauWithCounter(monitor='val_loss', factor=0.5, patience=max(1, self.params.get('early_patience', 10)//4), verbose=1),
            LambdaCallback(on_epoch_end=lambda e, l: print(f"Epoch {e+1}: LR={K.get_value(self.model.optimizer.learning_rate):.6f}")),
            KLAnneal(self, self.params.get('kl_weight', 1e-3), self.params.get('kl_anneal_epochs', 10))
        ]
        if not isinstance(y_train, dict) or not isinstance(y_val, dict):
            raise TypeError('y_train/y_val must be dicts of heads')
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks, verbose=1)
        train_preds, train_unc = self.predict_with_uncertainty(x_train, self.params.get('mc_samples', 50))
        val_preds, val_unc = self.predict_with_uncertainty(x_val, self.params.get('mc_samples', 50))
        try:
            self.calculate_mae(y_train[self.output_names[plotted_index]], train_preds[plotted_index])
            self.calculate_r2(y_train[self.output_names[plotted_index]], train_preds[plotted_index])
        except Exception as e:
            print(f'Metric calculation error: {e}')
        return history, train_preds, train_unc, val_preds, val_unc

    def predict_with_uncertainty(self, x_test, mc_samples=100):
        if self.model is None:
            raise ValueError('Model not built.')
        sample = self.model(x_test[:1], training=True)
        sample = [sample] if not isinstance(sample, list) else sample
        heads = len(sample)
        n = x_test.shape[0]
        d = sample[0].shape[-1]
        means = [np.zeros((n, d), dtype=np.float32) for _ in range(heads)]
        m2 = [np.zeros_like(means[0]) for _ in range(heads)]
        counts = [0]*heads
        for _ in tqdm(range(mc_samples), desc='MC'):
            preds = self.model(x_test, training=False)
            preds = [preds] if not isinstance(preds, list) else preds
            for h in range(heads):
                arr = preds[h].numpy()
                if arr.ndim == 1:
                    arr = np.expand_dims(arr, -1)
                counts[h] += 1
                delta = arr - means[h]
                means[h] += delta / counts[h]
                delta2 = arr - means[h]
                m2[h] += delta * delta2
        stds = []
        for h in range(heads):
            var = np.full_like(means[h], np.nan) if counts[h] < 2 else m2[h]/(counts[h]-1)
            stds.append(np.sqrt(np.maximum(var, 0)))
        return means, stds

    def save(self, file_path):
        self.model.save(file_path)
        print(f'Model saved to {file_path}')
    def load(self, file_path):
        self.model = load_model(file_path, custom_objects={
            'composite_loss': composite_loss,
            'compute_mmd': compute_mmd,
            'r2_metric': r2_metric,
            'mae_magnitude': mae_magnitude
        })
        print(f'Predictor model loaded from {file_path}')

    def calculate_mae(self, y_true, y_pred):
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            y_true = np.reshape(y_true, (-1, 1))
            y_true = np.concatenate([y_true, np.zeros_like(y_true)], axis=1)
        mag_true = y_true[:, 0:1]
        mag_pred = y_pred[:, 0:1]
        mae = float(np.mean(np.abs(mag_true.flatten() - mag_pred.flatten())))
        print(f'MAE (magnitude): {mae}')
        return mae
    def calculate_r2(self, y_true, y_pred):
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            y_true = np.reshape(y_true, (-1, 1))
            y_true = np.concatenate([y_true, np.zeros_like(y_true)], axis=1)
        mag_true = y_true[:, 0:1]
        mag_pred = y_pred[:, 0:1]
        ss_res = np.sum((mag_true - mag_pred)**2, axis=0)
        ss_tot = np.sum((mag_true - np.mean(mag_true, axis=0))**2, axis=0)
        r2 = float(np.mean(1 - (ss_res / (ss_tot + np.finfo(float).eps))))
        print(f'R2 (magnitude): {r2}')
        return r2


if __name__ == '__main__':
    plug = Plugin({"predicted_horizons": [1, 3], "plotted_horizon": 1})
    plug.build_model((24, 3), None, {})
    print('Outputs:', plug.output_names)