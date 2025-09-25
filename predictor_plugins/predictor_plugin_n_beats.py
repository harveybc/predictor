#!/usr/bin/env python
"""Deterministic N-BEATS style predictor using shared base plus optional positional encoding."""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from .common.losses import mae_magnitude
from .common.base import BaseDeterministicKerasPredictor
from .common.positional_encoding import positional_encoding


class Plugin(BaseDeterministicKerasPredictor):
    plugin_params = {
        "stack_blocks": 3,
        "block_layers": 2,
        "block_units": 256,
        "activation": "relu",
        "learning_rate": 1e-3,
        "predicted_horizons": [1],
        "batch_size": 32,
        "early_patience": 10,
        "mc_samples": 10,
    "positional_encoding": False,
    }
    plugin_debug_vars = [
        "stack_blocks","block_layers","block_units","activation","learning_rate","predicted_horizons","batch_size","early_patience","mc_samples","positional_encoding"
    ]

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)
        time_steps, channels = input_shape
        ph = self.params['predicted_horizons']
        act = self.params['activation']
        blocks = self.params['stack_blocks']
        layers = self.params['block_layers']
        units = self.params['block_units']
        inp = Input(shape=(time_steps, channels), name='input_layer')
        if self.params.get("positional_encoding", False):
            pe = positional_encoding(time_steps, channels)
            inp_pe = Lambda(lambda t, pe=pe: t + pe, name="add_positional_encoding")(inp)
        else:
            inp_pe = inp
        # Flatten input (Keras native) for fully-connected blocks
        flat = Flatten(name="flatten_inputs")(inp_pe)
        residual = flat  # initial residual (backcast target)
        # Collect per-horizon forecast components from each block
        per_horizon_components = {h: [] for h in ph}

        for b in range(blocks):
            x_block = residual
            for l in range(layers):
                x_block = Dense(units, activation=act, name=f"b{b}_dense{l}")(x_block)
            # Produce backcast to update residual
            backcast = Dense(time_steps * channels, activation='linear', name=f"b{b}_backcast")(x_block)
            # Subtract backcast (reshape back to same flat size already)
            residual = Lambda(lambda t: t[0] - t[1], name=f"b{b}_residual")([residual, backcast])
            # Forecast components per horizon
            for h in ph:
                comp = Dense(1, activation='linear', name=f"b{b}_h{h}_forecast")(x_block)
                per_horizon_components[h].append(comp)

        outputs = []
        self.output_names = []
        for h in ph:
            # Sum all components for this horizon
            if len(per_horizon_components[h]) == 1:
                agg = per_horizon_components[h][0]
            else:
                agg = Add(name=f"agg_h{h}")(per_horizon_components[h])
            # Identity Lambda just to keep naming convention consistent
            out = Lambda(lambda t: t, name=f"output_horizon_{h}")(agg)
            outputs.append(out)
            self.output_names.append(f"output_horizon_{h}")
        self.model = Model(inputs=inp, outputs=outputs, name="NBEATSPredictor")
        opt = AdamW(learning_rate=self.params['learning_rate'])
        loss_dict = {nm: 'huber' for nm in self.output_names}
        metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}
        self.model.compile(optimizer=opt, loss=loss_dict, metrics=metrics_dict)
        self.model.summary(line_length=140)

if __name__ == '__main__':
    plug = Plugin({"predicted_horizons": [1,3], "plotted_horizon": 1, "positional_encoding": True})
    plug.build_model((24,4), None, {})
    print('Outputs:', plug.output_names)
