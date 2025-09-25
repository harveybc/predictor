#!/usr/bin/env python
"""Clean deterministic multi-horizon N-BEATS style predictor plugin.

Simplified residual stacking -> aggregated forecast -> per-horizon linear heads.
Outputs: one Dense head per horizon named output_horizon_<H> with shape (batch,1).
Uncertainty: zeros placeholder (future MC dropout possible). Uses shared composite_loss_multihead.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LambdaCallback
import tensorflow.keras.backend as K
from .common.losses import mae_magnitude, r2_metric, composite_loss_multihead as composite_loss, compute_mmd
from .common.callbacks import ReduceLROnPlateauWithCounter, EarlyStoppingWithPatienceCounter

class Plugin:
    plugin_params={"batch_size":32,"learning_rate":1e-3,"activation":"relu","l2_reg":1e-4,
                   "mmd_lambda":0.0,"sigma_mmd":1.0,"nbeats_num_blocks":3,"nbeats_units":128,
                   "nbeats_layers":3,"predicted_horizons":[1],"early_patience":10,"mc_samples":10}
    plugin_debug_vars=["batch_size","learning_rate","l2_reg","mmd_lambda","sigma_mmd","nbeats_num_blocks","nbeats_units","nbeats_layers","predicted_horizons"]
    def __init__(self, config=None):
        self.params=self.plugin_params.copy()
        if config: self.params.update(config)
        self.model=None; self.output_names=[]
    def set_params(self, **kwargs):
        for k,v in kwargs.items(): self.params[k]=v
    def get_debug_info(self):
        return {k:self.params.get(k) for k in self.plugin_debug_vars}
    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())
    def _block(self, x, units, layers, b, act):
        r=x
        for i in range(layers):
            r=Dense(units, activation=act, name=f"block{b}_dense{i+1}")(r)
        forecast=Dense(1, activation="linear", name=f"block{b}_forecast")(r)
        back=Dense(x.shape[-1], activation="linear", name=f"block{b}_backcast")(r)
        updated=Lambda(lambda t: t[0]-t[1], name=f"block{b}_residual")([x, back])
        return updated, forecast
    def build_model(self, input_shape, x_train, config):
        if config: self.params.update(config)
        ph=self.params["predicted_horizons"]; B=self.params.get("nbeats_num_blocks",3)
        U=self.params.get("nbeats_units",128); L=self.params.get("nbeats_layers",3)
        act=self.params.get("activation","relu"); l2_reg_v=self.params.get("l2_reg",1e-4)
        inputs=Input(shape=input_shape, name="input_layer")
        flat=Lambda(lambda t: tf.reshape(t,[tf.shape(t)[0],-1]), name="flatten")(inputs)
        residual=flat; forecasts=[]
        for b in range(1,B+1):
            residual, fc=self._block(residual,U,L,b,act); forecasts.append(fc)
        agg = Lambda(lambda t: tf.reduce_sum(tf.stack(t,axis=0),axis=0), name="forecast_sum")(forecasts) if len(forecasts)>1 else forecasts[0]
        outputs=[]; self.output_names=[]
        for h in ph:
            o=Dense(1, activation="linear", kernel_regularizer=l2(l2_reg_v), name=f"output_horizon_{h}")(agg)
            outputs.append(o); self.output_names.append(f"output_horizon_{h}")
        self.model=Model(inputs=inputs, outputs=outputs, name=f"NBEATSPredictor_{len(ph)}H")
        opt=Adam(learning_rate=self.params.get("learning_rate",1e-3))
        mmd_lambda=self.params.get("mmd_lambda",0.0); sigma=self.params.get("sigma_mmd",1.0)
        loss_dict={}
        #!/usr/bin/env python
        """Deterministic N-BEATS style predictor using shared base classes.

        Only build_model is implemented here; training, saving, metrics, and uncertainty
        handling are provided by BaseDeterministicKerasPredictor.
        """
        from __future__ import annotations
        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Input, Dense, Lambda
        from tensorflow.keras.optimizers import AdamW
        from .common.losses import mae_magnitude
        from .common.base import BaseDeterministicKerasPredictor


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
            }
            plugin_debug_vars = [
                "stack_blocks","block_layers","block_units","activation","learning_rate","predicted_horizons","batch_size","early_patience","mc_samples"
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
                flat = tf.reshape(inp, (-1, time_steps * channels))
                residual = flat
                forecast_heads = {h: [] for h in ph}
                for b in range(blocks):
                    x = residual
                    for l in range(layers):
                        x = Dense(units, activation=act, name=f"b{b}_dense{l}")(x)
                    theta_f = Dense(units, activation=act, name=f"b{b}_theta_f")(x)
                    backcast = Dense(time_steps * channels, activation='linear', name=f"b{b}_backcast")(theta_f)
                    residual = tf.keras.layers.subtract([residual, backcast], name=f"b{b}_residual")
                    for h in ph:
                        fh = Dense(1, activation='linear', name=f"b{b}_h{h}_forecast")(theta_f)
                        forecast_heads[h].append(fh)
                outputs = []
                self.output_names = []
                for h in ph:
                    agg = Lambda(lambda tensors: tf.add_n(tensors), name=f"agg_h{h}")(forecast_heads[h])
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
            plug = Plugin({"predicted_horizons": [1,3], "plotted_horizon": 1})
            plug.build_model((24,4), None, {})
            print('Outputs:', plug.output_names)
