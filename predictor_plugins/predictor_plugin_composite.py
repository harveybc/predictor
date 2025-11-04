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
from .common.losses import mae_magnitude, composite_loss_multihead as composite_loss, random_normal_initializer_44, composite_loss_noreturns, r2_metric
from .common.bayesian import posterior_mean_field, prior_fn
from .common.base import BaseBayesianKerasPredictor
from .common.positional_encoding import positional_encoding


class Plugin(BaseBayesianKerasPredictor):
    plugin_params = {
        "batch_size": 32,
        "branch_units": 32,
        "merged_units": 128,
        "learning_rate": 0.001,
        "activation": "relu",
        "l2_reg": 1e-7,
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
        """
        Multi-branch model creation with two explicit modes controlled by config:
        - close_window_only = True:
            * Branch A (temporal): LSTM over full window of CLOSE (shape: (w,1)).
            * Branch B15/B30 (HF): if process_HF, take last 16 cols at last row (t):
                - first 8 => 15m minisquence (8x1) -> LSTM(16)
                - next 8  => 30m minisquence (8x1) -> LSTM(16)
            * Branch C (point): all other non-HF columns from last row only (vector) -> 1x1 Conv1D + GAP (acts like a Dense).
            * Merge: concat(A(64), B15(16), B30(16), C(32)) -> fusion MLP -> Bayesian heads (unchanged).
        - close_window_only = False:
            * Build your prior Conv1D backbone as before (unchanged path).
        Notes:
        - CLOSE default index = 3 (OPEN=0, HIGH=1, LOW=2, CLOSE=3) since DATE_TIME is not in the window tensor.
        - Latest tick is at row index -1 in each window (baseline time t). See sliding_windows.py for details.
        """
        # --------------------------- merge runtime config ---------------------------
        if config:
            self.params.update(config)

        # --------------------------- unpack common params ---------------------------
        w, c = input_shape                                   # window length and channel count
        ph = self.params["predicted_horizons"]               # list of horizons
        act = self.params.get("activation", "relu")          # nonlinearity
        l2_reg_v = self.params.get("l2_reg", 1e-7)           # L2
        use_returns = self.params.get("use_returns", False)  # loss variant
        use_pe = self.params.get("positional_encoding", False)

        # Flags for process each data type in its own branch, close_window, and process hf
        close_window_only = bool(self.params.get("close_window_only", False))
        process_HF        = bool(self.params.get("process_HF", False))

        # Default CLOSE channel index (0:OPEN,1:HIGH,2:LOW,3:CLOSE)
        close_idx = int(self.params.get("close_channel", 3))
        if not (0 <= close_idx < c):
            raise ValueError(f"Invalid close_channel={close_idx} for c={c}")

        # --------------------------- single Keras Input -----------------------------
        inputs = Input(shape=(w, c), name="input_layer")

        # Optional positional encoding over the window tensor (if you use it elsewhere)
        if use_pe:
            pe = positional_encoding(w, c)                   # (w, c) constant PE
            x_in = Lambda(lambda t, pe=pe: t + pe, name="add_positional_encoding")(inputs)
        else:
            x_in = inputs

        # ========================================================================== #
        # MODE 1: close_window_only == True  ->  Multi-branch per your new spec     #
        # ========================================================================== #
        if close_window_only:
            # ----------------------- Branch A: CLOSE full window -------------------
            # Slice the full CLOSE sequence: shape = (batch, w, 1)
            A_seq = Lambda(lambda t, ch=close_idx: t[:, :, ch:ch+1],
                        name="A_slice_close_full")(x_in)

            # Temporal encoder over CLOSE (choose LSTM as requested)
            # BiLSTM(32) => vector(64) keeps parsimony yet capacity
            A_repr = Bidirectional(
                LSTM(32, return_sequences=False),
                name="A_bilstm_close"
            )(A_seq)  # (batch, 64)

            # -------------------- Extract latest tick row (time t) -----------------
            # As per sliding_windows.py, the most recent row is the LAST one: index -1
            # (baseline time t). This is crucial for building the point/HF branches.
            last_row = Lambda(lambda t: t[:, -1, :], name="last_row_t")(x_in)  # (batch, c)

            # ----------------------- Discover HF columns if requested --------------
            # If process_HF is True, we must pull the last 16 columns as:
            #   [CLOSE_15m_tick_1 ... CLOSE_15m_tick_8, CLOSE_30m_tick_1 ... CLOSE_30m_tick_8]
            # across all datasets that have HF appended at the end.
            hf15_idx, hf30_idx = [], []
            if process_HF:
                if c < 16:
                    raise ValueError("process_HF=True but there are fewer than 16 columns.")
                hf_base = c - 16                               # starting index of the HF block
                hf15_idx = list(range(hf_base, hf_base + 8))   # first 8 = 15m ticks
                hf30_idx = list(range(hf_base + 8, c))         # next  8 = 30m ticks

            # ----------------------- Branch B15: HF 15m minisquence ----------------
            # We gather the 8 values for 15m from the last row only and treat them as (8,1) temporal mini-sequence.
            if hf15_idx:
                B15_vec = Lambda(lambda r, idx=hf15_idx: tf.gather(r, indices=idx, axis=1),
                                name="B15_gather_lastrow")(last_row)            # (batch, 8)
                B15_seq = Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                name="B15_as_seq")(B15_vec)                      # (batch, 8, 1)
                B15_repr = LSTM(16, return_sequences=False, name="B15_lstm")(B15_seq)  # (batch, 16)
            else:
                # If not present, supply zeros to keep merge dimensions consistent
                B15_repr = Lambda(lambda r: tf.zeros_like(r[:, :16]),
                                name="B15_dummy")(last_row)                     # (batch, 16)

            # ----------------------- Branch B30: HF 30m minisquence ----------------
            if hf30_idx:
                B30_vec = Lambda(lambda r, idx=hf30_idx: tf.gather(r, indices=idx, axis=1),
                                name="B30_gather_lastrow")(last_row)            # (batch, 8)
                B30_seq = Lambda(lambda x: tf.expand_dims(x, axis=-1),
                                name="B30_as_seq")(B30_vec)                      # (batch, 8, 1)
                B30_repr = LSTM(16, return_sequences=False, name="B30_lstm")(B30_seq)  # (batch, 16)
            else:
                B30_repr = Lambda(lambda r: tf.zeros_like(r[:, :16]),
                                name="B30_dummy")(last_row)                     # (batch, 16)

            # ----------------------- Branch C: point features at t -----------------
            # Point features = all columns EXCEPT CLOSE and (optionally) HF columns.
            # We take ONLY the latest tick (last row) values and push them through a compact block.
            def _point_indices(total_c, close_i, hf15, hf30):
                hfset = set(hf15) | set(hf30)
                return [j for j in range(total_c) if j != close_i and j not in hfset]

            point_idx = _point_indices(c, close_idx, hf15_idx, hf30_idx)

            if len(point_idx) > 0:
                C_vec = Lambda(lambda r, idx=point_idx: tf.gather(r, indices=idx, axis=1),
                            name="C_point_gather")(last_row)                   # (batch, M)
                # Use a 1x1 Conv1D over the feature axis to respect your "conv1d model" wording.
                # Reshape to (batch, M, 1) -> Conv1D(32,k=1) -> GlobalAverage over "M" -> Dense(32)
                C_seq = Lambda(lambda x: tf.expand_dims(x, axis=-1),
                            name="C_point_expand")(C_vec)                      # (batch, M, 1)
                C_tmp = Conv1D(32, kernel_size=1, padding="valid",
                            activation=act, kernel_regularizer=l2(l2_reg_v),
                            name="C_point_conv1x1")(C_seq)                     # (batch, M, 32)
                C_gav = Lambda(lambda t: tf.reduce_mean(t, axis=1),
                            name="C_point_gavg")(C_tmp)                        # (batch, 32)
                C_repr = C_gav                                                   # (batch, 32)
            else:
                C_repr = Lambda(lambda r: tf.zeros_like(r[:, :32]),
                                name="C_point_dummy")(last_row)                   # (batch, 32)

            # ----------------------- Merge branches (64 + 16 + 16 + 32 = 128) ------
            merged = Lambda(lambda xs: tf.concat(xs, axis=-1),
                            name="merge_concat")([A_repr, B15_repr, B30_repr, C_repr])  # (batch, 128)

            # Fusion MLP: 128 -> 128 -> 64
            F_h1 = Dense(128, activation=act, kernel_regularizer=l2(l2_reg_v), name="F_dense1")(merged)
            F_h1 = Lambda(lambda t: tf.nn.dropout(t, rate=0.2), name="F_dropout")(F_h1)
            F_h2 = Dense(64, activation=act, kernel_regularizer=l2(l2_reg_v), name="F_dense2")(F_h1)

            # ----------------------- Bayesian heads (unchanged semantics) ----------
            DenseFlipout = tfp.layers.DenseFlipout
            KL_WEIGHT = self.kl_weight_var
            mmd_lambda = self.params.get("mmd_lambda", 0.0)
            sigma_mmd  = self.params.get("sigma_mmd", 1.0)

            outputs = []
            self.output_names = []
            for horizon in ph:
                suf = f"_h{horizon}"

                # Light shaping per head
                H = Dense(32, activation=act, kernel_regularizer=l2(l2_reg_v), name=f"H_dense{suf}")(F_h2)

                flip_name = f"flipout{suf}"
                flip_layer = DenseFlipout(
                    units=1,
                    activation="linear",
                    kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flip_name: posterior_mean_field(dt, sh, bs, tr, nm),
                    kernel_prior_fn=lambda dt, sh, bs, tr, nm=flip_name: prior_fn(dt, sh, bs, tr, nm),
                    kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * KL_WEIGHT,
                    name=flip_name,
                )
                bayes = Lambda(lambda t, fl=flip_layer: fl(t), name=f"bayes_out{suf}")(H)
                bias  = Dense(1, activation="linear", kernel_initializer=random_normal_initializer_44, name=f"bias{suf}")(H)
                out   = Add(name=f"output_horizon_{horizon}")([bayes, bias])

                outputs.append(out)
                self.output_names.append(f"output_horizon_{horizon}")

            # Build & compile model (identical compile path to your original)
            self.model = Model(inputs=inputs, outputs=outputs, name=f"CNN3Branch_{len(ph)}H")
            optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))

            loss_dict = {}
            if use_returns:
                for i, nm in enumerate(self.output_names):
                    loss_dict[nm] = (lambda idx=i: (lambda yt, yp: composite_loss(
                        yt, yp, head_index=idx, mmd_lambda=mmd_lambda, sigma=sigma_mmd,
                        p=0, i=0, d=0, list_last_signed_error=[], list_last_stddev=[],
                        list_last_mmd=[], list_local_feedback=[])))()
            else:
                for i, nm in enumerate(self.output_names):
                    loss_dict[nm] = (lambda idx=i: (lambda yt, yp: composite_loss_noreturns(
                        yt, yp, head_index=idx, mmd_lambda=mmd_lambda, sigma=sigma_mmd,
                        p=0, i=0, d=0, list_last_signed_error=[], list_last_stddev=[],
                        list_last_mmd=[], list_local_feedback=[])))()
            metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}
            self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
            self.model.summary(line_length=140)
            return  # Done with mode 1

        # ========================================================================== #
        # MODE 2: close_window_only == False  ->  Keep your previous backbone        #
        # ========================================================================== #
        # (This is your original Conv1D + per-head Conv1D + BiLSTM build; unchanged.)
        # ---------------------------- original params ------------------------------
        initial_layer_size = self.params.get("initial_layer_size", 128)
        layer_size_divisor = self.params.get("layer_size_divisor", 2)
        intermediate_layers = int(self.params.get("intermediate_layers", 2))
        head_layers = int(self.params.get("head_layers", 2))

        x = x_in
        num_layers = max(1, intermediate_layers)
        sizes = [initial_layer_size] + [
            max(8, initial_layer_size // (layer_size_divisor ** i)) for i in range(1, num_layers)
        ]

        for i, filters_i in enumerate(sizes):
            x = Conv1D(
                filters=filters_i,
                kernel_size=3,
                strides=2,
                padding="same",
                activation=act,
                kernel_regularizer=l2(l2_reg_v),
                name=f"conv_{i+1}",
            )(x)

        last_root_filters = sizes[-1]
        merged = x
        outputs = []
        self.output_names = []
        DenseFlipout = tfp.layers.DenseFlipout
        KL_WEIGHT = self.kl_weight_var
        mmd_lambda = self.params.get("mmd_lambda", 0.0)
        sigma_mmd = self.params.get("sigma_mmd", 1.0)
        for horizon in ph:
            suf = f"_h{horizon}"
            head_num_layers = max(1, head_layers)
            base_head_filters = max(8, last_root_filters // 2)
            head_sizes = [base_head_filters] + [
                max(8, base_head_filters // (layer_size_divisor ** i)) for i in range(1, head_num_layers)
            ]
            h_in = merged
            for j, f_j in enumerate(head_sizes):
                h_in = Conv1D(
                    filters=f_j,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation=act,
                    kernel_regularizer=l2(l2_reg_v),
                    name=f"head_conv{j+1}{suf}",
                )(h_in)
            last_head_filters = head_sizes[-1]
            lstm_total_units = max(8, last_head_filters // 2)
            lstm_out = Bidirectional(
                LSTM(max(1, lstm_total_units // 2), return_sequences=False),
                name=f"bilstm{suf}",
            )(h_in)

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
        if use_returns:
            for i, nm in enumerate(self.output_names):
                loss_dict[nm] = (lambda idx=i: (lambda yt, yp: composite_loss(
                    yt, yp, head_index=idx, mmd_lambda=mmd_lambda, sigma=sigma_mmd,
                    p=0, i=0, d=0, list_last_signed_error=[], list_last_stddev=[],
                    list_last_mmd=[], list_local_feedback=[])))()
        else:
            for i, nm in enumerate(self.output_names):
                loss_dict[nm] = (lambda idx=i: (lambda yt, yp: composite_loss_noreturns(
                    yt, yp, head_index=idx, mmd_lambda=mmd_lambda, sigma=sigma_mmd,
                    p=0, i=0, d=0, list_last_signed_error=[], list_last_stddev=[],
                    list_last_mmd=[], list_local_feedback=[])))()
        metrics_dict = {nm: [mae_magnitude] for nm in self.output_names}
        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        self.model.summary(line_length=140)


if __name__ == "__main__":  # Minimal sanity check
    plugin = Plugin()
    plugin.build_model((24, 3), None, {"predicted_horizons": [1]})
    print(plugin.output_names)