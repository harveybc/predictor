"""Matched causal branch alternatives for selected-feature weekly forecasting.

All arms preserve six ordered time positions through fusion and core. ARCH_A is
the established FS4 model. The other arms replace only its branch stack.
"""

from __future__ import annotations

from tools import fs4_temporal_predictor as P

ARMS = ("ARCH_0", "ARCH_A", "ARCH_B", "ARCH_C")


def build_model(spec: P.PredictorSpec, n_features: int, input_mode: str,
                latent_dim: int | None, arm: str):
    if arm not in ARMS:
        raise P.Refusal(f"UNKNOWN_ARCHITECTURE: {arm}")
    if input_mode != "RAW" or latent_dim is not None:
        raise P.Refusal("I6A_RAW_ONLY")
    if n_features < 1:
        raise P.Refusal("AT_LEAST_ONE_FEATURE")
    if arm == "ARCH_A":
        return P.build_predictor(spec, n_features, input_mode, latent_dim)
    keras = P._keras()
    L = keras.layers
    inp = keras.Input((spec.window, 2 * n_features), name="raw_windows")
    if arm == "ARCH_0":
        # Fixed causal alignment: last sample of each non-overlapping 4-hour bin.
        # No learned per-feature branch; all cross-feature mixing begins at fusion.
        h = L.Lambda(lambda x: x[:, 3::4, :], output_shape=(spec.latent_steps, 2 * n_features),
                     name="fixed_causal_downsample")(inp)
    elif arm == "ARCH_B":
        G = P._grouped_layer_class()
        h = G(n_features, spec.branch_filters, spec.branch_kernel, activation="relu", name="branch_stem")(inp)
        for j, dilation in enumerate((1, 2)):
            u = G(n_features, spec.branch_filters, spec.branch_kernel, activation="relu",
                  dilation_rate=dilation, name=f"branch_res{j}_a")(h)
            u = G(n_features, spec.branch_filters, spec.branch_kernel, dilation_rate=dilation,
                  name=f"branch_res{j}_b")(u)
            h = L.Add(name=f"branch_res{j}_add")([h, u])
            h = G(n_features, spec.branch_filters, spec.branch_kernel, strides=2,
                  activation="relu", name=f"branch_down{j + 1}")(h)
    else:
        # One independent recurrent branch per selected feature. Recurrent
        # processing follows causal 24->12->6 compression, preserving order.
        parts = []
        for i in range(n_features):
            x = L.Lambda(lambda t, start=2 * i: t[:, :, start:start + 2],
                         output_shape=(spec.window, 2), name=f"feature_{i}_slice")(inp)
            x = L.Conv1D(spec.branch_filters, spec.branch_kernel, strides=2, padding="causal",
                         activation="relu", name=f"feature_{i}_down1")(x)
            x = L.Conv1D(spec.branch_filters, spec.branch_kernel, strides=2, padding="causal",
                         activation="relu", name=f"feature_{i}_down2")(x)
            x = L.GRU(spec.branch_filters, return_sequences=True, name=f"feature_{i}_gru")(x)
            parts.append(x)
        h = parts[0] if len(parts) == 1 else L.Concatenate(axis=-1, name="branch_concat")(parts)
    h = L.Conv1D(spec.fuse_filters, 1, padding="causal", activation="relu", name="channel_fusion")(h)
    for j, dilation in enumerate(spec.core_dilations):
        u = L.Conv1D(spec.core_filters, spec.core_kernel, padding="causal", dilation_rate=dilation,
                     activation="relu", name=f"core{j}_a")(h)
        u = L.Conv1D(spec.core_filters, spec.core_kernel, padding="causal", dilation_rate=dilation,
                     name=f"core{j}_b")(u)
        h = L.Add(name=f"core{j}_residual")([h, u])
    h = L.Activation("relu", name=P.CORE_OUTPUT_NAME)(h)
    last = L.Lambda(lambda t: t[:, -1, :], output_shape=(spec.core_filters,),
                    name=P.HEAD_INPUT_NAME)(h)
    model = keras.Model(inp, L.Dense(1, name="head")(last), name=f"I6A_{arm}")
    model.fs4_architecture = P.digest({"study": "I6A", "arm": arm, "spec": spec.to_dict(),
                                      "n_features": n_features, "input_mode": input_mode})
    return model
