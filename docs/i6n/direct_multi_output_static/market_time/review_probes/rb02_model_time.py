#!/usr/bin/env python3
"""RB02 - independent falsification of the market-time MODEL path.

Plan MARKET_TIME_PAIR_PLAN.md sections 3 and 6, model items:
  E. the ACTUAL per-example signed offsets and closure flags reach the model
     under controlled weights, and an ordinal-index encoding would be caught
  F. the two context channels are exact (adjacent elapsed hours / 24,
     closure-crossed) and the 24 is a fixed unit conversion, never fitted
  G. 19 branches -> 76 encoded channels + 2 context = 78 into the core
  H. serialization / reload of ALL inputs including the time channels
  I. the old DS2 family is unchanged, and no DS2/ASOF artifact is adoptable by
     shape alone

Each area carries a fail-closed counterexample written by this reviewer.
CPU only, no fit (no gradient step is taken; weights are SET, not learned),
no TEST row. argv[1] = a workdir. Exit 0; the JSON on stdout is the verdict.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.getcwd())

from tools.i6dmo import model as M                   # noqa: E402

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
WORK.mkdir(parents=True, exist_ok=True)
R: dict = {"probe": "RB02_model_time", "controls": {}}
FEATURES = [f"f{i:02d}" for i in range(19)]


def build(family="short_mg2"):
    return M.build_dmo(M.model_config(FEATURES, family=family))


b = build()
cfg = b.config
ident = b.component_manifests()
R["controls"]["G1_channel_arithmetic"] = {
    "branches": len(cfg["features"]),
    "fusion_output_shape": list(b.fusion_model.output_shape[1:]),
    "core_input_shape": list(b.core_model.input_shape[1:]),
    "core_input_channels": int(b.core_model.input_shape[-1]),
    "expected": 19 * 4 + 2,
    "slots": cfg["slots"],
    "time_encoder_identity": ident.get("time_encoder"),
    "encoder_has_no_weights": not any(
        lay.weights for lay in b.forecast_model.layers
        if lay.__class__.__name__ == "DynamicPhysicalTimeEncoder"),
    "model_input_names": [t.name.split(":")[0].split("/")[0]
                          for t in b.forecast_model.inputs],
}
g1 = R["controls"]["G1_channel_arithmetic"]
g1["verdict"] = "PASS" if (g1["core_input_channels"] == 78 and g1["slots"] == 13
                           and g1["encoder_has_no_weights"]
                           and set(g1["model_input_names"]) ==
                           {"observations", "offsets_hours", "closure_crossed"}) else "FAIL"

# -------------------------------------------------- E/F. controlled-weight time probe
# Controlled weights: every branch/core/head kernel is set so the network is an exact
# linear readout of ONE core input channel. No training occurs.
rng = np.random.default_rng(5)


def zero_branches(bundle):
    """CONTROLLED WEIGHTS: force every per-feature branch to output exactly zero.

    After this, the fused tensor is identically 0, so the core and the heads see
    ONLY what the time encoder contributes: the sinusoidal code on the actual
    offsets and the two context channels. Any change in the predictions is then
    attributable to the per-example time inputs alone. No gradient step is taken.
    """
    touched = []
    for layer in bundle.forecast_model.layers:
        if not layer.name.startswith("branch"):
            continue
        ws = layer.get_weights()
        if ws:
            layer.set_weights([np.zeros_like(w) for w in ws])
            touched.append(layer.name)
    return touched


def encoder_output(bundle, x, offsets, closure):
    """Read the encoder's output tensor directly for an explicit, weightless check."""
    import tensorflow as tf
    enc = next(l for l in bundle.forecast_model.layers
               if l.__class__.__name__ == "DynamicPhysicalTimeEncoder")
    fused = bundle.fusion_model(np.asarray(x, dtype=np.float32), training=False)
    out = enc([fused, tf.constant(offsets, dtype=tf.float32),
               tf.constant(closure, dtype=tf.float32)])
    return np.asarray(out, dtype=np.float64), np.asarray(fused, dtype=np.float64)


branch_layers_zeroed = zero_branches(b)
n, slots = 4, cfg["slots"]
x = rng.normal(size=(n, slots, 19)).astype(np.float32)
# realistic market-time offsets: 12 x 2 open-market hours, one window crossing a weekend
offs = np.tile(np.arange(-24, 1, 2, dtype=np.float64), (n, 1))
offs[1, :5] -= 48.0          # a weekend closure between slot 4 and slot 5
offs[2, :9] -= 47.0          # a DST-shifted closure of 47 h
closure = np.zeros((n, slots), dtype=np.float64)
closure[1, 5] = 1.0
closure[2, 9] = 1.0

enc, fused = encoder_output(b, x, offs, closure)
adj = np.zeros_like(offs)
adj[:, 1:] = np.diff(offs, axis=1)
want_ctx = np.stack([adj / 24.0, closure], axis=-1)
ctx = enc[:, :, -2:]
code = enc[:, :, :-2] - fused
# the same deterministic sinusoidal rule, evaluated by this reviewer on the ACTUAL offsets
width = fused.shape[-1]
ch = np.arange(width)
rates = np.power(10000.0, -2.0 * (ch // 2) / width)
angle = offs[:, :, None] * rates[None, None, :]
want_code = np.where(ch % 2 == 0, np.sin(angle), np.cos(angle))

R["controls"]["F1_context_channels_exact"] = {
    "context_shape": list(ctx.shape),
    "max_abs_context_error": float(np.abs(ctx - want_ctx).max()),
    "adjacent_channel_is_delta_over_24": bool(
        np.allclose(ctx[:, :, 0], adj / 24.0, rtol=0, atol=1e-6)),
    "closure_channel_is_the_flag": bool(np.array_equal(ctx[:, :, 1], closure)),
    "first_slot_delta_zero": bool(np.all(ctx[:, 0, 0] == 0.0)),
    "max_abs_sinusoid_error_vs_independent_recomputation":
        float(np.abs(code - want_code).max()),
}
f1 = R["controls"]["F1_context_channels_exact"]
f1["verdict"] = "PASS" if (f1["max_abs_context_error"] <= 1e-6
                           and f1["max_abs_sinusoid_error_vs_independent_recomputation"] <= 1e-5
                           and f1["first_slot_delta_zero"]) else "FAIL"

# E1 the ACTUAL offsets reach the model: ordinal indices must give a DIFFERENT output
ordinal = np.tile(np.arange(-12, 1, 1, dtype=np.float64), (n, 1))     # slot order, not hours
enc_ord, _ = encoder_output(b, x, ordinal, closure)
pred_actual = M.predict_matrix(b.forecast_model, [x, offs, closure])
pred_ordinal = M.predict_matrix(b.forecast_model, [x, ordinal, closure])
pred_noclosure = M.predict_matrix(b.forecast_model, [x, offs, np.zeros_like(closure)])
R["controls"]["E1_actual_time_reaches_the_model"] = {
    "controlled_weights": "every branch kernel and bias set to zero so the fused "
                          "tensor is identically 0 and only the time encoder "
                          "contributes; no fit, no gradient step",
    "branch_layers_zeroed": len(branch_layers_zeroed),
    "fused_is_identically_zero": float(np.abs(fused).max()),
    "encoded_differs_for_ordinal_vs_actual":
        float(np.abs(enc_ord - enc).max()),
    "predictions_differ_for_ordinal_vs_actual":
        float(np.abs(pred_ordinal - pred_actual).max()),
    "predictions_differ_when_closure_flags_are_zeroed":
        float(np.abs(pred_noclosure - pred_actual).max()),
    "note": "a nonzero difference under zeroed branches proves the per-example "
            "time inputs are consumed by the core and the heads; it proves "
            "nothing about predictive quality.",
}
e1 = R["controls"]["E1_actual_time_reaches_the_model"]
e1["verdict"] = "PASS" if (e1["fused_is_identically_zero"] == 0.0
                           and e1["encoded_differs_for_ordinal_vs_actual"] > 0
                           and e1["predictions_differ_for_ordinal_vs_actual"] > 0
                           and e1["predictions_differ_when_closure_flags_are_zeroed"] > 0) \
    else "FAIL"

# E2 would an ORDINAL-INDEX encoding be CAUGHT? Feed slot indices where hours belong
# and look for any typed refusal from the model or its contract surface.
caught = {}
try:
    M.predict_matrix(b.forecast_model, [x, ordinal, closure])
    caught["model_call_with_ordinal_offsets"] = "ACCEPTED_SILENTLY"
except Exception as exc:                              # noqa: BLE001
    caught["model_call_with_ordinal_offsets"] = type(exc).__name__ + ":" + str(exc)[:140]
# a positive (future) offset is physically impossible for a causal window
future = offs.copy()
future[:, -1] = 5.0
try:
    M.predict_matrix(b.forecast_model, [x, future, closure])
    caught["model_call_with_a_positive_future_offset"] = "ACCEPTED_SILENTLY"
except Exception as exc:                              # noqa: BLE001
    caught["model_call_with_a_positive_future_offset"] = \
        type(exc).__name__ + ":" + str(exc)[:140]
# the data path's own guard, for contrast
try:
    from tools.i6dmo import market_grid as MG

    class _A:
        actual_offsets_s = (offs * 3600).astype(np.int64)
        closure_crossed = closure.astype(bool)
        slot_instants = np.zeros((n, slots), dtype=np.int64)
    _A.actual_offsets_s[:, -1] = 3600            # last offset not zero
    MG.time_inputs(_A())
    caught["bundle_time_inputs_with_nonzero_last_offset"] = "ACCEPTED_SILENTLY"
except Exception as exc:                              # noqa: BLE001
    caught["bundle_time_inputs_with_nonzero_last_offset"] = \
        type(exc).__name__ + ":" + str(exc)[:140]
# a fixed per-batch grid (one row of offsets broadcast) must be refused by build()
try:
    import tensorflow as tf
    enc_layer = M.DynamicPhysicalTimeEncoder()
    enc_layer([tf.zeros((n, slots, 76)), tf.zeros((slots,)), tf.zeros((n, slots))])
    caught["encoder_with_a_fixed_per_batch_grid"] = "ACCEPTED_SILENTLY"
except Exception as exc:                              # noqa: BLE001
    caught["encoder_with_a_fixed_per_batch_grid"] = type(exc).__name__ + ":" + str(exc)[:140]
R["controls"]["E2_ordinal_encoding_would_be_caught"] = {
    **caught,
    "verdict": "PASS" if all(v != "ACCEPTED_SILENTLY" for v in caught.values()) else "FAIL",
}

# ------------------------------------------- H. serialization / reload of ALL inputs
out = WORK / "mg2_model"
M.save_dmo(b, out, extra={"review": "RB02"})
rb, rb_doc = M.load_dmo(out)
pred_reload = M.predict_matrix(rb.forecast_model, [x, offs, closure])
R["controls"]["H1_serialization_reload"] = {
    "input_names_recorded": json.loads((out / "MODEL.json").read_text()).get("input_names"),
    "market_time_metadata_recorded":
        json.loads((out / "MODEL.json").read_text()).get("market_time"),
    "max_abs_prediction_difference_after_reload":
        float(np.abs(pred_reload - pred_actual).max()),
    "reloaded_core_input_channels": int(rb.core_model.input_shape[-1]),
}
h1 = R["controls"]["H1_serialization_reload"]
h1["slot_order_is_not_wall_clock_recorded"] = bool(
    (h1["market_time_metadata_recorded"] or {}).get("slot_order_is_not_wall_clock"))
h1["time_inputs_recorded"] = (h1["market_time_metadata_recorded"] or {}).get("time_inputs")
h1["verdict"] = "PASS" if (h1["max_abs_prediction_difference_after_reload"] == 0.0
                           and h1["reloaded_core_input_channels"] == 78
                           and h1["slot_order_is_not_wall_clock_recorded"]
                           and h1["time_inputs_recorded"] ==
                           ["offsets_hours", "closure_crossed"]) else "FAIL"
# H counterexample: a reload whose metadata was stripped must be refused
import shutil                                          # noqa: E402
forged = WORK / "mg2_model_forged"
shutil.copytree(out, forged, dirs_exist_ok=True)
doc = json.loads((forged / "MODEL.json").read_text())
doc["market_time"] = None
(forged / "MODEL.json").write_text(json.dumps(doc))
try:
    M.load_dmo(forged)
    h1["counterexample_stripped_market_metadata"] = "NOT_CAUGHT"
except Exception as exc:                                # noqa: BLE001
    h1["counterexample_stripped_market_metadata"] = type(exc).__name__ + ":" + str(exc)[:160]
h1["counterexample_caught"] = h1["counterexample_stripped_market_metadata"] != "NOT_CAUGHT"
if not h1["counterexample_caught"]:
    h1["verdict"] = "FAIL"

# ------------------------------------- I. DS2 unchanged, no adoption by shape alone
ds2 = M.build_dmo(M.model_config(FEATURES, family="short"))
ds2_ident = ds2.component_manifests()
ds2_out = WORK / "ds2_model"
M.save_dmo(ds2, ds2_out)
ds2_reload, _ = M.load_dmo(ds2_out)
xs = rng.normal(size=(3, 13, 19)).astype(np.float32)
ds2_pred = M.predict_matrix(ds2.forecast_model, xs)
ds2_pred2 = M.predict_matrix(ds2_reload.forecast_model, xs)
R["controls"]["I1_ds2_unchanged_and_not_adoptable"] = {
    "ds2_single_input": [t.name.split(":")[0].split("/")[0] for t in ds2.forecast_model.inputs],
    "ds2_positional_encoding": ds2_ident["positional_encoding"]["layer"],
    "ds2_time_encoder_field": ds2_ident.get("time_encoder"),
    "ds2_core_input_channels": int(ds2.core_model.input_shape[-1]),
    "ds2_reload_identical": float(np.abs(ds2_pred - ds2_pred2).max()) == 0.0,
    "model_config_sha256_differs":
        ident["model_config_sha256"] != ds2_ident["model_config_sha256"],
}
i1 = R["controls"]["I1_ds2_unchanged_and_not_adoptable"]
# counterexamples: the market config must not accept a DS2 body, and a DS2
# artifact must not load as a market-time one by shape.
try:
    M.normalize_model_config({**cfg, "offsets_hours": list(range(-24, 1, 2))})
    i1["counterexample_market_config_with_elapsed_offsets"] = "NOT_CAUGHT"
except Exception as exc:                                # noqa: BLE001
    i1["counterexample_market_config_with_elapsed_offsets"] = \
        type(exc).__name__ + ":" + str(exc)[:140]
forged2 = WORK / "ds2_model_claiming_market"
shutil.copytree(ds2_out, forged2, dirs_exist_ok=True)
doc2 = json.loads((forged2 / "MODEL.json").read_text())
doc2["market_time"] = {"rule": "MG2_MARKET_EXACT", "slot_order_is_not_wall_clock": True,
                       "time_inputs": ["offsets_hours", "closure_crossed"]}
(forged2 / "MODEL.json").write_text(json.dumps(doc2))
try:
    M.load_dmo(forged2)
    i1["counterexample_ds2_artifact_claiming_market_time"] = "NOT_CAUGHT"
except Exception as exc:                                # noqa: BLE001
    i1["counterexample_ds2_artifact_claiming_market_time"] = \
        type(exc).__name__ + ":" + str(exc)[:140]
i1["verdict"] = "PASS" if (
    i1["ds2_single_input"] == ["observations"]
    and i1["ds2_core_input_channels"] == 76
    and i1["ds2_time_encoder_field"] is None
    and i1["ds2_reload_identical"]
    and i1["model_config_sha256_differs"]
    and i1["counterexample_market_config_with_elapsed_offsets"] != "NOT_CAUGHT"
    and i1["counterexample_ds2_artifact_claiming_market_time"] != "NOT_CAUGHT") else "FAIL"

R["verdict"] = {k: v.get("verdict") for k, v in R["controls"].items()}
print(json.dumps(R, indent=2, sort_keys=True, default=str))
