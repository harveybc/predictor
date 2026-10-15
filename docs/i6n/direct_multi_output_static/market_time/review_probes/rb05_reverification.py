#!/usr/bin/env python3
"""RB05 - BOUNDED re-verification of the four blockers of REVIEW_MARKET_TIME.md B.1.

Scope is exactly those four claims plus the pin discipline. The already-passing
plan-section-6 battery (RB01/RB02 PASS areas) is NOT re-run, and DMO10 is not
reopened.

  B-1  normalize_model_config accepts the sealed model block, is byte-equal to
       model_config(..., family=...), and both sealed designs build 3-input /
       78-channel networks; a non-null fixed offsets ladder is refused; the
       time_encoder identity (24 h conversion, fitted false, two context
       channels) verifies.
  B-2  the trainer consumes i6dmo.market_bundle.v1: schema read BEFORE body
       verification, rule/data-identity authentication, boundary/purge/finiteness
       re-applied, and the REAL CLI completes on BOTH sealed designs against real
       bundles on a 48-row subset for 2 epochs with a market_time receipt block.
  B-3  campaign._design_checks / _seal_family accept and VALIDATE the sealed
       market-time schema, refuse contradictions typed, and the improvised
       fixture design builder is gone.
  B-4  ordinal / impossible ladders are caught typed at the encoder AND at the
       trainer's input assembly - the two cases RB02 E2 found accepted.

CPU only, no GPU, no TEST row, no credential. argv[1] = a workdir.
Exit 0; the JSON on stdout is the verdict.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.getcwd())

from tests.i6dmo import _synth                                             # noqa: E402
from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4   # noqa: E402
from tools.i6dmo import campaign, data, design as D, market_grid, model as M, train as T  # noqa: E402

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
WORK.mkdir(parents=True, exist_ok=True)
ROOT = Path("docs/i6n/direct_multi_output_static")
SEALED = {"mg2": ROOT / "EXPERIMENT.mg2.json", "mg12": ROOT / "EXPERIMENT.mg12.json"}
FAMILY = {"mg2": "short_mg2", "mg12": "long_mg12"}
R: dict = {"probe": "RB05_reverification", "pin": "116bbef4", "controls": {}}


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ------------------------------------------------------------------ B-1
b1: dict = {}
for tag, path in SEALED.items():
    row = {}
    eff = D.effective_config(D.load_design(path))
    block = eff["model"]
    row["effective_config_sha256"] = eff["effective_config_sha256"]
    row["model_block_keys"] = sorted(block)
    try:
        norm = M.normalize_model_config(block)
        row["normalize"] = "ACCEPTED"
    except Exception as exc:                                    # noqa: BLE001
        norm = None
        row["normalize"] = type(exc).__name__ + ":" + str(exc)[:160]
    control = M.model_config(list(MANIFEST_NO_STLFSI4), family=FAMILY[tag])
    row["byte_equal_to_model_config_control"] = bool(
        norm is not None and canon(norm) == canon(control))
    row["normalized_sha256"] = (hashlib.sha256(canon(norm).encode()).hexdigest()[:16]
                                if norm is not None else None)
    row["control_sha256"] = hashlib.sha256(canon(control).encode()).hexdigest()[:16]
    try:
        built = M.build_dmo(block)
        row["build_dmo"] = "ACCEPTED"
        row["input_names"] = [t.name.split(":")[0].split("/")[0]
                              for t in built.forecast_model.inputs]
        row["core_input_channels"] = int(built.core_model.input_shape[-1])
        row["head_names"] = list(built.forecast_model.output_names)
    except Exception as exc:                                    # noqa: BLE001
        row["build_dmo"] = type(exc).__name__ + ":" + str(exc)[:160]
    # counterexample: a NON-NULL fixed elapsed ladder must be refused
    try:
        M.normalize_model_config({**block, "offsets_hours": list(range(-24, 1, 2))})
        row["counterexample_non_null_fixed_ladder"] = "NOT_CAUGHT"
    except Exception as exc:                                    # noqa: BLE001
        row["counterexample_non_null_fixed_ladder"] = type(exc).__name__ + ":" + str(exc)[:140]
    # counterexample: a tampered time_encoder identity must be refused
    te = dict(block.get("time_encoder") or {})
    row["time_encoder_identity"] = te
    row["time_encoder_declares_24"] = te.get("unit_conversion_hours") in (24, 24.0)
    row["time_encoder_fitted_false"] = te.get("fitted") is False
    row["time_encoder_two_context_channels"] = te.get("context_channels") == [
        "adjacent_elapsed_hours_over_24", "closure_crossed"]
    try:
        M.normalize_model_config({**block, "time_encoder": {**te, "unit_conversion_hours": 12}})
        row["counterexample_tampered_unit_conversion"] = "NOT_CAUGHT"
    except Exception as exc:                                    # noqa: BLE001
        row["counterexample_tampered_unit_conversion"] = type(exc).__name__ + ":" + str(exc)[:140]
    try:
        M.normalize_model_config({**block, "time_encoder": {**te, "fitted": True}})
        row["counterexample_tampered_fitted_flag"] = "NOT_CAUGHT"
    except Exception as exc:                                    # noqa: BLE001
        row["counterexample_tampered_fitted_flag"] = type(exc).__name__ + ":" + str(exc)[:140]
    row["verdict"] = "PASS" if (
        row["normalize"] == "ACCEPTED" and row["build_dmo"] == "ACCEPTED"
        and row["byte_equal_to_model_config_control"]
        and row.get("core_input_channels") == 78
        and set(row.get("input_names") or []) ==
        {"observations", "offsets_hours", "closure_crossed"}
        and row["time_encoder_declares_24"] and row["time_encoder_fitted_false"]
        and row["time_encoder_two_context_channels"]
        and row["counterexample_non_null_fixed_ladder"] != "NOT_CAUGHT"
        and row["counterexample_tampered_unit_conversion"] != "NOT_CAUGHT"
        and row["counterexample_tampered_fitted_flag"] != "NOT_CAUGHT") else "FAIL"
    b1[tag] = row
b1["verdict"] = "PASS" if all(v["verdict"] == "PASS" for k, v in b1.items()
                              if k != "verdict") else "FAIL"
R["controls"]["B1_sealed_model_block_builds"] = b1

# ------------------------------------------------------------------ B-2
src = _synth.make_input(WORK / "input")
mg2_dir, mg12_dir, ds2_dir = WORK / "mg2", WORK / "mg12", WORK / "ds2"
market_grid.build_market_bundle(src, market_grid.market_design(_synth.load_design()),
                                mg2_dir, code_commit="REVIEW")
market_grid.build_market_bundle(src, market_grid.market_design(
    data.long_data_view(_synth.load_design())), mg12_dir, code_commit="REVIEW")
elapsed_design = market_grid._elapsed_design(market_grid.market_design(_synth.load_design()))
data.build_frozen_bundle(src, elapsed_design, ds2_dir, code_commit="REVIEW")

b2: dict = {"bundles_built": ["mg2", "mg12", "ds2(elapsed, for the cross-adoption controls)"]}
for tag, path, bdir in (("mg2", SEALED["mg2"], mg2_dir), ("mg12", SEALED["mg12"], mg12_dir)):
    row = {}
    eff = D.effective_config(D.load_design(path))
    try:
        loaded = T.load_data_bundle(bdir, eff, path)
        row["load_data_bundle"] = "LOADED"
        row["kind"] = loaded.get("kind")
        row["splits_loaded"] = sorted(k for k in loaded if k in ("fit", "inner", "validation"))
        row["fit_arrays"] = sorted(loaded["fit"])
        parts = T.market_time_inputs(loaded["fit"])
        row["assembled_input_count"] = len(parts)
        row["assembled_shapes"] = [list(np.asarray(p).shape) for p in parts]
    except Exception as exc:                                    # noqa: BLE001
        row["load_data_bundle"] = type(exc).__name__ + ":" + str(exc)[:200]
    # counterexample 1: a DS2 (elapsed) bundle under a market-time design must be a typed
    # BUNDLE_SCHEMA refusal, read from the DECLARED schema, not a digest error
    try:
        T.load_data_bundle(ds2_dir, eff, path)
        row["counterexample_elapsed_bundle_under_market_design"] = "NOT_CAUGHT"
    except Exception as exc:                                    # noqa: BLE001
        row["counterexample_elapsed_bundle_under_market_design"] = \
            str(getattr(exc, "code", type(exc).__name__)) + ":" + str(exc)[:140]
    # counterexample 2: the market bundle under the ELAPSED design must also be refused
    try:
        eff_elapsed = D.effective_config(D.load_design(ROOT / "EXPERIMENT.short.json"))
        T.load_data_bundle(bdir, eff_elapsed)
        row["counterexample_market_bundle_under_elapsed_design"] = "NOT_CAUGHT"
    except Exception as exc:                                    # noqa: BLE001
        row["counterexample_market_bundle_under_elapsed_design"] = \
            str(getattr(exc, "code", type(exc).__name__)) + ":" + str(exc)[:140]
    # counterexample 3: the WRONG market rule (MG12 bundle under the MG2 design)
    other = mg12_dir if tag == "mg2" else mg2_dir
    try:
        T.load_data_bundle(other, eff, path)
        row["counterexample_wrong_market_rule"] = "NOT_CAUGHT"
    except Exception as exc:                                    # noqa: BLE001
        row["counterexample_wrong_market_rule"] = \
            str(getattr(exc, "code", type(exc).__name__)) + ":" + str(exc)[:140]
    # the REAL CLI, exactly as a fit unit invokes it, bounded subset and 2 epochs
    out = WORK / f"cli_{tag}"
    run = subprocess.run(
        [sys.executable, "-m", "tools.i6dmo.train", "--design", str(path),
         "--bundle", str(bdir), "--out", str(out), "--device", "cpu", "--json"],
        capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.getcwd(),
             "I6DMO_TEST_MAX_ROWS": "48",
             "I6DMO_TEST_TRAINING_OVERRIDES": json.dumps({"max_epochs": 2})})
    row["cli_returncode"] = run.returncode
    receipt_path = out / "RECEIPT.json"
    if receipt_path.exists():
        rec = json.loads(receipt_path.read_text())
        row["receipt_status"] = rec.get("status")
        row["receipt_scientific"] = rec.get("scientific")
        row["receipt_market_time"] = rec.get("market_time")
        row["receipt_device"] = rec.get("device")
        row["receipt_epochs"] = (rec.get("training") or {}).get("epochs_completed")
        row["receipt_artifacts"] = sorted((rec.get("artifacts") or {}))
        row["receipt_effective_config_sha256"] = (rec.get("identity") or {}).get(
            "effective_config_sha256")
    else:
        row["receipt_status"] = "NO_RECEIPT"
        row["cli_stderr_tail"] = run.stderr[-300:]
    row["verdict"] = "PASS" if (
        row.get("load_data_bundle") == "LOADED"
        and row.get("assembled_input_count") == 3
        and row["cli_returncode"] == 0
        and row.get("receipt_status") in ("COMPLETED", "COMPLETE")
        and isinstance(row.get("receipt_market_time"), dict)
        and row.get("counterexample_elapsed_bundle_under_market_design", "").startswith(
            "BUNDLE_SCHEMA")
        and row.get("counterexample_market_bundle_under_elapsed_design") != "NOT_CAUGHT"
        and row.get("counterexample_wrong_market_rule") != "NOT_CAUGHT") else "FAIL"
    b2[tag] = row
b2["verdict"] = "PASS" if all(b2[t]["verdict"] == "PASS" for t in ("mg2", "mg12")) else "FAIL"
R["controls"]["B2_trainer_consumes_the_market_bundle"] = b2

# ------------------------------------------------------------------ B-3
b3: dict = {}
adapter = campaign._adapter(None)
for tag, path in SEALED.items():
    doc = json.loads(Path(path).read_text())
    try:
        campaign._seal_family(doc, adapter, tag, mode=campaign.MODE_MARKET_TIME)
        b3[f"seal_family_{tag}"] = "ACCEPTED"
    except Exception as exc:                                    # noqa: BLE001
        b3[f"seal_family_{tag}"] = type(exc).__name__ + ":" + str(exc)[:200]
short_doc = json.loads(Path(SEALED["mg2"]).read_text())
contradictions = {
    "history_span_hours_alongside_market_time": {"history_span_hours": 24},
    "spacing_hours_alongside_market_time": {"spacing_hours": 2},
    "input_block_alongside_market_time": {"input": {"history_span_hours": 24}},
    "elapsed_span_fixed_true": {"market_time": {**short_doc["market_time"],
                                                "elapsed_span_fixed": True}},
    "require_exact_market_slots_false": {"market_time": {**short_doc["market_time"],
                                                         "require_exact_market_slots": False}},
    "include_origin_false": {"market_time": {**short_doc["market_time"],
                                             "include_origin": False}},
}
for name, patch in contradictions.items():
    try:
        campaign._seal_family({**short_doc, **patch}, adapter, "short",
                              mode=campaign.MODE_MARKET_TIME)
        b3[f"refusal_{name}"] = "NOT_CAUGHT"
    except Exception as exc:                                    # noqa: BLE001
        b3[f"refusal_{name}"] = type(exc).__name__ + ":" + str(exc)[:140]
# an ELAPSED design inside a market-time campaign must be refused typed, and the
# market-time design must stay refused on the ELAPSED path (both directions)
elapsed_doc = json.loads((ROOT / "EXPERIMENT.short.json").read_text())
try:
    campaign._seal_family(elapsed_doc, adapter, "short", mode=campaign.MODE_MARKET_TIME)
    b3["refusal_elapsed_design_in_market_time_mode"] = "NOT_CAUGHT"
except Exception as exc:                                        # noqa: BLE001
    b3["refusal_elapsed_design_in_market_time_mode"] = \
        type(exc).__name__ + ":" + str(exc)[:140]
try:
    campaign._seal_family(short_doc, adapter, "short", mode=campaign.MODE_ELAPSED)
    b3["refusal_market_design_in_elapsed_mode"] = "NOT_CAUGHT"
except Exception as exc:                                        # noqa: BLE001
    b3["refusal_market_design_in_elapsed_mode"] = \
        type(exc).__name__ + ":" + str(exc)[:140]
# the typed token init_campaign raises for an elapsed design in a market-time campaign
b3["typed_token_in_init"] = ("ELAPSED_DESIGN_IN_MARKET_TIME_CAMPAIGN"
                             in Path("tools/i6dmo/campaign.py").read_text())
# the elapsed path must be unchanged: the real elapsed design still seals in elapsed mode
try:
    campaign._seal_family(elapsed_doc, adapter, "short", mode=campaign.MODE_ELAPSED)
    b3["elapsed_path_unchanged"] = "ACCEPTED"
except Exception as exc:                                        # noqa: BLE001
    b3["elapsed_path_unchanged"] = type(exc).__name__ + ":" + str(exc)[:140]
try:
    from tests.i6dmo import ops_fakes
    b3["improvised_write_market_time_designs_still_present"] = hasattr(
        ops_fakes, "write_market_time_designs")
except Exception as exc:                                        # noqa: BLE001
    b3["improvised_write_market_time_designs_still_present"] = repr(exc)[:120]
b3["verdict"] = "PASS" if (
    b3["seal_family_mg2"] == "ACCEPTED" and b3["seal_family_mg12"] == "ACCEPTED"
    and all(v != "NOT_CAUGHT" for k, v in b3.items() if k.startswith("refusal_"))
    and b3["typed_token_in_init"] is True
    and b3["elapsed_path_unchanged"] == "ACCEPTED"
    and b3["improvised_write_market_time_designs_still_present"] is False) else "FAIL"
R["controls"]["B3_campaign_accepts_and_validates_the_sealed_designs"] = b3

# ------------------------------------------------------------------ B-4
import tensorflow as tf                                         # noqa: E402

n, slots = 4, 13
x = np.zeros((n, slots, 19), dtype=np.float32)
good = np.tile(np.arange(-24, 1, 2, dtype=np.float64), (n, 1))
closure = np.zeros((n, slots), dtype=np.float64)
ordinal = np.tile(np.arange(-12, 1, 1, dtype=np.float64), (n, 1))
future = good.copy()
future[:, -1] = 5.0
b4: dict = {}
built = M.build_dmo(M.model_config(list(MANIFEST_NO_STLFSI4), family="short_mg2"))


def encoder_call(offs):
    enc = next(l for l in built.forecast_model.layers
               if l.__class__.__name__ == "DynamicPhysicalTimeEncoder")
    return enc([tf.zeros((n, slots, 76)), tf.constant(offs, dtype=tf.float32),
                tf.constant(closure, dtype=tf.float32)])


for name, offs in (("ordinal_slot_indices", ordinal), ("positive_future_offset", future)):
    for where, call in (("encoder", lambda o=offs: encoder_call(o)),
                        ("model_predict", lambda o=offs: M.predict_matrix(
                            built.forecast_model, [x, o, closure])),
                        ("trainer_assembly", lambda o=offs: T.market_time_inputs(
                            {"x": x, "offsets_hours": o, "closure_crossed": closure}))):
        try:
            call()
            b4[f"{name}__{where}"] = "ACCEPTED_SILENTLY"
        except Exception as exc:                                # noqa: BLE001
            b4[f"{name}__{where}"] = \
                str(getattr(exc, "code", type(exc).__name__)) + ":" + str(exc)[:120]
# positive control: the legitimate ladder must still be accepted everywhere
for where, call in (("encoder", lambda: encoder_call(good)),
                    ("model_predict", lambda: M.predict_matrix(
                        built.forecast_model, [x, good, closure])),
                    ("trainer_assembly", lambda: T.market_time_inputs(
                        {"x": x, "offsets_hours": good, "closure_crossed": closure}))):
    try:
        call()
        b4[f"legitimate_ladder__{where}"] = "ACCEPTED"
    except Exception as exc:                                    # noqa: BLE001
        b4[f"legitimate_ladder__{where}"] = \
            str(getattr(exc, "code", type(exc).__name__)) + ":" + str(exc)[:120]
b4["verdict"] = "PASS" if (
    all(v != "ACCEPTED_SILENTLY" for k, v in b4.items()
        if k.startswith(("ordinal_slot_indices__", "positive_future_offset__")))
    and all(b4[f"legitimate_ladder__{w}"] == "ACCEPTED"
            for w in ("encoder", "model_predict", "trainer_assembly"))) else "FAIL"
R["controls"]["B4_ordinal_and_impossible_ladders_caught"] = b4

# ------------------------------------------------------- pin discipline + acceptance
pins = json.loads((ROOT / "market_time/INTEGRATION_PINS.json").read_text())
pin_rows = {}
for tag, path in SEALED.items():
    entry = pins["designs"][tag]
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    eff = D.effective_config(D.load_design(path))
    pin_rows[tag] = {
        "file_sha256_matches": hashlib.sha256(raw).hexdigest() == entry["file_sha256"],
        "design_sha256_matches": D.design_sha256(doc) == entry["design_sha256"],
        "seal_sha256_matches": doc["seal_sha256"] == entry["seal_sha256"],
        "effective_matches": eff["effective_config_sha256"] == entry["effective_config_sha256"],
        "supersedes_recorded": entry.get("supersedes"),
    }
acc = (ROOT / "ACCEPTANCE.md").read_text()
pin_rows["acceptance_rows"] = {
    "has_data_rows": any(t in acc for t in ("MTD0", "MTD1", "DATA")),
    "mtp_ids": sorted(set(t for t in acc.split() if t.startswith("MTP") or
                          t.startswith("MTD") or t.startswith("MTM"))),
    "references_test_market_time_fit_path": "test_market_time_fit_path" in acc,
    "mtp08_10_present": all(t in acc for t in ("MTP08", "MTP09", "MTP10")),
}
pin_rows["pins_asserted_by_the_suite"] = "INTEGRATION_PINS" in \
    (Path("tests/i6dmo/test_ops_market_time_pair.py").read_text()
     + Path("tests/i6dmo/test_market_time_fit_path.py").read_text()
     + Path("tests/i6dmo/ops_fakes.py").read_text())
pin_rows["verdict"] = "PASS" if (
    all(all(v for k, v in pin_rows[t].items() if k.endswith("matches"))
        for t in ("mg2", "mg12"))
    and pin_rows["pins_asserted_by_the_suite"]) else "FAIL"
R["controls"]["PIN_discipline_and_acceptance_rows"] = pin_rows

R["verdict"] = {k: v.get("verdict") for k, v in R["controls"].items()}
print(json.dumps(R, indent=2, sort_keys=True, default=str))
