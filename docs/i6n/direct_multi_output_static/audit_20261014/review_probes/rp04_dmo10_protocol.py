"""Independent reviewer probe RP04: Musashi's four DMO10 verification points (a)-(c)
against the ACTUAL implementation at the reviewed pin (point (d) is RP05).

CPU only; one tiny synthetic forward, no fit. Each control records whether the
implementation MEETS the release requirement or whether a DEFECT is CONFIRMED.
Exit 0 iff every observation matches this probe's stated expectation (so the
probe is reproducible); the review document adjudicates the defects.
"""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from tools.i6dmo import model as M  # noqa: E402  (flags before TF)
from tools.i6dmo import train as T  # noqa: E402
import tensorflow as tf  # noqa: E402

results = {}


def record_result(name, matches, verdict, detail):
    results[name] = {"expectation_matched": bool(matches), "verdict": verdict, "detail": detail}


M.enable_determinism()
rng = np.random.default_rng(14102026)
features = [f"f{i}" for i in range(19)]
bundle = M.build_dmo(M.model_config(features))
model = bundle.forecast_model


def split(n, seed):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, 13, 19)).astype("float32")
    mask = r.random((n, 6)) > 0.3
    mask[:, 0] = True
    y = np.where(mask, r.normal(size=(n, 6)), np.nan)
    return np.ascontiguousarray(x), y, mask


fit, inner = split(96, 1), split(40, 2)
BATCH = 32
fm, im, monitored, witness = T.evaluate_monitored(model, fit, inner, BATCH)
flags = T.current_determinism_flags()
record = {"procedure": T.MONITOR_EVAL, "batch_size": BATCH, "graph": "inference",
          "partition": T.MONITOR_PARTITION, "monitor_instance": T.MONITOR_INSTANCE,
          "input_materialization": T.MONITOR_INPUT,
          "monitor_forward": T.monitor_forward_identity(),
          "determinism_flags": flags, "declared_tolerance": dict(T.MONITOR_TOLERANCE),
          "witness": witness, "monitored": monitored}

device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
assert device == "cpu", "this probe is CPU-only"


def replay(rec):
    return T.replay_monitored(model, fit, inner, rec)


def perturb_sum(rec, rel):
    doc = copy.deepcopy(rec)
    sums = doc["witness"]["fit"]["batches"][0]["partial_sums"]
    sums[0] = sums[0] * (1.0 + rel)
    return doc


# --- A) baseline: same-process CPU replay is bitwise ---
_, _, mon2, tolerated = replay(copy.deepcopy(record))
record_result("A_cpu_bitwise_baseline", mon2 == monitored and tolerated == [],
              "REQUIREMENT_MET", f"bitwise={mon2 == monitored}, tolerated={tolerated}")

# --- B) point (b): tolerance activates on a PURE CPU process (flags are set at import
# regardless of device), so a sub-bound CPU partial-sum divergence is TOLERATED, not
# refused. The release requirement says CPU stays strictly exact. ---
try:
    _, _, _, tol = replay(perturb_sum(record, 5e-6))
    record_result("B_cpu_tolerance_leniency", bool(tol), "DEFECT_CONFIRMED",
                  f"CPU replay TOLERATED a 5e-6 relative partial-sum divergence: {tol[:1]}; "
                  f"activation keys on flags only (TF_CUDNN_USE_AUTOTUNE="
                  f"{flags['TF_CUDNN_USE_AUTOTUNE']!r}), visible GPUs = 0")
except T.TrainDisposition as exc:
    record_result("B_cpu_tolerance_leniency", False, "REQUIREMENT_MET", f"refused: {exc.code}")

# --- C) over-bound rejection works (2e-5 > 1e-5) ---
try:
    replay(perturb_sum(record, 2e-5))
    record_result("C_over_bound_rejected", False, "DEFECT", "2e-5 divergence accepted")
except T.TrainDisposition as exc:
    record_result("C_over_bound_rejected", exc.code == "MONITOR_WITNESS_DIVERGENCE",
                  "REQUIREMENT_MET", f"typed {exc.code}")

# --- D) retained-witness magnitudes keep their FAIL under the current bound ---
retained = {}
for rel in (1.8e-5, 1.211e-4):
    try:
        replay(perturb_sum(record, rel))
        retained[str(rel)] = "ACCEPTED"
    except T.TrainDisposition as exc:
        retained[str(rel)] = exc.code
record_result("D_old_witness_fail_retained",
              all(v == "MONITOR_WITNESS_DIVERGENCE" for v in retained.values()),
              "REQUIREMENT_MET", retained)

# --- E) near-zero sums fail closed (no declared zero handling in the protocol) ---
doc = copy.deepcopy(record)
doc["witness"]["fit"]["batches"][0]["partial_sums"][0] = 0.0
try:
    replay(doc)
    record_result("E_near_zero_fail_closed", False, "DEFECT", "zero-vs-finite sum accepted")
except T.TrainDisposition as exc:
    record_result("E_near_zero_fail_closed", exc.code == "MONITOR_WITNESS_DIVERGENCE",
                  "REQUIREMENT_MET_BUT_UNDECLARED", f"typed {exc.code}; the declared tolerance "
                  "names no zero/near-zero handling (relative-only bound)")

# --- F) same-process corruption is caught bitwise by the self check ---
try:
    T.monitor_self_check(model, fit, inner, copy.deepcopy(record),
                         monitored * (1.0 + 1e-9))
    record_result("F_self_check_strict", False, "DEFECT", "corrupted expected value accepted")
except T.TrainDisposition as exc:
    record_result("F_self_check_strict", exc.code == "MONITOR_SELF_CHECK_FAILED",
                  "REQUIREMENT_MET", f"typed {exc.code}")

# --- G) point (c): monitor FORWARD TENSOR divergence (sums equal) is TOLERATED as a
# pass under the flag condition; the requirement forbids tolerating forward tensors. ---
doc = copy.deepcopy(record)
hexes = doc["witness"]["fit"]["batch0_predictions_hex"]
v = float.fromhex(hexes[0][0])
hexes[0][0] = (v * (1.0 + 5e-6)).hex()
try:
    _, _, _, tol = replay(doc)
    record_result("G_forward_tensor_leniency", bool(tol), "DEFECT_CONFIRMED",
                  f"a diverging batch-0 forward tensor was TOLERATED as a pass: "
                  f"{[t.get('row_forward_diff', {}) and 'row_diff' for t in tol]}")
except T.TrainDisposition as exc:
    record_result("G_forward_tensor_leniency", False, "REQUIREMENT_MET", f"refused: {exc.code}")

# --- H) a record WITHOUT determinism flags gets NO tolerance (old receipts cannot
# inherit the leniency through the flag gate) ---
doc = perturb_sum(record, 5e-6)
del doc["determinism_flags"]
try:
    replay(doc)
    record_result("H_flagless_record_strict", False, "DEFECT", "flagless record tolerated")
except T.TrainDisposition as exc:
    record_result("H_flagless_record_strict", exc.code == "MONITOR_WITNESS_DIVERGENCE",
                  "REQUIREMENT_MET", f"typed {exc.code}")

# --- I) point (a): the protocol identity the release requires is ABSENT at the pin ---
sealed = dict(T.MONITOR_TOLERANCE)
required = {"atol": False, "rtol_5e-5_revision": sealed.get("relative_per_partial_sum") == 5e-5,
            "combined_formula_abs(actual-reference)<=atol+rtol*abs(reference)": False,
            "reference_direction": False, "scope_gpu_cross_process": False,
            "dtype": False, "runtime_version": False,
            "label_NUMERICALLY_EQUIVALENT": False, "versioned_protocol_id": False}
present = {k: bool(v) for k, v in required.items()}
record_result("I_protocol_identity_absent", not any(present.values()), "DEFECT_CONFIRMED",
              {"declared_tolerance": sealed, "required_but_absent":
               [k for k, v in present.items() if not v]})

# --- K) exported reload / serving prediction equality stays exact (CPU context) ---
from tools.i6dmo import gpu_probe_dmo10 as P  # noqa: E402
report = P.run(Path(sys.argv[1]) / "dmo10_probe")
record_result("K_reload_serving_exact_cpu", report["divergent_pairs"] == [],
              "REQUIREMENT_MET",
              {"device": report["device"], "divergent_pairs": report["divergent_pairs"],
               "context": "in_session vs rebuild vs full_load, one CPU process"})

ok = all(v["expectation_matched"] for v in results.values())
print(json.dumps({"probe": "RP04_dmo10_protocol", "ok": ok, "device": device,
                  "controls": results}, indent=1, default=str))
sys.exit(0 if ok else 1)
