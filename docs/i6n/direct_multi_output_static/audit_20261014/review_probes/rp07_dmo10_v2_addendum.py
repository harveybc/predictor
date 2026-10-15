"""Independent reviewer probe RP07 (addendum @a33cfe70): the sealed
i6dmo.monitor_numeric_equivalence.v2 protocol against Musashi's points (a)-(d).

CPU only, tiny synthetic forward, no fit. ``_gpu_visible`` is monkeypatched
in memory (never on disk) to exercise the proven-scope branches on this
CPU-only host. Exit 0 iff every control behaves as the release requires.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from tools.i6dmo import model as M  # noqa: E402  (flags before TF)
from tools.i6dmo import train as T  # noqa: E402
import tensorflow as tf  # noqa: E402

results = {}


def check(name, ok, detail):
    results[name] = {"ok": bool(ok), "detail": detail}


M.enable_determinism()
assert not tf.config.list_physical_devices("GPU"), "CPU-only probe"
bundle = M.build_dmo(M.model_config([f"f{i}" for i in range(19)]))
model = bundle.forecast_model


def split(n, seed):
    r = np.random.default_rng(seed)
    x = np.ascontiguousarray(r.normal(size=(n, 13, 19)).astype("float32"))
    mask = r.random((n, 6)) > 0.3
    mask[:, 0] = True
    return x, np.where(mask, r.normal(size=(n, 6)), np.nan), mask


fit, inner = split(96, 1), split(40, 2)
fm, im, monitored, witness = T.evaluate_monitored(model, fit, inner, 32)
protocol = T.floor_protocol_identity()


def base_record(device="gpu", process_uuid="f" * 32):
    return {"procedure": T.MONITOR_EVAL, "batch_size": 32, "graph": "inference",
            "partition": T.MONITOR_PARTITION, "monitor_instance": T.MONITOR_INSTANCE,
            "input_materialization": T.MONITOR_INPUT,
            "monitor_forward": T.monitor_forward_identity(),
            "determinism_flags": T.current_determinism_flags(),
            "floor_protocol": copy.deepcopy(protocol),
            "recording": {"device": device, "process_uuid": process_uuid},
            "declared_tolerance": dict(T.MONITOR_TOLERANCE),
            "witness": copy.deepcopy(witness), "monitored": monitored}


def perturb(rec, rel):
    doc = copy.deepcopy(rec)
    for name in ("fit", "inner"):
        for row in doc["witness"][name]["batches"]:
            row["partial_sums"] = [v * (1.0 + rel) for v in row["partial_sums"]]
    doc["monitored"] = doc["monitored"] * (1.0 + rel)
    return doc


def replay(rec):
    return T.replay_monitored(model, fit, inner, rec)


# --- (a) sealed protocol identity ---
need = {"id": "i6dmo.monitor_numeric_equivalence.v2",
        "formula": "abs(actual - reference) <= atol + rtol * abs(reference)",
        "reference": "recorded_fit_time_value", "rtol": 5e-5, "atol": 1e-9,
        "scope": "gpu_cross_process_monitor_replay_only", "dtype": "float64",
        "label": "NUMERICALLY_EQUIVALENT"}
check("a_protocol_sealed",
      all(protocol.get(k) == v for k, v in need.items())
      and protocol.get("required_flags") == T.current_determinism_flags()
      and protocol.get("runtime", {}).get("tensorflow") == tf.__version__
      and len(protocol.get("protocol_sha256", "")) == 64,
      {k: protocol.get(k) for k in list(need) + ["runtime"]})

# a foreign protocol identity is a typed mismatch, never inherited
foreign = perturb(base_record(), 2e-5)
foreign["floor_protocol"] = dict(protocol, rtol=1e-3, protocol_sha256="0" * 64)
try:
    replay(foreign)
    check("a_foreign_protocol_refused", False, "foreign protocol accepted")
except T.TrainDisposition as exc:
    check("a_foreign_protocol_refused", exc.code == "MONITOR_PROTOCOL_MISMATCH", f"typed {exc.code}")

# an OLD receipt (pre-v2: no sealed protocol AND no recording identity) keeps its
# FAIL at the retained magnitudes; it can never inherit the floor
old_fail = {}
for rel in (1.8244613879748877e-5, 1.211e-4):
    doc = perturb(base_record(), rel)
    del doc["floor_protocol"]
    del doc["recording"]  # pre-v2 records carry neither field
    try:
        replay(doc)
        old_fail[str(rel)] = "ACCEPTED"
    except T.TrainDisposition as exc:
        old_fail[str(rel)] = exc.code
check("a_old_witness_fail_retained",
      all(v == "MONITOR_WITNESS_DIVERGENCE" for v in old_fail.values()),
      {"outcomes": old_fail,
       "note": ("a doctored protocol-less record that CLAIMS recording.device=gpu and is "
                "replayed off-GPU types CROSS_DEVICE_NOT_COMPARABLE instead of FAIL - "
                "recorded separately, never graded equal; real pre-v2 receipts have no "
                "recording field and keep their FAIL as shown here")})

# --- (b) CPU and same-process strictly exact; scope proven, not inferred ---
outcomes = {}
for name, rec in (("cpu_recorded", perturb(base_record(device="cpu"), 2e-5)),
                  ("same_process", perturb(base_record(
                      process_uuid=T.PROCESS_IDENTITY["uuid"]), 2e-5))):
    try:
        replay(rec)
        outcomes[name] = "ACCEPTED"
    except T.TrainDisposition as exc:
        outcomes[name] = exc.code
# a GPU-recorded record replayed WITHOUT a GPU: separately typed, never graded
try:
    replay(perturb(base_record(device="gpu"), 2e-5))
    outcomes["gpu_record_no_gpu"] = "ACCEPTED"
except T.TrainDisposition as exc:
    outcomes["gpu_record_no_gpu"] = exc.code
check("b_scope_proven_strict",
      outcomes == {"cpu_recorded": "MONITOR_WITNESS_DIVERGENCE",
                   "same_process": "MONITOR_WITNESS_DIVERGENCE",
                   "gpu_record_no_gpu": "CROSS_DEVICE_NOT_COMPARABLE"}, outcomes)

# proven gpu_cross_process scope (simulated GPU visibility, in memory only):
_real = T._gpu_visible
T._gpu_visible = lambda: True
try:
    _, _, got, report = replay(perturb(base_record(device="gpu"), 2e-5))
    check("b_gpu_cross_process_floor",
          report["status"] == "NUMERICALLY_EQUIVALENT"
          and report["context"] == "gpu_cross_process" and bool(report["within_floor"])
          and got == monitored,
          {"status": report["status"], "context": report["context"],
           "monitored_abs_delta": report.get("monitored_abs_delta")})
    # beyond the floor stays a hard divergence even in the proven scope
    try:
        replay(perturb(base_record(device="gpu"), 1.211e-4))
        check("b_over_bound_refused_in_scope", False, "1.211e-4 accepted in scope")
    except T.TrainDisposition as exc:
        check("b_over_bound_refused_in_scope", exc.code == "MONITOR_WITNESS_DIVERGENCE",
              f"typed {exc.code}")
finally:
    T._gpu_visible = _real

# boundary and near-zero through the sealed formula itself
check("b_boundary_and_near_zero",
      T.floor_satisfied(5e-10, 0.0) and not T.floor_satisfied(2e-9, 0.0)
      and T.floor_satisfied(2.0 + (1e-9 + 5e-5 * 2.0), 2.0)
      and not T.floor_satisfied(2.0 + (1e-9 + 5e-5 * 2.0) * 1.01, 2.0),
      "atol handles near-zero; boundary inclusive; over-bound refused")

# same-process corruption caught by the strict self-check
try:
    T.monitor_self_check(model, fit, inner, base_record(
        process_uuid=T.PROCESS_IDENTITY["uuid"]), monitored * (1.0 + 1e-9))
    check("b_self_check_strict", False, "corrupted expected accepted")
except T.TrainDisposition as exc:
    check("b_self_check_strict", exc.code == "MONITOR_SELF_CHECK_FAILED", f"typed {exc.code}")

# --- (c) forward tensors: diagnostics only, never bitwise-claimed, never leniency ---
doc = base_record(device="cpu")  # strict context: no tolerance anywhere
hexes = doc["witness"]["fit"]["batch0_predictions_hex"]
hexes[0][0] = (float.fromhex(hexes[0][0]) * (1.0 + 5e-6)).hex()
_, _, got, report = replay(doc)
diag = report["row_forward_diagnostics"].get("fit") or {}
check("c_forward_tensors_diagnostic_only",
      report["status"] == "EXACT" and report["within_floor"] == []
      and diag.get("n_divergent_rows") == 1 and got == monitored,
      {"status": report["status"], "diagnostics": {k: diag.get(k) for k in
       ("n_divergent_rows", "max_relative_delta")},
       "note": ("sums/monitored comparison EXACT; forward divergence carried as a recorded "
                "diagnostic; replay_monitor labels this EXACT_SUMS_WITH_FORWARD_DIAGNOSTICS, "
                "never bitwise")})

# exported reload / serving equality stays exact in its named CPU context
from tools.i6dmo import gpu_probe_dmo10 as P  # noqa: E402
probe = P.run(Path(sys.argv[1]) / "dmo10_probe")
check("c_reload_serving_exact_cpu", probe["divergent_pairs"] == [],
      {"device": probe["device"], "divergent_pairs": probe["divergent_pairs"]})

# --- (d) decision impact: reported per unit, carries the near-tie flip honestly ---
near_tie = [1.0, 0.90, 0.90 * (1.0 - 0.5 * 1.8e-5)] + [1.0] * 10
separated = [1.0, 0.5, 0.25, 0.125] + [1.0] * 10
r_tie = T.stopping_decision_impact(near_tie, min_delta=1e-8, patience=10,
                                   relative_uncertainty=T.MEASURED_MONITOR_UNCERTAINTY_REL)
r_sep = T.stopping_decision_impact(separated, min_delta=1e-8, patience=10,
                                   relative_uncertainty=T.MEASURED_MONITOR_UNCERTAINTY_REL)
check("d_decision_impact_honest",
      r_tie["could_change_best_epoch"] is True and r_sep["could_change_best_epoch"] is False
      and "never claims cross-context decision invariance" in r_tie["note"]
      and T.MEASURED_MONITOR_UNCERTAINTY_REL == 1.8244613879748877e-5,
      {"near_tie": {k: r_tie[k] for k in ("could_change_best_epoch",
                                          "could_change_patience_or_stop", "best_epoch")},
       "separated": {k: r_sep[k] for k in ("could_change_best_epoch",
                                           "could_change_patience_or_stop")}})

# training knobs remain untouched in the sealed design
from tools.i6dmo import design as D  # noqa: E402
doc = json.loads((ROOT / "docs/i6n/direct_multi_output_static/EXPERIMENT.short.json").read_text())
eff = D.effective_config(D.validate_design(doc))
check("d_training_knobs_untouched",
      {k: eff["training"][k] for k in ("max_epochs", "patience", "min_delta")}
      == {"max_epochs": 200, "patience": 10, "min_delta": 1e-8},
      eff["training"])

ok = all(v["ok"] for v in results.values())
print(json.dumps({"probe": "RP07_dmo10_v2_addendum", "ok": ok, "controls": results},
                 indent=1, default=str))
sys.exit(0 if ok else 1)
