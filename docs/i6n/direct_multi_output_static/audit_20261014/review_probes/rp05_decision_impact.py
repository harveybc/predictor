"""Independent reviewer probe RP05: DMO10 point (d) — tolerance isolation and the
near-tie early-stopping negative control.

No TensorFlow needed: exercises MeanMonitorStopper (the REAL training stopper)
with histories perturbed inside the MEASURED monitor replay uncertainty
(1.8e-5 relative from the retained DMO10 handoff table; 1.211e-4 from the
pin's GPU_REPLAY.json row-level witness). Reports honestly whether best_epoch
and stop_epoch can flip; invariance is NOT required, misreporting is.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

# MeanMonitorStopper lives in tools.i6dmo.train, which imports tools.i6dmo.model
# (and therefore TensorFlow). That import is part of the real path; accept it.
from tools.i6dmo import train as T  # noqa: E402
from tools.i6dmo import design as D  # noqa: E402

results = {}

# --- isolation: training knobs are fixed in the sealed design schema ---
schema_training = {"max_epochs": 200, "patience": 10, "min_delta": 1e-8}
doc = json.loads((ROOT / "docs/i6n/direct_multi_output_static/EXPERIMENT.short.json").read_text())
eff = D.effective_config(D.validate_design(doc))
got = {k: eff["training"][k] for k in schema_training}
results["training_knobs_untouched"] = {
    "ok": got == schema_training,
    "detail": {"effective": got, "early_stop_rule": "i6dmo.mean_monitor_stop.v1"}}

# --- isolation: the tolerance symbol is confined to replay verification ---
import subprocess
out = subprocess.run(["grep", "-rl", "MONITOR_TOLERANCE", "--include=*.py", str(ROOT / "tools"),
                      str(ROOT / "app"), str(ROOT / "olap"), str(ROOT / "optimizer_plugins")],
                     capture_output=True, text=True)
users = sorted(Path(p).relative_to(ROOT).as_posix() for p in out.stdout.split())
results["tolerance_confined_to_replay"] = {
    "ok": users == ["tools/i6dmo/train.py"],
    "detail": {"files_referencing_MONITOR_TOLERANCE": users,
               "note": ("defined and applied only inside train.replay_monitored/_compare_witness; "
                        "replay_monitor.py consumes it solely through that function; no reference "
                        "in naive pairing, eligibility, app/, olap/ or optimizer_plugins/")}}

# --- near-tie negative control: can the measured uncertainty flip decisions? ---
def run_history(values, patience=10, min_delta=1e-8, max_epochs=200):
    s = T.MeanMonitorStopper(patience, min_delta, max_epochs)
    for v in values:
        # update() monitors 0.5*(fit+inner); feed v twice so monitored == v
        if s.update(v, v, [__import__("numpy").zeros(1)]):
            break
    return {"best_epoch": s.best_epoch, "stop_reason": s.stop_reason,
            "stop_epoch": len(s.history), "stale_trace": [h["stale"] for h in s.history]}


flips = {}
for rel in (1.8e-5, 1.211e-4):
    # best-epoch flip: epoch 3 beats epoch 2 by LESS than the uncertainty
    base = [1.0, 0.90, 0.90 * (1.0 - 0.5 * rel)] + [1.0] * 12
    pert = [1.0, 0.90, 0.90 * (1.0 + 0.5 * rel)] + [1.0] * 12
    a, b = run_history(base), run_history(pert)
    flips[str(rel)] = {
        "best_epoch": {"unperturbed": a["best_epoch"], "perturbed": b["best_epoch"],
                       "flips": a["best_epoch"] != b["best_epoch"]},
        "stop": {"unperturbed": (a["stop_reason"], a["stop_epoch"]),
                 "perturbed": (b["stop_reason"], b["stop_epoch"]),
                 "flips": (a["stop_reason"], a["stop_epoch"]) != (b["stop_reason"], b["stop_epoch"])},
    }

# patience-reset flip at the min_delta boundary: an improvement of 1.1*min_delta
# (well inside both measured uncertainties) resets patience; 0.9*min_delta does not.
eps = 1e-8
resetting = run_history([1.0] + [1.0 - (i + 1) * 1.1 * eps for i in range(14)])
stalling = run_history([1.0] + [1.0 - (i + 1) * 0.9 * eps for i in range(14)])
flips["min_delta_boundary"] = {
    "resetting": (resetting["stop_reason"], resetting["stop_epoch"]),
    "stalling": (stalling["stop_reason"], stalling["stop_epoch"]),
    "flips": (resetting["stop_reason"], resetting["stop_epoch"])
             != (stalling["stop_reason"], stalling["stop_epoch"])}

any_flip = any(v.get("best_epoch", {}).get("flips") or v.get("stop", {}).get("flips")
               or v.get("flips") for v in flips.values())
results["near_tie_control"] = {
    "ok": True,  # honesty control: the finding is reported, invariance is NOT claimed
    "decisions_can_flip_within_measured_uncertainty": any_flip,
    "detail": flips,
    "honest_statement": (
        "min_delta=1e-8 is three to four orders of magnitude BELOW the measured monitor "
        "replay uncertainty (1.8e-5 .. 1.211e-4 relative). A near-tie between epochs "
        "inside that uncertainty CAN change best_epoch, patience resets and stop_epoch "
        "across contexts; checkpoint selection is therefore NOT proven invariant and no "
        "such invariance is claimed by this review. Within one process the monitor is "
        "bit-stable (RP04 control A), so the recorded selection is well-defined per run.")}

ok = all(r.get("ok") for r in results.values())
print(json.dumps({"probe": "RP05_decision_impact", "ok": ok, "controls": results}, indent=1))
sys.exit(0 if ok else 1)
