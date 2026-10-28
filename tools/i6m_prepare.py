"""Authenticate retained selected features and materialize bounded worker inputs.

This does not read TEST or copy historical scoring populations into new
horizons. Parent lineage remains local/unverified. Execution uses a separate
sealed design; the originally approved planner remains historical evidence.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.i6e_horizon_plan import build_plan
from tools.i6m_data import file_hash, load_input, targets, partitions
from tools.i6m_queue import Queue, atomic_json, digest, seal


def code_identity():
    base = Path(__file__).resolve().parents[1]
    files = list((base/"tools").glob("i6m_*.py"))
    files += [base/"tools"/name for name in ("i6e_horizon_plan.py", "i6e_clock_windows.py", "i6e_masked_model.py", "i6e_compact_run.py", "fs4_temporal_predictor.py", "modular_candidate_evaluator.py")]
    files += sorted((base/"predictor_plugins/modular_temporal").glob("*.py"))
    return digest({str(p.relative_to(base)): file_hash(p) for p in sorted(files)})


def prepare(parent, output, predecessor=None):
    source = parent / "INPUT.npz"
    expected = "897caee164286bedea61c4caab361213cf703d3f50e60ba12cee543d9bfa28c9"
    if file_hash(source) != expected:
        raise ValueError("retained source differs from authenticated statistical input")
    from tools.i7_weekly_branch_regimes import _verify_seal
    prior = json.loads((parent/"DESIGN.json").read_text())
    _verify_seal(prior, "design_sha256")
    members = prior["members"]
    if len(members) != 20 or len(set(members)) != 20:
        raise ValueError("selected feature order mismatch")
    if (output/"DESIGN.json").exists():
        from tools.i6m_queue import verify
        design = verify(json.loads((output/"DESIGN.json").read_text()))
        if design["code_sha256"] != code_identity():
            raise ValueError("runtime code differs: create a successor, do not silently change running design")
        Queue(output/"QUEUE.db", design)
        return design
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as a:
        arrays = {k: a[k].copy() for k in ("ts", "x", "price")}
    np.savez_compressed(output/"INPUT.npz", **arrays)
    mask = arrays["ts"] < 1704067200
    np.savez_compressed(output/"TRAIN.npz", **{k:v[mask] for k,v in arrays.items()})
    ts, x, price = load_input(output/"INPUT.npz", file_hash(output/"INPUT.npz"))
    plan = build_plan()
    population = {}
    for h in plan["horizons_hours"]:
        y, _ = targets(ts, price, h)
        weeks = []
        origins = []
        for w in plan["validation_weeks"]:
            _, _, si = partitions(ts, y, w, h)
            weeks.append(dict(ordinal=w["ordinal"], rows=len(si), rows_sha256=digest(ts[si].tolist())))
            origins.extend(ts[si].tolist())
        population[str(h)] = dict(n=len(origins), rows_sha256=digest(origins), weeks=weeks)
    fit = dict(batch_size=256, huber_delta=1., learning_rate=.001, loss="huber", max_epochs=50,
               max_seconds=3600., max_updates=100000, min_delta=0., monitor="train_validation_mean",
               monitor_every=1, patience=5, weight_decay=.0001)
    predecessor_sha = None
    if predecessor:
        import shutil
        from tools.i6m_queue import verify
        old = verify(json.loads((predecessor/"DESIGN.json").read_text()))
        if old["parent_input_sha256"] != expected or old["members"] != members or old["fit"] != fit:
            raise ValueError("receipt-only predecessor scientific contract differs")
        for name in ("INPUT.npz", "TRAIN.npz"):
            if file_hash(predecessor/name) != old[name.split(".")[0].lower()+"_sha256"]:
                raise ValueError("predecessor input digest mismatch")
            shutil.copyfile(predecessor/name, output/name)
        predecessor_sha = old["code_sha256"]
        for cell in (predecessor/"cells").iterdir():
            terminal = cell/"CELL.json"
            if not terminal.exists():
                continue
            record = verify(json.loads(terminal.read_text()))
            if record["status"] != "FAILED":
                continue
            if not any(s in record.get("reason", "") for s in ("TRAIN_COST_FAILED", "multiple values for keyword")):
                continue
            destination = output/"cells"/cell.name
            destination.mkdir(parents=True, exist_ok=True)
            # Preserve original failed records separately; never relabel history.
            shutil.copyfile(terminal, destination/"PREDECESSOR_FAILED.json")
            for name in ("FIT.json", "MODEL.keras", "ARIMA_FITS.json", "PREDICTIONS.npz"):
                if (cell/name).exists():
                    shutil.copyfile(cell/name, destination/name)
    design = seal(dict(schema="predictor.i6m.execution.v1", arms=plan["arms"], weeks=plan["validation_weeks"],
                       seed=plan["seed"], models=["NEURAL", "ARIMA"], fit=fit, members=members,
                       plan_sha256=plan["plan_sha256"], code_sha256=code_identity(),
                       input_sha256=file_hash(output/"INPUT.npz"), train_sha256=file_hash(output/"TRAIN.npz"),
                       parent_input_sha256=expected, parent_design_sha256=prior["design_sha256"],
                       population=population, test_read=False, lineage="LOCAL_RETAINED_UNVERIFIED",
                       skill_formula="(MAE_h - naive_h) / naive_h", skill_negative_is_better=True,
                       repetitions=1, cost_pilot="exact first-week TRAIN fit adopted without repeat",
                       receipt_only_predecessor_code_sha256=predecessor_sha))
    atomic_json(output/"DESIGN.json", design)
    Queue(output/"QUEUE.db", design)
    atomic_json(output/"STATUS.json", Queue(output/"QUEUE.db", design).status())
    return design


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--predecessor", type=Path)
    a = p.parse_args()
    print(json.dumps(dict(status="PREPARED", design_sha256=prepare(a.parent, a.output, a.predecessor)["sha256"])))


if __name__ == "__main__":
    main()
