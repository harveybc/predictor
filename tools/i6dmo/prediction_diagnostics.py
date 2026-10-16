"""Read-only diagnostics of authenticated closed I6-DMO predictions.

No TensorFlow, fitting or TEST access. Original campaigns are never written.
The separate diagnostic outbox uses the existing governed OLAP protocol.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from tools.i6dmo import worker
from tools.i6dmo.outbox import HttpWarehouse, Outbox
from tools.i6n_swarm.common import atomic_json, file_sha256

SPLITS = ("fit", "inner", "validation")
TEST_START = 1735689600
DIAGNOSTIC_VERSION = "i6dmo.prediction_diagnostics.v1"


def validate(z, split):
    """Reject malformed/support-nonfinite arrays, duplicate origins and TEST."""
    if split not in SPLITS:
        raise ValueError("SPLIT_FORBIDDEN")
    if z["row_id"].ndim != 1 or z["row_id"].dtype.kind not in "iu":
        raise ValueError("ORIGIN_IDS")
    n, h = len(z["row_id"]), len(z["horizons_hours"])
    if z["horizons_hours"].ndim != 1 or z["horizons_hours"].dtype.kind not in "iu" \
            or np.any(z["horizons_hours"] <= 0) or len(set(z["horizons_hours"].tolist())) != h:
        raise ValueError("HORIZON_IDENTITY")
    if n and (np.any(np.diff(z["row_id"]) <= 0) or np.any(z["row_id"] >= TEST_START)):
        raise ValueError("ORIGIN_ORDER_OR_TEST")
    if z["origin_price"].shape != (n,) or not np.isfinite(z["origin_price"]).all():
        raise ValueError("ORIGIN_PRICE")
    if z["mask"].dtype != np.dtype(bool):
        raise ValueError("MASK_DTYPE")
    for key in ("mask", "predicted_price", "target_price"):
        if z[key].shape != (n, h):
            raise ValueError("ARRAY_SHAPE")
    for key in ("predicted_price", "target_price"):
        if not np.isfinite(z[key][z["mask"]]).all():
            raise ValueError("NONFINITE_SUPPORTED_PRICE")
    if np.any((z["row_id"][:, None] + 3600 * z["horizons_hours"][None, :] >= TEST_START) & z["mask"]):
        raise ValueError("SUPPORTED_TARGET_IN_TEST")
    if split == "validation" and n and np.any(z["row_id"] < 1704067200):
        raise ValueError("DEVELOPMENT_BOUNDARY")


def _head(z, horizon):
    return z["horizons_hours"].tolist().index(horizon)


def describe(z, horizon, origin_ids=None):
    """Own-horizon paired price metrics and descriptive, untuned signal diagnostics."""
    j = _head(z, horizon)
    mask = z["mask"][:, j].copy()
    if origin_ids is not None:
        mask &= np.isin(z["row_id"], origin_ids)
    n = int(mask.sum())
    if not n:
        return dict(state="NO_SUPPORT", n=0)
    p, y, pred = z["origin_price"][mask], z["target_price"][mask, j], z["predicted_price"][mask, j]
    actual, movement = y - p, pred - p
    mean = lambda a: math.fsum(a.tolist()) / n
    mae, naive = mean(np.abs(pred - y)), mean(np.abs(actual))
    sd, sd_actual = float(np.std(movement)), float(np.std(actual))
    nonzero = actual != 0
    return dict(state="COMPLETE", n=n, mae=mae, naive_mae=naive,
                relative_error=(mae - naive) / naive if naive else None,
                mean_predicted_movement=mean(movement), mean_actual_movement=mean(actual),
                residual_bias=mean(pred - y), movement_std=sd, actual_movement_std=sd_actual,
                movement_ratio_to_naive=mean(np.abs(movement)) / naive if naive else None,
                movement_correlation=float(np.corrcoef(movement, actual)[0, 1])
                if sd > 1e-15 and sd_actual > 1e-15 else None,
                direction_accuracy_nonzero_actual=float(np.mean(np.sign(movement[nonzero]) ==
                                                                np.sign(actual[nonzero])))
                if nonzero.any() else None,
                near_naive_fraction=float(np.mean(np.abs(movement) <= .01 * naive)))


def common_origins(a, ha, b, hb):
    """Intersection of valid origin timestamps, with exact current-price agreement."""
    ia, ib = _head(a, ha), _head(b, hb)
    rows = np.intersect1d(a["row_id"][a["mask"][:, ia]], b["row_id"][b["mask"][:, ib]])
    pa = a["origin_price"][np.isin(a["row_id"], rows)]
    pb = b["origin_price"][np.isin(b["row_id"], rows)]
    if not np.array_equal(pa, pb):
        raise ValueError("COMMON_ORIGIN_PRICE_MISMATCH")
    return rows


def load_fit(root, arm):
    """Authenticate the frozen unit and every artifact before reading predictions."""
    plan = json.loads((root / "PLAN.json").read_text())
    unit = next(u for u in plan["units"] if u["unit_id"] == f"{arm}.fit")
    folder = root / "terminals" / unit["identity_sha256"]
    stored = worker._verify_stored(folder, unit)
    if stored["terminal"]["status"] != "COMPLETE":
        raise ValueError("FIT_NOT_COMPLETE")
    arrays = {}
    arts = stored["terminal"]["artifacts"]
    for split in SPLITS:
        path = folder / "out" / arts[f"predictions/{split}.npz"]["path"]
        with np.load(path, allow_pickle=False) as z:
            arrays[split] = {k: z[k] for k in z.files}
        validate(arrays[split], split)
    return arrays, dict(unit_identity_sha256=unit["identity_sha256"],
                       terminal_sha256=file_sha256(folder / "TERMINAL.json"),
                       predictions={s: arts[f"predictions/{s}.npz"]["sha256"] for s in SPLITS})


def run(sequence_root):
    """Compute immutable, non-promotional diagnostics without writing source roots."""
    data, provenance, configs, results, weekly, common = {}, {}, {}, {}, {}, {}
    for family in ("short", "long"):
        root = Path(sequence_root) / family / "campaign"
        effective = json.loads((root / "EFFECTIVE_CONFIG.json").read_text())
        configs[family] = effective["adapter_effective"]
        provenance[family] = dict(campaign_id=json.loads((root / "PLAN.json").read_text())["campaign_id"],
                                  plan_sha256=file_sha256(root / "PLAN.json"),
                                  effective_sha256=file_sha256(root / "EFFECTIVE_CONFIG.json"), arms={})
        for arm in ("r0", "r2"):
            key = f"{family}.{arm}"
            data[key], provenance[family]["arms"][arm] = load_fit(root, arm)
            results[key] = {}
            for split, z in data[key].items():
                if z["horizons_hours"].tolist() != configs[family]["horizons_hours"]:
                    raise ValueError("EFFECTIVE_HORIZONS_MISMATCH")
                results[key][split] = {str(int(h)): describe(z, int(h)) for h in z["horizons_hours"]}
            # Weekly diagnostics focus only on the questioned boundary heads.
            z = data[key]["validation"]
            horizon = 12 if family == "short" else 18
            weeks = [datetime.fromtimestamp(int(t), timezone.utc).strftime("%G-W%V") for t in z["row_id"]]
            weekly[key] = {w: describe(z, horizon, z["row_id"][np.array(weeks) == w]) for w in sorted(set(weeks))}
    for arm in ("r0", "r2"):
        common[arm] = {}
        for split in SPLITS:
            a, b = data[f"short.{arm}"][split], data[f"long.{arm}"][split]
            ids = common_origins(a, 12, b, 18)
            common[arm][split] = dict(n_origins=len(ids), short_12=describe(a, 12, ids),
                                     long_18=describe(b, 18, ids))
    a, b = configs["short"], configs["long"]
    differences = {k: dict(short=a["model"].get(k), long=b["model"].get(k))
                   for k in sorted(set(a["model"]) | set(b["model"])) if a["model"].get(k) != b["model"].get(k)}
    return dict(schema=DIAGNOSTIC_VERSION, test_open=False, fits_started=0, provenance=provenance,
                near_naive_threshold="abs(predicted_price-origin_price) <= 0.01 * own split/head naive MAE",
                model_differences=differences, same_training_config=a["training"] == b["training"],
                results=results, common_origins=common, weekly_boundary_heads=weekly,
                qualification="Descriptive DEVELOPMENT diagnostics; common origins do not control horizon/grid/weights/objective; no causal attribution or promotion")


def reports(doc, code_commit):
    """Governed OLAP facts, with analysis identity distinct from scientific fit identity."""
    import hashlib
    digest = hashlib.sha256(json.dumps(doc, sort_keys=True, allow_nan=False).encode()).hexdigest()
    output = []
    def add(key, split, rows, tags):
        metrics = [dict(metric="Diagnostic_" + name, value=float(value), split=split,
                        horizon=int(h), unit="count" if name == "n" else "ratio" if any(
                            t in name for t in ("ratio", "relative", "fraction", "accuracy", "correlation")) else "price")
                   for h, row in rows.items() for name, value in row.items()
                   if name != "state" and value is not None]
        if metrics:
            output.append(dict(experiment_key=f"i6dmo:boundary:{digest[:16]}:{key}"[:128],
                experiment_set_key=f"i6dmo:boundary:{digest[:16]}", actor="predictor", lake="LOCAL_RETAINED",
                lineage="UNVERIFIED", config_sha256=digest, code_commit=code_commit,
                project="predictor", phase="I6DMO_DIAGNOSTIC", tags={**tags, "test_read": "false",
                    "analysis": DIAGNOSTIC_VERSION}, metrics=metrics))
    for key, splits in doc["results"].items():
        for split, rows in splits.items():
            add(f"{key}:{split}", split, rows, dict(arm=key, support="own"))
    for arm, splits in doc["common_origins"].items():
        for split, pair in splits.items():
            for family, h in (("short", 12), ("long", 18)):
                add(f"{arm}:{family}:{split}:common", split, {str(h): pair[f"{family}_{h}"]},
                    dict(arm=f"{family}.{arm}", support="common_12_18_origins"))
    for key, weeks in doc["weekly_boundary_heads"].items():
        h = 12 if key.startswith("short") else 18
        for week, row in weeks.items():
            add(f"{key}:{week}", "validation", {str(h): row}, dict(arm=key, week=week, support="own_week"))
    if "causal_audit" in doc:
        audit = doc["causal_audit"]
        for family, evidence in audit["families"].items():
            for split, checks in evidence["splits"].items():
                metrics = [dict(metric="Audit_" + k, value=float(v), split={"FIT": "fit", "INNER": "inner",
                    "DEVELOPMENT": "validation"}[split], horizon=None,
                    unit="bool" if isinstance(v, bool) else "count") for k, v in checks.items()]
                metrics.append(dict(metric="Audit_Unverified_Source_Features", value=len(
                    audit["upstream_limitations"]["unverified_features"]), split="validation", horizon=None, unit="count"))
                output.append(dict(experiment_key=f"i6dmo:boundary:{digest[:16]}:audit:{family}:{split}",
                    experiment_set_key=f"i6dmo:boundary:{digest[:16]}", actor="predictor", lake="LOCAL_RETAINED",
                    lineage="UNVERIFIED", config_sha256=digest, code_commit=code_commit,
                    project="predictor", phase="I6DMO_DIAGNOSTIC", tags=dict(analysis=DIAGNOSTIC_VERSION,
                        family=family, test_read="false", scope="retained_arrays_not_upstream_vintages"), metrics=metrics))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--storage-root", type=Path)
    args = parser.parse_args()
    status_path = args.output / "STATUS.json"
    if args.status:
        state = json.loads(status_path.read_text())
        if file_sha256(args.output / "RESULT.json") != state["result_sha256"]:
            raise ValueError("DIAGNOSTIC_RESULT_CHANGED")
        print(json.dumps(state, sort_keys=True))
        return
    if not args.sequence_root or not args.code_commit:
        parser.error("--sequence-root and --code-commit required for computation")
    if args.output.resolve().is_relative_to(args.sequence_root.resolve()):
        parser.error("output must be outside the immutable sequence")
    doc = run(args.sequence_root)
    if args.audit:
        if not args.input_dir or not args.storage_root:
            parser.error("--audit requires --input-dir and --storage-root")
        from tools.i6dmo.prediction_causal_audit import audit
        doc["causal_audit"] = audit(args.sequence_root, args.input_dir, args.storage_root)
    atomic_json(args.output / "RESULT.json", doc)
    box = Outbox(args.output / "outbox.sqlite")
    try:
        for report in reports(doc, args.code_commit):
            box.enqueue(report, source_unit="boundary_diagnostic")
        if args.publish:
            dep = json.loads((args.sequence_root / "short/campaign/private/deployment.json").read_text())
            wh = dep["warehouse"]
            box.pump(HttpWarehouse(wh["url"], wh["token_env"]))
        state = dict(schema=DIAGNOSTIC_VERSION, result_sha256=file_sha256(args.output / "RESULT.json"),
                     compute="COMPLETE", fits_started=0, test_open=False, publication=box.summary())
        atomic_json(status_path, state)
        print(json.dumps(state, sort_keys=True))
    finally:
        box.close()


if __name__ == "__main__":
    main()
