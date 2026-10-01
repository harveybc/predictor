#!/usr/bin/env python3
"""Frozen per-horizon forecast evidence for the naive-eligibility gate (owner orders b327b771, section 5).

Schema ``predictor.forecast_naive_evidence.v1`` is produced from an accepted
candidate receipt plus the validation NPZ it was scored on. Per horizon it records
model MAE and MSE, the SAME-ROW naive MAE and MSE, skill (1 - model/naive) and
delta (model - naive), and the row count. The record binds:

* artifact identity: candidate cid, model and weights sha256, predictor revision;
* population identity: dataset id, row count, sha256 of the ordered row ids, first
  and last forecast origins, target names, sample period and horizon unit;
* scaler identity and metric space (the scale the errors are expressed in);
* split provenance: ``held_out_validation`` (or ``chronological_oof``); the
  reserved trading test is never an admissible provenance;
* naive definition: persistence -- the last observed value of each target at the
  forecast origin, repeated for every horizon;
* the frozen primary metric (MAE) declared before any selection, MSE secondary.

Missing or nonfinite values are written as ``null`` with ``NOT_AVAILABLE`` and a
reason, never as zero. A zero naive error gives ``skill: null`` with reason
``ZERO_NAIVE`` (strict improvement is impossible). ``evidence_sha256`` is the
sha256 of the canonical JSON (sorted keys, compact separators) of the record
WITHOUT that field; ``verify`` recomputes it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "predictor.forecast_naive_evidence.v1"
ALLOWED_PROVENANCE = ("held_out_validation", "chronological_oof")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _pair(model, naive):
    if not (_finite(model) and _finite(naive)):
        return {"skill": None, "delta": None, "status": "NOT_AVAILABLE", "reason": "missing or nonfinite metric"}
    if naive == 0:
        return {"skill": None, "delta": model - naive, "status": "NOT_AVAILABLE", "reason": "ZERO_NAIVE"}
    return {"skill": 1 - model / naive, "delta": model - naive, "status": "OK"}


def build(receipt, validation_npz, *, provenance="held_out_validation", campaign_id=None, asset=None):
    if provenance not in ALLOWED_PROVENANCE:
        raise ValueError("provenance must be held_out_validation or chronological_oof; the reserved test is never admissible")
    if receipt.get("data", {}).get("test_used") is not False:
        raise ValueError("receipt does not certify test_used=false")
    raw = Path(validation_npz).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["digests"]["validation_sha256"]:
        raise ValueError("validation NPZ bytes differ from the receipt")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        row_ids = z["row_ids"].astype(str).tolist()
        origins = z["timestamps"].astype(np.int64)
        sample_hours = None
        if z["target_timestamps"].shape[1]:
            step = int(z["target_timestamps"][0, 0] - origins[0]) // int(z["horizons"][0])
            sample_hours = step / 3600
    iso = lambda t: dt.datetime.fromtimestamp(int(t), dt.timezone.utc).isoformat()
    per_horizon = []
    for h, m in sorted(receipt["per_horizon"].items(), key=lambda kv: int(kv[0])):
        mae, mse, nmae, nmse = (m.get(k) for k in ("MAE", "MSE", "baseline_MAE", "baseline_MSE"))
        per_horizon.append({
            "horizon": int(h), "rows": len(row_ids),
            "model_MAE": mae if _finite(mae) else None, "naive_MAE": nmae if _finite(nmae) else None,
            "model_MSE": mse if _finite(mse) else None, "naive_MSE": nmse if _finite(nmse) else None,
            "MAE": _pair(mae, nmae), "MSE": _pair(mse, nmse)})
    record = {
        "schema": SCHEMA,
        "frozen_metric": {"primary": "MAE", "secondary": "MSE",
                          "frozen": "declared before selection; never switched after seeing results"},
        "artifact": {"candidate_cid": receipt.get("candidate", {}).get("cid"),
                     "model_sha256": receipt["digests"]["model_sha256"],
                     "weights_sha256": receipt["digests"]["weights_sha256"],
                     "predictor_revision": receipt.get("bridge", {}).get("predictor_revision"),
                     "campaign_id": campaign_id},
        "population": {"dataset_id": receipt["data"]["dataset_id"], "asset": asset,
                       "targets": receipt["data"]["target_names"], "rows": len(row_ids),
                       "row_ids_sha256": hashlib.sha256("\n".join(row_ids).encode()).hexdigest(),
                       "first_origin": iso(origins.min()), "last_origin": iso(origins.max()),
                       "sample_hours": sample_hours, "horizon_unit": f"steps of {sample_hours} h",
                       "validation_sha256": receipt["digests"]["validation_sha256"]},
        "scale": {"metric_space": receipt["data"]["metric_space"], "scaler_identity": receipt["data"]["scaler_identity"],
                  "reduction": "mean over rows x targets of absolute / squared error, per horizon"},
        "split": {"provenance": provenance, "test_used": False, "reserved_trading_test": False},
        "naive": {"definition": "persistence: last observed value of each target at the forecast origin, "
                                "repeated for every horizon; same rows, targets, scaler and reduction"},
        "per_horizon": per_horizon}
    record["evidence_sha256"] = hashlib.sha256(canonical(record).encode()).hexdigest()
    return record


def verify(record):
    body = {k: v for k, v in record.items() if k != "evidence_sha256"}
    return hashlib.sha256(canonical(body).encode()).hexdigest() == record.get("evidence_sha256")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--provenance", default="held_out_validation")
    parser.add_argument("--campaign-id")
    parser.add_argument("--asset")
    args = parser.parse_args()
    record = build(json.loads(Path(args.receipt).read_text()), args.validation, provenance=args.provenance,
                   campaign_id=args.campaign_id, asset=args.asset)
    Path(args.out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(record["evidence_sha256"])


if __name__ == "__main__":
    main()
