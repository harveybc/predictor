"""TRAIN-only, selected-set I5-P reconstruction probe on governed parquet bytes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tools.i5p_reconstructed_inputs import (
    PINNED_EXTRACTOR_ROOT, data_sha256, load_pinned_extractor,
    reconstruct_selected_inputs, sha256_file,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--target", default="Y_s_1h")
    parser.add_argument("--feature-parquet", action="append", required=True)
    parser.add_argument("--target-parquet", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--extractor-root", default=str(PINNED_EXTRACTOR_ROOT))
    parser.add_argument("--output", help="write an atomic JSON receipt in addition to stdout")
    args = parser.parse_args(argv)
    if args.n < 1:
        raise ValueError("N_MUST_BE_POSITIVE")
    X, _ = load_pinned_extractor(args.extractor_root)
    frozen = json.loads(Path(args.freeze).read_text())
    digest = X.digest({k: v for k, v in frozen.items() if k not in ("freeze_sha256", "frozen_utc")})
    if digest != frozen.get("freeze_sha256"):
        raise ValueError("FREEZE_DIGEST_MISMATCH")
    members = tuple(frozen["winners"]["EURUSD"][args.target]["RAW"]["members"])
    if not members or len(members) != len(set(members)):
        raise ValueError("FROZEN_MEMBERS_INVALID")
    identity = "phase1-eurusd-final:94d20c038d55e152"
    expected = X.CORPORA[identity]["files"]
    files = {}
    for name in args.feature_parquet:
        path = Path(name)
        role = f"eurusd_ps1_{path.parent.name}"
        if role not in expected or role in files or sha256_file(path) != expected[role]:
            raise ValueError(f"FEATURE_PARQUET_IDENTITY_MISMATCH: {path}")
        files[role] = path
    if set(files) != set(expected):
        raise ValueError("FEATURE_PARQUET_POPULATION_INCOMPLETE")
    ids = ts = None
    columns = {}
    for role, path in sorted(files.items()):
        names = set(pq.ParquetFile(path).schema_arrow.names)
        wanted = [feature for feature in members if feature in names]
        table = pq.read_table(path, columns=["row_id", "t_decision_utc", *wanted])
        current_ids = table["row_id"].to_numpy(zero_copy_only=False).astype("int64")
        current_ts = table["t_decision_utc"].to_numpy(zero_copy_only=False).astype("datetime64[s]").astype("int64")
        if ids is None:
            ids, ts = current_ids, current_ts
        elif not np.array_equal(ids, current_ids) or not np.array_equal(ts, current_ts):
            raise ValueError(f"FEATURE_ROW_ALIGNMENT_MISMATCH: {role}")
        for feature in wanted:
            if feature in columns:
                raise ValueError(f"DUPLICATE_FEATURE: {feature}")
            columns[feature] = table[feature].to_numpy(zero_copy_only=False).astype("float64")
    if set(columns) != set(members):
        raise ValueError("SELECTED_FEATURE_MISSING")
    raw = np.column_stack([columns[name] for name in members])
    targets = pq.read_table(args.target_parquet, columns=["row_id", "t_decision_utc", args.target])
    target_ids = targets["row_id"].to_numpy(zero_copy_only=False).astype("int64")
    target_ts = targets["t_decision_utc"].to_numpy(zero_copy_only=False).astype("datetime64[s]").astype("int64")
    if not np.array_equal(ids, target_ids) or not np.array_equal(ts, target_ts):
        raise ValueError("TARGET_ROW_ALIGNMENT_MISMATCH")
    y = targets[args.target].to_numpy(zero_copy_only=False).astype("float64")
    root = Path(args.results_root)
    refs = {}
    fold = None
    for name in members:
        claim = {"schema": "fs4.extractibility.task.v1", "population_id": "EURUSD", "identity": identity,
                 "feature_id": name, "fold_id": "inner_2023", "arm": "TRAINED_ENCODER_V2", "seed": 0}
        path = root / X.task_digest(claim) / "result.json"
        if not path.is_file():
            raise ValueError(f"DONOR_MISSING: {name}")
        rec = json.loads(path.read_text())
        if rec.get("task_id") != X.task_digest(claim):
            raise ValueError(f"DONOR_IDENTITY_MISMATCH: {name}")
        if fold is None:
            fold = rec["fold"]["val"]
        elif fold != rec["fold"]["val"]:
            raise ValueError("DONOR_FOLD_MISMATCH")
        refs[name] = {"terminal": path, "sha256": sha256_file(path)}
    eligible = np.flatnonzero((ts >= fold[0]) & (ts < fold[1]) & np.isfinite(y) & np.isfinite(raw).all(axis=1))
    eligible = eligible[eligible >= 23]
    if len(eligible) < args.n:
        raise ValueError(f"TOO_FEW_COMPLETE_ORIGINS: {len(eligible)}")
    rows = eligible[-args.n:]
    truth = y[rows].copy()
    naive = np.zeros(args.n, dtype="float64")
    out = reconstruct_selected_inputs(timestamps=ts, row_ids=ids, values=raw, columns=members,
                                      scored_rows=rows, target=truth, naive=naive, checkpoints=refs,
                                      expected_data_sha256=data_sha256(ts, ids, raw, members, rows, truth, naive),
                                      extractor_root=args.extractor_root)
    if not np.array_equal(out.row_ids, ids[rows]) or not np.array_equal(out.target, truth) or \
            not np.array_equal(out.naive, naive):
        raise ValueError("TARGET_OR_SUPPORT_CHANGED")
    report = {"schema": "i5p.real_train_probe.v1", "split": "TRAIN", "population": "EURUSD",
              "target": args.target, "n": args.n, "feature_ids": list(members),
              "first_row_id": int(out.row_ids[0]), "last_row_id": int(out.row_ids[-1]),
              "input_data_sha256": data_sha256(ts, ids, raw, members, rows, truth, naive),
              "target_file_sha256": sha256_file(args.target_parquet),
              "mean_abs_reconstruction_change": float(np.mean(np.abs(out.values - raw[rows]))),
              "reconstruction_receipt": out.receipt}
    report["result_sha256"] = X.digest(report)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
