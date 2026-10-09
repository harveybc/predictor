#!/usr/bin/env python3
"""Resumable weekly I6-B branch-donor campaign.

Each unit rebuilds the exact production branch graph and pretrains its branches
from the rolling four-year fit corpus available strictly before one validation
week.  Scoring targets and TEST are intentionally absent from this interface.
The ``worker`` command starts every week in a fresh process so TensorFlow state
cannot accumulate between units.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from predictor_plugins.modular_temporal import build_modular
from tools import fs4_temporal_predictor as P
from tools import i6b_branch_pretraining as I6B
from tools import i6d_weekly_walk_forward as I6D


UTC = dt.timezone.utc
RECEIPT_SCHEMA = "predictor.i6b.weekly_donor_receipt.v1"
STATUS_SCHEMA = "predictor.i6b.weekly_donor_campaign.status.v1"


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _parse(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def load_design(path):
    design = json.loads(Path(path).read_text(encoding="utf-8"))
    return I6D.verify_weekly_design(design)


def prepare_week_windows(store, design, week):
    """Create one value/mask window bank per selected feature.

    Scaling and windows use only rows in ``[fit_start, cutoff)``.  Prior rows
    from the validation calendar may become fit history in later weeks, but no
    row in the week being scored or any future week can enter these arrays.
    """
    members = list(design["members"])
    missing = [member for member in members if member not in store.col]
    if missing:
        raise ValueError(f"missing selected members: {missing}")
    fit_start = _parse(week["fit_start"])
    cutoff = _parse(week["cutoff"])
    try:
        expected = cutoff.replace(year=cutoff.year - 4)
    except ValueError:  # February 29 follows the protocol's calendar clamp.
        expected = cutoff.replace(year=cutoff.year - 4, day=28)
    if fit_start != expected:
        raise ValueError("weekly donor support must span exactly four calendar years")
    lo = int(np.searchsorted(store.ts, int(fit_start.timestamp()), side="left"))
    hi = int(np.searchsorted(store.ts, int(cutoff.timestamp()), side="left"))
    if hi - lo < 24:
        raise ValueError("insufficient pre-cutoff support for 24-hour windows")
    origins = np.arange(lo, hi, dtype="int64")
    spec = P.PredictorSpec(window=24)
    windows = {}
    mapping = {}
    for index, member in enumerate(members):
        branch = f"branch_{index:03d}"
        values = np.asarray(store.X[:, [store.col[member]]], dtype="float64")
        scaler = P.Standardiser.fit(values[lo:hi])
        array, kept = P._inputs_for(
            spec, None, values, scaler, origins,
            timestamps=store.ts, min_timestamp=int(fit_start.timestamp()),
        )
        if kept.size == 0 or int(store.ts[kept[-1]]) >= int(cutoff.timestamp()):
            raise ValueError("weekly windows crossed their exclusive cutoff")
        windows[branch] = np.asarray(array, dtype="float32")
        mapping[branch] = member
    support = {
        "fit_start": week["fit_start"],
        "cutoff_exclusive": week["cutoff"],
        "last_origin_utc": dt.datetime.fromtimestamp(int(store.ts[hi - 1]), UTC).isoformat().replace("+00:00", "Z"),
        "source_digests": dict(sorted(store.digests.items())),
        "outer_validation_scoring_rows_read": False,
    }
    corpus = {
        "dataset_id": f"{store.population}.rolling4y.week_{int(week['ordinal']):03d}",
        "support": support,
    }
    return windows, corpus, mapping


def _receipt_body(design, ordinal, mapping, report):
    week = design["weeks"][ordinal]
    rows = []
    by_name = {row["branch"]: row for row in report["branches"]}
    if set(by_name) != set(mapping):
        raise ValueError("pretraining report does not cover the selected feature mapping")
    for branch, feature in mapping.items():
        row = by_name[branch]
        rows.append({
            "branch": branch, "feature": feature,
            "model_sha256": row["model_sha256"],
            "weights_sha256": row.get("weights_sha256"),
            "data_sha256": row.get("data_sha256"),
            "artifact": row.get("artifact"), "manifest": row.get("manifest"),
        })
    return {
        "schema": RECEIPT_SCHEMA, "status": "COMPLETE",
        "design_sha256": design["design_sha256"], "week_ordinal": ordinal,
        "week": week, "donor_count": len(rows), "donors": rows,
    }


def write_week_receipt(root, design, ordinal, mapping, report):
    body = _receipt_body(design, ordinal, mapping, report)
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    _atomic_json(Path(root) / f"week_{ordinal:03d}" / "WEEK_RECEIPT.json", receipt)
    return receipt


def _load_receipt(path, design, ordinal):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = value.pop("receipt_sha256", None)
    if digest != canonical_sha256(value):
        raise ValueError("receipt digest mismatch")
    if (value.get("schema") != RECEIPT_SCHEMA or value.get("status") != "COMPLETE"
            or value.get("design_sha256") != design["design_sha256"]
            or value.get("week_ordinal") != ordinal
            or value.get("week") != design["weeks"][ordinal]
            or value.get("donor_count") != len(design["members"])):
        raise ValueError("weekly donor receipt identity mismatch")
    expected = [(f"branch_{i:03d}", name) for i, name in enumerate(design["members"])]
    actual = [(row.get("branch"), row.get("feature")) for row in value.get("donors", [])]
    if actual != expected:
        raise ValueError("weekly donor receipt population mismatch")
    return {**value, "receipt_sha256": digest}


def campaign_status(root, design):
    root = Path(root)
    complete = []
    receipts = []
    for ordinal in range(len(design["weeks"])):
        path = root / f"week_{ordinal:03d}" / "WEEK_RECEIPT.json"
        if path.is_file():
            receipts.append(_load_receipt(path, design, ordinal))
            complete.append(ordinal)
    pending = [ordinal for ordinal in range(len(design["weeks"])) if ordinal not in complete]
    body = {
        "schema": STATUS_SCHEMA,
        "status": "COMPLETE" if not pending else "IN_PROGRESS",
        "design_sha256": design["design_sha256"],
        "completed_weeks": len(complete), "total_weeks": len(design["weeks"]),
        "completed_donors": sum(row["donor_count"] for row in receipts),
        "total_donors": len(design["weeks"]) * len(design["members"]),
        "pending_weeks": pending,
        "receipt_sha256": [row["receipt_sha256"] for row in receipts],
    }
    if root.exists():
        _atomic_json(root / "CAMPAIGN_STATUS.json", body)
    return body


def _feature_store(paths, population, members, fit_start, cutoff):
    """Read selected feature columns only, with a strict pre-cutoff predicate."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    members = list(members)
    blocks = []
    digests = {}
    for label, filenames in paths:
        for filename in filenames:
            path = Path(filename)
            parquet = pq.ParquetFile(path)
            available = set(parquet.schema.names)
            selected = [name for name in members if name in available]
            table = pq.read_table(
                path,
                columns=["t_decision_utc", "row_id", *selected],
                filters=[
                    ("t_decision_utc", ">=", pa.scalar(fit_start)),
                    ("t_decision_utc", "<", pa.scalar(cutoff)),
                ],
            )
            blocks.append((label, path, table, selected))
            digests[f"{label}:{path.parent.name}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    columns = {}
    reference_ids = None
    reference_ts = None
    for label in ("base", "history"):
        same_label = [(path, table, selected) for block_label, path, table, selected in blocks
                      if block_label == label]
        if not same_label:
            continue
        label_ids = None
        label_ts = None
        for path, table, selected in same_label:
            ids = table.column("row_id").to_numpy().astype("int64")
            raw_ts = table.column("t_decision_utc").cast("int64").to_numpy()
            unit = str(table.column("t_decision_utc").type)
            scale = 10**9 if "ns" in unit else 10**6 if "us" in unit else 10**3 if "ms" in unit else 1
            ts = (raw_ts // scale).astype("int64")
            if label_ids is None:
                label_ids, label_ts = ids, ts
            elif not np.array_equal(ids, label_ids) or not np.array_equal(ts, label_ts):
                raise ValueError(f"feature batches disagree on {label} row identity")
            for name in selected:
                if name in columns and label == "base":
                    raise ValueError(f"duplicate selected feature column: {name}")
                columns.setdefault(name, {})[label] = table.column(name).to_numpy(zero_copy_only=False)
        if reference_ids is None:
            reference_ids, reference_ts = label_ids, label_ts
        else:
            reference_ids = np.concatenate([reference_ids, label_ids])
            reference_ts = np.concatenate([reference_ts, label_ts])
    missing = [name for name in members if name not in columns]
    if missing:
        raise ValueError(f"missing selected members: {missing}")
    arrays = []
    for name in members:
        parts = []
        for label in ("base", "history"):
            if label in columns[name]:
                parts.append(np.asarray(columns[name][label], dtype="float64"))
            else:
                label_rows = next((len(table) for block_label, _, table, _ in blocks if block_label == label), 0)
                parts.append(np.full(label_rows, np.nan, dtype="float64"))
        arrays.append(np.concatenate(parts))
    order = np.argsort(reference_ts, kind="stable")
    return SimpleNamespace(
        population=population, names=members, col={name: i for i, name in enumerate(members)},
        X=np.column_stack(arrays)[order], ts=reference_ts[order],
        row_ids=reference_ids[order], digests=digests,
    )


def run_week(args):
    design = load_design(args.design)
    ordinal = args.week
    if not 0 <= ordinal < len(design["weeks"]):
        raise ValueError("week is outside the sealed population")
    destination = Path(args.output) / f"week_{ordinal:03d}"
    receipt_path = destination / "WEEK_RECEIPT.json"
    if receipt_path.is_file():
        return _load_receipt(receipt_path, design, ordinal)
    week = design["weeks"][ordinal]
    store = _feature_store(
        (("base", args.train_features), ("history", args.history_features)),
        args.population, design["members"], _parse(week["fit_start"]), _parse(week["cutoff"]),
    )
    windows, corpus, mapping = prepare_week_windows(store, design, design["weeks"][ordinal])
    import tensorflow as tf
    tf.keras.utils.set_random_seed(int(design["config"]["seed"]))
    bundle = build_modular(I6D._branch_only_base_config(design["config"]))
    settings = {
        "seed": args.seed, "max_epochs": args.max_epochs, "patience": args.patience,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "inner_tail_fraction": args.inner_tail_fraction, "purge_rows": 24,
    }
    report = I6B.train_branch_donors(bundle, windows, destination, corpus, settings)
    receipt = write_week_receipt(args.output, design, ordinal, mapping, report)
    campaign_status(args.output, design)
    return receipt


def worker(args):
    design = load_design(args.design)
    common = [
        "--design", args.design, "--output", args.output, "--population", args.population,
        "--seed", str(args.seed), "--max-epochs", str(args.max_epochs),
        "--patience", str(args.patience), "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--inner-tail-fraction", str(args.inner_tail_fraction),
    ]
    for path in args.train_features:
        common.extend(["--train-feature", path])
    for path in args.history_features:
        common.extend(["--history-feature", path])
    for ordinal in range(args.shard_index, len(design["weeks"]), args.shard_count):
        receipt = Path(args.output) / f"week_{ordinal:03d}" / "WEEK_RECEIPT.json"
        if receipt.is_file():
            continue
        command = [sys.executable, str(Path(__file__).resolve()), "run-week", "--week", str(ordinal), *common]
        subprocess.run(command, check=True)
    return campaign_status(args.output, design)


def _data_arguments(parser):
    parser.add_argument("--design", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--population", default="eurusd")
    parser.add_argument("--train-feature", dest="train_features", action="append", required=True)
    parser.add_argument("--history-feature", dest="history_features", action="append", required=True)
    parser.add_argument("--seed", type=int, default=6102026)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--inner-tail-fraction", type=float, default=0.2)


def build_parser():
    parser = argparse.ArgumentParser(description="Weekly exact branch-donor campaign")
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run-week", help="train one sealed weekly donor set")
    _data_arguments(one)
    one.add_argument("--week", type=int, required=True)
    many = commands.add_parser("worker", help="run one resumable process-isolated shard")
    _data_arguments(many)
    many.add_argument("--shard-index", type=int, required=True)
    many.add_argument("--shard-count", type=int, required=True)
    status = commands.add_parser("status", help="verify retained weekly receipts")
    status.add_argument("--design", required=True)
    status.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run-week":
        result = run_week(args)
    elif args.command == "worker":
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("invalid shard index/count")
        result = worker(args)
    else:
        result = campaign_status(args.output, load_design(args.design))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
