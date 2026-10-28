#!/usr/bin/env python3
"""NON_GOVERNING real-data support probe; never fits or ranks a model."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.code_root).resolve()))
    from tools.pool_selection_v2 import Engine, purged_folds, ts_hash
    data, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(admitted=str(data / "admitted"), csv_dir=str(data / "sel15"),
                             store=[str(data / "eurusd_h1_layer1_pool_r4"), str(data / "eurusd_h1_layer2_pool")],
                             catalog=str(data / "eurusd_h1_catalog/FEATURE_CATALOG.csv"), out=str(out),
                             fit_end="2024-01-01", stage="contrast", shard="0/2")
    engine = Engine(config)
    # This selects the witnessed task for geometry only, not a claim of informative fitness.
    engine.informative = ["opener3"]
    result = []
    for unit in engine.arms():
        key, stage, task, model, columns, fold, count, _, meta = unit
        if not (stage == "contrast" and task == "opener3" and meta["family"] == "px.er"
                and meta["ctx"] == "full" and fold == 0 and meta["arm"] in ("none", "with")):
            continue
        selected = engine.row_set(task, meta["row_cols"])
        bars = engine.bars[selected]
        _, train, val = list(purged_folds(len(selected), count))[fold]
        result.append({"key": key, "arm": meta["arm"], "n_input_columns": len(columns),
                       "n_row_columns": len(meta["row_cols"]), "rows_train": len(train), "rows_val": len(val),
                       "support_hash": ts_hash(engine.ts_ns[bars]), "val_hash": ts_hash(engine.ts_ns[bars[val]])})
    if len(result) != 2 or len({(r["support_hash"], r["val_hash"], r["rows_train"], r["rows_val"]) for r in result}) != 1:
        raise RuntimeError("PAIRING_PROBE_FAILED")
    report = {"scope": "NON_GOVERNING_SUPPORT_GEOMETRY_ONLY", "model_fits": 0, "store_sha256": engine.pins, "arms": result}
    (out / "PROBE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
