"""Standalone DMO10 monitored-value replay: the REAL subprocess scoring entrypoint.

Runs in a FRESH interpreter whose environment carries the static-algorithm
determinism flags before TensorFlow initializes (importing ``tools.i6dmo.model``
first guarantees the order; the init-order witness refuses otherwise)::

    python -m tools.i6dmo.replay_monitor --design D --bundle B --unit U [--json]

Loads the trained unit through the canonical rebuild+load_weights path, rebuilds
the fit/inner arrays from the frozen bundle and the unit's fit-only scales, and
recomputes the monitored value through the recorded identical procedure
(witness compared batch by batch). Exit codes: 0 MATCH or TOLERATED (within the
pre-declared bound, reported); 5 typed disposition; 4 unexpected error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tools.i6dmo import model as M  # noqa: F401  (flags before TF; init-order witness)
from tools.i6dmo import train as T


def replay_unit(design_path, bundle_dir, unit_dir):
    unit = Path(unit_dir)
    effective = T.effective_from_design(design_path)
    data = T.load_data_bundle(bundle_dir, effective)
    scales = json.loads((unit / "SCALES.json").read_text(encoding="utf-8"))
    training = json.loads((unit / "TRAINING.json").read_text(encoding="utf-8"))
    record = training["monitor_evaluation"]
    fit_rows, _ = T.trainable_rows(data["fit"])
    inner_rows, _ = T.trainable_rows(data["inner"])
    loaded, document = M.load_dmo(unit / "model")
    fm, im, monitored, tolerated = T.replay_monitored(
        loaded.forecast_model, T.transform(fit_rows, scales), T.transform(inner_rows, scales),
        record, ids=(fit_rows["row_id"], inner_rows["row_id"]))
    recorded = record["monitored"]
    return {"schema": "i6dmo.replay_monitor.v1",
            "status": "MATCH" if monitored == recorded else "TOLERATED",
            "bitwise": monitored == recorded, "monitored": monitored, "recorded": recorded,
            "tolerated_batches": tolerated,
            "tf_imported_before_flags": M.INIT_ORDER["tf_imported_before_flags"],
            "determinism_flags": T.current_determinism_flags(),
            "weights_sha256": document["weights_sha256"], "load_path": document["load_path"],
            "fit_counts": fm["counts"], "inner_counts": im["counts"]}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.i6dmo.replay_monitor")
    parser.add_argument("--design", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        verdict = replay_unit(args.design, args.bundle, args.unit)
    except (T.TrainDisposition, T.ContractError) as exc:
        print(json.dumps({"schema": "i6dmo.replay_monitor.v1", "status": "DIVERGED",
                          "code": exc.code, "detail": str(exc)}))
        return 5
    print(json.dumps(verdict, sort_keys=True) if args.json
          else f"{verdict['status']} monitored={verdict['monitored']!r} recorded={verdict['recorded']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
