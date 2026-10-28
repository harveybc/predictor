"""Fit-kind child for the real-chain ops tests: writes predictions, trains NOTHING.

It honours the ops child protocol (``--kind fit run-unit --spec --out --progress``) and stands in
ONLY for the trainer. Every later stage of the chain is the real code: the worker claims and stores
this terminal, the REAL report writer (``tools.i6dmo.reporting`` through ``unit_runner``) scores the
predictions against the REAL market bundle with the default loader, and the REAL pair/strategy hook
(``tools.i6dmo.strategy_adapter``) decides from those authenticated reports.

Predictions are ``target + k * (target - origin)`` on the bundle's own validation rows, so the
model error is exactly ``|k|`` times the naive error on every valid head: ``k < 1`` beats the naive
(eligible head), ``k > 1`` loses to it. ``k`` comes from ``I6DMO_FIXTURE_ERROR_<FAMILY>``
(defaults: short 0.1 -> six eligible short heads, long 1.5 -> zero eligible long heads).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from tools.i6dmo import market_grid

DEFAULT_ERROR = {"short": 0.1, "long": 1.5}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("command")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    if args.kind != "fit" or spec.get("kind") != "fit":
        print("KIND_MISMATCH", file=sys.stderr)
        return 4
    counter = os.environ.get("I6DMO_FAKE_COUNTER")
    if counter:
        with open(counter, "a") as handle:
            handle.write(f"fit:{spec['unit_id']}\n")
    family = spec["family"]
    k = float(os.environ.get(f"I6DMO_FIXTURE_ERROR_{family.upper()}", DEFAULT_ERROR[family]))
    design = spec["effective_config"]["long"]["design"] if family == "long" else spec["effective_config"]["design"]
    arrays = market_grid.load_market_split(Path(spec["bundle"]["path"]), "validation")
    target, origin = arrays["target_price"], arrays["origin_price"][:, None]
    pred = target + k * (target - origin)
    out = Path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    path = out / "predictions" / "validation.npz"
    np.savez(path, row_id=arrays["row_id"], horizons_hours=np.asarray(design["horizons_hours"]),
             predicted_price=pred, mask=arrays["mask"])
    Path(args.progress).write_text(json.dumps(dict(done=1, total=1, unit="step", phase="fixture",
                                                   updated_at=time.time())))
    terminal = dict(schema="i6dmo.terminal.v1", unit_id=spec["unit_id"], identity_sha256=spec["identity_sha256"],
                    status="COMPLETE", disposition="PREDICTION_FIXTURE_NO_TRAINING",
                    artifacts={"predictions/validation.npz": {"path": "predictions/validation.npz",
                                                               "sha256": _sha(path)}},
                    costs=dict(wall_seconds=0.0, fit_seconds=0.0, score_seconds=0.0, cgroup_peak_bytes=None),
                    eligibility={}, gates={}, reports=[],
                    trainer_identity=dict(weights_sha256="fixture-no-weights-" + family))
    (out / "TERMINAL.json").write_text(json.dumps(terminal, sort_keys=True))
    print(json.dumps(dict(terminal_sha256=_sha(out / "TERMINAL.json"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
