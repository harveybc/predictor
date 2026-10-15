"""Independent reviewer control RP02: per-head naive pairing (identical rows, original price).

Positive control plus two reviewer fail-closed mutations the candidate's own
tests do not already exercise: a mutated ROW ID and a mutated price SCALE.
Exit 0 iff pairing holds and both mutations fail typed.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from tests.i6dmo import _synth  # noqa: E402
from tools.i6dmo import data, naive  # noqa: E402

results = {}


def check(name, ok, detail):
    results[name] = {"ok": bool(ok), "detail": detail}


design = _synth.load_design()
work = Path(sys.argv[1])
src = _synth.make_input(work / "input")
bundle = data.load_input(src, design)
table = data.build_origin_table(bundle, design)
arrays = data.materialize(bundle, table, "VALIDATION")
horizons = tuple(arrays.horizons)
pred = np.where(arrays.mask, arrays.price_origin[:, None] + 2e-4, np.nan)

# --- positive: paired metrics agree; naive on identical row ids, original price ---
report = naive.paired_metrics(arrays, pred, design, input_dir=src,
                              checkpoint="reviewer-rp02", sealed_at="2026-10-14T00:00:00+00:00")
replay = naive.naive_replay(src, arrays, horizons)
same_rows = all(report["per_horizon"][str(h)]["row_ids_sha256"]
                == replay[str(h)]["row_ids_sha256"] for h in horizons)
manual_ok = True
for j, h in enumerate(horizons):
    m = arrays.mask[:, j]
    manual_naive = math.fsum(np.abs(arrays.y_price[m, j] - arrays.price_origin[m]).tolist()) / int(m.sum())
    if abs(report["per_horizon"][str(h)]["naive_mae"] - manual_naive) > 1e-12 * max(1.0, manual_naive):
        manual_ok = False
check("paired_identical_rows_original_price",
      report["agreement"]["status"] == "AGREE" and report["scale"] == "original_price"
      and same_rows and manual_ok,
      f"agreement={report['agreement']['status']} scale={report['scale']} "
      f"row_ids_match={same_rows} manual_naive_match={manual_ok}")

# --- reviewer mutation 1: one mutated row id must fail typed, never re-pair ---
mutated = type(arrays)(**{**arrays.__dict__})
rids = np.asarray(mutated.row_ids).copy()
j = 0
row = int(np.flatnonzero(arrays.mask[:, j])[5])
rids[row] = "EURUSD|2030-01-01T00:00:00+00:00"  # a bar that does not exist in INPUT.npz
mutated.row_ids = rids
try:
    naive.naive_replay(src, mutated, horizons)
    check("mutated_row_id_rejected", False, "replay accepted a row id with no retained bar")
except naive.NaiveReplayMismatch as exc:
    check("mutated_row_id_rejected", True, f"typed NaiveReplayMismatch: {str(exc)[:90]}")

# --- reviewer mutation 2: a scale mutation of the retained prices must fail typed ---
scaled = work / "scaled"
scaled.mkdir()
for name in ("CLOCK.json", "CALENDAR.json", "INPUT.npz"):
    (scaled / name).write_bytes((src / name).read_bytes())
with np.load(scaled / "INPUT.npz") as z:
    price = z["price"].copy()
_synth.rewrite_input(scaled, price=price * 1.0001)  # a different price scale on disk
try:
    naive.paired_metrics(arrays, pred, design, input_dir=scaled,
                         checkpoint="reviewer-rp02", sealed_at="2026-10-14T00:00:00+00:00")
    check("scaled_prices_rejected", False, "paired metrics accepted a rescaled retained price")
except naive.NaiveReplayMismatch as exc:
    check("scaled_prices_rejected", True, f"typed NaiveReplayMismatch: {str(exc)[:90]}")

ok = all(v["ok"] for v in results.values())
print(json.dumps({"probe": "RP02_naive_pairing", "ok": ok, "controls": results}, indent=1))
sys.exit(0 if ok else 1)
