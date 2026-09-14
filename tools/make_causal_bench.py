#!/usr/bin/env python3
"""A small synthetic trajectory with known structure, split chronologically.

P2 of `docs/handoffs/MUSASHI_TO_SATOSHI_CAUSAL_PIPELINE_AND_OFFLINE_DOIN_2026_09_14.md`.

Three seeds are not three temporal partitions: this writes **one** trajectory and cuts it in
time — train, then an embargo of one horizon, then validation, then an embargo, then test —
so that a test window can never overlap a training window through a lookback or a horizon.

The trajectory carries the components the causal tests need to distinguish: a trend, two
periodic components, level jumps and spikes, noise of declared variance, and a missing
stretch left missing. Availability is a **simulated** clock with a declared rule (a row of a
step-long grid is complete one step after its timestamp), never an observation of a data
provider.

The successor of the deployed fixture set: it does not overwrite it, and carries its own
roles and contracts.

usage:
  make_causal_bench.py --out DIR [--seed 20260915] [--rows 1440] [--horizon 6] [--lookback 24]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

COLUMNS = ("DATE_TIME", "level", "seasonal_daily", "seasonal_weekly", "jump", "spike",
           "noise", "signal", "observed")
#: What each column is for, so a test can say which regime it exercises.
REGIMES = {
    "trend": "level",
    "periodic_daily": "seasonal_daily",
    "periodic_weekly": "seasonal_weekly",
    "jumps": "jump",
    "extremes": "spike",
    "noise": "noise",
    "missingness": "observed",
}


class Stream:
    """xorshift32: declared, tiny, identical on every machine."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF or 1

    def uniform(self) -> float:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state / 0xFFFFFFFF

    def normal(self) -> float:
        # Box–Muller from two uniforms; no numpy, no global state
        u1 = max(self.uniform(), 1e-12)
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def trajectory(seed: int, rows: int, step_hours: int, noise_sd: float) -> list:
    stream = Stream(seed)
    per_day = max(1, 24 // step_hours)
    per_week = per_day * 7
    missing_from, missing_to = int(rows * 0.62), int(rows * 0.64)
    jump_at = {int(rows * 0.35): 4.0, int(rows * 0.75): -3.0}
    spike_at = {int(rows * 0.28): 9.0, int(rows * 0.55): -8.0, int(rows * 0.88): 7.0}
    level, jumped = 10.0, 0.0
    out = []
    for index in range(rows):
        level += 0.004                                   # trend
        jumped += jump_at.get(index, 0.0)                # level jumps, persistent
        daily = 1.5 * math.sin(2 * math.pi * index / per_day)
        weekly = 0.8 * math.sin(2 * math.pi * index / per_week)
        spike = spike_at.get(index, 0.0)                 # extremes, transient
        noise = noise_sd * stream.normal()
        signal = level + jumped + daily + weekly
        observed = signal + spike + noise
        present = not (missing_from <= index < missing_to)
        out.append({"level": level + jumped, "seasonal_daily": daily, "seasonal_weekly": weekly,
                    "jump": jumped, "spike": spike, "noise": noise, "signal": signal,
                    "observed": observed if present else None})
    return out


def write_csv(path: Path, rows: list, start: datetime, step: timedelta) -> dict:
    lines = [",".join(COLUMNS)]
    for index, row in enumerate(rows):
        moment = start + index * step
        values = [moment.strftime("%Y-%m-%d %H:%M:%S")]
        for column in COLUMNS[1:]:
            value = row[column]
            values.append("" if value is None else f"{value:.6f}")
        lines.append(",".join(values))
    body = "\n".join(lines) + "\n"
    path.write_text(body, encoding="utf-8")
    return {"file": path.name, "rows": len(rows), "columns": list(COLUMNS),
            "sha256": hashlib.sha256(body.encode()).hexdigest(), "bytes": len(body.encode()),
            "first_event": lines[1].split(",")[0], "last_event": lines[-1].split(",")[0],
            "missing_rows": sum(1 for r in rows if r["observed"] is None)}


def contract(step_hours: int) -> dict:
    return {"event_time_column": "DATE_TIME", "available_time_column": "DATE_TIME",
            "timezone": "NAIVE_WALL_CLOCK", "time_unit": None, "frequency": f"{step_hours}h",
            "availability": {"label": "WINDOW_START", "completion_lag_max": f"{step_hours}h",
                             "timezone_evidence": "PRODUCER_STATEMENT",
                             "use_class": "OFFLINE_DAY_GRANULAR"}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--rows", type=int, default=1440)
    parser.add_argument("--step-hours", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=6, help="rows predicted ahead")
    parser.add_argument("--lookback", type=int, default=24, help="rows a window looks back")
    parser.add_argument("--noise-sd", type=float, default=0.25)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 1)
    step = timedelta(hours=args.step_hours)
    rows = trajectory(args.seed, args.rows, args.step_hours, args.noise_sd)

    # chronological split with an embargo of one horizon on each side of a boundary, so a
    # window of one partition can never see a row of the next
    embargo = args.horizon + args.lookback
    train_end = int(args.rows * 0.6)
    validation_start = train_end + embargo
    validation_end = validation_start + int(args.rows * 0.15)
    test_start = validation_end + embargo
    partitions = {"train": (0, train_end),
                  "validation": (validation_start, validation_end),
                  "test": (test_start, args.rows)}
    files = {}
    for name, (lo, hi) in partitions.items():
        record = write_csv(out / f"causal_{name}.csv", rows[lo:hi], start + lo * step, step)
        record.update({"partition": name, "row_range": [lo, hi],
                       "first_index": lo, "last_index": hi - 1})
        files[record["file"]] = record
    whole = write_csv(out / "causal_whole.csv", rows, start, step)
    whole["partition"] = "whole"
    files[whole["file"]] = whole

    manifest = {
        "schema": "causal_bench.v1",
        "generator": "predictor/tools/make_causal_bench.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed, "rows": args.rows, "step_hours": args.step_hours,
        "horizon": args.horizon, "lookback": args.lookback, "embargo_rows": embargo,
        "noise_sd": args.noise_sd,
        "partitions": {name: {"row_range": list(span)} for name, span in partitions.items()},
        "regimes": REGIMES,
        "availability": "SIMULATED: a row of the grid is complete one step after its "
                        "timestamp, by construction of this generator; it is not an "
                        "observation of any data provider",
        "column_roles": {"time": "DATE_TIME",
                         "features": ["observed"],
                         "targets": ["signal"],
                         "metadata": ["level", "seasonal_daily", "seasonal_weekly", "jump",
                                      "spike", "noise"]},
        "files": files,
        "contracts": {name: contract(args.step_hours) for name in files},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out).replace(str(Path.home()), "~"),
                      "partitions": {k: v["row_range"] for k, v in
                                     manifest["partitions"].items()},
                      "embargo_rows": embargo,
                      "files": {k: v["sha256"][:16] for k, v in files.items()},
                      "missing_rows": {k: v["missing_rows"] for k, v in files.items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
