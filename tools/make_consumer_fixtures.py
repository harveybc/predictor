#!/usr/bin/env python3
"""Deterministic fixtures with the schemas the four consumers actually read.

N3 of `docs/handoffs/MUSASHI_TO_SATOSHI_TEMPORAL_SEMANTICS_AND_REAL_HOST_ADOPTION_2026_09_14.md`.

These exist so that preprocessor, feature-eng, feature-extractor and predictor can be driven
through the **new lake host** with data that is synthetic by construction: no financial file
is renamed as synthetic, and nothing here carries a producer's rights. Every file is
reproducible from a declared seed, and the manifest records generator, seed, schema, roles,
horizon, frequency, the time columns and the digest of every output.

Availability is not invented and not smuggled into the data either: the consumers read every
column of the file as a feature, so an extra availability column would end up inside their
tensors (feature-extractor failed exactly that way on the first attempt). This generator is
the producer, so it *declares the rule it applied*: a row of a `step`-long grid becomes
available one step after its timestamp. The contracts encode that as `WINDOW_START` with
`completion_lag_max = step` — a construction fact, recorded in the manifest, not a guess
about somebody else's pipeline.

usage:
  make_consumer_fixtures.py --out DIR [--seed 20260914]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

#: Columns of the 4h "typical price" series predictor and preprocessor consume.
PRICE_COLUMNS = ("DATE_TIME", "typical_price")

#: The feature columns feature-extractor's preprocessor plugin expects, in the order its
#: sample data uses them. Values are synthetic; the schema is what makes the run possible.
FEATURE_COLUMNS = (
    "DATE_TIME", "RSI", "MACD", "MACD_Histogram", "MACD_Signal", "EMA", "Stochastic_%K",
    "Stochastic_%D", "ADX", "DI+", "DI-", "ATR", "CCI", "WilliamsR", "Momentum", "ROC",
    "OPEN", "HIGH", "LOW", "CLOSE", "BC-BO", "BH-BL", "BH-BO", "BO-BL", "S&P500_Close",
    "vix_close", "CLOSE_15m_ti",
)

#: The hourly OHLC schema feature-eng's technical-indicator plugin reads. The names are the
#: ones its `forex_15m` header mapping resolves to (`app/config.py`), and its date column is
#: `DATE_TIME` — taken from the code that reads it, not from a convention.
OHLC_COLUMNS = ("DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "volume")


class Stream:
    """A small deterministic generator: no numpy, no global state, reproducible anywhere."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF or 1

    def next(self) -> float:
        # xorshift32: declared, tiny and identical on every machine
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state / 0xFFFFFFFF


def walk(stream: Stream, rows: int, start: float, drift: float, noise: float):
    value = start
    for index in range(rows):
        value += drift + noise * (stream.next() - 0.5)
        yield max(value, 0.01) + 0.05 * math.sin(index / 12.0)


def write_csv(path: Path, header, rows) -> dict:
    body = ",".join(header) + "\n" + "".join(",".join(str(v) for v in row) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")
    return {"file": path.name, "rows": len(rows), "columns": list(header),
            "sha256": hashlib.sha256(body.encode()).hexdigest(), "bytes": len(body.encode())}


def stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def price_series(name: str, seed: int, rows: int, start: datetime, step: timedelta,
                 out: Path) -> dict:
    """`DATE_TIME,typical_price,available_time` on a fixed grid."""
    stream = Stream(seed)
    values = list(walk(stream, rows, 1.0, 0.0002, 0.01))
    lines = []
    for index, value in enumerate(values):
        moment = start + index * step
        lines.append((stamp(moment), f"{value:.6f}"))
    record = write_csv(out / name, PRICE_COLUMNS, lines)
    record.update({"seed": seed, "frequency": f"{int(step.total_seconds())}s",
                   "event_time_column": "DATE_TIME", "available_time_column": "DATE_TIME",
                   "first_event": lines[0][0], "last_event": lines[-1][0],
                   "availability_rule": "a row is available one step after its timestamp, by "
                                        "construction of this generator"})
    return record


def feature_series(name: str, seed: int, rows: int, start: datetime, step: timedelta,
                   out: Path) -> dict:
    """The 27-column normalised schema, plus the availability column."""
    stream = Stream(seed)
    lines = []
    base = list(walk(stream, rows, 0.5, 0.0, 0.02))
    for index, value in enumerate(base):
        moment = start + index * step
        row = [stamp(moment)]
        for position in range(1, len(FEATURE_COLUMNS)):
            row.append(f"{(value + 0.01 * position + 0.001 * (index % 7)) % 1.0:.6f}")
        lines.append(row)
    record = write_csv(out / name, FEATURE_COLUMNS, lines)
    record.update({"seed": seed, "frequency": f"{int(step.total_seconds())}s",
                   "event_time_column": "DATE_TIME", "available_time_column": "DATE_TIME",
                   "first_event": lines[0][0], "last_event": lines[-1][0]})
    return record


def ohlc_series(name: str, seed: int, rows: int, start: datetime, step: timedelta,
                out: Path) -> dict:
    """Hourly OHLC for feature-eng's technical indicators."""
    stream = Stream(seed)
    lines = []
    for index, close in enumerate(walk(stream, rows, 100.0, 0.01, 0.8)):
        moment = start + index * step
        spread = 0.2 + 0.3 * stream.next()
        open_ = close - 0.5 * spread
        lines.append((stamp(moment), f"{open_:.5f}", f"{close + spread:.5f}",
                      f"{max(open_ - spread, 0.01):.5f}", f"{close:.5f}",
                      f"{1000 + 500 * stream.next():.2f}"))
    record = write_csv(out / name, OHLC_COLUMNS, lines)
    record.update({"seed": seed, "frequency": f"{int(step.total_seconds())}s",
                   "event_time_column": "DATE_TIME", "available_time_column": "DATE_TIME",
                   "first_event": lines[0][0], "last_event": lines[-1][0]})
    return record


def contract_of(record: dict) -> dict:
    """The contract this generator's construction supports: a row of a grid is complete one
    step after its own timestamp, and the timestamps are naive wall clock by construction."""
    seconds = int(record["frequency"].rstrip("s"))
    hours = seconds // 3600
    return {"event_time_column": record["event_time_column"],
            "available_time_column": record["available_time_column"],
            "timezone": "NAIVE_WALL_CLOCK", "time_unit": None,
            "frequency": f"{hours}h" if seconds % 3600 == 0 else record["frequency"],
            "availability": {"label": "WINDOW_START", "completion_lag_max": f"{hours}h",
                             "timezone_evidence": "PRODUCER_STATEMENT",
                             "use_class": "OFFLINE_DAY_GRANULAR"}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--rows", type=int, default=900)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc).replace(tzinfo=None)
    four_hours, one_hour = timedelta(hours=4), timedelta(hours=1)

    files, roles = {}, {}
    # predictor and preprocessor: three 4h price series, train / validation / test
    for index, part in enumerate(("train", "validation", "test")):
        name = f"synthetic_typical_price_4h_{part}.csv"
        files[name] = price_series(name, args.seed + index, args.rows, start, four_hours, out)
        roles.setdefault("predictor", {})[f"x_{part}_file"] = name
        roles["predictor"][f"y_{part}_file"] = name
    roles["preprocessor"] = {"input_file": "synthetic_typical_price_4h_train.csv"}

    # feature-extractor: the 27-column normalised schema, three partitions
    for index, part in enumerate(("train", "validation", "test")):
        name = f"synthetic_features_4h_{part}.csv"
        files[name] = feature_series(name, args.seed + 10 + index, args.rows, start, four_hours, out)
        roles.setdefault("feature-extractor", {})[f"x_{part}_file"] = name
        roles["feature-extractor"][f"y_{part}_file"] = name

    # feature-eng: hourly OHLC
    name = "synthetic_ohlc_1h.csv"
    files[name] = ohlc_series(name, args.seed + 20, args.rows * 2, start, one_hour, out)
    roles["feature-eng"] = {"input_file": name}

    manifest = {
        "schema": "consumer_fixtures.v1",
        "generator": "predictor/tools/make_consumer_fixtures.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed, "rows_per_series": args.rows,
        "synthetic": "every value is produced by the declared generator from the declared "
                     "seed; no financial file is copied or renamed here",
        "grid_start_utc": iso(start), "files": files, "roles": roles,
        "horizon": {"predictor": "the configuration's predicted horizons; the fixture itself "
                                 "carries no target column"},
        "contracts": {name: contract_of(record) for name, record in files.items()},
    }
    manifest["contracts_sha256"] = {
        name: hashlib.sha256(json.dumps(contract, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
        for name, contract in manifest["contracts"].items()}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out).replace(str(Path.home()), "~"),
                      "files": {n: r["sha256"][:16] for n, r in files.items()},
                      "rows": {n: r["rows"] for n, r in files.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
