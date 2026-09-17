#!/usr/bin/env python3
"""The three contracted toy resources as D3 units, obtained through governance (J3).

Design 07 §4 names the D3 bank as the synthetic D2 bank plus the toy resources the
`governance_smoke` lake serves under real availability contracts (`WINDOW_START`, `4h`/`1h`
completion lag, `PRODUCER_STATEMENT` time zone). They are the only inputs with a duration
contract, so they are the only place the amendment's §2 (durations to samples, exact, never
truncated) is exercised on real bytes rather than on a fixture.

Each resource is DELIVERED, not read: a DATASETS campaign declares every (lake, resource,
role) up front, `GovHttp.governed_download` receives and confirms each one (`X-Delivery-ID`),
and the confirmed bytes are materialised as the arrays `df_snapshot` understands:

    OBSERVED.npy      (T, V) float64, from the numeric columns in file order
    TIMESTAMPS.npy    (T,)   int64 seconds since the epoch, from the resource's time column
    TOY.json          the unit record: contract digest over the delivered bytes, the lake's
                      availability block and frequency, variables, train boundary, delivery id

Timestamps that go backwards refuse the resource; duplicates are kept, as `df_public_contract`
keeps them. The time zone is what the contract says it is (`NAIVE_WALL_CLOCK`): the values are
converted as naive UTC and the record says so, because inventing an offset would be worse.

usage:
  df_d3_toy_resources.py --gov-url URL --api-key-file FILE --campaign-sha256 SHA --unit UNIT
      --lake governance_smoke --resource NAME --role ROLE --contract CONTRACT.json --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FAMILY_OF = {"synthetic_typical_price": "toy_price", "synthetic_ohlc": "toy_ohlc",
             "synthetic_features": "toy_features"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def family_of(resource: str) -> str:
    for prefix, family in FAMILY_OF.items():
        if resource.startswith(prefix):
            return family
    raise SystemExit(f"REFUSED: {resource!r} is not one of the three contracted toy resources")


def materialise(csv_bytes: bytes, *, resource: str, resource_contract: dict, out_dir: Path,
                delivery: dict, unit_id: str) -> dict:
    """Confirmed CSV bytes -> OBSERVED.npy, TIMESTAMPS.npy, TOY.json. Refuses backwards time."""
    import pandas as pd

    time_column = resource_contract.get("event_time_column") or "DATE_TIME"
    frame = pd.read_csv(io.BytesIO(csv_bytes))
    if time_column not in frame.columns:
        raise SystemExit(f"REFUSED: {resource}: no time column {time_column!r}")
    ts = pd.to_datetime(frame[time_column], errors="raise")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")          # NAIVE_WALL_CLOCK, taken as UTC and said so
    seconds = (ts.astype("int64") // 10 ** 9).to_numpy(dtype=np.int64)
    diffs = np.diff(seconds)
    if (diffs < 0).any():
        raise SystemExit(f"REFUSED: {resource}: timestamps go backwards at "
                         f"{int((diffs < 0).sum())} rows")
    numeric = [c for c in frame.columns if c != time_column
               and pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric:
        raise SystemExit(f"REFUSED: {resource}: no numeric column")
    observed = frame[numeric].to_numpy(dtype=np.float64)
    out_dir.mkdir(parents=True, exist_ok=False)
    np.save(out_dir / "OBSERVED.npy", observed, allow_pickle=False)
    np.save(out_dir / "TIMESTAMPS.npy", seconds, allow_pickle=False)
    files = [{"name": name, "bytes": (out_dir / name).stat().st_size,
              "sha256": hashlib.sha256((out_dir / name).read_bytes()).hexdigest(),
              "role": role} for name, role in (("OBSERVED.npy", "OBSERVED"),
                                              ("TIMESTAMPS.npy", "TIMESTAMPS"))]
    n = int(observed.shape[0])
    record = {"schema": "d3_toy_unit.v1", "unit_id": unit_id, "family": family_of(resource),
              "dataset_id": f"toy.governance_smoke.{Path(resource).stem}",
              "lake": "governance_smoke", "resource": resource,
              "delivery": {k: delivery.get(k) for k in
                           ("delivery_id", "sha256", "bytes", "verification_state", "cached",
                            "availability_label", "availability_completion_lag_max",
                            "availability_use", "timezone_evidence", "time_column")},
              "source_sha256": hashlib.sha256(csv_bytes).hexdigest(),
              "files": files, "variables": numeric, "n_samples": n,
              "train_end": int(n * 0.6),
              "timestamp_meaning": "PERIOD_START",
              "timezone": resource_contract.get("timezone", "UNKNOWN"),
              "timezone_note": "NAIVE_WALL_CLOCK taken as UTC for the integer clock; the "
                               "contract's own statement is the authority",
              "resource_contract": {"frequency": resource_contract.get("frequency"),
                                    "availability": resource_contract.get("availability"),
                                    "event_time_column": time_column},
              "duplicate_timestamps": int((diffs == 0).sum()),
              "materialised_utc": now_iso(), "contract_sha256": ""}
    body = {k: v for k, v in record.items() if k != "contract_sha256"}
    record["contract_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    (out_dir / "TOY.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def deliver_and_materialise(gov, *, campaign_sha256: str, unit_id: str, lake: str,
                            resource: str, role: str, cache_dir: Path,
                            resource_contract: dict, out_dir: Path) -> dict:
    status, info = gov.governed_download(campaign_sha256, unit_id, lake, resource, role,
                                         str(cache_dir))
    if status != 200:
        raise SystemExit(f"REFUSED: download of {resource}: http {status}")
    csv_bytes = Path(info["path"]).read_bytes()
    if hashlib.sha256(csv_bytes).hexdigest() != info["sha256"]:
        raise SystemExit(f"REFUSED: {resource}: cached bytes do not match the confirmed digest")
    return materialise(csv_bytes, resource=resource, resource_contract=resource_contract,
                       out_dir=out_dir, delivery=info, unit_id=unit_id)


def main(argv=None) -> int:
    GR = _load("governed_run")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--campaign-sha256", required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--lake", default="governance_smoke")
    parser.add_argument("--resource", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--contract", type=Path, required=True,
                        help="the resource contract as the lake declares it (JSON)")
    parser.add_argument("--cache-dir", type=Path, default=Path(GR.DEFAULT_CACHE).expanduser())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), args.unit)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    record = deliver_and_materialise(gov, campaign_sha256=args.campaign_sha256,
                                     unit_id=args.unit, lake=args.lake, resource=args.resource,
                                     role=args.role, cache_dir=args.cache_dir,
                                     resource_contract=contract, out_dir=args.out)
    print(json.dumps({"unit_id": record["unit_id"], "n_samples": record["n_samples"],
                      "variables": len(record["variables"]),
                      "delivery_id": record["delivery"]["delivery_id"],
                      "verification_state": record["delivery"]["verification_state"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
