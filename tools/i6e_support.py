"""Audit origin-anchored 72h resolution support without fitting or reading TEST.

Usage: python -m tools.i6e_support --train-targets <parquet> --output <json>
Only timestamps are read. Missing market hours are never synthesized.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def support_report(timestamps):
    """Count exact past endpoints and exact future support on a common population."""
    ts = np.asarray(timestamps, dtype=np.int64)
    if ts.ndim != 1 or not len(ts) or np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps must be nonempty and strictly increasing")
    future = np.isin(ts + 72 * 3600, ts)
    common = future.copy()
    input_common = np.ones(len(ts), dtype=bool)
    arms = []
    for spacing, samples in ((72, 3), (36, 5), (18, 9)):
        support = np.ones(len(ts), dtype=bool)
        for j in range(samples):
            support &= np.isin(ts - j * spacing * 3600, ts)
        common &= support
        input_common &= support
        arms.append(dict(spacing_hours=spacing, samples=samples, span_hours=144,
                         input_origins=int(support.sum()),
                         exact_future_origins=int((support & future).sum())))
    eligible = ts[common].astype("<i8")
    return dict(schema="predictor.i6e.support.v1", horizon_hours=72,
                status="SUPPORTED" if len(eligible) else "EMPTY_SUPPORT",
                input_rows=len(ts), arms=arms, common_origins=len(eligible),
                masked_clock_origins=int((future & (ts >= ts[0] + 144 * 3600)).sum()),
                input_missing_origins=int((~input_common).sum()),
                future_missing_origins=int((~future).sum()),
                common_timestamps_sha256=hashlib.sha256(eligible.tobytes()).hexdigest(),
                support_only=True, model_fitted=False, test_read=False)


def main():
    """Write a reproducible TRAIN-only support report and input-file identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(args.train_targets, columns=["t_decision_utc"])
    ts = table.column(0).cast(pa.timestamp("s", tz="UTC")).cast(pa.int64()).to_numpy()
    report = support_report(ts)
    value = hashlib.sha256()
    with args.train_targets.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    report["input_sha256"] = value.hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
