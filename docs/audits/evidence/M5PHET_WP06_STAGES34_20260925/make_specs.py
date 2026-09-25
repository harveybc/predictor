"""WP06 stage 3: one m5phet.pipeline.v1 spec per design candidate, differing from `baseline_hand` ONLY in the
representation.

The design job (WP06 stage 2) emitted four candidate representations for the household series. Stage 3 fits one model
per candidate. What a fit needs beyond a representation -- a grouping, an encoder per group, a core, a preprocessing
plan -- is NOT part of the candidate, so it is held at exactly what `baseline_hand` uses (one group holding every meter
column, the `tcn` encoder family of `fused_branches`, no per-feature preprocessor declared). That is the only way the
table can attribute a difference to the representation: a candidate fitted with another grouping or another encoder
would differ from the baseline in three things at once.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = json.loads((HERE.parent / "wp06" / "candidates.json").read_text())["candidates"]
HAND = json.loads(Path(sys.argv[1]).read_text())

FEATURES = list(HAND["features"])

out = HERE / "specs"
out.mkdir(exist_ok=True)
written = []
for candidate in CANDIDATES:
    cid = candidate["candidate_id"]
    spec = {
        "schema": "m5phet.pipeline.v1",
        "stage": f"candidate_{cid}",
        "chosen_by": "HUMAN",
        "why": ("WP06 stage 3: the design candidate %r, fitted with the rest of the pipeline held at exactly what the "
                "baseline_hand stage uses (one group of every meter column, the 'tcn' encoder family of "
                "'fused_branches', no per-feature preprocessor declared, the same epochs, patience, seed, batch size "
                "and deterministic ops). The representation is therefore the only thing that differs from the "
                "baseline, and the only thing a difference in the table can be attributed to." % cid),
        "provenance": "DEVELOPMENT",
        "execution_authorized": False,
        "decisions": [],
        "features": FEATURES,
        "representation": candidate,
        "grouping": json.loads(json.dumps(HAND["grouping"])),
        "extractors": json.loads(json.dumps(HAND["extractors"])),
        "preprocessing": {},
        "core": json.loads(json.dumps(HAND["core"])),
    }
    spec["grouping"]["why"] = ("held at the baseline_hand grouping: every meter column in one block. The candidate is a "
                               "REPRESENTATION, and it declares no grouping; taking another one here would confound "
                               "the contrast this stage exists to measure.")
    path = out / f"spec_candidate_{cid}.json"
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    written.append({"candidate": cid, "windows": candidate["windows"], "lags": candidate["lags"],
                    "features": candidate["features"], "spec": str(path)})
print(json.dumps(written, indent=2))
