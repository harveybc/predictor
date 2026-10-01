"""Lane H: compare two REPLAY_DIGESTS documents (two workers). DEVELOPMENT.

Exact layer: the Kalman artifacts, fitted-state digests and output digests, the input identities, and every ELIGIBLE arm's
input matrices. The non-causal smoother control is excluded from the exact layer (it is a rejected control and uses LAPACK
matrix algebra, so it is not expected to be bitwise portable). Numeric layer: prediction digests of the ridge learner
(LAPACK eigen-decomposition, not bitwise portable across CPUs; agreement is then judged on the metrics by value)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def compare(a: dict, b: dict) -> dict:
    diffs = []
    for k in sorted(set(a["inputs"]) | set(b["inputs"])):
        if a["inputs"].get(k) != b["inputs"].get(k):
            diffs.append(["inputs", k])
    for v in sorted(set(a["kalman"]) | set(b["kalman"])):
        for g in sorted(set(a["kalman"].get(v, {})) | set(b["kalman"].get(v, {}))):
            for f in ("artifact_sha256", "fitted_state_digest", "output_digest"):
                if a["kalman"].get(v, {}).get(g, {}).get(f) != b["kalman"].get(v, {}).get(g, {}).get(f):
                    diffs.append(["kalman", v, g, f])
    excluded = sorted(k for k in set(a["arms_exact_inputs"]) | set(b["arms_exact_inputs"]) if "NONCAUSAL" in k)
    for k in sorted(set(a["arms_exact_inputs"]) | set(b["arms_exact_inputs"])):
        if k in excluded:
            continue
        if a["arms_exact_inputs"].get(k) != b["arms_exact_inputs"].get(k):
            diffs.append(["arms_exact_inputs", k])
    num = sorted(k for k in set(a["arms_numeric_predictions"]) | set(b["arms_numeric_predictions"])
                 if a["arms_numeric_predictions"].get(k) != b["arms_numeric_predictions"].get(k))
    return {"exact_core_equal": not diffs, "exact_differences": diffs, "noncausal_control_excluded": excluded,
            "numeric_predictions_equal": not num, "numeric_differing": num}


def main():
    a, b = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    print(json.dumps(compare(a, b), indent=1))


if __name__ == "__main__":
    main()
