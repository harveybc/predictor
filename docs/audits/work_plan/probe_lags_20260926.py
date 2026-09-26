"""Compare retained train-only estimators without collapsing missing clock rows."""
import json
from pathlib import Path
import numpy as np


def collect():
    root = Path.home() / ".local/state/crispdm-data-foundation/e1_household_successor_v3"
    with np.load(root / "DATA.npz", allow_pickle=False) as z:
        y, origins, window = z["Y"], z["train_origins"], int(z["window"][0])
    v = y[int(origins.min()) - window + 1:int(origins.max()) + 1]
    compressed = v[np.isfinite(v)]
    centered = compressed - compressed.mean()
    rows = []
    for lag in (1, 60, 1440, 10080):
        left, right = v[:-lag], v[lag:]
        mask = np.isfinite(left) & np.isfinite(right)
        fixed_acf = float(np.dot(centered[:-lag], centered[lag:]) / np.dot(centered, centered))
        rows.append({"lag_minutes": lag, "published_fixed_denominator_acf": fixed_acf,
                     "n_over_n_minus_k_rescaled_acf": fixed_acf * len(compressed) / (len(compressed) - lag),
                     "compressed_pearson": float(np.corrcoef(compressed[:-lag], compressed[lag:])[0, 1]),
                     "clock_preserving_pearson": float(np.corrcoef(left[mask], right[mask])[0, 1]),
                     "clock_preserving_pairs": int(mask.sum())})
    return {"scope": "retained training support only; no reserve read", "clock_rows": len(v),
            "missing_rows": int((~np.isfinite(v)).sum()), "rows": rows,
            "claim": "Lag-truncated Pearson is not merely n/(n-k) scaling of fixed-denominator ACF"}


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
