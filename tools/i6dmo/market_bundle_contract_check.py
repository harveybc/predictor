"""Integration check: the data lane's market bundles satisfy the model lane's consumed contract.

Read-only. Usage: ``python -m tools.i6dmo.market_bundle_contract_check <dir-holding mg2/ and mg12/>``.
Verifies per split that row_id/x/origin_price/target_price/mask/offsets_hours/closure_crossed exist with
the right shapes, that the last offset is exactly 0, the first closure flag 0, closure flags binary,
offsets vary per row (never a fixed elapsed ladder) and increase monotonically, and the head count
matches the family. Exit 0 only on CONTRACT_OK.
"""
import json, sys
import numpy as np
from pathlib import Path
S = Path(sys.argv[1])
REQUIRED = {"row_id", "x", "origin_price", "target_price", "mask", "offsets_hours", "closure_crossed"}
ok = True
for name, slots, heads in (("mg2", 13, [2, 3, 4, 5, 6, 12]), ("mg12", 13, [18, 24, 36, 48, 60, 72])):
    d = S / name
    man = json.loads((d / "BUNDLE.json").read_text())
    rule = man.get("rule") or (man.get("market_time") or {}).get("rule")
    print(f"== {name}: schema {man.get('schema')} rule {rule}")
    print("   manifest fixed elapsed ladder present?", "offsets_hours" in man,
          "| horizons", man.get("horizons_hours"), "| features", len(man.get("features", [])))
    for split in ("fit", "inner", "validation"):
        with np.load(d / f"{split}.npz", allow_pickle=False) as z:
            keys = set(z.files)
            missing = sorted(REQUIRED - keys)
            n = int(z["row_id"].shape[0])
            off, cc, x = z["offsets_hours"], z["closure_crossed"], z["x"]
            checks = dict(
                shapes=bool(off.shape == (n, slots) and cc.shape == (n, slots) and x.shape[1] == slots),
                last_offset_zero=bool(np.all(off[:, -1] == 0)),
                first_closure_zero=bool(np.all(cc[:, 0] == 0)),
                closure_binary=bool(set(np.unique(cc).tolist()) <= {0, 1}),
                offsets_vary_per_row=bool(len(np.unique(off[:, 0])) > 1),
                monotonic_increasing=bool(np.all(np.diff(off, axis=1) > 0)),
                heads_ok=bool(z["target_price"].shape[1] == len(heads)),
                closures_seen=int(cc.sum()),
            )
            bad = bool(missing) or not all(v for k, v in checks.items() if k != "closures_seen")
            ok = ok and not bad
            print(f"   {split}: n={n} missing={missing}", checks)
print("CONTRACT_OK" if ok else "CONTRACT_FAILED")
sys.exit(0 if ok else 3)
