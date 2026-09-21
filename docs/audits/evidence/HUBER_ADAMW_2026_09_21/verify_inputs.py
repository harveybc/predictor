"""Read-only materialization check: python verify_inputs.py RUN_ROOT.

This rederives inputs from the delivery; it is not a new temporal-causality test.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
import df_e1_pilot as P


def verify(root):
    design = json.loads((root / "DESIGN.json").read_text())
    source = Path(design["source_run"]["root"])
    original = json.loads((source / "DESIGN.json").read_text())
    deliveries = json.loads((root / "DELIVERIES.json").read_text())["units"]
    panel = Path(deliveries["mae_adam_s1"]["path"])
    assert P.sha_file(panel) == design["source_run"]["panel_sha256"]
    assert P.sha_file(source / "DATA.npz") == design["source_run"]["data_sha256"]
    frame = pd.read_parquet(panel).iloc[slice(*original["dev_subpartition"]["rows"])].reset_index(drop=True)
    contract = P.L.TaskContract(**original["contract"])
    resolved = P.L.resolve(frame, contract)
    enumerated = P.L.enumerate_windows(resolved, contract)
    tensor = P.L.build_tensors(resolved, enumerated, "train", contract, None)
    scaler = P.L.fit_scaler(tensor)
    del tensor
    with np.load(source / "DATA.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["scaler_mean"], scaler["mean"])
        np.testing.assert_array_equal(z["scaler_sd"], scaler["sd"])
        np.testing.assert_array_equal(z["Y"], resolved["targets"][:, 0].astype(np.float64))
        np.testing.assert_array_equal(z["Xs"], ((resolved["inputs"].astype(np.float64)
                                      - scaler["mean"]) / scaler["sd"]).astype(np.float32))
        tr, va = enumerated["splits"]["train"], enumerated["splits"]["validation"]
        origins = np.asarray(tr["origin_ids"], dtype=np.int64)
        origins = origins[np.asarray(tr["target_mask"], dtype=bool)[:, 0]]
        np.testing.assert_array_equal(z["train_origins"], origins)
        origins = np.asarray(va["origin_ids"], dtype=np.int64)
        mask = np.asarray(va["target_mask"], dtype=bool)[:, 0]
        lookup = origins + enumerated["horizon"] - P.DAY
        ok = lookup >= 0
        ok[ok] = np.isfinite(z["Y"][lookup[ok]])
        np.testing.assert_array_equal(z["eval_origins"], origins[mask & ok])
    return {"reconstruction": "EXACT", "contract_sha256": contract.sha256(),
            "checked": ["scaler_mean", "scaler_sd", "Y", "Xs", "train_origins", "eval_origins"]}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2))
