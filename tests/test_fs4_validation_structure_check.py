"""No-score structural check of the prepared 2024 parquets: sha256 against a receipt, then the wrapper's own loader, structure only."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tools import fs4_validation_structure_check as V

sys.path.insert(0, str(Path(__file__).parent))
from test_fs4_weekly_wrapper import _consolidated, _inputs  # noqa: E402


def _layout(tmp_path):
    p = _inputs(tmp_path / "src", freq="4h")
    root = tmp_path / "state"
    (root / "validation_2024/eth").mkdir(parents=True)
    vf = pd.read_parquet(p["val_features"])
    vt = pd.read_parquet(p["val_targets"])
    vf["row_id"] = np.arange(len(vf)) + 100000
    vt["row_id"] = vf["row_id"]
    vt.loc[vt.index[-2:], ["Y_s_1h", "Y_s_2h"]] = np.nan
    vf.to_parquet(root / "validation_2024/eth/features_train.parquet")
    vt.to_parquet(root / "validation_2024/eth/targets_train.parquet")
    receipt = {"schema": "fs4.validation2024_data_receipt.v1", "A_eth": {"files": {
        n: {"sha256": hashlib.sha256((root / "validation_2024/eth" / n).read_bytes()).hexdigest()} for n in ("features_train.parquet", "targets_train.parquet")}}}
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    cons = _consolidated()
    (tmp_path / "cons.json").write_text(json.dumps(cons))
    return p, root


def test_sha_mismatch_is_reported_and_structure_is_not_computed(tmp_path, capsys):
    p, root = _layout(tmp_path)
    (root / "validation_2024/eth/targets_train.parquet").write_bytes(b"corrupt")
    rc = V.main(["--receipt", str(tmp_path / "receipt.json"), "--state-root", str(root), "--population", "ETH", "--consolidated", str(tmp_path / "cons.json"),
                 "--train-features", str(p["train_features"]), "--train-targets", str(p["train_targets"]), "--bar-hours", "4"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 3 and out["status"] == "SHA256_MISMATCH_OR_MISSING" and out["sha256"]["mismatched"] == ["validation_2024/eth/targets_train.parquet"] and "structure" not in out


def test_structure_only_no_scores_and_typed_findings(tmp_path, capsys):
    p, root = _layout(tmp_path)
    rc = V.main(["--receipt", str(tmp_path / "receipt.json"), "--state-root", str(root), "--population", "ETH", "--consolidated", str(tmp_path / "cons.json"),
                 "--train-features", str(p["train_features"]), "--train-targets", str(p["train_targets"]), "--bar-hours", "4"])
    out = json.loads(capsys.readouterr().out)
    s = out["structure"]
    assert rc == 0 and out["status"] == "STRUCTURE_OK" and out["sha256"]["all_equal_receipt"] and out["test_read"] is False
    assert s["scored"] is False and s["metrics_computed"] == 0 and s["weeks_expected"] == 52 and s["weeks_with_zero_rows"] == []
    assert s["rows_at_or_after_2025"] == 0 and s["plan_members_missing"] == [] and s["row_ids_unique"] is True
    assert s["row_id_offset"] == 0 or s["row_id_offset"] >= 0
    text = json.dumps(out)
    assert "mae" not in text and "skill" not in text                               # no score vocabulary in the output
