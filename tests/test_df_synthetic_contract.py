"""C128 x C129: synthetic units become sealed common contracts only after
their arrays verify; truth files are bound and labelled as truth."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("df_synthetic_contract", ROOT / "tools/df_synthetic_contract.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    out = tmp_path_factory.mktemp("bank") / "bank"
    subprocess.run([sys.executable, "-B", str(ROOT / "tools/df_synthetic_bank.py"), "--out", str(out), "--limit", "3"],
                   check=True, capture_output=True)
    return out


def units(bank):
    return sorted(p for p in bank.iterdir() if p.is_dir())


def test_units_seal_as_synthetic_contracts(bank):
    for u in units(bank):
        c = S.unit_contract(u)
        assert S.C.validate(c) == []
        assert c["bank"] == "SYNTHETIC" and c["license"]["state"] == "NOT_APPLICABLE_GENERATED"
        rec = json.loads((u / "UNIT.json").read_text())
        assert c["partitions"]["boundaries"] == rec["partitions"]
        roles = {f["name"]: f["role"] for f in c["files"]}
        assert roles["observed_signal.npy"] == "OBSERVED" and roles["clean_signal.npy"] == "CLEAN_TRUTH"
        v = c["variables"][0]
        assert v["unit"]["value"] == "1" and v["unit"]["evidence"][0]["sha256"] == rec["generator"]["code_sha256"]


def test_the_contract_is_deterministic(bank):
    u = units(bank)[0]
    assert S.unit_contract(u) == S.unit_contract(u)


def test_a_tampered_array_refuses(tmp_path, bank):
    import shutil
    u = tmp_path / "unit"
    shutil.copytree(units(bank)[0], u)
    x = np.load(u / "observed_signal.npy")
    x.flat[0] += 1.0
    np.save(u / "observed_signal.npy", x)
    with pytest.raises(S.C.ContractRefusal, match="does not match its recorded digest"):
        S.unit_contract(u)
