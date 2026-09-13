"""C160: wavelet_mad is OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL. It yields one
aggregate figure per training segment and never a per-timestamp value; no
operator, router, selector, agent, feature-matrix or transform module reaches
it; and every SNR calibration figure is numerically identical to the code at
ad30cf3."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import df_snr as d  # noqa: E402
import df_snapshot as snap  # noqa: E402
from test_df_snr import _fit_snapshot, _train_snapshot, _write_units  # noqa: E402

BASE = "ad30cf3"


# ------------------------------------------------------------- structure
def test_contract_state_and_no_public_kernel():
    assert d.ESTIMATORS["wavelet_mad"]["contract_state"] == "OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL"
    assert all("kernel" not in e for e in d.ESTIMATORS.values())
    assert not any(callable(v) for v in d.ESTIMATORS["wavelet_mad"].values())
    assert d.estimator_declarations()["wavelet_mad"]["contract_state"] == "OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL"
    assert not hasattr(d, "_nv_wavelet_mad")


def test_call_graph_no_consumer_reaches_the_offline_kernel():
    sys.modules.pop("df_causal_battery", None)
    spec = importlib.util.spec_from_file_location("df_causal_battery", ROOT / "tools" / "df_causal_battery.py")
    bat = importlib.util.module_from_spec(spec)
    sys.modules["df_causal_battery"] = bat
    spec.loader.exec_module(bat)
    ev = bat.snr_isolation_evidence(ROOT / "tools")
    assert ev["passed"], ev
    assert ev["modules_loading_df_snr"] == [] and ev["kernel_identifier_references_outside_df_snr"] == []
    assert ev["df_snr_kernel_referenced_by"] == ["_KERNELS"]
    assert ev["df_snr_KERNELS_referenced_by"] == ["_KERNELS", "_point"]    # its definition and one consumer


# -------------------------------------------------------------- behaviour
def test_negative_consumer_cannot_build_per_timestamp_values():
    rng = np.random.default_rng(0)
    x = np.sin(np.arange(512) / 9.0) + rng.standard_normal(512) * 0.3
    params = dict(d.ESTIMATORS["wavelet_mad"]["parameters"])
    per_t = []
    with pytest.raises(d.SnrRefusal, match="never yields per-timestamp values"):
        for t in range(63, 512):
            per_t.append(d._KERNELS["wavelet_mad"](x[t - 63:t + 1], params))
    assert per_t == []
    with pytest.raises(d.SnrRefusal, match="a bare array never grants it"):
        [d.estimate(x[t - 63:t + 1], (0, 64), "wavelet_mad", do_bootstrap=False) for t in range(63, 512)]
    # an expanding prefix of the train partition is not a training segment
    with pytest.raises(d.SnrRefusal, match="one figure per training segment"):
        for t in range(64, 300):
            d.estimate_offline_train_diagnostic(_fit_snapshot(x, 300, end=t + 1), 0, do_bootstrap=False)
    # a calibration-role snapshot is not a training segment either
    c, L = _train_snapshot(x, 300)
    with pytest.raises(d.SnrRefusal, match="one figure per training segment"):
        d.estimate_offline_train_diagnostic(snap.FitSnapshot.from_contract(c, "CALIBRATION", L), 0)
    with pytest.raises(d.SnrRefusal, match="FitSnapshot"):
        d.estimate_offline_train_diagnostic(x[:300], 0)
    one = d.estimate_offline_train_diagnostic(_fit_snapshot(x, 300), 0, do_bootstrap=False)
    assert one["status"] == d.STATUS_OK and one["train"] == [0, 300]
    rows = d.estimate_real(x[None], (0, 300), estimators=["wavelet_mad"])
    assert rows[0]["status"] == d.STATUS_NI and rows[0]["reason"].startswith("refused")


# ---------------------------------------------------------------- parity
@pytest.fixture(scope="module")
def old_snr(tmp_path_factory):
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE}:tools/df_snr.py"], capture_output=True,
                             check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"base {BASE} not readable from git: {exc}")
    p = tmp_path_factory.mktemp("old") / "df_snr_ad30cf3.py"
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("df_snr_ad30cf3", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strip(rows):
    return [{k: v for k, v in r.items() if k != "code_sha256"} for r in rows]


def test_calibration_is_numerically_identical_to_base(old_snr, tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    _write_units(str(bank))
    bs = dict(d.DEFAULT_BOOTSTRAP, B=20)
    new_doc, new_rows = d.calibrate(str(bank), bootstrap=bs)
    old_doc, old_rows = old_snr.calibrate(str(bank), bootstrap=bs)
    assert _strip(new_rows) == _strip(old_rows)
    assert json.dumps(new_doc["grouped_table"], sort_keys=True) == json.dumps(old_doc["grouped_table"], sort_keys=True)
    assert new_doc["least_biased_per_perturbation"] == old_doc["least_biased_per_perturbation"]
    wm = [r for r in new_rows if r["estimator"] == "wavelet_mad" and r["metric"] == "noise_variance"]
    assert wm and any(r["value"] is not None for r in wm)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_estimates_identical_to_base_on_synthetic_units(old_snr, seed):
    rng = np.random.default_rng(seed)
    n = 2048
    x = np.sin(2 * np.pi * np.arange(n) / 128) + rng.standard_normal(n) * (0.2 * seed)
    x[500 + seed] = np.nan
    train = (0, 1228)
    bs = dict(d.DEFAULT_BOOTSTRAP, B=20)
    for name in d.ESTIMATORS:
        old = old_snr.estimate(x, train, name, bootstrap=bs)
        if name == "wavelet_mad":
            new = d.estimate_offline_train_diagnostic(_fit_snapshot(x, train[1]), 0, name, bootstrap=bs)
        else:
            new = d.estimate(x, train, name, bootstrap=bs)
        for k in ("noise_variance", "signal_variance", "snr_db", "status", "reason", "segment_used", "train",
                  "n_missing_in_train", "bootstrap"):
            assert new.get(k) == old.get(k), (name, k)
        if new["snr_db"] is not None:
            assert math.isfinite(new["snr_db"])
