"""C128 tests for tools/df_synthetic_bank.py (CPU, numpy, temp dirs)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "df_synthetic_bank.py"
_spec = importlib.util.spec_from_file_location("df_synthetic_bank", _PATH)
bank = importlib.util.module_from_spec(_spec)
sys.modules["df_synthetic_bank"] = bank
_spec.loader.exec_module(bank)

ARRAYS = ("clean_signal", "additive_noise", "observed_signal",
          "missing_mask")


def cell(family, pert="white", snr=10, n=2048, miss="none", **kw):
    return bank._cell_params(bank._cell(family, pert, snr, n, miss, **kw))


def test_regeneration_bit_identical_all_families_and_perturbations():
    params = [cell(f) for f in bank.CLEAN_FAMILIES if f != "null"]
    params += [cell("sinusoid", p) for p in bank.PERTURBATIONS
               if p != "null"]
    params += [cell("steps", miss="mcar"), cell("sinusoid", miss="blocks")]
    for p in params:
        a, b = bank.regenerate(p, 11), bank.regenerate(p, 11)
        for k in ARRAYS:
            assert a[k].tobytes() == b[k].tobytes(), (p, k)
            assert bank.array_digest(a[k]) == bank.array_digest(b[k])
        assert a["events"] == b["events"]
        c = bank.regenerate(p, 12)
        assert not np.array_equal(a["additive_noise"], c["additive_noise"])


def test_noise_seed_change_leaves_clean_unchanged():
    p = cell("bumps", "ar1", 5)
    q = dict(p, noise_seed_tag=1)
    a, b = bank.regenerate(p, 11), bank.regenerate(q, 11)
    assert bank.derived_seeds(p, 11)["clean"] == \
        bank.derived_seeds(q, 11)["clean"]
    assert np.array_equal(a["clean_signal"], b["clean_signal"])
    assert a["events"] == b["events"]
    assert not np.array_equal(a["additive_noise"], b["additive_noise"])
    # changing SNR or perturbation also leaves clean unchanged
    c = bank.regenerate(cell("bumps", "student_t", -5), 11)
    assert np.array_equal(a["clean_signal"], c["clean_signal"])


@pytest.mark.parametrize("family", ["sinusoid", "multiband", "multivariate"])
@pytest.mark.parametrize("snr", [20, 10, 5, 0, -5])
def test_declared_snr_matches_realized_train_snr(family, snr):
    g = bank.regenerate(cell(family, "white", snr, n=8192), 13)
    parts = g["partitions"]
    rs = bank.realized_snr(g["clean_signal"], g["additive_noise"], parts)
    for val in rs["train"]:
        assert abs(val - snr) < 0.5, (family, snr, val)


def test_inf_snr_gives_zero_noise():
    g = bank.regenerate(cell("sinusoid", "white", "inf"), 11)
    assert np.all(g["additive_noise"] == 0)
    assert not np.any(np.signbit(g["additive_noise"]))


def test_null_signal_and_null_noise_controls():
    g = bank.regenerate(cell("null", "white", "-inf", n=512), 11)
    assert np.all(g["clean_signal"] == 0)
    assert np.any(g["additive_noise"] != 0)
    rec = bank.build_unit_record(bank._cell("null", "white", "-inf", 512),
                                 11, g)
    assert rec["declared_snr_db"] == "-inf"
    assert set(rec["realized_snr_db"]["train"]) == {"-inf"}
    g = bank.regenerate(cell("sinusoid", "null", "inf", n=8192), 11)
    assert np.all(g["additive_noise"] == 0)
    rec = bank.build_unit_record(bank._cell("sinusoid", "null", "inf",
                                            8192), 11, g)
    assert set(rec["realized_snr_db"]["confirmation"]) == {"inf"}
    g = bank.regenerate(cell("null", "null", "nan"), 11)
    assert np.all(g["clean_signal"] == 0) and np.all(g["additive_noise"] == 0)
    with pytest.raises(ValueError):
        bank.regenerate(cell("null", "white", 10), 11)


def test_events_located_exactly():
    n = 2048
    g = bank.regenerate(cell("impulses", "null", "inf"), 11)
    x = g["clean_signal"][0]
    ev = [e for e in g["events"] if e["type"] == "impulse"]
    assert len(ev) == np.count_nonzero(x) == 8
    for e in ev:
        assert x[e["index"]] == e["height"]

    g = bank.regenerate(cell("steps"), 12)
    x = g["clean_signal"][0]
    for e in g["events"]:
        assert e["type"] == "step"
        assert x[e["index"]] == e["level_after"]
        assert x[e["index"] - 1] == e["level_before"]

    g = bank.regenerate(cell("bumps"), 11)
    x = g["clean_signal"][0]
    for e in g["events"]:
        assert x[e["index"]] == e["height"]
        assert np.argmax(x[e["index"] - 5:e["index"] + 6]) == 5

    g = bank.regenerate(cell("motif"), 11)
    x = g["clean_signal"][0]
    for e in g["events"]:
        i = e["index"]
        assert np.array_equal(x[i:i + e["length"]],
                              e["gain"] * bank.MOTIF_TEMPLATE)

    for fam, key in (("regime_mean", "mean"), ("regime_variance",
                                               "amplitude"),
                     ("regime_frequency", "period")):
        g = bank.regenerate(cell(fam), 11)
        bnd = g["clean_params"]["per_variable"][0]["boundaries"]
        evb = [e["index"] for e in g["events"]
               if e["type"] == "regime_boundary"]
        assert evb == bnd and len(bnd) == 3
        tr = bank.partitions(n)["train"]
        assert tr[0] <= bnd[0] < tr[1]      # a switch inside train

    g = bank.regenerate(cell("trend_piecewise"), 11)
    x = g["clean_signal"][0]
    for e in g["events"]:
        i = e["index"]
        assert np.isclose(x[i + 1] - x[i], e["slope_after"])
        assert np.isclose(x[i - 1] - x[i - 2], e["slope_before"])


def test_multivariate_structure_and_correlated_noise():
    g = bank.regenerate(cell("multivariate", "correlated", 0, n=8192), 11)
    assert g["clean_signal"].shape == (3, 8192)
    lo = g["clean_params"]["loadings"]
    assert len(lo) == 3
    z = g["additive_noise"] / g["noise_scale"][:, None]
    cov = np.array(bank.PERTURBATIONS["correlated"]["covariance"])
    assert np.allclose(np.corrcoef(z), cov, atol=0.06)


def test_missing_mask_matches_nans_in_observed_only():
    for miss, fam in (("mcar", "steps"), ("blocks", "multivariate"),
                      ("none", "sinusoid")):
        g = bank.regenerate(cell(fam, miss=miss), 11)
        m = g["missing_mask"]
        assert m.dtype == bool
        assert np.array_equal(np.isnan(g["observed_signal"]), m)
        assert not np.isnan(g["clean_signal"]).any()
        assert not np.isnan(g["additive_noise"]).any()
        assert np.array_equal(g["metric_support"], ~m)
        ok = ~m
        assert np.array_equal(g["observed_signal"][ok],
                              (g["clean_signal"] + g["additive_noise"])[ok])
        if miss == "mcar":
            assert abs(m.mean() - 0.10) < 0.02
        if miss == "blocks":
            assert m.sum() == 3 * (16 + 64 + 32)
        if miss == "none":
            assert not m.any()


@pytest.mark.parametrize("n", bank.LENGTHS)
def test_partitions_contiguous_chronological(n):
    p = bank.partitions(n)
    assert list(p) == ["train", "calibration", "confirmation"]
    assert p["train"][0] == 0 and p["confirmation"][1] == n
    assert p["train"][1] == p["calibration"][0]
    assert p["calibration"][1] == p["confirmation"][0]
    assert abs((p["train"][1]) / n - 0.6) < 0.01
    assert abs((p["calibration"][1] - p["calibration"][0]) / n - 0.2) < 0.01


def test_noise_scale_uses_only_train_part():
    g = bank.regenerate(cell("chirp", "white", 5), 11)
    parts = g["partitions"]
    clean = g["clean_signal"].copy()
    s0, _ = bank.noise_scale(clean, parts, 5)
    assert np.array_equal(s0, g["noise_scale"])
    a, b = parts["confirmation"]
    clean[:, a:b] = 1e6 * np.arange(b - a)
    c0, c1 = parts["calibration"]
    clean[:, c0:c1] = -42.0
    s1, _ = bank.noise_scale(clean, parts, 5)
    assert s1.tobytes() == s0.tobytes()
    t0, t1 = parts["train"]
    clean[:, t0] += 10.0
    s2, _ = bank.noise_scale(clean, parts, 5)
    assert not np.array_equal(s2, s0)


def test_materializer_write_once_digests_and_manifest(tmp_path):
    out = tmp_path / "bank"
    assert bank.main(["--out", str(out), "--limit", "8"]) == 0
    assert bank.main(["--out", str(out), "--limit", "8"]) == 2
    man = json.loads((out / "BANK_MANIFEST.json").read_text())
    assert man["units_materialized"] == 8 and man["complete"] is False
    assert man["generator"]["version"] == bank.GENERATOR_VERSION
    assert man["generator"]["code_sha256"] == bank.code_sha256()
    assert "/home/" not in (out / "BANK_MANIFEST.json").read_text()
    for u in man["units"]:
        ud = out / u["dir"]
        for k in bank.ARRAY_NAMES:
            assert (ud / f"{k}.npy").is_file()
        assert (ud / "events.json").is_file()
        text = (ud / "UNIT.json").read_text()
        assert "/home/" not in text
        rec = json.loads(text)
        assert rec["generator"]["code_sha256"] == bank.code_sha256()
        assert set(rec["realized_snr_db"]) == {"train", "calibration",
                                               "confirmation"}
        assert bank.verify_unit(ud)["ok"]
    # tamper one array: digests must catch it
    ud = out / man["units"][0]["dir"]
    a = np.load(ud / "clean_signal.npy")
    a[0, 0] += 1.0
    np.save(ud / "clean_signal.npy", a)
    v = bank.verify_unit(ud, regenerate_check=False)
    assert not v["ok"] and "clean_signal" in v["mismatches"]


def test_contract_fields_constants():
    c = bank._cell("multivariate", "white", 10, 2048, "mcar")
    g = bank.regenerate(bank._cell_params(c), 12)
    rec = bank.build_unit_record(c, 12, g)
    f = bank.contract_fields(rec)
    assert f["bank"] == "SYNTHETIC"
    assert f["license_state"] == "NOT_APPLICABLE_GENERATED"
    assert f["semantics"] == "KNOWN_BY_CONSTRUCTION"
    assert f["unit"] == "1"
    assert f["frequency"] == "1 sample"
    assert f["timestamp_meaning"] == "SAMPLE_INDEX"
    assert f["availability_rule"] == "SAMPLE_INDEX"
    assert "NaN" in f["missingness_encoding"]
    assert f["missingness_policy"].startswith("MCAR")
    assert f["partitions"] == bank.partitions(2048)
    assert f["variable_names"] == ["v0", "v1", "v2"]
    assert f["generator"]["version"] == bank.GENERATOR_VERSION
    assert f["generator"]["code_sha256"] == bank.code_sha256()
    assert f["generator"]["seed"] == 12
    json.dumps(f, allow_nan=False)


def test_matrix_predeclared_no_duplicates_and_counts():
    cells = bank.predeclared_matrix()
    keys = [json.dumps({k: v for k, v in c.items() if k != "blocks"},
                       sort_keys=True) for c in cells]
    assert len(keys) == len(set(keys))
    assert bank.predeclared_matrix() == cells          # deterministic
    for c in cells:
        bank._validate_cell(bank._cell_params(c))
    # A: every non-null family x white x full grid at primary length
    for fam in bank.CLEAN_FAMILIES:
        if fam == "null":
            continue
        got = {c["snr_db"] for c in cells if c["family"] == fam
               and c["perturbation"] == "white" and c["length"] == 2048
               and c["missingness"]["kind"] == "none"}
        assert got == {str(s) for s in bank.SNR_GRID_DB}
    # B: every non-null perturbation x {sinusoid, multiband} x grid
    for pert in bank.PERTURBATIONS:
        if pert == "null":
            continue
        for fam in ("sinusoid", "multiband"):
            got = {c["snr_db"] for c in cells if c["family"] == fam
                   and c["perturbation"] == pert}
            assert {str(s) for s in bank.SNR_GRID_DB} <= got
    # C: controls on all lengths
    for fam, pert in (("null", "white"), ("sinusoid", "null"),
                      ("null", "null")):
        assert {c["length"] for c in cells if c["family"] == fam
                and c["perturbation"] == pert} == set(bank.LENGTHS)
    # D: missingness x 3 families
    miss = [c for c in cells if c["missingness"]["kind"] != "none"]
    assert len({c["family"] for c in miss}) == 3
    assert {c["missingness"]["kind"] for c in miss} == {"mcar", "blocks"}
    plan = bank.unit_plan()
    assert len(plan) == len(cells) * len(bank.SEEDS)
    assert len(cells) < len(bank.CLEAN_FAMILIES) * len(bank.PERTURBATIONS) \
        * len(bank.SNR_GRID_DB) * len(bank.LENGTHS) * 3
    ids = [bank.unit_id(c, s) for c, s in plan]
    assert len(ids) == len(set(ids))
    counts = bank._counts(plan)
    for dim in ("family", "perturbation", "snr_db", "length",
                "missingness"):
        assert sum(counts[dim].values()) == len(plan)
    man = bank.build_manifest([], [], None, 0)
    assert man["matrix"]["cells_declared"] == len(cells)
    assert man["matrix"]["units_declared"] == len(plan)
    assert man["matrix"]["counts_declared_units"] == json.loads(
        json.dumps(counts))
    assert man["matrix"]["cartesian_sweep"] is False
