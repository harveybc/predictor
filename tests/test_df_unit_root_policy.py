"""C148: the finite-memory ADF/KPSS policy is declared and hashed, exact and
block rows are distinguished with their temporal universe, blocks are
contiguous (never thinned), n is never reduced silently, and on series small
enough to run exactly the block machinery (with a test-only block size) agrees
with the exact decision at 5% on known stationary and unit-root series.

Run as a script with --write to regenerate the agreement fixture."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/c148_unit_root_block_agreement.v1.json"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load("df_profile_univariate")
# Frozen when the policy was declared, before any agreement was measured.
DECLARED_POLICY_SHA256 = "dcea5b87627e8e768f460f8eb0db3ea2e2fbe54769158ab960916b32bfa6beec"
TEST_BLOCK = 5_000
N = 20_000
SEEDS = (1, 2, 3, 4, 5)


def series(kind, seed):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(N)
    if kind == "unit_root":
        return np.cumsum(e)
    x = np.empty(N)
    x[0] = e[0]
    for i in range(1, N):
        x[i] = 0.5 * x[i - 1] + e[i]
    return x


def by_metric(rows):
    return {r["metric"]: r for r in rows}


def agreement_table():
    policy = dict(U.UNIT_ROOT_POLICY, exact_max_n=TEST_BLOCK)
    cases = []
    for kind in ("stationary", "unit_root"):
        for seed in SEEDS:
            x = series(kind, seed)
            exact = by_metric(U.unit_root_rows("d", {"variable_id": "v"}, "train", x))
            block = by_metric(U.unit_root_rows("d", {"variable_id": "v"}, "train", x, policy=policy))
            case = {"kind": kind, "seed": seed}
            for test in ("adf", "kpss"):
                ex = exact[f"{test}_pvalue"]["value"] < 0.05
                offs = {o: block[f"{test}_pvalue_block_{o}"]["value"] < 0.05 for o in U.UNIT_ROOT_BLOCK_OFFSETS}
                case[test] = {"exact_reject_5pct": bool(ex), "block_reject_5pct": offs,
                              "agree": sum(v == ex for v in offs.values())}
            cases.append(case)
    total = sum(c[t]["agree"] for c in cases for t in ("adf", "kpss"))
    return {"schema": "crispdm.data_foundation.c148_unit_root_block_agreement.v1", "n": N,
            "test_block_length": TEST_BLOCK, "seeds": list(SEEDS),
            "stationary": "AR(1) phi=0.5 gaussian", "unit_root": "gaussian random walk",
            "cases": cases, "agreements": total, "comparisons": 3 * 2 * len(cases),
            "agreement_fraction": total / (3 * 2 * len(cases))}


def test_policy_is_declared_and_hash_frozen():
    assert U.UNIT_ROOT_EXACT_MAX_N == 200_000 == U.UNIT_ROOT_POLICY["exact_max_n"]
    assert U.UNIT_ROOT_POLICY_SHA256 == U.policy_sha256(U.UNIT_ROOT_POLICY) == DECLARED_POLICY_SHA256
    doc = U.__doc__.lower()
    assert "never a causality gate" in doc and "never an eligibility gate" in doc


def test_blocks_are_contiguous_offsets_never_thinned():
    mode, blocks = U.unit_root_blocks(10, 10 + 23, {"exact_max_n": 10})
    assert mode == "BLOCK_APPROX"
    assert blocks == [("start", 10, 20), ("middle", 16, 26), ("end", 23, 33)]
    assert all(e - s == 10 for _, s, e in blocks)
    assert U.unit_root_blocks(0, 10, {"exact_max_n": 10}) == ("EXACT", [("exact", 0, 10)])


def test_exact_rows_declare_policy_and_universe():
    x = np.full(3000, np.nan)
    x[100:2900] = series("stationary", 9)[:2800]
    rows = by_metric(U.unit_root_rows("d", {"variable_id": "v"}, "calibration", x))
    for m in ("adf_statistic", "adf_pvalue", "kpss_statistic", "kpss_pvalue"):
        p = rows[m]["estimator"]["params"]
        assert rows[m]["status"] == "COMPLETED"
        assert p["policy"] == "EXACT" and p["temporal_universe"] == [100, 2900] and p["run_length"] == 2800
        assert p["policy_sha256"] == U.UNIT_ROOT_POLICY_SHA256
    assert not [m for m in rows if "block" in m]


def test_block_rows_three_offsets_spread_and_exact_not_run():
    x = series("unit_root", 3)
    policy = dict(U.UNIT_ROOT_POLICY, exact_max_n=TEST_BLOCK)
    rows = by_metric(U.unit_root_rows("d", {"variable_id": "v"}, "train", x, policy=policy))
    for m in ("adf_statistic", "adf_pvalue", "kpss_statistic", "kpss_pvalue"):
        assert rows[m]["status"] == "NOT_RUN" and "BLOCK_APPROX" in rows[m]["reason"]
        assert rows[m]["estimator"]["params"]["run_length"] == N          # n is declared, not reduced
    expected = {"start": [0, 5000], "middle": [7500, 12500], "end": [15000, 20000]}
    for test in ("adf", "kpss"):
        for o, uni in expected.items():
            r = rows[f"{test}_statistic_block_{o}"]
            p = r["estimator"]["params"]
            assert r["status"] == "COMPLETED" and p["policy"] == "BLOCK_APPROX"
            assert p["temporal_universe"] == uni and p["block_length"] == TEST_BLOCK and p["block_offset"] == o
            if test == "adf":
                assert p["maxlag"] == U.schwert_lag(TEST_BLOCK)
        vals = [rows[f"{test}_statistic_block_{o}"]["value"] for o in expected]
        assert abs(rows[f"{test}_statistic_block_spread"]["value"] - (max(vals) - min(vals))) < 1e-12


def test_block_agreement_with_exact_at_5pct_matches_fixture():
    table = agreement_table()
    assert table == json.loads(FIXTURE.read_text())
    assert table["agreement_fraction"] >= 0.9
    # the ADF decision on each known process is the textbook one for exact and for every block
    for c in table["cases"]:
        expected = c["kind"] == "stationary"
        assert c["adf"]["exact_reject_5pct"] is expected
        assert all(v is expected for v in c["adf"]["block_reject_5pct"].values())


def test_a_constant_block_is_inconclusive_not_failed():
    # Found in the C151 worst-case smoke: a monthly 0/1 indicator materialized at 5 minutes is
    # constant inside whole blocks; ADF refuses a constant input and KPSS divides by zero.
    x = series("unit_root", 4)
    x[7_500:12_500] = 0.0                      # exactly the middle block with TEST_BLOCK on N
    policy = dict(U.UNIT_ROOT_POLICY, exact_max_n=TEST_BLOCK)
    rows = by_metric(U.unit_root_rows("d", {"variable_id": "v"}, "train", x, policy=policy))
    for test in ("adf", "kpss"):
        for what in ("statistic", "pvalue"):
            mid = rows[f"{test}_{what}_block_middle"]
            assert mid["status"] == "INCONCLUSIVE" and mid["reason"] == "ZERO_VARIANCE_BLOCK"
            assert rows[f"{test}_{what}_block_start"]["status"] == "COMPLETED"
            assert rows[f"{test}_{what}_block_spread"]["reason"] == "NOT_EVERY_OFFSET_COMPLETED"
    assert not [r for r in rows.values() if r["status"] == "FAILED"]


def test_resource_gate_refusal_never_calls_the_library(monkeypatch):
    import statsmodels.tsa.stattools as st

    def boom(*a, **k):
        raise AssertionError("library invoked despite NOT_RUN_RESOURCE_BOUND")
    monkeypatch.setattr(st, "adfuller", boom)
    monkeypatch.setattr(st, "kpss", boom)
    refuse = lambda group, key=None, partition=None, **s: {"decision": "NOT_RUN_RESOURCE_BOUND", "window": None}
    rows = U.unit_root_rows("d", {"variable_id": "v"}, "train", series("stationary", 1)[:3000], gate=refuse)
    assert {r["status"] for r in rows} == {"NOT_RUN"} and {r["reason"] for r in rows} == {"NOT_RUN_RESOURCE_BOUND"}


if __name__ == "__main__" and "--write" in sys.argv:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(agreement_table(), indent=1, sort_keys=True) + "\n")
    print(FIXTURE.read_text())
