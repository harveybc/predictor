"""C169: every ADF/KPSS resource estimate binds its block before the library runs,
the three blocks of one run are three identities, and the v2 recompute of a
v1-shaped root (identity missing) derives each block from the profile
statistic rows, without running ADF or KPSS, with offered == distinct."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("df_memory_plan")
U = _load("df_profile_univariate")
L = _load("load_data_foundation")
RV2 = _load("df_resource_estimate_v2")
SHA = "d" * 64
BLOCK = 5_000
POLICY = dict(U.UNIT_ROOT_POLICY, exact_max_n=BLOCK)
PARTS = [("train", 0, 20_000), ("calibration", 20_000, 23_000), ("confirmation", 23_000, 26_000)]
CTX = {"T": 26_000, "n_train": 20_000, "has_ts": False, "rg_rows": 26_000, "k_pairs": 1, "V_matrix": 1}


def column(seed, constant_middle=False):
    x = np.cumsum(np.random.default_rng(seed).standard_normal(26_000))
    if constant_middle:
        x[7_500:12_500] = 0.0                      # the middle train block is constant: no gate call for it
    return x


def test_mirrors_equal_the_module():
    assert P.UNIT_ROOT_POLICY_SHA256 == U.UNIT_ROOT_POLICY_SHA256
    for rs, re_, B in ((0, 10, 10), (3, 40, 10), (10, 33, 10), (0, 200_001, 200_000)):
        assert P.unit_root_block_ranges(rs, re_, B) == U.unit_root_blocks(rs, re_, {"exact_max_n": B})


def test_c166_5_counterexample_now_three_identities():
    """The PRE item 5 probe, through the same public API: three blocks, three distinct estimate rows."""
    rows = []
    planner = P.Planner(run_id="pre", bank="SYNTHETIC", dataset_id="pre_ds", budget_bytes=8 << 30,
                        code_sha256="0" * 64, context={"T": 20000, "n_train": 20000, "has_ts": False,
                                                       "rg_rows": 20000, "k_pairs": 0, "V_matrix": 1},
                        sink=rows.append)
    x = np.cumsum(np.random.default_rng(3).standard_normal(20000))
    U.unit_root_rows("pre_ds", {"variable_id": "v"}, "train", x, gate=planner.for_module("df_profile_univariate"),
                     policy=POLICY)
    for group in ("unit_root_adf", "unit_root_kpss"):
        est = [r for r in rows if r["metric"] == group]
        ident = {hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in est}
        assert len(est) == 3 and len(ident) == 3
        offs = [r["params"]["block_identity"]["block_offset"] for r in est]
        assert offs == ["start", "middle", "end"]
        assert [r["params"]["block_identity"]["range_partition"] for r in est] == [[0, 5000], [7500, 12500],
                                                                                  [15000, 20000]]
        assert all(P.validate_resource_row(r) == [] for r in est)


def test_gate_refuses_a_unit_root_estimate_without_or_with_a_wrong_identity():
    pl = P.Planner(run_id="r", bank="FINANCIAL", dataset_id="d", budget_bytes=4 << 30, code_sha256=SHA, context=CTX)
    with pytest.raises(P.PlanRefusal, match="must bind its block identity"):
        pl.gate("df_profile_univariate", "unit_root_adf", "v", "train", n_run=5000, lag=P.schwert_lag(5000))
    good = P.unit_root_block_identity(offset="start", block=(0, 5000), run=(0, 20000), partition_start=0,
                                      lag=P.schwert_lag(5000), policy_sha256=SHA, variant="BLOCK_APPROX")
    for bad, needle in ((dict(good, n_used=4999), "n_used"), (dict(good, block_offset="exact"), "disagree"),
                        (dict(good, lag_used=3), "lag"), (dict(good, range_absolute=[1, 5001]), "absolute"),
                        (dict(good, block_offset="late"), "not in")):
        with pytest.raises(P.PlanRefusal, match=needle):
            pl.gate("df_profile_univariate", "unit_root_adf", "v", "train", n_run=5000, lag=P.schwert_lag(5000),
                    variant="BLOCK_APPROX", block_identity=bad)
    with pytest.raises(P.PlanRefusal, match="only unit-root"):
        pl.gate("df_profile_univariate", "acf", "v", "train", n=10, block_identity=good)
    d = pl.gate("df_profile_univariate", "unit_root_adf", "v", "train", n_run=5000, lag=P.schwert_lag(5000),
                variant="BLOCK_APPROX", block_identity=good)
    assert d["decision"] == "RUN_BOUNDED"


def test_preflight_emits_one_identity_per_possible_block():
    rows = []
    T = 700_000
    meta = {"T": T, "variables": ["v"], "partitions": {"train": [0, 420_000], "calibration": [420_000, 560_000],
                                                       "confirmation": [560_000, T]},
            "has_ts": False, "rg_rows": 100_000, "text_timestamp": False}
    pl = P.Planner(run_id="r", bank="FINANCIAL", dataset_id="d", budget_bytes=6 << 30, code_sha256=SHA,
                   context={"T": T, "n_train": 420_000, "has_ts": False, "rg_rows": 100_000}, sink=rows.append,
                   stage="PREFLIGHT_METADATA_UPPER_BOUND")
    P.preflight(meta, pl)
    adf = [r for r in rows if r["metric"] == "unit_root_adf"]
    by_part = {p: [r["params"]["block_identity"] for r in adf if r["partition"] == p] for p in meta["partitions"]}
    assert [i["block_offset"] for i in by_part["train"]] == ["start", "middle", "end"]
    assert [i["block_offset"] for i in by_part["calibration"]] == ["exact"]
    assert by_part["calibration"][0]["range_absolute"] == [420_000, 560_000]
    assert {i["universe_basis"] for ids in by_part.values() for i in ids} == {"PARTITION_UPPER_BOUND"}
    assert len({json.dumps(r, sort_keys=True) for r in rows}) == len(rows)


# ------------------------------------------------------------------ v1-shaped fixture root
def legacy_gate(sink, ds="fixture.ds", budget=6 << 30):
    """The C162 child's gate: the same estimates, written WITHOUT the block identity."""
    def gate(group, key=None, partition=None, **sizes):
        sizes.pop("block_identity", None)
        variant = sizes.pop("variant", "EXACT")
        est, formula, params = P.estimate("df_profile_univariate", group, sizes, CTX)
        fits = est <= budget
        decision = ("RUN_EXACT" if variant == "EXACT" else "RUN_BOUNDED") if fits else "NOT_RUN_RESOURCE_BOUND"
        sink({"run_id": "fixture_run", "bank": "SYNTHETIC", "dataset_id": ds, "variable_id": key,
              "partition": partition, "module": "df_profile_univariate", "metric": group, "estimator": variant,
              "estimated_peak_bytes": est, "formula": formula,
              "params": dict(params, variant=variant, stage="IN_TASK_BEFORE_ALLOCATION",
                             constants_sha256=P.constants_sha256()),
              "budget_bytes": budget, "decision": decision, "code_sha256": SHA})
        return {"decision": decision, "window": None}
    return gate


def fixture_root(base: Path, modern=False) -> Path:
    root = base / "profiles_fixture_v1"
    adir = root / "attempts" / "0001__fixture.ds" / "attempt-1"
    adir.mkdir(parents=True)
    (root / "terminals").mkdir()
    est, prof = [], []
    for vid, col in (("fixture.a", column(1)), ("fixture.b", column(2, constant_middle=True))):
        if modern:
            pl = P.Planner(run_id="fixture_run", bank="SYNTHETIC", dataset_id="fixture.ds", budget_bytes=6 << 30,
                           code_sha256=SHA, context=CTX, sink=est.append)
            gate = pl.for_module("df_profile_univariate")
        else:
            gate = legacy_gate(est.append)
        prof += [{"module": "df_profile_univariate", "row": r}
                 for r in U.variable_rows("fixture.ds", vid, col, PARTS, gate, POLICY)]
    pre = []
    for p, s, e in PARTS:
        n = e - s
        variant = "EXACT" if n <= P.UNIT_ROOT_EXACT_MAX_N else "BLOCK_APPROX"
        legacy_gate(pre.append)("counts", "fixture.a", p, n=n)
        legacy_gate(pre.append)("unit_root_adf", "fixture.a", p, n_run=n, lag=P.schwert_lag(n), variant=variant)
        legacy_gate(pre.append)("unit_root_kpss", "fixture.a", p, n_run=n, variant=variant)
    pre.append(dict(pre[-1], variable_id="fixture.big", params=dict(pre[-1]["params"], variant="BLOCK_APPROX"),
                    estimator="BLOCK_APPROX", decision="RUN_BOUNDED"))
    for rows, name in ((est, "resource_estimates.jsonl"), (pre, "preflight_estimates.jsonl"),
                       (prof, "profile.jsonl")):
        (adir / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    (root / "PROFILE_RUN_RECEIPT.json").write_text(json.dumps({"datasets": [], "counts": {}}))
    return root


def test_recompute_restores_block_identity_without_running_the_tests(tmp_path, monkeypatch):
    import statsmodels.tsa.stattools as st
    root = fixture_root(tmp_path)

    def boom(*a, **k):
        raise AssertionError("ADF/KPSS must not run in the recompute")
    monkeypatch.setattr(st, "adfuller", boom)
    monkeypatch.setattr(st, "kpss", boom)
    s = RV2.recompute([root], tmp_path / "out")
    child = [json.loads(line) for line in (root / "attempts/0001__fixture.ds/attempt-1/resource_estimates.jsonl")
             .read_text().splitlines()]
    ur = [r for r in child if r["metric"] in ("unit_root_adf", "unit_root_kpss")]
    # a: 3 identical adf + 3 identical kpss in train (4 collapse); b: middle block constant, 2 + 2 (2 collapse)
    assert s["v1"]["collapsed_rows"] == 6
    assert s["v1"]["collapsed_by_stage_metric"] == {"CHILD_RUNTIME:unit_root_adf": 3,
                                                   "CHILD_RUNTIME:unit_root_kpss": 3}
    assert s["v2"]["offered_equals_distinct"] and s["v2"]["collapsed_rows"] == 0
    assert s["unresolved_v1_rows"] == 0 and s["every_v1_row_mapped"] and s["v2"]["validation_refusals"] == 0
    assert s["estimate_recompute"] == {"recomputed": len(ur)} and s["unpaired_statistic_cells"] == 0
    assert s["adf_kpss_rerun"] is False
    v2 = [json.loads(line) for line in (tmp_path / "out/df_fact_resource_estimate_v2.jsonl").read_text().splitlines()]
    assert len(v2) == s["v2"]["rows"] == s["v1"]["rows_offered"]
    assert len((tmp_path / "out/V1_TO_V2_MAP.jsonl").read_text().splitlines()) == s["v1"]["rows_offered"]

    def offsets(vid, part, metric):
        return [(r["block_offset"], r["range_start"], r["range_end"], r["run_start"], r["run_end"], r["lag_used"])
                for r in v2 if r["variable_id"] == vid and r["partition"] == part and r["metric"] == metric
                and r["stage"] == "CHILD_RUNTIME"]
    lag = P.schwert_lag(BLOCK)
    assert offsets("fixture.a", "train", "unit_root_adf") == [("start", 0, 5000, 0, 20000, lag),
                                                               ("middle", 7500, 12500, 0, 20000, lag),
                                                               ("end", 15000, 20000, 0, 20000, lag)]
    assert offsets("fixture.b", "train", "unit_root_kpss") == [("start", 0, 5000, 0, 20000, None),
                                                                ("end", 15000, 20000, 0, 20000, None)]
    assert offsets("fixture.a", "calibration", "unit_root_adf") == [("exact", 20000, 23000, 20000, 23000,
                                                                     P.schwert_lag(3000))]
    assert {r["unit_root_policy_sha256"] for r in v2 if r["identity_kind"] == "UNIT_ROOT_BLOCK"
            and r["stage"] == "CHILD_RUNTIME"} == {U.policy_sha256(POLICY)}
    kinds = {(r["stage"], r["identity_kind"], r["derivation"]) for r in v2}
    assert ("PREFLIGHT_METADATA_UPPER_BOUND", "UNIT_ROOT_PREFLIGHT_V1_ALL_OFFSETS", "CARRIED_OVER") in kinds
    assert ("PREFLIGHT_METADATA_UPPER_BOUND", "UNIT_ROOT_BLOCK", "RECOMPUTED_FROM_PROFILE_STATISTICS") in kinds
    assert ("CHILD_RUNTIME", "NOT_UNIT_ROOT", "CARRIED_OVER") in kinds
    assert all(L.validate_row("df_fact_resource_estimate_v2", r) == [] for r in v2)


def test_current_planner_rows_are_taken_as_emitted_and_already_distinct(tmp_path):
    root = fixture_root(tmp_path, modern=True)
    s = RV2.recompute([root], tmp_path / "out")
    assert s["v1"]["collapsed_rows"] == 0 and s["v2"]["offered_equals_distinct"]
    # child unit-root rows: a train 3+3, b train 2+2, and 1+1 exact per later partition for each variable
    assert s["v2"]["derivations"]["EMITTED_WITH_BLOCK_IDENTITY"] == (3 + 3) + (2 + 2) + 2 * 2 * (1 + 1)
    assert s["unresolved_v1_rows"] == 0
    assert len(RV2.root_v2_rows(root)) == s["v2"]["rows"]


def test_output_is_write_once_and_never_inside_a_root_or_the_state_tree(tmp_path, monkeypatch):
    root = fixture_root(tmp_path)
    RV2.recompute([root], tmp_path / "out")
    with pytest.raises(SystemExit, match="write-once"):
        RV2.recompute([root], tmp_path / "out")
    with pytest.raises(SystemExit, match="inside a root"):
        RV2.recompute([root], root / "v2")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    with pytest.raises(SystemExit, match="~/.local"):
        RV2.recompute([root], tmp_path / "home/.local/state/v2")


def test_v2_table_constraints_are_mirrored():
    base = {"run_id": "r", "bank": "SYNTHETIC", "dataset_id": "d", "variable_id": "v", "partition": "train",
            "module": "df_profile_univariate", "metric": "unit_root_adf", "estimator": "BLOCK_APPROX",
            "estimated_peak_bytes": 1, "formula": "f", "params": {}, "budget_bytes": 2, "decision": "RUN_BOUNDED",
            "stage": "CHILD_RUNTIME", "attempt": "a/attempt-1", "identity_kind": "UNIT_ROOT_BLOCK",
            "block_offset": "middle", "universe_basis": "LONGEST_FINITE_RUN", "partition_start": 0,
            "range_start": 10, "range_end": 20, "run_start": 0, "run_end": 30, "n_used": 10, "lag_used": 2,
            "unit_root_policy_sha256": SHA, "variant": "BLOCK_APPROX", "derivation": "CARRIED_OVER",
            "v1_row_sha256": SHA, "code_sha256": SHA}
    assert L.validate_row("df_fact_resource_estimate_v2", base) == []
    for over in ({"block_offset": "NONE"}, {"range_end": 40}, {"n_used": 9}, {"variant": "EXACT"},
                 {"identity_kind": "NOT_UNIT_ROOT"}, {"block_offset": "exact"}):
        assert L.validate_row("df_fact_resource_estimate_v2", dict(base, **over)), over
    ddl = L.ddl()
    assert "CREATE TABLE IF NOT EXISTS public.df_fact_resource_estimate_v2" in ddl
    assert "CREATE TABLE IF NOT EXISTS public.df_fact_resource_estimate (" in ddl     # v1 kept as history
