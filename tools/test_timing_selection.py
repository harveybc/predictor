"""Acceptance tests for the timing-target selection extension (targets v2 + ledger, price pool, driver).

Mapped to the plan audit's acceptance list (MUSASHI_TIMING_SELECTION_PLAN_AUDIT_2026_10_19): A01 ledger and
purges, A02 perturbation never changes inputs <= t and only permitted labels, A03 finality tested, A04 ties /
weekend clock / all origins accounted, A05 ordered inventory and profiles, A10 DAG invalidation, A12 idempotent
publication, snapshots and manifest-consumer negative tests.  Capabilities are loaded lazily through ``_cap``.
A02/A03 against the real expert need ``TIMING_ZIGZAG_MODULE`` (+ ``TIMING_ZIGZAG_SHA256``); otherwise they skip.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _cap(name: str):
    try:
        return importlib.import_module(f"tools.{name}")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.endswith(name):
            pytest.fail(f"capability absent: tools/{name}.py")
        raise


def _bars(n=600, seed=0, start="2020-01-01", gap_at=None, gap_hours=49):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 1.10 + 0.01 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 2e-4, n).cumsum() * 0.05
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 1.5e-4, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 1.5e-4, n))
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    if gap_at is not None:
        idx = idx.where(np.arange(n) < gap_at, idx + pd.Timedelta(hours=gap_hours - 1))
    return pd.DataFrame({"OPEN": open_, "HIGH": high, "LOW": low, "CLOSE": close}, index=pd.DatetimeIndex(idx).rename("DATE_TIME"))


class _FakeZZ:
    """Stand-in expert: pivots at fixed bars (alternating), confirmation ``lag`` bars after the next pivot;
    pivots listed in ``unconfirmed`` never confirm."""
    PEAK, VALLEY = 1, -1

    def __init__(self, bars_at=None, lag=3, unconfirmed=()):
        self.bars_at, self.lag, self.unconfirmed = bars_at, lag, set(unconfirmed)

    class Params:
        def __init__(self, depth=3, deviation_pips=36.3, backstep=2):
            self.depth, self.deviation_pips, self.backstep = depth, deviation_pips, backstep

    def hindsight_pivots(self, high, low, p):
        n = len(high)
        at = self.bars_at if self.bars_at is not None else range(24, n - 30, 24)
        out, kind = [], self.VALLEY
        for b in at:
            if b < n:
                out.append((b, kind, float(low[b] if kind == self.VALLEY else high[b])))
                kind = -kind
        return out

    def confirmation_bars(self, high, low, pivots, p, max_lag=400):
        idx = [b for b, _, _ in pivots]
        out = []
        for j, b in enumerate(idx):
            c = (idx[j + 1] + self.lag) if j + 1 < len(idx) else -1
            out.append(-1 if (j in self.unconfirmed or c >= len(high)) else c)
        return out


# ----------------------------------------------------------------------------- A01 ledger and purges
def test_a01_ledger_fields_support_and_purge_at_a_cut():
    tp = _cap("timing_pivot_targets")
    bars = _bars(300)
    zz = _FakeZZ()
    cut = pd.Timestamp("2020-01-06", tz="UTC")                       # bar 120 (a pivot bar)
    frame, led = tp.build_targets(bars, zz, zz.Params(), k_bars=2, cuts=[cut], with_ledger=True)
    for col in ("origin_time", "event_time", "t1_label_available_time", "t1_support_end", "expert_trade_id", "label_group_id", "t1_status"):
        assert col in led.columns
    y, st = frame["timing_pivot_k2"].to_numpy(), led["t1_status"].to_numpy()
    sup = pd.DatetimeIndex(led["t1_support_end"])
    before = (frame.timestamp < cut).to_numpy()
    crossing = before & (sup >= cut)
    assert crossing.sum() > 2                                          # more than the k-bar window: support, not k
    assert (st[crossing] == "PURGED_SUPPORT_CROSSES_CUT").all() and np.isnan(y[crossing]).all()
    labelled = st == "LABELLED"
    assert (pd.DatetimeIndex(led.loc[labelled & before, "t1_support_end"]) < cut).all()
    assert (pd.DatetimeIndex(led["t1_label_available_time"])[labelled] > pd.DatetimeIndex(led["origin_time"])[labelled]).all()
    # earliest-event semantics on a clean pivot
    assert y[22] == 1 and y[23] == 1 and y[24] == 0 and y[21] == 0      # valley at 24 labels (22, 23]


def test_a01_unconfirmed_and_end_of_data_are_censored_never_none():
    tp = _cap("timing_pivot_targets")
    bars = _bars(200)
    zz = _FakeZZ(unconfirmed={2})                                       # the third pivot never confirms
    frame, led = tp.build_targets(bars, zz, zz.Params(), k_bars=2, cuts=[], with_ledger=True)
    st = led["t1_status"]
    assert (st == "CENSORED_UNCONFIRMED").any() and frame.loc[(st == "CENSORED_UNCONFIRMED").to_numpy(), "timing_pivot_k2"].isna().all()
    assert (st.iloc[-5:] != "LABELLED").all()                           # the tail cannot be certified
    assert frame["timing_expert_leg_remaining_pips"].iloc[:24].isna().all()   # before the first pivot: NO_LEG


def test_a01_t2_on_the_confirmation_bar_and_t3_remaining_distance():
    tp = _cap("timing_pivot_targets")
    bars = _bars(300)
    zz = _FakeZZ()
    piv = zz.hindsight_pivots(bars.HIGH.to_numpy(), bars.LOW.to_numpy(), zz.Params())
    conf = zz.confirmation_bars(bars.HIGH.to_numpy(), bars.LOW.to_numpy(), piv, zz.Params())
    frame = tp.build_targets(bars, zz, zz.Params(), k_bars=2, cuts=[])
    a = frame["timing_action_confirmed"].to_numpy()
    assert a[conf[0]] == 1 and a[conf[1]] == -1 and a[piv[0][0]] == 0
    t = piv[0][0] + 5
    assert frame["timing_expert_leg_remaining_pips"].iloc[t] == pytest.approx((piv[1][2] - bars.CLOSE.iloc[t]) / 1e-4)


# ----------------------------------------------------------------------------- A04 ties, weekend clock, all origins
def test_a04_two_pivots_in_one_window_take_the_earliest_and_are_flagged():
    tp = _cap("timing_pivot_targets")
    bars = _bars(120)
    zz = _FakeZZ(bars_at=[30, 31, 60, 90])                             # valley 30, peak 31
    frame, led = tp.build_targets(bars, zz, zz.Params(), k_bars=2, cuts=[], with_ledger=True)
    assert frame["timing_pivot_k2"].iloc[29] == 1 and bool(led["both_kinds_in_window"].iloc[29])
    assert frame["timing_pivot_k2"].iloc[30] == 2 and not bool(led["both_kinds_in_window"].iloc[30])


def test_a04_tie_on_one_bar_is_refused():
    tp = _cap("timing_pivot_targets")
    bars = _bars(80)
    zz = _FakeZZ(bars_at=[30, 30, 60])
    with pytest.raises(tp.TimingError, match="PIVOT_TIE"):
        tp.build_targets(bars, zz, zz.Params(), k_bars=2, cuts=[])


def test_a04_k_counts_observed_bars_across_a_weekend_and_every_origin_is_accounted():
    tp = _cap("timing_pivot_targets")
    bars = _bars(200, gap_at=50, gap_hours=49)
    zz = _FakeZZ(bars_at=[51, 80, 110, 140, 170])
    frame, led = tp.build_targets(bars, zz, zz.Params(), k_bars=2, cuts=["2020-01-08"], with_ledger=True)
    assert frame["timing_pivot_k2"].iloc[49] == 1                       # pivot 2 observed bars ahead, 50 elapsed hours
    assert bool(led["window_crosses_gap"].iloc[49]) and led["window_gap_hours"].iloc[49] == 49
    tab = tp.status_table(led, {"A": ["2020-01-01", "2020-01-08"], "B": ["2020-01-08", "2020-02-01"]})
    assert tab["A"]["origins"] + tab["B"]["origins"] == len(bars)
    assert all(sum(tab[p][t].values()) == tab[p]["origins"] for p in tab for t in ("t1", "t2", "t3"))


# ----------------------------------------------------------------------------- A02 / A03 finality and perturbation
@pytest.fixture
def real_zz():
    tp = _cap("timing_pivot_targets")
    path, digest = os.environ.get("TIMING_ZIGZAG_MODULE"), os.environ.get("TIMING_ZIGZAG_SHA256")
    if not path or not digest:
        pytest.skip("TIMING_ZIGZAG_MODULE / TIMING_ZIGZAG_SHA256 not set")
    return tp.load_zigzag_module(path, digest)


def test_a03_finality_rule_holds_under_truncation_and_future_perturbation(real_zz):
    tp = _cap("timing_pivot_targets")
    rng = np.random.default_rng(7)
    n = 2500
    c = 1.1 * np.exp(np.cumsum(rng.normal(0, 1.2e-3, n)))
    o = np.r_[c[0], c[:-1]]
    w = np.abs(rng.normal(0, 6e-4, n)) * c
    bars = pd.DataFrame({"OPEN": o, "HIGH": np.maximum(o, c) + w, "LOW": np.minimum(o, c) - w, "CLOSE": c},
                        index=pd.date_range("2021-01-04", periods=n, freq="h", tz="UTC"))
    p = real_zz.Params(depth=3, deviation_pips=36.3, backstep=2)
    rep = tp.finality_check(bars, real_zz, p, k_bars=2, cut_positions=[900, 1700], check_t2=True)
    assert rep["all_final_rows_unchanged"] is True
    assert any(r[m]["t1"]["changed_among_not_final"] >= 0 for r in rep["cuts"] for m in ("truncation", "future_perturbation"))


def test_a02_future_perturbation_never_changes_pool_inputs_at_or_before_t_and_a_centred_feature_is_caught():
    pp = _cap("timing_price_pool")
    bars = _bars(900)
    assert pp.perturbation_check(pp.price_pool, bars, 600, seed=1)["passed"] is True

    def centred(b):
        return pd.DataFrame({"c": b.CLOSE.rolling(9, center=True).mean()}, index=b.index)

    bad = pp.perturbation_check(centred, bars, 600, seed=1)
    assert bad["passed"] is False and bad["future_perturbation"]["cells_differing"] > 0


def test_a05_quality_profile_and_declared_units():
    pp = _cap("timing_price_pool")
    bars = _bars(400)
    q = pp.quality_profile(pp.price_pool(bars))
    c = q["columns"]["px.rv_24"]
    assert q["duplicate_timestamps"] == 0 and c["warmup_rows"] == 24 and c["interior_nonfinite"] == 0
    assert c["window_unit"] == "market bars" and q["columns"]["px.elapsed_hours"]["window_unit"] == "elapsed hours"


def test_availability_gate_refuses_centred_filters_and_future_availability():
    pp = _cap("timing_price_pool")
    with pytest.raises(pp.AvailabilityError, match="FORBIDDEN_TRANSFORM"):
        pp.availability_gate([{"feature_id": "x_stl_trend", "transform": "stl", "availability_time": "timestamp"}])
    with pytest.raises(pp.AvailabilityError, match="FUTURE_AVAILABILITY"):
        pp.availability_gate([{"feature_id": "x", "transform": "identity", "availability_time": "timestamp+1h"}])


def test_within_day_variance_flags_daily_constant_features():
    pp = _cap("timing_price_pool")
    idx = pd.date_range("2020-01-01", periods=240, freq="h", tz="UTC")
    rep = pp.within_day_variance(pd.DataFrame({"d": np.repeat(np.arange(10), 24).astype(float), "h": np.arange(240.0)}, index=idx))
    assert rep["d"]["constant_within_day"] is True and rep["h"]["constant_within_day"] is False


def test_engine_mi_codes_keep_three_classes_distinct():
    ff = _cap("feature_filter_selection")
    assert len(np.unique(ff._codes(np.array([0, 1, 2] * 100, dtype=float), 8))) == 3


def test_fdr_screen_controls_the_null_and_keeps_a_real_signal():
    ts = _cap("timing_selection")
    rng = np.random.default_rng(1)
    n = 3000
    y = rng.integers(0, 3, n)
    X = pd.DataFrame({f"n{i}": rng.normal(size=n) for i in range(20)})
    X["sig"] = (y == 1).astype(float) + rng.normal(0, 0.5, n)
    rep = ts.univariate_screen(X, y, head="classification", q=0.1)
    assert rep["sig"]["pass_fdr"] is True and sum(r["pass_fdr"] for k, r in rep.items() if k != "sig") <= 3


# ----------------------------------------------------------------------------- A10 DAG
def test_a10_expert_hash_change_invalidates_every_target_dependent_stage_only():
    ts = _cap("timing_selection")
    leaves = {k: "x" for k in ("lake", "zigzag_module", "expert_params", "k_bars", "target_semantics", "splits", "inventory_generator",
                               "inventory_bundle", "fractal_store", "pool_code", "candidates_config", "pairwise_params", "filter_params",
                               "recipe_config", "causal_config", "comparison_sets", "adapter_revision")}
    a = ts.dag_identities(leaves)
    b = ts.dag_identities({**leaves, "zigzag_module": "y"})
    plan = ts.plan_invalidation(a, b)
    for name, node in ts.DAG_NODES.items():
        if node["target_dependent"]:
            assert plan[name] == "RECOMPUTE", name
        else:
            assert plan[name] == "REUSE", name


# ----------------------------------------------------------------------------- driver end to end, A12
def _write_zz(path: Path, extra: str = "") -> str:
    path.write_text(
        "PEAK, VALLEY = 1, -1\n"
        "class Params:\n    def __init__(self, depth=3, deviation_pips=36.3, backstep=2):\n"
        "        self.depth, self.deviation_pips, self.backstep = depth, deviation_pips, backstep\n"
        "def hindsight_pivots(high, low, p):\n    out=[]; kind=-1\n"
        "    for b in range(24, len(high)-30, 24):\n        out.append((b, kind, float(low[b] if kind==-1 else high[b]))); kind=-kind\n"
        "    return out\n"
        "def confirmation_bars(high, low, pivots, p, max_lag=400):\n    idx=[b for b,_,_ in pivots]\n"
        "    return [(idx[j+1]+3 if j+1 < len(idx) and idx[j+1]+3 < len(high) else -1) for j in range(len(idx))]\n" + extra)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path, zz_path, zz_sha):
    bars = _bars(2000, start="2020-01-01")
    lake = tmp_path / "lake.parquet"
    if not lake.exists():
        pd.DataFrame({"datetime": bars.index.tz_convert(None), "open": bars.OPEN, "high": bars.HIGH, "low": bars.LOW,
                      "close": bars.CLOSE}).to_parquet(lake, index=False)
    return {
        "schema": "predictor.timing_selection.v1", "population_id": "SYNTH_TIMING",
        "lake": {"path": str(lake), "sha256": hashlib.sha256(lake.read_bytes()).hexdigest(), "load": ["2020-01-01", "2020-03-15"]},
        "zigzag": {"module": str(zz_path), "sha256": zz_sha, "params": {"depth": 3, "deviation_pips": 36.3, "backstep": 2}, "k_bars": 2},
        "splits": {"train": ["2020-01-01", "2020-02-15"], "inner": ["2020-02-15", "2020-03-01"], "inner_a_end": "2020-02-22",
                   "confirm": ["2020-03-01", "2020-03-15"], "folds": ["2020-01-20", "2020-02-05"]},
        "candidates": {"inventory": None, "price_pool": True, "fractal": None},
        "pairwise_params": {"lags_hours": [0, 1, 2, 6], "min_support": 50, "dcor_max_rows": 128},
        "filter": {"seed": 1, "k_grid": [4, 8], "recipe": {"method": "MRMR", "k": 8}},
        "causal": {"enabled": False}, "comparison_sets": {"baseline_two": ["px.rv_24", "cal.hod_sin"]},
        "n_shards": 1, "seed": 1,
    }


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    ts = _cap("timing_selection")
    tmp = tmp_path_factory.mktemp("e2e")
    zz = tmp / "zz.py"
    cfg = _config(tmp, zz, _write_zz(zz))
    (tmp / "cfg.json").write_text(json.dumps(cfg))
    out = tmp / "out"
    rep = ts.run(tmp / "cfg.json", out)
    return ts, tmp, cfg, out, rep


def test_end_to_end_seals_a_gate_shaped_timing_proxy_manifest(e2e):
    ts, tmp, cfg, out, rep = e2e
    m = json.loads((out / "SELECTION_MANIFEST.json").read_text())
    assert m["selection_name"] == "TIMING_PROXY_SELECTION" and m["final_selection"] is False and m["proxy_only"] is True
    assert m["targets"][0]["role"] == "PRIMARY_ANTICIPATORY" and m["test_read"] is False
    assert ts.gate_digest(m, "manifest_sha256") == m["manifest_sha256"]
    inv = json.loads((out / "INVENTORY.json").read_text())
    assert inv["count"] == len(inv["ordered_columns"]) == sum(inv["counts_by_source"].values()) == rep["features"]
    assert json.loads((out / "TARGETS.json").read_text())["finality"]["all_final_rows_unchanged"] is True
    assert json.loads((out / "PERTURBATION.json").read_text())["all_passed"] is True
    assert (out / "FROZEN_SUBSET.json").exists() and (out / "READ_LEDGER.json").exists()
    assert set(rep["recipe"]["targets"]) == {"timing_pivot_k2", "timing_valley_k2", "timing_peak_k2"}


def test_a10_rerun_with_a_changed_expert_reuses_only_target_independent_stages(e2e, tmp_path):
    ts, tmp, cfg, out, rep = e2e
    zz2 = tmp_path / "zz2.py"
    cfg2 = dict(cfg, zigzag=dict(cfg["zigzag"], module=str(zz2), sha256=_write_zz(zz2, "# changed expert bytes\n")))
    (tmp_path / "cfg2.json").write_text(json.dumps(cfg2))
    rep2 = ts.run(tmp_path / "cfg2.json", tmp_path / "out2", reuse_from=out)
    plan = rep2["dag_plan"]
    assert plan["pool_features"] == "REUSE" and plan["targets_ledger"] == "RECOMPUTE" and plan["phase3_filters"] == "RECOMPUTE"


def test_a12_publication_is_idempotent_with_exact_readback_and_snapshots_verify(e2e, tmp_path):
    ts, tmp, cfg, out, rep = e2e
    db = tmp_path / "olap.sqlite"
    first = ts.publish(out, db)
    second = ts.publish(out, db)
    assert first["all_match"] and second["all_match"] and sum(first["inserted"].values()) > 0
    assert all(v == 0 for v in second["inserted"].values())
    snap = ts.snapshot(out, [tmp_path / "copy_a", tmp_path / "copy_b"])
    assert all(c["verified"] for c in snap["copies"]) and snap["independent_physical_copy"] is False


def test_a12_manifest_consumer_refuses_without_an_independent_decision_and_when_tampered(e2e):
    ts, tmp, cfg, out, rep = e2e
    gate = _cap("selected_manifest_gate")
    m = json.loads((out / "SELECTION_MANIFEST.json").read_text())
    sel = m["selected_features"]
    r = gate.evaluate(m, sel)
    assert r["admitted"] is False and any("DECISION" in x for x in r["reasons"])
    tampered = dict(m, selected_features=sel + ["px.rv_1"])
    assert any("DIGEST_MISMATCH" in x for x in gate.evaluate(tampered, sel)["reasons"])
    # a synthetic decision by an independent role proves the manifest is otherwise gate-conformant
    d = {"schema": gate.DECISION_SCHEMA, "decision_id": m["independent_decision_record"]["decision_id"], "decider_role": "auditor",
         "decided_on": "synthetic-test", "manifest_sha256": m["manifest_sha256"], "verdict": gate.ACCEPT_VERDICT}
    d["record_sha256"] = gate.canonical_sha256(d, "record_sha256")
    assert gate.evaluate(m, sel, decision_record=d)["admitted"] is True
    rejected = next(f["name"] for f in m["features"] if f["state"] == "rejected")
    assert any("CONSUMED_NOT_SELECTED" in x for x in gate.evaluate(m, sel + [rejected], decision_record=d)["reasons"])
    self_cert = dict(d, decider_role="producer")
    self_cert["record_sha256"] = gate.canonical_sha256(self_cert, "record_sha256")
    assert gate.evaluate(m, sel, decision_record=self_cert)["admitted"] is False
