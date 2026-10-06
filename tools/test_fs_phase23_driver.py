"""RED acceptance tests for the Phase-2/Phase-3 feature-selection driver.

Each acceptance point of FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md section 6
is one test.  Capabilities are loaded lazily through ``_cap`` so that a PRE run fails with
``capability absent: <module>`` for a missing implementation, never with a broken import.
"""
from __future__ import annotations

import gzip
import importlib
import json
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


# --------------------------------------------------------------------------- fixtures

FEATURES = ["f_alpha", "f_beta", "f_gamma", "f_delta", "f_eps", "f_zeta"]
TARGETS = [("Y_t_1h", 1), ("Y_t_4h", 4)]


def _synthetic_frames(n: int = 720, seed: int = 7, bar_hours: int = 1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq=f"{bar_hours}h", tz="UTC")
    a = rng.normal(size=n)
    frame = pd.DataFrame({"t_decision_utc": ts, "row_id": np.arange(n)})
    frame["f_alpha"] = a
    frame["f_beta"] = a.copy()                       # byte-identical alias
    frame["f_gamma"] = 2.0 * a + 1.0                 # exact affine alias
    frame["f_delta"] = a ** 2 + 0.01 * rng.normal(size=n)   # nonlinear, low Pearson (a symmetric)
    frame["f_eps"] = rng.normal(size=n)              # independent
    sparse = np.full(n, np.nan)
    sparse[:10] = rng.normal(size=10)
    frame["f_zeta"] = sparse                          # insufficient support
    targets = pd.DataFrame({"t_decision_utc": ts, "row_id": np.arange(n)})
    targets["Y_t_1h"] = np.roll(a, -1) + 0.5 * rng.normal(size=n)
    targets["Y_t_4h"] = np.roll(a, -4) + 0.5 * rng.normal(size=n)
    targets.loc[n - 4:, "Y_t_4h"] = np.nan
    targets.loc[n - 1:, "Y_t_1h"] = np.nan
    return frame, targets


def _population(tmp_path: Path, *, n: int = 720, seed: int = 7, columns=None, name: str = "SYN"):
    man = _cap("fs_phase23_manifest")
    frame, targets = _synthetic_frames(n=n, seed=seed)
    if columns is not None:
        frame = frame[["t_decision_utc", "row_id", *columns]]
    data = tmp_path / f"data_{name}"
    data.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(data / "features_train.parquet", index=False)
    targets.to_parquet(data / "targets_train.parquet", index=False)
    features_meta = []
    for i, f in enumerate(FEATURES):
        features_meta.append({
            "feature_id": f, "family": "syn", "source": "lake:syn/a.parquet" if f != "f_eps" else "lake:syn/b.parquet",
            "transform": "identity" if f in ("f_alpha", "f_beta") else f"t_{i}", "unit": "u",
            "availability_time": "t (bar end)", "clock": "OBSERVED", "support_h": 1.0,
            "train_coverage": 1.0 if f != "f_zeta" else 10 / n, "source_bytes": 8 * n,
        })
    manifest = man.build_manifest(
        population_id="SYN", identity=f"phase1-final:SYN:{'ab' * 8}",
        identity_source={"file": "synthetic", "field": "synthetic"},
        phase1={"plan_sha256": "0" * 64},
        data={"features_file": "features_train.parquet", "features_sha256": man.sha256_file(data / "features_train.parquet"),
              "targets_file": "targets_train.parquet", "targets_sha256": man.sha256_file(data / "targets_train.parquet"),
              "timestamp_column": "t_decision_utc", "row_id_column": "row_id", "bar_hours": 1, "train_rows": n},
        features=features_meta,
        targets=[{"target_id": t, "column": t, "family": "Y_t", "head": "short", "horizon_hours": h} for t, h in TARGETS],
        folds=[{"fold_id": "inner_1", "train_rows": [0, 300]}, {"fold_id": "inner_2", "train_rows": [0, 500]}],
        causal_supported=[{"feature_id": "f_alpha", "target_id": "Y_t_1h", "horizon_hours": 1}],
        contract_source="synthetic_test",
    )
    path = tmp_path / f"{name}_MANIFEST.json"
    man.write_manifest(manifest, path)
    return path, data


def _plan_and_run(tmp_path, manifest_path, data, *, n_shards=3, host="h1", name="run"):
    camp = _cap("feature_pairwise_campaign")
    state = tmp_path / f"state_{name}"
    plan = camp.plan(manifest_path=manifest_path, state_root=state, n_shards=n_shards,
                     hosts=[{"host_id": host, "size_class": "large"}])
    summary = camp.run_worker(plan_path=state / "PLAN.json", state_root=state, data_root=data, host_id=host)
    return camp, state, plan, summary


def _terminal_rows(state: Path, table="feature_pair_metrics"):
    rows = []
    for path in sorted((state / "terminals").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rows.extend(json.load(fh)["rows"][table])
    return rows


# --------------------------------------------------------------------------- manifest

def test_committed_manifests_bind_phase1_identities_and_denominators():
    man = _cap("fs_phase23_manifest")
    root = REPO / "fs_phase23" / "manifest"
    eur = man.load_manifest(root / "EURUSD_MANIFEST.json")
    eth = man.load_manifest(root / "ETH_MANIFEST.json")
    assert eur["identity"] == "phase1-eurusd-final:94d20c038d55e152"
    assert eth["identity"] == "phase1-final:ETH:29d2f745f5d9e87c"
    assert len(eur["features"]) == 366 and len(eur["targets"]) == 14 and len(eur["folds"]) == 5
    assert len(eth["features"]) == 83 and len(eth["targets"]) == 6 and len(eth["folds"]) == 3
    assert eur["expected_pairs"] == 66795 and eth["expected_pairs"] == 3403
    assert len({f["feature_id"] for f in eur["causal_supported"]}) == 12
    assert len({f["feature_id"] for f in eth["causal_supported"]}) == 3
    for m in (eur, eth):
        assert m["causal_supported_label"] == "CAUSAL_SUPPORTED"
        assert m["final_selection"] is False
        assert all(set(f) == {"fold_id", "train_rows"} for f in m["folds"])   # no validation ranges retained
        assert m["data"]["features_sha256"] and m["data"]["targets_sha256"]
        assert man.manifest_digest(m) == m["manifest_sha256"]
        text = json.dumps(m)
        assert "/home/" not in text and "127.0.0.1" not in text


def test_manifest_identity_is_column_order_invariant(tmp_path):
    man = _cap("fs_phase23_manifest")
    p1, _ = _population(tmp_path, name="A")
    p2, _ = _population(tmp_path, name="B", columns=list(reversed(FEATURES)))
    a, b = man.load_manifest(p1), man.load_manifest(p2)
    assert [f["feature_id"] for f in a["features"]] == [f["feature_id"] for f in b["features"]]
    assert man.population_digest(a) == man.population_digest(b)


# --------------------------------------------------------------------------- phase 2 worker

def test_plan_is_deterministic_and_shards_cover_every_pair(tmp_path):
    camp = _cap("feature_pairwise_campaign")
    manifest_path, _ = _population(tmp_path)
    s1 = camp.plan(manifest_path=manifest_path, state_root=tmp_path / "s1", n_shards=4,
                   hosts=[{"host_id": "small", "size_class": "small"}, {"host_id": "big", "size_class": "large"}])
    s2 = camp.plan(manifest_path=manifest_path, state_root=tmp_path / "s2", n_shards=4,
                   hosts=[{"host_id": "small", "size_class": "small"}, {"host_id": "big", "size_class": "large"}])
    assert s1["plan_sha256"] == s2["plan_sha256"]
    assert sum(s["pair_count"] for s in s1["shards"]) == 15
    pairs = set()
    for shard in s1["shards"]:
        for left, right in camp.shard_pairs(s1, shard["shard_index"]):
            assert left < right
            pairs.add((left, right))
    assert len(pairs) == 15
    assert {s["host_id"] for s in s1["shards"]} == {"small", "big"}
    small_cost = max(s["estimated_cost"] for s in s1["shards"] if s["host_id"] == "small")
    big_cost = min(s["estimated_cost"] for s in s1["shards"] if s["host_id"] == "big")
    assert small_cost <= big_cost
    assert all(len(s["unit_id"]) == 64 for s in s1["shards"])


def test_future_row_mutation_does_not_change_earlier_fold_results(tmp_path):
    man = _cap("fs_phase23_manifest")
    manifest_path, data = _population(tmp_path, name="base")
    camp, state, plan, _ = _plan_and_run(tmp_path, manifest_path, data, name="base")
    base = {r["row_key"]: r for r in _terminal_rows(state) if r["fold_id"] == "inner_1"}
    # mutate rows after fold inner_1 (rows >= 300), rewrite parquet and manifest digest
    frame = pd.read_parquet(data / "features_train.parquet")
    frame.loc[300:, "f_alpha"] = frame.loc[300:, "f_alpha"].to_numpy()[::-1] * 3.0 + 7.0
    frame.loc[300:, "f_eps"] = 0.0
    frame.to_parquet(data / "features_train.parquet", index=False)
    m = man.load_manifest(manifest_path)
    m["data"]["features_sha256"] = man.sha256_file(data / "features_train.parquet")
    m["identity"] = "phase1-final:SYN:" + "cd" * 8
    man.write_manifest(m, manifest_path)
    _, state2, _, _ = _plan_and_run(tmp_path, manifest_path, data, name="mut")
    mutated = {r["row_key"]: r for r in _terminal_rows(state2) if r["fold_id"] == "inner_1"}
    assert len(base) == len(mutated) > 0
    strip = lambda r: {k: v for k, v in r.items() if k not in ("row_key", "run_id", "unit_id")}
    key = lambda r: (r["left"], r["right"], r["fold_id"], r["metric"], r["lag_hours"])
    a = {key(r): strip(r) for r in base.values()}
    b = {key(r): strip(r) for r in mutated.values()}
    assert a == b


def test_column_order_does_not_change_pair_results(tmp_path):
    pa, da = _population(tmp_path, name="ord_a")
    pb, db = _population(tmp_path, name="ord_b", columns=list(reversed(FEATURES)))
    _, sa, _, _ = _plan_and_run(tmp_path, pa, da, name="ord_a")
    _, sb, _, _ = _plan_and_run(tmp_path, pb, db, name="ord_b")
    key = lambda r: (r["left"], r["right"], r["fold_id"], r["metric"], r["lag_hours"])
    strip = lambda r: {k: v for k, v in r.items() if k not in ("row_key", "run_id", "unit_id")}
    ra = {key(r): strip(r) for r in _terminal_rows(sa)}
    rb = {key(r): strip(r) for r in _terminal_rows(sb)}
    assert ra == rb
    ga = {(r["left"], r["right"]): r["gate_state"] for r in _terminal_rows(sa, "feature_pair_gate")}
    gb = {(r["left"], r["right"]): r["gate_state"] for r in _terminal_rows(sb, "feature_pair_gate")}
    assert ga == gb


def test_restart_is_byte_identical_and_never_recomputes_complete_units(tmp_path):
    manifest_path, data = _population(tmp_path)
    camp, state, plan, first = _plan_and_run(tmp_path, manifest_path, data)
    files = sorted((state / "terminals").glob("*.json.gz"))
    assert len(files) == len(plan["shards"]) and first["computed"] == len(files)
    before = {f.name: (f.read_bytes(), f.stat().st_ino) for f in files}
    second = camp.run_worker(plan_path=state / "PLAN.json", state_root=state, data_root=data, host_id="h1")
    assert second["computed"] == 0 and second["adopted_existing"] == len(files)
    after = {f.name: (f.read_bytes(), f.stat().st_ino) for f in sorted((state / "terminals").glob("*.json.gz"))}
    assert before == after
    # a fresh state root recomputes the same bytes (payload identity excludes wall-clock fields)
    _, state2, _, _ = _plan_and_run(tmp_path, manifest_path, data, name="again")
    for f in files:
        t1 = camp.load_terminal(f)
        t2 = camp.load_terminal(state2 / "terminals" / f.name)
        assert t1["rows_sha256"] == t2["rows_sha256"]
        assert t1["rows"] == t2["rows"]


def test_restart_submission_produces_no_duplicate_rows(tmp_path):
    camp = _cap("feature_pairwise_campaign")
    manifest_path, data = _population(tmp_path)
    _, state, plan, _ = _plan_and_run(tmp_path, manifest_path, data)
    wh_path = tmp_path / "wh.db"
    r1 = camp.follow_once(plan_path=state / "PLAN.json", state_root=state, terminal_dirs=[state / "terminals"],
                          warehouse_path=wh_path, data_root=data)
    r2 = camp.follow_once(plan_path=state / "PLAN.json", state_root=state, terminal_dirs=[state / "terminals"],
                          warehouse_path=wh_path, data_root=data)
    wh = camp.open_warehouse(wh_path)
    counts = wh.reconcile(plan["identity"])["tables"]
    expected_rows = sum(t["row_counts"]["feature_pair_metrics"] for t in
                        (camp.load_terminal(p) for p in (state / "terminals").glob("*.json.gz")))
    assert counts["feature_pair_metrics"]["count"] == expected_rows
    assert r2["submitted"] == 0 and r1["submitted"] == len(plan["shards"])
    # submitting the very same rows again is ignored by the unique key
    rows = _terminal_rows(state)[:5]
    receipt = wh.submit_rows(plan["identity"], "feature_pair_metrics", rows)
    assert receipt["inserted"] == 0 and receipt["duplicates_ignored"] == 5
    assert wh.reconcile(plan["identity"])["tables"]["feature_pair_metrics"]["count"] == expected_rows


def test_synthetic_alias_is_grouped_with_a_representative(tmp_path):
    camp = _cap("feature_pairwise_campaign")
    manifest_path, data = _population(tmp_path)
    _, state, plan, _ = _plan_and_run(tmp_path, manifest_path, data)
    gate = {(r["left"], r["right"]): r for r in _terminal_rows(state, "feature_pair_gate")}
    assert gate[("f_alpha", "f_beta")]["gate_state"] == "BYTE_IDENTICAL"
    assert gate[("f_alpha", "f_gamma")]["gate_state"] == "EXACT_AFFINE"
    assert gate[("f_alpha", "f_eps")]["gate_state"] == "DISTINCT"
    assert gate[("f_alpha", "f_beta")]["same_source"] is True and gate[("f_alpha", "f_eps")]["same_source"] is False
    groups = camp.alias_groups_from_gate_rows(list(gate.values()), camp.load_manifest(manifest_path))
    members = {g["alias_group_id"]: set(g["members"]) for g in groups}
    assert {"f_alpha", "f_beta", "f_gamma"} in members.values()
    group = next(g for g in groups if set(g["members"]) == {"f_alpha", "f_beta", "f_gamma"})
    assert group["representative"] in group["members"]
    assert group["disposition"] == "ALIAS_GROUP" and group["dropped_columns"] == []


def test_nonlinear_pair_is_caught_by_mi_and_dcor_with_low_pearson(tmp_path):
    manifest_path, data = _population(tmp_path)
    _, state, _, _ = _plan_and_run(tmp_path, manifest_path, data)
    rows = {(r["left"], r["right"], r["metric"]): r for r in _terminal_rows(state) if r["fold_id"] == "TRAIN" and r["lag_hours"] == 0}
    pear = rows[("f_alpha", "f_delta", "pearson")]
    mi = rows[("f_alpha", "f_delta", "mutual_information")]
    dcor = rows[("f_alpha", "f_delta", "distance_correlation")]
    assert pear["state"] == "MEASURED" and abs(pear["value"]) < 0.2
    assert mi["state"] == "MEASURED" and mi["value"] > 0.5
    assert dcor["state"] == "MEASURED" and dcor["value"] > 0.4
    assert mi["estimator"].startswith("quantile_bins") and "seed" in mi["params"]
    indep = rows[("f_alpha", "f_eps", "mutual_information")]
    assert indep["value"] < mi["value"] / 4


def test_insufficient_support_abstains_with_diagnostic(tmp_path):
    manifest_path, data = _population(tmp_path)
    _, state, _, _ = _plan_and_run(tmp_path, manifest_path, data)
    rows = [r for r in _terminal_rows(state) if "f_zeta" in (r["left"], r["right"])]
    assert rows
    assert all(r["state"] == "INSUFFICIENT_SUPPORT" and r["value"] is None for r in rows)
    assert all(r["support_n"] <= 10 and r["reason"] for r in rows)
    stab = [r for r in _terminal_rows(state, "feature_pair_stability") if "f_zeta" in (r["left"], r["right"])]
    assert stab and all(r["valid_folds"] == 0 and r["state"] == "INSUFFICIENT_SUPPORT" for r in stab)


def test_lagged_cross_correlation_uses_elapsed_hours(tmp_path):
    manifest_path, data = _population(tmp_path)
    _, state, _, _ = _plan_and_run(tmp_path, manifest_path, data)
    rows = {(r["left"], r["right"], r["metric"], r["lag_hours"]): r for r in _terminal_rows(state) if r["fold_id"] == "TRAIN"}
    lags = sorted({k[3] for k in rows})
    assert lags == [0, 1, 2, 6, 24, 48, 168]
    r = rows[("f_alpha", "f_eps", "xcorr_x_leads", 24)]
    assert r["state"] == "MEASURED" and r["support_n"] == 720 - 24
    assert ("f_alpha", "f_eps", "xcorr_y_leads", 24) in rows
    assert rows[("f_alpha", "f_eps", "xcorr", 0)]["value"] == rows[("f_alpha", "f_eps", "pearson", 0)]["value"]


def test_no_process_opens_validation_or_test(tmp_path, monkeypatch):
    man = _cap("fs_phase23_manifest")
    camp = _cap("feature_pairwise_campaign")
    manifest_path, data = _population(tmp_path)
    opened = []
    real = pd.read_parquet

    def spy(path, *a, **k):
        opened.append(Path(path).name)
        return real(path, *a, **k)
    monkeypatch.setattr(pd, "read_parquet", spy)
    _plan_and_run(tmp_path, manifest_path, data)
    assert set(opened) <= {"features_train.parquet"}
    m = man.load_manifest(manifest_path)
    m["data"]["features_file"] = "features_validation.parquet"
    (data / "features_validation.parquet").write_bytes((data / "features_train.parquet").read_bytes())
    m["data"]["features_sha256"] = man.sha256_file(data / "features_validation.parquet")
    bad = tmp_path / "bad.json"
    man.write_manifest(m, bad)
    with pytest.raises(camp.CampaignError, match="validation|test"):
        camp.plan(manifest_path=bad, state_root=tmp_path / "bad_state", n_shards=2, hosts=[{"host_id": "h", "size_class": "large"}])
    m["data"]["features_file"] = "features_test.parquet"
    man.write_manifest(m, bad)
    with pytest.raises(camp.CampaignError, match="validation|test"):
        camp.plan(manifest_path=bad, state_root=tmp_path / "bad_state2", n_shards=2, hosts=[{"host_id": "h", "size_class": "large"}])


def test_foreign_identity_warehouse_receipt_is_rejected(tmp_path):
    camp = _cap("feature_pairwise_campaign")
    manifest_path, data = _population(tmp_path)
    _, state, plan, _ = _plan_and_run(tmp_path, manifest_path, data)
    wh = camp.open_warehouse(tmp_path / "wh.db")
    unit = plan["shards"][0]["unit_id"]
    terminal = camp.load_terminal(state / "terminals" / f"{unit}.json.gz")
    good = wh.submit_rows(plan["identity"], "feature_pair_metrics", terminal["rows"]["feature_pair_metrics"])
    foreign = dict(good, run_id="phase1-final:OTHER:" + "ff" * 8)
    with pytest.raises(camp.CampaignError, match="identity"):
        camp.accept_receipt(plan, terminal, foreign)
    tampered = dict(good, rows_sha256="0" * 64)
    with pytest.raises(camp.CampaignError, match="digest"):
        camp.accept_receipt(plan, terminal, tampered)
    assert camp.accept_receipt(plan, terminal, good)["unit_id"] == unit


def test_corrupt_terminal_is_quarantined_and_valid_one_adopted(tmp_path):
    camp = _cap("feature_pairwise_campaign")
    manifest_path, data = _population(tmp_path)
    _, state, plan, _ = _plan_and_run(tmp_path, manifest_path, data)
    remote = tmp_path / "remote"
    remote.mkdir()
    files = sorted((state / "terminals").glob("*.json.gz"))
    for f in files:
        (remote / f.name).write_bytes(f.read_bytes())
    victim = files[0]
    t = camp.load_terminal(victim)
    t["rows"]["feature_pair_metrics"][0]["value"] = 123.0     # content no longer matches terminal_sha256
    with gzip.open(remote / victim.name, "wt", encoding="utf-8") as fh:
        json.dump(t, fh)
    coord = tmp_path / "coord"
    res = camp.follow_once(plan_path=state / "PLAN.json", state_root=coord, terminal_dirs=[remote],
                           warehouse_path=tmp_path / "wh.db", data_root=data)
    assert res["quarantined"] == 1 and res["adopted"] == len(files) - 1
    assert (coord / "quarantine" / victim.name).exists()
    assert not (coord / "adopted" / victim.name).exists()
    assert res["phase2_closed"] is False
    status = json.loads((coord / "STATUS.json").read_text())
    assert status["populations"]["SYN"]["pairwise_v1"]["complete"] == len(files) - 1


def test_phase2_closure_fails_on_one_missing_disposition(tmp_path):
    camp = _cap("feature_pairwise_campaign")
    manifest_path, data = _population(tmp_path)
    _, state, plan, _ = _plan_and_run(tmp_path, manifest_path, data)
    coord = tmp_path / "coord"
    res = camp.follow_once(plan_path=state / "PLAN.json", state_root=coord, terminal_dirs=[state / "terminals"],
                           warehouse_path=tmp_path / "wh.db", data_root=data)
    assert res["phase2_closed"] is True
    closure = json.loads((coord / "PHASE_2_COMPLETE.json").read_text())
    assert closure["expected_pairs"] == 15 and closure["observed_pairs"] == 15
    assert closure["generated_from_evidence"] is True
    # drop one disposition from one adopted terminal, re-seal it, and close again
    path = sorted((coord / "adopted").glob("*.json.gz"))[0]
    t = camp.load_terminal(path)
    dropped = t["rows"]["feature_pair_metrics"].pop()
    resealed = camp.seal_terminal({k: v for k, v in t.items() if k not in ("rows_sha256", "terminal_sha256", "row_counts")})
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(resealed, fh)
    (coord / "PHASE_2_COMPLETE.json").unlink()
    with pytest.raises(camp.CampaignError) as err:
        camp.close_phase2(plan_path=state / "PLAN.json", state_root=coord, warehouse_path=tmp_path / "wh.db", data_root=data)
    assert dropped["row_key"] in str(err.value) or "missing" in str(err.value)
    assert not (coord / "PHASE_2_COMPLETE.json").exists()


# --------------------------------------------------------------------------- phase 3

def _closed_phase2(tmp_path):
    camp = _cap("feature_pairwise_campaign")
    manifest_path, data = _population(tmp_path)
    _, state, plan, _ = _plan_and_run(tmp_path, manifest_path, data)
    coord = tmp_path / "coord"
    res = camp.follow_once(plan_path=state / "PLAN.json", state_root=coord, terminal_dirs=[state / "terminals"],
                           warehouse_path=tmp_path / "wh.db", data_root=data, chain_phase3=False)
    assert res["phase2_closed"]
    return camp, manifest_path, data, state, plan, coord


def test_mrmr_and_jmi_reject_incomplete_matrices_or_mixed_populations(tmp_path):
    sel = _cap("feature_filter_selection")
    camp, manifest_path, data, state, plan, coord = _closed_phase2(tmp_path)
    mats = sel.load_phase2_matrices(coord / "PHASE2_MATRICES.npz")
    manifest = camp.load_manifest(manifest_path)
    feats, X, y = sel.load_train_matrix(manifest, data, "Y_t_1h")
    good = sel.run_filter_methods(manifest, mats, feats, X, y, target_id="Y_t_1h", seed=1)
    assert {"MRMR", "JMI"} <= set(good["methods"])
    broken = dict(mats)
    mi = broken["mi"].copy()
    ia, ib = mats["features"].index("f_alpha"), mats["features"].index("f_delta")   # two admissible features
    assert mats["admissible"][ia] and mats["admissible"][ib]
    mi[ia, ib] = mi[ib, ia] = np.nan
    broken["mi"] = mi
    with pytest.raises(sel.SelectionError, match="incomplete"):
        sel.run_filter_methods(manifest, broken, feats, X, y, target_id="Y_t_1h", seed=1)
    mixed = dict(mats, population_id="OTHER")
    with pytest.raises(sel.SelectionError, match="population"):
        sel.run_filter_methods(manifest, mixed, feats, X, y, target_id="Y_t_1h", seed=1)
    mixed2 = dict(mats, identity="phase1-final:OTHER:" + "00" * 8)
    with pytest.raises(sel.SelectionError, match="population|identity"):
        sel.run_filter_methods(manifest, mixed2, feats, X, y, target_id="Y_t_1h", seed=1)


def test_phase3_outputs_label_causal_support_and_declare_no_winner(tmp_path):
    camp, manifest_path, data, state, plan, coord = _closed_phase2(tmp_path)
    res = camp.run_phase3(plan_path=state / "PLAN.json", state_root=coord, data_root=data, warehouse_path=tmp_path / "wh.db")
    closure = camp.close_phase3(plan_path=state / "PLAN.json", state_root=coord, warehouse_path=tmp_path / "wh.db")
    assert (coord / "PHASE_3_FILTER_COMPLETE.json").exists()
    cand = json.loads((coord / "CANDIDATES_FOR_VALIDATION.json").read_text())
    assert cand["predictive_winner"] is None and cand["uses_test_split"] is False and cand["final_selection"] is False
    assert cand["k_grid"] == [4, 8, 12, 16, 24, 32]
    methods = {c["method"] for c in cand["candidates"]}
    assert {"SPEARMAN_CLUSTER", "MRMR", "JMI", "ALL_ADMISSIBLE", "UNIVARIATE_MI", "CAUSAL_SUPPORTED", "RANDOM_K"} <= methods
    assert all(c["k"] <= cand["admissible_features"]["SYN"] for c in cand["candidates"])
    causal = [c for c in cand["candidates"] if c["method"] == "CAUSAL_SUPPORTED" and c["target_id"] == "Y_t_1h"]
    assert causal and all(c["label"] == "CAUSAL_SUPPORTED" and c["is_final_selection"] is False for c in causal)
    assert "f_alpha" in causal[0]["members"] or "f_beta" in causal[0]["members"] or "f_gamma" in causal[0]["members"]
    rank_rows = camp.open_warehouse(tmp_path / "wh.db").read_run(plan["identity"], "feature_filter_rankings")
    terms = {"relevance", "redundancy", "complementarity", "causality", "cost"}
    assert rank_rows and all(terms <= set(r["terms"]) for r in rank_rows)
    assert closure["targets"] == 2 and closure["generated_from_evidence"] is True
    # alias members are not counted as independent evidence: at most one of the alias trio per subset
    for c in cand["candidates"]:
        if c["method"] in ("MRMR", "JMI", "SPEARMAN_CLUSTER"):
            assert len({"f_alpha", "f_beta", "f_gamma"} & set(c["members"])) <= 1


def test_follow_chains_phase3_automatically_when_phase2_closes(tmp_path):
    camp = _cap("feature_pairwise_campaign")
    manifest_path, data = _population(tmp_path)
    _, state, plan, _ = _plan_and_run(tmp_path, manifest_path, data)
    coord = tmp_path / "coord"
    res = camp.follow_once(plan_path=state / "PLAN.json", state_root=coord, terminal_dirs=[state / "terminals"],
                           warehouse_path=tmp_path / "wh.db", data_root=data)
    assert res["phase2_closed"] and res["phase3_closed"]
    assert (coord / "PHASE_2_COMPLETE.json").exists() and (coord / "PHASE_3_FILTER_COMPLETE.json").exists()
    status = json.loads((coord / "STATUS.json").read_text())
    assert status["phase"] == "PHASE_3_COMPLETE"


# --------------------------------------------------------------------------- status

def test_status_reports_expected_complete_failed_active_rate_and_eta(tmp_path):
    st = _cap("feature_selection_phase23_status")
    camp = _cap("feature_pairwise_campaign")
    manifest_path, data = _population(tmp_path)
    state = tmp_path / "state"
    plan = camp.plan(manifest_path=manifest_path, state_root=state, n_shards=3, hosts=[{"host_id": "h1", "size_class": "large"}])
    camp.run_worker(plan_path=state / "PLAN.json", state_root=state, data_root=data, host_id="h1", max_shards=1)
    (state / "claims").mkdir(exist_ok=True)
    camp.write_claim(state, plan["shards"][2]["unit_id"], host_id="h1")
    (state / "failures").mkdir(exist_ok=True)
    (state / "failures" / f"{plan['shards'][1]['unit_id']}.x.json").write_text(json.dumps(
        {"unit_id": plan["shards"][1]["unit_id"], "reason": "synthetic failure", "host_id": "h1"}))
    status = st.build_status(plan_paths=[state / "PLAN.json"], terminal_dirs=[state / "terminals"],
                             state_roots=[state], out_path=state / "STATUS.json")
    cell = status["populations"]["SYN"]["pairwise_v1"]
    assert cell["expected"] == 3 and cell["complete"] == 1 and cell["failed"] == 1 and cell["active"] == 1
    assert cell["rate_units_per_hour"] is None or cell["rate_units_per_hour"] >= 0
    assert "eta_seconds" in cell and "by_host" in cell and cell["by_host"]["h1"]["expected"] == 3
    assert cell["failures"][0]["reason"] == "synthetic failure"
    assert status["generated_at"] and status["schema"].startswith("fs_phase23")
    assert json.loads((state / "STATUS.json").read_text())["populations"]["SYN"]["pairwise_v1"]["complete"] == 1


# --------------------------------------------------------------------------- regressions 2026-10-06 (follower incident)

def test_phase3_terminal_digest_survives_write_read_with_multi_k_subsets(tmp_path):
    """Integer K keys sorted differently in memory and after JSON reload made every phase-3 terminal 'corrupt'."""
    camp = _cap("feature_pairwise_campaign")
    doc = {"schema": "fs_phase23.filter_terminal.v1", "unit_id": "u" * 64, "target_id": "Y",
           "methods": {"MRMR": {"subsets": {4: ["a"], 8: ["b"], 12: ["c"], 16: ["d"], 24: ["e"], 32: ["f"]}}},
           "rows": {"feature_filter_rankings": [{"row_key": "k1", "terms": {}}], "feature_filter_subsets": [{"row_key": "k2", "k": 12}]}}
    sealed = camp.seal_terminal(doc)
    path = tmp_path / "t.json.gz"
    camp.write_terminal(path, sealed)
    back = camp.load_terminal(path)
    assert camp._digest({k: v for k, v in back.items() if k != "terminal_sha256"}) == back["terminal_sha256"]
    assert camp._phase3_terminal_valid(path)


def test_follow_survives_store_exception_on_one_unit(tmp_path):
    """A store/engine exception while submitting one unit is recorded and the other units still get receipts."""
    camp = _cap("feature_pairwise_campaign")
    manifest_path, data = _population(tmp_path)
    _, state, plan, _ = _plan_and_run(tmp_path, manifest_path, data)
    victim = plan["shards"][1]["unit_id"]

    class Flaky:
        def __init__(self, inner):
            self.inner = inner

        def submit_rows(self, run_id, table, rows, **kw):
            if rows and rows[0].get("unit_id") == victim:
                raise RuntimeError('Constraint Error: Duplicate key "row_identity_sha256: deadbeef" violates primary key constraint.')
            return self.inner.submit_rows(run_id, table, rows)

        def read_run(self, *a, **k):
            return self.inner.read_run(*a, **k)

        def reconcile(self, *a, **k):
            return self.inner.reconcile(*a, **k)

    healthy = camp.open_warehouse(tmp_path / "wh.db")
    coord = tmp_path / "coord"
    res = camp.follow_once(plan_path=state / "PLAN.json", state_root=coord, terminal_dirs=[state / "terminals"],
                           warehouse_path=tmp_path / "wh.db", data_root=data, warehouse=Flaky(healthy))
    assert res["submitted"] == len(plan["shards"]) - 1
    assert [e["unit_id"] for e in res["submit_errors"]] == [victim] and "Duplicate key" in res["submit_errors"][0]["reason"]
    assert (coord / "submit_failures" / f"{victim}.json").exists()
    assert res["phase2_closed"] is False and res["closure_error"] is None
    # the store recovers: the next pass submits only the failed unit and closes both phases
    res2 = camp.follow_once(plan_path=state / "PLAN.json", state_root=coord, terminal_dirs=[state / "terminals"],
                            warehouse_path=tmp_path / "wh.db", data_root=data, warehouse=healthy)
    assert res2["submitted"] == 1 and res2["submit_errors"] == [] and res2["phase2_closed"] and res2["phase3_closed"]
    assert not (coord / "submit_failures" / f"{victim}.json").exists()


def test_run_phase3_quarantines_and_recomputes_a_corrupt_phase3_terminal(tmp_path):
    camp, manifest_path, data, state, plan, coord = _closed_phase2(tmp_path)
    camp.run_phase3(plan_path=state / "PLAN.json", state_root=coord, data_root=data, warehouse_path=tmp_path / "wh.db")
    path = sorted((coord / "phase3" / "terminals").glob("*.json.gz"))[0]
    doc = camp.load_terminal(path)
    doc["admissible_count"] = doc["admissible_count"] + 1          # content no longer matches its digest
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh)
    with pytest.raises(camp.CampaignError, match="corrupt"):
        camp.close_phase3(plan_path=state / "PLAN.json", state_root=coord, warehouse_path=tmp_path / "wh.db")
    res = camp.run_phase3(plan_path=state / "PLAN.json", state_root=coord, data_root=data, warehouse_path=tmp_path / "wh.db")
    assert res["quarantined"] == 1 and res["computed"] == 1 and res["existing"] == len(plan["targets"]) - 1
    assert list((coord / "phase3" / "quarantine").glob("*.json.gz"))
    assert camp.close_phase3(plan_path=state / "PLAN.json", state_root=coord, warehouse_path=tmp_path / "wh.db")["targets"] == 2


def test_rebuild_pk_index_preserves_rows_and_digests(tmp_path):
    pytest.importorskip("duckdb")
    try:
        from tools import fs_phase23_warehouse as wh_mod
    except ImportError:
        pytest.skip("DATA warehouse module not present")
    camp = _cap("feature_pairwise_campaign")
    rebuild = _cap("fs_phase23_deploy.rebuild_duckdb_pk_index")
    manifest_path, data = _population(tmp_path)
    _, state, plan, _ = _plan_and_run(tmp_path, manifest_path, data)
    db = tmp_path / "store.duckdb"
    wh = wh_mod.open_warehouse(db)
    unit = plan["shards"][0]["unit_id"]
    t = camp.load_terminal(state / "terminals" / f"{unit}.json.gz")
    for table, rows in t["rows"].items():
        wh.submit_rows(plan["identity"], table, rows, host_role="coordinator")
    before = wh.reconcile(plan["identity"])["tables"]
    wh.close()
    assert rebuild.main(["--warehouse", str(db)]) == 0
    wh = wh_mod.open_warehouse(db)
    after = wh.reconcile(plan["identity"])["tables"]
    assert {k: (v["count"], v["rows_sha256"]) for k, v in before.items()} == {k: (v["count"], v["rows_sha256"]) for k, v in after.items()}
    receipt = wh.submit_rows(plan["identity"], "feature_pair_gate", t["rows"]["feature_pair_gate"], host_role="coordinator")
    assert receipt["inserted"] == 0 and receipt["duplicates_ignored"] == len(t["rows"]["feature_pair_gate"])
    wh.close()
