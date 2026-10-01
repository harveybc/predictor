"""M04 DOIN execution contract (MS13). Synthetic inputs prove mechanics only.

T1 reversible flat<->nested mapping; T2 invalid/conditional parameters fail before
fit; T3 candidates persisted before execution; T4 resume without repeating
completed work; T5 paired-seed incumbent from verified candidates only; T6 real
engine -> receipt -> independent checkpoint rescoring VERIFIED, tampering REFUTED;
T7 heartbeat cadence/fields; T8 optimizer.plugins surface returns incumbent flat.
"""
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pytest

from tools import modular_doin_campaign as camp
from tools import modular_search_space as ss

ROOT = Path(__file__).resolve().parents[1]
SPACE = json.loads((ROOT / "examples/config/modular_doin/ecl_l24_h24_search_space_v1.json").read_text())
DEFAULT = json.loads((ROOT / "examples/config/modular_doin/ecl_l24_h24_default_r0_v1.json").read_text())
FEATURES = [f"c{i}" for i in range(321)]
BASE = {"feature_names": FEATURES, "window": 24, "sample_hours": 1, "horizons": list(range(1, 25)),
        "target_feature_indices": list(range(321)),
        "objective": {"metric": "MAE", "split": "validation", "higher_is_better": False, "unit": "z_train"},
        "evaluator_fixed": {"max_updates": 1000000, "max_seconds": 3600.0}, "donors": {}}


def flat_default(**overrides):
    flat = {**DEFAULT, "train.seed": 2021, "train.huber_delta": 1.0}
    flat.update(overrides)
    return {k: v for k, v in flat.items() if v is not None}


# ------------------------------------------------------------------ T1 --
def test_default_round_trip_is_exact():
    flat = flat_default()
    nested = ss.from_flat(flat, BASE, SPACE)
    assert ss.to_flat(nested, SPACE) == flat
    assert ss.from_flat(ss.to_flat(nested, SPACE), BASE, SPACE) == nested
    assert len(nested["model"]["branches"]) == 321
    assert nested["model"]["core"]["params"]["stage_channels"] == [32, 16, 8]
    assert nested["model"]["core"]["params"]["time_factors"] == [2, 1, 1]


def test_sampled_round_trips_are_exact():
    rng = random.Random(7)
    for _ in range(60):
        flat = camp.propose(SPACE, rng, fixed={"branch.regime": "R0", "core.regime": "R0"})
        flat["train.seed"] = 2022
        nested = ss.from_flat(flat, BASE, SPACE)
        assert ss.to_flat(nested, SPACE) == flat


def test_corrected_default_matches_integrated_engine_and_old_default_is_refused():
    mt = pytest.importorskip("predictor_plugins.modular_temporal")
    v2 = json.loads((ROOT / "examples/config/modular_doin/ecl_l24_h24_default_r0_v2_full_grid.json").read_text())
    space_v2 = json.loads((ROOT / "examples/config/modular_doin/ecl_l24_h24_search_space_v2_full_grid.json").read_text())
    nested = ss.from_flat({**v2, "branch.grouping_size": 321, "train.seed": 2021, "train.huber_delta": 1.0},
                          BASE, space_v2)
    normalized = mt._normalize(nested["model"])
    assert normalized["core"]["params"]["d_model"] == 64
    assert normalized["branch_steps"] == 24 and normalized["output_steps"] == 6
    old = ss.from_flat(flat_default(**{"branch.grouping_size": 321}), BASE, SPACE)  # branch_steps 12
    with pytest.raises(ValueError, match="branch_steps must equal window"):
        mt._normalize(old["model"])


# ------------------------------------------------------------------ T2 --
@pytest.mark.parametrize("overrides,match", [
    ({"core.heads": 8, "core.d_model": 32, "core.stage_channels_0": 24, "core.stage_channels_1": 12}, None),
    ({"core.d_model": 32, "core.heads": 8, "core.stage_channels_0": 48}, "strictly decrease"),
    ({"core.time_factor_0": 3}, "multiply"),
    ({"model.output_steps": 6, "model.branch_steps": 6, "core.time_factor_0": 2}, "multiply"),
    ({"train.loss": "mae"}, "inactive"),
    ({"train.huber_delta": None}, "missing"),
    ({"branch.dilation_rate": 2}, "outside declared choices"),
    ({"core.dropout": 0.9}, "outside"),
    ({"core.stage_count": 4}, "missing"),
    ({"train.learning_rate": 1.0}, "outside"),
])
def test_invalid_combinations_fail_before_fit(overrides, match):
    flat = flat_default(**overrides)
    if match is None:
        ss.from_flat(flat, BASE, SPACE)
        return
    with pytest.raises(ss.SearchSpaceError, match=match):
        ss.from_flat(flat, BASE, SPACE)


def test_engine_capability_gate_refuses_unsupported_dilation():
    space = copy.deepcopy(SPACE)
    space["bounds"]["branch.dilation_rate"] = {"choices": [1, 2]}
    with pytest.raises(ss.SearchSpaceError, match="unsupported by engine"):
        ss.from_flat(flat_default(**{"branch.dilation_rate": 2}), BASE, space)


def test_donor_regimes_require_declared_donors():
    with pytest.raises(ss.SearchSpaceError, match="donor"):
        ss.from_flat(flat_default(**{"branch.regime": "R1"}), BASE, SPACE)
    base = copy.deepcopy(BASE)
    base["donors"] = {f"1:branch_{i}": f"/donors/b{i}.keras" for i in range(321)}
    nested = ss.from_flat(flat_default(**{"branch.regime": "R2"}), base, SPACE)
    assert nested["model"]["branches"][5]["donor"] == "/donors/b5.keras"
    assert ss.to_flat(nested, SPACE)["branch.regime"] == "R2"


def test_nonexpressible_nested_candidate_is_refused():
    nested = ss.from_flat(flat_default(), BASE, SPACE)
    nested["model"]["branches"][3]["params"]["channels"] = 32
    with pytest.raises(ss.SearchSpaceError, match="uniform"):
        ss.to_flat(nested, SPACE)


# ------------------------------------------------------------ campaign --
def make_campaign(tmp_path, seeds=(2021, 2022)):
    for split in ("train", "validation"):
        (tmp_path / f"{split}.npz").write_bytes(split.encode())
    decl = {"campaign_id": "test", "search_space": SPACE, "base": BASE, "paired_seeds": list(seeds),
            "default_candidate": DEFAULT, "default_huber_delta": 1.0,
            "data": {s: {"path": str(tmp_path / f"{s}.npz"), "sha256": camp.sha_file(tmp_path / f"{s}.npz")}
                     for s in ("train", "validation")},
            "resources": {"train": {"cap": "1G", "wall": "1h", "timeout_seconds": 60},
                          "verify": {"cap": "1G", "wall": "1h", "timeout_seconds": 60}},
            "executor": {}}
    return camp.Campaign.create(tmp_path / "campaign", decl)


class FakeExecutor:
    def __init__(self, campaign, values, verdict="VERIFIED", crash_on=None):
        self.campaign, self.values, self.verdict, self.crash_on = campaign, values, verdict, crash_on
        self.trained, self.verified = [], []

    def train(self, nested, output_root, declaration):
        cid = ss.digest(nested)
        row = self.campaign.db.execute("SELECT status FROM candidates WHERE cid=?", (cid,)).fetchone()
        assert row["status"] == "running"  # T3: persisted and marked before execution
        if self.crash_on == cid:
            raise KeyboardInterrupt  # simulates the launcher dying mid-run
        self.trained.append(cid)
        value = self.values[(nested["evaluator"]["loss"], nested["evaluator"]["seed"])]
        return {"status": "completed", "objective": value, "receipt_path": str(output_root / "r.json"),
                "data_sha256": {k: declaration["data"][k]["sha256"] for k in ("train", "validation")},
                "observed_updates": 10, "selected_epoch": 2, "per_update_seconds": 0.1, "cgroup_peak_bytes": 1}

    def verify(self, receipt, output_root, declaration):
        self.verified.append(receipt)
        verdict = self.verdict(receipt) if callable(self.verdict) else self.verdict
        return {"status": "completed", "verdict": verdict}


VALUES = {("huber", 2021): 0.30, ("huber", 2022): 0.32, ("mae", 2021): 0.29, ("mae", 2022): 0.35}


def enqueue_default(campaign):
    h, m = camp.paired_loss_arms(DEFAULT, 1.0)
    return campaign.enqueue(h, "default_huber") + campaign.enqueue(m, "default_mae")


def test_candidates_persist_before_execution_and_incumbent_uses_all_paired_seeds(tmp_path):
    campaign = make_campaign(tmp_path)
    added = enqueue_default(campaign)
    assert len(added) == 4
    assert {r[0] for r in campaign.db.execute("SELECT status FROM candidates")} == {"queued"}
    executor = FakeExecutor(campaign, VALUES)
    assert campaign.run(executor) == 4
    status = campaign.status()
    assert status["counts"] == {"verified": 4}
    inc = status["incumbent"]
    # huber mean 0.31 beats mae mean 0.32 although mae has the single best seed (0.29)
    assert inc["mean_objective"] == pytest.approx(0.31)
    labels = {t["config_id"]: t["label"] for t in status["standings"]}
    assert labels[inc["config_id"]] == "default_huber"


def test_resume_skips_completed_and_requeues_interrupted(tmp_path):
    campaign = make_campaign(tmp_path)
    enqueue_default(campaign)
    order = [r[0] for r in campaign.db.execute("SELECT cid FROM candidates ORDER BY position")]
    first = FakeExecutor(campaign, VALUES)
    campaign.run(first, max_candidates=2)
    assert first.trained == order[:2]
    # a launcher died while running candidate 3: its attempt row stays 'running' with a dead pid
    campaign.db.execute("UPDATE candidates SET status='running' WHERE cid=?", (order[2],))
    campaign.db.execute("INSERT INTO attempts(cid, attempt, kind, launcher_pid, started, status, output_root)"
                        " VALUES(?,?,?,?,?,?,?)", (order[2], 1, "train", 2**22 + 12345, time.time(), "running", "x"))
    reopened = camp.Campaign(campaign.root)
    second = FakeExecutor(reopened, VALUES)
    reopened.run(second)
    assert second.trained == order[2:]  # completed work was not repeated
    attempts = reopened.db.execute("SELECT attempt, status FROM attempts WHERE cid=? AND kind='train'"
                                   " ORDER BY attempt", (order[2],)).fetchall()
    assert [tuple(a) for a in attempts] == [(1, "interrupted"), (2, "completed")]


def test_refuted_candidates_never_become_incumbent(tmp_path):
    campaign = make_campaign(tmp_path)
    enqueue_default(campaign)
    campaign.run(FakeExecutor(campaign, VALUES, verdict="REFUTED"))
    assert campaign.status()["incumbent"] is None
    assert campaign.status()["counts"] == {"refuted": 4}


def test_receipt_with_foreign_data_digest_is_failed(tmp_path):
    campaign = make_campaign(tmp_path)
    enqueue_default(campaign)

    class Foreign(FakeExecutor):
        def train(self, nested, output_root, declaration):
            out = FakeExecutor.train(self, nested, output_root, declaration)
            out["data_sha256"] = {"train": "0" * 64, "validation": "1" * 64}
            return out

    campaign.run(Foreign(campaign, VALUES), max_candidates=1)
    row = campaign.db.execute("SELECT status, error FROM attempts WHERE kind='train'").fetchone()
    assert row["status"] == "failed" and "declared" in row["error"]


def test_blocked_donor_candidates_are_persisted_and_released(tmp_path):
    campaign = make_campaign(tmp_path)
    flat = {**DEFAULT, "branch.regime": "R1", "train.huber_delta": 1.0}
    added = campaign.enqueue(flat, "r1")
    rows = campaign.db.execute("SELECT status, blocked_reason FROM candidates").fetchall()
    assert len(added) == 2 and all(r["status"] == "blocked" and "donor" in r["blocked_reason"] for r in rows)
    campaign.declaration["base"]["donors"] = {f"1:branch_{i}": f"/d/{i}.keras" for i in range(321)}
    assert len(campaign.unblock()) == 2


def test_campaign_refuses_changed_data(tmp_path):
    campaign = make_campaign(tmp_path)
    decl = campaign.declaration
    (tmp_path / "train.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        camp.Campaign.create(tmp_path / "other", decl)


# ------------------------------------------------------------------ T8 --
def test_optimizer_plugin_surface_returns_incumbent_flat(tmp_path):
    from optimizer_plugins.modular_doin_optimizer import Plugin

    campaign = make_campaign(tmp_path)
    holder = {}

    class Lazy(FakeExecutor):
        def __init__(self):
            pass

        def train(self, nested, output_root, declaration):
            self.campaign = holder["c"]
            self.values, self.crash_on, self.trained = VALUES, None, []
            return FakeExecutor.train(self, nested, output_root, declaration)

        def verify(self, receipt, output_root, declaration):
            return {"status": "completed", "verdict": "VERIFIED"}

    holder["c"] = campaign
    plugin = Plugin(executor=Lazy())
    plugin.set_params(modular_campaign_root=str(campaign.root))
    plugin._campaign = lambda: campaign
    flat = plugin.optimize(None, None, {})
    assert flat["train.loss"] == "huber" and "train.seed" not in flat
    assert plugin.to_flat(plugin.to_nested({**flat, "train.seed": 2021}, campaign.declaration),
                          campaign.declaration) == {**flat, "train.seed": 2021}


# ------------------------------------------------------------- T6 / T7 --
def synthetic_inputs(tmp_path, features=("a", "b")):
    def arrays(split, offset):
        rng = np.random.default_rng(offset)
        n = 40
        x = rng.normal(size=(n, 24, len(features))).astype("float32")
        ts = np.arange(n, dtype="int64") * 3600 + offset * 3600
        return dict(windows=x, targets=np.repeat(x[:, -1:, :1], 2, axis=1) * 0.5,
                    row_ids=np.array([f"{split}-{i}" for i in range(n)]), timestamps=ts,
                    target_timestamps=ts[:, None] + np.array([1, 2]) * 3600,
                    dataset_id=np.array("synthetic-m04"), split=np.array(split),
                    feature_names=np.array(features), target_names=np.array(features[:1]),
                    horizons=np.array([1, 2]), timestamp_unit=np.array("seconds"),
                    metric_space=np.array("z_train"), scaler_identity=np.array("synthetic"),
                    scaler_scale=np.array([1.0]))
    paths = []
    for split, offset in (("train", 100), ("validation", 1000)):
        path = tmp_path / f"{split}.npz"
        np.savez(path, **arrays(split, offset))
        paths.append(path)
    return paths


def small_candidate():
    space = copy.deepcopy(json.loads(
        (ROOT / "examples/config/modular_doin/ecl_l24_h24_search_space_v2_full_grid.json").read_text()))
    base = {**BASE, "feature_names": ["a", "b"], "horizons": [1, 2], "target_feature_indices": [0],
            "evaluator_fixed": {"max_updates": 1000, "max_seconds": 120.0}}
    flat = flat_default(**{"branch.grouping_size": 1, "branch.channels": 8, "core.d_model": 32,
                           "core.heads": 2, "core.blocks": 1, "core.ff_dim": 64, "core.stage_channels_0": 24,
                           "core.stage_channels_1": 12, "train.max_epochs": 30, "train.batch_size": 32,
                           "train.patience": 3, "model.branch_steps": 24, "core.time_factor_0": 2,
                           "core.time_factor_1": 2, "core.time_factor_2": 1})
    flat["train.max_epochs"] = 30
    return ss.from_flat(flat, base, space)


def test_real_engine_receipt_is_independently_rescored_and_tamper_refuted(tmp_path):
    pytest.importorskip("tensorflow")
    from tools import modular_candidate_evaluator as evaluator
    from tools import modular_checkpoint_scorer as scorer
    from tools.modular_heartbeat import Heartbeat, progress_adapter

    train, validation = synthetic_inputs(tmp_path)
    nested = small_candidate()
    nested["evaluator"]["max_epochs"] = 2
    beat_path = tmp_path / "heartbeat.json"
    with Heartbeat(beat_path, interval=0.5, identity={"test": "T6"}) as beat:
        receipt = evaluator.evaluate_candidate(nested, train, validation, tmp_path / "out", progress=progress_adapter(beat))
    records = [json.loads(line) for line in beat_path.with_suffix(".jsonl").read_text().splitlines()]
    assert records[-1]["stage"] == "done"
    assert any(r["stage"] in ("fit", "validated", "score", "save") for r in records)
    assert all("resources" in r and "eta" in r for r in records)
    assert receipt["training"]["observed_updates"] > 0
    receipt_path = tmp_path / "out" / "evaluation.json"
    result = scorer.verify(receipt_path, validation, tmp_path / "verification.json")
    assert result["verdict"] == "VERIFIED", result["problems"]
    assert result["batch_size"] == nested["evaluator"]["batch_size"]  # replays the receipt's inference batch
    assert result["exact_match"] is True  # same batch, same device class: bitwise-equal rescoring
    assert result["objective"]["rescored_value"] == pytest.approx(receipt["objective"]["value"], rel=1e-5)
    # tamper: a forged receipt metric is refuted by rescoring
    forged = json.loads(receipt_path.read_text())
    forged["metrics"]["MAE"] *= 0.5
    forged["objective"]["value"] *= 0.5
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(json.dumps(forged))
    assert scorer.verify(forged_path, validation, tmp_path / "v2.json")["verdict"] == "REFUTED"


def test_heartbeat_interval_is_bounded(tmp_path):
    from tools.modular_heartbeat import Heartbeat

    with pytest.raises(ValueError):
        Heartbeat(tmp_path / "h.json", interval=61)
    with Heartbeat(tmp_path / "h.json", interval=0.2) as beat:
        beat.update(stage="fit", epoch=1, updates=4, batches_per_epoch=10, elapsed_seconds=2.0, max_epochs=3)
        time.sleep(0.7)
    records = [json.loads(line) for line in (tmp_path / "h.jsonl").read_text().splitlines()]
    times = [r["time"] for r in records]
    assert max(b - a for a, b in zip(times, times[1:])) < 60
    fit = [r for r in records if r["stage"] == "fit"]
    assert fit and fit[0]["eta"]["per_update_seconds"] == pytest.approx(0.5)


def test_two_host_runners_never_claim_the_same_candidate_and_verify_on_training_host(tmp_path):
    campaign = make_campaign(tmp_path)
    enqueue_default(campaign)
    other = camp.Campaign(campaign.root)  # a second runner, own sqlite connection
    a = campaign.claim("worker_a")
    b = other.claim("worker_b")
    assert a[0]["cid"] != b[0]["cid"] and a[1] == b[1] == "train"
    for claimed, owner in ((a, campaign), (b, other)):
        row, kind, attempt, root = claimed
        owner._record(row, kind, attempt, {"status": "completed", "objective": 0.3, "receipt_path": "r"})
    # verification of a's checkpoint is claimed only by worker_a
    vb = other.claim("worker_b")
    assert vb[1] == "verify" and vb[0]["cid"] == b[0]["cid"]
    va = campaign.claim("worker_a")
    assert va[1] == "verify" and va[0]["cid"] == a[0]["cid"]
    hosts = dict(campaign.db.execute("SELECT cid, host FROM attempts WHERE kind='verify'").fetchall())
    assert hosts == {a[0]["cid"]: "worker_a", b[0]["cid"]: "worker_b"}


def test_operational_hold_blocks_without_dropping_and_survives_unblock(tmp_path):
    campaign = make_campaign(tmp_path)
    enqueue_default(campaign)
    held = campaign.hold(lambda f: f["train.loss"] == "mae", "BLOCKED_COST_01")
    assert len(held) == 2
    campaign.declaration["base"]["donors"] = {}
    assert campaign.unblock() == []  # donor unblocking never releases an operational hold
    campaign.run(FakeExecutor(campaign, VALUES))
    counts = campaign.status()["counts"]
    assert counts == {"verified": 2, "blocked": 2}
    assert campaign.release_hold("BLOCKED_COST_01") == 2
    assert campaign.status()["counts"] == {"verified": 2, "queued": 2}


def test_concurrent_runners_claim_disjoint_candidates(tmp_path):
    """Two runners racing on one sqlite queue (BEGIN IMMEDIATE claim): no candidate twice."""
    import threading

    campaign = make_campaign(tmp_path)
    enqueue_default(campaign)
    h, m = camp.paired_loss_arms({**DEFAULT, "core.dropout": 0.1}, 0.5)
    campaign.enqueue(h, "x_huber")
    campaign.enqueue(m, "x_mae")
    claimed, errors = {"worker_a": [], "worker_b": []}, []
    barrier = threading.Barrier(2)

    def runner(host):
        try:
            own = camp.Campaign(campaign.root)
            barrier.wait()
            while True:
                got = own.claim(host)
                if got is None:
                    return
                claimed[host].append(got[0]["cid"])
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=runner, args=(h,)) for h in claimed]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    everything = claimed["worker_a"] + claimed["worker_b"]
    assert len(everything) == len(set(everything)) == 8


def test_cpu_request_reports_no_gpu_and_arms_nothing(monkeypatch):
    from tools.modular_heartbeat import gpu_facts

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu_facts() == {"cpu_only": True, "cuda_visible_devices": "", "fallback_raise_armed": False}


def test_placement_exclusion_keeps_candidate_for_the_admitting_host(tmp_path):
    campaign = make_campaign(tmp_path)
    enqueue_default(campaign)
    first = campaign.db.execute("SELECT config_id FROM candidates ORDER BY position").fetchone()[0]
    campaign.declaration["placement"] = {"exclude": {"worker_a": [first]}}
    got = campaign.claim("worker_a")
    assert got[0]["config_id"] != first
    got_b = campaign.claim("worker_b")
    assert got_b[0]["config_id"] == first


def test_deterministic_campaign_accepts_only_exact_rescoring(tmp_path):
    campaign = make_campaign(tmp_path)
    campaign.declaration["verification"] = {"require_exact_match": True}
    enqueue_default(campaign)

    class NearMiss(FakeExecutor):
        def verify(self, receipt, output_root, declaration):
            return {"status": "completed", "verdict": "VERIFIED", "exact_match": False}

    campaign.run(NearMiss(campaign, VALUES))
    assert campaign.status()["counts"] == {"finding": 4}
    assert campaign.status()["incumbent"] is None


# ----------------------------------------------------- corrected design (lane D) --
SPACE_V2 = json.loads((ROOT / "examples/config/modular_doin/ecl_l24_h24_search_space_v2_full_grid.json").read_text())
DEFAULT_V2 = json.loads((ROOT / "examples/config/modular_doin/ecl_l24_h24_default_r0_v2_full_grid.json").read_text())


def test_full_grid_engine_requires_branch_steps_equal_window():
    nested = ss.from_flat({**DEFAULT_V2, "train.seed": 2021, "train.huber_delta": 1.0}, BASE, SPACE_V2)
    assert nested["model"]["branch_steps"] == 24
    assert nested["model"]["core"]["params"]["time_factors"] == [2, 2, 1]
    assert nested["model"]["core"]["params"]["stage_channels"] == [32, 16, 8]
    space = copy.deepcopy(SPACE_V2)
    space["bounds"]["model.branch_steps"] = {"choices": [12, 24]}
    with pytest.raises(ss.SearchSpaceError, match="full-grid"):
        ss.from_flat({**DEFAULT_V2, "model.branch_steps": 12, "core.time_factor_0": 2, "core.time_factor_1": 1,
                      "train.seed": 2021, "train.huber_delta": 1.0}, BASE, space)
    # the corrected design changes every candidate identity, even for identical flat values
    old = ss.from_flat({**DEFAULT, "model.branch_steps": 24, "core.time_factor_0": 2, "core.time_factor_1": 2,
                        "train.seed": 2021, "train.huber_delta": 1.0}, BASE, SPACE)
    assert ss.digest(old) != ss.digest(nested)


def test_superseded_old_arch_candidates_are_never_dispatched(tmp_path):
    (tmp_path / "old").mkdir()
    old = make_campaign(tmp_path / "old")
    enqueue_default(old)
    old.run(FakeExecutor(old, VALUES), max_candidates=1)  # some old rows verified, some still queued
    decl = {**old.declaration, "campaign_id": "corrected", "search_space": SPACE_V2,
            "default_candidate": DEFAULT_V2}
    new = camp.Campaign.create(tmp_path / "new", decl)
    assert new.import_superseded(old.root, "old architecture") == 4
    assert new.import_superseded(old.root, "old architecture") == 0  # idempotent
    h, m = camp.paired_loss_arms(DEFAULT_V2, 1.0)
    fresh = new.enqueue(h, "corrected_huber") + new.enqueue(m, "corrected_mae")
    new.hold(lambda f: True, "AWAIT_LANE_A")
    assert new.unblock() == []
    new.release_hold("AWAIT_LANE_A")
    executor = FakeExecutor(new, VALUES)
    new.run(executor)
    assert sorted(executor.trained) == sorted(fresh)
    rows = new.db.execute("SELECT status FROM candidates WHERE cid LIKE 'test:%'").fetchall()
    assert [r[0] for r in rows] == ["SUPERSEDED_OLD_ARCH"] * 4
    assert new.claim("worker_b") is None


def test_unpinned_campaign_refuses_to_run(tmp_path):
    campaign = make_campaign(tmp_path)
    campaign.declaration["require_pin"] = True
    campaign.declaration["executor"] = {"predictor_revision": "PENDING_LANE_A_INTEGRATED_COMMIT"}
    enqueue_default(campaign)
    with pytest.raises(RuntimeError, match="not pinned"):
        campaign.run(FakeExecutor(campaign, VALUES))
    assert campaign.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0


def test_cancelled_superseded_rows_are_never_claimed_and_attempts_stay(tmp_path):
    campaign = make_campaign(tmp_path)
    enqueue_default(campaign)
    campaign.enqueue({**DEFAULT, "branch.regime": "R1", "train.huber_delta": 1.0}, "r1")  # donor-blocked
    campaign.run(FakeExecutor(campaign, VALUES), max_candidates=1)
    before = [tuple(r) for r in campaign.db.execute("SELECT * FROM attempts ORDER BY started")]
    changed = campaign.cancel_unstarted("2026-10-01 owner order ac125db9: old design")
    assert changed == 5  # 3 queued + 2 donor-blocked; the started candidate keeps its status
    statuses = dict(campaign.db.execute("SELECT status, COUNT(*) FROM candidates GROUP BY status").fetchall())
    assert statuses == {"CANCELLED_SUPERSEDED": 5, "completed": 1}  # trained, verification pending
    assert campaign.unblock() == [] and campaign.release_hold("BLOCKED_COST_01") == 0
    assert campaign.claim("worker_a") is None and campaign.claim("worker_b") is None
    assert [tuple(r) for r in campaign.db.execute("SELECT * FROM attempts ORDER BY started")] == before
    reason = campaign.db.execute("SELECT blocked_reason FROM candidates WHERE label='r1'").fetchone()[0]
    assert reason.startswith("2026-10-01 owner order ac125db9") and "was blocked" in reason


# ------------------------------------------------- candidate budget (addendum 256c61a6) --
from tools import modular_candidate_budget as budget_model  # noqa: E402

BUDGET = {"populations": {"train_windows": 18341, "validation_windows": 2609},
          "candidates_per_week": 100,
          "calibration": [
              {"fused_width": 176, "host_peak_bytes": 4543332352, "device_peak_bytes": 392229632,
               "seconds_per_update": 0.0254, "source": "PIN_df9ae31c grouped32 GPU pilot"},
              {"fused_width": 5136, "host_peak_bytes": 4726960128 + 663703552, "device_peak_bytes": 1.2e9,
               "seconds_per_update": 0.06, "source": "per-feature profile + measured GPU increment (fixture)"}],
          "caps": {"fused_width": 2000, "host_ram_bytes": 6.0e9, "materialization_bytes": 5.0e9}}


def corrected(grouping, **extra):
    return {**DEFAULT_V2, "branch.grouping_size": grouping, **extra}


def test_budget_prices_every_dimension_with_its_source():
    nested = ss.from_flat({**corrected(32), "train.seed": 2021, "train.huber_delta": 1.0}, BASE, SPACE_V2)
    priced = budget_model.price(nested, BUDGET, parameters=487672)
    assert set(priced) == set(budget_model.DIMENSIONS)
    assert priced["fused_width"]["value"] == 11 * 16 and priced["branches"]["value"] == 11
    assert priced["fused_time"]["value"] == 24  # never collapsed
    assert priced["materialization_bytes"]["value"] == (18341 + 2609) * 24 * 176 * 4
    assert priced["host_ram_bytes"]["value"] == 4543332352
    assert "pilot" in priced["host_ram_bytes"]["source"]
    assert priced["weekly_gpu_seconds"]["value"] == pytest.approx(priced["gpu_seconds_per_candidate"]["value"] * 100)


def test_budget_overflow_becomes_named_deferred_row_never_truncated(tmp_path):
    campaign = make_campaign(tmp_path)
    campaign.declaration.update(search_space=SPACE_V2, budget=BUDGET)
    campaign.space = SPACE_V2
    added = campaign.enqueue({**corrected(1), "train.huber_delta": 1.0}, "per_feature")
    rows = campaign.db.execute("SELECT status, blocked_reason, nested FROM candidates").fetchall()
    assert len(added) == 2 and all(r["status"] == "deferred" for r in rows)
    reason = rows[0]["blocked_reason"]
    assert reason.startswith("DEFERRED_BUDGET:") and "fused_width=5136>2000" in reason
    assert "materialization_bytes" in reason
    nested = json.loads(rows[0]["nested"])
    assert len(nested["model"]["branches"]) == 321 and nested["model"]["branch_steps"] == 24  # verbatim
    assert campaign.claim("worker_a") is None  # deferred rows are never dispatched


def test_deferred_candidate_returns_when_cap_or_measurement_changes(tmp_path):
    campaign = make_campaign(tmp_path)
    campaign.declaration.update(search_space=SPACE_V2, budget=copy.deepcopy(BUDGET))
    campaign.space = SPACE_V2
    campaign.enqueue({**corrected(1), "train.huber_delta": 1.0}, "per_feature")
    assert campaign.reconsider_deferred() == []  # nothing changed
    campaign.declaration["budget"]["caps"].update(fused_width=6000, materialization_bytes=2.0e10)
    released = campaign.reconsider_deferred()
    assert len(released) == 2
    assert {r[0] for r in campaign.db.execute("SELECT status FROM candidates")} == {"queued"}


def test_unmeasured_dimension_with_a_cap_defers_instead_of_fitting(tmp_path):
    nested = ss.from_flat({**corrected(32), "train.seed": 2021, "train.huber_delta": 1.0}, BASE, SPACE_V2)
    budget = {**BUDGET, "calibration": BUDGET["calibration"][:1], "caps": {"vram_bytes": 1e9}}
    _, overflow = budget_model.check(nested, budget)
    assert overflow == []  # width 176 is measured exactly
    nested8 = ss.from_flat({**corrected(8), "train.seed": 2021, "train.huber_delta": 1.0}, BASE, SPACE_V2)
    _, overflow8 = budget_model.check(nested8, budget)
    assert overflow8 == ["vram_bytes=UNMEASURED"]


def test_doin_proposals_respect_every_cap():
    rng = random.Random(3)
    space = copy.deepcopy(SPACE_V2)
    for _ in range(20):
        flat = camp.propose_within_budget(space, rng, BASE, BUDGET,
                                          fixed={"branch.regime": "R0", "core.regime": "R0"})
        trial = {**flat, "train.seed": 2021}
        if trial["train.loss"] == "huber":
            trial.setdefault("train.huber_delta", 1.0)
        nested = ss.from_flat(trial, BASE, space)
        assert budget_model.check(nested, BUDGET)[1] == []
        assert nested["model"]["branch_steps"] == 24


def test_runner_preflight_refuses_absent_or_dirty_pin_worktree(tmp_path, monkeypatch):
    import subprocess as sp

    decl = {"executor": {"predictor_python": "/py", "predictor_checkout": "/pin"}, "hosts": {}}
    monkeypatch.setenv("M04_SSH_worker_b", "alias-not-written")
    executor = camp.RemoteExecutor(decl, "worker_b", tmp_path)
    replies = iter([sp.CompletedProcess([], 3, "MISSING\n", ""),
                    sp.CompletedProcess([], 0, "b" * 40 + " 0\n", ""),
                    sp.CompletedProcess([], 0, "a" * 40 + " 2\n", ""),
                    sp.CompletedProcess([], 0, "a" * 40 + " 0\n", "")])
    monkeypatch.setattr(camp.subprocess, "run", lambda *a, **k: next(replies))
    for match in ("absent", "pin is", "modified"):
        with pytest.raises(RuntimeError, match=match):
            executor.preflight("a" * 40)
    assert executor.preflight("a" * 40)["clean"] is True


# ------------------------------------------- frozen forecast evidence (b327b771 s5) --
def _evidence_inputs(tmp_path):
    from tools import modular_forecast_evidence as fe  # noqa: F401
    train, validation = synthetic_inputs(tmp_path)
    receipt = {"data": {"test_used": False, "dataset_id": "synthetic-m04", "target_names": ["a"],
                        "metric_space": "z_train", "scaler_identity": "synthetic"},
               "digests": {"validation_sha256": camp.sha_file(validation), "model_sha256": "m" * 64,
                           "weights_sha256": "w" * 64},
               "candidate": {"cid": "c" * 64}, "bridge": {"predictor_revision": "r" * 40},
               "per_horizon": {"1": {"MAE": 0.5, "MSE": 0.4, "baseline_MAE": 0.4, "baseline_MSE": 0.3},
                               "2": {"MAE": 0.3, "MSE": 0.2, "baseline_MAE": 0.0, "baseline_MSE": 0.0},
                               "3": {"MAE": float("nan"), "MSE": 0.2, "baseline_MAE": 0.5, "baseline_MSE": 0.4}}}
    return receipt, validation


def test_forecast_evidence_is_frozen_hashed_and_same_row(tmp_path):
    from tools import modular_forecast_evidence as fe
    receipt, validation = _evidence_inputs(tmp_path)
    record = fe.build(receipt, validation, campaign_id="t", asset="synthetic")
    assert record["schema"] == "predictor.forecast_naive_evidence.v1" and fe.verify(record)
    h1, h2, h3 = record["per_horizon"]
    assert h1["MAE"]["status"] == "OK" and h1["MAE"]["delta"] == pytest.approx(0.1) and h1["rows"] == 40
    assert h2["MAE"] == {"skill": None, "delta": 0.3, "status": "NOT_AVAILABLE", "reason": "ZERO_NAIVE"}
    assert h3["model_MAE"] is None and h3["MAE"]["status"] == "NOT_AVAILABLE"
    assert record["split"] == {"provenance": "held_out_validation", "test_used": False, "reserved_trading_test": False}
    assert record["population"]["rows"] == 40 and len(record["population"]["row_ids_sha256"]) == 64
    tampered = json.loads(json.dumps(record))
    tampered["per_horizon"][0]["model_MAE"] = 0.1
    assert not fe.verify(tampered)


def test_forecast_evidence_refuses_test_provenance_and_foreign_rows(tmp_path):
    from tools import modular_forecast_evidence as fe
    receipt, validation = _evidence_inputs(tmp_path)
    with pytest.raises(ValueError, match="reserved test"):
        fe.build(receipt, validation, provenance="trading_test")
    receipt["digests"]["validation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differ"):
        fe.build(receipt, validation)


def test_forecast_evidence_optional_seasonal_naive(tmp_path):
    from tools import modular_forecast_evidence as fe
    receipt, validation = _evidence_inputs(tmp_path)
    record = fe.build(receipt, validation, seasonal_period_steps=2)
    assert record["seasonal_naive"]["period_steps"] == 2 and fe.verify(record)
    h1, h2, h3 = record["per_horizon"]
    assert h1["seasonal_naive"]["MAE"]["status"] in ("OK", "NOT_AVAILABLE") and "naive_MAE" in h1["seasonal_naive"]
    assert h3["seasonal_naive"]["status"] == "NOT_AVAILABLE"  # h > period
    assert "seasonal_naive" not in fe.build(receipt, validation)  # absent unless declared


def test_recover_adopts_a_finished_remote_outcome_instead_of_retraining(tmp_path):
    campaign = make_campaign(tmp_path)
    enqueue_default(campaign)
    row, kind, attempt, root = campaign.claim("worker_a")
    campaign.db.execute("UPDATE attempts SET launcher_pid=? WHERE cid=?", (2**22 + 777, row["cid"]))  # dead orchestrator
    outcome = {"status": "completed", "objective": 0.31, "receipt_path": "r", "observed_updates": 9,
               "data_sha256": {k: campaign.declaration["data"][k]["sha256"] for k in ("train", "validation")}}

    class Adopting(FakeExecutor):
        host = "worker_a"

        def fetch_outcome(self, output_root, host):
            return outcome if output_root == str(root) else None

    executor = Adopting(campaign, VALUES)
    campaign.run(executor)
    assert row["cid"] not in executor.trained  # never retrained
    att = campaign.db.execute("SELECT status, objective, error FROM attempts WHERE cid=? AND kind='train'",
                              (row["cid"],)).fetchall()
    assert len(att) == 1 and att[0]["status"] == "completed" and att[0]["objective"] == 0.31
    assert "adopted" in att[0]["error"]


def test_per_horizon_table_generator_reads_receipts_and_seasonal_naive(tmp_path):
    from tools import modular_doin_per_horizon_table as table
    campaign = make_campaign(tmp_path)
    _, validation = synthetic_inputs(tmp_path / "v") if (tmp_path / "v").mkdir() is None else (None, None)
    receipts = []

    class Receipting(FakeExecutor):
        def train(self, nested, output_root, declaration):
            out = FakeExecutor.train(self, nested, output_root, declaration)
            path = output_root / "r.json"
            path.write_text(json.dumps({"candidate": {"cid": ss.digest(nested)}, "digests": {"model_sha256": "m"},
                                        "metrics": {"MAE": out["objective"], "baseline_MAE": 0.5,
                                                    "skill_MAE": 1 - out["objective"] / 0.5},
                                        "per_horizon": {"1": {"MAE": 0.6, "baseline_MAE": 0.5, "skill_MAE": -0.2},
                                                        "2": {"MAE": 0.2, "baseline_MAE": 0.5, "skill_MAE": 0.6}}}))
            receipts.append(path)
            return {**out, "receipt_path": str(path)}

    enqueue_default(campaign)
    campaign.run(Receipting(campaign, VALUES))
    report = table.build(campaign.root / "queue.sqlite", validation, 1)
    assert report["candidates"] == 4 and len(report["pairs"]) == 2
    assert report["negative_skill_vs_persistence_by_horizon"] == {1: 4}
    row = report["rows"][0]
    assert row["h1_seasonal_MAE"] is not None and row["h2_seasonal_MAE"] is None  # h2 > period 1
    huber = next(p for p in report["pairs"] if p["seeds"] == {2021: 0.30, 2022: 0.32})
    assert huber["mean"] == pytest.approx(0.31) and huber["spread"] == pytest.approx(0.02)


def test_worker_verifies_donor_binding_and_contract_before_build(tmp_path, monkeypatch):
    import hashlib as hl
    from predictor_plugins.modular_temporal import provenance
    from tools import modular_heartbeat as hb

    donor = tmp_path / "branch_0.keras"
    for name, text in (("branch_0.keras", "k"), ("branch_0.manifest.json", "m"), ("branch_0.manifest.v2.json", "v")):
        (tmp_path / name).write_text(text)
    sha = lambda t: hl.sha256(t.encode()).hexdigest()
    binding = {"index_sha256": "i" * 64, "amendment_sha256": "a" * 64, "required_contract": "OPERATIONAL",
               "donors": {str(donor): {"keras": sha("k"), "manifest": sha("m"), "manifest_v2": sha("v")}}}
    config = {"modular_candidate": {"donor_binding": binding}}
    monkeypatch.setattr(provenance, "donor_provenance", lambda p: {"conditioning_contract": "OPERATIONAL"})
    assert hb.verify_donor_binding(config)["result"] == "VERIFIED_BEFORE_BUILD"
    monkeypatch.setattr(provenance, "donor_provenance", lambda p: {"conditioning_contract": "UNKNOWN"})
    with pytest.raises(ValueError, match="DONOR_CONTRACT_MISMATCH"):
        hb.verify_donor_binding(config)
    monkeypatch.setattr(provenance, "donor_provenance", lambda p: {"conditioning_contract": "OPERATIONAL"})
    (tmp_path / "branch_0.manifest.v2.json").write_text("tampered")
    with pytest.raises(ValueError, match="DONOR_BINDING_MISMATCH"):
        hb.verify_donor_binding(config)
    assert hb.verify_donor_binding({"modular_candidate": {}}) is None  # R0: nothing to check


def test_from_flat_embeds_only_the_used_donor_binding():
    base = copy.deepcopy(BASE)
    base["donors"] = {f"1:branch_{i}": f"/d/branch_{i}.keras" for i in range(321)}
    base["donors"]["core:1"] = "/d/core.keras"
    base["donor_binding"] = {"index_sha256": "i", "amendment_sha256": "a", "required_contract": "OPERATIONAL",
                             "donors": {p: {"keras": "k", "manifest": "m", "manifest_v2": "v"}
                                        for p in base["donors"].values()}}
    r1 = ss.from_flat({**DEFAULT_V2, "branch.regime": "R1", "train.seed": 2021, "train.huber_delta": 1.0}, base, SPACE_V2)
    assert len(r1["modular_candidate"]["donor_binding"]["donors"]) == 321  # branches only, no core
    r0 = ss.from_flat({**DEFAULT_V2, "train.seed": 2021, "train.huber_delta": 1.0}, base, SPACE_V2)
    assert "donor_binding" not in r0["modular_candidate"]


def test_r1r2_table_pairs_by_seed_lists_refusals_and_closure(tmp_path):
    from tools import modular_doin_r1r2_table as r12
    campaign = make_campaign(tmp_path)
    (tmp_path / "v").mkdir()
    _, validation = synthetic_inputs(tmp_path / "v")
    values = {("huber", 2021): 0.40, ("huber", 2022): 0.38, ("mae", 2021): 0.36, ("mae", 2022): 0.37}

    class Receipting(FakeExecutor):
        def train(self, nested, output_root, declaration):
            out = FakeExecutor.train(self, nested, output_root, declaration)
            path = output_root / "r.json"
            per = {"1": {"MAE": 0.6, "baseline_MAE": 0.5, "skill_MAE": -0.2},
                   "2": {"MAE": 0.2, "baseline_MAE": 0.5, "skill_MAE": 0.6}}
            path.write_text(json.dumps({"candidate": {"cid": ss.digest(nested)}, "digests": {"model_sha256": "m"},
                                        "metrics": {"MAE": out["objective"], "baseline_MAE": 0.5,
                                                    "skill_MAE": 1 - out["objective"] / 0.5}, "per_horizon": per}))
            return {**out, "receipt_path": str(path)}

    h, m = camp.paired_loss_arms(DEFAULT, 1.0)
    campaign.enqueue(h, "default_R0_huber")
    campaign.enqueue(m, "default_branch_core_R1_mae")
    campaign.run(Receipting(campaign, values))
    campaign.db.execute("UPDATE candidates SET status='REFUSED_BY_ENGINE', blocked_reason='Donor manifest mismatch' "
                        "WHERE seed=2022 AND label='default_branch_core_R1_mae' AND 0")
    ref_id = campaign.db.execute("SELECT config_id FROM candidates WHERE label='default_R0_huber'").fetchone()[0]
    report = r12.build(campaign.root / "queue.sqlite", validation, 1, {"config_id": ref_id}, {"note": "old"},
                       {"cpu_seconds": 1.0}, {"value": None, "status": "NOT_AVAILABLE", "reason": "no published L24 row"})
    (c,) = report["contrasts"]
    assert c["paired_difference_vs_R0"] == {2021: pytest.approx(-0.04), 2022: pytest.approx(-0.01)}
    assert c["mean_difference"] == pytest.approx(-0.025) and c["R0_spread"] == pytest.approx(0.02)
    assert c["label_rule"].startswith("STRICT_MINIMUM; gap exceeds")  # 0.025 > 0.02 and > 0.01
    assert c["pretraining_cost"] == {"cpu_seconds": 1.0}
    assert all(r["comparability"].startswith("NOT_COMPARABLE") for r in report["closure"])
    assert report["closure"][0]["model_MAE"] == pytest.approx(0.365)


def test_enqueue_with_explicit_seeds_and_per_config_eligibility(tmp_path):
    campaign = make_campaign(tmp_path, seeds=(2022, 2023))
    h, m = camp.paired_loss_arms(DEFAULT, 1.0)
    campaign.enqueue(h, "seeds_default")          # declared 2022/2023
    campaign.enqueue(m, "sweep", seeds=[2021, 2022])  # contrast on the reference's seeds
    seeds = sorted(r[0] for r in campaign.db.execute("SELECT seed FROM candidates WHERE label='sweep'"))
    assert seeds == [2021, 2022]
    values = {("huber", 2022): 0.4, ("huber", 2023): 0.41, ("mae", 2021): 0.30, ("mae", 2022): 0.31}
    campaign.run(FakeExecutor(campaign, values))
    table, ranked = campaign.standings()
    assert {t["label"]: t["eligible"] for t in table} == {"seeds_default": True, "sweep": True}
    assert ranked[0]["label"] == "sweep" and ranked[0]["per_seed"] == {"2021": 0.30, "2022": 0.31}


def test_optional_seasonal_residual_parameter_round_trips_and_is_optional():
    space = copy.deepcopy(SPACE_V2)
    space["bounds"]["model.target_residual"] = {"choices": ["none", "seasonal_naive_24"]}
    flat = {**DEFAULT_V2, "train.seed": 2021, "train.huber_delta": 1.0, "model.target_residual": "seasonal_naive_24"}
    nested = ss.from_flat(flat, BASE, space)
    assert nested["model"]["target_residual"]["period"] == 24
    assert len(nested["model"]["target_residual"]["target_features"]) == 321
    assert ss.to_flat(nested, space) == flat
    plain = {**flat, "model.target_residual": "none"}
    assert "target_residual" not in ss.from_flat(plain, BASE, space)["model"]
    with pytest.raises(ss.SearchSpaceError, match="missing"):
        ss.from_flat({k: v for k, v in flat.items() if k != "model.target_residual"}, BASE, space)
    with pytest.raises(ss.SearchSpaceError, match="not declared"):
        ss.from_flat(flat, BASE, SPACE_V2)  # older spaces never accept it silently
    old = {**DEFAULT_V2, "train.seed": 2021, "train.huber_delta": 1.0}
    assert ss.to_flat(ss.from_flat(old, BASE, SPACE_V2), SPACE_V2) == old  # older spaces unchanged
