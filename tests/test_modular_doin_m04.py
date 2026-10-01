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


def test_default_matches_engine_defaults_when_engine_available():
    mt = pytest.importorskip("predictor_plugins.modular_temporal")
    nested = ss.from_flat(flat_default(**{"branch.grouping_size": 321}), BASE, SPACE)
    normalized = mt._normalize(nested["model"])
    assert normalized["core"]["params"]["d_model"] == 64
    assert normalized["branch_steps"] == 12 and normalized["output_steps"] == 6


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
    space = copy.deepcopy(SPACE)
    base = {**BASE, "feature_names": ["a", "b"], "horizons": [1, 2], "target_feature_indices": [0],
            "evaluator_fixed": {"max_updates": 1000, "max_seconds": 120.0}}
    flat = flat_default(**{"branch.grouping_size": 1, "branch.channels": 8, "core.d_model": 32,
                           "core.heads": 2, "core.blocks": 1, "core.ff_dim": 64, "core.stage_channels_0": 24,
                           "core.stage_channels_1": 12, "train.max_epochs": 30, "train.batch_size": 32,
                           "train.patience": 3})
    flat["train.max_epochs"] = 30
    return ss.from_flat(flat, base, space)


def test_real_engine_receipt_is_independently_rescored_and_tamper_refuted(tmp_path):
    pytest.importorskip("tensorflow")
    from tools import modular_candidate_evaluator as evaluator
    from tools import modular_checkpoint_scorer as scorer
    from tools.modular_heartbeat import Heartbeat

    train, validation = synthetic_inputs(tmp_path)
    nested = small_candidate()
    nested["evaluator"]["max_epochs"] = 2
    beat_path = tmp_path / "heartbeat.json"
    with Heartbeat(beat_path, interval=0.5, identity={"test": "T6"}) as beat:
        receipt = evaluator.evaluate_candidate(nested, train, validation, tmp_path / "out", progress=beat.update)
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
