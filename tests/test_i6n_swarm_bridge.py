"""Lane B evaluation bridge: fold fan-out, ledger guard, adoption, binding, fold-mean fitness.

Fake evaluator subprocess on CPU (declared fixture device); no TensorFlow, no data.
"""
import json
import os
import sys
import threading

import pytest

from tools import i6n_search as search
from tools.i6n_swarm import bridge as B, common, ledger as L

FAKE = r'''
import json, os, sys, hashlib
req = json.load(sys.stdin)
ctx = req["context"]
count_dir = os.environ["FAKE_COUNT_DIR"]
open(os.path.join(count_dir, "%s-%d" % (ctx["fold"], len(os.listdir(count_dir)))), "w").write(json.dumps(req.get("attempt")))
if ctx.get("fail"):
    sys.stderr.write("evaluator refused"); sys.exit(1)
digest = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
sup = req["support"]
factor = float(ctx.get("factor", 0.9))
if ctx.get("wrong_config"):
    req["config"]["history_hours"] += 1
receipt = dict(status="ok", counts=sup["counts"], naive_sums=sup["naive_sums"], support_digest=sup["support_digest"],
               error_sums={k: v * factor for k, v in sup["naive_sums"].items()},
               metadata=dict(candidate_receipt=dict(schema_version="i6n.candidate_receipt.v1", experiment=ctx["experiment"],
                   fold=ctx["fold"], effective_config_sha256=digest(req["config"]), horizon_hours=req["config"]["horizon"],
                   genotype_canonical_sha256=ctx["genotype_expected"], seed=6102026, rows=[], attempt=req.get("attempt"))))
print(json.dumps(receipt))
'''


def support(scale):
    return dict(counts=dict(train=100, inner=10), naive_sums=dict(train=1.0 * scale, inner=0.1 * scale),
                support_digest=dict(train="1" * 64, inner="2" * 64))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    fake = tmp_path / "fake_eval.py"
    fake.write_text(FAKE)
    counts = tmp_path / "counts"
    counts.mkdir()
    monkeypatch.setenv("FAKE_COUNT_DIR", str(counts))
    genome = search.default_genome(12)
    wire = dict(schema="i6n-shared-deap-v1", genome=search.normalize_genome(genome),
                identity=search.effective_identity(genome), context="0" * 64, generation=0)
    folds = []
    for i, factor in enumerate((0.9, 0.7), 1):
        folds.append(dict(fold=f"F{i}", support=support(i),
                          context=dict(experiment="exp1", fold=f"F{i}", factor=factor,
                                       genotype_expected=wire["identity"])))
    swarm = dict(schema="i6n.swarm.context.v1", experiment="exp1", plan_sha256="p" * 64, horizon=12,
                 dag_node="search:h12", budget_new_candidates=24, folds=folds)
    node = dict(schema="i6n.swarm.node.v1", owner="n1", host_key="preferred",
                device=dict(cls="cpu_fixture", cuda_visible_devices="", fixture=True),
                exclusive_host=True, evaluator_argv=[sys.executable, str(fake)], workdir=str(tmp_path),
                launcher=None, lease=dict(ttl_seconds=5, renew_seconds=0.2, max_wait_seconds=5, poll_seconds=0.1),
                logs_dir=str(tmp_path / "logs"), fixture=True)
    led = L.Ledger(tmp_path / "ledger.sqlite3")
    bridge = B.Bridge(node, led, boot_id="boot-1")
    return bridge, led, wire, swarm, counts


def calls(counts):
    return sorted(os.listdir(counts))


def test_fold_mean_fitness_uses_search_score_per_fold(setup):
    bridge, led, wire, swarm, counts = setup
    result = bridge.evaluate_wire(wire, 0, swarm)
    expected = []
    for fold in swarm["folds"]:
        expected.append(search._score(result["metrics"]["folds"][fold["fold"]], fold["support"])[0])
    assert result["fitness"] == pytest.approx(sum(expected) / 2)
    assert result["fitness"] == pytest.approx((0.9 + 0.7) / 2)
    assert result["candidate_identity"] == wire["identity"]
    assert result["hyper_dict"] == search.effective_config(wire["genome"])
    assert result["seed"] == 6102026 and result["direction"] == "minimize"
    assert len(calls(counts)) == 2
    first_attempt = json.loads((counts / calls(counts)[0]).read_text())
    assert first_attempt["index"] == 1 and first_attempt["fencing_token"]


def test_reevaluation_adopts_every_fold_without_fitting(setup):
    bridge, led, wire, swarm, counts = setup
    first = bridge.evaluate_wire(wire, 0, swarm)
    second = B.Bridge(dict(bridge.node, owner="n2", host_key="secondary"), led, boot_id="boot-9").evaluate_wire(wire, 0, swarm)
    assert second["fitness"] == first["fitness"]
    assert len(calls(counts)) == 2


def test_failed_fold_rejects_candidate_and_keeps_completed_fold(setup):
    bridge, led, wire, swarm, counts = setup
    swarm["folds"][1]["context"]["fail"] = True
    with pytest.raises(search.ScoreRejected):
        bridge.evaluate_wire(wire, 0, swarm)
    with pytest.raises(search.ScoreRejected):
        bridge.evaluate_wire(wire, 0, swarm)
    assert len(calls(counts)) == 2  # fold F1 adopted, F2 terminal FAILED: nothing repeated
    states = sorted(u["state"] for u in led.units())
    assert states == ["COMPLETE", "FAILED"]


def test_receipt_for_a_different_config_is_never_published(setup):
    bridge, led, wire, swarm, counts = setup
    swarm["folds"][0]["context"]["wrong_config"] = True
    with pytest.raises(search.ScoreRejected):
        bridge.evaluate_wire(wire, 0, swarm)
    assert all(led.terminal(u["fit_id"]) is None or u["fold"] != "F1" for u in led.units())
    assert not [i for i in led.outbox_pending() if i["payload"]["fold"] == "F1" and i["payload"]["disposition"] != "FAILED"]


def test_wire_identity_must_match_genome(setup):
    bridge, led, wire, swarm, counts = setup
    bad = dict(wire, identity="f" * 64)
    with pytest.raises(search.ScoreRejected):
        bridge.evaluate_wire(bad, 0, swarm)
    assert calls(counts) == []


def test_held_lease_waits_then_adopts_peer_terminal(setup):
    bridge, led, wire, swarm, counts = setup
    fold = swarm["folds"][0]
    fit = B.fold_fit_id(swarm, fold, search.effective_config(wire["genome"]))
    led.ensure_unit(fit, B.unit_spec(swarm, fold, wire, search.effective_config(wire["genome"]), bridge.node))
    peer = led.claim(fit, "peer", "secondary", "b", "rtx4090", 30)
    config = search.effective_config(wire["genome"])
    receipt = dict(status="ok", **support(1), error_sums=dict(train=0.5, inner=0.05),
                   metadata=dict(candidate_receipt=dict(schema_version="i6n.candidate_receipt.v1", experiment="exp1",
                       fold="F1", effective_config_sha256=common.sha256_json(config), horizon_hours=12,
                       genotype_canonical_sha256=wire["identity"], seed=6102026, rows=[])))
    doc = B.receipt_document(fit, swarm, fold, wire, config, receipt)
    threading.Timer(0.5, lambda: led.complete(fit, peer["token"], doc)).start()
    result = bridge.evaluate_wire(wire, 0, swarm)
    assert result["metrics"]["folds"]["F1"]["error_sums"]["train"] == 0.5
    assert [c.split("-")[0] for c in calls(counts)] == ["F2"]


def test_budget_of_new_candidates_is_enforced_by_identity(setup):
    bridge, led, wire, swarm, counts = setup
    swarm["budget_new_candidates"] = 1
    bridge.evaluate_wire(wire, 0, swarm)
    other = search.default_genome(12)
    other["input_transform"] = "difference"
    wire2 = dict(wire, genome=search.normalize_genome(other), identity=search.effective_identity(other))
    for fold in swarm["folds"]:
        fold["context"]["genotype_expected"] = wire2["identity"]
    with pytest.raises(search.ScoreRejected, match="BUDGET"):
        bridge.evaluate_wire(wire2, 0, swarm)


def test_gpu_node_without_device_refuses_before_any_launch(setup):
    bridge, led, wire, swarm, counts = setup
    node = dict(bridge.node, fixture=False, device=dict(cls="rtx5090", cuda_visible_devices="", required_gpu_name="x"))
    with pytest.raises(search.ScoreRejected, match="fallback"):
        B.Bridge(node, led, boot_id="b").evaluate_wire(wire, 0, swarm)
    assert calls(counts) == []
