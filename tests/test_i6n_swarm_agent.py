"""Lane B worker agent: boot recovery, freeze/annual fixed units, DOIN node-config sync.

Fake freeze/evaluator subprocesses on CPU; systemctl is a recorder; nothing is installed.
"""
import json
import sys
from pathlib import Path

import pytest

from tests import i6n_swarm_fixtures as F
from tests.test_i6n_swarm_campaign import Recorder, complete_candidate, other_genome
from tools import i6n_search as search
from tools.i6n_swarm import agent as A, common, coordinator as K, ledger as L, plan as P

@pytest.fixture(autouse=True)
def _reference_aggregator(monkeypatch):
    """These tests exercise B; E's score_folds is exercised by E's suite and the integration check."""
    from tools.i6n_swarm import bridge as _b
    monkeypatch.setattr(_b, "resolve_score_folds", lambda: _b._reference_score_folds)


FAKE_FREEZE = r'''
import json, sys, os, hashlib
args = sys.argv[1:]
opt = {args[i]: args[i + 1] for i in range(0, len(args) - 1) if args[i].startswith("--") and not args[i + 1].startswith("--")}
out = opt["--output"]
os.makedirs(out)
cell = json.load(open(opt["--config"]))
scale = 1 + len(opt["--fold"]) % 3 + (0.5 if "--score-outer" in args else 0)
digest = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
binding = dict(i6n_horizon=cell["h"], i6n_support=dict(counts=dict(train=50, inner=5),
               naive_sums=dict(train=1.0 * scale, inner=0.1 * scale),
               support_digest=dict(train=digest(["t", opt["--fold"]]), inner=digest(["i", opt["--fold"]]))),
               i6n_evaluation_context=dict(schema="fixture", fold=opt["--fold"], experiment=opt["--experiment"],
                                           score_outer="--score-outer" in args),
               i6n_evaluator_command=["x"], i6n_evaluator_timeout=60)
json.dump(binding, open(os.path.join(out, "BINDING.json"), "w"))
print(json.dumps(binding))
'''

FAKE_EVAL = r'''
import json, sys, hashlib
req = json.load(sys.stdin)
digest = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
sup = req["support"]
print(json.dumps(dict(status="ok", counts=sup["counts"], naive_sums=sup["naive_sums"], support_digest=sup["support_digest"],
    error_sums={k: v * 0.95 for k, v in sup["naive_sums"].items()},
    metadata=dict(candidate_receipt=dict(schema_version="i6n.candidate_receipt.v1", experiment=req["context"].get("experiment", "exp-fx"),
        fold=req["context"]["fold"], effective_config_sha256=digest(req["config"]), horizon_hours=req["config"]["horizon"],
        genotype_canonical_sha256=hashlib.sha256(json.dumps({"schema": "i6n-search-v1", "config": req["config"]},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest(),
        seed=6102026, rows=[], attempt=req["attempt"])))))
'''


@pytest.fixture
def campaign(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "freeze.py").write_text(FAKE_FREEZE)
    (scripts / "evaluate.py").write_text(FAKE_EVAL)
    cfg = F.private_config(tmp_path, frozen=False, fake_evaluator=[sys.executable, str(scripts / "evaluate.py")])
    root = tmp_path / "campaign"
    P.build(cfg, root)
    K.start(root, roles=[], systemctl=Recorder())
    node = json.loads((root / "private" / "hosts" / "preferred" / "node.json").read_text())
    node.update(freeze_argv=[sys.executable, str(scripts / "freeze.py")],
                ledger=dict(path=str(root / "state" / "ledger.sqlite3")), shared_data_root=str(tmp_path / "shared"))
    return root, node, tmp_path


def test_freeze_units_run_outside_gpu_and_score_outer_only_for_weeks(campaign):
    root, node, tmp = campaign
    led = L.Ledger(root / "state" / "ledger.sqlite3")
    done = A.run_fixed_units(node, led, stages=("freeze",), boot_id="boot-1")
    freeze_units = [u for u in led.units() if u["stage"] == "freeze"]
    assert done["completed"] == len(freeze_units) and all(u["state"] == "COMPLETE" for u in led.units(stage="freeze"))
    for u in freeze_units:
        binding = led.terminal(u["fit_id"])["receipt"]["shared_receipt"]
        assert binding["i6n_evaluation_context"]["score_outer"] is u["dag_node"].startswith("freeze_weeks")
    K.tick(root, now=4000.0)
    states = {n["node_id"]: n["state"] for n in led.dag_nodes()}
    assert states["search:h12"] == "QUEUED" and states["freeze_weeks:h12"] == "COMPLETE"


def test_node_sync_renders_ready_domains_and_starts_node_once(campaign):
    root, node, tmp = campaign
    led = L.Ledger(root / "state" / "ledger.sqlite3")
    A.run_fixed_units(node, led, stages=("freeze",), boot_id="boot-1")
    K.tick(root, now=4000.0)
    node["doin_node_config"] = str(tmp / "worker" / "doin_node.json")
    rec = Recorder()
    first = A.sync_node(node, led, systemctl=rec)
    cfg = json.loads(Path(node["doin_node_config"]).read_text())
    assert [d["optimization_config"]["i6n_horizon"] for d in cfg["domains"]] == [12, 72]
    assert all(d["higher_is_better"] is False for d in cfg["domains"])
    assert first["action"] == "started"
    second = A.sync_node(node, led, systemctl=rec)
    assert second["action"] == "unchanged"
    assert [c for c in rec.calls if c[0] in ("start", "restart")] == [["start", "i6n-swarm-fx1-preferred-node.service"]]


def test_node_restart_deferred_while_this_host_is_fitting(campaign):
    root, node, tmp = campaign
    led = L.Ledger(root / "state" / "ledger.sqlite3")
    node["doin_node_config"] = str(tmp / "worker" / "doin_node.json")
    rec = Recorder(active={"i6n-swarm-fx1-preferred-node.service"})
    Path(node["doin_node_config"]).parent.mkdir(parents=True)
    Path(node["doin_node_config"]).write_text("{}")
    A.run_fixed_units(node, led, stages=("freeze",), boot_id="boot-1")
    K.tick(root, now=4000.0)
    some = led.units(stage="freeze")[0]
    led.ensure_unit(common.sha256_json("busy"), dict(some["spec"], fold="busy", dag_node="search:h12", stage="search"))
    led.claim(common.sha256_json("busy"), node["owner"], node["host_key"], "boot-1", "rtx5090", 600)
    out = A.sync_node(node, led, systemctl=rec)
    assert out["action"] == "deferred_restart_host_busy"
    assert not [c for c in rec.calls if c[0] in ("start", "restart")]


def test_boot_recovery_interrupts_previous_boot_of_this_host_only(campaign):
    root, node, tmp = campaign
    led = L.Ledger(root / "state" / "ledger.sqlite3")
    units = led.units(stage="freeze")
    led.claim(units[0]["fit_id"], "x", node["host_key"], "boot-old", "cpu", 3600, exclusive_host=False)
    led.claim(units[1]["fit_id"], "y", "secondary", "boot-s", "cpu", 3600, exclusive_host=False)
    changed = A.recover(node, led, boot_id="boot-new")
    assert [c["fit_id"] for c in changed] == [units[0]["fit_id"]]


def test_annual_units_run_with_fencing_and_complete(campaign):
    root, node, tmp = campaign
    led = L.Ledger(root / "state" / "ledger.sqlite3")
    A.run_fixed_units(node, led, stages=("freeze",), boot_id="boot-1")
    K.tick(root, now=4000.0)
    for h in (12, 72):
        complete_candidate(root, h, search.default_genome(h), 0.9)
        complete_candidate(root, h, other_genome(h), 0.8)
        led.dag_event(f"search:h{h}", "SEARCH_TERMINAL", {"stop_reason": "generation_budget"})
    K.tick(root, now=5000.0)
    annual = led.units(stage="annual")
    assert len(annual) == 2 * 2 * 2
    out = A.run_fixed_units(node, led, stages=("annual",), boot_id="boot-1")
    assert out["completed"] == len(annual)
    attempts = [a for u in annual for a in led.attempts(u["fit_id"])]
    assert all(a["index"] == 1 and a["outcome"] == "COMPLETE" for a in attempts)
    K.tick(root, now=6000.0)
    states = {n["node_id"]: n["state"] for n in led.dag_nodes()}
    assert states["annual:h12"] == "COMPLETE" and states["annual:h72"] == "COMPLETE"
