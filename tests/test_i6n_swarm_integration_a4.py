"""Integration A: A.4 wiring (host-relative v3 freeze, I6N_DATA_ROOT, direct multi-output arms).

Fabricated fixtures and stub child processes labelled as such; no fit, no data, no TEST.
"""
import csv
import json
import os
import sys
from pathlib import Path

import pytest

from tests import i6n_swarm_fixtures as F
from tests.test_i6n_swarm_campaign import Recorder
from tools import i6n_historical_evaluator as A
from tools import i6n_search as search
from tools.i6n_swarm import agent as AG, bridge as B, collect as C, common, coordinator as K, ledger as L, plan as P

STUB_FREEZE = r'''
import json, os, sys
# STUB (not lane A's freeze): records argv/env and writes a binding of the requested schema.
args = sys.argv[1:]
opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--") and not args[i + 1].startswith("--")}
os.makedirs(opt["--output"])
cell = json.load(open(opt["--config"]))
schema = "i6n.historical_evaluator.v3" if "--data-root" in args else "i6n.historical_evaluator.v2"
binding = dict(i6n_horizon=cell["h"], i6n_support=dict(counts=dict(train=5, inner=2), naive_sums=dict(train=1.0, inner=0.5),
               support_digest=dict(train="1" * 64, inner="2" * 64)),
               i6n_evaluation_context=dict(schema=schema, fold=opt["--fold"], argv=args, env_root=os.environ.get("I6N_DATA_ROOT")),
               i6n_evaluator_command=["x"], i6n_evaluator_timeout=60)
if "--family" in args:
    binding.update(i6n_family=opt["--family"], i6n_offsets=[2, 3, 4, 5, 6, 12] if opt["--family"] == "short" else [18, 24, 36, 48, 60, 72])
json.dump(binding, open(os.path.join(opt["--output"], "BINDING.json"), "w"))
print(json.dumps(binding))
'''


@pytest.fixture(autouse=True)
def _reference_aggregator(monkeypatch):
    monkeypatch.setattr(B, "resolve_score_folds", lambda: B._reference_score_folds)


@pytest.fixture
def campaign(tmp_path):
    (tmp_path / "freeze.py").write_text(STUB_FREEZE)
    cfg = F.private_config(tmp_path, frozen=False)
    root = tmp_path / "campaign"
    P.build(cfg, root)
    K.start(root, roles=[], systemctl=Recorder())
    node = json.loads((root / "private" / "hosts" / "preferred" / "node.json").read_text())
    node.update(freeze_argv=[sys.executable, str(tmp_path / "freeze.py")],
                ledger=dict(path=str(root / "state" / "ledger.sqlite3")))
    return root, node, tmp_path


def test_freeze_is_always_host_relative_and_families_get_their_own_binding(campaign):
    root, node, tmp = campaign
    led = L.Ledger(root / "state" / "ledger.sqlite3")
    AG.run_fixed_units(node, led, stages=("freeze",), boot_id="b")
    units = led.units(stage="freeze")
    assert units and all(u["state"] == "COMPLETE" for u in units)
    family_units = [u for u in units if u["dag_node"].startswith("freeze_family:")]
    assert {u["dag_node"] for u in family_units} == {"freeze_family:short", "freeze_family:long"}
    for u in units:
        ctx = led.terminal(u["fit_id"])["receipt"]["shared_receipt"]["i6n_evaluation_context"]
        assert "--data-root" in ctx["argv"] and ctx["env_root"] == node["shared_data_root"]
        assert ctx["schema"] == "i6n.historical_evaluator.v3"
        assert ("--family" in ctx["argv"]) == u["dag_node"].startswith("freeze_family:")
        assert ("--score-outer" in ctx["argv"]) == (not u["dag_node"].startswith("freeze_folds:"))
        pin = led.terminal(u["fit_id"])["receipt"]["candidate_receipt"]["binding"]
        assert set(pin) == {"relpath", "sha256"}
    # the family binding of a week is a different frozen object than the week's endpoint binding
    assert len({u["fit_id"] for u in units}) == len(units)


def test_production_refuses_a_v2_absolute_binding(campaign):
    root, node, tmp = campaign
    led = L.Ledger(root / "state" / "ledger.sqlite3")
    stub = tmp / "freeze.py"
    stub.write_text(STUB_FREEZE.replace('if "--data-root" in args', 'if False'))
    prod = dict(node, fixture=False)
    unit = led.units(dag_node="freeze_folds:h12")[0]
    assert AG._freeze(prod, led, unit, "b") == "FAILED"
    assert "BINDING_NOT_HOST_RELATIVE_V3" in led.unit(unit["fit_id"])["cause"]
    # and the coordinator refuses any frozen v2 binding in a production DAG node
    AG.run_fixed_units(node, led, stages=("freeze",), boot_id="b")  # fixture node: v2 stub bindings accepted
    payload = dict(P.load(root)["payload"], mode="production")
    node_row = next(n for n in led.dag_nodes() if n["node_id"] == "freeze_folds:h72")
    state, cause, _, frozen = K._freeze_node(led, payload, node_row, led.terminals())
    assert state == "FAILED" and cause.startswith("BINDING_NOT_HOST_RELATIVE_V3") and frozen == {}


def test_evaluate_subprocess_gets_this_hosts_data_root(tmp_path):
    node = dict(owner="o", host_key="h", device=dict(cls="cpu_fixture", cuda_visible_devices="", fixture=True),
                fixture=True, shared_data_root=str(tmp_path / "root-on-this-host"),
                lease=dict(ttl_seconds=5, renew_seconds=1, max_wait_seconds=1, poll_seconds=0.1))
    env = B.Bridge(node, L.Ledger(tmp_path / "l.sqlite3"))._env()
    assert env["I6N_DATA_ROOT"] == str(tmp_path / "root-on-this-host")


def _finalists(effective):
    nodes = {}
    for h, cfg in effective.items():
        best = dict(role="BEST", fitness=0.9 - h / 1000, effective_config=cfg, size_class="S1",
                    genotype_canonical_sha256=search.effective_identity({k: cfg[k] for k in search._GENOME_KEYS}),
                    search_candidate_sha256=common.sha256_json(["s", h, "B"]))
        g2 = search.default_genome(h)
        g2["input_transform"] = "difference"
        cfg2 = search.effective_config(g2)
        ctrl = dict(role="MINIMAL_CONTROL", fitness=0.95, effective_config=cfg2, size_class="S1",
                    genotype_canonical_sha256=search.effective_identity(g2),
                    search_candidate_sha256=common.sha256_json(["s", h, "C"]))
        nodes[f"search:h{h}"] = dict(summary=dict(finalists=[best, ctrl]))
    return nodes


def test_multi_output_arms_use_a4_config_and_family_bindings(tmp_path):
    weeks = ["W2024-01-07", "W2024-01-14"]
    template = F.c_arms_template(weeks, "e" * 64, families=("short",))
    payload = dict(inputs=dict(multi_output=F.pin(tmp_path / "arms.json", template)), experiment="exp-fx",
                   evaluator_capabilities=[K.MULTI_OUTPUT_CAPABILITY], mode="fixture")
    hs = (2, 3, 4, 5, 6, 12)
    effective = {h: search.effective_config(search.default_genome(h)) for h in hs}
    nodes = _finalists(effective)
    node = dict(node_id="multi_output:short", spec=dict(family="short", horizons=list(hs), anchor_horizon=12))
    led = L.Ledger(tmp_path / "l.sqlite3")
    state, cause, _, _ = K._multi_node(led, payload, node, nodes, family_bindings={})
    assert state == "WAIT_DEPENDENCY" and cause.startswith("FAMILY_BINDING_PENDING")
    bindings = {w: (dict(i6n_horizon=12, i6n_family="short", i6n_offsets=list(hs),
                         i6n_support=F.support(i + 1), i6n_evaluation_context=dict(schema="v3", week=w)), "9" * 64)
                for i, w in enumerate(weeks)}
    state, cause, created, summary = K._multi_node(led, payload, node, nodes, family_bindings=bindings)
    assert created and state == "QUEUED" and summary["ready_arms"] == 4
    units = led.units(stage="multi_output")
    assert len(units) == 4
    anchor = min(hs, key=lambda h: (nodes[f"search:h{h}"]["summary"]["finalists"][0]["fitness"], h))
    for u in units:
        cfg = u["spec"]["effective_config"]
        role = "BEST" if u["spec"]["recipe_slot"] == "R1" else "MINIMAL_CONTROL"
        source = next(f for f in nodes[f"search:h{anchor}"]["summary"]["finalists"] if f["role"] == role)
        assert cfg == A.multi_output_arm_config(source["effective_config"], list(hs))
        assert u["effective_config_sha256"] == A.arm_config_sha256(cfg)
        assert cfg["output_mode"] == "direct_multi_output" and cfg["output_offsets"] == list(hs)


def test_collect_reports_direct_head_rows_and_pooled_as_diagnostic(tmp_path):
    cfg = F.private_config(tmp_path)
    root = tmp_path / "campaign"
    P.build(cfg, root)
    K.start(root, roles=[], systemctl=Recorder())
    led = L.Ledger(root / "state" / "ledger.sqlite3")
    sup = F.support(1)
    config = A.multi_output_arm_config(search.effective_config(search.default_genome(12)), [2, 3, 4, 5, 6, 12])
    fit = common.sha256_json(["arm", 1])
    led.ensure_unit(fit, dict(stage="multi_output", dag_node="multi_output:short", horizon=12, fold="W2024-01-07",
                              experiment="exp-fx", genotype_canonical_sha256="a" * 64,
                              effective_config_sha256=common.sha256_json(config), search_candidate_sha256="b" * 64,
                              device_class="gpu_admitted", fixture=True, effective_config=config,
                              request=dict(context={}, support=sup)))
    c = led.claim(fit, "w", "preferred", "b", "cpu_fixture", 60)
    heads = [dict(head_role="DIRECT_HEAD", head_offset_hours=h, split=s, population="S", n=3,
                  sum_abs_error=0.3 * h, naive_sum_abs_error=0.6 * h) for h in (2, 12) for s in ("TRAIN", "INNER")]
    cr = dict(fold="W2024-01-07", experiment="exp-fx", horizon_hours=12, effective_config_sha256=common.sha256_json(config),
              seed=6102026, rows=heads)
    led.complete(fit, c["token"], dict(fit_id=fit, fold="W2024-01-07", experiment="exp-fx",
                                       effective_config_sha256=common.sha256_json(config), genotype_canonical_sha256="a" * 64,
                                       shared_receipt=dict(status="ok", counts=sup["counts"], support_digest=sup["support_digest"],
                                                           error_sums=dict(train=1.0, inner=0.1), naive_sums=sup["naive_sums"]),
                                       candidate_receipt=cr))
    C.collect(root, publisher=None)
    rows = [r for r in csv.DictReader(open(root / "collect" / "RESULTS.csv")) if r["stage"] == "multi_output"]
    kinds = {r["row_kind"] for r in rows}
    assert kinds == {"POOLED_DIAGNOSTIC", "DIRECT_HEAD"}
    direct = [r for r in rows if r["row_kind"] == "DIRECT_HEAD"]
    assert {(int(r["head_offset_hours"]), r["split"]) for r in direct} == {(2, "TRAIN"), (2, "INNER"), (12, "TRAIN"), (12, "INNER")}
    assert all(float(r["skill_mae"]) == pytest.approx(-0.5) for r in direct)


def _caps(tmp_path, **admission):
    receipt = F.pin(tmp_path / "pilot.json", dict(state="COMPLETED", observed_peak_bytes=1_944_051_712))
    entry = dict(receipt=receipt, child_wall_seconds=3600, **admission)
    return P._class_caps(dict(mode="production", class_admission={"S1": entry}), [dict(class_id="S1")])["S1"]


def test_class_cap_coverage_adjustment_raises_never_lowers(tmp_path):
    import math
    base = _caps(tmp_path)
    assert base["child_memory_bytes"] == math.ceil(1.25 * 1_944_051_712) and base["coverage_adjustment_bytes"] == 0
    adj = _caps(tmp_path, coverage_adjustment_bytes=9 * 1024**2, coverage_derivation="d" * 64)
    assert adj["child_memory_bytes"] == math.ceil(1.25 * (1_944_051_712 + 9 * 1024**2))
    assert adj["child_memory_bytes"] > base["child_memory_bytes"] and adj["coverage_derivation_sha256"] == "d" * 64
    assert adj["measured_peak_bytes"] == 1_944_051_712 + 9 * 1024**2 and adj["pilot_peak_bytes"] == 1_944_051_712
    for bad in (dict(coverage_adjustment_bytes=-1, coverage_derivation="d" * 64),
                dict(coverage_adjustment_bytes=True, coverage_derivation="d" * 64),
                dict(coverage_adjustment_bytes=5.0, coverage_derivation="d" * 64),
                dict(coverage_adjustment_bytes=5)):
        with pytest.raises(ValueError):
            _caps(tmp_path, **bad)
    pinned = _caps(tmp_path, coverage_adjustment_bytes=51 * 1024**2,
                   coverage_derivation=F.pin(tmp_path / "derivation.json", dict(rows="S2 h12 vs h36 pilot")))
    assert pinned["coverage_derivation_sha256"] == common.file_sha256(tmp_path / "derivation.json")
