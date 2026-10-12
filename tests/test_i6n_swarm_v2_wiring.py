"""Generated production DOIN configs must select E's joint v2 optimizer."""
import pytest
from optimizer_plugins.i6n_optimizer import Plugin
from tools import i6n_search_plan as E
from tools.i6n_swarm.plan import render_doin_config


def template():
    plan = E.build_plan()
    return dict(campaign_id="wiring", node_label="wiring", port=18000, data_dir="/unused",
                bootstrap_peers=[], shared_min_peers=2, claim_timeout=600,
                predictor_root="/unused", search=plan["search"], node_config="/unused/node.json",
                mode="production", e_plan=plan, admitted_classes=["S1"])


def context():
    folds = []
    for i, f in enumerate(("2023Q1_DCA", "2023Q2_DCA", "2023Q3_DCA", "2023Q4")):
        folds.append(dict(fold=f, context=dict(fold=f), origins_sha256=dict(train="a"*64, inner="b"*64),
                          support=dict(counts=dict(train=100, inner=20),
                                       naive_sums=dict(train=10, inner=4),
                                       support_digest=dict(train=str(i+1)*64, inner=str(i+5)*64))))
    return dict(folds=folds, plan_sha256=E.build_plan()["plan_sha256"])


def test_production_config_selects_joint_optimizer_and_minimal_seed():
    cfg = render_doin_config(template(), {"12": context()})["domains"][0]["optimization_config"]
    assert cfg["i6n_plan"]["plan_sha256"] == E.build_plan()["plan_sha256"]
    assert cfg["i6n_folds"]["2023Q1_DCA"]["i6n_origins_sha256"] == dict(train="a"*64, inner="b"*64)
    pop = Plugin().create_shared_population(8, cfg, seed=6102026)
    assert all(w["schema"] == "i6n-shared-deap-v2" for w in pop["population"])
    seed = pop["population"][0]["genotype"]
    assert seed["width"] == 8 and seed["reducer_filters"] == [4]


def test_production_refuses_missing_joint_plan():
    t = template()
    t.pop("e_plan")
    with pytest.raises(ValueError, match="joint"):
        render_doin_config(t, {"12": context()})


def test_production_refuses_an_unsealed_plan_copy():
    t = template()
    t["e_plan"] = dict(t["e_plan"], seed=1)
    with pytest.raises(ValueError):
        render_doin_config(t, {"12": context()})
