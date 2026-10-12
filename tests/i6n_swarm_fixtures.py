"""Tiny generated fixtures for lane B campaign tests (no real data, no TEST rows)."""
import json
import sys
from pathlib import Path

from tools import i6n_search as search
from tools.i6n_swarm import common

GIB = 1024**3


def pin(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True))
    return dict(path=str(path), sha256=common.file_sha256(path))


def support(scale):
    return dict(counts=dict(train=100, inner=10), naive_sums=dict(train=1.0 * scale, inner=0.1 * scale),
                support_digest=dict(train=common.sha256_json(["t", scale]), inner=common.sha256_json(["i", scale])))


def binding(root, name, horizon, scale, **context):
    ctx = dict(schema="fixture.binding.v1", name=name, fixture=True)
    ctx.update(context)
    return pin(root / "bindings" / f"{name}.json", dict(
        i6n_horizon=horizon, i6n_support=support(scale), i6n_evaluation_context=ctx,
        i6n_evaluator_command=[sys.executable, "-m", "tools.i6n_historical_evaluator", "evaluate"],
        i6n_evaluator_timeout=600))


def build_inputs(tmp_path, horizons=(12, 72), folds=("2023Q1", "2023Q2"), weeks=("W2024-01-07", "W2024-01-14"),
                 frozen=True, wait_folds=(), dca=()):
    root = tmp_path / "inputs"
    plan = dict(schema="i6n.search_plan.v1", horizons_hours=list(horizons), seed=6102026, replicas=1,
                search=dict(population_size=2, generations=3, new_candidates_per_horizon=4, mutation=0.8,
                            immigration=0.2, crossover=False, tolerance=1e-4, patience=3, direction="minimize"),
                budgets=dict(new_candidates_per_horizon_max=4, folds_per_candidate=len(folds),
                             new_fits_per_horizon_max=4 * len(folds), total_new_fits_max=4 * len(folds) * len(horizons),
                             max_physical_attempts_per_identity_fold=3, cache_hits_count_as_new_fits=False,
                             oom_retry_with_lower_cap=False),
                classes=[dict(class_id="S1", max_samples=25, max_width=32, min_delta_x_hours=1,
                              child_memory_bytes=None, child_wall_seconds=None,
                              admission="FROM_ADMITTED_COST_PILOT_RECEIPT"),
                         dict(class_id="S3", max_samples=433, max_width=128, min_delta_x_hours=1,
                              child_memory_bytes=None, child_wall_seconds=None,
                              admission="FROM_ADMITTED_COST_PILOT_RECEIPT")],
                folds=dict(fold_ids=list(folds), quarters=list(folds), all_folds_required_before_ranking=True),
                initial_genotypes={str(h): dict(seed_genotype_canonical_sha256=search.effective_identity(search.default_genome(h)),
                                                alternative_genotype_canonical_sha256=None) for h in horizons},
                finalists=dict(rule="best complete fitness + minimal control (seed), or small alternative"),
                fitness=dict(aggregation="unweighted_mean_over_folds"),
                annual=dict(max_dispositions=1272), multi_output=dict(max_dispositions=212))
    plan["plan_sha256"] = common.sha256_json(plan)
    def fold_entry(h, i, f):
        if f in dca:
            f_c = f + "_DCA"
            entry = dict(fold_id=f_c, unit_id=f_c, cutoff_utc=f"2023-0{3 * i + 3}-24T21:00:00Z",
                         state="READY_WITH_DECLARED_CLOCK_ASSUMPTION", cause=None,
                         declared_assumption="DCA_2019_EU_DST_POLICY_2019-03-11_2019-12-01",
                         unit_sha256=common.sha256_json([h, f_c]), cell_config=None,
                         binding=binding(root, f"fold-{h}-{f_c}", h, i + 1))
            return entry
        entry = dict(fold_id=f, unit_id=f, cutoff_utc=f"2023-0{3 * i + 3}-24T21:00:00Z", state="READY", cause=None,
                     unit_sha256=common.sha256_json([h, f]), cell_config=None, binding=None)
        if frozen:
            entry["binding"] = binding(root, f"fold-{h}-{f}", h, i + 1)
        else:
            entry["cell_config"] = pin(root / "cells" / f"{f}_h{h}.json", dict(schema="historical_cell.v1", h=h, unit=f))
            if f in wait_folds:
                entry.update(state="WAIT_DEPENDENCY", cause="CLOCK_SCOPE_CERTIFICATE missing", cell_config=None)
        return entry
    views = dict(schema="i6n.weekly_views.v1", experiment="exp-fx",
                 folds={str(h): [fold_entry(h, i, f) for i, f in enumerate(folds)] for h in horizons},
                 weeks=[dict(week_id=w, unit_id=w, cutoff_utc="2024-01-05T22:00:00Z", release_at_utc="2024-01-07T22:00:00Z",
                             horizons=list(horizons), eligible=True, cause={}, unit_sha256=common.sha256_json(w),
                             cell_configs={str(h): pin(root / "cells" / f"{w}_h{h}.json", dict(h=h, w=w)) for h in horizons},
                             bindings={str(h): binding(root, f"week-{h}-{w}", h, 3 + i) for h in horizons} if frozen else None,
                             binding=None)
                        for i, w in enumerate(weeks)]
                 + [dict(week_id="W2024-12-29", unit_id="W2024-12-29", cutoff_utc="2024-12-27T22:00:00Z",
                         release_at_utc="2024-12-29T22:00:00Z", horizons=[], eligible=False,
                         cause={str(h): "NO_ELIGIBLE_ROWS" for h in horizons}, unit_sha256=common.sha256_json("x"),
                         cell_configs={}, bindings=None, binding=None)])
    multi_doc = dict(schema="i6n.multi_output.arms.v1", arms=[
        dict(family="short", arm_id=f"short-{w}", week_id=w, horizons=[2, 3, 4, 5, 6, 12],
             config=dict(kind="multi_output_fixture", family="short"),
             binding=binding(root, f"multi-short-{w}", 12, 5)) for w in weeks])
    return dict(search_plan=pin(root / "plan_e.json", plan), weekly_views=pin(root / "views_c.json", views),
                multi_output=pin(root / "multi_c.json", multi_doc))


def private_config(tmp_path, fake_evaluator=None, frozen=True, wait_folds=(), dca=(), **overrides):
    inputs = build_inputs(tmp_path, frozen=frozen, wait_folds=wait_folds, dca=dca)
    token = tmp_path / "private" / "ledger.token"
    token.parent.mkdir(parents=True, exist_ok=True)
    token.write_text("fixture-token")
    token.chmod(0o600)
    evaluator = fake_evaluator or [sys.executable, "-m", "tools.i6n_historical_evaluator", "evaluate"]
    cfg = dict(
        schema="i6n.swarm_campaign.private.v1", campaign_id="fx1", mode="fixture", experiment="exp-fx",
        seed=6102026, prior_candidates=None, evaluator_capabilities=["VALIDATION_OUTER_WEEK"],
        shared_data_root=str(tmp_path / "inputs"), **inputs,
        hosts=[dict(role="coordinator", ledger_host=True, devices=[], python=sys.executable,
                    workdir=str(Path.cwd()), data_dir=str(tmp_path / "data-coord"), doin_port=None),
               dict(role="preferred", ledger_host=False, python=sys.executable, workdir=str(Path.cwd()),
                    data_dir=str(tmp_path / "data-pref"), doin_port=18470, evaluator_argv=evaluator,
                    devices=[dict(cls="cpu_fixture", cuda_visible_devices="", fixture=True)]),
               dict(role="secondary", ledger_host=False, python=sys.executable, workdir=str(Path.cwd()),
                    data_dir=str(tmp_path / "data-sec"), doin_port=18471, evaluator_argv=evaluator,
                    devices=[dict(cls="cpu_fixture", cuda_visible_devices="", fixture=True)])],
        ledger=dict(bind="127.0.0.1", port=18790, token_file=str(token), url="http://127.0.0.1:18790"),
        launcher=None,
        doin=dict(node_command=[sys.executable, "-m", "doin_node.cli"], shared_min_peers=2, claim_timeout=600,
                  peers={"preferred": "127.0.0.1:18470", "secondary": "127.0.0.1:18471"}),
        lease=dict(ttl_seconds=5, renew_seconds=0.2, max_wait_seconds=5, poll_seconds=0.1),
        class_admission={"S1": dict(source="FIXTURE_DECLARED_NOT_MEASURED", child_memory_bytes=2 * GIB,
                                    child_wall_seconds=600, vram_bytes=GIB, small_pilot=True),
                         "S3": dict(source="FIXTURE_DECLARED_NOT_MEASURED", child_memory_bytes=2 * GIB,
                                    child_wall_seconds=600, vram_bytes=GIB, small_pilot=False)},
        tick_interval_seconds=300)
    cfg.update(overrides)
    return pin(tmp_path / "private" / "campaign.json", cfg)["path"]
