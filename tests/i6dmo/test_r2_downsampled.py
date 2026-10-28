"""Explicit downsampled R2 arms; synthetic metadata only, no science or services."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.i6dmo import design as D, r2_design as R2, r2_runner as R


def reviewed_objective(source, tmp_path):
    review = tmp_path / "INDEPENDENT_REVIEW.synthetic.md"
    review.write_text("Synthetic review fixture, never a scientific execution authority.\n")
    return R2.lead_objective(source, arm="mg12m", ruling=dict(ruled_by="synthetic independent reviewer",
        review_commit="f" * 40, review_verdict="ACCEPT_FOR_RELEASE", review_document=str(review)))


@pytest.fixture
def long_source(tmp_path):
    source = tmp_path / "long_source"
    short = R2.MATCHED_DIR.parent / "EXPERIMENT.short.json"
    D.seal_matched_long(short, source)
    design = D.load_design(source / "EXPERIMENT.mg12m.json")
    effective = D.effective_config(design)
    seal = json.loads((R2.MATCHED_DIR / "POPULATION_SEAL.json").read_text())
    seal["population_digest"] = "6" * 64
    seal["identities"]["arms"] = {"mg12m": {"experiment_design_sha256": effective["design_sha256"]}}
    seal["identities"]["matched_population_rules_sha256"] = D.LONG_MATCHED_POPULATION_RULES["rules_sha256"]
    seal["naive_loss_scale"] = {str(h): .01 for h in D.LONG_HORIZONS}
    (source / "POPULATION_SEAL.json").write_text(json.dumps(seal))
    capacity = json.loads((R2.MATCHED_DIR / "CAPACITY_MATCHED.json").read_text())["arms"]["mg2m"]
    capacity.update(arm="mg12m", family="long_mg12", sealed_design_sha256=effective["design_sha256"])
    (source / "CAPACITY_MATCHED.json").write_text(json.dumps({"arms": {"mg12m": capacity}}))
    (tmp_path / "DATA_MODEL_RETURN.md").write_text("Synthetic owner monitor ruling; not scientific evidence.\n")
    return source


def test_short_contract_and_ruled_objective_remain_exactly_legacy():
    for name, actual in (("R2_CONTRAST_DESIGN.json", R2.build_contract()),
                        ("R2_CONTRAST_DESIGN.ruled.json", R2.rule_objective(R2.build_contract(), R2.lead_objective()))):
        assert json.loads((R2.MATCHED_DIR / name).read_text()) == actual


def test_explicit_long_contract_and_monitor_ruling(long_source, tmp_path):
    contract = R2.build_contract(long_source, arm="mg12m")
    assert contract["shared"]["arm"] == "mg12m"
    assert contract["shared"]["horizons_hours"] == D.LONG_HORIZONS
    assert contract["shared"]["market_time"] == dict(rule="MG12_MARKET_EXACT", observations=13,
        spacing_open_market_hours=12, span_open_market_hours=144)
    assert contract["selection"]["comparison_sha256"] == "22d9ac6ce7ce5a5ff3a78f3ce5d2b29672cdd79d7f0d35d6f2ae155963a01fa3"
    objective = R2.lead_objective(long_source, arm="mg12m")
    assert objective["version"] == "1.0.0"
    assert objective["fit_settings"]["monitor_version"] == "i6dmo.branch_mean_fit_inner.v1"
    ruled = R2.rule_objective(contract, objective)
    assert R2.validate_r2_design(ruled, matched_dir=long_source)["status"] == "PREPARING_INDEPENDENT_REVIEW"
    with pytest.raises(R2.R2DesignError, match="R2_INDEPENDENT_REVIEW_REQUIRED"):
        R2.validate_r2_design(ruled, for_execution=True, matched_dir=long_source)
    objective = reviewed_objective(long_source, tmp_path)
    ruled = R2.rule_objective(contract, objective)
    assert R2.validate_r2_design(ruled, for_execution=True, matched_dir=long_source)["status"] == "READY_FOR_EXECUTION"
    assert R2.source_dir("mg12m") == R2.MATCHED_DIR.parent / "r2_downsampled/long_source"
    with pytest.raises(R2.R2DesignError):
        R2.validate_r2_design(ruled, for_execution=True, matched_dir=R2.MATCHED_DIR)
    wrong = copy.deepcopy(objective)
    wrong["fit_settings"]["monitor"] = "validation_loss_on_purged_INNER_inputs"
    with pytest.raises(R2.R2DesignError, match="R2_OBJECTIVE_NOT_EXECUTABLE"):
        R2.validate_r2_design(R2.rule_objective(contract, wrong), for_execution=True, matched_dir=long_source)


def test_bind_uses_contract_arm_not_short_default(long_source):
    contract = R2.build_contract(long_source, arm="mg12m")
    shared = contract["shared"]
    prepared = dict(designs={"mg12m": dict(seal_sha256=shared["experiment_seal_sha256"],
        runtime=shared["runtime"])}, data_pins={"common_population": {"population_sha256": shared["population_sha256"]}})
    R._bind(contract, prepared)
    prepared["designs"] = {"mg2m": prepared["designs"].pop("mg12m")}
    with pytest.raises(Exception, match="R2_CONTRACT_BINDING"):
        R._bind(contract, prepared)


def test_write_long_ruled_requires_actual_accepted_review(long_source, tmp_path):
    base, out = tmp_path / "base.json", tmp_path / "ruled.json"
    R2.seal(long_source, base, arm="mg12m")
    with pytest.raises(R2.R2DesignError, match="R2_INDEPENDENT_REVIEW_REQUIRED"):
        R2.write_ruled(base, out, long_source)
    assert not out.exists()
    objective = reviewed_objective(long_source, tmp_path)
    result = R2.write_ruled(base, out, long_source, objective=objective)
    ruled = json.loads(out.read_text())
    assert result["seal_sha256"] == ruled["seal_sha256"]
    assert ruled["pretraining"]["objective"] == objective
    assert R2.validate_r2_design(ruled, for_execution=True, matched_dir=long_source)["status"] == "READY_FOR_EXECUTION"
    (tmp_path / "INDEPENDENT_REVIEW.synthetic.md").write_text("Changed review bytes\n")
    with pytest.raises(R2.R2DesignError, match="R2_OBJECTIVE_UNREVIEWED"):
        R2.validate_r2_design(ruled, for_execution=True, matched_dir=long_source)


def long_metrics(score, arm="mg12m"):
    from tests.i6dmo.test_raw_long_operations import _metrics
    metrics = _metrics(arm, score)
    metrics["matched"].update(fit_runtime="2.21.0", fit_gpu_class="synthetic-same-gpu-class",
                               fit_identity={}, capacity={"capacity_sha256": "c" * 64})
    return metrics


def import_record():
    record = dict(schema="i6dmo.r2_import.v1", arm="mg12m", trainable=True,
        donor_terminals={f"f{i:02d}": f"{i:02d}" * 32 for i in range(19)}, branches={})
    record["import_sha256"] = R2._import_record_sha(record)
    return record


def test_long_comparison_validates_actual_metadata_cache_and_same_gpu():
    a, b, record = long_metrics(.1), long_metrics(-.1), import_record()
    b["matched"]["fit_identity"]["r2_import_sha256"] = record["import_sha256"]
    assert R2.decide_r2(a, b, r2_import_record=record)["selected"] == "R2"
    for change in (lambda m: m["per_horizon"].pop("72"),
                   lambda m: m["matched"]["selection"].update(score=-99.),
                   lambda m: m["c2"].update(state="INCOMPLETE"),
                   lambda m: m["matched"].update(fit_gpu_class="another-class")):
        changed = copy.deepcopy(b)
        change(changed)
        with pytest.raises(R2.R2DesignError):
            R2.decide_r2(a, changed, r2_import_record=record)


def test_mean_monitor_selects_same_weight_fit_inner_mean_and_restores():
    from tools.i6dmo import model as M
    class EpochModel:
        def __init__(self):
            self.epoch = 0
            self.iterations = 0
            self.optimizer = SimpleNamespace(iterations=SimpleNamespace(numpy=lambda: self.iterations))
            self.seen = []
        def compile(self, **kwargs):
            self.compiled = kwargs
        def reset_metrics(self):
            pass
        def train_on_batch(self, x, y):
            self.epoch += 1
            self.iterations += 1
            return 999.  # online training loss must NOT select
        def __call__(self, x, training=False):
            self.seen.append((self.epoch, float(x[0, 0, 0]), training))
            loss = ((0., 1., 8.) if x[0, 0, 0] == 0 else (9., 4., 1.))[self.epoch - 1]
            return x + np.sqrt(loss)
        def get_weights(self):
            return [np.array([self.epoch], dtype=np.float32)]
        def set_weights(self, values):
            self.epoch = int(values[0][0])
    model = EpochModel()
    settings = dict(max_epochs=3, patience=10, batch_size=2, learning_rate=1e-3, weight_decay=1e-4,
                    min_delta=1e-8, max_updates=100, max_seconds=60., seed=6102026)
    fit, inner = np.zeros((2, 13, 1)), np.ones((2, 13, 1))
    result = M.fit_branch_mean_monitor(model, fit, inner, settings, loss="mse")
    assert result["selected_epoch"] == 2 and model.epoch == 2
    assert [r["monitored_loss"] for r in result["history"]] == pytest.approx([4.5, 2.5, 4.5])
    assert all(not training for _, _, training in model.seen)
    assert result["monitor_version"] == "i6dmo.branch_mean_fit_inner.v1"


def test_pretrain_full_cgroup_peak_is_measured_without_writes_or_RSS_substitution(tmp_path):
    proc = tmp_path / "self-cgroup"
    root = tmp_path / "cgroup"
    group = root / "attempt.scope"
    group.mkdir(parents=True)
    peak = group / "memory.peak"
    peak.write_text("123456789\n")
    proc.write_text("0::/attempt.scope\n")
    before = peak.stat()
    result = R.pretrain_memory_cost(proc, root)
    assert result["cgroup_peak_bytes"] == 123456789 and result["cgroup_peak_state"] == "MEASURED"
    assert peak.stat().st_mtime_ns == before.st_mtime_ns
    peak.unlink()
    assert R.pretrain_memory_cost(proc, root)["cgroup_peak_bytes"] is None
    assert R.pretrain_memory_cost(proc, root)["cgroup_peak_state"] == "UNAVAILABLE"


def test_context_uses_long_design_horizons_and_reports_without_short_alias(tmp_path, monkeypatch):
    from tools.i6dmo import train as T
    source = R2.source_dir("mg12m")
    design = D.load_design(source / "EXPERIMENT.mg12m.json")
    calls = []
    def load(path, digest, horizons, features, input_dir=None):
        calls.append((path, digest, horizons, len(features), input_dir))
        return "synthetic-population"
    monkeypatch.setattr(T, "load_matched_population", load)
    spec = dict(_folder=str(tmp_path), effective_config=dict(design=design, code_commit="synthetic"),
        pins=dict(common_population=dict(path="synthetic-pop", population_sha256="p" * 64),
                  matched_input=dict(path="synthetic-input")), unit_id="r2.pretrain.00", family="long",
        kind="pretrain", campaign_id="synthetic", identity_sha256="i" * 64)
    _, effective, pop = R._context(spec)
    assert pop == "synthetic-population" and effective["horizons_hours"] == D.LONG_HORIZONS
    assert calls[0][2] == D.LONG_HORIZONS and calls[0][3] == 19
    report = R._report(spec, "synthetic", [R._metric("Long_H72", .1, "ratio", "validation", 72)])
    assert report["tags"]["family"] == "long" and report["metrics"][0]["horizon"] == 72


def test_canonical_long_contract_is_validated_by_arm_not_contract_location(tmp_path):
    contract = R2.build_contract(arm="mg12m")
    assert contract["shared"]["population_sha256"].startswith("360f3bc1")
    ruled = R2.rule_objective(contract, reviewed_objective(R2.source_dir("mg12m"), tmp_path))
    path = tmp_path / "any-location.json"
    path.write_text(json.dumps(ruled))
    assert R._contract(path)[1] == ruled
    wrong = copy.deepcopy(ruled)
    wrong["shared"]["arm"] = "mg2m"
    path.write_text(json.dumps(wrong))
    with pytest.raises(Exception, match="R2_CONTRACT_NOT_EXECUTABLE"):
        R._contract(path)


@pytest.mark.parametrize("arm", ["mg2m", "mg12m"])
def test_custom_selected_design_prepare_keeps_full_original_pair(tmp_path, monkeypatch, arm):
    from tools.i6dmo import matched_runner as MR
    objective = R2.lead_objective() if arm == "mg2m" else reviewed_objective(R2.source_dir(arm), tmp_path)
    contract = R2.rule_objective(R2.build_contract(arm=arm), objective)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    calls = []
    def prepare(**kwargs):
        calls.append(kwargs)
        shared = contract["shared"]
        matched = dict(designs={arm: dict(seal_sha256=shared["experiment_seal_sha256"], runtime=shared["runtime"])},
                       data_pins={"common_population": {"population_sha256": shared["population_sha256"]}})
        R._atomic(kwargs["out"], matched)
        return matched
    monkeypatch.setattr(MR, "prepare", prepare)
    selected = R2.source_dir(arm) / f"EXPERIMENT.{arm}.json"
    R.prepare(deployment=tmp_path / "deployment.json", input_dir=tmp_path / "synthetic",
              out=tmp_path / "prepared.json", contract=path, arm_design=selected, arm=arm)
    pair = MR.ARMS if arm == "mg2m" else MR.LONG_ARMS
    assert set(calls[0]["designs"]) == set(pair)
    assert calls[0]["designs"][arm] == selected
    other = next(a for a in pair if a != arm)
    assert calls[0]["designs"][other].is_file()


@pytest.mark.parametrize("budget_stop", [False, True])
def test_mean_monitor_patience_ten_and_hard_budget_never_emit_partial_checkpoint(budget_stop):
    from tools.i6dmo import model as M
    class ConstantModel:
        def __init__(self):
            self.weight = 0
            self.updates = 0
            self.optimizer = SimpleNamespace(iterations=SimpleNamespace(numpy=lambda: self.updates))
        def compile(self, **kwargs):
            pass
        def reset_metrics(self):
            pass
        def train_on_batch(self, x, y):
            self.weight += 1
            self.updates += 1
            return 1.
        def __call__(self, x, training=False):
            return x + 1
        def get_weights(self):
            return [np.array([self.weight], dtype=np.float32)]
        def set_weights(self, weights):
            self.weight = int(weights[0][0])
    model = ConstantModel()
    x = np.zeros((2, 13, 1))
    settings = dict(max_epochs=20, patience=10, batch_size=2, learning_rate=1e-3, weight_decay=1e-4,
                    min_delta=1e-8, max_updates=1 if budget_stop else 100, max_seconds=60., seed=6102026)
    if budget_stop:
        with pytest.raises(M.DonorImportError, match="R2_PRETRAINING_BUDGET_STOP"):
            M.fit_branch_mean_monitor(model, x, x, settings)
    else:
        result = M.fit_branch_mean_monitor(model, x, x, settings)
        assert result["stop_reason"] == "patience" and result["observed_updates"] == 11
        assert result["selected_epoch"] == model.weight == 1


def test_long_pretrain_routes_versioned_o1_without_changing_registered_loss(monkeypatch):
    from tools.i6dmo import model as M, provenance
    from predictor_plugins.modular_temporal import objectives as O
    objective = R2.lead_objective(arm="mg12m")
    bundle = M.build_dmo(M.model_config(provenance.MANIFEST, family="long_mg12_matched"))
    feature = bundle.config["features"][0]
    calls = []
    def fit(ae, x, inner, settings, loss="mse"):
        calls.append((ae, settings, loss))
        return dict(selected_epoch=1, selected_updates=1, observed_updates=1, best_validation_loss=1.,
                    best_monitored_loss=1., stop_reason="max_epochs", stop_class="max_epochs",
                    elapsed_seconds=0., history=[], monitor_version=R2.MEAN_MONITOR_VERSION,
                    restored_best_weights=True, reconstruction={"state": "MEASURED"})
    monkeypatch.setattr(M, "fit_branch_mean_monitor", fit)
    monkeypatch.setattr(O.AutoencoderReconstruction, "fit", lambda *a: pytest.fail("legacy loop used for long"))
    windows = np.zeros((2, 13, 1), dtype=np.float32)
    result = M.pretrain_branch(bundle, feature, windows, windows, objective)
    assert len(calls) == 1 and calls[0][2] == "mse"
    assert result["fit"]["monitor_version"] == R2.MEAN_MONITOR_VERSION
    assert result["objective_identity"]["version"] == "1.0.0"
    with pytest.raises(M.DonorImportError, match="R2_OBJECTIVE_UNKNOWN"):
        M.pretrain_branch(bundle, feature, windows, windows, R2.lead_objective())


@pytest.mark.parametrize("bad_c2", [False, True])
def test_long_runner_comparison_authenticates_child_c2_and_emits_governed_reports(tmp_path, bad_c2):
    import hashlib
    from tests.i6dmo.test_raw_pair import store
    from tools.i6dmo.outbox import seal_report
    a, b, record = long_metrics(.1), long_metrics(-.1), import_record()
    b["matched"]["fit_identity"]["r2_import_sha256"] = record["import_sha256"]
    upstream = {}
    for regime, metrics in (("r0", a), ("r2", b)):
        child_c2 = copy.deepcopy(metrics["c2"])
        if bad_c2 and regime == "r0":
            child_c2["input_sha256"] = "0" * 64
        eligibility = {h: dict(n=row["n"], n_declared=row["n_declared"], n_failed=0,
            state="COMPLETE", mae=row["mae"], naive=row["naive_mae"], relative_error=row["relative_error"],
            eligible=row["eligible_mae_below_naive"], row_ids_sha256=row["row_ids_sha256"])
            for h, row in metrics["per_horizon"].items()}
        upstream[regime + ".report"] = store(tmp_path, regime + ".report", "long",
            {"metrics": ("METRICS.json", metrics)}, eligibility=eligibility,
            lineage=dict(arm="mg12m", common_population_sha256="a" * 64, c2=child_c2),
            selection=metrics["matched"]["selection"])
    record_sha = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    upstream["r2.fit"] = store(tmp_path, "r2.fit", "long", {
        "R2_IMPORT.json": ("R2_IMPORT.json", record),
        "RECEIPT": ("RECEIPT.json", {"artifacts": {"R2_IMPORT.json": record_sha}})})
    spec = dict(unit_id="r2.comparison", identity_sha256="f" * 64, kind="comparison", family="long",
        campaign_id="synthetic-r2-long", upstream=upstream, effective_config=dict(code_commit="c" * 40,
        data_pins={"common_population": {"population_sha256": "a" * 64}}))
    result = R.run_comparison(spec, tmp_path / "result", tmp_path / "progress.json")
    if bad_c2:
        assert result["status"] == "FAILED" and result["disposition"].startswith("MATCHED_REPORT_C2")
    else:
        assert result["status"] == "COMPLETE" and result["disposition"] == "R2_SELECTED", result
        for report in result["reports"]:
            assert seal_report(report)["report_sha256"]


def test_new_pretrain_terminal_captures_peak_but_explicit_old_null_is_not_rewritten(monkeypatch):
    measured = dict(cgroup_peak_bytes=123456789, cgroup_peak_state="MEASURED", cgroup_peak_source="synthetic")
    monkeypatch.setattr(R, "pretrain_memory_cost", lambda: dict(measured))
    spec = dict(kind="pretrain", unit_id="r2.pretrain.00")
    terminal = R._terminal(spec, "FAILED", "SYNTHETIC_BEFORE_FIT", costs={"fit_seconds": 0.})
    assert terminal["costs"]["cgroup_peak_bytes"] == measured["cgroup_peak_bytes"]
    old = {"cgroup_peak_bytes": None}
    assert R._terminal(spec, "COMPLETE", "SYNTHETIC", costs=old)["costs"] == old
