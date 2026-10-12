"""Generated data only; no historical fits or external clock certification."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_i6n_historical_cell import fixture as historical_fixture, write_json, repin
from tools import i6n_historical_evaluator as evaluator
from tools import i6n_search as search
from tools.i6n_contracts import paired_metrics as c2


@pytest.fixture
def prepared(historical_fixture, tmp_path):
    cfg, _ = historical_fixture
    cfg.update(horizon_hours=2, samples=13, spacing_hours=2)
    cfg["resources"]["max_samples"] = 100
    # Plan section 2 stopping is mandatory for every frozen fold.
    cfg["training"].update(max_epochs=50, patience=5, min_delta=1e-6, seed=6102026)
    # 50 plan epochs need an update budget sized for them (callers may enlarge data).
    cfg["resources"]["max_updates"] = 10_000
    path = tmp_path / "base.json"
    write_json(path, cfg)
    binding = evaluator.freeze(path, tmp_path / "bridge", experiment="unit-exp", fold="F0")
    genome = search.default_genome(2)
    genome.update(delta_x=2, core=cfg["model"])
    request = dict(config=search.effective_config(genome), generation=0,
                   context=binding["i6n_evaluation_context"], support=binding["i6n_support"])
    return cfg, path, binding, request


def test_freeze_emits_isolated_command_and_explicit_timeout(prepared):
    cfg, _, binding, request = prepared
    assert "i6n_evaluator" not in binding
    assert binding["i6n_evaluator_command"] == [
        sys.executable, "-m", "tools.i6n_historical_evaluator", "evaluate"]
    assert binding["i6n_evaluator_timeout"] == cfg["resources"]["max_seconds"] + 120
    persisted = evaluator.Path(request["context"]["output_root"]) / "BINDING.json"
    assert json.loads(persisted.read_text()) == binding


def test_thread_callback_rejected_before_loading_or_output(tmp_path, monkeypatch):
    root = tmp_path / "not-created"
    monkeypatch.setattr(evaluator, "load_request", lambda *a: pytest.fail("thread reached request loading"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(evaluator.evaluate, {"context": {"output_root": str(root)}})
        with pytest.raises(ValueError, match="main thread.*i6n_evaluator_command"):
            future.result()
    assert not root.exists()


def test_existing_timer_rejected_before_loading_or_output(tmp_path, monkeypatch):
    root = tmp_path / "not-created"
    monkeypatch.setattr(evaluator.signal, "getitimer", lambda *a: (1., 0.))
    monkeypatch.setattr(evaluator, "load_request", lambda *a: pytest.fail("timer reached request loading"))
    with pytest.raises(ValueError, match="existing wall timer"):
        evaluator.evaluate({"context": {"output_root": str(root)}})
    assert not root.exists()


def test_generated_binding_dispatches_command_from_executor_without_fit(prepared, monkeypatch):
    from optimizer_plugins import i6n_optimizer as optimizer
    _, _, binding, request = prepared
    calls = []
    def command(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout='{"status":"no-fit-test"}')
    monkeypatch.setattr(optimizer.subprocess, "run", command)
    monkeypatch.setattr(evaluator, "evaluate", lambda *a: pytest.fail("inline callback invoked"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(optimizer.Plugin()._evaluate, binding, request).result() == {"status": "no-fit-test"}
    argv, kwargs = calls[0]
    assert len(calls) == 1 and argv == binding["i6n_evaluator_command"]
    assert kwargs["timeout"] == binding["i6n_evaluator_timeout"]
    assert kwargs["shell"] is False and kwargs["check"] is True
    assert json.loads(kwargs["input"]) == request


def test_all_endpoint_genes_mapped_and_derived_fields_checked(prepared):
    base, _, _, request = prepared
    g = search.default_genome(2)
    g.update(delta_x=1, multiplier=6, input_transform="difference")
    g["core"].update(entrance="conv_transformer", width=32, transformer_blocks=2,
        reducer_filters=[16, 8, 4], reducer_strides=[2, 1, 2], adapter="lstm", adapter_width=8)
    effective = search.effective_config(g)
    cfg = evaluator.candidate_config(base, effective)
    assert cfg["model"] == effective["core"]
    assert (cfg["samples"], cfg["spacing_hours"], cfg["horizon_hours"]) == (25, 1, 2)
    assert effective["input_transform"] == "difference"
    for key in ("samples", "final_steps", "history_hours"):
        changed = deepcopy(effective)
        changed[key] += 1
        with pytest.raises(ValueError, match="effective"):
            evaluator.candidate_config(base, changed)
    changed = deepcopy(effective)
    changed["unknown_gene"] = 1
    with pytest.raises(ValueError, match="effective"):
        evaluator.candidate_config(base, changed)
    for change in ({"input_transform": "not_implemented"}, {"delta_y": 1},
                   {"output_offsets": [1, 2]}, {"input_offsets": [-2, 0]}):
        with pytest.raises(ValueError, match="effective"):
            evaluator.candidate_config(base, {**effective, **change})


def test_trajectory_is_supervised_not_rejected_or_renamed(prepared, monkeypatch):
    base, _, _, request = prepared
    g = search.default_genome(2)
    g.update(output_mode="trajectory", delta_y=1, delta_x=2, core=base["model"])
    request["config"] = search.effective_config(g)
    cfg = evaluator.candidate_config(base, request["config"])
    assert evaluator.cell.output_offsets(cfg) == (1, 2) and cfg["model"]["n_horizons"] == 2
    seen = []
    def stop(*args, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("reached trajectory data preparation")
    monkeypatch.setattr(evaluator.cell, "prepare", stop)
    with pytest.raises(RuntimeError, match="trajectory data"):
        evaluator.evaluate(request)
    assert seen == [{"trajectory": True}]


def test_backward_difference_masks_first_slot_gaps_and_missing_sources():
    x = np.ones((1, 5, 41), np.float32)
    x[0, :, 0] = [2, 5, 9, 12, 20]
    x[0, 2, -1] = 0
    x[0, 4, 3] = 0
    original = x.copy()
    result = evaluator.transform_windows(x, "difference")
    np.testing.assert_array_equal(result[0, :, 0], [0, 3, 0, 0, 8])
    np.testing.assert_array_equal(result[0, :, -1], [0, 1, 0, 0, 1])
    assert result[0, 4, 3] == 0 and result[0, 4, 2] == 0
    np.testing.assert_array_equal(x, original)
    future = x.copy()
    future[:, -1, :] = 200
    np.testing.assert_array_equal(evaluator.transform_windows(future, "difference")[:, :-1], result[:, :-1])


def test_frozen_support_independent_of_window_and_transform(prepared):
    base, _, binding, request = prepared
    before = evaluator.support(evaluator.cell.prepare(base), base)
    for dx in (1, 2):
        g = search.default_genome(2)
        g.update(delta_x=dx, input_transform="difference", core=base["model"])
        cfg = evaluator.candidate_config(base, search.effective_config(g))
        actual = evaluator.support(evaluator.cell.prepare(cfg), cfg)
        assert actual == before == binding["i6n_support"]
    assert before["counts"] == {"train": 7, "inner": 7}
    assert before["naive_sums"]["train"] > 0


def test_naive_support_uses_c2_float64_arithmetic(prepared):
    base, _, _, _ = prepared
    with np.load(base["input"]["path"]) as a:
        ts, x, price = a["ts"], a["x"], a["price"].astype(np.float32)
    np.savez(base["input"]["path"], ts=ts, x=x, price=price)
    repin(base)
    data = evaluator.cell.prepare(base)
    frozen = evaluator.support(data, base)
    _, current, future = evaluator.observations(data, base, "train")
    assert current.dtype == future.dtype == np.float64
    assert frozen["naive_sums"]["train"] == evaluator.math.fsum(abs(float(a)-float(b)) for a, b in zip(current, future))


def test_candidate_may_not_drop_a_frozen_endpoint(prepared, monkeypatch):
    from dataclasses import replace
    _, _, _, request = prepared
    original = evaluator.cell.prepare
    def changed(cfg, **kwargs):
        data = original(cfg, **kwargs)
        part = data["parts"].train
        masks = list(part.mask)
        masks[next(i for i, row in enumerate(masks) if any(row))] = (False,)
        data["parts"] = replace(data["parts"], train=replace(part, mask=tuple(masks)))
        return data
    monkeypatch.setattr(evaluator.cell, "prepare", changed)
    monkeypatch.setattr(evaluator, "fit_candidate", lambda *a: pytest.fail("changed support reached fit"))
    with pytest.raises(ValueError, match="intersection/drop forbidden"):
        evaluator.evaluate(request)


@pytest.mark.parametrize("field", ["counts", "naive_sums", "support_digest"])
def test_support_mismatch_never_fits(prepared, monkeypatch, field):
    _, _, _, request = prepared
    request["support"][field]["train"] = "0"*64 if field == "support_digest" else 123
    monkeypatch.setattr(evaluator, "fit_candidate", lambda *a: pytest.fail("mismatch reached fit"))
    with pytest.raises(ValueError, match="support"):
        evaluator.evaluate(request)


def test_tiny_real_difference_fit_shared_receipt_reload_and_reuse(prepared, monkeypatch):
    _, _, binding, request = prepared
    request["config"]["input_transform"] = "difference"
    receipt = evaluator.evaluate(request)
    score, _ = search._score(receipt, request["support"])
    assert score == receipt["fitness"]
    assert receipt["status"] == "ok" and receipt["metadata"]["checkpoint_reload_verified"]
    assert receipt["metadata"]["search_space_scope"] == "ENDPOINT_AND_TRAJECTORY"
    assert receipt["metadata"]["candidate_receipt"]["output_mode"] == "endpoint"
    assert receipt["metadata"]["effective_config"] == request["config"]
    assert receipt["metadata"]["training"]["observed_updates"] > 0
    assert receipt["metadata"]["training"]["early_stop"]["rule"]["monitor"] == "train_inner_mean"
    for split in ("train", "inner"):
        metrics = receipt["metadata"]["endpoint_metrics"][split]
        assert metrics["mae"] == receipt["error_sums"][split]/receipt["counts"][split]
        assert metrics["unit"] == "PRICE" and metrics["scale"] == "ORIGINAL_PRICE"
        retained = receipt["metadata"]["c2_receipts"][split]
        native = json.loads(evaluator.Path(retained["path"]).read_text())
        c2.verify_result(native)
        group = next(g for g in native["groups"] if g["population_kind"] == "S" and g["role"] == "management_origin")
        assert group["n"] == request["support"]["counts"][split]
        assert group["coverage"] == 1 and group["sum_abs_error"] == receipt["error_sums"][split]
    root = evaluator.Path(receipt["metadata"]["artifact_directory"])
    with np.load(root / "scores.npz") as rows:
        for split in ("train", "inner"):
            expected = np.abs(rows[split+"_prediction"] - rows[split+"_future"])
            assert receipt["error_sums"][split] == pytest.approx(expected.sum())
    monkeypatch.setattr(evaluator, "fit_candidate", lambda *a: pytest.fail("retrained identical candidate"))
    request["generation"] = 8
    assert evaluator.evaluate(request) == receipt
    from optimizer_plugins.i6n_optimizer import Plugin
    genome = {k: request["config"][k] for k in search._GENOME_KEYS}
    monkeypatch.setattr(search, "sample_genome", lambda h, rng: deepcopy(genome))
    # Keep the no-refit monkeypatch effective for this main-thread cache check.
    binding = deepcopy(binding)
    binding.pop("i6n_evaluator_command")
    binding.pop("i6n_evaluator_timeout")
    binding["i6n_evaluator"] = "tools.i6n_historical_evaluator:evaluate"
    plugin = Plugin()
    state = plugin.create_shared_population(1, binding)
    forwarded = plugin.evaluate_single_genome(state["population"][0], 0, binding)
    assert forwarded["metrics"] == receipt and forwarded["fitness"] == score
    result_path, status_path = root / "RESULT.json", root / "STATUS.json"
    result = json.loads(result_path.read_text())
    status = json.loads(status_path.read_text())
    input_path = root / "train.input.json"
    original = input_path.read_bytes()
    document = json.loads(original)
    document["forecasts"][0]["prediction"] += .1
    result["artifacts"][input_path.name] = write_json(input_path, document)
    status["result_sha256"] = write_json(result_path, result)
    write_json(status_path, status)
    with pytest.raises(ValueError, match="metric input differs"):
        evaluator.evaluate(request)
    input_path.write_bytes(original)
    result["artifacts"][input_path.name] = evaluator.cell.file_hash(input_path)
    status["result_sha256"] = write_json(result_path, result)
    write_json(status_path, status)
    (root / "model.keras").write_bytes(b"broken")
    with pytest.raises(ValueError, match="digest"):
        evaluator.evaluate(request)


def test_failure_terminal_and_context_pins(prepared, monkeypatch):
    _, config_path, _, request = prepared
    calls = []
    def fail(*args):
        calls.append(1)
        raise RuntimeError("deliberate failure")
    monkeypatch.setattr(evaluator, "fit_candidate", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        evaluator.evaluate(request)
    with pytest.raises(ValueError, match="retry"):
        evaluator.evaluate(request)
    assert calls == [1]
    config_path.write_text(config_path.read_text()+" ")
    with pytest.raises(ValueError, match="digest"):
        evaluator.evaluate(request)
