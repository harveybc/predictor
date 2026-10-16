import copy
import json

import numpy as np
import pytest

from tools.i6dmo import resolution_batch as B


def fixture_result(spacing, score):
    return dict(state="COMPLETE", metrics={"VALIDATION": [dict(horizon=h, n=10,
                mae=(h/10000)*(1+score), naive_mae=h/10000, predicted_movement_mae=.001,
                pairs_sha256="same", relative_error=score) for h in B.LONG]})


def test_plan_order_and_bounds():
    c = B.cells()
    assert len(c) == 19
    assert [x["id"] for x in c[:4]] == ["short_ds2", "short_ds4", "long_ds2", "long_ds4"]
    assert all(x["slots"] == 13 for x in c[:4])
    assert sorted(set(x["slots"] for x in c[4:])) == [6, 9, 13, 18, 24]
    assert all(x["horizons"] == [72] and x["spacing"] == "SELECTED_LONG" for x in c[4:])


def test_installed_environment_retains_cuda_library_paths_and_rejects_injection():
    text = B.role_environment(dict(environment=dict(LD_LIBRARY_PATH="/example/cuda/lib",
                                                    XLA_FLAGS="--xla_gpu_cuda_data_dir=/example/nvcc")))
    assert 'Environment="LD_LIBRARY_PATH=/example/cuda/lib"' in text
    assert 'Environment="XLA_FLAGS=--xla_gpu_cuda_data_dir=/example/nvcc"' in text
    with pytest.raises(ValueError, match="ENVIRONMENT_INVALID"):
        B.role_environment(dict(environment=dict(LD_LIBRARY_PATH="x\nExecStart=bad")))


def test_choose_spacing_negative_is_better_tie_retains_ds2():
    r = {f"long_ds{s}": fixture_result(s, v) for s, v in ((2, .1), (4, -.01))}
    assert B.selected_spacing(r) == 4
    r["long_ds4"] = fixture_result(4, .1-1e-9)
    assert B.selected_spacing(r) == 2
    r["long_ds4"]["state"] = "FAILED"
    assert B.selected_spacing(r) is None


@pytest.mark.parametrize("key,value", [("n", 11), ("naive_mae", .01), ("pairs_sha256", "other")])
def test_selection_refuses_unpaired(key, value):
    r = {f"long_ds{s}": fixture_result(s, .1) for s in (2, 4)}
    r["long_ds4"]["metrics"]["VALIDATION"][0][key] = value
    with pytest.raises(ValueError, match="UNPAIRED"):
        B.selected_spacing(r)


def test_no_premature_single_head_dispatch():
    spec = dict(cells=B.cells())
    assert B.next_cell(spec, {})["id"] == "short_ds2"
    r = {x["id"]: dict(state="FAILED") for x in spec["cells"][:4]}
    assert B.next_cell(spec, r) is None
    r.update({f"long_ds{s}": fixture_result(s, .1) for s in (2, 4)})
    assert B.next_cell(spec, r)["id"] == "h72_n6_linear"


def test_naive_each_horizon_original_scale_exact_mask():
    a = dict(origin_ts=np.array([1, 2, 3]), origin_price=np.array([1., 2., 3.]),
             target_price=np.array([[2., 5.], [4., np.nan], [6., 9.]]),
             mask=np.array([[1, 1], [1, 0], [1, 1]], bool))
    r = B.metrics(a, np.array([[2., 4.], [4., 0.], [6., 8.]]), [2, 72])
    assert r[0]["naive_mae"] == 2
    assert r[0]["relative_error"] == -1
    assert r[1]["naive_mae"] == 5
    assert r[1]["mae"] == 1
    assert r[1]["relative_error"] == -.8
    assert r[0]["n"] == 3 and r[1]["n"] == 2


def test_terminal_adoption_and_corruption(tmp_path):
    cell = B.cells()[0]
    folder = tmp_path / "results" / cell["id"]
    folder.mkdir(parents=True)
    for name in ("CONFIG.json", "TRAINING.json", "model.weights.h5", "FIT.npz", "INNER.npz", "VALIDATION.npz"):
        (folder / name).write_bytes(b"original")
    d = dict(cell=cell, batch_id="unit-test", state="COMPLETE",
             population=dict(manifest_sha256="population"),
             files={p.name: B.file_sha256(p) for p in folder.iterdir()})
    d["sha256"] = B.digest(d)
    B.put(folder / "RESULT.json", d)
    spec = dict(batch_id="unit-test", population_digest="population")
    assert B.terminal(tmp_path, cell, spec) == d
    (folder / "FIT.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="TERMINAL_FILE"):
        B.terminal(tmp_path, cell, spec)
    d["state"] = "FAILED"
    B.put(folder / "RESULT.json", d)
    with pytest.raises(ValueError, match="TERMINAL_IDENTITY"):
        B.terminal(tmp_path, cell, spec)


def test_metadata_only_terminal_is_not_a_fit(tmp_path):
    cell = B.cells()[0]
    d = dict(cell=cell, batch_id="unit-test", state="COMPLETE", files={},
             population=dict(manifest_sha256="population"))
    d["sha256"] = B.digest(d)
    B.put(tmp_path/"results"/cell["id"]/"RESULT.json", d)
    with pytest.raises(ValueError, match="TERMINAL_ARTIFACTS"):
        B.terminal(tmp_path, cell, dict(batch_id="unit-test", population_digest="population"))


def test_close_repairs_status_after_interruption(tmp_path, monkeypatch):
    closed = dict(state="CLOSED", complete=19)
    B.put(tmp_path/"CLOSE.json", closed)
    B.put(tmp_path/"STATUS.json", dict(state="RUNNING"))
    monkeypatch.setattr(B, "verify_code", lambda spec: None)
    stops = []
    monkeypatch.setattr(B, "stop_timers", lambda spec: stops.append(True))
    assert B.tick(dict(root=str(tmp_path))) == closed
    assert B.read(tmp_path/"STATUS.json") == closed
    assert stops == [True]


def test_code_and_plan_changes_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "REPO", tmp_path)
    (tmp_path / "code.py").write_text("pass")
    s = dict(code_files={"code.py": B.file_sha256(tmp_path / "code.py")}, cells=B.cells(),
             seed=B.SEED, training=B.TRAINING)
    B.verify_code(s)
    altered = copy.deepcopy(s)
    altered["training"]["patience"] = 3
    with pytest.raises(ValueError, match="PLAN_CHANGED"):
        B.verify_code(altered)
    (tmp_path / "code.py").write_text("other")
    with pytest.raises(ValueError, match="CODE_CHANGED"):
        B.verify_code(s)


def test_outbox_retry_does_not_fit(tmp_path, monkeypatch):
    from tools.i6dmo.outbox import Outbox
    r = fixture_result(2, .1)
    r.update(cell=B.cells()[0], actual_spacing=2, population={}, parameters=4882,
             training_seconds=1., epochs=2, best_epoch=1, sha256="a"*64)
    s = dict(batch_id="test-batch", code_commit="b"*40)
    report = B.report(s, r)
    box = Outbox(tmp_path / "outbox.sqlite3")
    try:
        assert box.enqueue(report, source_unit="test") == box.enqueue(report, source_unit="test")
        assert box.summary()["expected"] == 1
        class Down:
            def post_report(self, report):
                raise ConnectionError("unavailable")
        assert box.pump(Down())["pending"] == 1
        assert box.summary()["expected"] == 1
    finally:
        box.close()
