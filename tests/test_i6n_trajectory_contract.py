"""Lane A trajectory contracts on invented data; no model fitting here.

Requirement -> test map lives in docs/i6n/swarm/A_RETURN.md.
"""

from dataclasses import replace
from datetime import timedelta
import json

import numpy as np
import pytest

from tests._i6n_trajectory_fixture import build, genome, week_open, write_json
from tools import i6n_historical_cell as cell
from tools import i6n_historical_evaluator as evaluator
from tools import i6n_search as search
from tools.i6m_data import file_hash


def traj(base, h, dx, dy=None, multiplier=1):
    effective = search.effective_config(genome(search, h, dx=dx, dy=dy, multiplier=multiplier,
                                               core=base["model"]))
    return evaluator.candidate_config(base, effective), effective


def repin(cfg):
    cfg["input"]["sha256"] = file_hash(cfg["input"]["path"])
    path = cell.Path(cfg["source_clock"]["path"])
    clock = json.loads(path.read_text())
    clock["input_sha256"] = cfg["input"]["sha256"]
    cfg["source_clock"]["sha256"] = write_json(path, clock)


def at(week, hour, dst_from=None):
    return week_open(week, dst_from) + timedelta(hours=hour)


def rows(part):
    return {t: m for t, m in zip(part.origins, part.mask)}


# --- independent dx/dy, exact derived grid, no truncation ---------------------------------

@pytest.mark.parametrize("h,dx,dy,m,samples,offsets", [
    (12, 3, 6, 1, 9, (6, 12)),
    (72, 36, 24, 1, 3, (24, 48, 72)),
    (72, 36, 24, 6, 13, (24, 48, 72)),
    (72, 1, 24, 6, 433, (24, 48, 72)),
    (12, 6, 3, 2, 5, (3, 6, 9, 12)),
])
def test_independent_dx_dy_map_to_exact_heads(tmp_path, h, dx, dy, m, samples, offsets):
    base = build(tmp_path, horizon=h, samples=3, spacing=1)
    cfg, effective = traj(base, h, dx, dy, m)
    assert (cfg["samples"], cfg["spacing_hours"], cfg["horizon_hours"]) == (samples, dx, h)
    assert cell.output_offsets(cfg) == offsets == tuple(effective["output_offsets"])
    assert cfg["model"]["n_horizons"] == len(offsets)
    assert cfg["training"]["head_weights"] == [1.] * len(offsets)
    cell.authenticated(cfg, trajectory=True)
    with pytest.raises(ValueError, match="one horizon"):
        cell.authenticated(cfg)


def test_endpoint_alias_is_the_same_cell(tmp_path):
    base = build(tmp_path, horizon=12, samples=3, spacing=1)
    endpoint, _ = traj(base, 12, 3)
    alias, _ = traj(base, 12, 3, dy=12)
    assert endpoint == alias and cell.output_offsets(endpoint) == (12,)


@pytest.mark.parametrize("h,dx,dy", [(12, 3, 5), (5, 1, 2), (72, 36, 7), (12, 5, 6), (72, 7, 24)])
def test_non_divisor_horizons_rejected(tmp_path, h, dx, dy):
    with pytest.raises(ValueError):
        search.effective_config(genome(search, h, dx=dx, dy=dy))


def test_forged_trajectory_fields_rejected(tmp_path):
    base = build(tmp_path, horizon=12, samples=3, spacing=1)
    _, effective = traj(base, 12, 3, 6)
    for change in ({"delta_y": 4}, {"output_offsets": [4, 8, 12]}, {"output_offsets": [12]},
                   {"output_mode": "endpoint"}):
        with pytest.raises(ValueError, match="effective"):
            evaluator.candidate_config(base, {**effective, **change})
    forged = json.loads(json.dumps(effective))
    forged["core"]["n_horizons"] = 1
    with pytest.raises(ValueError, match="effective"):
        evaluator.candidate_config(base, forged)


def test_trajectory_head_weights_must_be_equal(tmp_path):
    base = build(tmp_path, horizon=12, samples=3, spacing=1)
    cfg, _ = traj(base, 12, 3, 6)
    cfg["training"]["head_weights"] = [1., 2.]
    with pytest.raises(ValueError, match="equal"):
        cell.authenticated(cfg, trajectory=True)


# --- endpoint support identical; missing intermediate never filters endpoint ---------------

def test_endpoint_support_rows_and_naive_identical_and_missing_intermediate(tmp_path):
    base = build(tmp_path, horizon=12, samples=9, spacing=3, drop={(10, 50)})
    endpoint_cfg, _ = traj(base, 12, 3)
    trajectory_cfg, _ = traj(base, 12, 3, 6)
    endpoint = cell.prepare(endpoint_cfg)
    trajectory = cell.prepare(trajectory_cfg, trajectory=True)
    assert evaluator.support(trajectory, trajectory_cfg) == evaluator.support(endpoint, endpoint_cfg)
    for split in ("train", "inner"):
        a = evaluator.observations(endpoint, endpoint_cfg, split)
        b = evaluator.observations(trajectory, trajectory_cfg, split)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(endpoint["fit_indices"], trajectory["fit_indices"])
    assert endpoint["scaler"].sha256() == trajectory["scaler"].sha256()
    assert trajectory["target_center"][-1] == endpoint["target_center"][0]
    assert trajectory["target_scale"][-1] == endpoint["target_scale"][0]
    train = rows(trajectory["parts"].train)
    assert train[at(10, 44)] == (False, True)     # missing t+6, endpoint kept
    assert train[at(10, 38)] == (True, False)     # missing t+12, intermediate-only row
    assert train[at(10, 50)] == (False, False)    # missing origin
    x, y, mask, origins = cell.materialize(trajectory, trajectory["parts"].train, trajectory_cfg)
    assert mask.shape == y.shape == (len(origins), 2) and np.isfinite(y).all()
    assert mask[:, 1].sum() == evaluator.support(endpoint, endpoint_cfg)["counts"]["train"]
    assert mask[:, 0].sum() > mask[:, 1].sum() and (mask[:, 0] & ~mask[:, 1]).any()
    assert int(at(10, 38).timestamp()) in origins.tolist()
    assert np.all(y[~mask] == 0)


def test_label_lookup_is_exact_elapsed_hours(tmp_path):
    base = build(tmp_path, horizon=12, samples=9, spacing=3, drop={(10, 50)})
    cfg, _ = traj(base, 12, 3, 6)
    data = cell.prepare(cfg, trajectory=True)
    part = data["parts"].train
    forged = [list(m) for m in part.mask]
    forged[part.origins.index(at(10, 44))][0] = True   # claims the missing t+6 bar
    data["parts"] = replace(data["parts"], train=replace(part, mask=tuple(map(tuple, forged))))
    with pytest.raises(ValueError, match="exact"):
        cell.materialize(data, data["parts"].train, cfg)


def test_dst_shifted_weeks_use_exact_utc_elapsed_hours(tmp_path):
    base = build(tmp_path, horizon=12, samples=9, spacing=3, dst_from=100)
    cfg, _ = traj(base, 12, 3, 6)
    data = cell.prepare(cfg, trajectory=True)
    six = dict(base, horizon_hours=6)
    single = cell.prepare(six)
    twelve = cell.prepare(base)
    inner = data["parts"].inner
    assert inner.origins[0].hour != at(10, 0).hour  # shifted open is in the grid
    for name in ("train", "inner"):
        part = getattr(data["parts"], name)
        for t, targets in zip(part.origins, part.target_instants):
            assert targets == (t + timedelta(hours=6), t + timedelta(hours=12))
        assert [m[0] for m in part.mask] == [m[0] for m in getattr(single["parts"], name).mask]
        assert [m[1] for m in part.mask] == [m[0] for m in getattr(twelve["parts"], name).mask]
    close = week_open(207, 100) + timedelta(days=5)
    last = max(t for t, m in zip(inner.origins, inner.mask) if m[1])
    assert last + timedelta(hours=12) < close - timedelta(hours=1) <= last + timedelta(hours=13)


# --- heads without support -----------------------------------------------------------------

def test_head_without_train_support_is_excluded_not_zero_error(tmp_path):
    keep = lambda w, h: w != 10 or h % 12 == 0
    base = build(tmp_path, horizon=12, samples=9, spacing=3, keep=keep)
    cfg, _ = traj(base, 12, 3, 6)
    data = cell.prepare(cfg, trajectory=True)
    assert data["head_status"] == ["NO_TRAIN_SUPPORT", "TRAINED"]
    assert data["head_trainable"].tolist() == [False, True]
    raw_inner = np.asarray(data["parts"].inner.mask)
    assert raw_inner[:, 0].any()
    for part in (data["parts"].train, data["parts"].inner):
        _, y, mask, _ = cell.materialize(data, part, cfg)
        assert not mask[:, 0].any() and mask[:, 1].any() and np.isfinite(y).all()


def test_endpoint_without_train_support_refused(tmp_path):
    keep = lambda w, h: w != 10 or h <= 12
    base = build(tmp_path, horizon=12, samples=9, spacing=3, keep=keep)
    cfg, _ = traj(base, 12, 3, 6)
    with pytest.raises(ValueError, match="NO_ELIGIBLE_ENDPOINT_TRAIN"):
        cell.prepare(cfg, trajectory=True)


# --- future perturbation -------------------------------------------------------------------

def test_future_perturbation_changes_only_the_affected_labels(tmp_path):
    base = build(tmp_path, horizon=12, samples=9, spacing=3)
    cfg, _ = traj(base, 12, 3, 6)
    data = cell.prepare(cfg, trajectory=True)
    before = {n: cell.materialize(data, getattr(data["parts"], n), cfg) for n in ("train", "inner")}
    with np.load(cfg["input"]["path"]) as a:
        ts, x, price = a["ts"], a["x"].copy(), a["price"].copy()
    late = ts >= data["parts"].fit_cutoff.timestamp()
    x[late] += 1000
    price[late] *= 2
    target = int(at(10, 50).timestamp())
    price[ts == target] *= 1.01
    np.savez(cfg["input"]["path"], ts=ts, x=x, price=price)
    repin(cfg)
    changed = cell.prepare(cfg, trajectory=True)
    assert changed["scaler"].sha256() == data["scaler"].sha256()
    after = {n: cell.materialize(changed, getattr(changed["parts"], n), cfg) for n in ("train", "inner")}
    for n in ("train", "inner"):
        np.testing.assert_array_equal(before[n][0], after[n][0])
        np.testing.assert_array_equal(before[n][2], after[n][2])
    # TRAIN-only scales legitimately refit; raw inner log-returns are untouched.
    raw = lambda d, y: y * d["target_scale"] + d["target_center"]
    ib, ia = before["inner"], after["inner"]
    np.testing.assert_allclose(raw(data, ib[1])[ib[2]], raw(changed, ia[1])[ia[2]], rtol=0, atol=1e-7)
    xb, yb, mb, ob = before["train"]
    ya = after["train"][1]
    origins = ob.tolist()
    expected = np.zeros_like(mb)
    expected[origins.index(target - 6 * 3600), 0] = True
    expected[origins.index(target - 12 * 3600), 1] = True
    expected[origins.index(target), :] = mb[origins.index(target)]
    # Center/scale were refit on TRAIN labels, so compare the raw log-returns.
    raw_before = yb * data["target_scale"] + data["target_center"]
    raw_after = ya * changed["target_scale"] + changed["target_center"]
    differs = ~np.isclose(raw_before, raw_after, rtol=0, atol=1e-7) & mb
    np.testing.assert_array_equal(differs, expected & mb)


# --- plan stopping, attempt lineage --------------------------------------------------------

def test_freeze_requires_plan_section_2_stopping(tmp_path):
    base = build(tmp_path, horizon=12, samples=9, spacing=3)
    for key, value in (("patience", 1), ("max_epochs", 10), ("min_delta", 0.), ("seed", 1)):
        cfg = json.loads(json.dumps(base))
        cfg["training"][key] = value
        path = tmp_path / f"bad-{key}.json"
        write_json(path, cfg)
        with pytest.raises(ValueError, match="PLAN_STOPPING"):
            evaluator.freeze(path, tmp_path / f"out-{key}", experiment="unit", fold="F1")
        assert not (tmp_path / f"out-{key}").exists()


@pytest.fixture
def frozen12(tmp_path):
    base = build(tmp_path, horizon=12, samples=9, spacing=3)
    path = tmp_path / "base.json"
    write_json(path, base)
    binding = evaluator.freeze(path, tmp_path / "bridge", experiment="unit-exp", fold="2023Q1")
    _, effective = traj(base, 12, 3, 6)
    request = dict(config=effective, generation=0, context=binding["i6n_evaluation_context"],
                   support=binding["i6n_support"])
    return base, binding, request


def test_binding_context_carries_experiment_and_fold(frozen12):
    _, binding, request = frozen12
    ctx = binding["i6n_evaluation_context"]
    assert ctx["schema"] == "i6n.historical_evaluator.v2"
    assert (ctx["experiment"], ctx["fold"]) == ("unit-exp", "2023Q1")


def test_attempt_lineage_linked_counted_and_bounded(frozen12, monkeypatch):
    _, _, request = frozen12
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("external worker loss")
    monkeypatch.setattr(evaluator, "fit_candidate", fail)
    with pytest.raises(RuntimeError):
        evaluator.evaluate(request)
    with pytest.raises(ValueError, match="retry"):
        evaluator.evaluate(request)
    root = evaluator.candidate_root(request)
    first = file_hash(root / "STATUS.json")
    for bad in ({"index": 2, "parent_status_sha256": "0" * 64, "fencing_token": "t"},
                {"index": 2, "parent_status_sha256": None, "fencing_token": "t"},
                {"index": 4, "parent_status_sha256": first, "fencing_token": "t"},
                {"index": 2, "parent_status_sha256": first, "fencing_token": "t", "extra": 1}):
        with pytest.raises(ValueError, match="attempt|parent"):
            evaluator.evaluate(dict(request, attempt=bad))
    assert len(calls) == 1
    second = dict(request, attempt={"index": 2, "parent_status_sha256": first, "fencing_token": "lease-7"})
    with pytest.raises(RuntimeError):
        evaluator.evaluate(second)
    status = json.loads((root.parent / (root.name + ".attempt2") / "STATUS.json").read_text())
    assert status["state"] == "FAILED"
    assert status["attempt"] == {"index": 2, "parent_status_sha256": first, "fencing_token": "lease-7"}
    with pytest.raises(ValueError, match="retry"):
        evaluator.evaluate(second)
    assert len(calls) == 2
    second_digest = file_hash(root.parent / (root.name + ".attempt2") / "STATUS.json")
    with pytest.raises(ValueError, match="attempt"):
        evaluator.evaluate(dict(request, attempt={"index": 3, "parent_status_sha256": first,
                                                  "fencing_token": "x"}))
    third = dict(request, attempt={"index": 3, "parent_status_sha256": second_digest,
                                   "fencing_token": "lease-8"})
    with pytest.raises(RuntimeError):
        evaluator.evaluate(third)
    assert len(calls) == 3
    third_digest = file_hash(root.parent / (root.name + ".attempt3") / "STATUS.json")
    # The physical-attempt cap is three, including linked recoveries.
    with pytest.raises(ValueError, match="attempt"):
        evaluator.evaluate(dict(request, attempt={"index": 4, "parent_status_sha256": third_digest,
                                                  "fencing_token": "x"}))
    assert len(calls) == 3


def test_outer_week_never_reaches_stopping_or_fitness(tmp_path, monkeypatch):
    import tools.i6n_training as training
    base = build(tmp_path, horizon=12, samples=9, spacing=3)
    path = tmp_path / "base.json"
    write_json(path, base)
    binding = evaluator.freeze(path, tmp_path / "bridge", experiment="unit-exp",
                               fold="W2024-01-07", score_outer=True)
    _, effective = traj(base, 12, 3, 6)
    request = dict(config=effective, generation=0, context=binding["i6n_evaluation_context"],
                   support=binding["i6n_support"])
    cfg = evaluator.candidate_config(base, effective)
    data = cell.prepare(cfg, trajectory=True)
    expected = {n: cell.materialize(data, getattr(data["parts"], n), cfg) for n in ("train", "inner")}
    outer = data["parts"].evaluation.origins[0].timestamp()
    seen = []

    def capture(model, x_train, y_train, m_train, x_inner, y_inner, m_inner, config, progress=None):
        seen.append(1)
        for (x, y, m), name in (((x_train, y_train, m_train), "train"), ((x_inner, y_inner, m_inner), "inner")):
            np.testing.assert_array_equal(y, expected[name][1])
            np.testing.assert_array_equal(m, expected[name][2])
            assert expected[name][3].max() < outer
        raise RuntimeError("captured stopping inputs")
    monkeypatch.setattr(training, "fit_masked", capture)
    with pytest.raises(RuntimeError, match="captured"):
        evaluator.evaluate(request)
    assert seen == [1]
