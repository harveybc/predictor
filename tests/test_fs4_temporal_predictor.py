"""R0 modular temporal predictor for the phase-4 weekly comparison (plan section 4, FS4-02/03/05).

Written before the implementation. The spec/identity tests are pure python and run on the
coordinator; the TensorFlow fits run only where FS4_TF_TESTS=1 (a worker under crispdm-run).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from tools import fs4_temporal_predictor as P

TF = os.environ.get("FS4_TF_TESTS") == "1"
needs_tf = pytest.mark.skipif(not TF, reason="TensorFlow fits run on a worker with FS4_TF_TESTS=1")


# ----------------------------------------------------------------------------- pure python
def test_spec_is_fixed_and_budget_is_identical_for_every_subset():
    spec = P.PredictorSpec()
    assert spec.window == 24 and spec.latent_steps == 6
    assert spec.sha256() == P.PredictorSpec().sha256()
    a = P.input_identity(spec, ["f_b", "f_a", "f_c"], "RAW", None)
    b = P.input_identity(spec, ["f_c", "f_a", "f_b"], "RAW", None)
    assert a == b                                                        # FS4-02 column permutation
    assert a != P.input_identity(spec, ["f_a", "f_b"], "RAW", None)
    assert P.budget_sha256(spec) == hashlib.sha256(P.canonical(spec.budget()).encode()).hexdigest()
    assert "n_features" not in spec.budget() and "epochs" in P.canonical(spec.budget())


def test_input_modes_and_encoder_identity_bind_the_latent_source():
    spec = P.PredictorSpec()
    assert P.INPUT_MODES == ("RAW", "TRAINED_ENCODER", "RANDOM_ENCODER")
    with pytest.raises(P.Refusal, match="ENCODER_IDENTITY_REQUIRED"):
        P.input_identity(spec, ["f_a"], "TRAINED_ENCODER", None)
    with pytest.raises(P.Refusal, match="UNKNOWN_INPUT_MODE"):
        P.input_identity(spec, ["f_a"], "MLP", None)
    assert P.input_identity(spec, ["f_a"], "RANDOM_ENCODER", "e" * 64) != P.input_identity(spec, ["f_a"], "TRAINED_ENCODER", "e" * 64)


def test_windows_use_only_preceding_rows_and_drop_short_histories():
    X = np.arange(40, dtype="float64").reshape(20, 2)
    win, kept = P.make_windows(X, np.array([3, 5, 19]), window=4)
    assert kept.tolist() == [3, 5, 19] and win.shape == (3, 4, 2)
    assert win[0, -1, 0] == X[3, 0] and win[0, 0, 0] == X[0, 0]            # ends at the origin, earlier rows before
    win2, kept2 = P.make_windows(X, np.array([1, 3]), window=4)
    assert kept2.tolist() == [3] and win2.shape == (1, 4, 2)                 # origin 1 has no 4-row history
    assert np.all(np.diff(win[2, :, 0]) > 0)                                 # time order preserved


def test_future_rows_never_enter_a_window():
    X = np.random.default_rng(0).normal(size=(50, 3))
    win, _ = P.make_windows(X, np.array([10, 20]), window=5)
    Y = X.copy()
    Y[21:] = 1e9                                                             # perturb everything after the last origin
    win2, _ = P.make_windows(Y, np.array([10, 20]), window=5)
    assert np.array_equal(win, win2)


def test_standardiser_fits_on_fit_rows_only():
    X = np.vstack([np.zeros((10, 2)), np.full((5, 2), 100.0)])
    st = P.Standardiser.fit(X[:10])
    Z = st.apply(X)
    assert np.allclose(Z[:10], 0.0) and Z[10, 0] > 0
    assert st.sha256() == P.Standardiser.fit(X[:10]).sha256()


def test_raw_lag3_uses_elapsed_hours_and_keeps_raw_support():
    ts = np.array([3600 * i for i in range(50) if i != 30], dtype="int64")
    X = ts[:, None].astype("float64") / 3600
    st = P.Standardiser.fit(X[:28])
    origins = np.array([23, 24, 30, 32], dtype="int64")
    raw, raw_kept = P._inputs_for(P.PredictorSpec(), None, X, st, origins, ts, 0)
    lag, lag_kept = P.raw_lag3_windows(P.PredictorSpec(), X, st, origins, ts, 0)
    assert np.array_equal(raw_kept, lag_kept)
    assert lag.shape == raw.shape
    assert lag[0, :3, 1].tolist() == [0, 0, 0]  # no pre-fit history
    assert lag[0, -1, 0] == pytest.approx((20 - st.mean[0]) / st.sd[0])
    assert lag[2, -1, 0] == pytest.approx((28 - st.mean[0]) / st.sd[0])  # hour 31 minus 3 elapsed hours
    assert lag[3, -1, 1] == 0  # hour 33 minus 3 is the missing grid slot
    changed = X.copy()
    changed[ts > ts[32] - 3 * 3600] = 1e8
    lag_changed, _ = P.raw_lag3_windows(P.PredictorSpec(), changed, st, [32], ts, 0)
    assert np.array_equal(lag[3:4], lag_changed)


def test_raw_lag3_refuses_missing_grid_or_shifted_support():
    X = np.arange(40, dtype="float64")[:, None]
    st = P.Standardiser.fit(X)
    with pytest.raises(P.Refusal, match="HOURLY_TIMESTAMPS_REQUIRED"):
        P.raw_lag3_windows(P.PredictorSpec(), X, st, [30], None, 0)
    with pytest.raises(P.Refusal, match="RAW_LAG3_SUPPORT_MISMATCH"):
        P.require_same_support([30, 31], [31, 30])


def test_raw_lag3_diagnostic_identity_is_versioned_and_distinct():
    spec = P.PredictorSpec()
    raw = P.input_identity(spec, ["a", "b"], "RAW", None)
    lag = P.raw_lag3_identity(spec, ["b", "a"], "f" * 64)
    assert lag == P.raw_lag3_identity(spec, ["a", "b"], "f" * 64)
    assert lag != raw
    assert lag != P.raw_lag3_identity(spec, ["a", "b"], "0" * 64)


def _diagnostic_fixture():
    from types import SimpleNamespace
    from tools import fs4_weekly_wrapper as W

    y = np.linspace(-0.2, 0.2, 5)
    store = SimpleNamespace(population="EURUSD", digests={"train": "abc"}, row_id_offset=0,
                            targets={"return_h1": y}, col={"a": 0}, record_ids=np.array(["0", "1", "2", "3", "4"]),
                            range_idx=lambda start, end: np.arange(5))
    task = {"schema": W.TASK_SCHEMA, "task_id": "task1", "plan_sha256": "plan1", "population_id": "EURUSD",
            "split": "validation", "input_mode": "RAW", "target_id": "return_h1", "horizon_hours": 1,
            "set_id": "set1", "members": ["a"], "seed": W.SEED, "predictor_spec_sha256": P.PredictorSpec().sha256(),
            "week": {"start": "2024-01-01T00:00:00Z", "end": "2024-01-08T00:00:00Z"}}
    reference = {"schema": W.RESULT_SCHEMA, "task_id": "task1", "plan_sha256": "plan1", "population_id": "EURUSD",
                 "disposition": "COMPLETED", "input_mode": "RAW", "split": "validation", "target_id": "return_h1",
                 "seed": W.SEED, "predictor_spec_sha256": P.PredictorSpec().sha256(),
                 "horizon_hours": 1, "set_id": "set1", "members": ["a"], "week_start": task["week"]["start"],
                 "week_end": task["week"]["end"], "n_scored": 5, "rows_sha256": W.rows_digest(store.record_ids),
                 "naive": {"rows_sha256": W.rows_digest(store.record_ids)},
                 "metrics": {"mae": 0.3, "mse": 0.4, "naive_mae": float(np.mean(np.abs(y))),
                             "naive_mse": float(np.mean(y ** 2))},
                 "input_sha256": W.digest({"files": store.digests, "members": ["a"], "target": "return_h1",
                                           "input_mode": "RAW", "row_id_offset": 0})}
    reference["result_sha256"] = W.digest(reference)
    return task, reference, store


def test_raw_lag3_reference_rejects_wrong_rows_target_naive_or_source():
    task, reference, store = _diagnostic_fixture()
    assert P.validate_raw_lag3_reference(task, reference, store).tolist() == list(range(5))
    for field, value in (("rows_sha256", "0" * 64), ("target_id", "other"),
                         ("seed", 99), ("predictor_spec_sha256", "2" * 64),
                         ("input_sha256", "1" * 64)):
        changed = {**reference, field: value}
        changed["result_sha256"] = P.digest({k: v for k, v in changed.items() if k != "result_sha256"})
        with pytest.raises(P.Refusal, match="RAW_LAG3_REFERENCE_MISMATCH"):
            P.validate_raw_lag3_reference(task, changed, store)
    changed = {**reference, "metrics": {**reference["metrics"], "naive_mae": 999.0}}
    changed["result_sha256"] = P.digest({k: v for k, v in changed.items() if k != "result_sha256"})
    with pytest.raises(P.Refusal, match="RAW_LAG3_REFERENCE_MISMATCH"):
        P.validate_raw_lag3_reference(task, changed, store)
    with pytest.raises(P.Refusal, match="RAW_LAG3_REFERENCE_MISMATCH"):
        P.validate_raw_lag3_reference(task, reference | {"result_sha256": "0" * 64}, store)


def test_raw_lag3_diagnostic_refuses_score_support_before_training(monkeypatch):
    from tools import fs4_weekly_wrapper as W

    task, reference, store = _diagnostic_fixture()
    task["week"]["fit_start"] = "2024-01-01T00:00:00Z"
    store.X = np.arange(40, dtype="float64")[:, None]
    store.ts = np.arange(40, dtype="int64") * 3600
    store.targets["return_h1"] = np.r_[np.full(5, 0.1), np.zeros(35)]
    store.record_ids = np.arange(40).astype(str)
    store.range_idx = lambda start, end: np.arange(5)
    reference["metrics"]["naive_mae"] = float(np.mean(np.abs(store.targets["return_h1"][:5])))
    reference["metrics"]["naive_mse"] = float(np.mean(store.targets["return_h1"][:5] ** 2))
    reference["result_sha256"] = P.digest({k: v for k, v in reference.items() if k != "result_sha256"})
    monkeypatch.setattr(W, "run_task", lambda *args, **kw: pytest.fail("training must not run"))
    with pytest.raises(P.Refusal, match="RAW_LAG3_SUPPORT_MISMATCH"):
        P.run_raw_lag3_diagnostic(task, reference, store)


def test_raw_lag3_receipt_is_distinct_and_rejects_postfit_row_drift(monkeypatch):
    from tools import fs4_weekly_wrapper as W

    task, reference, store = _diagnostic_fixture()
    task["week"]["fit_start"] = "1970-01-01T00:00:00Z"
    store.X = np.arange(50, dtype="float64")[:, None]
    store.ts = np.arange(50, dtype="int64") * 3600
    store.targets["return_h1"] = np.linspace(-0.1, 0.1, 50)
    store.record_ids = np.arange(50).astype(str)
    store.range_idx = lambda start, end: np.arange(30, 35)
    scored = store.range_idx(None, None)
    reference["rows_sha256"] = W.rows_digest(store.record_ids[scored])
    reference["naive"]["rows_sha256"] = reference["rows_sha256"]
    reference["metrics"]["naive_mae"] = float(np.mean(np.abs(store.targets["return_h1"][scored])))
    reference["metrics"]["naive_mse"] = float(np.mean(store.targets["return_h1"][scored] ** 2))
    reference.update(fit_population_digest="fit", inner_population_digest="inner", fit_rows=30, inner_rows=5,
                     standardiser_sha256="st", seed=0)
    reference["result_sha256"] = P.digest({k: v for k, v in reference.items() if k != "result_sha256"})
    fitted = {**reference, "metrics": {**reference["metrics"], "mae": 0.3, "mse": 0.4},
              "model_sha256": "m", "cost": {"fit_seconds": 1.0}}
    monkeypatch.setattr(W, "run_task", lambda *args, **kw: fitted)
    record = P.run_raw_lag3_diagnostic(task, reference, store)
    assert record["schema"] == P.RAW_LAG3_SCHEMA and record["input_mode"] == "RAW_LAG3_DIAGNOSTIC"
    assert record["identity"] != P.input_identity(P.PredictorSpec(), ["a"], "RAW", None)
    assert record["raw_reference_sha256"] == reference["result_sha256"]
    assert record["result_sha256"] == P.digest({k: v for k, v in record.items() if k != "result_sha256"})
    monkeypatch.setattr(W, "run_task", lambda *args, **kw: {**fitted, "rows_sha256": "0" * 64})
    with pytest.raises(P.Refusal, match="RAW_LAG3_SUPPORT_MISMATCH"):
        P.run_raw_lag3_diagnostic(task, reference, store)


# ----------------------------------------------------------------------------- tensorflow
def _synthetic(n=600, f=3, window=24, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = 0.6 * np.roll(X[:, 0], 1) - 0.3 * np.roll(X[:, 1], 2) + 0.05 * rng.normal(size=n)
    return X, y


@needs_tf
def test_raw_lag3_fits_real_cpu_model_without_changing_target_rows():
    X, y = _synthetic(n=360, f=2)
    ts = np.arange(360, dtype="int64") * 3600
    spec = P.PredictorSpec(max_epochs=1, batch_size=32)
    rep = P.fit_named(spec, X, ["a", "b"], y, np.arange(23, 230), np.arange(230, 280),
                      input_mode="RAW", encoder=None, seed=0, timestamps=ts, min_timestamp=0,
                      raw_lag_hours=3)
    assert rep.fit_windows == 207 and rep.inner_windows == 50
    assert rep.raw_lag_hours == 3 and rep.input_identity != P.input_identity(spec, ["a", "b"], "RAW", None)
    scored = np.arange(280, 300)
    pred = rep.predict(X, scored)
    changed = X.copy()
    changed[297:] = 1e6
    assert np.array_equal(pred, rep.predict(changed, scored))


@needs_tf
def test_architecture_keeps_time_until_the_head_and_fuses_channels():
    spec = P.PredictorSpec(max_epochs=1)
    model = P.build_predictor(spec, n_features=3, input_mode="RAW", latent_dim=None)
    names = [type(l).__name__ for l in model.layers]
    assert "Flatten" not in names and "GlobalAveragePooling1D" not in names and "GlobalMaxPooling1D" not in names
    core = model.get_layer(P.CORE_OUTPUT_NAME)
    assert tuple(core.output.shape[1:]) == (spec.latent_steps, spec.core_filters)   # rank-3 until the head
    assert model.get_layer(P.HEAD_INPUT_NAME).output.shape[1:] == (spec.core_filters,)
    stem = model.get_layer("branch_stem")
    assert stem.groups == 3 and stem.fpg == spec.branch_filters                           # one causal branch per feature (grouped)
    assert model.get_layer("branch_down2").groups == 3 and model.get_layer("channel_fusion").filters == spec.fuse_filters
    assert all(getattr(l, "padding", "causal") == "causal" for l in model.layers if type(l).__name__ == "Conv1D")
    other = P.build_predictor(spec, n_features=7, input_mode="RAW", latent_dim=None)
    assert P.architecture_sha256(model) == P.architecture_sha256(other)          # identical architecture family
    assert P.count_params(model) != P.count_params(other)


@needs_tf
def test_fit_is_deterministic_and_future_perturbation_leaves_weights_unchanged(tmp_path):
    X, y = _synthetic()
    spec = P.PredictorSpec(max_epochs=3, batch_size=64, patience=2)
    fit_idx = np.arange(24, 400)
    inner_idx = np.arange(400, 480)
    rep1 = P.fit_predictor(spec, X, y, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=0)
    rep2 = P.fit_predictor(spec, X, y, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=0)
    assert rep1.weights_sha256 == rep2.weights_sha256
    Xp, yp = X.copy(), y.copy()
    Xp[480:] = 1e6
    yp[480:] = -1e6
    rep3 = P.fit_predictor(spec, Xp, yp, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=0)
    assert rep3.weights_sha256 == rep1.weights_sha256                                   # FS4-03 / BW02
    assert rep1.epochs_run <= spec.max_epochs and rep1.updates > 0 and rep1.best_epoch >= 1
    assert rep1.budget_sha256 == P.budget_sha256(spec)
    pred = P.predict(rep1, X, np.arange(480, 600))
    assert pred.shape == (120,) and np.all(np.isfinite(pred))
    rep_seed = P.fit_predictor(spec, X, y, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=1)
    assert rep_seed.weights_sha256 != rep1.weights_sha256                               # FS4-04


@needs_tf
def test_column_permutation_gives_identical_fit_and_budget_is_the_same_for_any_subset():
    X, y = _synthetic(f=4)
    spec = P.PredictorSpec(max_epochs=2, batch_size=64)
    fit_idx, inner_idx = np.arange(24, 400), np.arange(400, 480)
    names = ["f_c", "f_a", "f_d", "f_b"]
    rep = P.fit_named(spec, X, names, y, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=0)
    perm = [3, 1, 0, 2]
    rep_p = P.fit_named(spec, X[:, perm], [names[i] for i in perm], y, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=0)
    assert rep.input_identity == rep_p.input_identity and rep.weights_sha256 == rep_p.weights_sha256
    small = P.fit_named(spec, X[:, :2], names[:2], y, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=0)
    assert small.budget_sha256 == rep.budget_sha256 and small.architecture_sha256 == rep.architecture_sha256


EXTRACTOR = os.environ.get("FS4_EXTRACTOR_CODE")
needs_runner = pytest.mark.skipif(not (TF and EXTRACTOR), reason="needs FS4_TF_TESTS=1 and FS4_EXTRACTOR_CODE (pinned feature-extractor checkout)")
IDENTITY = "synthetic-train:v1"
ROLE = "synthetic_train"


def _ts(y, m, d):
    import datetime as dt
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())


@pytest.fixture(scope="module")
def runner_root(tmp_path_factory):
    """Retained terminals of the REAL phase-4 runner (feature-extractor app/fs4_task_runner.py) on a tiny corpus."""
    import io
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    P.load_extractor(EXTRACTOR)
    from app import fs4_extractibility as X
    from app import fs4_task_runner as R
    from app import univariate_temporal as U

    d = tmp_path_factory.mktemp("runner")
    rng = np.random.default_rng(0)
    ts = np.arange(_ts(2018, 1, 1), _ts(2020, 3, 1), 3600, dtype=np.int64)
    ts = ts[((ts // 86400 + 3) % 7) != 6]
    n = ts.size
    cols = {}
    for name in ("feat_a", "feat_b"):
        e = rng.normal(size=n)
        v = np.zeros(n)
        for t in range(1, n):
            v[t] = 0.9 * v[t - 1] + e[t]
        v[rng.random(n) < 0.05] = np.nan
        cols[name] = v
    path = str(d / "train.parquet")
    pq.write_table(pa.table({"t_decision_utc": pa.array(ts * 10 ** 9, pa.timestamp("ns", tz="UTC")),
                             "row_id": np.arange(n, dtype=np.int64), **cols}), path)
    reg = d / "registry.json"
    reg.write_text(json.dumps({IDENTITY: {"population_id": "SYN", "bar_seconds": 3600, "files": {ROLE: U.sha256_file(path)}}}))
    out = d / "out"
    for feat in cols:
        for arm in ("RANDOM_ENCODER", "TRAINED_ENCODER"):
            payload = {"schema": R.TASK_SCHEMA, "population_id": "SYN", "identity": IDENTITY, "feature_id": feat,
                       "fold_id": "inner_2019", "arm": arm, "seed": 0}
            claim = {**payload, "task_id": X.task_digest(payload), "attempt": 1, "lease_until": 0}
            buf = io.StringIO()
            rc = R.main(["--input", f"{ROLE}={path}", "--corpus-registry", str(reg), "--allow-cpu-training", "--max-epochs", "2",
                         "--patience", "1", "--max-fit-windows", "256", "--min-fit-windows", "64", "--min-scoring-windows", "16",
                         "--output-root", str(out)], stdin=io.StringIO(json.dumps(claim)), stdout=buf)
            assert rc == 0, buf.getvalue()
    return {"root": out, "ts": ts, "cols": cols, "spec": P.EncoderSpec(fold_id="inner_2019")}


def _bank(rr, arm, ts=None, cols=None, root=None, features=("feat_a", "feat_b")):
    ts = rr["ts"] if ts is None else ts
    X = np.column_stack([(rr["cols"] if cols is None else cols)[f] for f in features])
    return P.RunnerEncoderBank(rr["spec"], features, ts, X, arm=arm, population_id="SYN", identity=IDENTITY,
                               results_root=rr["root"] if root is None else root, code_dir=EXTRACTOR)


@needs_runner
def test_bank_loads_real_runner_files_and_verifies_their_digests(runner_root):
    rr = runner_root
    trained, rnd = _bank(rr, "TRAINED_ENCODER"), _bank(rr, "RANDOM_ENCODER")
    idx = np.arange(5000, 5100)
    zt, kept = trained.latents(idx)
    zr, _ = rnd.latents(idx)
    assert zt.shape == (100, 6, 16) and kept.tolist() == idx.tolist() and np.all(np.isfinite(zt))      # (B, 6, F*D): time preserved
    assert not np.allclose(zt, zr) and trained.weights_sha256 != rnd.weights_sha256
    assert trained.optimizer_steps == 0 and rnd.optimizer_steps == 0 and trained.n_features == 2 and trained.latent_dim == 8
    short, kshort = trained.latents(np.array([3, 5000]))                      # an origin without a 24-hour history is dropped
    assert kshort.tolist() == [5000] and short.shape[0] == 1


@needs_runner
def test_bank_refuses_a_tampered_or_missing_runner_terminal(runner_root, tmp_path):
    import json
    import shutil

    rr = runner_root
    root = tmp_path / "copy"
    shutil.copytree(rr["root"], root)
    victim = sorted(root.glob("*/result.json"))
    rec = [json.loads(p.read_text()) for p in victim]
    target = next(p for p, r in zip(victim, rec) if r["arm"] == "TRAINED_ENCODER" and r["feature_id"] == "feat_a")
    r = json.loads(target.read_text())
    r["weights"]["chosen_weights_sha256"] = "0" * 64
    target.write_text(json.dumps(r))
    with pytest.raises(P.Refusal, match="ENCODER_IDENTITY_MISMATCH"):
        _bank(rr, "TRAINED_ENCODER", root=root)
    with pytest.raises(P.Refusal, match="RUNNER_RESULT_MISSING"):
        _bank(rr, "TRAINED_ENCODER", cols={**rr["cols"], "feat_missing": rr["cols"]["feat_a"]}, features=("feat_a", "feat_missing"))
    h5 = next(root.glob("*/chosen.weights.h5"))
    h5.write_bytes(h5.read_bytes() + b"x")
    with pytest.raises(P.Refusal, match="ENCODER_FILE_IDENTITY_MISMATCH|ENCODER_IDENTITY_MISMATCH"):
        for arm_bank in ("TRAINED_ENCODER",):
            _bank(rr, arm_bank, root=root, features=("feat_a",))
            _bank(rr, arm_bank, root=root, features=("feat_b",))


@needs_runner
def test_latents_never_see_rows_after_the_origin(runner_root):
    rr = runner_root
    base = _bank(rr, "TRAINED_ENCODER")
    idx = np.arange(6000, 6040)
    z0, _ = base.latents(idx)
    cols = {k: v.copy() for k, v in rr["cols"].items()}
    for k in cols:
        cols[k][6040:] = 1e6
    z1, _ = _bank(rr, "TRAINED_ENCODER", cols=cols).latents(idx)
    assert np.array_equal(z0, z1)                                             # FS4-03


@needs_runner
def test_encoder_arms_fit_with_a_frozen_bank_and_the_same_budget(runner_root):
    rr = runner_root
    X = np.column_stack([rr["cols"]["feat_a"], rr["cols"]["feat_b"]])
    y = np.nan_to_num(np.roll(X[:, 0], 1))
    spec = P.PredictorSpec(max_epochs=2, batch_size=64, patience=1)
    fit_idx, inner_idx = np.arange(100, 6000), np.arange(6000, 6600)
    reps = {}
    for arm in ("RANDOM_ENCODER", "TRAINED_ENCODER"):
        bank = _bank(rr, arm)
        reps[arm] = P.fit_predictor(spec, X, y, fit_idx, inner_idx, input_mode=arm, encoder=bank, seed=0, features=("feat_a", "feat_b"))
        assert reps[arm].encoder_sha256 == bank.weights_sha256 and bank.optimizer_steps == 0
        pred = P.predict(reps[arm], X, np.arange(6600, 6700))
        assert pred.shape == (100,) and np.all(np.isfinite(pred))
    raw = P.fit_predictor(spec, X, y, fit_idx, inner_idx, input_mode="RAW", encoder=None, seed=0)
    assert len({r.budget_sha256 for r in reps.values()} | {raw.budget_sha256}) == 1
    assert reps["RANDOM_ENCODER"].weights_sha256 != reps["TRAINED_ENCODER"].weights_sha256


@needs_tf
def test_each_feature_has_its_own_branch_and_branches_do_not_mix_before_fusion():
    keras = P._keras()
    spec = P.PredictorSpec(max_epochs=1)
    model = P.build_predictor(spec, n_features=3, input_mode="RAW", latent_dim=None)
    branches = keras.Model(model.input, model.get_layer("branch_down2").output)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, spec.window, 3)).astype("float32")
    z0 = np.asarray(branches.predict(x, verbose=0))
    assert z0.shape == (5, spec.latent_steps, 3 * spec.branch_filters)                       # time axis kept: 24 -> 6
    x1 = x.copy()
    x1[:, :, 1] += 5.0                                                                       # perturb feature 1 only
    z1 = np.asarray(branches.predict(x1, verbose=0))
    bf = spec.branch_filters
    assert np.array_equal(z0[..., :bf], z1[..., :bf]) and np.array_equal(z0[..., 2 * bf:], z1[..., 2 * bf:])
    assert not np.allclose(z0[..., bf:2 * bf], z1[..., bf:2 * bf])
    x2 = x.copy()
    x2[:, -1, 0] += 5.0                                                                      # causal: only the last step moves
    z2 = np.asarray(branches.predict(x2, verbose=0))
    assert np.array_equal(z0[:, :-1, :bf], z2[:, :-1, :bf]) and not np.allclose(z0[:, -1, :bf], z2[:, -1, :bf])   # the newest row reaches the last latent step


@needs_tf
def test_grouped_layer_matches_independent_causal_convs():
    keras = P._keras()
    G = P._grouped_layer_class()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 24, 3)).astype("float32")
    layer = G(3, 2, 3, 1, None, name="g")
    y = np.asarray(layer(x))
    w, b = layer.get_weights()
    for g in range(3):                                                    # reference: group g is its own causal Conv1D on channel g
        ref = keras.layers.Conv1D(2, 3, padding="causal")
        ref.build((None, 24, 1))
        ref.set_weights([w[g].reshape(3, 1, 2), b[g]])
        assert np.allclose(np.asarray(ref(x[:, :, g:g + 1])), y[:, :, 2 * g:2 * g + 2], atol=1e-5)
    strided = np.asarray(G(3, 2, 3, 2, None, name="s")(x))
    assert strided.shape == (4, 12, 6)


# ----------------------------------------------------------------------------- origin support of the frozen encoder (owner order 2026-10-07)
from tools import fs4_encoder_alignment as AL


def test_analytic_support_of_both_alignments_is_exact():
    even = AL.summarize_reads(AL.analytic_reads(AL.ALIGNMENT_AS_TRAINED))
    odd = AL.summarize_reads(AL.analytic_reads(AL.ALIGNMENT_ORIGIN_COVERING))
    assert [e["max_position"] for e in even] == [0, 4, 8, 12, 16, 20]                 # EVEN: the last step stops 3 rows before the origin
    assert even[-1]["lag_rows_vs_origin"] == 3 and even[-1]["min_position"] == 12
    assert [o["max_position"] for o in odd] == [3, 7, 11, 15, 19, 23] and odd[-1]["lag_rows_vs_origin"] == 0
    assert all(e["max_position"] <= 23 for e in even + odd)                            # neither reads a row after the origin
    assert P.RunnerEncoderBank.alignment == AL.ALIGNMENT_AS_TRAINED and P.RunnerEncoderBank.last_step_lag_rows == 3


def _evidence(leak_free=True, matches=True, odd_worse=2, n=4):
    structure = {"as_trained": {"last_step_reads_no_future": leak_free, "empirical_matches_analytic": [True] * 6,
                                "last_step_lag_rows": 3, "last_step_lag_hours": 3}}
    replays = [{"terminal_matches_replay": matches, "replay_even_mae": 1.0, "replay_odd_mae": 2.0 if i < odd_worse else 0.5} for i in range(n)]
    return structure, replays


def test_choice_rule_is_structural_and_never_silent():
    ok = AL.choose_alignment(*_evidence())
    assert ok["status"] == "CERTIFIED" and ok["alignment"] == AL.ALIGNMENT_AS_TRAINED and ok["caveats"]
    assert ok["evidence_not_a_gate"]["odd_phase_better_terminals"] == 2                     # mixed MAE evidence does not decide soundness
    assert AL.choose_alignment(*_evidence(odd_worse=0))["status"] == "CERTIFIED"             # even if the untrained phase reconstructs better everywhere
    assert AL.choose_alignment(*_evidence(matches=False))["status"] == "NOT_CERTIFIED"       # the replay is not the runner's score
    assert AL.choose_alignment(*_evidence(leak_free=False))["status"] == "NOT_CERTIFIED"
    s, r = _evidence()
    s["as_trained"]["empirical_matches_analytic"][2] = False
    assert AL.choose_alignment(s, r)["status"] == "NOT_CERTIFIED"                             # derived support differs from the measured model
    assert AL.choose_alignment(_evidence()[0], [])["status"] == "NOT_CERTIFIED"                # no replayed terminal, no certificate


def test_certificate_is_bound_to_the_rule_digest_and_tamper_evident(tmp_path):
    structure, replays = _evidence()
    structure["as_trained"]["last_step_lag_hours"] = 3
    replays = [dict(r, population_id="EURUSD", probe_r2_last_latent_to_row_at_lag={AL.ALIGNMENT_AS_TRAINED: {"0": 0.5, "3": 0.9}, AL.ALIGNMENT_ORIGIN_COVERING: {"0": 0.9}}, terminal_mae=1.0, naive_mae=0.5, odd_over_even_mae=1.0) for r in replays]
    cert = AL.write_cert(tmp_path / "c.json", structure, replays, AL.choose_alignment(structure, replays), "a" * 64, "deffa53")
    assert AL.load_cert(tmp_path / "c.json", "a" * 64)["cert_sha256"] == cert["cert_sha256"]
    with pytest.raises(AL.Refusal, match="RULE_MISMATCH"):
        AL.load_cert(tmp_path / "c.json", "b" * 64)
    import json as _json
    d = _json.loads((tmp_path / "c.json").read_text())
    d["alignment"] = AL.ALIGNMENT_ORIGIN_COVERING
    (tmp_path / "c.json").write_text(_json.dumps(d))
    with pytest.raises(AL.Refusal, match="CORRUPT"):
        AL.load_cert(tmp_path / "c.json")


@needs_runner
def test_real_runner_encoder_reads_exactly_what_the_analysis_states(runner_root):
    rr = runner_root
    X, U = P.load_extractor(EXTRACTOR)
    hp = X.Hyper(**next(iter(P.index_runner_results(rr["root"]).values()))[0]["hyper"])
    encoder, decoder, training = X.build_models(hp, calendar_dim=len(U.CALENDAR_SPEC))
    rec, directory = P.index_runner_results(rr["root"])[("SYN", IDENTITY, "feat_a", "inner_2019", "TRAINED_ENCODER")]
    training.load_weights(str(directory / "chosen.weights.h5"))                                   # the REAL trained weights
    odd = AL.odd_phase_encoder(X, U, hp, encoder)
    for model, phase in ((encoder, AL.ALIGNMENT_AS_TRAINED), (odd, AL.ALIGNMENT_ORIGIN_COVERING)):
        emp = AL.empirical_reads(model, len(U.CALENDAR_SPEC))
        ana = AL.analytic_reads(phase)
        assert [sorted(e) for e in emp] == [sorted(a) for a in ana], phase                        # measured on the Keras model == derived
        assert max(max(e) for e in emp) <= 23                                                      # no future read at the origin
    assert max(AL.empirical_reads(encoder, len(U.CALENDAR_SPEC))[-1]) == 20                       # trained phase: lag 3 h at the last step
    assert max(AL.empirical_reads(odd, len(U.CALENDAR_SPEC))[-1]) == 23                           # origin-covering: lag 0
    # the odd-phase model really is the same weights: same shapes and a different (phase-shifted) function
    x = {k: np.random.default_rng(0).normal(size=(3, 24, c)).astype("float32") for k, c in
         (("signal", 1), ("observed_mask", 1), ("delta_time", 1), ("calendar", len(U.CALENDAR_SPEC)))}
    assert encoder.predict(x, verbose=0).shape == odd.predict(x, verbose=0).shape == (3, 6, 8)
    assert not np.allclose(encoder.predict(x, verbose=0), odd.predict(x, verbose=0))


@needs_runner
def test_bank_last_latent_step_never_reads_the_last_three_rows_and_the_origin(runner_root):
    rr = runner_root
    ts = rr["ts"]
    hours = (ts - ts[0]) // 3600
    contiguous = np.where(np.isin(hours[:-1] + 1, hours) & (np.diff(ts) == 3600))[0]
    origin = next(int(i) for i in range(6000, 7000) if np.all(hours[i - 24:i + 1] == hours[i] - np.arange(24, -1, -1)))
    base_cols = rr["cols"]
    bank = _bank(rr, "TRAINED_ENCODER")
    z0, _ = bank.latents(np.array([origin]))
    changed = {}
    for lag in range(0, 8):                                                           # perturb the value `lag` rows before the origin
        r = origin - lag
        cols = {k: v.copy() for k, v in base_cols.items()}
        for k in cols:
            if np.isfinite(cols[k][r]):
                cols[k][r] += 10.0
        z1, _ = _bank(rr, "TRAINED_ENCODER", cols=cols).latents(np.array([origin]))
        D = 8
        last = [not np.array_equal(z0[0, 5, f * D:(f + 1) * D], z1[0, 5, f * D:(f + 1) * D]) for f in range(2)]
        changed[lag] = any(last)
    assert [changed[l] for l in (0, 1, 2)] == [False, False, False]                   # the last step ignores origin, origin-1, origin-2
    assert changed[3] is True                                                          # ... and starts reading at origin-3 (lag 3 h)
    assert bank.alignment == AL.ALIGNMENT_AS_TRAINED and bank.last_step_lag_rows == 3


@needs_runner
def test_replay_on_a_real_runner_terminal_reproduces_its_score_under_the_trained_alignment(runner_root, tmp_path):
    rr = runner_root
    import json as _json
    rec, directory = P.index_runner_results(rr["root"])[("SYN", IDENTITY, "feat_a", "inner_2019", "TRAINED_ENCODER")]
    path = next(iter(rr["root"].parent.glob("train.parquet")))
    registry = _json.loads((rr["root"].parent / "registry.json").read_text())
    out = AL.replay_terminal(EXTRACTOR, {ROLE: str(path)}, directory, registry=registry)
    assert out["terminal_matches_replay"] is True and abs(out["replay_even_mae"] - rec["metrics"]["mae"]) < 1e-6 * rec["metrics"]["mae"]
    assert np.isfinite(out["replay_odd_mae"]) and out["odd_over_even_mae"] > 0
    p = out["probe_r2_last_latent_to_row_at_lag"]
    assert set(p) == {AL.ALIGNMENT_AS_TRAINED, AL.ALIGNMENT_ORIGIN_COVERING} and "0" in p[AL.ALIGNMENT_AS_TRAINED]
