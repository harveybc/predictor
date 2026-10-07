"""R0 modular temporal predictor for the phase-4 weekly comparison (plan section 4, FS4-02/03/05).

Written before the implementation. The spec/identity tests are pure python and run on the
coordinator; the TensorFlow fits run only where FS4_TF_TESTS=1 (a worker under crispdm-run).
"""
from __future__ import annotations

import hashlib
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


# ----------------------------------------------------------------------------- tensorflow
def _synthetic(n=600, f=3, window=24, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = 0.6 * np.roll(X[:, 0], 1) - 0.3 * np.roll(X[:, 1], 2) + 0.05 * rng.normal(size=n)
    return X, y


@needs_tf
def test_architecture_keeps_time_until_the_head_and_fuses_channels():
    spec = P.PredictorSpec(max_epochs=1)
    model = P.build_predictor(spec, n_features=3, input_mode="RAW", latent_dim=None)
    names = [type(l).__name__ for l in model.layers]
    assert "Flatten" not in names and "GlobalAveragePooling1D" not in names and "GlobalMaxPooling1D" not in names
    core = model.get_layer(P.CORE_OUTPUT_NAME)
    assert tuple(core.output.shape[1:]) == (spec.latent_steps, spec.core_filters)   # rank-3 until the head
    assert model.get_layer(P.HEAD_INPUT_NAME).output.shape[1:] == (spec.core_filters,)
    assert len([l for l in model.layers if l.name.startswith("branch_")]) >= 3          # one causal branch per feature
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
