"""Retsu probes for PS01..PS10 (PROCEDIMIENTO_CANONICO.md), part C of ORDEN_SATOSHI_OPERACION.md.

Synthetic CPU fixtures only: no training, no GPU, no TEST rows, no campaign root, no ssh/systemctl/
warehouse. Tests whose name carries ``simulated_boundary`` substitute an external boundary (runtime
version check, model loading/fit, filesystem/rsync/mirror, warehouse HTTP) with a fake or a temp
directory; the enforcement under test is the REAL code of ``tools.i6dmo.prequential_test`` /
``tools.i6dmo.weekly_test`` between those boundaries. They are never GPU or end-to-end evidence.

Assertion vocabulary: ``npz_bytes_equal`` compares file bytes (sha256); ``json_without_clock_equal``
compares JSON receipts after removing the wall-clock fields ``created_at``/``at``.
"""
import copy
import hashlib
import json
import shutil
from datetime import timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tools.i6dmo import continuation as C, prequential_test as S, weekly18 as W

UTC = timezone.utc
stamp = lambda v: int(S.CD.data._instant(v).timestamp())  # noqa: E731
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
STAMPS = ("created_at", "at", "posted_at", "updated_at")


def npz_bytes_equal(a, b):
    return sha(a) == sha(b)


def json_without_clock_equal(a, b):
    def load(p):
        doc = json.loads(Path(p).read_text())
        for key in STAMPS:
            doc.pop(key, None)
        return doc
    return load(a) == load(b)


def tree_json_without_clock(folder):
    out = {}
    for p in sorted(Path(folder).rglob("*")):
        if not p.is_file():
            continue
        if p.suffix == ".json":
            doc = json.loads(p.read_text())
            for key in STAMPS:
                doc.pop(key, None)
            out[str(p.relative_to(folder))] = ("json_without_clock", W.B.digest(doc))
        else:
            out[str(p.relative_to(folder))] = ("bytes", sha(p))
    return out


# --------------------------------------------------------------------------- shared synthetic data

PERIODS = {
    1: dict(train_start="2021-01-05T22:00:00+00:00", train_end="2025-01-05T22:00:00+00:00",
            validation_end="2025-01-10T22:00:00+00:00"),
    2: dict(train_start="2021-01-12T22:00:00+00:00", train_end="2025-01-12T22:00:00+00:00",
            validation_end="2025-01-17T22:00:00+00:00"),
}


def arrays(week, shift=0.0):
    """Two origins inside TEST week ``week`` (labels mature before its validation_end)."""
    t0 = stamp(PERIODS[week]["train_end"]) + 12 * 3600
    origins = np.array([t0, t0 + 7200])
    a = dict(x=np.ones((2, 13, 1)), offsets_hours=np.tile(np.arange(-24, 1, 2), (2, 1)),
             closure_crossed=np.zeros((2, 13), bool), origin_ts=origins, row_ids=origins,
             origin_price=np.array([1.0, 2.0]))
    o = dict(a, mask=np.ones((2, 1), bool), target_price=a["origin_price"][:, None] + 0.2 + shift)
    fit_t = stamp(PERIODS[week]["train_start"]) + 86400
    fit = dict(x=np.ones((3, 13, 1)), offsets_hours=np.tile(np.arange(-24, 1, 2), (3, 1)),
               closure_crossed=np.zeros((3, 13), bool), origin_ts=np.array([fit_t, fit_t + 7200, fit_t + 14400]),
               row_ids=np.array([fit_t, fit_t + 7200, fit_t + 14400]), origin_price=np.array([1.0, 1.5, 2.0]),
               mask=np.ones((3, 1), bool), target_price=np.array([[1.1], [1.6], [2.1]]))
    return a, o, fit


def record(week):
    return dict(cell=dict(id=f"short__WEEK{week}", horizons=[2], period=PERIODS[week]),
                model=dict(horizons_hours=[2]), scales=dict(input_mean=[9.0], input_std=[2.0], naive_mae=[0.2],
                                                            scales_sha256=f"sc{week}" * 16),
                naive_cache={}, population={})


def initial_donor():
    return dict(id="validation-donor", available_by="2024-12-27T22:00:00+00:00",
                scales=dict(input_mean=[1.0], input_std=[1.0], naive_mae=[0.1]), weights_sha256="0" * 64)


def _weight_bias(donor):
    """A number derived from the donor's weight digest: the fake model's 'state'."""
    return int(donor["weights_sha256"][:8], 16) / 16 ** 8


class StatefulFake:
    """Fake backend whose PREDICTION depends on the donor's weight digest and whose UPDATE derives the
    next checkpoint's weights from the exact refit inputs. Records every call."""

    def __init__(self):
        self.events, self.update_inputs = [], []

    def predict(self, donor, a):
        assert set(a) == S.PREDICT_KEYS
        self.events.append(("predict", donor["id"]))
        return a["origin_price"][:, None] * float(donor["scales"]["input_std"][0]) + 0.1 + _weight_bias(donor)

    def update(self, donor, fit, observed, record, folder):
        folder = Path(folder)
        assert (folder.parent / "PREDICTION.json").is_file() and (folder.parent / "SCORE.json").is_file()
        digest = W.B.digest(dict(donor=donor["weights_sha256"],
                                 fit={k: np.asarray(v).tolist() for k, v in sorted(fit.items())},
                                 observed={k: np.asarray(v).tolist() for k, v in sorted(observed.items())}))
        self.events.append(("update", donor["id"]))
        self.update_inputs.append(digest)
        folder.mkdir(parents=True, exist_ok=True)
        W.B.put(folder / "UPDATE.json", dict(donor_id=donor["id"], inputs_sha256=digest))
        return dict(next_checkpoint=dict(id="ckpt-" + digest[:12], root="worker", files={},
                                         scales=dict(record["scales"]),
                                         available_by=record["cell"]["period"]["validation_end"],
                                         weights_sha256=digest),
                    epochs=3, best_epoch=1, stop_reason="patience", training_seconds=1.0, parameters=1)


class StatelessFake(StatefulFake):
    """PRE: the audit's earlier fake, whose prediction ignores the donor's weights (kept to document
    which assertions it fails)."""

    def predict(self, donor, a):
        assert set(a) == S.PREDICT_KEYS
        self.events.append(("predict", donor["id"]))
        return a["origin_price"][:, None] * float(donor["scales"]["input_std"][0]) + 0.1


def chain(base, fake=StatefulFake, shift2=0.0, shift1=0.0):
    """Weeks 1 and 2 in order; week-2 targets shifted by ``shift2`` (week-1 by ``shift1``)."""
    base = Path(base)
    backend = fake()
    out = {}
    donor = initial_donor()
    for week, shift in ((1, shift1), (2, shift2)):
        a, o, fit = arrays(week, shift)
        reader = lambda name, a=a, o=o, fit=fit: copy.deepcopy({"PREDICT": a, "OBSERVED": o, "FIT": fit}[name])  # noqa: E731
        folder = base / f"week{week}"
        out[week] = S.execute_week(record(week), donor, folder, reader, backend, input_sha256=f"in{week}")
        donor = out[week]["next_checkpoint"]
    return out, backend


# --------------------------------------------------------------------------- PS02 / PS03 / PS04

def test_ps02_order_predict_seal_score_update_next_week(tmp_path):
    res, be = chain(tmp_path)
    assert [e[0] for e in be.events] == ["predict", "update", "predict", "update"]
    assert res[2]["inference_donor"] == res[1]["next_checkpoint"]["id"]
    assert be.events[2] == ("predict", res[1]["next_checkpoint"]["id"])


@pytest.mark.parametrize("fake", [StatefulFake, pytest.param(
    StatelessFake, marks=pytest.mark.xfail(strict=True, reason="PRE: the state-free fake cannot show the "
                                                               "positive control (week-2 depends on the week-1 refit)"))])
def test_ps03_current_week_outcomes_never_change_its_sealed_prediction_positive_control(tmp_path, fake):
    (ra, ba), (rb, bb) = chain(tmp_path / "a", fake), chain(tmp_path / "b", fake, shift1=50.0)
    # week-1 sealed forecast: NPZ bytes equal and seal receipt equal (without clock)
    assert npz_bytes_equal(tmp_path / "a/week1/PREDICTED.npz", tmp_path / "b/week1/PREDICTED.npz")
    assert json_without_clock_equal(tmp_path / "a/week1/PREDICTION.json", tmp_path / "b/week1/PREDICTION.json")
    assert ra[1]["prediction_sha256"] == rb[1]["prediction_sha256"]
    # the perturbed week-1 outcome moved week-1's score and its refit inputs...
    assert ra[1]["metrics"]["TEST"] != rb[1]["metrics"]["TEST"] and ba.update_inputs[0] != bb.update_inputs[0]
    # ...hence the checkpoint that serves week 2 (next-week only), and so week 2's sealed forecast MAY change:
    assert ra[1]["next_checkpoint"]["weights_sha256"] != rb[1]["next_checkpoint"]["weights_sha256"]
    assert not npz_bytes_equal(tmp_path / "a/week2/PREDICTED.npz", tmp_path / "b/week2/PREDICTED.npz")  # positive control


@pytest.mark.parametrize("fake", [StatefulFake, StatelessFake])
def test_ps04_future_week_outcomes_never_change_an_earlier_refit_or_sealed_prediction(tmp_path, fake):
    (ra, ba), (rb, bb) = chain(tmp_path / "a", fake), chain(tmp_path / "b", fake, shift2=50.0)
    assert tree_json_without_clock(tmp_path / "a/week1") == tree_json_without_clock(tmp_path / "b/week1")
    assert npz_bytes_equal(tmp_path / "a/week1/PREDICTED.npz", tmp_path / "b/week1/PREDICTED.npz")
    assert npz_bytes_equal(tmp_path / "a/week1/TEST.npz", tmp_path / "b/week1/TEST.npz")
    assert ba.update_inputs[0] == bb.update_inputs[0] and ra[1]["next_checkpoint"] == rb[1]["next_checkpoint"]
    # week 2's OWN seal is also unchanged by its own outcomes (PS03 at week 2); only its score and refit move
    assert npz_bytes_equal(tmp_path / "a/week2/PREDICTED.npz", tmp_path / "b/week2/PREDICTED.npz")
    assert ra[2]["metrics"]["TEST"] != rb[2]["metrics"]["TEST"] and ba.update_inputs[1] != bb.update_inputs[1]


# --------------------------------------------------------------------------- PS05 / edge: no-pairs monitor

def test_ps05_rows_outside_the_four_year_window_and_immature_labels_are_refused():
    p = PERIODS[1]
    a, o, fit = arrays(1)
    S.validate_update_span(p, fit, o, [2])
    early = dict(fit, origin_ts=np.array([stamp(p["train_start"]) - 3600, fit["origin_ts"][1], fit["origin_ts"][2]]))
    with pytest.raises(ValueError, match="TRAIN_CUTOFF_OR_LABEL_MATURITY"):
        S.validate_update_span(p, early, o, [2])
    late_label = dict(fit, origin_ts=np.array([stamp(p["train_end"]) - 3600] * 3))
    with pytest.raises(ValueError, match="TRAIN_CUTOFF_OR_LABEL_MATURITY"):
        S.validate_update_span(p, late_label, o, [2])
    with pytest.raises(ValueError, match="NOT_FOUR_CALENDAR_YEARS"):
        S.validate_update_span(dict(p, train_start="2021-01-05T23:00:00+00:00"), fit, o, [2])


def test_edge_week_without_pairs_cannot_drive_early_stopping():
    a, o, fit = arrays(1)
    with pytest.raises(ValueError, match="NO_OBSERVED_MONITOR_PAIRS"):
        S.validate_update_span(PERIODS[1], fit, dict(o, mask=np.zeros((2, 1), bool)), [2])


# --------------------------------------------------------------------------- PS07 scalers

def test_ps07_checkpoint_identity_binds_its_scalers_and_a_swapped_scaler_is_refused_on_resume(tmp_path):
    folder = tmp_path / "ckpt"
    folder.mkdir()
    (folder / "forecast_model.keras").write_bytes(b"k")
    own = dict(input_mean=[1.0], input_std=[1.0], naive_mae=[0.1], scales_sha256="a" * 64)
    other = dict(own, input_mean=[2.0], scales_sha256="b" * 64)
    c1 = S.checkpoint_files(tmp_path, folder, dict(horizons_hours=[2]), own, "w" * 64, PERIODS[1]["validation_end"])
    c2 = S.checkpoint_files(tmp_path, folder, dict(horizons_hours=[2]), other, "w" * 64, PERIODS[1]["validation_end"])
    assert c1["id"] != c2["id"] and c1["weights_sha256"] == c2["weights_sha256"]
    a, o, fit = arrays(1)
    assert not np.array_equal(S.inference_inputs(a, own)[0], S.inference_inputs(a, other)[0])
    donor = dict(initial_donor(), id=c1["id"], scales=own)
    reader = lambda name: copy.deepcopy({"PREDICT": a, "OBSERVED": o, "FIT": fit}[name])  # noqa: E731
    S.execute_week(record(1), donor, tmp_path / "w", reader, StatefulFake(), input_sha256="in")
    with pytest.raises(ValueError, match="SEALED_FORECAST_IDENTITY"):
        S.execute_week(record(1), dict(donor, id=c2["id"], scales=other), tmp_path / "w", reader, StatefulFake(),
                       input_sha256="in")


def test_ps07_inference_scaler_is_the_donors_and_refit_scaler_is_the_periods_train_scaler(tmp_path):
    res, be = chain(tmp_path)
    p1 = np.load(tmp_path / "week1" / "PREDICTED.npz")["prediction"]
    np.testing.assert_allclose(p1[:, 0], np.array([1.0, 2.0]) + 0.1 + _weight_bias(initial_donor()))
    assert res[1]["next_checkpoint"]["scales"]["scales_sha256"] == record(1)["scales"]["scales_sha256"]
    p2 = np.load(tmp_path / "week2" / "PREDICTED.npz")["prediction"]
    np.testing.assert_allclose(p2[:, 0], np.array([1.0, 2.0]) * record(1)["scales"]["input_std"][0] + 0.1
                               + _weight_bias(res[1]["next_checkpoint"]))


# --------------------------------------------------------------------------- PS09 resumption

def test_ps09_crash_between_seal_and_update_resumes_from_the_same_donor_without_repredicting(tmp_path):
    a, o, fit = arrays(1)
    reader = lambda name: copy.deepcopy({"PREDICT": a, "OBSERVED": o, "FIT": fit}[name])  # noqa: E731

    class Crash(StatefulFake):
        def update(self, *args):
            raise RuntimeError("simulated crash after STARTED")
    with pytest.raises(RuntimeError):
        S.execute_week(record(1), initial_donor(), tmp_path, reader, Crash(), input_sha256="in")
    sealed = sha(tmp_path / "PREDICTED.npz")
    be2 = StatefulFake()
    res = S.execute_week(record(1), initial_donor(), tmp_path, reader, be2, input_sha256="in")
    assert be2.events == [("update", "validation-donor")] and sha(tmp_path / "PREDICTED.npz") == sealed
    assert res["prediction_sha256"] == sealed


def test_ps09_stale_update_receipt_from_another_donor_is_refused_before_any_load_simulated_boundary(tmp_path):
    """Simulated boundary: model loading is forbidden by a mock; the receipt guard of the REAL update runs."""
    a, o, fit = arrays(1)
    folder = tmp_path / "update"
    folder.mkdir()
    W.B.put(folder / "UPDATE.json", dict(donor_id="some-other-donor", next_checkpoint=dict(files={})))
    backend = S.GpuBackend(dict(gpu_data=str(tmp_path)), tmp_path)
    rec = dict(record(1), model=dict(horizons_hours=[2]))
    donor = dict(initial_donor(), model=dict(horizons_hours=[2]))
    with patch.object(S.GpuBackend, "load", side_effect=AssertionError("load must not run")):
        with pytest.raises(ValueError, match="UPDATE_DONOR_CHANGED"):
            backend.update(donor, fit, o, rec, folder)


# --------------------------------------------------------------------------- PS06 (C1) simulated boundary

@pytest.fixture(scope="module")
def tf_runtime():
    tf = pytest.importorskip("tensorflow")
    from tools.i6dmo import continuation_models as M
    return tf, M


def _tiny_model(M, seed):
    """A minimal 3-input, one-head Keras model with the trainer's input/output protocol."""
    keras = M.keras
    keras.utils.set_random_seed(seed)
    x = keras.Input((13, 1), name="x")
    off = keras.Input((13,), name="offsets")
    clo = keras.Input((13,), name="closure")
    h = keras.layers.Concatenate()([keras.layers.Flatten()(x), off, clo])
    heads = [keras.layers.Dense(1, name=f"head{i}")(h) for i in range(2)]      # two heads: list output protocol
    return keras.Model([x, off, clo], heads)


def _checkpoint(tmp_path, M, seed):
    from predictor_plugins.modular_temporal import weights_hash
    model = _tiny_model(M, seed)
    folder = tmp_path / "donor"
    folder.mkdir(parents=True)
    model.save(folder / "forecast_model.keras")
    files = {"donor/forecast_model.keras": sha(folder / "forecast_model.keras")}
    return dict(id=f"donor-{seed}", root="worker", model=dict(horizons_hours=[2, 3]),
                scales=dict(input_mean=[1.0], input_std=[1.0], naive_mae=[0.1, 0.2], scales_sha256="d" * 64),
                weights_sha256=weights_hash(model), archive="donor/forecast_model.keras", files=files,
                available_by="2024-12-27T22:00:00+00:00"), model


def _runtime_boundary(M):
    """Only the runtime-version/GPU-presence checks are substituted; load_model and hashing are real."""
    return (patch.object(M.tf, "__version__", "2.21.0"),
            patch("tools.i6dmo.prequential_test.importlib.metadata.version", return_value="3.13.2"),
            patch.object(M.tf.config, "list_physical_devices", return_value=[object()]))


def _fit_stub(initial_from, perturb=True):
    """Stands in for train.fit_dmo at the fit boundary; the initial hash is COMPUTED with the real
    weights_hash on ``initial_from(bundle.forecast_model)`` (never copied from the donor record)."""
    from predictor_plugins.modular_temporal import weights_hash

    def fit_dmo(bundle, fit, inner, training, seed, *, ids=None, record_device="cpu", progress=None):
        model = bundle.forecast_model
        initial = weights_hash(initial_from(model))
        if perturb:
            model.set_weights([w + 1e-3 for w in model.get_weights()])
        restored = weights_hash(model)
        return dict(initial_weights_sha256=initial, restored_weights_sha256=restored, epochs_completed=2,
                    best_epoch=1, stop_reason="patience", fit_seconds=0.5, history=[], early_stop=dict(rule="stub"))
    return fit_dmo


def test_ps06_real_initial_weights_guard_accepts_only_the_hash_computed_from_the_loaded_checkpoint_simulated_boundary(tmp_path, tf_runtime):
    """Simulated boundary (runtime check + fit): the REAL GpuBackend.load/update run, including
    DONOR_WEIGHTS_CHANGED, WARM_START_WEIGHTS_CHANGED and SAVED_CHECKPOINT_RELOAD."""
    tf, M = tf_runtime
    from predictor_plugins.modular_temporal import weights_hash
    from tools.i6dmo import train as T
    donor, model = _checkpoint(tmp_path, M, 6102026)
    a, o, fit = arrays(1)
    two = lambda d: dict(d, mask=np.ones((len(d["origin_ts"]), 2), bool),  # noqa: E731
                         target_price=np.repeat(d["target_price"], 2, axis=1) + np.array([[0.0, 0.05]]))
    o, fit = two(o), two(fit)
    rec = dict(record(1), model=donor["model"], scales=donor["scales"])
    rec["cell"] = dict(rec["cell"], horizons=[2, 3])
    backend = S.GpuBackend(dict(gpu_data=str(tmp_path)), tmp_path)
    p1, p2, p3 = _runtime_boundary(M)
    with p1, p2, p3:
        # (1) correct hash: computed by weights_hash on the model loaded from the .keras archive
        with patch.object(T, "fit_dmo", _fit_stub(lambda m: m)):
            r = backend.update(donor, fit, o, rec, tmp_path / "u1")
        reloaded = M.keras.models.load_model(tmp_path / donor["archive"], compile=False, safe_mode=True)
        assert r["initial_weights_sha256"] == donor["weights_sha256"] == weights_hash(reloaded)
        assert r["next_checkpoint"]["weights_sha256"] == r["restored_weights_sha256"] != r["initial_weights_sha256"]
        assert (tmp_path / "u1" / "forecast_model.keras").is_file() and r["next_checkpoint"]["scales"] == donor["scales"]
        # (2) a trainer that re-initializes: the initial hash is computed from OTHER weights -> typed refusal
        fresh = _tiny_model(M, 1)
        with patch.object(T, "fit_dmo", _fit_stub(lambda m: fresh)):
            with pytest.raises(ValueError, match="WARM_START_WEIGHTS_CHANGED"):
                backend.update(donor, fit, o, rec, tmp_path / "u2")
        assert not (tmp_path / "u2" / "UPDATE.json").exists()
        # (3) a donor record whose declared hash does not match the archive bytes -> refused at load
        with patch.object(T, "fit_dmo", _fit_stub(lambda m: m)):
            with pytest.raises(ValueError, match="DONOR_WEIGHTS_CHANGED"):
                backend.update(dict(donor, weights_sha256="f" * 64), fit, o, rec, tmp_path / "u3")
        # (4) a stub that merely copies the donor's declared value is NOT what the guard trusts: with the
        #     archive bytes swapped for another model, load refuses before any fit
        other, _ = _checkpoint(tmp_path / "other", M, 2)
        shutil.copyfile(tmp_path / "other" / other["archive"], tmp_path / donor["archive"])
        with patch.object(T, "fit_dmo", lambda *a, **k: dict(initial_weights_sha256=donor["weights_sha256"])):
            with pytest.raises(ValueError, match="ARTIFACT_DIGEST"):
                backend.update(donor, fit, o, rec, tmp_path / "u4")


# --------------------------------------------------------------------------- PS10 (C2) simulated boundary

def _complete_result(spec, cell, heads, folder, unsupported_last_head=False):
    """A COMPLETE RESULT.json + TEST.npz shaped like a real unit (synthetic metrics, no TEST rows)."""
    n = 4
    origins = np.arange(n) * 7200 + stamp(cell["period"]["train_end"]) + 12 * 3600
    mask = np.ones((n, len(heads)), bool)
    if unsupported_last_head:
        mask[:, -1] = False                                        # one head without support this week
    target = np.where(mask, 1.0 + 0.01 * (1 + np.arange(len(heads)))[None, :], np.nan)   # never equal to origin
    arrays_ = dict(origin_ts=origins, origin_price=np.full(n, 1.0), target_price=target, mask=mask)
    metrics = C.measured_metrics(arrays_, np.full((n, len(heads)), 1.005), heads)
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "TEST.npz", prediction=np.full((n, len(heads)), 1.005), origin_ts=origins,
                        origin_price=arrays_["origin_price"], target_price=target, mask=mask)
    r = dict(batch_id=spec["batch_id"], cell=cell, state="COMPLETE", files={"TEST.npz": sha(folder / "TEST.npz")},
             metrics=dict(TEST=metrics), training_seconds=1.0, parameters=1, epochs=3, best_epoch=1,
             stop_reason="patience", actual_spacing=2, population={}, test_read=True,
             qualification=S.QUALIFICATION, inference_donor="d", prediction_sha256="p" * 64)
    r["sha256"] = W.B.digest(r)
    W.B.put(folder / "RESULT.json", r)


def _skipped_result(spec, cell, folder):
    folder.mkdir(parents=True)
    r = dict(batch_id=spec["batch_id"], cell=cell, state="SKIPPED_NO_EVAL_ROWS", files={}, test_read=True,
             next_checkpoint=dict(id="d"))
    r["sha256"] = W.B.digest(r)
    W.B.put(folder / "RESULT.json", r)


def _campaign(tmp_path, unsupported_last_head=False):
    from tools.i6dmo import weekly_test_data as D
    root, art, gpu = tmp_path / "root", tmp_path / "artifacts", tmp_path / "gpu_root"
    cells = []
    for fam, heads in (("short", D.SHORT), ("long", D.LONG)):
        for week, period in PERIODS.items():
            cells.append(dict(id=f"{fam}__WEEK{week}", family=fam, period=period, horizons=heads, kind="conv",
                              stage="prequential_test", group="short", head="lstm", slots=13, core=[2], spacing=2))
    spec = dict(root=str(root), artifacts=str(art), gpu_root=str(gpu), ssh_alias="sim-alias", cells=cells,
                batch_id=S.NAME, code_commit="c" * 40, code_files={}, training=W.B.TRAINING, seed=W.B.SEED,
                population_digest="p" * 64, qualification=S.QUALIFICATION, evaluation_split="TEST",
                warehouse=dict(url="http://warehouse.invalid", token_env="SIM_TOKEN"),
                local_mirror=str(tmp_path / "mirror-local"), mirror_alias="sim-mirror",
                remote_mirror=str(tmp_path / "mirror-remote"))
    root.mkdir()
    for c in cells:
        folder = gpu / "results" / c["id"]
        if c["period"] is PERIODS[2]:
            _complete_result(spec, c, c["horizons"], folder, unsupported_last_head)
        else:
            _skipped_result(spec, c, folder)
    return spec


@pytest.fixture
def campaign(tmp_path):
    """A finished-compute campaign on a temp filesystem: 2 short + 2 long cells (one SKIPPED each)."""
    return _campaign(tmp_path)


class Boundaries:
    """Substitutes for rsync/ssh/systemctl (run_cmd), the storage probe, the mirror and the warehouse."""

    def __init__(self, spec, warehouse, mirror_state):
        self.spec, self.warehouse, self.mirror_state, self.calls = spec, warehouse, mirror_state, []

    def run_cmd(self, args, timeout=600):
        self.calls.append([str(a) for a in args])
        if args[0] == "rsync":                                     # simulated remote -> local artifact sync
            shutil.copytree(self.spec["gpu_root"], self.spec["artifacts"], dirs_exist_ok=True)
        return ""

    def mirror(self, spec, artifacts, files):
        return dict(state=self.mirror_state, files=len(files), local_bad=[], remote_bad=[], manifest_sha256=W.B.digest(files))

    def __enter__(self):
        from tools.i6dmo import outbox, weekly_test as P
        self._stack = [S.context(), patch.object(W, "run_cmd", self.run_cmd), patch.object(W, "mirror", self.mirror),
                       patch.object(P, "storage_guard", lambda path: None),
                       patch.object(outbox, "HttpWarehouse", lambda url, token_env: self.warehouse)]
        for s in self._stack:
            s.__enter__()
        return self

    def __exit__(self, *exc):
        for s in reversed(self._stack):
            s.__exit__(*exc)
        return False

    @property
    def systemctl_calls(self):
        return [c for c in self.calls if "systemctl" in c]


def test_ps10_pending_readback_then_idempotent_retry_simulated_boundary(campaign):
    """Simulated boundary (warehouse/rsync/mirror): a readback outage leaves items POSTED, the next tick
    completes delivery without re-posting, and publication is written only when nothing is pending."""
    from tests.i6dmo.ops_fakes import FakeWarehouse
    from tools.i6dmo import weekly_test as P
    from tools.i6dmo.outbox import Outbox
    wh = FakeWarehouse(fail_query=5)                                # 5 items: 4 cells + annual
    with Boundaries(campaign, wh, "VERIFIED") as b:
        first = P.tick(campaign)
        assert first["state"] == "COMPUTE_COMPLETE" and first["publication"]["pending"] == 5
        assert first["publication"]["delivered"] == 0 and wh.posts == 5
        assert not (Path(campaign["artifacts"]) / "PUBLICATION.json").exists() and not b.systemctl_calls
        second = P.tick(campaign)
        assert second["publication"] == dict(second["publication"], pending=0, mismatch=0, delivered=5)
        assert wh.posts == 5                                        # idempotent: readback only, no re-POST
        box = Outbox.open_readonly(Path(campaign["root"]) / "outbox.sqlite3")
        try:
            assert len(box.items()) == 5 and all(i["state"] == "DELIVERED" for i in box.items())
        finally:
            box.close()
        assert (Path(campaign["artifacts"]) / "PUBLICATION.json").is_file()
        assert (Path(campaign["artifacts"]) / "READBACK.json").is_file()


def test_ps10_failed_mirror_prevents_close_and_touches_no_timer_simulated_boundary(campaign):
    from tests.i6dmo.ops_fakes import FakeWarehouse
    from tools.i6dmo import weekly_test as P
    with Boundaries(campaign, FakeWarehouse(), "MISMATCH") as b:
        state = P.tick(campaign)
        assert state["publication"]["pending"] == 0 and state["mirrors"]["state"] == "MISMATCH"
        assert state["state"] == "COMPUTE_COMPLETE" and not (Path(campaign["root"]) / "CLOSE.json").exists()
        assert b.systemctl_calls == []
        again = P.tick(campaign)                                    # retry keeps the same answer, still no CLOSE
        assert again["mirrors"]["state"] == "MISMATCH" and not (Path(campaign["root"]) / "CLOSE.json").exists()


def test_ps10_later_closure_stops_only_this_campaigns_own_timers_simulated_boundary(campaign):
    """Simulated boundary: the injected systemctl recorder must see exactly the two campaign timers."""
    from tests.i6dmo.ops_fakes import FakeWarehouse
    from tools.i6dmo import weekly_test as P
    with Boundaries(campaign, FakeWarehouse(), "MISMATCH") as b:
        P.tick(campaign)
        assert not (Path(campaign["root"]) / "CLOSE.json").exists()
    with Boundaries(campaign, FakeWarehouse(), "VERIFIED") as b:
        state = P.tick(campaign)
        assert state["state"] == "CLOSED" and (Path(campaign["root"]) / "CLOSE.json").is_file()
        assert W.read(Path(campaign["root"]) / "CLOSE.json")["state"] == "CLOSED"
        units = [c[-1] for c in b.systemctl_calls]
        assert units == [S.NAME + "-gpu.timer", S.NAME + "-tick.timer"]
        assert all(c[-3:-1] == ["disable", "--now"] for c in b.systemctl_calls)
        assert b.systemctl_calls[0][:2] == ["ssh", campaign["ssh_alias"]]        # gpu timer on the worker
        assert b.systemctl_calls[1][0] == "systemctl"                            # tick timer locally
        closed = P.tick(campaign)                                   # closed campaign: no more calls
        assert closed["state"] == "CLOSED" and len(b.systemctl_calls) == 2
    annual = W.read(Path(campaign["artifacts"]) / "ANNUAL_TEST.json")
    assert annual["families"]["short"]["state"] == annual["families"]["long"]["state"] == "COMPLETE"
    assert all(m["n"] == 4 for m in annual["families"]["short"]["metrics"])


def test_ps10_closure_with_a_head_lacking_annual_pairs_must_still_close_simulated_boundary(tmp_path):
    """Simulated boundary: same fakes as above; the only COMPLETE week leaves the last head unsupported."""
    from tests.i6dmo.ops_fakes import FakeWarehouse
    from tools.i6dmo import weekly_test as P
    spec = _campaign(tmp_path, unsupported_last_head=True)
    with Boundaries(spec, FakeWarehouse(), "VERIFIED") as b:
        state = P.tick(spec)
    assert state["state"] == "CLOSED" and (Path(spec["root"]) / "CLOSE.json").exists() and len(b.systemctl_calls) == 2
    annual = W.read(Path(spec["artifacts"]) / "ANNUAL_TEST.json")
    last = annual["families"]["short"]["metrics"][-1]
    assert last["n"] == 0 and last["mae"] is None and last["naive_mae"] is None    # metrics untouched: no fake zero
    assert annual["coverage"]["families"]["short"]["per_horizon"][str(last["horizon"])]["pairs_scored"] == 0


def test_ps10_observation_head_without_annual_pairs_renders_sin_soporte_simulated_boundary(tmp_path):
    """Was the observation of the crash (TypeError on '{m[mae]:.12f}' of None, closure lost after publication);
    now records the fixed behaviour: the row renders n 0 / sin soporte and the head is listed without support."""
    from tests.i6dmo.ops_fakes import FakeWarehouse
    from tools.i6dmo import weekly_test as P, weekly_test_data as D
    spec = _campaign(tmp_path, unsupported_last_head=True)
    with Boundaries(spec, FakeWarehouse(), "VERIFIED") as b:
        P.tick(spec)
        assert W.read(Path(spec["artifacts"]) / "PUBLICATION.json")["pending"] == 0
    text = (Path(spec["artifacts"]) / "RESULTS.md").read_text()
    head = D.SHORT[-1]
    assert f"| short | {head} h | n 0 / sin soporte | n 0 / sin soporte | n 0 / sin soporte | 0 |" in text
    assert f"| short | {head} h | 0.000000000000" not in text
    cov = W.read(Path(spec["artifacts"]) / "COVERAGE.json")["families"]["short"]["per_horizon"][str(head)]
    assert cov["weeks_with_pairs"] == 0 and cov["weeks_without_support"] == 2 and cov["pairs_scored"] == 0
    assert cov["weeks_without_support_ids"] == ["short__WEEK1", "short__WEEK2"]


# --------------------------------------------------------------------------- PS01 initial donors + no-pairs week

MODEL = dict(horizons_hours=[2, 3, 4, 5, 6, 12, 18], tag="retained")


def _member(base, name, train_end, validation_end, *, state="VERIFIED", weights="1" * 64):
    folder = base / name
    folder.mkdir(parents=True)
    (folder / "forecast_model.keras").write_bytes(("keras-" + name).encode())
    (folder / "model.weights.h5").write_bytes(("h5-" + name).encode())
    W.B.put(folder / "CONFIG.json", dict(model=MODEL))
    W.B.put(folder / "SCALES.json", dict(input_mean=[0.0], input_std=[1.0], naive_mae=[0.1], scales_sha256=name * 8))
    W.B.put(folder / "TRAINING.json", dict(initial_weights_sha256="0" * 64))
    W.B.put(folder / "ARCHIVE.json", dict(state=state, archive_sha256=sha(folder / "forecast_model.keras"),
                                          weights_sha256=weights))
    names = ("forecast_model.keras", "model.weights.h5", "CONFIG.json", "SCALES.json", "TRAINING.json", "ARCHIVE.json")
    return dict(id="member-" + name, folder=name, period=dict(train_end=train_end, validation_end=validation_end),
                files={n: sha(folder / n) for n in names})


def _prior(base, model):
    root = base / "prior"
    root.mkdir()
    cells, files = [], {}
    for cid, state, period in (("short__A", "SKIPPED_NO_EVAL_ROWS", PERIODS[1]), ("short__B", "COMPLETE", PERIODS[2])):
        cell = dict(id=cid, family="short", period=period, horizons=[2])
        folder = root / cid
        folder.mkdir()
        rec = dict(cell=cell, parent_state=state, model=model, scales=dict(input_mean=[0.0], input_std=[1.0],
                   naive_mae=[0.1], scales_sha256="s" * 64), naive_cache={}, population={})
        if state == "COMPLETE":
            a, o, fit = arrays(2)
            np.savez_compressed(folder / "FIT.npz", **fit)
            np.savez_compressed(folder / "TEST.npz", **o)
            files[cid + "/FIT.npz"] = sha(folder / "FIT.npz")
            files[cid + "/TEST.npz"] = sha(folder / "TEST.npz")
        W.B.put(folder / "PERIOD.json", rec)
        files[cid + "/PERIOD.json"] = sha(folder / "PERIOD.json")
        cells.append(cell)
    doc = dict(cells=cells, files=files)
    doc["sha256"] = W.B.digest(doc)
    W.B.put(root / "PREPARED.json", doc)
    return root, doc


@pytest.fixture
def world(tmp_path):
    prior, doc = _prior(tmp_path, dict(MODEL))
    members = {}
    for fam in ("short", "long"):
        members[fam] = [_member(tmp_path, f"{fam}-early", "2024-12-15T22:00:00+00:00", "2024-12-20T22:00:00+00:00"),
                        _member(tmp_path, f"{fam}-late", "2024-12-22T22:00:00+00:00", "2024-12-27T22:00:00+00:00")]
    pair = tmp_path / "pair.json"
    W.B.put(pair, dict(families={fam: dict(members=ms) for fam, ms in members.items()}))
    spec = dict(prepared_source=str(prior), prepared_source_sha256=doc["sha256"], data_root=str(tmp_path / "data"),
                pair=str(pair))
    return dict(tmp=tmp_path, spec=spec, members=members)


def test_ps01_initial_donor_is_the_last_validation_member_with_exact_bytes_and_hashes(world):
    with S.context():
        p = S.prepare(world["spec"])
    for fam in ("short", "long"):
        ini, late = p["initial"][fam], world["members"][fam][1]
        assert ini["source_member"] == late["id"] and ini["available_by"] == "2024-12-27T22:00:00+00:00"
        assert ini["weights_sha256"] == "1" * 64 and ini["root"] == "data"
        copied = Path(world["spec"]["data_root"]) / "initial" / fam
        assert sha(copied / "forecast_model.keras") == late["files"]["forecast_model.keras"]
        assert ini["scales"]["scales_sha256"] == (f"{fam}-late") * 8
        assert ini["id"] == W.B.digest({k: v for k, v in ini.items() if k != "id"})
    assert p["sha256"] == W.B.digest({k: v for k, v in p.items() if k != "sha256"})
    assert [c["predecessor"] for c in p["cells"]] == [None, "short__A"]
    with np.load(Path(world["spec"]["data_root"]) / "short__B" / "PREDICT.npz") as z:
        assert set(z.files) == S.PREDICT_KEYS
    assert not (Path(world["spec"]["data_root"]) / "short__A" / "PREDICT.npz").exists()


def test_ps01_a_member_finishing_inside_test_or_unverified_is_refused(world):
    late = world["members"]["short"][1]
    with S.context():
        bad = dict(late, period=dict(late["period"], validation_end="2025-01-10T22:00:00+00:00"))
        W.B.put(world["tmp"] / "pair.json", dict(families=dict(short=dict(members=[bad]), long=dict(members=world["members"]["long"]))))
        with pytest.raises(ValueError, match="INITIAL_DONOR_FROM_TEST"):
            S.prepare(dict(world["spec"], data_root=str(world["tmp"] / "d1")))
        unverified = _member(world["tmp"], "short-unverified", "2024-12-29T22:00:00+00:00", "2024-12-31T22:00:00+00:00",
                             state="UNVERIFIED")
        W.B.put(world["tmp"] / "pair.json", dict(families=dict(short=dict(members=[late, unverified]),
                                                               long=dict(members=world["members"]["long"]))))
        with pytest.raises(ValueError, match="VALIDATION_ARCHIVE_NOT_VERIFIED"):
            S.prepare(dict(world["spec"], data_root=str(world["tmp"] / "d2")))


def test_edge_no_pairs_week_carries_the_donor_forward_without_mae_or_training_simulated_boundary(world, tmp_path):
    """Simulated boundary: GpuBackend.load is forbidden by a mock; the REAL run() scheduling executes."""
    with S.context():
        p = S.prepare(world["spec"])
        gpu_root = tmp_path / "gpu"
        spec = dict(world["spec"], gpu_root=str(gpu_root), gpu_data=world["spec"]["data_root"], cells=p["cells"],
                    population_digest=p["sha256"], batch_id=S.NAME, training=W.B.TRAINING, seed=W.B.SEED,
                    evaluation_split="TEST", qualification=S.QUALIFICATION, code_files={})
        with patch.object(S.GpuBackend, "load", side_effect=AssertionError("no model load for a no-pairs week")):
            S.run(spec)
        r = W.read(gpu_root / "results" / "short__A" / "RESULT.json")
        assert r["state"] == "SKIPPED_NO_EVAL_ROWS" and "metrics" not in r and r["files"] == {}
        assert r["next_checkpoint"] == p["initial"]["short"]
        assert W.read(gpu_root / "PROGRESS.json")["state"] == "SKIPPED_NO_EVAL_ROWS"
        failed = dict(batch_id=S.NAME, cell=p["cells"][0], state="FAILED", files={}, test_read=True, reason="x")
        failed["sha256"] = W.B.digest(failed)
        W.B.put(gpu_root / "results" / "short__A" / "RESULT.json", failed)
        S.run(spec)
        assert W.read(gpu_root / "PROGRESS.json")["state"] == "BLOCKED_DEPENDENCY"
