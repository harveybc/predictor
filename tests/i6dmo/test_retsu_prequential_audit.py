"""Retsu audit probes for PS01..PS10 (PROCEDIMIENTO_CANONICO.md) at 811a7544: synthetic CPU fixtures only.

No training, no GPU, no TEST rows, no campaign root. Backends are fakes that record calls; the
enforcement under test is ``tools.i6dmo.prequential_test`` itself (prepare, execute_week, run,
validate_update_span, checkpoint_files, GpuBackend.update receipt guard).
"""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tools.i6dmo import prequential_test as S, weekly18 as W

UTC = timezone.utc
stamp = lambda v: int(S.CD.data._instant(v).timestamp())  # noqa: E731
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def _tree(folder):
    """Every file byte-hashed, except that JSON receipts are compared without their wall-clock stamps."""
    out = {}
    for p in sorted(Path(folder).rglob("*")):
        if not p.is_file():
            continue
        if p.suffix == ".json":
            doc = json.loads(p.read_text())
            for key in ("created_at", "at"):
                doc.pop(key, None)
            out[str(p.relative_to(folder))] = W.B.digest(doc)
        else:
            out[str(p.relative_to(folder))] = sha(p)
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


class Recorder:
    """Fake backend: deterministic prediction from inputs only; records exactly what update sees."""

    def __init__(self):
        self.events, self.update_inputs = [], []

    def predict(self, donor, a):
        assert set(a) == S.PREDICT_KEYS
        self.events.append(("predict", donor["id"]))
        return a["origin_price"][:, None] * float(donor["scales"]["input_std"][0]) + 0.1

    def update(self, donor, fit, observed, record, folder):
        folder = Path(folder)
        assert (folder.parent / "PREDICTION.json").is_file() and (folder.parent / "SCORE.json").is_file()
        digest = W.B.digest(dict(donor=donor["id"], fit={k: np.asarray(v).tolist() for k, v in sorted(fit.items())},
                                 observed={k: np.asarray(v).tolist() for k, v in sorted(observed.items())}))
        self.events.append(("update", donor["id"]))
        self.update_inputs.append(digest)
        folder.mkdir(parents=True, exist_ok=True)
        W.B.put(folder / "UPDATE.json", dict(donor_id=donor["id"], inputs_sha256=digest))
        scales = dict(record["scales"])
        return dict(next_checkpoint=dict(id="ckpt-" + digest[:12], root="worker", files={}, scales=scales,
                                         available_by=record["cell"]["period"]["validation_end"],
                                         weights_sha256="w" + digest[:12]),
                    epochs=3, best_epoch=1, stop_reason="patience", training_seconds=1.0, parameters=1)


def initial_donor():
    return dict(id="validation-donor", available_by="2024-12-27T22:00:00+00:00",
                scales=dict(input_mean=[1.0], input_std=[1.0], naive_mae=[0.1]), weights_sha256="w0")


def chain(base, shift2=0.0, shift1=0.0):
    """Weeks 1 and 2 in order; week-2 targets shifted by ``shift2`` (week-1 by ``shift1``)."""
    base = Path(base)
    backend = Recorder()
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

def test_ps02_order_and_ps03_current_week_targets_never_touch_the_sealed_prediction(tmp_path):
    (res_a, be_a), (res_b, be_b) = chain(tmp_path / "a"), chain(tmp_path / "b", shift1=50.0)
    assert [e[0] for e in be_a.events] == ["predict", "update", "predict", "update"]
    for week in (1, 2):
        assert sha(tmp_path / "a" / f"week{week}" / "PREDICTED.npz") == sha(tmp_path / "b" / f"week{week}" / "PREDICTED.npz")
        assert res_a[week]["prediction_sha256"] == res_b[week]["prediction_sha256"]
    # the perturbed week-1 outcome changes week-1's SCORE and the week-1 update inputs (next-week only)...
    assert res_a[1]["metrics"]["TEST"] != res_b[1]["metrics"]["TEST"]
    assert be_a.update_inputs[0] != be_b.update_inputs[0]
    # ...and therefore the checkpoint that serves week 2, but never the sealed week-1 forecast bytes
    assert res_a[1]["next_checkpoint"]["id"] != res_b[1]["next_checkpoint"]["id"]
    assert res_a[2]["inference_donor"] == res_a[1]["next_checkpoint"]["id"]


def test_ps04_future_week_targets_never_change_an_earlier_refit_or_prediction(tmp_path):
    (res_a, be_a), (res_b, be_b) = chain(tmp_path / "a"), chain(tmp_path / "b", shift2=50.0)
    assert _tree(tmp_path / "a" / "week1") == _tree(tmp_path / "b" / "week1")          # every week-1 byte
    assert be_a.update_inputs[0] == be_b.update_inputs[0]                              # week-1 refit inputs
    assert res_a[1]["next_checkpoint"] == res_b[1]["next_checkpoint"]
    assert res_a[2]["prediction_sha256"] == res_b[2]["prediction_sha256"]              # week-2 seal (PS03)
    assert res_a[2]["metrics"]["TEST"] != res_b[2]["metrics"]["TEST"]                  # only its score moved
    assert be_a.update_inputs[1] != be_b.update_inputs[1]                              # only week-2's refit


# --------------------------------------------------------------------------- PS05 / edge: SIN_PARES monitor

def test_ps05_rows_outside_the_four_year_window_and_immature_labels_are_refused():
    p = PERIODS[1]
    a, o, fit = arrays(1)
    S.validate_update_span(p, fit, o, [2])
    early = dict(fit, origin_ts=np.array([stamp(p["train_start"]) - 3600, fit["origin_ts"][1], fit["origin_ts"][2]]))
    with pytest.raises(ValueError, match="TRAIN_CUTOFF_OR_LABEL_MATURITY"):
        S.validate_update_span(p, early, o, [2])
    late_label = dict(fit, origin_ts=np.array([stamp(p["train_end"]) - 3600] * 3))   # origin inside, t+2h past cutoff
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
    x_own = S.inference_inputs(a, own)[0]
    x_other = S.inference_inputs(a, other)[0]
    assert not np.array_equal(x_own, x_other)                       # a swapped scaler changes the model input
    donor = dict(initial_donor(), id=c1["id"], scales=own)
    reader = lambda name: copy.deepcopy({"PREDICT": a, "OBSERVED": o, "FIT": fit}[name])  # noqa: E731
    S.execute_week(record(1), donor, tmp_path / "w", reader, Recorder(), input_sha256="in")
    with pytest.raises(ValueError, match="SEALED_FORECAST_IDENTITY"):
        S.execute_week(record(1), dict(donor, id=c2["id"], scales=other), tmp_path / "w", reader, Recorder(),
                       input_sha256="in")


def test_ps07_inference_scaler_is_the_donors_and_refit_scaler_is_the_periods_train_scaler(tmp_path):
    res, be = chain(tmp_path)
    # week 1 inferred with the VALIDATION donor's scaler (input_std 1 -> prediction = price + .1)
    p1 = np.load(tmp_path / "week1" / "PREDICTED.npz")["prediction"]
    np.testing.assert_allclose(p1[:, 0], np.array([1.0, 2.0]) + 0.1)
    # the checkpoint written by week 1's refit carries week 1's TRAIN scaler, and week 2 infers with it
    assert res[1]["next_checkpoint"]["scales"]["scales_sha256"] == record(1)["scales"]["scales_sha256"]
    p2 = np.load(tmp_path / "week2" / "PREDICTED.npz")["prediction"]
    np.testing.assert_allclose(p2[:, 0], np.array([1.0, 2.0]) * record(1)["scales"]["input_std"][0] + 0.1)


# --------------------------------------------------------------------------- PS09 resumption

def test_ps09_crash_between_seal_and_update_resumes_from_the_same_donor_without_repredicting(tmp_path):
    a, o, fit = arrays(1)
    reader = lambda name: copy.deepcopy({"PREDICT": a, "OBSERVED": o, "FIT": fit}[name])  # noqa: E731

    class Crash(Recorder):
        def update(self, *args):
            raise RuntimeError("simulated crash after STARTED")
    be = Crash()
    with pytest.raises(RuntimeError):
        S.execute_week(record(1), initial_donor(), tmp_path, reader, be, input_sha256="in")
    sealed = sha(tmp_path / "PREDICTED.npz")
    be2 = Recorder()
    res = S.execute_week(record(1), initial_donor(), tmp_path, reader, be2, input_sha256="in")
    assert be2.events == [("update", "validation-donor")] and sha(tmp_path / "PREDICTED.npz") == sealed
    assert res["prediction_sha256"] == sealed


def test_ps09_stale_update_receipt_from_another_donor_is_refused_before_any_load(tmp_path):
    """GpuBackend.update: an UPDATE.json left by a different donor is never adopted (no TF model is loaded)."""
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


# --------------------------------------------------------------------------- PS01 initial donors + SIN_PARES

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
    """A parent prepared root with one SKIPPED cell and one COMPLETE cell (tiny arrays)."""
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


def _pair(base, members):
    pair = base / "pair.json"
    W.B.put(pair, dict(families={fam: dict(members=ms) for fam, ms in members.items()}))
    return pair


@pytest.fixture
def world(tmp_path):
    model = dict(MODEL)
    prior, doc = _prior(tmp_path, model)
    members = {}
    for fam in ("short", "long"):
        members[fam] = [_member(tmp_path, f"{fam}-early", "2024-12-15T22:00:00+00:00", "2024-12-20T22:00:00+00:00"),
                        _member(tmp_path, f"{fam}-late", "2024-12-22T22:00:00+00:00", "2024-12-27T22:00:00+00:00")]
    pair = _pair(tmp_path, members)
    spec = dict(prepared_source=str(prior), prepared_source_sha256=doc["sha256"], data_root=str(tmp_path / "data"),
                pair=str(pair))
    return dict(tmp=tmp_path, spec=spec, members=members, model=model)


def test_ps01_initial_donor_is_the_last_validation_member_with_exact_bytes_and_hashes(world):
    with S.context():
        p = S.prepare(world["spec"])
    for fam in ("short", "long"):
        ini = p["initial"][fam]
        late = world["members"][fam][1]
        assert ini["source_member"] == late["id"] and ini["available_by"] == "2024-12-27T22:00:00+00:00"
        assert ini["weights_sha256"] == "1" * 64 and ini["root"] == "data"
        copied = Path(world["spec"]["data_root"]) / "initial" / fam
        assert sha(copied / "forecast_model.keras") == late["files"]["forecast_model.keras"]
        assert ini["scales"]["scales_sha256"] == (f"{fam}-late") * 8          # the donor's OWN scaler
        assert ini["id"] == W.B.digest({k: v for k, v in ini.items() if k != "id"})
    assert p["sha256"] == W.B.digest({k: v for k, v in p.items() if k != "sha256"})
    assert [c["predecessor"] for c in p["cells"]] == [None, "short__A"]
    # the COMPLETE cell got a PREDICT.npz with NO target/mask keys; the SKIPPED one got none
    with np.load(Path(world["spec"]["data_root"]) / "short__B" / "PREDICT.npz") as z:
        assert set(z.files) == S.PREDICT_KEYS
    assert not (Path(world["spec"]["data_root"]) / "short__A" / "PREDICT.npz").exists()


def test_ps01_a_member_finishing_inside_test_or_unverified_is_refused(world):
    late = world["members"]["short"][1]
    base = Path(late["folder"]).parent
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


def test_edge_sin_pares_week_carries_the_donor_forward_without_mae_or_training(world, tmp_path):
    with S.context():
        p = S.prepare(world["spec"])
        gpu_root = tmp_path / "gpu"
        spec = dict(world["spec"], gpu_root=str(gpu_root), gpu_data=world["spec"]["data_root"], cells=p["cells"],
                    population_digest=p["sha256"], batch_id=S.NAME, training=W.B.TRAINING, seed=W.B.SEED,
                    evaluation_split="TEST", qualification=S.QUALIFICATION, code_files={})
        with patch.object(S.GpuBackend, "load", side_effect=AssertionError("no model load for SIN_PARES")):
            S.run(spec)
        r = W.read(gpu_root / "results" / "short__A" / "RESULT.json")
        assert r["state"] == "SKIPPED_NO_EVAL_ROWS" and "metrics" not in r and r["files"] == {}
        assert r["next_checkpoint"] == p["initial"]["short"]                 # checkpoint carried forward unchanged
        assert W.read(gpu_root / "PROGRESS.json")["state"] == "SKIPPED_NO_EVAL_ROWS"
        # the successor becomes eligible on a SKIPPED predecessor; a FAILED terminal would block it
        failed = dict(batch_id=S.NAME, cell=p["cells"][0], state="FAILED", files={}, test_read=True, reason="x")
        failed["sha256"] = W.B.digest(failed)
        W.B.put(gpu_root / "results" / "short__A" / "RESULT.json", failed)
        S.run(spec)
        assert W.read(gpu_root / "PROGRESS.json")["state"] == "BLOCKED_DEPENDENCY"
