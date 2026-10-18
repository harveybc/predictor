"""Part A of ORDEN_SATOSHI_OPERACION.md: bounded transient recovery (RECOVERY_DESIGN.md).

Synthetic CPU fixtures only: fake backend, injected clock, no sleep, no poll, no GPU, no TEST data,
no campaign root. The enforcement under test is ``prequential_test.run`` + ``prequential_recovery``.
"""
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.i6dmo import prequential_recovery as R, prequential_test as S, weekly18 as W

BASE = "9beb31b6"
SCIENTIFIC = ("tools/i6dmo/model.py", "tools/i6dmo/train.py", "tools/i6dmo/weekly_test_data.py",
              "tools/i6dmo/continuation.py", "tools/i6dmo/continuation_data.py", "tools/i6dmo/continuation_models.py",
              "tools/i6dmo/continuation_strategy.py", "tools/i6dmo/continuation_close_operator.py",
              "tools/i6dmo/data.py", "tools/i6dmo/market_grid.py", "tools/i6dmo/weekly_test_source.py")
stamp = lambda v: int(S.CD.data._instant(v).timestamp())  # noqa: E731
PERIODS = {
    1: dict(id="WEEKLY-2025-01-05", train_start="2021-01-05T22:00:00+00:00", train_end="2025-01-05T22:00:00+00:00",
            validation_end="2025-01-10T22:00:00+00:00"),
    2: dict(id="WEEKLY-2025-01-12", train_start="2021-01-12T22:00:00+00:00", train_end="2025-01-12T22:00:00+00:00",
            validation_end="2025-01-17T22:00:00+00:00"),
}


def _arrays(week):
    t0 = stamp(PERIODS[week]["train_end"]) + 12 * 3600
    origins = np.array([t0, t0 + 7200])
    a = dict(x=np.ones((2, 13, 1)), offsets_hours=np.tile(np.arange(-24, 1, 2), (2, 1)),
             closure_crossed=np.zeros((2, 13), bool), origin_ts=origins, row_ids=origins,
             origin_price=np.array([1.0, 2.0]))
    o = dict(a, mask=np.ones((2, 1), bool), target_price=a["origin_price"][:, None] + 0.2)
    return a, o, dict(o)


@pytest.fixture
def world(tmp_path):
    """Two families x two weeks, prepared data + initial donors, all synthetic."""
    data = tmp_path / "data"
    cells, files, previous = [], {}, {}
    for week in (1, 2):
        for family in ("short", "long"):
            cid = f"{family}__{PERIODS[week]['id']}"
            cell = dict(id=cid, family=family, horizons=[2], period=PERIODS[week], stage="prequential_test",
                        predecessor=previous.get(family))
            previous[family] = cid
            cells.append(cell)
            folder = data / cid
            folder.mkdir(parents=True)
            a, o, fit = _arrays(week)
            np.savez_compressed(folder / "PREDICT.npz", **a)
            np.savez_compressed(folder / "OBSERVED.npz", **o)
            np.savez_compressed(folder / "FIT.npz", **fit)
            W.B.put(folder / "PERIOD.json", dict(cell=cell, parent_state="COMPLETE", model=dict(horizons_hours=[2]),
                                                 scales=dict(input_mean=[0.0], input_std=[1.0], naive_mae=[0.1],
                                                             scales_sha256="s" * 64),
                                                 naive_cache={}, population={}))
            for name in ("PREDICT.npz", "OBSERVED.npz", "FIT.npz", "PERIOD.json"):
                files[cid + "/" + name] = W.file_sha256(folder / name)
    initial = {f: dict(id=f"initial-{f}", root="data", model=dict(horizons_hours=[2]), files={},
                       scales=dict(input_mean=[0.0], input_std=[1.0], naive_mae=[0.1], scales_sha256="i" * 64),
                       weights_sha256="w" * 64, available_by="2024-12-27T22:00:00+00:00") for f in ("short", "long")}
    doc = dict(schema="i6dmo.prequential_test.v1", cells=cells, files=files, initial=initial)
    doc["sha256"] = W.B.digest(doc)
    W.B.put(data / "PREPARED.json", doc)
    gpu_root = tmp_path / "gpu"
    spec = dict(gpu_root=str(gpu_root), gpu_data=str(data), cells=cells, population_digest=doc["sha256"],
                batch_id=S.NAME, training=W.B.TRAINING, seed=W.B.SEED, evaluation_split="TEST",
                qualification=S.QUALIFICATION, code_files={})
    return dict(spec=spec, root=gpu_root, data=data, cells=cells)


class Clock:
    def __init__(self, start=1_000.0):
        self.now = start

    def __call__(self):
        return self.now


class FakeBackend:
    """Deterministic prediction from inputs; the update mirrors the real receipt/adoption contract."""

    def __init__(self, failures=None, target="short__WEEKLY-2025-01-05"):
        self.failures = list(failures or [])   # exceptions raised by successive update calls of ``target``
        self.target = target
        self.fits, self.predicts, self.donors = 0, 0, []
        self.raise_after_receipt = False

    def predict(self, donor, a):
        self.predicts += 1
        return a["origin_price"][:, None] + 0.1

    def update(self, donor, fit, observed, record, folder):
        self.donors.append(dict(id=donor["id"], seed=W.B.SEED, model=record["model"], scales=record["scales"]))
        receipt = Path(folder) / "UPDATE.json"
        if receipt.exists():
            r = W.read(receipt)
            if r["donor_id"] != donor["id"]:
                raise ValueError("UPDATE_DONOR_CHANGED")
            return r                                           # adopted, never refit
        exc = self.failures.pop(0) if self.failures and record["cell"]["id"] == self.target else None
        if exc is not None and not self.raise_after_receipt:
            raise exc
        self.fits += 1
        Path(folder).mkdir(parents=True, exist_ok=True)
        r = dict(next_checkpoint=dict(id="after-" + donor["id"], root="worker", files={},
                                      scales=record["scales"], available_by=record["cell"]["period"]["validation_end"],
                                      model=record["model"], weights_sha256="n" * 64),
                 donor_id=donor["id"], epochs=3, best_epoch=1, stop_reason="patience", training_seconds=1.0,
                 parameters=1, initial_weights_sha256=donor["weights_sha256"])
        W.B.put(receipt, r)
        if exc is not None:
            raise exc
        return r


def _run(world, backend, clock, gate=None):
    gate = gate or (lambda spec: dict(foreign_gpu_processes=0, memory_available=16 * 1024 ** 3, cause=None))
    with S.context():
        S.run(world["spec"], clock=clock, backend_factory=lambda spec, root: backend, gate=gate)
    return W.read(world["root"] / "PROGRESS.json")


def _results(world):
    with S.context():
        return {c["id"]: r for c in world["cells"] if (r := W.terminal(world["root"], c, world["spec"]))}


def _ledger(world, cid):
    return R.ledger(world["root"] / "results" / cid)


# --------------------------------------------------------------------------- classification
def test_classification_is_by_type_and_typed_code_never_by_message():
    gpu = R.TransientProcessError("GPU_NOT_VISIBLE", "the retained runtime is right but enumerates no GPU")
    assert R.classify(gpu) == (R.TRANSIENT_PROCESS, "GPU_NOT_VISIBLE")
    assert R.classify(ValueError("RUNTIME_VERSION_MISMATCH")) == (R.PERMANENT, "RUNTIME_VERSION_MISMATCH")
    # the same message on both types: the TYPE decides
    assert R.classify(ValueError("RUNTIME_VERSION_MISMATCH: GPU temporarily unavailable"))[0] == R.PERMANENT
    assert R.classify(R.TransientProcessError("GPU_NOT_VISIBLE", "RUNTIME_VERSION_MISMATCH"))[0] == R.TRANSIENT_PROCESS
    for code in ("SEALED_FORECAST_IDENTITY", "ARTIFACT_DIGEST:x/y", "DONOR_NOT_YET_AVAILABLE",
                 "TRAIN_CUTOFF_OR_LABEL_MATURITY", "DONOR_WEIGHTS_CHANGED"):
        assert R.classify(ValueError(code))[0] == R.PERMANENT
    assert R.classify(ValueError("SOME_NEW_CODE"))[0] == R.UNKNOWN      # unlisted code: never assumed transient
    assert R.classify(TypeError("x"))[0] == R.UNKNOWN
    assert R.classify(TimeoutError())[0] == R.TRANSIENT_PROCESS
    assert R.classify(MemoryError())[0] == R.PERMANENT
    assert R.MAX_RETRIES == 2 and R.COOLDOWNS == (30, 60)


def test_runtime_version_and_gpu_visibility_are_split_by_type(monkeypatch, tmp_path):
    """GpuBackend.load: a version mismatch is a PERMANENT ValueError; no GPU is a TransientProcessError."""
    import types
    fake_tf = types.SimpleNamespace(__version__="2.21.0", config=types.SimpleNamespace(list_physical_devices=lambda k: []))
    fake_models = types.SimpleNamespace(tf=fake_tf)
    monkeypatch.setitem(__import__("sys").modules, "tools.i6dmo.continuation_models", fake_models)
    monkeypatch.setattr(S.importlib.metadata, "version", lambda name: "3.13.2")
    backend = S.GpuBackend(dict(gpu_data=str(tmp_path)), tmp_path)
    with pytest.raises(R.TransientProcessError, match="GPU_NOT_VISIBLE"):
        backend.load(dict(root="worker", files={}, archive="x.keras"))
    fake_tf.__version__ = "2.18.0"
    with pytest.raises(ValueError, match="RUNTIME_VERSION_MISMATCH"):
        backend.load(dict(root="worker", files={}, archive="x.keras"))


# --------------------------------------------------------------------------- recovery
def test_transient_after_seal_recovers_with_identical_prediction_and_same_donor(world):
    clock = Clock()
    backend = FakeBackend(failures=[R.TransientProcessError("GPU_NOT_VISIBLE")])
    cid = "short__WEEKLY-2025-01-05"
    p = _run(world, backend, clock)
    assert p["state"] == "RETRY_PENDING" and p["cell"] == cid and p["attempt"] == 1
    assert p["next_attempt_at"] == clock.now + 30
    folder = world["root"] / "results" / cid
    assert not (folder / "RESULT.json").exists()                       # a transient is never a terminal
    sealed = W.file_sha256(folder / "PREDICTED.npz")
    seal = W.read(folder / "PREDICTION.json")
    assert (folder / "attempts" / "a01" / "FAILURE.json").is_file()   # evidence kept
    # during the cooldown the independent family's eligible week runs
    p = _run(world, backend, clock)
    assert p["state"] == "COMPLETE" and p["cell"] == "long__WEEKLY-2025-01-05"
    clock.now += 30
    p = _run(world, backend, clock)
    assert p["state"] == "COMPLETE" and p["cell"] == cid and p["attempt"] == 2
    assert backend.predicts == 2                                        # one per family: the retry never re-predicted
    assert W.file_sha256(folder / "PREDICTED.npz") == sealed and W.read(folder / "PREDICTION.json") == seal
    r = _results(world)[cid]
    assert r["state"] == "COMPLETE" and r["prediction_sha256"] == sealed and r["inference_donor"] == "initial-short"
    short_donors = [d for d in backend.donors if d["id"] == "initial-short"]
    assert len(short_donors) == 2 and short_donors[0] == short_donors[1]   # same donor, seed, model, scales
    led = _ledger(world, cid)["attempts"]
    assert [a["outcome"] for a in led] == ["TRANSIENT", "COMPLETE"] and led[0]["cause_code"] == "GPU_NOT_VISIBLE"
    assert R.LEDGER not in r["files"] and not any(k.startswith("attempts/") for k in r["files"])


def test_cooldown_is_enforced_by_the_injected_clock_without_sleeping(world, monkeypatch):
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *a: pytest.fail("sleep called"))
    clock = Clock()
    backend = FakeBackend(failures=[TimeoutError("io"), TimeoutError("io")])
    cid = "short__WEEKLY-2025-01-05"
    assert _run(world, backend, clock)["state"] == "RETRY_PENDING"
    # the long family runs during the cooldown; once it is past its own eligible work, nothing runs
    assert _run(world, backend, clock)["cell"] == "long__WEEKLY-2025-01-05"
    assert _run(world, backend, clock)["cell"] == "long__WEEKLY-2025-01-12"
    clock.now += 29
    p = _run(world, backend, clock)
    assert p["state"] == "RETRY_PENDING" and p["cell"] == cid and p["next_attempt_at"] == 1_030.0
    assert len(_ledger(world, cid)["attempts"]) == 1                    # nothing attempted inside the cooldown
    clock.now += 1
    assert _run(world, backend, clock)["state"] == "RETRY_PENDING"      # attempt 2 failed: cooldown 60
    assert _ledger(world, cid)["attempts"][-1]["next_attempt_at"] == clock.now + 60
    clock.now += 59
    assert _run(world, backend, clock)["state"] == "RETRY_PENDING"
    clock.now += 1
    assert _run(world, backend, clock)["state"] == "COMPLETE"
    assert [a["outcome"] for a in _ledger(world, cid)["attempts"]] == ["TRANSIENT", "TRANSIENT", "COMPLETE"]


def test_transient_budget_exhaustion_is_an_explicit_terminal(world):
    clock = Clock()
    backend = FakeBackend(failures=[R.TransientProcessError("GPU_NOT_VISIBLE")] * 3)
    cid = "short__WEEKLY-2025-01-05"
    for wait in (0, 30, 60):
        clock.now += wait
        p = _run(world, backend, clock)
        assert p["cell"] == cid, p
    assert p["state"] == "FAILED"
    r = _results(world)[cid]
    assert r["state"] == "FAILED" and r["cause_class"] == R.TRANSIENT_PROCESS and r["cause_code"] == "GPU_NOT_VISIBLE"
    assert r["reason"].startswith("TRANSIENT_EXHAUSTED:") and "audited recovery" in r["required_action"]
    assert [a["outcome"] for a in r["attempts"]] == ["TRANSIENT", "TRANSIENT", "EXHAUSTED"]
    assert (world["root"] / "results" / cid / "PREDICTED.npz").is_file()   # the sealed prediction is retained
    # the independent family drains; then the successor is blocked, typed; nothing is retried
    while (p := _run(world, backend, clock))["state"] == "COMPLETE":
        assert p["cell"].startswith("long__")
    assert p["state"] == "BLOCKED_DEPENDENCY" and p["cells"]["short__WEEKLY-2025-01-12"]["dependency"] == cid
    assert len(_ledger(world, cid)["attempts"]) == 3


@pytest.mark.parametrize("exc,cause_class,code", [
    (ValueError("ARTIFACT_DIGEST:update/x"), R.PERMANENT, "ARTIFACT_DIGEST"),
    (ValueError("DONOR_NOT_YET_AVAILABLE"), R.PERMANENT, "DONOR_NOT_YET_AVAILABLE"),
    (ValueError("RUNTIME_VERSION_MISMATCH"), R.PERMANENT, "RUNTIME_VERSION_MISMATCH"),
    (TypeError("unexpected"), R.UNKNOWN, "TypeError"),
])
def test_permanent_and_unknown_causes_are_never_retried(world, exc, cause_class, code):
    clock = Clock()
    backend = FakeBackend(failures=[exc])
    cid = "short__WEEKLY-2025-01-05"
    p = _run(world, backend, clock)
    assert p["state"] == "FAILED" and p["cell"] == cid
    r = _results(world)[cid]
    assert r["cause_class"] == cause_class and r["cause_code"] == code and len(r["attempts"]) == 1
    assert r["attempts"][0]["next_attempt_at"] is None
    clock.now += 10_000
    for _ in range(3):
        _run(world, backend, clock)
    assert len(_ledger(world, cid)["attempts"]) == 1 and backend.fits == 2   # only the two long weeks fitted
    assert _results(world)[cid]["cause_code"] == code                          # the terminal is untouched


def test_completed_update_is_adopted_on_retry_without_a_second_fit(world):
    clock = Clock()
    backend = FakeBackend(failures=[R.TransientProcessError("GPU_NOT_VISIBLE")])
    backend.raise_after_receipt = True                                 # the fit completed, the receipt is durable
    cid = "short__WEEKLY-2025-01-05"
    assert _run(world, backend, clock)["state"] == "RETRY_PENDING"
    assert (world["root"] / "results" / cid / "update" / "UPDATE.json").is_file() and backend.fits == 1
    clock.now += 30
    p = _run(world, backend, clock)
    assert p["cell"] == cid and p["state"] == "COMPLETE" and p["attempt"] == 2
    r = _results(world)[cid]
    assert r["state"] == "COMPLETE" and r["next_checkpoint"]["id"] == "after-initial-short"
    assert backend.fits == 1                                           # the retry adopted the receipt: no second fit


def test_other_family_runs_during_a_cooldown_and_never_skips_its_own_dependency(world):
    clock = Clock()
    backend = FakeBackend(failures=[R.TransientProcessError("GPU_NOT_VISIBLE")])
    assert _run(world, backend, clock)["state"] == "RETRY_PENDING"
    order = []
    for _ in range(3):
        p = _run(world, backend, clock)
        order.append((p["state"], p.get("cell")))
    assert order[:2] == [("COMPLETE", "long__WEEKLY-2025-01-05"), ("COMPLETE", "long__WEEKLY-2025-01-12")]
    assert order[2][0] == "RETRY_PENDING"                              # short week 2 never ran ahead of week 1
    assert "short__WEEKLY-2025-01-12" not in _results(world)
    view = R.classify_pending(world["cells"], _results(world), world["root"], clock.now)
    assert view["short__WEEKLY-2025-01-12"] == dict(state="WAIT_DEPENDENCY", dependency="short__WEEKLY-2025-01-05")


def test_ledger_resumes_attempt_numbering_after_a_simulated_restart(world):
    clock = Clock()
    cid = "short__WEEKLY-2025-01-05"
    assert _run(world, FakeBackend(failures=[TimeoutError()]), clock)["state"] == "RETRY_PENDING"
    before = json.loads((world["root"] / "results" / cid / R.LEDGER).read_text())
    # a new process: fresh backend object, fresh module state, only the ledger on disk persists
    clock.now += 30
    fresh = FakeBackend(failures=[TimeoutError()])
    while _run(world, fresh, clock).get("cell") != cid:
        pass
    after = R.ledger(world["root"] / "results" / cid)["attempts"]
    assert after[0] == before["attempts"][0] and after[1]["index"] == 2 and after[1]["next_attempt_at"] == clock.now + 60
    clock.now += 60
    while _run(world, FakeBackend(), clock).get("cell") != cid:
        pass
    assert [a["index"] for a in _ledger(world, cid)["attempts"]] == [1, 2, 3]
    assert _results(world)[cid]["state"] == "COMPLETE"


def test_abrupt_interruption_is_unknown_and_cannot_silently_restart_a_fit(world):
    clock = Clock()
    cid = world["cells"][0]["id"]
    folder = world["root"] / "results" / cid
    R.begin_attempt(folder, clock.now)
    backend = FakeBackend()
    p = _run(world, backend, clock)
    assert p["state"] == "FAILED" and p["cell"] == cid
    r = _results(world)[cid]
    assert r["cause_class"] == R.UNKNOWN and len(r["attempts"]) == 1
    assert r["cause_code"] == "InterruptedAttempt"
    assert backend.fits == backend.predicts == 0
    raw = (folder / "RESULT.json").read_bytes()
    _run(world, backend, clock)
    assert (folder / "RESULT.json").read_bytes() == raw


@pytest.mark.parametrize("exc", [ValueError("DONOR_WEIGHTS_CHANGED"), TypeError("unknown"),
                                R.TransientProcessError("GPU_NOT_VISIBLE")])
def test_crash_after_terminal_ledger_does_not_reopen_an_attempt(world, exc):
    clock = Clock()
    cid = world["cells"][0]["id"]
    folder = world["root"] / "results" / cid
    for index in range(3 if isinstance(exc, R.TransientProcessError) else 1):
        attempt = R.begin_attempt(folder, clock.now)
        failure = R.record_failure(folder, attempt, exc, clock.now)
        clock.now += (30, 60, 60)[index]
    assert not failure["retry"]
    raw = (folder / R.LEDGER).read_bytes()
    backend = FakeBackend()
    assert _run(world, backend, clock)["state"] == "FAILED"
    assert backend.fits == backend.predicts == 0
    assert (folder / R.LEDGER).read_bytes() == raw


def test_crash_after_completed_ledger_reuses_update_without_an_extra_attempt(world, monkeypatch):
    clock = Clock()
    backend = FakeBackend()
    original = R.finish_attempt
    def killed_after_finish(*args, **kwargs):
        original(*args, **kwargs)
        raise SystemExit("simulated process exit before RESULT")
    with monkeypatch.context() as m:
        m.setattr(R, "finish_attempt", killed_after_finish)
        with pytest.raises(SystemExit):
            _run(world, backend, clock)
    cid = world["cells"][0]["id"]
    folder = world["root"] / "results" / cid
    sealed = W.file_sha256(folder / "PREDICTED.npz")
    ledger_bytes = (folder / R.LEDGER).read_bytes()
    assert _run(world, backend, clock)["state"] == "COMPLETE"
    assert backend.fits == backend.predicts == 1
    assert W.file_sha256(folder / "PREDICTED.npz") == sealed
    assert (folder / R.LEDGER).read_bytes() == ledger_bytes


def test_resource_wait_has_no_finite_eta(world):
    clock = Clock()
    busy = lambda spec: dict(foreign_gpu_processes=1, memory_available=16 * 1024 ** 3, cause=None)
    _run(world, FakeBackend(), clock, gate=busy)
    out = S.enrich_status(dict(state="RUNNING", eta=dict(seconds=99.0)),
                          world["spec"], world["root"], {}, now=clock.now)
    assert out["state"] == "WAIT_RESOURCES"
    assert not out["recovery"]["eta_estimable"] and out["eta"]["seconds"] is None


@pytest.mark.parametrize("fault", ["missing_receipt", "changed_donor"])
def test_completed_attempt_finalization_never_refits_or_rewrites_ledger(world, monkeypatch, fault):
    clock = Clock()
    backend = FakeBackend()
    original = R.finish_attempt
    def killed_after_finish(*args, **kwargs):
        original(*args, **kwargs)
        raise SystemExit("before RESULT")
    with monkeypatch.context() as m:
        m.setattr(R, "finish_attempt", killed_after_finish)
        with pytest.raises(SystemExit):
            _run(world, backend, clock)
    cid = world["cells"][0]["id"]
    folder = world["root"] / "results" / cid
    receipt = folder / "update" / "UPDATE.json"
    if fault == "missing_receipt":
        receipt.unlink()
    else:
        W.B.put(receipt, dict(W.read(receipt), donor_id="incorrect"))
    raw = (folder / R.LEDGER).read_bytes()
    assert _run(world, backend, clock)["state"] == "FAILED"
    assert backend.fits == backend.predicts == 1
    assert (folder / R.LEDGER).read_bytes() == raw
    assert (folder / "FINALIZATION_FAILURE.json").is_file()


def test_historical_failed_terminal_is_never_deleted_or_retried(world):
    clock = Clock()
    cid = "short__WEEKLY-2025-01-05"
    failed = dict(batch_id=S.NAME, cell=world["cells"][0], state="FAILED", files={}, test_read=True,
                  reason="ValueError:SEALED_FORECAST_IDENTITY")
    failed["sha256"] = W.B.digest(failed)
    path = world["root"] / "results" / cid / "RESULT.json"
    W.B.put(path, failed)
    raw = path.read_bytes()
    backend = FakeBackend()
    for _ in range(4):
        clock.now += 100
        _run(world, backend, clock)
    assert path.read_bytes() == raw and not (world["root"] / "results" / cid / R.LEDGER).exists()
    assert backend.fits == 2 and "short__WEEKLY-2025-01-12" not in _results(world)
    assert W.read(world["root"] / "PROGRESS.json")["state"] == "BLOCKED_DEPENDENCY"


def test_resource_block_consumes_no_attempt_and_is_typed(world):
    clock = Clock()
    backend = FakeBackend()
    busy = lambda spec: dict(foreign_gpu_processes=1, memory_available=16 * 1024 ** 3, cause=None)  # noqa: E731
    p = _run(world, backend, clock, gate=busy)
    assert p["state"] == "WAIT_RESOURCES" and p["foreign_gpu_processes"] == 1 and backend.fits == 0
    assert not (world["root"] / "results" / "short__WEEKLY-2025-01-05" / R.LEDGER).exists()
    down = lambda spec: dict(foreign_gpu_processes=None, memory_available=None, cause="NVIDIA_SMI_UNAVAILABLE:OSError")  # noqa: E731
    assert _run(world, backend, clock, gate=down)["cause"].startswith("NVIDIA_SMI_UNAVAILABLE")


# --------------------------------------------------------------------------- status
def test_status_distinguishes_states_and_never_promises_an_eta_under_a_block(world):
    clock = Clock()
    backend = FakeBackend(failures=[R.TransientProcessError("GPU_NOT_VISIBLE")] * 3)
    _run(world, backend, clock)                                        # short W1 attempt 1 -> RETRY_PENDING
    results = _results(world)
    tick_state = dict(state="RUNNING", eta=dict(kind="MEASURED_FIT_MEAN", seconds=99.0))
    out = S.enrich_status(dict(tick_state), world["spec"], world["root"], results, now=clock.now)
    cells = out["recovery"]["cells"]
    assert out["state"] == "RETRY_PENDING" and out["eta"] == dict(kind="NOT_ESTIMABLE", seconds=None,
                                                                   note=out["recovery"]["eta_note"])
    assert cells["short__WEEKLY-2025-01-05"]["state"] == "RETRY_PENDING"
    assert cells["short__WEEKLY-2025-01-05"]["next_attempt_at"] == clock.now + 30
    assert cells["short__WEEKLY-2025-01-12"] == dict(state="WAIT_DEPENDENCY", dependency="short__WEEKLY-2025-01-05")
    assert cells["long__WEEKLY-2025-01-05"]["state"] == "ELIGIBLE"
    # exhaust: terminal block, successor BLOCKED_DEPENDENCY, campaign BLOCKED (never RUNNING)
    for wait in (30, 60):
        clock.now += wait
        while _run(world, backend, clock).get("cell") != "short__WEEKLY-2025-01-05":
            pass
    results = _results(world)
    W.B.progress(world["root"], state="RUNNING", cell="long__WEEKLY-2025-01-12")   # a stale/live RUNNING child
    out = S.enrich_status(dict(tick_state), world["spec"], world["root"], results, now=clock.now)
    cells = out["recovery"]["cells"]
    assert out["state"] == "BLOCKED" and out["eta"]["kind"] == "NOT_ESTIMABLE" and out["eta"]["seconds"] is None
    blocked = cells["short__WEEKLY-2025-01-05"]
    assert blocked["state"] == "BLOCKED_TERMINAL" and blocked["cause_code"] == "GPU_NOT_VISIBLE"
    assert blocked["attempts"] == 3 and blocked["dependents"] == ["short__WEEKLY-2025-01-12"]
    assert cells["short__WEEKLY-2025-01-12"]["state"] == "BLOCKED_DEPENDENCY"
    assert out["recovery"]["blocked"] == ["short__WEEKLY-2025-01-05", "short__WEEKLY-2025-01-12"]
    # no block: the tick's own measured ETA and RUNNING state are left alone
    ok = S.enrich_status(dict(tick_state), world["spec"], world["root"], {}, now=clock.now)
    assert ok["state"] == "RUNNING" and ok["eta"]["seconds"] == 99.0 and ok["recovery"]["eta_estimable"]


def test_scientific_files_are_byte_identical_to_the_base():
    for rel in SCIENTIFIC:
        base = subprocess.check_output(["git", "show", BASE + ":" + rel], cwd=W.REPO)
        assert (W.REPO / rel).read_bytes() == base, rel
    assert S.TRAINING == dict(batch_size=256, learning_rate=.001, weight_decay=.0001, patience=10,
                              min_delta=1e-8, max_epochs=200) and W.B.SEED == 6102026
    assert "max_epochs" not in open(R.__file__).read() and "MemoryMax" not in open(R.__file__).read()
