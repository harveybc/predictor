"""C36 (order 2026-09-11): the terminal knows where the run wrote.

Two defects, both of them about WHEN a value is read.

  * `results_dir` was read before the body ran. The run only learns
    where it writes once its configuration is merged, so the value was
    always `None`, and an outbox failure left no gap file anywhere —
    precisely the situation the gap file exists for. It is now lazy,
    like `config` and `campaign_key` already were;
  * the gap was written to one fixed filename. Two runs into the same
    results directory meant the second erased the first run's only
    record that the cube never heard from it. The file is now named by
    the envelope digest and created with O_EXCL: one gap per run,
    write-once.

Every terminal state is exercised, with the outbox both working and
failing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from olap import outbox as ob                                # noqa: E402
from olap import terminal as T                               # noqa: E402


@pytest.fixture
def outbox(tmp_path, monkeypatch):
    root = tmp_path / "outbox"
    monkeypatch.setenv(ob.OUTBOX_ENV, str(root))
    ob.ensure_outbox(root)
    return root


@pytest.fixture
def broken_outbox(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("the local outbox is unwritable")
    monkeypatch.setattr(T.ob, "emit", boom)


def run(fn, *, results_dir=None, config=None):
    return T.terminal_run(
        fn, campaign_key=lambda: "predictor::c36",
        producer="predictor",
        config=config if config is not None else (lambda: {}),
        results_dir=results_dir)


# ------------------------------------------------- exactly one terminal
@pytest.mark.parametrize("state", list(T.TERMINAL_STATES))
def test_every_terminal_state_emits_exactly_one_envelope(outbox, state):
    def body():
        if state == T.COMPLETE:
            return {"ok": True}
        if state == T.FAILED:
            raise RuntimeError("boom")
        if state == T.REFUSED:
            raise SystemExit("REFUSED: the gate said no")
        return {"terminal_state": state}

    if state in (T.FAILED, T.REFUSED):
        with pytest.raises(BaseException):
            run(body)
    else:
        run(body)
    entries = ob.pending_entries(outbox)
    assert len(entries) == 1, [p.name for p in entries]
    doc = json.loads(entries[0].read_text())["document"]
    assert doc["terminal"]["state"] == state
    assert doc["result_class"] == T.RESULT_CLASS_FOR[state]


# --------------------------------------------------------- laziness
def test_results_dir_is_resolved_after_the_body_runs(tmp_path, outbox,
                                                     broken_outbox):
    """The heart of the defect: the directory does not exist yet when
    terminal_run is called."""
    shared: dict = {}
    late = tmp_path / "results/phase_1_daily"

    def body():
        shared["results_dir"] = late
        return {"ok": True}

    out = run(body, results_dir=lambda: shared.get("results_dir"))
    assert out is not None
    gaps = sorted(late.glob(f"{T.OPERATIONAL_GAP_STEM}-*.json"))
    assert len(gaps) == 1, (
        "an outbox failure must leave a gap beside the results; with an "
        "eagerly-read results_dir there was nowhere to write it")
    gap = json.loads(gaps[0].read_text())
    assert gap["terminal_state"] == T.COMPLETE
    assert gap["campaign_key"] == "predictor::c36"
    assert gap["run_id"]


def test_an_eagerly_captured_none_writes_no_gap(tmp_path, outbox,
                                                broken_outbox):
    """The PRE behaviour, kept as a test so the regression is named."""
    shared: dict = {}

    def body():
        shared["results_dir"] = tmp_path / "results"
        return {"ok": True}

    run(body, results_dir=shared.get("results_dir"))     # eager: None
    assert not list(tmp_path.rglob(f"{T.OPERATIONAL_GAP_STEM}*")), (
        "this is what the defect looked like")


def test_a_plain_path_still_works(tmp_path, outbox, broken_outbox):
    d = tmp_path / "results"
    run(lambda: {"ok": True}, results_dir=d)
    assert len(list(d.glob(f"{T.OPERATIONAL_GAP_STEM}-*.json"))) == 1


# -------------------------------------------------------- write-once
def test_two_runs_into_one_directory_keep_two_gaps(tmp_path, outbox,
                                                   broken_outbox):
    d = tmp_path / "results"
    run(lambda: {"ok": True}, results_dir=d,
        config=lambda: {"load_config": "a.json"})
    run(lambda: {"terminal_state": T.INCONCLUSIVE}, results_dir=d,
        config=lambda: {"load_config": "b.json"})
    gaps = sorted(d.glob(f"{T.OPERATIONAL_GAP_STEM}-*.json"))
    assert len(gaps) == 2, (
        "a fixed filename let the second run erase the first run's only "
        "evidence that the cube never heard from it")
    states = {json.loads(g.read_text())["terminal_state"] for g in gaps}
    assert states == {T.COMPLETE, T.INCONCLUSIVE}


def test_the_gap_file_is_write_once(tmp_path, outbox, broken_outbox):
    d = tmp_path / "results"
    d.mkdir(parents=True)
    env = {"envelope_sha256": "a" * 64, "campaign_key": "k",
           "identity": {"run_id": "r"}, "terminal": {"state": "COMPLETE"}}
    target = d / T.gap_name(env)
    target.write_text('{"pre_existing": true}')
    out = T.emit_terminal(env, results_dir=d)
    assert out["state"] == "OPERATIONAL_GAP"
    assert json.loads(target.read_text()) == {"pre_existing": True}, (
        "an existing gap for THIS envelope is the same gap, not one to "
        "overwrite")


def test_the_gap_names_the_envelope_it_belongs_to():
    env = {"envelope_sha256": "abcdef0123456789" + "0" * 48}
    assert T.gap_name(env) == \
        f"{T.OPERATIONAL_GAP_STEM}-abcdef0123456789.json"


# ------------------------------------------------- the call site itself
def test_main_binds_results_dir_lazily():
    """The correction must be at the call site, not only in the API."""
    body = (REPO / "app/main.py").read_text()
    assert "results_dir=lambda:" in body, (
        "app/main.py must pass a callable; a value read there is read "
        "before the run has one")
    assert "results_dir=shared.get(\"results_dir\"))" not in body
