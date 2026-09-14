"""GOV-N4: a permanently refused terminal is visible, traceable and disposable; it
never disappears, never turns FAILED into COMPLETED, and stops blocking once adjudicated."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_outbox_subject", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_outbox_subject"] = module
    spec.loader.exec_module(module)
    return module


GR = _load("governed_run")


def _terminal(status="COMPLETED", metrics=None, deliveries=("a" * 32,)):
    return {"schema": "governed_terminal.v1", "generation": 1, "status": status,
            "reason": None if status == "COMPLETED" else "X", "started_at": "2026-09-14T00:00:00Z",
            "finished_at": "2026-09-14T00:00:01Z", "costs": {"wall_seconds": 1.0},
            "deliveries": list(deliveries), "artifacts": [], "metrics": metrics or [], "tags": {}}


def _envelope(unit="u1", **kw):
    return {"campaign_sha256": "c" * 64, "unit_id": unit, "terminal": _terminal(**kw)}


def _refuse(text):
    def sender(payload):
        raise GR.GovernedRunError(text)
    return sender


def _accept(payload):
    return {"terminal_sha256": "f" * 64}


def test_failure_classes_never_decide_invalidity_from_a_4xx_alone():
    assert GR.classify_failure("terminal refused: http 503 terminal lake unreachable") == "TRANSIENT"
    assert GR.classify_failure("ConnectionRefusedError: [Errno 111]") == "TRANSIENT"
    assert GR.classify_failure("terminal refused: http 401 unauthorized") == "CONFIGURATION"
    assert GR.classify_failure("terminal refused: http 404 unknown campaign") == "CONFIGURATION"
    assert GR.classify_failure("terminal refused: http 400 invalid metric") == "REFUSED_BY_SERVER"
    assert GR.classify_failure("terminal refused: http 422 completed terminal lacks verified campaign data") == "REFUSED_BY_SERVER"
    assert GR.classify_failure("terminal accounting and terminal lake diverge") == "UNRESOLVED"


def test_flush_records_failures_and_status_tells_the_classes_apart(tmp_path):
    outbox = GR.TerminalOutbox(tmp_path / "outbox")
    bad = outbox.put(_envelope("u1", metrics=[{"metric": "Naive MAE", "split": None, "horizon": None, "unit": None,
                                               "value": 1.0, "std_dev": None, "min_value": None, "max_value": None}]))
    down = outbox.put(_envelope("u2"))
    outbox.flush(lambda p: _refuse("terminal refused: http 400 invalid metric")(p) if p["unit_id"] == "u1"
                 else _refuse("terminal refused: http 503 terminal lake unreachable")(p))
    outbox.flush(lambda p: _refuse("terminal refused: http 400 invalid metric")(p) if p["unit_id"] == "u1"
                 else _refuse("ConnectionRefusedError: down")(p))
    health = outbox.status()
    by_unit = {p["unit_id"]: p for p in health["pending"]}
    assert by_unit["u1"]["class"] == "REFUSED_BY_SERVER" and by_unit["u1"]["attempts"] == 2
    assert by_unit["u2"]["class"] == "TRANSIENT" and by_unit["u2"]["attempts"] == 2
    assert health["awaiting_adjudication"] == 1 and health["recoverable"] == 1 and health["unresolved"] == 0
    assert (bad.path.with_name(bad.path.name[:-5] + ".failure")).is_file()
    # the transient one recovers and its sidecar goes away; the refused one stays
    assert outbox.flush(lambda p: _accept(p) if p["unit_id"] == "u2" else _refuse("terminal refused: http 400 x")(p)) == {
        "sent": 1, "pending": 1, "failures": {bad.path.name: "GovernedRunError: terminal refused: http 400 x"}}
    assert not (down.path.with_name(down.path.name[:-5] + ".failure")).exists()
    assert outbox.status()["sent"] == 1 and outbox.status()["awaiting_adjudication"] == 1


def test_dispose_moves_the_envelope_unchanged_with_a_write_once_record(tmp_path):
    outbox = GR.TerminalOutbox(tmp_path / "outbox")
    item = outbox.put(_envelope("u1"))
    raw = item.path.read_bytes()
    outbox.flush(_refuse("terminal refused: http 400 invalid metric"))
    with pytest.raises(GR.GovernedRunError, match="states its reason"):
        outbox.dispose(item.path.name, "INVALID_ENVELOPE", " ")
    with pytest.raises(GR.GovernedRunError, match="unknown disposition"):
        outbox.dispose(item.path.name, "DELETED", "no")
    with pytest.raises(GR.GovernedRunError, match="needs the accepted successor"):
        outbox.dispose(item.path.name, "SUPERSEDED", "fixed")
    record = outbox.dispose(item.path.name, "INVALID_ENVELOPE", "metric name carries a space; producer fixed in 08a4c04")
    moved = tmp_path / "outbox" / "adjudicated" / item.path.name
    assert moved.read_bytes() == raw and not item.path.exists()
    disposition = json.loads((tmp_path / "outbox" / "adjudicated" / (item.path.name[:-5] + ".disposition.json")).read_text())
    assert disposition == record and disposition["decision"] == "INVALID_ENVELOPE"
    assert disposition["failure"]["class"] == "REFUSED_BY_SERVER" and disposition["envelope_sha256"] == item.path.name[:-5]
    with pytest.raises(GR.GovernedRunError, match="not a pending envelope"):
        outbox.dispose(item.path.name, "INVALID_ENVELOPE", "twice")
    health = outbox.status()
    assert health["pending"] == [] and [a["decision"] for a in health["adjudicated"]] == ["INVALID_ENVELOPE"]
    # nothing was deleted anywhere under the outbox
    assert sorted(p.name for p in (tmp_path / "outbox" / "adjudicated").iterdir()) == sorted(
        [item.path.name, item.path.name[:-5] + ".disposition.json", item.path.name[:-5] + ".failure"])


def test_supersede_keeps_outcome_and_deliveries_and_links_the_successor(tmp_path):
    outbox = GR.TerminalOutbox(tmp_path / "outbox")
    bad_metric = [{"metric": "Naive MAE", "split": "train", "horizon": 9, "unit": None, "value": 1.0,
                   "std_dev": None, "min_value": None, "max_value": None}]
    item = outbox.put(_envelope("u1", status="FAILED", metrics=bad_metric))
    outbox.flush(_refuse("terminal refused: http 400 invalid metric"))
    fixed = _terminal(status="FAILED", metrics=[dict(bad_metric[0], metric="Naive_MAE")])
    with pytest.raises(GR.GovernedRunError, match="keeps the original outcome"):
        outbox.supersede(item.path.name, dict(fixed, status="COMPLETED", reason=None), _accept, "no")
    with pytest.raises(GR.GovernedRunError, match="keeps the original deliveries"):
        outbox.supersede(item.path.name, dict(fixed, deliveries=["b" * 32]), _accept, "no")
    sent_payloads = []

    def accept(payload):
        sent_payloads.append(payload)
        return {"terminal_sha256": "9" * 64}

    record = outbox.supersede(item.path.name, fixed, accept, "metric key canonicalised")
    assert record["decision"] == "SUPERSEDED" and record["successor_terminal_sha256"] == "9" * 64
    assert record["successor_generation"] == 2
    successor = sent_payloads[0]["terminal"]
    assert successor["generation"] == 2 and successor["status"] == "FAILED"
    assert successor["tags"]["supersedes_envelope_sha256"] == item.path.name[:-5]
    health = outbox.status()
    assert health["sent"] == 1 and health["pending"] == [] and len(health["adjudicated"]) == 1
    # a successor the server refuses leaves both envelopes pending, nothing adjudicated
    other = outbox.put(_envelope("u3", metrics=bad_metric))
    with pytest.raises(GR.GovernedRunError, match="not accepted"):
        outbox.supersede(other.path.name, _terminal(metrics=[dict(bad_metric[0], metric="Naive_MAE")]),
                         lambda p: {"error": "still refused"}, "try")
    assert len(outbox.status()["pending"]) == 2 and other.path.is_file()


def test_adjudicated_envelopes_do_not_block_a_new_run_but_pending_ones_do(tmp_path):
    outbox = GR.TerminalOutbox(tmp_path / "outbox")
    item = outbox.put(_envelope("u1"))
    outbox.flush(_refuse("terminal refused: http 400 invalid metric"))
    assert GR._send_pending.__name__ == "_send_pending"
    outbox.dispose(item.path.name, "INVALID_ENVELOPE", "closed")
    assert outbox.flush(_accept) == {"sent": 0, "pending": 0, "failures": {}}
    pending = outbox.put(_envelope("u2"))
    result = outbox.flush(_refuse("ConnectionRefusedError: down"))
    assert result["pending"] == 1 and pending.path.is_file()
    assert outbox.status()["recoverable"] == 1
