"""RP32: the outbox race of an earlier round, reproduced on a fixture. Two concurrent processes were given
the SAME terminal outbox directory; one moved an envelope to sent/ while the other was flushing it, and the
loser died with FileNotFoundError. The fixture reproduces exactly that, then proves what the accounting is
after it: every terminal reaches the server AT LEAST once and lands in sent/ EXACTLY once (no loss, no
duplicate unit in the accounting), and the operational rule that follows — never share one outbox between two
concurrent processes — is stated by the third test, which shows a private outbox per process has no race."""
import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GR = _load("governed_run")
N = 24


def _terminal(i):
    return {"campaign_sha256": "a" * 64, "unit_id": f"unit_{i:03d}",
            "terminal": {"schema": "governed_terminal.v1", "generation": 1, "status": "COMPLETED", "reason": None,
                         "started_at": "2026-09-19T00:00:00Z", "finished_at": "2026-09-19T00:00:01Z",
                         "costs": {"wall_seconds": 1.0, "cpu_seconds": 1.0}, "deliveries": [], "artifacts": [],
                         "metrics": [], "tags": {"i": str(i)}}}


def _fill(root):
    ob = GR.TerminalOutbox(root)
    for i in range(N):
        ob.put(_terminal(i))
    return ob


def _server(received_dir):
    """A server that records every accepted report as one write-once file per (unit, generation)."""
    received_dir = Path(received_dir)
    received_dir.mkdir(parents=True, exist_ok=True)

    def sender(envelope):
        u = envelope["unit_id"]
        path = received_dir / f"{u}.{envelope['terminal']['generation']}.{__import__('os').getpid()}.json"
        path.write_text(json.dumps(envelope))
        return {"terminal_sha256": "b" * 64}
    return sender


def _worker(root, received, out, barrier_file):
    """One process: waits for its sibling to exist, then flushes the SHARED outbox."""
    import time
    Path(barrier_file).write_text("x")
    while len(list(Path(barrier_file).parent.glob("barrier_*"))) < 2:
        time.sleep(0.005)
    ob = GR.TerminalOutbox(root)
    try:
        res = ob.flush(_server(received))
        Path(out).write_text(json.dumps({"ok": True, **res}))
    except BaseException as e:                                  # the race's real signature
        Path(out).write_text(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))


def _run_pair(tmp_path, root_a, root_b):
    ctx = mp.get_context("fork")
    received, bar = tmp_path / "received", tmp_path / "bar"
    bar.mkdir()
    ps = []
    for k, root in enumerate((root_a, root_b)):
        p = ctx.Process(target=_worker, args=(str(root), str(received), str(tmp_path / f"out{k}.json"), str(bar / f"barrier_{k}")))
        p.start()
        ps.append(p)
    for p in ps:
        p.join(60)
    return [json.loads((tmp_path / f"out{k}.json").read_text()) for k in (0, 1)], received


def test_RP32_two_processes_on_one_outbox_race_and_the_loser_dies_on_a_missing_envelope(tmp_path):
    root = tmp_path / "shared"
    _fill(root)
    results, received = _run_pair(tmp_path, root, root)                 # THE SAME directory for both
    errors = [r["error"] for r in results if not r["ok"]]
    failures = [f for r in results if r["ok"] for f in r.get("failures", {}).values()]
    assert errors or failures, f"the race did not occur in this run: {results}"
    assert any("FileNotFoundError" in e for e in errors + failures), f"a different failure: {errors + failures}"


def test_RP32_after_the_race_no_terminal_is_lost_and_none_is_duplicated_in_the_accounting(tmp_path):
    root = tmp_path / "shared"
    _fill(root)
    results, received = _run_pair(tmp_path, root, root)
    ob = GR.TerminalOutbox(root)
    sent = sorted(p.name for p in (root / "sent").glob("*.json"))
    pending = sorted(p.name for p in (root / "pending").glob("*.json"))
    # no loss: every envelope is still exactly one file, in sent/ or pending/, and the two sets do not overlap
    assert len(set(sent) | set(pending)) == N and not (set(sent) & set(pending))
    # no duplicate in the accounting: one unit is one sent envelope (the file name is the envelope's digest)
    units_sent = [json.loads((root / "sent" / n).read_text())["unit_id"] for n in sent]
    assert len(units_sent) == len(set(units_sent))
    # the server may have been told twice about the same unit (at-least-once delivery); its own write-once
    # accounting per (unit, generation) is what makes the repeat harmless, and the repeat is visible:
    got = [p.name.split(".")[0] for p in Path(received).glob("*.json")]
    assert set(units_sent) <= set(got)
    repeats = {u for u in got if got.count(u) > 1}
    assert repeats <= set(units_sent)
    # a second flush of what is left never loses it either
    res = ob.flush(_server(received))
    assert res["pending"] == 0 and len(list((root / "sent").glob("*.json"))) == N


def test_RP32_a_private_outbox_per_process_has_no_race(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _fill(a)
    _fill(b)
    results, received = _run_pair(tmp_path, a, b)
    assert all(r["ok"] and r["sent"] == N and r["pending"] == 0 and not r["failures"] for r in results), results
    assert len(list((a / "sent").glob("*.json"))) == N and len(list((b / "sent").glob("*.json"))) == N
