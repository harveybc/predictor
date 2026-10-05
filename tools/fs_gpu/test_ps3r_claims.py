"""Tests for the shared-claim race rule, the skip rule and the relay-cycle settle gate."""

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ps3r_claims as C  # noqa: E402

BATCH = "batch_002"
FEATURE = "fx.test.logret_1h"


def _write_terminal(directory: Path, valid: bool = True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    body = b'{"row_kind":"feature_summary"}\n'
    (directory / "results.jsonl").write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    (directory / "run_manifest.json").write_text(json.dumps({"status": "COMPLETED", "results_sha256": digest if valid else "0" * 64, "wall_seconds": 100.0}))


def _heartbeat(path: Path, cycle: int, age_seconds: int = 0) -> None:
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=age_seconds)
    path.write_text(json.dumps({"schema": "ps3r_relay_heartbeat.v1", "cycle": cycle, "relayed_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "pushes_ok": True}))


def test_arbitrate_home_role_wins_any_conflict_even_when_later():
    claims = {
        "worker_a": {"state": "CLAIMED", "claimed_at_utc": "2026-10-05T04:00:00Z"},
        "worker_b": {"state": "CLAIMED", "claimed_at_utc": "2026-10-05T04:00:30Z"},
    }
    assert C.arbitrate(claims) == "worker_b"
    assert C.arbitrate(claims, home_role="worker_a") == "worker_a"
    claims["worker_b"]["claimed_at_utc"] = "2026-10-05T03:59:59Z"
    assert C.arbitrate(claims) == "worker_b"


def test_arbitrate_without_home_claim_earliest_wins_and_ignores_abandoned_or_bad_timestamps():
    claims = {
        "worker_a": {"state": "CLAIMED", "claimed_at_utc": "2026-10-05T04:00:00Z"},
        "worker_b": {"state": "ABANDONED", "claimed_at_utc": "2026-10-05T03:00:00Z"},
    }
    assert C.arbitrate(claims) == "worker_a"
    claims["worker_a"]["claimed_at_utc"] = "garbage"
    assert C.arbitrate(claims) is None
    assert C.arbitrate({}) is None


def test_decide_skips_valid_terminal_anywhere(tmp_path):
    own = tmp_path / "own"
    peer = tmp_path / "peer"
    _write_terminal(peer)
    result = C.decide(tmp_path / "claims", BATCH, FEATURE, "worker_a", own, [peer])
    assert result["action"] == "SKIP_TERMINAL"
    assert result["terminal"]["kind"] == "COMPLETED"
    assert not (tmp_path / "claims").exists()


def test_decide_does_not_skip_on_digest_mismatch_but_skips_failed_marker(tmp_path):
    own = tmp_path / "own"
    _write_terminal(own, valid=False)
    result = C.decide(tmp_path / "claims", BATCH, FEATURE, "worker_b", own)
    assert result["action"] == "CLAIMED"
    (own / "FAILED.codex.json").write_text("{}")
    result = C.decide(tmp_path / "claims2", BATCH, FEATURE, "worker_b", own)
    assert result["action"] == "SKIP_TERMINAL" and result["terminal"]["kind"] == "FAILED"


def test_decide_skips_visible_peer_claim_and_ignores_legacy(tmp_path):
    root = tmp_path / "claims"
    cell = root / BATCH / FEATURE
    cell.mkdir(parents=True)
    (cell / "claim.json").write_text(json.dumps({"feature_id": FEATURE, "claimed_at_utc": "2026-10-04T04:54:43Z"}))
    first = C.decide(root, BATCH, FEATURE, "worker_a", tmp_path / "out")
    assert first["action"] == "CLAIMED" and first["legacy"]
    second = C.decide(root, BATCH, FEATURE, "worker_b", tmp_path / "out_b")
    assert second["action"] == "SKIP_PEER_CLAIM" and second["peer"] == "worker_a"
    assert (cell / "claim.worker_a.json").is_file()
    assert not (cell / "claim.worker_b.json").exists()


def test_decide_requires_fresh_heartbeat_when_asked(tmp_path):
    beat = tmp_path / "hb.json"
    result = C.decide(tmp_path / "claims", BATCH, FEATURE, "worker_a", tmp_path / "out", heartbeat=beat, max_heartbeat_age=300)
    assert result["action"] == "SKIP_RELAY_STALE" and result["heartbeat_age_seconds"] is None
    _heartbeat(beat, cycle=7, age_seconds=600)
    assert C.decide(tmp_path / "claims", BATCH, FEATURE, "worker_a", tmp_path / "out", heartbeat=beat, max_heartbeat_age=300)["action"] == "SKIP_RELAY_STALE"
    _heartbeat(beat, cycle=7, age_seconds=10)
    result = C.decide(tmp_path / "claims", BATCH, FEATURE, "worker_a", tmp_path / "out", heartbeat=beat, max_heartbeat_age=300)
    assert result["action"] == "CLAIMED" and result["heartbeat_cycle"] == 7


def test_settle_stealer_waits_three_cycles_then_arbitrates(tmp_path):
    root = tmp_path / "claims"
    beat = tmp_path / "hb.json"
    _heartbeat(beat, cycle=10)
    assert C.decide(root, BATCH, FEATURE, "worker_a", tmp_path / "out", heartbeat=beat)["action"] == "CLAIMED"
    assert C.settle(root, BATCH, FEATURE, "worker_a", heartbeat=beat)["action"] == "WAIT"
    _heartbeat(beat, cycle=12)
    assert C.settle(root, BATCH, FEATURE, "worker_a", heartbeat=beat)["action"] == "WAIT"
    _heartbeat(beat, cycle=13)
    result = C.settle(root, BATCH, FEATURE, "worker_a", heartbeat=beat)
    assert result["action"] == "WIN" and result["competitors"] == []


def test_race_stealer_abandons_before_training_home_never_waits(tmp_path):
    root_a = tmp_path / "a"  # worker_a's local claims dir (stealer)
    root_b = tmp_path / "b"  # worker_b's local claims dir (home of batch_002)
    beat_a = tmp_path / "hb_a.json"
    _heartbeat(beat_a, cycle=5)
    a = C.decide(root_a, BATCH, FEATURE, "worker_a", tmp_path / "out_a", heartbeat=beat_a, max_heartbeat_age=300)
    assert a["action"] == "CLAIMED"
    # worker_b claims a few seconds later, before the relay delivered worker_a's claim
    b = C.decide(root_b, BATCH, FEATURE, "worker_b", tmp_path / "out_b", heartbeat=beat_a)
    assert b["action"] == "CLAIMED"
    # the home role does not wait for relay cycles and wins immediately
    assert C.settle(root_b, BATCH, FEATURE, "worker_b", heartbeat=beat_a)["action"] == "WIN"
    C.mark(root_b, BATCH, FEATURE, "worker_b", "RUNNING")
    # the stealer is still inside its gate; the relay delivers the home claim meanwhile
    assert C.settle(root_a, BATCH, FEATURE, "worker_a", heartbeat=beat_a)["action"] == "WAIT"
    C.claim_path(root_a, BATCH, FEATURE, "worker_b").write_text(C.claim_path(root_b, BATCH, FEATURE, "worker_b").read_text())
    _heartbeat(beat_a, cycle=8)
    lose = C.settle(root_a, BATCH, FEATURE, "worker_a", heartbeat=beat_a)
    assert lose["action"] == "LOSE" and lose["winner"] == "worker_b"
    assert json.loads(C.claim_path(root_a, BATCH, FEATURE, "worker_a").read_text())["state"] == "ABANDONED"
    # relayed back, the abandoned claim blocks nobody; the RUNNING home claim blocks the stealer
    C.claim_path(root_b, BATCH, FEATURE, "worker_a").write_text(C.claim_path(root_a, BATCH, FEATURE, "worker_a").read_text())
    assert C.decide(root_a, BATCH, FEATURE, "worker_a", tmp_path / "out_a", heartbeat=beat_a, max_heartbeat_age=300)["action"] == "SKIP_PEER_CLAIM"


def test_stealer_wins_when_no_home_claim_arrives_within_gate(tmp_path):
    root = tmp_path / "claims"
    beat = tmp_path / "hb.json"
    _heartbeat(beat, cycle=1)
    assert C.decide(root, BATCH, FEATURE, "worker_a", tmp_path / "out", heartbeat=beat, max_heartbeat_age=300)["action"] == "CLAIMED"
    _heartbeat(beat, cycle=4)
    assert C.settle(root, BATCH, FEATURE, "worker_a", heartbeat=beat)["action"] == "WIN"


def test_same_second_claims_go_to_home_role(tmp_path):
    root = tmp_path / "claims"
    stamp = "2026-10-05T04:00:00Z"
    for role in C.ROLES:
        C.atomic_write_json(C.claim_path(root, BATCH, FEATURE, role), dict(C.new_claim(role, BATCH, FEATURE), claimed_at_utc=stamp))
    assert C.settle(root, BATCH, FEATURE, "worker_b")["action"] == "WIN"
    assert C.settle(root, BATCH, FEATURE, "worker_a")["action"] == "LOSE"


def test_stale_peer_claim_is_reclaimable_only_when_asked(tmp_path):
    root = tmp_path / "claims"
    old = C.new_claim("worker_b", BATCH, FEATURE)
    old["claimed_at_utc"] = "2026-10-01T00:00:00Z"
    C.atomic_write_json(C.claim_path(root, BATCH, FEATURE, "worker_b"), old)
    assert C.decide(root, BATCH, FEATURE, "worker_a", tmp_path / "out")["action"] == "SKIP_PEER_CLAIM"
    result = C.decide(root, BATCH, FEATURE, "worker_a", tmp_path / "out", stale_after_seconds=36000)
    assert result["action"] == "CLAIMED" and result["reclaimed_from_stale"] == ["worker_b"]
    assert C.settle(root, BATCH, FEATURE, "worker_a", stale_after_seconds=36000)["action"] == "WIN"


def test_completed_peer_claim_counts_as_terminal(tmp_path):
    root = tmp_path / "claims"
    C.atomic_write_json(C.claim_path(root, BATCH, FEATURE, "worker_b"), dict(C.new_claim("worker_b", BATCH, FEATURE), state="COMPLETED", results_sha256="ab" * 32))
    result = C.decide(root, BATCH, FEATURE, "worker_a", tmp_path / "out")
    assert result["action"] == "SKIP_TERMINAL" and result["terminal"]["results_sha256"] == "ab" * 32


def test_mark_reads_terminal_and_keeps_history(tmp_path):
    root = tmp_path / "claims"
    out = tmp_path / "out"
    _write_terminal(out)
    C.decide(root, BATCH, FEATURE, "worker_a", tmp_path / "nothing")
    C.mark(root, BATCH, FEATURE, "worker_a", "RUNNING")
    payload = C.mark(root, BATCH, FEATURE, "worker_a", "COMPLETED", results_sha256=C.valid_terminal(out))
    assert payload["state"] == "COMPLETED" and [h["state"] for h in payload["history"]] == ["CLAIMED", "RUNNING"]
    assert payload["results_sha256"] == hashlib.sha256(b'{"row_kind":"feature_summary"}\n').hexdigest()


def test_ledger_merges_roots_and_flags_contested(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    C.decide(root_a, BATCH, FEATURE, "worker_a", tmp_path / "o1")
    C.decide(root_b, BATCH, FEATURE, "worker_b", tmp_path / "o2")
    C.decide(root_b, BATCH, "other.feature", "worker_b", tmp_path / "o3")
    book = C.ledger({"worker_a": root_a, "worker_b": root_b})
    assert book["counts"]["cells"] == 2
    assert book["contested_cells"] == [f"{BATCH}::{FEATURE}"]
    assert book["cells"][f"{BATCH}::other.feature"]["winner"] == "worker_b"


def test_cli_exit_codes(tmp_path, capsys):
    root = tmp_path / "claims"
    args = ["--claims-root", str(root), "--batch", BATCH, "--feature", FEATURE]
    assert C.main(["decide", *args, "--role", "worker_a", "--out-dir", str(tmp_path / "out"), "--extra", '{"cap":"8000M"}']) == C.EXIT_CLAIMED
    assert C.main(["decide", *args, "--role", "worker_b", "--out-dir", str(tmp_path / "out")]) == C.EXIT_SKIP
    beat = tmp_path / "hb.json"
    _heartbeat(beat, cycle=1)
    assert C.main(["settle", *args, "--role", "worker_a", "--heartbeat", str(beat)]) == C.EXIT_WAIT
    assert C.main(["settle", *args, "--role", "worker_a"]) == C.EXIT_WIN
    assert C.main(["mark", *args, "--role", "worker_a", "--state", "RUNNING"]) == 0
    assert json.loads(C.claim_path(root, BATCH, FEATURE, "worker_a").read_text())["cap"] == "8000M"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
