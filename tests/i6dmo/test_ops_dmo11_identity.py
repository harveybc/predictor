"""DMO11: completed identity reused; changed config/data/code cannot adopt; interrupted units stay visible."""
from tests.i6dmo.ops_fakes import CODE, facts_provider, fit_count, make_campaign, run_worker


def test_dmo11_same_identity_is_adopted_without_refit(tmp_path):
    counter = tmp_path / "calls.txt"
    first = make_campaign(tmp_path, name="first")
    assert run_worker(first, "preferred", counter)["result"] == "COMPLETE"
    second = make_campaign(tmp_path, name="second", prior_roots=[first])
    out = run_worker(second, "preferred", counter)
    assert out["result"] == "ADOPTED" and out["unit_id"] == "short.fit"
    assert fit_count(counter) == 1


def test_dmo11_changed_code_data_or_config_cannot_adopt(tmp_path):
    counter = tmp_path / "calls.txt"
    first = make_campaign(tmp_path, name="first")
    run_worker(first, "preferred", counter)
    code = make_campaign(tmp_path, name="code", prior_roots=[first], code_commit="e" * 40)
    data = make_campaign(tmp_path, name="data", prior_roots=[first], data_pins={"corpus": {"sha256": "f" * 64}})
    config = make_campaign(tmp_path, name="config", prior_roots=[first], training__learning_rate=0.0005)
    for root in (code, data, config):
        assert run_worker(root, "preferred", counter)["result"] == "COMPLETE"
    assert fit_count(counter) == 4


def test_dmo11_tampered_prior_terminal_is_rejected(tmp_path):
    import json
    from tools.i6dmo import campaign
    counter = tmp_path / "calls.txt"
    first = make_campaign(tmp_path, name="first")
    run_worker(first, "preferred", counter)
    terminal = next((first / "terminals").glob("*/TERMINAL.json"))
    doc = json.loads(terminal.read_text())
    doc["identity"]["code_commit"] = "9" * 40  # identity components no longer hash to identity_sha256
    terminal.write_text(json.dumps(doc))
    second = make_campaign(tmp_path, name="second", prior_roots=[first])
    assert run_worker(second, "preferred", counter)["result"] == "COMPLETE"
    assert fit_count(counter) == 2
    assert any(e["event"] == "PRIOR_REJECTED" for e in campaign.events(second))


def test_dmo11_interrupted_attempt_remains_visible_and_recovers(tmp_path):
    from tools.i6dmo import queue, status, worker
    counter = tmp_path / "calls.txt"
    root = make_campaign(tmp_path)
    q = queue.Queue(root / "state" / "queue.sqlite3", clock=lambda: 1000.0)
    claimed = q.claim("short.fit", owner="lost-supervisor", role="preferred", host_key="host-preferred",
                      boot_id="boot-1", device_class="rtx5090", heavy=True, ttl=10)
    assert claimed["result"] == "CLAIMED"
    q.close()
    recovered = worker.recover_role(root, "preferred", executor=worker.LocalExecutor(),
                                    facts_provider=facts_provider(), clock=lambda: 2000.0)
    assert recovered["interrupted"] == ["short.fit"]
    unit = status.compute(root, level="unit", unit="short.fit", now=2000.0)
    assert unit["state"] == "INTERRUPTED" and unit["attempts"][0]["outcome"] == "INTERRUPTED"
    camp = status.compute(root, level="campaign", now=2000.0)
    assert camp["state"] != "COMPLETE" and camp["counts"]["INTERRUPTED"] == 1
    assert run_worker(root, "preferred", counter, clock=lambda: 2001.0)["result"] == "COMPLETE"
    unit = status.compute(root, level="unit", unit="short.fit", now=2002.0)
    assert [a["outcome"] for a in unit["attempts"]] == ["INTERRUPTED", "COMPLETE"]
    assert unit["attempts"][1]["kind"] == "RECOVERY"
    assert CODE  # identity pinned by the helper
