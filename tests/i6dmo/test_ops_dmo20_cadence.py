"""DMO20: static/quarterly/monthly/weekly have distinct identities; TEST reads cannot influence selection."""
import pytest

from tests.i6dmo.ops_fakes import fit_count, make_campaign, run_worker


def _components(cadence):
    return dict(unit="short.fit", family="short", kind="fit", cadence=cadence, effective_config_sha256="a" * 64,
                data_sha256="b" * 64, code_commit="c" * 40, seed=6102026, upstream=[])


def test_dmo20_cadences_have_distinct_identities():
    from tools.i6dmo import campaign
    ids = {c: campaign.unit_identity(_components(c)) for c in campaign.CADENCES}
    assert set(campaign.CADENCES) == {"NONE", "QUARTERLY", "MONTHLY", "WEEKLY"}
    assert len(set(ids.values())) == 4
    with pytest.raises(ValueError, match="UNKNOWN_CADENCE"):
        campaign.unit_identity(_components("DAILY"))


def test_dmo20_static_terminal_cannot_be_adopted_by_a_cadence_campaign(tmp_path):
    from tools.i6dmo import campaign
    counter = tmp_path / "calls.txt"
    static = make_campaign(tmp_path, name="static")
    run_worker(static, "preferred", counter)
    quarterly = make_campaign(tmp_path, name="quarterly", prior_roots=[static], retraining_frequency="QUARTERLY")
    a = {u["unit_id"]: u for u in campaign.load_plan(static)["units"]}
    b = {u["unit_id"]: u for u in campaign.load_plan(quarterly)["units"]}
    assert a["short.fit"]["identity_sha256"] != b["short.fit"]["identity_sha256"]
    assert campaign.load_plan(static)["campaign_id"] != campaign.load_plan(quarterly)["campaign_id"]
    assert run_worker(quarterly, "preferred", counter)["result"] == "COMPLETE"
    assert fit_count(counter) == 2


def test_dmo20_gate_built_from_test_reads_is_refused(tmp_path):
    from tools.i6dmo import campaign
    root = make_campaign(tmp_path)
    with pytest.raises(campaign.Refused, match="TEST_FEEDBACK"):
        campaign.record_gate(root, "DATA_PROVENANCE", status="PASS", evidence={},
                             splits_read=["train", "validation", "test"])


def test_dmo20_selection_inputs_ignore_test_terminals(tmp_path):
    from tools.i6dmo import campaign, queue
    root = make_campaign(tmp_path)
    for role in ("preferred", "coordinator"):
        campaign.tick(root)
        run_worker(root, role, tmp_path / "c.txt")
    before = campaign.selection_inputs(root)
    assert before["units"] == ["short.report"] and before["eligibility"]["short"]
    q = queue.Queue(root / "state" / "queue.sqlite3")
    q.ensure_unit("final.pair_test", dict(identity_sha256="f" * 64, kind="final_test", family="pair",
                                          device="cpu", depends_on=[], stage="FINAL_PAIR_TEST", test_open=True),
                  state="QUEUED")
    claim = q.claim("final.pair_test", owner="x", role="coordinator", host_key="h", boot_id="b",
                    device_class="cpu", heavy=False, ttl=60)
    q.complete("final.pair_test", claim["token"],
               dict(status="COMPLETE", eligibility={"2": dict(mae=0.0, naive=1.0, eligible=True, n=1)}),
               terminal_sha256="e" * 64)
    q.close()
    assert campaign.selection_inputs(root) == before
