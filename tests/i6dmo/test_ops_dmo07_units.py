"""DMO07: exactly one validation fit; reports never call fit; two output families are separate units."""
import pytest

from tests.i6dmo.ops_fakes import fit_count, make_campaign, run_worker, trainer_entrypoints


def test_dmo07_plan_has_one_fit_per_family_and_no_per_week_or_per_head_fits(tmp_path):
    from tools.i6dmo import campaign
    root = make_campaign(tmp_path)
    units = campaign.load_plan(root)["units"]
    fits = [u for u in units if u["kind"] == "fit"]
    assert sorted((u["unit_id"], u["family"]) for u in fits) == [("long.fit", "long"), ("short.fit", "short")]
    short = [u for u in units if u["family"] == "short"]
    assert sorted(u["unit_id"] for u in short) == ["short.fit", "short.report"]
    report = next(u for u in units if u["unit_id"] == "short.report")
    assert report["calls_fit"] is False and report["depends_on"] == ["short.fit"]
    assert all(not u["test_open"] for u in units)


def test_dmo07_full_validation_runs_one_fit_per_family_and_reports_do_not_fit(tmp_path):
    from tools.i6dmo import campaign
    counter = tmp_path / "calls.txt"
    root = make_campaign(tmp_path)
    assert run_worker(root, "preferred", counter)["result"] == "COMPLETE"
    assert run_worker(root, "preferred", counter)["result"] == "IDLE"  # nothing else is a GPU fit yet
    campaign.tick(root)
    assert run_worker(root, "coordinator", counter)["unit_id"] == "short.report"
    assert fit_count(counter) == 1
    campaign.tick(root)  # short closed -> long design sealed with its own identity
    plan = {u["unit_id"]: u for u in campaign.load_plan(root)["units"]}
    assert plan["long.fit"]["identity_sha256"] and plan["long.fit"]["identity_sha256"] != plan["short.fit"]["identity_sha256"]
    assert run_worker(root, "preferred", counter)["unit_id"] == "long.fit"
    campaign.tick(root)
    assert run_worker(root, "coordinator", counter)["unit_id"] == "long.report"
    lines = counter.read_text().split()
    assert lines.count("fit:short.fit") == 1 and lines.count("fit:long.fit") == 1
    assert fit_count(counter) == 2 and "report:short.report" in lines


def test_dmo07_report_entry_identical_to_fit_entry_is_refused():
    from tools.i6dmo import campaign
    entries = trainer_entrypoints()
    entries["report"] = list(entries["fit"])
    with pytest.raises(ValueError, match="REPORT_ENTRY_CALLS_FIT"):
        campaign.validate_entrypoints(entries)
