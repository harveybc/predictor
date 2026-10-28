"""The observer cannot confuse reused scoring time with new fitting cost."""

from tools.horizon_campaign_monitor import status
from tools.i6m_queue import Queue, seal
from tools.i6e_horizon_plan import experiment_grid, weekly_schedule


def test_cost_unknown_and_scoped_counters(tmp_path):
    d = seal(dict(arms=experiment_grid()[0], weeks=weekly_schedule(2024), seed=6102026, input_sha256="a"*64))
    q = Queue(tmp_path/"QUEUE.db", d)
    job = q.claim("preferred", "NEURAL")
    value = status(tmp_path, horizon=2, week=0)
    assert value["counts"]["RUNNING"] == 1
    assert value["counts"]["QUEUED"] == 1
    assert value["annual_completion_eta_seconds"] is None
    assert value["active"][0]["slot"] == "preferred"


def test_adoption_is_not_a_measured_fit_or_annual_result(tmp_path):
    d = seal(dict(arms=experiment_grid()[0], weeks=weekly_schedule(2024), seed=6102026, input_sha256="a"*64))
    q = Queue(tmp_path/"QUEUE.db", d)
    job = q.claim("preferred", "NEURAL")
    cell = seal(dict(job_id=job["id"], lease=job["lease"], design_sha256=q.identity, status="COMPLETED",
                     model=job["model"], arm=job["arm"], week=job["week"], reused_fit=True,
                     elapsed_seconds=1, metrics=dict(n=10, mae=.1, naive_mae=.2)))
    q.complete(job, cell)
    observed = status(tmp_path, horizon=2)
    assert observed["timing_classes"][0]["measured_cells"] == 0
    assert observed["results"][0]["scope"] == "PARTIAL_YEAR_NOT_ELIGIBLE"
    assert observed["results"][0]["skill_percent"] == -50
