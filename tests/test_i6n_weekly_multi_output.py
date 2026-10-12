"""Lane C: direct multi-output arms manifest (plan section 4.6), synthetic calendar only.

A template bound to search finalists resolved at stage activation; never a
transferred 'optimal' recipe. No fit, no prices, no TEST rows.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from tools.i6n_contracts import eligibility as c1
from tools import i6n_weekly_campaign as wc
from tools import i6n_weekly_multi_output as mo

UTC = timezone.utc
H = timedelta(hours=1)
DEV = c1.SplitBounds("DEVELOPMENT", datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC), None)
E_PLAN = "6" * 64


def _plan(drop_week=None):
    cal = wc.nominal_fx_calendar(date(2022, 12, 4), date(2024, 12, 29),
                                 calendar_id="FX_NY_17_ASSUMED_NOT_BROKER", version="i6n-v1")
    _, weeks = c1.load_calendar(cal)
    obs = []
    for w in weeks:
        t = w.market_open + H
        while t <= w.market_close and t < wc.TEST_START:
            if w.week_id != drop_week:
                obs.append(int(t.timestamp()))
            t += H
    return wc.build_plan(calendar_config=cal, observed_seconds=obs,
                         clock_scope={"scope_id": "S", "start": datetime(2019, 12, 1, tzinfo=UTC),
                                      "end": datetime(2025, 1, 1, tzinfo=UTC)},
                         history_years=1, development_split=DEV, fold_year=None, experiment="I6N_TEST")


PLAN = _plan()


def finalists(family="short", fitness=None, basis="INNER_FOLD_FITNESS", skip=()):
    hs = mo.FAMILIES[family]
    fitness = fitness or {h: 1.0 - 0.01 * i for i, h in enumerate(hs)}
    out = {}
    for h in hs:
        if h in skip:
            continue
        out[str(h)] = {role: {"search_candidate_sha256": f"{h:02d}{role[0]}".ljust(64, "a")[:64],
                              "genotype_canonical_sha256": f"{h:02d}{role[0]}".ljust(64, "b")[:64],
                              "fitness": fitness[h] + (0.0 if role == "BEST" else 0.05),
                              "selection_basis": basis}
                       for role in ("BEST", "MINIMAL_CONTROL")}
    return out


def test_template_covers_all_eligible_weeks_both_families_within_budget():
    m = mo.build_arms_manifest(PLAN, e_plan_sha256=E_PLAN)
    assert m["state"] == "WAIT_DEPENDENCY" and m["cause"] == "FINALISTS_UNRESOLVED"
    assert m["families"] == {"short": [2, 3, 4, 5, 6, 12], "long": [18, 24, 36, 48, 60, 72]}
    assert m["budget"] == {"max_dispositions": 212, "planned_dispositions": 212, "recipes_per_family": 2}
    arms = m["arms"]
    assert len(arms) == 212 and len({a["arm_id"] for a in arms}) == 212
    weekly = [u["unit_id"] for u in PLAN["units"] if u["kind"] == "WEEKLY_REFIT"]
    for fam in ("short", "long"):
        for slot in ("R1", "R2"):
            got = [a["week_id"] for a in arms if a["family"] == fam and a["recipe_slot"] == slot]
            assert got == weekly  # every eligible week, weekly refit, none merged
    for a in arms:
        assert a["state"] == "WAIT_DEPENDENCY" and a["recipe"] is None and a["binding"] is None
        assert a["config"] is None and a["horizons"] == m["families"][a["family"]]
        assert a["week_id"][1:] < "2025"
    assert m["transfer_claim"] == "TEMPLATE_BOUND_TO_SEARCH_FINALISTS_NOT_A_TRANSFERRED_OPTIMUM"


def test_per_head_c1_masks_keep_valid_heads_and_type_unsupported_ones():
    m = mo.build_arms_manifest(PLAN, e_plan_sha256=E_PLAN)
    last = next(a for a in m["arms"] if a["family"] == "long" and a["week_id"] == "W2024-12-29")
    states = {h: v["state"] for h, v in last["heads"].items()}
    assert states == {"18": "READY", "24": "READY", "36": "READY", "48": "NO_ELIGIBLE_ROWS",
                      "60": "NO_ELIGIBLE_ROWS", "72": "NO_ELIGIBLE_ROWS"}
    unit = next(u for u in PLAN["units"] if u["unit_id"] == "W2024-12-29")
    v = unit["support"]["MANAGEMENT"]["18"]["VALIDATION"]
    assert last["heads"]["18"]["VALIDATION"] == {"n": v["n"], "origins_sha256": v["origins_sha256"]}
    assert last["heads"]["72"]["VALIDATION"]["n"] == 0


def test_week_without_any_supported_head_is_excluded_and_recorded():
    p = _plan(drop_week="2024-08-04")
    m = mo.build_arms_manifest(p, e_plan_sha256=E_PLAN)
    assert {"unit_id": "W2024-08-04", "family": "short", "cause": "NO_ELIGIBLE_HEAD"} in m["excluded_weeks"]
    assert m["budget"]["planned_dispositions"] == 208
    assert all(a["week_id"] != "W2024-08-04" for a in m["arms"])


def test_budget_is_enforced_never_by_dropping_weeks():
    with pytest.raises(mo.MultiOutputError) as e:
        mo.build_arms_manifest(PLAN, e_plan_sha256=E_PLAN, max_dispositions=100)
    assert e.value.code == "BUDGET_EXCEEDED"
    with pytest.raises(mo.MultiOutputError):
        mo.build_arms_manifest(PLAN, e_plan_sha256=E_PLAN, recipes_per_family=3)


def test_resolution_binds_recipes_from_inner_fold_finalists_only():
    m = mo.build_arms_manifest(PLAN, e_plan_sha256=E_PLAN)
    fin = {**finalists("short"), **finalists("long")}
    r = mo.resolve_arms(m, fin)
    assert r["state"] == "READY" and r["template_sha256"] == m["manifest_sha256"]
    short = r["recipes"]["short"]
    assert short["anchor_horizon"] == 12  # lowest BEST inner-fold fitness in the family
    assert short["R1"]["search_candidate_sha256"] == fin["12"]["BEST"]["search_candidate_sha256"]
    assert short["R2"]["search_candidate_sha256"] == fin["12"]["MINIMAL_CONTROL"]["search_candidate_sha256"]
    arm = next(a for a in r["arms"] if a["family"] == "short" and a["recipe_slot"] == "R2")
    assert arm["state"] == "READY" and arm["recipe"]["role"] == "MINIMAL_CONTROL"
    assert arm["config"] is None and arm["binding"] is None  # produced by E/A and B at activation
    assert mo.resolve_arms(m, fin) == r
    with pytest.raises(mo.MultiOutputError):
        mo.resolve_arms(r, fin)


def test_resolution_refuses_validation_selected_or_incomplete_finalists():
    m = mo.build_arms_manifest(PLAN, e_plan_sha256=E_PLAN)
    with pytest.raises(mo.MultiOutputError) as e:
        mo.resolve_arms(m, {**finalists("short", basis="ANNUAL_VALIDATION_2024"), **finalists("long")})
    assert e.value.code == "SELECTION_BASIS_NOT_INNER_FOLD"
    partial = mo.resolve_arms(m, {**finalists("short", skip=(5,)), **finalists("long")})
    assert partial["state"] == "PARTIAL"
    assert partial["recipes"]["short"] is None and partial["recipes"]["long"] is not None
    assert {a["state"] for a in partial["arms"] if a["family"] == "short"} == {"WAIT_DEPENDENCY"}
    assert all(a["cause"] == "FINALISTS_INCOMPLETE: h5" for a in partial["arms"] if a["family"] == "short")


def test_tie_breaks_on_the_shorter_horizon():
    m = mo.build_arms_manifest(PLAN, e_plan_sha256=E_PLAN)
    flat = {h: 0.9 for h in mo.FAMILIES["long"]}
    r = mo.resolve_arms(m, {**finalists("short"), **finalists("long", fitness=flat)})
    assert r["recipes"]["long"]["anchor_horizon"] == 18
