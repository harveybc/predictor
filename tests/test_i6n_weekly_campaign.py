"""Lane C: weekly campaign builder (calendar, units, folds, support, coverage).

Synthetic calendars and timestamps only; no prices, no fits, no TEST rows.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from tools.i6n_contracts import eligibility as c1
from tools.i6n_weekly_data import build_weekly_partitions
from tools import i6n_weekly_campaign as wc

UTC = timezone.utc
H = timedelta(hours=1)
DEV = c1.SplitBounds("DEVELOPMENT", datetime(2024, 1, 1, tzinfo=UTC),
                     datetime(2025, 1, 1, tzinfo=UTC), None)
# Existing nominal calendar identity bound by the retained CLOCK.json.
RETAINED_CALENDAR_SHA256 = "347be3faf00e51833a1eb3070b3ce97212c4649698096f7aa92ed4b2687952a7"


def utc(*a):
    return datetime(*a, tzinfo=UTC)


def calendar(first=date(2022, 12, 4), last=date(2024, 12, 29)):
    return wc.nominal_fx_calendar(first, last, calendar_id="FX_NY_17_ASSUMED_NOT_BROKER",
                                  version="i6n-v1")


def full_observed(cal, drop=()):
    _, weeks = c1.load_calendar(cal)
    stamps = []
    for w in weeks:
        t = w.market_open + H  # bar-end clock: first bar ends one hour after the open
        while t <= w.market_close and t < wc.TEST_START:
            if t not in drop:
                stamps.append(int(t.timestamp()))
            t += H
    return stamps


def plan(cal=None, observed=None, **kw):
    cal = cal or calendar()
    obs = full_observed(cal) if observed is None else observed
    args = dict(calendar_config=cal, observed_seconds=obs,
                clock_scope={"scope_id": "TEST_SCOPE", "start": utc(2019, 12, 1),
                             "end": utc(2025, 1, 1)},
                horizons=wc.BUSINESS_HORIZONS, roles=("MANAGEMENT", "ENTRY"),
                fit_role="MANAGEMENT", history_years=1, inner_days=28,
                max_context_hours=432, development_split=DEV, fold_year=None,
                experiment="I6N_TEST")
    args.update(kw)
    return wc.build_plan(**args)


def unit(p, unit_id):
    return next(u for u in p["units"] if u["unit_id"] == unit_id)


# --------------------------------------------------------------------------
# Calendar identity and enumeration


def test_nominal_calendar_reproduces_retained_identity():
    cal = calendar(date(2018, 12, 30), date(2024, 12, 29))
    assert c1.canonical_digest(cal) == RETAINED_CALENDAR_SHA256
    assert cal["market_calendar"]["calendar_id"] == "FX_NY_17_ASSUMED_NOT_BROKER"


def test_all_2024_weeks_enumerated_with_weekly_refit():
    p = plan()
    weeks = [u for u in p["units"] if u["kind"] == "WEEKLY_REFIT"]
    ids = [u["week_id"] for u in weeks]
    assert ids[0] == "2023-12-31" and ids[-1] == "2024-12-29" and len(ids) == 53
    sundays = [date.fromisoformat(i) for i in ids]
    assert all(b - a == timedelta(days=7) for a, b in zip(sundays, sundays[1:]))
    cutoffs = [u["fit_cutoff"] for u in weeks]
    assert len(set(cutoffs)) == 53  # one refit per week, never one model per year
    for u in weeks:
        assert u["fit_cutoff"] < u["evaluation_week"]["market_open"] == u["release_at"]
        assert u["refit_window"] == {"start": u["fit_cutoff"], "end": u["release_at"]}
        assert u["scored_outer"] is True
    assert p["calendar"]["certification"] == wc.NOMINAL_CERTIFICATION


def test_four_calendar_years_and_purged_inner_per_unit():
    p = plan(history_years=4, cal=calendar(date(2018, 12, 30)))
    u = unit(p, "W2024-01-07")
    assert u["fit_cutoff"] == "2024-01-05T22:00:00+00:00"
    assert u["history_start"] == "2020-01-05T22:00:00+00:00"
    assert u["inner_start"] == "2023-12-08T22:00:00+00:00"
    assert u["context_start"] == "2019-12-18T22:00:00+00:00"
    tr, inn = u["splits"]["TRAIN"], u["splits"]["INNER"]
    assert tr["origin_end"] == tr["target_end"] == tr["label_available_by"] == u["inner_start"]
    assert inn["origin_end"] == inn["target_end"] == inn["label_available_by"] == u["fit_cutoff"]


# --------------------------------------------------------------------------
# Mandatory edge cases


def test_dst_weeks_keep_elapsed_hour_semantics():
    p = plan()
    spring, fall = unit(p, "W2024-03-10"), unit(p, "W2024-11-03")
    assert spring["evaluation_week"]["market_open"] == "2024-03-10T21:00:00+00:00"
    assert spring["fit_cutoff"] == "2024-03-08T22:00:00+00:00"   # 47 h closure
    assert fall["evaluation_week"]["market_open"] == "2024-11-03T22:00:00+00:00"
    assert fall["fit_cutoff"] == "2024-11-01T21:00:00+00:00"     # 49 h closure
    for u in (spring, fall, unit(p, "W2024-01-14"), unit(p, "W2024-07-14")):
        v = u["support"]["MANAGEMENT"]["72"]["VALIDATION"]
        assert v["n"] == 24 and v["status"] == "SUPPORTED"  # [A, B-72h) on a full grid
        cut = datetime.fromisoformat(u["fit_cutoff"])
        assert datetime.fromisoformat(u["inner_start"]) == cut - timedelta(hours=28 * 24)


def test_leap_year_history_start_and_leap_day_origins():
    from tools.i6n_weekly_data import _years_before
    assert _years_before(utc(2024, 2, 29, 22), 1) == utc(2023, 2, 28, 22)
    assert _years_before(utc(2024, 2, 29, 22), 4) == utc(2020, 2, 29, 22)
    p = plan(history_years=4, cal=calendar(date(2018, 12, 30)))
    u = unit(p, "W2024-03-03")
    start, cut = (datetime.fromisoformat(u[k]) for k in ("history_start", "fit_cutoff"))
    assert (cut - start).days == 1461  # four calendar years include Feb 29 2020
    leap = [t for t in full_observed(calendar())
            if utc(2024, 2, 29) <= datetime.fromtimestamp(t, UTC) < utc(2024, 3, 1)]
    assert len(leap) == 24  # the leap day contributes origins inside INNER
    assert u["support"]["MANAGEMENT"]["2"]["INNER"]["n"] > 0


def test_2024_2025_boundary_closes_test_and_typed_no_support():
    p = plan()
    last = unit(p, "W2024-12-29")
    s2 = last["support"]["MANAGEMENT"]["2"]["VALIDATION"]
    s72 = last["support"]["MANAGEMENT"]["72"]["VALIDATION"]
    assert s2["n"] > 0 and s2["exclusions"].get("TARGET_IN_TEST_CLOSED", 0) > 0
    assert s72["n"] == 0 and s72["status"] == "NO_ELIGIBLE_ROWS"
    assert last["horizons"]["72"]["state"] == "NO_ELIGIBLE_ROWS"
    assert last["horizons"]["2"]["state"] == "READY"
    for u in p["units"]:
        for role in u["support"].values():
            for splits in role.values():
                v = splits.get("VALIDATION")
                if v and v.get("last_eligible_target"):
                    assert v["last_eligible_target"] < "2025-01-01"


def test_test_rows_in_source_are_refused():
    obs = full_observed(calendar())
    with pytest.raises(wc.CampaignError) as err:
        plan(observed=obs + [int(utc(2025, 1, 2, 3).timestamp())])
    assert err.value.code == "TEST_ROWS_PRESENT"


def test_min_data_coverage_is_exact_and_never_shortens_history():
    cal = calendar()
    obs = full_observed(cal)
    base = unit(plan(), "W2023-12-31")
    assert base["context_start"] == "2022-12-11T22:00:00+00:00"
    ctx = int(datetime.fromisoformat(base["context_start"]).timestamp())
    first_ok = max(t for t in obs if t <= ctx)
    u_ok = unit(plan(observed=[t for t in obs if t >= first_ok]), "W2023-12-31")
    assert u_ok["data"]["state"] == "READY"
    late_obs = [t for t in obs if t > ctx]
    late = datetime.fromtimestamp(late_obs[0], UTC)
    u = unit(plan(observed=late_obs), "W2023-12-31")
    assert u["data"]["state"] == "WAIT_DEPENDENCY"
    missing = u["data"]["missing"]
    assert missing == [{"object": "SUCCESSOR_INPUT_ROWS", "start": u["context_start"],
                        "end": c1.iso(late), "detail": missing[0]["detail"]}]
    assert u["history_start"] == u_ok["history_start"]  # history never shortened
    assert all(h["state"] == "WAIT_DEPENDENCY" for h in u["horizons"].values())
    assert u["support"]["MANAGEMENT"]["2"]["TRAIN"]["label_state"] == "WAIT_DEPENDENCY"
    assert u["support"]["MANAGEMENT"]["2"]["TRAIN"]["n"] is None
    assert u["support"]["MANAGEMENT"]["2"]["TRAIN"]["n_operational"] > 0
    p_scope = plan(clock_scope={"scope_id": "X", "start": late, "end": utc(2025, 1, 1)})
    m = unit(p_scope, "W2023-12-31")["data"]["missing"]
    assert [x["object"] for x in m] == ["CLOCK_SCOPE_CERTIFICATE"]
    assert unit(p_scope, "W2024-01-14")["data"]["state"] == "READY"


def test_minimum_label_support_threshold_is_typed():
    p = plan(min_support={"TRAIN": 10 ** 9, "INNER": 1, "VALIDATION": 1})
    u = unit(p, "W2024-01-14")
    assert u["horizons"]["2"]["state"] == "NO_ELIGIBLE_ROWS"
    assert "TRAIN" in u["horizons"]["2"]["cause"]


def test_horizon_without_calendar_support_is_never_scored():
    p = plan(horizons=(2, 96, 120))
    for u in p["units"]:
        for h in ("96", "120"):
            assert u["horizons"][h]["state"] == "NO_ELIGIBLE_ROWS"
            assert u["support"]["MANAGEMENT"][h]["VALIDATION"]["n"] == 0
    assert p["dispositions"]["by_horizon"]["96"] == 0
    assert p["dispositions"]["by_horizon"]["120"] == 0


def test_incomplete_weeks_stay_listed_with_typed_exclusions():
    cal = calendar()
    _, weeks = c1.load_calendar(cal)
    wk = next(w for w in weeks if w.week_id == "2024-07-07")
    gone = {wk.market_open + k * H for k in range(30, 60)}
    dead = next(w for w in weeks if w.week_id == "2024-08-04")
    gone |= {dead.market_open + k * H for k in range(0, 121)}
    p = plan(cal=cal, observed=full_observed(cal, drop=gone))
    u, full = unit(p, "W2024-07-07"), unit(plan(), "W2024-07-07")
    v, vf = u["support"]["MANAGEMENT"]["2"]["VALIDATION"], full["support"]["MANAGEMENT"]["2"]["VALIDATION"]
    assert 0 < v["n"] < vf["n"]
    assert v["exclusions"]["INPUT_UNAVAILABLE"] > 0
    assert v["origins_enumerated"] == v["n"] + sum(v["exclusions"].values())
    d = unit(p, "W2024-08-04")
    assert d["horizons"]["2"]["state"] == "NO_ELIGIBLE_ROWS"
    assert "W2024-08-04" in [x["unit_id"] for x in p["units"]]


def test_counts_match_weekly_partitions_reference_exactly():
    cal = calendar()
    _, weeks = c1.load_calendar(cal)
    drop = {utc(2024, 3, 5, 14), utc(2024, 3, 6, 2), utc(2024, 2, 21, 9)}
    obs = full_observed(cal, drop=drop)
    p = plan(cal=cal, observed=obs)
    stamps = [datetime.fromtimestamp(t, UTC) for t in obs]
    origins = sorted({w.market_open + k * H for w in weeks
                      for k in range(int((w.market_close - w.market_open) / H))})
    for uid in ("W2024-03-10", "W2024-12-29", "W2023-12-31"):
        u = unit(p, uid)
        ref = build_weekly_partitions(
            origins=origins, weeks=weeks, observed_targets={t: t for t in stamps},
            input_available_at={t: t for t in stamps}, horizons=list(wc.BUSINESS_HORIZONS),
            evaluation_week_id=u["week_id"], evaluation_split=DEV,
            source_coverage_start=stamps[0], history_years=1, inner_days=28,
            history_span_hours=24, role=c1.Role.MANAGEMENT, coverage_mode="complete")
        for name, part in (("TRAIN", ref.train), ("INNER", ref.inner), ("VALIDATION", ref.evaluation)):
            for j, h in enumerate(wc.BUSINESS_HORIZONS):
                chosen = [int(t.timestamp()) for t, m in zip(part.origins, part.mask) if m[j]]
                got = u["support"]["MANAGEMENT"][str(h)][name]
                assert got["n"] == len(chosen), (uid, name, h)
                assert got["origins_sha256"] == c1.canonical_digest(chosen)


def test_trajectory_head_offsets_have_own_support():
    offs = wc.head_offsets(wc.BUSINESS_HORIZONS)
    assert offs[72] == (18, 24, 36, 48, 54, 72)
    assert offs[5] == (5,)
    assert offs[12] == (3, 4, 6, 8, 9, 12)
    p = plan()
    u = unit(p, "W2024-01-14")
    assert u["support"]["MANAGEMENT"]["54"]["VALIDATION"]["n"] > 0
    assert u["support"]["MANAGEMENT"]["54"]["business_horizon"] is False


# --------------------------------------------------------------------------
# 2023 inner folds


def test_2023_quarter_folds_last_complete_market_cutoff():
    p = plan(fold_year=2023, history_years=4, cal=calendar(date(2018, 12, 30)),
             clock_scope={"scope_id": "S", "start": utc(2018, 12, 1),
                                                         "end": utc(2025, 1, 1)})
    folds = [u for u in p["units"] if u["kind"] == "INNER_FOLD"]
    assert [f["fold_id"] for f in folds] == ["2023Q1", "2023Q2", "2023Q3", "2023Q4"]
    assert [f["fit_cutoff"] for f in folds] == [
        "2023-03-31T21:00:00+00:00", "2023-06-30T21:00:00+00:00",
        "2023-09-29T21:00:00+00:00", "2023-12-29T22:00:00+00:00"]
    assert [f["history_start"] for f in folds] == [
        "2019-03-31T21:00:00+00:00", "2019-06-30T21:00:00+00:00",
        "2019-09-29T21:00:00+00:00", "2019-12-29T22:00:00+00:00"]
    for f in folds:
        assert f["scored_outer"] is False and f["splits"]["VALIDATION"] is None
        assert f["structural_evaluation_split"]["split_id"].startswith("STRUCTURAL_UNSCORED_")
        assert f["data"]["state"] == "READY"


def test_folds_before_certificate_scope_wait_with_exact_object():
    p = plan(fold_year=2023, history_years=4, cal=calendar(date(2018, 12, 30)))
    states = {u["fold_id"]: u["data"] for u in p["units"] if u["kind"] == "INNER_FOLD"}
    assert states["2023Q4"]["state"] == "READY"
    for q in ("2023Q1", "2023Q2", "2023Q3"):
        assert states[q]["state"] == "WAIT_DEPENDENCY"
        assert states[q]["missing"][0]["object"] == "CLOCK_SCOPE_CERTIFICATE"
        assert states[q]["missing"][0]["end"] == "2019-12-01T00:00:00+00:00"


def test_plan_is_deterministic_and_sealed():
    a, b = plan(), plan()
    assert a["plan_sha256"] == b["plan_sha256"]
    body = dict(a)
    body.pop("plan_sha256")
    assert c1.canonical_digest(body) == a["plan_sha256"]
    for u in a["units"]:
        v = dict(u)
        v.pop("unit_sha256")
        assert c1.canonical_digest(v) == u["unit_sha256"]
    assert a["dispositions"]["weekly_fits_per_arm"] == sum(a["dispositions"]["by_horizon"].values())
