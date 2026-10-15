#!/usr/bin/env python3
"""PROV01 - independent check of the two FRED publication clocks against the
recovered ALFRED point-in-time evidence.

Inputs (retained under market_time/provenance/):
  DGS3MO_realtime_start_2019_2024.csv
  DGS7_realtime_start_2019_2024.csv
Each row: observation_date, initial_value, realtime_start_date.

The producer_recovery.v1.json record declares, for both FRED features, the
clock "as-of last value from (D+2) 00:00 UTC, NaN if older than 168 h".
This probe asks one falsifiable question per control:

  C1  Is every observation's realtime_start strictly AFTER its own
      observation date?  (no same-day database availability)
  C2  For how many observations is the ALFRED realtime_start date strictly
      LATER than the declared availability instant (D+2) 00:00 UTC?  Those
      observations are NOT covered by the declared clock on this artifact.
  C3  Are the long lags explained by weekends/holidays (i.e. is the lag
      pattern a function of the observation weekday)?
  C4  Fail-closed counterexample: a doctored row whose realtime_start equals
      its observation date must be reported as a violation by C1, and a
      doctored row whose realtime_start is one day earlier must flip C2's
      coverage count.  If the controls cannot see an injected violation they
      measure nothing.
  C5  Revision control: the recovered artifact must name every observation
      whose first published value differs from the late-anchor vintage.

Exit code 0 always; the JSON verdict is the result.  CPU only, no network.
"""
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(HERE), "provenance")
SERIES = ["DGS3MO", "DGS7"]
DECLARED_LAG_DAYS = 2  # "(D+2) 00:00 UTC"


def load(series):
    path = os.path.join(PROV, "%s_realtime_start_2019_2024.csv" % series)
    rows = []
    with open(path) as f:
        head = f.readline().strip().split(",")
        assert head == ["observation_date", "initial_value",
                        "realtime_start_date"], head
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) != 3:
                continue
            rows.append((p[0], p[1], p[2]))
    return rows


def controls(rows):
    c1 = []        # realtime_start <= observation_date
    not_covered = []  # realtime_start date later than D + DECLARED_LAG_DAYS
    by_weekday = {}
    lag_hist = {}
    for obs, val, rts in rows:
        o = dt.date.fromisoformat(obs)
        r = dt.date.fromisoformat(rts)
        lag = (r - o).days
        lag_hist[lag] = lag_hist.get(lag, 0) + 1
        wd = o.strftime("%a")
        by_weekday.setdefault(wd, {}).setdefault(lag, 0)
        by_weekday[wd][lag] += 1
        if lag <= 0:
            c1.append({"obs": obs, "realtime_start": rts})
        if lag > DECLARED_LAG_DAYS:
            not_covered.append({"obs": obs, "obs_weekday": wd,
                                "realtime_start": rts, "lag_days": lag})
    return {
        "n": len(rows),
        "C1_realtime_start_not_after_observation": c1,
        "C1_verdict": "PASS" if not c1 else "FAIL",
        "C2_declared_instant": "(D+%d) 00:00 UTC" % DECLARED_LAG_DAYS,
        "C2_observations_not_covered_by_declared_clock": len(not_covered),
        "C2_fraction_not_covered": round(len(not_covered) / max(1, len(rows)), 4),
        "C2_examples": not_covered[:10],
        "C3_lag_histogram_calendar_days": {str(k): v for k, v in sorted(lag_hist.items())},
        "C3_lag_by_observation_weekday": {
            k: {str(a): b for a, b in sorted(v.items())}
            for k, v in sorted(by_weekday.items())},
    }


out = {"probe": "PROV01_fred_publication_clock",
       "declared_clock_under_test":
           "as-of last value from (D+2) 00:00 UTC, NaN if older than 168 h "
           "(tools/i6dmo/producer_recovery.v1.json, both FRED features)",
       "evidence_semantics":
           "ALFRED realtime_start is the DATE the value entered the FRED/ALFRED "
           "database. It is an UPPER bound on public availability: the H.15 "
           "release itself is posted at 4:15pm ET on the observation day. A "
           "realtime_start later than the declared instant therefore refutes "
           "coverage BY THIS ARTIFACT, and does not by itself prove look-ahead.",
       "series": {}}

for s in SERIES:
    rows = load(s)
    res = controls(rows)

    # C4 fail-closed counterexamples, injected by this probe
    bad_same_day = list(rows)
    bad_same_day[10] = (rows[10][0], rows[10][1], rows[10][0])
    inj1 = controls(bad_same_day)
    victim = next(r for r in rows
                  if (dt.date.fromisoformat(r[2])
                      - dt.date.fromisoformat(r[0])).days > DECLARED_LAG_DAYS)
    earlier = (dt.date.fromisoformat(victim[0])
               + dt.timedelta(days=DECLARED_LAG_DAYS)).isoformat()
    bad_cov = [(o, v, earlier if o == victim[0] else r) for (o, v, r) in rows]
    inj2 = controls(bad_cov)
    res["C4_counterexamples"] = {
        "same_day_row_injected": {
            "row": {"obs": rows[10][0], "forced_realtime_start": rows[10][0]},
            "C1_verdict_on_doctored_table": inj1["C1_verdict"],
            "caught": inj1["C1_verdict"] == "FAIL",
        },
        "coverage_row_pulled_earlier": {
            "row": {"obs": victim[0], "true_realtime_start": victim[2],
                    "forced_realtime_start": earlier},
            "not_covered_true": res["C2_observations_not_covered_by_declared_clock"],
            "not_covered_doctored": inj2["C2_observations_not_covered_by_declared_clock"],
            "caught": (inj2["C2_observations_not_covered_by_declared_clock"]
                       == res["C2_observations_not_covered_by_declared_clock"] - 1),
        },
    }
    out["series"][s] = res

# C5 revision control, read back from the retained manifest
man = json.load(open(os.path.join(PROV, "ALFRED_PROVENANCE.json")))
out["C5_revision_control"] = {
    s: {"anchor_vintage": man["series"][s]["revision_test"]["vintage"],
        "compared": man["series"][s]["revision_test"]["compared"],
        "observations_revised":
            man["series"][s]["revision_test"]["observations_revised"],
        "revised": man["series"][s]["revision_test"]["revised_sample"]}
    for s in SERIES}

both = [out["series"][s]["C1_verdict"] for s in SERIES]
out["verdict"] = {
    "no_same_day_availability": "PASS" if set(both) == {"PASS"} else "FAIL",
    "declared_(D+2)_clock_covered_by_this_artifact":
        all(out["series"][s]["C2_observations_not_covered_by_declared_clock"] == 0
            for s in SERIES),
    "counterexamples_caught": all(
        out["series"][s]["C4_counterexamples"][k]["caught"]
        for s in SERIES
        for k in ("same_day_row_injected", "coverage_row_pulled_earlier")),
}
print(json.dumps(out, indent=2, sort_keys=True))
