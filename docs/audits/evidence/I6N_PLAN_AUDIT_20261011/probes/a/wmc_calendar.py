"""WMC calendar checks for the I6-N audit (pure python + numpy timestamps only).

ASSUMPTION (not the broker calendar): NY Forex convention, O = Sunday 17:00
America/New_York, C = Friday 17:00 America/New_York, offsets 12+12, entry cut
Thursday 00:00 America/New_York. Bars are labelled by their END time in UTC
(inferred from the retained INPUT.npz clock; see bar_clock.out).
Reads only the 'ts' array of an INPUT.npz whose rows are < 2025-01-01 (no TEST).
"""
import sys, json
from datetime import datetime, timedelta, timezone, date
from zoneinfo import ZoneInfo
import numpy as np

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
SHORT = [2, 3, 4, 5, 6, 12]
LONG = [18, 24, 36, 48, 60, 72]
H = SHORT + LONG
HOUR = 3600
TEST_START = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())

def ep(d):
    return int(d.timestamp())

def week(sunday, f=12, e=12):
    O = datetime(sunday.year, sunday.month, sunday.day, 17, tzinfo=NY)
    fri = sunday + timedelta(days=5); thu = sunday + timedelta(days=4)
    C = datetime(fri.year, fri.month, fri.day, 17, tzinfo=NY)
    cut = datetime(thu.year, thu.month, thu.day, 0, tzinfo=NY)
    o, c, k = ep(O), ep(C), ep(cut)
    return dict(O=o, C=c, A=o + e * HOUR, B=c - f * HOUR, cut=k)

def counts(w, ts_set=None, closed=()):
    """E (calendar) and optionally M (exact observed t and t+h) per h and role."""
    A, B, cut = w["A"], w["B"], w["cut"]
    out = {}
    for h in H:
        row = {}
        for role, end in (("management", B), ("entry", min(cut, B))):
            ts = np.arange(A, end, HOUR)
            ts = ts[ts + h * HOUR < B]
            if closed:
                ok = np.ones(len(ts), bool)
                for (a, b) in closed:  # holiday closure [a,b): neither origin nor target inside
                    ok &= ~((ts >= a) & (ts < b)) & ~((ts + h * HOUR >= a) & (ts + h * HOUR < b))
                ts = ts[ok]
            row[role + "_E"] = int(len(ts))
            row[role + "_E_targets_in_2025"] = int(np.sum(ts + h * HOUR >= TEST_START))
            if ts_set is not None:
                m = np.array([(t in ts_set) and ((t + h * HOUR) in ts_set) and (t + h * HOUR < TEST_START) for t in ts.tolist()], bool)
                row[role + "_M"] = int(m.sum())
        out[h] = row
    return out

def fmt(x):
    return datetime.fromtimestamp(x, UTC).strftime("%a %Y-%m-%d %H:%M UTC")

def show(title, w, c):
    print(f"\n## {title}: O={fmt(w['O'])} A={fmt(w['A'])} cut={fmt(w['cut'])} B={fmt(w['B'])} C={fmt(w['C'])} B-A={(w['B']-w['A'])//HOUR}h A->cut={(w['cut']-w['A'])//HOUR}h")
    keys = list(next(iter(c.values())).keys())
    print("h\t" + "\t".join(keys))
    for h in H:
        print(f"{h}\t" + "\t".join(str(c[h][k]) for k in keys))

def main(npz):
    with np.load(npz, allow_pickle=False) as a:
        ts = a["ts"].astype(np.int64).copy()
    assert ts[-1] < TEST_START
    tset = set(ts.tolist())
    res = {}
    # 1. normal week, DST weeks, holiday weeks
    cases = {
        "normal_2024-06-09": date(2024, 6, 9),
        "dst_start_2024-03-10": date(2024, 3, 10),
        "dst_end_2024-11-03": date(2024, 11, 3),
        "eu_us_dst_gap_2024-03-24": date(2024, 3, 24),
        "christmas_2024-12-22": date(2024, 12, 22),
        "newyear_last_validation_week_2024-12-29": date(2024, 12, 29),
        "first_2024_week_2023-12-31": date(2023, 12, 31),
        "thanksgiving_2024-11-24": date(2024, 11, 24),
    }
    for name, sun in cases.items():
        w = week(sun)
        c = counts(w, tset)
        show(name, w, c); res[name] = dict(week={k: fmt(v) for k, v in w.items()}, counts=c)
    # Holiday variant V2 (ASSUMPTION): closure 17:00 NY eve -> 17:00 NY holiday
    def closure(y, m, d):
        a = datetime(y, m, d, 17, tzinfo=NY) - timedelta(days=1)
        return (ep(a), ep(a + timedelta(days=1)))
    v2 = {"christmas_2024-12-22_closure": (date(2024, 12, 22), [closure(2024, 12, 25)]),
          "newyear_2024-12-29_closure": (date(2024, 12, 29), [closure(2025, 1, 1)])}
    for name, (sun, cl) in v2.items():
        w = week(sun); c = counts(w, None, cl); show(name + " (V2: holiday treated as no-origin/no-target closure)", w, c)
        res[name] = dict(counts=c)
    # 2. all market weeks whose A falls in 2024: totals of E and M per h/role
    tot = {h: dict(management_E=0, management_M=0, entry_E=0, entry_M=0, management_E_targets_in_2025=0) for h in H}
    sun = date(2023, 12, 31); weeks = 0; zero_weeks = {h: 0 for h in H}
    while True:
        w = week(sun)
        if w["A"] >= TEST_START:
            break
        c = counts(w, tset); weeks += 1
        for h in H:
            for k in tot[h]:
                tot[h][k] += c[h][k]
            if c[h]["management_M"] == 0:
                zero_weeks[h] += 1
        sun += timedelta(days=7)
    print(f"\n## VALIDATION-2024 market weeks with A in 2024: {weeks}")
    print("h\tmgmt_E\tmgmt_M\tentry_E\tentry_M\tmgmt_E_targets>=2025\tweeks_with_zero_mgmt_M")
    for h in H:
        t = tot[h]
        print(f"{h}\t{t['management_E']}\t{t['management_M']}\t{t['entry_E']}\t{t['entry_M']}\t{t['management_E_targets_in_2025']}\t{zero_weeks[h]}")
    res["validation_2024_totals"] = dict(weeks=weeks, totals=tot, zero_weeks=zero_weeks)
    # 3. 2023 weeks: per-quarter candidate inner weeks, the 4-week h=72 population
    q = [date(2023, 2, 12), date(2023, 5, 14), date(2023, 8, 13), date(2023, 11, 12)]
    s = 0
    for sun in q:
        c = counts(week(sun), tset); s += c[72]["management_M"]
    print(f"\n## Example 4 quarterly 2023 inner weeks (ILLUSTRATIVE, not named in the plan): h=72 management_M total = {s}")
    res["inner_2023_example_h72_mgmt_M"] = s
    # 4. long-family availability by remaining hours over entry origins (normal week)
    w = week(date(2024, 6, 9))
    avail = {}
    for t in range(w["A"], w["cut"], HOUR):
        n = sum(1 for h in LONG if t + h * HOUR < w["B"])
        avail[n] = avail.get(n, 0) + 1
    print("\n## Entry origins by number of calendar-eligible long horizons (normal week):", dict(sorted(avail.items())))
    res["entry_long_availability_normal_week"] = avail
    # 5. missing input slots for 2024 h=72 management origins (exact observed origins only)
    origins = []
    sun = date(2023, 12, 31)
    while True:
        w = week(sun)
        if w["A"] >= TEST_START:
            break
        for t in range(w["A"], w["B"] - 72 * HOUR, HOUR):
            if t in tset:
                origins.append(t)
        sun += timedelta(days=7)
    origins = np.array(origins)
    print(f"\n## Input-slot absence for {len(origins)} observed 2024 h=72 management origins")
    miss = {}
    for (L, d) in ((144, 36), (144, 72), (432, 1), (432, 8), (288, 1), (72, 1), (24, 1), (8, 2)):
        n = L // d + 1
        offs = np.arange(n - 1, -1, -1) * d * HOUR
        slots = origins[:, None] - offs[None, :]
        present = np.isin(slots, ts)
        frac = 1 - present.mean()
        any_missing = float((~present).any(axis=1).mean())
        miss[f"L{L}_d{d}_n{n}"] = dict(missing_slot_fraction=round(float(frac), 4), origins_with_any_missing=round(any_missing, 4),
                                        min_present=int(present.sum(axis=1).min()), max_present=int(present.sum(axis=1).max()))
        print(f"L={L} delta={d} n={n}: missing slot fraction={frac:.4f}; origins with >=1 missing slot={any_missing:.4f}; present per window min/max={present.sum(axis=1).min()}/{present.sum(axis=1).max()}")
    res["input_absence_h72_2024"] = miss
    json.dump(res, open(sys.argv[2], "w"), indent=1, default=str)

if __name__ == "__main__":
    main(sys.argv[1])
