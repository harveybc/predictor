"""Calendar attribution for I6N-AUD-07/-08/-10/-11/-29 (addendum, lane calendar).

Every count is computed under an EXPLICIT calendar given by parameters; no
number in the output is valid without the calendar id printed next to it.

Calendars
  NY(f,e,cut)       ASSUMED Forex convention: O = Sunday 17:00 America/New_York,
                    C = Friday 17:00 America/New_York, no holidays.
  OBS(f,e,cut,lab)  DATA-OBSERVED session proxy derived ONLY from the ts array:
                    weeks are split at Saturday 00:00 UTC; O and C come from
                    the first/last bar of each week under bar-label convention
                    lab (END: bar labelled L covers [L-1h, L); OPEN: [L, L+1h)).
                    The contract (MWC:53-55) says a CSV gap alone does not
                    identify a market closure, so OBS is a proxy, never the
                    calendar. The week containing 2024-12-31 cannot reveal C
                    without reading TEST rows; its C is taken from NY and the
                    week is flagged HYBRID.
  A = O + e, B = C - f; entry cut = <weekday> <hh:mm> America/New_York of the
  week (Saturday split + 4 or 5 days).

Eligibility (MWC:69-88): management E: A <= t < B and A <= t+h < B (t hourly
on the grid of A); entry E: additionally t < min(cut, B). M = E and t, t+h are
both observed bars and t+h < 2025-01-01 (the retained INPUT has no TEST rows).

I6-E objects (read-only, imported from the audited worktree at HEAD 269c2d13):
  tools/i6e_horizon_plan.py weekly_schedule(2024)  -> the 53 segments
  tools/i6m_data.py targets/partitions             -> the I6-E scoring rows
The population digest is recomputed with the expression of
tools/i6m_queue.py:17-19 and compared with DESIGN.json of I6-E-M.

Reads ONLY the 'ts' member of the INPUT archive (price is replaced by ones,
which leaves targets() validity unchanged because it depends on ts only).
Usage:
  python calendar_attribution.py <INPUT.npz> <AUDITED_WORKTREE> <I6EM_DESIGN.json> <out.json>
"""
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
HOUR = 3600
SHORT = [2, 3, 4, 5, 6, 12]
LONG = [18, 24, 36, 48, 60, 72]
H = SHORT + LONG
TEST_START = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())
VAL_START = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp())
SAT0 = int(datetime(2012, 4, 28, tzinfo=UTC).timestamp())  # a Saturday 00:00 UTC
CUTS = {"WED_1200": (3, 12), "THU_0000": (4, 0), "THU_1200": (4, 12)}  # days after Sunday, hour
OFFSETS = [6, 12, 18]
INNER_2023_SUNDAYS = ["2023-02-12", "2023-05-14", "2023-08-13", "2023-11-12"]  # as in probes/a/wmc_calendar.py
NORMAL_SUNDAY = "2024-06-09"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fmt_ny(x):
    return datetime.fromtimestamp(int(x), UTC).astimezone(NY).strftime("%a %Y-%m-%d %H:%M %Z")


def fmt_utc(x):
    return datetime.fromtimestamp(int(x), UTC).strftime("%a %Y-%m-%d %H:%M UTC")


def ny_epoch(day, hour):
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=NY).timestamp())


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ny_weeks():
    """NY calendar skeleton: one record per Sunday from 2022-12-25 to 2024-12-29."""
    out = []
    sun = datetime(2022, 12, 25).date()
    while sun <= datetime(2024, 12, 29).date():
        out.append(dict(sunday=sun.isoformat(), O=ny_epoch(sun, 17), C=ny_epoch(sun + timedelta(days=5), 17), hybrid=False, label="END"))
        sun += timedelta(days=7)
    return out


def obs_weeks(ts, label):
    """Data-observed proxy: Saturday-00:00-UTC split, O/C from first/last bar."""
    key = (ts - SAT0) // (7 * 86400)
    out = []
    lo = int(datetime(2022, 12, 24, tzinfo=UTC).timestamp())
    for k in np.unique(key):
        sat = SAT0 + int(k) * 7 * 86400
        if sat < lo:
            continue
        w = ts[key == k]
        sun = datetime.fromtimestamp(sat, UTC).date() + timedelta(days=1)
        if label == "END":
            O, C = int(w[0]) - HOUR, int(w[-1])
        else:
            O, C = int(w[0]), int(w[-1]) + HOUR
        hybrid = False
        if sun.isoformat() == "2024-12-29":
            C = ny_epoch(sun + timedelta(days=5), 17)  # not observable without TEST rows
            hybrid = True
        out.append(dict(sunday=sun.isoformat(), O=O, C=C, hybrid=hybrid, label=label, n_bars=int(len(w))))
    return out


def finish(weeks, f, e, cut):
    dd, hh = CUTS[cut]
    for w in weeks:
        sun = datetime.fromisoformat(w["sunday"]).date()
        w["A"] = w["O"] + e * HOUR
        w["B"] = w["C"] - f * HOUR
        w["cut"] = ny_epoch(sun + timedelta(days=dd), hh)
    return weeks


def origins(w, h, role):
    end = w["B"] if role == "management" else min(w["cut"], w["B"])
    t = np.arange(w["A"], max(end, w["A"]), HOUR, dtype=np.int64)
    return t[t + h * HOUR < w["B"]]


def week_counts(w, tset):
    res = {}
    for h in H:
        r = {}
        for role in ("management", "entry"):
            t = origins(w, h, role)
            m = np.array([(x in tset) and ((x + h * HOUR) in tset) and (x + h * HOUR < TEST_START) for x in t.tolist()], bool)
            r[role + "_E"] = int(len(t))
            r[role + "_M"] = int(m.sum())
            r[role + "_E_target_ge_2025"] = int(np.sum(t + h * HOUR >= TEST_START))
        res[h] = r
    return res


def check_identities(w):
    """Calendar-independent identities; returns number of violations."""
    BA = (w["B"] - w["A"]) // HOUR
    cA = (min(w["cut"], w["B"]) - w["A"]) // HOUR
    bad = 0
    for h in H:
        if len(origins(w, h, "management")) != max(0, BA - h):
            bad += 1
        if len(origins(w, h, "entry")) != max(0, min(max(cA, 0), BA - h)):
            bad += 1
        t = origins(w, h, "management")
        if len(t) and (w["B"] - t.max()) // HOUR != h + 1:  # strict h < R(t): last origin has R = h+1
            bad += 1
    n = long_availability(w)
    if len(n) and int(n.min()) != sum(1 for h in LONG if h < (w["B"] - min(w["cut"], w["B"])) // HOUR + 1):
        bad += 1  # min #longs over entry origins == #{h in LONG : h < (B-cut)+1}
    return bad


def long_availability(w):
    t = np.arange(w["A"], min(w["cut"], w["B"]), HOUR, dtype=np.int64)
    n = np.array([sum(1 for h in LONG if x + h * HOUR < w["B"]) for x in t.tolist()], int)
    return n


def run_config(name, weeks, tset):
    val = [w for w in weeks if VAL_START <= w["A"] < TEST_START]
    ident_bad = sum(check_identities(w) for w in weeks)
    totals = {h: dict(management_E=0, management_M=0, entry_E=0, entry_M=0,
                      management_E_target_ge_2025=0, entry_E_target_ge_2025=0, zero_mgmt_M_weeks=0) for h in H}
    BA_hist, cutA_hist, h72_slots, Bcut_hist, odd = {}, {}, {}, {}, []
    min_longs, lt2, zero_long = 99, 0, 0
    for w in val:
        c = week_counts(w, tset)
        for h in H:
            for k in ("management_E", "management_M", "entry_E", "entry_M", "management_E_target_ge_2025", "entry_E_target_ge_2025"):
                totals[h][k] += c[h][k]
            totals[h]["zero_mgmt_M_weeks"] += int(c[h]["management_M"] == 0)
        BA = (w["B"] - w["A"]) // HOUR
        BA_hist[BA] = BA_hist.get(BA, 0) + 1
        cA = (min(w["cut"], w["B"]) - w["A"]) // HOUR
        cutA_hist[cA] = cutA_hist.get(cA, 0) + 1
        Bc = (w["B"] - min(w["cut"], w["B"])) // HOUR
        Bcut_hist[Bc] = Bcut_hist.get(Bc, 0) + 1
        odd.append(dict(sunday=w["sunday"], O=fmt_ny(w["O"]), C=fmt_ny(w["C"]), B_minus_A=int(BA), A_to_entry_end=int(cA), hybrid=w["hybrid"], n_bars=w.get("n_bars")))
        t72 = origins(w, 72, "management")
        if len(t72):
            s = (fmt_ny(t72[0])[:3] + fmt_ny(t72[0])[14:20], fmt_ny(t72[-1])[:3] + fmt_ny(t72[-1])[14:20])
            h72_slots[s] = h72_slots.get(s, 0) + 1
        n = long_availability(w)
        if len(n):
            min_longs = min(min_longs, int(n.min()))
            lt2 += int((n < 2).sum())
            zero_long += int((n == 0).sum())
    modal = max(set((o["B_minus_A"], o["A_to_entry_end"]) for o in odd), key=lambda p: sum(1 for o in odd if (o["B_minus_A"], o["A_to_entry_end"]) == p))
    odd = [o for o in odd if (o["B_minus_A"], o["A_to_entry_end"]) != modal]
    normal = next((w for w in weeks if w["sunday"] == NORMAL_SUNDAY), None)
    normal_c = week_counts(normal, tset) if normal else None
    normal_av = long_availability(normal) if normal else np.array([], int)
    inner = [w for w in weeks if w["sunday"] in INNER_2023_SUNDAYS]
    inner_E = sum(len(origins(w, 72, "management")) for w in inner)
    inner_M = sum(week_counts(w, tset)[72]["management_M"] for w in inner)
    first = min(val, key=lambda w: w["A"])
    last = max(val, key=lambda w: w["A"])
    return dict(
        calendar=name, n_validation_weeks=len(val), identity_violations_all_weeks=ident_bad,
        n_weeks_checked=len(weeks), B_minus_A_hours_hist=dict(sorted(BA_hist.items())),
        A_to_entry_end_hours_hist=dict(sorted(cutA_hist.items())), B_minus_entry_end_hours_hist=dict(sorted(Bcut_hist.items())),
        modal_B_minus_A_and_A_to_entry_end=list(modal), weeks_deviating_from_modal=odd, h72_mgmt_slot_NY_hist={f"{a}..{b}": v for (a, b), v in sorted(h72_slots.items())},
        validation_totals=totals,
        validation_entry_long_availability=dict(min=min_longs, origins_with_lt2_longs=lt2, origins_with_0_longs=zero_long),
        normal_week=dict(sunday=NORMAL_SUNDAY, A=fmt_ny(normal["A"]), B=fmt_ny(normal["B"]), cut=fmt_ny(normal["cut"]),
                         counts=normal_c, entry_long_availability={int(k): int(v) for k, v in zip(*np.unique(normal_av, return_counts=True))}) if normal else None,
        inner_2023_four_weeks_h72=dict(sundays=INNER_2023_SUNDAYS, E=inner_E, M=inner_M, found=len(inner)),
        first_validation_week=dict(sunday=first["sunday"], O=fmt_ny(first["O"]), A=fmt_ny(first["A"]), B=fmt_ny(first["B"]),
                                   counts=week_counts(first, tset), hybrid=first["hybrid"]),
        last_validation_week=dict(sunday=last["sunday"], O=fmt_ny(last["O"]), A=fmt_ny(last["A"]), B=fmt_ny(last["B"]),
                                  C=fmt_ny(last["C"]), hybrid=last["hybrid"], counts=week_counts(last, tset),
                                  decomposition={h: {role: dict(
                                      origin_2024_target_2024=int(np.sum((origins(last, h, role) < TEST_START) & (origins(last, h, role) + h * HOUR < TEST_START))),
                                      origin_2024_target_2025_straddle=int(np.sum((origins(last, h, role) < TEST_START) & (origins(last, h, role) + h * HOUR >= TEST_START))),
                                      origin_2025=int(np.sum(origins(last, h, role) >= TEST_START)))
                                      for role in ("management", "entry")} for h in H}),
        _weeks=weeks,
    )


def i6e_segments(worktree, ts, design_path, cals):
    plan = load_module(f"{worktree}/tools/i6e_horizon_plan.py", "i6e_horizon_plan_audited")
    data = load_module(f"{worktree}/tools/i6m_data.py", "i6m_data_audited")
    segs = plan.weekly_schedule(2024)
    design = json.load(open(design_path))
    ones = np.ones(len(ts))
    digest = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    out = dict(n_segments=len(segs), first=segs[0], last=segs[-1], design_weeks_equal=(design["weeks"] == segs), per_h={}, per_calendar={})
    pops = {}
    for h in [2, 3, 4, 5, 6, 24, 48, 72]:
        y, _ = data.targets(ts, ones, h)
        rows = []
        per_seg = []
        for w in segs:
            _, _, si = data.partitions(ts, y, w, h, require_fit=False)
            rows.extend(ts[si].tolist())
            per_seg.append(len(si))
        pops[h] = (np.array(rows, np.int64), per_seg)
        dz = design["population"][str(h)]
        out["per_h"][h] = dict(n=len(rows), rows_sha256_match=(digest(rows) == dz["rows_sha256"]), design_n=dz["n"],
                              last_segment_rows=per_seg[-1], segments_with_zero_rows=int(sum(1 for x in per_seg if x == 0)))
    seg_cut = np.array([data.epoch(w["fit_cutoff"]) for w in segs], np.int64)
    for name, weeks in cals.items():
        val = [w for w in weeks if VAL_START <= w["A"] < TEST_START]
        allw = weeks
        # post-reopen bars visible to each I6-E fit: observed labels L with O < L < fit_cutoff (END labels: info at L)
        post, after_open = 0, 0
        for c in seg_cut.tolist():
            prev = [w for w in allw if w["O"] < c < w["C"]]
            if prev:
                O = prev[-1]["O"]
                after_open += 1
                # END label L: information complete at L, so L in (O, c); OPEN label L: complete at L+1h <= c, so L in [O, c)
                post += int(np.sum((ts > O) & (ts < c))) if prev[-1]["label"] == "END" else int(np.sum((ts >= O) & (ts < c)))
        r = dict(segments_with_fit_cutoff_after_O=after_open, observed_bars_between_O_and_fit_cutoff_total=post, per_h={})
        for h, (rows, per_seg) in pops.items():
            ok = np.zeros(len(rows), bool)
            for w in allw:
                t = origins(w, h, "management")
                ok |= np.isin(rows, t)
            r["per_h"][h] = dict(i6e_rows=int(len(rows)), i6e_rows_mgmt_eligible=int(ok.sum()), i6e_rows_not_mgmt_eligible=int((~ok).sum()))
        # last segment (2-day fragment) vs the straddling market week
        last_seg = segs[-1]
        s0, s1 = data.epoch(last_seg["score_start"]), data.epoch(last_seg["score_end"])
        straddle = [w for w in val if w["A"] < s1 and w["B"] > s0]
        sd = {}
        for w in straddle:
            for h in H:
                t = origins(w, h, "management")
                sd[h] = dict(week_mgmt_E=int(len(t)), in_last_segment=int(np.sum((t >= s0) & (t < s1))),
                             target_ge_2025=int(np.sum(t + h * HOUR >= TEST_START)))
        r["last_segment_vs_straddling_week"] = dict(weeks=[w["sunday"] for w in straddle], per_h=sd)
        out["per_calendar"][name] = r
    return out


def main():
    npz, worktree, design_path, out_json = sys.argv[1:5]
    raw = open(npz, "rb").read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        tsb = z.read("ts.npy")
    ts = np.load(io.BytesIO(tsb), allow_pickle=False).astype(np.int64)
    assert ts[-1] < TEST_START, "TEST rows present"
    tset = set(ts.tolist())
    print("# calendar_attribution.py")
    print(f"input_archive_sha256={sha(raw)} ts_member_sha256={sha(tsb)} rows={len(ts)} first={fmt_utc(ts[0])} last={fmt_utc(ts[-1])}")
    for rel in ("tools/i6e_horizon_plan.py", "tools/i6m_data.py"):
        print(f"audited_module {rel} sha256={sha(open(f'{worktree}/{rel}', 'rb').read())}")
    print(f"i6em_design_sha256={sha(open(design_path, 'rb').read())}")
    configs = {}
    for f in OFFSETS:
        for cut in CUTS:
            configs[f"NY_f{f}_e{f}_{cut}"] = lambda f=f, cut=cut: finish(ny_weeks(), f, f, cut)
            configs[f"OBS-END_f{f}_e{f}_{cut}"] = lambda f=f, cut=cut: finish(obs_weeks(ts, "END"), f, f, cut)
    configs["OBS-OPEN_f12_e12_THU_0000"] = lambda: finish(obs_weeks(ts, "OPEN"), 12, 12, "THU_0000")
    results = {}
    for name, mk in configs.items():
        results[name] = run_config(name, mk(), tset)
    primary = ["NY_f12_e12_THU_0000", "OBS-END_f12_e12_THU_0000", "OBS-OPEN_f12_e12_THU_0000"]

    print("\n## 1. Calendar-independent identities (checked on every week of every calendar)")
    tot_w = sum(r["n_weeks_checked"] for r in results.values())
    tot_bad = sum(r["identity_violations_all_weeks"] for r in results.values())
    print(f"management_E == max(0,(B-A)-h); entry_E == max(0,min(cut-A,(B-A)-h)); last origin has R(t)=B-t=h+1 (strict h<R); min #longs over entry origins == #{{h in LONG: h < (B-cut)+1}}: "
          f"{len(results)} calendars, {tot_w} week-instances, 12 h, violations={tot_bad}")

    print("\n## 2. Primary calendars: VALIDATION 2024 (weeks with A in 2024)")
    for name in primary:
        r = results[name]
        print(f"\n### {name}: weeks={r['n_validation_weeks']} B-A hist={r['B_minus_A_hours_hist']} A->entry_end hist={r['A_to_entry_end_hours_hist']}")
        print(f"h72 management slot (NY local, first..last origin) hist={r['h72_mgmt_slot_NY_hist']}")
        print("h\tmgmt_E\tmgmt_M\tentry_E\tentry_M\tmgmt_E_t+h>=2025\tentry_E_t+h>=2025\tweeks_mgmt_M=0")
        for h in H:
            t = r["validation_totals"][h]
            print(f"{h}\t{t['management_E']}\t{t['management_M']}\t{t['entry_E']}\t{t['entry_M']}\t{t['management_E_target_ge_2025']}\t{t['entry_E_target_ge_2025']}\t{t['zero_mgmt_M_weeks']}")
        nw = r["normal_week"]
        print(f"normal week {nw['sunday']}: A={nw['A']} cut={nw['cut']} B={nw['B']}; mgmt_E per h="
              + ",".join(f"{h}:{nw['counts'][h]['management_E']}" for h in H)
              + "; entry_E per h=" + ",".join(f"{h}:{nw['counts'][h]['entry_E']}" for h in H))
        print(f"normal week entry-origin count by #valid longs={nw['entry_long_availability']}; VALIDATION entry long availability={r['validation_entry_long_availability']}")
        fw = r["first_validation_week"]
        print(f"first VALIDATION week sunday={fw['sunday']} O={fw['O']} A={fw['A']} B={fw['B']}: h72 mgmt E/M={fw['counts'][72]['management_E']}/{fw['counts'][72]['management_M']}; h60 E/M={fw['counts'][60]['management_E']}/{fw['counts'][60]['management_M']}")
        for o in r["weeks_deviating_from_modal"]:
            print(f"  week deviating from this calendar's modal (B-A, A->entry_end)={tuple(r['modal_B_minus_A_and_A_to_entry_end'])}: {o}")
        lw = r["last_validation_week"]
        print(f"last VALIDATION week sunday={lw['sunday']} A={lw['A']} B={lw['B']} C={lw['C']} hybrid_C_from_NY={lw['hybrid']}: mgmt_E (t+h>=2025) per h="
              + ",".join(f"{h}:{lw['counts'][h]['management_E']}({lw['counts'][h]['management_E_target_ge_2025']})" for h in H))
        print("  decomposition of the straddling week, management role, per h: origin2024&target2024 / origin2024&target2025 (straddle) / origin2025 (TEST time)")
        print("  " + "; ".join(f"{h}:{v['management']['origin_2024_target_2024']}/{v['management']['origin_2024_target_2025_straddle']}/{v['management']['origin_2025']}" for h, v in lw["decomposition"].items()))
        print("  same, entry role: " + "; ".join(f"{h}:{v['entry']['origin_2024_target_2024']}/{v['entry']['origin_2024_target_2025_straddle']}/{v['entry']['origin_2025']}" for h, v in lw["decomposition"].items()))
        iw = r["inner_2023_four_weeks_h72"]
        print(f"illustrative 2023 inner weeks {iw['sundays']} (found {iw['found']}): h72 mgmt E={iw['E']} M={iw['M']}")

    print("\n## 3. Sensitivity grid (offsets f=e in {6,12,18}, entry cut in {WED_1200,THU_0000,THU_1200} NY)")
    print("calendar\tval_weeks\tB-A_hist\tA->entry_end_hist\tB-entry_end_hist\tmgmt_E_h72\tmgmt_M_h72\tmgmt_M_h48\tmgmt_M_h18\tentry_M_h18\tmin_longs_entry\tentry_origins_<2_longs\tentry_origins_0_longs\th72_t+h>=2025\th2_t+h>=2025\tinner2023_h72_E/M\tfirst_week_h72_E")
    for name, r in results.items():
        t = r["validation_totals"]
        la = r["validation_entry_long_availability"]
        iw = r["inner_2023_four_weeks_h72"]
        print(f"{name}\t{r['n_validation_weeks']}\t{r['B_minus_A_hours_hist']}\t{r['A_to_entry_end_hours_hist']}\t{r['B_minus_entry_end_hours_hist']}\t{t[72]['management_E']}\t{t[72]['management_M']}\t{t[48]['management_M']}\t{t[18]['management_M']}\t{t[18]['entry_M']}\t{la['min']}\t{la['origins_with_lt2_longs']}\t{la['origins_with_0_longs']}\t{t[72]['management_E_target_ge_2025']}\t{t[2]['management_E_target_ge_2025']}\t{iw['E']}/{iw['M']}\t{r['first_validation_week']['counts'][72]['management_E']}")

    print("\n## 4. I6-E segments (tools/i6e_horizon_plan.py weekly_schedule(2024)) and scoring rows (tools/i6m_data.py partitions)")
    seg = i6e_segments(worktree, ts, design_path, {n: results[n]["_weeks"] for n in primary})
    print(f"n_segments={seg['n_segments']} first={seg['first']} last={seg['last']} DESIGN.json weeks identical={seg['design_weeks_equal']}")
    print("h\tI6E_rows(recomputed)\tDESIGN_n\trows_sha256_match\tlast_segment_rows\tsegments_with_0_rows")
    for h, v in seg["per_h"].items():
        print(f"{h}\t{v['n']}\t{v['design_n']}\t{v['rows_sha256_match']}\t{v['last_segment_rows']}\t{v['segments_with_zero_rows']}")
    for name, r in seg["per_calendar"].items():
        print(f"\n### {name}: segments whose fit_cutoff (Mon 00:00 UTC) is after market O = {r['segments_with_fit_cutoff_after_O']} of {seg['n_segments']}; observed bars with O < label < fit_cutoff summed over segments = {r['observed_bars_between_O_and_fit_cutoff_total']}")
        print("h\tI6E_rows\tof_which_WMC_mgmt_eligible\tof_which_NOT_eligible")
        for h, v in r["per_h"].items():
            print(f"{h}\t{v['i6e_rows']}\t{v['i6e_rows_mgmt_eligible']}\t{v['i6e_rows_not_mgmt_eligible']}")
        ls = r["last_segment_vs_straddling_week"]
        print(f"last I6-E segment (2-day fragment) vs straddling market week(s) {ls['weeks']}: h: week_mgmt_E / origins inside the fragment / t+h>=2025")
        print("  " + "; ".join(f"{h}:{v['week_mgmt_E']}/{v['in_last_segment']}/{v['target_ge_2025']}" for h, v in ls["per_h"].items()))

    for r in results.values():
        r.pop("_weeks")
    json.dump(dict(results=results, i6e_segments=seg), open(out_json, "w"), indent=1, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
