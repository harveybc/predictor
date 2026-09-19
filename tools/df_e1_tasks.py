#!/usr/bin/env python3
"""RP23: E1 task contracts from the GOVERNED bytes of the DEV families (canonical panels registered in
the census, digest-checked against the dataset contract). No reserve is opened, nothing is re-censused.

Per family: entities, unit, timestamps (format, zone as declared by the producer, DST handling), gaps,
structural zeros (a client that does not exist yet) vs measurements, missing values, availability and
revisions as the producer documents them; column roles (target candidates, features, metadata, controls)
declared, never "every column is a target". Per split (train / validation / test by time, purge = W + h):
the windows really usable after masks (non-finite inputs or targets), purge and horizon; their support
in physical time; candidate contexts in physical units derived from TRAIN only ((W - 1) * delta_t, the
periodicities really measured on train by the spectrum, the model's reach); cost per context. Controls
(future/prefix, train-only scale, gaps, DST) are exercised by tests against this very pipeline.
Eligibility is separated: catalogue (licence), task (the contract above), reserve (not judged here).

    python tools/df_e1_tasks.py --out E1_TASKS.json [--families uci_321,uci_235] [--panels-root DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

STATE = Path.home() / ".local/state/crispdm-data-foundation"
PANELS = STATE / "public_panels_c126_v2"
FAMILIES = {
    "uci_321": {"dataset_id": "public.uci.321.electricityloaddiagrams20112014", "panel": "uci_321_electricityloaddiagrams20112014",
                "panel_sha256": "7cf225c4869e88abcb12b7fd274d532756e41fedbe3094ebc805cc803fc6a6a8", "timestamp": "timestamp_label", "ts_format": "%Y-%m-%d %H:%M:%S",
                "delta_seconds": 900, "unit": "kW (15-min average consumption per client)",
                "producer_notes": ["timestamps in Portuguese local wall clock (producer: 'All time labels report to Portuguese hour')",
                                   "clients created after 2011 have zero consumption before their first record (structural zeros, not measurements)",
                                   "DST (producer statement): 96 values per day kept on the grid; the March change day is a 23-HOUR day whose missing wall-clock hour carries zeros; the October change day is a 25-HOUR day whose repeated hour is aggregated into one — hours of 23/25-hour days, NOT 23/25 records (RP28 trace: rows with labels 01:00-01:45 of the last Sunday of March are zero for ~2/3 of the active clients in the raw file and in the panel alike; October shows no zero signature)",
                                   "availability delay and revision policy: UNKNOWN in the contract; the file is a static archive (no revisions)"],
                "source": "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014", "roles": {"timestamp": ["timestamp_label"]},
                "structural_zero_rule": "a client's rows before its first non-zero value are STRUCTURAL zeros (client not yet existing); zeros after that are measurements"},
    "uci_235": {"dataset_id": "public.uci.235.individual_household_electric_power_consumption", "panel": "uci_235_individual_household_power",
                "panel_sha256": "b3192c0bcb117b2ee120a906dbcfb9550cd907abff74fea9bc2b1aa320ebc8db", "timestamp": "timestamp_label", "ts_format": "%d/%m/%Y %H:%M:%S",
                "delta_seconds": 60, "unit": "mixed: kW (global active/reactive power), V (voltage), A (intensity), Wh (sub-metering 1-3)",
                "producer_notes": ["one household near Paris, one-minute sampling, 2006-12 to 2010-11",
                                   "about 1.25 % of rows have missing measurements while keeping their timestamps (producer statement); missing = NaN in the panel",
                                   "time zone: UNKNOWN in the contract (local French time presumed, DST not documented by the producer)",
                                   "availability delay and revision policy: UNKNOWN; static archive"],
                "source": "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption", "roles": {"timestamp": ["timestamp_label"]},
                "structural_zero_rule": "no structural zeros: the household exists throughout; zeros are measurements (e.g. sub-metering 3 off)"},
}
DEFAULT_ROLES = {"uci_321": {"targets": "every client column (kW)", "features": "the same columns (lagged) plus calendar (hour, weekday, DST flag) derived from the timestamp",
                             "metadata": ["timestamp_label"], "controls": "seasonal naive (daily and weekly), persistence"},
                 "uci_235": {"targets": ["Global_active_power"], "features": ["Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"],
                             "metadata": ["timestamp_label"], "controls": "persistence, seasonal naive (daily)"}}
SPLIT_FRACTIONS = {"train": 0.7, "validation": 0.15, "test": 0.15}
HORIZONS = {"uci_321": [1, 4, 96], "uci_235": [1, 60, 1440]}          # steps: next step, one hour, one day
CONTEXT_CANDIDATES_STEPS = {"uci_321": [24, 96, 192, 672], "uci_235": [60, 180, 1440, 10080]}


def _spectrum():
    import importlib.util
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("df_e1_spectrum", here / "df_e1_spectrum.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_panel(fam: dict) -> tuple:
    path = PANELS / fam["panel"] / "panel.parquet"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != fam["panel_sha256"]:
        raise SystemExit(f"REFUSED: {path} digest {digest[:12]} is not the census contract's {fam['panel_sha256'][:12]}")
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df[fam["timestamp"]], format=fam["ts_format"])
    return df, ts, {"path": str(path), "sha256": digest, "receipt": json.loads((PANELS / fam["panel"] / "PARSE_RECEIPT.json").read_text())}


def time_structure(ts: pd.Series, delta: int) -> dict:
    d = ts.diff().dt.total_seconds().dropna()
    counts = d.value_counts()
    irregular = counts[counts.index != delta]
    dst_days = {}
    for day, n in ts.dt.floor("D").value_counts().items():
        expected = 86400 // delta
        if n != expected:
            dst_days[str(day.date())] = int(n)
    return {"rows": int(ts.size), "start": str(ts.iloc[0]), "end": str(ts.iloc[-1]), "delta_seconds_declared": delta,
            "delta_seconds_observed": {str(int(k)): int(v) for k, v in counts.head(5).items()}, "irregular_steps": int(irregular.sum()),
            "monotone": bool(ts.is_monotonic_increasing), "duplicates": int(ts.duplicated().sum()),
            "days_with_row_count_not_nominal": dict(sorted(dst_days.items())[:12]), "days_with_row_count_not_nominal_total": len(dst_days)}


def column_structure(df: pd.DataFrame, fam: dict, ts: pd.Series) -> dict:
    num = df.select_dtypes(include=[np.number])
    out = {"n_numeric": int(num.shape[1]), "missing_fraction_overall": float(num.isna().mean().mean()), "columns": {}}
    for c in num.columns:
        v = num[c].to_numpy(dtype=float)
        finite = np.isfinite(v)
        nz = np.flatnonzero(finite & (v != 0))
        first_nz = int(nz[0]) if nz.size else None
        struct = int(first_nz) if (fam["structural_zero_rule"].startswith("a client") and first_nz is not None) else 0
        out["columns"][c] = {"missing": int((~finite).sum()), "zeros": int((finite & (v == 0)).sum()), "structural_zero_rows": struct,
                             "measured_zero_rows": int((finite & (v == 0)).sum()) - struct, "first_non_zero_row": first_nz,
                             "first_non_zero_time": str(ts.iloc[first_nz]) if first_nz is not None else None,
                             "quantiles": [float(q) for q in np.nanquantile(v, [0, 0.5, 1])] if finite.any() else None, "constant": bool(np.nanstd(v) == 0)}
    if len(out["columns"]) > 24:
        cols = out["columns"]
        out["columns_summary"] = {"structural_zero_rows_quantiles": [float(q) for q in np.quantile([x["structural_zero_rows"] for x in cols.values()], [0, .25, .5, .75, 1])],
                                  "missing_total": int(sum(x["missing"] for x in cols.values())), "constant_columns": [k for k, x in cols.items() if x["constant"]]}
        out["columns"] = {k: cols[k] for k in list(cols)[:8]}
        out["columns_note"] = "first 8 of the columns shown; the summary covers all"
    return out


def usable_windows(mask_ok: np.ndarray, valid_target: np.ndarray, n: int, W: int, h: int, splits: dict) -> dict:
    """Windows whose W inputs are all valid and whose target at +h is valid, per split, with purge = W + h."""
    out = {}
    ok = np.asarray(mask_ok, dtype=bool)
    csum = np.concatenate([[0], np.cumsum(ok)])
    for name, (lo, hi) in splits.items():
        origins = np.arange(max(lo, W - 1), min(hi, n - h))
        full = (csum[origins + 1] - csum[origins - W + 1]) == W
        tgt = valid_target[origins + h]
        out[name] = {"origins": int(origins.size), "usable": int((full & tgt).sum()), "range": [int(lo), int(hi)]}
    return out


def family_contract(key: str, contexts: list, horizons: list) -> dict:
    fam = FAMILIES[key]
    df, ts, gov = load_panel(fam)
    n = len(df)
    num = df.select_dtypes(include=[np.number])
    delta = fam["delta_seconds"]
    tstruct = time_structure(ts, delta)
    cstruct = column_structure(df, fam, ts)
    # splits by time, purge derived per (W, h)
    edges = {}
    cur = 0
    for name, frac in SPLIT_FRACTIONS.items():
        edges[name] = [cur, cur + int(n * frac)]
        cur = edges[name][1]
    edges["test"][1] = n
    # masks: an input row is valid if every used column is finite; a target is valid if finite and NOT a structural zero
    used = list(num.columns)
    vals = num[used].to_numpy(dtype=float)
    finite_rows = np.isfinite(vals).all(axis=1)
    struct = np.zeros(vals.shape, dtype=bool)
    for j, c in enumerate(used):
        first = cstruct["columns"].get(c, {}).get("first_non_zero_row")
        if fam["structural_zero_rule"].startswith("a client"):
            nz = np.flatnonzero(np.isfinite(vals[:, j]) & (vals[:, j] != 0))
            if nz.size:
                struct[:nz[0], j] = True
    target_valid_any = np.isfinite(vals).any(axis=1)
    target_valid_all = np.isfinite(vals).all(axis=1) & ~struct.any(axis=1)
    # train-only spectra with DECLARED resolution (RP28): per-variable and aggregate grains; bands outside the support are NO_RESUELTO
    lo, hi = edges["train"]
    SP = _spectrum()
    rule = "every declared target/feature column of the family (electricity: a declared sample of 32 clients by column order, NOT representative by proof)"
    cols = used[: min(32, len(used))]
    Xtr = num[cols].to_numpy(dtype=float)[lo:hi]
    nperseg = int(min(1 << 15, Xtr.shape[0]))
    bands_s = {"daily_24h": 86400.0, "half_day_12h": 43200.0, "weekly_168h": 7 * 86400.0, "35_days": 35 * 86400.0}
    spec = SP.per_variable_and_aggregate(Xtr, cols, delta, nperseg, bands_s, rule)
    agg = spec["aggregate"] or {}
    peaks = [{"period_hours": p["period_seconds"] / 3600.0, "share_of_total_power": p["share"]} for p in agg.get("peaks", [])]
    peaks_named = {f"{k}_share": (v.get("share") if v.get("state") == "MEDIDO" else v.get("state")) for k, v in (agg.get("bands") or {}).items()}
    # sensitivity to the train support: the same bands on the last third of train
    short = SP.per_variable_and_aggregate(Xtr[-max(Xtr.shape[0] // 3, 32):], cols, delta, nperseg, bands_s, rule)
    sens = {f"{k}_share_last_third": (v.get("share") if v.get("state") == "MEDIDO" else v.get("state")) for k, v in ((short["aggregate"] or {}).get("bands") or {}).items()}
    dst = {}
    if delta == 900:                                                     # electricity: producer statement on the March change day (one hour of zeros)
        for year in sorted(set(ts.dt.year)):
            march = ts[(ts.dt.year == year) & (ts.dt.month == 3) & (ts.dt.weekday == 6)]
            if march.empty:
                continue
            day = march.dt.floor("D").max()
            arr = num.to_numpy(dtype=float)
            hours = {}
            for hr in range(0, 6):
                sel = ((ts.dt.floor("D") == day) & (ts.dt.hour == hr)).to_numpy()
                block = arr[sel]
                hours[hr] = {"rows": int(sel.sum()), "rows_all_zero": int((block == 0).all(axis=1).sum()) if block.size else 0}
            dst[str(day.date())] = {"hours_0_to_5": hours, "producer_statement": "one hour of zeros on the March change day",
                                    "reproduced_as_all_clients_zero": any(v["rows"] > 0 and v["rows_all_zero"] == v["rows"] for v in hours.values()),
                                    "trace": "RP28_DST_TRACE.json: labels 01:00-01:45 zero for most ACTIVE clients (raw == panel); disposition AMBIGUOUS_SUPPORT",
                                    "ambiguous_labels": [(day + pd.Timedelta(hours=1, minutes=m)).strftime("%Y-%m-%d %H:%M:%S") for m in (0, 15, 30, 45)]}
        for year in sorted(set(ts.dt.year)):
            octo = ts[(ts.dt.year == year) & (ts.dt.month == 10) & (ts.dt.weekday == 6)]
            if octo.empty:
                continue
            day = octo.dt.floor("D").max()
            dst[str(day.date())] = {"producer_statement": "two hours aggregated into one on the October change day", "trace": "no zero signature; aggregation not observable from values",
                                    "disposition": "AMBIGUOUS_SUPPORT for the labels 01:00-02:00", "ambiguous_labels": [(day + pd.Timedelta(hours=1, minutes=m)).strftime("%Y-%m-%d %H:%M:%S") for m in (0, 15, 30, 45, 60)]}
    windows = {}
    for W in contexts:
        for h in horizons:
            purge = W + h
            splits = {"train": [edges["train"][0], edges["train"][1] - purge], "validation": [edges["validation"][0], edges["validation"][1] - purge],
                      "test": [edges["test"][0], edges["test"][1] - h]}
            uw = usable_windows(finite_rows, target_valid_all, n, W, h, splits)
            uw_any = usable_windows(finite_rows, target_valid_any, n, W, h, splits)
            per_col = []
            for j in range(min(vals.shape[1], 400)):
                tv = np.isfinite(vals[:, j]) & ~struct[:, j]
                per_col.append({k: v["usable"] for k, v in usable_windows(finite_rows, tv, n, W, h, splits).items()})
            per_col_q = {k: [int(q) for q in np.quantile([pc[k] for pc in per_col], [0, 0.25, 0.5, 0.75, 1])] for k in ("train", "validation", "test")}
            windows[f"W{W}_h{h}"] = {"context_steps": W, "context_physical_seconds": (W - 1) * delta, "context_physical_hours": (W - 1) * delta / 3600,
                                    "horizon_steps": h, "horizon_physical_seconds": h * delta, "purge": purge,
                                    "usable_windows_all_targets_valid": {k: v["usable"] for k, v in uw.items()},
                                    "usable_windows_any_target_valid": {k: v["usable"] for k, v in uw_any.items()},
                                    "usable_windows_per_target_column_quantiles": per_col_q,
                                    "usable_note": "'all targets valid' requires every column measured (never true early for a panel of clients that appear over time); "
                                                   "the per-column quantiles are the honest support of a multi-output task with masks; 'any' is the upper bound",
                                    "origins": {k: v["origins"] for k, v in uw.items()},
                                    "support_physical_days": {k: v["usable"] * delta / 86400 for k, v in uw.items()},
                                    "cost_note": "cost per window grows linearly with W x p; the pilot fixes seconds per update per (W, p)"}
    return {"family": key, "dataset_id": fam["dataset_id"], "governed_bytes": gov, "unit": fam["unit"], "producer_notes": fam["producer_notes"], "source": fam["source"],
            "structural_zero_rule": fam["structural_zero_rule"], "time": tstruct, "columns": cstruct, "roles": DEFAULT_ROLES[key],
            "splits_by_time": edges,
            "train_only_periodicities": {"columns_used": len(cols), "column_rule": rule, "top_peaks": peaks, "bands": peaks_named, "sensitivity_last_third_of_train": sens,
                                         "resolution": agg.get("resolution"), "per_variable": {c: {"bands": {k: (v.get("share") if v.get("state") == "MEDIDO" else v.get("state")) for k, v in e["bands"].items()},
                                                                                                       "n_missing_imputed": e["n_missing_imputed"]} for c, e in spec["per_variable"].items()},
                                         "note": "Welch on the TRAIN block only with a declared segment/resolution; a band without bins is NO_RESUELTO, never a measured zero; "
                                                 "batch characterisation of train, never an online filter; the aggregate weighs variables equally after unit-power normalisation"},
            "dst_march_change_days": dst,
            "windows": windows,
            "eligibility": {"catalogue": "OPEN_ATTRIBUTION (CC-BY-4.0) — licence eligibility, not adequacy",
                            "task": "TASK_CONTRACT_DECLARED (this document): usable windows counted per split after masks, purge and horizon; contexts in physical units; roles declared",
                            "reserve": "NOT_JUDGED_HERE (no reserve is opened by this order)"},
            "independence_note": "blocks without overlap are NOT independent replicates; columns (clients) are not independent replicates either: the cross-sectional "
                                 "dependence between clients is a measured fact (see the pair relations of the D1 profiles) and the unit of replication must be declared by task"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", default="uci_321,uci_235")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists")
    doc = {"schema": "e1_tasks.v1", "families": {}, "prior_exposure": {
        "public.uci.501.beijing_multisite_air_quality": "characterised by the D1/D2 fronts (coverage_v2 RESULT rows in the census); read for descriptors, never used for model selection; RESERVE_CANDIDATE keeps its result-level reserve defensible only for the modelling results, not for descriptive knowledge",
        "public.uci.374.appliances_energy_prediction": "same: descriptors exist in the census; not opened by this order"},
           "catalogue_note": "UCI is a repository, not one physical source: the four families are distinct producers/entities/panels; sharing the catalogue proves no dependence"}
    for key in args.families.split(","):
        doc["families"][key] = family_contract(key, CONTEXT_CANDIDATES_STEPS[key], HORIZONS[key])
        print(json.dumps({"family": key, "rows": doc["families"][key]["time"]["rows"], "irregular_steps": doc["families"][key]["time"]["irregular_steps"],
                          "dst_days": doc["families"][key]["time"]["days_with_row_count_not_nominal_total"],
                          "windows_per_column_median": {k: {sp: q[2] for sp, q in v["usable_windows_per_target_column_quantiles"].items()} for k, v in doc["families"][key]["windows"].items()},
                          "bands": doc["families"][key]["train_only_periodicities"]["bands"], "dst": doc["families"][key]["dst_march_change_days"],
                          "peaks_hours": [round(p["period_hours"], 2) for p in doc["families"][key]["train_only_periodicities"]["top_peaks"]]}, indent=1))
    args.out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
