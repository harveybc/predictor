#!/usr/bin/env python3
"""C120 read-only candidate multivariate panel inventory (DRAFT).

Reproduces PANEL_INVENTORY_DRAFT.json from bytes already on disk.
Reads only: file bytes (for sha256), schemas (parquet footer / CSV
header / .tsf header), the time column (first/last timestamp), and
existing declaration or artifact files (license records, provenance,
DAG, temporal contracts, terminal census). It reads no target, label,
forecast, score, model result or metric, and ranks nothing. It
downloads nothing and writes only the output JSON.

Usage: python panel_inventory_draft.py [--out PANEL_INVENTORY_DRAFT.json]
Roots may be overridden with env vars C120_ROOT_<ID> (ID upper-cased,
non-alphanumerics -> _).
"""
import argparse
import calendar
import datetime as dt
import glob
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

HOME = Path.home()
GH = HOME / "Documents/GitHub"


def _sm_root():
    try:
        import statsmodels
        return Path(statsmodels.__file__).parent / "datasets"
    except Exception:
        return None


ROOTS = {
    "financial-data": GH / "financial-data",
    "predictor": GH / "predictor",
    "synthetic-datagen": GH / "synthetic-datagen",
    "agent-multi-share": HOME / ".local/share/agent-multi",
    "crispdm-successors": HOME / ".local/state/crispdm-successors",
    "python-env:statsmodels.datasets": _sm_root(),
}
for k in list(ROOTS):
    env = "C120_ROOT_" + re.sub(r"[^A-Za-z0-9]", "_", k).upper()
    if os.environ.get(env):
        ROOTS[k] = Path(os.environ[env])

TIME_COLS = ["DATE_TIME", "open_time", "timestamp", "datetime", "Date",
             "date", "time", "fundingTime", "period", "TimePeriod",
             "record_date", "settlementDate", "tradeReportDate"]
SKIP_NAMES = {".gitkeep", "__init__.py"}
CONDITIONS = [
    "c1_independent_terminal", "c2_semantic_numeric_measurable",
    "c3_dag_causal_active_complete_binding", "c4_role_input_feature",
    "c5_semantic_type_unit_license_declared",
    "c6_missing_sentinel_policy", "c7_temporal_contract_and_mask",
    "c8_missingness_and_observations",
]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def loc(root, rel):
    return {"root_id": root, "relative_path": rel}


def rp(root, rel):
    r = ROOTS.get(root)
    return None if r is None else Path(r) / rel


# ---------------------------------------------------------------- bytes
def hash_target(root, rel, patterns=None, exclude=None):
    base = rp(root, rel)
    if base is None or not base.exists():
        return {"exists": False}
    if base.is_file():
        return {"exists": True, "kind": "file", "bytes": base.stat().st_size,
                "sha256": sha_file(base), "file_count": 1,
                "_files": [base]}
    files = []
    for dp, dn, fn in os.walk(base):
        dn[:] = sorted(d for d in dn if d not in ("__pycache__", ".git"))
        for f in fn:
            if f in SKIP_NAMES or f.endswith(".pyc"):
                continue
            p = Path(dp) / f
            r = p.relative_to(base).as_posix()
            if patterns and not any(re.search(x, r) for x in patterns):
                continue
            if exclude and any(re.search(x, r) for x in exclude):
                continue
            files.append(p)
    files.sort(key=lambda p: p.relative_to(base).as_posix())
    lines, total = [], 0
    for p in files:
        s = p.stat().st_size
        total += s
        lines.append(f"{p.relative_to(base).as_posix()}\t{s}\t{sha_file(p)}\n")
    manifest = "".join(lines).encode()
    return {"exists": True, "kind": "directory", "bytes": total,
            "sha256": hashlib.sha256(manifest).hexdigest(),
            "sha256_rule": "sha256 of sorted 'relpath\\tbytes\\tsha256\\n' lines",
            "file_count": len(files), "_files": files}


# ------------------------------------------------------- schema + time
def _ts_str(v):
    if v is None:
        return None
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    return str(v)


def parquet_info(p):
    pf = pq.ParquetFile(p)
    sch = pf.schema_arrow
    cols = [(f.name, str(f.type)) for f in sch]
    tcol = next((c for c in TIME_COLS if c in sch.names), None)
    first = last = None
    if tcol and pf.metadata.num_rows:
        col = pq.read_table(p, columns=[tcol]).column(tcol)
        mm = pc.min_max(col)
        first, last = _ts_str(mm["min"].as_py()), _ts_str(mm["max"].as_py())
    return {"columns": cols, "time_column": tcol, "rows": pf.metadata.num_rows,
            "first": first, "last": last,
            "time_rule": "min/max of the time column (lexical for strings)"}


def csv_info(p):
    with open(p, "rb") as f:
        header = f.readline().decode(errors="replace").rstrip("\r\n")
        first_line = f.readline().decode(errors="replace").rstrip("\r\n")
        f.seek(0, 2)
        size = f.tell()
        back = min(size, 65536)
        f.seek(size - back)
        tail = f.read().decode(errors="replace").rstrip("\r\n").split("\n")
        last_line = tail[-1].rstrip("\r")
    names = header.split(",")
    tcol = next((c for c in TIME_COLS if c in names), None)
    if header.startswith("{") or not tcol:
        return {"columns": [(n, "UNKNOWN_CSV_TEXT") for n in names] if not header.startswith("{") else [],
                "time_column": None, "rows": None, "first": None, "last": None,
                "note": "no recognised time column in header"}
    i = names.index(tcol)
    with open(p, "rb") as f:
        rows = sum(1 for _ in f) - 1
    return {"columns": [(n, "UNTYPED_CSV_TEXT") for n in names],
            "time_column": tcol, "rows": rows,
            "first": first_line.split(",")[i], "last": last_line.split(",")[i],
            "time_rule": "time column of first and last data line (file order)"}


def _add(ts, freq, n):
    if freq == "10_minutes":
        return ts + dt.timedelta(minutes=10 * n)
    if freq == "hourly":
        return ts + dt.timedelta(hours=n)
    if freq == "daily":
        return ts + dt.timedelta(days=n)
    if freq == "weekly":
        return ts + dt.timedelta(weeks=n)
    if freq == "monthly":
        m = ts.month - 1 + n
        y, m = ts.year + m // 12, m % 12 + 1
        return ts.replace(year=y, month=m,
                          day=min(ts.day, calendar.monthrange(y, m)[1]))
    return None


def tsf_info(zip_path):
    """Header attributes + per-series name/start/length. Values are not
    parsed: the length is the count of comma-separated fields."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".tsf")]
        member = names[0]
        msha = hashlib.sha256()
        attrs, hdr = [], {}
        series = []
        in_data = False
        with z.open(member) as fb:
            for raw in fb:
                msha.update(raw)
                line = raw.decode(errors="replace").strip()
                if not line or line.startswith("#"):
                    continue
                if not in_data:
                    if line.startswith("@attribute"):
                        _, n, t = line.split()[:3]
                        attrs.append((n, t))
                    elif line.lower() == "@data":
                        in_data = True
                    elif line.startswith("@"):
                        k, *v = line[1:].split(maxsplit=1)
                        hdr[k] = v[0] if v else None
                    continue
                parts = line.split(":")
                vals = parts[-1]
                rec = dict(zip([a[0] for a in attrs], parts[:len(attrs)]))
                rec["length"] = vals.count(",") + 1
                series.append(rec)
    freq = hdr.get("frequency")
    firsts, lasts = [], []
    if any(a[0] == "start_timestamp" for a in attrs):
        for s in series:
            ts = dt.datetime.strptime(s["start_timestamp"], "%Y-%m-%d %H-%M-%S")
            end = _add(ts, freq, s["length"] - 1)
            firsts.append(ts)
            if end:
                lasts.append(end)
    lengths = [s["length"] for s in series]
    types = sorted({s.get("series_type") for s in series if s.get("series_type")})
    return {
        "tsf_member": member, "tsf_member_sha256": msha.hexdigest(),
        "header": hdr, "attributes": attrs, "series_count": len(series),
        "series_types": types,
        "length_min": min(lengths), "length_max": max(lengths),
        "first": min(firsts).isoformat() if firsts else None,
        "last": max(lasts).isoformat() if lasts else None,
        "time_rule": ("start_timestamp + (length-1) x declared frequency"
                      if firsts else "UNKNOWN: the .tsf declares no start_timestamp attribute"),
        "variables": [{"name": s.get("series_name"), "type": "float (tsf text)",
                       **({"series_type": s["series_type"]} if "series_type" in s else {})}
                      for s in series],
    }


def schema_union(files, base):
    per, first, last, rows_max, tcols = [], [], [], 0, set()
    distinct = {}
    for p in files:
        suf = p.suffix.lower()
        try:
            if suf == ".parquet":
                info = parquet_info(p)
            elif suf == ".csv":
                info = csv_info(p)
            else:
                continue
        except Exception as e:  # recorded, never silently dropped
            per.append({"member": p.relative_to(base).as_posix() if base.is_dir() else p.name,
                        "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        rel = p.relative_to(base).as_posix() if base.is_dir() else p.name
        per.append({"member": rel, "rows": info["rows"], "time_column": info["time_column"],
                    "n_columns": len(info["columns"])})
        for n, t in info["columns"]:
            distinct.setdefault((n, t), 0)
            distinct[(n, t)] += 1
        if info["time_column"]:
            tcols.add(info["time_column"])
        if info["first"]:
            first.append(info["first"])
        if info["last"]:
            last.append(info["last"])
        rows_max = max(rows_max, info["rows"] or 0)
    tabular = [m for m in per if "error" not in m]
    return {
        "members": per, "tabular_member_count": len(tabular),
        "time_columns": sorted(tcols),
        "first": min(first) if first else None,
        "last": max(last) if last else None,
        "rows_max_member": rows_max,
        "variables_distinct": [{"column": n, "physical_type": t, "members_with_column": c}
                               for (n, t), c in sorted(distinct.items())],
    }


def provenance_sources(root, rel):
    base = rp(root, rel)
    out = {}
    if base is None or not base.exists():
        return out
    for p in glob.glob(str(base / "**/provenance.json"), recursive=True):
        try:
            s = str(json.load(open(p)).get("source"))
        except Exception:
            s = "UNPARSEABLE"
        out[s] = out.get(s, 0) + 1
    return out


def license_scan(root, rel):
    """Search provenance/README/data_dictionary under the target for a
    license declaration. Returns found lines with file sha256."""
    base = rp(root, rel)
    hits = []
    if base is None or not base.exists() or base.is_file():
        return hits
    for pat in ("**/provenance.json", "**/README.md", "**/data_dictionary.md"):
        for p in glob.glob(str(base / pat), recursive=True):
            t = open(p, errors="replace").read()
            for m in re.finditer(r"[^\n]*(licen[cs]e|terms of (use|service)|public domain|CC[- ]BY)[^\n]*", t, re.I):
                hits.append({"file": loc(root, Path(p).relative_to(ROOTS[root]).as_posix()),
                             "sha256": sha_file(p), "line": m.group(0)[:200]})
    return hits


# ------------------------------------------------------ join artifacts
FD_CENSUS = "features/census"


def artifact_state():
    fd = rp("financial-data", FD_CENSUS)
    st = {"dag": {}, "temporal": {}, "terminal_census_datasets": set(),
          "semantic_census_files": [], "in_progress_not_read": []}
    dagp = fd / "FEATURE_DAG.v4.json"
    if dagp.exists():
        d = json.load(open(dagp))
        for n in d.get("nodes", []):
            k = n.get("dataset_id")
            e = st["dag"].setdefault(k, {"file": loc("financial-data", f"{FD_CENSUS}/FEATURE_DAG.v4.json"),
                                         "sha256": sha_file(dagp), "causal_active": 0,
                                         "causal_active_complete_binding": 0, "nodes": 0})
            e["nodes"] += 1
            if n.get("class") == "CAUSAL_ACTIVE":
                e["causal_active"] += 1
                if (n.get("binding") or {}).get("complete") is True:
                    e["causal_active_complete_binding"] += 1
    for v in ("v1", "v2"):
        p = fd / f"ETH_H4_TEMPORAL_CONTRACT.{v}.json"
        if p.exists():
            d = json.load(open(p))
            ds = (d.get("dataset") or {}).get("dataset_id")
            if not ds:
                m = re.search(r'"dataset_id": "([^"]+)"', json.dumps(d))
                ds = m.group(1) if m else None
            st["temporal"].setdefault(ds, []).append(
                {"file": loc("financial-data", f"{FD_CENSUS}/{p.name}"), "sha256": sha_file(p),
                 "mask_declared": bool(d.get("mask_artifact"))})
    # any newer temporal contract / semantic census: existence only (another
    # agent is producing them; not read here)
    for p in sorted(glob.glob(str(fd / "ETH_H4_TEMPORAL_CONTRACT.v[3-9]*.json"))) + \
            sorted(glob.glob(str(fd / "*SEMANTIC*"))) + \
            sorted(glob.glob(str(fd / "*CHARACTERI*"))):
        st["in_progress_not_read"].append(loc("financial-data", f"{FD_CENSUS}/{Path(p).name}"))
    for p in glob.glob(str(fd / "artifacts/census-*.json")):
        t = open(p).read()
        for ds in set(re.findall(r'"dataset_id": "([^"]+)"', t)):
            st["terminal_census_datasets"].add(ds)
    ev = rp("predictor", "docs/audits/evidence/TERMINALS_V4_DISPOSITION.v1.json")
    st["terminals_v4"] = ({"file": loc("predictor", "docs/audits/evidence/TERMINALS_V4_DISPOSITION.v1.json"),
                           "sha256": sha_file(ev)} if ev and ev.exists() else None)
    return st


ABSENT = "ABSENT: no artifact for this dataset found on disk"


def join_artifacts(dataset_id, st):
    a = {c: ABSENT for c in CONDITIONS}
    if dataset_id in st["terminal_census_datasets"] and st.get("terminals_v4"):
        a["c1_independent_terminal"] = {
            "state": "PRESENT_DATASET_LEVEL",
            "evidence": st["terminals_v4"],
            "note": "dataset appears in the lake terminal census accepted by TERMINALS_V4 "
                    "(scope: physical/statistical only); per-column join not evaluated here"}
    dg = st["dag"].get(dataset_id)
    if dg:
        a["c3_dag_causal_active_complete_binding"] = {"state": "PRESENT", **dg}
    tc = st["temporal"].get(dataset_id)
    if tc:
        a["c7_temporal_contract_and_mask"] = {"state": "PRESENT", "contracts": tc}
    a["c5_semantic_type_unit_license_declared"] = (
        "ABSENT: TERMINALS_V4 scope SEMANTIC_DECLARATIONS_KNOWN = 0 lake-wide"
        if dataset_id in st["terminal_census_datasets"] else ABSENT)
    a["c8_missingness_and_observations"] = "NOT_EVALUATED: requires the member-by-member join and a bound temporal mask"
    return a


# ---------------------------------------------------------- panel specs
MONASH = [
    # id, zip, record, family, dependence
    ("electricity_weekly", 4656141, "electricity_demand_uci_portugal_clients"),
    ("solar_10_minutes", 4656144, "solar_power_nrel_alabama_2006"),
    ("solar_weekly", 4656151, "solar_power_nrel_alabama_2006"),
    ("pedestrian_counts", 4656626, "urban_pedestrian_melbourne_sensors"),
    ("weather", 4654822, "weather_australia_bom_stations"),
    ("tourism_monthly", 4656096, "tourism_kaggle_competition"),
    ("hospital", 4656014, "health_hospital_expsmooth"),
    ("saugeenday", 4656058, "hydrology_saugeen_river"),
    ("us_births", 4656049, "demography_us_births"),
]

FD_PANELS = [
    # panel_id, rel, family, instrument/source, provider_decl, frequency, dataset_id
    ("fd_crypto_spot_top50", "market_data/crypto/spot_top50", "F_BINANCE", "Binance spot pairs (32 instruments)", "5m/15m/1h/4h (file names)"),
    ("fd_crypto_perpetuals", "market_data/crypto/perpetuals", "F_BINANCE", "Binance USDT-M perpetuals (10 instruments)", "5m/15m/1h/4h (file names)"),
    ("fd_crypto_funding_rates", "market_data/crypto/funding_rates", "F_BINANCE", "Binance perpetual funding rates (10 instruments)", "event (funding times)"),
    ("fd_trading_asset_data", "features/trading_asset_data", "F_BINANCE+F_HISTDATA", "stage copies of Binance spot/perp and HistData FX bars (50 assets)", "5m/15m/1h/4h (file names)"),
    ("fd_forex_g10", "market_data/forex/g10", "F_HISTDATA", "HistData G10 FX pairs (10 instruments)", "5m/15m/1h/4h (file names)"),
    ("fd_forex_emerging", "market_data/forex/emerging_markets", "F_YAHOO", "Yahoo Finance EM FX (10 instruments)", "daily (file names)"),
    ("fd_equities_yahoo", "market_data/equities", "F_YAHOO", "Yahoo Finance ETFs, global and US indices (54 instruments)", "daily (file names)"),
    ("fd_commodities_yahoo", "market_data/commodities", "F_YAHOO", "Yahoo Finance commodity futures (14 instruments)", "daily (file names)"),
    ("fd_fred", "macro_economic/fred", "F_FRED", "FRED series (135 series, 15 groups)", "mixed (per series; not declared at panel level)"),
    ("fd_economic_calendar", "economic_calendar", "F_FRED+F_FXMACRODATA", "FRED release actuals and FXMacroData events", "event"),
    ("fd_oecd_cli", "macro_economic/oecd", "F_OECD", "OECD composite leading indicators (SDMX)", "monthly (file name)"),
    ("fd_bea_nipa", "macro_economic/bea", "F_BEA", "BEA NIPA tables", "quarterly (file name)"),
    ("fd_bls", "macro_economic/bls", "F_BLS", "BLS public series", "UNKNOWN"),
    ("fd_treasury_avg_rates", "macro_economic/yield_curves", "F_TREASURY", "US Treasury FiscalData average interest rates", "monthly record_date (not declared)"),
    ("fd_cftc_cot", "alternative_data/cot_reports", "F_CFTC", "CFTC disaggregated COT (raw zips)", "weekly (not declared)"),
    ("fd_finra_short", "alternative_data/short_interest", "F_FINRA", "FINRA consolidated short interest and Reg SHO daily", "semi-monthly / daily (not declared)"),
    ("fd_coinmetrics_community", "alternative_data", "F_COINMETRICS", "CoinMetrics Community on-chain metrics BTC and ETH", "daily (not declared)"),
    ("fd_cryptoquant", "alternative_data/cryptoquant", "F_CRYPTOQUANT", "CryptoQuant exchange/miner/stablecoin flows", "daily (not declared)"),
    ("fd_onchain_snapshots_misc", "alternative_data", "F_ONCHAIN_MISC", "Blockchain.com, mempool.space, Etherscan, DeFiLlama snapshots", "snapshots (not declared)"),
]
FD_SUBSETS = {
    "fd_coinmetrics_community": [r"coinmetrics_community/"],
    "fd_onchain_snapshots_misc": [r"onchain_btc/blockchain_com/", r"onchain_btc/mempool_space/",
                                  r"onchain_eth/etherscan_free_snapshots/", r"defi_metrics/defillama/"],
}

FAMILY_REPRESENTATIVE = {
    "F_BINANCE": "eth_h4_successor", "F_HISTDATA": "fd_forex_g10", "F_YAHOO": "fd_equities_yahoo",
    "F_FRED": "fd_fred", "F_OECD": "fd_oecd_cli", "F_BEA": "fd_bea_nipa", "F_BLS": "fd_bls",
    "F_TREASURY": "fd_treasury_avg_rates", "F_CFTC": "fd_cftc_cot", "F_FINRA": "fd_finra_short",
    "F_COINMETRICS": "fd_coinmetrics_community", "F_CRYPTOQUANT": "fd_cryptoquant",
    "F_ONCHAIN_MISC": "fd_onchain_snapshots_misc", "F_PREDICTOR_LEGACY": "predictor_legacy_eurusd_1h_phase1",
    "F_ETT": "etth1",
}


def build():
    st = artifact_state()
    panels = []

    def finalize(p, h, sch=None):
        p["bytes"] = h.get("bytes")
        p["sha256"] = h.get("sha256")
        p["hash_kind"] = h.get("kind")
        p["file_count"] = h.get("file_count")
        if h.get("sha256_rule"):
            p["sha256_rule"] = h["sha256_rule"]
        panels.append(p)

    # --- ETH successor ---------------------------------------------------
    succ_id = "financial_data.project3.ethusdt_4h_tech_stat.model_ready.successor_stage22_rerun.v1"
    rel = "eth_h4_stage22_rerun_v1/successor"
    h = hash_target("crispdm-successors", rel)
    base = rp("crispdm-successors", rel)
    sch = schema_union([f for f in h["_files"] if f.suffix == ".parquet"], base)
    summ = rp("financial-data", f"{FD_CENSUS}/ETH_H4_STAGE22_RERUN_SUMMARY.v1.json")
    declared = json.load(open(summ)).get("successor_dataset_sha256")
    member_sha = {f.name: sha_file(f) for f in h["_files"]}
    vars_ = [v for v in sch["variables_distinct"] if v["column"] != "DATE_TIME"]
    p = {
        "panel_id": "eth_h4_successor", "dataset_id": succ_id, "independence_family": "F_BINANCE",
        "location_logical": loc("crispdm-successors", rel),
        "member_sha256": member_sha,
        "declared_successor_dataset_sha256": {"value": declared,
                                              "source": loc("financial-data", f"{FD_CENSUS}/ETH_H4_STAGE22_RERUN_SUMMARY.v1.json"),
                                              "matches_member": [k for k, v in member_sha.items() if v == declared]},
        "source_or_instrument": "ETHUSDT spot 4h (upstream features/trading_asset_data/ethusdt/4h.parquet per PRE_RUN_MANIFEST)",
        "provider": "Binance Spot (per market_data/crypto/spot_top50/ethusdt/provenance.json 'source'); derived by Project 3 stage22 rerun",
        "frequency": "4h (declared nominal_bar_seconds 14400 in ETH_H4_TEMPORAL_CONTRACT.v2 for the historical dataset; not re-declared for the successor here)",
        "time_range": {"first": sch["first"], "last": sch["last"], "time_column": sch["time_columns"]},
        "rows": sch["rows_max_member"],
        "variables": vars_, "n_variables": len(vars_),
        "license": "UNKNOWN",
        "license_why": "no license declaration found for Binance data in financial-data (README.md: 'No license file is present'); "
                       "provenance.json carries no license field; lake census and crispdm inventory record license UNKNOWN/UNDECLARED. "
                       "License determination is owned by a separate agent (not read, not written here).",
        "license_source": None,
        "citation": None,
        "dependence": {"shares_feed_instrument_with": ["eth_h4_model_ready_v1", "fd_crypto_spot_top50", "fd_trading_asset_data",
                                                        "fd_crypto_perpetuals", "fd_crypto_funding_rates", "synthetic_datagen_all"],
                       "underlying": "ETHUSDT Binance spot bars", "modelled": "NOT_MODELLED"},
        "artifacts_present_for_join": join_artifacts(succ_id, st),
    }
    a = p["artifacts_present_for_join"]
    a["c2_semantic_numeric_measurable"] = "ABSENT_AT_INVENTORY_TIME: successor semantic census being built by another agent (not read)"
    a["c4_role_input_feature"] = "ABSENT_AT_INVENTORY_TIME: role declarations expected from the in-progress semantic census (not read)"
    a["c7_temporal_contract_and_mask"] = (a["c7_temporal_contract_and_mask"] if isinstance(a["c7_temporal_contract_and_mask"], dict)
                                          else "ABSENT_FOR_THIS_DATASET_ID: contracts v1/v2 bind the historical model_ready.v1 dataset; v3 for the successor is in progress (not read)")
    a["c1_independent_terminal"] = (a["c1_independent_terminal"] if isinstance(a["c1_independent_terminal"], dict)
                                    else "ABSENT_FOR_THIS_DATASET_ID: lake terminal census covers model_ready.v1 and the EURUSD legacy dataset only; successor characterization in progress (not read)")
    p["in_progress_artifacts_seen_not_read"] = st["in_progress_not_read"]
    finalize(p, h)

    # --- ETH historical model_ready.v1 (predictor examples) ---------------
    hist_id = "financial_data.project3.ethusdt_4h_tech_stat.model_ready.v1"
    rel = "examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv"
    h = hash_target("predictor", rel)
    ci = csv_info(rp("predictor", rel))
    p = {
        "panel_id": "eth_h4_model_ready_v1", "dataset_id": hist_id, "independence_family": "F_BINANCE",
        "location_logical": loc("predictor", rel),
        "source_or_instrument": "ETHUSDT spot 4h (historical export; superseded by the stage22 successor)",
        "provider": "Binance-derived Project 3 export (per crispdm_dataset_inventory.v1 'provider')",
        "frequency": "4h (nominal_bar_seconds 14400 per ETH_H4_TEMPORAL_CONTRACT.v2)",
        "time_range": {"first": ci["first"], "last": ci["last"], "time_column": ci["time_column"]},
        "rows": ci["rows"],
        "variables": [{"column": n, "physical_type": t} for n, t in ci["columns"] if n != "DATE_TIME"],
        "n_variables": len(ci["columns"]) - 1,
        "license": "UNKNOWN", "license_why": "crispdm_dataset_inventory.v1 records license_id UNDECLARED (metadata_issues LICENSE_UNDECLARED)",
        "license_source": {"file": loc("predictor", "examples/research/crispdm_dataset_inventory.v1.json"),
                           "sha256": sha_file(rp("predictor", "examples/research/crispdm_dataset_inventory.v1.json")),
                           "states": "license_id UNDECLARED"},
        "citation": None,
        "dependence": {"shares_feed_instrument_with": ["eth_h4_successor", "fd_crypto_spot_top50", "fd_trading_asset_data", "synthetic_datagen_all"],
                       "underlying": "same ETHUSDT series as the successor", "modelled": "NOT_MODELLED"},
        "artifacts_present_for_join": join_artifacts(hist_id, st),
    }
    finalize(p, h)

    # --- predictor legacy ------------------------------------------------
    leg_id = "predictor.legacy.eurusd_1h.phase1_test.v1"
    rel = "examples/data/phase_1/base_d6.csv"
    h = hash_target("predictor", rel)
    ci = csv_info(rp("predictor", rel))
    p = {
        "panel_id": "predictor_legacy_eurusd_1h_phase1", "dataset_id": leg_id, "independence_family": "F_PREDICTOR_LEGACY",
        "location_logical": loc("predictor", rel),
        "source_or_instrument": "EURUSD 1h per dataset_id naming in crispdm_dataset_inventory.v1; instrument not otherwise declared",
        "provider": "UNKNOWN (inventory: 'legacy predictor dataset; provider not recorded')",
        "frequency": "UNKNOWN_DECLARED (inventory measured median sampling 3600 s)",
        "time_range": {"first": ci["first"], "last": ci["last"], "time_column": ci["time_column"]},
        "rows": ci["rows"],
        "variables": [{"column": n, "physical_type": t} for n, t in ci["columns"] if n != "DATE_TIME"],
        "n_variables": len(ci["columns"]) - 1,
        "license": "UNKNOWN", "license_why": "crispdm_dataset_inventory.v1 records license_id UNDECLARED",
        "license_source": {"file": loc("predictor", "examples/research/crispdm_dataset_inventory.v1.json"),
                           "sha256": sha_file(rp("predictor", "examples/research/crispdm_dataset_inventory.v1.json")),
                           "states": "license_id UNDECLARED"},
        "citation": None,
        "dependence": {"shares_feed_instrument_with": ["fd_forex_g10", "fd_trading_asset_data", "predictor_legacy_other_ohlc"],
                       "underlying": "EURUSD (if the dataset_id naming is right); provider unknown so feed overlap with HistData is UNKNOWN",
                       "modelled": "NOT_MODELLED"},
        "artifacts_present_for_join": join_artifacts(leg_id, st),
        "structural_notes": ["OHLC plus four bar-difference columns: 8 non-time columns; derived columns are functions of OHLC"],
    }
    finalize(p, h)

    rel = "examples/data"
    h = hash_target("predictor", rel, patterns=[r"^phase_[0-9_]+/base_d[0-9]+\.csv$"], exclude=[r"^phase_1/base_d6\.csv$"])
    sch = schema_union(h["_files"], rp("predictor", rel))
    p = {
        "panel_id": "predictor_legacy_other_ohlc", "dataset_id": "UNREGISTERED (no dataset_id found)", "independence_family": "F_PREDICTOR_LEGACY",
        "location_logical": loc("predictor", rel), "member_rule": "phase_*/base_d*.csv except phase_1/base_d6.csv",
        "source_or_instrument": "UNKNOWN (no provenance or dictionary in these directories)",
        "provider": "UNKNOWN", "frequency": "UNKNOWN",
        "time_range": {"first": sch["first"], "last": sch["last"], "time_column": sch["time_columns"],
                       "note": "lexical min/max across members of first/last lines"},
        "members": sch["members"], "variables": sch["variables_distinct"],
        "n_variables": len([v for v in sch["variables_distinct"] if v["column"] != "DATE_TIME"]),
        "license": "UNKNOWN", "license_why": "no license, provenance or README found in these directories", "license_source": None,
        "citation": None,
        "dependence": {"shares_feed_instrument_with": ["predictor_legacy_eurusd_1h_phase1"], "underlying": "UNKNOWN; phase_1 splits look like siblings of base_d6", "modelled": "NOT_MODELLED"},
        "artifacts_present_for_join": join_artifacts("__none__", st),
    }
    finalize(p, h)

    # --- financial-data families -----------------------------------------
    fd_readme = rp("financial-data", "README.md")
    fd_readme_sha = sha_file(fd_readme)
    for pid, rel, fam, inst, freq in FD_PANELS:
        h = hash_target("financial-data", rel, patterns=FD_SUBSETS.get(pid))
        if not h.get("exists"):
            continue
        sch = schema_union(h["_files"], rp("financial-data", rel))
        prov = {}
        for f in h["_files"]:
            if f.name == "provenance.json":
                try:
                    s = str(json.load(open(f)).get("source"))
                except Exception:
                    s = "UNPARSEABLE"
                prov[s] = prov.get(s, 0) + 1
        lic_hits = [x for x in license_scan("financial-data", rel)
                    if not FD_SUBSETS.get(pid) or any(re.search(r, x["file"]["relative_path"]) for r in FD_SUBSETS[pid])]
        iso = re.compile(r"^\d{4}([-Q]\d|$)")
        if not (sch["first"] and sch["last"] and iso.match(str(sch["first"])) and iso.match(str(sch["last"]))):
            sch["time_unreliable_raw"] = [sch["first"], sch["last"]]
            sch["first"] = sch["last"] = None
        deps = [q for q, _, f2, _, _ in FD_PANELS if q != pid and (set(f2.split("+")) & set(fam.split("+")))]
        if "F_BINANCE" in fam:
            deps += ["eth_h4_successor", "eth_h4_model_ready_v1"]
        if pid == "fd_coinmetrics_community":
            deps += ["fd_crypto_spot_top50 (same BTC/ETH assets, different measurement source)", "eth_h4_successor (ETH asset)"]
        if pid == "fd_forex_g10":
            deps += ["predictor_legacy_eurusd_1h_phase1 (EURUSD instrument)"]
        p = {
            "panel_id": pid, "dataset_id": f"UNREGISTERED: financial-data/{rel}", "independence_family": fam,
            "location_logical": loc("financial-data", rel),
            **({"member_rule": FD_SUBSETS[pid]} if pid in FD_SUBSETS else {}),
            "source_or_instrument": inst,
            "provider": {"provenance_json_source_counts": prov} if prov else "UNKNOWN (no provenance.json in scope)",
            "frequency": freq,
            "time_range": {"first": sch["first"], "last": sch["last"], "time_columns": sch["time_columns"],
                           "note": "min/max across tabular members; string time columns compared lexically; mixed timezones not normalised"},
            "tabular_member_count": sch["tabular_member_count"], "rows_max_member": sch["rows_max_member"],
            "variables": sch["variables_distinct"],
            "n_variables": len([v for v in sch["variables_distinct"] if v["column"] not in TIME_COLS]),
            "n_variables_rule": "distinct (column, physical type) pairs across members excluding recognised time columns; long-format tables (FRED/OECD/BEA/Treasury/CoinMetrics) hold one value column per member, so series count is tabular_member_count or the key-column cardinality (not computed)",
            "license": "UNKNOWN",
            "license_why": ("no license declaration found in provenance.json / README.md / data_dictionary.md in scope; "
                            "financial-data README.md 'License' section states no license file is present and no reuse permission is granted "
                            "beyond rights supplied by the original sources" if not lic_hits
                            else "license-like lines found; see license_scan_hits; not adjudicated (LICENSE_REVIEW_REQUIRED)"),
            "license_source": {"file": loc("financial-data", "README.md"), "sha256": fd_readme_sha,
                               "states": "No license file is present. No permission to reuse the repository contents is granted beyond rights supplied by the original data sources."},
            "license_scan_hits": lic_hits,
            "citation": None,
            "dependence": {"shares_provider_feed_or_instrument_with": sorted(set(deps)), "modelled": "NOT_MODELLED"},
            "artifacts_present_for_join": join_artifacts("__none__", st),
        }
        if lic_hits:
            p["license"] = "LICENSE_REVIEW_REQUIRED"
        finalize(p, h)

    # --- Monash / Zenodo ---------------------------------------------------
    man_p = rp("agent-multi-share", "t2_public_data_manifest_20260906.json")
    man = json.load(open(man_p))
    man_sha = sha_file(man_p)
    fam_of = {k: f for k, _, f in MONASH}
    for lid, rec, fam in MONASH:
        md = man["datasets"][lid]
        rel = f"t2_public_raw/{md['local_relpath']}"
        h = hash_target("agent-multi-share", rel)
        ti = tsf_info(rp("agent-multi-share", rel))
        recp = rp("agent-multi-share", f"t2_public_raw/record_{rec}.json")
        recj = json.load(open(recp))
        rmd = recj.get("metadata", {})
        lic_id = (rmd.get("license") or {}).get("id", "UNKNOWN")
        deps = [k for k, f in fam_of.items() if f == fam and k != lid]
        notes = []
        if ti["series_count"] < 5:
            notes.append(f"only {ti['series_count']} series: cannot supply 5 variables")
        if ti["length_max"] < 2000:
            notes.append(f"longest series has {ti['length_max']} observations (< 2000)")
        if ti["header"].get("equallength") == "false":
            notes.append("unequal series lengths: a common time grid must be declared before members can form one panel")
        if ti["first"] is None:
            notes.append("no start_timestamp attribute: time range and alignment are UNKNOWN")
        if ti["series_types"]:
            notes.append(f"series_type values {ti['series_types']}: per-station groups have {len(ti['series_types'])} variables")
        p = {
            "panel_id": f"monash_{lid}", "dataset_id": f"UNREGISTERED: zenodo:{rec} {ti['tsf_member']}",
            "independence_family": f"M_{fam}",
            "location_logical": loc("agent-multi-share", rel),
            "tsf_member_sha256": ti["tsf_member_sha256"],
            "manifest": {"file": loc("agent-multi-share", "t2_public_data_manifest_20260906.json"), "sha256": man_sha,
                         "declared_sha256": md["sha256"], "matches": md["sha256"] == h["sha256"]},
            "source_or_instrument": re.sub(r"<[^>]+>|&nbsp;", " ", rmd.get("description", "")).strip(),
            "provider": "Monash Time Series Forecasting Repository via Zenodo (distributor); upstream as described in the record",
            "frequency": ti["header"].get("frequency"),
            "time_range": {"first": ti["first"], "last": ti["last"], "rule": ti["time_rule"]},
            "tsf_header": ti["header"], "series_count": ti["series_count"],
            "series_length_min": ti["length_min"], "series_length_max": ti["length_max"],
            "variables": ti["variables"], "n_variables": ti["series_count"],
            "license": lic_id.upper() if lic_id != "UNKNOWN" else "UNKNOWN",
            "license_source": {"file": loc("agent-multi-share", f"t2_public_raw/record_{rec}.json"), "sha256": sha_file(recp),
                               "states": f"metadata.license.id = {lic_id}", "license_text_sha256": "UNAVAILABLE (record gives an identifier, not text)"},
            "citation": md.get("citation"),
            "doi": rmd.get("doi"),
            "dependence": {"shares_underlying_series_with": deps,
                           "shares_distributor_with": [f"monash_{k}" for k, _, _ in MONASH if k != lid],
                           "modelled": "NOT_MODELLED"},
            "artifacts_present_for_join": join_artifacts("__none__", st),
            "structural_notes": notes,
        }
        finalize(p, h)

    # --- ETTh1 -----------------------------------------------------------
    md = man["datasets"]["etth1"]
    rel = f"t2_public_raw/{md['local_relpath']}"
    h = hash_target("agent-multi-share", rel)
    ci = csv_info(rp("agent-multi-share", rel))
    licp = rp("agent-multi-share", "t2_public_raw/etth1__LICENSE")
    p = {
        "panel_id": "etth1", "dataset_id": "UNREGISTERED: ETDataset ETT-small/ETTh1.csv", "independence_family": "F_ETT",
        "location_logical": loc("agent-multi-share", rel),
        "manifest": {"file": loc("agent-multi-share", "t2_public_data_manifest_20260906.json"), "sha256": man_sha,
                     "declared_sha256": md["sha256"], "matches": md["sha256"] == h["sha256"], "admission": md.get("admission")},
        "source_or_instrument": "one electricity transformer (ETT-small h1): load columns and oil temperature",
        "provider": "ETDataset GitHub repository (zhouhaoyi/ETDataset)",
        "frequency": "UNKNOWN_DECLARED (file name 'h1'; not declared in bytes read)",
        "time_range": {"first": ci["first"], "last": ci["last"], "time_column": ci["time_column"]},
        "rows": ci["rows"],
        "variables": [{"column": n, "physical_type": t} for n, t in ci["columns"] if n != "date"],
        "n_variables": len(ci["columns"]) - 1,
        "license": "LICENSE_REVIEW_REQUIRED",
        "license_declared": "CC-BY-ND-4.0",
        "license_why": "the LICENSE file is Creative Commons Attribution-NoDerivatives 4.0; whether per-variable preprocessing outputs are "
                       "derivatives is unresolved; manifest admission is EXCLUDED_FROM_T2_CONFIRMATORY",
        "license_source": {"file": loc("agent-multi-share", "t2_public_raw/etth1__LICENSE"), "sha256": sha_file(licp),
                           "states": open(licp, errors="replace").readline().strip()},
        "citation": md.get("citation"),
        "dependence": {"shares_provider_feed_or_instrument_with": [], "modelled": "NOT_MODELLED"},
        "artifacts_present_for_join": join_artifacts("__none__", st),
    }
    finalize(p, h)

    # --- statsmodels dev units --------------------------------------------
    smr = ROOTS["python-env:statsmodels.datasets"]
    if smr and smr.exists():
        for mod in ("co2", "elnino", "sunspots", "nile"):
            rel = f"{mod}/{mod}.csv"
            h = hash_target("python-env:statsmodels.datasets", rel)
            if not h.get("exists"):
                continue
            src = smr / mod / "data.py"
            txt = open(src).read()
            m = re.search(r'COPYRIGHT\s*=\s*"""(.*?)"""', txt, re.S)
            with open(smr / rel) as f:
                header = f.readline().strip().split(",")
                rows = sum(1 for _ in f)
            p = {
                "panel_id": f"statsmodels_{mod}", "dataset_id": f"UNREGISTERED: statsmodels.datasets.{mod}",
                "independence_family": f"S_statsmodels_{mod}",
                "location_logical": loc("python-env:statsmodels.datasets", rel),
                "source_or_instrument": re.search(r'TITLE\s*=\s*(?:"""(.*?)"""|__doc__)', txt, re.S).group(1) or "see data.py __doc__",
                "provider": "statsmodels package bundled dataset",
                "frequency": "UNKNOWN_DECLARED_IN_BYTES_READ",
                "time_range": "UNKNOWN: not extracted (panel structurally excluded)",
                "rows": rows, "variables": [{"column": c, "physical_type": "UNTYPED_CSV_TEXT"} for c in header],
                "n_variables": len(header),
                "license": "UNKNOWN",
                "license_declared_text": m.group(1).strip() if m else None,
                "license_why": "statsmodels data.py COPYRIGHT string is a packager statement, not a license from the original source; recorded but not adjudicated",
                "license_source": {"file": loc("python-env:statsmodels.datasets", f"{mod}/data.py"), "sha256": sha_file(src)},
                "citation": None,
                "dependence": {"modelled": "NOT_MODELLED", "shares_provider_feed_or_instrument_with": []},
                "artifacts_present_for_join": join_artifacts("__none__", st),
                "structural_notes": ["univariate or month-as-column table; fewer than 2000 observations except co2; not a >=5-variable panel"],
                "excluded_class": "DEVELOPMENT_ONLY_UNIVARIATE",
            }
            finalize(p, h)

    # --- synthetic ---------------------------------------------------------
    rel = "."
    h = hash_target("synthetic-datagen", rel, patterns=[r"\.(csv|parquet)$"])
    p = {
        "panel_id": "synthetic_datagen_all", "dataset_id": "UNREGISTERED: synthetic-datagen csv/parquet files",
        "independence_family": "X_SYNTHETIC", "location_logical": loc("synthetic-datagen", "**/*.csv|*.parquet"),
        "source_or_instrument": "synthetic series and fixtures, including examples/data/ethusdt_4h_full_8yr.csv (declared in AGENTS.md as an ETH/USDT 4h input fixture)",
        "provider": "synthetic-datagen generators and copied real fixtures",
        "frequency": "UNKNOWN", "time_range": "NOT_EXTRACTED: excluded class", "variables": "NOT_EXTRACTED: excluded class",
        "n_variables": None,
        "license": "UNKNOWN", "license_why": "no LICENSE file at repository root", "license_source": None, "citation": None,
        "dependence": {"shares_provider_feed_or_instrument_with": ["eth_h4_successor", "eth_h4_model_ready_v1"], "modelled": "NOT_MODELLED"},
        "artifacts_present_for_join": join_artifacts("__none__", st),
        "excluded_class": "SYNTHETIC_NOT_AN_INDEPENDENT_REAL_PANEL",
    }
    finalize(p, h)

    # --- counts_toward_six --------------------------------------------------
    for p in panels:
        why = []
        fam = p["independence_family"]
        rep = FAMILY_REPRESENTATIVE.get(fam)
        if "+" in fam:
            why.append(f"mixed family {fam}: copies/derivations of other feeds, dependence NOT_MODELLED")
        elif rep and rep != p["panel_id"]:
            why.append(f"not independent: shares family {fam} with representative {rep}; dependence NOT_MODELLED")
        dep = p.get("dependence", {})
        if fam.startswith("M_") and dep.get("shares_underlying_series_with"):
            why.append(f"shares underlying series with {dep['shares_underlying_series_with']}; dependence NOT_MODELLED")
        if p.get("excluded_class"):
            why.append(p["excluded_class"])
        if p["license"] in ("UNKNOWN", "LICENSE_REVIEW_REQUIRED"):
            why.append(f"license {p['license']}")
        missing = [c for c, v in p["artifacts_present_for_join"].items()
                   if not (isinstance(v, dict) and str(v.get("state", "")).startswith("PRESENT"))]
        if missing:
            why.append("join artifacts absent or not evaluable: " + ", ".join(missing))
        why += [f"structural: {n}" for n in p.get("structural_notes", [])]
        if isinstance(p.get("n_variables"), int) and p["n_variables"] < 5:
            why.append(f"structural: n_variables {p['n_variables']} < 5")
        p["counts_toward_six"] = not why
        p["why_not"] = why

    return {
        "schema": "c120.panel_inventory_draft.v1",
        "status": "DRAFT_READ_ONLY_GRANTS_NOTHING",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "roots_logical": sorted(ROOTS),
        "rules": [
            "no target, label, forecast, score, model result or metric was read or used",
            "no download; bytes on disk only",
            "license never inferred from provider reputation; unit never inferred from a column name",
            "counts_toward_six requires family independence, an unambiguous declared license and all eight join artifacts present",
        ],
        "conditions": CONDITIONS,
        "panel_count": len(panels),
        "counts_toward_six_total": sum(p["counts_toward_six"] for p in panels),
        "panels": panels,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("PANEL_INVENTORY_DRAFT.json")))
    a = ap.parse_args()
    doc = build()
    Path(a.out).write_text(json.dumps(doc, indent=1, default=str, sort_keys=False))
    print(json.dumps({"panels": doc["panel_count"], "counts_toward_six": doc["counts_toward_six_total"]}))


if __name__ == "__main__":
    main()
