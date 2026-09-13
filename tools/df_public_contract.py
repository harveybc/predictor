#!/usr/bin/env python3
"""C126 x C129: the public bank as canonical panels with common contracts.

For each dataset in the custody manifest of C126:

1. every raw archive and every license evidence page is re-hashed and must
   equal the manifest before anything is parsed;
2. the archive is parsed with rules declared here (separator, decimal mark,
   missing markers, timestamp format), without interpolation, filling,
   deduplication or re-ordering. A non-monotonic series refuses;
3. a canonical panel (parquet: a timestamp label column plus one column per
   variable) and a parse receipt are written write-once into a new root;
4. a common contract is sealed.

What the contract declares comes only from evidence:

* a unit, a missing-value statement or a timezone is declared only when its
  statement is FOUND at run time in the official page kept in custody (or,
  for the Jena station, in the source's own CSV header). The matched text
  and the evidence digest are stored. If the statement is not found, the
  field is UNKNOWN. Nothing is inferred from a column name;
* columns the source says come from a third party (rp5.ru weather in UCI 374,
  China Meteorological Administration weather in UCI 501) carry
  TERMS_REQUIRE_REVIEW, since the distributor's license may not cover them;
* the timestamp meaning (period start or end) is UNKNOWN unless stated.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RAW_ROOT = Path.home() / ".local/share/crispdm-data-foundation/public_raw_c126_20260912"
PANEL_ROOT = Path.home() / ".local/state/crispdm-data-foundation/public_panels_c126_v1"
RAW_LOGICAL = "crispdm-data-foundation/public_raw_c126_20260912"
PANEL_LOGICAL = "crispdm-data-foundation/public_panels_c126_v1"
UNKNOWN = "UNKNOWN"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


C = _load("df_contract")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def page_text(raw: bytes) -> str:
    t = raw.decode("utf-8", "replace")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S | re.I)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t)


# ------------------------------------------------------------ declarations
# Units and statements are REGEX over the official page text; the match is
# the evidence. `third_party` names columns sourced from a third party.
SOURCES = {
    "uci_321_electricityloaddiagrams20112014": {
        "dataset_id": "public.uci.321.electricityloaddiagrams20112014",
        "provider": "UCI Machine Learning Repository", "parser": "uci_321", "frequency_seconds": 900,
        "semantics": ("ELECTRICITY_CONSUMPTION_PER_CLIENT",
                      r"This data set contains electricity consumption of 370 points/clients\."),
        "unit_all": ("kW", r"Values are in kW of each 15 min\."),
        "timezone": ("Europe/Lisbon local wall clock", r"All time labels report to Portuguese hour\."),
        "missing": r"Data set has no missing values\.",
        "sentinel": (r"Some clients were created after 2011\. In these cases consumption were considered zero\.",
                     "SOURCE_DECLARED_ZERO_BEFORE_CLIENT_CREATION; March DST hour zeros declared by the source"),
        "units": {}, "third_party": (),
    },
    "uci_235_individual_household_power": {
        "dataset_id": "public.uci.235.individual_household_electric_power_consumption",
        "provider": "UCI Machine Learning Repository", "parser": "uci_235", "frequency_seconds": 60,
        "semantics": ("HOUSEHOLD_ELECTRICITY_MEASUREMENT",
                      r"measurements gathered in a house located in Sceaux"),
        "unit_all": None, "timezone": None,
        "missing": r"The dataset contains some missing values in the measurements \(nearly 1,25% of the rows\)\.",
        "sentinel": None,
        "units": {"Global_active_power": ("kW", r"global_active_power: household global minute-averaged active power \(in kilowatt\)"),
                  "Global_reactive_power": ("kW", r"global_reactive_power: household global minute-averaged reactive power \(in kilowatt\)"),
                  "Voltage": ("V", r"voltage: minute-averaged voltage \(in volt\)"),
                  "Global_intensity": ("A", r"global_intensity: household global minute-averaged current intensity \(in ampere\)"),
                  "Sub_metering_1": ("Wh", r"sub_metering_1: energy sub-metering No\. 1 \(in watt-hour of active energy\)"),
                  "Sub_metering_2": ("Wh", r"sub_metering_2: energy sub-metering No\. 2 \(in watt-hour of active energy\)"),
                  "Sub_metering_3": ("Wh", r"sub_metering_3: energy sub-metering No\. 3 \(in watt-hour of active energy\)")},
        "third_party": (),
    },
    "uci_374_appliances_energy": {
        "dataset_id": "public.uci.374.appliances_energy_prediction",
        "provider": "UCI Machine Learning Repository", "parser": "uci_374", "frequency_seconds": 600,
        "semantics": ("LOW_ENERGY_HOUSE_MONITORING", r"The data set is at 10 min for about 4\.5 months\."),
        "unit_all": None, "timezone": None, "missing": None, "sentinel": None,
        "units": {"Appliances": ("Wh", r"Appliances, energy use in Wh"),
                  "lights": ("Wh", r"lights, energy use of light fixtures in the house in Wh"),
                  **{f"T{i}": ("degC", rf"T{i}, Temperature in [^,]+, in Celsius") for i in range(1, 10)},
                  **{f"RH_{i}": ("%", rf"RH_{i}, Humidity in [^,]+, in %") for i in range(1, 10)},
                  "T_out": ("degC", r"To, Temperature outside \(from Chievres weather station\), in Celsius"),
                  "Press_mm_hg": ("mmHg", r"Pressure \(from Chievres weather station\), in mm Hg"),
                  "RH_out": ("%", r"RH_out, Humidity outside \(from Chievres weather station\), in %"),
                  "Windspeed": ("m/s", r"Wind speed \(from Chievres weather station\), in m/s"),
                  "Visibility": ("km", r"Visibility \(from Chievres weather station\), in km"),
                  "Tdewpoint": ("degC", r"Tdewpoint \(from Chievres weather station\), \S{0,3}C"),
                  "rv1": ("1", r"rv1, Random variable 1, nondimensional"),
                  "rv2": ("1", r"rv2, Random variable 2, nondimensional")},
        "excluded": {"rv1": "SOURCE_DECLARED_RANDOM_CONTROL", "rv2": "SOURCE_DECLARED_RANDOM_CONTROL"},
        "third_party": ("T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"),
        "third_party_statement": r"downloaded from a public data set from Reliable Prognosis",
    },
    "uci_501_beijing_multisite_air_quality": {
        "dataset_id": "public.uci.501.beijing_multisite_air_quality",
        "provider": "UCI Machine Learning Repository", "parser": "uci_501", "frequency_seconds": 3600,
        "semantics": ("URBAN_AIR_QUALITY_AND_METEOROLOGY",
                      r"This data set includes hourly air pollutants data from 12 nationally-controlled air-quality monitoring sites\."),
        "unit_all": None, "timezone": None, "missing": r"Missing data are denoted as NA\.", "sentinel": None,
        "units": {"PM2.5": ("ug/m3", r"PM2\.5: PM2\.5 concentration \(ug/m\^3\)"),
                  "PM10": ("ug/m3", r"PM10: PM10 concentration \(ug/m\^3\)"),
                  "SO2": ("ug/m3", r"SO2: SO2 concentration \(ug/m\^3\)"),
                  "NO2": ("ug/m3", r"NO2: NO2 concentration \(ug/m\^3\)"),
                  "CO": ("ug/m3", r"CO: CO concentration \(ug/m\^3\)"),
                  "O3": ("ug/m3", r"O3: O3 concentration \(ug/m\^3\)"),
                  "TEMP": ("degC", r"TEMP: temperature \(degree Celsius\)"),
                  "PRES": ("hPa", r"PRES: pressure \(hPa\)"),
                  "DEWP": ("degC", r"DEWP: dew point temperature \(degree Celsius\)"),
                  "RAIN": ("mm", r"RAIN: precipitation \(mm\)"),
                  "WSPM": ("m/s", r"WSPM: wind speed \(m/s\)")},
        "third_party": ("TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"),
        "third_party_statement": r"matched with the nearest weather station from the China Meteorological Administration",
    },
    "mpi_bgc_jena_weather_mpi_roof_2020_2024": {
        "dataset_id": "public.mpi_bgc.jena_weather_station_beutenberg.2020_2024",
        "provider": "Max Planck Institute for Biogeochemistry", "parser": "jena", "frequency_seconds": 600,
        "semantics": ("WEATHER_STATION_MEASUREMENT", r"Terms of Use \(as per Creative Commons CC-BY-4\.0\)"),
        "unit_all": None, "timezone": None, "missing": None, "sentinel": None,
        "units": "FROM_SOURCE_HEADER", "third_party": (),
    },
}


def find(pattern: str, text: str):
    m = re.search(pattern, text)
    return m.group(0) if m else None


# ----------------------------------------------------------------- parsers
def _member(zf: zipfile.ZipFile, name: str) -> bytes:
    return zf.read(name)


def parse_uci_321(archives):
    (_, zf), = archives
    raw = _member(zf, "LD2011_2014.txt")
    df = pd.read_csv(io.BytesIO(raw), sep=";", decimal=",", quotechar='"', header=0, index_col=0, dtype=str)
    labels = list(df.index)
    values = df.apply(lambda s: pd.to_numeric(s, errors="raise")).astype("float64")
    return labels, values, {"LD2011_2014.txt": sha_bytes(raw)}, {c: c for c in values.columns}


def parse_uci_235(archives):
    (_, zf), = archives
    raw = _member(zf, "household_power_consumption.txt")
    df = pd.read_csv(io.BytesIO(raw), sep=";", header=0, dtype=str, na_values=["?", ""], keep_default_na=False)
    labels = (df["Date"] + " " + df["Time"]).tolist()
    values = df.drop(columns=["Date", "Time"]).apply(lambda s: pd.to_numeric(s, errors="raise")).astype("float64")
    return labels, values, {"household_power_consumption.txt": sha_bytes(raw)}, {c: c for c in values.columns}


def parse_uci_374(archives):
    (_, zf), = archives
    raw = _member(zf, "energydata_complete.csv")
    df = pd.read_csv(io.BytesIO(raw), header=0, dtype=str, keep_default_na=False)
    labels = df["date"].tolist()
    values = df.drop(columns=["date"]).apply(lambda s: pd.to_numeric(s.str.strip(), errors="raise")).astype("float64")
    return labels, values, {"energydata_complete.csv": sha_bytes(raw)}, {c: c for c in values.columns}


def parse_uci_501(archives):
    (_, zf), = archives
    inner_name = next(n for n in zf.namelist() if n.endswith(".zip"))
    inner = zipfile.ZipFile(io.BytesIO(zf.read(inner_name)))
    members = sorted(n for n in inner.namelist() if n.endswith(".csv"))
    frames, digests, labels = {}, {}, None
    for n in members:
        raw = inner.read(n)
        digests[n] = sha_bytes(raw)
        df = pd.read_csv(io.BytesIO(raw), header=0, dtype=str, na_values=["NA"], keep_default_na=False)
        station = df["station"].iloc[0]
        if (df["station"] != station).any():
            raise C.ContractRefusal([f"{n}: more than one station in one file"])
        lab = (df["year"].str.zfill(4) + "-" + df["month"].str.zfill(2) + "-" + df["day"].str.zfill(2)
               + " " + df["hour"].str.zfill(2) + ":00:00").tolist()
        if labels is None:
            labels = lab
        elif lab != labels:
            raise C.ContractRefusal([f"{n}: station timestamps differ; not one aligned grid"])
        for col in df.columns:
            if col in ("No", "year", "month", "day", "hour", "station"):
                continue
            s = df[col]
            frames[f"{station}__{col}"] = s if col == "wd" else pd.to_numeric(s, errors="raise").astype("float64")
    values = pd.DataFrame(frames)
    source_col = {c: c.split("__", 1)[1] for c in values.columns}
    return labels, values, digests, source_col


def parse_jena(archives):
    frames, digests = [], {}
    for _, zf in archives:
        for n in sorted(zf.namelist()):
            raw = zf.read(n)
            digests[n] = sha_bytes(raw)
            frames.append((n, pd.read_csv(io.BytesIO(raw), header=0, dtype=str, encoding="latin-1",
                                          skipinitialspace=True, keep_default_na=False)))
    header = list(frames[0][1].columns)
    for n, df in frames:
        if list(df.columns) != header:
            raise C.ContractRefusal([f"{n}: header differs from the first file"])
    df = pd.concat([f for _, f in frames], ignore_index=True)
    labels = df["Date Time"].tolist()
    values = df.drop(columns=["Date Time"]).apply(lambda s: pd.to_numeric(s, errors="raise")).astype("float64")
    return labels, values, digests, {c: c for c in values.columns}


PARSERS = {"uci_321": parse_uci_321, "uci_235": parse_uci_235, "uci_374": parse_uci_374,
           "uci_501": parse_uci_501, "jena": parse_jena}
TIMESTAMP_FORMATS = {"uci_321": "%Y-%m-%d %H:%M:%S", "uci_235": "%d/%m/%Y %H:%M:%S", "uci_374": "%Y-%m-%d %H:%M:%S",
                     "uci_501": "%Y-%m-%d %H:%M:%S", "jena": "%d.%m.%Y %H:%M:%S"}


# ------------------------------------------------------------------- build
def _verified_inputs(entry: dict, raw_root: Path):
    archives, files = [], []
    for f in entry["files"]:
        p = raw_root / "data" / f["name"]
        b = p.read_bytes()
        if sha_bytes(b) != f["sha256"] or len(b) != f["bytes"]:
            raise C.ContractRefusal([f"{f['name']}: bytes differ from the custody manifest"])
        archives.append((f["name"], zipfile.ZipFile(io.BytesIO(b))))
        files.append({"name": f"{RAW_LOGICAL}/data/{f['name']}", "bytes": len(b), "sha256": f["sha256"],
                      "role": "RAW_ARCHIVE"})
    ev = entry["license"]["evidence"]
    ev_bytes = (raw_root / ev["file"]).read_bytes()
    if sha_bytes(ev_bytes) != ev["sha256"]:
        raise C.ContractRefusal([f"{ev['file']}: license evidence differs from the custody manifest"])
    files.append({"name": f"{RAW_LOGICAL}/{ev['file']}", "bytes": len(ev_bytes), "sha256": ev["sha256"],
                  "role": "LICENSE_EVIDENCE"})
    return archives, files, page_text(ev_bytes), ev


def build(logical_id: str, raw_root: Path = RAW_ROOT, panel_root: Path = PANEL_ROOT) -> dict:
    spec = SOURCES[logical_id]
    manifest = json.loads((raw_root / "PUBLIC_RAW_MANIFEST.json").read_text())
    entry = next(d for d in manifest["datasets"] if d["logical_id"] == logical_id)
    out = Path(panel_root) / logical_id
    if out.exists():
        raise C.ContractRefusal([f"{out.name}: canonical panel exists; write-once"])
    archives, files, text, ev = _verified_inputs(entry, raw_root)
    if find(r"This dataset is licensed under a Creative Commons Attribution 4\.0 International \(CC BY 4\.0\) license\.|"
            r"Terms of Use \(as per Creative Commons CC-BY-4\.0\)", text) is None:
        raise C.ContractRefusal([f"{logical_id}: the license statement is not in the kept evidence page"])

    labels, values, member_digests, source_col = PARSERS[spec["parser"]](archives)
    ts = pd.to_datetime(pd.Series(labels), format=TIMESTAMP_FORMATS[spec["parser"]], errors="raise")
    diffs = ts.diff().dt.total_seconds().iloc[1:]
    backwards = diffs[diffs < 0]
    if len(backwards):
        # Refused, never sorted or deduplicated. Every backward jump is listed
        # so the source anomaly can be reviewed and a rule declared for it.
        detail = [{"row": int(ix), "jump_seconds": float(v),
                   "labels_around": [str(x) for x in labels[max(0, int(ix) - 3):int(ix) + 3]]}
                  for ix, v in backwards.items()][:20]
        raise C.ContractRefusal([f"{logical_id}: timestamps go backwards at {len(backwards)} rows; "
                                 "the source order is not chronological",
                                 "anomalies: " + json.dumps(detail)])
    duplicates = int((diffs == 0).sum())

    ev_ref = {"source": f"{RAW_LOGICAL}/{ev['file']}", "sha256": ev["sha256"]}
    receipt_units, variables = {}, []
    header_sha = next(iter(member_digests.values()))
    for col in values.columns:
        src = source_col[col]
        unit_val, unit_ev = UNKNOWN, []
        if spec["units"] == "FROM_SOURCE_HEADER":
            m = re.fullmatch(r"(.+?) \((.+)\)", src.strip())
            if m:
                unit_val = m.group(2)
                unit_ev = [{"source": f"CSV header column label {src!r}", "sha256": header_sha}]
        elif spec.get("unit_all"):
            quote = find(spec["unit_all"][1], text)
            if quote:
                unit_val, unit_ev = spec["unit_all"][0], [dict(ev_ref, source=f"{ev_ref['source']}: {quote}")]
        elif src in spec["units"]:
            u, pat = spec["units"][src]
            quote = find(pat, text)
            if quote:
                unit_val, unit_ev = u, [dict(ev_ref, source=f"{ev_ref['source']}: {quote}")]
        receipt_units[col] = unit_val
        sem_quote = find(spec["semantics"][1], text)
        semantics = ({"type": spec["semantics"][0], "description": f"source column {src!r}",
                      "evidence": [dict(ev_ref, source=f"{ev_ref['source']}: {sem_quote}")]} if sem_quote
                     else {"type": UNKNOWN, "description": f"source column {src!r}", "evidence": []})
        tp = src in spec.get("third_party", ())
        tp_quote = find(spec.get("third_party_statement", r"(?!)"), text) if tp else None
        excluded = spec.get("excluded", {}).get(src)
        is_text = values[col].dtype == object
        missing_quote = find(spec["missing"], text) if spec.get("missing") else None
        sentinel = spec.get("sentinel")
        sentinel_quote = find(sentinel[0], text) if sentinel else None
        variables.append(C.variable(
            spec["dataset_id"], col,
            semantics=semantics, unit={"value": unit_val, "evidence": unit_ev},
            producer={"kind": "SOURCE_MEASUREMENT", "reference": f"{spec['provider']} ({src})"},
            physical_type="string" if is_text else "float64",
            frequency_nominal_seconds=spec["frequency_seconds"], event_time="SOURCE_TIMESTAMP_LABEL",
            available_time_rule=UNKNOWN,
            missingness={"encoding": "typed null in the canonical panel (source markers: " + {
                "uci_235": "'?' or empty", "uci_501": "NA"}.get(spec["parser"], "none observed") + ")",
                "policy": f"NO_IMPUTATION; source statement: {missing_quote}" if missing_quote else "NO_IMPUTATION"},
            sentinels={"values": [0.0] if sentinel_quote else [],
                       "policy": f"{sentinel[1]}: {sentinel_quote}" if sentinel_quote else UNKNOWN},
            role="EXCLUDED" if excluded else "INPUT_CANDIDATE",
            license_state="TERMS_REQUIRE_REVIEW" if tp else "OPEN_ATTRIBUTION",
            original_fields={"source_column": src, "excluded_reason": excluded,
                             "third_party_statement": tp_quote}))

    out.mkdir(parents=True)
    # A missing observation is a typed null in the panel, never a NaN that
    # could be mistaken for a computed value.
    columns = {}
    for c in values.columns:
        if values[c].dtype == object:
            columns[c] = pa.array([None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
                                   for v in values[c].tolist()], pa.string())
        else:
            arr = values[c].to_numpy(dtype="float64")
            columns[c] = pa.array(arr, type=pa.float64(), mask=np.isnan(arr))
    table = pa.table({"timestamp_label": pa.array([str(x) for x in labels], pa.string()), **columns})
    panel_path = out / "panel.parquet"
    pq.write_table(table, panel_path, compression="zstd")
    tz = spec.get("timezone")
    tz_quote = find(tz[1], text) if tz else None
    receipt = {"schema": "crispdm.data_foundation.public_parse_receipt.v1", "logical_id": logical_id,
               "parser": spec["parser"], "timestamp_format": TIMESTAMP_FORMATS[spec["parser"]],
               "rows": len(labels), "columns": len(values.columns), "member_sha256": member_digests,
               "duplicate_timestamp_labels": duplicates, "rows_reordered": 0, "rows_filled": 0, "rows_dropped": 0,
               "units_declared": {k: v for k, v in receipt_units.items() if v != UNKNOWN},
               "units_unknown": sorted(k for k, v in receipt_units.items() if v == UNKNOWN),
               "missing_values": int(values.select_dtypes("float64").isna().sum().sum())}
    receipt_path = out / "PARSE_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    for p, role in ((panel_path, "DERIVED_CANONICAL_PANEL"), (receipt_path, "PARSE_RECEIPT")):
        b = p.read_bytes()
        files.append({"name": f"{PANEL_LOGICAL}/{logical_id}/{p.name}", "bytes": len(b), "sha256": sha_bytes(b),
                      "role": role})
    contract = {
        "schema": C.DATASET_SCHEMA, "dataset_id": spec["dataset_id"], "version": entry["files"][0]["retrieved_at_utc"],
        "bank": "PUBLIC", "files": files, "content_sha256": "", "contract_sha256": "",
        "source": {"provider": spec["provider"], "official_url": entry["license"]["evidence"]["fetched_from"] or UNKNOWN,
                   "citation": entry.get("citation") or UNKNOWN, "doi": entry.get("doi") or UNKNOWN,
                   "upstream_owner": entry.get("upstream_owner") or UNKNOWN},
        "license": {"state": "OPEN_ATTRIBUTION", "id": "CC-BY-4.0", "url": entry["license"]["url"] or UNKNOWN,
                    "text_sha256": "UNAVAILABLE", "attribution_required": "YES",
                    "redistribution": "ALLOWED_WITH_ATTRIBUTION", "derivatives": "ALLOWED_WITH_ATTRIBUTION",
                    "evidence": [dict(ev_ref, source=f"{ev_ref['source']}: {entry['license']['evidence']['statement']}")]},
        "time": {"frequency_nominal_seconds": spec["frequency_seconds"],
                 "timezone": f"{tz[0]} (source: {tz_quote})" if tz_quote else UNKNOWN,
                 "timestamp_meaning": UNKNOWN, "range_start": str(labels[0]), "range_end": str(labels[-1]),
                 "availability_rule": UNKNOWN, "availability_delay_seconds": UNKNOWN},
        "panel": {"aligned_common_grid": True, "n_series": len(values.columns),
                  "alignment_rule": f"one table in source order; {duplicates} duplicate timestamp labels kept as found"},
        "partitions": {"scheme": "CHRONOLOGICAL_FRACTIONS",
                       "fractions": {"train": 0.6, "calibration": 0.2, "confirmation": 0.2},
                       "boundaries": C.chronological_partitions(len(labels)), "sealed_periods_excluded": [],
                       "frozen_before_profile": True},
        "dependence": [], "variables": variables,
        "original_fields": {"custody_manifest_entry": entry, "parse_receipt": receipt},
    }
    sealed = C.seal(contract)
    (out / "CONTRACT.json").write_text(json.dumps(sealed, indent=1, sort_keys=True) + "\n")
    return sealed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", action="append", choices=sorted(SOURCES), default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    ap.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    a = ap.parse_args(argv)
    for lid in (sorted(SOURCES) if a.all else a.dataset):
        c = build(lid, a.raw_root, a.panel_root)
        rec = c["original_fields"]["parse_receipt"]
        print(json.dumps({"dataset_id": c["dataset_id"], "contract_sha256": c["contract_sha256"], "rows": rec["rows"],
                          "variables": len(c["variables"]), "units_unknown": len(rec["units_unknown"]),
                          "duplicate_labels": rec["duplicate_timestamp_labels"], "missing": rec["missing_values"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
