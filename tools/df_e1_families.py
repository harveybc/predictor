#!/usr/bin/env python3
"""RP15: E1_FAMILIES.json from the EXISTING census in the warehouse (no re-census, no reserve opened).

Selects, does not recharacterise: every dataset of the census (df_dim_dataset, 715 rows) is
classified by bank and licence state; public candidates are described from their contract
(variables, sampling, range, provider, licence) and from the coverage ledger (df_fact_coverage_v2:
how many profile rows the D1/D2 fronts produced for them). Criteria are DECLARED per family, not
a universal "20 cycles": a periodic family reports the cycles of its dominant declared period in
the range; an aperiodic (or unknown-period) family reports the number of non-overlapping
context + horizon blocks it supports for the E1 geometry and requires an adequacy diagnostic
before use. Source deficits are recorded, not hidden.

    python tools/df_e1_families.py --out E1_FAMILIES.json [--warehouse-url URL --token-env VAR]
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

SCHEMA = "e1_families.v1"
E1_GEOMETRY = {"context_window": 96, "horizon": 1, "min_blocks_aperiodic": 200, "min_variables": 6, "min_targets": 2,
               "why": "context 96 covers one daily cycle at 15-min sampling and four at hourly; the block count is the support in "
                      "independent (non-overlapping) context + horizon spans, the unit that matters for a process without an identifiable period"}
DAY = 86400.0


def _query(url: str, token: str, sql: str) -> list:
    request = urllib.request.Request(f"{url}/api/v1/query?" + urllib.parse.urlencode({"sql": sql}), headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=120) as answer:
        return json.loads(answer.read())["rows"]


def _parse(ts: str):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except (ValueError, TypeError):
            continue
    return None


def describe(row: dict, coverage: dict) -> dict:
    c = json.loads(row["contract_json"])
    t, lic, src = c.get("time") or {}, c.get("license") or {}, c.get("source") or {}
    freq = t.get("frequency_nominal_seconds")
    start, end = _parse(str(t.get("range_start"))), _parse(str(t.get("range_end")))
    variables = c.get("variables") or []
    targets = [v["name"] for v in variables if v.get("role") in ("INPUT_CANDIDATE", "TARGET_CANDIDATE", "TARGET") and v.get("name") not in ("timestamp", "date", "time")]
    entry = {"dataset_id": row["dataset_id"], "bank": row["bank"], "license_state": row["license_state"], "license_id": lic.get("id"),
             "provider": src.get("provider"), "n_variables": row["n_variables"], "semantics": sorted({(v.get("semantics") or {}).get("type") for v in variables if (v.get("semantics") or {}).get("type")}),
             "sampling_seconds": freq, "range": [t.get("range_start"), t.get("range_end")], "panel_series": (c.get("panel") or {}).get("n_series"),
             "targets_candidate": len(targets), "coverage_rows": coverage.get(row["dataset_id"], {}).get("rows", 0),
             "coverage_states": coverage.get(row["dataset_id"], {}).get("states", {}), "exclusions": [], "adequacy": {}}
    if isinstance(freq, (int, float)) and start and end and freq > 0:
        rows_est = int((end - start).total_seconds() / freq) + 1
        entry["rows_estimated"] = rows_est
        daily = DAY / freq
        entry["adequacy"] = {"declared_period_samples": daily if daily >= 2 else None, "declared_period": "daily (24 h) — declared from the sampling, "
                             "to be CONFIRMED by the profile's spectrum before use",
                             "cycles_in_range": rows_est / daily if daily >= 2 else None,
                             "blocks_context_plus_horizon": rows_est // (E1_GEOMETRY["context_window"] + E1_GEOMETRY["horizon"]),
                             "aperiodic_rule": f"if no period is confirmed, adequacy = blocks >= {E1_GEOMETRY['min_blocks_aperiodic']} plus a "
                                               f"stationarity / dependence diagnostic from the D1 profiles (ADF/KPSS/correlation time rows in coverage)"}
    else:
        entry["rows_estimated"] = None
        entry["exclusions"].append("sampling or range unknown in the contract")
    if row["bank"] != "PUBLIC":
        entry["exclusions"].append(f"bank {row['bank']}: {'licence pending evidence' if 'PENDING' in (row['license_state'] or '') else 'generated, not a public family'}")
    if row["license_state"] not in ("OPEN_ATTRIBUTION", "OPEN"):
        entry["exclusions"].append(f"licence state {row['license_state']} does not allow research use without evidence")
    if (row["n_variables"] or 0) < E1_GEOMETRY["min_variables"]:
        entry["exclusions"].append(f"fewer than {E1_GEOMETRY['min_variables']} variables")
    if len(targets) < E1_GEOMETRY["min_targets"]:
        entry["exclusions"].append(f"fewer than {E1_GEOMETRY['min_targets']} candidate targets")
    if entry["adequacy"].get("blocks_context_plus_horizon") is not None and entry["adequacy"]["blocks_context_plus_horizon"] < E1_GEOMETRY["min_blocks_aperiodic"]:
        entry["exclusions"].append(f"fewer than {E1_GEOMETRY['min_blocks_aperiodic']} context+horizon blocks")
    entry["eligible"] = not entry["exclusions"]
    return entry


def build(rows: list, coverage: dict) -> dict:
    described = [describe(r, coverage) for r in rows]
    banks = {}
    for d in described:
        banks.setdefault(d["bank"], {}).setdefault(d["license_state"], 0)
        banks[d["bank"]][d["license_state"]] += 1
    eligible = sorted([d for d in described if d["eligible"]], key=lambda d: -(d["rows_estimated"] or 0) * (d["n_variables"] or 0))
    # DEV and RESERVE from distinct datasets/domains: the two largest supports to DEV, the rest RESERVE candidates (never opened here)
    for i, d in enumerate(eligible):
        d["split_proposal"] = "DEV" if i < 2 else "RESERVE_CANDIDATE_NOT_OPENED"
        d["split_rule"] = "distinct datasets (never the same dataset split by rows); DEV = the two largest supports; the reserve is not generated, read or profiled here"
    excluded_public = [d for d in described if d["bank"] == "PUBLIC" and not d["eligible"]]
    return {"schema": SCHEMA, "geometry": E1_GEOMETRY, "census_datasets": len(described), "by_bank_and_license": banks,
            "families": eligible, "public_excluded": excluded_public,
            "non_public_summary": {b: sum(v.values()) for b, v in banks.items() if b != "PUBLIC"},
            "source_deficit": {"public_families_eligible": len(eligible),
                               "statement": "the census holds four public multivariate families (all UCI, CC-BY-4.0); the financial bank (198) is "
                                            "INTERNAL_RESEARCH_ONLY_PENDING_EVIDENCE and the synthetic bank (513) is generated: E1 cannot draw a "
                                            "development AND a reserve family from independent public SOURCES beyond these four; resolving it is "
                                            "engineering (licence evidence for financial sources or ingestion of further public multivariate "
                                            "datasets through the governed census), not a selection choice",
                               "no_universal_cycle_minimum": "cycles_in_range is reported where a daily period is declared; adequacy for an "
                                                             "aperiodic process is the block count plus the D1 stationarity/dependence diagnostic"},
            "references_fixed_before_development": ["naive", "seasonal naive", "a pertinent multivariate statistical model (VAR)", "DLinear", "PatchTST",
                                                    "iTransformer", "DUET", "one of MTST/Pathformer/TimeMixer chosen before development results"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    parser.add_argument("--token-env", default="DATA_GOV_LAKE_TOKEN")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists")
    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(f"REFUSED: no warehouse token in ${args.token_env}")
    rows = _query(args.warehouse_url, token, 'SELECT dataset_id, bank, license_state, n_variables, contract_json FROM "main"."df_dim_dataset" LIMIT 5000')
    cov = _query(args.warehouse_url, token, 'SELECT dataset_id, state, count(*) n FROM "main"."df_fact_coverage_v2" WHERE dataset_id LIKE \'public.%\' GROUP BY 1, 2 LIMIT 1000')
    coverage = {}
    for r in cov:
        d = coverage.setdefault(r["dataset_id"], {"rows": 0, "states": {}})
        d["rows"] += int(r["n"])
        d["states"][r["state"]] = int(r["n"])
    doc = build(rows, coverage)
    doc["source"] = {"warehouse": args.warehouse_url, "tables": ["df_dim_dataset", "df_fact_coverage_v2"]}
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"census_datasets": doc["census_datasets"], "by_bank": doc["by_bank_and_license"], "eligible": [(f["dataset_id"], f["split_proposal"]) for f in doc["families"]],
                      "excluded_public": [(f["dataset_id"], f["exclusions"]) for f in doc["public_excluded"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
