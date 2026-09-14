#!/usr/bin/env python3
"""Flow v3 adoption, P0.2: validate installed resource_contracts against physical bytes.

Loads the data-gov config that will be deployed, instantiates the in-process
`files_lake` plugin exactly as the service does (same params), and for every
resource with a contract:

  * AS_IS governed download: delivered sha256 == sha256 of the file on disk,
    contract digest == sha256(canonical contract), delivered bytes == file size;
  * a day-range governed download (CUT): every kept row has
    from <= available_time < to + 1 day, the cut is a byte subset of the
    source (header + kept lines), and the row count equals an independent
    pandas count;
  * the contract's columns exist and parse under the declared timezone mode.

Also confirms every resource *without* a contract is refused
("resource availability contract required"). Read-only against the data;
cuts are materialised under a throwaway cuts_dir given on the command line.

usage: p02_validate_contracts.py <data-gov checkout> <config.json> <throwaway cuts dir>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def main(argv) -> int:
    data_gov, config_path, cuts_dir = Path(argv[1]).resolve(), Path(argv[2]).resolve(), Path(argv[3]).resolve()
    sys.path.insert(0, str(data_gov))
    from lake_plugins.errors import UnsupportedError
    from lake_plugins.files_lake import Plugin

    config = json.loads(config_path.read_text(encoding="utf-8"))
    lake_cfg = next(l for l in config["lakes"] if l["plugin"] == "files_lake")
    params = {k: v for k, v in lake_cfg.items() if k != "plugin"}
    root = (data_gov / params["root_path"]).resolve()
    params["root_path"] = str(root)
    params["cuts_dir"] = str(cuts_dir / "cuts")
    params["spool_dir"] = str(cuts_dir / "spool")
    plugin = Plugin()
    plugin.set_params(**params)
    contracts = params.get("resource_contracts") or {}
    report = {"schema": "flow_v3_p02_contract_validation.v1", "lake_id": params["lake_id"],
              "root_path_relative": lake_cfg["root_path"], "resources": {}, "refused_without_contract": {}}
    ok = True
    for rid, contract in contracts.items():
        path = root / rid
        entry = {"contract": contract,
                 "contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()}
        disk_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        info = plugin.governed_download(rid)
        info["handle"].close()
        entry["as_is"] = {
            "delivery": info["delivery"], "sha256": info["sha256"], "bytes": info["bytes"],
            "source_sha256": info["source_sha256"], "time_column": info["time_column"],
            "availability_contract_sha256": info["availability_contract_sha256"],
            "matches_disk": info["sha256"] == disk_sha == info["source_sha256"] and info["bytes"] == path.stat().st_size,
            "contract_digest_matches": info["availability_contract_sha256"] == entry["contract_sha256"],
        }
        df = pd.read_csv(path)
        ts = pd.to_datetime(df[contract["available_time_column"]], format="ISO8601")
        ev = pd.to_datetime(df[contract["event_time_column"]], format="ISO8601")
        entry["columns_present"] = {contract["event_time_column"]: contract["event_time_column"] in df.columns,
                                    contract["available_time_column"]: contract["available_time_column"] in df.columns}
        entry["available_time_ge_event_time_all_rows"] = bool((ts >= ev).all())
        # a one-year cut inside the resource, at day granularity
        year = int(ts.min().year) + 1
        start, end = f"{year}-01-01", f"{year}-12-31"
        lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
        cut = plugin.governed_download(rid, start, end)
        data = cut["handle"].read()
        cut["handle"].close()
        cut_df = pd.read_csv(Path(cut["path"]))
        cts = pd.to_datetime(cut_df[contract["available_time_column"]], format="ISO8601")
        expected_rows = int(((ts >= lo) & (ts < hi)).sum())
        src_lines = set(path.read_bytes().splitlines())
        cut_lines = data.splitlines()
        entry["cut"] = {
            "range": [start, end], "delivery": cut["delivery"], "sha256": cut["sha256"], "bytes": cut["bytes"],
            "rows": int(len(cut_df)), "expected_rows_independent_pandas": expected_rows,
            "all_rows_inside_range": bool(((cts >= lo) & (cts < hi)).all()),
            "max_available_time_in_cut": str(cts.max()),
            "last_information_complete_by_(t+1h)": str(cts.max() + pd.Timedelta(hours=1)),
            "range_end_exclusive": str(hi),
            "complete_before_range_end_under_1h_bound": bool(cts.max() + pd.Timedelta(hours=1) <= hi),
            "byte_subset_of_source": all(line in src_lines for line in cut_lines),
            "delivered_sha256_matches_bytes": hashlib.sha256(data).hexdigest() == cut["sha256"],
            "availability_contract_sha256": cut["availability_contract_sha256"],
        }
        entry["ok"] = all([
            entry["as_is"]["matches_disk"], entry["as_is"]["contract_digest_matches"],
            all(entry["columns_present"].values()), entry["available_time_ge_event_time_all_rows"],
            entry["cut"]["rows"] == expected_rows, entry["cut"]["all_rows_inside_range"],
            entry["cut"]["byte_subset_of_source"], entry["cut"]["delivered_sha256_matches_bytes"],
            entry["cut"]["complete_before_range_end_under_1h_bound"],
            entry["cut"]["availability_contract_sha256"] == entry["contract_sha256"],
        ])
        ok &= entry["ok"]
        report["resources"][rid] = entry
    for item in plugin.discover():
        rid = item["resource_id"]
        if rid in contracts:
            continue
        try:
            plugin.governed_download(rid)
            report["refused_without_contract"][rid] = "SERVED (defect)"
            ok = False
        except UnsupportedError as exc:
            report["refused_without_contract"][rid] = str(exc)
    report["resources_without_contract"] = len(report["refused_without_contract"])
    report["all_refusals_are_contract_required"] = all(
        v == "resource availability contract required" for v in report["refused_without_contract"].values())
    ok &= report["all_refusals_are_contract_required"]
    report["ok"] = ok
    json.dump(report, sys.stdout, indent=1, default=str)
    sys.stdout.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
