"""Build a small, navigable I6-A result catalog from verified closures.

The warehouse and retained cell receipts remain authoritative. This catalog
contains only a digest-bound summary, never the large cell population.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import urllib.parse
from pathlib import Path

from tools.fs4_candidates import digest
from tools.i6a_architectures import ARMS
from tools.i6a_publish_olap import experiment_set_key, request_json, token_from_file
from tools.i6a_weekly_arch_pilot import target_horizon_hours


def target_id(closure):
    if closure.get("schema") == "i6a.architecture_closure.v1":
        return "Y_s_1h"
    if closure.get("schema") == "i6a.architecture_closure.v2":
        return closure.get("target_id")
    raise ValueError("UNKNOWN_CLOSURE_SCHEMA")


def validate(closure, publication):
    target = target_id(closure)
    if (closure.get("sha256") != digest({k: v for k, v in closure.items() if k != "sha256"})
            or closure.get("state") != "COMPLETE" or closure.get("problems")
            or closure.get("expected_cells") != 208 or closure.get("verified_cells") != 208
            or closure.get("expected_weeks") != 52 or closure.get("year") != 2024
            or closure.get("split") != "validation" or closure.get("test_read") is not False
            or closure.get("selected_set_conditioned_on_validation") is not True):
        raise ValueError("INCOMPLETE_OR_INVALID_CLOSURE")
    if not isinstance(target, str) or not target.startswith(("Y_s_", "Y_l_")):
        raise ValueError("INVALID_TARGET")
    if (publication.get("state") != "PUBLISHED" or publication.get("target") != target
            or publication.get("closure_sha256") != closure["sha256"]
            or publication.get("reports") != 208
            or publication.get("experiment_set_key") != experiment_set_key(target)
            or publication.get("metric_horizon") != target_horizon_hours(target)):
        raise ValueError("WAREHOUSE_PUBLICATION_NOT_VERIFIED")
    arms = closure.get("arms", {})
    if set(arms) != set(ARMS):
        raise ValueError("INCOMPLETE_ARM_POPULATION")
    naive_values = []
    for arm in ARMS:
        row = arms[arm]
        mae, naive = row.get("mean_weekly_mae"), row.get("mean_weekly_naive_mae")
        wins = row.get("weeks_better_than_naive")
        if (type(mae) not in (int, float) or not math.isfinite(mae) or mae < 0
                or type(naive) not in (int, float) or not math.isfinite(naive) or naive <= 0
                or type(wins) is not int or not 0 <= wins <= 52):
            raise ValueError("INVALID_ARM_METRICS")
        naive_values.append(naive)
    if len(set(naive_values)) != 1:
        raise ValueError("UNPAIRED_NAIVE")
    return target


def summarize(closure, publication):
    target = validate(closure, publication)
    arms = closure["arms"]
    best_name = min(("ARCH_A", "ARCH_B", "ARCH_C"),
                    key=lambda arm: arms[arm]["mean_weekly_mae"])
    best = arms[best_name]
    naive = best["mean_weekly_naive_mae"]
    delta = best["mean_weekly_mae"] - naive
    arm_values = ", ".join(f"{name} {arms[name]['mean_weekly_mae']:.9f}" for name in ARMS)
    pooled = ""
    if "pooled_mae" in best and "pooled_naive_mae" in best:
        pooled = (f" Row-pooled MAE {best['pooled_mae']:.9f} versus paired naive "
                  f"{best['pooled_naive_mae']:.9f} over {best['pooled_scored_rows']} rows.") if "pooled_scored_rows" in best else ""
    gate = "passes" if delta < 0 else "does not pass"
    return (f"EURUSD {target}, 2024 weekly validation: 52/52 weeks and 208/208 matched cells "
            f"(one seed, four-year rolling refits). Best learned arm {best_name} has mean weekly "
            f"MAE {best['mean_weekly_mae']:.9f} versus same-row naive {naive:.9f} "
            f"(delta {delta:+.9f}; {best['weeks_better_than_naive']}/52 weeks beat naive)."
            f" All-arm mean weekly MAE: {arm_values}.{pooled} It {gate} the strict annual naive gate. Feature selection used this "
            f"validation year, so this is development evidence, not an external confirmation; "
            f"TEST was not read. Closure `{closure['sha256']}`; warehouse readback 208 reports "
            f"and 1,664 metric rows in `{experiment_set_key(target)}`.")


def warehouse_publication(closure, url, token_file):
    target = target_id(closure)
    set_key = experiment_set_key(target)
    sql = ("SELECT COUNT(DISTINCT r.report_sha256) AS n, COUNT(m.report_sha256) AS metrics "
           "FROM gov_report r LEFT JOIN gov_metric m ON m.report_sha256 = r.report_sha256 "
           f"WHERE r.experiment_set_key = '{set_key}' AND m.horizon = {target_horizon_hours(target)} LIMIT 1")
    endpoint = url.rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"sql": sql})
    response = request_json(endpoint, token_from_file(token_file), "GET")
    rows = response.get("rows") if isinstance(response, dict) else None
    if (not isinstance(rows, list) or len(rows) != 1
            or int(rows[0].get("n", -1)) != 208
            or int(rows[0].get("metrics", -1)) != 1664):
        raise ValueError("WAREHOUSE_READBACK_COUNT_MISMATCH")
    return {"state": "PUBLISHED", "target": target, "reports": 208,
            "closure_sha256": closure["sha256"], "experiment_set_key": set_key,
            "metric_horizon": target_horizon_hours(target)}


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def record(root, closure, publication):
    target = validate(closure, publication)
    paragraph = summarize(closure, publication)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".catalog.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target_root = root / "I6A" / "EURUSD" / "2024_validation"
        manifest = target_root / f"{target}.json"
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            if previous["closure_sha256"] != closure["sha256"]:
                raise ValueError("CATALOG_IDENTITY_CONFLICT")
        entry = {"schema": "i6a.result_catalog.v1", "target": target,
                 "closure_sha256": closure["sha256"], "paragraph": paragraph,
                 "status": "PUBLISHED", "test_read": False}
        atomic_text(manifest, json.dumps(entry, sort_keys=True, indent=2) + "\n")
        atomic_text(target_root / f"{target}.md", f"# {target}: I6-A validation\n\n{paragraph}\n")
        entries = [json.loads(p.read_text()) for p in target_root.glob("Y_*.json")]
        entries.sort(key=lambda e: e["target"])
        lines = ["# Results", "", "## I6-A / EURUSD / 2024 validation", "",
                 "Development evidence only; TEST remains sealed. Each page is generated from a complete, warehouse-published closure.", ""]
        lines.extend(f"- [{e['target']}](I6A/EURUSD/2024_validation/{e['target']}.md): "
                     f"{e['paragraph']}" for e in entries)
        atomic_text(root / "INDEX.md", "\n".join(lines) + "\n")
    return target_root / f"{target}.md"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--closure", type=Path, required=True)
    ap.add_argument("--status", type=Path)
    ap.add_argument("--warehouse-url")
    ap.add_argument("--token-file", type=Path)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args(argv)
    if bool(args.status) == bool(args.warehouse_url) or bool(args.warehouse_url) != bool(args.token_file):
        raise ValueError("SPECIFY_STATUS_OR_WAREHOUSE_READBACK")
    closure = json.loads(args.closure.read_text())
    publication = (json.loads(args.status.read_text()) if args.status
                   else warehouse_publication(closure, args.warehouse_url, args.token_file))
    print(record(args.root, closure, publication))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
