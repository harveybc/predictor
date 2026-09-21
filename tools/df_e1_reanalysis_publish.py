#!/usr/bin/env python3
"""RP57: publish the comparison table to the warehouse as a REANALYSIS, identified as one.

What this is. `tools/forecast_comparison.py` recomputes, from preserved arrays, what an already
executed run measured. That is not a new experiment and it must never enter the cube as if it were:
it is a reanalysis of a run that already has its own campaigns and terminals.

So it travels the same governed route as any other result — campaign registered before anything is
read, delivery of the SAME panel bytes verified, terminal through the outbox, campaign reconciled,
content checked in the warehouse — and every row it writes says what it is:

    purpose      E1_REANALYSIS
    of_run       the run whose arrays were recomputed
    classification NON_GOVERNING
    new_training false

The numbers themselves are the table's: MAE per method on each scale, the skill against EACH
declared reference, and the scaled errors with their period. Nothing is derived here that the report
does not already contain.

    python tools/df_e1_reanalysis_publish.py --report REPORT.json --run RUN_ROOT --root STATE_DIR \\
        --run-id satoshi-e1-reanalysis-20260920 --api-key-file KEY [--gov-url http://127.0.0.1:5055]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNIT = "rp57_comparison"
PURPOSE = "E1_REANALYSIS"


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def metrics_of(report: dict) -> list:
    """Every published number, named by scale, method and reference. No new quantity is invented."""
    U = _module("df_utility_run")
    out = []
    for scale, table in report["by_scale"].items():
        for method, row in table.items():
            out.append(U._metric(f"e1.reanalysis.mae.{scale}.{method}", row["mae"], "kW" if scale == "raw_kW"
                                 else scale, split="validation", horizon=report["horizon"]))
            for reference, against in (row.get("versus") or {}).items():
                value = against.get("mae_skill_percent")
                if value is None:                       # a zero-error reference has no skill, and none is invented
                    continue
                out.append(U._metric(f"e1.reanalysis.mae_skill_percent.{scale}.{method}.vs_{reference}",
                                     value, "percent", split="validation", horizon=report["horizon"]))
    for method, row in report["by_scale"]["raw_kW"].items():
        for key, unit in (("persistence_scaled_error_horizon_train", "ratio_persistence_h"),
                          ("conventional_mase_m1_train_slice", "mase_m1"),
                          ("conventional_mase_m1440_train_slice", "mase_m1440")):
            if row.get(key) is not None:
                out.append(U._metric(f"e1.reanalysis.{key}.{method}", row[key], unit,
                                     split="validation", horizon=report["horizon"]))
    for pair, d in (report.get("paired_differences") or {}).items():
        out.append(U._metric(f"e1.reanalysis.paired_abs_error_difference.{pair}",
                             d["mean_abs_error_difference"], "kW", split="validation",
                             horizon=report["horizon"]))
    return out


def publish(report_path: Path, run_root: Path, state_root: Path, *, run_id: str, gov_url: str,
            api_key_file: Path, lake: str, resource: str, outbox_dir: str | None = None) -> dict:
    G = _module("df_e1_governed")
    U = _module("df_utility_run")
    E0 = _module("df_mod_e0_close")
    report = json.loads(Path(report_path).read_text())
    run_design = json.loads((Path(run_root) / "DESIGN.json").read_text())
    state_root = Path(state_root)
    state_root.mkdir(parents=True, exist_ok=True)
    design = {"schema": "df_e1_reanalysis.v1", "design_sha256": run_design["design_sha256"],
              "purpose": PURPOSE, "of_run": str(run_root), "pilots": [], "cells": [{"cell_id": UNIT}]}
    (state_root / "DESIGN.json").write_text(json.dumps(design, indent=1))
    started = U._z(__import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    G.acquire(run_id=run_id, root=state_root, lake=lake, resource=resource, unit_id=UNIT,
              gov_url=gov_url, api_key_file=api_key_file, design_sha256=design["design_sha256"],
              cache_dir=state_root / "cache", expect_sha256=run_design["governed_bytes"]["sha256"])
    terminal = U._terminal(
        status="COMPLETED", reason=None,
        cost={"wall_seconds": 0.0, "cpu_seconds": 0.0},
        metrics=metrics_of(report),
        started=started, finished=U._z(__import__("datetime").datetime.now(__import__("datetime").timezone.utc)),
        tags={"purpose": PURPOSE, "classification": "NON_GOVERNING", "phase": "DEVELOPMENT",
              "unit": UNIT, "design_sha256": design["design_sha256"], "new_training": "false",
              "of_run": str(run_root), "report_sha256": G.sha_file(Path(report_path)),
              "reading": "a recomputation from preserved arrays of a run that is already closed; it is "
                         "not a new experiment and never replaces that run's own terminals"})
    reported = G.report_terminal(state_root, UNIT, terminal, gov_url=gov_url,
                                 api_key_file=api_key_file, outbox_dir=outbox_dir, started_at=started)
    receipt = {"schema": "df_e1_reanalysis_publication.v1", "run_id": run_id, "unit": UNIT,
               "purpose": PURPOSE, "of_run": str(run_root), "report": str(report_path),
               "report_sha256": G.sha_file(Path(report_path)),
               "metrics_published": len(terminal["metrics"]),
               "campaign_sha256": reported["campaign_sha256"],
               "terminal_sha256": (reported.get("receipt") or {}).get("terminal_sha256"),
               "reconciliation": reported["reconciliation"], "flushed": reported["flushed"]}
    (state_root / "PUBLICATION.json").write_text(json.dumps(receipt, indent=1, default=str))
    return receipt


def verify(state_root: Path, *, url: str, token: str) -> dict:
    """The cube's own rows against the terminal the client got: content, not a status column."""
    E0 = _module("df_mod_e0_close")
    receipt = json.loads((Path(state_root) / "PUBLICATION.json").read_text())
    held = E0.warehouse_terminals(url, token, receipt["campaign_sha256"])
    row = (held.get("current") or {}).get(UNIT)
    problems = []
    if row is None:
        problems.append("the warehouse holds no terminal for this reanalysis")
    else:
        if row.get("terminal_sha256") != receipt["terminal_sha256"]:
            problems.append("the warehouse's terminal digest is not the one the client received")
        if len(row.get("metrics") or []) != receipt["metrics_published"]:
            problems.append(f"the warehouse holds {len(row.get('metrics') or [])} metric rows, "
                            f"the client published {receipt['metrics_published']}")
        tags = json.loads(row.get("tags_json") or "{}")
        if tags.get("purpose") != PURPOSE or tags.get("new_training") != "false":
            problems.append(f"the stored rows do not identify themselves as a reanalysis: {tags}")
    return {"unit": UNIT, "campaign_sha256": receipt["campaign_sha256"], "problems": problems,
            "metric_rows": len(((row or {}).get("metrics") or [])), "identified_as_reanalysis": not problems,
            "complete": not problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--gov-url", default="http://127.0.0.1:5055")
    ap.add_argument("--api-key-file", type=Path, required=True)
    ap.add_argument("--lake", default="public_panels")
    ap.add_argument("--resource", default="uci_235_individual_household_power/panel.parquet")
    ap.add_argument("--outbox-dir", default=None)
    ap.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    ap.add_argument("--warehouse-token-file", type=Path, default=None)
    a = ap.parse_args(argv)
    receipt = publish(a.report, a.run, a.root, run_id=a.run_id, gov_url=a.gov_url,
                      api_key_file=a.api_key_file, lake=a.lake, resource=a.resource,
                      outbox_dir=a.outbox_dir)
    out = {"publication": receipt}
    if a.warehouse_token_file:
        token = a.warehouse_token_file.read_text().strip().strip('"').strip("'")
        out["warehouse"] = verify(a.root, url=a.warehouse_url, token=token)
    print(json.dumps(out, indent=1, default=str))
    return 0 if out.get("warehouse", {"complete": True})["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
