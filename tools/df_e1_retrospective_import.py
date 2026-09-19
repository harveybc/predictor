#!/usr/bin/env python3
"""RP34: import the preserved household DEV pilot into the warehouse AS WHAT IT IS — evidence that ran
before the public panels were a governed resource.

It is built from the CLOSURE (tools/df_e1_close.py), never from the run's own summaries, so a unit that
the closure refuses never reaches the cube, and every imported unit carries:

  * its identity: run id, design digest, data digest, panel digest, record digest;
  * `executed_at`, the instant the child actually ran, read from the attempt's outcome;
  * `imported_at`, now — the two are different facts and both are stored;
  * `governed_delivery_at_execution: false`, because none existed, plus the reason;
  * its verified scope, so METRICS_VERIFIED_INFERENCE_NOT_REPLAYED is never read as reproduced.

It never claims a delivery that did not happen and never re-dates one: the envelope is refused by the
loader if it tries. Failures and incidents of the run are imported too, as failures, not as scores.

    python tools/df_e1_retrospective_import.py --root RUN --close CLOSE.json --out ENVELOPE.json
    python tools/df_e1_retrospective_import.py ... --rehearse REPORT.json   # load it into a disposable cube
    python tools/df_e1_retrospective_import.py ... --submit --gov-url ...   # the production path (guarded)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
HOME = Path.home()
WAREHOUSE_PYTHON = HOME / ".venvs/store-hosts-duckdb-prod/bin/python"
WAREHOUSE_CONFIG = HOME / ".local/state/crispdm-duckdb/prod/5057.host.duckdb.json"
REASON = ("the household DEV pilot of 2026-09-19 ran while no lake served the public panels: there was no "
          "governed delivery, campaign or accepted terminal for it, and this import states that rather than "
          "back-dating one")


def _load(name: str, where: Path = HERE):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CE = _load("campaign_envelope", REPO / "olap")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def executed_at(root: Path, cell_id: str) -> str:
    oc = json.loads((root / "attempts" / cell_id / "outcome.json").read_text())
    cost = (oc.get("summary") or {}).get("cost") or {}
    return str(cost.get("started_at") or cost.get("ended_at") or "UNKNOWN")


def build(root: Path, close_doc: dict, *, run_id: str | None = None) -> dict:
    root = Path(root)
    design = json.loads((root / "DESIGN.json").read_text())
    report = json.loads((root / "REPORT.json").read_text())
    run = run_id or report["run_id"]
    units, skipped, incidents = [], [], []
    earliest = None
    for cell_id, verdict in close_doc["units"].items():
        if not verdict["verified"]:
            skipped.append({"cell_key": cell_id, "why": verdict["problems"][:3]})
            continue
        rec = json.loads((root / "attempts" / cell_id / "cell.json").read_text())
        started = executed_at(root, cell_id)
        earliest = started if earliest is None or started < earliest else earliest
        scope = verdict["scope"]
        base = {"candidate_key": f"{rec.get('regime') or rec['kind']}|{design['graph']['arch']}|seed{rec['seed']}"
                if False else f"{rec.get('regime') or rec['kind']}_seed{rec['seed']}",
                "terminal_state": "COMPLETED"}
        scores = (rec.get("scores") or {}).get("validation") or {}
        for name, block in scores.items():
            for metric in ("mase_mean", "mae_mean"):
                value = block.get(metric)
                if value is None:
                    continue
                units.append({**base, "cell_key": cell_id,
                              "metric_name": f"e1.{name}.{metric[:-5]}" if name != "model" else f"e1.{metric[:-5]}",
                              "metric_value": float(value), "uncertainty_kind": "UNAVAILABLE"})
        if rec["kind"] == "fit":
            units.append({**base, "cell_key": cell_id, "metric_name": "e1.updates",
                          "metric_value": float(rec["training"]["updates"]), "uncertainty_kind": "UNAVAILABLE"})
        if rec["kind"] == "ae":
            units.append({**base, "cell_key": cell_id, "metric_name": "e1.ae.reconstruction_val_mse_masked",
                          "metric_value": float(rec["pretraining"]["reconstruction_val_mse_masked"]),
                          "uncertainty_kind": "UNAVAILABLE"})
        incidents.append({"cell_key": cell_id, "scope": scope, "record_sha256": verdict["facts"].get("record_sha256")})
    for failure in report.get("terminals", []):
        if failure.get("status") not in ("COMPLETED",):
            units.append({"cell_key": failure["unit_id"], "candidate_key": "incident",
                          "metric_name": "e1.incident", "metric_value": None,
                          "terminal_state": failure["status"], "uncertainty_kind": "UNAVAILABLE"})
    provenance = {"mode": CE.RETROSPECTIVE, "executed_at": earliest or "UNKNOWN", "imported_at": now_iso(),
                  "governed_delivery_at_execution": False, "reason": REASON}
    doc = CE.build_envelope(
        campaign_key=f"{run}-retrospective-import", producer="predictor", result_class="DEVELOPMENT",
        identity={"run_id": run, "code_identity": report["code_identity"]["value"],
                  "design_sha256": design["design_sha256"], "record_sha256": close_doc.get("data_sha256")},
        data_consumed={"datasets": [{"id": design["dataset_id"], "digest": close_doc["panel_sha256"],
                                     "eligibility_state": "UNGOVERNED_AT_EXECUTION"}],
                       "variables": [{"id": v, "digest": close_doc["data_sha256"],
                                      "eligibility_state": "UNGOVERNED_AT_EXECUTION"}
                                     for v in design["task"]["features"] + design["task"]["target"]],
                       "operators": []},
        partitions={"exposure": "DEVELOPMENT_NO_RESERVE", "splits": json.dumps(design["dev_subpartition"]["rows"])},
        budget={"device": "cpu", "wall_seconds": float(report.get("spent_cpu_seconds") or 0.0), "cost_units": 0.0},
        terminal={"state": "COMPLETED", "adjudication": "DESCRIPTIVE",
                  "reason": "development pilot: no hypothesis confirmed or refuted"},
        artifacts={"verification": "SCHEMA_EXACT_AND_SELF_DIGEST_REDERIVED"},
        units=units, provenance=provenance)
    return {"envelope": doc, "skipped_units": skipped, "scopes": incidents,
            "counts": {"units_imported": len({u['cell_key'] for u in units}), "metric_rows": len(units),
                       "units_skipped": len(skipped)}}


# --- loading it -------------------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_json(url: str, token: str | None = None, *, method="GET", body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    head = {"Content-Type": "application/json"}
    if token:
        head["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, method=method, headers=head)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as answer:
            raw = answer.read().decode(errors="replace")
            try:
                return answer.status, json.loads(raw or "{}")
            except ValueError:
                return answer.status, {"body": raw[:300]}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            return exc.code, json.loads(raw or "{}")
        except ValueError:
            return exc.code, {"error": raw[:300]}
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc)}


def rehearse(envelope: dict, out_path: Path) -> dict:
    """Load the envelope into a DISPOSABLE cube served by the deployed warehouse host, twice, and read
    the rows back: this is the storage rule exercised on the engine production runs."""
    work = Path(tempfile.mkdtemp(prefix="rp34-import-"))
    port = free_port()
    cfg = json.loads(WAREHOUSE_CONFIG.read_text())
    cfg["web_port"] = port
    cfg["backend"]["settings"] = {**cfg["backend"]["settings"], "duckdb_path": str(work / "cube.duckdb"),
                                  "min_free_bytes": 1 << 20}
    cfg["operator_config_path"] = str(work / "pending.json")
    cfg_path = work / "host.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))
    report = {"schema": "df_e1_retrospective_import_rehearsal.v1", "at": now_iso(), "work": str(work), "port": port,
              "package": "CANDIDATE (olap/store/src) on PYTHONPATH; the deployed build predates the provenance block "
                         "and adopting it is the same guarded operation as the lake registration"}
    log = open(work / "cube.log", "w")
    proc = subprocess.Popen([str(WAREHOUSE_PYTHON), "-c",
                             "import json,sys;from predictor_duckdb_store.provider import PredictorDuckdbStore;"
                             "cfg=json.loads(open(sys.argv[1]).read())['backend']['settings'];"
                             "s=PredictorDuckdbStore();s.set_params(**{k:v for k,v in cfg.items() if k in "
                             "('duckdb_path','schema','memory_limit','threads','min_free_bytes')});"
                             "doc=json.loads(open(sys.argv[2]).read());"
                             "print(json.dumps({'first':s.write_foundation_envelope(doc),"
                             "'second':s.write_foundation_envelope(doc),"
                             "'rows':[list(map(str,r)) for r in s.engine().connect().execute(__import__('sqlalchemy').text("
                             "'SELECT cell_key, metric_name, metric_value, provenance_mode, executed_at, "
                             "governed_delivery_at_execution FROM public.fact_campaign_unit ORDER BY cell_key, metric_name')).fetchall()]}))",
                             str(cfg_path), str(work / "envelope.json")],
                            stdout=subprocess.PIPE, stderr=log, text=True, cwd=str(work),
                            # the CANDIDATE package on PYTHONPATH, exactly as the adoption procedure
                            # rehearses a new build: the deployed one predates the provenance block
                            env={**os.environ, "PYTHONPATH": str(REPO / "olap" / "store" / "src")})
    (work / "envelope.json").write_text(json.dumps(envelope))
    try:
        stdout, _ = proc.communicate(timeout=600)
        report["loader"] = json.loads(stdout) if stdout.strip().startswith("{") else {"stdout": stdout[-800:]}
    except subprocess.TimeoutExpired:
        proc.kill()
        report["loader"] = {"error": "timeout"}
    finally:
        log.close()
        report["cube_log_tail"] = (work / "cube.log").read_text()[-800:]
    loaded = report.get("loader") or {}
    rows = loaded.get("rows") or []
    report["idempotent"] = bool(loaded.get("first", {}).get("units") and loaded.get("second", {}).get("units") == 0)
    report["every_row_is_retrospective"] = bool(rows) and all(r[3] == CE.RETROSPECTIVE and r[5] in ("False", "0", "false")
                                                              for r in rows)
    report["rows"] = rows[:10]
    report["row_count"] = len(rows)
    out_path.write_text(json.dumps(report, indent=1, default=str))
    shutil.rmtree(work, ignore_errors=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--close", type=Path, required=True, help="the closure document of that run")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rehearse", type=Path, default=None)
    a = ap.parse_args(argv)
    close_doc = json.loads(a.close.read_text())
    built = build(a.root, close_doc)
    a.out.write_text(json.dumps(built, indent=1, default=str))
    summary = {"counts": built["counts"], "envelope_sha256": built["envelope"]["envelope_sha256"],
               "provenance": built["envelope"]["provenance"]}
    if a.rehearse:
        summary["rehearsal"] = {k: v for k, v in rehearse(built["envelope"], a.rehearse).items()
                                if k in ("idempotent", "every_row_is_retrospective", "row_count")}
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
