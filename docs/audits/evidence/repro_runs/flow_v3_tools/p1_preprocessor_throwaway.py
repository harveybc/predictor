#!/usr/bin/env python3
"""Flow v3 adoption, P1: a governed preprocessor run on the throwaway stack.

Same disposable services as p03 (OLAP lake on a throwaway PostgreSQL database,
data-gov with the in-process `predictor_examples` lake and the deployed
contracts). The preprocessor's `tools/governed_run.py` (profile over data-gov's
`tools/governed_exec.py`) consumes `phase_1/normalized_d4.csv` as `input_file`
with the repository's own phase_1b downsampled configuration, and its terminal
(row counts of the twelve split files, artifact hashes, one verified delivery)
is reconciled between accounting and the cube. A second run into the same
output directory must be REFUSED without downloading.

Output: JSON on stdout; local roots printed as ~.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
P03 = importlib.util.spec_from_file_location("p03_helpers", HERE.with_name("p03_throwaway_governed_runs.py"))
p03 = importlib.util.module_from_spec(P03)
P03.loader.exec_module(p03)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preprocessor-checkout", required=True, type=Path)
    ap.add_argument("--predictor-checkout", required=True, type=Path, help="serves examples/data_downsampled")
    ap.add_argument("--data-gov-checkout", required=True, type=Path)
    ap.add_argument("--olap-lake-checkout", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--pg-db", required=True)
    a = ap.parse_args(argv)
    prep, predictor = a.preprocessor_checkout.resolve(), a.predictor_checkout.resolve()
    data_gov, olap, work = a.data_gov_checkout.resolve(), a.olap_lake_checkout.resolve(), a.work_dir.resolve()
    if work.exists():
        raise SystemExit(f"work dir exists: {work}")
    if not a.pg_db.startswith("data_gov_v3_throwaway_"):
        raise SystemExit("refusing: throwaway database must be named data_gov_v3_throwaway_*")
    work.mkdir(parents=True)
    git = lambda repo, *args: subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout.strip()  # noqa: E731
    result = {"schema": "flow_v3_p1_preprocessor_throwaway.v1",
              "code_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
              "preprocessor_commit": git(prep, "rev-parse", "HEAD"),
              "preprocessor_clean": git(prep, "status", "--porcelain", "--untracked-files=all") == "",
              "data_gov_commit": git(data_gov, "rev-parse", "HEAD"), "pg_db": a.pg_db, "runs": {}}
    actor_key, lake_token, salt = "throwaway-actor-" + os.urandom(8).hex(), "throwaway-lake-" + os.urandom(8).hex(), "throwaway-salt"
    key_file = work / "actor.key"
    key_file.write_text(actor_key + "\n", encoding="ascii")
    key_file.chmod(0o600)
    olap_port, gov_port = p03.port(), p03.port()
    gov_url = f"http://127.0.0.1:{gov_port}"
    lake_root = predictor / "examples/data_downsampled"
    (work / "olap.json").write_text(json.dumps({
        "pipeline_plugin": "default_pipeline", "web_plugin": "default_web", "query_plugin": "sql_query",
        "web_host": "127.0.0.1", "web_port": olap_port, "holdout_start": "2025-01-01", "lake_id": "olap_cube"}, indent=1))
    (work / "governance.json").write_text(json.dumps({
        "pipeline_plugin": "default_pipeline", "web_plugin": "default_web", "access_plugin": "default_access",
        "accounting_plugin": "default_accounting", "role_plugin": "default_role", "web_host": "127.0.0.1",
        "web_port": gov_port, "accounting_db": str(work / "accounting.sqlite"), "spool_dir": str(work / "gov-spool"),
        "cuts_dir": str(work / "gov-cuts"), "save_config": None, "password_salt": salt, "secret_key": "throwaway",
        "principals": {"preprocessor": {"kind": "service", "role": "service",
                                        "api_key_hash": hashlib.sha256(f"{salt}:{actor_key}".encode()).hexdigest()}},
        "policies": [
            {"principal": "preprocessor", "lake": "predictor_examples", "verbs": ["discover", "coverage", "read", "download"], "deny_from": "2025-01-01"},
            {"principal": "preprocessor", "lake": "olap_cube", "verbs": ["discover", "query", "write_terminal"], "deny_from": "2025-01-01"}],
        "lakes": [
            {"plugin": "files_lake", "lake_id": "predictor_examples", "title": "predictor examples", "kind": "files_inventory",
             "root_path": str(lake_root), "include_globs": ["**/*.csv"], "time_column": "DATE_TIME",
             "holdout_start": "2025-01-01", "cuts_dir": str(work / "lake-cuts"), "spool_dir": str(work / "lake-spool"),
             "resource_contracts": {r: p03.CONTRACT for r in p03.RESOURCES}},
            {"plugin": "http_lake", "lake_id": "olap_cube", "base_url": f"http://127.0.0.1:{olap_port}", "holdout_start": "2025-01-01"}]},
        indent=1))
    base_cfg = json.loads((prep / "examples/config_downsampled/phase_1b.json").read_text(encoding="utf-8"))
    base_cfg["input_file"] = str(lake_root / "phase_1/normalized_d4.csv")
    cfg_dir = work / "phase_1b_governed"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "config.json"
    cfg_path.write_text(json.dumps(base_cfg, indent=1))
    result["config_sha256"] = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    p03.pg_admin("predictor_olap", f'CREATE DATABASE "{a.pg_db}"')
    olap_env = {"DATA_GOV_LAKE_TOKEN": lake_token, "PGDATABASE": a.pg_db}
    procs = {}

    def governed(key, out_dir, log):
        cmd = [sys.executable, "tools/governed_run.py", "--load_config", str(cfg_path), "--experiment-key", key,
               "--gov-url", gov_url, "--api-key-file", str(key_file), "--lake", "predictor_examples",
               "--lake-root", str(lake_root), "--metrics-lake", "olap_cube", "--out-dir", str(out_dir),
               "--cache-dir", str(work / "cache"), "--outbox-dir", str(work / "outbox")]
        with open(log, "ab") as handle:
            return subprocess.run(cmd, cwd=prep, env=dict(os.environ, DATA_GOV_CHECKOUT=str(data_gov), CUDA_VISIBLE_DEVICES=""),
                                  stdout=handle, stderr=subprocess.STDOUT).returncode

    try:
        procs["olap"] = p03.start(olap, work / "olap.json", olap_env, work / "olap.log")
        p03.wait_health(f"http://127.0.0.1:{olap_port}/healthz", [procs["olap"]])
        procs["gov"] = p03.start(data_gov, work / "governance.json", {"DATA_GOV_LAKE_TOKEN": lake_token}, work / "governance.log")
        p03.wait_health(f"{gov_url}/healthz", list(procs.values()))
        out = work / "run1"
        rc = governed("fv3-prep-completed", out, work / "run1.log")
        r = p03.receipt(out)
        camp = r.get("campaign_sha256")
        rows = p03.pg(a.pg_db, "SELECT status, reason FROM public.gov_terminal WHERE campaign_sha256=%s", (camp,))
        metrics = p03.pg(a.pg_db, "SELECT m.metric, m.split, m.value FROM public.gov_terminal_metric m JOIN public.gov_terminal t USING (terminal_sha256) WHERE t.campaign_sha256=%s ORDER BY m.split", (camp,))
        datasets = p03.pg(a.pg_db, "SELECT role, resource_id, sha256, verification_state, availability_contract_sha256 FROM public.gov_terminal_dataset d JOIN public.gov_terminal t USING (terminal_sha256) WHERE t.campaign_sha256=%s", (camp,))
        artifacts = p03.pg(a.pg_db, "SELECT role, bytes FROM public.gov_terminal_artifact x JOIN public.gov_terminal t USING (terminal_sha256) WHERE t.campaign_sha256=%s ORDER BY role", (camp,))
        st, rec = p03.request(f"{gov_url}/api/v2/campaigns/{camp}/reconcile", token=actor_key, headers={"X-Campaign-SHA256": camp}) if camp else (None, {})
        produced = sorted(p.name for p in out.iterdir())
        e1 = {"exit_code": rc, "status": r.get("status"), "reason": r.get("reason"), "campaign_sha256": camp,
              "terminal_pending": r.get("terminal_pending"), "code_identity": r.get("code_identity"),
              "inputs": r.get("inputs"), "cube_rows": rows, "metrics": metrics, "datasets": datasets,
              "artifacts": artifacts, "reconcile_http": rec, "produced_files": produced,
              "log_tail": (work / "run1.log").read_text(errors="replace")[-1500:] if rc else ""}
        e1["ok"] = (rc == 0 and r.get("status") == "COMPLETED" and [x[0] for x in rows] == ["COMPLETED"]
                    and len(metrics) == 12 and all(m[0] == "rows" and m[2] > 0 for m in metrics)
                    and len(datasets) == 1 and datasets[0][0] == "input_file" and datasets[0][3] == "VERIFIED_TRANSFER"
                    and datasets[0][2] == "6412c3cdc42942be2a4de5ba893682a33cfe63613f63ab94571a1ead523c18ee"
                    and rec.get("missing_units") == [] and rec.get("accounting_only") == [] and rec.get("lake_only") == []
                    and len(artifacts) >= 12)
        result["runs"]["1_completed"] = e1
        rc2 = governed("fv3-prep-refused", out, work / "run2.log")
        r2 = p03.receipt(out)
        rows2 = p03.pg(a.pg_db, "SELECT status, reason FROM public.gov_terminal WHERE campaign_sha256=%s", (r2.get("campaign_sha256"),))
        result["runs"]["2_refused_stale_outputs"] = {
            "exit_code": rc2, "status": r2.get("status"), "reason": r2.get("reason"), "cube_rows": rows2,
            "no_downloads": "inputs" not in r2,
            "ok": rc2 != 0 and r2.get("status") == "REFUSED" and [x[0] for x in rows2] == ["REFUSED"] and "inputs" not in r2}
        result["throwaway_cube_counts_final"] = p03.gov_counts(a.pg_db)
        result["ok"] = all(v["ok"] for v in result["runs"].values())
    finally:
        for p in procs.values():
            p03.stop(p)
        p03.pg_admin("predictor_olap", f'DROP DATABASE IF EXISTS "{a.pg_db}"')
        result["pg_db_dropped"] = a.pg_db not in {r[0] for r in p03.pg("predictor_olap", "SELECT datname FROM pg_database")}
        shutil.rmtree(work / "cache", ignore_errors=True)
    sys.stdout.write(json.dumps(result, indent=1, default=str).replace(str(Path.home()), "~") + "\n")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
