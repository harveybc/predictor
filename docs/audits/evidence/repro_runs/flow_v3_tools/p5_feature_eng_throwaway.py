#!/usr/bin/env python3
"""GOV-N5 on the throwaway stack: the real feature-eng pipeline (tech_indicator plugin)
on governed synthetic inputs, through feature-eng's tools/governed_run.py.

Disposable services as in p03 plus a `feature_eng_fixtures` lake rooted at the
feature-eng checkout's tests/data/governed (synthetic OHLC with an explicit
contract and scope). Runs:

  1. COMPLETED  input A, fresh out-dir      -> metrics rows/columns of output_file,
                                              every produced CSV/PNG hashed as artifacts
  2. REFUSED    same out-dir                -> stale outputs, no download
  3. COMPLETED  input B (another fixture)   -> a different delivery hash and a different
                                              campaign identity: lineage follows the input
  4. FAILED     bogus plugin name           -> terminal with cost and its delivery kept
  5. retry: a second flush sends nothing    -> no duplicate terminal
The feature-eng checkout must be clean; its plugins resolve from that checkout's
own egg-info (created here with `setup.py egg_info`, ignored by git), never from
another checkout on sys.path. Output: JSON on stdout; local roots printed as ~.
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
import time
from pathlib import Path

HERE = Path(__file__).resolve()
P03 = importlib.util.spec_from_file_location("p03_helpers", HERE.with_name("p03_throwaway_governed_runs.py"))
p03 = importlib.util.module_from_spec(P03)
P03.loader.exec_module(p03)

def fixture_contract(entry: dict) -> dict:
    """The contract of a synthetic fixture, from its manifest entry (we are the producer)."""
    return {"event_time_column": entry["time_column"], "available_time_column": entry["time_column"],
            "timezone": "NAIVE_WALL_CLOCK", "time_unit": None, "frequency": entry["frequency"],
            "availability": {"label": entry["label"], "completion_lag_max": entry["completion_lag_max"],
                             "timezone_evidence": "PRODUCER_STATEMENT", "use_class": "OFFLINE_DAY_GRANULAR"}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-eng-checkout", required=True, type=Path)
    ap.add_argument("--data-gov-checkout", required=True, type=Path)
    ap.add_argument("--olap-lake-checkout", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--pg-db", required=True)
    ap.add_argument("--feature-eng-python", default=sys.executable,
                    help="interpreter of feature-eng's own environment (requirements-governed.txt)")
    a = ap.parse_args(argv)
    feng, data_gov, olap, work = (a.feature_eng_checkout.resolve(), a.data_gov_checkout.resolve(),
                                  a.olap_lake_checkout.resolve(), a.work_dir.resolve())
    if work.exists():
        raise SystemExit(f"work dir exists: {work}")
    if not a.pg_db.startswith("data_gov_v3_throwaway_"):
        raise SystemExit("refusing: throwaway database must be named data_gov_v3_throwaway_*")
    work.mkdir(parents=True)
    git = lambda *args: subprocess.run(["git", *args], cwd=feng, capture_output=True, text=True).stdout.strip()  # noqa: E731
    # plugins resolve through entry points: give this checkout its own egg-info (git-ignored)
    py = str(Path(a.feature_eng_python).expanduser().absolute())  # not resolve(): a venv python is a symlink
    egg = subprocess.run([py, "setup.py", "egg_info"], cwd=feng, capture_output=True, text=True)
    fixtures = feng / "tests/data/governed"
    manifest = json.loads((fixtures / "MANIFEST.json").read_text(encoding="utf-8"))
    result = {"schema": "flow_v3_n5_feature_eng_throwaway.v1", "code_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
              "feature_eng_commit": git("rev-parse", "HEAD"),
              "feature_eng_clean": git("status", "--porcelain", "--untracked-files=all") == "",
              "egg_info_rc": egg.returncode, "feature_eng_python": py.replace(str(Path.home()), "~"),
              "fixtures_manifest": manifest, "pg_db": a.pg_db, "runs": {}}
    actor_key, lake_token, salt = "throwaway-actor-" + os.urandom(8).hex(), "throwaway-lake-" + os.urandom(8).hex(), "throwaway-salt"
    key_file = work / "actor.key"
    key_file.write_text(actor_key + "\n"); key_file.chmod(0o600)
    olap_port, gov_port = p03.port(), p03.port()
    gov_url = f"http://127.0.0.1:{gov_port}"
    (work / "olap.json").write_text(json.dumps({
        "pipeline_plugin": "default_pipeline", "web_plugin": "default_web", "query_plugin": "sql_query",
        "web_host": "127.0.0.1", "web_port": olap_port, "holdout_start": "2025-01-01", "lake_id": "olap_cube"}, indent=1))
    (work / "governance.json").write_text(json.dumps({
        "pipeline_plugin": "default_pipeline", "web_plugin": "default_web", "access_plugin": "default_access",
        "accounting_plugin": "default_accounting", "role_plugin": "default_role", "web_host": "127.0.0.1",
        "web_port": gov_port, "accounting_db": str(work / "accounting.sqlite"), "spool_dir": str(work / "gov-spool"),
        "cuts_dir": str(work / "gov-cuts"), "save_config": None, "password_salt": salt, "secret_key": "throwaway",
        "principals": {"feature-eng": {"kind": "service", "role": "service",
                                       "api_key_hash": hashlib.sha256(f"{salt}:{actor_key}".encode()).hexdigest()}},
        "policies": [
            {"principal": "feature-eng", "lake": "feature_eng_fixtures", "verbs": ["discover", "coverage", "read", "download"], "deny_from": "2025-01-01"},
            {"principal": "feature-eng", "lake": "olap_cube", "verbs": ["discover", "query", "write_terminal"], "deny_from": "2025-01-01"}],
        "lakes": [
            {"plugin": "files_lake", "lake_id": "feature_eng_fixtures", "title": "feature-eng synthetic fixtures", "kind": "lake",
             "root_path": str(fixtures), "include_globs": ["*.csv"], "holdout_start": "2025-01-01",
             "cuts_dir": str(work / "lake-cuts"), "spool_dir": str(work / "lake-spool"),
             "time_columns": {f["file"]: f["time_column"] for f in manifest["files"]},
             "resource_contracts": {f["file"]: fixture_contract(f) for f in manifest["files"]}},
            {"plugin": "http_lake", "lake_id": "olap_cube", "base_url": f"http://127.0.0.1:{olap_port}", "holdout_start": "2025-01-01"}]},
        indent=1))
    base_cfg = {"input_file": str(fixtures / "synthetic_ohlc_1h_a.csv"), "output_file": "./indicators_output.csv",
                "save_log": "./debug_log.json", "save_config": "./output_config.json", "plugin": "tech_indicator",
                "dataset_type": "forex_15m", "tech_indicators": True, "seasonality_columns": True,
                "correlation_analysis": False, "distribution_plot": True, "quiet_mode": True,
                "high_freq_dataset": None, "sp500_dataset": None, "vix_dataset": str(fixtures / "synthetic_vix_daily.csv"),
                "economic_calendar": None,
                "forex_datasets": None}
    cfg_dir = work / "phase_synthetic"
    cfg_dir.mkdir()

    def config(name, **changes):
        path = cfg_dir / f"{name}.json"
        path.write_text(json.dumps({**base_cfg, **changes}, indent=1))
        return path

    p03.pg_admin("predictor_olap", f'CREATE DATABASE "{a.pg_db}"')
    olap_env = {"DATA_GOV_LAKE_TOKEN": lake_token, "PGDATABASE": a.pg_db}
    procs = {}
    outbox, cache = work / "outbox", work / "cache"

    def governed(key, cfg, out_dir, log, *extra):
        cmd = [py, "tools/governed_run.py", "--load_config", str(cfg), "--experiment-key", key,
               "--gov-url", gov_url, "--api-key-file", str(key_file), "--lake", "feature_eng_fixtures",
               "--lake-root", str(fixtures), "--metrics-lake", "olap_cube", "--out-dir", str(out_dir),
               "--cache-dir", str(cache), "--outbox-dir", str(outbox), *extra]
        with open(log, "ab") as handle:
            rc = subprocess.run(cmd, cwd=feng, env=dict(os.environ, DATA_GOV_CHECKOUT=str(data_gov), CUDA_VISIBLE_DEVICES=""),
                                stdout=handle, stderr=subprocess.STDOUT).returncode
        r = p03.receipt(out_dir)
        camp = r.get("campaign_sha256")
        rows = p03.pg(a.pg_db, "SELECT status, reason, costs_json FROM public.gov_terminal WHERE campaign_sha256=%s", (camp,)) if camp else []
        metrics = p03.pg(a.pg_db, "SELECT m.metric, m.split, m.value FROM public.gov_terminal_metric m JOIN public.gov_terminal t USING (terminal_sha256) WHERE t.campaign_sha256=%s ORDER BY 1,2", (camp,)) if camp else []
        datasets = p03.pg(a.pg_db, "SELECT role, resource_id, sha256, verification_state, availability_contract_sha256 FROM public.gov_terminal_dataset d JOIN public.gov_terminal t USING (terminal_sha256) WHERE t.campaign_sha256=%s", (camp,)) if camp else []
        artifacts = p03.pg(a.pg_db, "SELECT role, sha256, bytes FROM public.gov_terminal_artifact x JOIN public.gov_terminal t USING (terminal_sha256) WHERE t.campaign_sha256=%s ORDER BY role", (camp,)) if camp else []
        rec = p03.request(f"{gov_url}/api/v2/campaigns/{camp}/reconcile", token=actor_key, headers={"X-Campaign-SHA256": camp})[1] if camp else {}
        produced = sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else []
        return {"exit_code": rc, "status": r.get("status"), "reason": r.get("reason"), "campaign_sha256": camp,
                "terminal_pending": r.get("terminal_pending"), "inputs": r.get("inputs"), "cube_rows": rows,
                "metrics": metrics, "datasets": datasets, "artifacts": artifacts, "reconcile": rec, "produced_files": produced,
                "log_tail": Path(log).read_text(errors="replace")[-1200:] if rc else ""}

    try:
        procs["olap"] = p03.start(olap, work / "olap.json", olap_env, work / "olap.log")
        p03.wait_health(f"http://127.0.0.1:{olap_port}/healthz", [procs["olap"]])
        procs["gov"] = p03.start(data_gov, work / "governance.json", {"DATA_GOV_LAKE_TOKEN": lake_token}, work / "governance.log")
        p03.wait_health(f"{gov_url}/healthz", list(procs.values()))
        runs = result["runs"]
        out1 = work / "run1"
        r1 = governed("fv3-feng-a", config("a"), out1, work / "run1.log")
        exact = rec_ok = lambda rec: rec.get("missing_units") == [] and rec.get("accounting_only") == [] and rec.get("lake_only") == []  # noqa: E731
        csv_artifacts = [x for x in r1["artifacts"] if x[0].startswith("csv")]
        r1["ok"] = (r1["exit_code"] == 0 and r1["status"] == "COMPLETED" and [x[0] for x in r1["cube_rows"]] == ["COMPLETED"]
                    and {m[0] for m in r1["metrics"]} == {"rows", "columns"} and all(m[2] > 0 for m in r1["metrics"])
                    and len(r1["datasets"]) == 2 and all(d[3] == "VERIFIED_TRANSFER" for d in r1["datasets"])
                    and {d[2] for d in r1["datasets"]} == {manifest["files"][0]["sha256"], manifest["files"][2]["sha256"]}
                    and len(csv_artifacts) >= 2 and exact(r1["reconcile"]))
        runs["1_completed_input_a"] = r1
        r2 = governed("fv3-feng-stale", config("a"), out1, work / "run2.log")
        r2["ok"] = r2["exit_code"] != 0 and r2["status"] == "REFUSED" and [x[0] for x in r2["cube_rows"]] == ["REFUSED"] and "inputs" not in (p03.receipt(out1))
        runs["2_refused_stale_outputs"] = r2
        out3 = work / "run3"
        r3 = governed("fv3-feng-b", config("b", input_file=str(fixtures / "synthetic_ohlc_1h_b.csv")), out3, work / "run3.log")
        main_b = [d for d in r3["datasets"] if d[0] == "input_file"]
        r3["ok"] = (r3["exit_code"] == 0 and r3["status"] == "COMPLETED" and len(r3["datasets"]) == 2
                    and main_b and main_b[0][2] == manifest["files"][1]["sha256"] and main_b[0][2] != manifest["files"][0]["sha256"]
                    and r3["campaign_sha256"] != r1["campaign_sha256"] and exact(r3["reconcile"]))
        runs["3_completed_input_b_changes_lineage"] = r3
        out4 = work / "run4"
        r4 = governed("fv3-feng-failed", config("bogus", plugin="no_such_plugin_fv3"), out4, work / "run4.log")
        costs = json.loads(r4["cube_rows"][0][2]) if r4["cube_rows"] else {}
        r4["ok"] = (r4["exit_code"] != 0 and r4["status"] == "FAILED" and [x[0] for x in r4["cube_rows"]] == ["FAILED"]
                    and costs.get("wall_seconds", -1) >= 0 and len(r4["datasets"]) == 2 and r4["metrics"] == [])
        runs["4_failed_bogus_plugin"] = r4
        before = p03.gov_counts(a.pg_db)
        flush = subprocess.run([sys.executable, str(data_gov / "tools/governed_exec.py"), "--flush", "--gov-url", gov_url,
                                "--api-key-file", str(key_file), "--outbox-dir", str(outbox)], cwd=feng, capture_output=True, text=True)
        after = p03.gov_counts(a.pg_db)
        runs["5_retry_sends_nothing"] = {"flush_stdout": flush.stdout.strip(), "rc": flush.returncode, "before": before, "after": after,
                                        "ok": flush.returncode == 0 and '"sent": 0' in flush.stdout and before == after}
        result["throwaway_cube_counts_final"] = after
        result["ok"] = all(v["ok"] for v in runs.values())
    finally:
        for p in procs.values():
            p03.stop(p)
        time.sleep(1)
        p03.pg_admin("predictor_olap", f'DROP DATABASE IF EXISTS "{a.pg_db}"')
        result["pg_db_dropped"] = a.pg_db not in {r[0] for r in p03.pg("predictor_olap", "SELECT datname FROM pg_database")}
        shutil.rmtree(cache, ignore_errors=True)
    sys.stdout.write(json.dumps(result, indent=1, default=str).replace(str(Path.home()), "~") + "\n")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
