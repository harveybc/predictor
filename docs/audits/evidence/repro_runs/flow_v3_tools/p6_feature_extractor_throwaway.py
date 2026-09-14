#!/usr/bin/env python3
"""GOV-N5 for feature-extractor, on the throwaway stack, after the port the owner decided.

feature-extractor was blocked on predictor's preprocessor API drift
(`run_preprocessing(self, target_plugin, config)` since predictor 9b7d611). The port is in
`app/preprocessing_api.py`; this script proves the ported pipeline under governance:

  1. COMPLETED  six governed inputs, fresh out-dir -> the metrics of `save_log`,
                encoder/decoder/log/config hashed as artifacts, six verified deliveries over
                three distinct resources (three transferred, three from the governed cache)
  2. REFUSED    same out-dir                       -> stale outputs, no download
  3. FAILED     bogus encoder plugin               -> terminal with cost, deliveries kept
  4. retry: a second flush sends nothing           -> no duplicate terminal

Everything is disposable: services on free ports, a `data_gov_v3_throwaway_*` database that
is dropped at the end, and a fixtures lake rooted at the feature-extractor checkout's
`examples/data/phase_3`, which the lake only ever reads. Output: JSON on stdout, with local
roots printed as `~`.
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

RESOURCES = ["normalized_d1.csv", "normalized_d2.csv", "normalized_d3.csv"]
CONTRACT = {"event_time_column": "DATE_TIME", "available_time_column": "DATE_TIME",
            "timezone": "NAIVE_WALL_CLOCK", "time_unit": None, "frequency": "1h",
            "availability": {"label": "WINDOW_END", "completion_lag_max": "1h",
                             "timezone_evidence": "PRODUCER_STATEMENT",
                             "use_class": "OFFLINE_DAY_GRANULAR"}}
METRICS = {"final_training_mae_logged", "final_validation_mae_logged", "execution_time_seconds"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-extractor-checkout", required=True, type=Path)
    ap.add_argument("--data-gov-checkout", required=True, type=Path)
    ap.add_argument("--olap-lake-checkout", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--pg-db", required=True)
    ap.add_argument("--feature-extractor-python", default=sys.executable,
                    help="interpreter of feature-extractor's own environment (requirements-governed.txt)")
    a = ap.parse_args(argv)
    fext, data_gov, olap, work = (a.feature_extractor_checkout.resolve(), a.data_gov_checkout.resolve(),
                                  a.olap_lake_checkout.resolve(), a.work_dir.resolve())
    if work.exists():
        raise SystemExit(f"work dir exists: {work}")
    if not a.pg_db.startswith("data_gov_v3_throwaway_"):
        raise SystemExit("refusing: throwaway database must be named data_gov_v3_throwaway_*")
    work.mkdir(parents=True)
    git = lambda *args: subprocess.run(["git", *args], cwd=fext, capture_output=True, text=True).stdout.strip()  # noqa: E731
    py = str(Path(a.feature_extractor_python).expanduser().absolute())
    egg = subprocess.run([py, "setup.py", "egg_info"], cwd=fext, capture_output=True, text=True)
    fixtures = fext / "examples/data/phase_3"
    digests = {name: hashlib.sha256((fixtures / name).read_bytes()).hexdigest() for name in RESOURCES}
    result = {"schema": "flow_v3_n5_feature_extractor_throwaway.v1",
              "code_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
              "feature_extractor_commit": git("rev-parse", "HEAD"),
              "feature_extractor_python": py.replace(str(Path.home()), "~"),
              "egg_info_rc": egg.returncode, "resource_sha256": digests, "pg_db": a.pg_db, "runs": {}}
    actor_key = "throwaway-actor-" + os.urandom(8).hex()
    lake_token = "throwaway-lake-" + os.urandom(8).hex()
    salt = "throwaway-salt"
    key_file = work / "actor.key"
    key_file.write_text(actor_key + "\n")
    key_file.chmod(0o600)
    olap_port, gov_port = p03.port(), p03.port()
    gov_url = f"http://127.0.0.1:{gov_port}"
    (work / "olap.json").write_text(json.dumps({
        "pipeline_plugin": "default_pipeline", "web_plugin": "default_web", "query_plugin": "sql_query",
        "web_host": "127.0.0.1", "web_port": olap_port, "holdout_start": "2025-01-01",
        "lake_id": "olap_cube"}, indent=1))
    (work / "governance.json").write_text(json.dumps({
        "pipeline_plugin": "default_pipeline", "web_plugin": "default_web", "access_plugin": "default_access",
        "accounting_plugin": "default_accounting", "role_plugin": "default_role", "web_host": "127.0.0.1",
        "web_port": gov_port, "accounting_db": str(work / "accounting.sqlite"),
        "spool_dir": str(work / "gov-spool"), "cuts_dir": str(work / "gov-cuts"), "save_config": None,
        "password_salt": salt, "secret_key": "throwaway",
        "principals": {"feature-extractor": {
            "kind": "service", "role": "service",
            "api_key_hash": hashlib.sha256(f"{salt}:{actor_key}".encode()).hexdigest()}},
        "policies": [
            {"principal": "feature-extractor", "lake": "feature_extractor_examples",
             "verbs": ["discover", "coverage", "read", "download"], "deny_from": "2025-01-01"},
            {"principal": "feature-extractor", "lake": "olap_cube",
             "verbs": ["discover", "query", "write_terminal"], "deny_from": "2025-01-01"}],
        "lakes": [
            {"plugin": "files_lake", "lake_id": "feature_extractor_examples",
             "title": "feature-extractor phase_3 examples", "kind": "lake", "root_path": str(fixtures),
             "include_globs": ["normalized_d*.csv"], "holdout_start": "2025-01-01",
             "cuts_dir": str(work / "lake-cuts"), "spool_dir": str(work / "lake-spool"),
             "time_columns": {name: "DATE_TIME" for name in RESOURCES},
             "resource_contracts": {name: CONTRACT for name in RESOURCES}},
            {"plugin": "http_lake", "lake_id": "olap_cube",
             "base_url": f"http://127.0.0.1:{olap_port}", "holdout_start": "2025-01-01"}]}, indent=1))

    base_cfg = json.loads((fext / "examples/config/phase_4_2/phase_4_2_small.json").read_text())
    base_cfg.update({
        "epochs": 1, "kl_anneal_epochs": 1, "start_from_epoch": 0, "quiet_mode": True,
        "x_train_file": str(fixtures / "normalized_d1.csv"), "y_train_file": str(fixtures / "normalized_d1.csv"),
        "x_validation_file": str(fixtures / "normalized_d2.csv"),
        "y_validation_file": str(fixtures / "normalized_d2.csv"),
        "x_test_file": str(fixtures / "normalized_d3.csv"), "y_test_file": str(fixtures / "normalized_d3.csv"),
        "save_log": "./debug_out.json", "save_config": "./config_out.json"})
    cfg_dir = work / "configs"
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
               "--gov-url", gov_url, "--api-key-file", str(key_file), "--lake", "feature_extractor_examples",
               "--lake-root", str(fixtures), "--metrics-lake", "olap_cube", "--out-dir", str(out_dir),
               "--cache-dir", str(cache), "--outbox-dir", str(outbox), *extra]
        with open(log, "ab") as handle:
            rc = subprocess.run(cmd, cwd=fext,
                                env=dict(os.environ, DATA_GOV_CHECKOUT=str(data_gov),
                                         CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(fext)),
                                stdout=handle, stderr=subprocess.STDOUT).returncode
        r = p03.receipt(out_dir)
        camp = r.get("campaign_sha256")
        q = lambda sql, args=None: p03.pg(a.pg_db, sql, args) if camp else []  # noqa: E731
        rows = q("SELECT status, reason, costs_json FROM public.gov_terminal WHERE campaign_sha256=%s", (camp,))
        metrics = q("SELECT m.metric, m.value FROM public.gov_terminal_metric m "
                    "JOIN public.gov_terminal t USING (terminal_sha256) "
                    "WHERE t.campaign_sha256=%s ORDER BY 1", (camp,))
        datasets = q("SELECT role, resource_id, sha256, verification_state, availability_contract_sha256 "
                     "FROM public.gov_terminal_dataset d JOIN public.gov_terminal t USING (terminal_sha256) "
                     "WHERE t.campaign_sha256=%s ORDER BY role", (camp,))
        artifacts = q("SELECT role, sha256, bytes FROM public.gov_terminal_artifact x "
                      "JOIN public.gov_terminal t USING (terminal_sha256) "
                      "WHERE t.campaign_sha256=%s ORDER BY role", (camp,))
        rec = p03.request(f"{gov_url}/api/v2/campaigns/{camp}/reconcile", token=actor_key,
                          headers={"X-Campaign-SHA256": camp})[1] if camp else {}
        produced = sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else []
        return {"exit_code": rc, "status": r.get("status"), "reason": r.get("reason"),
                "campaign_sha256": camp, "terminal_pending": r.get("terminal_pending"),
                "inputs": r.get("inputs"), "cube_rows": rows, "metrics": metrics, "datasets": datasets,
                "artifacts": artifacts, "reconcile": rec, "produced_files": produced,
                "log_tail": Path(log).read_text(errors="replace")[-1500:] if rc else ""}

    try:
        procs["olap"] = p03.start(olap, work / "olap.json", olap_env, work / "olap.log")
        p03.wait_health(f"http://127.0.0.1:{olap_port}/healthz", [procs["olap"]])
        procs["gov"] = p03.start(data_gov, work / "governance.json",
                                 {"DATA_GOV_LAKE_TOKEN": lake_token}, work / "governance.log")
        p03.wait_health(f"{gov_url}/healthz", list(procs.values()))
        exact = lambda rec: (rec.get("missing_units") == [] and rec.get("accounting_only") == []  # noqa: E731
                             and rec.get("lake_only") == [])
        runs = result["runs"]
        out1 = work / "run1"
        r1 = governed("fv3-fext-a", config("a"), out1, work / "run1.log")
        measured = {m[0] for m in r1["metrics"]}
        r1["ok"] = (r1["exit_code"] == 0 and r1["status"] == "COMPLETED"
                    and [x[0] for x in r1["cube_rows"]] == ["COMPLETED"]
                    and METRICS <= measured
                    and len(r1["datasets"]) == 6
                    # six keys over three distinct resources: three are transferred and the
                    # repeats are served from the governed cache, which is the protocol working
                    and {d[3] for d in r1["datasets"]} <= {"VERIFIED_TRANSFER", "VERIFIED_CACHE"}
                    and sum(d[3] == "VERIFIED_TRANSFER" for d in r1["datasets"]) == 3
                    and {d[2] for d in r1["datasets"]} == set(digests.values())
                    and all(len(d[4] or "") == 64 for d in r1["datasets"])
                    and {x[0] for x in r1["artifacts"]} >= {"encoder", "decoder"}
                    and exact(r1["reconcile"]))
        runs["1_completed_six_governed_inputs"] = r1
        r2 = governed("fv3-fext-stale", config("a"), out1, work / "run2.log")
        r2["ok"] = (r2["exit_code"] != 0 and r2["status"] == "REFUSED"
                    and [x[0] for x in r2["cube_rows"]] == ["REFUSED"]
                    and "inputs" not in p03.receipt(out1))
        runs["2_refused_stale_outputs"] = r2
        out3 = work / "run3"
        r3 = governed("fv3-fext-failed", config("bogus", encoder_plugin="no_such_plugin_fv3"),
                      out3, work / "run3.log")
        costs = json.loads(r3["cube_rows"][0][2]) if r3["cube_rows"] else {}
        r3["ok"] = (r3["exit_code"] != 0 and r3["status"] == "FAILED"
                    and [x[0] for x in r3["cube_rows"]] == ["FAILED"]
                    and costs.get("wall_seconds", -1) >= 0 and len(r3["datasets"]) == 6
                    and r3["metrics"] == [])
        runs["3_failed_bogus_encoder"] = r3
        before = p03.gov_counts(a.pg_db)
        flush = subprocess.run([sys.executable, str(data_gov / "tools/governed_exec.py"), "--flush",
                                "--gov-url", gov_url, "--api-key-file", str(key_file),
                                "--outbox-dir", str(outbox)], cwd=fext, capture_output=True, text=True)
        after = p03.gov_counts(a.pg_db)
        runs["4_retry_sends_nothing"] = {"flush_stdout": flush.stdout.strip(), "rc": flush.returncode,
                                         "before": before, "after": after,
                                         "ok": flush.returncode == 0 and '"sent": 0' in flush.stdout
                                         and before == after}
        result["throwaway_cube_counts_final"] = after
        result["ok"] = all(v["ok"] for v in runs.values())
    finally:
        for p in procs.values():
            p03.stop(p)
        time.sleep(1)
        p03.pg_admin("predictor_olap", f'DROP DATABASE IF EXISTS "{a.pg_db}"')
        result["pg_db_dropped"] = a.pg_db not in {r[0] for r in p03.pg("predictor_olap",
                                                                      "SELECT datname FROM pg_database")}
        shutil.rmtree(cache, ignore_errors=True)
    sys.stdout.write(json.dumps(result, indent=1, default=str).replace(str(Path.home()), "~") + "\n")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
