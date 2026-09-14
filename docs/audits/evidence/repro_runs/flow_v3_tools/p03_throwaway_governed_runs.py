#!/usr/bin/env python3
"""Flow v3 adoption, P0.3 on a throwaway stack: governed CPU micro-runs of predictor
through data-gov, terminal on a throwaway PostgreSQL database, deliberate outage of the
terminal destination, outbox retention and exactly-once retry.

Services started by this script (all disposable, ephemeral ports, own state dirs):
  * OLAP lake (predictor/olap/lake) on PostgreSQL database <pg-db> created here and
    dropped at the end; the real cube is never touched.
  * data-gov with an in-process files_lake `predictor_examples` that serves the
    predictor checkout's examples/data_downsampled with the SAME resource_contracts
    entry as the deployed config, and one service principal.

Runs (predictor `tools/governed_run.py`, CPU, from the clean checkout given):
  1. COMPLETED  fresh output dir                     -> terminal + 6 verified deliveries
  2. REFUSED    same output dir as run 1             -> stale outputs refused, no download
  3. FAILED     bogus predictor plugin               -> predictor exit != 0
  4. OUTAGE     OLAP lake killed while training      -> terminal pending in the outbox,
                lake restarted, flush loads it exactly once, replay is idempotent
Output: JSON on stdout. Nothing here restarts or reads the production services.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CONTRACT = {"event_time_column": "DATE_TIME", "available_time_column": "DATE_TIME",
            "timezone": "NAIVE_WALL_CLOCK", "time_unit": None, "frequency": "4h"}
RESOURCES = ["phase_1/normalized_d4.csv", "phase_1/normalized_d5.csv", "phase_1/normalized_d6.csv"]
TOY_CONFIG = "examples/config/phase_1_daily/phase_1_ann_1575_1d_config.json"
EXTRA = ["--epochs", "2", "--max_steps_train", "300", "--max_steps_test", "300", "--mc_samples", "2",
         "--execution_purpose", "ARCHIVAL_REPLAY_NON_AUTHORITATIVE"]


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(url, *, token, body=None, headers=None, method=None):
    h = {"Authorization": f"Bearer {token}", **(headers or {})}
    data = None
    if body is not None:
        data = json.dumps(body, allow_nan=False).encode("ascii")
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def wait_health(url, procs, deadline=60):
    until = time.monotonic() + deadline
    while time.monotonic() < until:
        for p in procs:
            if p.poll() is not None:
                raise RuntimeError(f"service exited early: {p.args[:3]} rc={p.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"not ready: {url}")


def start(checkout, config, env, log):
    handle = open(log, "ab")
    e = dict(os.environ, **env, PYTHONPATH=str(checkout))
    p = subprocess.Popen([sys.executable, "app/main.py", "--load_config", str(config)],
                         cwd=checkout, env=e, stdout=handle, stderr=subprocess.STDOUT)
    p._log = handle
    return p


def stop(p, sig=signal.SIGTERM):
    if p.poll() is None:
        p.send_signal(sig)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=10)
    p._log.close()


def pg(dbname, sql, params=None):
    import psycopg2
    conn = psycopg2.connect(host=os.environ.get("PGHOST", "127.0.0.1"), port=os.environ.get("PGPORT", "5432"),
                            user=os.environ.get("PGUSER", "metabase"), password=os.environ["PGPASSWORD"], dbname=dbname)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchall() if cur.description else None
    finally:
        conn.close()


def pg_admin(dbname, sql):
    import psycopg2
    conn = psycopg2.connect(host=os.environ.get("PGHOST", "127.0.0.1"), port=os.environ.get("PGPORT", "5432"),
                            user=os.environ.get("PGUSER", "metabase"), password=os.environ["PGPASSWORD"], dbname=dbname)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
    finally:
        conn.close()


def gov_counts(dbname):
    out = {}
    for t in ("gov_terminal", "gov_terminal_metric", "gov_terminal_dataset", "gov_terminal_artifact"):
        out[t] = pg(dbname, f'SELECT count(*) FROM public."{t}"')[0][0]
    return out


def governed_run(predictor, key, out_dir, gov_url, key_file, cache, outbox, extra, log):
    cmd = [sys.executable, "tools/governed_run.py", "--load_config", TOY_CONFIG, "--experiment-key", key,
           "--gov-url", gov_url, "--api-key-file", str(key_file), "--lake", "predictor_examples",
           "--lake-root", "examples/data_downsampled", "--metrics-lake", "olap_cube", "--out-dir", str(out_dir),
           "--cache-dir", str(cache), "--outbox-dir", str(outbox), "--", *extra]
    handle = open(log, "ab")
    p = subprocess.Popen(cmd, cwd=predictor, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                         stdout=handle, stderr=subprocess.STDOUT)
    p._log = handle
    return p


def receipt(out_dir):
    path = Path(out_dir) / "GOVERNED_RUN.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictor-checkout", required=True, type=Path)
    ap.add_argument("--data-gov-checkout", required=True, type=Path)
    ap.add_argument("--olap-lake-checkout", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--pg-db", required=True, help="throwaway database name (created and dropped here)")
    a = ap.parse_args(argv)
    predictor, data_gov, olap = a.predictor_checkout.resolve(), a.data_gov_checkout.resolve(), a.olap_lake_checkout.resolve()
    work = a.work_dir.resolve()
    if work.exists():
        raise SystemExit(f"work dir exists: {work}")
    work.mkdir(parents=True)
    if not a.pg_db.startswith("data_gov_v3_throwaway_"):
        raise SystemExit("refusing: throwaway database must be named data_gov_v3_throwaway_*")
    result = {"schema": "flow_v3_p03_throwaway_governed_runs.v1", "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "predictor_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=predictor, capture_output=True, text=True).stdout.strip(),
              "predictor_clean": subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=predictor, capture_output=True, text=True).stdout.strip() == "",
              "data_gov_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=data_gov, capture_output=True, text=True).stdout.strip(),
              "pg_db": a.pg_db, "runs": {}}
    actor_key, lake_token, salt = "throwaway-actor-" + os.urandom(8).hex(), "throwaway-lake-" + os.urandom(8).hex(), "throwaway-salt"
    key_file = work / "actor.key"
    key_file.write_text(actor_key + "\n", encoding="ascii")
    key_file.chmod(0o600)
    olap_port, gov_port = port(), port()
    gov_url = f"http://127.0.0.1:{gov_port}"
    olap_cfg, gov_cfg = work / "olap.json", work / "governance.json"
    olap_cfg.write_text(json.dumps({
        "pipeline_plugin": "default_pipeline", "web_plugin": "default_web", "query_plugin": "sql_query",
        "web_host": "127.0.0.1", "web_port": olap_port, "holdout_start": "2025-01-01", "lake_id": "olap_cube"}, indent=1))
    gov_cfg.write_text(json.dumps({
        "pipeline_plugin": "default_pipeline", "web_plugin": "default_web", "access_plugin": "default_access",
        "accounting_plugin": "default_accounting", "role_plugin": "default_role", "web_host": "127.0.0.1",
        "web_port": gov_port, "accounting_db": str(work / "accounting.sqlite"), "spool_dir": str(work / "gov-spool"),
        "cuts_dir": str(work / "gov-cuts"), "save_config": None, "password_salt": salt, "secret_key": "throwaway",
        "principals": {"predictor": {"kind": "service", "role": "service",
                                     "api_key_hash": hashlib.sha256(f"{salt}:{actor_key}".encode()).hexdigest()}},
        "policies": [
            {"principal": "predictor", "lake": "predictor_examples", "verbs": ["discover", "coverage", "read", "download"], "deny_from": "2025-01-01"},
            {"principal": "predictor", "lake": "olap_cube", "verbs": ["discover", "query", "write_terminal"], "deny_from": "2025-01-01"}],
        "lakes": [
            {"plugin": "files_lake", "lake_id": "predictor_examples", "title": "predictor examples", "kind": "files_inventory",
             "root_path": str(predictor / "examples/data_downsampled"), "include_globs": ["**/*.csv"], "time_column": "DATE_TIME",
             "holdout_start": "2025-01-01", "cuts_dir": str(work / "lake-cuts"), "spool_dir": str(work / "lake-spool"),
             "resource_contracts": {r: CONTRACT for r in RESOURCES}},
            {"plugin": "http_lake", "lake_id": "olap_cube", "base_url": f"http://127.0.0.1:{olap_port}", "holdout_start": "2025-01-01"}]},
        indent=1))
    pg_admin("predictor_olap", f'CREATE DATABASE "{a.pg_db}"')
    olap_env = {"DATA_GOV_LAKE_TOKEN": lake_token, "PGDATABASE": a.pg_db}
    procs = {}
    try:
        procs["olap"] = start(olap, olap_cfg, olap_env, work / "olap.log")
        wait_health(f"http://127.0.0.1:{olap_port}/healthz", [procs["olap"]])
        procs["gov"] = start(data_gov, gov_cfg, {"DATA_GOV_LAKE_TOKEN": lake_token}, work / "governance.log")
        wait_health(f"{gov_url}/healthz", list(procs.values()))
        cache, outbox = work / "cache", work / "outbox"

        def finish(name, proc, out_dir, expect_status):
            rc = proc.wait()
            proc._log.close()
            r = receipt(out_dir)
            camp = r.get("campaign_sha256")
            rows = pg(a.pg_db, "SELECT status, reason FROM public.gov_terminal WHERE campaign_sha256=%s", (camp,)) if camp else []
            entry = {"exit_code": rc, "status": r.get("status"), "reason": r.get("reason"), "campaign_sha256": camp,
                     "terminal_pending": r.get("terminal_pending"), "cube_rows": rows,
                     "reconciliation": r.get("reconciliation"), "expected_status": expect_status,
                     "ok": r.get("status") == expect_status and [x[0] for x in rows] == [expect_status]}
            result["runs"][name] = entry
            return r, entry

        # 1. COMPLETED
        out1 = work / "run1"
        r, e1 = finish("1_completed", governed_run(predictor, "fv3-toy-completed", out1, gov_url, key_file, cache, outbox, EXTRA, work / "run1.log"), out1, "COMPLETED")
        ds = pg(a.pg_db, "SELECT role, resource_id, sha256, source_sha256, verification_state, availability_contract_sha256, delivery_kind, range_from, range_to FROM public.gov_terminal_dataset d JOIN public.gov_terminal t USING (terminal_sha256) WHERE t.campaign_sha256=%s ORDER BY role", (r.get("campaign_sha256"),))
        e1["terminal_datasets"] = [dict(zip(("role", "resource", "sha256", "source_sha256", "state", "contract", "kind", "from", "to"), row)) for row in ds]
        e1["metrics_rows"] = pg(a.pg_db, "SELECT count(*) FROM public.gov_terminal_metric m JOIN public.gov_terminal t USING (terminal_sha256) WHERE t.campaign_sha256=%s", (r.get("campaign_sha256"),))[0][0]
        e1["artifact_rows"] = pg(a.pg_db, "SELECT count(*) FROM public.gov_terminal_artifact x JOIN public.gov_terminal t USING (terminal_sha256) WHERE t.campaign_sha256=%s", (r.get("campaign_sha256"),))[0][0]
        e1["inputs"] = r.get("inputs")
        e1["code_identity"] = r.get("code_identity")
        e1["ok"] = e1["ok"] and len(ds) == 6 and all(row[4] in ("VERIFIED_TRANSFER", "VERIFIED_CACHE") for row in ds) and e1["metrics_rows"] > 0
        st, rec = request(f"{gov_url}/api/v2/campaigns/{r['campaign_sha256']}/reconcile", token=actor_key, headers={"X-Campaign-SHA256": r["campaign_sha256"]})
        e1["reconcile_http"] = rec

        # 2. REFUSED (stale outputs in the same directory)
        finish("2_refused_stale_outputs", governed_run(predictor, "fv3-toy-refused", out1, gov_url, key_file, cache, outbox, EXTRA, work / "run2.log"), out1, "REFUSED")
        result["runs"]["2_refused_stale_outputs"]["no_downloads_recorded"] = receipt(out1).get("inputs") is None or receipt(out1).get("campaign_sha256") != e1["campaign_sha256"]

        # 3. FAILED (predictor exits non-zero)
        out3 = work / "run3"
        finish("3_failed_predictor_exit", governed_run(predictor, "fv3-toy-failed", out3, gov_url, key_file, cache, outbox, EXTRA + ["--predictor_plugin", "no_such_plugin_fv3"], work / "run3.log"), out3, "FAILED")

        # 4. OUTAGE of the terminal destination while training
        out4 = work / "run4"
        p4 = governed_run(predictor, "fv3-toy-outage", out4, gov_url, key_file, cache, outbox, EXTRA + ["--epochs", "6"], work / "run4.log")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline and "command" not in receipt(out4) and p4.poll() is None:
            time.sleep(0.2)
        killed_at_stage = "training_started" if "command" in receipt(out4) else "not_started"
        stop(procs["olap"], signal.SIGKILL)
        rc4 = p4.wait()
        p4._log.close()
        r4 = receipt(out4)
        pending = sorted(p.name for p in (outbox / "pending").glob("*.json"))
        e4 = {"killed_at_stage": killed_at_stage, "exit_code": rc4, "status": r4.get("status"), "terminal_pending": r4.get("terminal_pending"),
              "outbox_pending_after_outage": pending, "campaign_sha256": r4.get("campaign_sha256")}
        procs["olap"] = start(olap, olap_cfg, olap_env, work / "olap.log")
        wait_health(f"http://127.0.0.1:{olap_port}/healthz", [procs["olap"]])
        before = gov_counts(a.pg_db)
        flush = subprocess.run([sys.executable, "tools/flush_governed_terminals.py", "--gov-url", gov_url, "--api-key-file", str(key_file), "--outbox-dir", str(outbox)],
                               cwd=predictor, capture_output=True, text=True)
        e4["flush_1"] = {"rc": flush.returncode, "stdout": flush.stdout.strip(), "stderr": flush.stderr.strip()[-300:]}
        after = gov_counts(a.pg_db)
        flush2 = subprocess.run([sys.executable, "tools/flush_governed_terminals.py", "--gov-url", gov_url, "--api-key-file", str(key_file), "--outbox-dir", str(outbox)],
                                cwd=predictor, capture_output=True, text=True)
        e4["flush_2"] = {"rc": flush2.returncode, "stdout": flush2.stdout.strip()}
        after2 = gov_counts(a.pg_db)
        rows4 = pg(a.pg_db, "SELECT status FROM public.gov_terminal WHERE campaign_sha256=%s", (r4.get("campaign_sha256"),))
        # replay the accepted envelope by hand: idempotent at the destination
        sent = sorted((outbox / "sent").glob("*.json"))
        replay = None
        for path in sent:
            env = json.loads(path.read_text())
            if env["campaign_sha256"] == r4.get("campaign_sha256"):
                st, body = request(f"{gov_url}/api/v2/campaigns/{env['campaign_sha256']}/units/{env['unit_id']}/terminal", token=actor_key,
                                   headers={"X-Campaign-SHA256": env["campaign_sha256"], "X-Unit-ID": env["unit_id"]}, body=env["terminal"])
                replay = {"http": st, "body": body}
        after3 = gov_counts(a.pg_db)
        st, rec4 = request(f"{gov_url}/api/v2/campaigns/{r4['campaign_sha256']}/reconcile", token=actor_key, headers={"X-Campaign-SHA256": r4["campaign_sha256"]})
        e4.update({"cube_counts_before_flush": before, "after_flush_1": after, "after_flush_2": after2, "after_replay": after3,
                   "cube_rows_for_campaign": rows4, "replay": replay, "reconcile_http": rec4,
                   "outbox_pending_after_flush": sorted(p.name for p in (outbox / "pending").glob("*.json")),
                   "outbox_sent": [p.name for p in sent]})
        e4["ok"] = (killed_at_stage == "training_started" and rc4 != 0 and r4.get("terminal_pending") is True and len(pending) == 1
                    and after["gov_terminal"] == before["gov_terminal"] + 1 and after2 == after and after3 == after
                    and [x[0] for x in rows4] == ["COMPLETED"] and replay and replay["http"] == 200 and replay["body"].get("already_stored") is True
                    and rec4.get("missing_units") == [] and rec4.get("accounting_only") == [] and rec4.get("lake_only") == []
                    and not e4["outbox_pending_after_flush"])
        result["runs"]["4_outage_outbox_exactly_once"] = e4
        result["throwaway_cube_counts_final"] = gov_counts(a.pg_db)
        result["states_in_cube"] = sorted({row[0] for row in pg(a.pg_db, "SELECT status FROM public.gov_terminal")})
        result["ok"] = all(v.get("ok") for v in result["runs"].values())
    finally:
        for p in procs.values():
            stop(p)
        time.sleep(1)
        pg_admin("predictor_olap", f'DROP DATABASE IF EXISTS "{a.pg_db}"')
        result["pg_db_dropped"] = a.pg_db not in {r[0] for r in pg("predictor_olap", "SELECT datname FROM pg_database")}
        shutil.rmtree(work / "cache", ignore_errors=True)
    json.dump(result, sys.stdout, indent=1, default=str)
    sys.stdout.write("\n")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
