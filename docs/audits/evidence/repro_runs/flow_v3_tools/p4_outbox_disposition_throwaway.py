#!/usr/bin/env python3
"""GOV-N4 on the throwaway stack: pending terminals are classified, adjudicated and
recovered without deleting evidence, and unaffected units are not held hostage.

Disposable services as in p03 (OLAP lake on a throwaway PostgreSQL database created and
dropped here; data-gov with the deployed predictor_examples contracts). Steps:

  1. governed run A completes (generation 1 accepted).
  2. a manufactured envelope of unit A, generation 2, whose metric name carries a space
     -> data-gov answers 400 -> class REFUSED_BY_SERVER, visible in --status.
  3. a wrong API key -> 401 -> class CONFIGURATION; the right key recovers it.
  4. supersede: corrected terminal (metric key fixed) sent as generation 3, same outcome,
     same deliveries -> accepted; the refused envelope is adjudicated SUPERSEDED, moved
     unchanged; replaying the successor answers already_stored; counts do not grow.
  5. a second manufactured envelope (generation 4, bad metric) -> adjudicated INVALID_ENVELOPE
     by explicit decision; its bytes remain under adjudicated/.
  6. isolation: governed run B while the OLAP lake is down -> pending TRANSIENT; lake back;
     flush sends B; A's adjudications stay; reconciliation of both campaigns is exact.
Output: JSON on stdout; local roots printed as ~.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
P03 = importlib.util.spec_from_file_location("p03_helpers", HERE.with_name("p03_throwaway_governed_runs.py"))
p03 = importlib.util.module_from_spec(P03)
P03.loader.exec_module(p03)


def load_governed_run(predictor: Path):
    spec = importlib.util.spec_from_file_location("governed_run_n4", predictor / "tools" / "governed_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["governed_run_n4"] = module
    spec.loader.exec_module(module)
    return module


def flush(predictor, gov_url, key_file, outbox, *extra):
    proc = subprocess.run([sys.executable, "tools/flush_governed_terminals.py", "--gov-url", gov_url,
                           "--api-key-file", str(key_file), "--outbox-dir", str(outbox), *extra],
                          cwd=predictor, capture_output=True, text=True)
    try:
        body = json.loads(proc.stdout)
    except ValueError:
        body = {"stdout": proc.stdout.strip()}
    return {"rc": proc.returncode, "body": body, "stderr": proc.stderr.strip()[-300:]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictor-checkout", required=True, type=Path)
    ap.add_argument("--data-gov-checkout", required=True, type=Path)
    ap.add_argument("--olap-lake-checkout", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--pg-db", required=True)
    a = ap.parse_args(argv)
    predictor, data_gov, olap, work = (a.predictor_checkout.resolve(), a.data_gov_checkout.resolve(),
                                       a.olap_lake_checkout.resolve(), a.work_dir.resolve())
    if work.exists():
        raise SystemExit(f"work dir exists: {work}")
    if not a.pg_db.startswith("data_gov_v3_throwaway_"):
        raise SystemExit("refusing: throwaway database must be named data_gov_v3_throwaway_*")
    work.mkdir(parents=True)
    GR = load_governed_run(predictor)
    result = {"schema": "flow_v3_n4_outbox_disposition_throwaway.v1",
              "code_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
              "predictor_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=predictor, capture_output=True, text=True).stdout.strip(),
              "predictor_clean": subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=predictor, capture_output=True, text=True).stdout.strip() == "",
              "pg_db": a.pg_db, "steps": {}}
    actor_key, lake_token, salt = "throwaway-actor-" + os.urandom(8).hex(), "throwaway-lake-" + os.urandom(8).hex(), "throwaway-salt"
    key_file, wrong_key_file = work / "actor.key", work / "wrong.key"
    key_file.write_text(actor_key + "\n"); key_file.chmod(0o600)
    wrong_key_file.write_text("not-the-key\n"); wrong_key_file.chmod(0o600)
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
        "principals": {"predictor": {"kind": "service", "role": "service",
                                     "api_key_hash": hashlib.sha256(f"{salt}:{actor_key}".encode()).hexdigest()}},
        "policies": [
            {"principal": "predictor", "lake": "predictor_examples", "verbs": ["discover", "coverage", "read", "download"], "deny_from": "2025-01-01"},
            {"principal": "predictor", "lake": "olap_cube", "verbs": ["discover", "query", "write_terminal"], "deny_from": "2025-01-01"}],
        "lakes": [
            {"plugin": "files_lake", "lake_id": "predictor_examples", "title": "predictor examples", "kind": "files_inventory",
             "root_path": str(predictor / "examples/data_downsampled"), "include_globs": ["**/*.csv"], "time_column": "DATE_TIME",
             "holdout_start": "2025-01-01", "cuts_dir": str(work / "lake-cuts"), "spool_dir": str(work / "lake-spool"),
             "resource_contracts": {r: p03.CONTRACT for r in p03.RESOURCES}},
            {"plugin": "http_lake", "lake_id": "olap_cube", "base_url": f"http://127.0.0.1:{olap_port}", "holdout_start": "2025-01-01"}]},
        indent=1))
    p03.pg_admin("predictor_olap", f'CREATE DATABASE "{a.pg_db}"')
    olap_env = {"DATA_GOV_LAKE_TOKEN": lake_token, "PGDATABASE": a.pg_db}
    procs = {}
    outbox_dir, cache = work / "outbox", work / "cache"
    steps = result["steps"]

    def gens(campaign):
        return p03.pg(a.pg_db, "SELECT generation, status, terminal_sha256 FROM public.gov_terminal WHERE campaign_sha256=%s ORDER BY generation", (campaign,))

    def reconcile(campaign):
        return p03.request(f"{gov_url}/api/v2/campaigns/{campaign}/reconcile", token=actor_key, headers={"X-Campaign-SHA256": campaign})[1]

    try:
        procs["olap"] = p03.start(olap, work / "olap.json", olap_env, work / "olap.log")
        p03.wait_health(f"http://127.0.0.1:{olap_port}/healthz", [procs["olap"]])
        procs["gov"] = p03.start(data_gov, work / "governance.json", {"DATA_GOV_LAKE_TOKEN": lake_token}, work / "governance.log")
        p03.wait_health(f"{gov_url}/healthz", list(procs.values()))

        # 1. run A completes
        out_a = work / "run_a"
        pa = p03.governed_run(predictor, "fv3-n4-a", out_a, gov_url, key_file, cache, outbox_dir, p03.EXTRA, work / "run_a.log")
        rc = pa.wait(); pa._log.close()
        ra = p03.receipt(out_a)
        camp_a = ra["campaign_sha256"]
        steps["1_run_a"] = {"exit_code": rc, "status": ra["status"], "generations": gens(camp_a), "ok": rc == 0 and ra["status"] == "COMPLETED"}
        sent_a = json.loads(sorted((outbox_dir / "sent").glob("*.json"))[0].read_text())

        # 2. manufactured generation 2 with a metric name the contract refuses
        outbox = GR.TerminalOutbox(outbox_dir)
        bad = json.loads(json.dumps(sent_a))
        bad["terminal"]["generation"] = 2
        bad["terminal"]["metrics"][0]["metric"] = "MAE (custom)"  # a key the contract refuses, unique once fixed
        bad_item = outbox.put(bad)
        f2 = flush(predictor, gov_url, key_file, outbox_dir)
        status2 = flush(predictor, gov_url, key_file, outbox_dir, "--status")["body"]
        steps["2_refused_by_server"] = {"flush": f2, "status": status2,
                                        "ok": f2["rc"] == 1 and status2["awaiting_adjudication"] == 1
                                        and status2["pending"][0]["class"] == "REFUSED_BY_SERVER"
                                        and "http 400" in status2["pending"][0]["last_error"] and bad_item.path.is_file()}

        # 3. configuration error: wrong key -> 401; the right key recovers a transient-free case
        good = json.loads(json.dumps(sent_a))
        good["terminal"]["generation"] = 5
        good["terminal"]["tags"] = {**good["terminal"]["tags"], "purpose": "configuration-class-probe"}
        good_item = outbox.put(good)
        f3 = flush(predictor, gov_url, wrong_key_file, outbox_dir)
        status3 = flush(predictor, gov_url, key_file, outbox_dir, "--status")["body"]
        cls3 = {p["file"]: p["class"] for p in status3["pending"]}
        f3b = flush(predictor, gov_url, key_file, outbox_dir)
        status3b = flush(predictor, gov_url, key_file, outbox_dir, "--status")["body"]
        # the class is that of the LAST attempt: under the wrong key every pending envelope is a
        # configuration failure; with the right key the sendable one goes and the refused one
        # shows its server refusal again
        steps["3_configuration_then_recovery"] = {
            "flush_wrong_key": f3, "classes_after_wrong_key": cls3, "flush_right_key": f3b,
            "pending_after": [p["file"][:12] + ":" + p["class"] for p in status3b["pending"]],
            "ok": set(cls3.values()) == {"CONFIGURATION"} and len(cls3) == 2
            and f3b["body"].get("sent") == 1 and [p["class"] for p in status3b["pending"]] == ["REFUSED_BY_SERVER"]
            and status3b["pending"][0]["file"] == bad_item.path.name}

        # 4. supersede the refused envelope with a corrected terminal (generation 3)
        corrected = json.loads(json.dumps(bad["terminal"]))
        corrected["metrics"][0]["metric"] = "MAE_custom"
        corrected_path = work / "corrected_terminal.json"
        corrected_path.write_text(json.dumps(corrected, indent=1))
        before = p03.gov_counts(a.pg_db)
        f4 = flush(predictor, gov_url, key_file, outbox_dir, "--supersede", bad_item.path.name, "--terminal", str(corrected_path),
                   "--reason", "metric name carried a space; the producer canonicalises keys since 08a4c04")
        after = p03.gov_counts(a.pg_db)
        record4 = f4["body"]
        successor_sha = record4.get("successor_terminal_sha256")
        replay_env = None
        for path in (outbox_dir / "sent").glob("*.json"):
            env = json.loads(path.read_text())
            if env["terminal"].get("generation") == 3:
                replay_env = env
        replay = p03.request(f"{gov_url}/api/v2/campaigns/{camp_a}/units/{replay_env['unit_id']}/terminal", token=actor_key,
                             headers={"X-Campaign-SHA256": camp_a, "X-Unit-ID": replay_env["unit_id"]}, body=replay_env["terminal"]) if replay_env else (None, {})
        after_replay = p03.gov_counts(a.pg_db)
        adjudicated_dir = outbox_dir / "adjudicated"
        moved = adjudicated_dir / bad_item.path.name
        steps["4_superseded"] = {
            "flush": f4, "generations": gens(camp_a), "counts_before": before, "counts_after": after, "after_replay": after_replay,
            "replay_http": replay[0], "replay_body": replay[1], "reconcile": reconcile(camp_a),
            "moved_unchanged": moved.is_file() and hashlib.sha256(moved.read_bytes()).hexdigest() == bad_item.path.name[:-5],
            "ok": f4["rc"] == 0 and record4.get("decision") == "SUPERSEDED" and record4.get("successor_generation") == 3
            and [g[0] for g in gens(camp_a)] == [1, 3, 5] and after["gov_terminal"] == before["gov_terminal"] + 1
            and replay[0] == 200 and replay[1].get("already_stored") is True and after_replay == after
            and successor_sha in {g[2] for g in gens(camp_a)} and not bad_item.path.exists()
            and moved.is_file() and hashlib.sha256(moved.read_bytes()).hexdigest() == bad_item.path.name[:-5]
            and reconcile(camp_a) == {"campaign_sha256": camp_a, "missing_units": [], "accounting_only": [], "lake_only": []}}

        # 5. an envelope closed by explicit decision
        bad2 = json.loads(json.dumps(sent_a))
        bad2["terminal"]["generation"] = 4
        bad2["terminal"]["metrics"][0]["metric"] = "R2 (adj)"
        bad2_item = outbox.put(bad2)
        f5 = flush(predictor, gov_url, key_file, outbox_dir)
        d5 = flush(predictor, gov_url, key_file, outbox_dir, "--dispose", bad2_item.path.name, "--reason",
                   "manufactured probe; no successor: the run's generation 1 terminal stands")
        status5 = flush(predictor, gov_url, key_file, outbox_dir, "--status")["body"]
        steps["5_invalid_envelope_disposed"] = {
            "flush": f5, "dispose": d5, "status": status5,
            "ok": f5["rc"] == 1 and d5["rc"] == 0 and d5["body"].get("decision") == "INVALID_ENVELOPE"
            and status5["pending"] == [] and sorted(x["decision"] for x in status5["adjudicated"]) == ["INVALID_ENVELOPE", "SUPERSEDED"]
            and (adjudicated_dir / bad2_item.path.name).is_file() and [g[0] for g in gens(camp_a)] == [1, 3, 5]}

        # 6. isolation: run B while the terminal lake is down, then recovery; A's adjudications stay
        out_b = work / "run_b"
        pb = p03.governed_run(predictor, "fv3-n4-b", out_b, gov_url, key_file, cache, outbox_dir, p03.EXTRA + ["--epochs", "6"], work / "run_b.log")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline and "command" not in p03.receipt(out_b) and pb.poll() is None:
            time.sleep(0.2)
        p03.stop(procs["olap"], signal.SIGKILL)
        rcb = pb.wait(); pb._log.close()
        rb = p03.receipt(out_b)
        status6 = flush(predictor, gov_url, key_file, outbox_dir, "--status")["body"]
        procs["olap"] = p03.start(olap, work / "olap.json", olap_env, work / "olap.log")
        p03.wait_health(f"http://127.0.0.1:{olap_port}/healthz", [procs["olap"]])
        f6 = flush(predictor, gov_url, key_file, outbox_dir)
        status6b = flush(predictor, gov_url, key_file, outbox_dir, "--status")["body"]
        camp_b = rb.get("campaign_sha256")
        steps["6_isolation_and_recovery"] = {
            "run_b_exit": rcb, "run_b_status": rb.get("status"), "pending_during_outage": [p["class"] for p in status6["pending"]],
            "flush_after_lake_back": f6, "status_after": {k: status6b[k] for k in ("recoverable", "awaiting_adjudication", "unresolved", "sent")},
            "generations_b": gens(camp_b), "reconcile_a": reconcile(camp_a), "reconcile_b": reconcile(camp_b),
            "ok": rcb != 0 and rb.get("terminal_pending") is True and [p["class"] for p in status6["pending"]] == ["TRANSIENT"]
            and f6["body"].get("sent") == 1 and status6b["pending"] == [] and len(status6b["adjudicated"]) == 2
            and [g[1] for g in gens(camp_b)] == ["COMPLETED"]
            and reconcile(camp_b)["missing_units"] == [] and reconcile(camp_a)["missing_units"] == []}
        result["throwaway_cube_counts_final"] = p03.gov_counts(a.pg_db)
        result["ok"] = all(s["ok"] for s in steps.values())
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
