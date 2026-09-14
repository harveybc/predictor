"""N5: a real pending terminal, its recovery, and the classification rules.

Musashi's finding 6: flushing an empty outbox proves that the flush wrote nothing, not that
recovering a pending terminal is idempotent. This creates the pending terminal for real — by
taking the terminal destination down while a run finishes — then recovers it through the
consumer's own wrapper and checks that the cube holds exactly one row, that resending the
same terminal adds none, and that a second download of the same input leaves a cache receipt.

Disposable throughout: the stack raised by tools/disposable_route_stack.py.
"""
from __future__ import annotations

import json, os, sqlite3, subprocess, sys, time, urllib.request
from pathlib import Path

REPO = Path("~/Documents/GitHub/predictor")
GITHUB = REPO.parent
#: a campaign key is an identity; each attempt of this test carries its own
ATTEMPT = ["1"]


def cube_rows(cube, campaign=None):
    with sqlite3.connect(f"file:{cube}?mode=ro", uri=True) as conn:
        if campaign:
            return conn.execute("SELECT terminal_sha256, status FROM gov_terminal "
                                "WHERE campaign_sha256=?", (campaign,)).fetchall()
        return conn.execute("SELECT count(*) FROM gov_terminal").fetchone()[0]


def run(consumer, label, stack, work, config, out_dir, extra=()):
    checkout = REPO if consumer == "predictor" else GITHUB / consumer
    venv = Path.home() / ".venvs" / consumer / "bin" / "python"
    command = [str(venv) if venv.is_file() else sys.executable,
               str(checkout / "tools" / "governed_run.py"),
               "--load_config", str(config), "--experiment-key", f"n5-{consumer}-{label}-{ATTEMPT[0]}",
               "--gov-url", stack["gov_url"], "--api-key-file", stack["key_file"],
               "--lake", stack["lake"], "--lake-root", stack["fixtures"],
               "--metrics-lake", stack["metrics_lake"], "--out-dir", str(out_dir),
               "--cache-dir", str(work / "cache"), "--outbox-dir", str(work / "outbox"),
               "--classification", "NON_GOVERNING", *extra]
    log = work / f"{label}.log"
    with open(log, "ab") as handle:
        code = subprocess.run(command, cwd=str(checkout), stdout=handle, stderr=subprocess.STDOUT,
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(checkout),
                                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                       DATA_GOV_CHECKOUT=str(GITHUB / "data-gov"))).returncode
    receipt = out_dir / "GOVERNED_RUN.json"
    state = json.loads(receipt.read_text()) if receipt.is_file() else {}
    return code, state, log


def outbox_status(work, stack):
    tool = GITHUB / "data-gov" / "tools" / "governed_exec.py"
    out = subprocess.run([sys.executable, str(tool), "--status", "--gov-url", stack["gov_url"],
                          "--api-key-file", stack["key_file"], "--outbox-dir", str(work / "outbox")],
                         capture_output=True, text=True)
    try:
        return json.loads(out.stdout or "{}")
    except ValueError:
        return {"stdout": out.stdout[-300:]}


def flush(work, stack):
    tool = GITHUB / "data-gov" / "tools" / "governed_exec.py"
    out = subprocess.run([sys.executable, str(tool), "--flush", "--gov-url", stack["gov_url"],
                          "--api-key-file", stack["key_file"], "--outbox-dir", str(work / "outbox")],
                         capture_output=True, text=True)
    try:
        return json.loads(out.stdout or "{}")
    except ValueError:
        return {"stdout": out.stdout[-300:]}


def main():
    stack = json.loads(Path(sys.argv[1]).read_text())
    ATTEMPT[0] = sys.argv[4] if len(sys.argv) > 4 else "1"
    work = Path(sys.argv[2]); work.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(stack["fixtures"]) / "MANIFEST.json").read_text())
    sys.path.insert(0, str(REPO / "tools"))
    from verify_consumer_adoption import fixture_config

    config = work / "preprocessor.json"
    config.write_text(json.dumps(fixture_config("preprocessor", "success", manifest,
                                                Path(stack["fixtures"])), indent=1))
    report = {"schema": "n5_real_retry.v1", "stack": {k: stack[k] for k in ("gov_url", "lake")}}

    # 1. the destination must fall *during* the run: taking it down first only makes the
    #    wrapper refuse before starting, which proves nothing about a pending terminal.
    #    A watcher kills the warehouse as soon as the pipeline has written its output.
    import threading

    # the moment the governed input lands in the cache, the delivery is done and the
    # terminal has not been sent yet: that is the window where a real outage strands it
    cache = work / "cache"

    def drop_destination():
        for _ in range(2400):
            if any(cache.rglob("*.csv")):
                os.kill(stack["warehouse_pid"], 15)
                return
            time.sleep(0.05)

    watcher = threading.Thread(target=drop_destination, daemon=True)
    watcher.start()
    code, state, _log = run("preprocessor", "pending", stack, work, config, work / "run")
    watcher.join(timeout=5)
    report["with_destination_down"] = {"exit_code": code, "status": state.get("status"),
                                       "terminal_pending": state.get("terminal_pending"),
                                       "campaign_sha256": state.get("campaign_sha256")}
    report["outbox_after_outage"] = outbox_status(work, stack)

    # 2. bring it back and recover through the wrapper's own outbox
    handle = open(work / "warehouse.log", "ab")
    subprocess.Popen(stack["warehouse_command"], cwd=stack["work"], stdout=handle,
                     stderr=subprocess.STDOUT, start_new_session=True,
                     env=dict(os.environ, DATA_GOV_LAKE_TOKEN=stack["lake_token"],
                              PYTHONUNBUFFERED="1"))
    for _ in range(60):
        try:
            with urllib.request.urlopen(stack["warehouse_url"] + "/healthz", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.5)
    report["recovery_flush"] = flush(work, stack)
    campaign = report["with_destination_down"]["campaign_sha256"]
    report["cube_rows_after_recovery"] = cube_rows(stack["cube"], campaign)

    # 3. resending the same terminal adds no row
    report["second_flush"] = flush(work, stack)
    report["cube_rows_after_second_flush"] = cube_rows(stack["cube"], campaign)

    # 4. a second run over the same input leaves a cache receipt
    code2, state2, _ = run("preprocessor", "cached", stack, work, config, work / "run_cached")
    report["second_run"] = {"exit_code": code2, "status": state2.get("status"),
                            "inputs": [{k: i.get(k) for k in ("role", "resource", "cached",
                                                              "verification_state")}
                                       for i in (state2.get("inputs") or [])]}
    report["ok"] = bool(
        report["with_destination_down"].get("terminal_pending")
        and report["recovery_flush"].get("sent") == 1
        and len(report["cube_rows_after_recovery"]) == 1
        and report["second_flush"].get("sent") == 0
        and report["cube_rows_after_second_flush"] == report["cube_rows_after_recovery"]
        and any(i.get("cached") for i in report["second_run"]["inputs"]))
    Path(sys.argv[3]).write_text(json.dumps(report, indent=1, default=str).replace(str(Path.home()), "~") + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "stack"}, indent=1, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
