#!/usr/bin/env python3
"""A3: adoption of the deployed stores, one consumer at a time, against the live services.

For each consumer this drives **its own** governed wrapper — the entry point the repository
really uses, not a client written here — through data-gov at :5055 to the deployed stores,
and records four things per consumer: a bounded success, a refusal of stale outputs, a
failure with its cost, and a retry that sends nothing. Every campaign is NON_GOVERNING: this
is transport and mechanics, and it grants nothing scientific.

What it does not do: train anything heavy, touch a financial resource without a contract,
restart a service, or write into a repository's data root.

usage:
  verify_consumer_adoption.py --api-key-file FILE --work DIR --out RECEIPT.json
      [--consumer preprocessor] [--consumer predictor] [--skip-success feature-eng]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GITHUB = REPO.parent
GOV_URL = "http://127.0.0.1:5055"
#: The lake data-gov serves the contracted sample resources from (a local files lake inside
#: data-gov). The deployed lake host serves `financial_files`, which has no contract yet, and
#: `governance_smoke`, the two-row synthetic fixture.
SAMPLE_LAKE = "predictor_examples"
SAMPLE_ROOT = REPO / "examples" / "data_downsampled"


def pg(sql: str, params=()):
    import psycopg2

    with psycopg2.connect(host=os.environ.get("PGHOST", "127.0.0.1"),
                          port=os.environ.get("PGPORT", "5432"),
                          dbname=os.environ.get("PGDATABASE", "predictor_olap"),
                          user=os.environ.get("PGUSER", "metabase"),
                          password=os.environ["PGPASSWORD"]) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def cube_counts() -> dict:
    tables = ("gov_terminal", "gov_terminal_metric", "gov_terminal_dataset", "gov_terminal_artifact")
    return {t: pg(f'SELECT count(*) FROM public."{t}"')[0][0] for t in tables}


def receipt_of(out_dir: Path) -> dict:
    path = out_dir / "GOVERNED_RUN.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def terminal_rows(campaign: str):
    if not campaign:
        return []
    return pg("SELECT status, reason, costs_json FROM public.gov_terminal WHERE campaign_sha256=%s",
              (campaign,))


def reconcile(campaign: str, key: str) -> dict:
    import urllib.request

    if not campaign:
        return {}
    request = urllib.request.Request(
        f"{GOV_URL}/api/v2/campaigns/{campaign}/reconcile",
        headers={"Authorization": f"Bearer {key}", "X-Campaign-SHA256": campaign})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def config_for(consumer: str, work: Path, case: str) -> Path:
    """A bounded configuration per consumer, written into the work directory."""
    directory = work / consumer / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{case}.json"
    if consumer == "preprocessor":
        template = json.loads((GITHUB / "preprocessor" / "examples" / "config_downsampled"
                               / "phase_1b.json").read_text(encoding="utf-8"))
        template.update({
            "input_file": str(SAMPLE_ROOT / "phase_1" / "normalized_d4.csv"),
            "plugin": "plugin_default" if case != "failure" else "no_such_plugin_a3",
            "dataset_prefix": "./base_", "target_prefix": "./normalized_",
            "normalization_config_a": "./norm_a.json", "normalization_config_b": "./norm_b.json",
            "output_file": "./preprocessed.csv", "save_log": "./debug_out.json",
            "save_config": "./config_out.json", "debug_file": "./debug_out_file.json",
            "quiet_mode": True, "trim_start_rows": 0,
            # the preprocessor consumes predictor's eligibility gate, which asks what a run
            # claims to be: this is a mechanical replay, not gated scientific evidence
            "execution_purpose": "ARCHIVAL_REPLAY_NON_AUTHORITATIVE",
        })
        body = template
    elif consumer == "predictor":
        base = json.loads((REPO / "examples" / "config" / "phase_1_daily"
                           / "phase_1_ann_1575_1d_config.json").read_text(encoding="utf-8"))
        base.update({"epochs": 1, "max_steps_train": 64, "max_steps_test": 64, "mc_samples": 2,
                     "early_patience": 1, "quiet_mode": True})
        if case == "failure":
            base["predictor_plugin"] = "no_such_plugin_a3"
        for key, value in list(base.items()):
            if isinstance(value, str) and value.startswith("examples/results/"):
                base[key] = "./" + Path(value).name
        base["save_log"] = "./debug_out.json"
        base["save_config"] = "./config_out.json"
        body = base
    else:
        raise SystemExit(f"no bounded configuration is defined for {consumer}")
    path.write_text(json.dumps(body, indent=1), encoding="utf-8")
    return path


def wrapper_of(consumer: str) -> tuple[Path, Path]:
    checkout = REPO if consumer == "predictor" else GITHUB / consumer
    return checkout, checkout / "tools" / "governed_run.py"


def python_of(consumer: str) -> str:
    """Each consumer runs in its own environment when it declares one."""
    venv = Path.home() / ".venvs" / consumer / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def run_case(consumer: str, case: str, work: Path, key_file: Path, key: str,
             out_dir: Path, config: Path, label: str = "") -> dict:
    checkout, wrapper = wrapper_of(consumer)
    # A campaign key is an identity: re-registering it with different content is refused by
    # the kernel, as it should be. Each attempt of this check carries its own label.
    experiment = f"a3-{consumer}-{case}" + (f"-{label}" if label else "")
    command = [python_of(consumer), str(wrapper), "--load_config", str(config),
               "--experiment-key", experiment, "--gov-url", GOV_URL,
               "--api-key-file", str(key_file), "--lake", SAMPLE_LAKE,
               "--lake-root", str(SAMPLE_ROOT), "--metrics-lake", "olap_cube",
               "--out-dir", str(out_dir), "--cache-dir", str(work / "cache"),
               "--outbox-dir", str(work / "outbox"), "--classification", "NON_GOVERNING"]
    log = work / consumer / f"{case}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with open(log, "ab") as handle:
        code = subprocess.run(command, cwd=str(checkout), stdout=handle, stderr=subprocess.STDOUT,
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES="",
                                       PYTHONPATH=str(checkout), OMP_NUM_THREADS="1",
                                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                       DATA_GOV_CHECKOUT=str(GITHUB / "data-gov"))).returncode
    state = receipt_of(out_dir)
    campaign = state.get("campaign_sha256")
    return {"case": case, "exit_code": code, "wall_seconds": round(time.monotonic() - started, 3),
            "status": state.get("status"), "reason": state.get("reason"),
            "campaign_sha256": campaign, "classification": state.get("classification"),
            "inputs": [{k: v for k, v in i.items()
                        if k in ("role", "resource", "sha256", "verification_state",
                                 "availability_contract_sha256")}
                       for i in (state.get("inputs") or [])],
            "cube_rows": terminal_rows(campaign),
            "reconcile": reconcile(campaign, key) if campaign else {},
            "produced": sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else [],
            "log_tail": log.read_text(errors="replace")[-1200:] if code else ""}


def flush(work: Path, key_file: Path) -> dict:
    tool = GITHUB / "data-gov" / "tools" / "governed_exec.py"
    result = subprocess.run([sys.executable, str(tool), "--flush", "--gov-url", GOV_URL,
                             "--api-key-file", str(key_file), "--outbox-dir", str(work / "outbox")],
                            capture_output=True, text=True)
    try:
        body = json.loads(result.stdout or "{}")
    except ValueError:
        body = {"stdout": result.stdout[-400:]}
    return {"exit_code": result.returncode, **body}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--consumer", action="append", default=[])
    parser.add_argument("--label", default="", help="suffix for the campaign keys of this attempt")
    args = parser.parse_args(argv)
    consumers = args.consumer or ["preprocessor", "predictor"]
    work = args.work.resolve()
    if work.exists():
        raise SystemExit(f"REFUSED: the work directory already exists: {work}")
    work.mkdir(parents=True)
    key = args.api_key_file.read_text().strip()

    report = {"schema": "consumer_adoption_check.v1", "gov_url": GOV_URL, "label": args.label,
              "sample_lake": SAMPLE_LAKE, "classification": "NON_GOVERNING",
              "cube_before": cube_counts(), "consumers": {}}
    for consumer in consumers:
        cases = {}
        out_dir = work / consumer / "run"
        cases["success"] = run_case(consumer, "success", work, args.api_key_file, key, out_dir,
                                    config_for(consumer, work, "success"), args.label)
        cases["stale"] = run_case(consumer, "stale", work, args.api_key_file, key, out_dir,
                                  config_for(consumer, work, "success"), args.label)
        cases["failure"] = run_case(consumer, "failure", work, args.api_key_file, key,
                                    work / consumer / "run_failure",
                                    config_for(consumer, work, "failure"), args.label)
        before = cube_counts()
        cases["retry"] = {"flush": flush(work, args.api_key_file), "cube_before": before,
                          "cube_after": cube_counts()}
        exact = lambda r: (r.get("missing_units") == [] and r.get("accounting_only") == []  # noqa: E731
                           and r.get("lake_only") == [])
        cases["retry"]["sends_nothing"] = (cases["retry"]["flush"].get("sent") == 0
                                           and before == cases["retry"]["cube_after"])
        verdict = {
            "success": (cases["success"]["status"] == "COMPLETED"
                        and cases["success"]["exit_code"] == 0
                        and bool(cases["success"]["inputs"])
                        and all(i.get("verification_state", "").startswith("VERIFIED")
                                for i in cases["success"]["inputs"])
                        and exact(cases["success"]["reconcile"])),
            "stale_refused": cases["stale"]["status"] == "REFUSED" and cases["stale"]["exit_code"] != 0,
            "failure_recorded": (cases["failure"]["status"] == "FAILED"
                                 and cases["failure"]["exit_code"] != 0
                                 and bool(cases["failure"]["cube_rows"])),
            "retry_sends_nothing": cases["retry"]["sends_nothing"],
        }
        report["consumers"][consumer] = {"cases": cases, "verdict": verdict,
                                         "adopted": all(verdict.values())}
    report["cube_after"] = cube_counts()
    report["cube_delta"] = {k: report["cube_after"][k] - report["cube_before"][k]
                            for k in report["cube_after"]}
    report["ok"] = all(c["adopted"] for c in report["consumers"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str).replace(str(Path.home()), "~")
                        + "\n", encoding="utf-8")
    shutil.rmtree(work / "cache", ignore_errors=True)
    print(json.dumps({"ok": report["ok"], "cube_delta": report["cube_delta"],
                      "consumers": {name: c["verdict"] for name, c in report["consumers"].items()}},
                     indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
