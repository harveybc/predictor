#!/usr/bin/env python3
"""C146 (order 2026-09-13): the two OOM terminations as non-governing attempts.

Nothing here is a result. The record states, from the kernel and systemd
journals and from the preserved roots on disk:

* each kernel OOM kill of a runner process (time, pid, anon RSS, scope kind)
  and the units that failed with result 'oom-kill' in the same window;
* for each module (profiles C130-C133, SNR calibration C134) the root it left,
  read-only, with its listing digest and the final receipts it lacks;
* what cannot be derived from disk, typed as such.

The per-dataset memory projection and the causal reproductions are in the
frozen PRE, bound here by the sha256 of its script and output. The record is
sealed with record_sha256 and yields df_fact_incident_attempt rows. Hosts do
not appear; roles do.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STATE = Path.home() / ".local/state/crispdm-data-foundation"
PRE = REPO / "docs/audits/evidence/repro_runs/c146_c165_pre_2026_09_13"
WINDOW = ("2026-09-12 21:30:00", "2026-09-12 22:45:00")
SCHEMA = "crispdm.data_foundation.c146_incident_record.v1"
KILL = re.compile(r"^(\S+) \S+ kernel: Out of memory: Killed process (\d+) \((\S+)\) total-vm:\d+kB, "
                  r"anon-rss:(\d+)kB")
MEMCG = re.compile(r"task_memcg=(\S+),task=\S+,pid=(\d+),")
UNIT_FAIL = re.compile(r"^(\S+) \S+ systemd\[\d+\]: (\S+): Failed with result 'oom-kill'")
ROOTS = {"C130_C133_PROFILES": ("profiles_c130_v1", ("PROFILE_RUN_RECEIPT.json",)),
         "C134_SNR_CALIBRATION": ("snr_calibration_c134_v1",
                                  ("SNR_CALIBRATION.v1.json", "SNR_CALIBRATION.v1.olap_rows.jsonl"))}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_sha256() -> str:
    return sha_bytes(Path(__file__).read_bytes())


def scope_kind(unit: str) -> str:
    for pat, kind in (("snap.code", "EDITOR_SESSION_SCOPE"), ("snap.firefox", "BROWSER_SCOPE"),
                      ("docker-", "CONTAINER_SCOPE")):
        if pat in unit:
            return kind
    return "OTHER_UNIT"


def journal(*args) -> list[str]:
    return subprocess.run(["journalctl", *args, "--since", WINDOW[0], "--until", WINDOW[1], "-o", "short-iso",
                           "--no-pager"], capture_output=True, text=True).stdout.splitlines()


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        h.update(str(p.relative_to(root)).encode())
        h.update(b"D" if p.is_dir() else sha_bytes(p.read_bytes()).encode())
    return h.hexdigest()


def build() -> dict:
    kills, scope_by_pid = [], {}
    for ln in journal("-k"):
        m = MEMCG.search(ln)
        if m:
            scope_by_pid[int(m.group(2))] = scope_kind(m.group(1))
        k = KILL.search(ln)
        if k:
            kills.append({"at": k.group(1), "pid": int(k.group(2)), "process_name": k.group(3),
                          "anon_rss_bytes": int(k.group(4)) * 1024})
    for k in kills:
        k["scope"] = scope_by_pid.get(k["pid"], "UNKNOWN")
    units = [{"at": m.group(1), "unit": scope_kind(m.group(2))} for m in map(UNIT_FAIL.search, journal()) if m]
    roots = {}
    for module, (name, receipts) in ROOTS.items():
        root = STATE / name
        roots[module] = {"root_name": name, "exists": root.is_dir(),
                         "files": sorted(str(p.relative_to(root)) for p in root.rglob("*")),
                         "read_only": not any(p.stat().st_mode & 0o222 for p in [root, *root.rglob("*")]),
                         "listing_sha256": tree_digest(root),
                         "missing_final_receipts": [r for r in receipts if not (root / r).exists()]}
    runner_kills = [k for k in kills if k["process_name"] == "python" and k["scope"] == "EDITOR_SESSION_SCOPE"]
    editor_failures = [u for u in units if u["unit"] == "EDITOR_SESSION_SCOPE"]
    attempts = []
    for i, k in enumerate(runner_kills, start=1):
        scope_end = next((u["at"] for u in editor_failures if u["at"] >= k["at"]), None)
        attempts.append({"attempt_id": f"C130_C133_PROFILES_ATTEMPT_{i}", "module": "C130_C133_PROFILES",
                         "terminated_by": "KERNEL_OOM_KILL_OF_A_RUNNER_WORKER", "event_at": k["at"],
                         "victim_pid": k["pid"], "victim_anon_rss_bytes": k["anon_rss_bytes"]})
        attempts.append({"attempt_id": f"C134_SNR_CALIBRATION_ATTEMPT_{i}", "module": "C134_SNR_CALIBRATION",
                         "terminated_by": "EDITOR_SESSION_SCOPE_STOPPED_AFTER_OOM", "event_at": scope_end,
                         "victim_pid": None, "victim_anon_rss_bytes": None})
    doc = {
        "schema": SCHEMA,
        "state": "NON_GOVERNING_ATTEMPTS",
        "kernel_oom_kills": kills,
        "units_failed_with_oom_kill": units,
        "attempts": attempts,
        "roots_left": roots,
        "not_derivable_from_disk": [
            "which dataset each killed worker was profiling: the runner wrote no job-start record "
            "(the PRE names projection-consistent candidates)",
            "the launch time of the first attempt: its roots were empty and were removed when the names "
            "were reused at the second launch; the kept roots belong to the second attempt",
        ],
        "bound_pre": {"script_sha256": sha_bytes(PRE.with_suffix(".py").read_bytes()),
                      "output_sha256": sha_bytes(PRE.with_suffix(".out").read_bytes())},
        "never": "no row, figure or root of these attempts is a result or is promoted",
        "code_sha256": code_sha256(),
    }
    doc["record_sha256"] = sha_bytes(json.dumps(doc, sort_keys=True).encode())
    return doc


def rows(doc: dict, run_id: str) -> list[dict]:
    out = []
    for a in doc["attempts"]:
        root = doc["roots_left"][a["module"]]
        last = a["attempt_id"].endswith(f"_{len(doc['attempts']) // 2}")
        out.append({"run_id": run_id, "attempt_id": a["attempt_id"], "module": a["module"],
                    "state": "NON_GOVERNING_ATTEMPT", "terminated_by": a["terminated_by"], "event_at": a["event_at"],
                    "victim_pid": a["victim_pid"], "victim_anon_rss_bytes": a["victim_anon_rss_bytes"],
                    "root_name": root["root_name"] if last else None,
                    "root_listing_sha256": root["listing_sha256"] if last else None,
                    "missing_receipts": root["missing_final_receipts"] if last else [],
                    "evidence": {"record_sha256": doc["record_sha256"],
                                 "root_note": "kept read-only" if last else "empty root removed; name reused"},
                    "code_sha256": doc["code_sha256"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="record JSON; refuses to overwrite")
    ap.add_argument("--rows", type=Path, help="df_fact_incident_attempt JSONL; refuses to overwrite")
    ap.add_argument("--run-id", default="c146_incident_record_v1")
    a = ap.parse_args(argv)
    for p in (a.out, a.rows):
        if p is not None and p.exists():
            raise SystemExit(f"REFUSED: {p.name} exists; the record is write-once")
    doc = build()
    if len(doc["attempts"]) != 4:
        raise SystemExit(f"REFUSED: expected two kills and two SNR stops, found {len(doc['attempts'])} attempts")
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    if a.rows:
        a.rows.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows(doc, a.run_id)))
    print(json.dumps({"record_sha256": doc["record_sha256"], "attempts": len(doc["attempts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
