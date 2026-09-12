"""PRE-R for the recovery addendum R1-R6 (order 2026-09-11).

Freezes the six durable facts Musashi's post-reboot inspection found,
through public interfaces only. READ-ONLY: this script starts, stops
and restarts nothing, and writes nothing outside its own stdout.

Private paths, host topology and account identifiers are redacted: every
path is printed relative to $HOME as `~/...`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
SHARE = HOME / ".local/share/agent-multi"
B4 = SHARE / "b4_campaign_results_v7_20260907"
T2 = SHARE / "t2_confirmatory_results_resource_successor_v1_20260909"
OUTBOX = HOME / ".local/share/predictor/olap_outbox"
UNIT_DIR = HOME / ".config/systemd/user"

FAILURES: list[str] = []


def redact(p) -> str:
    s = str(p)
    return s.replace(str(HOME), "~")


def item(n: int, title: str) -> None:
    print(f"\n=== PRE-R {n}: {title} ===")


def expect(cond: bool, msg: str) -> None:
    print(f"  [{'PRE-REPRODUCED' if cond else 'NOT-REPRODUCED'}] {msg}")
    if not cond:
        FAILURES.append(msg)


def systemctl(*args: str) -> str:
    try:
        return subprocess.run(("systemctl", "--user", *args),
                              capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as exc:                      # pragma: no cover
        return f"<unavailable: {type(exc).__name__}>"


# --------------------------------------------------------------- 1
item(1, "B4 v7 has 2 terminals, 1 partial, 9 not started and no final "
        "quarantine adjudication")
ledger = json.loads((B4 / "CAMPAIGN_LEDGER.json").read_text())
cells = ledger["cells"]
completed, partial, not_started = [], [], []
for key in sorted(cells):
    d = B4 / key
    if (d / "B4_CELL_TERMINAL.json").is_file():
        completed.append(key)
    elif d.is_dir():
        partial.append(key)
    else:
        not_started.append(key)
print(f"  declared cells           : {len(cells)}")
print(f"  COMPLETED (terminal)     : {len(completed)} {completed}")
print(f"  PARTIAL (dir, no term.)  : {len(partial)} {partial}")
print(f"  NOT_STARTED (no dir)     : {len(not_started)} {len(not_started) * ''}")
print(f"  CAMPAIGN_STOP present    : {(B4 / 'CAMPAIGN_STOP').exists()}")
print(f"  per-cell STOP on partial : "
      f"{[k for k in partial if (B4 / k / 'STOP').exists()]}")
ledger_states = sorted({c["status"] for c in cells.values()})
print(f"  ledger cell statuses     : {ledger_states}")
adjudications = sorted(p.name for p in B4.glob("*ADJUDICATION*")) + \
                sorted(p.name for p in B4.glob("*QUARANTINE*"))
print(f"  campaign adjudication    : {adjudications or 'ABSENT'}")
expect(len(completed) == 2 and len(partial) == 1 and len(not_started) == 9,
       "B4 v7 is physically 2 COMPLETED / 1 PARTIAL / 9 NOT_STARTED")
expect(ledger_states == ["PENDING"],
       "the ledger still calls every cell PENDING, including the two sealed")
expect(not adjudications,
       "no durable campaign-level quarantine adjudication exists")

# --------------------------------------------------------------- 2
item(2, "T2 has 242/242/242 while the heartbeat still says done=241")
names = sorted(p.name for p in (T2 / "units").iterdir())
kinds: dict[str, int] = {}
for n in names:
    kinds[n.split("_", 1)[0]] = kinds.get(n.split("_", 1)[0], 0) + 1
print(f"  unit artifacts by kind   : {kinds}")
hb = json.loads((T2 / "EXECUTOR_HEARTBEAT.json").read_text())
print(f"  heartbeat done           : {hb['done']}")
print(f"  heartbeat current_unit   : {hb['current_unit']}")
print(f"  heartbeat wall_consumed_s: {hb['wall_consumed_s']}")
cur = hb["current_unit"].replace("::", "__")
cur_record = (T2 / "units" / f"RECORD_{cur}.json")
print(f"  record of current_unit   : "
      f"{redact(cur_record)} exists={cur_record.is_file()}")
closures = sorted(p.name for p in T2.glob("*CLOSURE*")) + \
           sorted(p.name for p in T2.glob("*RECONCIL*"))
print(f"  durable closure record   : {closures or 'ABSENT'}")
expect(kinds.get("CLAIM") == 242 and kinds.get("ARRAYS") == 242
       and kinds.get("RECORD") == 242,
       "T2 holds exactly 242 claims, 242 arrays and 242 records")
expect(hb["done"] == 241 and cur_record.is_file(),
       "the heartbeat reports done=241 while the 242nd record is sealed "
       "on disk — telemetry disagreeing with durable evidence")
expect(not closures,
       "no durable 241->242 reconciliation/closure record exists")

# --------------------------------------------------------------- 3
item(3, "p1lr-decision@101 restarts on an ExecStartPre refusal that "
        "RestartPreventExitStatus cannot cover")
props = systemctl("show", "p1lr-decision@101.service",
                  "-p", "NRestarts", "-p", "Restart",
                  "-p", "RestartPreventExitStatus", "-p", "ExecMainStatus",
                  "-p", "Result", "-p", "ActiveState")
shown = dict(line.split("=", 1) for line in props.splitlines() if "=" in line)
for k in ("ActiveState", "Result", "Restart", "RestartPreventExitStatus",
          "ExecMainStatus", "NRestarts"):
    print(f"  {k:26s}: {shown.get(k, '<unavailable>')}")
gate = None
for drop in sorted((UNIT_DIR / "p1lr-decision@.service.d").glob("*.conf")):
    for line in drop.read_text().splitlines():
        if line.startswith("Environment=P1LR_SCREEN_GATE="):
            gate = Path(line.split("=", 2)[2])
print(f"  effective pinned gate     : {redact(gate)}")
print(f"  pinned gate exists        : {gate.is_file() if gate else 'n/a'}")
expect(gate is not None and not gate.is_file(),
       "the pinned screen gate does not exist, so ExecStartPre exits 4")
expect(int(shown.get("NRestarts", "0")) > 0
       and shown.get("ExecMainStatus") == "0",
       "systemd restarted the unit because the MAIN process never ran: "
       "RestartPreventExitStatus=4 does not see an ExecStartPre exit code")

# --------------------------------------------------------------- 4
item(4, "the Alpaca runner classifies a wrapped ConnectionError as fatal")
sys.path.insert(0, str(HOME / "Documents/GitHub/lts"))
from app.runner_retry_taxonomy import classify_runner_exception  # noqa: E402


class AlpacaPaperError(RuntimeError):
    """Mirrors lts app.alpaca_paper_lab.AlpacaPaperError."""


try:
    raise ConnectionError("connection refused")
except ConnectionError as exc:
    wrapped = AlpacaPaperError("account request failed: ConnectionError")
    wrapped.__cause__ = exc
direct = ConnectionError("connection refused")
print(f"  direct  ConnectionError   -> {classify_runner_exception(direct)}")
print(f"  wrapped in AlpacaPaperError -> "
      f"{classify_runner_exception(wrapped)}")
print(f"  __cause__ of the wrapper   -> {type(wrapped.__cause__).__name__}")
hb_path = HOME / ".local/state/lts/alpaca-model-runner-heartbeat.json"
if hb_path.is_file():
    ahb = json.loads(hb_path.read_text())
    print(f"  live runner state          : {ahb.get('state')}")
    print(f"  live runner phase          : {ahb.get('phase')}")
    print(f"  live runner error          : {ahb.get('error')}")
expect(classify_runner_exception(direct) == "transient"
       and classify_runner_exception(wrapped) == "fatal",
       "the same connection failure is transient when raw and fatal once "
       "the broker wraps it with `raise ... from`")

# --------------------------------------------------------------- 5
item(5, "the OLAP loader is alive, fresh and empty, yet healthy=false")
sys.path.insert(0, str(HOME / "Documents/GitHub/predictor"))
from olap import outbox as ob                                  # noqa: E402

counts = ob.counts(OUTBOX)
hbeat = json.loads((OUTBOX / "HEARTBEAT.json").read_text())
print(f"  outbox counts             : {counts}")
print(f"  heartbeat healthy         : {hbeat['healthy']}")
print(f"  heartbeat pending         : {hbeat['pending']}")
print(f"  heartbeat failed          : {hbeat['failed']}")
reasons = sorted((OUTBOX / "failed").glob("*.reason"))
for r in reasons:
    print(f"  dead-letter reason        : {r.read_text().strip()[:120]}")
print(f"  loader ActiveState        : "
      f"{systemctl('show', 'crispdm-olap-loader.service', '-p', 'ActiveState')}")
print(f"  loader NRestarts          : "
      f"{systemctl('show', 'crispdm-olap-loader.service', '-p', 'NRestarts')}")
expect(counts["pending"] == 0 and counts["failed"] == 1
       and hbeat["healthy"] is False,
       "a single historical dead-letter makes `healthy` false forever, "
       "and nothing distinguishes it from a dead loader")

# --------------------------------------------------------------- 6
item(6, "systemd ignores StartLimitIntervalSec in the loader unit")
unit = UNIT_DIR / "crispdm-olap-loader.service"
verify = subprocess.run(("systemd-analyze", "--user", "verify", str(unit)),
                        capture_output=True, text=True, timeout=60)
lines = [ln for ln in (verify.stdout + verify.stderr).splitlines()
         if "crispdm-olap-loader" in ln]
for ln in lines:
    print(f"  {redact(ln)}")
section, seen = None, {}
for raw in unit.read_text().splitlines():
    s = raw.strip()
    if s.startswith("[") and s.endswith("]"):
        section = s
    elif s.startswith("StartLimit"):
        seen[s.split("=")[0]] = section
print(f"  StartLimit* declared in   : {seen}")
expect(any("StartLimitIntervalSec" in ln and "ignoring" in ln
           for ln in lines)
       and seen.get("StartLimitIntervalSec") == "[Service]",
       "StartLimitIntervalSec sits in [Service], where systemd ignores it")

# ---------------------------------------------------------------
print(f"\n=== PRE-R SUMMARY: {6 - len({f for f in FAILURES})} items printed; "
      f"{len(FAILURES)} assertions NOT reproduced ===")
for f in FAILURES:
    print(f"  NOT-REPRODUCED: {f}")
print("READ-ONLY: no service was started, stopped or restarted; no "
      "durable artifact was written.")
sys.exit(1 if FAILURES else 0)
