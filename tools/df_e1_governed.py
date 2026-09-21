#!/usr/bin/env python3
"""RP33/RP38: the governed entry of an E1 run — the only way its bytes may arrive.

Acceptance, in the order it must happen:

  1. the campaign is registered BEFORE anything is prepared or fitted, declaring the lake, the
     resource and the role it will consume;
  2. the panel is obtained over HTTP from data-gov by THIS host's own identity, its digest verified
     against the stream's header and against the bytes on disk, and the delivery confirmed;
  3. the run consumes exactly those bytes: the prepared DATA records the delivered digest, and the
     closure refuses if what was read is not what was delivered;
  4. every unit ends with a terminal that goes outbox -> accounting, and the campaign reconciles;
  5. if the lake host is absent, unreachable or does not serve the resource, NO new work starts.
     A cached copy is used only when governance itself authorises it (the delivery says `cached`).

A local document with the terminal schema is not a terminal: nothing here writes one.

    python tools/df_e1_governed.py acquire --run-id ID --root ROOT --lake public_panels \\
        --resource uci_235_.../panel.parquet --api-key-file KEY [--gov-url http://127.0.0.1:5055]
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCHEMA = "df_e1_governed_acquisition.v1"
DEFAULT_GOV = "http://127.0.0.1:5055"


def _load(name: str, where: Path = HERE):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class GovernanceUnavailable(SystemExit):
    """The entry host is absent or does not serve the resource: no new work may start."""


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class _Lock:
    """One writer at a time for a receipt file. Children run in parallel and each was rewriting the
    whole file from its own copy, so the last writer erased the other units' deliveries."""

    def __init__(self, path: Path):
        self.path = Path(str(path) + ".lock")

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "w")
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()
        return False


def _merge_write(path: Path, unit_id: str, entry: dict, base: dict) -> dict:
    """Re-read, merge THIS unit, write atomically, all while holding the lock."""
    with _Lock(path):
        doc = json.loads(path.read_text()) if path.is_file() else base
        doc.setdefault("units", {})[unit_id] = entry
        doc["transfer"] = {"transferred_units": sum(1 for u in doc["units"].values() if not u.get("cached")),
                           "cache_reused_units": sum(1 for u in doc["units"].values() if u.get("cached")),
                           "bytes_first_transfer": next((u.get("bytes") for u in doc["units"].values()
                                                         if not u.get("cached")), None)}
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1, default=str))
        os.replace(tmp, path)
        return doc


def acquire(*, run_id: str, root: Path, lake: str, resource: str, unit_id: str = "prepare", role: str = "panel",
            gov_url: str = DEFAULT_GOV, api_key_file: Path, design_sha256: str,
            cache_dir: Path | None = None, expect_sha256: str | None = None, units: list | None = None) -> dict:
    """One campaign and one delivery PER UNIT (the shape data-gov completes: a delivery binds to
    campaign, actor and unit, so a unit without its own verified delivery cannot complete).

    The first unit transfers the bytes; the next ones are served from the verified cache, and the
    receipt records which, so transfer and reuse are measured instead of assumed.
    """
    GR = _load("governed_run")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "DELIVERIES.json"
    doc = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {
        "schema": SCHEMA, "at": now_iso(), "host": os.uname().nodename, "run_id": run_id,
        "design_sha256": design_sha256, "lake": lake, "resource": resource, "role": role, "units": {},
        "order": "each unit's campaign is registered BEFORE it reads anything; its delivery is verified before it runs"}
    if doc.get("design_sha256") != design_sha256 or doc.get("resource") != resource:
        raise SystemExit("REFUSED: this root already holds deliveries of another design or resource")
    with _Lock(receipt_path):                    # another child may have acquired it a moment ago
        current = json.loads(receipt_path.read_text()) if receipt_path.is_file() else doc
    if unit_id in (current.get("units") or {}):
        return current
    code_identity = GR.strict_code_identity(REPO)
    key = f"{run_id}-{unit_id}-data"
    token = Path(api_key_file).read_text().strip()
    gov = GR.GovHttp(gov_url, token, key)
    campaign = {"schema": "governed_campaign.v1", "campaign_key": key, "classification": "NON_GOVERNING",
                "project": "predictor", "code_identity": code_identity, "config_sha256": design_sha256,
                "input_mode": "DATASETS", "synthetic_spec_sha256": None, "units": [unit_id],
                "datasets": [{"lake": lake, "resource": resource, "role": role, "from": None, "to": None}],
                "terminal_lake": "olap_cube"}
    try:
        status, body = gov.submit_campaign(campaign)
    except GR.GovernedRunError as exc:
        raise GovernanceUnavailable(f"REFUSED: the governance host is unreachable ({exc}); no new work starts") from None
    if status not in (200, 201):
        raise GovernanceUnavailable(f"REFUSED: the campaign was not registered (http {status}: {body.get('error')}); "
                                    "no data is read and no unit runs")
    sha = body["campaign_sha256"]
    try:
        http, info = gov.governed_download(sha, unit_id, lake, resource, role,
                                           str(cache_dir or (Path.home() / ".cache/data-gov")))
    except GR.GovernedRunError as exc:
        raise GovernanceUnavailable(f"REFUSED: the delivery failed ({exc}); no new work starts") from None
    path = Path(info["path"])
    on_disk = sha_file(path)
    if on_disk != info["sha256"]:
        raise SystemExit("REFUSED: the delivered bytes on disk are not the ones the stream declared")
    if expect_sha256 and on_disk != expect_sha256:
        raise SystemExit(f"REFUSED: the delivered panel {on_disk} is not the characterised {expect_sha256}")
    entry = {"campaign_key": key, "campaign_sha256": sha, "code_identity": code_identity,
             "at": now_iso(), "host": os.uname().nodename,
             "delivery_id": info.get("delivery_id"), "sha256": info["sha256"],
             "bytes": info.get("bytes"), "cached": bool(info.get("cached")),
             "verification_state": info.get("verification_state"),
             "availability_use": info.get("availability_use"),
             "availability_label": info.get("availability_label"),
             "availability_contract_sha256": info.get("availability_contract_sha256"),
             "path": str(path), "bytes_on_disk_sha256": on_disk}
    return _merge_write(receipt_path, unit_id, entry, doc)


def require_delivery(root: Path, design: dict, unit_id: str = "prepare") -> dict:
    """What `prepare` and every unit call: it may only read a panel delivered to THAT unit."""
    path = Path(root) / "DELIVERIES.json"
    if not path.is_file():
        raise GovernanceUnavailable(
            "REFUSED: this run has no governed delivery. The panel is a governed resource and the run reads it only "
            "through data-gov; run `df_e1_governed.py acquire` first (and, if the resource is not served yet, the "
            "lake registration is the missing object, not the science).")
    doc = json.loads(path.read_text())
    if doc.get("design_sha256") != design["design_sha256"]:
        raise SystemExit("REFUSED: the delivery in this root belongs to another design")
    unit = (doc.get("units") or {}).get(unit_id)
    if unit is None:
        raise GovernanceUnavailable(f"REFUSED: unit {unit_id!r} has no governed delivery of its own; it does not run")
    delivered = Path(unit["path"])
    if not delivered.is_file():
        raise GovernanceUnavailable("REFUSED: the delivered bytes are gone from the cache; nothing is read from anywhere else")
    if sha_file(delivered) != unit["sha256"]:
        raise SystemExit("REFUSED: the delivered bytes changed after the delivery was confirmed")
    return {**doc, "delivery": unit, "campaign_sha256": unit["campaign_sha256"], "campaign_key": unit["campaign_key"]}


def report_terminal(root: Path, unit_id: str, terminal: dict, *, gov_url: str = DEFAULT_GOV,
                    api_key_file: Path, outbox_dir: str | None = None, started_at: str | None = None) -> dict:
    """Terminal -> outbox -> accounting, under THIS unit's own campaign and delivery."""
    GR = _load("governed_run")
    doc = json.loads((Path(root) / "DELIVERIES.json").read_text())
    unit = (doc.get("units") or {}).get(unit_id)
    if unit is None:
        raise GovernanceUnavailable(f"REFUSED: unit {unit_id!r} has no delivery, so it has no terminal to report")
    gov = GR.GovHttp(gov_url, Path(api_key_file).read_text().strip(), unit["campaign_key"])
    # one spool PER UNIT: children run in parallel and a shared spool made each of them flush the
    # others' envelopes — through its own campaign's client, and racing on the same file
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(outbox_dir or GR.DEFAULT_OUTBOX)).resolve() / "units" / unit_id)
    body = {**terminal, "deliveries": sorted({unit["delivery_id"]})}
    outbox.put({"campaign_sha256": unit["campaign_sha256"], "unit_id": unit_id, "terminal": body})
    receipts = {}

    def sender(envelope):
        status, receipt = gov.report_terminal(envelope["campaign_sha256"], envelope["unit_id"], envelope["terminal"])
        if status not in (200, 201):
            raise GR.GovernedRunError(f"terminal refused: http {status} {receipt.get('error', '')}".strip())
        GR._require_reconciled(gov, envelope["campaign_sha256"], envelope["unit_id"], before_run=False)
        receipts[envelope["unit_id"]] = receipt          # the SERVICE's own answer, kept for the receipt file
        return receipt
    flushed = outbox.flush(sender)
    status, rbody = gov.reconcile_campaign(unit["campaign_sha256"])
    reconciliation = {"http": status, "missing_units": rbody.get("missing_units"),
                      "accounting_only": rbody.get("accounting_only"), "lake_only": rbody.get("lake_only")}
    persisted = None
    if receipts.get(unit_id):                            # RP51: persist it from the client's own answer
        RC = _load("df_e1_receipts")
        persisted = RC.record_accepted(root, unit_id, campaign_sha256=unit["campaign_sha256"],
                                       campaign_key=unit["campaign_key"], terminal=body,
                                       receipt=receipts[unit_id], reconciliation=reconciliation,
                                       design_sha256=doc["design_sha256"], started_at=started_at)
    return {"flushed": flushed, "campaign_sha256": unit["campaign_sha256"], "receipt": receipts.get(unit_id),
            "persisted_receipt": persisted, "reconciliation": reconciliation}


def report_failed(root: Path, unit_id: str, reason: str, *, gov_url: str, api_key_file: Path,
                  outbox_dir: str | None = None) -> dict:
    """Close a unit that will never produce a result, with a FAILED terminal that says why.

    A registered campaign whose unit never closes stays PENDING for ever, and the population rule
    then names it for ever. When the work genuinely cannot be completed — the process died, or the
    fact was published under a successor campaign — the honest end is a terminal with status FAILED
    carrying the reason, not silence and not a COMPLETED terminal for work that did not happen.
    """
    U = _load("df_utility_run")
    started = U._z(U.now_iso())
    terminal = U._terminal(status="FAILED", reason=reason,
                           cost={"wall_seconds": 0.0, "cpu_seconds": 0.0}, metrics=[],
                           started=started, finished=U._z(U.now_iso()),
                           tags={"unit": unit_id, "classification": "NON_GOVERNING",
                                 "phase": "DEVELOPMENT", "closed_as": "FAILED",
                                 "reading": "this unit produced no result; the terminal records that "
                                            "fact so its campaign stops being an open one"})
    return report_terminal(root, unit_id, terminal, gov_url=gov_url, api_key_file=api_key_file,
                           outbox_dir=outbox_dir, started_at=started)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["acquire", "check", "fail"])
    ap.add_argument("--unit")
    ap.add_argument("--reason")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--lake", default="public_panels")
    ap.add_argument("--resource", default="uci_235_individual_household_power/panel.parquet")
    ap.add_argument("--role", default="panel")
    ap.add_argument("--gov-url", default=DEFAULT_GOV)
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    design = json.loads(a.design.read_text()) if a.design else json.loads((a.root / "DESIGN.json").read_text())
    if a.command == "fail":
        if not (a.unit and a.reason):
            raise SystemExit("REFUSED: a FAILED terminal names its unit and its reason")
        out = report_failed(a.root, a.unit, a.reason, gov_url=a.gov_url, api_key_file=a.api_key_file)
        print(json.dumps({"unit": a.unit, "campaign_sha256": out["campaign_sha256"],
                          "terminal_sha256": (out.get("receipt") or {}).get("terminal_sha256"),
                          "reconciliation": out["reconciliation"], "sent": out["flushed"]["sent"]},
                         indent=1, default=str))
        return 0 if out["flushed"]["sent"] else 1
    if a.command == "check":
        print(json.dumps(require_delivery(a.root, design), indent=1, default=str))
        return 0
    doc = None
    for unit_id in ["prepare"] + [c["cell_id"] for c in design["pilots"] + design["cells"]]:
        doc = acquire(run_id=a.run_id or "e1-run", root=a.root, lake=a.lake, resource=a.resource, role=a.role,
                      unit_id=unit_id, gov_url=a.gov_url, api_key_file=a.api_key_file,
                      design_sha256=design["design_sha256"], cache_dir=a.cache_dir,
                      expect_sha256=design["governed_bytes"]["sha256"])
    print(json.dumps(doc, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
