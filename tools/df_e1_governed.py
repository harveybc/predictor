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


def acquire(*, run_id: str, root: Path, lake: str, resource: str, role: str = "panel",
            gov_url: str = DEFAULT_GOV, api_key_file: Path, units: list, design_sha256: str,
            cache_dir: Path | None = None, expect_sha256: str | None = None) -> dict:
    """Register the campaign, take the delivery and record it. Returns the acquisition receipt."""
    GR = _load("governed_run")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "DELIVERIES.json"
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        if prior.get("design_sha256") != design_sha256 or prior.get("resource") != resource:
            raise SystemExit("REFUSED: this root already holds a delivery of another design or resource")
        return prior
    code_identity = GR.strict_code_identity(REPO)
    key = f"{run_id}-e1-data"
    token = Path(api_key_file).read_text().strip()
    gov = GR.GovHttp(gov_url, token, key)
    campaign = {"schema": "governed_campaign.v1", "campaign_key": key, "classification": "NON_GOVERNING",
                "project": "predictor", "code_identity": code_identity, "config_sha256": design_sha256,
                "input_mode": "DATASETS", "synthetic_spec_sha256": None, "units": list(units),
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
    unit = list(units)[0]
    try:
        http, info = gov.governed_download(sha, unit, lake, resource, role,
                                           str(cache_dir or (Path.home() / ".cache/data-gov")))
    except GR.GovernedRunError as exc:
        raise GovernanceUnavailable(f"REFUSED: the delivery failed ({exc}); no new work starts") from None
    path = Path(info["path"])
    on_disk = sha_file(path)
    if on_disk != info["sha256"]:
        raise SystemExit("REFUSED: the delivered bytes on disk are not the ones the stream declared")
    if expect_sha256 and on_disk != expect_sha256:
        raise SystemExit(f"REFUSED: the delivered panel {on_disk} is not the characterised {expect_sha256}")
    doc = {"schema": SCHEMA, "at": now_iso(), "host": os.uname().nodename, "run_id": run_id,
           "campaign_key": key, "campaign_sha256": sha, "code_identity": code_identity,
           "design_sha256": design_sha256, "lake": lake, "resource": resource, "role": role,
           "delivery": {"delivery_id": info.get("delivery_id"), "sha256": info["sha256"], "bytes": info.get("bytes"),
                        "cached": info.get("cached"), "verification_state": info.get("verification_state"),
                        "availability_use": info.get("availability_use"), "availability_label": info.get("availability_label"),
                        "availability_contract_sha256": info.get("availability_contract_sha256"),
                        "path": str(path)},
           "bytes_on_disk_sha256": on_disk,
           "order": "campaign registered BEFORE any preparation or fit; the panel was then delivered and verified"}
    receipt_path.write_text(json.dumps(doc, indent=1, default=str))
    return doc


def require_delivery(root: Path, design: dict) -> dict:
    """What `prepare` calls: the run may only read a panel that was delivered to it."""
    path = Path(root) / "DELIVERIES.json"
    if not path.is_file():
        raise GovernanceUnavailable(
            "REFUSED: this run has no governed delivery. The panel is a governed resource and the run reads it only "
            "through data-gov; run `df_e1_governed.py acquire` first (and, if the resource is not served yet, the "
            "lake registration is the missing object, not the science).")
    doc = json.loads(path.read_text())
    if doc.get("design_sha256") != design["design_sha256"]:
        raise SystemExit("REFUSED: the delivery in this root belongs to another design")
    delivered = Path(doc["delivery"]["path"])
    if not delivered.is_file():
        raise GovernanceUnavailable("REFUSED: the delivered bytes are gone from the cache; nothing is read from anywhere else")
    if sha_file(delivered) != doc["delivery"]["sha256"]:
        raise SystemExit("REFUSED: the delivered bytes changed after the delivery was confirmed")
    return doc


def report_terminal(root: Path, unit_id: str, terminal: dict, *, gov_url: str = DEFAULT_GOV,
                    api_key_file: Path, outbox_dir: str | None = None) -> dict:
    """Terminal -> outbox -> accounting, with the campaign of this run's delivery."""
    GR = _load("governed_run")
    doc = json.loads((Path(root) / "DELIVERIES.json").read_text())
    gov = GR.GovHttp(gov_url, Path(api_key_file).read_text().strip(), doc["campaign_key"])
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(outbox_dir or GR.DEFAULT_OUTBOX)).resolve())
    outbox.put({"campaign_sha256": doc["campaign_sha256"], "unit_id": unit_id, "terminal": terminal})
    flushed = GR._send_pending(gov, outbox)
    status, body = gov.reconcile_campaign(doc["campaign_sha256"])
    return {"flushed": flushed, "reconciliation": {"http": status, "missing_units": body.get("missing_units"),
                                                   "accounting_only": body.get("accounting_only"),
                                                   "lake_only": body.get("lake_only")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["acquire", "check"])
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
    if a.command == "check":
        print(json.dumps(require_delivery(a.root, design), indent=1, default=str))
        return 0
    units = [c["cell_id"] for c in design["pilots"] + design["cells"]]
    doc = acquire(run_id=a.run_id or "e1-run", root=a.root, lake=a.lake, resource=a.resource, role=a.role,
                  gov_url=a.gov_url, api_key_file=a.api_key_file, units=units,
                  design_sha256=design["design_sha256"], cache_dir=a.cache_dir,
                  expect_sha256=design["governed_bytes"]["sha256"])
    print(json.dumps(doc, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
