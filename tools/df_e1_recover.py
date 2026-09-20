#!/usr/bin/env python3
"""RP55: rebuild a run's local acquisition receipt from the governance service's own record.

Why this exists. `DELIVERIES.json` was written by a read-modify-write, so three children acquiring at
once overwrote each other and four units lost their entry — while the service still held their
campaign and their verified delivery. The write is now serialised (tools/df_e1_governed._merge_write),
and this tool repairs the roots that the defect already damaged.

Nothing here is invented. Each restored entry is copied from the service's accounting record for
THIS campaign and THIS unit, and is accepted only when

  * the campaign exists under the key the run's own scheme produces, for this design's config digest;
  * the unit has exactly one delivery there, in a verified state;
  * the delivered bytes are on this host and their digest is the digest the service recorded.

A unit whose delivery the service does not hold is NOT restored: it must be acquired again.
A unit that already has a local entry is left exactly as it is, and no terminal is ever invented here:
the report of a terminal remains the runner's own act.

    python tools/df_e1_recover.py --root ROOT --accounting-db PATH [--apply]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "restored from the governance service's own accounting record after a client-side receipt loss"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _open(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)          # read-only: the service owns it
    con.row_factory = sqlite3.Row
    return con


def service_record(con: sqlite3.Connection, campaign_key: str) -> dict | None:
    row = con.execute("SELECT campaign_sha256, body_json FROM governed_campaigns WHERE campaign_key = ?",
                      (campaign_key,)).fetchone()
    if row is None:
        return None
    body = json.loads(row["body_json"])
    deliveries = [dict(r) for r in con.execute(
        "SELECT * FROM governed_deliveries WHERE campaign_sha256 = ?", (row["campaign_sha256"],))]
    terminals = [dict(r) for r in con.execute(
        "SELECT unit_id, generation, status, terminal_sha256 FROM governed_terminals WHERE campaign_sha256 = ?",
        (row["campaign_sha256"],))]
    return {"campaign_sha256": row["campaign_sha256"], "body": body,
            "deliveries": deliveries, "terminals": terminals}


def plan(root: Path, db: Path, *, cache_dir: Path | None = None) -> dict:
    root = Path(root)
    design = json.loads((root / "DESIGN.json").read_text())
    doc = json.loads((root / "DELIVERIES.json").read_text())
    units = ["prepare"] + [c["cell_id"] for c in design["pilots"] + design["cells"]]
    cache = Path(cache_dir) if cache_dir else root / "cache" / doc["lake"]
    con = _open(Path(db))
    restore, keep, absent, refused, held = {}, [], [], {}, {}
    for unit in units:
        if unit in (doc.get("units") or {}):
            keep.append(unit)
            continue
        key = f"{doc['run_id']}-{unit}-data"
        record = service_record(con, key)
        if record is None:
            absent.append(unit)
            continue
        if record["body"].get("config_sha256") != design["design_sha256"]:
            refused[unit] = "the service's campaign belongs to another design"
            continue
        rows = [d for d in record["deliveries"] if d["unit_id"] == unit]
        if len(rows) != 1:
            refused[unit] = f"the service holds {len(rows)} deliveries for this unit, not one"
            continue
        row = rows[0]
        if not str(row["state"]).startswith("VERIFIED"):
            refused[unit] = f"the delivery is in state {row['state']}, not verified"
            continue
        path = cache / f"{row['sha256']}.parquet"
        if not path.is_file():
            refused[unit] = f"the delivered bytes are not on this host at {path}"
            continue
        on_disk = sha_file(path)
        if on_disk != row["sha256"]:
            refused[unit] = "the bytes on disk are not the bytes the service delivered"
            continue
        restore[unit] = {
            "campaign_key": key, "campaign_sha256": record["campaign_sha256"],
            "code_identity": record["body"].get("code_identity"),
            "at": row["verified_at"] or row["created_at"], "host": doc.get("host"),
            "delivery_id": row["delivery_id"], "sha256": row["sha256"], "bytes": row["bytes"],
            "cached": bool(row["cached"]), "verification_state": row["state"],
            "availability_contract_sha256": row["availability_contract_sha256"],
            "path": str(path), "bytes_on_disk_sha256": on_disk,
            "restored": SOURCE,
        }
        held[unit] = [{"generation": t["generation"], "status": t["status"],
                       "terminal_sha256": t["terminal_sha256"]} for t in record["terminals"]
                      if t["unit_id"] == unit]
    con.close()
    return {"root": str(root), "restore": restore, "already_present": keep, "unknown_to_the_service": absent,
            "refused": refused, "terminals_the_service_already_holds": held,
            "rule": "a delivery is copied from the service's record; a terminal is never written here"}


def apply(root: Path, db: Path, *, cache_dir: Path | None = None) -> dict:
    spec = importlib.util.spec_from_file_location("df_e1_governed",
                                                  Path(__file__).resolve().parent / "df_e1_governed.py")
    gov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gov)
    decided = plan(root, db, cache_dir=cache_dir)
    path = Path(root) / "DELIVERIES.json"
    for unit, entry in decided["restore"].items():
        base = json.loads(path.read_text())
        gov._merge_write(path, unit, entry, base)                   # the same serialised write the runner uses
    decided["applied_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    decided["applied"] = sorted(decided["restore"])
    return decided


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--accounting-db", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    out = (apply if a.apply else plan)(a.root, a.accounting_db, cache_dir=a.cache_dir)
    print(json.dumps(out, indent=1, default=str))
    return 1 if out["refused"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
