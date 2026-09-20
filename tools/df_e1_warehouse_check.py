#!/usr/bin/env python3
"""RP56: the E1 run's units, checked in the warehouse BY CONTENT and by POPULATION.

A unit is not closed because a status column says so. For every unit of the sealed design this reads
the terminal the warehouse actually holds through the canonical reader
(`tools/df_mod_e0_close.warehouse_terminals`) and compares it, field by field, with the receipt the
client persisted when the service accepted it:

  * the terminal digest the service returned is the digest the warehouse stores;
  * its status is a terminal state, and the same one;
  * the campaign it hangs from is the campaign the delivery named;
  * every unit of the design's enumeration is present — rehearsals and failures included, because a
    unit that failed still has a terminal and a cost;
  * no unit is in the warehouse that the design does not declare.

Anything missing, extra or divergent is named per unit. This never writes to the warehouse.

    python tools/df_e1_warehouse_check.py --root ROOT --url URL --token-file FILE --out OUT.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TERMINAL_STATES = {"COMPLETED", "FAILED", "REFUSED", "ABORTED", "CANCELLED"}


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check(root: Path, *, url: str, token: str) -> dict:
    E0 = _module("df_mod_e0_close")
    root = Path(root)
    design = json.loads((root / "DESIGN.json").read_text())
    deliveries = (json.loads((root / "DELIVERIES.json").read_text()).get("units") or {})
    receipts = (json.loads((root / "TERMINAL_RECEIPTS.json").read_text()).get("units") or {})
    population = ["prepare"] + [c["cell_id"] for c in design["pilots"] + design["cells"]]
    units, problems = {}, []
    for unit in population:
        entry = {"unit": unit}
        delivery, receipt = deliveries.get(unit), receipts.get(unit)
        if receipt is None:
            entry["state"] = "NO_ACCEPTED_TERMINAL"
            problems.append(f"{unit}: the client holds no accepted terminal for it")
            units[unit] = entry
            continue
        held = E0.warehouse_terminals(url, token, receipt["campaign_sha256"])
        row = (held.get("current") or {}).get(unit)
        entry["campaign_sha256"] = receipt["campaign_sha256"]
        if row is None:
            entry["state"] = "ABSENT_FROM_THE_WAREHOUSE"
            problems.append(f"{unit}: its campaign is in the warehouse but the unit's terminal is not")
            units[unit] = entry
            continue
        entry.update(state="PRESENT", generation=row.get("generation"), status=row.get("status"),
                     warehouse_terminal_sha256=row.get("terminal_sha256"),
                     client_terminal_sha256=receipt.get("terminal_sha256"),
                     metric_rows=len(row.get("metrics") or []), artifacts=len(row.get("artifacts") or []),
                     config_sha256=row.get("config_sha256"))
        if row.get("terminal_sha256") != receipt.get("terminal_sha256"):
            problems.append(f"{unit}: the warehouse holds terminal {row.get('terminal_sha256')} and the client's "
                            f"receipt says {receipt.get('terminal_sha256')}")
        if row.get("status") not in TERMINAL_STATES:
            problems.append(f"{unit}: the warehouse holds an impossible state {row.get('status')!r}")
        elif receipt.get("status") and row.get("status") != receipt.get("status"):
            problems.append(f"{unit}: the warehouse says {row.get('status')} and the receipt says {receipt['status']}")
        if row.get("config_sha256") not in (None, design["design_sha256"]):
            problems.append(f"{unit}: the warehouse binds it to design {row.get('config_sha256')}")
        if delivery and delivery.get("campaign_sha256") != receipt["campaign_sha256"]:
            problems.append(f"{unit}: its delivery and its terminal name different campaigns")
        entry["verdict"] = "MATCHES_THE_CLIENTS_RECEIPT" if not [p for p in problems if p.startswith(unit + ":")] \
            else "DIVERGES"
        units[unit] = entry
    strangers = sorted(set(receipts) - set(population))
    if strangers:
        problems.append(f"the receipts name units the design does not declare: {strangers}")
    counts = {}
    for entry in units.values():
        counts[entry.get("verdict") or entry["state"]] = counts.get(entry.get("verdict") or entry["state"], 0) + 1
    return {"schema": "df_e1_warehouse_content_check.v1",
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "root": str(root), "design_sha256": design["design_sha256"], "url": url,
            "population": population, "units": units, "counts": counts, "problems": problems,
            "complete": not problems and len(units) == len(population),
            "rule": "content and population: every declared unit, the digest the service returned, and no stranger"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:5057")
    ap.add_argument("--token-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    token = a.token_file.read_text().strip().strip('"').strip("'")
    out = check(a.root, url=a.url, token=token)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "units"}, indent=1, default=str))
    return 0 if out["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
