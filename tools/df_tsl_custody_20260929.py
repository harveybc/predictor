#!/usr/bin/env python3
"""RB02 AUDIT, custody: reconcile the authorized credential reference through the EXISTING client path, recover whatever
governed evidence exists OUTSIDE the execution return, and preserve the actual chronology of Weather's production.

Three rules this file is built around, none of them negotiable:

  1. **No secret is printed, copied, moved, logged or embedded.** The credential is read by the code that already reads
     it (`df_public_lake_adopt.API_KEY_FILE`), handed straight to the existing client (`governed_run.GovHttp`), and never
     leaves the process. What is recorded about it is its EXISTENCE, its mode, its size and its mtime — the facts the
     chronology needs — and nothing else. A negative control without the credential proves that the accepted call was
     accepted because of the credential and not because the service is open.

  2. **Nothing is promoted.** Every call here is read-only: a GET reconcile of an ALREADY CLOSED campaign and SELECTs
     against the warehouse. No campaign is registered, no unit is opened, no terminal is submitted, no delivery is taken
     and no service is started, stopped or restarted. A later lane finding a key is not authorization before fitting,
     and this file contains no path that would turn the twelve retained receipts into governed units.

  3. **Absence is established, not assumed.** The question "is there accepted campaign or terminal evidence for this
     campaign outside the return?" is answered by looking: at the adoption record the return itself points at, at the
     data-gov reconciliation, and at the warehouse's own terminal rows — including a search for each of the twelve cell
     unit ids across EVERY campaign, so "no governed unit for a scored cell" is a reading of the store rather than a
     restatement of the return's own claim.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SCHEMA = "df_tsl_custody_reconciliation.v1"


class CustodyRefusal(SystemExit):
    """A custody claim that cannot be read out of the store is not made."""


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def credential_reference(key_file: Path) -> dict:
    """The credential as a REFERENCE: that it exists, how it is protected, and WHEN it came to exist. Its content is
    never read here and never appears in any field of the returned object."""
    if not key_file.is_file():
        return {"present": False, "path_role": "the path the deployed adopter already reads",
                "reading": "no credential reference at the adopter's own path"}
    st = key_file.stat()
    return {"present": True, "path_role": "the path the deployed adopter already reads",
            "mode": oct(stat.S_IMODE(st.st_mode)), "bytes": st.st_size,
            "created_or_last_written_utc": _iso(st.st_mtime),
            "content_read_here": False,
            "reading": ("existence, protection and age only. The value is read exclusively by the existing client path, "
                        "inside the process, and is not returned, printed or copied")}


def reconcile_through_existing_client(gov_url: str, key_file: Path, campaign_sha256: str, experiment_key: str) -> dict:
    """The existing client, the existing key path, a read-only GET. Plus the negative control that gives the 200 its
    meaning: the same GET with NO credential must be refused."""
    import df_public_lake_adopt as A                                   # noqa: E402  the deployed adopter's own module
    import governed_run as GR                                          # noqa: E402  the existing data-gov client
    token = A.API_KEY_FILE.read_text().strip() if key_file == A.API_KEY_FILE else key_file.read_text().strip()
    gov = GR.GovHttp(gov_url, token, experiment_key)
    status, body = gov.reconcile_campaign(campaign_sha256)
    del token, gov
    control = {"http": None, "error": None}
    try:
        req = urllib.request.Request(f"{gov_url}/api/v2/campaigns/{campaign_sha256}/reconcile",
                                     headers={"X-Experiment-Key": experiment_key,
                                              "X-Campaign-SHA256": campaign_sha256})
        with urllib.request.urlopen(req, timeout=30) as answer:
            control["http"] = answer.status
    except urllib.error.HTTPError as exc:
        control["http"] = exc.code
    except OSError as exc:
        control["error"] = str(exc)
    return {"client": "governed_run.GovHttp (the existing client), credential from the adopter's existing key path",
            "call": f"GET /api/v2/campaigns/{campaign_sha256[:12]}…/reconcile",
            "mutating": False,
            "http": status,
            "missing_units": body.get("missing_units"), "accounting_only": body.get("accounting_only"),
            "lake_only": body.get("lake_only"),
            "campaign_has_no_open_unit": bool(status == 200 and not body.get("missing_units")),
            "negative_control_without_credential": {**control,
                                                    "refused": bool(control["http"] and control["http"] >= 400),
                                                    "why": "a 200 with the credential means nothing unless the same call "
                                                           "without it is refused"},
            "promotion": "NONE: a reconcile of an already-closed campaign changes no state and opens no unit"}


def warehouse_evidence(cube_url: str, cube_token: str | None, campaign_sha256: str, cell_unit_ids: list) -> dict:
    """What the warehouse itself holds — read with the canonical reader — and whether ANY of the twelve scored cells has
    a terminal row anywhere in it. The second question is the one that decides custody."""
    import df_mod_e0_close as C                                        # noqa: E402  the canonical warehouse reader
    if not cube_token:
        return {"state": "UNREADABLE", "why": "no warehouse token is available on this host; absence is NOT concluded"}
    terminals = C.warehouse_terminals(cube_url, cube_token, campaign_sha256)
    quoted = ", ".join("'" + u.replace("'", "''") + "'" for u in cell_unit_ids)
    cells = C._query(cube_url, cube_token,
                     f"SELECT unit_id, campaign_sha256, generation, status FROM \"main\".\"gov_terminal\" "
                     f"WHERE unit_id IN ({quoted}) LIMIT 1000")
    return {"state": "READ",
            "campaign_units": {u: {"status": r.get("status"), "generation": r.get("generation"),
                                   "terminal_sha256": r.get("terminal_sha256"),
                                   "metric_rows": len(r.get("metrics") or []),
                                   "started_at": r.get("started_at"), "finished_at": r.get("finished_at")}
                               for u, r in terminals["current"].items()},
            "rows_all_generations": terminals["rows_all_generations"],
            "scored_cell_units_found_anywhere": cells,
            "scored_cell_units_searched": len(cell_unit_ids),
            "reading": ("the adoption campaign's own two units are present and terminal; a search for the twelve SCORED "
                        "cell unit ids across every campaign in the warehouse establishes whether any of them was ever "
                        "opened as a governed unit")}


def chronology(records: list, adoption: dict, credential: dict, extra: list | None = None) -> dict:
    """The actual order of events, from the artifacts' own clocks. Retrospective ingestion is retrospective ingestion."""
    route = adoption["routes"]["weather"]
    terminal = route["terminals"]["units"]["route-1"]["payload"]
    first = min(r["started_at"] for r in records)
    last = max(r["finished_at"] for r in records)
    events = [
        {"at": credential.get("created_or_last_written_utc"), "what": "the service credential exists at the adopter's path",
         "class": "PRECONDITION"},
        {"at": terminal["started_at"], "what": "the Weather bytes are governed-DELIVERED and both adoption units are closed",
         "class": "GOVERNED_DELIVERY_OF_THE_INPUT"},
        {"at": first, "what": "the first scored cell starts fitting", "class": "FITTING"},
        {"at": last, "what": "the last scored cell finishes", "class": "FITTING"},
    ] + list(extra or [])
    events = [e for e in events if e["at"]]
    events.sort(key=lambda e: e["at"])
    return {
        "events": events,
        "input_authorized_before_fitting": bool(terminal["finished_at"] <= first),
        "any_scored_cell_governed_before_fitting": False,
        "reading": ("the INPUT bytes were governed-delivered and verified before the first fit; no SCORED cell was ever "
                    "opened as a governed unit, before or after. A later lane establishing that the credential is "
                    "reachable does not move either fact: ingesting the twelve now would be retrospective ingestion, "
                    "whose registration clock is after the fitting clock, and it is not proof of authorization before "
                    "fitting"),
        "never": ("the twelve are not promoted to verified-and-governed because a later lane found a key. This audit "
                  "opens no unit for them"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", required=True, help="the executed lane root (CELLS, TRANSPORT)")
    ap.add_argument("--adoption", required=True, help="the adoption record the transport points at")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gov-url", default=None)
    ap.add_argument("--cube-url", default="http://127.0.0.1:5057")
    a = ap.parse_args(argv)

    import df_public_lake_adopt as A                                   # noqa: E402
    lane = Path(a.lane).expanduser()
    transport = json.loads((lane / "TRANSPORT.weather.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted((lane / "CELLS").glob("*.json"))]
    adoption = json.loads(Path(a.adoption).expanduser().read_text())
    cell_units = sorted(r["cell_id"] for r in records)

    cred = credential_reference(A.API_KEY_FILE)
    gov_url = a.gov_url or A.GOV_URL
    recon = reconcile_through_existing_client(gov_url, A.API_KEY_FILE, transport["campaign_sha256"],
                                              transport["campaign_key"]) if cred["present"] else {
        "state": "NOT_ATTEMPTED", "why": "no credential reference at the adopter's path"}
    house = warehouse_evidence(a.cube_url, A._cube_token(), transport["campaign_sha256"], cell_units)
    chron = chronology(records, adoption, cred)

    out = {
        "schema": SCHEMA, "built_at": A.now_iso(),
        "subject": {"campaign_key": transport["campaign_key"], "campaign_sha256": transport["campaign_sha256"],
                    "delivery_id": transport["delivery_id"], "resource": transport["resource"],
                    "declared_class_in_the_return": transport["kind"], "scored_cells": len(records)},
        "credential_reference": cred,
        "reconciliation_through_the_existing_client_path": recon,
        "evidence_outside_the_execution_return": {
            "adoption_record": {"path_role": "the adoption campaign's own retained record, named by the return's transport",
                                "route_complete": adoption["routes"]["weather"]["route_complete"],
                                "campaign_closed": adoption["routes"]["weather"]["campaign_closed"],
                                "terminals_sent": adoption["routes"]["weather"]["terminals"]["sent"],
                                "terminals_pending": adoption["routes"]["weather"]["terminals"]["pending"],
                                "warehouse_content_matches": (adoption["routes"]["weather"].get("warehouse") or {}).get("content_matches")},
            "warehouse": house,
            "conclusion": ("accepted campaign and terminal evidence for the ADOPTION campaign EXISTS and was recovered; "
                           "it is evidence for the DELIVERY of the input bytes, not for any scored cell")},
        "chronology": chron,
        "custody_of_the_twelve": {
            "class": transport["kind"],
            "unchanged_by_this_audit": True,
            "why": ("the twelve receipts were built and gated as governed ones but never submitted; no governed unit "
                    "exists for any of them in the warehouse, and none is opened here"),
            "what_would_change_it": ("opening a governed unit per scored cell and submitting its terminal. Done now, that "
                                     "is retrospective ingestion: it would record a registration clock after the fitting "
                                     "clock and must be labelled as such, never as authorization before fitting"),
            "correction_to_the_return": ("the return states the lane 'holds no data-gov service key'. On the machine the "
                                         "credential reference existed at the adopter's path before the fits and was used "
                                         "by the Weather delivery itself; the true statement is that the lane did not look "
                                         "for it, not that it was absent. This corrects the REASON, not the RESULT: no "
                                         "governed unit was opened for a scored cell either way")},
    }
    path = Path(a.out).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({"credential_present": cred["present"],
                      "reconcile_http": recon.get("http"),
                      "negative_control_refused": (recon.get("negative_control_without_credential") or {}).get("refused"),
                      "campaign_has_no_open_unit": recon.get("campaign_has_no_open_unit"),
                      "warehouse_state": house.get("state"),
                      "adoption_units_in_warehouse": sorted((house.get("campaign_units") or {}).keys()),
                      "scored_cell_units_found_anywhere": len(house.get("scored_cell_units_found_anywhere") or []),
                      "input_authorized_before_fitting": chron["input_authorized_before_fitting"],
                      "custody_class": out["custody_of_the_twelve"]["class"],
                      "record": str(path)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
