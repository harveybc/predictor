#!/usr/bin/env python3
"""C120 (order 2026-09-12): the panel inventory for the per-variable bank.

Built without labels, scores or downloads. The scan itself is
`panel_inventory_scan.py`, a read-only draft written for C120 and reviewed
before publication; it is kept verbatim and pinned by digest, so the review
binds the bytes that ran. This wrapper makes its output publishable and
consistent with the member-by-member join:

* it drops the scan's wall-clock field, so the document is reproducible;
* it replaces the scan's ETH successor entry, written while the C114-C116
  artifacts were still being built, with facts from the published v5
  population ledger, after checking that ledger still binds today's census;
* it refuses if the inventory would count a panel the join did not admit;
* it adds the bank verdict, the exact deficit, the unused download
  authorization with its reason, and a shortlist of official sources whose
  licenses are read from the Zenodo community listings already on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCAN = HERE / "panel_inventory_scan.py"
SCAN_SHA256 = "7e52726aa2b0100e7020823bceb0289e2571da405411146cbddb9a8a7c4b0f71"
LEDGER = REPO / "docs/audits/evidence/PER_VARIABLE_DESIGN_V5_POPULATION.v1.json"
LEDGER_LOGICAL = "docs/audits/evidence/PER_VARIABLE_DESIGN_V5_POPULATION.v1.json"
CENSUS_LOGICAL = "features/census/ETH_H4_SUCCESSOR_SEMANTIC_CENSUS.v1.json"
SUCCESSOR_PANEL = "eth_h4_successor"
# Official public non-financial sources named by the inventory review; their
# titles, licenses, sizes and references are read from the listings, never typed.
SHORTLIST_IDS = (4656132, 4656140, 4656719, 5184708, 4659727, 4654909, 4656072)


class InventoryRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_scan():
    if _sha(SCAN.read_bytes()) != SCAN_SHA256:
        raise InventoryRefusal("the scan module is not the reviewed draft")
    return _load(SCAN, "panel_inventory_scan_reviewed")


def successor_facts(v5, scan, ledger: Path) -> tuple[dict, dict]:
    raw = Path(ledger).read_bytes()
    pop = v5.strict_json_loads(raw)
    census = Path(scan.ROOTS["financial-data"]) / CENSUS_LOGICAL
    if _sha(census.read_bytes()) != pop["inputs"]["census"]["sha256"]:
        raise InventoryRefusal("the population ledger does not bind today's successor census")
    rows = pop["ledger"]
    facts = {
        "source": {"file": {"root_id": "predictor", "relative_path": LEDGER_LOGICAL}, "sha256": _sha(raw)},
        "join": "per_variable_design_v5.derive_population, member by member",
        "candidates": pop["candidates"],
        "conditions_true": {c: sum(1 for r in rows if r["conditions"][c]) for c in v5.CONDITIONS},
        "eligible_variables": pop["eligible_variables"],
        "exclusion_reasons": pop["exclusion_reasons"],
        "population_sha256": pop["population_sha256"],
        "ledger_sha256": pop["ledger_sha256"],
    }
    return facts, pop


def shortlist(scan) -> tuple[list, list]:
    root = Path(scan.ROOTS["agent-multi-share"]) / "t2_public_raw"
    found = {}
    for lp in sorted(root.glob("zenodo_forecasting_p*.json")):
        raw = lp.read_bytes()
        doc = json.loads(raw)
        hits = (doc.get("hits") or {}).get("hits", []) if isinstance(doc, dict) else []
        for h in hits:
            rid = h.get("id")
            if rid not in SHORTLIST_IDS or rid in found:
                continue
            md = h.get("metadata") or {}
            creators = [c.get("name") for c in md.get("creators", [])]
            found[rid] = {
                "record": rid, "url": f"https://zenodo.org/records/{rid}", "doi": h.get("doi"),
                "title": md.get("title"),
                "license": (md.get("license") or {}).get("id", "UNKNOWN"),
                "license_scope": "the record's license identifier only; whether it covers the upstream "
                                 "provider's own terms is not verified, so a panel built from it needs "
                                 "license review before it counts",
                "publication_date": md.get("publication_date"),
                "citation": f"{'; '.join(creators)} ({md.get('publication_date')}). {md.get('title')}. "
                            f"Zenodo. https://doi.org/{h.get('doi')}",
                "upstream_references": md.get("references", []),
                "bytes": sum(f.get("size", 0) for f in h.get("files", [])),
                "listing": {"file": {"root_id": "agent-multi-share", "relative_path": f"t2_public_raw/{lp.name}"},
                            "sha256": _sha(raw)},
                "downloaded": False,
            }
    return [found[i] for i in SHORTLIST_IDS if i in found], [i for i in SHORTLIST_IDS if i not in found]


def build(ledger: Path = LEDGER) -> dict:
    scan = load_scan()
    v5 = _load(HERE / "per_variable_design_v5.py", "per_variable_design_v5_for_inventory")
    doc = scan.build()
    doc.pop("generated_at_utc", None)
    facts, pop = successor_facts(v5, scan, ledger)

    succ = next(p for p in doc["panels"] if p["panel_id"] == SUCCESSOR_PANEL)
    succ.pop("in_progress_artifacts_seen_not_read", None)
    succ["artifacts_present_for_join"] = facts
    succ["license"] = "UNKNOWN"
    succ["license_why"] = ("the C115 successor semantic census declares license UNKNOWN with license source NONE "
                           "for all 89 columns; no license document exists for these bytes")
    census = Path(scan.ROOTS["financial-data"]) / CENSUS_LOGICAL
    succ["license_source"] = {"file": {"root_id": "financial-data", "relative_path": CENSUS_LOGICAL},
                              "sha256": _sha(census.read_bytes())}
    why = [w for w in succ["why_not"] if not w.startswith("join artifacts")]
    if pop["eligible_variables"] < v5.MIN_VARIABLES_PER_PANEL:
        why.append(f"member-by-member join: {pop['eligible_variables']} eligible variables "
                   f"(fewer than {v5.MIN_VARIABLES_PER_PANEL})")
    succ["why_not"] = why
    succ["counts_toward_six"] = not why
    doc["counts_toward_six_total"] = sum(p["counts_toward_six"] for p in doc["panels"])

    qualifying = pop["panel_count"]
    if doc["counts_toward_six_total"] > qualifying:
        raise InventoryRefusal("the inventory would count a panel the member-by-member join did not admit")
    listed, not_found = shortlist(scan)
    needs = [{"panel_id": SUCCESSOR_PANEL, "source": "v5 population ledger",
              "conditions_failing": sorted(c for c, n in facts["conditions_true"].items() if n < facts["candidates"])}]
    needs += [{"panel_id": p["panel_id"], "source": "inventory entry",
               "needs": "all eight join artifacts", "license": p["license"],
               "structural_notes": p.get("structural_notes", [])}
              for p in doc["panels"]
              if p["license"].lower().startswith("cc-by") and not p.get("excluded_class")
              and not any(w.startswith("structural") or w.startswith("not independent") or "shares underlying" in w
                          for w in p["why_not"])]
    doc.update({
        "schema": "crispdm.panel_inventory.v1",
        "status": "READ_ONLY_GRANTS_NOTHING",
        "scan": {"module": "tools/panel_inventory_scan.py", "sha256": SCAN_SHA256,
                 "provenance": "read-only draft written for C120 and reviewed before publication"},
        "bank": {
            "verdict": "BANK_SUFFICIENT_FOR_REVIEW" if qualifying >= v5.MIN_PANELS else "BANK_INSUFFICIENT",
            "rule": "a panel qualifies only through the member-by-member join; the inventory never counts "
                    "a candidate the join has not admitted",
            "qualifying_panels": qualifying,
            "deficit": {"panels_required": v5.MIN_PANELS, "panels_qualifying": qualifying,
                        "panels_missing": max(0, v5.MIN_PANELS - qualifying),
                        "eligible_variables_required": v5.MIN_PANELS * v5.MIN_VARIABLES_PER_PANEL,
                        "eligible_variables_in_qualifying_panels": pop["members"]},
            "download_authorization": {
                "granted": "C120: up to 2 GiB of public non-financial datasets from official sources, "
                           "without credentials",
                "used": False, "bytes_downloaded": 0,
                "why_not_used": (
                    "A download adds bytes, not members. A downloaded panel counts only after its own terminals, "
                    "a semantic census with license evidence, a lineage DAG with producer bindings, and a temporal "
                    "contract and mask exist and the join admits five of its variables. None of those exists for "
                    "any candidate except the ETH successor, including the Monash panels already on disk; this "
                    "round built them for the successor only, where every variable is excluded for lack of a "
                    "declared license. No download could change the verdict before those artifacts exist."),
            },
            "closest_paths": needs,
            "shortlist_not_downloaded": listed,
            "shortlist_ids_not_found_on_disk": not_found,
        },
    })
    doc["rules"] = doc["rules"] + [
        "the ETH successor entry is taken from the v5 population ledger, not from the scan",
        "shortlist licenses are read from the Zenodo community listings on disk; nothing was fetched",
    ]
    text = json.dumps(doc, sort_keys=True, default=str)
    if "/home/" in text or str(Path.home()) in text:
        raise InventoryRefusal("absolute home path in the inventory")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise InventoryRefusal(f"{a.out.name} exists; write-once")
    doc = build()
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"panels": doc["panel_count"], "counts_toward_six": doc["counts_toward_six_total"],
                      "verdict": doc["bank"]["verdict"], "deficit": doc["bank"]["deficit"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
