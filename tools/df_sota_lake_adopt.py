#!/usr/bin/env python3
"""RP92 (SOTA-first): publish the OFFICIAL processed ECL benchmark (Time-Series-Library `electricity.csv`, the file the author
scripts read) as a governed resource, by the SAME tested operating procedure as the public panels (tools/df_public_lake_adopt.py):
inventory -> backup -> rehearsal on a disposable stack -> additive change -> restart -> post-check through the live route ->
write-once receipt, with the rollback the receipt names. Nothing here is a new procedure: this module REBINDS the public-panel
adopter to another lake (its own external host, unit and port) and another catalogue (the benchmark store's BUILD_RECEIPT),
so every rule that adopter enforces — additive change, rehearsal-bound bytes, failed post-check => rollback, no second adoption
of a live entry — applies unchanged.

What is published and what is NOT:
  * one resource: `thuml_tsl_electricity/electricity.csv`, the bytes whose sha256 equals the Hugging Face LFS object of the
    pinned dataset revision (BUILD_RECEIPT.json in the store root records origin, revision, digest, rows, columns, labels);
  * `untimed`: its labels are the distributor's wall-clock strings, not evidence of when a row became available; delivery is
    whole-resource AS_IS with an UNDECLARED scope; every date range is refused (holdout from the data's own first day);
  * use class BENCHMARK/PUBLIC; no financial data, no private holdout, no point-in-time claim.

    python tools/df_sota_lake_adopt.py inventory --out INVENTORY.json
    python tools/df_sota_lake_adopt.py contract  --out CONTRACT.json
    python tools/df_sota_lake_adopt.py rehearse  --out REHEARSAL.json
    python tools/df_sota_lake_adopt.py adopt     --state-dir DIR --rehearsal REHEARSAL.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path.home()
STORE_ROOT = HOME / ".local/state/crispdm-data-foundation/sota_benchmarks_v1"
LAKE_ID = "sota_benchmarks"
LAKE_HOST_PORT = 5060
LAKE_HOST_UNIT = "crispdm-data-lake-sota-benchmarks.service"
RESOURCES = {"thuml_tsl_electricity/electricity.csv": "thuml_tsl_electricity"}
TITLE = "SOTA benchmarks (official processed ECL, Time-Series-Library) — public benchmark archive"
DESCRIPTION = ("Read-only lake over the official processed public benchmarks the author code reads. BENCHMARK/PUBLIC use: "
               "whole-resource AS_IS deliveries only; every date range is refused; availability UNDECLARED.")


def _load_public_adopter():
    """A PRIVATE instance of the public-panel adopter module: rebinding its globals must never leak into the public-panel
    adopter that other code (and its tests) use in the same process."""
    name = "df_public_lake_adopt__sota_instance"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "df_public_lake_adopt.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def receipt() -> dict:
    doc = json.loads((STORE_ROOT / "BUILD_RECEIPT.json").read_text())
    return {r["resource"]: r for r in doc["results"] if r.get("status") == "BUILT"}


def first_day() -> str:
    """The archive's own first day: the holdout that withholds every date-ranged request (the public-panel rule, RP41)."""
    return min(r["first_label"][:10] for r in receipt().values())


def resource_contract(resource: str) -> dict:
    r = receipt()[resource]
    return {"event_time_column": r["time_column"], "available_time_column": r["time_column"], "timezone": "NAIVE_WALL_CLOCK",
            "time_unit": None, "frequency": f"{r['step_seconds_unique'][0]}s"}


def lake_entry() -> dict:
    """The lake entry and the declared sheet: bytes bound to the store receipt (whose digest equals the LFS object)."""
    rec = receipt()
    contracts, declared = {}, {}
    for resource in RESOURCES:
        r = rec.get(resource)
        if r is None:
            raise SystemExit(f"REFUSED: {resource} is not BUILT in {STORE_ROOT/'BUILD_RECEIPT.json'}")
        path = STORE_ROOT / resource
        if not path.is_file():
            raise SystemExit(f"REFUSED: {path} is not a file")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != r["sha256"] or not r.get("digest_matches_lfs_oid"):
            raise SystemExit(f"REFUSED: {resource} bytes are not the receipted official ones")
        contracts[resource] = resource_contract(resource)
        declared[resource] = {"dataset_id": r["dataset_id"], "sha256": digest, "bytes": path.stat().st_size, "rows": r["rows"], "columns": r["columns"],
                              "channels": r["channels"], "provenance": r["provenance"], "first_label": r["first_label"], "last_label": r["last_label"],
                              "label_format": r["label_format"], "licence": r["provenance"]["licence"], "use_class": r["use_class"],
                              "availability": "UNDECLARED: the labels are data, not evidence of publication time; no completion lag, "
                                              "no point-in-time and no live-equivalent eligibility",
                              "time_zone": "not stated by the distributor; the author code never interprets it as availability"}
    holdout = first_day()
    entry = {"plugin": "files_lake", "lake_id": LAKE_ID, "title": TITLE, "description": DESCRIPTION, "kind": "lake",
             "engine": "files_inventory", "root_path": str(STORE_ROOT), "include_globs": sorted(RESOURCES), "untimed": sorted(RESOURCES),
             "time_column": None, "time_columns": {}, "time_unit": None, "resource_contracts": contracts, "holdout_start": holdout}
    return {"entry": entry, "declared": declared}


def bind(A=None):
    """Rebind the public-panel adopter's module globals to THIS lake. Every function of that module resolves its globals at
    call time, so inventory/rehearse/adopt/_restore run unchanged against the benchmark lake."""
    A = A or _load_public_adopter()
    holdout = first_day()
    A.LAKE_ID = LAKE_ID
    A.RESOURCES = dict(RESOURCES)
    A.PANEL_ROOT = STORE_ROOT
    A.ARCHIVE_FIRST_DAY = {fam: holdout for fam in RESOURCES.values()}
    A.HOLDOUT_START = holdout
    A.LAKE_HOST_PORT = LAKE_HOST_PORT
    A.LAKE_HOST_UNIT = LAKE_HOST_UNIT
    A.lake_entry = lake_entry
    A.resource_contract = lambda family: resource_contract(next(r for r, f in RESOURCES.items() if f == family))

    def lake_host_config(*, port: int, state_dir: Path, token_file: Path | None = None) -> dict:
        built = lake_entry()
        return {"store_id": LAKE_ID, "title": TITLE, "description": DESCRIPTION, "kind": "lake", "engine": "files_inventory",
                "transport": "http", "web_host": "127.0.0.1", "web_port": int(port),
                "operator_config_path": str(Path(state_dir) / "public-panels.pending.json"),
                "backend": {"entry_point": "financial_files", "distribution": "financial-data-store",
                            "settings": {"root_path": str(STORE_ROOT), "include_globs": sorted(RESOURCES), "untimed": sorted(RESOURCES),
                                         "holdout_start": holdout, "holdout_reason": A.HOLDOUT_REASON,
                                         "resource_contracts": built["entry"]["resource_contracts"]}}}
    A.lake_host_config = lake_host_config

    def lake_entry_http(port: int, token: str | None = None) -> dict:
        entry = {"plugin": "http_lake", "lake_id": LAKE_ID, "title": TITLE, "description": DESCRIPTION, "kind": "lake",
                 "engine": "files_inventory", "base_url": f"http://127.0.0.1:{int(port)}", "holdout_start": holdout}
        if token:
            entry["lake_service_token"] = token
        return entry
    A.lake_entry_http = lake_entry_http
    # the binding names THIS module too: it is code that answers the route
    original_binding = A.rehearsal_binding

    def rehearsal_binding(_unused, after_cfg: dict) -> dict:
        out = original_binding(_unused, after_cfg)
        out["serving_tools_sha256"] = {**out.get("serving_tools_sha256", {}), "df_sota_lake_adopt.py": A.sha_file(Path(__file__).resolve())}
        out["lake_id"] = LAKE_ID
        return out
    A.rehearsal_binding = rehearsal_binding
    # inventory: the registration flag is this lake's
    original_inventory = A.inventory

    def inventory() -> dict:
        out = original_inventory()
        out["schema"] = "df_sota_lake_inventory.v1"
        out["benchmark_present"] = {r: (STORE_ROOT / r).is_file() for r in RESOURCES}
        out["benchmark_lake_registered"] = any(l.get("lake_id") == LAKE_ID for l in out["lakes"])
        return out
    A.inventory = inventory
    # default arguments were bound at the public module's definition time (port 5059): rebind them to THIS lake's port
    import functools
    original_adopt, original_deployed, original_route = A.adopt, A.deployed_external, A.route_checks
    A.adopt = functools.partial(original_adopt, lake_port=LAKE_HOST_PORT)
    A.deployed_external = functools.partial(original_deployed, port=LAKE_HOST_PORT)
    A.route_checks = functools.partial(original_route, lake=LAKE_ID)          # the route registers its campaign on THIS lake
    return A


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["inventory", "contract", "rehearse", "adopt"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--state-dir", type=Path, default=None)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--principals", nargs="*", default=["predictor", "satoshi-gamma", "satoshi-dragon"])
    ap.add_argument("--rehearsal", type=Path, default=None)
    ap.add_argument("--lake-port", type=int, default=LAKE_HOST_PORT)
    a = ap.parse_args(argv)
    A = bind()
    if a.command == "inventory":
        doc = A.inventory()
    elif a.command == "contract":
        doc = lake_entry()
    elif a.command == "rehearse":
        doc = A.rehearse(a.out or Path("REHEARSAL.json"), keep=a.keep)
    else:
        if a.state_dir is None:
            raise SystemExit("REFUSED: --state-dir is required")
        doc = A.adopt(a.state_dir, principals=a.principals, rehearsal=a.rehearsal, lake_port=a.lake_port)
        doc["schema"] = "df_sota_lake_adoption.v1"
        (a.state_dir / "RECEIPT.json").write_text(json.dumps(doc, indent=1, default=str))
    if a.out and a.command != "rehearse":
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({k: doc.get(k) for k in ("adopted", "refused", "route_ok", "benchmark_lake_registered", "failure") if k in doc} or
                     {"entry": doc.get("entry", {}).get("lake_id")}, indent=1, default=str))
    return 0 if (a.command != "adopt" or doc.get("adopted")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
