#!/usr/bin/env python3
"""C39-C41 (order 2026-09-11): characterization bound to bytes, with a
public pilot and a deterministic pass over the lake.

The C30 pilot produced 420 honest numbers that could not be checked:
no source, no digest, no window, no code, no protocol, no terminal. It
also read the first N columns BY POSITION, which pulled `DATE_TIME`
into the numeric universe and gave a timestamp a mean and a spectrum.

This is a SEPARATE producer, deliberately. The pilot's rows are never
edited or deleted; its tool is left exactly as it was, and everything
here is a new version beside it.

What changed:

  * every row binds source id and byte digest, the exact window and its
    digest, the executing code, the protocol, the side/role and units,
    and the terminal attempt it was born from;
  * temporal identifiers are EXCLUDED from the numeric universe and
    recorded as an axis contract instead;
  * three banks stay apart, and the public pilot uses only material
    the T2 manifest excludes from confirmation, so no confirmatory
    budget is spent;
  * the ledger reports coverage by CONCEPTUAL VARIABLE and by PHYSICAL
    APPEARANCE, because they are different populations and reporting
    one as the other is how 7,860 appearances once looked like 7,860
    variables. Every subject gets MEASURED, NOT_IDENTIFIABLE,
    UNAVAILABLE or FAILED — absence is a result.

CPU only. No selection, no confirmation, no GPU.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from olap import characterization as ch                      # noqa: E402
from eligibility.consumed import (NON_FEATURE_COLUMNS,        # noqa: E402
                                  code_identity,
                                  local_code_surface)

MEASURED = "MEASURED"
NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
UNAVAILABLE = "UNAVAILABLE"
FAILED = "FAILED"

#: the public material a pilot may touch: the T2 manifest's own
#: exclusion from confirmation is the guarantee that measuring it
#: spends no confirmatory budget.
PUBLIC_NON_CONFIRMATORY = "EXCLUDED_FROM_T2_CONFIRMATORY"

DEV_FRACTION = 0.7


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_obj(o) -> str:
    return hashlib.sha256(
        json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def is_temporal(column: str) -> bool:
    c = column.strip().lower()
    return (column in NON_FEATURE_COLUMNS or c in
            {"date", "datetime", "date_time", "timestamp", "time",
             "index", "period"})


def read_columns(path: Path):
    """Header plus per-column values, with temporal columns kept OUT of
    the numeric universe and returned as the axis instead."""
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        cols = {c: [] for c in header}
        for row in reader:
            for c in header:
                cols[c].append(row.get(c))
    axis = [c for c in header if is_temporal(c)]
    numeric = {}
    for c in header:
        if c in axis:
            continue
        vals = []
        for raw in cols[c]:
            try:
                vals.append(float(raw))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        numeric[c] = vals
    return header, axis, numeric, {c: cols[c] for c in axis}


def axis_contract(path: Path, axis_cols: list[str],
                  axis_values: dict) -> dict:
    """A timestamp is not a variable. It is the axis the variables are
    observed on, and what it deserves is a contract, not a mean."""
    if not axis_cols:
        return {"axis_columns": [], "state": "NO_TEMPORAL_IDENTIFIER",
                "note": "no column in this file identifies a row in "
                        "time; every column was treated as numeric"}
    col = axis_cols[0]
    vals = [v for v in axis_values[col] if v not in (None, "")]
    return {
        "axis_columns": axis_cols,
        "state": "DECLARED_AS_AXIS_NOT_MEASURED",
        "rows": len(vals),
        "first": vals[0] if vals else "UNAVAILABLE",
        "last": vals[-1] if vals else "UNAVAILABLE",
        "monotonic_non_decreasing": all(
            vals[i] <= vals[i + 1] for i in range(len(vals) - 1)),
        "axis_sha256": sha_obj(vals),
        "note": "excluded from the numeric universe: a mean, a "
                "spectrum or an entropy of a timestamp describes the "
                "calendar, not the variable",
    }


def window(values, *, fraction: float) -> tuple[list, dict]:
    """The development window, declared exactly."""
    n = len(values)
    end = int(n * fraction)
    contract = {
        "rule": f"first {fraction:.0%} of rows in file order",
        "rows_total": n, "rows_used": end,
        "index_start": 0, "index_end_exclusive": end,
    }
    return values[:end], contract


def make_binding(*, source_path: Path, source_sha: str,
                 window_contract: dict, code_digest: str,
                 side: str, contract_role: str, units: str,
                 terminal_attempt: str) -> dict:
    return {
        "source_id": source_path.name,
        "source_sha256": source_sha,
        "window_sha256": sha_obj(window_contract),
        "window_contract": json.dumps(window_contract, sort_keys=True),
        "code_identity": code_digest,
        "protocol_version": ch.PROTOCOL_VERSION,
        "side": side,
        "contract_role": contract_role,
        "units": units,
        "terminal_attempt": terminal_attempt,
    }


# --------------------------------------------------------------- banks
def synthetic_subjects(count: int, length: int):
    import numpy as np
    out = []
    for i in range(count):
        rng = np.random.default_rng(1000 + i)
        t = np.arange(length)
        clean = np.sin(2 * np.pi * t / (24 + 8 * i))
        noise = rng.normal(0, 0.2 + 0.1 * i, length)
        out.append({
            "variable_id": f"synthetic::sine{24 + 8 * i}::seed{1000 + i}",
            "values": (clean + noise).tolist(),
            "reference": clean.tolist(),
            "partition_key": "synthetic::development",
            "bank": ch.BANK_SYNTHETIC,
            "units": "dimensionless"})
    return out


def public_subjects(manifest_path: Path, raw_root: Path):
    """Only material the T2 manifest EXCLUDES from confirmation."""
    if not manifest_path.is_file():
        return [], {"state": UNAVAILABLE,
                    "reason": "no public manifest on this host"}
    doc = json.loads(manifest_path.read_text())
    datasets = doc.get("datasets", {})
    eligible, skipped = [], []
    for logical_id, entry in sorted(datasets.items()):
        admission = entry.get("admission")
        rel = entry.get("local_relpath", "")
        if admission != PUBLIC_NON_CONFIRMATORY:
            skipped.append({"dataset": logical_id,
                            "admission": admission,
                            "reason": "confirmatory-eligible material; "
                                      "measuring it here would spend "
                                      "confirmation"})
            continue
        path = raw_root / rel
        if not path.is_file() or path.suffix.lower() != ".csv":
            skipped.append({"dataset": logical_id,
                            "admission": admission,
                            "reason": "not a plain CSV this pilot "
                                      "reads"})
            continue
        eligible.append({"dataset": logical_id, "path": path,
                         "declared_sha256": entry.get("sha256")})
    return eligible, {"state": "PILOT",
                      "eligible": len(eligible),
                      "skipped": skipped}


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measured-at", required=True)
    ap.add_argument("--ledger", required=True, type=Path)
    ap.add_argument("--terminal-attempt", required=True)
    ap.add_argument("--public-manifest", type=Path, default=(
        Path.home() / ".local/share/agent-multi"
        / "t2_public_data_manifest_20260906.json"))
    ap.add_argument("--public-raw-root", type=Path, default=(
        Path.home() / ".local/share/agent-multi/t2_public_raw"))
    ap.add_argument("--financial-inventory", type=Path, default=(
        REPO / "examples/research/crispdm_dataset_inventory.v1.json"))
    ap.add_argument("--financial-columns", type=int, default=0,
                    help="0 = every non-temporal column")
    ap.add_argument("--synthetic-generators", type=int, default=3)
    ap.add_argument("--synthetic-length", type=int, default=512)
    ap.add_argument("--load", action="store_true")
    a = ap.parse_args(argv)

    started = time.perf_counter()
    code_digest, _inv = code_identity(
        REPO, local_code_surface(REPO), {})
    attempts, rows = [], []
    axis_contracts = []

    # ---- synthetic: the only bank where noise is identifiable ----
    for s in synthetic_subjects(a.synthetic_generators,
                                a.synthetic_length):
        binding = make_binding(
            source_path=Path(f"{s['variable_id']}.generated"),
            source_sha=sha_obj(s["values"]),
            window_contract={"rule": "generated in full",
                             "rows_total": len(s["values"]),
                             "rows_used": len(s["values"])},
            code_digest=code_digest, side="x",
            contract_role="input", units=s["units"],
            terminal_attempt=a.terminal_attempt)
        r = ch.characterize_series(
            s["values"], variable_id=s["variable_id"],
            partition_key=s["partition_key"],
            bank_authority=s["bank"], measured_at=a.measured_at,
            binding=binding, noise_reference=s["reference"])
        rows.extend(r)
        attempts.append({"variable_id": s["variable_id"],
                         "bank": s["bank"], "outcome": MEASURED,
                         "descriptors": len(r),
                         "noise_identifiable": True,
                         "appearance_id": s["variable_id"]})

    # ---- public pilot ----
    public, public_state = public_subjects(a.public_manifest,
                                           a.public_raw_root)
    for ds in public:
        path = ds["path"]
        try:
            source_sha = sha_file(path)
            _header, axis, numeric, axis_values = read_columns(path)
        except Exception as exc:                      # noqa: BLE001
            attempts.append({"variable_id": ds["dataset"],
                             "bank": ch.BANK_PUBLIC, "outcome": FAILED,
                             "reason": exc.__class__.__name__})
            continue
        contract = axis_contract(path, axis, axis_values)
        contract["dataset"] = ds["dataset"]
        axis_contracts.append(contract)
        if ds["declared_sha256"] and \
                ds["declared_sha256"] != source_sha:
            attempts.append({"variable_id": ds["dataset"],
                             "bank": ch.BANK_PUBLIC, "outcome": FAILED,
                             "reason": "the file on disk does not match "
                                       "the digest the manifest declares"})
            continue
        for column, values in sorted(numeric.items()):
            used, wcontract = window(values, fraction=DEV_FRACTION)
            binding = make_binding(
                source_path=path, source_sha=source_sha,
                window_contract=wcontract, code_digest=code_digest,
                side="x", contract_role="input", units="UNDECLARED",
                terminal_attempt=a.terminal_attempt)
            vid = f"{ds['dataset']}::{column}"
            try:
                r = ch.characterize_series(
                    used, variable_id=vid,
                    partition_key=f"{ds['dataset']}::development",
                    bank_authority=ch.BANK_PUBLIC,
                    measured_at=a.measured_at, binding=binding)
            except SystemExit as exc:
                attempts.append({"variable_id": vid,
                                 "bank": ch.BANK_PUBLIC,
                                 "outcome": FAILED,
                                 "reason": str(exc)[:120]})
                continue
            rows.extend(r)
            identifiable = [x for x in r if x["identifiable"]]
            attempts.append({
                "variable_id": vid, "bank": ch.BANK_PUBLIC,
                "outcome": MEASURED if identifiable
                else NOT_IDENTIFIABLE,
                "descriptors": len(r),
                "not_identifiable": len(r) - len(identifiable),
                "noise_identifiable": False,
                "appearance_id": f"{ds['dataset']}::{path.name}",
                "partition": "development",
                "confirmatory_material": False})

    # ---- financial: development only, by ROLE, never by position ----
    financial_state = {"state": UNAVAILABLE}
    if a.financial_inventory.is_file():
        doc = json.loads(a.financial_inventory.read_text())
        datasets = doc.get("datasets", [])
        financial_state = {"state": "MEASURED",
                           "datasets_declared": len(datasets)}
        for dsx in datasets:
            path = REPO / dsx["relative_path"]
            if not path.is_file():
                attempts.append({"variable_id": dsx["dataset_id"],
                                 "bank": ch.BANK_FINANCIAL,
                                 "outcome": UNAVAILABLE,
                                 "reason": "the declared file is not "
                                           "present in this checkout"})
                continue
            source_sha = sha_file(path)
            _header, axis, numeric, axis_values = read_columns(path)
            contract = axis_contract(path, axis, axis_values)
            contract["dataset"] = dsx["dataset_id"]
            axis_contracts.append(contract)
            columns = sorted(numeric)
            if a.financial_columns:
                columns = columns[:a.financial_columns]
            for column in columns:
                used, wcontract = window(numeric[column],
                                         fraction=DEV_FRACTION)
                binding = make_binding(
                    source_path=path, source_sha=source_sha,
                    window_contract=wcontract, code_digest=code_digest,
                    side="x", contract_role="input",
                    units="UNDECLARED",
                    terminal_attempt=a.terminal_attempt)
                vid = f"{dsx['dataset_id']}::{column}"
                r = ch.characterize_series(
                    used, variable_id=vid,
                    partition_key=f"{dsx['dataset_id']}::development",
                    bank_authority=ch.BANK_FINANCIAL,
                    measured_at=a.measured_at, binding=binding)
                rows.extend(r)
                identifiable = [x for x in r if x["identifiable"]]
                attempts.append({
                    "variable_id": vid, "bank": ch.BANK_FINANCIAL,
                    "outcome": MEASURED if identifiable
                    else NOT_IDENTIFIABLE,
                    "descriptors": len(r),
                    "not_identifiable": len(r) - len(identifiable),
                    "noise_identifiable": False,
                    "appearance_id": f"{dsx['dataset_id']}::{path.name}",
                    "provenance_state": dsx.get("profile_status",
                                                "UNKNOWN")})

    ch.assert_no_selection(rows)
    loaded = {}
    if a.load:
        from tools.backfill_campaign_envelopes import _engine
        loaded = ch.load_rows(_engine(), rows)

    banks: dict[str, int] = {}
    for r in rows:
        banks[r["bank_authority"]] = banks.get(r["bank_authority"], 0) + 1
    outcomes = {o: sum(1 for x in attempts if x["outcome"] == o)
                for o in (MEASURED, NOT_IDENTIFIABLE, UNAVAILABLE,
                          FAILED)}
    # C41: two populations, never conflated.
    conceptual = {x["variable_id"] for x in attempts}
    appearances = {x.get("appearance_id") for x in attempts
                   if x.get("appearance_id")}
    ledger = {
        "schema": "crispdm.characterization_ledger.v2",
        "measured_at": a.measured_at,
        "terminal_attempt": a.terminal_attempt,
        "code_identity": code_digest,
        "protocol_version": ch.PROTOCOL_VERSION,
        "attempts": attempts,
        "attempts_total": len(attempts),
        "outcomes": outcomes,
        "coverage": {
            "conceptual_variables_attempted": len(conceptual),
            "physical_appearances_attempted": len(appearances),
            "note": "a conceptual variable may appear in many files; "
                    "these are DIFFERENT populations and a count of "
                    "one is never a count of the other",
        },
        "axis_contracts": axis_contracts,
        "rows_total": len(rows),
        "rows_by_bank": banks,
        "rows_not_identifiable": sum(1 for r in rows
                                     if not r["identifiable"]),
        "rows_bound_to_source": sum(
            1 for r in rows
            if r.get("binding_state") == "BOUND_TO_SOURCE_BYTES"),
        "public_bank": public_state,
        "financial_bank": financial_state,
        "total_cost_seconds": round(
            sum(r["cost_seconds"] for r in rows), 6),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "loaded": loaded,
        "supersedes": "the C30 pilot rows are KEPT; this is a new "
                      "version beside them, never an edit",
        "selection_emitted": "NONE — this run measures variables and "
                             "chooses none",
        "confirmation_used": "NONE",
        "gpu_used": "NONE",
    }
    ledger["ledger_sha256"] = sha_obj(ledger)
    a.ledger.parent.mkdir(parents=True, exist_ok=True)
    a.ledger.write_text(json.dumps(ledger, indent=1, sort_keys=True)
                        + "\n")
    print(json.dumps({k: ledger[k] for k in
                      ("attempts_total", "outcomes", "coverage",
                       "rows_total", "rows_by_bank",
                       "rows_not_identifiable", "rows_bound_to_source",
                       "total_cost_seconds", "wall_seconds", "loaded",
                       "ledger_sha256")}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
