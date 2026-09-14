#!/usr/bin/env python3
"""D2-R3: governed review campaign over the CONSERVED D2 evidence (Flow v3).

One GOVERNING campaign is registered with data-gov before anything is read; the support
of every arm is re-derived from the conserved fresh-confirmation rows with the repaired
adjudicator (`tools/df_d2_support.py`); successor decisions and the old -> new table are
emitted under a NEW run id; one terminal reports the transition counts, the artifact
hashes and the measured cost, and accounting and the terminal lake are reconciled.

No operator is re-executed, no unit is regenerated and no published row is overwritten.
Input mode is SYNTHETIC: the evidence is the synthetic D2 bank, and the spec binds the
generator, design, seed tape, reserve, conserved tables and code digests, so the receipt
is current while the bytes keep their original production date and provenance. The
terminal grants nothing: it records a review, not an eligibility.

usage:
  df_d2_r3_campaign.py --gov-url URL --api-key-file FILE --design D.json --tables DIR
      --reserve ROOT --collected ROOT --out DIR [--campaign-key KEY] [--metrics-lake olap_cube]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GR = _load("governed_run")
D = _load("df_d2_design")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def evidence_spec(design: Path, tables: Path, reserve: Path, collected: Path) -> dict:
    """Everything the review consumes, by digest: generator, design, tape, reserve, the
    conserved tables and the code that produced and now re-reads them."""
    design_doc = json.loads(design.read_text(encoding="utf-8"))
    reserve_manifest = json.loads((reserve / "ROOT_MANIFEST.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((collected / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    tables_manifest = json.loads((tables / "TABLES_MANIFEST.json").read_text(encoding="utf-8"))
    files = {name: sha256_file(tables / f"{name}.jsonl") for name in sorted(tables_manifest["counts"])}
    declared = tables_manifest["sha256"]
    mismatch = {k: (files[k], declared.get(k)) for k in files if files[k] != declared.get(k)}
    if mismatch:
        raise SystemExit(f"REFUSED: conserved tables do not re-hash to their manifest: {sorted(mismatch)}")
    return {
        "schema": "d2_r3_synthetic_evidence_spec.v1",
        "bank": "SYNTHETIC_D2_FRESH_CONFIRMATION",
        "design_sha256": design_doc["design_sha256"],
        "design_file_sha256": sha256_file(design),
        "generator": reserve_manifest["generator"],
        "tape_sha256": reserve_manifest["tape_sha256"],
        "reserve_manifest_sha256": sha256_file(reserve / "ROOT_MANIFEST.json"),
        "reserve_units": len(reserve_manifest["units"]),
        "seed_tape_file_sha256": sha256_file(reserve / "SEED_TAPE.json"),
        "run_id_superseded": run_manifest["run_id"],
        "run_manifest_sha256": sha256_file(collected / "RUN_MANIFEST.json"),
        "terminals_by_status": run_manifest["terminals_by_status"],
        "conserved_tables": {name: {"sha256": files[name], "rows": tables_manifest["counts"][name]}
                             for name in sorted(files)},
        "tables_manifest_sha256": sha256_file(tables / "TABLES_MANIFEST.json"),
        "lab_code_sha256s_at_creation": run_manifest["code_sha256_at_creation"],
        "lab_code_sha256s_now": D.lab_code_sha256s(),
        "readjudication_code": {"df_d2_adjudicate": sha256_file(HERE / "df_d2_adjudicate.py"),
                                "df_d2_support": sha256_file(HERE / "df_d2_support.py"),
                                "df_d2_r3_campaign": sha256_file(Path(__file__))},
        "partition": "confirmation",
        "note": "the arrays and rows were produced before Flow v3; this receipt is current and does not "
                "claim a governed download of the original production",
    }


def metrics_from(impact: dict, universe: dict) -> list:
    def metric(name, value, unit=None, split=None):
        return {"metric": name, "split": split, "horizon": None, "unit": unit, "value": float(value),
                "std_dev": None, "min_value": None, "max_value": None}

    rows = [metric("decisions.published", impact["decisions_published"], "decisions"),
            metric("decisions.successor", impact["decisions_preview"], "decisions"),
            metric("decisions.changed", impact["changed"], "decisions"),
            metric("successor.rows", impact.get("successor_rows") or 0, "rows")]
    for transition, count in sorted(impact["transitions"].items()):
        old, new = [part.strip() for part in transition.split("->")]
        rows.append(metric(f"transition.{old}.to.{new}", count, "decisions"))
    for key in ("expected_rows", "observed_rows", "missing_rows", "events_disagree"):
        rows.append(metric(f"universe.{key}", universe.get(key, 0), "cells"))
    return rows


def artifacts_of(out: Path) -> list:
    roles = {"successor_decisions": "DECISIONS_SUCCESSOR.jsonl", "impact_table": "IMPACT_TABLE.json",
             "impact_rows": "IMPACT_ROWS.jsonl", "universe_check": "UNIVERSE_CHECK.json",
             "support_summary": "SUPPORT_SUMMARY.json", "support_table": "SUPPORT_TABLE.jsonl",
             "decisions_preview": "DECISIONS_PREVIEW.jsonl"}
    items = []
    for role, name in sorted(roles.items()):
        path = out / name
        if path.is_file():
            items.append({"role": role, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gov-url", default="http://127.0.0.1:5055")
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--reserve", type=Path, required=True)
    ap.add_argument("--collected", type=Path, required=True, help="collected fresh root (RUN_MANIFEST, DECISIONS.jsonl)")
    ap.add_argument("--out", type=Path, required=True, help="fresh output directory for this review")
    ap.add_argument("--campaign-key", default="d2-support-readjudication-r3")
    ap.add_argument("--unit", default="readjudication")
    ap.add_argument("--metrics-lake", default="olap_cube")
    ap.add_argument("--project", default="predictor")
    ap.add_argument("--outbox-dir", default=GR.DEFAULT_OUTBOX)
    a = ap.parse_args(argv)
    out = a.out.resolve()
    if out.exists():
        raise SystemExit(f"REFUSED: the output directory already exists: {out}")
    decisions = a.collected / "DECISIONS.jsonl"
    code_identity = GR.strict_code_identity(REPO)
    spec = evidence_spec(a.design, a.tables, a.reserve, a.collected)
    synthetic_spec_sha256 = sha256_text(canonical(spec))
    design_doc = json.loads(a.design.read_text(encoding="utf-8"))
    execution = {"schema": "d2_r3_review_execution.v1", "project": a.project, "unit": a.unit,
                 "evidence": synthetic_spec_sha256, "decisions_sha256": sha256_file(decisions),
                 "rules": {"denoising": D.sha_obj(design_doc["denoising_rules"]),
                           "snr": D.sha_obj(design_doc["snr_rules"])},
                 "outputs": ["DECISIONS_SUCCESSOR.jsonl", "IMPACT_TABLE.json", "IMPACT_ROWS.jsonl",
                             "UNIVERSE_CHECK.json", "SUPPORT_SUMMARY.json", "SUPPORT_TABLE.jsonl"]}
    config_sha256 = sha256_text(canonical(execution))
    gov = GR.GovHttp(a.gov_url, GR.load_api_key(a.api_key_file), a.campaign_key)
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(a.outbox_dir)).resolve())
    campaign = {"schema": "governed_campaign.v1", "campaign_key": a.campaign_key, "classification": "GOVERNING",
                "project": a.project, "code_identity": code_identity, "config_sha256": config_sha256,
                "input_mode": "SYNTHETIC", "synthetic_spec_sha256": synthetic_spec_sha256, "units": [a.unit],
                "datasets": [], "terminal_lake": a.metrics_lake}
    status, receipt = gov.submit_campaign(campaign)
    if status not in (200, 201):
        raise SystemExit(f"REFUSED: campaign refused: http {status} {receipt.get('error', '')}".strip())
    campaign_sha256 = receipt["campaign_sha256"]
    prior = GR._send_pending(gov, outbox)
    if prior["pending"]:
        raise SystemExit(f"REFUSED: a prior terminal remains pending: {prior}")
    GR._require_reconciled(gov, campaign_sha256, a.unit, before_run=True)

    started_at = GR._utc_now()
    wall = time.monotonic()
    successor_run_id = f"d2r3_{campaign_sha256[:24]}"
    support = subprocess.run(
        [sys.executable, str(HERE / "df_d2_support.py"), "--design", str(a.design), "--tables", str(a.tables),
         "--reserve", str(a.reserve), "--decisions", str(decisions), "--out", str(out),
         "--successor-run-id", successor_run_id, "--supersedes-run-id", spec["run_id_superseded"]],
        capture_output=True, text=True)
    status_name, reason = "COMPLETED", None
    if support.returncode != 0:
        status_name, reason = "FAILED", f"SUPPORT_TOOL_EXIT_{support.returncode}"
    impact_path, universe_path = out / "IMPACT_TABLE.json", out / "UNIVERSE_CHECK.json"
    impact = json.loads(impact_path.read_text(encoding="utf-8")) if impact_path.is_file() else {}
    universe = json.loads(universe_path.read_text(encoding="utf-8")) if universe_path.is_file() else {}
    if status_name == "COMPLETED" and not universe.get("ok"):
        status_name, reason = "INCONCLUSIVE", "UNIVERSE_CHECK_NOT_CLEAN"
    terminal = {"schema": "governed_terminal.v1", "generation": 1, "status": status_name, "reason": reason,
                "started_at": started_at, "finished_at": GR._utc_now(),
                "costs": {"wall_seconds": max(0.0, time.monotonic() - wall)}, "deliveries": [],
                "artifacts": artifacts_of(out),
                "metrics": metrics_from(impact, universe) if status_name == "COMPLETED" else [],
                "tags": {"purpose": "D2_R3_GOVERNED_REVIEW", "grants": "NONE",
                         "evidence": "CONSERVED_FRESH_CONFIRMATION", "availability_use": "SYNTHETIC_EVIDENCE",
                         "supersedes_run_id": spec["run_id_superseded"], "successor_run_id": successor_run_id,
                         "externally_reviewed": "false"}}
    envelope = {"campaign_sha256": campaign_sha256, "unit_id": a.unit, "terminal": terminal}
    outbox.put(envelope)
    flushed = GR._send_pending(gov, outbox)
    reconciliation = None
    if not flushed["pending"]:
        reconciliation = GR._require_reconciled(gov, campaign_sha256, a.unit, before_run=False)
    state = {"schema": "d2_r3_review_receipt.v1", "campaign_key": a.campaign_key, "campaign_sha256": campaign_sha256,
             "unit_id": a.unit, "classification": "GOVERNING", "code_identity": code_identity,
             "config_sha256": config_sha256, "synthetic_spec_sha256": synthetic_spec_sha256,
             "evidence_spec": spec, "execution_spec": execution, "successor_run_id": successor_run_id,
             "support_tool_exit": support.returncode, "support_tool_stderr": support.stderr.strip()[-2000:],
             "terminal": terminal, "outbox_flush": flushed, "reconciliation": reconciliation,
             "governing_scope": "a review of conserved evidence; it grants no eligibility and opens no D3"}
    (out / "R3_REVIEW_RECEIPT.json").write_text(
        json.dumps(state, indent=1, default=float).replace(str(Path.home()), "~") + "\n", encoding="utf-8")
    print(json.dumps({"campaign_sha256": campaign_sha256, "status": status_name, "reason": reason,
                      "successor_run_id": successor_run_id, "changed": impact.get("changed"),
                      "flush": flushed, "reconciliation": reconciliation}, indent=1))
    return 0 if (status_name == "COMPLETED" and not flushed["pending"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
