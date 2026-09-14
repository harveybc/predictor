#!/usr/bin/env python3
"""D2-R4 (B2): the portability comparison re-executed as a governed diagnostic record.

The comparator was repaired to enforce its frozen population (`df_d2_r4_replay.compare`);
this registers the re-comparison of the **conserved** replay files with data-gov and writes
one additive terminal with the denominators and the outcomes actually obtained.

Nothing is regenerated: no unit is recomputed, no estimator runs, no published row is
touched. Input mode is SYNTHETIC and the spec binds the three replay files, the conserved
SNR table and decisions, the design, the frozen subset and the comparator's own code, so
the record is current while the arrays keep their original provenance. The terminal is a
diagnostic: it grants nothing and changes no decision.

usage:
  df_d2_r4_diagnostic_campaign.py --gov-url URL --api-key-file FILE --design D.json
      --subset SUBSET.json --snr-table T.jsonl --decisions D.jsonl
      --replay-file R1.json --replay-file R2.json --replay-file R3.json --out DIR
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def evidence_spec(design: Path, subset: Path, snr_table: Path, decisions: Path, replays: list) -> dict:
    """Everything the comparison reads, by digest. A receipt no one has to take on trust."""
    return {"schema": "d2_r4_diagnostic_evidence_spec.v1",
            "design_file_sha256": sha256_file(design),
            "design_sha256": json.loads(design.read_text(encoding="utf-8"))["design_sha256"],
            "subset_sha256": sha256_file(subset),
            "snr_table_sha256": sha256_file(snr_table),
            "decisions_sha256": sha256_file(decisions),
            "replays": {json.loads(Path(p).read_text(encoding="utf-8"))["role"]: sha256_file(Path(p))
                        for p in replays},
            "comparator_code_sha256": sha256_file(HERE / "df_d2_r4_replay.py"),
            "adjudicator_code_sha256": sha256_file(HERE / "df_d2_adjudicate.py"),
            "note": "conserved replay files and conserved rows; nothing is recomputed here"}


def metrics_from(report: dict) -> list:
    coverage = report.get("coverage") or {}
    denominators = coverage.get("denominators") or {}
    stability = report.get("decision_stability") or {}
    values = {
        "population.selected_units": denominators.get("selected_units"),
        "population.expected_facts_per_role": denominators.get("expected_facts_per_role"),
        "population.roles": denominators.get("roles"),
        "population.expected_cells": denominators.get("expected_cells"),
        "population.compared_cells": denominators.get("compared_cells"),
        "population.regimes": denominators.get("regimes"),
        "coverage.missing_selected_facts": coverage.get("missing_selected_facts"),
        "coverage.duplicated_facts": len(coverage.get("duplicated_facts") or []),
        "coverage.carried_unselected_rows": coverage.get("carried_unselected_rows"),
        "coverage.substituted_facts": coverage.get("substituted_facts"),
        "coverage.substituted_non_estimates": coverage.get("substituted_non_estimates"),
        "comparison.bytes_equal": report.get("bytes_equal"),
        "comparison.within_tolerance": report.get("within_tolerance"),
        "comparison.outside_tolerance": report.get("outside_tolerance"),
        "comparison.identifiability_changed": report.get("identifiability_changed"),
        "comparison.max_abs_delta_db": report.get("max_abs_delta_db"),
        "decisions.compared": stability.get("compared"),
        "decisions.changed": stability.get("changed"),
    }
    for estimator, stats in sorted((report.get("per_estimator") or {}).items()):
        values[f"estimator.{estimator}.cells"] = stats.get("cells")
        values[f"estimator.{estimator}.bytes_equal"] = stats.get("bytes_equal")
        values[f"estimator.{estimator}.outside_tolerance"] = stats.get("outside_tolerance")
        values[f"estimator.{estimator}.max_abs_delta_db"] = stats.get("max_abs_delta_db")
    return [{"metric": GR.metric_key(name), "split": None, "horizon": None, "unit": None,
             "value": float(value), "std_dev": None, "min_value": None, "max_value": None}
            for name, value in values.items() if isinstance(value, (int, float))]


def artifacts_of(out: Path) -> list:
    roles = {"portability_report": "R4_PORTABILITY_REPORT.json", "cells": "R4_CELLS.jsonl"}
    artifacts = []
    for role, name in roles.items():
        path = out / name
        if path.is_file():
            artifacts.append({"role": role, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return artifacts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gov-url", default="http://127.0.0.1:5055")
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--snr-table", type=Path, required=True)
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--replay-file", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tolerance-db", type=float, default=1e-9)
    ap.add_argument("--campaign-key", default="d2-r4-portability-diagnostic")
    ap.add_argument("--unit", default="recomparison")
    ap.add_argument("--metrics-lake", default="olap_cube")
    ap.add_argument("--project", default="predictor")
    ap.add_argument("--outbox-dir", default=GR.DEFAULT_OUTBOX)
    a = ap.parse_args(argv)
    out = a.out.resolve()
    if out.exists():
        raise SystemExit(f"REFUSED: the output directory already exists: {out}")

    spec = evidence_spec(a.design, a.subset, a.snr_table, a.decisions, a.replay_file)
    synthetic_spec_sha256 = sha256_text(canonical(spec))
    execution = {"schema": "d2_r4_diagnostic_execution.v1", "project": a.project, "unit": a.unit,
                 "evidence": synthetic_spec_sha256, "tolerance_db": a.tolerance_db,
                 "outputs": ["R4_PORTABILITY_REPORT.json", "R4_CELLS.jsonl"],
                 "population_rule": "every selected key, replayed exactly once by every role"}
    config_sha256 = sha256_text(canonical(execution))
    code_identity = GR.strict_code_identity(REPO)
    gov = GR.GovHttp(a.gov_url, GR.load_api_key(a.api_key_file), a.campaign_key)
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(a.outbox_dir)).resolve())
    campaign = {"schema": "governed_campaign.v1", "campaign_key": a.campaign_key,
                "classification": "GOVERNING", "project": a.project, "code_identity": code_identity,
                "config_sha256": config_sha256, "input_mode": "SYNTHETIC",
                "synthetic_spec_sha256": synthetic_spec_sha256, "units": [a.unit],
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
    out.mkdir(parents=True)
    command = [sys.executable, "-B", str(HERE / "df_d2_r4_replay.py"), "--compare",
               "--design", str(a.design), "--subset", str(a.subset), "--out", str(out),
               "--snr-table", str(a.snr_table), "--decisions", str(a.decisions),
               "--tolerance-db", str(a.tolerance_db)]
    for path in a.replay_file:
        command += ["--replay-file", str(path)]
    run = subprocess.run(command, capture_output=True, text=True,
                         env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                  MKL_NUM_THREADS="1"))
    report_path = out / "R4_PORTABILITY_REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    verdict = report.get("verdict")
    if not report:
        status_name, reason = "FAILED", f"COMPARATOR_EXIT_{run.returncode}"
    elif verdict != "MEASURED":
        status_name = "INCONCLUSIVE"
        reason = "POPULATION_INCOMPLETE:" + ",".join(report.get("inconclusive_reasons") or [])
    else:
        status_name, reason = "COMPLETED", None
    terminal = {"schema": "governed_terminal.v1", "generation": 1, "status": status_name, "reason": reason,
                "started_at": started_at, "finished_at": GR._utc_now(),
                "costs": {"wall_seconds": max(0.0, time.monotonic() - wall)}, "deliveries": [],
                "artifacts": artifacts_of(out),
                "metrics": metrics_from(report) if report else [],
                "tags": {"purpose": "D2_R4_PORTABILITY_DIAGNOSTIC", "grants": "NONE",
                         "evidence": "CONSERVED_REPLAYS", "availability_use": "SYNTHETIC_EVIDENCE",
                         "population_enforced": "true", "externally_reviewed": "false"}}
    outbox.put({"campaign_sha256": campaign_sha256, "unit_id": a.unit, "terminal": terminal})
    flushed = GR._send_pending(gov, outbox)
    reconciliation = None
    if not flushed["pending"]:
        reconciliation = GR._require_reconciled(gov, campaign_sha256, a.unit, before_run=False)
    state = {"schema": "d2_r4_diagnostic_receipt.v1", "campaign_key": a.campaign_key,
             "campaign_sha256": campaign_sha256, "unit_id": a.unit, "classification": "GOVERNING",
             "code_identity": code_identity, "config_sha256": config_sha256,
             "synthetic_spec_sha256": synthetic_spec_sha256, "evidence_spec": spec,
             "execution_spec": execution, "comparator_exit": run.returncode,
             "comparator_stderr": run.stderr.strip()[-2000:],
             "verdict": verdict, "denominators": (report.get("coverage") or {}).get("denominators"),
             "substituted_states": report.get("substituted_states"),
             "decision_stability": {k: v for k, v in (report.get("decision_stability") or {}).items()
                                    if k != "changed_rows"},
             "terminal": terminal, "outbox_flush": flushed, "reconciliation": reconciliation,
             "scope": "a numerical diagnostic over conserved evidence; it grants nothing and "
                      "changes no decision"}
    (out / "R4_DIAGNOSTIC_RECEIPT.json").write_text(
        json.dumps(state, indent=1, default=float).replace(str(Path.home()), "~") + "\n", encoding="utf-8")
    print(json.dumps({"campaign_sha256": campaign_sha256, "status": status_name, "verdict": verdict,
                      "denominators": state["denominators"], "decision_stability": state["decision_stability"],
                      "flush": flushed, "reconciliation": reconciliation}, indent=1))
    return 0 if (status_name == "COMPLETED" and not flushed["pending"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
