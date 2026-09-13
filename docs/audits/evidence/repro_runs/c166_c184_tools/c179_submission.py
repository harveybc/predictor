#!/usr/bin/env python3
"""C179: a review submission that binds design, code, contracts, seed tape, roots, ledgers, decisions and the
OLAP load by digest. It is a submission, never a record of Musashi and never an authority: the gate keeps
refusing until an external record exists. Write-once. Hosts by role; home paths redacted.

  c179_submission.py --out <write-once json> --fresh-root <collected fresh root> --load-receipt <real receipt>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "tools"))
import df_d2_design as D  # noqa: E402

HOME = Path.home()
S = HOME / ".local/state/crispdm-data-foundation"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree(root: Path) -> dict:
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(root).as_posix()}\n{sha(p)}\n".encode())
    return {"files": len(files), "tree_sha256": h.hexdigest()}


def git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fresh-root", type=Path, required=True)
    ap.add_argument("--load-receipt", type=Path, required=True)
    ap.add_argument("--gate-refusal", type=Path, required=True, help="the gate's refusal output for the fresh decisions")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit("REFUSED: the submission is write-once")
    design_path = S / "d2_design_c171_v2/D2_DESIGN_V2.json"
    design = json.loads(design_path.read_text())
    problems = D.validate_design(design, require_current_code=True)
    tape = S / "d2_fresh_tape_c173_v1/SEED_TAPE.json"
    reserve = S / "d2_fresh_reserve_c173_v1"
    fresh_manifest = json.loads((a.fresh_root / "RUN_MANIFEST.json").read_text())
    adj = json.loads((a.fresh_root / "ADJUDICATION_SUMMARY.json").read_text())
    comparison = S / "d2_reanalysis_c172_v2_comparison"
    receipt = json.loads(a.load_receipt.read_text())
    doc = {
        "schema": "crispdm.data_foundation.d2_review_submission.v1",
        "kind": "SUBMISSION_FOR_EXTERNAL_REVIEW",
        "authority": "NONE: this document grants nothing; df_consumption_gate refuses until an external record exists",
        "stop": "D2_CURRENT_API_FRESH_CONFIRMATION_READY_FOR_MUSASHI_REVIEW",
        "code": {"predictor_tip": git("rev-parse", "HEAD"), "predictor_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                 "tracked_clean": git("status", "--porcelain", "--untracked-files=no") == "",
                 "lab_code_sha256s": D.lab_code_sha256s(), "lab_code_sha256": D.lab_code_sha256(),
                 "design_code_matches_tip": problems == [], "design_validation_problems": problems},
        "design": {"path": "d2_design_c171_v2/D2_DESIGN_V2.json", "design_id": design["design_id"],
                   "design_sha256": design["design_sha256"], "file_sha256": sha(design_path),
                   "dispersion_sha256": sha(S / "d2_design_c171_v2/C137_DISPERSION.json"),
                   "seal_summary_sha256": sha(S / "d2_design_c171_v2/SEAL_SUMMARY.json"),
                   "supersedes": json.loads((S / "d2_design_c171_v2/SEAL_SUMMARY.json").read_text())["supersedes"]},
        "seed_tape": {"path": "d2_fresh_tape_c173_v1/SEED_TAPE.json", "tape_sha256": json.loads(tape.read_text())["tape_sha256"],
                      "file_sha256": sha(tape)},
        "fresh_reserve": {"path": "d2_fresh_reserve_c173_v1", "root_manifest_sha256": sha(reserve / "ROOT_MANIFEST.json"),
                          **tree(reserve)},
        "historical_bank": {"path": "synthetic_bank_c128_v1", "bank_manifest_sha256": sha(S / "synthetic_bank_c128_v1/BANK_MANIFEST.json")},
        "historical_reanalysis": {"path": "d2_reanalysis_c172_v2", "label": D.HISTORICAL_MODE,
                                  "comparison_sha256": sha(comparison / "C172_COMPARISON.json"),
                                  "tables_manifest_sha256": sha(comparison / "tables/TABLES_MANIFEST.json"),
                                  "shard_roots": tree(S / "d2_reanalysis_c172_v2")},
        "fresh_confirmation": {"path": a.fresh_root.name, "run_id": fresh_manifest["run_id"],
                               "run_manifest_sha256": sha(a.fresh_root / "RUN_MANIFEST.json"),
                               "terminals": fresh_manifest["terminals"], "terminals_by_status": fresh_manifest["terminals_by_status"],
                               "invalidation_markers": fresh_manifest["invalidation_markers"],
                               "decisions_sha256": adj["decisions_sha256"], "decision_rows": adj["decision_rows"],
                               "decisions_by_kind": adj["decisions_by_kind"], "externally_reviewed": False,
                               "adjudication_summary_sha256": sha(a.fresh_root / "ADJUDICATION_SUMMARY.json")},
        "dispatch": {r: (sha(S / r / "DISPATCH_RECEIPT.json") if (S / r / "DISPATCH_RECEIPT.json").is_file() else None)
                     for r in ("d2_dispatch_c172_v2", "d2_dispatch_c174_coordinator", "d2_dispatch_c174_workers",
                               "d2_dispatch_c174_workers_v2")},
        "olap_load": {"receipt": a.load_receipt.name, "receipt_sha256": sha(a.load_receipt), "mode": receipt["mode"],
                      "historical_unchanged": receipt.get("historical_unchanged"),
                      "offered": receipt.get("offered"), "loader_code_sha256": receipt.get("loader_code_sha256")},
        "gate": {"refusal_sha256": sha(a.gate_refusal), "refusal": json.loads(a.gate_refusal.read_text())},
        "not_done_by_order": ["D3", "D4", "D5", "GPU", "feature selection", "models", "RL", "DOIN", "live", "venue"],
    }
    text = json.dumps(doc, indent=1, sort_keys=True).replace(str(HOME), "~")
    a.out.write_text(text + "\n")
    a.out.chmod(0o444)
    print(json.dumps({"submission_sha256": sha(a.out), "design_code_matches_tip": problems == [],
                      "tracked_clean": doc["code"]["tracked_clean"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
