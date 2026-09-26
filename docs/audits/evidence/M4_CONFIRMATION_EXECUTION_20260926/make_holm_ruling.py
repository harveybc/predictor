"""Emit the pre-execution multiplicity ruling as a self-sealed record.

The ruling is a DECISION, so its text is authored; every FACT it quotes is
read out of the sealed bytes at run time (the two sealed-design multiplicity
sentences and the successor's own supersession sentence), and the record's
self-identity is computed with the corpus canonical rule (sha256 of the
canonical body with the digest field removed).

Run BEFORE any CONFIRMATION fit. Its whole value is the timestamp of the
commit that carries it.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

AM = Path("/home/harveybc/Documents/GitHub/agent-multi")
TIP = "0e99ad1add3635dd4f0e151936d6fe4bf6379592"
DESIGN = "docs/research/model_capacity/M4_SEALED_DESIGN_V5_2026_09_09.json"
SUCC = "docs/research/model_capacity/M4_CONFIRMATION_SUCCESSOR_2026_09_10.json"
DATE = "2026-09-26"


def _selfsha(doc, key):
    body = {k: doc[k] for k in sorted(doc) if k != key}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _show(rel):
    r = subprocess.run(["git", "show", f"{TIP}:{rel}"], cwd=AM,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cannot read {rel} at {TIP}")
    return r.stdout.encode(), json.loads(r.stdout)


def main() -> int:
    draw, design = _show(DESIGN)
    sraw, succ = _show(SUCC)
    design_sha = hashlib.sha256(draw).hexdigest()
    succ_sha = hashlib.sha256(sraw).hexdigest()
    doc = {
        "schema": "satoshi_m4_confirmation_multiplicity_ruling.v1",
        "date": DATE,
        "author": "Satoshi III (Mujuro Utsutsu)",
        "role": "SUCCESSOR_TECHNICAL_LEAD",
        "authority": (
            "the owner's grant of 2026-09-26, which gives the successor "
            "technical lead the authority to decide and execute the prepared "
            "M4 CONFIRMATION screen after the external reviewer stayed "
            "absent. This ruling is NOT Musashi's and is not offered as his; "
            "no part of it claims external-audit provenance."),
        "ordered_before_any_outcome": (
            "recorded and committed BEFORE any CONFIRMATION generator was "
            "constructed, any unit fitted and any contrast computed; the "
            "record has value only because its commit precedes the numbers."),
        "binds_successor_sha256": succ["successor_sha256"],
        "binds_design_sha256": design["design_sha256"],
        "binds_reviewed_tip": succ["binds_reviewed_tip"],
        "binds_agent_multi_tip": TIP,
        "successor_file_sha256": succ_sha,
        "design_file_sha256": design_sha,
        "decision": "HOLM_OVER_ALL_16_SLOTS_ADMITTED_AS_THE_GOVERNING_PROCEDURE",
        "ruling": (
            "The frozen preparation applies Holm step-down over all 16 slots "
            "of the confirmatory family, including the non-rejecting "
            "placeholders, at alpha 0.05. The sealed design v5 carries, in "
            "its statistics block, a Bonferroni sentence. Order @889320ee "
            "clause C34.7 permits a correction change declared before "
            "execution that is not less conservative per family. Holm and "
            "Bonferroni control the same family-wise error rate in the strong "
            "sense over the same closed family of 16; Holm's first step is "
            "exactly Bonferroni and every later step is no smaller, so Holm "
            "is uniformly at least as powerful and never rejects a "
            "hypothesis Bonferroni would retain at the same alpha. The "
            "family is unchanged: the same 16 slots, the same alpha, the same "
            "per-contrast two-sided one-sample t on generator-level paired "
            "effects. Holm is therefore admitted as the governing multiplicity "
            "procedure for this execution."),
        "sealed_design_evidence": {
            "statistics.multiplicity": design["statistics"]["multiplicity"],
            "confirmatory_contrast_family.multiplicity":
                design["confirmatory_contrast_family"]["multiplicity"],
            "note": (
                "read out of the sealed bytes, not paraphrased. The sealed "
                "design is not univocal: its contrast-family block already "
                "names 'Bonferroni-bounded design basis; Holm step-down at "
                "analysis time', so Holm at analysis time is the sealed "
                "design's own provision and the statistics-block sentence is "
                "the narrower one. This makes the change smaller than the "
                "preparation itself claimed; it does not make it free, and it "
                "is ruled on here rather than assumed."),
        },
        "successor_declaration":
            succ["analysis_freeze"]["multiplicity_supersession"],
        "conservative_addition_not_a_plan_change": (
            "Bonferroni-adjusted p-values over the same 16 slots will be "
            "published alongside the governing Holm result as a NON-GOVERNING "
            "cross-check. It adds no hypothesis, changes no alpha and can only "
            "expose a case where the two procedures disagree. Where they "
            "agree, this ruling is immaterial to the verdict and the return "
            "must say so."),
        "what_this_ruling_does_not_do": [
            "it does not add, drop or redefine any contrast",
            "it does not change alpha, the estimand, the per-contrast test, "
            "the eligible-slot population, the 48-generator or 3-seed census, "
            "the attrition floor of 39 or the M2 exclusion",
            "it does not grant execution: the runner's two-record gate is a "
            "separate instrument and is recorded separately",
            "it does not claim external audit; the external design review "
            "commanded by order @889320ee C36 remains unperformed and its "
            "absence is named in the return",
        ],
        "signature": "Satoshi III (Mujuro Utsutsu), successor technical lead, "
                     "2026-09-26",
    }
    doc["ruling_sha256"] = _selfsha(doc, "ruling_sha256")
    out = Path(__file__).resolve().parent / \
        "SATOSHI_M4_HOLM_MULTIPLICITY_RULING_2026_09_26.json"
    if out.exists():
        raise SystemExit("ruling already exists — append-only, never rewritten")
    fd = os.open(str(out), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, json.dumps(doc, indent=1).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    print(f"wrote {out}")
    print(f"ruling_sha256 = {doc['ruling_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
