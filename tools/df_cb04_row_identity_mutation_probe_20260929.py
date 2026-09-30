"""Are the row-identity tests red against the WRONG design? Three mutations say yes.

A suite that passes is not evidence until it can fail. This probe substitutes one
digest at a time with a design that is defensible on its face and wrong, reruns the
repair tests, and records which tests die. A mutation that kills nothing is a test
gap, and the probe exits non-zero.

  M1  the shipped defect: every row takes the terminal-level primary identity.
  M2  row_role inside the identity: the same measurement stored as a primary in one
      terminal and a secondary in another then hides behind two digests, so the
      double count between terminals is invisible again.
  M3  evidence_class inside the identity: a published number and a measured one stop
      being comparable, and the separation the contract needs collapses into the
      digest instead of standing beside it.

Nothing is written outside --out. No service, warehouse or model is touched.

  python3 tools/df_cb04_row_identity_mutation_probe_20260929.py \
      --out docs/audits/evidence/cb04_row_identity_20260929/MUTATION_PROBE.json
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

from app import classification_receipt as cr            # noqa: E402
from app import classification_row_identity as ri       # noqa: E402

import test_classification_row_identity as suite        # noqa: E402

#: the classes that assert the REPAIR; the defect class asserts the shipped
#: behaviour and is expected to survive a mutation back towards it
REPAIR_CLASSES = (suite.EveryRowHasItsOwnIdentity,
                  suite.EvidenceClassIsSeparateFromIdentity,
                  suite.NoDoubleCountingBetweenTerminals,
                  suite.TheTagsAndTheReader)

_TRUE_IDENTITY = ri.row_identity_sha256
_TRUE_ROWS = ri.metric_rows_with_identity
_TRUE_OCCURRENCE = ri.metric_occurrence_sha256


def _run(classes):
    loader = unittest.TestLoader()
    tests = unittest.TestSuite(loader.loadTestsFromTestCase(c) for c in classes)
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(tests)
    killed = sorted(str(case).split(" ")[0]
                    for case, _ in (result.failures + result.errors))
    return {"ran": result.testsRun, "killed": killed, "killed_count": len(killed)}


def _mutation_primary_identity_for_every_row():
    """M1: the shipped defect, expressed as a mutation of the successor module."""
    def rows(receipt):
        out = _TRUE_ROWS(receipt)
        for row in out:
            row["row_identity_sha256"] = receipt["metric_identity_sha256"]
            row["metric_occurrence_sha256"] = _TRUE_OCCURRENCE(
                row_identity_sha256=receipt["metric_identity_sha256"],
                task_id=receipt["task_id"], corpus_id=receipt["corpus_id"],
                corpus_sha256=receipt["corpus_sha256"], provider=receipt["provider"],
                checkpoint_sha256=receipt["checkpoint_sha256"],
                supervision_regime=receipt["supervision_regime"],
                evidence_class=receipt["evidence_class"],
                evaluation_split=receipt["evaluation_split"],
                evaluation_population_sha256=receipt["evaluation_population_sha256"],
                protocol_sha256=receipt["protocol_sha256"],
                scorer_sha256=receipt["scorer_sha256"], seed=receipt["seed"])
        return out
    ri.metric_rows_with_identity = rows


def _mutation_row_role_inside_the_identity():
    """M2: role in the identity, so one measurement under two roles never collapses."""
    def rows(receipt):
        out = _TRUE_ROWS(receipt)
        for row in out:
            row["row_identity_sha256"] = cr.sha256_of(
                {"identity": row["row_identity_sha256"], "row_role": row["row_role"]})
            row["metric_occurrence_sha256"] = cr.sha256_of(
                {"occurrence": row["metric_occurrence_sha256"],
                 "row_role": row["row_role"]})
        return out
    ri.metric_rows_with_identity = rows


def _mutation_evidence_class_inside_the_identity():
    """M3: provenance folded into meaning."""
    def rows(receipt):
        out = _TRUE_ROWS(receipt)
        for row in out:
            row["row_identity_sha256"] = cr.sha256_of(
                {"identity": row["row_identity_sha256"],
                 "evidence_class": receipt["evidence_class"]})
        return out
    ri.metric_rows_with_identity = rows


MUTATIONS = {
    "M1_every_row_takes_the_terminal_level_primary_identity":
        _mutation_primary_identity_for_every_row,
    "M2_row_role_inside_the_identity": _mutation_row_role_inside_the_identity,
    "M3_evidence_class_inside_the_identity": _mutation_evidence_class_inside_the_identity,
}


def _restore():
    ri.row_identity_sha256 = _TRUE_IDENTITY
    ri.metric_rows_with_identity = _TRUE_ROWS
    ri.metric_occurrence_sha256 = _TRUE_OCCURRENCE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = {"schema": "cb04_row_identity_mutation_probe.v1",
              "what_this_is": ("three wrong designs for the per-row metric identity, each "
                               "substituted in turn, with the tests each one kills"),
              "what_this_is_not": ("a measurement of any model, a warehouse write, or a "
                                   "claim about the accepted rows already stored"),
              "baseline": None, "mutations": {}, "test_classes": [c.__name__
                                                                  for c in REPAIR_CLASSES]}
    _restore()
    report["baseline"] = _run(REPAIR_CLASSES)
    if report["baseline"]["killed"]:
        report["verdict"] = "BASELINE_NOT_GREEN"
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
        print(json.dumps(report["baseline"], indent=1))
        return 1

    for name, apply_mutation in MUTATIONS.items():
        _restore()
        apply_mutation()
        try:
            report["mutations"][name] = _run(REPAIR_CLASSES)
        finally:
            _restore()

    survivors = [name for name, outcome in report["mutations"].items()
                 if outcome["killed_count"] == 0]
    report["mutations_that_killed_nothing"] = survivors
    report["verdict"] = ("EVERY_MUTATION_IS_CAUGHT" if not survivors
                         else "A_MUTATION_SURVIVED_THE_SUITE_IS_INCOMPLETE")
    after = _run(REPAIR_CLASSES)
    report["suite_after_restore"] = after
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({"verdict": report["verdict"], "baseline": report["baseline"],
                      "mutations": {k: {"killed": v["killed_count"], "of": v["ran"]}
                                    for k, v in report["mutations"].items()},
                      "suite_after_restore": after, "written": args.out},
                     indent=1, sort_keys=True))
    return 0 if not survivors and not after["killed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
