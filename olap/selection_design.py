"""The variable-selection design — sealed before anything is
scored.

A selection design written after looking at results is not a
design, it is a description of what happened to work. So this
module builds the design object, seals it with a self digest, and
exposes a mechanical preflight that is structurally incapable of
producing a conclusion: the preflight checks that units, splits
and comparators can be constructed and that costs can be
measured, and it refuses to compute a score.

Everything the order requires to be frozen is frozen here as
DATA, not prose: the outer unit, the comparator set, what may be
fitted where, the budget rule, the stability requirement, the
non-inferiority margin, the extreme-preservation requirement, the
multiplicity procedure, the INCONCLUSIVE rule and the withdrawal
criterion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "crispdm.selection_design.v1"

# The outer unit. A seed is a repetition, never an independent
# observation — the failure this line exists to prevent.
OUTER_UNIT = "task_origin_series"
FORBIDDEN_OUTER_UNITS = ("seed", "model_seed", "replicate",
                         "epoch", "checkpoint")


class SelectionDesignRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def _self_sha(doc: dict, key: str = "design_sha256") -> str:
    body = {k: doc[k] for k in sorted(doc) if k != key}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------
# comparators
# --------------------------------------------------------------

COMPARATORS = (
    {
        "comparator_id": "all_mechanically_admissible",
        "description": "every variable that passed mechanical "
                       "admissibility — the ceiling a selector "
                       "must beat to be worth its cost",
        "fits_on": "nothing",
        "selection_rule": "identity",
        "required": True,
    },
    {
        "comparator_id": "stability_redundancy_filter",
        "description": "drop variables that are unstable across "
                       "training origins or redundant with an "
                       "already-kept variable",
        "fits_on": "training_only",
        "selection_rule": "threshold on within-training "
                          "stability and pairwise redundancy",
        "required": True,
    },
    {
        "comparator_id": "mutual_information_train_only",
        "description": "univariate mutual information with the "
                       "target, estimated inside training",
        "fits_on": "training_only",
        "selection_rule": "top-k by MI under the frozen budget",
        "required": True,
    },
    {
        "comparator_id": "regularised_linear",
        "description": "an L1/elastic-net linear model fitted "
                       "inside training; non-zero coefficients "
                       "select",
        "fits_on": "training_only",
        "selection_rule": "non-zero coefficients under the "
                          "frozen penalty path",
        "required": True,
    },
    {
        "comparator_id": "agent_multi_current_selector",
        "description": "the selector agent-multi uses today — "
                       "admitted ONLY if it passes the P4 "
                       "boundary audit",
        "fits_on": "training_only",
        "selection_rule": "as implemented",
        "required": False,
        "admission_condition": "P4_BOUNDARY_AUDIT_PASSED",
    },
    {
        "comparator_id": "random_same_size_control",
        "description": "a uniformly random subset of the same "
                       "size as the selector under test — a "
                       "selector that does not beat this has "
                       "shown nothing",
        "fits_on": "nothing",
        "selection_rule": "uniform without replacement, seeded "
                          "from the frozen tape",
        "required": True,
    },
)


def build_design(*, sealed_at: str, tasks: list[dict],
                 budget_rule: dict, bank_index_sha256: str,
                 eligibility_manifest_sha256: str | None,
                 excluded_comparators: list[dict] | None = None
                 ) -> dict:
    if not tasks:
        raise SelectionDesignRefusal(
            "a selection design with no task declares nothing")
    for t in tasks:
        for field in ("task_id", "objective", "baseline",
                      "outer_origins"):
            if field not in t:
                raise SelectionDesignRefusal(
                    f"task {t.get('task_id', '?')} is missing "
                    f"{field!r} — objective and baseline are "
                    "frozen per task, never chosen later")
        if t.get("outer_unit", OUTER_UNIT) in \
                FORBIDDEN_OUTER_UNITS:
            raise SelectionDesignRefusal(
                f"task {t['task_id']}: the outer unit may not be "
                f"{t['outer_unit']!r} — a seed is a repetition, "
                "never an independent observation")
    excluded = list(excluded_comparators or ())
    excluded_ids = {e["comparator_id"] for e in excluded}
    active = [c for c in COMPARATORS
              if c["comparator_id"] not in excluded_ids]
    missing_required = [c["comparator_id"] for c in COMPARATORS
                        if c["required"]
                        and c["comparator_id"] in excluded_ids]
    if missing_required:
        raise SelectionDesignRefusal(
            f"required comparators cannot be excluded: "
            f"{missing_required}")
    doc = {
        "schema": SCHEMA,
        "sealed_at": sealed_at,
        "status": "SEALED_BEFORE_ANY_SCORE",
        "binds": {
            "bank_index_sha256": bank_index_sha256,
            "eligibility_manifest_sha256":
                eligibility_manifest_sha256 or "UNAVAILABLE",
        },
        "outer_unit": {
            "unit": OUTER_UNIT,
            "statement": "the outer unit is a task, origin and "
                         "series; seeds and repetitions are "
                         "nested inside it",
            "forbidden_units": list(FORBIDDEN_OUTER_UNITS),
        },
        "split_discipline": {
            "outer_split": "UNTOUCHED — declared before "
                           "selection and never re-cut",
            "fitted_inside_training_only": [
                "imputation", "scaling", "transformations",
                "the selector itself",
            ],
            "validation_reuse": "each outer origin's validation "
                                "may be consulted ONCE per "
                                "comparator; repeated "
                                "consultation is manual "
                                "optimisation and is forbidden",
        },
        "tasks": tasks,
        "comparators": active,
        "excluded_comparators": excluded,
        "budget_rule": budget_rule,
        "frozen_analysis": {
            "selection_cost": "wall seconds and peak memory of "
                              "profiling, fitting and selecting, "
                              "recorded per comparator per outer "
                              "origin, and reported with every "
                              "result",
            "stability_across_origins": {
                "statistic": "Jaccard overlap of selected sets "
                             "between outer origins",
                "reported": "always, as a first-class result, "
                            "not only when it is favourable",
            },
            "global_non_inferiority": {
                "margin": 0.02,
                "scale": "relative to the all-admissible "
                         "comparator on the task's own "
                         "objective",
                "statement": "a selector that reduces the "
                             "variable count but loses more than "
                             "the margin does not advance",
            },
            "extreme_preservation": {
                "requirement": "error on the declared extreme "
                               "regime may not degrade beyond "
                               "the same margin",
                "rationale": "a preparation that improves the "
                             "average by erasing extremes is a "
                             "loss disguised as a gain",
            },
            "multiplicity": {
                "procedure": "Holm over the full comparator "
                             "family per task, alpha 0.05",
                "family_frozen_before_scoring": True,
            },
            "inconclusive_rule": "when the interval for the "
                                 "primary contrast contains both "
                                 "the non-inferiority margin and "
                                 "zero, the result is "
                                 "INCONCLUSIVE — never rounded "
                                 "toward the preferred outcome",
            "withdrawal_criterion": "a comparator that fails "
                                    "non-inferiority, or fails "
                                    "extreme preservation, or "
                                    "whose stability overlap is "
                                    "below the frozen floor, is "
                                    "WITHDRAWN from the path and "
                                    "is not re-entered by a "
                                    "later run",
            "stability_floor": 0.5,
        },
        "execution_boundary": {
            "confirmation": "NOT AUTHORIZED by this design — a "
                            "large confirmation requires an "
                            "external review record",
            "permitted_now": "a mechanical preflight that "
                             "constructs units, splits and "
                             "comparators and measures cost, "
                             "and produces NO score",
        },
    }
    doc["design_sha256"] = _self_sha(doc)
    return doc


def verify_design(doc: dict) -> dict:
    if doc.get("schema") != SCHEMA:
        raise SelectionDesignRefusal("not a selection design")
    if _self_sha(doc) != doc.get("design_sha256"):
        raise SelectionDesignRefusal(
            "selection design self digest does not re-derive — "
            "a design edited after sealing is not sealed")
    if doc.get("status") != "SEALED_BEFORE_ANY_SCORE":
        raise SelectionDesignRefusal(
            f"design status is {doc.get('status')!r}")
    return doc


# --------------------------------------------------------------
# mechanical preflight — constructs, measures, never scores
# --------------------------------------------------------------

SCORE_WORDS = ("score", "metric_value", "objective_value",
               "mae", "rmse", "auc", "r2", "accuracy")


def mechanical_preflight(design: dict, *,
                         available_units: dict) -> dict:
    """Prove the design is CONSTRUCTIBLE on the units in hand.

    Returns counts, feasibility and declared costs. It contains no
    model fit and no metric, and `assert_no_conclusion` re-checks
    that on the way out.
    """
    verify_design(design)
    per_task = []
    for t in design["tasks"]:
        tid = t["task_id"]
        units = available_units.get(tid, {})
        origins = list(t["outer_origins"])
        series = units.get("series", [])
        feasible = bool(origins) and bool(series)
        per_task.append({
            "task_id": tid,
            "outer_origins_declared": len(origins),
            "series_available": len(series),
            "outer_units_constructible": (
                len(origins) * len(series) if feasible else 0),
            "feasible": feasible,
            "reason": ("declared origins and available series "
                       "both present"
                       if feasible else
                       "no outer unit can be constructed from "
                       "the units in hand"),
            "comparators_constructible": [
                c["comparator_id"] for c in design["comparators"]],
        })
    out = {
        "schema": "crispdm.selection_preflight.v1",
        "design_sha256": design["design_sha256"],
        "mode": "MECHANICAL_ONLY_NO_SCORE",
        "tasks": per_task,
        "comparator_count": len(design["comparators"]),
        "conclusion": "NONE — a preflight establishes that the "
                      "design can be executed, and nothing about "
                      "whether any selector works",
        "next_gate": "external review of this design before any "
                     "scoring run",
    }
    assert_no_conclusion(out)
    return out


def assert_no_conclusion(preflight: dict) -> None:
    """A preflight that carries a score is not a preflight."""
    blob = json.dumps(preflight).lower()
    for word in SCORE_WORDS:
        if f'"{word}"' in blob:
            raise SelectionDesignRefusal(
                f"the preflight carries a {word!r} field — a "
                "mechanical preflight never scores")
    if preflight.get("mode") != "MECHANICAL_ONLY_NO_SCORE":
        raise SelectionDesignRefusal(
            "preflight mode is not MECHANICAL_ONLY_NO_SCORE")
