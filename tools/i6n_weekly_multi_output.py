"""I6-N lane C: direct multi-output arms manifest for the final stage (plan section 4.6).

The manifest is a TEMPLATE built before any fit from lane C's sealed weekly plan:

* two families, kept separate: short [2,3,4,5,6,12] and long [18,24,36,48,60,72];
* up to two recipe slots per family (R1, R2), refit EVERY eligible 2024 week, so
  at most 2 families x 2 recipes x 53 weeks = 212 dispositions (the plan budget);
* every arm carries the per-head C1 support of its weekly unit (TRAIN/INNER/
  VALIDATION ``n`` and ``origins_sha256`` per head). A head without support is
  typed NO_ELIGIBLE_ROWS inside the arm and never scored; the other heads keep
  their rows. A week where no head of the family is eligible is excluded and
  recorded, never silently skipped;
* the budget is enforced by refusing, never by dropping weeks.

Recipes are bound later, at stage activation, by :func:`resolve_arms` from the
search finalists (selected on the 2023 inner folds only). The pre-registered
rule per family: anchor = the family horizon whose BEST finalist has the lowest
inner-fold fitness (ties: shorter horizon); R1 = that BEST finalist, R2 = the
same horizon's MINIMAL_CONTROL (or small alternative). This is a template bound
to finalist ids, not a transferred 'optimal' recipe. Finalists selected on any
other basis (for example 2024 validation) are refused. Adapting the recipe to
multi-output heads (effective config, horizon-dependent genes) and freezing the
binding belong to lanes E/A and B; until then ``config`` and ``binding`` are null.

Pure standard library; no I/O besides the CLI, no fit, no prices, no TEST rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from tools.i6n_contracts.eligibility import canonical_digest
from tools.i6n_weekly_campaign import RUNNABLE

SCHEMA = "i6n.multi_output_arms.v1"
FAMILIES = {"short": (2, 3, 4, 5, 6, 12), "long": (18, 24, 36, 48, 60, 72)}
SLOTS = ("R1", "R2")
SLOT_ROLE = {"R1": "BEST", "R2": "MINIMAL_CONTROL"}
SELECTION_BASIS = "INNER_FOLD_FITNESS"
MAX_DISPOSITIONS = 212
TRANSFER_CLAIM = "TEMPLATE_BOUND_TO_SEARCH_FINALISTS_NOT_A_TRANSFERRED_OPTIMUM"
SPLITS = ("TRAIN", "INNER", "VALIDATION")


class MultiOutputError(ValueError):
    """Typed refusal with a stable ``code``."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _seal(body: Dict[str, Any], field: str) -> Dict[str, Any]:
    body = {k: v for k, v in body.items() if k != field}
    return dict(body, **{field: canonical_digest(body)})


def build_arms_manifest(plan: Mapping[str, Any], *, e_plan_sha256: str, recipes_per_family: int = 2,
                        max_dispositions: int = MAX_DISPOSITIONS) -> Dict[str, Any]:
    """Template arms for every eligible weekly unit of a sealed lane C plan."""
    if recipes_per_family not in (1, 2):
        raise MultiOutputError("RECIPES_PER_FAMILY", "one or two recipes per family")
    if not isinstance(e_plan_sha256, str) or len(e_plan_sha256) != 64:
        raise MultiOutputError("E_PLAN_REQUIRED")
    fit_role = plan["fit_role"]
    missing = [h for hs in FAMILIES.values() for h in hs if h not in plan["horizons_hours"]]
    if missing:
        raise MultiOutputError("PLAN_HORIZONS_MISSING", str(missing))
    slots = SLOTS[:recipes_per_family]
    arms: List[dict] = []
    excluded: List[dict] = []
    for family, hs in FAMILIES.items():
        for u in plan["units"]:
            if u["kind"] != "WEEKLY_REFIT":
                continue
            heads = {}
            for h in hs:
                state = u["horizons"][str(h)]["state"]
                support = u["support"][fit_role][str(h)]
                heads[str(h)] = {"state": "READY" if state in RUNNABLE else state,
                                 "cause": u["horizons"][str(h)]["cause"],
                                 **{s: {"n": support[s]["n"], "origins_sha256": support[s]["origins_sha256"]}
                                    for s in SPLITS if s in support}}
            if not any(v["state"] == "READY" for v in heads.values()):
                excluded.append({"unit_id": u["unit_id"], "family": family, "cause": "NO_ELIGIBLE_HEAD"})
                continue
            if u["evaluation_week"]["market_open"][:10] >= plan["test_start"][:10]:
                raise MultiOutputError("TEST_WEEK", u["unit_id"])
            for slot in slots:
                arms.append({"arm_id": f"MO-{family.upper()}-{slot}-{u['unit_id']}", "family": family,
                             "recipe_slot": slot, "recipe_role": SLOT_ROLE[slot], "week_id": u["unit_id"],
                             "unit_sha256": u["unit_sha256"], "cutoff_utc": u["fit_cutoff"],
                             "release_at_utc": u["release_at"], "horizons": list(hs), "heads": heads,
                             "state": "WAIT_DEPENDENCY", "cause": "FINALISTS_UNRESOLVED",
                             "recipe": None, "config": None, "binding": None})
    if len(arms) > max_dispositions:
        raise MultiOutputError("BUDGET_EXCEEDED", f"{len(arms)} > {max_dispositions}; no week is dropped")
    body = {"schema": SCHEMA, "experiment": plan["experiment"], "weekly_plan_sha256": plan["plan_sha256"],
            "e_plan_sha256": e_plan_sha256, "fit_role": fit_role,
            "families": {f: list(hs) for f, hs in FAMILIES.items()},
            "budget": {"max_dispositions": max_dispositions, "planned_dispositions": len(arms),
                       "recipes_per_family": recipes_per_family},
            "recipe_rule": {"selection_basis": SELECTION_BASIS,
                            "anchor": "family horizon with the lowest BEST inner-fold fitness; tie -> shorter h",
                            "R1": "BEST finalist of the anchor horizon",
                            "R2": "MINIMAL_CONTROL (or small alternative) of the anchor horizon",
                            "adaptation": "multi-output effective config and binding by E/A and B at activation"},
            "transfer_claim": TRANSFER_CLAIM, "refit": "weekly, every eligible 2024 week, own cutoff",
            "per_head_masks": "C1 per head from the weekly unit; ineligible heads typed, never scored",
            "state": "WAIT_DEPENDENCY", "cause": "FINALISTS_UNRESOLVED",
            "excluded_weeks": excluded, "arms": arms, "recipes": None}
    return _seal(body, "manifest_sha256")


def _family_recipe(family: str, finalists: Mapping[str, Any]) -> Optional[dict]:
    hs = FAMILIES[family]
    missing = [h for h in hs if str(h) not in finalists
               or any(r not in finalists[str(h)] for r in ("BEST", "MINIMAL_CONTROL"))]
    if missing:
        return {"missing": missing}
    for h in hs:
        for role in ("BEST", "MINIMAL_CONTROL"):
            f = finalists[str(h)][role]
            if f.get("selection_basis") != SELECTION_BASIS:
                raise MultiOutputError("SELECTION_BASIS_NOT_INNER_FOLD", f"h{h} {role}: {f.get('selection_basis')}")
            if not isinstance(f.get("fitness"), (int, float)) or isinstance(f.get("fitness"), bool):
                raise MultiOutputError("FINALIST_FITNESS_MISSING", f"h{h} {role}")
            for key in ("search_candidate_sha256", "genotype_canonical_sha256"):
                if not isinstance(f.get(key), str) or len(f[key]) != 64:
                    raise MultiOutputError("FINALIST_IDENTITY", f"h{h} {role} {key}")
    anchor = min(hs, key=lambda h: (finalists[str(h)]["BEST"]["fitness"], h))
    recipe = {"anchor_horizon": anchor}
    for slot, role in SLOT_ROLE.items():
        f = finalists[str(anchor)][role]
        recipe[slot] = {"role": role, "source_horizon": anchor,
                        "search_candidate_sha256": f["search_candidate_sha256"],
                        "genotype_canonical_sha256": f["genotype_canonical_sha256"],
                        "inner_fold_fitness": f["fitness"]}
    return recipe


def resolve_arms(manifest: Mapping[str, Any], finalists: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind recipe slots to finalist ids (inner-fold selection only). Pure and deterministic."""
    if manifest.get("schema") != SCHEMA or manifest.get("recipes") is not None:
        raise MultiOutputError("NOT_A_TEMPLATE", "resolve a sealed, unresolved template")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if canonical_digest(body) != manifest.get("manifest_sha256"):
        raise MultiOutputError("TEMPLATE_DAMAGED")
    recipes, causes = {}, {}
    for family in FAMILIES:
        r = _family_recipe(family, finalists)
        if "missing" in r:
            recipes[family] = None
            causes[family] = "FINALISTS_INCOMPLETE: " + ",".join(f"h{h}" for h in r["missing"])
        else:
            recipes[family] = r
    arms = []
    for arm in manifest["arms"]:
        recipe = recipes[arm["family"]]
        if recipe is None:
            arms.append(dict(arm, cause=causes[arm["family"]]))
        else:
            arms.append(dict(arm, state="READY", cause=None, recipe=recipe[arm["recipe_slot"]]))
    ready = [f for f in FAMILIES if recipes[f] is not None]
    state = "READY" if len(ready) == len(FAMILIES) else ("PARTIAL" if ready else "WAIT_DEPENDENCY")
    out = dict(body, template_sha256=manifest["manifest_sha256"], recipes=recipes, arms=arms, state=state,
               cause=None if state == "READY" else "; ".join(f"{k}: {v}" for k, v in causes.items()),
               finalists_sha256=canonical_digest(finalists))
    return _seal(out, "manifest_sha256")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--plan", type=Path, required=True)
    b.add_argument("--e-plan-sha256", required=True)
    b.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--finalists", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_arms_manifest(json.loads(args.plan.read_text()), e_plan_sha256=args.e_plan_sha256)
    else:
        result = resolve_arms(json.loads(args.manifest.read_text()), json.loads(args.finalists.read_text()))
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("state", "cause", "manifest_sha256")} |
                     {"dispositions": result["budget"]["planned_dispositions"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
