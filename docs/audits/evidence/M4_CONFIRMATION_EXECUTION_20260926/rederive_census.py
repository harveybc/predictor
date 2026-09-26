"""Independently re-derive the M4 CONFIRMATION census identity.

The census digest 12cfd9ad... is asserted by the preparation's own return and
by the frozen runner. This script never copies it: it reads the sealed bytes
out of the agent-multi object store, re-derives the eligible-slot population
from the governing adjudication STRUCTURES, constructs the 3024 unit ids from
scratch with its own string construction, computes the canonical digests here,
and only then compares with (a) the frozen runner's own planner and (b) the
digest the documents assert. A mismatch refuses execution.

Exit 0 only if every re-derivation agrees.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

AM = Path("/home/harveybc/Documents/GitHub/agent-multi")
EXECTREE = Path("/home/harveybc/Documents/GitHub/.worktrees/m4-conf-exec-20260926")
TIP = "0e99ad1add3635dd4f0e151936d6fe4bf6379592"
REVIEWED_TIP = "5e7a8fd430c8231a049baf03f00e720ba24ec994"
ASSERTED_CENSUS = ("12cfd9ad785b41e788ffce575ec575ab"
                   "2a78ab772151b5e3b94c8f0c71169ea0")
PINS = {
    "design": ("docs/research/model_capacity/M4_SEALED_DESIGN_V5_2026_09_09.json",
               "design_sha256",
               "d7280a92047d98898418fb7cd750b22c506a621eb381d9847e0fe926b7df69b9"),
    "amendment": ("docs/research/model_capacity/"
                  "M4_V5_NUMERIC_VALIDITY_AMENDMENT_1_2026_09_09.json",
                  "amendment_sha256",
                  "43e0804e1e6e583b10ddbe46b7d4cd752838b0473ccbc6496f0e458c49aedd4b"),
    "adjudication": ("docs/audits/evidence/"
                     "M4_V5_CALIBRATION_ADJUDICATION_ATTEMPT3_GOVERNING_2026_09_09.json",
                     "record_sha256",
                     "b35b6fd969aa162047bdfb55b8f9fcce01aa76864c388d29a1c36642ab051ade"),
    "successor": ("docs/research/model_capacity/M4_CONFIRMATION_SUCCESSOR_2026_09_10.json",
                  "successor_sha256",
                  "6a50d97ddfb3a8e8dd1b5fbc83ebd95e60e1c087b3fc5e01697c2d783a50608c"),
}

CHECKPOINT_KINDS = ("initialization", "calibration_stop",
                    "pre_stop", "post_stop_bounded")

FAILED = []
N = [0]


def check(ok, what, detail=""):
    N[0] += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {what}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(what)


def selfsha(doc, key):
    body = {k: doc[k] for k in sorted(doc) if k != key}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def show(rel):
    r = subprocess.run(["git", "show", f"{TIP}:{rel}"], cwd=AM,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cannot read {rel}")
    return json.loads(r.stdout)


def main() -> int:
    print("M4 CONFIRMATION census -- independent re-derivation")
    print(f"agent-multi tip : {TIP}")
    print(f"reviewed tip    : {REVIEWED_TIP}\n")
    docs = {}
    for name, (rel, key, pin) in PINS.items():
        d = show(rel)
        docs[name] = d
        check(d.get(key) == pin, f"{name}: declared {key} is the order pin")
        check(selfsha(d, key) == pin,
              f"{name}: {key} RE-DERIVES from the bytes", selfsha(d, key)[:16] + "...")

    adj, succ, design = docs["adjudication"], docs["successor"], docs["design"]

    # --- population, re-derived from the adjudication structures ---
    slots = adj["confirmation_slots"]
    elig = [s for s in slots
            if s["typed_status"] == "ELIGIBLE_UNDER_PROPOSED_RULE"]
    inelig = [s for s in slots
              if s["typed_status"] != "ELIGIBLE_UNDER_PROPOSED_RULE"]
    check(len(slots) == 28, "28 reserved slots", str(len(slots)))
    check(len(elig) == 21, "21 eligible slots", str(len(elig)))
    check(len(inelig) == 7, "7 typed ineligible slots", str(len(inelig)))
    check(sorted(s["cell"] for s in elig)
          == sorted(s["cell"] for s in succ["eligible_slots"]),
          "the successor's eligible slots are the adjudication's, BY IDENTITY")
    inc = sorted({u.rsplit("::", 1)[0]
                  for u in adj["incomplete_units_in_denominator"]})
    check(len(inc) == 2, "exactly two incomplete generators", ", ".join(inc))
    check(sum(1 for d in adj["dispersion"].values()
              if d.get("status") == "CALIBRATION_INCOMPLETE") == 0,
          "zero calibration-incomplete cells")
    check(adj["ladder"]["m2_minus_m1_paired_gain"] == -0.41982887,
          "M2 gain is the accepted -0.41982887",
          repr(adj["ladder"]["m2_minus_m1_paired_gain"]))

    # --- the frozen numbers, re-derived by formula where a formula exists ---
    per_slot = succ["confirmation_generators_per_eligible_slot"]
    seeds = succ["nested_seeds_per_generator"]
    check(per_slot == 48, "48 generators per eligible slot", str(per_slot))
    check(seeds == 3, "3 nested seeds per generator", str(seeds))
    import math
    floor = max(3, math.ceil(per_slot * (1 - succ["attrition"]["allowance"])))
    check(succ["attrition"]["min_complete_required"] == floor == 39,
          "attrition floor 39 re-derives as max(3, ceil(48*0.8))", str(floor))
    check(len(succ["contrast_family_16"]) == 16, "the family is 16 slots")
    check(succ["contrast_family_16"]
          == design["confirmatory_contrast_family"]["contrasts"],
          "the 16 slots are the SEALED design's family, unchanged")

    # --- unit ids, constructed here from scratch ---
    units = []
    for s in succ["eligible_slots"]:
        prefix, w = s["cell"].rsplit("::w", 1)
        fam, nz = prefix.split("::")
        for gi in range(per_slot):
            for ms in range(seeds):
                units.append(f"intervention::CONFIRMATION::{fam}::{nz}"
                             f"::w{int(w)}::g{gi}::s{ms}")
    units = sorted(units)
    check(len(units) == 21 * 48 * 3 == 3024,
          "3024 unit ids = 21 x 48 x 3", str(len(units)))
    check(len(set(units)) == len(units), "every unit id is distinct")
    check(all("::CONFIRMATION::" in u for u in units),
          "every unit id carries the CONFIRMATION role")

    ck = design["checkpoint_rules"]
    census = {
        "schema": "m4_confirmation_census.v1",
        "successor_sha256": succ["successor_sha256"],
        "eligible_slots": len(succ["eligible_slots"]),
        "generators_per_slot": per_slot,
        "seeds_per_generator": seeds,
        "units_total": len(units),
        "unit_ids_sha256": hashlib.sha256(
            json.dumps(units).encode()).hexdigest(),
        "checkpoint_kinds": list(CHECKPOINT_KINDS),
        "update_bounds": {
            "calibration_stop_max_updates":
                ck["calibration_stop"]["max_updates"],
            "cadence_updates": ck["calibration_stop"]["cadence_updates"],
            "post_stop_bounded_updates": 500,
        },
    }
    mine = selfsha(census, "census_sha256")
    print(f"\nre-derived census_sha256 = {mine}")
    check(mine == ASSERTED_CENSUS,
          "the re-derived census digest equals the one the documents assert")

    # --- cross-check against the frozen runner's own planner ---
    r = subprocess.run([sys.executable, "tools/m4_confirmation_runner.py", "plan"],
                       cwd=EXECTREE, capture_output=True, text=True)
    check(r.returncode == 0, "the frozen planner runs", r.stderr.strip()[:120])
    if r.returncode == 0:
        plan = json.loads(r.stdout)
        check(plan["census_sha256"] == mine,
              "the frozen planner's census digest equals mine")
        check(plan["units_total"] == 3024, "planner units_total 3024")
        check(plan["mode"] == "PLAN_ONLY_NO_AUTHORITY",
              "the planner claims no authority", plan["mode"])
        print("\nplanner: " + json.dumps(
            {k: plan[k] for k in ("units_total", "eligible_slots",
                                  "generators_per_slot", "seeds_per_generator",
                                  "checkpoints_per_unit", "update_bounds",
                                  "musashi_review_record_present",
                                  "owner_execution_record_present",
                                  "execution_open")}))

    Path(Path(__file__).resolve().parent / "census_rederived.json").write_text(
        json.dumps({"census": census, "census_sha256": mine,
                    "unit_ids_total": len(units),
                    "eligible_cells": sorted(s["cell"] for s in
                                             succ["eligible_slots"]),
                    "ineligible_cells": sorted(
                        {"cell": s["cell"], "typed_status": s["typed_status"]}
                        .__repr__() for s in inelig)}, indent=1) + "\n")
    print(f"\nchecks: {N[0]}   failed: {len(FAILED)}")
    if FAILED:
        print("CENSUS DOES NOT REPRODUCE -- execution REFUSED")
        return 1
    print("CENSUS REPRODUCES -- execution may proceed to the record gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
