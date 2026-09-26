#!/usr/bin/env python3
"""Independent re-verification of the M4 C32-C38 CONFIRMATION preparation.

The order is agent-multi@889320ee section "P2 - M4 C32-C38 confirmation
preparation".  It was executed, committed and pushed on 2026-09-10 in the
`agent-multi` repository, branch `satoshi/model-capacity-m3-20260908`,
tip `0e99ad1add3635dd4f0e151936d6fe4bf6379592`.  This script does NOT redo
that work.  It answers one question a reader should not have to take on
trust: *is the frozen plan still exactly the plan that was frozen, and is
it still unexecuted?*

Everything is read out of the git object store at the pinned ref, so the
script never needs a checkout of that branch and never touches it.  It
recomputes rather than reads back:

  1. every artifact's file digest at the pinned ref;
  2. the four order-pinned self-identities, recomputed from the bytes with
     the same canonical rule the protocol uses (sha256 of the JSON body
     with the self key removed, sort_keys=True);
  3. the five order facts of C32, re-derived from the adjudication
     STRUCTURES and not from any summary field;
  4. the C33 frozen policy numbers, against the order's own words;
  5. the C34 sixteen-contrast topology, re-derived from the successor's
     eligible-slot list -- including the arithmetic that the 21 eligible
     slots decompose as 10 two-width contrasts + 1 single-width contrast;
  6. that no CONFIRMATION array, score or ledger blob exists at the ref,
     and that neither external authority record is installed on this host.

Exit 0 means every check passed.  Any failure prints the check and exits 1.
No model is fitted, loaded or scored.  CPU only.  Nothing is written.

Usage:
    python3 reverify_m4_c32_c38.py [--repo /path/to/agent-multi]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- pins ----
# Copied from the order (agent-multi@889320ee, P2/C32) and from Musashi's
# accepting audit docs/audits/MUSASHI_AUDIT_M4_C31A_C31F_AND_EXTERNAL_
# CAMPAIGN_STATUS_2026_09_10.md at the reviewed tip.
REVIEWED_TIP = "5e7a8fd430c8231a049baf03f00e720ba24ec994"
RETURN_TIP = "0e99ad1add3635dd4f0e151936d6fe4bf6379592"
CYCLE_TIP = "89a3781f"
PRE_TIP = "d8b45eb4"

DESIGN_PATH = ("docs/research/model_capacity/"
               "M4_SEALED_DESIGN_V5_2026_09_09.json")
AMENDMENT_PATH = ("docs/research/model_capacity/"
                  "M4_V5_NUMERIC_VALIDITY_AMENDMENT_1_2026_09_09.json")
ADJUDICATION_PATH = ("docs/audits/evidence/"
                     "M4_V5_CALIBRATION_ADJUDICATION_ATTEMPT3_"
                     "GOVERNING_2026_09_09.json")
SUCCESSOR_PATH = ("docs/research/model_capacity/"
                  "M4_CONFIRMATION_SUCCESSOR_2026_09_10.json")

# file digests pinned by Musashi's audit, sections 1
FILE_SHA = {
    DESIGN_PATH:
        "0a0fb757d847749373259031cf99f0844d13fb37a8d135462c9f926b1930700a",
    AMENDMENT_PATH:
        "49c363953641ea7be2eff124654cf861d4214402ab921b7276d7a768ced61566",
    ADJUDICATION_PATH:
        "51247b7853f7da1d5e549810f6b680183e7f4a4c77a4c787188030b6e560ff4c",
}

# self-identities pinned by the order's C32 bullet list
SELF_ID = {
    DESIGN_PATH: (
        "design_sha256",
        "d7280a92047d98898418fb7cd750b22c506a621eb381d9847e0fe926b7df69b9"),
    AMENDMENT_PATH: (
        "amendment_sha256",
        "43e0804e1e6e583b10ddbe46b7d4cd752838b0473ccbc6496f0e458c49aedd4b"),
    ADJUDICATION_PATH: (
        "record_sha256",
        "b35b6fd969aa162047bdfb55b8f9fcce01aa76864c388d29a1c36642ab051ade"),
    # the successor's self-identity is not in the order (it did not exist
    # yet); it is pinned by the C32-C38 return packet, section 3.
    SUCCESSOR_PATH: (
        "successor_sha256",
        "6a50d97ddfb3a8e8dd1b5fbc83ebd95e60e1c087b3fc5e01697c2d783a50608c"),
}

# exact re-derived facts the order's C32 demands
ORDER_FACTS = {
    "eligible_slots": 21,
    "total_slots": 28,
    "ineligible_slots": 7,
    "incomplete_generators": 2,
    "calibration_incomplete_cells": 0,
    "m2_gain": -0.41982887,
}

# the C33 numbers the order fixes in words
C33 = {
    "min_learnable": 12,
    "of_generators": 16,
    "max_numerically_invalid": 0,
    "generators_per_eligible_slot": 48,
    "attrition_allowance": 0.20,
    "min_complete_required": 39,
    "nested_seeds": 3,
    "selection_rule_label": "CALIBRATION_DERIVED_AND_REVIEWED",
    "classification": "SCIENTIFIC_ANALYSIS_FREEZE",
    "m2_status": "DOES_NOT_ADVANCE_FROM_CALIBRATION",
}

# the deliverables of the cycle, with the digests measured at the return tip
DELIVERABLE_SHA = {
    "tools/m4_confirmation_protocol.py":
        "ddc180e285232f53ae639579296d2f0ba01d0630c132490e457e3f5764688e52",
    "tools/m4_confirmation_runner.py":
        "f4545bef7876e355b492110df51a2710e1c91d91178acc73e6453ef2791c3a1e",
    "tests/test_m4_confirmation_protocol.py":
        "454bc8b33eb0ff350d877a5b46416f6a24e08bc1487ef3bd6fa645801d8ade5d",
    SUCCESSOR_PATH:
        "0b257a98efca1a946a1c7ca378d8977036a758290d6c2111301bd2574bddf6c2",
    ("docs/audits/evidence/"
     "MUSASHI_M4_CONFIRMATION_DESIGN_REVIEW_TEMPLATE_2026_09_10.json"):
        "2dfd326600ca8ae3560fb23340d91167cf877e16e3356b9713d320a27ea3344b",
    ("docs/audits/evidence/"
     "OWNER_M4_CONFIRMATION_EXECUTION_TEMPLATE_2026_09_10.json"):
        "62885899f3b40645583273d93113733634724a53224dcad749bda2b5a7b38d87",
    "docs/audits/evidence/repro_runs/m4_c32_c38_post_2026_09_10.py":
        "58ffbf558bc1178e386c475fb969f3f1769da1f22120695d7d08f9acd16c469d",
    "docs/audits/evidence/repro_runs/m4_c32_c38_post_2026_09_10.out":
        "9fefb91b821fdf816c88a6d453c01a5d6e77574e9eb6a7fdd61cf6f991422964",
    ("docs/handoffs/"
     "GENERAL_SATOSHI_TO_MUSASHI_M4_C32_C38_RETURN_2026_09_10.md"):
        "54027aa3a68f2339e2650c6b3442d34557bf8fe9dd7bbdfa6667ada17916a432",
}

DEFAULT_REPO = Path("/home/harveybc/Documents/GitHub/agent-multi")


class Failed(Exception):
    pass


_results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    _results.append((bool(ok), label))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def selfsha(doc: dict, key: str) -> str:
    body = {k: doc[k] for k in sorted(doc) if k != key}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise Failed(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def blob(repo: Path, ref: str, path: str) -> bytes:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=repo,
                       capture_output=True)
    if r.returncode != 0:
        raise Failed(f"{path} absent at {ref}")
    return r.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(DEFAULT_REPO),
                    help="path to the agent-multi repository")
    ap.add_argument("--live-checkout", default=None,
                    help="path to a clean checkout of the return tip; when "
                         "given, the C37 battery, the planner and the "
                         "execute refusal are run there as well. The "
                         "guards read git, so they can only be exercised "
                         "in a real checkout.")
    a = ap.parse_args(argv)
    repo = Path(a.repo)

    print("M4 C32-C38 CONFIRMATION preparation -- independent "
          "re-verification")
    print(f"repo            : {repo}")
    print(f"order           : agent-multi@889320ee P2 (C32-C38)")
    print(f"reviewed tip    : {REVIEWED_TIP}")
    print(f"PRE / cycle / packet : {PRE_TIP} / {CYCLE_TIP} / "
          f"{RETURN_TIP[:8]}")
    print()

    # -- 0. the refs exist and the pushed tip is the local tip -------------
    print("-- 0. refs")
    for name, ref in (("reviewed tip", REVIEWED_TIP),
                      ("PRE", PRE_TIP), ("cycle", CYCLE_TIP),
                      ("return packet", RETURN_TIP)):
        t = git(repo, "cat-file", "-t", f"{ref}^{{commit}}").strip()
        check(t == "commit", f"{name} {ref[:8]} is a commit in this repo")
    local = git(repo, "rev-parse",
                "satoshi/model-capacity-m3-20260908").strip()
    remote = git(repo, "rev-parse",
                 "refs/remotes/origin/"
                 "satoshi/model-capacity-m3-20260908").strip()
    check(local == RETURN_TIP,
          "branch tip is the return packet commit", local)
    check(remote == RETURN_TIP,
          "the packet is PUSHED (origin tip equals it)", remote)
    print()

    # -- 1. deliverable digests -------------------------------------------
    print("-- 1. deliverables at the return tip")
    for path, want in sorted(DELIVERABLE_SHA.items()):
        got = hashlib.sha256(blob(repo, RETURN_TIP, path)).hexdigest()
        check(got == want, f"{path}", got)
    print()

    # -- 2. the pinned file digests of the accepted calibration evidence ---
    print("-- 2. accepted calibration evidence, file digests "
          "(Musashi audit section 1)")
    for path, want in sorted(FILE_SHA.items()):
        got = hashlib.sha256(blob(repo, RETURN_TIP, path)).hexdigest()
        check(got == want, f"{path}", got)
    print()

    # -- 3. the four self-identities, RECOMPUTED ---------------------------
    print("-- 3. self-identities recomputed from the bytes")
    docs: dict[str, dict] = {}
    for path, (key, want) in SELF_ID.items():
        doc = json.loads(blob(repo, RETURN_TIP, path))
        docs[path] = doc
        rec = selfsha(doc, key)
        check(doc.get(key) == want,
              f"{Path(path).name}: declared {key} is the pin")
        check(rec == want,
              f"{Path(path).name}: {key} RE-DERIVES from the bytes", rec)
    print()

    # -- 4. C32 order facts, re-derived from STRUCTURES --------------------
    print("-- 4. C32 order facts re-derived from adjudication structures")
    adj = docs[ADJUDICATION_PATH]
    slots = adj["confirmation_slots"]
    elig = [s for s in slots
            if s["typed_status"] == "ELIGIBLE_UNDER_PROPOSED_RULE"]
    inelig = [s for s in slots
              if s["typed_status"] != "ELIGIBLE_UNDER_PROPOSED_RULE"]
    inc_gens = sorted({u.rsplit("::", 1)[0]
                       for u in adj["incomplete_units_in_denominator"]})
    bad = [c for c, d in adj["dispersion"].items()
           if d.get("status") == "CALIBRATION_INCOMPLETE"]
    check(len(slots) == ORDER_FACTS["total_slots"],
          "28 reserved family/noise/width slots", str(len(slots)))
    check(len(elig) == ORDER_FACTS["eligible_slots"],
          "21 eligible slots", str(len(elig)))
    check(len(inelig) == ORDER_FACTS["ineligible_slots"],
          "7 typed ineligible slots", str(len(inelig)))
    check(len(inc_gens) == ORDER_FACTS["incomplete_generators"],
          "exactly two incomplete generators, kept in the denominator",
          ", ".join(inc_gens))
    check(len(bad) == ORDER_FACTS["calibration_incomplete_cells"],
          "zero calibration-incomplete cells", str(len(bad)))
    gain = adj["ladder"]["m2_minus_m1_paired_gain"]
    check(gain == ORDER_FACTS["m2_gain"],
          "M2 minus M1 paired gain is the accepted -0.41982887", repr(gain))
    check(adj.get("authority") ==
          "CANDIDATE_FOR_MUSASHI_REVIEW_NO_CONFIRMATION_AUTHORITY",
          "the adjudication claims NO confirmation authority of its own",
          str(adj.get("authority")))
    print()

    # -- 5. C33 frozen policy --------------------------------------------
    print("-- 5. C33 frozen policy in the successor")
    suc = docs[SUCCESSOR_PATH]
    er = suc["eligibility_rule"]
    at = suc["attrition"]
    check(er["min_learnable_under_frozen_budget"] == C33["min_learnable"],
          "eligibility threshold is 12")
    check(er["of_calibration_generators"] == C33["of_generators"],
          "out of 16 calibration generators")
    check(er["max_numerically_invalid"] == C33["max_numerically_invalid"],
          "and zero NUMERICALLY_INVALID permitted")
    check(suc["confirmation_generators_per_eligible_slot"]
          == C33["generators_per_eligible_slot"],
          "48 CONFIRMATION generators per eligible slot")
    check(suc["nested_seeds_per_generator"] == C33["nested_seeds"],
          "3 nested seeds per generator")
    check(at["allowance"] == C33["attrition_allowance"],
          "attrition allowance 0.20")
    check(at["min_complete_required"] == C33["min_complete_required"],
          "attrition floor 39 = max(3, ceil(48*0.8))")
    check(at["min_complete_required"]
          == max(3, -(-C33["generators_per_eligible_slot"]
                      * 4 // 5)),
          "the floor re-derives from the formula, not from the field")
    check(suc["selection_rule_label"] == C33["selection_rule_label"],
          "the rule is labelled CALIBRATION_DERIVED_AND_REVIEWED, "
          "never predeclared")
    check(suc["classification"] == C33["classification"],
          "classified SCIENTIFIC_ANALYSIS_FREEZE, not "
          "'scientific_change: NONE'")
    check(suc["m2_status"]["status"] == C33["m2_status"],
          "M2 is DOES_NOT_ADVANCE_FROM_CALIBRATION")
    check(suc["m2_status"]["calibration_gain"] == ORDER_FACTS["m2_gain"],
          "and carries the measured gain, not a re-estimate")
    check(len(suc["eligible_slots"]) == ORDER_FACTS["eligible_slots"]
          and len(suc["ineligible_slots"])
          == ORDER_FACTS["ineligible_slots"],
          "the successor copies the exact 21 + 7 slot population")
    elig_cells = {s["cell"] for s in suc["eligible_slots"]}
    check(elig_cells == {s["cell"] for s in elig},
          "and copies them BY IDENTITY from the governing adjudication")
    check(suc["binds_reviewed_tip"] == REVIEWED_TIP,
          "the successor binds the reviewed tip")
    check(suc["supersedes_design_sha256"] == SELF_ID[DESIGN_PATH][1],
          "supersedes the sealed v5 by digest (v5 bytes untouched)")
    print()

    # -- 6. C34 sixteen-contrast topology, re-derived ----------------------
    print("-- 6. C34 sixteen contrasts, topology re-derived from the "
          "eligible slots")
    fam = suc["contrast_family_16"]
    names = [c if isinstance(c, str) else c.get("contrast") for c in fam]
    check(len(fam) == 16, "the family has exactly 16 slots")
    check(names[-2] == "checkpoint_effect::primary_pair",
          "the 15th slot is checkpoint_effect::primary_pair")
    check(names[-1] == "incremental_prediction::M2_vs_M1",
          "the 16th slot is the M2 placeholder, kept so the family "
          "does not shrink")
    interventions = [n for n in names
                     if n.startswith("intervention_effect::")]
    check(len(interventions) == 14,
          "14 family/noise intervention contrasts")
    widths_by_pair: dict[str, set[str]] = {}
    for cell in sorted(elig_cells):
        fam_noise, width = cell.rsplit("::", 1)
        widths_by_pair.setdefault(fam_noise, set()).add(width)
    two, one, zero = [], [], []
    for n in interventions:
        pair = n.split("::", 1)[1]
        w = widths_by_pair.get(pair, set())
        (two if len(w) == 2 else one if len(w) == 1 else zero).append(pair)
    check(len(two) == 10, "10 contrasts have both frozen widths",
          ", ".join(sorted(two)))
    check(one == ["state_space::clean"] and
          widths_by_pair["state_space::clean"] == {"w16"},
          "state_space::clean is single-width and the width is named: w16")
    check(sorted(zero) == ["discontinuity::clean", "discontinuity::white",
                           "parity4::clean"],
          "three contrasts have no eligible width -> NOT_EVALUABLE p=1",
          ", ".join(sorted(zero)))
    check(2 * len(two) + len(one) == ORDER_FACTS["eligible_slots"],
          "the arithmetic closes: 10*2 + 1 = 21 eligible slots")
    af = suc["analysis_freeze"]
    check(af["unit"].startswith("the task GENERATOR"),
          "the independent unit is the generator")
    check("Holm" in af["multiplicity"] and "ALL 16" in af["multiplicity"],
          "Holm runs over ALL 16 slots including placeholders")
    check("SUPERSEDED" in af["multiplicity_supersession"]
          and "Bonferroni" in af["multiplicity_supersession"],
          "the Holm-supersedes-Bonferroni override is DECLARED, not "
          "slipped through")
    check("SECONDARY" in af["width_heterogeneity"],
          "width-specific effects are secondary heterogeneity only")
    check("NOT_EVALUABLE" in af["no_width_rule"]
          and "p=1" in af["no_width_rule"],
          "no eligible width gives a non-rejecting NOT_EVALUABLE")
    print()

    # -- 7. still PREPARED, NOT EXECUTED ----------------------------------
    print("-- 7. the plan is still unexecuted")
    tracked = git(repo, "ls-tree", "-r", "--name-only",
                  RETURN_TIP).splitlines()
    conf = [p for p in tracked
            if "M4_CONFIRMATION" in p or "m4_confirmation" in p]
    expected = {
        SUCCESSOR_PATH,
        "docs/audits/evidence/"
        "MUSASHI_M4_CONFIRMATION_DESIGN_REVIEW_TEMPLATE_2026_09_10.json",
        "docs/audits/evidence/"
        "OWNER_M4_CONFIRMATION_EXECUTION_TEMPLATE_2026_09_10.json",
        "tools/m4_confirmation_protocol.py",
        "tools/m4_confirmation_runner.py",
        "tests/test_m4_confirmation_protocol.py",
    }
    check(set(conf) == expected,
          "the only M4-CONFIRMATION paths are the plan, the two "
          "non-authorizing templates, the two tools and the battery",
          f"{len(conf)} paths: " + ", ".join(sorted(
              Path(p).name for p in conf)))
    # the observational artifacts a CONFIRMATION run would create: a unit
    # array, an arm record, a pre-result ledger, a verdict.  None may exist.
    arrays = [p for p in tracked
              if ("confirmation" in p.lower()
                  and (p.endswith((".jsonl", ".npy", ".npz"))
                       or "ledger" in p.lower()
                       or "run_report" in p.lower()))]
    check(arrays == [],
          "no CONFIRMATION array, arm record, ledger or verdict is "
          "tracked at the return tip", str(arrays))
    # Wherever the CONFIRMATION role string appears in a data file it must
    # be a RESERVATION and never an observation, and the cycle must not
    # have added one: the counts at the reviewed tip and at the return tip
    # must be identical.
    def role_census(ref: str) -> dict[str, tuple[int, int]]:
        g = subprocess.run(
            ["git", "grep", "-l", "--", "::CONFIRMATION::", ref,
             "--", "*.jsonl", "*.json"],
            cwd=repo, capture_output=True, text=True)
        out = {}
        for ln in g.stdout.splitlines():
            if ":" not in ln:
                continue
            path = ln.split(":", 1)[1]
            body = blob(repo, ref, path).decode("utf-8", "replace")
            out[path] = (body.count("::CONFIRMATION::"),
                         body.count("RESERVED::CONFIRMATION::"))
        return out
    before, after = role_census(REVIEWED_TIP), role_census(RETURN_TIP)
    check(all(t == r for t, r in after.values()),
          "every ::CONFIRMATION:: occurrence in a data file is a "
          "RESERVED:: reservation, never an observation",
          "; ".join(f"{Path(p).name}: {t} total / {r} reserved"
                    for p, (t, r) in sorted(after.items())))
    check(before == after,
          "the C32-C38 cycle added no CONFIRMATION record: the reserved "
          "census is byte-identical at the reviewed tip and the "
          "return tip",
          f"{sum(t for t, _ in after.values())} reservations, unchanged")
    check(sum(t for t, _ in after.values())
          == ORDER_FACTS["total_slots"] * C33[
              "generators_per_eligible_slot"],
          "the reservation count is 28 slots x 48 = 1344 -- reservations "
          "over ALL slots; the plan will use the 21 eligible "
          "(21x48x3 = 3024 units)")
    for post in ("docs/audits/evidence/repro_runs/"
                 "m4_c32_c38_post_2026_09_10.py",
                 "docs/audits/evidence/repro_runs/"
                 "m4_c32_c38_post_2026_09_10.out"):
        check(post in tracked,
              f"the sealed POST is retained: {Path(post).name}")
    for tpl, field in (
        ("docs/audits/evidence/"
         "MUSASHI_M4_CONFIRMATION_DESIGN_REVIEW_TEMPLATE_2026_09_10.json",
         "Musashi design review"),
        ("docs/audits/evidence/"
         "OWNER_M4_CONFIRMATION_EXECUTION_TEMPLATE_2026_09_10.json",
         "owner execution"),
    ):
        raw = blob(repo, RETURN_TIP, tpl).decode()
        check("<" in raw and ">" in raw,
              f"the {field} TEMPLATE still carries placeholders "
              "(it grants nothing)")
    for root in (Path.home() / ".local/state/m4_confirmation_authority",
                 Path("/var/lib/m4_confirmation_authority")):
        check(not root.exists(),
              f"no authority record installed at {root}")
    print()

    # -- 8. live checkout: battery, planner, execute refusal ---------------
    if a.live_checkout:
        wt = Path(a.live_checkout)
        print("-- 8. live checkout (the git-reading guards)")
        print(f"checkout: {wt}")
        head = git(wt, "rev-parse", "HEAD").strip()
        check(head == RETURN_TIP, "the checkout is at the return tip", head)
        py = sys.executable
        bat = subprocess.run(
            [py, "-m", "pytest", "tests/test_m4_confirmation_protocol.py",
             "-q"], cwd=wt, capture_output=True, text=True)
        tail = [ln for ln in bat.stdout.splitlines() if " passed" in ln
                or " failed" in ln]
        check(bat.returncode == 0 and any("28 passed" in t for t in tail),
              "the C37 acceptance battery is 28 passed",
              tail[-1] if tail else bat.stdout[-200:])
        pl = subprocess.run(
            [py, "tools/m4_confirmation_runner.py", "plan"],
            cwd=wt, capture_output=True, text=True)
        try:
            plan = json.loads(pl.stdout)
        except json.JSONDecodeError:
            plan = {}
        check(plan.get("units_total") == 3024,
              "the planner reports 3024 units = 21 x 48 x 3",
              str(plan.get("units_total")))
        check(plan.get("eligible_slots") == 21
              and plan.get("generators_per_slot") == 48
              and plan.get("seeds_per_generator") == 3
              and plan.get("checkpoints_per_unit") == 4,
              "and the slot / generator / seed / checkpoint census "
              "is the frozen one")
        check(plan.get("execution_open") is False
              and plan.get("musashi_review_record_present") is False
              and plan.get("owner_execution_record_present") is False,
              "execution_open is false and BOTH records are absent")
        check(plan.get("mode") == "PLAN_ONLY_NO_AUTHORITY",
              "the planner's own mode is PLAN_ONLY_NO_AUTHORITY",
              str(plan.get("mode")))
        print(f"       census_sha256 = {plan.get('census_sha256')}")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "must_not_exist"
            ex = subprocess.run(
                [py, "tools/m4_confirmation_runner.py", "execute",
                 "--out", str(out)],
                cwd=wt, capture_output=True, text=True)
            msg = (ex.stdout + ex.stderr).strip().splitlines()
            check(any("REFUSED" in m and "ABSENT" in m for m in msg),
                  "execute REFUSES at the two-record gate",
                  msg[-1] if msg else "")
            check(not out.exists(),
                  "and refuses BEFORE creating the output root -- no "
                  "directory, array or ledger")
        print()

    # -- 9. summary -------------------------------------------------------
    failed = [lab for ok, lab in _results if not ok]
    print("-- summary")
    print(f"checks: {len(_results)}   passed: "
          f"{len(_results) - len(failed)}   failed: {len(failed)}")
    if failed:
        for lab in failed:
            print(f"  FAILED: {lab}")
        return 1
    print("VERDICT: the M4 C32-C38 CONFIRMATION plan is intact, its "
          "identities re-derive, and it remains PREPARED AND NOT "
          "EXECUTED.")
    print("BLOCKED ON: the Musashi design-review record and the owner "
          "execution record. Neither exists. Only both together open "
          "execution.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Failed as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(1)
