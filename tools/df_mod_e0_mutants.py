#!/usr/bin/env python3
"""RP11: mutate every guard of the closure/metric code and show that the behavioural check goes RED
(the altered copy that the intact guard refuses is ACCEPTED under the mutant). Runs on a small real
campaign built like the tests do (real generator, real training with a tiny allowance), on copies;
every mutation is applied to a COPY of tools/ loaded under another name; the repository is untouched.

    python tools/df_mod_e0_mutants.py --out MUTANTS.json [--root EXISTING_FIXTURE_ROOT]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# (mutant id, file, original fragment, mutated fragment, case, message the intact guard emits)
MUTANTS = [
    ("M01_labels", "df_mod_e0_close.py",
     'if a is None or a.shape != value.shape or not np.array_equal(a, value):', 'if a is None or a.shape != value.shape:',
     "labels", "validation.y: not the generator's"),
    ("M02_denominator", "df_mod_e0_close.py",
     'if denom.shape != (p,) or not np.array_equal(denom, np.asarray(denom_expected)):', 'if denom.shape != (p,):',
     "denominator", "denominators are not the train seasonal-naive"),
    ("M03_rows", "df_mod_e0_close.py",
     'for key, value in (("rows", P["rows"]), ("y", P["y"]), ("naive", P["naive"]), ("oracle", P["oracle"])):',
     'for key, value in (("y", P["y"]), ("naive", P["naive"]), ("oracle", P["oracle"])):',
     "rows", "rows: not the generator's"),
    ("M04_mae", "df_mod_e0_close.py",
     'if (a_ is None) != (b_ is None) or (a_ is not None and abs(a_ - b_) > 1e-9):\n                    prob(f"{part}.{model}.{agg}',
     'if False:\n                    prob(f"{part}.{model}.{agg}',
     "mae", "validation.model.mae_mean"),
    ("M05_weights_unreadable", "df_mod_e0_close.py",
     'for why in replay_doc.get("problems") or []:\n            prob(f"replay: {why}")', 'for why in []:\n            prob(f"replay: {why}")',
     "unreadable", "weights unreadable"),
    ("M06_prediction_tolerance", "df_mod_e0_close.py",
     'if d is None or not np.isfinite(d) or d > tolerance["prediction_atol"]:', 'if d is None:',
     "parity", "validation predictions from the reloaded weights differ"),
    ("M07_frozen", "df_mod_e0_close.py",
     'if not frozen or rec.get("extractor_weight_change", 1.0) != 0.0:', 'if not frozen:',
     "unfrozen", "does not show a frozen extractor"),
    ("M08_non_finite_is_medido", "df_mod_e0.py",
     'if err.size == 0 or not np.isfinite(err).all() or not np.isfinite(denom[k]):', 'if err.size == 0:',
     "nan", "non-finite or empty arrays recorded as measured"),
    ("M09_parent", "df_mod_e0_close.py",
     'and abs(rc["mase_validation"] - file_mase) <= 1e-9', 'and True',
     "parent", "parent: the report's recorded outcome differs"),
    ("M10_all_verified_conjunction", "df_mod_e0_close.py",
     'out["all_verified"] = not out["not_verified"] and (out["parent_equal"] is not False)', 'out["all_verified"] = not out["not_verified"]',
     "parent", "ALL_VERIFIED_MUST_BE_FALSE"),
    ("M11_update_allowance", "df_mod_e0_close.py",
     'if int(tr.get("updates", -1)) > int(expected_rule["max_updates"]):', 'if False:',
     "updates", "exceed the allowance"),
    ("M12_strangers", "df_mod_e0_close.py",
     'strangers = sorted(set(on_disk) - set(pop["members"]) - set(pop["pilot_ids"]))\n    if strangers:', 'strangers = []\n    if strangers:',
     "stranger", "REFUSAL_EXPECTED"),
]


def _load_from(tools_dir: Path, name: str, alias: str):
    """Load `name` from a copy of tools/ under a distinct module name so that its own `_load` calls
    resolve inside that copy (HERE is the copy)."""
    for mod in [m for m in list(sys.modules) if m in ("df_mod_e0", "df_mod_e0_close", "df_mod_e0_design", "df_mod_e0_arch_design",
                                                        "df_mod_e0_verify", "df_mod_e0_run", "df_mod_e0_metrics")]:
        del sys.modules[mod]
    spec = importlib.util.spec_from_file_location(name, tools_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _apply(tools_copy: Path, file: str, old: str, new: str) -> bool:
    path = tools_copy / file
    text = path.read_text()
    if text.count(old) != 1:
        return False
    path.write_text(text.replace(old, new))
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None, help="an existing fixture root (built by tests/test_df_mod_e0_close.build_root)")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists")
    sys.path.insert(0, str(REPO / "tests"))
    T = importlib.import_module("test_df_mod_e0_close")
    CLOSE, E = T.CLOSE, T.E
    work = Path(tempfile.mkdtemp(prefix="mod-e0-mutants-"))
    t0 = time.process_time()
    if args.root is None:
        root = work / "fixture"
        T.build_root(root)
    else:
        root = args.root
    intact_out = work / "intact"
    intact = CLOSE.local_closure(root, intact_out, replays=True, workers=4)
    assert intact["closure"] == CLOSE.TOTAL, intact["not_verified"]
    campaign = {"root": root, "replays": intact_out / "replays"}
    results = {"schema": "df_mod_e0_mutants.v1", "fixture_root": str(root), "intact_closure": intact["closure"], "mutants": []}

    def altered_case(case: str, tmp: Path):
        """The altered copy of the case (as in the tests) and its root/out."""
        rootc, outc = T._copy(campaign, tmp, case)
        attempt = rootc / "attempts" / "H3__r1__s1__sequence"
        rec = json.loads((attempt / "cell.json").read_bytes())
        arr = T._load_arrays(attempt)
        new_arr = None
        if case == "labels":
            for part in ("train", "validation", "test"):
                arr[f"{part}_y"] = arr[f"{part}_pred"].copy()
            T._rescore(rec, arr)
            new_arr = arr
        elif case == "denominator":
            arr["denominator"] = arr["denominator"] * 100
            rec["mase_denominator"] = arr["denominator"].tolist()
            T._rescore(rec, arr)
            new_arr = arr
        elif case == "rows":
            for part in ("train", "validation", "test"):
                arr[f"{part}_rows"] = arr[f"{part}_rows"] + 100000
            new_arr = arr
        elif case == "mae":
            rec["scores"]["validation"]["model"]["mae_mean"] = 999999.0
        elif case == "unreadable":
            (attempt / "weights.weights.h5").write_bytes(b"not model weights")
        elif case == "parity":
            arr["validation_pred"] = arr["validation_naive"].copy()
            T._rescore(rec, arr)
            new_arr = arr
        elif case == "unfrozen":
            rec["extractor_weight_change"] = 0.5
        elif case == "nan":
            arr["validation_pred"][3, 2] = np.nan
            new_arr = arr
        elif case == "updates":
            rec["training"]["updates"] = 999999
        elif case == "parent":
            report = json.loads((rootc / "REPORT.json").read_text())
            report["cells"]["H3__r1__s1__sequence"]["mase_validation"] = 999999.0
            (rootc / "REPORT.json").write_text(json.dumps(report))
            return rootc, outc
        elif case == "stranger":
            shutil.copytree(rootc / "attempts" / "H2__h3__s1__profiles", rootc / "attempts" / "H2__h9__s1__profiles")
            return rootc, outc
        T._reseal(attempt, rec, new_arr)
        return rootc, outc

    for mid, file, old, new, case, message in MUTANTS:
        tools_copy = work / mid / "tools"
        shutil.copytree(HERE, tools_copy, ignore=shutil.ignore_patterns("__pycache__"))
        applied = _apply(tools_copy, file, old, new)
        entry = {"mutant": mid, "file": file, "case": case, "applied": applied, "expected_message_of_intact_guard": message}
        if not applied:
            entry["verdict"] = "MUTATION_SITE_NOT_FOUND"
            results["mutants"].append(entry)
            continue
        tmp = work / mid / "case"
        tmp.mkdir(parents=True)
        rootc, outc = altered_case(case, tmp)
        MUT = _load_from(tools_copy, "df_mod_e0_close", mid)
        try:
            with_mutant = MUT.local_closure(rootc, outc, replays=True, workers=2)
            unit = with_mutant["units"]["H3__r1__s1__sequence"]
            if message == "ALL_VERIFIED_MUST_BE_FALSE":
                accepted = bool(with_mutant["all_verified"])
            else:
                accepted = not any(message in q for q in unit["problems"])
            entry["under_mutant"] = {"closure": with_mutant["closure"], "all_verified": with_mutant["all_verified"], "unit_status": unit["status"],
                                     "unit_problems": unit["problems"][:6]}
            entry["mutant_accepts_the_altered_copy"] = accepted
        except MUT.ClosureRefusal as e:
            entry["under_mutant"] = {"refused": e.why}
            accepted = False
            entry["mutant_accepts_the_altered_copy"] = False
        if message == "REFUSAL_EXPECTED":
            accepted = "refused" not in (entry.get("under_mutant") or {})
            entry["mutant_accepts_the_altered_copy"] = accepted
        # the intact code on the same altered copy
        INTACT = _load_from(HERE, "df_mod_e0_close", "intact")
        try:
            with_intact = INTACT.local_closure(rootc, outc / "intact", replays=True, workers=2)
            unit_i = with_intact["units"]["H3__r1__s1__sequence"]
            if message == "ALL_VERIFIED_MUST_BE_FALSE":
                intact_red = not with_intact["all_verified"]
            else:
                intact_red = any(message in q for q in unit_i["problems"])
            entry["under_intact"] = {"closure": with_intact["closure"], "all_verified": with_intact["all_verified"], "unit_status": unit_i["status"]}
        except INTACT.ClosureRefusal as e:
            entry["under_intact"] = {"refused": e.why}
            intact_red = message == "REFUSAL_EXPECTED"
        entry["intact_guard_red"] = intact_red
        entry["verdict"] = "MUTANT_KILLED_BY_BEHAVIOUR_TEST" if (accepted and intact_red) else ("MUTANT_SURVIVED" if not accepted else "INTACT_NOT_RED")
        results["mutants"].append(entry)
        print(json.dumps({k: entry[k] for k in ("mutant", "case", "verdict")}), flush=True)
    results["summary"] = {"mutants": len(results["mutants"]), "killed": sum(m.get("verdict") == "MUTANT_KILLED_BY_BEHAVIOUR_TEST" for m in results["mutants"]),
                          "cpu_seconds": round(time.process_time() - t0, 1)}
    args.out.write_text(json.dumps(results, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(results["summary"]))
    shutil.rmtree(work, ignore_errors=True)
    return 0 if results["summary"]["killed"] == results["summary"]["mutants"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
