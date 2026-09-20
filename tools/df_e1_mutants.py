#!/usr/bin/env python3
"""RP32 POST battery for this round's new guards: each mutation removes ONE guard of the E1 loader,
the regimes module, the pilot runner or the weekly controller, and the test named beside it MUST fail.
A mutation the suite still passes is a guard that is not really tested.

    python tools/df_e1_mutants.py --out MUTANTS.json [--only M01]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

MUTANTS = [
    ("M01_extras_admitted", "tools/df_e1_loader.py",
     'raise ContractRefusal(f"columns not foreseen by the contract (declare a role or list them in `ignore`): {extra}")', 'extra = []',
     "tests/test_df_e1_loader.py", "test_RP27_an_unforeseen_numeric_column_is_refused_and_an_ignored_one_changes_nothing",
     "an unforeseen numeric column is admitted"),
    ("M02_train_split_covers_every_row", "tools/df_e1_loader.py",
     'edges[name] = [cur, cur + int(n * frac)]',
     'edges[name] = [0, n]',
     "tests/test_df_e1_loader.py", "test_RP27_the_scaler_is_fitted_on_train_windows_only_and_later_splits_cannot_alter_it",
     "the train split reaches later rows, so the scaler is no longer train-only"),
    ("M03_gap_ignored", "tools/df_e1_loader.py",
     'admissible = span_ok & inputs_finite', 'admissible = inputs_finite',
     "tests/test_df_e1_loader.py", "test_RP27_a_one_minute_jump_withdraws_every_window_whose_support_crosses_it",
     "a window whose support crosses a gap is admitted"),
    ("M04_activation_at_the_target", "tools/df_e1_loader.py",
     'tgt_valid = fin_t[origins + h] & act[origins, :]', 'tgt_valid = fin_t[origins + h] & act[origins + h, :]',
     "tests/test_df_e1_loader.py", "test_RP27_electricity_activation_is_causal_and_never_active_clients_are_named",
     "activity judged with a future row"),
    ("M05_R1_not_frozen", "tools/df_e1_regimes.py",
     'l.trainable = not (regime == "R1" and l.name in det)', 'l.trainable = True',
     "tests/test_df_e1_regimes.py", "test_RP29_gradients_weights_resume_and_reload_prove_the_regimes",
     "R1 does not freeze the detector"),
    ("M06_arrays_not_reverified", "tools/df_e1_pilot.py",
     'if not arrays.is_file() or sha_file(arrays) != rec.get("arrays_sha256"):', 'if not arrays.is_file():',
     "tests/test_df_e1_pilot.py", "test_RP30_units_share_the_initial_checkpoint_and_the_ae_and_are_verified_from_arrays",
     "altered arrays accepted"),
    ("M07_release_ignored", "tools/e3_weekly_controller.py",
     'avail = [r for r in self.releases if r.release <= bar_time]', 'avail = list(self.releases)',
     "tests/test_e3_weekly_controller.py", "test_RP31_release_and_fallback_are_executed_and_weeks_are_by_timestamp",
     "a model acts before its release"),
    ("M08_short_emitted", "tools/e3_weekly_controller.py",
     'action = HOLD\n                record.update(action=action, reason="INCOMPATIBLE_PROPOSAL_SHORT_REFUSED_HOLD")',
     'action = SHORT\n                record.update(action=action, reason="INCOMPATIBLE_PROPOSAL_SHORT_REFUSED_HOLD")',
     "tests/test_e3_weekly_controller.py", "test_RP31_close_to_flat_never_short_and_sizing_from_equity",
     "flat turns into short"),
    ("M09_week_by_counter", "tools/e3_weekly_controller.py",
     'monday = (ts - timedelta(days=ts.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)',
     'monday = ts.replace(hour=0, minute=0, second=0, microsecond=0)',
     "tests/test_e3_weekly_controller.py", "test_RP31_release_and_fallback_are_executed_and_weeks_are_by_timestamp",
     "the week is not the timestamp's week"),
    ("M10_pilot_scores_the_test_split", "tools/df_e1_pilot.py",
     'contract.splits = {"train": dev_train_days / (dev_train_days + dev_val_days), "validation": dev_val_days / (dev_train_days + dev_val_days)}',
     'contract.splits = {"train": 0.6, "validation": 0.2, "test": 0.2}',
     "tests/test_df_e1_pilot.py", "test_RP30_prepare_uses_the_real_loader_with_preflight_and_a_train_only_scaler_and_refuses_altered_bytes",
     "a test split exists in the pilot's enumerator"),
    # RP55: the guards this round added, each removed on its own
    ("M11_deliveries_written_without_the_lock", "tools/df_e1_governed.py",
     "    with _Lock(path):\n        doc = json.loads(path.read_text()) if path.is_file() else base",
     "    if True:\n        doc = json.loads(path.read_text()) if path.is_file() else base",
     "tests/test_df_e1_receipt_concurrency.py", "test_RP55_concurrent_children_never_erase_each_others_deliveries",
     "concurrent children can erase each other's deliveries"),
    ("M12_terminals_written_without_the_lock", "tools/df_e1_receipts.py",
     "    fcntl.flock(handle, fcntl.LOCK_EX)\n    try:",
     "    try:",
     "tests/test_df_e1_receipt_concurrency.py", "test_RP55_concurrent_children_never_erase_each_others_terminals",
     "concurrent children can erase each other's accepted terminals"),
    ("M13_chronology_compared_as_text", "tools/df_e1_close.py",
     "    a, b = _instant(earlier), _instant(later)\n    if a is None or b is None:\n        return False\n    return a < b",
     "    if not isinstance(earlier, str) or not isinstance(later, str):\n        return False\n    return earlier < later",
     "tests/test_df_e1_chronology.py", "test_RP55_the_same_instant_spelled_two_ways_is_not_a_violation",
     "the same instant spelled two ways reads as an inversion"),
    ("M14_the_first_report_file_wins", "tools/df_e1_pilot.py",
     "    complete = [(p, d) for p, d in candidates if d.get(\"stopped\") in (None, \"\")]",
     "    complete = [candidates[0]] if candidates else []",
     "tests/test_df_e1_governing_report.py", "test_RP55_the_report_of_the_run_that_finished_governs",
     "the table can describe an aborted attempt of the same root"),
    ("M15_one_spool_for_every_unit", "tools/df_e1_governed.py",
     'outbox = GR.TerminalOutbox(Path(os.path.expanduser(outbox_dir or GR.DEFAULT_OUTBOX)).resolve() / "units" / unit_id)',
     'outbox = GR.TerminalOutbox(Path(os.path.expanduser(outbox_dir or GR.DEFAULT_OUTBOX)).resolve())',
     "tests/test_df_e1_governed_route.py", "test_RP51_a_destination_that_goes_down_and_comes_back_loses_no_terminal",
     "parallel children share one spool and flush each other's envelopes"),
]


def run(only=None) -> dict:
    out = []
    python = sys.executable
    for mid, file, old, new, test_file, test_name, why in MUTANTS:
        if only and mid not in only:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "repo"
            work.mkdir()
            for d in ("tools", "tests"):
                shutil.copytree(REPO / d, work / d)
            for extra in ("docs",):
                (work / extra).mkdir(exist_ok=True)
            shutil.copytree(REPO / "docs/tres_temas_entrevista", work / "docs/tres_temas_entrevista", dirs_exist_ok=True)
            path = work / file
            text = path.read_text()
            if text.count(old) != 1:
                out.append({"mutant": mid, "applied": False, "killed": None, "why": f"fragment not unique in {file}"})
                continue
            path.write_text(text.replace(old, new))
            proc = subprocess.run([python, "-m", "pytest", str(work / test_file), "-x", "-q", "-k", test_name],
                                  cwd=work, capture_output=True, text=True, timeout=1800,
                                  env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"})
            killed = proc.returncode != 0
            out.append({"mutant": mid, "applied": True, "killed": killed, "guard_removed": why, "test": f"{test_file}::{test_name}",
                        "returncode": proc.returncode, "tail": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""})
    return {"schema": "df_e1_mutants.v1", "mutants": out, "killed": sum(1 for m in out if m.get("killed")), "total": len(out),
            "rule": "every mutation removes one guard; the named test must fail (killed = true). A survivor is an untested guard."}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    doc = run(a.only)
    a.out.write_text(json.dumps(doc, indent=1))
    print(json.dumps({"killed": doc["killed"], "total": doc["total"],
                      "survivors": [m["mutant"] for m in doc["mutants"] if m.get("killed") is not True]}, indent=1))
    return 0 if doc["killed"] == doc["total"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
