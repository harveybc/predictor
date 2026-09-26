"""Is the frozen M4 CONFIRMATION plan EXECUTABLE as frozen?

This asks a question nobody asked in the preparation cycle, and it can be
answered without constructing a single CONFIRMATION byte and without either
authority record.

Three facts, each established by running the frozen code or reading the frozen
bytes:

 1. the two-record gate still refuses, and still refuses BEFORE creating any
    directory, array or ledger;
 2. the sealed generator bank REFUSES to construct a CONFIRMATION generator
    unless an explicit allow_confirmation flag is passed (C33 kill 17);
 3. the sealed unit runner `_run_intervention_unit_v5` calls
    `gb.generate(u["role"], ...)` WITHOUT that flag, and
    `execute_confirmation` returns immediately after writing the pre-result
    ledger with the note that "unit execution proceeds only beyond this point".

Together: even with both records installed, the frozen code path would write a
3024-unit PENDING ledger and then fit ZERO units. The plan is frozen and
verifiable but it is not yet runnable; the C35 step-7 execution body was never
written. Closing that gap is code, not a plan change — but it is code that did
not exist when this screen was called ready.
"""
import inspect
import json
import subprocess
import sys
from pathlib import Path

EXECTREE = Path("/home/harveybc/Documents/GitHub/.worktrees/m4-conf-exec-20260926")
sys.path.insert(0, str(EXECTREE / "tools"))

FAILED = []
N = [0]


def check(ok, what, detail=""):
    N[0] += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {what}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILED.append(what)


def main() -> int:
    import m4_confirmation_protocol as cp
    import m4_confirmation_runner as cr
    import m4_generator_bank as gb

    print("M4 CONFIRMATION -- executability probe (no CONFIRMATION bytes)\n")

    # --- 1. the gate still refuses, before anything exists ---
    out = Path("/tmp/claude-1000/m4_gate_probe_should_never_exist")
    check(not out.exists(), "the probe output root does not exist beforehand")
    try:
        cr.execute_confirmation(repo_root=EXECTREE, out_root=out)
        check(False, "execute refuses at the two-record gate", "IT DID NOT REFUSE")
    except SystemExit as e:
        check("REFUSED" in str(e), "execute refuses at the two-record gate", str(e)[:100])
    check(not out.exists(), "and refuses BEFORE creating the output root")
    check(not cp.MUSASHI_REVIEW_RECORD_PATH.is_file(),
          "the design-review record is still ABSENT")
    check(not cp.OWNER_EXECUTION_RECORD_PATH.is_file(),
          "the owner execution record is still ABSENT")

    # --- 2. the sealed bank refuses to construct CONFIRMATION bytes ---
    refusal = None
    try:
        gb.generate("CONFIRMATION", "identity", "clean", 0)
        check(False, "the sealed bank refuses CONFIRMATION construction",
              "IT CONSTRUCTED ONE")
    except SystemExit as e:
        refusal = str(e)
        check("RESERVED" in refusal,
              "the sealed bank refuses CONFIRMATION construction", refusal[:90])
    sig = inspect.signature(gb.generate)
    check("allow_confirmation" in sig.parameters,
          "construction is opened only by an explicit allow_confirmation flag")
    check(sig.parameters["allow_confirmation"].default is False,
          "whose default is False")

    # --- 3. the sealed unit path never passes that flag ---
    import m4_v5_runner as rn
    src = inspect.getsource(rn._run_intervention_unit_v5)
    check("gb.generate(u[\"role\"]" in src.replace("'", '"'),
          "_run_intervention_unit_v5 constructs via gb.generate(u['role'], ...)")
    check("allow_confirmation" not in src,
          "and never passes allow_confirmation — so it CANNOT run a "
          "CONFIRMATION unit")
    esrc = inspect.getsource(cr.execute_confirmation)
    check("write_pre_result_ledger" in esrc,
          "execute_confirmation writes the pre-result ledger")
    # the note is split across source lines; compare on collapsed whitespace
    flat = " ".join(esrc.split()).replace('" "', "")
    check("unit execution proceeds only beyond this point" in flat,
          "and then RETURNS, with the note that unit execution lies beyond it")
    check("_run_intervention_unit_v5" not in esrc,
          "execute_confirmation contains no unit-execution loop at all")

    # --- what the census says would have to run ---
    plan = subprocess.run([sys.executable, "tools/m4_confirmation_runner.py", "plan"],
                          cwd=EXECTREE, capture_output=True, text=True)
    p = json.loads(plan.stdout)
    print(f"\nthe census the ledger would enumerate: {p['units_total']} units "
          f"= {p['eligible_slots']} slots x {p['generators_per_slot']} "
          f"generators x {p['seeds_per_generator']} seeds, "
          f"{p['checkpoints_per_unit']} checkpoints each")
    print(f"census_sha256 = {p['census_sha256']}")
    print("\nCONCLUSION: with both records installed the frozen code path would "
          "write a 3024-unit PENDING ledger and fit ZERO units. The C35 step-7 "
          "execution body does not exist in the frozen tools.")
    print(f"\nchecks: {N[0]}   failed: {len(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
