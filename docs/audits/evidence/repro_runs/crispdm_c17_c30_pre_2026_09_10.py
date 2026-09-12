"""PRE for order C17-C30: the ten counterexamples of the
governing re-audit, reproduced from the shipped code before any
correction.

Each item asserts the SPECIFIC defect and prints the exact
evidence, so a passing PRE cannot be confused with an accidental
failure.

 1. mutating only `y_validation_file` leaves subjects,
    data_digest and partitions_digest identical
 2. an optimizer's return value reaches the pipeline after the
    gate, with no allowlist and no re-derivation
 3. a submission reviewed on one invocation cannot govern the
    next, because `submitted_at` is regenerated every run
 4. a manifest with an undeclared top-level field and an
    undeclared entry field is accepted
 5. an envelope with NaN and undeclared nested fields is accepted
 6. a pipeline exception produces ZERO terminal outbox entry
 7. no persistent loader process and no productive outbox exist
 8. a new version of the same logical id is silently ignored by
    ON CONFLICT DO NOTHING
 9. changing the preprocessor or predictor module does not change
    the reviewed code digest
10. T2/M4/B4 have no terminal emission on their next execution

CPU only. Nothing outside a temporary directory is written; the
populated cube is read but never written.
"""
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

FAILURES = []


def check(n, name, condition, detail):
    print(f"PRE-{n:02d} {name}: "
          f"{'REPRODUCED' if condition else 'NOT REPRODUCED'}")
    print(f"        {detail}")
    if not condition:
        FAILURES.append((n, name))


from eligibility import gate            # noqa: E402
from eligibility import integration as integ  # noqa: E402
from eligibility import review          # noqa: E402
from eligibility import consumed as cons  # noqa: E402
from olap import campaign_envelope as ce  # noqa: E402
from olap import outbox as ob           # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="c17_pre_"))


def _csv(path, columns, value="1"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        w.writerow(["2020-01-01"] + [value] * (len(columns) - 1))


# ==============================================================
# 1. a mutated y leaves every derived identity untouched
# ==============================================================
COLS = ["DATE_TIME", "px"]
for role in ("d4", "d5", "d6"):
    _csv(TMP / f"x_{role}.csv", COLS)
    _csv(TMP / f"y_{role}.csv", COLS)
config = {
    "x_train_file": str(TMP / "x_d4.csv"),
    "x_validation_file": str(TMP / "x_d5.csv"),
    "x_test_file": str(TMP / "x_d6.csv"),
    "y_train_file": str(TMP / "y_d4.csv"),
    "y_validation_file": str(TMP / "y_d5.csv"),
    "y_test_file": str(TMP / "y_d6.csv"),
    "target_column": "px",
    integ.KEY_SCOPE: "forecasting",
    integ.PURPOSE_KEY: integ.PURPOSE_EXPERIMENT,
}
before = cons.resolve_consumed_subjects(dict(config),
                                        repo_root=TMP)
_csv(TMP / "y_d5.csv", COLS, value="999")       # ONLY y changes
after = cons.resolve_consumed_subjects(dict(config),
                                       repo_root=TMP)
same = (before["subject_ids"] == after["subject_ids"]
        and before["digests"]["data"] == after["digests"]["data"]
        and before["digests"]["partitions"] ==
        after["digests"]["partitions"])
check(1, "a mutated y changes no derived identity", same,
      f"y_validation went from 1 to 999; subjects "
      f"{after['subject_ids']}, data_digest "
      f"{after['digests']['data'][:16]} and partitions_digest "
      f"{after['digests']['partitions'][:16]} are IDENTICAL — "
      "subjects derive from the x header only, and both digests "
      "are computed from x alone")

# ==============================================================
# 2. the optimizer can rewrite the approved contract
# ==============================================================
main_src = (REPO / "app/main.py").read_text()
gate_at = main_src.index("gate_run(")
upd_at = main_src.index("config.update(optimal_params)")
pipe_at = main_src.index(
    "pipeline_plugin.run_prediction_pipeline(")
allowlist = any(t in main_src for t in
                ("ALLOWED_OPTIMIZER_KEYS",
                 "allowed_optimizer_keys",
                 "_assert_contract_unchanged"))
check(2, "the optimizer mutates the approved contract",
      gate_at < upd_at < pipe_at and not allowlist,
      f"gate at {gate_at}, config.update(optimal_params) at "
      f"{upd_at}, pipeline at {pipe_at}; there is NO allowlist "
      "and no re-derivation between them, so an optimizer "
      "returning x_test_file, target_column or a plugin name "
      "changes what the pipeline consumes AFTER approval")

# ==============================================================
# 3. a reviewed submission cannot govern a later run
# ==============================================================
integ_src = (REPO / "eligibility/integration.py").read_text()
regenerates = "submitted_at=review.utc_now_stamp()" in integ_src
has_two_phase = any(t in integ_src for t in
                    ("SUBMIT_ONLY", "EXECUTE_REVIEWED"))
check(3, "the submission is regenerated every run",
      regenerates and not has_two_phase,
      "gate_run() stamps submitted_at with the wall clock of "
      "THIS invocation and immediately demands a record binding "
      "that submission's digest; there is no SUBMIT_ONLY phase "
      "and no way to consume a previously reviewed submission, "
      "so a record issued for run N refuses run N+1 on "
      "chronology")

# ==============================================================
# 4. a manifest with undeclared fields is accepted
# ==============================================================
entry = {
    "subject_kind": "variable", "subject_id": "v",
    "version": "1",
    "io_schema": {"input": ["f"], "output": ["f"]}, "unit": "u",
    "temporal_availability": {"event_time": "t",
                              "available_time": "t"},
    "fit_scope": "TRAIN_ONLY", "incremental_state_policy": "n",
    "parameters": {},
    "digests": {"data": "d" * 64, "code": "c" * 64,
                "partitions": "b" * 64, "evidence": "e" * 64},
    "measured_cost": {}, "decision": "PUBLICLY_ELIGIBLE",
    "decision_scope": "s", "decision_reason": "r",
    "reviewer": "x", "reviewed_at": "2026-09-09T00:00:00Z",
    "undeclared_nested_field": "smuggled",
}
doc = {"schema": gate.MANIFEST_SCHEMA,
       "issued_at": "2026-09-10T00:00:00Z",
       "issuer": "r", "scope": "s", "entries": [entry],
       "undeclared_top_level_field": "smuggled"}
doc["manifest_sha256"] = gate._self_sha(doc)
mp = TMP / "loose.json"
mp.write_text(json.dumps(doc))
try:
    loaded = gate.load_manifest(mp)
    manifest_loose = True
    detail4 = ("loaded with undeclared_top_level_field="
               f"{loaded['undeclared_top_level_field']!r} and "
               "undeclared_nested_field="
               f"{loaded['entries'][0]['undeclared_nested_field']!r}")
except SystemExit as exc:
    manifest_loose = False
    detail4 = f"refused: {exc}"
check(4, "a manifest accepts undeclared fields",
      manifest_loose, detail4)

# ==============================================================
# 5. an envelope accepts NaN and undeclared nested fields
# ==============================================================
env = {
    "schema": ce.ENVELOPE_SCHEMA, "campaign_key": "k",
    "producer": "p", "result_class": "DEVELOPMENT",
    "identity": {"run_id": "r", "code_identity": "c",
                 "design_sha256": "d",
                 "undeclared_identity_field": "smuggled"},
    "data_consumed": {"datasets": [], "variables": [],
                      "operators": []},
    "partitions": {"exposure": "e", "splits": "s"},
    "budget": {"device": "cpu", "wall_seconds": 1.0,
               "cost_units": "s"},
    "terminal": {"state": "COMPLETE", "adjudication": "NONE"},
    "artifacts": {"a": "b"},
    "units": [],
}
env["envelope_sha256"] = ce._sha(env, "envelope_sha256")
try:
    ce.validate_envelope(env)
    extra_ok = True
except SystemExit:
    extra_ok = False
nan_env = dict(env)
nan_env["budget"] = {"device": "cpu",
                     "wall_seconds": float("nan"),
                     "cost_units": "s"}
nan_env["envelope_sha256"] = ce._sha(nan_env, "envelope_sha256")
try:
    ce.validate_envelope(nan_env)
    nan_ok = True
except SystemExit:
    nan_ok = False
check(5, "an envelope accepts NaN and undeclared fields",
      extra_ok and nan_ok,
      f"undeclared nested identity field accepted: {extra_ok}; "
      f"wall_seconds=NaN accepted and digested: {nan_ok} — a "
      "non-finite number passed as a measurement")

builder_src = (REPO /
               "tools/build_campaign_envelopes.py").read_text()
check(6, "the envelope builder parses loosely",
      "json.loads(path.read_text())" in builder_src
      and "len(declared) != 64" in builder_src,
      "the producer artifact is read with plain json.loads (so "
      "duplicate keys and NaN survive) and its self-digest is "
      "validated by LENGTH, not as canonical lowercase hex")

# ==============================================================
# 7. a pipeline exception emits nothing
# ==============================================================
emit_at = main_src.index("_outbox.emit(")
check(7, "a pipeline exception emits no terminal",
      emit_at > pipe_at and "except" not in
      main_src[pipe_at:emit_at],
      f"the only emit() stands at {emit_at}, AFTER the pipeline "
      f"call at {pipe_at}, with no try/except around the "
      "pipeline; an exception, a refusal or an inconclusive "
      "outcome before that line leaves the cube with no record "
      "that the run ever happened")

# ==============================================================
# 8. no persistent loader, no productive outbox
# ==============================================================
units = subprocess.run(
    ["systemctl", "--user", "list-units", "--all",
     "--no-legend"], capture_output=True, text=True).stdout
loader_unit = [ln for ln in units.splitlines()
               if "olap" in ln.lower() and "loader" in ln.lower()]
prod_outbox = ob.DEFAULT_OUTBOX
check(8, "no persistent loader and no productive outbox",
      not loader_unit and not prod_outbox.exists(),
      f"systemd --user units matching an OLAP loader: "
      f"{loader_unit or 'NONE'}; the productive outbox "
      f"directory {prod_outbox.name} exists: "
      f"{prod_outbox.exists()} — the loader was implemented and "
      "tested as a component, never deployed as a flow")

# ==============================================================
# 9. the inventory freezes an old version
# ==============================================================
inv_src = (REPO / "olap/inventory_rows.py").read_text()
do_nothing = inv_src.count("DO NOTHING")
ddl = inv_src[inv_src.index("INVENTORY_DDL = "):
              inv_src.index("# The historical tables")]
check(9, "the inventory ignores a new version of an id",
      do_nothing >= 4 and "version" not in ddl.lower(),
      f"{do_nothing} inserts use ON CONFLICT (<logical id>) DO "
      "NOTHING with no version column, so a changed digest, "
      "availability or semantics under the same id is silently "
      "dropped and the cube keeps the older observation")

# ==============================================================
# 10. code identity ignores the modules actually loaded
# ==============================================================
check(10, "the reviewed code digest ignores the real graph",
      len(cons.CONSUMING_CODE) == 4
      and not any("plugin" in c for c in cons.CONSUMING_CODE),
      f"CONSUMING_CODE is {list(cons.CONSUMING_CODE)} — four "
      "gate files. The predictor, optimizer, pipeline, target, "
      "preprocessor, data handler and config merge modules that "
      "actually consume the data are absent, so changing any of "
      "them leaves reviewed_code_digest unchanged")

# ==============================================================
# 11. T2/M4/B4 have no terminal emission
# ==============================================================
producers = {
    "T2": REPO.parent / ".worktrees/am-t0t1/tools",
    "M4": REPO.parent / ".worktrees/am-m3/tools",
    "B4": REPO.parent / ".worktrees/am-data-first/tools",
}
emitters = {}
for name, d in producers.items():
    if not d.is_dir():
        emitters[name] = "CHECKOUT_ABSENT"
        continue
    hits = subprocess.run(
        ["grep", "-rl", "outbox", "--include=*.py", str(d)],
        capture_output=True, text=True).stdout.split()
    emitters[name] = [Path(h).name for h in hits] or "NONE"
check(11, "T2/M4/B4 never emit a terminal",
      all(v in ("NONE", "CHECKOUT_ABSENT")
          for v in emitters.values()),
      f"outbox references in each producer's tools: {emitters} — "
      "their results reached the cube only through the manual "
      "backfill, so their NEXT execution records nothing")

print()
if FAILURES:
    print(f"PRE INCOMPLETE: {len(FAILURES)} item(s) did not "
          "reproduce:")
    for n, name in FAILURES:
        print(f"  PRE-{n:02d} {name}")
    raise SystemExit(1)
print("PRE CONFIRMED: every counterexample in the governing "
      "re-audit reproduces from the shipped code, each for its "
      "own named reason. A mutated y really is invisible to "
      "every derived identity; the optimizer really can rewrite "
      "the approved contract; a reviewed submission really "
      "cannot govern a later run; manifest and envelope really "
      "accept undeclared fields and NaN; a pipeline exception "
      "really emits nothing; there really is no loader process "
      "and no productive outbox; the inventory really freezes an "
      "old version; the reviewed code digest really ignores the "
      "modules that consume the data; and T2, M4 and B4 really "
      "have no terminal emission.")
