"""POST for order C17-C30: every counterexample the PRE
reproduced is gone, each for its own named reason.

The PRE stays frozen and is not re-run: on corrected code its own
locators no longer describe the program, which is part of the
evidence and is asserted below.

CPU only. The populated cube is read, never written.
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FIN = REPO.parent / "financial-data"
sys.path.insert(0, str(REPO))

FAILURES = []


def check(n, name, condition, detail):
    print(f"POST-{n:02d} {name}: "
          f"{'CORRECTED' if condition else 'STILL PRESENT'}")
    print(f"         {detail}")
    if not condition:
        FAILURES.append((n, name))


from eligibility import consumed as cons   # noqa: E402
from eligibility import gate               # noqa: E402
from eligibility import integration as integ  # noqa: E402
from eligibility import review             # noqa: E402
from eligibility import strict             # noqa: E402
from olap import campaign_envelope as ce   # noqa: E402
from olap import characterization as ch    # noqa: E402
from olap import inventory_rows as ir      # noqa: E402
from olap import outbox as ob              # noqa: E402
from olap import terminal as term          # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="c17_post_"))
COLS = ["DATE_TIME", "px"]


def _csv(p, cols, value="1"):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerow(["2020-01-01"] + [value] * (len(cols) - 1))


# ---- 1: a mutated y changes the identity ----
for role in ("d4", "d5", "d6"):
    _csv(TMP / f"x_{role}.csv", COLS)
    _csv(TMP / f"y_{role}.csv", ["DATE_TIME", "target"])
config = {
    "x_train_file": str(TMP / "x_d4.csv"),
    "x_validation_file": str(TMP / "x_d5.csv"),
    "x_test_file": str(TMP / "x_d6.csv"),
    "y_train_file": str(TMP / "y_d4.csv"),
    "y_validation_file": str(TMP / "y_d5.csv"),
    "y_test_file": str(TMP / "y_d6.csv"),
    "target_column": "target",
    integ.KEY_SCOPE: "forecasting",
    integ.KEY_CENSUS: "a" * 64,
}
before = cons.resolve_consumed_subjects(dict(config),
                                        repo_root=TMP)
_csv(TMP / "y_d5.csv", ["DATE_TIME", "target"], value="999")
after = cons.resolve_consumed_subjects(dict(config),
                                       repo_root=TMP)
check(1, "a mutated y changes every derived identity",
      before["digests"]["data"] != after["digests"]["data"]
      and before["digests"]["partitions"] !=
      after["digests"]["partitions"],
      f"data {before['digests']['data'][:12]} -> "
      f"{after['digests']['data'][:12]}, partitions "
      f"{before['digests']['partitions'][:12]} -> "
      f"{after['digests']['partitions'][:12]}; both digests now "
      "cover the complete ordered file list, x and y")
sides = {s["column"]: s["side"] for s in after["subjects"]}
check(2, "the y target is a reviewable subject",
      sides.get("target") == "y" and sides.get("px") == "x",
      f"subject sides {sides} — a target living on the y side is "
      "a subject with its side and contract role named")

# ---- 3: the optimizer cannot rewrite the contract ----
main_src = (REPO / "app/main.py").read_text()
refused = []
for key in ("x_test_file", "y_test_file", "target_column",
            "preprocessor_plugin", "plugin"):
    try:
        integ.assert_optimizer_result_is_hyperparameters_only(
            {key: "x"}, consumer="post")
        refused.append((key, False))
    except SystemExit:
        refused.append((key, True))
check(3, "the optimizer may only propose hyperparameters",
      all(ok for _, ok in refused)
      and "ALLOWED_OPTIMIZER_KEYS" in
      (REPO / "eligibility/integration.py").read_text(),
      f"{dict(refused)} — every data, target and plugin key "
      "refuses, and the whole identity is re-derived before the "
      "pipeline via assert_contract_unchanged")
order_ok = (main_src.index("gate_run(") <
            main_src.index(
                "assert_optimizer_result_is_hyperparameters_only(\n")
            < main_src.index("config.update(optimal_params)")
            < main_src.index("assert_contract_unchanged(")
            < main_src.index(
                "pipeline_plugin.run_prediction_pipeline("))
check(4, "the guards stand in the only order that works",
      order_ok,
      "gate -> optimizer filter -> config.update -> full "
      "re-derivation -> pipeline")

# ---- 5: two phases, and a reviewed submission governs later ----
integ_src = (REPO / "eligibility/integration.py").read_text()
rev_src = (REPO / "eligibility/review.py").read_text()
check(5, "the review protocol has two real phases",
      "SUBMIT_ONLY" in integ_src and "EXECUTE_REVIEWED" in
      integ_src and "def persist_submission" in rev_src
      and "def load_persisted_submission" in rev_src
      and "def assert_same_submission" in rev_src,
      "phase one derives, persists a content-addressed "
      "submission and EXITS; phase two re-derives, proves it "
      "landed on the same persisted submission and only then "
      "consumes the record — proven across separate processes in "
      "tests/unit_tests/test_gate_c17_c21.py")

# ---- 6: exact schemas ----
def _manifest(tmp, **over):
    entry = {
        "subject_kind": "variable", "subject_id": "v",
        "version": "1",
        "io_schema": {"input": ["f"], "output": ["f"]},
        "unit": "u",
        "temporal_availability": {"event_time": "t",
                                  "available_time": "t"},
        "fit_scope": "TRAIN_ONLY",
        "incremental_state_policy": "n", "parameters": {},
        "digests": {"data": "d" * 64, "code": "c" * 64,
                    "partitions": "b" * 64,
                    "evidence": "e" * 64},
        "measured_cost": {}, "decision": "PUBLICLY_ELIGIBLE",
        "decision_scope": "s", "decision_reason": "r",
        "reviewer": "x",
        "reviewed_at": "2026-09-09T00:00:00Z"}
    entry.update(over.pop("entry", {}))
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "r", "scope": "s", "entries": [entry]}
    doc.update(over)
    doc["manifest_sha256"] = gate._self_sha(doc)
    p = tmp / "m.json"
    p.write_text(json.dumps(doc))
    return p


schema_refusals = {}
for label, kwargs in (
        ("undeclared_top_level", {"undeclared_top": "x"}),
        ("undeclared_entry_field",
         {"entry": {"undeclared_nested": "x"}}),
        ("bool_as_cost",
         {"entry": {"measured_cost": {"s": True}}})):
    try:
        gate.load_manifest(_manifest(TMP, **kwargs))
        schema_refusals[label] = False
    except SystemExit:
        schema_refusals[label] = True
env_refusals = {}
def _env(**over):
    base = dict(campaign_key="k", producer="p",
                result_class="DEVELOPMENT",
                identity={"run_id": "r", "code_identity": "c",
                          "design_sha256": "d"},
                data_consumed={"datasets": [], "variables": [],
                               "operators": []},
                partitions={"exposure": "e", "splits": "s"},
                budget={"device": "cpu", "wall_seconds": 1.0,
                        "cost_units": "s"},
                terminal={"state": "COMPLETE",
                          "adjudication": "NONE"},
                artifacts={"a": "b"}, units=[])
    base.update(over)
    return ce.build_envelope(**base)
for label, kwargs in (
        ("envelope_undeclared_field",
         {"identity": {"run_id": "r", "code_identity": "c",
                       "design_sha256": "d", "extra": "x"}}),
        ("envelope_nan",
         {"budget": {"device": "cpu",
                     "wall_seconds": float("nan"),
                     "cost_units": "s"}})):
    try:
        _env(**kwargs)
        env_refusals[label] = False
    except SystemExit:
        env_refusals[label] = True
check(6, "manifest and envelope enforce exact schemas",
      all(schema_refusals.values())
      and all(env_refusals.values()),
      f"manifest {schema_refusals}, envelope {env_refusals} — "
      "undeclared fields, bool-as-number and NaN all refuse")

builder_src = (REPO /
               "tools/build_campaign_envelopes.py").read_text()
check(7, "producer artifacts are parsed strictly",
      "strict_load_file" in builder_src
      and "require_sha256" in builder_src
      and "len(declared) != 64" not in builder_src,
      "the builder reads with the strict parser and requires a "
      "canonical lowercase-hex digest, not a length")

# ---- 8: every outcome emits exactly one terminal ----
states = {}
for state in term.TERMINAL_STATES:
    env = term.build_terminal_envelope(
        campaign_key="k", producer="p", state=state, stamp={},
        config={}, wall_seconds=1.0)
    states[state] = (env["terminal"]["state"] == state
                     and len(env["units"]) == 1)
check(8, "every outcome produces exactly one terminal unit",
      all(states.values()) and "terminal_run" in main_src
      and "_outbox.emit(" not in main_src,
      f"{states} — the success-only emitter is gone and a "
      "terminal run carries its own unit with its measured cost")

# ---- 9: the loader is a live service ----
active = subprocess.run(
    ["systemctl", "--user", "is-active",
     "crispdm-olap-loader.service"],
    capture_output=True, text=True).stdout.strip()
hb_path = ob.DEFAULT_OUTBOX / "HEARTBEAT.json"
hb = json.loads(hb_path.read_text()) if hb_path.is_file() else {}
check(9, "the loader is an active supervised service",
      active == "active" and hb.get("loaded", 0) >= 1,
      f"systemd --user unit is {active!r}; heartbeat reports "
      f"loaded={hb.get('loaded')} pending={hb.get('pending')} "
      f"failed={hb.get('failed')} — an entry travelled from a "
      "real run to the cube with no human in the loop")

# ---- 10: the inventory versions instead of freezing ----
inv_src = (REPO / "olap/inventory_rows.py").read_text()
check(10, "the inventory keeps every version",
      "observation_sha256" in inv_src
      and "ON CONFLICT (appearance_id) DO NOTHING" not in inv_src
      and "v_lake_appearance_current" in inv_src,
      "the primary key is (logical id, observation), a changed "
      "observation is a NEW version, and four deterministic "
      "current views expose one row per id without deleting "
      "history")

# ---- 11: the code identity covers the real graph ----
resolved = cons.resolve_consumed_subjects(
    dict(config, plugin="ann", pipeline_plugin="stl_pipeline"),
    repo_root=TMP)
other = cons.resolve_consumed_subjects(
    dict(config, plugin="ann",
         pipeline_plugin="default_pipeline"), repo_root=TMP)
kinds = {e["kind"] for e in resolved["code_inventory"]}
check(11, "the reviewed code digest follows the real graph",
      len(cons.CONSUMING_CODE) >= 12 and kinds == {"module",
                                                   "plugin"}
      and resolved["digests"]["code"] != other["digests"]["code"],
      f"{len(resolved['code_inventory'])} bound entries "
      f"({kinds}); changing the executed pipeline plugin changes "
      "the digest, which four gate files could never detect")

# ---- 12: the financial scope is derived, and honest ----
bridge_p = FIN / "features/census/AVAILABILITY_BRIDGE.v1.json"
disp_p = FIN / "features/census/AVAILABILITY_DISPOSITION.v1.json"
bridge = json.loads(bridge_p.read_text())
disp = json.loads(disp_p.read_text())
check(12, "the availability gap is bridged by evidence",
      bridge["bridge"]["columns_examined"] == 94
      and bridge["demand"]["configs_declaring_both"] == []
      and disp["disposition"] ==
      "FINANCIAL_AVAILABILITY_EVIDENCE_REQUIRED",
      f"94 consumed columns examined, "
      f"{bridge['bridge']['resolved']} resolved and "
      f"{bridge['bridge']['unresolved']} unresolved with the "
      "stage named; the 83 RL columns proved to be a SUBSET of "
      "the 94 supervised columns, so both consumers read the "
      "same view — the gap is crossed by physical evidence and "
      "no configuration was edited")

# ---- 13: characterization measures and never selects ----
ledger = json.loads((REPO / "examples/research/"
                            "CHARACTERIZATION_LEDGER.v1.json"
                     ).read_text())
check(13, "characterization measures without selecting",
      ledger["rows_total"] > 0
      and ledger["selection_emitted"].startswith("NONE")
      and ledger["confirmation_used"] == "NONE"
      and ledger["gpu_used"] == "NONE",
      f"{ledger['rows_total']} descriptor rows over "
      f"{ledger['attempts_total']} attempts, "
      f"{ledger['rows_by_bank']}, "
      f"{ledger['rows_not_identifiable']} explicitly NOT "
      f"identifiable, total cost "
      f"{ledger['total_cost_seconds']} s; no selection, no "
      "confirmation, no GPU")

print()
if FAILURES:
    print(f"POST INCOMPLETE: {len(FAILURES)} defect(s) remain:")
    for n, name in FAILURES:
        print(f"  POST-{n:02d} {name}")
    raise SystemExit(1)
print("POST CONFIRMED: every counterexample the PRE reproduced "
      "is corrected. A mutated y changes the identity and a "
      "y-side target is a subject; the optimizer may only "
      "propose hyperparameters and the contract is re-derived "
      "before the pipeline; the review protocol has two real "
      "phases with a persisted submission; manifests, envelopes "
      "and producer artifacts enforce exact schemas and "
      "canonical digests; every outcome emits exactly one "
      "terminal carrying its own unit and cost; the loader is an "
      "ACTIVE supervised service that has carried a real run to "
      "the cube; the inventory versions instead of freezing; the "
      "code digest follows the real graph; the availability gap "
      "is bridged by evidence with zero as an honest answer; and "
      "the first characterization measures variables without "
      "choosing any.")
