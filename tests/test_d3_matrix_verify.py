"""K1: the verified matrix is sealed only when every unit, variable, operator and test of the
frozen population is present exactly once, its bytes match its receipt, and each verdict is
what its own tests say. The exploratory summary is a different thing and says so.

The first two rules are the reviewer's counterexamples (Musashi, 2026-09-16), the others the
cases the order names: a file modified after COLLECT, a failed test under a favourable
verdict, a row of another variable, a duplicated receipt. A fixture here is a throwaway run
root — FREEZE.json, collected/ with terminals and rows — never the real run.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


matrix = _load("df_d3_matrix")
design = _load("df_d3_design")

DESIGN = design.D3_AMENDMENT_V1["design_sha256"]
OPS = [{"kind": "op_a", "params": {}, "spec_sha256": "a" * 64, "twin": "op_a_centred"},
       {"kind": "op_b", "params": {}, "spec_sha256": "b" * 64, "twin": "NOT_APPLICABLE"}]
WORKER_SHA = "w" * 64


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rows_for(unit, variables, *, outcomes=None, verdict=None, n_samples=64):
    """A complete, consistent row set: 12 PASSED tests + 1 verdict per variable x operator."""
    rows = []
    for v in variables:
        for op in OPS:
            base = {"schema": "df_fact_d3_mechanics.v1", "run_id": "r", "host_role": "A",
                    "design_sha256": DESIGN, "bank": "SYNTHETIC", "unit_id": unit,
                    "family": "f", "dataset_id": f"synthetic.x.{unit}",
                    "contract_sha256": "c" * 64, "variable": v, "operator_kind": op["kind"],
                    "operator_group": "g", "operator_params": "{}",
                    "spec_sha256": op["spec_sha256"], "fit_scope": "NONE",
                    "n_samples": n_samples, "timestamp_meaning": "SAMPLE_INDEX",
                    "code_sha256": WORKER_SHA, "result_class": "MECHANICAL",
                    "classification": "NON_GOVERNING", "cpu_seconds": None, "value": None,
                    "detail": ""}
            for test in design.REQUIRED_TESTS:
                outcome = (outcomes or {}).get(test, "PASSED")
                rows.append(dict(base, test=test, outcome=outcome))
            rows.append(dict(base, test="verdict", outcome=verdict or "MECHANICALLY_ACCEPTED",
                             value=1.0 if (verdict or "MECHANICALLY_ACCEPTED")
                             == "MECHANICALLY_ACCEPTED" else 0.0,
                             detail=json.dumps({"failed": [], "scoped": [], "undecided": []})))
    return rows


class Fixture:
    """A throwaway run root with a frozen population and a collected ledger."""

    def __init__(self, root: Path, units):
        self.root = root
        self.units = {u: n for u, n in units}
        bank = root / "bank"
        for unit, n in units:
            d = bank / unit
            d.mkdir(parents=True)
            (d / "UNIT.json").write_text(json.dumps(
                {"unit_id": unit, "n_variables": n, "digests": {"observed_signal": "c" * 64}}))
        freeze = {"schema": "d3_mechanics_freeze.v1", "design_sha256": DESIGN,
                  "freeze_sha256": "", "operators": OPS,
                  "code_sha256s": {"df_d3_unit_worker": WORKER_SHA},
                  "bank": {"root": str(bank), "count": len(units),
                           "units": [{"unit_id": u, "n_variables": n, "family": "f",
                                      "path": str(bank / u)} for u, n in units]},
                  "toys": []}
        body = {k: v for k, v in freeze.items() if k != "freeze_sha256"}
        freeze["freeze_sha256"] = hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.freeze = freeze
        (root / "FREEZE.json").write_text(json.dumps(freeze))
        # the run's shards and dispatch: every unit belongs to shard "s", launched on role "A"
        (root / "shards" / "s").mkdir(parents=True)
        (root / "shards" / "s" / "MEMBERS.txt").write_text("".join(u + "\n" for u, _ in units))
        (root / "dispatch").mkdir()
        (root / "dispatch" / "DISPATCH_RECEIPT.json").write_text(json.dumps(
            {"schema": "df_dispatch_receipt.v1", "final": True,
             "jobs": {"r-s": {"attempt": 1, "role": "A", "status": "COMPLETED", "reason": ""}}}))
        # the campaign record the report conserved: the freeze and design it was registered with
        config = {"schema": "d3_mechanics_execution.v1", "run_id": "r",
                  "freeze_sha256": freeze["freeze_sha256"], "design_sha256": DESIGN}
        (root / "REPORT.json").write_text(json.dumps(
            {"schema": "d3_mechanics_report.v1", "run_id": "r",
             "config_sha256": hashlib.sha256(json.dumps(
                 config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
             "synthetic_spec": {"freeze_sha256": freeze["freeze_sha256"],
                                "design_sha256": DESIGN, "units": len(units)},
             "campaigns": {"synthetic": {"campaign_sha256": "c" * 64, "units": len(units)}}}))
        self.receipt = {"schema": "d3_mechanics_collect.v1", "run_id": "r", "mismatched": 0,
                        "verified": 0, "units": []}

    def reseal_freeze(self):
        """Rewrite FREEZE.json with a canonical digest after a change (the reproducer's step);
        the campaign record still names the ORIGINAL freeze."""
        body = {k: v for k, v in self.freeze.items() if k != "freeze_sha256"}
        self.freeze["freeze_sha256"] = hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        (self.root / "FREEZE.json").write_text(json.dumps(self.freeze))

    def complete(self, unit, rows=None, *, attempt=1, role="A", shard="s", receipt=True):
        rows = rows if rows is not None else rows_for(
            unit, [f"v{i}" for i in range(self.units[unit])])
        adir = self.root / "collected" / role / shard / "attempts" / unit / f"attempt-{attempt}"
        adir.mkdir(parents=True)
        body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
        (adir / "rows.jsonl").write_bytes(body)
        self.terminal(unit, attempt=attempt, role=role, shard=shard, status="COMPLETED",
                      output_sha256=sha(body), rows_written=len(rows))
        if receipt:
            self.receipt["units"].append({"unit": unit, "role": role, "shard": shard,
                                          "attempt": attempt, "status": "COMPLETED",
                                          "output_verified": True, "rows": len(rows),
                                          "wall_seconds": 1.0, "cpu_seconds": 1.0})
            self.receipt["verified"] += 1
        return adir / "rows.jsonl"

    def fail(self, unit, *, attempt=1, role="A", shard="s", status="RESOURCE_EXCEEDED",
             receipt=True):
        self.terminal(unit, attempt=attempt, role=role, shard=shard, status=status,
                      output_sha256=None, rows_written=0)
        if not receipt:
            return
        self.receipt["units"].append({"unit": unit, "role": role, "shard": shard,
                                      "attempt": attempt, "status": status, "rows": 0,
                                      "wall_seconds": 1.0, "cpu_seconds": 1.0})

    def terminal(self, unit, *, attempt, role, shard, status, output_sha256, rows_written):
        tdir = self.root / "collected" / role / shard / "terminals"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{unit}.attempt-{attempt}.json").write_text(json.dumps(
            {"dataset_id": unit, "status": status, "output_sha256": output_sha256,
             "rows_written": rows_written, "code_sha256": WORKER_SHA, "run_id": "r",
             "reason": "" if status == "COMPLETED" else "WALL_TIME_LIMIT",
             "wall_seconds": 1.0, "cpu_seconds": 1.0}))

    def seal(self, name="COLLECT.json"):
        (self.root / name).write_text(json.dumps(self.receipt))
        return name


def verify(fx, name="COLLECT.json"):
    fx.seal(name)
    return matrix.verify(fx.root, name)


def refusal_kinds(result):
    return sorted({r["kind"] for r in result["refusals"]})


# --- the reviewer's two counterexamples ------------------------------------------------------

def test_a_single_favourable_verdict_row_without_its_twelve_tests_is_refused(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    only_verdict = [r for r in rows_for("u", ["v0"]) if r["test"] == "verdict"]
    fx.complete("u", only_verdict)
    out = verify(fx)
    assert out["verified"] is False
    assert "MISSING_TESTS" in refusal_kinds(out)
    assert out["operators"] == {} or all(
        op["verdicts"] == {} for op in out["operators"].values())


def test_a_missing_rows_file_is_a_refusal_never_a_silent_zero(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    path = fx.complete("u")
    path.unlink()
    out = verify(fx)
    assert out["verified"] is False
    assert "MISSING_FILE" in refusal_kinds(out)
    assert out["units"]["missing"] == ["u"]
    # the exploratory summary refuses too: it never returns zero units from a receipt of one
    with pytest.raises(SystemExit):
        matrix.aggregate(fx.root)


# --- the order's cases -----------------------------------------------------------------------

def test_a_file_modified_after_collect_is_refused_by_its_receipt_digest(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    path = fx.complete("u")
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    rows[0]["outcome"] = "PASSED"          # same shape, different bytes
    rows[0]["detail"] = "edited after the fact"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    out = verify(fx)
    assert out["verified"] is False
    assert "DIGEST_MISMATCH" in refusal_kinds(out)


def test_a_failed_test_under_a_favourable_verdict_is_a_contradiction(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    rows = rows_for("u", ["v0"], outcomes={"future_perturbation": "FAILED"},
                    verdict="MECHANICALLY_ACCEPTED")
    fx.complete("u", rows)
    out = verify(fx)
    assert out["verified"] is False
    assert "VERDICT_CONTRADICTION" in refusal_kinds(out)
    # the recomputed verdict is what the tests say, and it is reported
    bad = [r for r in out["refusals"] if r["kind"] == "VERDICT_CONTRADICTION"][0]
    assert bad["recomputed"] == "MECHANICALLY_REFUSED" and bad["recorded"] == "MECHANICALLY_ACCEPTED"


def test_a_row_of_a_variable_the_unit_does_not_have_is_unexpected(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    rows = rows_for("u", ["v0"]) + rows_for("u", ["v7"])
    fx.complete("u", rows)
    out = verify(fx)
    assert out["verified"] is False
    assert "UNEXPECTED_VARIABLE" in refusal_kinds(out)


def test_a_duplicated_receipt_entry_is_refused_not_counted_twice(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    fx.complete("u")
    fx.receipt["units"].append(dict(fx.receipt["units"][0]))
    out = verify(fx)
    assert out["verified"] is False
    assert "DUPLICATE_RECEIPT" in refusal_kinds(out)


def test_a_duplicated_row_is_refused(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    rows = rows_for("u", ["v0"])
    fx.complete("u", rows + [rows[0]])
    out = verify(fx)
    assert "DUPLICATE_ROW" in refusal_kinds(out)


# --- population, ledger and the seal ---------------------------------------------------------

def test_a_complete_consistent_run_is_sealed_with_its_population(tmp_path):
    fx = Fixture(tmp_path, [("u1", 1), ("u2", 3)])
    fx.complete("u1")
    fx.complete("u2")
    out = verify(fx)
    assert out["verified"] is True and out["refusals"] == []
    assert out["population"] == {"units": 2, "variables": 4, "operators": 2, "tests": 12,
                                 "design_sha256": DESIGN}
    assert out["units"] == {"completed": 2, "failed": 0, "missing": [], "expected": 2}
    assert out["operators"]["op_a"]["verdicts"] == {"MECHANICALLY_ACCEPTED": 4}
    assert out["operators"]["op_a"]["tests"]["future_perturbation"] == {"PASSED": 4}
    assert out["rows"] == 2 * 4 * 13


def test_a_recorded_failed_unit_stays_in_the_denominator_and_does_not_block_the_seal(tmp_path):
    fx = Fixture(tmp_path, [("u1", 1), ("u2", 1)])
    fx.complete("u1")
    fx.fail("u2")
    out = verify(fx)
    assert out["verified"] is True
    assert out["units"] == {"completed": 1, "failed": 1, "missing": [], "expected": 2}
    assert out["failed_units"] == [{"unit": "u2", "attempt": 1, "status": "RESOURCE_EXCEEDED",
                                    "reason": "WALL_TIME_LIMIT"}]


def test_a_frozen_unit_with_no_terminal_at_all_is_missing_scope(tmp_path):
    fx = Fixture(tmp_path, [("u1", 1), ("u2", 1)])
    fx.complete("u1")
    out = verify(fx)
    assert out["verified"] is False
    assert out["units"]["missing"] == ["u2"] and "MISSING_UNIT" in refusal_kinds(out)


def test_attempts_resolve_from_the_ledger_highest_attempt_never_by_score(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    fx.fail("u", attempt=1, receipt=False)        # killed; the receipt names the retry only
    fx.complete("u", attempt=2)                   # the retry
    out = verify(fx)
    assert out["verified"] is True and out["units"]["completed"] == 1
    # a receipt that names the wrong attempt disagrees with the ledger
    fx.receipt["units"] = [dict(u, attempt=1) for u in fx.receipt["units"] if u["attempt"] == 2]
    out = verify(fx, "COLLECT.wrong.json")
    assert "LEDGER_DISAGREES" in refusal_kinds(out)


def test_a_unit_outside_the_frozen_population_is_refused(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    fx.complete("u")
    fx.units["ghost"] = 1
    fx.complete("ghost")
    out = verify(fx)
    assert "UNEXPECTED_UNIT" in refusal_kinds(out)


def test_rows_bound_to_another_design_or_spec_are_refused(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    rows = rows_for("u", ["v0"])
    rows[3]["spec_sha256"] = "9" * 64
    fx.complete("u", rows)
    out = verify(fx)
    assert "SPEC_MISMATCH" in refusal_kinds(out)


def test_the_exploratory_summary_says_it_is_not_a_verification(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    fx.complete("u")
    fx.seal()
    summary = matrix.aggregate(fx.root)
    assert summary["verified"] is False and summary["schema"] == "d3_mechanics_summary.v1"
    assert "not a verification" in summary["note"]


# --- L1: the reviewer's three omissions, frozen before the fix ------------------------------------

def test_a_missing_unit_contract_is_a_refusal_not_an_omitted_binding(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    fx.complete("u")
    (tmp_path / "bank" / "u" / "UNIT.json").unlink()
    out = verify(fx)
    assert out["verified"] is False and "CONTRACT_UNBOUND" in refusal_kinds(out)


def test_declared_toys_without_their_records_are_a_refusal(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    fx.complete("u")
    fx.freeze["toys"] = [{"resource": "synthetic_ohlc_1h.csv", "role": "input_file"}]
    fx.reseal_freeze()
    out = verify(fx)
    assert out["verified"] is False and "TOYS_UNBOUND" in refusal_kinds(out)


def test_a_freeze_whose_digest_does_not_seal_its_body_is_refused(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    fx.complete("u")
    frozen = json.loads((tmp_path / "FREEZE.json").read_text())
    frozen["freeze_sha256"] = "0" * 64
    (tmp_path / "FREEZE.json").write_text(json.dumps(frozen))
    out = verify(fx)
    assert out["verified"] is False and "FREEZE_IDENTITY" in refusal_kinds(out)


def test_a_resealed_freeze_that_is_not_the_one_the_campaign_was_registered_with_is_refused(tmp_path):
    """A self-digest alone does not prove this is the design that governed the run."""
    fx = Fixture(tmp_path, [("u", 1)])
    fx.complete("u")
    fx.freeze["bank"]["units"][0]["family"] = "edited"
    fx.reseal_freeze()
    out = verify(fx)
    assert out["verified"] is False and "CAMPAIGN_RECORD_MISMATCH" in refusal_kinds(out)


def test_a_missing_campaign_record_is_a_refusal(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    fx.complete("u")
    (tmp_path / "REPORT.json").unlink()
    out = verify(fx)
    assert "CAMPAIGN_RECORD_MISSING" in refusal_kinds(out)


def test_freeze_cardinalities_are_validated(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    fx.complete("u")
    fx.freeze["bank"]["count"] = 7
    fx.reseal_freeze()
    (tmp_path / "REPORT.json").write_text(json.dumps(dict(
        json.loads((tmp_path / "REPORT.json").read_text()),
        synthetic_spec={"freeze_sha256": fx.freeze["freeze_sha256"], "design_sha256": DESIGN})))
    out = verify(fx)
    assert "FREEZE_CARDINALITY" in refusal_kinds(out)


def test_an_identical_copy_under_an_unassigned_location_is_a_transport_copy_never_counted_twice(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    fx.complete("u")
    import shutil
    shutil.copytree(tmp_path / "collected" / "A" / "s", tmp_path / "collected" / "B" / "s")
    out = verify(fx)
    assert out["verified"] is False
    assert "UNASSIGNED_LOCATION" in refusal_kinds(out)
    assert out["transport_copies"] == [{"unit": "u", "attempt": 1, "role": "B", "shard": "s",
                                        "identical_to": {"role": "A", "shard": "s"}}]
    assert out["units"]["completed"] <= 1 and out["rows"] == 2 * 13


def test_a_contradictory_duplicate_attempt_is_refused_never_resolved_by_path_order(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    fx.complete("u")
    other = rows_for("u", ["v0"], outcomes={"future_perturbation": "FAILED"},
                     verdict="MECHANICALLY_REFUSED")
    # the same unit and attempt, under the shard's own role but as a second, different record
    fx.receipt["units"] = []
    fx.complete("u", other, role="A", shard="s2", receipt=True)
    (tmp_path / "shards" / "s2").mkdir()
    (tmp_path / "shards" / "s2" / "MEMBERS.txt").write_text("u\n")
    receipt = json.loads((tmp_path / "dispatch" / "DISPATCH_RECEIPT.json").read_text())
    receipt["jobs"]["r-s2"] = {"attempt": 1, "role": "A", "status": "COMPLETED", "reason": ""}
    (tmp_path / "dispatch" / "DISPATCH_RECEIPT.json").write_text(json.dumps(receipt))
    out = verify(fx)
    assert out["verified"] is False
    assert "CONTRADICTORY_ATTEMPT" in refusal_kinds(out)


def test_an_unknown_terminal_status_and_a_discordant_id_are_refused(tmp_path):
    fx = Fixture(tmp_path, [("u1", 1), ("u2", 1)])
    fx.complete("u1")
    fx.fail("u2", status="MAYBE")
    out = verify(fx)
    assert "UNKNOWN_STATUS" in refusal_kinds(out)
    fx2 = Fixture(tmp_path / "b", [("u", 1)])
    fx2.complete("u")
    tpath = tmp_path / "b" / "collected" / "A" / "s" / "terminals" / "u.attempt-1.json"
    t = json.loads(tpath.read_text())
    t["dataset_id"] = "someone_else"
    tpath.write_text(json.dumps(t))
    out = verify(fx2)
    assert "ID_DISCORDANT" in refusal_kinds(out)


def test_a_terminal_under_a_shard_the_unit_does_not_belong_to_is_unassigned(tmp_path):
    fx = Fixture(tmp_path, [("u", 1)])
    fx.complete("u", role="A", shard="elsewhere")
    out = verify(fx)
    assert "UNASSIGNED_LOCATION" in refusal_kinds(out)


# --- L2 §5: a composite replay inherits by digest and is judged from the union -----------------------

def composite_pair(tmp_path, *, measured_outcome="PASSED", drop_measured=False,
                   break_source=False):
    """A source run (all 12 tests) and a child run that measured only non_causal_twin for
    op_a, inheriting the rest from the source by digest."""
    src = Fixture(tmp_path / "src", [("u", 1)])
    src.complete("u")
    src.seal()
    child = Fixture(tmp_path / "child", [("u", 1)])
    inherits = {"source_root": str(tmp_path / "src"), "source_run_id": "r",
                "source_freeze_sha256": src.freeze["freeze_sha256"],
                "source_design_sha256": DESIGN, "source_receipt": "COLLECT.json",
                "inherited_tests": [t for t in design.REQUIRED_TESTS if t != "non_causal_twin"],
                "measured_tests": ["non_causal_twin"], "measured_operators": ["op_a"],
                "justification": "fixture"}
    child.freeze["inherits"] = inherits
    child.reseal_freeze()
    config = {"schema": "d3_mechanics_execution.v1", "run_id": "r",
              "freeze_sha256": child.freeze["freeze_sha256"], "design_sha256": DESIGN}
    (tmp_path / "child" / "REPORT.json").write_text(json.dumps(
        {"schema": "d3_mechanics_report.v1", "run_id": "r",
         "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True,
                                                    separators=(",", ":")).encode()).hexdigest(),
         "synthetic_spec": {"freeze_sha256": child.freeze["freeze_sha256"],
                            "design_sha256": DESIGN, "units": 1},
         "campaigns": {"synthetic": {"campaign_sha256": "c" * 64, "units": 1}}}))
    rows = []
    if not drop_measured:
        base = [r for r in rows_for("u", ["v0"]) if r["operator_kind"] == "op_a"][0]
        rows.append(dict(base, test="non_causal_twin", outcome=measured_outcome))
        verdict = ("MECHANICALLY_ACCEPTED" if measured_outcome == "PASSED" else
                   "MECHANICALLY_REFUSED" if measured_outcome == "FAILED" else "INCONCLUSIVE")
        rows.append(dict(base, test="verdict", outcome=verdict,
                         value=1.0 if verdict == "MECHANICALLY_ACCEPTED" else 0.0,
                         detail=json.dumps({"failed": [], "scoped": [], "undecided": [],
                                            "scope": ["non_causal_twin"]})))
    child.complete("u", rows)
    if break_source:
        (tmp_path / "src" / "collected" / "A" / "s" / "attempts" / "u" / "attempt-1"
         / "rows.jsonl").write_text("{}\n")
    return src, child


def test_a_composite_replay_joins_measured_and_inherited_tests_and_recomputes_the_verdict(tmp_path):
    src, child = composite_pair(tmp_path, measured_outcome="INSUFFICIENT_TEST")
    out = verify(child)
    assert out["verified"] is True, out["refusals"]
    assert out["inherits"]["measured_tests"] == ["non_causal_twin"]
    a, b = out["operators"]["op_a"], out["operators"]["op_b"]
    assert a["verdicts"] == {"INCONCLUSIVE": 1}            # measured twin undecided
    assert a["tests"]["non_causal_twin"] == {"INSUFFICIENT_TEST": 1}
    assert a["tests"]["prefix_all_available"] == {"PASSED": 1}   # inherited from the source
    assert b["verdicts"] == {"MECHANICALLY_ACCEPTED": 1}   # not measured: all from the source


def test_a_composite_replay_refuses_when_the_source_no_longer_verifies(tmp_path):
    src, child = composite_pair(tmp_path, break_source=True)
    out = verify(child)
    assert out["verified"] is False
    assert "INHERITED_SOURCE_UNVERIFIED" in refusal_kinds(out)


def test_a_composite_replay_refuses_a_measured_cell_that_is_missing(tmp_path):
    src, child = composite_pair(tmp_path, drop_measured=True)
    out = verify(child)
    assert out["verified"] is False
    assert "MISSING_CELL" in refusal_kinds(out)


def test_a_composite_replay_refuses_a_source_whose_freeze_is_not_the_declared_one(tmp_path):
    src, child = composite_pair(tmp_path)
    child.freeze["inherits"]["source_freeze_sha256"] = "9" * 64
    child.reseal_freeze()
    rec = json.loads((tmp_path / "child" / "REPORT.json").read_text())
    config = {"schema": "d3_mechanics_execution.v1", "run_id": "r",
              "freeze_sha256": child.freeze["freeze_sha256"], "design_sha256": DESIGN}
    rec["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True,
                                                     separators=(",", ":")).encode()).hexdigest()
    rec["synthetic_spec"]["freeze_sha256"] = child.freeze["freeze_sha256"]
    (tmp_path / "child" / "REPORT.json").write_text(json.dumps(rec))
    out = verify(child)
    assert "INHERITED_SOURCE_MISMATCH" in refusal_kinds(out)
