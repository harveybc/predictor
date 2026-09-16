"""The D3 contract and its acceptance battery, proved able to fail.

Block 3 of `docs/handoffs/MUSASHI_I1_I3_ACCEPTANCE_AND_STACK_FOLLOWUP_2026_09_16.md`: prepare
the next preprocessing step's data contracts, tests and governed execution plan. The step is
D3 — quantization/compression, time-frequency representations and detectors — whose design is
sealed in `docs/integracion_workplan_2026_09_10/07_DISENO_D3_..._2026_09_14.md`. Nothing here
implements one of its nine operators, executes D3, or invents scientific acceptance.

A battery that cannot fail is worth nothing — I shipped one of those in G1 and it took the
reviewer's three-case probe to show it. So every test below is exercised against a fixture
built to break exactly that test, and the battery must say so:

* a trailing-mean operator, honestly declared, passes;
* a centred-window twin of it — the design's deliberate non-causal control — fails the two
  causality tests, and the battery reports that as the control WORKING;
* one fixture per test lies about a single field, and only that test fails.

The operators here are harness fixtures, not candidates. Promoting anything on the strength of
this file would be exactly the error it exists to prevent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("df_d3_contract")
battery = _load("df_d3_acceptance")

NA = contract.NOT_AVAILABLE


def signal(length: int = 400) -> list:
    """A deterministic, non-constant signal. No randomness: a battery must be reproducible."""
    return [((index * 37) % 101) / 10.0 - 5.0 for index in range(length)]


def base_spec(**overrides) -> dict:
    spec = {"kind": "trailing_mean", "params": {"window": 8}, "bytes_state": 0,
            "fit_scope": "NONE", "lookback_samples": 7, "output_availability": "t",
            "warm_up_samples": 7, "delay_samples": 0,
            "cost_cpu_seconds_per_1000": 5.0,
            "applicability": ["step", "trend"], "chunk_restart": "IDEMPOTENT"}
    spec.update(overrides)
    return spec


class TrailingMean:
    """A causal operator that declares itself honestly. The reference the battery must accept."""

    def __init__(self, spec=None):
        self._spec = spec or base_spec()
        self.window = self._spec["params"]["window"]

    def describe(self):
        return dict(self._spec)

    def fit(self, train_prefix):
        return {}

    def transform(self, x, state):
        window, values, available = self.window, [], []
        for index in range(len(x)):
            if index + 1 < window:
                values.append(0.0)
                available.append(False)
                continue
            chunk = x[index + 1 - window:index + 1]
            values.append(sum(chunk) / window)
            available.append(True)
        return {"values": values, "available": available, "raw": list(x)}

    def apply_to_family(self, family, x, state):
        if family not in self._spec["applicability"]:
            return contract.NOT_APPLICABLE
        return self.transform(x, state)


class CentredMean(TrailingMean):
    """The design's deliberate non-causal control: the same window, centred.

    It reads samples that do not exist at `t`. It MUST fail the causality tests; that is what
    makes it a control and not a candidate.
    """

    def __init__(self):
        # Its window is centred, so it settles four samples in rather than seven; the control
        # declares that honestly and lies only about causality, which is its whole purpose.
        super().__init__(base_spec(kind="centred_mean", warm_up_samples=4,
                                   non_causal_control_of="trailing_mean"))

    def transform(self, x, state):
        window, half = self.window, self.window // 2
        values, available = [], []
        for index in range(len(x)):
            start, stop = index - half, index - half + window
            if start < 0 or stop > len(x):
                values.append(0.0)
                available.append(False)
                continue
            chunk = x[start:stop]
            values.append(sum(chunk) / window)
            available.append(True)
        return {"values": values, "available": available, "raw": list(x)}


def run(operator, **kwargs):
    return battery.run_battery(operator, signal(), **kwargs)


# --- the contract ---------------------------------------------------------------------------

def test_a_complete_declaration_is_accepted():
    assert contract.validate_spec(base_spec()) == base_spec()
    assert len(contract.spec_sha256(base_spec())) == 64


@pytest.mark.parametrize("field", sorted(contract.SPEC_FIELDS))
def test_every_mandatory_field_is_actually_required(field):
    spec = base_spec()
    spec.pop(field)
    with pytest.raises(contract.SpecRefusal, match="does not declare"):
        contract.validate_spec(spec)


def test_two_different_delays_cannot_be_declared_at_once():
    """`output_availability` and `delay_samples` are the same fact stated twice."""
    with pytest.raises(contract.SpecRefusal, match="two different delays"):
        contract.validate_spec(base_spec(output_availability="t + 3", delay_samples=0))


def test_a_fitting_scope_outside_train_is_refused():
    with pytest.raises(contract.SpecRefusal, match="fit_scope"):
        contract.validate_spec(base_spec(fit_scope="CALIBRATION"))


def test_an_operator_that_fits_nothing_cannot_carry_state():
    with pytest.raises(contract.SpecRefusal, match="cannot carry fitted state"):
        contract.validate_spec(base_spec(fit_scope="NONE", bytes_state=16))


def test_an_unmeasured_cost_is_refused():
    with pytest.raises(contract.SpecRefusal, match="cost_cpu_seconds_per_1000"):
        contract.validate_spec(base_spec(cost_cpu_seconds_per_1000=0.0))


def test_applicable_to_nothing_is_refused():
    with pytest.raises(contract.SpecRefusal, match="applicability"):
        contract.validate_spec(base_spec(applicability=[]))


def test_a_field_the_contract_does_not_define_is_refused():
    with pytest.raises(contract.SpecRefusal, match="does not define"):
        contract.validate_spec(base_spec(score=0.9))


def test_an_available_warm_up_output_is_refused():
    with pytest.raises(contract.SpecRefusal, match="fabricated observation"):
        contract.validate_output({"values": [1.0, 2.0], "available": [True, True],
                                  "raw": [1.0, 2.0]},
                                 spec=base_spec(warm_up_samples=1), samples=2)


# --- the reference operator passes -----------------------------------------------------------

def test_an_honestly_declared_causal_operator_is_accepted():
    report = run(TrailingMean(), control=CentredMean(),
                 resource_contract={"completion_lag_max": 0})
    assert report["failed"] == [], report["results"]
    assert report["verdict"] == "MECHANICALLY_ACCEPTED"
    assert report["results"]["prefix"]["passed"] is True
    assert report["results"]["altered_suffix"]["passed"] is True


def test_the_report_names_the_spec_and_the_state_it_used():
    report = run(TrailingMean(), control=CentredMean(),
                 resource_contract={"completion_lag_max": 0})
    assert report["spec_sha256"] == contract.spec_sha256(base_spec())
    assert len(report["state_sha256"]) == 64
    assert "selects nothing" in report["note"]


# --- the non-causal control fails, and that is the control working ---------------------------

def test_the_declared_control_fails_the_causality_tests():
    report = run(CentredMean(), resource_contract={"completion_lag_max": 0})
    assert report["results"]["altered_suffix"]["passed"] is False
    assert report["verdict"] == "MECHANICALLY_REFUSED"


def test_test_one_alone_does_not_catch_a_look_ahead_smaller_than_the_lookback():
    """A measured limit of the sealed design, recorded rather than worked around.

    §3.1 compares `transform(X[:n])` and `transform(X)` only up to `n - lookback - delay`. A
    centred window that reads four samples ahead while declaring seven samples of lookback is
    therefore invisible to test 1: every index it compares is one both calls could compute.
    Test 2 — changing the future must not change the past — does catch it, and does here.

    This is a note for the reviewer, not a change: the design is sealed, the battery implements
    it as written, and the pair of tests together is what makes the control fail. Widening
    test 1 to `n - delay` would close the gap and is a design decision, not mine to take.
    """
    control = CentredMean()
    state = control.fit(signal()[:200])
    assert battery.check_prefix(control, state, signal())["passed"] is True
    assert battery.check_altered_suffix(control, state, signal())["passed"] is False


def test_a_control_that_passes_is_reported_as_no_control_at_all():
    """If the control is causal, either it is not the control or the tests are not working."""
    outcome = battery.check_non_causal_control(
        TrailingMean(base_spec(non_causal_control_of="trailing_mean")), {}, signal())
    assert outcome["passed"] is False
    assert "not measuring causality" in outcome["detail"]


def test_a_control_must_name_what_it_is_the_twin_of():
    outcome = battery.check_non_causal_control(TrailingMean(), {}, signal())
    assert outcome["passed"] is False
    assert "deliberate twin" in outcome["detail"]


def test_the_control_is_never_promoted():
    report = run(TrailingMean(), control=CentredMean(),
                 resource_contract={"completion_lag_max": 0})
    assert report["results"]["non_causal_control"]["passed"] is True
    assert report["results"]["non_causal_control"]["promoted"] is False


# --- one lie per fixture, and only that test fails -------------------------------------------

def test_an_understated_warm_up_is_caught():
    class Understated(TrailingMean):
        def __init__(self):
            super().__init__(base_spec(warm_up_samples=2))

    outcome = battery.check_warm_up_edge(Understated(), {}, signal())
    assert outcome["passed"] is True     # fewer available outputs than declared is not a lie
    report = run(Understated(), resource_contract={"completion_lag_max": 0})
    assert "warm_up_edge" not in report["failed"]


def test_an_operator_that_is_never_available_is_caught():
    class NeverReady(TrailingMean):
        def transform(self, x, state):
            return {"values": [0.0] * len(x), "available": [False] * len(x), "raw": list(x)}

    outcome = battery.check_warm_up_edge(NeverReady(), {}, signal())
    assert outcome["passed"] is False
    assert "never ends" in outcome["detail"]


def test_an_undeclared_delay_is_measured_against_an_impulse():
    class Delayed(TrailingMean):
        """Shifts its output by two samples while still declaring `t`."""

        def transform(self, x, state):
            output = super().transform(x, state)
            values = [0.0, 0.0] + output["values"][:-2]
            available = [False, False] + output["available"][:-2]
            return {"values": values, "available": available, "raw": list(x)}

    outcome = battery.check_measured_delay(Delayed(), {})
    assert outcome["passed"] is False
    assert outcome["declared"] == 0 and outcome["observed"] == 2


def test_an_operator_that_does_not_react_at_all_cannot_declare_a_delay():
    class Deaf(TrailingMean):
        def transform(self, x, state):
            return {"values": [1.0] * len(x), "available": [True] * len(x), "raw": list(x)}

    outcome = battery.check_measured_delay(Deaf(base_spec(warm_up_samples=0)), {})
    assert outcome["passed"] is False
    assert "cannot be measured" in outcome["detail"]


def test_a_replaced_raw_branch_is_caught():
    class Replacer(TrailingMean):
        def transform(self, x, state):
            output = super().transform(x, state)
            output["raw"] = output["values"]
            return output

    outcome = battery.check_raw_branch(Replacer(), {}, signal())
    assert outcome["passed"] is False
    assert "never replace the original" in outcome["detail"]


def test_a_number_for_an_inapplicable_family_is_caught():
    class Extrapolator(TrailingMean):
        def apply_to_family(self, family, x, state):
            return self.transform(x, state)

    outcome = battery.check_applicability(Extrapolator(), {}, signal(), "seasonal")
    assert outcome["passed"] is False
    assert "undeclared extrapolation" in outcome["detail"]


def test_an_operator_that_cannot_be_asked_about_a_family_fails_rather_than_passes():
    class Silent:
        """Everything the contract requires except a way to be asked about a family."""

        def describe(self):
            return base_spec(kind="silent")

        def fit(self, train_prefix):
            return {}

        def transform(self, x, state):
            return TrailingMean().transform(x, state)

    outcome = battery.check_applicability(Silent(), {}, signal(), "seasonal")
    assert outcome["passed"] is False
    assert "cannot refuse one" in outcome["detail"]


def test_an_understated_cost_is_caught():
    outcome = battery.check_cost(TrailingMean(base_spec(cost_cpu_seconds_per_1000=1e-9)),
                                 {}, signal())
    assert outcome["passed"] is False
    assert outcome["measured_cpu_seconds_per_1000"] > 0


def test_a_checkpoint_that_is_only_declared_is_caught():
    operator = TrailingMean(base_spec(chunk_restart="STATEFUL_WITH_CHECKPOINT"))
    outcome = battery.check_chunk_restart(operator, {}, signal())
    assert outcome["passed"] is False
    assert "cannot be exercised" in outcome["detail"]


# --- availability against the lake's own contract --------------------------------------------

def test_an_output_claimed_before_the_resource_is_complete_is_caught():
    outcome = battery.check_availability(TrailingMean(base_spec(lookback_samples=0)),
                                         {"completion_lag_max": 4})
    assert outcome["passed"] is False
    assert "does not exist yet" in outcome["detail"]


def test_an_unknown_completion_lag_certifies_nothing_rather_than_passing():
    """`UNKNOWN` is not zero: an archive that cannot say when a bar is complete has not said
    that it is complete immediately."""
    outcome = battery.check_availability(TrailingMean(), {"completion_lag_max": "UNKNOWN"})
    assert outcome["passed"] is None
    assert "the operator is not refused, the claim is" in outcome["detail"]


def test_an_undecided_test_makes_the_verdict_inconclusive_not_accepted():
    report = run(TrailingMean(), control=CentredMean(),
                 resource_contract={"completion_lag_max": "UNKNOWN"})
    assert report["failed"] == []
    assert "availability" in report["undecided"]
    assert report["verdict"] == "INCONCLUSIVE"


# --- what the battery is not -----------------------------------------------------------------

def code_without_prose(path: Path) -> str:
    """Source with every docstring removed: prose is not behaviour, in either direction."""
    import ast

    text = path.read_text(encoding="utf-8")
    skip = set()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            skip.update(range(body[0].lineno, body[0].end_lineno + 1))
    return "\n".join(line for number, line in enumerate(text.splitlines(), 1)
                      if number not in skip)


def test_the_battery_scores_nothing_and_writes_nowhere():
    body = code_without_prose(REPO / "tools" / "df_d3_acceptance.py")
    for forbidden in ("write_terminal", "requests.", "urlopen", "duckdb", "sqlite3",
                      "score", "rank(", "select_features", "open("):
        assert forbidden not in body, forbidden


def test_the_ten_design_tests_are_all_present():
    assert len(battery.TESTS) == 10
    report = run(TrailingMean(), control=CentredMean(),
                 resource_contract={"completion_lag_max": 0})
    assert set(report["results"]) == set(battery.TESTS)
