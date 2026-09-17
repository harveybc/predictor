"""The amended D3 contract and battery (J2), proved against the actual harness callables.

Every rule here is exercised through `df_d3_acceptance.run_battery` or one of its `check_*`
functions and through `df_d3_contract`, never through a substitute implementation. The four
defects the reviewer reproduced are frozen as NEGATIVE tests first; then the positive controls
the order requires: a valid causal zero-lag operator and a valid delayed-output operator.

Then the cases the order names explicitly: fractional durations, late arrivals, missing
timestamps, a real '0s' contract, window padding and centring, full-series reconstruction and
normalisation, restart state, the same prefix with a changed future, and changed future fit
data. Wavelet support is derived from the library; a recursive filter carries state.

The operators here are harness fixtures. Nothing in this file promotes anything.
"""

from __future__ import annotations

import importlib.util
import json
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
design = _load("df_d3_design")

NA = contract.NOT_AVAILABLE


def signal(length: int = 300) -> list:
    return [((index * 37) % 101) / 10.0 - 5.0 for index in range(length)]


def x_of(length: int = 300, **kw) -> dict:
    return battery.make_input(signal(length), **kw)


def base_spec(**overrides) -> dict:
    spec = {"schema": contract.SPEC_SCHEMA, "kind": "trailing_mean",
            "params": {"window": 8}, "bytes_state": 0, "fit_scope": "NONE",
            "lookback_samples": 7, "warm_up_samples": 7, "delay_samples": 0,
            "output_availability": "t",
            "response_probe": {"kind": "impulse", "expected_onset_samples": 0},
            "non_causal_twin": {"kind": "centred_mean"},
            "support": {"kind": "FINITE", "samples": 8, "derivation": "window of 8 samples",
                        "boundary_mode": None},
            "cost_cpu_seconds_per_1000": 5.0,
            "applicability": ["step", "trend"], "chunk_restart": "IDEMPOTENT"}
    spec.update(overrides)
    return spec


class TrailingMean:
    """A causal, zero-lag operator that declares itself honestly: the positive control."""

    def __init__(self, spec=None):
        self._spec = spec or base_spec()
        self.window = self._spec["params"]["window"]

    def describe(self):
        return dict(self._spec)

    def fit(self, train):
        return {}

    def _emit(self, x):
        return contract.emission_times(x, lookback=self._spec["lookback_samples"],
                                       delay=self._spec["delay_samples"])

    def transform(self, x, state):
        v = x["values"]
        values, available = [], []
        for i in range(len(v)):
            if i + 1 < self.window:
                values.append(0.0)
                available.append(False)
                continue
            chunk = v[i + 1 - self.window:i + 1]
            values.append(sum(chunk) / self.window)
            available.append(True)
        return {"values": values, "available": available, "emitted_at": self._emit(x),
                "raw": list(v)}

    def apply_to_family(self, family, x, state):
        if family not in self._spec["applicability"]:
            return contract.NOT_APPLICABLE
        return self.transform(x, state)


class CentredMean(TrailingMean):
    """The design's deliberate non-causal twin: the same window, centred."""

    def __init__(self):
        super().__init__(base_spec(kind="centred_mean", warm_up_samples=4,
                                   lookback_samples=4,
                                   support={"kind": "FINITE", "samples": 5,
                                            "derivation": "centred window", "boundary_mode": None},
                                   non_causal_twin={"not_applicable": True,
                                                    "reason": "it is itself a twin"},
                                   non_causal_control_of="trailing_mean"))

    def transform(self, x, state):
        v = x["values"]
        window, half = self.window, self.window // 2
        values, available = [], []
        for i in range(len(v)):
            start, stop = i - half, i - half + window
            if start < 0 or stop > len(v):
                values.append(0.0)
                available.append(False)
                continue
            values.append(sum(v[start:stop]) / window)
            available.append(True)
        return {"values": values, "available": available, "emitted_at": self._emit(x),
                "raw": list(v)}


class DelayedMean(TrailingMean):
    """A valid operator whose output for t is emitted two periods later: the delayed control."""

    def __init__(self):
        super().__init__(base_spec(kind="delayed_mean", delay_samples=2,
                                   output_availability="t + 2",
                                   response_probe={"kind": "impulse",
                                                   "expected_onset_samples": 0},
                                   non_causal_twin={"not_applicable": True,
                                                    "reason": "control fixture"}))


CONTRACT_0S = {"frequency": "1s", "availability": {"label": "WINDOW_START",
                                                  "completion_lag_max": "0s",
                                                  "timezone_evidence": "PRODUCER_STATEMENT",
                                                  "use_class": "LIVE_EQUIVALENT"}}


def run(operator, **kw):
    return battery.run_battery(operator, x_of(), **kw)


# --- the amendment itself -------------------------------------------------------------------

def test_the_amendment_is_sealed_and_cites_the_original_bytes():
    assert design.validate_amendment(design.D3_AMENDMENT_V1) == []
    assert design.D3_AMENDMENT_V1["supersedes"]["sha256"] == design.ORIGINAL_SHA256
    assert len(design.D3_AMENDMENT_V1["operators"]) == 9


def test_the_amendment_digest_moves_when_a_rule_moves():
    doc = json.loads(json.dumps(design.D3_AMENDMENT_V1))
    doc["prefix_rule"] = "something weaker"
    assert "design digest does not re-derive" in design.validate_amendment(doc)


def test_readiness_has_three_separate_words():
    assert design.READINESS_STATES == ("INFRASTRUCTURE_PRESENT", "TEMPORAL_BATTERY_ACCEPTED",
                                       "SCIENTIFIC_UTILITY")


# --- the reviewer's four findings, frozen as negatives --------------------------------------

def test_finding_1_past_lookback_never_makes_a_late_input_available():
    """delay 0, lookback 7, resource lag 5 periods: v1 passed this. It must not."""

    class EmitsAtInputTime(TrailingMean):
        def transform(self, x, state):
            out = super().transform(x, state)
            out["emitted_at"] = list(x["timestamps"])          # ignores availability
            return out

    op = EmitsAtInputTime()
    x = x_of()
    late = {"frequency": "1s", "availability": {"label": "WINDOW_START",
                                               "completion_lag_max": "5s",
                                               "timezone_evidence": "PRODUCER_STATEMENT",
                                               "use_class": "OFFLINE_DAY_GRANULAR"}}
    # the harness refuses the output outright: an output emitted before its input is a leak
    with pytest.raises(contract.SpecRefusal, match="before its own input was available"):
        battery.check_availability_emission(op, battery.prefix(x, 180), x, late)


def test_finding_1_emission_must_follow_the_latest_consumed_availability():
    """An operator that emits at its own row's availability but consumes a LATER row."""

    class ConsumesAheadInWindow(TrailingMean):
        def transform(self, x, state):
            out = super().transform(x, state)
            # honest per-row availability, but the window looks back over inputs whose
            # availability may be later than this row's (late arrivals): must be >= max
            out["emitted_at"] = list(x["available_at"])
            return out

    op = ConsumesAheadInWindow()
    n = 300
    ts = list(range(n))
    avail = list(ts)
    avail[100] = 120                       # a late arrival inside later windows
    x = battery.make_input(signal(n), timestamps=ts, available_at=avail, period_seconds=1)
    outcome = battery.check_availability_emission(op, battery.prefix(x, 180), x, CONTRACT_0S)
    assert outcome["passed"] is False
    assert any(v["index"] in range(101, 108) for v in outcome["violations"])


def test_finding_2_a_real_zero_second_contract_is_valid():
    outcome = battery.check_availability_emission(TrailingMean(), battery.prefix(x_of(), 180),
                                                  x_of(), CONTRACT_0S)
    assert outcome["passed"] is True
    assert outcome["lag_seconds"] == 0.0 and outcome["lag_samples"] == 0


def test_finding_2_durations_are_parsed_as_the_producer_parses_them():
    assert battery.parse_duration_seconds("4h") == 4 * 3600.0
    assert battery.parse_duration_seconds("0s") == 0.0
    assert battery.parse_duration_seconds("UNKNOWN") is None
    assert battery.parse_duration_seconds(None) is None
    with pytest.raises(ValueError):
        battery.parse_duration_seconds("-1h")


def test_finding_2_a_fractional_sample_offset_is_refused_not_truncated():
    with pytest.raises(ValueError, match="FRACTIONAL_SAMPLE_OFFSET"):
        battery.duration_to_samples(90.0, 60)
    assert battery.duration_to_samples(120.0, 60) == 2
    x = x_of(period_seconds=60)
    late = {"frequency": "60s", "availability": {"label": "WINDOW_START",
                                                "completion_lag_max": "90s",
                                                "timezone_evidence": "PRODUCER_STATEMENT",
                                                "use_class": "OFFLINE_DAY_GRANULAR"}}
    outcome = battery.check_availability_emission(TrailingMean(), battery.prefix(x, 180), x,
                                                  late)
    assert outcome["passed"] is False
    assert "FRACTIONAL_SAMPLE_OFFSET" in outcome["detail"]


def test_finding_2_unknown_stays_unknown():
    unknown = {"availability": {"label": "UNKNOWN", "completion_lag_max": "UNKNOWN",
                                "timezone_evidence": "UNKNOWN",
                                "use_class": "OFFLINE_DAY_GRANULAR"}}
    outcome = battery.check_availability_emission(TrailingMean(), battery.prefix(x_of(), 180),
                                                  x_of(), unknown)
    assert outcome["passed"] is None and outcome["outcome"] == "UNKNOWN"


def test_finding_3_the_centred_twin_fails_prefix_invariance_itself():
    """v1 exempted the last `lookback` samples and let the centred twin through test 1."""
    x = x_of()
    outcome = battery.check_prefix_all_available(CentredMean(), battery.prefix(x, 180), x)
    assert outcome["passed"] is False
    assert outcome["compared"] > 0


def test_finding_3_there_is_no_lookback_exemption_in_the_prefix_rule():
    """A one-sample look-ahead hidden inside a declared lookback of 7 is still caught."""

    class PeeksOne(TrailingMean):
        def transform(self, x, state):
            v = x["values"]
            out = super().transform(x, state)
            out["values"] = [out["values"][i] + (v[i + 1] if i + 1 < len(v) else 0.0) * 1e-3
                             for i in range(len(v))]
            return out

    x = x_of()
    outcome = battery.check_prefix_all_available(PeeksOne(), battery.prefix(x, 180), x)
    assert outcome["passed"] is False


def test_finding_4_an_operator_with_no_twin_and_no_reason_is_refused():
    spec = base_spec(non_causal_twin={})
    with pytest.raises(contract.SpecRefusal, match="names the twin's kind"):
        contract.validate_spec(spec)


def test_finding_4_a_declared_twin_that_is_not_supplied_is_a_failure_not_undecided():
    report = run(TrailingMean(), resource_contract=CONTRACT_0S)
    assert report["results"]["non_causal_twin"]["passed"] is False
    assert report["verdict"] == "MECHANICALLY_REFUSED"


def test_finding_4_a_scoped_twin_needs_a_design_reason_and_is_reported_as_scoped():
    with pytest.raises(contract.SpecRefusal, match="design reason"):
        contract.validate_spec(base_spec(non_causal_twin={"not_applicable": True, "reason": ""}))
    op = TrailingMean(base_spec(non_causal_twin={"not_applicable": True,
                                                 "reason": "pointwise codec"}))
    report = run(op, resource_contract=CONTRACT_0S)
    assert report["scoped"] == ["non_causal_twin"]
    assert report["results"]["non_causal_twin"]["outcome"] == "NOT_APPLICABLE"
    assert report["verdict"] == "MECHANICALLY_ACCEPTED"


# --- positive controls ----------------------------------------------------------------------

def test_a_causal_zero_lag_operator_with_its_twin_is_review_ready():
    report = run(TrailingMean(), twin=CentredMean(), resource_contract=CONTRACT_0S)
    assert report["failed"] == [], report["results"]
    assert report["undecided"] == []
    assert report["review_ready"] is True
    assert report["results"]["non_causal_twin"]["twin_prefix_passed"] is False


def test_a_delayed_output_operator_is_valid_and_compared_only_after_emission():
    x = x_of()
    outcome = battery.check_prefix_all_available(DelayedMean(), battery.prefix(x, 180), x)
    assert outcome["passed"] is True
    avail = battery.check_availability_emission(DelayedMean(), battery.prefix(x, 180), x,
                                                CONTRACT_0S)
    assert avail["passed"] is True
    report = run(DelayedMean(), resource_contract=CONTRACT_0S)
    assert report["review_ready"] is True


def test_a_positive_delay_without_a_sampling_contract_cannot_be_turned_into_a_time():
    x = battery.make_input(signal(50), timestamps=list(range(0, 5000, 100)),
                           period_seconds=None)
    with pytest.raises(contract.SpecRefusal, match="declared sampling period"):
        contract.emission_times(x, lookback=7, delay=2)


# --- durations, late arrivals, missing timestamps --------------------------------------------

def test_late_arrivals_push_emission_forward_for_every_window_that_consumes_them():
    n = 40
    avail = list(range(n))
    avail[10] = 30
    x = battery.make_input(signal(n), timestamps=list(range(n)), available_at=avail,
                           period_seconds=1)
    emitted = contract.emission_times(x, lookback=7, delay=0)
    assert emitted[10] == 30 and emitted[17] == 30 and emitted[18] == 18


def test_a_row_available_before_its_own_timestamp_is_refused_at_the_input():
    with pytest.raises(contract.SpecRefusal, match="available before its own timestamp"):
        battery.make_input([1.0, 2.0], timestamps=[10, 20], available_at=[10, 15],
                           period_seconds=10)


def test_missing_timestamps_are_refused_rather_than_invented():
    with pytest.raises(contract.SpecRefusal, match="same length"):
        contract.validate_input({"values": [1.0, 2.0, 3.0], "timestamps": [0, 1],
                                 "available_at": [0, 1], "period_seconds": 1})


def test_holes_are_not_equally_spaced_observations():
    """With an irregular grid there is no period; a duration is never converted."""
    with pytest.raises(ValueError, match="no sampling contract"):
        battery.duration_to_samples(60.0, None)


# --- windows, reconstruction, normalisation -------------------------------------------------

def test_window_padding_at_the_start_is_unavailable_not_zero():
    x = x_of()
    outcome = battery.check_warm_up_edge(TrailingMean(), battery.prefix(x, 180), x)
    assert outcome["passed"] is True
    padded = TrailingMean(base_spec(warm_up_samples=0))
    with pytest.raises(contract.SpecRefusal, match="fabricated observation"):
        # the operator marks warm-up unavailable but declares zero warm-up: the declaration
        # is what is refused, since the mask and the declaration must agree
        battery.check_warm_up_edge(
            type("Lying", (TrailingMean,), {"transform": lambda self, x, s: dict(
                TrailingMean.transform(self, x, s), available=[True] * len(x["values"]))})(
                base_spec(warm_up_samples=7)), battery.prefix(x, 180), x)
    assert padded.describe()["warm_up_samples"] == 0


def test_a_full_series_reconstruction_is_not_a_causal_output():
    """Normalising by a whole-series statistic reads the future by construction."""

    class WholeSeriesZScore(TrailingMean):
        def transform(self, x, state):
            v = x["values"]
            mean = sum(v) / len(v)
            sd = (sum((a - mean) ** 2 for a in v) / len(v)) ** 0.5 or 1.0
            return {"values": [(a - mean) / sd for a in v], "available": [True] * len(v),
                    "emitted_at": self._emit(x), "raw": list(v)}

    op = WholeSeriesZScore(base_spec(warm_up_samples=0, lookback_samples=0,
                                     support={"kind": "POINTWISE", "samples": 1,
                                              "derivation": "claims pointwise",
                                              "boundary_mode": None}))
    x = x_of()
    assert battery.check_future_perturbation(op, battery.prefix(x, 180), x)["passed"] is False


def test_train_only_normalisation_is_causal():
    class TrainZScore(TrailingMean):
        def fit(self, train):
            v = train["values"]
            mean = sum(v) / len(v)
            sd = (sum((a - mean) ** 2 for a in v) / len(v)) ** 0.5 or 1.0
            return {"mean": mean, "sd": sd}

        def transform(self, x, state):
            v = x["values"]
            return {"values": [(a - state["mean"]) / state["sd"] for a in v],
                    "available": [True] * len(v), "emitted_at": self._emit(x), "raw": list(v)}

    op = TrainZScore(base_spec(kind="train_zscore", fit_scope="TRAIN_PREFIX_ONLY",
                               bytes_state=16, warm_up_samples=0, lookback_samples=0,
                               support={"kind": "POINTWISE", "samples": 1,
                                        "derivation": "pointwise", "boundary_mode": None},
                               non_causal_twin={"not_applicable": True,
                                                "reason": "pointwise codec"}))
    report = run(op, resource_contract=CONTRACT_0S)
    assert report["results"]["fit_scope_train_only"]["passed"] is True
    assert report["results"]["future_perturbation"]["passed"] is True


# --- state: fresh per branch, restart, changed future fit data --------------------------------

def test_a_transform_that_mutates_its_state_is_caught():
    class Mutator(TrailingMean):
        def fit(self, train):
            return {"calls": 0}

        def transform(self, x, state):
            state["calls"] += 1
            out = super().transform(x, state)
            out["values"] = [v + state["calls"] for v in out["values"]]
            return out

    x = x_of()
    outcome = battery.check_fresh_state_per_branch(
        Mutator(base_spec(fit_scope="TRAIN_PREFIX_ONLY", bytes_state=8)),
        battery.prefix(x, 180), x)
    assert outcome["passed"] is False


def test_changing_the_evaluated_future_does_not_change_a_train_only_fit():
    class TrainMean(TrailingMean):
        def fit(self, train):
            return {"m": sum(train["values"]) / len(train["values"])}

        def transform(self, x, state):
            out = super().transform(x, state)
            out["values"] = [v - state["m"] for v in out["values"]]
            return out

    op = TrainMean(base_spec(fit_scope="TRAIN_PREFIX_ONLY", bytes_state=8))
    x = x_of()
    assert battery.check_fit_scope_train_only(op, battery.prefix(x, 180), x)["passed"] is True


def test_a_fit_that_reads_the_evaluated_future_is_caught():
    """The harness hands only the train prefix; a fixture that smuggles the full series in
    through module state stands in for a fit that reached past the boundary."""
    leak = {}

    class LeakyFit(TrailingMean):
        def fit(self, train):
            return {"m": sum(leak.get("full", train["values"])) / len(leak.get("full",
                                                                             train["values"]))}

        def transform(self, x, state):
            leak["full"] = x["values"]
            out = super().transform(x, state)
            out["values"] = [v - state["m"] for v in out["values"]]
            return out

    op = LeakyFit(base_spec(fit_scope="TRAIN_PREFIX_ONLY", bytes_state=8))
    x = x_of()
    outcome = battery.check_fit_scope_train_only(op, battery.prefix(x, 180), x)
    assert outcome["passed"] is False


def test_an_idempotent_restart_reproduces_one_pass():
    x = x_of()
    assert battery.check_chunk_restart(TrailingMean(), battery.prefix(x, 180), x)["passed"] is True


def test_a_checkpoint_that_is_only_declared_is_caught():
    op = TrailingMean(base_spec(chunk_restart="STATEFUL_WITH_CHECKPOINT"))
    x = x_of()
    outcome = battery.check_chunk_restart(op, battery.prefix(x, 180), x)
    assert outcome["passed"] is False and "no checkpoint()" in outcome["detail"]


def test_a_stateful_checkpoint_resume_is_exercised():
    class Cumulative(TrailingMean):
        def __init__(self):
            super().__init__(base_spec(kind="cumulative_sum", chunk_restart="STATEFUL_WITH_CHECKPOINT",
                                       lookback_samples=0, warm_up_samples=0,
                                       support={"kind": "RECURSIVE", "samples": None,
                                                "derivation": "running sum state",
                                                "boundary_mode": None},
                                       non_causal_twin={"not_applicable": True,
                                                        "reason": "control fixture"}))
            self.total = 0.0

        def fit(self, train):
            return {"total": 0.0}

        def transform(self, x, state):
            total = state["total"]
            values = []
            for v in x["values"]:
                total += v
                values.append(total)
            self.total = total
            return {"values": values, "available": [True] * len(values),
                    "emitted_at": self._emit(x), "raw": list(x["values"])}

        def checkpoint(self):
            return json.dumps({"total": self.total})

        def resume(self, blob):
            return {"total": json.loads(blob)["total"]}

    x = x_of()
    assert battery.check_chunk_restart(Cumulative(), battery.prefix(x, 180), x)["passed"] is True


# --- probes and delay -------------------------------------------------------------------------

def test_the_declared_probe_measures_onset_not_group_delay():
    x = x_of()
    outcome = battery.check_response_probe(TrailingMean(), battery.prefix(x, 180), x)
    assert outcome["passed"] is True and outcome["observed"] == 0


def test_an_undeclared_onset_is_measured_against_the_probe():
    class Shifted(TrailingMean):
        def transform(self, x, state):
            out = super().transform(x, state)
            out["values"] = [0.0, 0.0] + out["values"][:-2]
            out["available"] = [False, False] + out["available"][:-2]
            return out

    x = x_of()
    outcome = battery.check_response_probe(Shifted(), battery.prefix(x, 180), x)
    assert outcome["passed"] is False and outcome["observed"] == 2


def test_an_unidentified_response_is_recorded_never_fabricated():
    op = TrailingMean(base_spec(response_probe={"kind": "impulse",
                                                "expected_onset_samples": "UNIDENTIFIED"}))
    x = x_of()
    outcome = battery.check_response_probe(op, battery.prefix(x, 180), x)
    assert outcome["passed"] is None and outcome["outcome"] == "UNIDENTIFIED"
    with pytest.raises(contract.SpecRefusal, match="never a fabricated zero"):
        contract.validate_spec(base_spec(response_probe={"kind": "impulse",
                                                         "expected_onset_samples": -1}))


# --- filter support ---------------------------------------------------------------------------

def test_wavelet_support_is_derived_from_the_library_not_from_two_to_the_L():
    pywt = pytest.importorskip("pywt")
    db4 = pywt.Wavelet("db4").dec_len
    haar = pywt.Wavelet("haar").dec_len
    assert (haar, db4) == (2, 8)
    support = lambda dec_len, L: (dec_len - 1) * (2 ** L - 1) + 1  # noqa: E731
    assert support(haar, 3) == 8 and support(db4, 3) == 50


def test_a_recursive_support_refuses_a_finite_sample_count():
    with pytest.raises(contract.SpecRefusal, match="dependency"):
        contract.validate_spec(base_spec(support={"kind": "RECURSIVE", "samples": 3,
                                                  "derivation": "butterworth order 2",
                                                  "boundary_mode": None}))


def test_a_support_larger_than_the_declared_lookback_is_refused():
    with pytest.raises(contract.SpecRefusal, match="reads more past"):
        contract.validate_spec(base_spec(support={"kind": "FINITE", "samples": 50,
                                                  "derivation": "db4 level 3",
                                                  "boundary_mode": "zero"}))


# --- the verdict --------------------------------------------------------------------------------

def test_review_ready_means_every_required_test_ran_and_passed():
    report = run(TrailingMean(), twin=CentredMean(), resource_contract=CONTRACT_0S)
    assert set(report["results"]) == set(design.REQUIRED_TESTS)
    assert report["review_ready"] is True


def test_an_undecided_required_test_is_never_review_ready():
    unknown = {"availability": {"label": "UNKNOWN", "completion_lag_max": "UNKNOWN",
                                "timezone_evidence": "UNKNOWN",
                                "use_class": "OFFLINE_DAY_GRANULAR"}}
    report = run(TrailingMean(), twin=CentredMean(), resource_contract=unknown)
    assert report["review_ready"] is False and report["verdict"] == "INCONCLUSIVE"
    assert "availability_emission" in report["undecided"]


def test_the_report_carries_the_amendment_digest():
    report = run(TrailingMean(), twin=CentredMean(), resource_contract=CONTRACT_0S)
    assert report["design_sha256"] == design.D3_AMENDMENT_V1["design_sha256"]


def test_the_battery_scores_nothing_and_writes_nowhere():
    import ast

    text = (REPO / "tools" / "df_d3_acceptance.py").read_text(encoding="utf-8")
    skip = set()
    for node in ast.walk(ast.parse(text)):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            skip.update(range(body[0].lineno, body[0].end_lineno + 1))
    code = "\n".join(l for i, l in enumerate(text.splitlines(), 1) if i not in skip)
    for forbidden in ("write_terminal", "urlopen", "duckdb", "sqlite3", "score", "rank(",
                      "select_features", "open("):
        assert forbidden not in code, forbidden
