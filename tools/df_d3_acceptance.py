#!/usr/bin/env python3
"""The ten D3 acceptance tests, executable against any operator that carries the contract.

Block 3 of `docs/handoffs/MUSASHI_I1_I3_ACCEPTANCE_AND_STACK_FOLLOWUP_2026_09_16.md`. The tests
are §3 of the sealed design
`docs/integracion_workplan_2026_09_10/07_DISENO_D3_CUANTIZACION_TIEMPO_FRECUENCIA_DETECTORES_2026_09_14.md`,
transcribed and made runnable. They are **mechanical**: causality, edges, restart, delay,
availability, cost, applicability and the raw branch. Not one of them scores an operator, and
passing them is not evidence that an operator is useful — only that what it declares about
itself is true.

    1  prefix                 transform(X[:n]) agrees with transform(X) on the settled prefix
    2  altered suffix         changing the future does not change the past
    3  edge / warm-up         the first warm_up_samples outputs are NOT_AVAILABLE, not 0
    4  chunk / restart        two chunks with a checkpoint reproduce one pass
    5  measured delay         a unit impulse shows the delay the operator declared
    6  non-causal control     the declared control FAILS 1-2; recorded, never promoted
    7  availability           no output before `label + completion_lag_max` of the resource
    8  cost                   a single-thread pilot stays within the declared budget
    9  applicability          an inapplicable family yields NOT_APPLICABLE, not a number
   10  raw branch             the output preserves `*_raw`

An operator is a small object with three methods, as the design's §1 requires:

    describe() -> spec          the declaration, validated by `df_d3_contract`
    fit(train_prefix) -> state  fitted on a TRAIN PREFIX only, or on nothing
    transform(x, state) -> {"values": [...], "available": [...], "raw": [...]}

`transform` is called with a plain sequence and must be a pure function of it and the state.

Nothing here writes to a store, opens a database or starts a governed run. It is a battery,
not a campaign.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("df_d3_contract", HERE / "df_d3_contract.py")
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)

#: The design's memory ceiling for one operator process.
MEMORY_CEILING_BYTES = 2 * 1024 ** 3

TESTS = ("prefix", "altered_suffix", "warm_up_edge", "chunk_restart", "measured_delay",
         "non_causal_control", "availability", "cost", "applicability", "raw_branch")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _settled(spec: dict, n: int) -> int:
    """How far into an output of `n` samples the values can no longer change.

    Beyond `n - lookback - delay` an output may still be waiting for samples the shorter call
    has not seen, so comparing there would measure the truncation rather than the operator.
    """
    return max(0, n - spec["lookback_samples"] - spec["delay_samples"])


def _values(output) -> list:
    return list(output["values"])


def _available(output) -> list:
    return [bool(flag) for flag in output["available"]]


def _masked(output) -> list:
    """Values with the unavailable ones replaced by the sentinel, which is what is compared."""
    return [value if flag else contract.NOT_AVAILABLE
            for value, flag in zip(_values(output), _available(output))]


def _run(operator, x, state):
    output = operator.transform(list(x), state)
    return contract.validate_output(output, spec=operator.describe(), samples=len(x))


def check_prefix(operator, state, signal) -> dict:
    """Test 1. Bit for bit on the settled prefix, not approximately."""
    spec = operator.describe()
    whole = _masked(_run(operator, signal, state))
    n = len(signal) // 2
    part = _masked(_run(operator, signal[:n], state))
    edge = _settled(spec, n)
    ok = part[:edge] == whole[:edge]
    return {"passed": ok, "compared_samples": edge,
            "detail": None if ok else
            f"the first {edge} outputs differ between a call on {n} samples and one on "
            f"{len(signal)}: the operator is reading past its own lookback"}


def check_altered_suffix(operator, state, signal) -> dict:
    """Test 2. Changing the future must not change the past."""
    spec = operator.describe()
    t = len(signal) // 2
    altered = list(signal[:t + 1]) + [value * -7.0 - 13.0 for value in signal[t + 1:]]
    before = _masked(_run(operator, signal, state))
    after = _masked(_run(operator, altered, state))
    edge = max(0, t + 1 - spec["delay_samples"])
    ok = before[:edge] == after[:edge]
    changed = [index for index in range(edge) if before[index] != after[index]]
    return {"passed": ok, "compared_samples": edge,
            "detail": None if ok else
            f"{len(changed)} outputs in [0, {edge}) moved when only X[{t + 1}:] changed; "
            f"first at index {changed[0]}"}


def check_warm_up_edge(operator, state, signal) -> dict:
    """Test 3. The warm-up is unavailable, not zero."""
    spec = operator.describe()
    output = _run(operator, signal, state)
    warm = spec["warm_up_samples"]
    available = _available(output)
    early_available = [index for index in range(min(warm, len(available)))
                       if available[index]]
    later = available[warm:]
    ok = not early_available and (not later or any(later))
    return {"passed": ok, "warm_up_samples": warm,
            "detail": None if ok else
            (f"outputs {early_available[:5]} are marked available inside the declared warm-up"
             if early_available else
             "no output is ever available: a warm-up that never ends is not a warm-up")}


def check_chunk_restart(operator, state, signal) -> dict:
    """Test 4. Two chunks with a checkpoint reproduce one pass."""
    spec = operator.describe()
    whole = _masked(_run(operator, signal, state))
    cut = len(signal) // 2
    if spec["chunk_restart"] == "IDEMPOTENT":
        # No carried state, so a restart re-reads its own lookback and nothing else. The claim
        # under test is about the SECOND chunk: fed the samples it declares it needs, it must
        # produce exactly what one pass produced for those samples.
        back = min(cut, spec["lookback_samples"] + spec["delay_samples"])
        tail = _masked(_run(operator, signal[cut - back:], state))
        rebuilt = tail[back:]
        expected = whole[cut:]
        ok = rebuilt == expected
        differing = [index for index in range(min(len(rebuilt), len(expected)))
                     if rebuilt[index] != expected[index]]
        return {"passed": ok, "mode": "IDEMPOTENT", "compared_samples": len(expected),
                "restart_lookback": back,
                "detail": None if ok else
                (f"restarting at sample {cut} with {back} samples of lookback reproduced "
                 f"{len(differing)} outputs differently; first at {cut + differing[0]}"
                 if differing else
                 "the restarted chunk produced a different number of outputs")}
    if not hasattr(operator, "checkpoint") or not hasattr(operator, "resume"):
        return {"passed": False, "mode": spec["chunk_restart"],
                "detail": "the operator declares STATEFUL_WITH_CHECKPOINT and offers no "
                          "checkpoint()/resume(): the declaration cannot be exercised"}
    head_output = _run(operator, signal[:cut], state)
    saved = operator.checkpoint()
    resumed = operator.resume(saved)
    tail_output = contract.validate_output(
        operator.transform(list(signal[cut:]), resumed), spec=spec,
        samples=len(signal) - cut)
    rebuilt = _masked(head_output) + _masked(tail_output)
    ok = rebuilt == whole
    return {"passed": ok, "mode": spec["chunk_restart"], "compared_samples": len(whole),
            "detail": None if ok else
            "resuming from the checkpoint did not reproduce a single pass"}


def check_measured_delay(operator, state, length: int = 256) -> dict:
    """Test 5. A unit impulse shows the declared delay, or the declaration is wrong."""
    spec = operator.describe()
    quiet = [0.0] * length
    position = length // 2
    impulse = list(quiet)
    impulse[position] = 1.0
    base = _run(operator, quiet, state)
    hit = _run(operator, impulse, state)
    moved = [index for index in range(length)
             if _available(hit)[index] and _available(base)[index]
             and _values(hit)[index] != _values(base)[index]]
    if not moved:
        return {"passed": False, "declared": spec["delay_samples"],
                "detail": "an impulse moved no available output at all: the delay cannot be "
                          "measured, so it cannot be declared"}
    observed = moved[0] - position
    ok = observed == spec["delay_samples"]
    return {"passed": ok, "declared": spec["delay_samples"], "observed": observed,
            "detail": None if ok else
            f"the impulse first moved the output {observed} samples after it arrived, and the "
            f"operator declares {spec['delay_samples']}"}


def check_non_causal_control(control, control_state, signal) -> dict:
    """Test 6. The declared control MUST fail 1-2. A control that passes is not a control."""
    if control is None:
        return {"passed": None, "detail": "no non-causal control was declared for this operator"}
    spec = control.describe()
    if not spec.get("non_causal_control_of"):
        return {"passed": False,
                "detail": "a control must name the operator it is the deliberate twin of"}
    prefix = check_prefix(control, control_state, signal)
    suffix = check_altered_suffix(control, control_state, signal)
    failed = not prefix["passed"] or not suffix["passed"]
    return {"passed": failed, "promoted": False,
            "control_prefix_passed": prefix["passed"],
            "control_altered_suffix_passed": suffix["passed"],
            "detail": None if failed else
            "the declared non-causal control passed the causality tests, so either it is not "
            "the control it claims to be or the tests are not measuring causality"}


def check_availability(operator, resource_contract) -> dict:
    """Test 7. No output before `label + completion_lag_max` of the resource it reads.

    The lake's availability contract is the authority, and `UNKNOWN` is not zero: an archive
    that cannot say when a bar is complete does not thereby say it is complete immediately.
    """
    spec = operator.describe()
    if not isinstance(resource_contract, dict):
        return {"passed": None, "detail": "no resource availability contract was supplied"}
    lag = resource_contract.get("completion_lag_max")
    if lag in (None, "", "UNKNOWN"):
        return {"passed": None, "lag": lag,
                "detail": "the resource declares an UNKNOWN completion lag, so no output "
                          "timing can be certified against it; the operator is not refused, "
                          "the claim is"}
    try:
        required = int(lag)
    except (TypeError, ValueError):
        return {"passed": False, "lag": lag,
                "detail": f"completion_lag_max {lag!r} is not a sample count"}
    total = spec["delay_samples"] + spec["warm_up_samples"]
    ok = spec["delay_samples"] >= 0 and total >= required if required else True
    ok = spec["delay_samples"] + spec["lookback_samples"] >= required
    return {"passed": ok, "lag": required,
            "operator_delay_plus_lookback": spec["delay_samples"] + spec["lookback_samples"],
            "detail": None if ok else
            f"the resource is complete only at t + {required} and the operator claims an "
            f"output at t + {spec['delay_samples']} with {spec['lookback_samples']} samples "
            "of lookback: it would read a bar that does not exist yet"}


def check_cost(operator, state, signal) -> dict:
    """Test 8. A single-thread pilot within the declared budget, and under the memory ceiling."""
    spec = operator.describe()
    started = time.process_time()
    _run(operator, signal, state)
    seconds = time.process_time() - started
    per_thousand = seconds * 1000.0 / max(1, len(signal))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    within_budget = per_thousand <= float(spec["cost_cpu_seconds_per_1000"])
    within_memory = peak <= MEMORY_CEILING_BYTES
    return {"passed": bool(within_budget and within_memory),
            "declared_cpu_seconds_per_1000": float(spec["cost_cpu_seconds_per_1000"]),
            "measured_cpu_seconds_per_1000": round(per_thousand, 6),
            "peak_rss_bytes": peak, "memory_ceiling_bytes": MEMORY_CEILING_BYTES,
            "detail": None if within_budget and within_memory else
            ("the pilot cost more than the operator declares"
             if not within_budget else "the pilot exceeded the memory ceiling")}


def check_applicability(operator, state, signal, family: str) -> dict:
    """Test 9. An inapplicable family yields NOT_APPLICABLE, not a number."""
    spec = operator.describe()
    if family in spec["applicability"]:
        return {"passed": None, "family": family,
                "detail": "the operator declares itself applicable to this family, so there "
                          "is nothing to refuse"}
    if not hasattr(operator, "apply_to_family"):
        return {"passed": False, "family": family,
                "detail": "the operator offers no way to be asked about a family, so it "
                          "cannot refuse one"}
    answer = operator.apply_to_family(family, list(signal), state)
    ok = answer == contract.NOT_APPLICABLE
    return {"passed": ok, "family": family, "answer": answer if ok else repr(answer)[:120],
            "detail": None if ok else
            "an operator asked for a family it does not declare must answer NOT_APPLICABLE; "
            "returning a number is an undeclared extrapolation"}


def check_raw_branch(operator, state, signal) -> dict:
    """Test 10. The raw branch survives the transformation, unchanged."""
    output = _run(operator, signal, state)
    raw = list(output["raw"])
    ok = raw == list(signal)
    return {"passed": ok, "detail": None if ok else
            "the preserved raw branch is not the input: a transformation may add a branch, "
            "never replace the original"}


def run_battery(operator, signal, *, train_prefix=None, control=None,
                resource_contract=None, inapplicable_family="unknown_family") -> dict:
    """Every test, against one operator, with its verdict stated rather than inferred."""
    spec = contract.validate_spec(operator.describe())
    prefix = list(train_prefix if train_prefix is not None else signal[:len(signal) // 2])
    state = operator.fit(prefix)
    control_state = control.fit(prefix) if control is not None else None
    results = {
        "prefix": check_prefix(operator, state, signal),
        "altered_suffix": check_altered_suffix(operator, state, signal),
        "warm_up_edge": check_warm_up_edge(operator, state, signal),
        "chunk_restart": check_chunk_restart(operator, state, signal),
        "measured_delay": check_measured_delay(operator, state),
        "non_causal_control": check_non_causal_control(control, control_state, signal),
        "availability": check_availability(operator, resource_contract),
        "cost": check_cost(operator, state, signal),
        "applicability": check_applicability(operator, state, signal, inapplicable_family),
        "raw_branch": check_raw_branch(operator, state, signal),
    }
    failed = sorted(name for name, outcome in results.items() if outcome["passed"] is False)
    undecided = sorted(name for name, outcome in results.items() if outcome["passed"] is None)
    return {"schema": "df_d3_acceptance.v1", "generated_utc": now(),
            "kind": spec["kind"], "spec_sha256": contract.spec_sha256(spec),
            "state_sha256": contract.state_sha256(state),
            "samples": len(signal), "results": results,
            "failed": failed, "undecided": undecided,
            "verdict": ("MECHANICALLY_ACCEPTED" if not failed and not undecided else
                        "MECHANICALLY_REFUSED" if failed else "INCONCLUSIVE"),
            "note": ("Mechanical acceptance only: causality, edges, restart, delay, "
                     "availability, cost, applicability and the raw branch. It says the "
                     "operator's declaration about itself is true. It says nothing about "
                     "whether the operator is useful, and it selects nothing.")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-tests", action="store_true")
    args = parser.parse_args(argv)
    if args.list_tests:
        print(json.dumps({"schema": "df_d3_acceptance.v1", "tests": list(TESTS),
                          "memory_ceiling_bytes": MEMORY_CEILING_BYTES}, indent=1))
        return 0
    parser.error("this module is a battery; import it and call run_battery(operator, signal)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
