#!/usr/bin/env python3
"""The D3 operator contract, v2: what an operator must declare before it may be measured.

Amended under J1 (`docs/integracion_workplan_2026_09_10/07A_ENMIENDA_TEMPORAL_D3_2026_09_16.md`,
sealed in `df_d3_design.D3_AMENDMENT_V1`). The first version conflated three different
instants into "delay" and let a declaration pass with no non-causal twin and no probe. Four
instants are now separate and each has its own field:

    event index            t                    the sample the output is FOR
    input availability     available_at[t]      the resource contract's label + completion lag
    output emission        emitted_at[t]        >= the latest availability of what it consumed,
                                                plus the declared emission delay
    signal response        response_probe       measured with a predeclared probe; NOT a
                                                statement about availability

Nothing here scores, selects or promotes anything. A complete diagnosis with abstention is a
valid outcome, and the raw branch is never removed.
"""

from __future__ import annotations

import hashlib
import json
import math

SPEC_SCHEMA = "d3_operator_spec.v3"

NOT_AVAILABLE = "NOT_AVAILABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"
UNIDENTIFIED = "UNIDENTIFIED"

FIT_SCOPES = ("NONE", "TRAIN_PREFIX_ONLY")
CHUNK_RESTARTS = ("IDEMPOTENT", "STATEFUL_WITH_CHECKPOINT")
PROBE_KINDS = ("impulse", "step", "level_shift", "variance_shift", "none")
SUPPORT_KINDS = ("FINITE", "RECURSIVE", "POINTWISE")

#: Mandatory fields and their admissible types. `bool` is never an admissible number.
SPEC_FIELDS = {
    "schema": (str,),
    "kind": (str,),
    "params": (dict,),
    "bytes_state": (int,),
    "fit_scope": (str,),
    "lookback_samples": (int,),
    "warm_up_samples": (int,),
    "delay_samples": (int,),                 # emission delay beyond input availability
    "output_availability": (str,),           # "t + delay_samples", stated for a reader
    "response_probe": (dict,),               # {"kind", "expected_onset_samples", "scale"}
    "non_causal_twin": (dict,),              # {"kind"} or {"not_applicable", "reason"}
    "support": (dict,),                      # {"kind", "samples", "derivation", "boundary_mode"}
    "cost_cpu_seconds_per_1000": (float, int),
    "applicability": (list,),
    "chunk_restart": (str,),
}
OPTIONAL_FIELDS = {"non_causal_control_of": (str,), "notes": (str,), "library_versions": (dict,)}


class SpecRefusal(Exception):
    """A declaration that cannot be checked is refused, never defaulted."""


def _refuse(message: str):
    raise SpecRefusal(message)


def availability_delay(text):
    """`t` -> 0, `t + N` -> N. Anything else is not an availability statement."""
    if not isinstance(text, str):
        return None
    cleaned = text.replace(" ", "")
    if cleaned == "t":
        return 0
    if cleaned.startswith("t+") and cleaned[2:].isdigit():
        return int(cleaned[2:])
    return None


def validate_spec(spec) -> dict:
    """Every mandatory field present, typed, and internally consistent. No defaults."""
    if not isinstance(spec, dict):
        _refuse("a spec must be a JSON object")
    missing = sorted(set(SPEC_FIELDS) - set(spec))
    if missing:
        _refuse(f"the spec does not declare {missing}")
    unknown = sorted(set(spec) - set(SPEC_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown:
        _refuse(f"the spec declares fields the contract does not define: {unknown}")
    for name, types in {**SPEC_FIELDS, **OPTIONAL_FIELDS}.items():
        if name not in spec:
            continue
        value = spec[name]
        if isinstance(value, bool) or not isinstance(value, types):
            _refuse(f"{name!r} must be {'/'.join(t.__name__ for t in types)}, "
                    f"got {type(value).__name__}")
    if spec["schema"] != SPEC_SCHEMA:
        _refuse(f"schema must be {SPEC_SCHEMA!r}; v1 conflated the instants, v2 left the "
                "probe's scale to the fixture")
    if not spec["kind"]:
        _refuse("'kind' must name the operator")
    if spec["fit_scope"] not in FIT_SCOPES:
        _refuse(f"'fit_scope' must be one of {list(FIT_SCOPES)}; calibration and confirmation "
                "partitions are never a fitting scope")
    if spec["chunk_restart"] not in CHUNK_RESTARTS:
        _refuse(f"'chunk_restart' must be one of {list(CHUNK_RESTARTS)}")
    for name in ("bytes_state", "lookback_samples", "warm_up_samples", "delay_samples"):
        if spec[name] < 0:
            _refuse(f"{name!r} cannot be negative")
    cost = float(spec["cost_cpu_seconds_per_1000"])
    if not math.isfinite(cost) or cost <= 0:
        _refuse("'cost_cpu_seconds_per_1000' must be a finite positive measurement")
    if not spec["applicability"]:
        _refuse("'applicability' must name at least one family or regime; an operator that "
                "declares itself applicable to nothing cannot be measured")
    if any(not isinstance(item, str) or not item for item in spec["applicability"]):
        _refuse("'applicability' must be a list of non-empty names")
    delay = availability_delay(spec["output_availability"])
    if delay is None:
        _refuse("'output_availability' must read 't' or 't + N': it says WHEN the output for "
                f"sample t is emitted, and {spec['output_availability']!r} does not")
    if delay != spec["delay_samples"]:
        _refuse(f"'output_availability' says t + {delay} and 'delay_samples' says "
                f"{spec['delay_samples']}: an operator may not declare two different emission "
                "delays")
    if spec["fit_scope"] == "NONE" and spec["bytes_state"] != 0:
        _refuse("an operator that fits nothing cannot carry fitted state")
    _validate_probe(spec["response_probe"])
    _validate_twin(spec["non_causal_twin"], spec["kind"])
    _validate_support(spec["support"], spec["lookback_samples"])
    return spec


PROBE_SCALE = "TRAIN_FIT"


def _validate_probe(probe: dict) -> None:
    if set(probe) != {"kind", "expected_onset_samples", "scale"}:
        _refuse("'response_probe' must declare exactly 'kind', 'expected_onset_samples' and "
                "'scale'; v2 declarations left the excitation's scale to the fixture")
    if probe["scale"] != PROBE_SCALE:
        _refuse(f"'response_probe.scale' must be {PROBE_SCALE!r}: the excitation is built from "
                "the training fit and the operator's declared resolution, never from the "
                "fixture")
    if probe["kind"] not in PROBE_KINDS:
        _refuse(f"'response_probe.kind' must be one of {list(PROBE_KINDS)}")
    onset = probe["expected_onset_samples"]
    if onset == UNIDENTIFIED:
        return
    if isinstance(onset, bool) or not isinstance(onset, int) or onset < 0:
        _refuse("'response_probe.expected_onset_samples' is a non-negative integer or "
                f"{UNIDENTIFIED!r}; a response the probe cannot identify is never a fabricated "
                "zero")
    if probe["kind"] == "none":
        _refuse("a probe of kind 'none' cannot expect an onset; declare UNIDENTIFIED")


def _validate_twin(twin: dict, kind: str) -> None:
    if twin.get("not_applicable") is True:
        if set(twin) != {"not_applicable", "reason"} or not str(twin.get("reason", "")).strip():
            _refuse("a NOT_APPLICABLE twin needs a design reason; absence is not a pass")
        return
    if set(twin) != {"kind"} or not isinstance(twin["kind"], str) or not twin["kind"]:
        _refuse("'non_causal_twin' names the twin's kind, or is {'not_applicable': true, "
                "'reason': ...}")
    if twin["kind"] == kind:
        _refuse("an operator cannot be its own non-causal twin")


def _validate_support(support: dict, lookback: int) -> None:
    if set(support) != {"kind", "samples", "derivation", "boundary_mode"}:
        _refuse("'support' must declare kind, samples, derivation and boundary_mode")
    if support["kind"] not in SUPPORT_KINDS:
        _refuse(f"'support.kind' must be one of {list(SUPPORT_KINDS)}")
    samples = support["samples"]
    if support["kind"] == "RECURSIVE":
        if samples is not None:
            _refuse("a RECURSIVE support has no finite sample count; its state is a dependency, "
                    "not a memory of order p")
    else:
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
            _refuse("a FINITE or POINTWISE support declares a non-negative sample count")
        if samples > lookback + 1:
            _refuse(f"support of {samples} samples exceeds lookback {lookback} + 1: the operator "
                    "reads more past than it declares")
    if not isinstance(support["derivation"], str) or not support["derivation"].strip():
        _refuse("'support.derivation' says how the support was derived (library, mode, level)")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def spec_sha256(spec: dict) -> str:
    return hashlib.sha256(canonical(validate_spec(spec)).encode("ascii")).hexdigest()


def state_sha256(state) -> str:
    if isinstance(state, (bytes, bytearray)):
        return hashlib.sha256(bytes(state)).hexdigest()
    return hashlib.sha256(canonical(state).encode("ascii")).hexdigest()


# --- inputs and outputs -------------------------------------------------------------------

def validate_input(x, *, name: str = "input") -> dict:
    """An operator input: values, timestamps and availability, all the same length.

    Timestamps and availability are integer seconds (or sample indices under SAMPLE_INDEX);
    `period_seconds` is the declared sampling contract or None when there is none, in which
    case no duration is ever converted to samples.
    """
    if not isinstance(x, dict):
        _refuse(f"{name} must be an object with values, timestamps, available_at")
    for key in ("values", "timestamps", "available_at"):
        if key not in x:
            _refuse(f"{name} does not carry {key!r}")
    n = len(x["values"])
    if len(x["timestamps"]) != n or len(x["available_at"]) != n:
        _refuse(f"{name}: values, timestamps and available_at must have the same length")
    period = x.get("period_seconds")
    if period is not None and (isinstance(period, bool) or not isinstance(period, (int, float))
                               or period <= 0):
        _refuse(f"{name}: period_seconds is a positive number or None")
    for i in range(n):
        if x["available_at"][i] < x["timestamps"][i]:
            _refuse(f"{name}: row {i} is available before its own timestamp")
    return x


def validate_output(output, *, spec: dict, x: dict) -> dict:
    """values, an availability mask, emission times and the raw branch, all of length n.

    An available output inside the declared warm-up is a fabricated observation. An available
    output emitted before the input it is for was available is a leak by the clock rather
    than by the index, and is refused here before any test runs.
    """
    n = len(x["values"])
    if not isinstance(output, dict):
        _refuse("an output must be an object with values, available, emitted_at and raw")
    for key in ("values", "available", "emitted_at", "raw"):
        if key not in output:
            _refuse(f"the output does not carry {key!r}")
    lengths = {key: len(output[key]) for key in ("values", "available", "emitted_at", "raw")}
    if len(set(lengths.values())) != 1 or lengths["values"] != n:
        _refuse(f"values, available, emitted_at and raw must all have {n} entries, got {lengths}")
    available = [bool(flag) for flag in output["available"]]
    if any(available[:spec["warm_up_samples"]]):
        _refuse(f"the first {spec['warm_up_samples']} outputs are declared warm-up and must "
                "be unavailable; an available warm-up output is a fabricated observation")
    for i in range(n):
        if available[i]:
            emitted = output["emitted_at"][i]
            if emitted is None or (isinstance(emitted, str)):
                _refuse(f"output {i} is available but carries no emission time")
            if emitted < x["available_at"][i]:
                _refuse(f"output {i} is emitted at {emitted}, before its own input was "
                        f"available at {x['available_at'][i]}")
    return output


def emission_times(x: dict, *, lookback: int, delay: int) -> list:
    """The earliest honest emission time of each output: the latest availability of the inputs
    in its declared window, plus the declared emission delay in whole periods.

    Shared by operators so none of them invents its own clock. With no sampling contract the
    delay cannot be turned into seconds, and a positive delay is then refused rather than
    guessed.
    """
    period = x.get("period_seconds")
    if delay and period is None:
        _refuse("an emission delay in samples needs a declared sampling period to become a time")
    out = []
    for t in range(len(x["values"])):
        start = max(0, t - lookback)
        latest = max(x["available_at"][start:t + 1])
        out.append(latest + (delay * period if delay else 0))
    return out


__all__ = ["SPEC_SCHEMA", "NOT_AVAILABLE", "NOT_APPLICABLE", "UNIDENTIFIED", "SpecRefusal",
           "SPEC_FIELDS", "OPTIONAL_FIELDS", "FIT_SCOPES", "CHUNK_RESTARTS", "PROBE_KINDS",
           "SUPPORT_KINDS", "validate_spec", "availability_delay", "spec_sha256",
           "state_sha256", "validate_input", "validate_output", "emission_times", "canonical"]
