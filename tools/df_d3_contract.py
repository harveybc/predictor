#!/usr/bin/env python3
"""The D3 operator contract: what an operator must declare before it may be measured.

Block 3 of `docs/handoffs/MUSASHI_I1_I3_ACCEPTANCE_AND_STACK_FOLLOWUP_2026_09_16.md` asks for
the next preprocessing step's **data contracts, tests and governed execution plan** — not for
its execution and not for a new design. The design is sealed in
`docs/integracion_workplan_2026_09_10/07_DISENO_D3_CUANTIZACION_TIEMPO_FRECUENCIA_DETECTORES_2026_09_14.md`
and is transcribed here, not extended: the mandatory specification fields of its §1, and
nothing invented beside them.

What this module is for: an operator that cannot state its own lookback, delay, warm-up,
availability and cost cannot be checked for causality, and an operator that is merely *believed*
causal is how a leak enters a data foundation. So the declaration comes first, and the battery
in `df_d3_acceptance.py` measures the declaration against the operator's behaviour.

Nothing here scores, selects or promotes anything. A complete diagnosis with abstention is a
valid outcome, and the raw branch is never removed.
"""

from __future__ import annotations

import hashlib
import json
import math

#: An output that does not exist yet. It is NOT zero, and it is not a number: a warm-up written
#: as 0.0 is a fabricated observation, which is the failure mode acceptance test 3 exists for.
NOT_AVAILABLE = "NOT_AVAILABLE"
#: An operator asked for a family it does not declare itself applicable to. Also not a number.
NOT_APPLICABLE = "NOT_APPLICABLE"

FIT_SCOPES = ("NONE", "TRAIN_PREFIX_ONLY")
CHUNK_RESTARTS = ("IDEMPOTENT", "STATEFUL_WITH_CHECKPOINT")

#: The specification fields §1 of the design makes mandatory, with their admissible types.
#: `params` and `bytes_state` describe the fitted state; the rest describe behaviour that the
#: acceptance battery then measures.
SPEC_FIELDS = {
    "kind": (str,),
    "params": (dict,),
    "bytes_state": (int,),
    "fit_scope": (str,),
    "lookback_samples": (int,),
    "output_availability": (str,),
    "warm_up_samples": (int,),
    "delay_samples": (int,),
    "cost_cpu_seconds_per_1000": (float, int),
    "applicability": (list,),
    "chunk_restart": (str,),
}
#: Optional, and meaningful only for a control: the spec it is the deliberate non-causal twin
#: of. A control is recorded and never promoted, so it has to be able to say what it is.
OPTIONAL_FIELDS = {"non_causal_control_of": (str,), "notes": (str,)}


class SpecRefusal(Exception):
    """A declaration that cannot be checked is refused, never defaulted."""


def _refuse(message: str):
    raise SpecRefusal(message)


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
    availability = spec["output_availability"]
    delay = availability_delay(availability)
    if delay is None:
        _refuse("'output_availability' must read 't' or 't + N': it says WHEN the output for "
                f"sample t is complete, and {availability!r} does not")
    if delay != spec["delay_samples"]:
        # A disagreement here is precisely a silent claim of zero delay. Acceptance test 5
        # then measures the number against an impulse, so the declaration cannot be both
        # self-consistent and wrong for free.
        _refuse(f"'output_availability' says t + {delay} and 'delay_samples' says "
                f"{spec['delay_samples']}: an operator may not declare two different delays")
    if spec["fit_scope"] == "NONE" and spec["bytes_state"] != 0:
        _refuse("an operator that fits nothing cannot carry fitted state")
    return spec


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


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def spec_sha256(spec: dict) -> str:
    """The identity of a declaration. The design requires every output to carry it."""
    return hashlib.sha256(canonical(validate_spec(spec)).encode("ascii")).hexdigest()


def state_sha256(state) -> str:
    """The identity of the fitted state, so a result can name the state that produced it."""
    if isinstance(state, (bytes, bytearray)):
        return hashlib.sha256(bytes(state)).hexdigest()
    return hashlib.sha256(canonical(state).encode("ascii")).hexdigest()


def validate_output(output, *, spec: dict, samples: int) -> dict:
    """An operator's output: values, an availability mask, and the raw branch it preserved.

    The raw branch is part of the contract rather than a convention, because the design says a
    transformation never replaces the original before it has shown utility — and a rule that
    lives only in prose is one nobody can fail.
    """
    if not isinstance(output, dict):
        _refuse("an output must be an object with 'values', 'available' and 'raw'")
    for key in ("values", "available", "raw"):
        if key not in output:
            _refuse(f"the output does not carry {key!r}")
    lengths = {key: len(output[key]) for key in ("values", "available", "raw")}
    if len(set(lengths.values())) != 1 or lengths["values"] != samples:
        _refuse(f"values, available and raw must all have {samples} entries, got {lengths}")
    if any(bool(flag) for flag in list(output["available"])[:spec["warm_up_samples"]]):
        _refuse(f"the first {spec['warm_up_samples']} outputs are declared warm-up and must "
                "be unavailable; an available warm-up output is a fabricated observation")
    return output


__all__ = ["NOT_AVAILABLE", "NOT_APPLICABLE", "SpecRefusal", "SPEC_FIELDS", "FIT_SCOPES",
           "CHUNK_RESTARTS", "OPTIONAL_FIELDS", "validate_spec", "availability_delay",
           "spec_sha256", "state_sha256", "validate_output", "canonical"]
