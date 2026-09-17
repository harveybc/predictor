#!/usr/bin/env python3
"""The D3 acceptance battery, v2: the required-test matrix of the sealed amendment.

Amended under J1/J2 (`df_d3_design.D3_AMENDMENT_V1`). Twelve mandatory tests; review-ready
means all of them ran and passed, and a scoped one (`non_causal_twin` marked NOT_APPLICABLE
with a design reason) is reported as scoped, never silently accepted.

    prefix_all_available    every output available at the cut: value, mask, emitted_at; no
                            lookback exemption; empty population is INSUFFICIENT_TEST
    future_perturbation     the future replaced by nine adversarial suffixes; the past is
                            unchanged
    warm_up_edge            the first warm_up_samples outputs are unavailable, not 0
    fit_scope_train_only    refit on the train prefix with a different evaluated future;
                            outputs before the cut are unchanged
    fresh_state_per_branch  a transform does not mutate the state it was given
    chunk_restart           two chunks with re-read lookback or a checkpoint reproduce one pass
    response_probe          the declared probe shows the declared onset, or UNIDENTIFIED with
                            a reason; never a fabricated zero
    non_causal_twin         the declared twin FAILS prefix/future; NOT_APPLICABLE only with a
                            reason; absence is a refusal
    availability_emission   emitted_at >= the latest availability of consumed inputs, with the
                            resource's durations parsed as the producer parses them
    cost_pilot              single-thread CPU within the declaration and under 2 GiB
    applicability           an undeclared family yields NOT_APPLICABLE, not a number
    raw_branch              the raw branch survives, unchanged

An operator is an object with `describe()`, `fit(train)`, `transform(x, state)`, optionally
`checkpoint()`/`resume(blob)` and `apply_to_family(family, x, state)`. `x` and `train` are
inputs of `df_d3_contract.validate_input`. Every test branch fits its own state.

Nothing here writes to a store, opens a database or starts a governed run.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    """One module object per name, shared with whoever loaded it first: a refusal raised by
    the contract must be the same class the caller catches."""
    import sys

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("df_d3_contract")
design = _load("df_d3_design")

MEMORY_CEILING_BYTES = design.D3_AMENDMENT_V1["memory_ceiling_bytes"]
TESTS = tuple(design.REQUIRED_TESTS)
SCOPEABLE = tuple(design.SCOPEABLE_TESTS)
FUTURE_PERTURBATIONS = tuple(design.FUTURE_PERTURBATIONS)
RANDOM_CUTS = design.CUT_MENU["random"]["count"]
CUT_SEED = design.CUT_MENU["random"]["seed"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# --- inputs ---------------------------------------------------------------------------------

def make_input(values, *, timestamps=None, available_at=None, period_seconds=None) -> dict:
    """An input under SAMPLE_INDEX semantics unless timestamps are given."""
    n = len(values)
    if timestamps is None:
        timestamps = list(range(n))
        period_seconds = 1 if period_seconds is None else period_seconds
    if available_at is None:
        available_at = list(timestamps)
    return contract.validate_input({"values": list(values), "timestamps": list(timestamps),
                                    "available_at": list(available_at),
                                    "period_seconds": period_seconds})


def prefix(x: dict, n: int) -> dict:
    return {"values": x["values"][:n], "timestamps": x["timestamps"][:n],
            "available_at": x["available_at"][:n], "period_seconds": x.get("period_seconds")}


def with_values(x: dict, values) -> dict:
    return {"values": list(values), "timestamps": list(x["timestamps"]),
            "available_at": list(x["available_at"]), "period_seconds": x.get("period_seconds")}


def _seeded(seed: int):
    import random

    return random.Random(seed)


def cut_menu(spec: dict, n: int, *, seed: int = CUT_SEED) -> list:
    """The sealed menu of §4: predetermined, seeded, never chosen after a result."""
    marks = set()
    j = 0
    while 2 ** j <= n:
        marks.update({2 ** j - 1, 2 ** j, 2 ** j + 1})
        j += 1
    w = spec["warm_up_samples"]
    marks.update({w - 1, w, w + 1})
    window = spec["params"].get("window") or spec["params"].get("segment") \
        or spec["params"].get("w")
    if isinstance(window, int) and window > 0:
        marks.update({k * window for k in (1, 2, 3)})
    marks.update({0, n - 2, n - 1})
    rng = _seeded(seed)
    marks.update(rng.randrange(0, n) for _ in range(RANDOM_CUTS))
    return sorted(m for m in marks if 0 <= m < n)


def perturbation_cuts(spec: dict, n: int, *, seed: int = CUT_SEED + 1) -> list:
    """The cuts the nine adversarial suffixes are applied at: the edges, the warm-up boundary
    and the seeded random draw of the menu. The D2 battery applied its suffixes at a seeded
    subset on long series too (`declared_cuts`, SUFFIX_ADVERSARIAL); the full power-of-two
    ladder stays with the prefix test, where every cut costs one transform rather than nine."""
    w = spec["warm_up_samples"]
    marks = {0, w - 1, w, w + 1, n - 2, n - 1}
    rng = _seeded(seed)
    marks.update(rng.randrange(0, n) for _ in range(RANDOM_CUTS))
    return sorted(m for m in marks if 0 <= m < n)


# --- outputs --------------------------------------------------------------------------------

def _run(operator, x: dict, state):
    output = operator.transform(x, state)
    return contract.validate_output(output, spec=operator.describe(), x=x)


def _triple(output, i):
    return (output["values"][i] if output["available"][i] else contract.NOT_AVAILABLE,
            bool(output["available"][i]),
            output["emitted_at"][i] if output["available"][i] else None)


def _same_value(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _same_row(p, q) -> bool:
    return _same_value(p[0], q[0]) and p[1] == q[1] and p[2] == q[2]


def _fit(operator, train: dict):
    return operator.fit(copy.deepcopy(train))


# --- the twelve tests -----------------------------------------------------------------------

def check_prefix_all_available(operator, train, x) -> dict:
    """§3: every output available at the cut, with value, mask and emission, at every cut."""
    spec = operator.describe()
    whole = _run(operator, x, _fit(operator, train))
    compared, failures = 0, []
    for c in cut_menu(spec, len(x["values"])):
        part = _run(operator, prefix(x, c + 1), _fit(operator, train))
        for i in range(c + 1):
            # A delayed representation is compared only once it has been emitted for both
            # runs; an output the full run has not emitted by the cut is not yet a fact.
            if not whole["available"][i]:
                continue
            if whole["emitted_at"][i] > x["available_at"][c]:
                continue
            compared += 1
            if not _same_row(_triple(whole, i), _triple(part, i)):
                failures.append({"cut": c, "index": i, "full": _triple(whole, i),
                                 "prefix": _triple(part, i)})
                break
    if compared == 0:
        return {"passed": None, "outcome": "INSUFFICIENT_TEST", "compared": 0,
                "detail": "no output was available at any cut, so nothing was tested"}
    ok = not failures
    return {"passed": ok, "compared": compared, "failures": failures[:5],
            "detail": None if ok else
            f"{len(failures)} cuts changed an output that was already available: the operator "
            "reads past the cut"}


def _perturb(kind: str, x: dict, c: int, seed: int) -> list:
    tail = list(x["values"][c + 1:])
    m = len(tail)
    rng = _seeded(seed + c)
    if kind == "zeros":
        return [0.0] * m
    if kind == "large_constant":
        return [1e6] * m
    if kind == "other_seed_noise":
        return [rng.gauss(0.0, 5.0) for _ in range(m)]
    if kind == "reversed":
        return list(reversed(x["values"]))[c + 1:]
    if kind == "nan_blocks":
        return [float("nan") if (i // 3) % 2 == 0 else v for i, v in enumerate(tail)]
    if kind == "impulse_at_t_plus_1":
        return [(tail[0] + 1e3 if m else 0.0)] + tail[1:]
    if kind == "step":
        return [v + 50.0 for v in tail]
    if kind == "chirp":
        return [10.0 * math.sin(2 * math.pi * (0.01 + 0.2 * i / max(m, 1)) * i)
                for i in range(m)]
    if kind == "regime_change":
        return [v * 10.0 + 100.0 for v in tail]
    raise ValueError(kind)


def check_future_perturbation(operator, train, x) -> dict:
    """§4: the future replaced by every adversarial suffix; outputs <= cut do not move."""
    spec = operator.describe()
    base = _run(operator, x, _fit(operator, train))
    n = len(x["values"])
    cuts = [c for c in perturbation_cuts(spec, n) if c < n - 1]
    compared, failures = 0, []
    for c in cuts:
        for kind in FUTURE_PERTURBATIONS:
            altered = with_values(x, list(x["values"][:c + 1]) + _perturb(kind, x, c, CUT_SEED))
            other = _run(operator, altered, _fit(operator, train))
            for i in range(c + 1):
                if not base["available"][i]:
                    continue
                compared += 1
                if not _same_row(_triple(base, i), _triple(other, i)):
                    failures.append({"cut": c, "suffix": kind, "index": i})
                    break
    if compared == 0:
        return {"passed": None, "outcome": "INSUFFICIENT_TEST", "compared": 0,
                "detail": "no available output preceded any cut"}
    ok = not failures
    return {"passed": ok, "compared": compared, "failures": failures[:5],
            "detail": None if ok else
            f"{len(failures)} (cut, suffix) pairs moved an output at or before the cut"}


def check_warm_up_edge(operator, train, x) -> dict:
    spec = operator.describe()
    output = _run(operator, x, _fit(operator, train))
    warm = spec["warm_up_samples"]
    available = [bool(f) for f in output["available"]]
    early = [i for i in range(min(warm, len(available))) if available[i]]
    later = available[warm:]
    ok = not early and (not later or any(later))
    return {"passed": ok, "warm_up_samples": warm,
            "detail": None if ok else
            (f"outputs {early[:5]} are marked available inside the declared warm-up" if early
             else "no output is ever available: a warm-up that never ends is not a warm-up")}


def check_fit_scope_train_only(operator, train, x) -> dict:
    """§4: refit on the same train prefix, evaluate a different future; the past is unchanged.

    The fit only ever sees `train`, which is strictly before the evaluated future. If any
    output before the train boundary moves when the future changes, the operator's fit or
    transform reached past the boundary.
    """
    spec = operator.describe()
    boundary = len(train["values"])
    n = len(x["values"])
    if boundary >= n:
        return {"passed": None, "outcome": "INSUFFICIENT_TEST",
                "detail": "the train prefix covers the whole series; no evaluated future exists"}
    base = _run(operator, x, _fit(operator, train))
    altered = with_values(x, list(x["values"][:boundary])
                          + _perturb("regime_change", x, boundary - 1, CUT_SEED + 11))
    other = _run(operator, altered, _fit(operator, train))
    compared = 0
    for i in range(boundary):
        if not base["available"][i]:
            continue
        compared += 1
        if not _same_row(_triple(base, i), _triple(other, i)):
            return {"passed": False, "boundary": boundary, "index": i,
                    "detail": "an output before the train boundary moved when only the "
                              "evaluated future changed"}
    if compared == 0:
        return {"passed": None, "outcome": "INSUFFICIENT_TEST", "boundary": boundary,
                "detail": "no available output lies before the train boundary"}
    if spec["fit_scope"] == "TRAIN_PREFIX_ONLY":
        a = contract.state_sha256(_state_facts(_fit(operator, train)))
        b = contract.state_sha256(_state_facts(_fit(operator, train)))
        if a != b:
            return {"passed": False, "boundary": boundary,
                    "detail": "fitting the same train prefix twice gave two different states"}
    return {"passed": True, "boundary": boundary, "compared": compared}


def _state_facts(state):
    try:
        json.dumps(state, sort_keys=True)
        return state
    except TypeError:
        return repr(state)


def check_fresh_state_per_branch(operator, train, x) -> dict:
    """§4: a transform must not mutate the state it was handed."""
    state = _fit(operator, train)
    before = contract.state_sha256(_state_facts(copy.deepcopy(state)))
    first = _run(operator, x, state)
    after = contract.state_sha256(_state_facts(copy.deepcopy(state)))
    second = _run(operator, x, state)
    same = all(_same_row(_triple(first, i), _triple(second, i)) for i in range(len(x["values"])))
    ok = before == after and same
    return {"passed": ok,
            "detail": None if ok else
            ("the state changed after a transform" if before != after else
             "two transforms with the same state gave different outputs")}


def check_chunk_restart(operator, train, x) -> dict:
    """§4, separately from the fit: a restart reproduces one pass."""
    spec = operator.describe()
    whole = _run(operator, x, _fit(operator, train))
    n = len(x["values"])
    cut = n // 2
    if spec["chunk_restart"] == "IDEMPOTENT":
        back = min(cut, spec["lookback_samples"] + spec["delay_samples"])
        tail_x = {"values": x["values"][cut - back:], "timestamps": x["timestamps"][cut - back:],
                  "available_at": x["available_at"][cut - back:],
                  "period_seconds": x.get("period_seconds")}
        tail = _run(operator, tail_x, _fit(operator, train))
        compared, bad = 0, None
        for i in range(cut, n):
            if not whole["available"][i]:
                continue
            compared += 1
            if not _same_row(_triple(whole, i), _triple(tail, i - cut + back)):
                bad = i
                break
        if compared == 0:
            return {"passed": None, "outcome": "INSUFFICIENT_TEST", "mode": "IDEMPOTENT",
                    "detail": "no available output after the restart point"}
        return {"passed": bad is None, "mode": "IDEMPOTENT", "compared": compared,
                "restart_lookback": back,
                "detail": None if bad is None else
                f"restarting at {cut} with {back} samples of lookback differs at {bad}"}
    if not hasattr(operator, "checkpoint") or not hasattr(operator, "resume"):
        return {"passed": False, "mode": spec["chunk_restart"],
                "detail": "STATEFUL_WITH_CHECKPOINT declared with no checkpoint()/resume()"}
    state = _fit(operator, train)
    head = _run(operator, prefix(x, cut), state)
    blob = operator.checkpoint()
    resumed = operator.resume(blob)
    tail_x = {"values": x["values"][cut:], "timestamps": x["timestamps"][cut:],
              "available_at": x["available_at"][cut:], "period_seconds": x.get("period_seconds")}
    tail = contract.validate_output(operator.transform(tail_x, resumed), spec=spec, x=tail_x)
    rebuilt = [_triple(head, i) for i in range(cut)] + [_triple(tail, i) for i in range(n - cut)]
    bad = next((i for i in range(n) if whole["available"][i]
                and not _same_row(_triple(whole, i), rebuilt[i])), None)
    return {"passed": bad is None, "mode": spec["chunk_restart"],
            "detail": None if bad is None else f"resume from the checkpoint differs at {bad}"}


def _probe_series(kind: str, length: int, position: int, *, baseline: float, sigma: float,
                  amplitude: float, gain: float) -> tuple:
    """Quiet and excited branches over the SAME seeded noise, on the training fit's scale."""
    rng = _seeded(CUT_SEED + 3)
    quiet = [baseline + sigma * rng.gauss(0.0, 1.0) for _ in range(length)]
    hit = list(quiet)
    if kind == "impulse":
        hit[position] += amplitude
    elif kind in ("step", "level_shift"):
        for i in range(position, length):
            hit[i] += amplitude
    elif kind == "variance_shift":
        for i in range(position, length):
            hit[i] = baseline + (quiet[i] - baseline) * gain
    return quiet, hit


def _unidentified(probe, reason, **facts):
    return {"passed": None, "outcome": "UNIDENTIFIED", "probe": probe["kind"],
            "identifiable": False, "first_change_observed": None, "matches_declared": None,
            "declared": probe["expected_onset_samples"], "detail": reason, **facts}


def check_response_probe(operator, train, x, *, length: int = 256) -> dict:
    """§6 as amended (K2): the excitation is built from the training fit and the operator's
    declared resolution; three facts are recorded apart — whether the excitation is
    identifiable (declared), the first change observed (measured), and whether it matches the
    declared onset. A declared-identifiable excitation that moves nothing FAILS; a first change
    later than declared FAILS; only a declared UNIDENTIFIED abstains, and it costs the verdict."""
    spec = operator.describe()
    probe = spec["response_probe"]
    if probe["kind"] == "none" or probe["expected_onset_samples"] == contract.UNIDENTIFIED:
        return _unidentified(probe, "the operator declares that no probe identifies its "
                                    "response onset; recorded as such, not as zero")
    finite = sorted(v for v in train["values"] if isinstance(v, (int, float))
                    and not (isinstance(v, float) and math.isnan(v)))
    if len(finite) < 2:
        return _unidentified(probe, "the training fit has fewer than two finite samples")
    baseline = finite[int(0.10 * (len(finite) - 1))]
    scale = finite[int(0.90 * (len(finite) - 1))] - baseline
    if not scale > 0:
        return _unidentified(probe, "constant training fit: no excitation is identifiable on a "
                                    "degenerate domain", baseline=baseline, scale=scale)
    state = _fit(operator, train)
    resolution = operator.probe_resolution(state, baseline=baseline, scale=scale,
                                           sigma=0.01 * scale)
    excitation = {"kind": probe["kind"], "baseline": baseline, "scale": scale,
                  "sigma": 0.01 * scale, "amplitude": resolution.get("amplitude"),
                  "gain": resolution.get("amplitude") if resolution.get("gain") else None,
                  "reason": resolution.get("reason")}
    if resolution.get("amplitude") == contract.UNIDENTIFIED:
        return _unidentified(probe, f"no identifiable excitation from the fit: "
                                    f"{resolution.get('reason')}", excitation=excitation)
    amplitude = float(resolution["amplitude"])
    if not amplitude > 0:
        return {"passed": False, "outcome": "FAILED", "probe": probe["kind"],
                "identifiable": False, "first_change_observed": None, "matches_declared": None,
                "declared": probe["expected_onset_samples"], "excitation": excitation,
                "detail": "the operator declared a non-positive resolution"}
    position = length // 2
    quiet, hit = _probe_series(probe["kind"], length, position, baseline=baseline,
                               sigma=0.01 * scale, amplitude=amplitude,
                               gain=amplitude if resolution.get("gain") else 1.0)
    base = _run(operator, make_input(quiet), _fit(operator, train))
    moved = _run(operator, make_input(hit), _fit(operator, train))
    first = next((i for i in range(position, length)
                  if base["available"][i] and moved["available"][i]
                  and not _same_value(base["values"][i], moved["values"][i])), None)
    facts = {"probe": probe["kind"], "identifiable": True, "excitation": excitation,
             "declared": probe["expected_onset_samples"],
             "emitted_after_impact": sum(1 for i in range(position, length)
                                         if base["available"][i] and moved["available"][i])}
    if first is None:
        return {"passed": False, "outcome": "FAILED", "first_change_observed": None,
                "matches_declared": False,
                "detail": "the operator declared this excitation identifiable and no available "
                          "output moved: the declaration is contradicted, not abstained", **facts}
    observed = first - position
    ok = observed == probe["expected_onset_samples"]
    return {"passed": ok, "outcome": "PASSED" if ok else "FAILED", "observed": observed,
            "first_change_observed": observed, "matches_declared": ok,
            "detail": None if ok else
            f"the {probe['kind']} first moved the output {observed} samples after it arrived; "
            f"the operator declares {probe['expected_onset_samples']}", **facts}


def check_non_causal_twin(operator, twin, train, x) -> dict:
    """§5 as amended (K3): the twin MUST demonstrate a causality failure; a twin without any
    observable comparison is INSUFFICIENT_TEST, never a detection and never a pass."""
    declared = operator.describe()["non_causal_twin"]
    if declared.get("not_applicable") is True:
        return {"passed": None, "scoped": True, "outcome": "NOT_APPLICABLE",
                "reason": declared["reason"],
                "detail": "no twin by design; the independent temporal tests still apply"}
    if twin is None:
        return {"passed": False, "scoped": False,
                "detail": f"the operator declares twin {declared['kind']!r} and none was "
                          "supplied to the battery: absence is not a pass"}
    twin_spec = twin.describe()
    if twin_spec["kind"] != declared["kind"]:
        return {"passed": False, "scoped": False,
                "detail": f"the supplied twin is {twin_spec['kind']!r}, the declaration names "
                          f"{declared['kind']!r}"}
    if twin_spec.get("non_causal_control_of") != operator.describe()["kind"]:
        return {"passed": False, "scoped": False,
                "detail": "the twin does not name the operator it is the deliberate twin of"}
    whole = _run(twin, x, _fit(twin, train))
    emissions = sum(1 for a in whole["available"] if a)
    p = check_prefix_all_available(twin, train, x)
    f = check_future_perturbation(twin, train, x)
    comparisons = int(p.get("compared") or 0) + int(f.get("compared") or 0)
    facts = {"scoped": False, "promoted": False, "twin_emissions": emissions,
             "twin_comparisons": comparisons, "twin_prefix_passed": p["passed"],
             "twin_future_passed": f["passed"]}
    if p["passed"] is False or f["passed"] is False:
        return {"passed": True, "outcome": "PASSED", "detail": None, **facts}
    if comparisons == 0:
        return {"passed": None, "outcome": "INSUFFICIENT_TEST",
                "detail": f"the twin emitted {emissions} outputs and no comparison was "
                          "observable at any cut: nothing was detected and nothing was shown; "
                          "absence of evidence is not evidence", **facts}
    return {"passed": False, "outcome": "FAILED",
            "detail": f"the declared twin was compared {comparisons} times and never failed "
                      "causality: either it is not the twin it claims to be or the tests are "
                      "not measuring causality", **facts}


def parse_duration_seconds(text):
    """The producer's own parser. Returns None for UNKNOWN/None, never zero."""
    if text in (None, "", "UNKNOWN"):
        return None
    import pandas as pd

    delta = pd.Timedelta(str(text))
    if pd.isna(delta) or delta < pd.Timedelta(0):
        raise ValueError(f"invalid completion lag {text!r}")
    return float(delta.total_seconds())


def duration_to_samples(seconds: float, period_seconds):
    """Exact division only. A fractional offset is refused, never truncated."""
    if period_seconds is None:
        raise ValueError("no sampling contract: a duration cannot become a sample count")
    ratio = seconds / float(period_seconds)
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(f"FRACTIONAL_SAMPLE_OFFSET: {seconds}s is {ratio} periods")
    return int(round(ratio))


def check_availability_emission(operator, train, x, resource_contract) -> dict:
    """§1-§2: emitted_at >= the latest availability of consumed inputs; durations as the
    producer parses them; UNKNOWN is undecided; a fractional offset is refused."""
    spec = operator.describe()
    if not isinstance(resource_contract, dict):
        return {"passed": None, "outcome": "INSUFFICIENT_TEST",
                "detail": "no resource availability contract was supplied"}
    block = resource_contract.get("availability") or resource_contract
    lag_text = block.get("completion_lag_max")
    try:
        lag = parse_duration_seconds(lag_text)
    except ValueError as exc:
        return {"passed": False, "lag": lag_text, "detail": str(exc)}
    if lag is None:
        return {"passed": None, "outcome": "UNKNOWN", "lag": lag_text,
                "detail": "the resource declares an UNKNOWN completion lag, so no emission "
                          "time can be certified; the claim is undecided, not the operator"}
    period = x.get("period_seconds")
    frequency = resource_contract.get("frequency")
    if period is None and frequency:
        try:
            period = parse_duration_seconds(frequency)
        except ValueError as exc:
            return {"passed": False, "detail": f"frequency {frequency!r}: {exc}"}
    lag_samples = None
    if period is not None:
        try:
            lag_samples = duration_to_samples(lag, period)
        except ValueError as exc:
            return {"passed": False, "lag": lag_text, "period_seconds": period,
                    "detail": str(exc)}
    # The input the battery hands the operator carries BOTH availabilities, and the later
    # one wins: the contract's nominal lag never erases a late arrival the snapshot recorded,
    # and a recorded arrival never claims to precede what the contract says is complete.
    shifted = {"values": x["values"], "timestamps": x["timestamps"],
               "available_at": [max(a, t + lag) for a, t in
                                zip(x["available_at"], x["timestamps"])],
               "period_seconds": period}
    output = _run(operator, shifted, _fit(operator, train))
    lower = contract.emission_times(shifted, lookback=spec["lookback_samples"],
                                    delay=spec["delay_samples"]) if period is not None or \
        not spec["delay_samples"] else None
    violations, checked = [], 0
    for i in range(len(x["values"])):
        if not output["available"][i]:
            continue
        checked += 1
        emitted = output["emitted_at"][i]
        if emitted < shifted["available_at"][i]:
            violations.append({"index": i, "emitted_at": emitted,
                               "available_at": shifted["available_at"][i]})
        elif lower is not None and emitted < lower[i]:
            violations.append({"index": i, "emitted_at": emitted, "consumed_until": lower[i]})
    if checked == 0:
        return {"passed": None, "outcome": "INSUFFICIENT_TEST",
                "detail": "no output was available under the resource's availability"}
    ok = not violations
    return {"passed": ok, "lag_seconds": lag, "lag_samples": lag_samples,
            "period_seconds": period, "checked": checked, "violations": violations[:5],
            "detail": None if ok else
            f"{len(violations)} outputs were emitted before an input they consume was available"}


def check_cost_pilot(operator, train, x) -> dict:
    spec = operator.describe()
    state = _fit(operator, train)
    started = time.process_time()
    _run(operator, x, state)
    seconds = time.process_time() - started
    per_thousand = seconds * 1000.0 / max(1, len(x["values"]))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    within_budget = per_thousand <= float(spec["cost_cpu_seconds_per_1000"])
    within_memory = peak <= MEMORY_CEILING_BYTES
    return {"passed": bool(within_budget and within_memory),
            "declared_cpu_seconds_per_1000": float(spec["cost_cpu_seconds_per_1000"]),
            "measured_cpu_seconds_per_1000": round(per_thousand, 6),
            "peak_rss_bytes": peak, "memory_ceiling_bytes": MEMORY_CEILING_BYTES,
            "detail": None if within_budget and within_memory else
            ("the pilot cost more than the operator declares" if not within_budget
             else "the pilot exceeded the memory ceiling")}


def check_applicability(operator, train, x, family: str) -> dict:
    spec = operator.describe()
    if family in spec["applicability"]:
        return {"passed": None, "outcome": "INSUFFICIENT_TEST", "family": family,
                "detail": "the operator declares itself applicable to this family"}
    if not hasattr(operator, "apply_to_family"):
        return {"passed": False, "family": family,
                "detail": "the operator offers no way to be asked about a family, so it "
                          "cannot refuse one"}
    answer = operator.apply_to_family(family, x, _fit(operator, train))
    ok = answer == contract.NOT_APPLICABLE
    return {"passed": ok, "family": family,
            "detail": None if ok else
            "an operator asked for a family it does not declare must answer NOT_APPLICABLE"}


def check_raw_branch(operator, train, x) -> dict:
    output = _run(operator, x, _fit(operator, train))
    ok = all(_same_value(a, b) for a, b in zip(output["raw"], x["values"])) \
        and len(output["raw"]) == len(x["values"])
    return {"passed": ok, "detail": None if ok else
            "the preserved raw branch is not the input"}


# --- the battery ----------------------------------------------------------------------------

def run_battery(operator, x, *, train=None, twin=None, resource_contract=None,
                inapplicable_family="unknown_family") -> dict:
    x = contract.validate_input(x)
    spec = contract.validate_spec(operator.describe())
    train = train if train is not None else prefix(x, max(1, len(x["values"]) * 6 // 10))
    contract.validate_input(train, name="train")
    results = {
        "prefix_all_available": check_prefix_all_available(operator, train, x),
        "future_perturbation": check_future_perturbation(operator, train, x),
        "warm_up_edge": check_warm_up_edge(operator, train, x),
        "fit_scope_train_only": check_fit_scope_train_only(operator, train, x),
        "fresh_state_per_branch": check_fresh_state_per_branch(operator, train, x),
        "chunk_restart": check_chunk_restart(operator, train, x),
        "response_probe": check_response_probe(operator, train, x),
        "non_causal_twin": check_non_causal_twin(operator, twin, train, x),
        "availability_emission": check_availability_emission(operator, train, x,
                                                             resource_contract),
        "cost_pilot": check_cost_pilot(operator, train, x),
        "applicability": check_applicability(operator, train, x, inapplicable_family),
        "raw_branch": check_raw_branch(operator, train, x),
    }
    whole = _run(operator, x, _fit(operator, train))
    coverage = {"n": len(x["values"]), "emitted": sum(1 for a in whole["available"] if a),
                "inputs_available": sum(1 for v in x["values"]
                                        if not (isinstance(v, float) and math.isnan(v)))}
    failed = sorted(k for k, r in results.items() if r["passed"] is False)
    scoped = sorted(k for k, r in results.items()
                    if r["passed"] is None and r.get("scoped") and k in SCOPEABLE)
    undecided = sorted(k for k, r in results.items()
                       if r["passed"] is None and k not in scoped)
    verdict = ("MECHANICALLY_REFUSED" if failed else
               "INCONCLUSIVE" if undecided else "MECHANICALLY_ACCEPTED")
    return {"schema": "df_d3_acceptance.v2", "generated_utc": now(),
            "design_sha256": design.D3_DESIGN_CURRENT["design_sha256"],
            "kind": spec["kind"], "spec_sha256": contract.spec_sha256(spec),
            "samples": len(x["values"]), "results": results,
            "required_tests": list(TESTS), "failed": failed, "scoped": scoped,
            "undecided": undecided,
            "coverage": coverage,
            "review_ready": verdict == "MECHANICALLY_ACCEPTED",
            "verdict": verdict,
            "note": ("Mechanical acceptance only: what the operator declares about itself is "
                     "true. It says nothing about utility and selects nothing. A scoped test is "
                     "reported as scoped; an undecided one makes the verdict INCONCLUSIVE.")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-tests", action="store_true")
    args = parser.parse_args(argv)
    if args.list_tests:
        print(json.dumps({"schema": "df_d3_acceptance.v2", "tests": list(TESTS),
                          "scopeable": list(SCOPEABLE),
                          "design_sha256": design.D3_DESIGN_CURRENT["design_sha256"]}, indent=1))
        return 0
    parser.error("this module is a battery; import it and call run_battery(operator, x)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
