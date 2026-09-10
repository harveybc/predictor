"""C22: exactly one durable terminal per run, whatever happens.

A run can end five ways — it completes, it fails, it is
inconclusive, it is refused by a gate, or it is quarantined. The
previous emitter only fired on the success path, after the
pipeline returned, so an exception, a refusal or an inconclusive
outcome left the cube with no record that the run had ever
happened.

`terminal_run()` wraps the whole run instead. Exactly one envelope
is emitted on every exit path, carrying the outcome, and for a
failure the PHASE and TYPE of the failure — without dressing a
crash up as a scientific result. If the outbox itself cannot be
written, that is recorded as a typed operational gap next to the
results rather than a print that scrolls away.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from olap import outbox as ob
from olap.campaign_envelope import UNAVAILABLE, build_envelope

COMPLETE = "COMPLETE"
FAILED = "FAILED"
INCONCLUSIVE = "INCONCLUSIVE"
REFUSED = "REFUSED"
QUARANTINED = "QUARANTINED"
TERMINAL_STATES = (COMPLETE, FAILED, INCONCLUSIVE, REFUSED,
                   QUARANTINED)

# A refusal is a gate saying no. It is an operational verdict
# about permission, never a scientific result, so it is stored
# NON_GOVERNING like a quarantine.
RESULT_CLASS_FOR = {
    COMPLETE: "DEVELOPMENT",
    FAILED: "NON_GOVERNING",
    INCONCLUSIVE: "DEVELOPMENT",
    REFUSED: "NON_GOVERNING",
    QUARANTINED: "NON_GOVERNING",
}

OPERATIONAL_GAP_NAME = "OLAP_OUTBOX_OPERATIONAL_GAP.json"


class TerminalEmissionGap(Exception):
    """The outbox itself failed. Never silent."""


def build_terminal_envelope(*, campaign_key: str, producer: str,
                            state: str, stamp: dict,
                            config: dict,
                            failure_phase: str = UNAVAILABLE,
                            failure_type: str = UNAVAILABLE,
                            wall_seconds=UNAVAILABLE) -> dict:
    if state not in TERMINAL_STATES:
        raise ValueError(f"unknown terminal state {state!r}")
    stamp = stamp or {}
    digests = stamp.get("digests", {}) or {}
    return build_envelope(
        campaign_key=campaign_key,
        producer=producer,
        result_class=RESULT_CLASS_FOR[state],
        identity={
            "run_id": str(config.get("output_file",
                                     UNAVAILABLE)),
            "code_identity": str(digests.get("code",
                                             UNAVAILABLE)),
            "design_sha256": str(stamp.get("manifest_sha256",
                                           UNAVAILABLE)),
        },
        data_consumed={
            "datasets": [],
            "variables": [
                {"id": s,
                 "digest": str(digests.get("data", UNAVAILABLE)),
                 "eligibility_state": str(stamp.get(
                     "eligibility_status", UNAVAILABLE))}
                for s in stamp.get("subject_ids", [])],
            "operators": []},
        partitions={
            "exposure": str(stamp.get("execution_purpose",
                                      UNAVAILABLE)),
            "splits": str(digests.get("partitions",
                                      UNAVAILABLE))},
        budget={"device": "cpu",
                "wall_seconds": wall_seconds,
                "cost_units": ("wall_seconds"
                               if wall_seconds != UNAVAILABLE
                               else UNAVAILABLE)},
        terminal={
            "state": state,
            "adjudication": str(stamp.get(
                "eligibility_status", UNAVAILABLE)),
            # the failure is described, never promoted into a
            # scientific finding
            "failure_phase": failure_phase,
            "failure_type": failure_type},
        artifacts={
            "results_file": str(
                config.get("results_file", UNAVAILABLE)),
            # C23: born at the producer's terminal point, not
            # translated from a summary after the fact
            "verification": "BORN_AT_PRODUCER_TERMINAL"},
        # C23: a terminal run IS a unit. Emitting a campaign
        # summary with units=[] was exactly the shape the order
        # rejected, so the run appears with its own identity and
        # its measured cost.
        units=[{
            "cell_key": str(stamp.get("dataset_id",
                                      UNAVAILABLE)),
            "candidate_key": str(config.get(
                "plugin", UNAVAILABLE)),
            "metric_name": "run_wall_seconds",
            "metric_value": wall_seconds,
            "uncertainty_kind": UNAVAILABLE,
            "uncertainty_low": UNAVAILABLE,
            "uncertainty_high": UNAVAILABLE,
            "terminal_state": state,
            "epoch_count": (int(config["epochs"])
                            if isinstance(config.get("epochs"),
                                          int)
                            else UNAVAILABLE),
            "checkpoint_count": UNAVAILABLE,
        }])


def emit_terminal(envelope: dict, *, results_dir: Path | None
                  ) -> dict:
    """Emit, or record a typed operational gap beside the
    results."""
    try:
        return ob.emit(envelope, kind="envelope")
    except Exception as exc:                    # noqa: BLE001
        gap = {
            "schema": "crispdm.olap_outbox_operational_gap.v1",
            "campaign_key": envelope.get("campaign_key"),
            "envelope_sha256": envelope.get("envelope_sha256"),
            "terminal_state":
                envelope.get("terminal", {}).get("state"),
            "error_type": exc.__class__.__name__,
            "consequence": "this run produced results but the "
                           "cube did not receive its terminal; "
                           "the gap is recorded here so the "
                           "absence is visible instead of "
                           "silent",
        }
        if results_dir:
            try:
                d = Path(results_dir)
                d.mkdir(parents=True, exist_ok=True)
                (d / OPERATIONAL_GAP_NAME).write_text(
                    json.dumps(gap, indent=1, sort_keys=True)
                    + "\n")
            except Exception:                   # noqa: BLE001
                pass
        return {"outbox_entry": UNAVAILABLE,
                "state": "OPERATIONAL_GAP", "gap": gap}


def terminal_run(fn, *, campaign_key, producer: str,
                 config, results_dir: Path | None = None):
    """Run `fn` and emit EXACTLY ONE terminal, whatever happens.

    `config` may be a dict OR a zero-argument callable returning
    one. The callable form exists because the run only builds its
    configuration part-way through: capturing the dict up front
    would freeze an empty object and report UNAVAILABLE for
    everything the run actually did.

    `fn` may return the string INCONCLUSIVE (or a dict carrying
    `terminal_state`) to declare an outcome that is neither a
    success nor a failure.
    """
    started = time.monotonic()
    state, phase, ftype = COMPLETE, UNAVAILABLE, UNAVAILABLE
    result = None
    try:
        result = fn()
        if isinstance(result, dict) and result.get(
                "terminal_state") in TERMINAL_STATES:
            state = result["terminal_state"]
            phase = result.get("failure_phase", UNAVAILABLE)
            ftype = result.get("failure_type", UNAVAILABLE)
        elif result == INCONCLUSIVE:
            state = INCONCLUSIVE
    except SystemExit as exc:
        # a gate refusal is a SystemExit carrying REFUSED:
        state = (REFUSED if "REFUSED" in str(exc) else FAILED)
        phase = "run"
        ftype = "SystemExit"
        _finish(state, phase, ftype, campaign_key, producer,
                config, results_dir, started)
        raise
    except BaseException as exc:                # noqa: BLE001
        state, phase = FAILED, "run"
        ftype = exc.__class__.__name__
        tb = traceback.extract_tb(exc.__traceback__)
        if tb:
            phase = f"{Path(tb[-1].filename).name}:" \
                    f"{tb[-1].lineno}"
        _finish(state, phase, ftype, campaign_key, producer,
                config, results_dir, started)
        raise
    _finish(state, phase, ftype, campaign_key, producer, config,
            results_dir, started)
    return result


def _finish(state, phase, ftype, campaign_key, producer, config,
            results_dir, started):
    # both the key and the config are read at FINISH time: the run
    # only knows which experiment it is once its configuration is
    # merged
    campaign_key = (campaign_key() if callable(campaign_key)
                    else campaign_key)
    config = config() if callable(config) else config
    config = config if isinstance(config, dict) else {}
    env = build_terminal_envelope(
        campaign_key=campaign_key, producer=producer,
        state=state, stamp=config.get("eligibility_stamp") or {},
        config=config, failure_phase=phase, failure_type=ftype,
        wall_seconds=round(time.monotonic() - started, 6))
    out = emit_terminal(env, results_dir=results_dir)
    config["olap_terminal"] = {
        "state": state, "outbox": out.get("state"),
        "entry": out.get("outbox_entry")}
    print(f"olap terminal: {state} -> {out.get('state')} "
          f"{out.get('outbox_entry')}")
    return out
