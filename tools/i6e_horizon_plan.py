"""Generate the I6-E multihorizon design without fitting or reading market data.

The native clock is hourly. ``horizon_hours`` is always physical future time;
``forecast_steps`` is only its expression on a particular sampled clock.
Equivalent hourly/coarse requests share one arm. This planner is not a worker
or evidence that the generalized forecasting executor has been implemented.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

HORIZONS = (2, 3, 4, 5, 6, 24, 48, 72, 96, 120)
STEPS = (1, 2, 3, 4)
UTC = timezone.utc


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def experiment_grid():
    """Return unique executable specifications and all requested comparisons.

    Every arm covers 2*h physical hours including the origin. No fractional
    timestamps, interpolation, smoothing, or duplicate fits are introduced.
    """
    unique = {}
    requests = []
    for horizon in HORIZONS:
        spacings = [("DOWNSAMPLED", step, horizon // step)
                    for step in STEPS if horizon % step == 0]
        spacings.append(("HOURLY", horizon, 1))
        for mode, steps, spacing in spacings:
            arm_id = f"h{horizon:03d}_d{spacing:03d}"
            unique.setdefault(arm_id, dict(
                arm_id=arm_id, horizon_hours=horizon, spacing_hours=spacing,
                forecast_steps=steps, samples=2 * horizon // spacing + 1,
                context_hours=2 * horizon,
                stage="HOURLY" if spacing == 1 else "DOWNSAMPLING",
                naive_label=f"naive para el horizonte de {horizon} horas",
                naive_horizon_hours=horizon, metric_unit="USD_per_EUR",
                diagnostic_naive_horizons_hours=list(range(spacing, horizon + 1, spacing))
                if mode == "DOWNSAMPLED" else [horizon],
            ))
            requests.append(dict(mode=mode, requested_steps=steps,
                                 horizon_hours=horizon, arm_id=arm_id))
    arms = sorted(unique.values(), key=lambda a: (
        0 if a["stage"] == "DOWNSAMPLING" else 1,
        a["samples"], a["horizon_hours"], a["spacing_hours"]))
    return arms, requests


def weekly_schedule(year):
    """Cover a complete calendar year with Monday refits and partial edge weeks.

    The fit cutoff is the Monday of each scoring week; the fit interval starts
    four calendar years earlier. This returns boundaries, not available rows.
    """
    if type(year) is not int or not 1904 <= year <= 2099:
        raise ValueError("unsupported calendar year")
    start = datetime(year, 1, 1, tzinfo=UTC)
    end = datetime(year + 1, 1, 1, tzinfo=UTC)
    cutoff = start - timedelta(days=start.weekday())
    weeks = []
    while cutoff < end:
        try:
            fit_start = cutoff.replace(year=cutoff.year - 4)
        except ValueError:
            fit_start = cutoff.replace(year=cutoff.year - 4, day=28)
        iso = lambda time: time.isoformat().replace("+00:00", "Z")
        weeks.append(dict(
            ordinal=len(weeks), fit_start=iso(fit_start), fit_cutoff=iso(cutoff),
            score_start=iso(max(start, cutoff)),
            score_end=iso(min(end, cutoff + timedelta(days=7))),
        ))
        cutoff += timedelta(days=7)
    return weeks


def compare_prices(current, future, predicted, *, horizon_hours, naive_horizon_hours):
    """Compare price predictions ONLY to the naive for that physical horizon.

    Callers must first verify source/row identities and t+h timestamps. Inputs
    are original positive prices, not scaled labels or logarithmic returns.
    The result uses row-weighted MAE/MSE; it never averages weekly means.
    """
    if (type(horizon_hours) is not int or horizon_hours <= 0
            or type(naive_horizon_hours) is not int or naive_horizon_hours != horizon_hours):
        raise ValueError("model and naive horizon must be the same positive physical hours")
    arrays = [tuple(float(value) for value in values)
              for values in (current, future, predicted)]
    if (not arrays[0] or any(len(a) != len(arrays[0]) for a in arrays)
            or any(not math.isfinite(v) or v <= 0 for a in arrays for v in a)):
        raise ValueError("same-row nonempty positive finite original prices required")
    error = [p - y for p, y in zip(arrays[2], arrays[1])]
    baseline = [p - y for p, y in zip(arrays[0], arrays[1])]
    n = len(error)
    mae = math.fsum(abs(e) for e in error) / n
    naive = math.fsum(abs(e) for e in baseline) / n
    return dict(
        n=n, horizon_hours=horizon_hours, naive_horizon_hours=naive_horizon_hours,
        naive_label=f"naive para el horizonte de {horizon_hours} horas",
        mae=mae, mse=math.fsum(e * e for e in error) / n,
        naive_mae=naive, naive_mse=math.fsum(e * e for e in baseline) / n,
        skill_mae=None if naive == 0 else 1 - mae / naive, unit="USD_per_EUR",
    )


def build_plan():
    """Emit a deterministic design, explicitly distinct from a deployed campaign."""
    arms, requests = experiment_grid()
    validation = weekly_schedule(2024)
    plan = dict(
        schema="predictor.i6e.multihorizon.plan.v1",
        status="PLANNED_NOT_EXECUTED", execution_ready=False, test_read=False,
        horizons_hours=list(HORIZONS), seed=6102026, repetitions=1,
        arms=arms, comparison_requests=requests,
        stage_order=["DOWNSAMPLING", "HOURLY"],
        unique_arms=len(arms), downsampled_arms=28, hourly_arms=10,
        validation_weeks=validation, validation_cell_dispositions_expected=len(arms) * len(validation),
        measured_scored_rows=None, test_weeks_reserved=weekly_schedule(2025),
        model=dict(
            regime="R0", features=20, branch_channels=16, branch_kernel=3,
            positional_encoding="physical_hours_immediately_after_fusion",
            projection_channels=32, transformer_blocks=2, heads=2, ff_dim=64,
            core_channels=[24, 16, 8], core_strides=[1, 1, 1],
            scalar_head="last_temporal_position", expected_parameters=34921,
            parameter_count_status="VERIFIED_COMPACT_ONLY_GENERALIZATION_PENDING",
        ),
        training=dict(train_calendar_years=4, refit="weekly", inner_weeks=4,
                      max_epochs=50, patience=5, batch_size=256,
                      learning_rate=.001, monitor="train_validation_mean"),
        automation=dict(
            transition="automatic_after_all_stage_compute_dispositions",
            warehouse_delivery="durable_idempotent_outbox_with_exact_readback",
            prior_terminal_reuse="exact_full_contract_and_population_match_only",
            peak_memory_margin=1.25, preferred_gpu_role="preferred_worker",
            preferred_worker_gpu_jobs=1, use_secondary_preferred_worker_gpu=False,
        ),
        pending_implementation=[
            "Generalize the model adapter for 7 and 2h+1 hourly samples without editing sealed I6E code.",
            "Authenticate per-horizon input manifests, price endpoints and origin populations.",
            "Implement and test the leased workers, stage follower, OLAP outbox and measured-cost status.",
            "Measure full child/supervisor memory and actual parameter counts before production placement.",
        ],
    )
    plan["plan_sha256"] = hashlib.sha256(_canonical(plan).encode()).hexdigest()
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = build_plan()
    if args.command == "check":
        if args.output is None or json.loads(args.output.read_text()) != plan:
            raise ValueError("saved plan differs from the governing generated design")
        print(json.dumps(dict(status="PLAN_VERIFIED_NOT_EXECUTED", plan_sha256=plan["plan_sha256"])))
        return
    body = json.dumps(plan, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            if json.loads(args.output.read_text()) != plan:
                raise ValueError("refusing to overwrite a different plan; use a successor path")
        else:
            with args.output.open("x") as handle:
                handle.write(body)
        print(json.dumps(dict(status=plan["status"], output=str(args.output),
                              arms=plan["unique_arms"], plan_sha256=plan["plan_sha256"])))
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
