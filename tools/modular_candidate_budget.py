"""Candidate budget model for the modular DOIN search (owner addendum 256c61a6).

Every candidate is priced on explicit dimensions BEFORE it can be queued:

  raw_channels            distinct input columns the candidate reads
  expanded_channels       channels after source transforms (sum of branch inputs)
  branches                number of branch encoders
  fused_width             sum of branch output widths (not the number of files)
  fused_time              time steps kept through fusion (the full window)
  materialization_bytes   float32 fused representation over the declared populations
  parameters              trainable model size (exact, from the built graph when available)
  host_ram_bytes          whole-cgroup peak, from measured calibration points
  vram_bytes              TF device peak, from measured calibration points
  gpu_seconds_per_candidate  max_epochs x updates/epoch x measured s/update
  weekly_gpu_seconds      the per-candidate figure x candidates per week declared

Each value carries its SOURCE: "exact" (arithmetic on the config), "built"
(Keras graph), or a named measured calibration point (profile/pilot receipt)
with the interpolation used. A dimension without a measurement is reported as
UNMEASURED and is never silently treated as fitting.

A candidate over ANY declared cap becomes a NAMED DEFERRED row listing every
overflowing dimension. It is never truncated, never has inputs dropped and never
has time collapsed to fit: the configuration is kept verbatim and re-evaluated
when a cap or a calibration measurement changes.

Dimensional bottlenecks (fusion width, latent compression) reduce MODEL cost;
they do not reduce the cost of generating, profiling or materializing features
upstream of the branches. Feature-generation cost is budgeted by its own lane.
"""
from __future__ import annotations

import math

DIMENSIONS = ("raw_channels", "expanded_channels", "branches", "fused_width", "fused_time",
              "materialization_bytes", "parameters", "host_ram_bytes", "vram_bytes",
              "gpu_seconds_per_candidate", "weekly_gpu_seconds")
UNMEASURED = "UNMEASURED"


def _interpolate(points, x, key):
    """Piecewise-linear in fused_width over measured points; a single point only prices its own width."""
    pts = sorted((p["fused_width"], p[key], p["source"]) for p in points if p.get(key) is not None)
    if not pts:
        return None, UNMEASURED
    exact = [p for p in pts if p[0] == x]
    if exact:
        return max(p[1] for p in exact), exact[0][2]
    if len(pts) == 1:
        return None, f"{UNMEASURED} (one measured point at width {pts[0][0]}: {pts[0][2]})"
    if x < pts[0][0]:
        a, b, kind = pts[0], pts[1], "extrapolated"
    elif x > pts[-1][0]:
        a, b, kind = pts[-2], pts[-1], "extrapolated"
    else:
        a, b = next((pts[i], pts[i + 1]) for i in range(len(pts) - 1) if pts[i][0] <= x <= pts[i + 1][0])
        kind = "interpolated"
    if a[0] == b[0]:
        return max(a[1], b[1]), f"{kind} at width {a[0]}"
    value = a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0])
    return value, f"{kind} in fused_width between {a[2]} (w={a[0]}) and {b[2]} (w={b[0]})"


def price(nested, budget, *, parameters=None):
    """Value and source for every dimension of one nested modular candidate."""
    model, ev = nested["model"], nested["evaluator"]
    pops = budget["populations"]
    branches = model["branches"]
    widths = [b["params"].get("channels", 16) for b in branches]
    fused_width = int(sum(widths))
    raw = len({f for b in branches for f in b["features"]})
    expanded = int(sum(len(b["features"]) for b in branches))
    windows = pops["train_windows"] + pops["validation_windows"]
    values = {
        "raw_channels": (raw, "exact"),
        "expanded_channels": (expanded, "exact"),
        "branches": (len(branches), "exact"),
        "fused_width": (fused_width, "exact: sum of branch output widths"),
        "fused_time": (model["window"], "exact: branches preserve the window"),
        "materialization_bytes": (windows * model["window"] * fused_width * 4,
                                  f"exact: ({pops['train_windows']}+{pops['validation_windows']}) windows x "
                                  f"{model['window']} x {fused_width} x 4 B"),
        "parameters": (parameters, "built") if parameters is not None else (None, UNMEASURED + " (not built)"),
    }
    points = budget.get("calibration", [])
    for dim, key in (("host_ram_bytes", "host_peak_bytes"), ("vram_bytes", "device_peak_bytes")):
        values[dim] = _interpolate(points, fused_width, key)
    per_update, source = _interpolate(points, fused_width, "seconds_per_update")
    if per_update is None:
        values["gpu_seconds_per_candidate"] = (None, source)
        values["weekly_gpu_seconds"] = (None, source)
    else:
        updates = ev["max_epochs"] * math.ceil(pops["train_windows"] / ev["batch_size"])
        per = updates * per_update
        values["gpu_seconds_per_candidate"] = (per, f"upper bound {updates} updates x s/update {source}")
        weekly = budget.get("candidates_per_week")
        values["weekly_gpu_seconds"] = ((per * weekly, f"{weekly} candidates/week x per-candidate")
                                        if weekly else (None, UNMEASURED + " (candidates_per_week undeclared)"))
    return {dim: {"value": values[dim][0], "source": values[dim][1]} for dim in DIMENSIONS}


def check(nested, budget, *, parameters=None):
    """Return (priced, overflow) where overflow names every dimension over its cap.

    An UNMEASURED dimension that has a cap is reported as overflowing only when
    ``budget['unmeasured_policy'] == 'defer'`` (default); it is never assumed to fit.
    """
    priced = price(nested, budget, parameters=parameters)
    caps = budget.get("caps", {})
    policy = budget.get("unmeasured_policy", "defer")
    overflow = []
    for dim, cap in caps.items():
        if dim not in DIMENSIONS:
            raise ValueError(f"unknown budget dimension {dim}")
        value = priced[dim]["value"]
        if value is None:
            if policy == "defer":
                overflow.append(f"{dim}=UNMEASURED")
        elif value > cap:
            overflow.append(f"{dim}={value:.6g}>{cap:.6g}")
    return priced, overflow
