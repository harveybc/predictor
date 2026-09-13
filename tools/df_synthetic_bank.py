#!/usr/bin/env python3
"""C128 versioned known-truth synthetic signal bank.

Self-generated, seeded, numpy only, no downloads, no targets, no
models. Every unit carries the clean signal, the realized additive
perturbation, the observation (clean + noise, then missingness as NaN),
the missing mask, the metric support, the events placed by construction
and the exact provenance needed to rebuild all of it bit-identically:

    unit/
      clean_signal.npy     (V, N) float64, never masked
      additive_noise.npy   (V, N) float64, never masked
      observed_signal.npy  (V, N) float64, NaN where missing
      missing_mask.npy     (V, N) bool, True where missing
      metric_support.npy   (V, N) bool, == ~missing_mask
      events.json          every impulse/step/bump/motif/regime
                           boundary/trend knot, with index and type
      UNIT.json            params, derived seeds, partitions, declared
                           and realized SNR, digests, generator identity

Array names follow agent-multi tools/t1_known_truth_bank.py; this bank
adds missing_mask.npy and events.json.

Order of operations per unit (fixed, auditable):
  1. partitions [start, end) from the length alone (60/20/20);
  2. clean from seed_clean = H(seed, "clean", family, length, V);
  3. unit-variance base noise from seed_noise = H(seed, "noise",
     perturbation, params, length, V, noise_seed_tag);
  4. per-variable noise scale from the TRAIN partition's clean
     variance and the declared SNR only;
  5. observed = clean + noise;
  6. missing mask from seed_missing = H(seed, "missing", ...), applied
     to observed only.

CLI: python tools/df_synthetic_bank.py --out DIR [--limit K]
DIR must not exist (write-once)."""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

GENERATOR_VERSION = "df_synthetic_bank.c128.v1"
UNIT_SCHEMA = "predictor.df_synthetic_unit.v1"
MANIFEST_SCHEMA = "predictor.df_synthetic_bank_manifest.v1"
MODULE_RELPATH = "tools/df_synthetic_bank.py"

SNR_GRID_DB = ("inf", 20, 10, 5, 0, -5)
LENGTHS = (512, 2048, 8192)
PRIMARY_LENGTH = 2048
SEEDS = (11, 12, 13)
PARTITION_FRACTIONS = (("train", 0.6), ("calibration", 0.2),
                       ("confirmation", 0.2))
MULTIVARIATE_V = 3

CLEAN_FAMILIES = (
    "sinusoid", "multiband", "trend_linear", "trend_piecewise",
    "seasonal", "chirp", "impulses", "steps", "bumps", "motif",
    "regime_mean", "regime_variance", "regime_frequency",
    "multivariate", "null")

PERTURBATIONS = {
    "white": {},
    "ar1": {"phi": 0.8},
    "pink": {"exponent": 1.0},
    "contaminated": {"epsilon": 0.05, "kappa": 5.0},
    "student_t": {"df": 3.0},
    "correlated": {"covariance": [[1.0, 0.6, 0.3],
                                  [0.6, 1.0, 0.6],
                                  [0.3, 0.6, 1.0]]},
    "heteroscedastic": {"envelope": "linear", "start": 1.0,
                        "end": 2.5},
    "null": {},
}

MISSINGNESS = {
    "none": {"kind": "none"},
    "mcar": {"kind": "mcar", "rate": 0.10},
    "blocks": {"kind": "blocks",
               # [start as fraction of N, length in samples]; one block
               # per partition at N=2048.
               "blocks": [[0.2, 16], [0.7, 64], [0.9, 32]]},
}

MOTIF_LENGTH = 32
_u = np.arange(MOTIF_LENGTH) / MOTIF_LENGTH
MOTIF_TEMPLATE = np.hanning(MOTIF_LENGTH) * np.sin(2 * np.pi * 2 * _u)
MOTIF_TEMPLATE.setflags(write=False)
del _u

ARRAY_NAMES = ("clean_signal", "additive_noise", "observed_signal",
               "missing_mask", "metric_support")


# ----------------------------- identity -----------------------------
def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def generator_identity() -> dict:
    return {"version": GENERATOR_VERSION, "module": MODULE_RELPATH,
            "code_sha256": code_sha256(), "numpy": np.__version__}


def array_digest(a: np.ndarray) -> str:
    """sha256 over {dtype, shape} header + raw C-order bytes."""
    h = hashlib.sha256()
    h.update(json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)},
                        sort_keys=True).encode())
    h.update(b"\n")
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def derive_seed(*parts) -> int:
    blob = json.dumps({"generator": GENERATOR_VERSION,
                       "parts": list(parts)}, sort_keys=True)
    return int(hashlib.sha256(blob.encode()).hexdigest()[:16], 16)


def derived_seeds(cell: dict, seed: int) -> dict:
    n, v = cell["length"], cell["n_variables"]
    return {
        "clean": derive_seed(seed, "clean", cell["family"], n, v),
        "noise": derive_seed(seed, "noise", cell["perturbation"],
                             cell["perturbation_params"], n, v,
                             cell.get("noise_seed_tag", 0)),
        "missing": derive_seed(seed, "missing", cell["missingness"],
                               n, v),
    }


def _jnum(x) -> object:
    """JSON-safe number: non-finite floats become strings."""
    x = float(x)
    if np.isnan(x):
        return "nan"
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


# ----------------------------- partitions ---------------------------
def partitions(n: int) -> dict:
    """Chronological 60/20/20 [start, end), from the length alone."""
    a, b = n * 6 // 10, n * 8 // 10
    return {"train": [0, a], "calibration": [a, b],
            "confirmation": [b, n]}


# ----------------------------- clean families -----------------------
def _strata(rng, n: int, m: int, margin_lo: int, margin_hi: int
            ) -> list:
    """One index per equal stratum, inside [lo+margin_lo, hi-margin_hi).
    Guarantees ordering, non-overlap and presence in every stratum."""
    edges = np.linspace(0, n, m + 1).astype(int)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a, b = lo + margin_lo, hi - margin_hi
        if b <= a:
            raise ValueError("REFUSED: strata too short for events")
        out.append(int(rng.integers(a, b)))
    return out


def _sin(t, amp, period, phase):
    return amp * np.sin(2 * np.pi * t / period + phase)


def _clean_univariate(family: str, n: int, rng, j: int):
    t = np.arange(n, dtype=np.float64)
    ev = []
    if family == "null":
        return np.zeros(n), {}, ev
    if family == "sinusoid":
        p = {"amplitude": float(rng.uniform(0.5, 2.0)),
             "period": float(rng.uniform(16, 128)),
             "phase": float(rng.uniform(0, 2 * np.pi))}
        return _sin(t, p["amplitude"], p["period"], p["phase"]), p, ev
    if family == "multiband":
        k = int(rng.integers(2, 5))
        periods = np.sort(np.exp(rng.uniform(np.log(8), np.log(256),
                                             k)))
        comps = [{"amplitude": float(rng.uniform(0.3, 1.5)),
                  "period": float(pr),
                  "phase": float(rng.uniform(0, 2 * np.pi))}
                 for pr in periods]
        x = np.zeros(n)
        for c in comps:
            x += _sin(t, c["amplitude"], c["period"], c["phase"])
        return x, {"n_components": k, "components": comps}, ev
    if family == "trend_linear":
        p = {"intercept": float(rng.uniform(-1, 1)),
             "slope_per_sample": float(rng.uniform(-3, 3) / n)}
        return p["intercept"] + p["slope_per_sample"] * t, p, ev
    if family == "trend_piecewise":
        knots = _strata(rng, n, 3, n // 20, n // 20)
        slopes = [float(rng.uniform(-3, 3) / n) for _ in range(4)]
        intercept = float(rng.uniform(-1, 1))
        slope_t = np.full(n, slopes[0])
        for k, idx in enumerate(knots):
            slope_t[idx:] = slopes[k + 1]
            ev.append({"variable": j, "type": "trend_knot",
                       "index": idx, "slope_before": slopes[k],
                       "slope_after": slopes[k + 1]})
        x = intercept + np.concatenate(([0.0], np.cumsum(slope_t)[:-1]))
        return x, {"intercept": intercept, "knots": knots,
                   "slopes_per_sample": slopes}, ev
    if family == "seasonal":
        period = int(rng.choice([24, 48, 96]))
        nh = 3
        harm = [{"harmonic": h,
                 "amplitude": float(rng.uniform(0.5, 1.5) / h),
                 "phase": float(rng.uniform(0, 2 * np.pi))}
                for h in range(1, nh + 1)]
        x = np.zeros(n)
        for c in harm:
            x += _sin(t, c["amplitude"], period / c["harmonic"],
                      c["phase"])
        return x, {"period": period, "n_harmonics": nh,
                   "harmonics": harm}, ev
    if family == "chirp":
        p = {"amplitude": float(rng.uniform(0.5, 2.0)),
             "f0_cycles_per_sample": float(rng.uniform(1 / 256, 1 / 128)),
             "f1_cycles_per_sample": float(rng.uniform(1 / 16, 1 / 8)),
             "phase": float(rng.uniform(0, 2 * np.pi))}
        ph = 2 * np.pi * (p["f0_cycles_per_sample"] * t
                          + (p["f1_cycles_per_sample"]
                             - p["f0_cycles_per_sample"])
                          * t ** 2 / (2 * (n - 1))) + p["phase"]
        return p["amplitude"] * np.sin(ph), p, ev
    if family == "impulses":
        m = max(4, n // 256)
        idx = _strata(rng, n, m, 1, 1)
        hts = [float(rng.uniform(2, 5) * rng.choice([-1, 1]))
               for _ in idx]
        x = np.zeros(n)
        for i, h in zip(idx, hts):
            x[i] = h
            ev.append({"variable": j, "type": "impulse", "index": i,
                       "height": h})
        return x, {"indices": idx, "heights": hts}, ev
    if family == "steps":
        m = max(3, n // 512)
        idx = _strata(rng, n, m, 1, 1)
        levels = [float(rng.uniform(-1, 1))]
        levels += [float(rng.uniform(-3, 3)) for _ in idx]
        x = np.full(n, levels[0])
        for k, i in enumerate(idx):
            x[i:] = levels[k + 1]
            ev.append({"variable": j, "type": "step", "index": i,
                       "level_before": levels[k],
                       "level_after": levels[k + 1]})
        return x, {"indices": idx, "levels": levels}, ev
    if family == "bumps":
        m = max(3, n // 400)
        widths = [float(rng.uniform(2, 8)) for _ in range(m)]
        halfw = int(np.ceil(4 * max(widths)))
        centers = _strata(rng, n, m, halfw + 1, halfw + 1)
        hts = [float(rng.uniform(1, 4)) for _ in range(m)]
        x = np.zeros(n)
        for c, w, h in zip(centers, widths, hts):
            hw = int(np.ceil(4 * w))
            s = slice(c - hw, c + hw + 1)
            x[s] = h * np.exp(-0.5 * ((t[s] - c) / w) ** 2)
            ev.append({"variable": j, "type": "bump", "index": c,
                       "width_sd": w, "height": h,
                       "support_halfwidth": hw})
        return x, {"centers": centers, "widths_sd": widths,
                   "heights": hts, "support": "compact |t-c|<=ceil(4w)"
                   }, ev
    if family == "motif":
        m = max(3, n // 512)
        pos = _strata(rng, n, m, 0, MOTIF_LENGTH)
        gains = [float(rng.uniform(1, 2)) for _ in pos]
        x = np.zeros(n)
        for p0, g in zip(pos, gains):
            x[p0:p0 + MOTIF_LENGTH] = g * MOTIF_TEMPLATE
            ev.append({"variable": j, "type": "motif", "index": p0,
                       "length": MOTIF_LENGTH, "gain": g,
                       "template_digest": array_digest(MOTIF_TEMPLATE)})
        return x, {"positions": pos, "gains": gains,
                   "template": "hanning(32)*sin(2*pi*2*arange(32)/32)",
                   "template_digest": array_digest(MOTIF_TEMPLATE)}, ev
    if family in ("regime_mean", "regime_variance", "regime_frequency"):
        bnd = _strata(rng, n, 3, n // 20, n // 20)
        amp = float(rng.uniform(0.5, 1.5))
        period = float(rng.uniform(16, 64))
        phase = float(rng.uniform(0, 2 * np.pi))
        if family == "regime_mean":
            vals = [float(rng.uniform(-3, 3)) for _ in range(4)]
            level = np.full(n, vals[0])
            for k, i in enumerate(bnd):
                level[i:] = vals[k + 1]
            x = _sin(t, amp, period, phase) + level
            key = "mean"
        elif family == "regime_variance":
            vals = [float(rng.uniform(0.3, 3.0)) for _ in range(4)]
            a_t = np.full(n, vals[0])
            for k, i in enumerate(bnd):
                a_t[i:] = vals[k + 1]
            x = a_t * np.sin(2 * np.pi * t / period + phase)
            key = "amplitude"
        else:
            vals = [float(rng.uniform(8, 96)) for _ in range(4)]
            per_t = np.full(n, vals[0])
            for k, i in enumerate(bnd):
                per_t[i:] = vals[k + 1]
            ph = 2 * np.pi * np.concatenate(
                ([0.0], np.cumsum(1.0 / per_t)[:-1])) + phase
            x = amp * np.sin(ph)
            key = "period"
        for k, i in enumerate(bnd):
            ev.append({"variable": j, "type": "regime_boundary",
                       "regime": family.split("_", 1)[1], "index": i,
                       f"{key}_before": vals[k],
                       f"{key}_after": vals[k + 1]})
        p = {"boundaries": bnd, f"{key}_per_regime": vals,
             "phase": phase}
        if family != "regime_variance":
            p["amplitude"] = amp
        if family != "regime_frequency":
            p["period"] = period
        return x, p, ev
    raise ValueError(f"REFUSED: unknown family {family!r}")


def clean_signal(family: str, n: int, v: int, seed_clean: int):
    """Returns (clean (V, N), params, events)."""
    rng = np.random.default_rng(seed_clean)
    if family == "multivariate":
        if v < MULTIVARIATE_V:
            raise ValueError("REFUSED: multivariate needs V >= 3")
        t = np.arange(n, dtype=np.float64)
        latent_c = [{"amplitude": float(rng.uniform(0.5, 1.5)),
                     "period": float(rng.uniform(64, 256)),
                     "phase": float(rng.uniform(0, 2 * np.pi))}
                    for _ in range(2)]
        latent = np.zeros(n)
        for c in latent_c:
            latent += _sin(t, c["amplitude"], c["period"], c["phase"])
        loadings = [float(rng.uniform(0.5, 1.5) * rng.choice([-1, 1]))
                    for _ in range(v)]
        private = [{"amplitude": float(rng.uniform(0.2, 0.6)),
                    "period": float(rng.uniform(8, 48)),
                    "phase": float(rng.uniform(0, 2 * np.pi))}
                   for _ in range(v)]
        x = np.empty((v, n))
        for j in range(v):
            x[j] = loadings[j] * latent + _sin(
                t, private[j]["amplitude"], private[j]["period"],
                private[j]["phase"])
        return x, {"latent_components": latent_c,
                   "loadings": loadings,
                   "private_components": private}, []
    rows, per_var, events = [], [], []
    for j in range(v):
        x, p, ev = _clean_univariate(family, n, rng, j)
        rows.append(x)
        per_var.append(p)
        events += ev
    return np.vstack(rows), {"per_variable": per_var}, events


# ----------------------------- perturbations ------------------------
def base_noise(kind: str, params: dict, n: int, v: int, parts: dict,
               seed_noise: int):
    """Unit theoretical variance per variable (before envelope
    normalization for heteroscedastic). Returns (base, extra)."""
    rng = np.random.default_rng(seed_noise)
    extra = {}
    if kind == "null":
        return np.zeros((v, n)), extra
    if kind == "white":
        return rng.standard_normal((v, n)), extra
    if kind == "ar1":
        phi = float(params["phi"])
        c = np.sqrt(1 - phi ** 2)
        e = rng.standard_normal((v, n))
        x = np.empty((v, n))
        x[:, 0] = e[:, 0]                       # stationary start
        for i in range(1, n):
            x[:, i] = phi * x[:, i - 1] + c * e[:, i]
        return x, extra
    if kind == "pink":
        w = rng.standard_normal((v, n))
        f = np.abs(np.fft.fftfreq(n))
        h = np.zeros(n)
        h[f > 0] = f[f > 0] ** (-float(params["exponent"]) / 2)
        h *= np.sqrt(n / np.sum(h ** 2))        # (1/N) sum H^2 == 1
        extra["shaping"] = ("real(ifft(fft(w)*H)), H=|f|^(-exponent/2), "
                            "H(0)=0, (1/N)sum H^2=1")
        return np.real(np.fft.ifft(np.fft.fft(w, axis=1) * h, axis=1)
                       ), extra
    if kind == "contaminated":
        eps, kap = float(params["epsilon"]), float(params["kappa"])
        z = rng.standard_normal((v, n))
        out = rng.random((v, n)) < eps
        return (z * np.where(out, kap, 1.0)
                / np.sqrt(1 - eps + eps * kap ** 2)), extra
    if kind == "student_t":
        df = float(params["df"])
        if df <= 2:
            raise ValueError("REFUSED: student_t needs df > 2")
        return rng.standard_t(df, (v, n)) * np.sqrt((df - 2) / df), extra
    if kind == "correlated":
        cov = np.asarray(params["covariance"], dtype=np.float64)
        if cov.shape != (v, v) or not np.allclose(np.diag(cov), 1.0):
            raise ValueError("REFUSED: covariance must be VxV, unit diag")
        chol = np.linalg.cholesky(cov)
        return chol @ rng.standard_normal((v, n)), extra
    if kind == "heteroscedastic":
        s0, s1 = float(params["start"]), float(params["end"])
        env = s0 + (s1 - s0) * np.arange(n) / (n - 1)
        a, b = parts["train"]
        norm = float(np.sqrt(np.mean(env[a:b] ** 2)))
        extra["envelope_train_rms_normalizer"] = norm
        extra["envelope"] = "start+(end-start)*t/(N-1), / train RMS"
        return rng.standard_normal((v, n)) * (env / norm)[None, :], extra
    raise ValueError(f"REFUSED: unknown perturbation {kind!r}")


def noise_scale(clean: np.ndarray, parts: dict, snr_db) -> tuple:
    """Per-variable scale from the TRAIN partition's clean variance and
    the declared SNR only. Returns (scale, train_clean_variance)."""
    a, b = parts["train"]
    var = clean[:, a:b].var(axis=1)
    snr = str(snr_db)
    if snr in ("inf", "nan"):
        return np.zeros(clean.shape[0]), var
    if snr == "-inf":                      # null signal: unit reference
        return np.ones(clean.shape[0]), var
    if np.any(var <= 0):
        raise ValueError("REFUSED: zero train clean variance at finite SNR")
    return np.sqrt(var / 10 ** (float(snr_db) / 10.0)), var


def missing_mask(spec: dict, n: int, v: int, seed_missing: int):
    kind = spec["kind"]
    if kind == "none":
        return np.zeros((v, n), dtype=bool), {}
    if kind == "mcar":
        rng = np.random.default_rng(seed_missing)
        return rng.random((v, n)) < float(spec["rate"]), {}
    if kind == "blocks":
        m = np.zeros((v, n), dtype=bool)
        realized = []
        for frac, ln in spec["blocks"]:
            s = int(frac * n)
            e = min(n, s + int(ln))
            m[:, s:e] = True
            realized.append([s, e])
        return m, {"blocks_realized_start_end": realized}
    raise ValueError(f"REFUSED: unknown missingness {kind!r}")


# ----------------------------- SNR ----------------------------------
def _snr_db(sv: float, nv: float):
    if sv == 0 and nv == 0:
        return "nan"
    if nv == 0:
        return "inf"
    if sv == 0:
        return "-inf"
    return float(10 * np.log10(sv / nv))


def realized_snr(clean, noise, parts, support=None) -> dict:
    out = {}
    for name, (a, b) in parts.items():
        row = []
        for j in range(clean.shape[0]):
            c, z = clean[j, a:b], noise[j, a:b]
            if support is not None:
                s = support[j, a:b]
                c, z = c[s], z[s]
            if c.size == 0:
                row.append("nan")
                continue
            row.append(_snr_db(float(c.var()), float(z.var())))
        out[name] = row
    return out


# ----------------------------- regeneration -------------------------
def _validate_cell(cell: dict) -> None:
    fam, pert, snr = cell["family"], cell["perturbation"], str(
        cell["snr_db"])
    if fam not in CLEAN_FAMILIES or pert not in PERTURBATIONS:
        raise ValueError("REFUSED: unknown family/perturbation")
    if cell["length"] < 64:
        raise ValueError("REFUSED: length too short")
    null_sig, null_noise = fam == "null", pert == "null"
    expect = ("nan" if null_sig and null_noise else "-inf" if null_sig
              else "inf" if null_noise else None)
    if expect is not None and snr != expect:
        raise ValueError(f"REFUSED: {fam}+{pert} requires snr {expect}")
    if expect is None and snr not in [str(s) for s in SNR_GRID_DB]:
        raise ValueError(f"REFUSED: snr {snr} not in declared grid")


def regenerate(unit_params: dict, seed: int) -> dict:
    """Rebuild clean, noise, observed, mask (bit-identical) plus
    events and provenance from the unit params and the seed alone."""
    cell = unit_params
    _validate_cell(cell)
    n, v = int(cell["length"]), int(cell["n_variables"])
    parts = partitions(n)                  # before any statistic
    seeds = derived_seeds(cell, seed)
    clean, cparams, events = clean_signal(cell["family"], n, v,
                                          seeds["clean"])
    base, extra = base_noise(cell["perturbation"],
                             cell["perturbation_params"], n, v, parts,
                             seeds["noise"])
    scale, train_var = noise_scale(clean, parts, cell["snr_db"])
    noise = base * scale[:, None] + 0.0    # +0.0 folds -0.0 to 0.0
    observed = clean + noise
    mask, miss_extra = missing_mask(cell["missingness"], n, v,
                                    seeds["missing"])
    observed[mask] = np.nan
    return {"clean_signal": clean, "additive_noise": noise,
            "observed_signal": observed, "missing_mask": mask,
            "metric_support": ~mask, "events": events,
            "clean_params": cparams, "partitions": parts,
            "derived_seeds": seeds, "noise_scale": scale,
            "train_clean_variance": train_var,
            "noise_extra": extra, "missing_extra": miss_extra}


def unit_id(cell: dict, seed: int) -> str:
    miss = cell["missingness"]["kind"]
    uid = (f"{cell['family']}__{cell['perturbation']}__"
           f"snr{cell['snr_db']}__{miss}__n{cell['length']}__"
           f"v{cell['n_variables']}__seed{seed}")
    if cell.get("noise_seed_tag", 0):
        uid += f"__nt{cell['noise_seed_tag']}"
    return uid


def _cell_params(cell: dict) -> dict:
    return {k: cell[k] for k in ("family", "perturbation",
                                 "perturbation_params", "snr_db",
                                 "length", "n_variables", "missingness",
                                 "noise_seed_tag")}


def build_unit_record(cell: dict, seed: int, g: dict) -> dict:
    n, v = cell["length"], cell["n_variables"]
    events_doc = {"unit_id": unit_id(cell, seed), "events": g["events"]}
    return {
        "schema": UNIT_SCHEMA,
        "unit_id": unit_id(cell, seed),
        "generator": generator_identity(),
        "unit_params": _cell_params(cell),
        "seed": seed,
        "derived_seeds": g["derived_seeds"],
        "family": cell["family"],
        "perturbation": cell["perturbation"],
        "perturbation_params": cell["perturbation_params"],
        "missingness": cell["missingness"],
        "n_variables": v, "n_samples": n,
        "variable_names": [f"v{j}" for j in range(v)],
        "partitions": g["partitions"],
        "partitions_materialized_before_any_statistic": True,
        "clean_params": g["clean_params"],
        "noise_model": {
            "base": "unit theoretical variance per variable",
            "scale_reference": "TRAIN partition clean variance (ddof=0)",
            "train_clean_variance": [_jnum(x)
                                     for x in g["train_clean_variance"]],
            "scale_per_variable": [_jnum(x) for x in g["noise_scale"]],
            **g["noise_extra"]},
        "declared_snr_db": str(cell["snr_db"]),
        "declared_snr_db_per_variable": [str(cell["snr_db"])] * v,
        "realized_snr_db": realized_snr(
            g["clean_signal"], g["additive_noise"], g["partitions"]),
        "realized_snr_db_on_support": realized_snr(
            g["clean_signal"], g["additive_noise"], g["partitions"],
            g["metric_support"]),
        "realized_snr_definition": "10*log10(var(clean)/var(noise)) "
                                   "per variable per partition",
        "missing": {"fraction_per_variable": [
            float(x) for x in g["missing_mask"].mean(axis=1)],
            "encoding": "NaN in observed_signal; missing_mask True",
            **g["missing_extra"]},
        "n_events": len(g["events"]),
        "events_sha256": hashlib.sha256(_dumps(events_doc).encode()
                                        ).hexdigest(),
        "digest_scheme": "sha256(json{dtype,shape} + '\\n' + C bytes)",
        "digests": {k: array_digest(g[k]) for k in ARRAY_NAMES},
    }


def _dumps(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, allow_nan=False)


# ----------------------------- contract facts -----------------------
def contract_fields(unit_manifest: dict) -> dict:
    """Plain facts for the common data contract (mapped elsewhere)."""
    m = unit_manifest
    miss = m["missingness"]
    policy = ("NONE" if miss["kind"] == "none" else
              f"MCAR rate={miss['rate']}" if miss["kind"] == "mcar" else
              f"BLOCKS {miss['blocks']} (start fraction, length)")
    return {
        "bank": "SYNTHETIC",
        "license_state": "NOT_APPLICABLE_GENERATED",
        "semantics": "KNOWN_BY_CONSTRUCTION",
        "unit": "1",
        "frequency": "1 sample",
        "timestamp_meaning": "SAMPLE_INDEX",
        "availability_rule": "SAMPLE_INDEX",
        "missingness_encoding": "NaN in observed_signal; missing_mask "
                                "True where missing; clean and noise "
                                "never masked",
        "missingness_policy": policy,
        "partitions": m["partitions"],
        "variable_names": list(m["variable_names"]),
        "generator": {"version": m["generator"]["version"],
                      "code_sha256": m["generator"]["code_sha256"],
                      "module": m["generator"]["module"],
                      "seed": m["seed"],
                      "derived_seeds": m["derived_seeds"]},
    }


# ----------------------------- matrix -------------------------------
def _cell(family, perturbation, snr_db, length, missingness="none",
          n_variables=None):
    if n_variables is None:
        n_variables = (MULTIVARIATE_V if family == "multivariate"
                       or perturbation == "correlated" else 1)
    return {"family": family, "perturbation": perturbation,
            "perturbation_params": PERTURBATIONS[perturbation],
            "snr_db": str(snr_db), "length": length,
            "n_variables": n_variables,
            "missingness": MISSINGNESS[missingness],
            "noise_seed_tag": 0}


def predeclared_matrix() -> list:
    """Predeclared blocks, NOT a cartesian sweep; cells deduplicated in
    order (a cell claimed by several blocks lists all of them):
    A  every non-null clean family x white x full SNR grid at N=2048
       (the null family's SNR is -inf by construction; it is covered
       by the controls, which include N=2048);
    B  every non-null perturbation x {sinusoid, multiband} x full SNR
       grid at N=2048 (correlated noise uses V=3 of the same family;
       white cells coincide with block A);
    C  controls null+white (-inf), sinusoid+null noise (inf) and
       null+null (nan) at every length {512, 2048, 8192};
    D  missingness {mcar rate 0.10, 3 declared blocks} x {sinusoid,
       steps, multivariate} x white at 10 dB, N=2048;
    every cell crossed with the seed tuple (11, 12, 13)."""
    blocks = []
    for fam in CLEAN_FAMILIES:
        if fam == "null":
            continue
        for snr in SNR_GRID_DB:
            blocks.append(("A", _cell(fam, "white", snr, PRIMARY_LENGTH)))
    for pert in PERTURBATIONS:
        if pert == "null":
            continue
        for fam in ("sinusoid", "multiband"):
            for snr in SNR_GRID_DB:
                blocks.append(("B", _cell(fam, pert, snr,
                                          PRIMARY_LENGTH)))
    for n in LENGTHS:
        blocks.append(("C", _cell("null", "white", "-inf", n)))
        blocks.append(("C", _cell("sinusoid", "null", "inf", n)))
        blocks.append(("C", _cell("null", "null", "nan", n)))
    for miss in ("mcar", "blocks"):
        for fam in ("sinusoid", "steps", "multivariate"):
            blocks.append(("D", _cell(fam, "white", 10, PRIMARY_LENGTH,
                                      miss)))
    cells, index = [], {}
    for name, c in blocks:
        key = json.dumps(c, sort_keys=True)
        if key in index:
            if name not in cells[index[key]]["blocks"]:
                cells[index[key]]["blocks"].append(name)
            continue
        index[key] = len(cells)
        cells.append({**c, "blocks": [name]})
    return cells


def unit_plan(limit=None) -> list:
    plan = [(c, s) for c in predeclared_matrix() for s in SEEDS]
    return plan if limit is None else plan[:limit]


def _counts(plan) -> dict:
    return {
        "family": dict(Counter(c["family"] for c, _ in plan)),
        "perturbation": dict(Counter(c["perturbation"] for c, _ in plan)),
        "snr_db": dict(Counter(c["snr_db"] for c, _ in plan)),
        "length": dict(Counter(str(c["length"]) for c, _ in plan)),
        "missingness": dict(Counter(c["missingness"]["kind"]
                                    for c, _ in plan)),
        "block": dict(Counter(b for c, _ in plan for b in c["blocks"])),
    }


# ----------------------------- materialization ----------------------
def materialize_unit(cell: dict, seed: int, out_dir: Path) -> dict:
    g = regenerate(_cell_params(cell), seed)
    rec = build_unit_record(cell, seed, g)
    ud = out_dir / rec["unit_id"]
    ud.mkdir(exist_ok=False)
    for k in ARRAY_NAMES:
        np.save(ud / f"{k}.npy", g[k])
    (ud / "events.json").write_text(
        _dumps({"unit_id": rec["unit_id"], "events": g["events"]}))
    (ud / "UNIT.json").write_text(_dumps(rec))
    return rec


def verify_unit(unit_dir: Path, regenerate_check: bool = True) -> dict:
    unit_dir = Path(unit_dir)
    rec = json.loads((unit_dir / "UNIT.json").read_text())
    bad = []
    arrays = {}
    for k in ARRAY_NAMES:
        a = np.load(unit_dir / f"{k}.npy", allow_pickle=False)
        arrays[k] = a
        if array_digest(a) != rec["digests"][k]:
            bad.append(k)
    ev_bytes = (unit_dir / "events.json").read_bytes()
    if hashlib.sha256(ev_bytes).hexdigest() != rec["events_sha256"]:
        bad.append("events")
    if regenerate_check:
        g = regenerate(rec["unit_params"], rec["seed"])
        for k in ARRAY_NAMES:
            if array_digest(g[k]) != rec["digests"][k]:
                bad.append(f"regenerate:{k}")
    return {"unit_id": rec["unit_id"], "ok": not bad, "mismatches": bad}


def build_manifest(records, plan_used, limit, total_bytes) -> dict:
    full = unit_plan()
    cells = predeclared_matrix()
    return {
        "schema": MANIFEST_SCHEMA,
        "generator": generator_identity(),
        "matrix": {
            "rule": predeclared_matrix.__doc__,
            "cartesian_sweep": False,
            "cells_declared": len(cells),
            "seeds": list(SEEDS),
            "units_declared": len(full),
            "counts_declared_units": _counts(full),
            "snr_grid_db": [str(s) for s in SNR_GRID_DB],
            "lengths": list(LENGTHS),
            "clean_families": list(CLEAN_FAMILIES),
            "perturbations": PERTURBATIONS,
            "missingness": MISSINGNESS,
            "partition_fractions": dict(PARTITION_FRACTIONS),
        },
        "limit": limit,
        "complete": len(records) == len(full),
        "units_materialized": len(records),
        "counts_materialized_units": _counts(plan_used),
        "bytes_materialized": total_bytes,
        "units": [{"unit_id": r["unit_id"], "dir": r["unit_id"],
                   "blocks": c["blocks"], "seed": r["seed"],
                   "unit_json_sha256": r["_unit_json_sha256"]}
                  for r, (c, _) in zip(records, plan_used)],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory; must not exist")
    ap.add_argument("--limit", type=int, default=None,
                    help="materialize only the first K planned units")
    args = ap.parse_args(argv)
    out = args.out
    if out.exists():
        print(f"REFUSED: {out.name} exists (write-once)", file=sys.stderr)
        return 2
    t0 = time.process_time()
    out.mkdir(parents=True, exist_ok=False)
    plan = unit_plan(args.limit)
    records = []
    for cell, seed in plan:
        rec = materialize_unit(cell, seed, out)
        rec["_unit_json_sha256"] = hashlib.sha256(
            (out / rec["unit_id"] / "UNIT.json").read_bytes()).hexdigest()
        records.append(rec)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    manifest = build_manifest(records, plan, args.limit, total)
    (out / "BANK_MANIFEST.json").write_text(_dumps(manifest))
    total += (out / "BANK_MANIFEST.json").stat().st_size
    print(json.dumps({"units": len(records),
                      "cells_declared": manifest["matrix"]["cells_declared"],
                      "bytes": total,
                      "cpu_seconds": round(time.process_time() - t0, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
