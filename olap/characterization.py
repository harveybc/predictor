"""C30: the first per-variable characterization.

This measures variables. It does not choose them.

Nothing here fits a model, scores a selector, promotes an operator
or touches confirmation-reserved data. Every descriptor is
empirical and is named for what it is: a compressed length under a
declared compressor is a compressed length, not a measure of
information content; an entropy under a declared quantisation is
that quantisation's entropy, not the variable's; and a noise
estimate is only reported where a reference, a replicate or a
generator makes noise identifiable at all.

The cost of computing each descriptor is measured and stored
beside it, because a descriptor nobody can afford is not a
descriptor anyone will use.

Bank separation is structural, not advisory:

  * synthetic   — calibrates a diagnostic against known truth;
  * public      — evaluates transfer outside the origin domain;
  * financial   — development and domain revalidation only.

A row carries the authority of its bank, and the loader stores it
that way, so a query can never silently mix them.
"""
from __future__ import annotations

import bz2
import hashlib
import json
import math
import time
import zlib

SCHEMA_NAME = "public"
CHARACTERIZATION_SCHEMA = "crispdm.variable_characterization.v1"

# The banks, and what a measurement taken on each may support.
BANK_SYNTHETIC = "SYNTHETIC_KNOWN_MECHANISM_CALIBRATION_ONLY"
BANK_PUBLIC = "PUBLIC_FORECASTING_EVIDENCE"
BANK_FINANCIAL = "FINANCIAL_DOMAIN_DEVELOPMENT_ONLY"
BANKS = (BANK_SYNTHETIC, BANK_PUBLIC, BANK_FINANCIAL)

# Descriptors are declared with their contract, so a reader knows
# exactly what was computed rather than inferring it from a name.
COMPRESSOR = "zlib level 9 over float64 little-endian bytes"
QUANTISATION = "64 equal-width bins over the observed range"

DDL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.fact_variable_characterization (
  variable_id        TEXT NOT NULL,
  partition_key      TEXT NOT NULL,
  bank_authority     TEXT NOT NULL,
  descriptor         TEXT NOT NULL,
  value              DOUBLE PRECISION,
  value_text         TEXT NOT NULL,
  descriptor_contract TEXT NOT NULL,
  identifiable       BOOLEAN NOT NULL,
  cost_seconds       DOUBLE PRECISION NOT NULL,
  observation_sha256 TEXT NOT NULL,
  measured_at        TEXT NOT NULL,
  loaded_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (variable_id, partition_key, descriptor,
               observation_sha256)
);

CREATE OR REPLACE VIEW {SCHEMA_NAME}.v_variable_characterization_current AS
SELECT DISTINCT ON (variable_id, partition_key, descriptor) *
FROM {SCHEMA_NAME}.fact_variable_characterization
ORDER BY variable_id, partition_key, descriptor,
         loaded_at DESC, observation_sha256 DESC;

CREATE INDEX IF NOT EXISTS idx_fact_var_char_bank
  ON {SCHEMA_NAME}.fact_variable_characterization
     (bank_authority, descriptor);
"""


class CharacterizationRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def _sha(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, round(time.perf_counter() - t0, 9)


def characterize_series(values, *, variable_id: str,
                        partition_key: str,
                        bank_authority: str,
                        measured_at: str,
                        noise_reference=None) -> list[dict]:
    """Measure ONE variable on ONE development partition."""
    import numpy as np

    if bank_authority not in BANKS:
        raise CharacterizationRefusal(
            f"{variable_id}: unknown bank authority "
            f"{bank_authority!r} — a measurement whose authority "
            "is unknown is not stored")
    arr = np.asarray(values, dtype="float64")
    rows: list[dict] = []

    def add(descriptor, value, contract, cost,
            identifiable=True):
        # the observation identity is stamped HERE so every exit
        # path carries it — the early return for a series with
        # too few finite values used to skip it
        row = {
            "variable_id": variable_id,
            "partition_key": partition_key,
            "bank_authority": bank_authority,
            "descriptor": descriptor,
            "value": (float(value)
                      if isinstance(value, (int, float))
                      and math.isfinite(float(value)) else None),
            "value_text": ("UNAVAILABLE" if value is None
                           else str(value)),
            "descriptor_contract": contract,
            "identifiable": bool(identifiable),
            "cost_seconds": cost,
            "measured_at": measured_at,
        }
        row["observation_sha256"] = _sha({
            "variable_id": row["variable_id"],
            "partition_key": row["partition_key"],
            "descriptor": row["descriptor"],
            "value_text": row["value_text"],
            "contract": row["descriptor_contract"],
            "bank": row["bank_authority"]})
        rows.append(row)

    n = arr.size
    finite = arr[np.isfinite(arr)]

    # ---- coverage and quality ----
    v, c = _timed(lambda: int(n))
    add("n_observations", v, "count of rows read", c)
    v, c = _timed(lambda: int(np.count_nonzero(np.isnan(arr))))
    add("missing_count", v, "NaN entries", c)
    v, c = _timed(lambda: int(n - np.count_nonzero(
        np.isfinite(arr))))
    add("non_finite_count", v, "NaN or infinite entries", c)
    v, c = _timed(lambda: int(n - np.unique(arr).size))
    add("duplicate_count", v, "rows minus distinct values", c)
    v, c = _timed(lambda: float(
        1.0 - finite.size / n if n else 0.0))
    add("missingness_fraction", v, "1 - finite/total", c)

    if finite.size < 3:
        add("insufficient_finite_observations", None,
            "fewer than 3 finite values: distribution, "
            "dependence and compression descriptors are not "
            "identifiable", 0.0, identifiable=False)
        return rows

    # ---- distribution and scale ----
    for name, fn, contract in (
        ("mean", lambda: float(np.mean(finite)), "arithmetic mean"),
        ("median", lambda: float(np.median(finite)), "50th percentile"),
        ("std", lambda: float(np.std(finite, ddof=1)),
         "sample standard deviation"),
        ("iqr", lambda: float(np.subtract(
            *np.percentile(finite, [75, 25]))),
         "robust scale: p75 - p25"),
        ("mad", lambda: float(np.median(
            np.abs(finite - np.median(finite)))),
         "median absolute deviation"),
        ("min", lambda: float(np.min(finite)), "minimum"),
        ("max", lambda: float(np.max(finite)), "maximum"),
        ("p01", lambda: float(np.percentile(finite, 1)),
         "1st percentile — extreme tail"),
        ("p99", lambda: float(np.percentile(finite, 99)),
         "99th percentile — extreme tail"),
        ("constant", lambda: float(
            np.min(finite) == np.max(finite)),
         "1.0 when the variable never varies"),
    ):
        val, cost = _timed(fn)
        add(name, val, contract, cost)

    # ---- stability across windows ----
    def window_stability():
        k = max(2, min(8, finite.size // 32))
        chunks = np.array_split(finite, k)
        means = np.array([ch.mean() for ch in chunks
                          if ch.size])
        scale = np.std(finite, ddof=1)
        return float(np.std(means, ddof=1) / scale) if (
            scale > 0 and means.size > 1) else 0.0
    val, cost = _timed(window_stability)
    add("window_mean_dispersion", val,
        "std of per-window means divided by the overall std; "
        "windows are equal splits, not a regime model", cost)

    # ---- dependence ----
    def acf(lag):
        def _f():
            a, b = finite[:-lag], finite[lag:]
            if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
                return None
            return float(np.corrcoef(a, b)[0, 1])
        return _f
    for lag in (1, 5, 20):
        if finite.size > lag + 2:
            val, cost = _timed(acf(lag))
            add(f"autocorrelation_lag{lag}", val,
                f"Pearson correlation at lag {lag}", cost,
                identifiable=val is not None)

    def unit_root_proxy():
        d = np.diff(finite)
        if d.size < 3 or np.std(finite) == 0:
            return None
        return float(np.std(d, ddof=1) / np.std(finite, ddof=1))
    val, cost = _timed(unit_root_proxy)
    add("difference_to_level_dispersion", val,
        "std of first differences over std of levels — a "
        "DESCRIPTIVE stationarity proxy, not a hypothesis test",
        cost, identifiable=val is not None)

    # ---- spectral content ----
    def spectral_centroid():
        if finite.size < 8:
            return None
        spec = np.abs(np.fft.rfft(finite - finite.mean()))
        freqs = np.fft.rfftfreq(finite.size)
        total = spec.sum()
        return float((spec * freqs).sum() / total) if total > 0 \
            else None
    val, cost = _timed(spectral_centroid)
    add("spectral_centroid", val,
        "amplitude-weighted mean frequency of the "
        "mean-removed series", cost,
        identifiable=val is not None)

    # ---- compressibility, named honestly ----
    def compressed_ratio():
        raw = finite.astype("<f8").tobytes()
        return float(len(zlib.compress(raw, 9)) / len(raw))
    val, cost = _timed(compressed_ratio)
    add("compressed_length_ratio", val,
        f"compressed bytes over raw bytes under {COMPRESSOR}; "
        "this is a property of the compressor and the data, NOT "
        "an information content and NOT Kolmogorov complexity",
        cost)

    def discrete_entropy():
        import numpy as _np
        hist, _ = _np.histogram(finite, bins=64)
        p = hist[hist > 0] / hist.sum()
        return float(-(p * _np.log2(p)).sum())
    val, cost = _timed(discrete_entropy)
    add("discrete_entropy_bits", val,
        f"Shannon entropy under {QUANTISATION}; the value is a "
        "property of that quantisation, not of the variable",
        cost)

    # ---- noise, only where it is identifiable ----
    if noise_reference is None:
        add("noise_estimate", None,
            "NOT IDENTIFIABLE: no reference, replicate or "
            "generator makes the clean component observable for "
            "this variable, so no signal-to-noise figure is "
            "reported", 0.0, identifiable=False)
    else:
        def snr():
            ref = np.asarray(noise_reference, dtype="float64")
            if ref.shape != arr.shape:
                return None
            resid = finite - ref[np.isfinite(arr)]
            rp, np_ = float(np.var(ref)), float(np.var(resid))
            return (10.0 * math.log10(rp / np_)
                    if np_ > 0 and rp > 0 else None)
        val, cost = _timed(snr)
        add("signal_to_noise_db", val,
            "10*log10(var(reference)/var(observed-reference)); "
            "identifiable ONLY because a known clean reference "
            "was supplied", cost, identifiable=val is not None)

    return rows


def assert_no_selection(rows: list[dict]) -> None:
    """C30: a characterization measures. It never ranks."""
    forbidden = ("rank", "selected", "chosen", "score",
                 "importance", "promoted", "eligible")
    for r in rows:
        d = r["descriptor"].lower()
        for word in forbidden:
            if word in d:
                raise CharacterizationRefusal(
                    f"descriptor {r['descriptor']!r} names a "
                    "selection; C30 measures variables and never "
                    "chooses them")


def ensure_tables(engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(DDL)


def load_rows(engine, rows: list[dict]) -> dict:
    """Versioned and idempotent, exactly like the inventory."""
    from sqlalchemy import text
    assert_no_selection(rows)
    loaded = 0
    with engine.begin() as conn:
        conn.exec_driver_sql(DDL)
        for r in rows:
            res = conn.execute(text(f"""
                INSERT INTO
                  {SCHEMA_NAME}.fact_variable_characterization
                  (variable_id, partition_key, bank_authority,
                   descriptor, value, value_text,
                   descriptor_contract, identifiable,
                   cost_seconds, observation_sha256, measured_at)
                VALUES (:v, :p, :b, :d, :val, :vt, :c, :i, :cost,
                        :o, :m)
                ON CONFLICT (variable_id, partition_key,
                             descriptor, observation_sha256)
                DO NOTHING
            """), {"v": r["variable_id"],
                   "p": r["partition_key"],
                   "b": r["bank_authority"],
                   "d": r["descriptor"], "val": r["value"],
                   "vt": r["value_text"],
                   "c": r["descriptor_contract"],
                   "i": r["identifiable"],
                   "cost": r["cost_seconds"],
                   "o": r["observation_sha256"],
                   "m": r["measured_at"]})
            loaded += res.rowcount or 0
    return {"fact_variable_characterization": loaded}
