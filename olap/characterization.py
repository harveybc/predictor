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

-- C39 (order 2026-09-11): a descriptor row must say WHAT was measured.
--
-- The 420 pilot rows carried a value and a contract and nothing that
-- ties them to bytes: no source, no digest, no window, no code, no
-- protocol, and no terminal they were born from. Two different files
-- yielding the same descriptive number produced the SAME
-- observation_sha256 and collided as one observation.
--
-- These columns are ADDITIVE. The pilot rows are never edited or
-- deleted; they keep their identity and are labelled for what they
-- were, and every new measurement is a new version beside them.
ALTER TABLE {SCHEMA_NAME}.fact_variable_characterization
  ADD COLUMN IF NOT EXISTS source_id TEXT;
ALTER TABLE {SCHEMA_NAME}.fact_variable_characterization
  ADD COLUMN IF NOT EXISTS source_sha256 TEXT;
ALTER TABLE {SCHEMA_NAME}.fact_variable_characterization
  ADD COLUMN IF NOT EXISTS window_sha256 TEXT;
ALTER TABLE {SCHEMA_NAME}.fact_variable_characterization
  ADD COLUMN IF NOT EXISTS window_contract TEXT;
ALTER TABLE {SCHEMA_NAME}.fact_variable_characterization
  ADD COLUMN IF NOT EXISTS code_identity TEXT;
ALTER TABLE {SCHEMA_NAME}.fact_variable_characterization
  ADD COLUMN IF NOT EXISTS protocol_version TEXT;
ALTER TABLE {SCHEMA_NAME}.fact_variable_characterization
  ADD COLUMN IF NOT EXISTS side TEXT;
ALTER TABLE {SCHEMA_NAME}.fact_variable_characterization
  ADD COLUMN IF NOT EXISTS contract_role TEXT;
ALTER TABLE {SCHEMA_NAME}.fact_variable_characterization
  ADD COLUMN IF NOT EXISTS units TEXT;
ALTER TABLE {SCHEMA_NAME}.fact_variable_characterization
  ADD COLUMN IF NOT EXISTS terminal_attempt TEXT;
ALTER TABLE {SCHEMA_NAME}.fact_variable_characterization
  ADD COLUMN IF NOT EXISTS measurement_sha256 TEXT;
ALTER TABLE {SCHEMA_NAME}.fact_variable_characterization
  ADD COLUMN IF NOT EXISTS binding_state TEXT;

UPDATE {SCHEMA_NAME}.fact_variable_characterization
   SET binding_state = 'PILOT_UNBOUND_TO_SOURCE_BYTES'
 WHERE binding_state IS NULL;

-- C39/C38: current means the latest MEASUREMENT, not the latest load.
DROP VIEW IF EXISTS {SCHEMA_NAME}.v_variable_characterization_current;
CREATE VIEW {SCHEMA_NAME}.v_variable_characterization_current AS
SELECT DISTINCT ON (variable_id, partition_key, descriptor) *
FROM {SCHEMA_NAME}.fact_variable_characterization
ORDER BY variable_id, partition_key, descriptor,
         measured_at DESC, observation_sha256 DESC;

CREATE INDEX IF NOT EXISTS idx_fact_var_char_bank
  ON {SCHEMA_NAME}.fact_variable_characterization
     (bank_authority, descriptor);
CREATE INDEX IF NOT EXISTS idx_fact_var_char_source
  ON {SCHEMA_NAME}.fact_variable_characterization (source_sha256);
"""


class CharacterizationRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def _sha(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _timed(fn):
    """Measure a descriptor, and its cost, without letting one
    descriptor kill the measurement.

    C40: a variable whose range overflows float64 made `np.histogram`
    raise, and the exception escaped the whole characterization — so a
    single pathological column silenced every other descriptor of
    every other variable in the batch. A descriptor that cannot be
    computed is a typed ABSENCE, which is a result; it is not an
    excuse to report nothing.
    """
    t0 = time.perf_counter()
    try:
        value = fn()
    except Exception as exc:                            # noqa: BLE001
        value = _NotComputed(exc.__class__.__name__)
    return value, round(time.perf_counter() - t0, 9)


class _NotComputed:
    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason


#: C39: the facts a descriptor row must bind before it is a
#: measurement of anything. A value with no source is a number.
BINDING_KEYS = ("source_id", "source_sha256", "window_sha256",
                "window_contract", "code_identity", "protocol_version",
                "side", "contract_role", "units", "terminal_attempt")

PROTOCOL_VERSION = "crispdm.characterization_protocol.v2"


def verify_binding(binding: dict, *, variable_id: str) -> dict:
    """Every binding field present and non-empty, nothing undeclared."""
    if not isinstance(binding, dict):
        raise CharacterizationRefusal(
            f"{variable_id}: the observation binding is not a mapping")
    missing = [k for k in BINDING_KEYS
               if not str(binding.get(k, "")).strip()]
    if missing:
        raise CharacterizationRefusal(
            f"{variable_id}: the observation binding is missing "
            f"{missing} — a descriptor that cannot be traced to the "
            "bytes it was computed from is not a measurement")
    undeclared = sorted(set(binding) - set(BINDING_KEYS))
    if undeclared:
        raise CharacterizationRefusal(
            f"{variable_id}: undeclared binding fields {undeclared}")
    return {k: str(binding[k]) for k in BINDING_KEYS}


def characterize_series(values, *, variable_id: str,
                        partition_key: str,
                        bank_authority: str,
                        measured_at: str,
                        binding: dict,
                        noise_reference=None,
                        timestamps=None,
                        imputation: str | None = None) -> list[dict]:
    """Measure ONE variable on ONE development partition.

    `binding` is required (C39): the source and its byte digest, the
    exact window, the code and protocol that produced the numbers, the
    side/role the variable is consumed under, its units, and the
    terminal attempt this measurement was born from. All of it enters
    the observation identity, so two different files that happen to
    give the same descriptive number are two observations — they used
    to collide as one.
    """
    import numpy as np

    if bank_authority not in BANKS:
        raise CharacterizationRefusal(
            f"{variable_id}: unknown bank authority "
            f"{bank_authority!r} — a measurement whose authority "
            "is unknown is not stored")
    bound = verify_binding(binding, variable_id=variable_id)
    arr = np.asarray(values, dtype="float64")
    rows: list[dict] = []

    def add(descriptor, value, contract, cost,
            identifiable=True):
        # the observation identity is stamped HERE so every exit
        # path carries it — the early return for a series with
        # too few finite values used to skip it
        if isinstance(value, _NotComputed):
            rows.append({
                "variable_id": variable_id,
                "partition_key": partition_key,
                "bank_authority": bank_authority,
                "descriptor": descriptor,
                "value": None,
                "value_text": f"NOT_COMPUTABLE:{value.reason}",
                "descriptor_contract": (
                    f"{contract} — NOT IDENTIFIABLE: computing it "
                    f"raised {value.reason} on this variable"),
                "identifiable": False,
                "cost_seconds": cost,
                "measured_at": measured_at,
                "binding_state": "BOUND_TO_SOURCE_BYTES",
                **bound,
            })
            rows[-1]["observation_sha256"] = _sha({
                "variable_id": variable_id,
                "partition_key": partition_key,
                "descriptor": descriptor,
                "value_text": rows[-1]["value_text"],
                "identifiable": False,
                "contract": rows[-1]["descriptor_contract"],
                "bank": bank_authority, "binding": bound})
            rows[-1]["measurement_sha256"] = _sha({
                "observation": rows[-1]["observation_sha256"],
                "measured_at": measured_at, "cost_seconds": cost})
            return
        numeric = (isinstance(value, (int, float))
                   and not isinstance(value, bool))
        finite_value = numeric and math.isfinite(float(value))
        if numeric and not finite_value:
            # C40: an inf or NaN never leaves as an identifiable
            # number with numeric-looking text. It is a typed absence.
            value_text, identifiable = "NOT_FINITE", False
        elif value is None:
            value_text = "UNAVAILABLE"
        else:
            value_text = str(value)
        row = {
            "variable_id": variable_id,
            "partition_key": partition_key,
            "bank_authority": bank_authority,
            "descriptor": descriptor,
            "value": float(value) if finite_value else None,
            "value_text": value_text,
            "descriptor_contract": contract,
            "identifiable": bool(identifiable),
            "cost_seconds": cost,
            "measured_at": measured_at,
            "binding_state": "BOUND_TO_SOURCE_BYTES",
            **bound,
        }
        # C39: the SCIENTIFIC identity covers every fact that decides
        # what was measured. Cost and the instant of measurement are
        # performance, not science, so they carry their own identity
        # and cannot silently collapse two observations into one.
        row["observation_sha256"] = _sha({
            "variable_id": row["variable_id"],
            "partition_key": row["partition_key"],
            "descriptor": row["descriptor"],
            "value_text": row["value_text"],
            "identifiable": row["identifiable"],
            "contract": row["descriptor_contract"],
            "bank": row["bank_authority"],
            "binding": bound})
        row["measurement_sha256"] = _sha({
            "observation": row["observation_sha256"],
            "measured_at": measured_at,
            "cost_seconds": cost})
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

    # ---- dependence, on the ORIGINAL time axis ----
    #
    # C40: these used to run on `finite`, the array with non-finite
    # entries REMOVED. Deleting a row before shifting closes the gap,
    # so "lag 1" became "the next value that happens to exist" — with
    # one NaN in the middle of a daily series, the reported lag-1
    # autocorrelation pairs observations two days apart. The pairs are
    # now taken at the original separation and only kept when BOTH
    # members are finite.
    has_gaps = bool(n - finite.size)

    def acf(lag):
        def _f():
            a, b = arr[:-lag], arr[lag:]
            keep = np.isfinite(a) & np.isfinite(b)
            a, b = a[keep], b[keep]
            if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
                return None
            return float(np.corrcoef(a, b)[0, 1])
        return _f
    for lag in (1, 5, 20):
        if n > lag + 2:
            val, cost = _timed(acf(lag))
            add(f"autocorrelation_lag{lag}", val,
                f"Pearson correlation over pairs separated by {lag} "
                "positions on the ORIGINAL axis, keeping only pairs "
                "whose members are both finite; missing rows are "
                "skipped, never closed up", cost,
                identifiable=val is not None)

    def unit_root_proxy():
        # differences between ADJACENT original positions, both finite
        a, b = arr[:-1], arr[1:]
        keep = np.isfinite(a) & np.isfinite(b)
        d = b[keep] - a[keep]
        if d.size < 3 or np.std(finite) == 0:
            return None
        return float(np.std(d, ddof=1) / np.std(finite, ddof=1))
    val, cost = _timed(unit_root_proxy)
    add("difference_to_level_dispersion", val,
        "std of first differences over std of levels, differences "
        "taken between ADJACENT original positions that are both "
        "finite — a DESCRIPTIVE stationarity proxy, not a hypothesis "
        "test", cost, identifiable=val is not None)

    # ---- spectral content ----
    #
    # C40: a Fourier transform assumes a regular grid. Running it over
    # the compacted finite values silently resamples the series onto a
    # grid that does not exist. With gaps it either refuses, or uses an
    # imputation the CALLER predeclared — and then says which one.
    def spectral_centroid():
        if has_gaps and not imputation:
            return None
        series = finite if not has_gaps else _impute(arr, imputation)
        if series is None or series.size < 8:
            return None
        spec = np.abs(np.fft.rfft(series - series.mean()))
        freqs = np.fft.rfftfreq(series.size)
        total = spec.sum()
        return float((spec * freqs).sum() / total) if total > 0 \
            else None
    val, cost = _timed(spectral_centroid)
    add("spectral_centroid", val,
        "amplitude-weighted mean frequency of the mean-removed "
        "series on a REGULAR grid"
        + (f"; gaps filled by the predeclared imputation "
           f"{imputation!r}" if (has_gaps and imputation)
           else ("; NOT IDENTIFIABLE because the series has gaps and "
                 "no imputation was predeclared — a spectrum of a "
                 "compacted series describes a grid that does not "
                 "exist" if has_gaps else "; the series has no gaps")),
        cost, identifiable=val is not None)

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
            # C40: exact alignment. The observed and the reference are
            # compared position by position on the original axis, and
            # only where BOTH are finite — so a gap in either never
            # shifts one series against the other.
            ref = np.asarray(noise_reference, dtype="float64")
            if ref.shape != arr.shape:
                return None
            keep = np.isfinite(arr) & np.isfinite(ref)
            if keep.sum() < 3:
                return None
            resid = arr[keep] - ref[keep]
            rp, np_ = float(np.var(ref[keep])), float(np.var(resid))
            return (10.0 * math.log10(rp / np_)
                    if np_ > 0 and rp > 0 else None)
        val, cost = _timed(snr)
        add("signal_to_noise_db", val,
            "10*log10(var(reference)/var(observed-reference)); "
            "identifiable ONLY because a known clean reference "
            "was supplied", cost, identifiable=val is not None)

    return rows


#: imputations a caller may PREDECLARE for the spectral descriptor.
#: There is no default: filling gaps is a modelling decision, and one
#: made silently inside a measurement is the worst kind.
IMPUTATIONS = ("linear_interpolation", "mean_fill")


def _impute(arr, how: str | None):
    import numpy as np
    if how not in IMPUTATIONS:
        raise CharacterizationRefusal(
            f"unknown imputation {how!r}; declare one of "
            f"{list(IMPUTATIONS)} or accept that the spectrum is not "
            "identifiable for a series with gaps")
    finite_mask = np.isfinite(arr)
    if finite_mask.sum() < 2:
        return None
    idx = np.arange(arr.size)
    if how == "mean_fill":
        out = arr.copy()
        out[~finite_mask] = float(np.mean(arr[finite_mask]))
        return out
    return np.interp(idx, idx[finite_mask], arr[finite_mask])


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
                   cost_seconds, observation_sha256, measured_at,
                   source_id, source_sha256, window_sha256,
                   window_contract, code_identity,
                   protocol_version, side, contract_role, units,
                   terminal_attempt, measurement_sha256,
                   binding_state)
                VALUES (:v, :p, :b, :d, :val, :vt, :c, :i, :cost,
                        :o, :m, :src, :srcsha, :wsha, :wc, :code,
                        :proto, :side, :role, :units, :term,
                        :msha, :bstate)
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
                   "m": r["measured_at"],
                   "src": r.get("source_id"),
                   "srcsha": r.get("source_sha256"),
                   "wsha": r.get("window_sha256"),
                   "wc": r.get("window_contract"),
                   "code": r.get("code_identity"),
                   "proto": r.get("protocol_version"),
                   "side": r.get("side"),
                   "role": r.get("contract_role"),
                   "units": r.get("units"),
                   "term": r.get("terminal_attempt"),
                   "msha": r.get("measurement_sha256"),
                   "bstate": r.get("binding_state",
                                   "PILOT_UNBOUND_TO_SOURCE_BYTES")})
            loaded += res.rowcount or 0
    return {"fact_variable_characterization": loaded}
