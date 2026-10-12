"""I6N causal traceability audit: one bounded, read-only, CPU-only run.

``run`` executes every check of docs/handoffs/RETSU_I6N_CAUSAL_TRACE_AUDIT_2026_10_12.md
against retained campaign evidence and writes the deliverables atomically under
one output directory. ``status`` reads the terminal status back. Nothing under
a campaign root is written; no model is fitted; no TEST row (>= 2025-01-01) is
used: every external series is filtered on its date column before any value is
read, and the campaign inputs end on 2024-12-31 by construction.

Every verdict is one of PROVEN_SAFE, PROVEN_LEAK, RISK_UNRESOLVED or
NOT_APPLICABLE. States are counted, never averaged. Missing evidence stays
RISK_UNRESOLVED. Mutation checks use exact equality; every tolerance used
elsewhere is declared in SPEC.json before the run and is reported beside the
exact result, never instead of it.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PROVEN_SAFE = "PROVEN_SAFE"
PROVEN_LEAK = "PROVEN_LEAK"
RISK_UNRESOLVED = "RISK_UNRESOLVED"
NOT_APPLICABLE = "NOT_APPLICABLE"
STATES = (PROVEN_SAFE, PROVEN_LEAK, RISK_UNRESOLVED, NOT_APPLICABLE)
SCHEMA = "i6n.causal_trace_audit.v1"
REQUIRED_TOLERANCES = ("mutation", "digest", "identity_equality", "metric_replay")
MINIMAL_SPEC = {
    "schema": "i6n.causal_trace_audit.spec.v1",
    "tolerances": {"mutation": "exact", "digest": "exact", "identity_equality": "exact",
                   "metric_replay": "exact"},
}
HOUR = 3600
TEST_START = 1735689600  # 2025-01-01T00:00:00Z
REPO = Path(__file__).resolve().parents[1]


class DigestMismatch(ValueError):
    """A retained artifact does not carry the digest the evidence declares."""


# ----------------------------------------------------------------------------- primitives

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def verify_digest(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise DigestMismatch(f"{Path(path).name}: expected {expected[:12]} got {actual[:12]}")
    return True


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    if spec.get("schema") != MINIMAL_SPEC["schema"]:
        raise ValueError("spec schema mismatch")
    tolerances = spec.get("tolerances")
    if not isinstance(tolerances, dict):
        raise ValueError("spec must declare tolerances before the run")
    for key in REQUIRED_TOLERANCES:
        if key not in tolerances:
            raise ValueError(f"tolerance {key} must be declared before the run")
        if tolerances[key] != "exact":
            raise ValueError(f"tolerance {key} must be exact")
    return spec


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(jsonable(value), indent=1, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def iso(seconds):
    return datetime.fromtimestamp(int(seconds), timezone.utc).isoformat()


def epoch(text):
    return int(datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp())


class Findings:
    def __init__(self):
        self.rows = []

    def add(self, finding_id, section, assertion, state, evidence, affected_claims, detail=None):
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}")
        self.rows.append(dict(id=finding_id, section=section, assertion=assertion, state=state,
                              evidence=evidence, affected_claims=list(affected_claims),
                              detail=jsonable(detail) if detail is not None else None))
        return state

    def counts(self):
        return {s: sum(1 for r in self.rows if r["state"] == s) for s in STATES}

    def worst(self):
        for state in (PROVEN_LEAK, RISK_UNRESOLVED, PROVEN_SAFE, NOT_APPLICABLE):
            if any(r["state"] == state for r in self.rows):
                return state
        return NOT_APPLICABLE

    def to_csv(self, path, redactor=lambda s: s):
        with Path(path).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "section", "assertion", "state", "evidence", "affected_claims", "detail"])
            for r in self.rows:
                writer.writerow([r["id"], r["section"], r["assertion"], r["state"], redactor(r["evidence"]),
                                 ";".join(r["affected_claims"]),
                                 redactor(json.dumps(r["detail"], sort_keys=True)) if r["detail"] is not None else ""])


class Status:
    def __init__(self, out_dir, findings):
        self.out, self.findings, self.started = Path(out_dir), findings, time.time()
        self.phases = {}

    def write(self, state, phase=None, extra=None):
        payload = dict(schema=SCHEMA + ".status", state=state, phase=phase,
                       elapsed_seconds=round(time.time() - self.started, 1),
                       updated_utc=datetime.now(timezone.utc).isoformat(),
                       counts=self.findings.counts(), findings=len(self.findings.rows),
                       phase_seconds=self.phases)
        if extra:
            payload.update(extra)
        atomic_json(self.out / "STATUS.json", payload)


_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_HOME = re.compile(r"/home/[^/\s\"']+")
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_GPU = re.compile(r"GPU-[0-9a-fA-F-]{8,}")


def redact(text, mapping):
    out = str(text)
    for key in sorted(mapping, key=len, reverse=True):
        out = out.replace(key, mapping[key])
    out = _GPU.sub("<gpu-uuid>", out)
    out = _UUID.sub("<uuid>", out)
    out = _IPV4.sub("<ip>", out)
    out = _HOME.sub("<home>", out)
    return out


# ----------------------------------------------------------------------------- pure checks

def origins_sha256(origins):
    return canonical_digest(sorted(int(t) for t in origins))


def window_timestamps(t, samples, spacing_hours):
    return [int(t) - (samples - 1 - i) * spacing_hours * HOUR for i in range(samples)]


def _window(ts, values, t, samples, spacing_hours):
    """Own right-anchored sampler: exact-time lookup, zeros and a False mask when absent."""
    stamps = np.asarray(window_timestamps(t, samples, spacing_hours), np.int64)
    index = np.searchsorted(ts, stamps)
    present = (index < len(ts)) & (ts[np.minimum(index, len(ts) - 1)] == stamps)
    out = np.zeros((samples, values.shape[1]), values.dtype)
    out[present] = values[index[present]]
    return out, present, stamps


def window_mutation_checks(ts, x, t, samples, spacing_hours, sentinel, transform=None):
    ts = np.asarray(ts, np.int64)
    x = np.asarray(x, np.float64)
    f = (lambda v: v) if transform is None else transform
    base, present, stamps = _window(ts, f(x), t, samples, spacing_hours)
    future = x.copy()
    future[ts > t] += sentinel
    after, _, _ = _window(ts, f(future), t, samples, spacing_hours)
    origin = x.copy()
    origin[ts == t] += sentinel
    at_origin, _, _ = _window(ts, f(origin), t, samples, spacing_hours)
    eq = lambda a, b: bool(np.array_equal(a, b, equal_nan=True))  # noqa: E731  exact; NaN placeholders compare equal to themselves
    result = dict(origin=int(t), window_timestamps=[int(s) for s in stamps], present=present.tolist(),
                  rightmost_is_origin=int(stamps[-1]) == int(t),
                  others_before_origin=bool(np.all(stamps[:-1] < t)),
                  grid_exact=bool(np.all(np.diff(stamps) == spacing_hours * HOUR)),
                  future_rows_unchanged=eq(base, after),
                  origin_row_changes_rightmost=(not eq(base[-1], at_origin[-1])) and eq(base[:-1], at_origin[:-1]))
    if not result["future_rows_unchanged"]:
        result["state"] = PROVEN_LEAK
    elif all(result[k] for k in ("rightmost_is_origin", "others_before_origin", "grid_exact",
                                 "origin_row_changes_rightmost")):
        result["state"] = PROVEN_SAFE
    else:
        result["state"] = RISK_UNRESOLVED
    return result


def impulse_after_t(ts, x, t, transform, sentinel=1e6):
    ts = np.asarray(ts, np.int64)
    base = transform(np.asarray(x, np.float64))
    hit = np.asarray(x, np.float64).copy()
    hit[ts > t] += sentinel
    moved = transform(hit)
    past = ts <= t
    return dict(rows_at_or_before_t_unchanged=bool(np.array_equal(base[past], moved[past], equal_nan=True)),
                rows_after_t_changed=bool(not np.array_equal(base[~past], moved[~past], equal_nan=True)) if (~past).any() else None)


def _label(ts, price, t, horizon_hours):
    oi = int(np.searchsorted(ts, t))
    ti = int(np.searchsorted(ts, t + horizon_hours * HOUR))
    ok = oi < len(ts) and ti < len(ts) and ts[oi] == t and ts[ti] == t + horizon_hours * HOUR
    return (math.log(price[ti]) - math.log(price[oi]) if ok else None), ok, ti


def target_mutation_check(ts, x, price, t, samples, spacing_hours, horizon_hours, sentinel=0.5):
    ts, price = np.asarray(ts, np.int64), np.asarray(price, np.float64)
    base_label, ok, ti = _label(ts, price, t, horizon_hours)
    base_window, _, _ = _window(ts, np.asarray(x, np.float64), t, samples, spacing_hours)
    moved = price.copy()
    if ok:
        moved[ti] += sentinel
    new_label, _, _ = _label(ts, moved, t, horizon_hours)
    window_after, _, _ = _window(ts, np.asarray(x, np.float64), t, samples, spacing_hours)
    return dict(origin=int(t), target=int(t + horizon_hours * HOUR), target_observed=bool(ok),
                target_is_t_plus_4_elapsed_hours=bool(ok and ts[ti] - t == horizon_hours * HOUR),
                label_changes=bool(ok and new_label != base_label),
                input_unchanged=bool(np.array_equal(base_window, window_after, equal_nan=True)))


def mask_parity(model_rows, naive_rows):
    model = [list(map(int, r)) for r in model_rows]
    naive = [list(map(int, r)) for r in naive_rows]
    ms, ns = {tuple(r) for r in model}, {tuple(r) for r in naive}
    return dict(model_count=len(model), naive_count=len(naive), counts_equal=len(model) == len(naive),
                identical=model == naive, same_set=ms == ns,
                only_model=[list(r) for r in sorted(ms - ns)][:50], only_naive=[list(r) for r in sorted(ns - ms)][:50],
                model_sha256=origins_sha256([r[0] for r in model]), naive_sha256=origins_sha256([r[0] for r in naive]))


def replay_rows(current, future, prediction):
    current, future, prediction = (np.asarray(a, np.float64) for a in (current, future, prediction))
    n = int(len(future))
    sae = math.fsum(np.abs(prediction - future).tolist())
    nsae = math.fsum(np.abs(current - future).tolist())
    return dict(n=n, sum_abs_error=sae, naive_sum_abs_error=nsae,
                mae=sae / n if n else None, naive_mae=nsae / n if n else None)


def fold_fitness(error_sums, naive_sums):
    return math.fsum(error_sums[s] / naive_sums[s] for s in sorted(("train", "inner"))) / 2


def inner_only_fitness(error_sums, naive_sums):
    return error_sums["inner"] / naive_sums["inner"]


def aggregate(values):
    values = list(values)
    return math.fsum(values) / len(values)


def compare_exact(a, b):
    return a == b


def scaler_digest(x, fit_indices):
    from tools.fs4_temporal_predictor import Standardiser
    values = np.where(np.isfinite(x), x, np.nan)
    return Standardiser.fit(values[np.asarray(fit_indices)]).sha256()


def fitted_state_check(x, fit_indices, perturb, sentinel=1e6):
    base = scaler_digest(x, fit_indices)
    moved = np.asarray(x, np.float64).copy()
    moved[np.asarray(perturb)] += sentinel
    after = scaler_digest(moved, fit_indices)
    return dict(byte_identical=base == after, base=base, after=after)


def fs4_union(winners):
    members = set()
    for target in winners.values():
        members.update(target["RAW"]["members"])
    return sorted(members)


def selection_chronology(selection_max_time, fold_cutoff):
    return PROVEN_LEAK if epoch(selection_max_time) > epoch(fold_cutoff) else PROVEN_SAFE


def availability_lags(ts, column, raw, abs_tol=0.0, max_days=21):
    """For every value transition of a dated source, the first hour at which the column shows it."""
    ts = np.asarray(ts, np.int64)
    column = np.asarray(column, np.float64)
    dates = sorted(raw)
    lags, matched, total, cursor, unmatched = [], 0, 0, 0, []
    previous = None
    for d in dates:
        v = raw[d]
        if v is None or not math.isfinite(v) or (previous is not None and v == previous):
            previous = v if v is not None else previous
            continue
        previous = v
        day = epoch(d + "T00:00:00+00:00")
        if day < ts[0] - max_days * 86400 or day > ts[-1]:
            continue
        total += 1
        start = max(cursor, int(np.searchsorted(ts, day - max_days * 86400)))
        stop = int(np.searchsorted(ts, day + max_days * 86400))
        segment = column[start:stop]
        hits = np.flatnonzero(np.abs(segment - v) <= abs_tol) if abs_tol else np.flatnonzero(segment == v)
        if not len(hits):
            unmatched.append(d)
            continue
        j = start + int(hits[0])
        lags.append((int(ts[j]) - day) / HOUR)
        matched += 1
        cursor = j
    out = dict(matched=matched, total_transitions=total, unmatched_example=unmatched[:5])
    if lags:
        arr = np.asarray(lags)
        out.update(lag_hours=lags if len(lags) <= 400 else lags[:400], min=float(arr.min()),
                   p50=float(np.median(arr)), max=float(arr.max()),
                   share_below_20h=float(np.mean(arr < 20)), share_below_0h=float(np.mean(arr < 0)))
    return out


def asof_alignment(ts, column, raw, lag_hours=range(-24, 97), abs_tol=1e-9):
    """Identify the as-of lag L (hours after the source date's 00:00 UTC) under which the
    hourly column equals the forward-filled dated series; robust to repeated values."""
    ts = np.asarray(ts, np.int64)
    column = np.asarray(column, np.float64)
    dates = [d for d in sorted(raw) if raw[d] is not None and math.isfinite(raw[d])]
    if not dates:
        return dict(best_lag_hours=None, best_match_rate=0.0, compared=0)
    days = np.asarray([epoch(d + "T00:00:00+00:00") for d in dates], np.int64)
    values = np.asarray([raw[d] for d in dates], np.float64)
    finite = np.isfinite(column)
    best, table = None, {}
    for lag in lag_hours:
        idx = np.searchsorted(days + lag * HOUR, ts, side="right") - 1
        ok = (idx >= 0) & finite
        expected = values[np.maximum(idx, 0)]
        hit = ok & (np.abs(column - expected) <= abs_tol)
        rate = float(hit.sum() / ok.sum()) if ok.any() else 0.0
        table[lag] = rate
        if best is None or rate > best[1]:
            best = (lag, rate, int(ok.sum()))
    return dict(best_lag_hours=best[0], best_match_rate=best[1], compared=best[2],
                rate_at_lag={str(k): round(v, 4) for k, v in table.items() if v >= 0.5})


def asof_state(alignment, bound_hours, min_rate=0.95):
    if alignment["best_lag_hours"] is None or alignment["best_match_rate"] < min_rate:
        return RISK_UNRESOLVED
    return PROVEN_LEAK if alignment["best_lag_hours"] < bound_hours else PROVEN_SAFE


def availability_state(lags, bound_hours, min_match_share=0.5):
    if not lags.get("matched") or lags["matched"] < min_match_share * max(1, lags.get("total_transitions", 1)):
        return RISK_UNRESOLVED
    return PROVEN_LEAK if lags["min"] < bound_hours else PROVEN_SAFE


# ----------------------------------------------------------------------------- campaign run

def _quarter_cutoff(weeks, year, quarter):
    """Close of the last calendar week entirely inside the quarter (plan folds.rule)."""
    start = datetime(year, 3 * (quarter - 1) + 1, 1, tzinfo=timezone.utc)
    end = datetime(year + (quarter == 4), 1 if quarter == 4 else 3 * quarter + 1, 1, tzinfo=timezone.utc)
    inside = [w for w in weeks if epoch(w["market_open"]) >= start.timestamp() and epoch(w["market_close"]) <= end.timestamp()]
    return max(epoch(w["market_close"]) for w in inside)


def _independent_eligibility(ts, weeks, start, end, horizon, entry_delay, force_flat_hours, span_hours):
    """Independent re-derivation of the MANAGEMENT endpoint mask on the hourly grid."""
    present = set(int(t) for t in ts)
    first = int(ts[0]) + span_hours * HOUR
    eligible, rejected = [], {}
    for w in weeks:
        o, c = epoch(w["market_open"]), epoch(w["market_close"])
        ff = c - force_flat_hours * HOUR
        for t in range(max(o, start), min(c, end), HOUR):
            q = t + horizon * HOUR
            reasons = []
            if t < o + entry_delay * HOUR:
                reasons.append("ORIGIN_BEFORE_OPERATIONAL_START")
            if t >= ff:
                reasons.append("ORIGIN_AT_OR_AFTER_FORCE_FLAT")
            if q >= ff:
                reasons.append("TARGET_AT_OR_AFTER_FORCE_FLAT")
            if q >= c:
                reasons.append("TARGET_AT_OR_AFTER_MARKET_CLOSE")
            if t not in present or t < first:
                reasons.append("INPUT_UNAVAILABLE")
            if q not in present:
                reasons.append("TARGET_NOT_OBSERVED")
            if q >= end:
                reasons.append("TARGET_AT_OR_AFTER_SPLIT_END")
            if reasons:
                rejected[reasons[0]] = rejected.get(reasons[0], 0) + 1
            else:
                eligible.append(t)
    return eligible, rejected


def _read_fred(path, end_epoch):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["date", "value"])
    out = {}
    for d, v in zip(table.column("date").to_pylist(), table.column("value").to_pylist()):
        if epoch(str(d) + "T00:00:00+00:00") >= end_epoch:
            continue  # never read a TEST-dated value
        try:
            out[str(d)] = float(v)
        except (TypeError, ValueError):
            out[str(d)] = None
    return out


def _read_yahoo(path, end_epoch):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["Date", "Close", "Adj Close"])
    rows = []
    for d, c, a in zip(table.column("Date").to_pylist(), table.column("Close").to_pylist(),
                       table.column("Adj Close").to_pylist()):
        day = d.date().isoformat()
        if epoch(day + "T00:00:00+00:00") >= end_epoch:
            continue
        rows.append((day, c, a))
    return rows


def _derive(series, formula):
    """series: ordered list of (date, value). Returns {date: derived} per closed formula."""
    out = {}
    dates = [d for d, _ in series]
    vals = [v for _, v in series]
    for k, d in enumerate(dates):
        v = vals[k]
        if v is None or not math.isfinite(v):
            continue
        if formula == "level":
            out[d] = v
        elif formula.startswith("diff_") or formula.startswith("logret_"):
            spec_lag = formula.split("_")[1]
            if spec_lag.endswith("cal"):  # value as of d minus the value as of d - N calendar days
                n = int(spec_lag[:-3])
                day = datetime.fromisoformat(d) - timedelta(days=n)
                j = max(i for i, dd in enumerate(dates) if dd <= day.date().isoformat()) if dates[0] <= day.date().isoformat() else None
                prev = vals[j] if j is not None else None
            else:
                lag = int(spec_lag)
                prev = vals[k - lag] if k - lag >= 0 else None
            if prev is None or not math.isfinite(prev):
                continue
            out[d] = v - prev if formula.startswith("diff") else (math.log(v / prev) if v > 0 and prev > 0 else None)
    return out


def _read_fx(path, end_epoch):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["datetime", "open", "high", "low", "close"])
    stamps = table.column("datetime").cast("int64").to_numpy() // 10**9
    keep = stamps < end_epoch
    cols = {c: table.column(c).to_numpy(zero_copy_only=False).astype(np.float64)[keep] for c in ("open", "high", "low", "close")}
    return stamps[keep].astype(np.int64), cols


def _match_rate(ts, column, vendor_ts, vendor_values, lag, tol):
    index = np.searchsorted(vendor_ts, ts + lag)
    ok = (index < len(vendor_ts)) & (vendor_ts[np.minimum(index, len(vendor_ts) - 1)] == ts + lag)
    both = ok & np.isfinite(column)
    if not both.any():
        return 0.0, 0
    v = vendor_values[index[both]]
    good = np.isfinite(v) & (np.abs(v - column[both]) <= tol)
    return float(good.mean()), int(both.sum())


class Audit:
    def __init__(self, args):
        self.args = args
        self.spec = load_spec(args.spec)
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.findings = Findings()
        self.status = Status(self.out, self.findings)
        self.campaign = Path(args.campaign_root)
        self.data = Path(args.data_root)
        self.cas = Path(args.cas_root)
        self.state = Path(args.state_root)
        self.fin = Path(args.fin_root)
        self.island = args.island
        self.redactions = {str(self.campaign): "<campaign-root>", str(self.data): "<data-root>",
                           str(self.cas): "<cas-root>", str(self.state): "<state-root>", str(self.fin): "<fin-root>",
                           str(REPO): "<repo>"}
        self.graph = dict(schema=SCHEMA + ".trace_graph", nodes=[], edges=[])
        self.alignment_cases = []
        self.feature_rows = []
        self.mask_parity = {}
        self.metric_replay = {}
        self.inputs_read = {}
        os.environ["I6N_DATA_ROOT"] = str(self.data)

    # -- helpers --------------------------------------------------------------------------
    def r(self, text):
        return redact(text, self.redactions)

    def note_input(self, path):
        path = Path(path)
        key = self.r(str(path))
        if key not in self.inputs_read and path.is_file():
            self.inputs_read[key] = sha256_file(path)
        return self.inputs_read.get(key)

    def node(self, node_id, kind, **fields):
        self.graph["nodes"].append(dict(id=node_id, kind=kind, **jsonable(fields)))

    def edge(self, src, dst, **fields):
        base = dict(producer_path=None, producer_commit=None, artifact_path=None, sha256=None, schema=None,
                    row_identity=None, min_source_timestamp=None, max_source_timestamp=None,
                    fitting_interval=None, output_timestamp=None, availability_timestamp=None,
                    consumer=dst, provenance_state=RISK_UNRESOLVED)
        base.update(jsonable(fields))
        self.graph["edges"].append(dict(source=src, target=dst, **base))

    def phase(self, name):
        self.status.phases[name] = round(time.time() - self.status.started, 1)
        self.status.write("RUNNING", phase=name)

    # -- A. identity -----------------------------------------------------------------------
    def phase_identity(self):
        spec, F = self.spec["object"], self.findings
        manifest = json.loads((self.campaign / "config/MANIFEST.json").read_text())
        self.note_input(self.campaign / "config/MANIFEST.json")
        islands = sorted({n["island_id"] for n in manifest["nodes"] if n["horizon"] == 4})
        for i, island in enumerate(islands):
            self.redactions[island] = f"ISLAND_{i}"
        self.islands = islands
        F.add("A0", "identity", "sequence manifest binds the sealed plan digest and the H4 config digest",
              PROVEN_SAFE if manifest["plan_sha256"] == spec["plan_sha256"]
              and manifest["optimization_config_sha256_by_horizon"]["4"] == spec["h4_config_sha256"] else RISK_UNRESOLVED,
              "config/MANIFEST.json", ["H4 identity"])
        config_path = self.campaign / "config/h4" / self.island / "doin_node.json"
        self.note_input(config_path)
        cfg = json.loads(config_path.read_text())
        oc = cfg["domains"][0]["optimization_config"]
        self.oc = oc
        plan = dict(oc["i6n_plan"])
        plan.pop("plan_sha256"), plan.pop("seal")
        seal = canonical_digest(plan)
        F.add("A1", "identity", "sealed search plan digest recomputes from the embedded plan (keys plan_sha256/seal excluded)",
              PROVEN_SAFE if seal == spec["plan_sha256"] else RISK_UNRESOLVED,
              f"recomputed {seal[:16]} declared {spec['plan_sha256'][:16]}", ["sealed plan identity"],
              dict(plan_version=oc["i6n_plan"]["plan_version"]))
        oc_digest = canonical_digest(oc)
        F.add("A2", "identity", "H4 scientific config digest = sha256 of canonical optimization_config",
              PROVEN_SAFE if oc_digest == spec["h4_config_sha256"] else RISK_UNRESOLVED, f"recomputed {oc_digest[:16]}",
              ["H4 identity"])
        same = []
        for island in islands:
            other = json.loads((self.campaign / "config/h4" / island / "doin_node.json").read_text())
            same.append(canonical_digest(other["domains"][0]["optimization_config"]) == oc_digest)
        active = self.campaign / "active" / self.island / "doin_node.json"
        active_ok = None
        if active.exists():
            self.note_input(active)
            active_ok = canonical_digest(json.loads(active.read_text())["domains"][0]["optimization_config"]) == oc_digest
        F.add("A3", "identity", "all four island configs and the live active config carry the same H4 optimization_config",
              PROVEN_SAFE if all(same) and active_ok else RISK_UNRESOLVED,
              f"islands equal={same} active equal={active_ok}", ["H4 identity"])
        self.node("plan", "sealed_plan", sha256=spec["plan_sha256"], version=oc["i6n_plan"]["plan_version"],
                  optimization_objective=oc["i6n_plan"]["fitness"]["optimization_objective"],
                  per_fold_rule=oc["i6n_plan"]["fitness"]["per_fold"])
        self.node("h4_config", "scientific_config", sha256=spec["h4_config_sha256"])
        self.edge("plan", "h4_config", artifact_path="config/h4/<island>/doin_node.json", sha256=spec["h4_config_sha256"],
                  schema="doin_node optimization_config", provenance_state=PROVEN_SAFE, producer_commit=spec["predictor_commit"])
        # folds
        self.folds = {}
        for fold in spec["folds"]:
            item = oc["i6n_folds"][fold]
            ctx = item["i6n_evaluation_context"]
            base = self.data / ctx["base_config"]["relpath"]
            frozen_path = self.data / ctx["frozen_support"]["relpath"]
            states = []
            for path, expected in ((base, ctx["base_config"]["sha256"]), (frozen_path, ctx["frozen_support"]["sha256"])):
                try:
                    verify_digest(path, expected)
                    states.append(True)
                except (DigestMismatch, OSError) as exc:
                    states.append(str(exc))
                self.note_input(path)
            frozen = json.loads(frozen_path.read_text())
            cell_cfg = json.loads(base.read_text())
            support_ok = frozen["support"] == item["i6n_support"]
            self.folds[fold] = dict(ctx=ctx, base=base, frozen_path=frozen_path, frozen=frozen, cell=cell_cfg,
                                    origins_sha256=item["i6n_origins_sha256"], support=item["i6n_support"])
            F.add(f"A5.{fold}", "identity", f"{fold}: cell config and frozen support carry the declared digests; frozen support equals the config binding",
                  PROVEN_SAFE if all(s is True for s in states) and support_ok else RISK_UNRESOLVED,
                  f"digests={states} support_equal={support_ok}", ["fold identity"])
            for key in ("input", "source_clock"):
                path = (base.parent / cell_cfg[key]["path"]).resolve()
                try:
                    verify_digest(path, cell_cfg[key]["sha256"])
                    ok = True
                except (DigestMismatch, OSError) as exc:
                    ok = str(exc)
                self.note_input(path)
                F.add(f"A6.{fold}.{key}", "identity", f"{fold}: cell {key} bytes carry the pinned digest",
                      PROVEN_SAFE if ok is True else RISK_UNRESOLVED, f"{self.r(str(path))} {ok}", ["fold identity"])
            self.node(f"cell:{fold}", "cell_config", sha256=ctx["base_config"]["sha256"], input_sha256=cell_cfg["input"]["sha256"],
                      samples=cell_cfg["samples"], spacing_hours=cell_cfg["spacing_hours"])
            self.node(f"frozen:{fold}", "frozen_support", sha256=ctx["frozen_support"]["sha256"], counts=frozen["support"]["counts"],
                      naive_sums=frozen["support"]["naive_sums"], support_digest=frozen["support"]["support_digest"])
        # source bytes
        prep = json.loads((self.data / "inputs/extended/PREPARATION.json").read_text())
        self.note_input(self.data / "inputs/extended/PREPARATION.json")
        missing = []
        for archive in prep["archives"]:
            found = any(Path(root).joinpath(archive["member"]).exists() for root in (self.campaign, self.data, self.state))
            if not found:
                missing.append(dict(member=archive["member"], sha256=archive["sha256"], year=archive["year"]))
            self.node(f"source:{archive['member']}", "source_archive", sha256=archive["sha256"], year=archive["year"],
                      present_locally=not any(m["member"] == archive["member"] for m in missing))
            self.edge(f"source:{archive['member']}", "input:extended", artifact_path=archive["member"], sha256=archive["sha256"],
                      schema="histdata minute csv in zip", row_identity="minute bar -> hourly close",
                      producer_path="tools/i6n_weekly_history_extension.py", producer_commit=spec["predictor_commit"],
                      provenance_state=RISK_UNRESOLVED if missing else PROVEN_SAFE)
        F.add("A7", "identity", "retained source bytes (annual minute archives) are present locally and verifiable",
              RISK_UNRESOLVED if missing else PROVEN_SAFE,
              "PREPARATION.json names six archives; none is present under the campaign, data or state roots",
              ["source provenance"], dict(missing=missing, note="price reconciliation is a narrative claim of PREPARATION.json; the bytes are not on this host"))
        self.prep = prep

    # -- B. data ---------------------------------------------------------------------------
    def phase_data(self):
        from tools.i6m_data import load_input
        from tools.i6n_clock_successor import corrected_timestamps
        F = self.findings
        ext = self.data / "inputs/extended"
        prim = self.data / "inputs/primary"
        clock = json.loads((ext / "CLOCK.json").read_text())
        self.note_input(ext / "CLOCK.json")
        self.features = list(clock["features"])
        try:
            ts, x, price = load_input(ext / "INPUT.npz", clock["input_sha256"])
            ok = True
        except ValueError as exc:
            ok = str(exc)
        F.add("B1", "data", "extended INPUT: digest verified, timestamps strictly increasing, hourly-aligned, no duplicates, no row >= 2025",
              PROVEN_SAFE if ok is True else RISK_UNRESOLVED, f"load_input: {ok}", ["source timestamps"],
              dict(rows=int(len(ts)), first=iso(ts[0]), last=iso(ts[-1]), semantics=clock["timestamp_semantics"],
                   availability_policy=clock["availability_policy"]))
        self.ts, self.x, self.price = ts, x, price
        pclock = json.loads((prim / "CLOCK.json").read_text())
        self.note_input(prim / "CLOCK.json")
        pts, px, pprice = load_input(prim / "INPUT.npz", pclock["input_sha256"])
        k = len(ts) - len(pts)
        suffix = bool(np.array_equal(ts[k:], pts) and np.array_equal(x[k:], px, equal_nan=True) and np.array_equal(price[k:], pprice))
        F.add("B2", "data", "primary INPUT is an exact suffix of the extended INPUT (same values, same clock)",
              PROVEN_SAFE if suffix else RISK_UNRESOLVED, f"extension_rows={k}", ["fold 2023Q4 input identity"])
        self.primary = (pts, px, pprice)
        # parent
        parent = self.state / self.spec["paths"]["parent_input"]
        parent_ok = None
        shift = {}
        try:
            verify_digest(parent, self.prep["source_input_sha256"])
            self.note_input(parent)
            with np.load(parent, allow_pickle=False) as z:
                ots, ox, oprice = z["ts"].copy(), z["x"].copy(), z["price"].copy()
            keep = ots >= epoch("2019-03-11T00:00:00+00:00")
            ots, ox, oprice = ots[keep], ox[keep], oprice[keep]
            corrected = corrected_timestamps(ots)
            parent_ok = bool(np.array_equal(corrected, ts) and np.array_equal(ox, x, equal_nan=True) and np.array_equal(oprice, price))
            delta = (corrected - ots) // HOUR
            shift = {str(int(k)): int(v) for k, v in zip(*np.unique(delta, return_counts=True))}
        except (DigestMismatch, OSError, ValueError) as exc:
            parent_ok = str(exc)
        F.add("B3", "data", "extended INPUT values (x, price) are byte-identical to the parent input after the documented clock correction; only timestamps moved later or stayed",
              PROVEN_SAFE if parent_ok is True else RISK_UNRESOLVED, f"parent {self.prep['source_input_sha256'][:16]} compare={parent_ok}",
              ["feature values", "clock conversion"], dict(shift_hours_histogram=shift, transform=self.prep["transform"]))
        self.node("input:parent", "parent_input", sha256=self.prep["source_input_sha256"], rows_used=int(keep.sum()) if parent_ok is True else None)
        self.node("input:extended", "campaign_input", sha256=clock["input_sha256"], rows=int(len(ts)), first=iso(ts[0]), last=iso(ts[-1]))
        self.node("input:primary", "campaign_input", sha256=pclock["input_sha256"], rows=int(len(pts)), first=iso(pts[0]), last=iso(pts[-1]))
        self.edge("input:parent", "input:extended", producer_path="tools/i6n_weekly_history_extension.py:materialize",
                  producer_commit=self.spec["object"]["predictor_commit"], artifact_path="inputs/extended/INPUT.npz",
                  sha256=clock["input_sha256"], schema="npz ts/x/price", row_identity="hourly bar end UTC",
                  min_source_timestamp=iso(ts[0]), max_source_timestamp=iso(ts[-1]), output_timestamp="= bar end",
                  availability_timestamp="AT_TIMESTAMP (declared)", provenance_state=PROVEN_SAFE if parent_ok is True else RISK_UNRESOLVED)
        self.edge("input:extended", "input:primary", artifact_path="inputs/primary/INPUT.npz", sha256=pclock["input_sha256"],
                  schema="npz ts/x/price", row_identity="exact suffix", provenance_state=PROVEN_SAFE if suffix else RISK_UNRESOLVED)

    # -- C. feature provenance ---------------------------------------------------------------
    def phase_features(self):
        F, spec = self.findings, self.spec
        design = json.loads((self.state / spec["paths"]["parent_design"]).read_text())
        self.note_input(self.state / spec["paths"]["parent_design"])
        freeze = json.loads((self.state / spec["paths"]["fs4_test_freeze"]).read_text())
        self.note_input(self.state / spec["paths"]["fs4_test_freeze"])
        plan = json.loads((self.state / spec["paths"]["fs4_weekly_plan"]).read_text())
        self.note_input(self.state / spec["paths"]["fs4_weekly_plan"])
        i6b = json.loads((self.state / spec["paths"]["i6b_manifest"]).read_text())
        self.note_input(self.state / spec["paths"]["i6b_manifest"])
        freeze_ok = freeze["freeze_sha256"] == canonical_digest({k: v for k, v in freeze.items() if k not in ("freeze_sha256", "frozen_utc")})
        winners = freeze["winners"]["EURUSD"]
        union = fs4_union(winners)
        equal = (sorted(self.features) == union == sorted(design["members"]) == sorted(i6b["selected_features"]))
        order_equal = list(self.features) == list(design["members"]) == list(i6b["selected_features"])
        sel_start = min(w["start"] for w in plan["weeks"])
        sel_end = max(w["end"] for w in plan["weeks"])
        F.add("C1", "feature_selection", "the 20 campaign features are exactly the union of the 14 EURUSD FS4 RAW winners (TEST_FREEZE), as declared by DESIGN members and the I6-B manifest",
              PROVEN_SAFE if equal and freeze_ok and i6b["freeze_sha256"] == freeze["freeze_sha256"] else RISK_UNRESOLVED,
              f"union={len(union)} order_equal={order_equal} freeze_seal_ok={freeze_ok}", ["feature identity"],
              dict(targets=sorted(winners), validation_year=plan["validation_year"], selection_arm=plan["selection_arm"],
                   selection_interval=[sel_start, sel_end], evaluation_mode=freeze["evaluation_mode"]))
        self.node("fs4:test_freeze", "feature_selector", sha256=freeze["freeze_sha256"], selection_split="VALIDATION",
                  selection_year=plan["validation_year"], selection_interval=[sel_start, sel_end],
                  winners=len(winners), union=len(union))
        self.node("features:20", "selected_features", members=self.features)
        self.edge("fs4:test_freeze", "features:20", producer_path="tools/fs4_weekly_campaign.py (FS4 weekly selection)",
                  artifact_path=spec["paths"]["fs4_test_freeze"], sha256=self.inputs_read.get(self.r(str(self.state / spec["paths"]["fs4_test_freeze"]))),
                  schema="fs4 TEST_FREEZE winners", row_identity="(target, RAW arm) winner set",
                  min_source_timestamp=sel_start, max_source_timestamp=sel_end,
                  fitting_interval=f"selection scored on VALIDATION weeks {sel_start}..{sel_end}; fits rolling 4y",
                  output_timestamp=freeze["frozen_utc"], availability_timestamp=sel_end, provenance_state=PROVEN_LEAK)
        # per feature x fold rows
        members_by_feature = {f: sorted(t for t, w in winners.items() if f in w["RAW"]["members"]) for f in self.features}
        for fold in spec["object"]["folds"]:
            cutoff = self.folds[fold]["frozen"]["splits"]["train"]["plan"]["splits"][0]["end"]  # inner_start
            fit_cutoff = self.folds[fold]["frozen"]["splits"]["inner"]["plan"]["splits"][0]["end"]
            state = selection_chronology(sel_end, fit_cutoff)
            for f in self.features:
                self.feature_rows.append(dict(feature=f, fold=fold, fold_fit_cutoff=fit_cutoff, train_inner_boundary=cutoff,
                                              selection_split="VALIDATION", selection_year=plan["validation_year"],
                                              selection_max_time=sel_end, winners_containing=";".join(members_by_feature[f]),
                                              chronology_state=state))
            F.add(f"C2.{fold}", "feature_selection", f"{fold}: every one of the 20 features was admitted by a selection that consumed outcomes up to {sel_end}, after the fold cutoff {fit_cutoff}",
                  state, "TEST_FREEZE winners x WEEKLY_PLAN weeks", ["unbiased estimate of the 2023 fold score", "scientific promotion"],
                  dict(classification="PROVEN_FUTURE_SELECTION_LEAK", reclassify_h4_score_as="RETROSPECTIVE_DEVELOPMENT_ONLY"))
        # transformations and availability
        self.phase("features.availability")
        bounds = spec["availability_bounds_hours_after_source_date_utc_midnight"]
        end_epoch = epoch(spec["object"]["test_boundary"])
        tol = spec["tolerances"]["transform_identification"]["abs"]
        vtol = spec["tolerances"]["vendor_identification"]["abs"]
        fx_cache = {}
        for f in self.features:
            src = spec["feature_sources"][f]
            col = self.x[:, self.features.index(f)]
            path = self.fin / src["path"]
            row = dict(feature=f, kind=src["kind"], formula=src["formula"], source_path=src["path"])
            if not path.exists():
                row.update(availability_state=RISK_UNRESOLVED, note="raw source file absent")
                self.availability_rows = getattr(self, "availability_rows", []) + [row]
                continue
            row["source_sha256"] = self.note_input(path)
            if src["kind"] == "fred":
                raw = _read_fred(path, end_epoch)
                series = [(d, raw[d]) for d in sorted(raw)]
                variants = [src["formula"]] if src["formula"] == "level" else [src["formula"], src["formula"] + "cal", src["formula"].split("_")[0] + "_7cal"]
                best = None
                for variant in variants:
                    derived = _derive(series, variant)
                    align = asof_alignment(self.ts, col, derived, abs_tol=tol)
                    if best is None or align["best_match_rate"] > best[1]["best_match_rate"]:
                        best = (variant, align, derived)
                lags = availability_lags(self.ts, col, best[2], abs_tol=0.0 if src["formula"] == "level" else tol)
                strict = bounds["strict"][f]
                row.update(formula_identified=best[0], asof=best[1], transition_lags={k: v for k, v in lags.items() if k != "lag_hours"},
                           strict_bound=strict, loose_bound=bounds["loose"],
                           availability_state=asof_state(best[1], strict), loose_state=asof_state(best[1], bounds["loose"]),
                           support=f"as-of forward fill, source date + {best[1]['best_lag_hours']} h" if best[1]["best_lag_hours"] is not None else None)
            elif src["kind"] == "yahoo":
                rows = _read_yahoo(path, end_epoch)
                best = None
                for variant, idx in (("Close", 1), ("Adj Close", 2)):
                    derived = _derive([(r[0], r[idx]) for r in rows], src["formula"])
                    align = asof_alignment(self.ts, col, derived, abs_tol=tol)
                    if best is None or align["best_match_rate"] > best[1]["best_match_rate"]:
                        best = (variant, align, derived)
                lags = availability_lags(self.ts, col, best[2], abs_tol=tol)
                strict = bounds["strict"][f]
                row.update(price_column=best[0], asof=best[1], transition_lags={k: v for k, v in lags.items() if k != "lag_hours"},
                           strict_bound=strict, loose_bound=bounds["loose"],
                           availability_state=asof_state(best[1], strict), loose_state=asof_state(best[1], bounds["loose"]),
                           support=f"trailing daily closes, as-of source date + {best[1]['best_lag_hours']} h" if best[1]["best_lag_hours"] is not None else None)
            else:  # fx hourly vendor identification
                if path not in fx_cache:
                    fx_cache[path] = _read_fx(path, end_epoch)
                vts, v = fx_cache[path]
                formulas = {}
                with np.errstate(all="ignore"):
                    logret = np.full(len(vts), np.nan)
                    logret[1:] = np.log(v["close"][1:] / v["close"][:-1])
                    formulas["logret_1h"] = logret
                    formulas["log_co"] = np.log(v["close"] / v["open"])
                    hl = v["high"] - v["low"]
                    formulas["close_loc"] = np.where(hl > 0, (v["close"] - v["low"]) / hl, np.nan)
                    sq = np.full(len(vts), np.nan)
                    for i in range(5, len(vts)):
                        w = logret[i - 4:i + 1]
                        sq[i] = math.sqrt(float(np.sum(w * w))) if np.isfinite(w).all() else np.nan
                    formulas["rv5_sqrt_sum_sq"] = sq
                    sd0 = np.full(len(vts), np.nan)
                    sd1 = np.full(len(vts), np.nan)
                    for i in range(5, len(vts)):
                        w = logret[i - 4:i + 1]
                        if np.isfinite(w).all():
                            sd0[i], sd1[i] = float(np.std(w)), float(np.std(w, ddof=1))
                    formulas["rv5_std_ddof0"], formulas["rv5_std_ddof1"] = sd0, sd1
                # price clock: which vendor datetime carries the campaign close (bar end) price
                clock_rates = {lag: _match_rate(self.ts, self.price, vts, v["close"], lag, 1e-9)[0] for lag in range(-6 * HOUR, 7 * HOUR, HOUR)}
                if f == "px.rv5":
                    candidates = {k: formulas[k] for k in ("rv5_sqrt_sum_sq", "rv5_std_ddof0", "rv5_std_ddof1")}
                elif src["formula"] in formulas:
                    candidates = {src["formula"]: formulas[src["formula"]]}
                else:
                    candidates = {}
                results = {}
                for name, values in candidates.items():
                    for lag in range(-6 * HOUR, 7 * HOUR, HOUR):
                        rate, n = _match_rate(self.ts, col, vts, values, lag, vtol)
                        results[f"{name}@{lag // HOUR:+d}h"] = dict(match_rate=rate, compared=n)
                best = max(results.items(), key=lambda kv: kv[1]["match_rate"]) if results else None
                price_lag = max(clock_rates.items(), key=lambda kv: kv[1])
                row.update(vendor_price_clock_match={str(k // HOUR): v for k, v in clock_rates.items()},
                           identification=results, best=best[0] if best else None,
                           best_match_rate=best[1]["match_rate"] if best else None)
                if best is None or best[1]["match_rate"] < 0.95:
                    row.update(availability_state=RISK_UNRESOLVED,
                               note="formula not reproduced from the independent vendor bars; producer code absent from every sibling repository")
                elif price_lag[1] < 0.5:
                    row.update(availability_state=RISK_UNRESOLVED,
                               note="formula reproduced but the vendor bar clock cannot be tied to the retained close (price match below 0.5 at every lag)")
                else:
                    feature_lag = int(best[0].split("@")[1].rstrip("h")) * HOUR
                    same_bar = feature_lag == price_lag[0]
                    row.update(feature_bar_lag_hours=feature_lag // HOUR, price_bar_lag_hours=price_lag[0] // HOUR,
                               availability_state=PROVEN_SAFE if same_bar or feature_lag < price_lag[0] else PROVEN_LEAK,
                               support="bar ending at t (same bar as the retained close)" if same_bar else f"bar offset {feature_lag // HOUR:+d}h relative to the retained close bar")
            self.availability_rows = getattr(self, "availability_rows", []) + [row]
            F.add(f"C3.{f}", "transformations", f"{f}: value at origin t never consumes an observation later than t (support interval and publication bound)",
                  row["availability_state"], f"{src['kind']} {src['formula']}", ["input causality"],
                  {k: row[k] for k in row if k in ("asof", "transition_lags", "formula_identified", "strict_bound", "loose_state", "best", "best_match_rate", "support", "note", "price_column", "vendor_price_clock_match")})
            self.node(f"feature:{f}", "selected_feature", source=src["path"], formula=src["formula"], availability_state=row["availability_state"])
            self.edge(f"raw:{src['path']}", f"feature:{f}", producer_path="PS1 feature builder (code not retained in any sibling repository)",
                      artifact_path="ps1/batch_00*/features_train.parquet", schema="parquet t_decision_utc + feature columns",
                      row_identity="hourly decision time", sha256=row.get("source_sha256"),
                      availability_timestamp=f"min lag {row.get('lags', {}).get('min')} h after source date" if row.get("lags") else row.get("support"),
                      provenance_state=row["availability_state"])
            self.edge(f"feature:{f}", "input:parent", provenance_state=PROVEN_SAFE if self.findings.rows and any(r["id"] == "B3" and r["state"] == PROVEN_SAFE for r in self.findings.rows) else RISK_UNRESOLVED,
                      row_identity="column of x", schema="float64")
        self.phase("features.done")

    # -- D. folds, masks, fitted state -------------------------------------------------------
    def phase_folds(self):
        from tools import i6n_historical_cell as cell, i6n_historical_evaluator as evaluator
        from tools.i6m_data import load_input
        F, spec = self.findings, self.spec
        self.prepared = {}
        calendar = json.loads((self.data / "inputs/extended/CALENDAR.json").read_text())
        self.note_input(self.data / "inputs/extended/CALENDAR.json")
        weeks = calendar["market_calendar"]["weeks"]
        year_weeks = [w for w in weeks if w["week_id"].startswith("2023") or w["week_id"].startswith("2022-12")]
        for fold in spec["object"]["folds"]:
            self.phase(f"folds.{fold}")
            info = self.folds[fold]
            cfg = cell.read_config(info["base"])
            data = cell.prepare(cfg, trajectory=True)
            self.prepared[fold] = (cfg, data)
            parts = data["parts"]
            frozen = info["frozen"]
            # D1 support reproduces
            support = evaluator.support(data, cfg, portable=True)
            F.add(f"D1.{fold}", "folds", f"{fold}: campaign code re-derives the frozen support (counts, naive sums, support digests) from the pinned inputs",
                  PROVEN_SAFE if support == frozen["support"] else RISK_UNRESOLVED, "evaluator.support == FROZEN.support", ["fold support"])
            # D2 cutoff recomputation
            quarter = int(fold[5])
            cutoff = _quarter_cutoff(weeks, 2023, quarter)
            fit_cutoff = int(parts.fit_cutoff.timestamp())
            F.add(f"D2.{fold}", "folds", f"{fold}: fold cutoff recomputed from the sealed calendar (close of the last week entirely inside the quarter) equals the frozen fit cutoff; TRAIN = 4 calendar years, INNER = 28 days",
                  PROVEN_SAFE if cutoff == fit_cutoff and parts.history_start == parts.fit_cutoff.replace(year=2019)
                  and parts.inner_start == parts.fit_cutoff - timedelta(days=28) else RISK_UNRESOLVED,
                  f"recomputed {iso(cutoff)} frozen {iso(fit_cutoff)}", ["fold boundaries"],
                  dict(history_start=parts.history_start.isoformat(), inner_start=parts.inner_start.isoformat(), fit_cutoff=parts.fit_cutoff.isoformat()))
            # D3 independent eligibility
            span = (cfg["samples"] - 1) * cfg["spacing_hours"]
            ts = data["ts"]
            hs, inner_start, cut = (int(parts.history_start.timestamp()), int(parts.inner_start.timestamp()), fit_cutoff)
            my_train, rej_train = _independent_eligibility(ts, weeks, hs, inner_start, 4, calendar["entry_delay_after_open_hours"],
                                                           calendar["force_flat_before_close_hours"], span)
            my_inner, rej_inner = _independent_eligibility(ts, weeks, inner_start, cut, 4, calendar["entry_delay_after_open_hours"],
                                                           calendar["force_flat_before_close_hours"], span)
            frozen_train = [epoch(k["origin"]) for k in frozen["splits"]["train"]["population"]["keys"]]
            frozen_inner = [epoch(k["origin"]) for k in frozen["splits"]["inner"]["population"]["keys"]]
            digests_ok = (origins_sha256(my_train) == info["origins_sha256"]["train"] and origins_sha256(my_inner) == info["origins_sha256"]["inner"])
            F.add(f"D3.{fold}", "folds", f"{fold}: an independent re-derivation of the MANAGEMENT endpoint mask (calendar, 12h entry delay, 12h forced flat, exact t and t+4h bars, purge at the TRAIN/INNER boundary) reproduces the frozen origin populations and the config origin digests exactly",
                  PROVEN_SAFE if my_train == frozen_train and my_inner == frozen_inner and digests_ok else RISK_UNRESOLVED,
                  f"train {len(my_train)}/{len(frozen_train)} inner {len(my_inner)}/{len(frozen_inner)} digests_ok={digests_ok}", ["business mask", "paired rows"],
                  dict(rejections_train=rej_train, rejections_inner=rej_inner))
            # D4 overlap, purge, force flat, weekend, DST
            train_targets = np.asarray(my_train) + 4 * HOUR
            inner_targets = np.asarray(my_inner) + 4 * HOUR
            overlap = bool(set(my_train) & set(my_inner))
            purge_ok = bool(train_targets.max() < inner_start) and bool(min(my_inner) >= inner_start) and bool(inner_targets.max() < cut)
            opens = np.asarray([epoch(w["market_open"]) for w in weeks], np.int64)
            closes = np.asarray([epoch(w["market_close"]) for w in weeks], np.int64)
            week_of = {}
            for t in my_inner + my_train:
                wi = int(np.searchsorted(opens, t, side="right")) - 1
                week_of[t] = (int(opens[wi]), int(closes[wi]))
            ff_ok = all(t + 4 * HOUR < week_of[t][1] - 12 * HOUR and t >= week_of[t][0] + 12 * HOUR for t in my_inner + my_train)
            crosses = sum(1 for t in my_inner + my_train if not (week_of[t][0] <= t + 4 * HOUR < week_of[t][1]))
            present = set(int(v) for v in ts)
            elapsed_ok = all(t + 4 * HOUR in present for t in my_inner + my_train)
            # counterexamples: origins in [force_flat-4h, force_flat) must be rejected
            counter = [t for (o, c) in {week_of[t] for t in my_inner} for t in range(c - 16 * HOUR, c - 12 * HOUR, HOUR)]
            counter_rejected = all(t not in set(my_inner) for t in counter)
            F.add(f"D4.{fold}", "folds", f"{fold}: TRAIN/INNER origins are disjoint; every TRAIN target precedes inner_start (purge) and every INNER target precedes the cutoff; every target is strictly before forced-flat and inside its own market week (no weekend gap counted as 4 market hours); every (t, t+4h) pair is an exact elapsed 14400 s with both bars observed; origins whose target would cross forced-flat are rejected",
                  PROVEN_SAFE if not overlap and purge_ok and ff_ok and crosses == 0 and elapsed_ok and counter_rejected else PROVEN_LEAK,
                  f"overlap={overlap} purge_ok={purge_ok} forced_flat_ok={ff_ok} week_crossings={crosses} elapsed_ok={elapsed_ok} forced_flat_counterexamples_rejected={counter_rejected} ({len(counter)})",
                  ["business mask", "weekend handling", "DST handling"],
                  dict(longest_input_history_hours=span, note="INNER windows read bars before inner_start (past information); no INNER label is used by any TRAIN row"))
            # D5 calendar vs data clock (session boundary deviations in 2023 weeks)
            deviations = []
            for w in year_weeks:
                o, c = epoch(w["market_open"]), epoch(w["market_close"])
                bars = ts[(ts > o) & (ts <= c)]
                if not len(bars):
                    continue
                first_bar, last_bar = int(bars[0]), int(bars[-1])
                if first_bar != o + HOUR or last_bar != c:
                    deviations.append(dict(week_id=w["week_id"], open=iso(o), first_bar=iso(first_bar), close=iso(c), last_bar=iso(last_bar)))
            F.add(f"D5.{fold}", "folds", f"{fold}: the sealed calendar's session boundaries agree with the retained bar clock in every 2023 week (first bar = open + 1 h, last bar = close)",
                  PROVEN_SAFE if not deviations else RISK_UNRESOLVED, f"{len(deviations)} deviating weeks", ["business mask under DST"],
                  dict(deviations=deviations[:12], note="a deviation means the mask was applied with a nominal calendar that differs from the data clock in that week; eligibility, not label causality"))
            # D6 fitted state: scaler and target scale from TRAIN only; perturbations
            raw_ts, raw_x, raw_price = load_input(cfg["input"]["path"], cfg["input"]["sha256"])
            fit = data["fit_indices"]
            scaler_sha = data["scaler"].sha256()
            inner_rows = np.flatnonzero((raw_ts >= inner_start) & (raw_ts < cut))
            later_rows = np.flatnonzero(raw_ts >= cut)
            fit_ts = raw_ts[fit]
            inner_check = fitted_state_check(raw_x, fit, inner_rows)
            later_check = fitted_state_check(raw_x, fit, later_rows)
            control = fitted_state_check(raw_x, fit, np.asarray(fit)[:5])
            labels_ok = True
            moved_price = raw_price.copy()
            moved_price[inner_rows] *= 1.5
            moved_price[later_rows] *= 1.5
            sel = np.array([int(t.timestamp()) for t, m in zip(parts.train.origins, parts.train.mask) if m[-1]], np.int64)
            labels, found = cell._head_labels(raw_ts, raw_price, sel, 4)
            labels2, _ = cell._head_labels(raw_ts, moved_price, sel, 4)
            labels_ok = bool(np.array_equal(labels, labels2)) and float(labels.mean()) == data["target_center"][-1] and float(labels.std()) == data["target_scale"][-1]
            F.add(f"D6.{fold}", "fitted_state", f"{fold}: feature scaler (median/mean/sd) and endpoint target center/scale are fitted on TRAIN endpoint rows only: perturbing all INNER values and all later-fold values leaves them byte-identical, perturbing TRAIN rows changes them (control)",
                  PROVEN_SAFE if inner_check["byte_identical"] and later_check["byte_identical"] and not control["byte_identical"] and labels_ok else PROVEN_LEAK,
                  f"scaler {scaler_sha[:16]} inner_identical={inner_check['byte_identical']} later_identical={later_check['byte_identical']} control_changed={not control['byte_identical']} labels_identical={labels_ok}",
                  ["preprocessing causality"],
                  dict(fit_rows=int(len(fit)), fit_interval=[iso(fit_ts.min()), iso(fit_ts.max())], first_scored_inner=iso(min(my_inner)),
                       first_scored_train=iso(min(my_train)), max_fit_time_lt_first_inner=bool(fit_ts.max() < min(my_inner)),
                       max_fit_time_lt_first_train=bool(fit_ts.max() < min(my_train))))
            F.add(f"D7.{fold}", "fitted_state", f"{fold}: for the TRAIN split, which enters the ranking at weight 0.5, the scaler and target scale consume TRAIN rows later than the scored origin (max_fit_time > first_scored_time)",
                  PROVEN_LEAK if fit_ts.max() >= min(my_train) else PROVEN_SAFE,
                  f"max_fit_time={iso(fit_ts.max())} first_scored_train={iso(min(my_train))}", ["ranking contract", "TRAIN score as out-of-sample"],
                  dict(note="in-sample normalization inside the declared TRAIN interval; not a future-of-fold leak, but it disqualifies the TRAIN half of the per-fold fitness as an out-of-sample quantity"))
            # D8 windows for INNER origins byte-identical under later-fold perturbation
            moved_x = raw_x.copy()
            moved_x[later_rows] += spec["mutation"]["sentinel"]
            from tools.i6e_compact_run import scaled_inputs
            scaled_moved, _ = scaled_inputs(np.where(np.isfinite(moved_x), moved_x, np.nan), fit)
            inner_origins = np.asarray(my_inner, np.int64)
            w0 = cell.windows(raw_ts, data["scaled"], inner_origins, cfg["spacing_hours"], cfg["samples"])
            w1 = cell.windows(raw_ts, scaled_moved, inner_origins, cfg["spacing_hours"], cfg["samples"])
            F.add(f"D8.{fold}", "fitted_state", f"{fold}: every INNER input window is byte-identical after perturbing all rows at or after the fold cutoff",
                  PROVEN_SAFE if np.array_equal(w0, w1) else PROVEN_LEAK, f"{len(inner_origins)} windows x {cfg['samples']} slots", ["input causality"])
            self.node(f"scaler:{fold}", "fitted_state", sha256=scaler_sha, fitting_interval=[iso(fit_ts.min()), iso(fit_ts.max())], rows=int(len(fit)))
            self.node(f"target_scale:{fold}", "fitted_state", center=data["target_center"][-1], scale=data["target_scale"][-1],
                      fitting_interval=[iso(sel.min()), iso(sel.max() + 4 * HOUR)])
            inp = "input:extended" if "extended" in cfg["input"]["path"] else "input:primary"
            self.edge(inp, f"scaler:{fold}", producer_path="tools/i6n_historical_cell.py:prepare -> tools/fs4_temporal_predictor.py:Standardiser.fit",
                      producer_commit=spec["object"]["predictor_commit"], sha256=scaler_sha, schema="median/mean/sd float32 arrays",
                      row_identity="TRAIN endpoint-eligible origins", min_source_timestamp=iso(fit_ts.min()), max_source_timestamp=iso(fit_ts.max()),
                      fitting_interval=[iso(fit_ts.min()), iso(fit_ts.max())], provenance_state=PROVEN_SAFE if inner_check["byte_identical"] and later_check["byte_identical"] else PROVEN_LEAK)
            self.edge(inp, f"target_scale:{fold}", producer_path="tools/i6n_historical_cell.py:prepare (_head_labels mean/std)", producer_commit=spec["object"]["predictor_commit"],
                      row_identity="TRAIN endpoint labels log(p[t+4h]/p[t])", min_source_timestamp=iso(sel.min()), max_source_timestamp=iso(sel.max() + 4 * HOUR),
                      fitting_interval=[iso(sel.min()), iso(sel.max() + 4 * HOUR)], provenance_state=PROVEN_SAFE if labels_ok else PROVEN_LEAK)
            self.edge(inp, f"frozen:{fold}", producer_path="tools/i6n_historical_evaluator.py:freeze", producer_commit=spec["object"]["predictor_commit"],
                      artifact_path=info["ctx"]["frozen_support"]["relpath"], sha256=info["ctx"]["frozen_support"]["sha256"], schema=frozen["schema"],
                      row_identity="C1 population keys (origin ISO) per split", min_source_timestamp=iso(hs), max_source_timestamp=iso(cut),
                      provenance_state=PROVEN_SAFE if my_train == frozen_train and my_inner == frozen_inner else RISK_UNRESOLVED)
            self.mask_parity[fold] = dict(frozen_vs_independent=dict(train=mask_parity([(t, t + 4 * HOUR) for t in frozen_train], [(t, t + 4 * HOUR) for t in my_train]),
                                                                   inner=mask_parity([(t, t + 4 * HOUR) for t in frozen_inner], [(t, t + 4 * HOUR) for t in my_inner])),
                                          config_origins_sha256=info["origins_sha256"],
                                          independent_origins_sha256=dict(train=origins_sha256(my_train), inner=origins_sha256(my_inner)))

    # -- E. archive census, metric replay, mask parity on retained rows ----------------------------
    def phase_archive(self):
        F, spec = self.findings, self.spec
        bodies = []
        for path in sorted(glob.glob(str(self.cas / "sha256/*/*/*"))):
            raw = Path(path).read_bytes()
            digest = Path(path).name
            if hashlib.sha256(raw).hexdigest() != digest:
                F.add(f"E0.{digest[:12]}", "archive", "CAS object bytes match their content address", RISK_UNRESOLVED, digest[:16], ["archive integrity"])
                continue
            try:
                payload = json.loads(raw.decode())
            except ValueError:
                continue
            pl = payload.get("payload", payload)
            if isinstance(pl, dict) and pl.get("schema") == "i6n.archive.island_candidate.v1":
                bodies.append((digest, pl))
        chain_path = self.campaign / "doin/h4" / self.island / "chain.db"
        self.note_input(chain_path)
        con = sqlite3.connect(f"file:{chain_path}?mode=ro", uri=True)
        refs = [json.loads(p)["body_digest"] for (p,) in con.execute("select payload from transactions where tx_type='candidate_evaluated'")]
        optimae = [(bi, json.loads(p).get("performance"), peer) for bi, p, peer in con.execute("select block_index,payload,peer_id from transactions where tx_type='optimae_accepted' order by block_index")]
        con.close()
        have = {d for d, _ in bodies}
        F.add("E1", "archive", "every candidate body referenced on the local chain is present in the local content-addressed store with a verified digest",
              PROVEN_SAFE if all(r in have for r in refs) else RISK_UNRESOLVED, f"{sum(r in have for r in refs)}/{len(refs)} referenced bodies present; {len(bodies)} candidate bodies",
              ["archive integrity"])
        frozen_support = {f: self.folds[f]["support"] for f in spec["object"]["folds"]}
        scaler_by_fold = {f: self.prepared[f][1]["scaler"].sha256() for f in spec["object"]["folds"]} if hasattr(self, "prepared") else {}
        center_by_fold = {f: self.prepared[f][1]["target_center"][-1] for f in spec["object"]["folds"]} if hasattr(self, "prepared") else {}
        census, bad_identity, bad_scaler, bad_center, bad_init, bad_stop, replay = [], [], [], [], [], [], {}
        champion = None
        for digest, pl in bodies:
            folds = [f["fold"] for f in pl["per_fold"]]
            identity_ok = (pl["plan_sha256"] == spec["object"]["plan_sha256"] and pl["horizon_hours"] == 4 and folds == spec["object"]["folds"]
                           and all(f["support_digest"] == frozen_support[f["fold"]]["support_digest"] and f["counts"] == frozen_support[f["fold"]]["counts"]
                                   and f["naive_sums"] == frozen_support[f["fold"]]["naive_sums"] for f in pl["per_fold"]))
            if not identity_ok:
                bad_identity.append(pl["genotype_canonical_sha256"][:12])
            inits = {f["receipt"]["metadata"]["training"]["initial_weights_sha256"] for f in pl["per_fold"]}
            restored = {f["receipt"]["metadata"]["training"]["restored_weights_sha256"] for f in pl["per_fold"]}
            iters = {f["receipt"]["metadata"]["training"]["initial_optimizer_iterations"] for f in pl["per_fold"]}
            if len(inits) != 1 or len(restored) != 4 or iters != {0}:
                bad_init.append(pl["genotype_canonical_sha256"][:12])
            for f in pl["per_fold"]:
                md = f["receipt"]["metadata"]
                if scaler_by_fold and md["feature_scaler_sha256"] != scaler_by_fold[f["fold"]]:
                    bad_scaler.append((pl["genotype_canonical_sha256"][:12], f["fold"]))
                if center_by_fold and md["target_center"][-1] != center_by_fold[f["fold"]]:
                    bad_center.append((pl["genotype_canonical_sha256"][:12], f["fold"], self.r(pl["island_id"]),
                                       abs(md["target_center"][-1] - center_by_fold[f["fold"]])))
                stop = md["candidate_receipt"]["stopping"]
                if stop["monitor"] != "train_inner_mean" or not stop["plan_conformant"] or "outer_metrics" in md or md["training"]["early_stop"]["rule"]["monitor"] != "train_inner_mean":
                    bad_stop.append((pl["genotype_canonical_sha256"][:12], f["fold"]))
            per_fold = {f["fold"]: fold_fitness(f["error_sums"], f["naive_sums"]) for f in pl["per_fold"]}
            agg = aggregate(per_fold.values())
            inner_only = aggregate(inner_only_fitness(f["error_sums"], f["naive_sums"]) for f in pl["per_fold"]) - 1
            train_only = aggregate(f["error_sums"]["train"] / f["naive_sums"]["train"] for f in pl["per_fold"]) - 1
            recorded = pl["fitness"]
            replay[pl["genotype_canonical_sha256"]] = dict(island=self.r(pl["island_id"]), body_sha256=digest, recorded_fitness=recorded,
                                                           replayed_fitness=agg - 1, exact_equal=compare_exact(recorded, agg - 1),
                                                           isclose_1e12=math.isclose(recorded, agg - 1, rel_tol=1e-12, abs_tol=1e-12),
                                                           per_fold_relative={k: v - 1 for k, v in per_fold.items()},
                                                           per_fold_recorded={f["fold"]: f["relative_error_vs_naive"] for f in pl["per_fold"]},
                                                           inner_only_relative=inner_only, train_only_relative=train_only,
                                                           inner_per_fold={f["fold"]: inner_only_fitness(f["error_sums"], f["naive_sums"]) - 1 for f in pl["per_fold"]},
                                                           output_mode=pl["effective_config"]["output_mode"], is_champion=pl["is_champion"])
            census.append(dict(genotype=pl["genotype_canonical_sha256"][:12], island=self.r(pl["island_id"]), fitness=recorded, inner_only=inner_only, output_mode=pl["effective_config"]["output_mode"]))
            if pl["genotype_canonical_sha256"] == spec["object"]["champion_genotype_sha256"]:
                champion = (digest, pl)
        F.add("E2", "archive", "every archived candidate binds the sealed plan, horizon 4, exactly the four frozen folds and the frozen support (counts, naive sums, digests)",
              PROVEN_SAFE if bodies and not bad_identity else RISK_UNRESOLVED, f"{len(bodies) - len(bad_identity)}/{len(bodies)}", ["ranking population identity"], dict(bad=bad_identity))
        F.add("E3", "archive", "every archived fold fit on every island used the fold's TRAIN-only feature scaler (digest equality with the local recomputation)",
              PROVEN_SAFE if bodies and not bad_scaler else RISK_UNRESOLVED, f"{len(bad_scaler)} mismatches over {4 * len(bodies)} fold fits", ["preprocessing causality across islands"], dict(bad=bad_scaler[:20]))
        F.add("E3b", "archive", "every archived fold fit's endpoint target center equals the local TRAIN-only recomputation exactly (cross-host float64 reduction; no tolerance declared, so inexact equality stays unresolved)",
              PROVEN_SAFE if bodies and not bad_center else RISK_UNRESOLVED,
              f"{len(bad_center)} inexact over {4 * len(bodies)} fold fits; max abs diff {max((b[3] for b in bad_center), default=0.0)!r}",
              ["inverse-transform parameters across islands"], dict(islands=sorted({b[2] for b in bad_center}), examples=bad_center[:8]))
        F.add("E4", "archive", "each candidate's four folds start from one identical initial weight digest (seeded build) with optimizer iterations 0 and end in four distinct restored checkpoints; no fold or candidate inherits weights",
              PROVEN_SAFE if bodies and not bad_init else RISK_UNRESOLVED, f"{len(bad_init)} violations", ["fold independence"], dict(bad=bad_init))
        F.add("E5", "archive", "early stopping monitored mean(TRAIN, INNER) at plan settings; no outer VALIDATION/TEST metric exists in any fold receipt",
              PROVEN_SAFE if bodies and not bad_stop else RISK_UNRESOLVED, f"{len(bad_stop)} violations", ["stopping contract"], dict(bad=bad_stop[:20]))
        exact = sum(1 for r in replay.values() if r["exact_equal"])
        close = sum(1 for r in replay.values() if r["isclose_1e12"])
        F.add("E6", "metrics", "each archived candidate's aggregate fitness replays from its retained per-fold error/naive sums as mean over folds of 0.5*(TRAIN ratio + INNER ratio) - 1",
              PROVEN_SAFE if replay and exact == len(replay) else RISK_UNRESOLVED, f"exact {exact}/{len(replay)}; isclose(1e-12) {close}/{len(replay)}", ["metric parity"])
        F.add("E7", "metrics", "the in-sample TRAIN split enters every per-fold fitness at weight 0.5 (plan per_fold rule); the ranking is not an INNER-only quantity",
              PROVEN_LEAK, "formula replayed on all archived candidates; plan fitness.per_fold", ["ranking contract", "unbiased estimate"],
              dict(note="not a temporal leak across the fold boundary; TRAIN error is in-sample (D7)"))
        self.metric_replay.update(census=census, replay=replay, chain_optimae=[dict(block=b, performance=p, peer=pe[:8]) for b, p, pe in optimae])
        if champion is None:
            F.add("E8", "champion", "the provisional champion genotype body is retained locally", RISK_UNRESOLVED, "no CAS body carries the genotype", ["champion"])
            return
        digest, pl = champion
        rep = replay[pl["genotype_canonical_sha256"]]
        chain_equal = any(p == spec["object"]["advertised_fitness"] for _, p, _ in optimae)
        F.add("E8", "champion", "champion fold scores and aggregate fitness replay from retained sums and equal the advertised value on the chain (metric parity)",
              PROVEN_SAFE if rep["exact_equal"] and rep["recorded_fitness"] == spec["object"]["advertised_fitness"] and chain_equal else RISK_UNRESOLVED,
              f"replayed {rep['replayed_fitness']!r} advertised {spec['object']['advertised_fitness']!r} chain_equal={chain_equal}", ["metric parity"], rep)
        artifacts = [f["receipt"]["metadata"]["artifact_directory"] for f in pl["per_fold"]]
        local = [Path(a).exists() for a in artifacts]
        F.add("E9", "champion", "champion paired rows (scores.npz, *.input.json, model.keras) are retained on this host for row-level replay",
              PROVEN_SAFE if all(local) else RISK_UNRESOLVED, f"{sum(local)}/4 fold artifact directories present locally",
              ["champion row-level parity", "champion causal validity"],
              dict(missing=[self.r(a) for a, ok in zip(artifacts, local) if not ok], island=self.r(pl["island_id"]),
                   c2_receipt_sha256={f["fold"]: f["receipt"]["metadata"]["c2_receipts"]["inner"]["sha256"] for f in pl["per_fold"]}))
        self.metric_replay["champion"] = dict(genotype=pl["genotype_canonical_sha256"], body_sha256=digest, island=self.r(pl["island_id"]),
                                              effective_config=pl["effective_config"], replay=rep,
                                              per_fold=[dict(fold=f["fold"], counts=f["counts"], error_sums=f["error_sums"], naive_sums=f["naive_sums"],
                                                             endpoint_metrics=f["receipt"]["metadata"]["endpoint_metrics"],
                                                             fit_cutoff=f["receipt"]["metadata"]["fit_cutoff"], best_epoch=f["receipt"]["metadata"]["training"]["best_epoch"],
                                                             initial_weights_sha256=f["receipt"]["metadata"]["training"]["initial_weights_sha256"],
                                                             restored_weights_sha256=f["receipt"]["metadata"]["training"]["restored_weights_sha256"],
                                                             feature_scaler_sha256=f["receipt"]["metadata"]["feature_scaler_sha256"])
                                                        for f in pl["per_fold"]],
                                              artifacts_local=local)
        self.node("champion", "candidate", genotype=pl["genotype_canonical_sha256"], island=self.r(pl["island_id"]), fitness=rep["recorded_fitness"])
        for f in pl["per_fold"]:
            md = f["receipt"]["metadata"]
            fid = f"fit:champion:{f['fold']}"
            self.node(fid, "model_fit", fold=f["fold"], restored_weights_sha256=md["training"]["restored_weights_sha256"],
                      initial_weights_sha256=md["training"]["initial_weights_sha256"], best_epoch=md["training"]["best_epoch"], device=md["device"])
            self.edge(f"frozen:{f['fold']}", fid, provenance_state=PROVEN_SAFE, row_identity="frozen support", sha256=self.folds[f["fold"]]["ctx"]["frozen_support"]["sha256"])
            self.edge(f"scaler:{f['fold']}", fid, provenance_state=PROVEN_SAFE, sha256=md["feature_scaler_sha256"])
            self.edge(fid, f"score:champion:{f['fold']}", producer_path="tools/i6n_historical_evaluator.py:score_receipt (C2)", producer_commit=spec["object"]["predictor_commit"],
                      artifact_path=self.r(md["artifact_directory"]), sha256=md["c2_receipts"]["inner"]["sha256"], schema="i6n.paired_metrics.result.v2",
                      row_identity="(origin, target=origin+4h) per split", fitting_interval=[md["history_start"], md["inner_start"]],
                      output_timestamp=md["fit_cutoff"], provenance_state=RISK_UNRESOLVED if not Path(md["artifact_directory"]).exists() else PROVEN_SAFE)
            self.node(f"score:champion:{f['fold']}", "fold_score", relative=f["relative_error_vs_naive"], counts=f["counts"], error_sums=f["error_sums"], naive_sums=f["naive_sums"])
            self.edge(f"score:champion:{f['fold']}", "fitness:champion", producer_path="tools/i6n_search_plan.py:score_folds", provenance_state=PROVEN_SAFE)
        self.node("fitness:champion", "aggregate_fitness", value=rep["recorded_fitness"], inner_only=rep["inner_only_relative"])
        self.edge("fitness:champion", "chain:optimae", provenance_state=PROVEN_SAFE if chain_equal else RISK_UNRESOLVED, artifact_path="doin/h4/<island>/chain.db")
        self.node("chain:optimae", "chain_record", accepted=len(optimae))

    def phase_local_rows(self):
        """Row-level replay and model/naive parity on every retained local candidate."""
        F, spec = self.findings, self.spec
        local = {}
        code_mismatch, predecessor_hits = set(), set()
        for fold in spec["object"]["folds"]:
            root = self.data / self.folds[fold]["ctx"]["output_root"]["relpath"] / "candidates"
            for cand in sorted(root.glob("*")):
                result_path = cand / "RESULT.json"
                if not result_path.exists():
                    continue
                status = json.loads((cand / "STATUS.json").read_text())
                if status.get("state") != "COMPLETED" or sha256_file(result_path) != status["result_sha256"]:
                    continue
                result = json.loads(result_path.read_text())
                receipt = result["receipt"]
                md = receipt["metadata"]
                known = spec.get("code_identity_known_predecessors", {})
                for rel, digest in result["identity"]["cell"]["code"].items():
                    path = REPO / rel
                    if (not path.exists() or sha256_file(path) != digest) and known.get(rel, {}).get("sha256") != digest:
                        code_mismatch.add(rel)
                for key, path in (("bridge_sha256", REPO / "tools/i6n_historical_evaluator.py"), ("search_sha256", REPO / "tools/i6n_search.py")):
                    rel = str(path.relative_to(REPO))
                    digest = result["identity"][key]
                    if sha256_file(path) != digest:
                        if known.get(rel, {}).get("sha256") == digest:
                            predecessor_hits.add(rel)
                        else:
                            code_mismatch.add(rel)
                genotype = md["candidate_receipt"]["genotype_canonical_sha256"]
                with np.load(cand / "scores.npz", allow_pickle=False) as z:
                    scores = {k: z[k].copy() for k in z.files if k.startswith(("train_", "inner_")) and "_h" not in k}
                rows = {}
                for split in ("train", "inner"):
                    doc = json.loads((cand / f"{split}.input.json").read_text())
                    model_rows = [(epoch(fc["origin"]), epoch(fc["target_timestamp"])) for fc in doc["forecasts"]]
                    naive_rows = [(int(t), int(t) + 4 * HOUR) for t in scores[f"{split}_origins"]]
                    pop_rows = [(epoch(k["origin"]), epoch(k["origin"]) + 4 * HOUR) for k in self.folds[fold]["frozen"]["splits"][split]["population"]["keys"]]
                    parity = mask_parity(model_rows, naive_rows)
                    parity["frozen_population_identical"] = model_rows == pop_rows
                    rp = replay_rows(scores[f"{split}_current"], scores[f"{split}_future"], scores[f"{split}_prediction"])
                    doc_pred = np.array([fc["prediction"] for fc in doc["forecasts"]])
                    doc_cur = np.array([fc["price_origin"] for fc in doc["forecasts"]])
                    doc_fut = np.array([fc["price_target"] for fc in doc["forecasts"]])
                    doc_rp = replay_rows(doc_cur, doc_fut, doc_pred)
                    rows[split] = dict(parity=parity, replay=rp, document_replay=doc_rp,
                                       scores_equal_document=bool(np.array_equal(doc_pred, scores[f"{split}_prediction"]) and np.array_equal(doc_fut, scores[f"{split}_future"]) and np.array_equal(doc_cur, scores[f"{split}_current"])),
                                       receipt_error_sum=receipt["error_sums"][split], receipt_naive_sum=receipt["naive_sums"][split],
                                       exact_error=compare_exact(rp["sum_abs_error"], receipt["error_sums"][split]),
                                       exact_naive=compare_exact(rp["naive_sum_abs_error"], receipt["naive_sums"][split]),
                                       isclose_error=math.isclose(rp["sum_abs_error"], receipt["error_sums"][split], rel_tol=1e-12, abs_tol=1e-12),
                                       target_minus_origin_unique=sorted({b - a for a, b in model_rows}),
                                       scale=doc["forecasts"][0]["scale"] if doc["forecasts"] else None, unit=doc["forecasts"][0]["unit"] if doc["forecasts"] else None)
                fitness = fold_fitness(receipt["error_sums"], receipt["naive_sums"])
                local[f"{fold}:{cand.name[:12]}"] = dict(fold=fold, candidate_dir=cand.name[:12], genotype=genotype, rows=rows,
                                                        fold_fitness_replayed=fitness, fold_fitness_recorded=receipt["fitness"],
                                                        exact_fitness=compare_exact(fitness, receipt["fitness"]), output_mode=md["candidate_receipt"]["output_mode"],
                                                        samples=md["candidate_receipt"]["samples"], delta_x=md["candidate_receipt"]["delta_x_hours"])
        self.local_candidates = local
        F.add("F0", "identity", "the code identity recorded in every retained local RESULT.json (cell, evaluator, search, contracts, model, training) equals the audited commit's files",
              PROVEN_SAFE if local and not code_mismatch else RISK_UNRESOLVED, f"{len(code_mismatch)} mismatching files; {len(predecessor_hits)} files match the declared predecessor commit", ["code provenance"],
              dict(mismatch=sorted(code_mismatch), predecessor=[dict(path=rel, **spec.get("code_identity_known_predecessors", {}).get(rel, {})) for rel in sorted(predecessor_hits)]))
        ok_parity = all(r["parity"]["identical"] and r["parity"]["frozen_population_identical"] and r["scores_equal_document"] and r["target_minus_origin_unique"] == [4 * HOUR]
                        and r["scale"] == "ORIGINAL_PRICE" for c in local.values() for r in c["rows"].values())
        F.add("F1", "mask_parity", "on every retained local fold fit, the model rows, the naive rows and the frozen C1 population are the same ordered (origin, origin+4h) identities, in original price scale",
              PROVEN_SAFE if local and ok_parity else (RISK_UNRESOLVED if not local else PROVEN_LEAK), f"{len(local)} local fold fits", ["paired rows", "scale"])
        ok_sums = all(r["exact_error"] and r["exact_naive"] for c in local.values() for r in c["rows"].values()) and all(c["exact_fitness"] for c in local.values())
        close_sums = all(r["isclose_error"] for c in local.values() for r in c["rows"].values())
        F.add("F2", "metrics", "on every retained local fold fit, endpoint error sums and naive sums replay exactly from the paired rows, and the fold fitness replays exactly from the sums",
              PROVEN_SAFE if local and ok_sums else RISK_UNRESOLVED, f"exact={ok_sums} isclose(1e-12)={close_sums}", ["metric parity"])
        self.mask_parity["local_candidates"] = {k: {s: v["rows"][s]["parity"] for s in v["rows"]} for k, v in local.items()}
        self.metric_replay["local_candidates"] = local

    # -- G. alignment cases with the retained model on CPU ----------------------------------------
    def phase_alignment(self):
        from tools import i6n_historical_cell as cell, i6n_historical_evaluator as evaluator
        from tools.i6m_data import load_input
        F, spec = self.findings, self.spec
        sentinel = spec["mutation"]["sentinel"]
        tol = spec["tolerances"]["forward_replay_gpu_to_cpu"]
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        tf.config.experimental.enable_op_determinism()
        import tools.i6e_masked_model  # noqa: F401  registers ClockView
        import tools.i6n_contracts.temporal_ops  # noqa: F401
        import predictor_plugins.modular_temporal.components  # noqa: F401
        chosen = []
        for fold in ("2023Q1_DCA", "2023Q4"):
            modes = {}
            for key, cand in self.local_candidates.items():
                if cand["fold"] == fold and cand["output_mode"] not in modes:
                    modes[cand["output_mode"]] = key
            chosen.extend(modes.values())
        if not chosen:
            F.add("G0", "alignment", "a retained local model is available for forward mutation checks", RISK_UNRESOLVED, "no local candidate", ["tensor causality"])
            return
        all_states = []
        for key in chosen:
            cand = self.local_candidates[key]
            fold = cand["fold"]
            root = self.data / self.folds[fold]["ctx"]["output_root"]["relpath"] / "candidates"
            cand_dir = next(p for p in root.glob(cand["candidate_dir"] + "*") if (p / "RESULT.json").exists())
            result = json.loads((cand_dir / "RESULT.json").read_text())
            effective = result["identity"]["effective_config"]
            base_cfg = cell.read_config(self.folds[fold]["base"])
            cfg = evaluator.candidate_config(base_cfg, effective)
            data = cell.prepare(cfg, trajectory=True)
            raw_ts, raw_x, raw_price = load_input(cfg["input"]["path"], cfg["input"]["sha256"])
            parts = data["parts"]
            offsets = data["offsets"]
            x, y, mask, origins = cell.materialize(data, parts.inner, cfg)
            batch = cfg["training"]["batch_size"]
            for i in range(0, len(x), batch):
                x[i:i + batch] = evaluator.transform_windows(x[i:i + batch], effective["input_transform"])
            model = tf.keras.models.load_model(cand_dir / "model.keras", compile=False)
            with np.load(cand_dir / "scores.npz", allow_pickle=False) as z:
                retained = dict(origins=z["inner_origins"].copy(), prediction=z["inner_prediction"].copy(), current=z["inner_current"].copy())
            weeks = json.loads((self.data / "inputs/extended/CALENDAR.json").read_text())["market_calendar"]["weeks"]
            opens = {epoch(w["market_open"]): w for w in weeks}
            week_open = sorted(opens)
            origin_list = [int(t) for t, m in zip(origins, mask[:, -1]) if m]
            cases = {}
            cases["ordinary"] = next((t for t in origin_list if datetime.fromtimestamp(t, timezone.utc).weekday() == 2 and datetime.fromtimestamp(t, timezone.utc).hour == 12), origin_list[len(origin_list) // 2])
            first_by_week, last_by_week = {}, {}
            for t in origin_list:
                wk = week_open[int(np.searchsorted(week_open, t, side="right")) - 1]
                first_by_week.setdefault(wk, t)
                last_by_week[wk] = t
            cases["week_open"] = first_by_week[sorted(first_by_week)[1]] if len(first_by_week) > 1 else min(origin_list)
            cases["pre_close"] = last_by_week[sorted(last_by_week)[0]]
            for label, day in (("dst_transition_us", "2023-03-12"), ("dst_transition_eu", "2023-03-26"), ("dst_transition_us_end", "2023-11-05")):
                after = epoch(day + "T00:00:00+00:00")
                hit = next((t for t in origin_list if t >= after), None)
                if hit is not None and hit - after < 3 * 86400:
                    cases[label] = hit
            missing = next((int(t) for t, m, row in zip(origins, mask[:, -1], x) if m and (row[:, -1] == 0).any()), None)
            if missing is not None:
                cases["missing_bar"] = missing
            cases["fold_boundary_first_inner"] = origin_list[0]
            train_origins = [int(t.timestamp()) for t, m in zip(parts.train.origins, parts.train.mask) if m[-1]]
            cases["fold_boundary_last_train"] = train_origins[-1]
            inner_index = {int(t): i for i, t in enumerate(origins)}

            def forward(window):
                return model(window[None].astype(np.float32), training=False).numpy()[0]

            def transform(values):  # the full pipeline applied to raw values with the fitted scaler held fixed
                s = data["scaler"].apply(np.where(np.isfinite(values), values, np.nan))
                out = np.empty((len(values), values.shape[1] * 2), np.float32)
                out[:, 0::2], out[:, 1::2] = s, np.isfinite(values)
                return out

            for label, t in cases.items():
                t = int(t)
                check = window_mutation_checks(raw_ts, raw_x, t, cfg["samples"], cfg["spacing_hours"], sentinel, transform=transform)
                stamps = check["window_timestamps"]
                # own window + model forward under the three mutations, batch size 1 each
                def pipeline_window(values):
                    w = cell.windows(raw_ts, transform(values), np.asarray([t], np.int64), cfg["spacing_hours"], cfg["samples"])
                    return evaluator.transform_windows(w, effective["input_transform"])[0]
                base_w = pipeline_window(raw_x)
                future = raw_x.copy(); future[raw_ts > t] += sentinel
                origin = raw_x.copy(); origin[raw_ts == t] += sentinel
                after_w, origin_w = pipeline_window(future), pipeline_window(origin)
                out_base, out_after, out_origin = forward(base_w), forward(after_w), forward(origin_w)
                tgt = target_mutation_check(raw_ts, raw_x, raw_price, t, cfg["samples"], cfg["spacing_hours"], 4)
                impulse = impulse_after_t(raw_ts, raw_x, t, transform, sentinel)
                traj = dict(output_offsets=list(offsets), offsets_are_1_2_3_4=list(offsets) == [1, 2, 3, 4] if len(offsets) == 4 else None,
                            endpoint_is_last_head=offsets[-1] == 4)
                parity = None
                if t in inner_index:
                    j = int(np.flatnonzero(retained["origins"] == t)[0]) if (retained["origins"] == t).any() else None
                    if j is not None:
                        mat_equal = bool(np.array_equal(x[inner_index[t]], base_w))
                        cpu_price = float(cell.price_prediction(np.asarray([retained["current"][j]]), np.asarray([out_base[-1]]), data["target_center"][-1], data["target_scale"][-1])[0])
                        parity = dict(materialized_window_equals_own_window=mat_equal, retained_prediction=float(retained["prediction"][j]), cpu_prediction=cpu_price,
                                      abs_diff=abs(cpu_price - float(retained["prediction"][j])),
                                      within_declared_diagnostic_tolerance=bool(abs(cpu_price - float(retained["prediction"][j])) <= tol["atol"] + tol["rtol"] * abs(float(retained["prediction"][j]))))
                case = dict(candidate=key, genotype=cand["genotype"], output_mode=cand["output_mode"], fold=fold, case=label, origin=t, origin_iso=iso(t),
                            window_timestamps=stamps, window_iso=[iso(s) for s in stamps], present=check["present"],
                            rightmost_is_origin=check["rightmost_is_origin"], others_before_origin=check["others_before_origin"], grid_exact=check["grid_exact"],
                            spacing_hours=cfg["spacing_hours"], samples=cfg["samples"],
                            input_tensor_unchanged_after_future_perturbation=bool(np.array_equal(base_w, after_w)),
                            model_output_unchanged_after_future_perturbation=bool(np.array_equal(out_base, out_after)),
                            rightmost_changes_when_origin_row_perturbed=bool(not np.array_equal(base_w[-1], origin_w[-1])) and bool(np.array_equal(base_w[:-1], origin_w[:-1])),
                            model_output_changes_when_origin_row_perturbed=bool(not np.array_equal(out_base, out_origin)),
                            target=tgt, impulse_after_t=impulse, trajectory=traj, retained_prediction_parity=parity,
                            transform_check_state=check["state"])
                leak = not (case["input_tensor_unchanged_after_future_perturbation"] and case["model_output_unchanged_after_future_perturbation"]
                            and impulse["rows_at_or_before_t_unchanged"] and tgt["input_unchanged"])
                safe = (case["rightmost_is_origin"] and case["others_before_origin"] and case["grid_exact"] and case["rightmost_changes_when_origin_row_perturbed"]
                        and case["model_output_changes_when_origin_row_perturbed"] and tgt["target_is_t_plus_4_elapsed_hours"] and tgt["label_changes"]
                        and (traj["offsets_are_1_2_3_4"] is not False) and traj["endpoint_is_last_head"])
                case["state"] = PROVEN_LEAK if leak else (PROVEN_SAFE if safe else RISK_UNRESOLVED)
                all_states.append(case["state"])
                self.alignment_cases.append(case)
            self.node(f"fit:local:{key}", "model_fit", fold=fold, genotype=cand["genotype"], samples=cfg["samples"], spacing_hours=cfg["spacing_hours"], offsets=list(offsets))
            self.edge(f"scaler:{fold}", f"fit:local:{key}", provenance_state=PROVEN_SAFE, artifact_path=self.r(str(cand_dir / "model.keras")), sha256=sha256_file(cand_dir / "model.keras"))
        worst = PROVEN_LEAK if PROVEN_LEAK in all_states else (RISK_UNRESOLVED if RISK_UNRESOLVED in all_states else PROVEN_SAFE)
        F.add("G1", "alignment", "for every alignment case (ordinary, week-open, pre-close, DST transitions, missing bar, fold boundaries) the rightmost input is exactly t, every other slot is earlier on the right-anchored delta_x grid, the H4 target is exactly t+14400 s, trajectory heads are +1,+2,+3,+4 h, perturbing any source row after t leaves the input tensor and the CPU model output byte-identical, perturbing row t changes only the rightmost slot and the output, and perturbing the target price changes only the label",
              worst, f"{len(self.alignment_cases)} cases on {len(chosen)} retained local models; exact equality", ["tensor/target causality"],
              dict(states={s: all_states.count(s) for s in set(all_states)}))
        parity_rows = [c["retained_prediction_parity"] for c in self.alignment_cases if c["retained_prediction_parity"]]
        self.metric_replay["forward_replay_diagnostic"] = dict(
            note="CPU forward replay of retained GPU models; declared diagnostic tolerance only, not serialization-justified; no verdict rests on it",
            within_tolerance=sum(1 for p in parity_rows if p["within_declared_diagnostic_tolerance"]), cases=len(parity_rows),
            rtol=tol["rtol"], atol=tol["atol"], max_abs_diff=max((p["abs_diff"] for p in parity_rows), default=None),
            materialized_window_equals_own_window=all(p["materialized_window_equals_own_window"] for p in parity_rows) if parity_rows else None)

    # -- H. outputs ---------------------------------------------------------------------------
    def write_outputs(self, terminal, shard_plan=None):
        out = self.out
        red = self.r
        with (out / "FEATURE_PROVENANCE.csv").open("w", newline="") as stream:
            fields = ["feature", "fold", "fold_fit_cutoff", "train_inner_boundary", "selection_split", "selection_year", "selection_max_time",
                      "winners_containing", "chronology_state", "transform_kind", "transform_formula", "availability_state", "availability_min_lag_hours",
                      "availability_strict_bound_hours", "support_interval", "source_sha256"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            avail = {r["feature"]: r for r in getattr(self, "availability_rows", [])}
            for row in self.feature_rows:
                a = avail.get(row["feature"], {})
                writer.writerow(dict(row, transform_kind=a.get("kind"), transform_formula=a.get("formula"), availability_state=a.get("availability_state"),
                                     availability_min_lag_hours=(a.get("asof") or {}).get("best_lag_hours"),
                                     availability_strict_bound_hours=a.get("strict_bound"), support_interval=a.get("support"), source_sha256=a.get("source_sha256")))
        with (out / "ALIGNMENT_CASES.jsonl").open("w") as stream:
            for case in self.alignment_cases:
                stream.write(red(json.dumps(jsonable(case), sort_keys=True, allow_nan=False)) + "\n")
        atomic_json(out / "MASK_PARITY.json", json.loads(red(json.dumps(jsonable(self.mask_parity), allow_nan=False))))
        atomic_json(out / "METRIC_REPLAY.json", json.loads(red(json.dumps(jsonable(self.metric_replay), allow_nan=False))))
        atomic_json(out / "TRACE_GRAPH.json", json.loads(red(json.dumps(jsonable(self.graph), allow_nan=False))))
        self.findings.to_csv(out / "FINDINGS.csv", red)
        atomic_json(out / "FEATURE_AVAILABILITY.json", json.loads(red(json.dumps(jsonable(getattr(self, "availability_rows", [])), allow_nan=False))))
        atomic_json(out / "INPUTS_READ.json", dict(schema=SCHEMA + ".inputs", inputs={red(k): v for k, v in self.inputs_read.items()}))
        extra = dict(terminal=terminal, worst=self.findings.worst())
        if shard_plan:
            extra["shard_plan"] = shard_plan
        self.status.write(terminal, phase="done", extra=extra)
        names = ["STATUS.json", "TRACE_GRAPH.json", "FEATURE_PROVENANCE.csv", "ALIGNMENT_CASES.jsonl", "MASK_PARITY.json", "METRIC_REPLAY.json",
                 "FINDINGS.csv", "FEATURE_AVAILABILITY.json", "INPUTS_READ.json", "SPEC.json"]
        lines = []
        for name in names:
            if (out / name).exists():
                lines.append(f"{sha256_file(out / name)}  {name}")
        for rel in ("tools/i6n_causal_trace_audit.py", "tests/test_i6n_causal_trace_audit.py"):
            lines.append(f"{sha256_file(REPO / rel)}  {rel}")
        for key, digest in sorted(self.inputs_read.items()):
            lines.append(f"{digest}  input:{key}")
        (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")

    def run(self):
        budget = self.spec["envelope"]["soft_phase_budget_seconds"]
        shard = None
        order = [("identity", self.phase_identity), ("data", self.phase_data), ("features", self.phase_features),
                 ("folds", self.phase_folds), ("archive", self.phase_archive), ("local_rows", self.phase_local_rows),
                 ("alignment", self.phase_alignment)]
        terminal = "COMPLETE"
        for position, (name, fn) in enumerate(order):
            elapsed = time.time() - self.status.started
            if elapsed > budget:
                remaining = [n for n, _ in order[position:]]
                shard = dict(reason="soft phase budget exhausted", elapsed_seconds=round(elapsed, 1), measured_phase_seconds=self.status.phases,
                             not_run=remaining, proposal="run each remaining phase as its own command with the same spec; no ETA is asserted")
                for n in remaining:
                    self.findings.add(f"SHARD.{n}", n, f"phase {n} executed inside the envelope", RISK_UNRESOLVED, "not run: envelope", ["all"])
                terminal = "INCOMPLETE_ENVELOPE"
                break
            self.phase(name)
            try:
                fn()
            except Exception as exc:  # fail closed: the phase's assertions stay unresolved
                self.findings.add(f"ERR.{name}", name, f"phase {name} completed without an unhandled error", RISK_UNRESOLVED,
                                  f"{type(exc).__name__}: {self.r(str(exc))[:300]}", ["all assertions of this phase"])
                terminal = "COMPLETE_WITH_PHASE_ERRORS"
        self.phase("write")
        self.write_outputs(terminal, shard)
        return terminal


def status(args):
    path = Path(args.out) / "STATUS.json"
    if not path.exists():
        print(json.dumps(dict(state="NOT_RUN")))
        return 3
    print(path.read_text())
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--campaign-root", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--cas-root", required=True)
    run.add_argument("--state-root", required=True)
    run.add_argument("--fin-root", required=True)
    run.add_argument("--island", required=True, help="control island id (redacted in every output)")
    st = sub.add_parser("status")
    st.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "status":
        return status(args)
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print("refusing: CUDA_VISIBLE_DEVICES must be the empty string (CPU only)", file=sys.stderr)
        return 2
    sys.dont_write_bytecode = True
    audit = Audit(args)
    audit.status.write("RUNNING", phase="start")
    terminal = audit.run()
    print(json.dumps(dict(state=terminal, counts=audit.findings.counts(), worst=audit.findings.worst())))
    return 0 if terminal == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
