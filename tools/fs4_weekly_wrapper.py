#!/usr/bin/env python3
"""Phase-4 weekly wrapper: BUSINESS weekly walk-forward comparison of candidate feature sets.

Adapts ``tools/fs_close_weekly.py`` (calendar, protocol, contract discipline) to the phase-4 plan
(section 4) without creating a second calendar: validation weeks come from
``fs_close_weekly.build_protocol``; the point-in-time window from ``tools.business_asof_window``;
the modular temporal predictor from ``tools/fs4_temporal_predictor.py``.

Per candidate set (one target, ordered members, phase-3 identity) and per eligible VALIDATION week:
FULL_RETRAIN_ROLLING_4Y on rows with ``available_time <= cutoff - purge`` (purge = target horizon),
internal early stopping on the chronological tail of the fit window, the same seed, a zero-return
naive on the IDENTICAL scored rows and scale, one disposition per week (COMPLETED or FAILED with
its reason; never a missing week), and the cost of every week. The winner per target is chosen by
the predeclared aggregate over ALL weeks (``AGGREGATE_RULE``) with the predeclared tie-break
(``TIE_RULE``). TEST is refused until a freeze record authorises exactly one traversal (FS4-12).

The literature static mode keeps its own identity in ``tools/fs_close_refit.py``
(``LITERATURE_STATIC_VALIDATION_DIAGNOSTIC``); this module refuses any other evaluation mode.

Unit of work for the controller (``tools/fs4_weekly_campaign.py``): ``run-task`` reads one task
JSON on stdin and writes one result JSON on stdout; logs go to stderr.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from tools import fs4_temporal_predictor as P  # noqa: E402
from tools import fs_close_weekly as W  # noqa: E402
from tools.business_asof_window import AsOfRow, SupportSpec, resolve_asof_window  # noqa: E402
from tools.business_weekly_protocol import EvaluationSplit  # noqa: E402
from tools.fs4_candidates import canonical, digest  # noqa: E402

UTC = dt.timezone.utc
PLAN_SCHEMA = "fs4.weekly_plan.v1"
TASK_SCHEMA = "fs4.weekly_task.v1"
RESULT_SCHEMA = "fs4.weekly_task_result.v1"
PRODUCTION_TRAINER = "R0_TEMPORAL_CONV1D"
BUSINESS_MODE = W.BUSINESS_MODE
UPDATE_MODE = W.UPDATE_MODE.value
SEED = 0
STAGE1_MODES = ("RAW",)
STAGE2_MODES = ("RANDOM_ENCODER", "TRAINED_ENCODER")
STAGE2_FAMILIES = ("SPEARMAN_CLUSTER", "MRMR", "JMI", "MRMR_CAUSAL", "JMI_CAUSAL", "UNIVARIATE_MI", "CAUSAL_SUPPORTED", "RANDOM_K",
                   "ALL_ADMISSIBLE")
STAGE2_TOP_N = 3
STAGE2_RULE_PATH = HERE.parent / "docs" / "fs4" / "STAGE2_RULE.md"
MIN_FIT_ROWS = 100
NAIVE_RULE = "zero log-return (prediction 0) on the identical scored rows, same target and raw_log_return scale"
SCORED_ROWS_RULE = ("rows whose decision time lies in [week.start, week.end) and whose bound target is finite; "
                    "every origin needs a full window history, otherwise the week FAILS with the count")
AGGREGATE_RULE = ("score(set, input_mode) = mean over ALL sealed VALIDATION weeks of the weekly skill_mae = "
                  "(naive_mae - mae) / naive_mae with the zero-return naive on the identical scored rows and scale; "
                  "never the best week; a set with any week not COMPLETED is ineligible and its dispositions stay in "
                  "the denominator; the feature-set winner per target is chosen on the RAW arm; the frozen encoder "
                  "arms (TRAINED_ENCODER, RANDOM_ENCODER control) are paired contrasts on identical rows and never "
                  "substitute the later R0/R1/R2 comparison")
TIE_RULE = "ties -> fewer features, then lower total fit seconds over all weeks, then set_id"


class Refusal(ValueError):
    """A contract violation that must not become a weekly result."""


# ----------------------------------------------------------------------------- helpers
def _iso(value: dt.datetime) -> str:
    return W._iso(value)


def _parse(text: str) -> dt.datetime:
    return W._parse(text)


def _now() -> str:
    return W._now()


def cgroup_peak_bytes() -> int | None:
    """Peak memory of this process's cgroup (the crispdm-run scope); None when unreadable."""
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        return int((Path("/sys/fs/cgroup") / rel.lstrip("/") / "memory.peak").read_text())
    except (OSError, IndexError, ValueError):
        return None


def _cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(r.ru_utime + r.ru_stime + c.ru_utime + c.ru_stime)


def _hex(d: str) -> str:
    return d.split(":", 1)[1] if d.startswith("sha256:") else d


def rows_digest(record_ids) -> str:
    return hashlib.sha256("|".join(str(r) for r in record_ids).encode()).hexdigest()


def code_sha256() -> str:
    h = hashlib.sha256()
    for f in (Path(__file__), HERE / "fs4_temporal_predictor.py", HERE / "fs_close_weekly.py", HERE / "business_asof_window.py"):
        h.update(f.read_bytes())
    return h.hexdigest()


def _week_dict(w) -> dict:
    return {"ordinal": w.ordinal, "split": w.split.value, "start": _iso(w.start), "end": _iso(w.end),
            "cutoff": _iso(w.cutoff), "fit_start": _iso(w.fit_start)}


def stage2_rule_sha256() -> str:
    return hashlib.sha256(STAGE2_RULE_PATH.read_bytes()).hexdigest()


def plan_digest(plan: dict) -> str:
    return digest({k: v for k, v in plan.items() if k not in ("plan_sha256", "built_utc")})


# ----------------------------------------------------------------------------- plan
def build_plan(consolidated: list[dict], seals: list[dict], *, validation_year: int, input_modes=("RAW",),
               bar_hours: dict, spec: P.PredictorSpec | None = None, encoder_spec: P.EncoderSpec | None = None,
               evaluation_mode: str = BUSINESS_MODE, extractibility_sha256: str | None = None,
               stage2_modes=STAGE2_MODES) -> dict:
    if evaluation_mode != BUSINESS_MODE:
        raise Refusal(f"EVALUATION_MODE_NOT_BUSINESS: {evaluation_mode!r}; LITERATURE_STATIC and monthly modes keep their "
                      "own identity (tools/fs_close_refit.py static diagnostic) and never enter this wrapper")
    input_modes = tuple(input_modes)
    if not input_modes or any(m not in P.INPUT_MODES for m in input_modes) or len(set(input_modes)) != len(input_modes):
        raise Refusal("INPUT_MODES_INVALID")
    spec = spec or P.PredictorSpec()
    encoder_spec = encoder_spec or P.EncoderSpec()
    seals_by_pop = {s["population_id"]: s for s in seals}
    populations = {}
    sets = []
    denominator = {}
    for cons in consolidated:
        pop = cons["population_id"]
        seal = seals_by_pop.get(pop)
        if seal is None or seal.get("inputs", {}).get("consolidated_sha256") != cons.get("consolidated_sha256"):
            raise Refusal(f"FRONTIER_SEAL_MISMATCH: {pop}")
        if pop not in bar_hours or type(bar_hours[pop]) is not int or bar_hours[pop] < 1:
            raise Refusal(f"BAR_HOURS_REQUIRED: {pop}")
        frontier = set(seal["frontier_set_ids"])
        populations[pop] = {"identity": cons["identity"], "consolidated_sha256": cons["consolidated_sha256"],
                            "phase3_closure_sha256": cons.get("phase3_closure_sha256"), "frontier_seal_sha256": seal["seal_sha256"],
                            "frontier_rule_sha256": seal["rule_sha256"], "extractibility_closure_sha256": seal["inputs"]["extractibility"]["closure_sha256"],
                            "bar_hours": bar_hours[pop]}
        denominator[pop] = dict(seal["denominator"])
        for s in cons["sets"]:
            if s["set_id"] in frontier:
                sets.append({k: s[k] for k in ("set_id", "population_id", "identity", "target_id", "horizon_hours", "members",
                                               "n_features", "methods", "control_methods", "phase3_unit_id")})
    if not sets:
        raise Refusal("NO_SETS_IN_FRONTIER")
    protocol = W.build_protocol(validation_year, "0" * 64)
    weeks = [_week_dict(w) for w in protocol.weeks() if w.split is EvaluationSplit.VALIDATION]
    test_weeks = [_week_dict(w) for w in protocol.weeks() if w.split is EvaluationSplit.TEST]
    plan = {
        "schema": PLAN_SCHEMA, "evaluation_mode": BUSINESS_MODE, "update_mode": UPDATE_MODE,
        "validation_year": validation_year, "rolling_calendar_years": 4, "seed": SEED,
        "input_modes": list(input_modes), "stage2_modes": list(stage2_modes), "stage2_rule_sha256": stage2_rule_sha256(),
        "stage2_families": list(STAGE2_FAMILIES), "stage2_top_n": STAGE2_TOP_N, "trainer": PRODUCTION_TRAINER,
        "predictor_spec": spec.to_dict(), "predictor_spec_sha256": spec.sha256(), "budget_sha256": P.budget_sha256(spec),
        "encoder_spec": encoder_spec.to_dict(), "encoder_spec_sha256": encoder_spec.sha256(),
        "populations": populations, "denominator": denominator,
        "sets": sorted(sets, key=lambda s: (s["population_id"], s["target_id"], s["set_id"])),
        "weeks": weeks, "test_weeks": test_weeks, "weeks_in_validation_year": len(weeks), "test_weeks_sealed": len(test_weeks),
        "support": {"purge": "target horizon hours", "input_lookback": f"{spec.window} rows; a fit origin needs all {spec.window} rows at or after fit_start", "inner_validation_weeks": 1,
                    "availability": "features available at the bar end (as-of columns from PS1); available_time == decision time"},
        "preprocessing": "fit-row median imputation and fit-row z-score per feature, refit every week",
        "naive": NAIVE_RULE, "scored_rows": SCORED_ROWS_RULE, "aggregate_rule": AGGREGATE_RULE, "tie_rule": TIE_RULE,
        "selection_arm": "RAW",
        "test": "EXTERNAL TEST (the following year) is sealed; opened once by the controller after TEST_FREEZE.json",
        "extractibility_closure_sha256": extractibility_sha256,
        "final_selection": False,
    }
    plan["plan_sha256"] = plan_digest(plan)
    plan["built_utc"] = _now()
    return plan


def enumerate_tasks(plan: dict, split: str = "validation", *, set_ids=None, test_authorization: str | None = None,
                    modes=None) -> list[dict]:
    if split not in ("validation", "test"):
        raise Refusal("SPLIT_INVALID")
    weeks = plan["weeks"] if split == "validation" else plan["test_weeks"]
    if split == "test" and not test_authorization:
        raise Refusal("TEST_SEALED: test tasks require the freeze digest")
    out = []
    for s in plan["sets"]:
        if set_ids is not None and s["set_id"] not in set_ids:
            continue
        for mode in (plan["input_modes"] if modes is None else modes):
            for w in weeks:
                payload = {"schema": TASK_SCHEMA, "plan_sha256": plan["plan_sha256"], "population_id": s["population_id"],
                           "identity": s["identity"], "set_id": s["set_id"], "target_id": s["target_id"],
                           "horizon_hours": s["horizon_hours"], "members": list(s["members"]), "n_features": s["n_features"],
                           "input_mode": mode, "stage": 1 if mode in plan["input_modes"] else 2, "split": split, "validation_year": plan["validation_year"],
                           "week": {k: w[k] for k in ("ordinal", "start", "end", "cutoff", "fit_start")}, "seed": plan["seed"],
                           "predictor_spec_sha256": plan["predictor_spec_sha256"], "encoder_spec_sha256": plan["encoder_spec_sha256"]}
                if split == "test":
                    payload["test_authorization"] = test_authorization
                payload["task_id"] = digest(payload)
                out.append(payload)
    return out


# ----------------------------------------------------------------------------- data
class DataStore:
    """TRAIN + VALIDATION rows of one population, sorted by decision time, with input digests."""

    META = ("t_decision_utc", "row_id")

    def __init__(self, population: str, names, X, ts, row_ids, targets: dict, digests: dict, bar_hours: int,
                 validation_first_read_utc: str):
        self.population = population
        self.names = list(names)
        self.col = {n: i for i, n in enumerate(self.names)}
        self.X = X
        self.ts = ts
        self.row_ids = row_ids
        self.record_ids = np.array([f"{int(r):012d}" for r in row_ids])
        self.rid_index = {r: i for i, r in enumerate(self.record_ids)}
        self.targets = targets
        self.digests = digests
        self.bar_hours = bar_hours
        self.validation_first_read_utc = validation_first_read_utc

    @staticmethod
    def _read(feature_files, targets_file, label):
        import pyarrow.parquet as pq

        X = None
        row_ids = None
        names = []
        digests = {}
        for fp in feature_files:
            fp = Path(fp)
            digests[f"{label}:{fp.name}@{fp.parent.name}"] = hashlib.sha256(fp.read_bytes()).hexdigest()
            pf = pq.ParquetFile(fp)
            ids = pf.read(columns=["row_id"]).column("row_id").to_numpy()
            if row_ids is None:
                row_ids = ids
            elif not np.array_equal(ids, row_ids):
                raise Refusal(f"ROW_ID_MISMATCH between feature batches: {fp}")
            cols = [n for n in pf.schema.names if n not in DataStore.META]
            block = np.column_stack([pf.read(columns=[n]).column(n).to_numpy(zero_copy_only=False).astype("float64", copy=False)
                                     for n in cols]) if cols else np.empty((len(ids), 0))
            X = block if X is None else np.hstack([X, block])
            names.extend(cols)
        tp = Path(targets_file)
        digests[f"{label}:{tp.name}"] = hashlib.sha256(tp.read_bytes()).hexdigest()
        tt = pq.read_table(tp)
        if not np.array_equal(tt.column("row_id").to_numpy(), row_ids):
            raise Refusal(f"ROW_ID_MISMATCH between features and targets: {tp}")
        tcol = tt.column("t_decision_utc")
        if "timestamp" not in str(tcol.type):
            raise Refusal("TARGETS_DECISION_TIME_MUST_BE_TIMESTAMP")
        unit = str(tcol.type)
        raw = tcol.cast("int64").to_numpy()
        ts = raw // (10**9 if "ns" in unit else 10**6 if "us" in unit else 10**3 if "ms" in unit else 1)
        import pyarrow as pa
        targets = {f.name: tt.column(f.name).to_numpy(zero_copy_only=False).astype("float64") for f in tt.schema
                   if f.name not in DataStore.META and f.name.startswith("Y")
                   and (pa.types.is_floating(f.type) or pa.types.is_integer(f.type))}
        if len(set(names)) != len(names):
            raise Refusal("DUPLICATE_FEATURE_COLUMNS")
        return names, X, ts.astype("int64"), row_ids.astype("int64"), targets, digests

    @classmethod
    def from_paths(cls, population: str, train_features, train_targets, val_features, val_targets, *, bar_hours: int) -> "DataStore":
        names_t, Xt, ts_t, ids_t, tg_t, dig_t = cls._read(train_features, train_targets, "train")
        validation_first_read_utc = _now()
        names_v, Xv, ts_v, ids_v, tg_v, dig_v = cls._read(val_features, val_targets, "validation")
        if set(tg_v) != set(tg_t):
            raise Refusal("TRAIN_AND_VALIDATION_TARGETS_DIFFER")
        # validation batches are split differently from the TRAIN file: align by NAME on the columns both hold; a plan member
        # missing from either side is refused per task (MEMBERS_NOT_IN_INPUTS), never silently dropped
        common = [n for n in names_t if n in set(names_v)]
        if not common:
            raise Refusal("TRAIN_AND_VALIDATION_SHARE_NO_FEATURE_COLUMNS")
        Xt = Xt[:, [names_t.index(n) for n in common]]
        Xv = Xv[:, [names_v.index(n) for n in common]]
        names_t = common
        X = np.vstack([Xt, Xv])
        ts = np.concatenate([ts_t, ts_v])
        row_ids = np.concatenate([ids_t, ids_v])
        targets = {n: np.concatenate([tg_t[n], tg_v[n]]) for n in tg_t}
        if len(set(row_ids.tolist())) != len(row_ids):
            raise Refusal("DUPLICATE_ROW_IDS across TRAIN and VALIDATION inputs")
        order = np.argsort(ts, kind="stable")
        return cls(population, names_t, X[order], ts[order], row_ids[order], {n: v[order] for n, v in targets.items()},
                   {**dig_t, **dig_v}, bar_hours, validation_first_read_utc)

    @classmethod
    def from_train_only(cls, population: str, train_features, train_targets, *, bar_hours: int) -> "DataStore":
        """Cost-pilot store: TRAIN files only, no VALIDATION byte is opened."""
        names, X, ts, ids, tg, dig = cls._read(train_features, train_targets, "train")
        order = np.argsort(ts, kind="stable")
        return cls(population, names, X[order], ts[order], ids[order], {n: v[order] for n, v in tg.items()}, dig, bar_hours, "NOT_READ_TRAIN_ONLY_PILOT")

    @property
    def max_ts(self) -> int:
        return int(self.ts[-1])

    def range_idx(self, start_iso: str, end_iso: str) -> np.ndarray:
        lo = np.searchsorted(self.ts, int(_parse(start_iso).timestamp()), side="left")
        hi = np.searchsorted(self.ts, int(_parse(end_iso).timestamp()), side="left")
        return np.arange(lo, hi)

    def row_ids_between(self, start_iso: str, end_iso: str) -> list[str]:
        return [str(r) for r in self.record_ids[self.range_idx(start_iso, end_iso)]]

    def target_values(self, target: str, record_ids) -> np.ndarray:
        return self.targets[target][[self.rid_index[r] for r in record_ids]]


# ----------------------------------------------------------------------------- trainer
def r0_trainer_factory(names, encoder_spec: P.EncoderSpec):
    def trainer(spec, X, y, fit_idx, inner_idx, input_mode, encoder, seed):
        return P.fit_named(spec, X, names, y, fit_idx, inner_idx, input_mode=input_mode, encoder=encoder, seed=seed)
    return trainer


def _encoder_for(input_mode: str, members, Xsub, store, task, encoder_spec: P.EncoderSpec, seed: int, results_root, extractor_code):
    if input_mode == "RAW":
        return None
    if results_root is None or extractor_code is None:
        raise Refusal("ENCODER_ARMS_NEED --runner-results and --extractor-code (the runner's retained terminals and pinned code)")
    try:
        return P.RunnerEncoderBank(encoder_spec, members, store.ts, Xsub, arm=input_mode, population_id=store.population,
                                   identity=task["identity"], results_root=results_root, code_dir=extractor_code, seed=seed)
    except P.Refusal as exc:
        raise Refusal(str(exc)) from exc


# ----------------------------------------------------------------------------- one task
def _failed(base: dict, reason: str, started: float) -> dict:
    out = dict(base)
    out.update({"disposition": "FAILED", "reason": reason, "metrics_absent_reason": reason,
                "cost": out.get("cost") or {"fit_seconds": 0.0, "wall_seconds": time.time() - started, "epochs": 0, "best_epoch": 0,
                                            "updates": 0, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "peak_cgroup_bytes": cgroup_peak_bytes(), "cpu_seconds": _cpu_seconds(),
                                            "n_params": 0}})
    out["result_sha256"] = digest({k: v for k, v in out.items() if k != "result_sha256"})
    return out


def run_task(task: dict, store: DataStore, *, trainer=None, test_freeze: dict | None = None, results_root=None, extractor_code=None,
             spec: P.PredictorSpec | None = None, encoder_spec: P.EncoderSpec | None = None) -> dict:
    started = time.time()
    if task.get("schema") != TASK_SCHEMA or not task.get("task_id"):
        raise Refusal("TASK_SCHEMA_INVALID")
    if task.get("population_id") != store.population:
        raise Refusal(f"POPULATION_MISMATCH: task {task.get('population_id')} store {store.population}")
    if task.get("seed") != SEED:
        raise Refusal("SEED_MISMATCH")
    split = task.get("split")
    if split == "test":
        if test_freeze is None or not test_freeze.get("freeze_sha256") or task.get("test_authorization") != test_freeze["freeze_sha256"]:
            raise Refusal("TEST_SEALED: no freeze record authorises this test task")
    elif split != "validation":
        raise Refusal("SPLIT_INVALID")
    spec = spec or P.PredictorSpec()
    encoder_spec = encoder_spec or P.EncoderSpec()
    if task.get("predictor_spec_sha256") != spec.sha256() or task.get("encoder_spec_sha256") != encoder_spec.sha256():
        raise Refusal("SPEC_MISMATCH: the task was planned with another predictor/encoder spec")
    protocol = W.build_protocol(int(task["validation_year"]), task["plan_sha256"])
    want = EvaluationSplit.VALIDATION if split == "validation" else EvaluationSplit.TEST
    week = next((w for w in protocol.weeks() if w.split is want and _iso(w.start) == task["week"]["start"]), None)
    if week is None or _week_dict(week) != {**task["week"], "split": split}:
        raise Refusal("WEEK_NOT_IN_CALENDAR")
    target = task["target_id"]
    if target not in store.targets:
        raise Refusal(f"TARGET_NOT_IN_INPUTS: {target}")
    members = P.canonical_features(task["members"])
    missing = [m for m in members if m not in store.col]
    if missing:
        raise Refusal(f"MEMBERS_NOT_IN_INPUTS: {missing[:5]}")
    horizon = int(task["horizon_hours"])
    purge = dt.timedelta(hours=horizon)
    support = SupportSpec(input_lookback=dt.timedelta(0), target_horizon=purge,
                          maximum_holding_support=dt.timedelta(0), inner_validation_weeks=1)
    y = store.targets[target]
    finite = np.isfinite(y)
    lo = np.searchsorted(store.ts, int(week.fit_start.timestamp()), side="left")
    hi = np.searchsorted(store.ts, int((week.cutoff - purge).timestamp()), side="right")
    rows = []
    # an origin needs a full window of `spec.window` rows inside the rolling four years: no input row predates fit_start
    for i in range(lo + spec.window - 1, hi):
        if not finite[i]:
            continue
        et = dt.datetime.fromtimestamp(int(store.ts[i]), UTC)
        rows.append(AsOfRow(record_id=str(store.record_ids[i]), event_time=et, available_time=et, target_available_time=et + purge,
                            row_digest=hashlib.sha256(f"{store.record_ids[i]}|{int(store.ts[i])}".encode()).hexdigest()))
    cols = [store.col[m] for m in members]
    Xsub = store.X[:, cols]
    base = {
        "schema": RESULT_SCHEMA, "status": "COMPLETE", "task_id": task["task_id"], "plan_sha256": task["plan_sha256"],
        "population_id": store.population, "identity": task["identity"], "set_id": task["set_id"], "target_id": target,
        "horizon_hours": horizon, "members": list(members), "n_features": len(members), "input_mode": task["input_mode"],
        "split": split, "week_start": task["week"]["start"], "week_end": task["week"]["end"], "cutoff": task["week"]["cutoff"],
        "fit_start": task["week"]["fit_start"], "seed": SEED, "evaluation_mode": BUSINESS_MODE, "update_mode": UPDATE_MODE,
        "trainer": PRODUCTION_TRAINER if trainer is None else "INJECTED_STAND_IN_TEST_ONLY",
        "predictor_spec_sha256": spec.sha256(), "budget_sha256": P.budget_sha256(spec), "encoder_spec_sha256": encoder_spec.sha256(),
        "input_sha256": digest({"files": store.digests, "members": list(members), "target": target, "input_mode": task["input_mode"]}),
        "code_sha256": code_sha256(), "validation_first_read_utc": store.validation_first_read_utc,
        "naive": {"rule": NAIVE_RULE, "scale": "raw_log_return", "unit": "log_return"},
    }
    try:
        window = resolve_asof_window(week, rows, support)
    except Exception as exc:
        return _failed(base | {"n_scored": 0, "rows_sha256": rows_digest([]), "fit_rows": 0, "inner_rows": 0, "purged_rows": 0,
                               "fit_population_digest": "", "fit_max_event_time": task["week"]["cutoff"]},
                       f"as-of resolution failed: {type(exc).__name__}: {exc}", started)
    fit_idx = np.array([store.rid_index[r.record_id] for r in window.fit_rows], dtype="int64")
    inner_idx = np.array([store.rid_index[r.record_id] for r in window.inner_validation_rows], dtype="int64")
    base.update({"fit_rows": int(fit_idx.size), "inner_rows": int(inner_idx.size), "purged_rows": int(window.purged_count),
                 "fit_population_digest": _hex(window.fit_population_digest), "inner_population_digest": _hex(window.inner_validation_population_digest),
                 "fit_max_event_time": _iso(max(r.event_time for r in window.fit_rows)) if window.fit_rows else task["week"]["cutoff"],
                 "fit_end": _iso(window.fit_end), "inner_validation_start": _iso(window.inner_validation_start),
                 "inner_validation_end": _iso(window.inner_validation_end)})
    scored_all = store.range_idx(task["week"]["start"], task["week"]["end"])
    scored = scored_all[finite[scored_all]]
    scored_ids = [str(r) for r in store.record_ids[scored]]
    base.update({"n_scored": int(scored.size), "rows_sha256": rows_digest(scored_ids), "n_rows_in_week": int(scored_all.size)})
    base["naive"]["rows_sha256"] = base["rows_sha256"]
    if fit_idx.size < MIN_FIT_ROWS or inner_idx.size == 0:
        return _failed(base, f"too few fit rows: fit {fit_idx.size} inner {inner_idx.size}", started)
    if scored.size == 0:
        return _failed(base, "no scored row with a finite target in the week", started)
    standardiser = P.Standardiser.fit(Xsub[fit_idx])
    base["standardiser_sha256"] = standardiser.sha256()
    try:
        encoder = _encoder_for(task["input_mode"], members, Xsub, store, task, encoder_spec, SEED, results_root, extractor_code)
    except Refusal as exc:
        if "RUNNER_RESULT_MISSING" in str(exc):     # a member without a runner terminal: a typed disposition, never dropped
            return _failed(base, f"ENCODER_NOT_AVAILABLE: {exc}", started)
        raise
    fit = trainer or r0_trainer_factory(list(members), encoder_spec)
    t0 = time.time()
    try:
        rep = fit(spec, Xsub, y, fit_idx, inner_idx, task["input_mode"], encoder, SEED)
    except P.Refusal as exc:
        return _failed(base, f"trainer refused: {exc}", started)
    fit_seconds = time.time() - t0
    pred = np.asarray(rep.predict(Xsub, scored), dtype="float64").reshape(-1)
    yt = y[scored]
    cost = {"fit_seconds": float(fit_seconds), "wall_seconds": float(time.time() - started), "epochs": int(rep.epochs_run),
            "best_epoch": int(rep.best_epoch), "updates": int(rep.updates), "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "peak_cgroup_bytes": cgroup_peak_bytes(), "cpu_seconds": _cpu_seconds(), "n_params": int(rep.n_params)}
    base.update({"model_sha256": rep.weights_sha256, "architecture_sha256": rep.architecture_sha256,
                 "encoder_sha256": rep.encoder_sha256, "trainer_budget_sha256": rep.budget_sha256, "cost": cost})
    if not np.all(np.isfinite(pred)):
        return _failed(base, f"prediction unavailable for {int(np.sum(~np.isfinite(pred)))} of {pred.size} origins (insufficient history)", started)
    err = pred - yt
    naive_mae = float(np.mean(np.abs(yt)))
    metrics = {"mae": float(np.mean(np.abs(err))), "mse": float(np.mean(err ** 2)), "naive_mae": naive_mae, "naive_mse": float(np.mean(yt ** 2))}
    base.update({"disposition": "COMPLETED", "reason": None, "metrics": metrics,
                 "skill_mae": (naive_mae - metrics["mae"]) / naive_mae if naive_mae > 0 else None,
                 "beats_naive": metrics["mae"] < naive_mae})
    base["result_sha256"] = digest({k: v for k, v in base.items() if k != "result_sha256"})
    return base


# ----------------------------------------------------------------------------- aggregation
def aggregate_results(results: list[dict], weeks: list[str], *, sets_n_features: dict) -> dict:
    """Per set: the mean over ALL sealed weeks; any non-COMPLETED week makes the set ineligible (kept in the denominator)."""
    by_set: dict[str, dict[str, dict]] = {}
    for r in results:
        if r.get("status") != "COMPLETE":
            continue
        by_set.setdefault(r["set_id"], {})[r["week_start"]] = r
    out = {}
    for set_id, per_week in by_set.items():
        dispositions = []
        skills, maes, naives, fit_seconds = [], [], [], 0.0
        for w in weeks:
            r = per_week.get(w)
            if r is None:
                dispositions.append({"week_start": w, "disposition": "MISSING", "reason": "no terminal result"})
                continue
            fit_seconds += float(r.get("cost", {}).get("fit_seconds", 0.0))
            if r["disposition"] == "COMPLETED":
                skills.append(r["skill_mae"] if r.get("skill_mae") is not None else float("nan"))
                maes.append(r["metrics"]["mae"])
                naives.append(r["metrics"]["naive_mae"])
                dispositions.append({"week_start": w, "disposition": "COMPLETED", "reason": None, "skill_mae": r.get("skill_mae"),
                                     "mae": r["metrics"]["mae"], "naive_mae": r["metrics"]["naive_mae"], "n_scored": r.get("n_scored"),
                                     "fit_seconds": r.get("cost", {}).get("fit_seconds")})
            else:
                dispositions.append({"week_start": w, "disposition": r["disposition"], "reason": r.get("reason"), "n_scored": r.get("n_scored")})
        completed = sum(1 for d in dispositions if d["disposition"] == "COMPLETED")
        sk = np.array(skills, dtype="float64")
        out[set_id] = {
            "set_id": set_id, "weeks": len(weeks), "weeks_completed": completed, "eligible": completed == len(weeks) and len(weeks) > 0,
            "n_features": int(sets_n_features.get(set_id, 0)),
            "mean_weekly_skill_mae": float(np.nanmean(sk)) if sk.size else None,
            "std_weekly_skill_mae": float(np.nanstd(sk)) if sk.size else None,
            "mean_model_mae": float(np.mean(maes)) if maes else None, "mean_naive_mae": float(np.mean(naives)) if naives else None,
            "weeks_beating_naive": int(np.sum(sk > 0)) if sk.size else 0,
            "beats_naive_on_aggregate": bool(maes) and float(np.mean(maes)) < float(np.mean(naives)),
            "fit_seconds_total": float(fit_seconds), "dispositions": dispositions,
        }
    return out


def choose_winner(aggregates: dict) -> dict | None:
    eligible = [a for a in aggregates.values() if a["eligible"] and a["mean_weekly_skill_mae"] is not None
                and np.isfinite(a["mean_weekly_skill_mae"])]
    if not eligible:
        return None
    ranked = sorted(eligible, key=lambda a: (-a["mean_weekly_skill_mae"], a["n_features"], a["fit_seconds_total"], a["set_id"]))
    best = ranked[0]
    tied = [a for a in ranked[1:] if a["mean_weekly_skill_mae"] == best["mean_weekly_skill_mae"]]
    tie_break = "none"
    if any(a["n_features"] > best["n_features"] for a in tied):
        tie_break = "fewer_features"
    elif any(a["n_features"] == best["n_features"] and a["fit_seconds_total"] > best["fit_seconds_total"] for a in tied):
        tie_break = "lower_cost"
    elif tied:
        tie_break = "set_id"
    return {"set_id": best["set_id"], "score": best["mean_weekly_skill_mae"], "n_features": best["n_features"],
            "fit_seconds_total": best["fit_seconds_total"], "weeks": best["weeks"], "weeks_completed": best["weeks_completed"],
            "beats_naive_on_aggregate": best["beats_naive_on_aggregate"], "tie_break": tie_break,
            "aggregate_rule": AGGREGATE_RULE, "tie_rule": TIE_RULE, "ranked": [a["set_id"] for a in ranked]}


def stage2_set_ids(plan: dict, raw_aggregates: dict) -> dict:
    """The stage-2 list, computed mechanically from the stage-1 RAW aggregates (STAGE2_RULE.md).

    raw_aggregates: population -> target -> {set_id: aggregate} as produced by ``aggregate_results``.
    """
    sets = {s["set_id"]: s for s in plan["sets"]}
    chosen: dict[str, list] = {}
    for pop, targets in sorted(raw_aggregates.items()):
        for target, agg in sorted(targets.items()):
            for fam in plan["stage2_families"]:
                cand = [a for a in agg.values() if fam in sets[a["set_id"]]["methods"] and a["eligible"]
                        and a["mean_weekly_skill_mae"] is not None and np.isfinite(a["mean_weekly_skill_mae"])]
                cand.sort(key=lambda a: (-a["mean_weekly_skill_mae"], a["n_features"], a["fit_seconds_total"], a["set_id"]))
                for rank, a in enumerate(cand[:plan["stage2_top_n"]], 1):
                    chosen.setdefault(a["set_id"], []).append({"population_id": pop, "target_id": target, "family": fam, "rank": rank})
    for sid, s in sets.items():
        if "ALL_ADMISSIBLE" in s["methods"]:
            chosen.setdefault(sid, []).append({"population_id": s["population_id"], "target_id": s["target_id"],
                                               "family": "ALL_ADMISSIBLE", "rank": 0, "reason": "reference arm, always included"})
    body = {"stage2_rule_sha256": plan["stage2_rule_sha256"], "plan_sha256": plan["plan_sha256"],
            "set_ids": sorted(chosen), "reasons": {k: chosen[k] for k in sorted(chosen)},
            "source_aggregates_sha256": digest({p: {t: {k: {x: v[x] for x in ("mean_weekly_skill_mae", "weeks_completed", "eligible", "n_features", "fit_seconds_total")}
                                                        for k, v in a.items()} for t, a in tt.items()} for p, tt in raw_aggregates.items()})}
    body["list_sha256"] = digest(body)
    return body


def encoder_comparison(aggregates_by_mode: dict, raw_mode: str = "RAW") -> dict:
    """Per set: RAW vs TRAINED vs RANDOM mean weekly skill on identical weeks; RANDOM is the control."""
    out = {}
    raw = aggregates_by_mode.get(raw_mode, {})
    for sid, a_raw in raw.items():
        row = {"raw": a_raw["mean_weekly_skill_mae"]}
        for mode in STAGE2_MODES:
            a = aggregates_by_mode.get(mode, {}).get(sid)
            row[mode.lower()] = a["mean_weekly_skill_mae"] if a else None
            row[mode.lower() + "_weeks_completed"] = a["weeks_completed"] if a else None
        if row.get("trained_encoder") is not None and row["raw"] is not None:
            row["trained_minus_raw"] = row["trained_encoder"] - row["raw"]
        if row.get("trained_encoder") is not None and row.get("random_encoder") is not None:
            row["trained_minus_random"] = row["trained_encoder"] - row["random_encoder"]
        if len(row) > 1 and any(k in row for k in ("trained_minus_raw", "trained_minus_random")):
            out[sid] = row
    return out


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run-task", help="stdin: one task JSON (controller claim); stdout: one result JSON")
    run.add_argument("--population", required=True)
    run.add_argument("--train-features", nargs="+", required=True)
    run.add_argument("--train-targets", required=True)
    run.add_argument("--task-file", help="claim JSON file (else $FS4_CLAIM_JSON, else stdin)")
    run.add_argument("--val-features", nargs="+")
    run.add_argument("--val-targets")
    run.add_argument("--bar-hours", type=int, required=True)
    run.add_argument("--trainer", choices=(PRODUCTION_TRAINER,), default=PRODUCTION_TRAINER)
    run.add_argument("--runner-results", help="directory of the phase-4 runner's retained <task_id>/result.json + chosen.weights.h5 (encoder arms)")
    run.add_argument("--extractor-code", help="pinned feature-extractor checkout (the runner's builders), encoder arms only")
    run.add_argument("--pilot-train-only", action="store_true", help="cost pilot: read TRAIN only; refuses any week ending after TRAIN")
    run.add_argument("--test-freeze", help="TEST_FREEZE.json; required only for test-split tasks")
    a = ap.parse_args(argv)
    # crispdm-run gives its child /dev/null as stdin: the claim also comes from --task-file or $FS4_CLAIM_JSON
    import os
    if a.task_file:
        task = json.loads(Path(a.task_file).read_text())
    elif os.environ.get("FS4_CLAIM_JSON"):
        task = json.loads(os.environ["FS4_CLAIM_JSON"])
    else:
        task = json.load(sys.stdin)
    if a.pilot_train_only:
        store = DataStore.from_train_only(a.population, a.train_features, a.train_targets, bar_hours=a.bar_hours)
        if _parse(task["week"]["end"]).timestamp() > store.max_ts:
            print(canonical({"error": "PILOT_WEEK_ENDS_AFTER_TRAIN", "task_id": task.get("task_id")}), file=sys.stderr)
            return 2
    elif not (a.val_features and a.val_targets):
        print(canonical({"error": "VALIDATION_FILES_REQUIRED unless --pilot-train-only"}), file=sys.stderr)
        return 2
    else:
        store = DataStore.from_paths(a.population, a.train_features, a.train_targets, a.val_features, a.val_targets, bar_hours=a.bar_hours)
    freeze = json.loads(Path(a.test_freeze).read_text()) if a.test_freeze else None
    try:
        result = run_task(task, store, test_freeze=freeze, results_root=a.runner_results, extractor_code=a.extractor_code)
    except Refusal as exc:
        print(canonical({"error": str(exc), "task_id": task.get("task_id")}), file=sys.stderr)
        return 2
    print(canonical(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
