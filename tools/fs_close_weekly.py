#!/usr/bin/env python3
"""FS-CLOSE primary closure: BUSINESS weekly walk-forward over the external VALIDATION year.

Musashi corrections 2026-10-05 section 3. The static single-fit closure is kept apart as
LITERATURE_STATIC_VALIDATION_DIAGNOSTIC (tools/fs_close_refit.py); it cannot choose the business
manifest. This module reuses the repository's weekly framework and adds no second one:

  tools.business_weekly_protocol   calendar, WeekSpec, four calendar years as-of, procedure identity
  tools.business_asof_window       point-in-time fit population per week (purged, digested)
  tools.business_weekly_training   FULL_RETRAIN traversal with per-week model identity
  tools.business_weekly_score      paired same-row naive scoring and the durable weekly ledger
  tools.business_objective_firewall  TEST stays sealed (phase VALIDATION_SELECTION, never opened)

Per scored week (every complete Monday-aligned week of the validation year, derived from the
calendar) and per frozen K candidate set: the declared ridge head (6-output, one model per
forecast family) and the declared logistic head (one per barrier cell) are refit on exactly the
four calendar years available at the cutoff, then score only the following week against the
same-row naive (zero return / fit prior). The contract (candidate sets, heads, support, naives,
aggregate rule) is sealed to disk BEFORE any validation row is read. Resuming never refits or
rescores a terminal week.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from tools import fs_close_refit as R  # noqa: E402
from tools.business_asof_window import AsOfRow, SupportSpec  # noqa: E402
from tools.business_objective_firewall import BusinessObjectiveFirewall  # noqa: E402
from tools.business_weekly_protocol import (  # noqa: E402
    WEEK,
    BusinessWeeklyProtocol,
    DispositionStatus,
    EvaluationMode,
    EvaluationSplit,
    UpdateMode,
    WeekSpec,
    subtract_calendar_years,
)
from tools.business_weekly_score import (  # noqa: E402
    FINANCIAL_TASK,
    ForecastFamily,
    HorizonForecast,
    ScoreRefusal,
    WeekRelease,
    WeeklyScoreLedger,
    financial_task_contract,
)
from tools.business_weekly_training import (  # noqa: E402
    TaskFamily,
    TrainingRequest,
    TrainingResult,
    TrainingStatus,
    run_weekly_training,
)

UTC = dt.timezone.utc
CLOSURE_SCHEMA = "fs_close_closure_record.v2"
CONTRACT_SCHEMA = "fs_close_weekly_contract.v1"
BUSINESS_MODE = EvaluationMode.BUSINESS_WEEKLY_WALK_FORWARD.value
UPDATE_MODE = UpdateMode.FULL_RETRAIN_ROLLING_4Y
CANDIDATE_KINDS = ("ALL_ADMISSIBLE", "PRED_BEST", "PLUS_CAUSAL", "PLUS_EXTRACTIBILITY_EVIDENCE", "KNOCKOFF")
FAMILIES = {
    # family -> (kind, target columns, purge hours)
    "short": ("regression", [f"Y_s_{h}h" for h in (1, 2, 3, 4, 5, 6)], 6),
    "long": ("regression", [f"Y_l_{h}h" for h in (24, 48, 72, 96, 120, 144)], 144),
    "barrier_s6": ("barrier", ["Y_b_s6"], 6),
    "barrier_l144": ("barrier", ["Y_b_l144"], 144),
}
HORIZON_LABEL = {"short": ["1h", "2h", "3h", "4h", "5h", "6h"], "long": ["24h", "48h", "72h", "96h", "120h", "144h"]}
AGGREGATE_RULE = ("score = mean over the 14 cells (12 forecast horizons: mean weekly skill_mae vs zero-return naive; "
                  "2 barriers: mean weekly skill of log-loss vs fit-prior naive) over ALL scored weeks, never the best week; "
                  "ties -> fewer features, then lower total fit seconds, then ALL_ADMISSIBLE, PRED_BEST, PLUS_CAUSAL, "
                  "PLUS_EXTRACTIBILITY_EVIDENCE, KNOCKOFF")


class WeeklyClosureError(RuntimeError):
    """A contract violation that must stop the closure explicitly."""


# ----------------------------------------------------------------------------- helpers
def _iso(value: dt.datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(text: str) -> dt.datetime:
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _now() -> str:
    return _iso(dt.datetime.now(UTC))


def _first_monday_on_or_after(d: dt.datetime) -> dt.datetime:
    return d + dt.timedelta(days=(7 - d.weekday()) % 7)


def _last_monday_on_or_before(d: dt.datetime) -> dt.datetime:
    return d - dt.timedelta(days=d.weekday())


def build_protocol(validation_year: int, procedure_digest: str) -> BusinessWeeklyProtocol:
    """Validation = every complete Monday-aligned week inside the year; test = the following year's weeks."""
    jan1 = dt.datetime(validation_year, 1, 1, tzinfo=UTC)
    next_jan1 = dt.datetime(validation_year + 1, 1, 1, tzinfo=UTC)
    after_jan1 = dt.datetime(validation_year + 2, 1, 1, tzinfo=UTC)
    validation_start = _first_monday_on_or_after(jan1)
    validation_end = _last_monday_on_or_before(next_jan1)
    test_start = validation_end
    test_end = _last_monday_on_or_before(after_jan1)
    return BusinessWeeklyProtocol(
        evaluation_mode=EvaluationMode.BUSINESS_WEEKLY_WALK_FORWARD,
        update_mode=UPDATE_MODE,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
        procedure_digest=procedure_digest,
    )


def business_closure_or_none(record) -> dict | None:
    """The record only if it is a BUSINESS weekly walk-forward closure; anything else is not a closure."""
    if not isinstance(record, dict):
        return None
    if record.get("schema") != CLOSURE_SCHEMA or record.get("evaluation_mode") != BUSINESS_MODE:
        return None
    if record.get("update_mode") != UPDATE_MODE.value or not record.get("winner"):
        return None
    return record


def require_business_closure(record) -> dict:
    out = business_closure_or_none(record)
    if out is None:
        raise WeeklyClosureError(
            f"CLOSURE_NOT_BUSINESS: evaluation_mode {record.get('evaluation_mode') if isinstance(record, dict) else None!r} "
            f"is not {BUSINESS_MODE}; a static single fit cannot choose the business manifest")
    return out


# ----------------------------------------------------------------------------- frozen candidate sets
def _consensus_order(lists: list[list[str]]) -> list[str]:
    """Mean-rank consensus across lists (Borda); names absent from a list get rank len+1."""
    if not lists:
        return []
    universe = sorted({f for lst in lists for f in lst})
    score = {}
    for f in universe:
        ranks = [lst.index(f) + 1 if f in lst else len(lst) + 1 for lst in lists]
        score[f] = (sum(ranks) / len(ranks), f)
    return sorted(universe, key=lambda f: score[f])


def freeze_candidate_sets(plan: dict, k_primary: int) -> dict:
    """One frozen K list per candidate set and family, from TRAIN-only rankings in the plan.

    Per-target|fold rankings are consolidated by mean rank over the family's targets and folds.
    Never reads VALIDATION: the only input is the sealed plan.
    """
    frozen = {}
    for s in plan.get("sets", []):
        if s.get("set_kind") not in CANDIDATE_KINDS:
            continue
        if s.get("k") not in (None, k_primary):
            continue
        per_family = {}
        for fam, (_, targets, _) in FAMILIES.items():
            if "features_by" in s:
                by = s["features_by"]
                if "*" in by and len(by) == 1:
                    order = list(by["*"])
                else:
                    lists = [list(v) for key, v in by.items() if key.split("|")[0] in targets or key.startswith("*|")]
                    order = _consensus_order(lists) if lists else []
                feats = order[: s["k"]] if s.get("k") else order
            else:
                feats = list(s["features"])
            if not feats:
                raise WeeklyClosureError(f"FROZEN_SET_EMPTY: {s['set_id']} {fam}")
            per_family[fam] = feats
        frozen[s["set_id"]] = {"set_kind": s["set_kind"], "k": s.get("k"), "set_sha256": s.get("set_sha256") or R.set_sha256(s),
                               "families": per_family,
                               "frozen_sha256": _digest(per_family)}
    if not frozen:
        raise WeeklyClosureError("NO_CANDIDATE_SETS_TO_FREEZE")
    return frozen


def seal_contract(plan: dict, k_primary: int, validation_year: int, out_dir: Path) -> dict:
    frozen = freeze_candidate_sets(plan, k_primary)
    body = {
        "schema": CONTRACT_SCHEMA,
        "evaluation_mode": BUSINESS_MODE,
        "update_mode": UPDATE_MODE.value,
        "validation_year": validation_year,
        "rolling_calendar_years": 4,
        "k_primary": k_primary,
        "candidate_sets": frozen,
        "heads": {"regression": R.HEAD_PARAMS["ridge"] | {"outputs": "one multi-output ridge per forecast family (6 horizons), one digest"},
                  "barrier": R.HEAD_PARAMS["ridge"]["barrier"]},
        "support": {fam: {"kind": kind, "targets": targets, "purge_hours": purge, "input_lookback_hours": 0,
                          "inner_validation_weeks": 1, "availability": "features available at the bar end (as-of columns from PS1)"}
                    for fam, (kind, targets, purge) in FAMILIES.items()},
        "naive": {"regression": "zero return on the same scored rows", "barrier": "fit-row class prior on the same scored rows"},
        "scored_rows": "rows whose decision time lies in the scored week and whose 12 forecast targets are all finite "
                       "(identical origins across horizons); barriers use their own supported rows",
        "imputation": "fit-row median per feature; inputs standardised with fit-row statistics",
        "aggregate_rule": AGGREGATE_RULE,
        "seed": R.SEED,
        "test": "EXTERNAL TEST (the following year) is never read; firewall phase stays VALIDATION_SELECTION",
        "plan_population_sha256": plan.get("population_sha256"),
    }
    body["contract_sha256"] = _digest(body)
    body["sealed_utc"] = _now()
    path = Path(out_dir) / "weekly_contract.json"
    if path.is_file():
        prior = json.loads(path.read_text())
        if prior.get("contract_sha256") != body["contract_sha256"]:
            raise WeeklyClosureError("CONTRACT_CHANGED: a sealed weekly contract exists with a different digest")
        return prior
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True))
    os.replace(tmp, path)
    return body


# ----------------------------------------------------------------------------- data
def _load_train(feature_files, targets_file, population):
    """TRAIN loader without a folds file (the weekly protocol derives its own calendar)."""
    import pyarrow.parquet as pq

    col_index = {name: i for i, name in enumerate(population)}
    X = None
    row_ids = None
    ts = None
    digests = {}
    seen = set()
    for fp in feature_files:
        fp = Path(fp)
        digests["train:" + fp.name + "@" + fp.parent.name] = R.sha256_file(fp)
        pf = pq.ParquetFile(fp)
        ids = pf.read(columns=["row_id"]).column("row_id").to_numpy()
        if row_ids is None:
            row_ids = ids
            tcol = pf.read(columns=["t_decision_utc"]).column("t_decision_utc")
            ts = (tcol.cast("int64").to_numpy() // 10**9) if "ns" in str(tcol.type) else tcol.cast("int64").to_numpy()
            X = np.empty((len(row_ids), len(population)), dtype="float64")
        elif not np.array_equal(ids, row_ids):
            raise R.RefitError(f"ROW_ID_MISMATCH between feature batches: {fp}")
        for name in [n for n in pf.schema.names if n in col_index]:
            X[:, col_index[name]] = pf.read(columns=[name]).column(name).to_numpy(zero_copy_only=False).astype("float64", copy=False)
            seen.add(name)
    missing = [c for c in population if c not in seen]
    if missing:
        raise R.RefitError(f"POPULATION_NOT_IN_INPUTS: {len(missing)} first {missing[:5]}")
    tp = Path(targets_file)
    digests["train:" + tp.name] = R.sha256_file(tp)
    tdf = pq.read_table(tp).to_pandas()
    if not np.array_equal(tdf["row_id"].to_numpy(), row_ids):
        raise R.RefitError("ROW_ID_MISMATCH between features and targets")
    bound = R.assert_train_only(ts)
    return X, row_ids, tdf, None, digests, bound


def _target_matrix(tdf, fam: str) -> np.ndarray:
    kind, cols, _ = FAMILIES[fam]
    if kind == "regression":
        return np.column_stack([tdf[c].to_numpy(dtype="float64") for c in cols])
    raw = tdf[cols[0]].to_numpy(dtype="float64")
    y = np.full(raw.shape, np.nan)
    ok = np.isfinite(raw) & np.isin(raw, (-1.0, 0.0, 1.0))
    y[ok] = raw[ok] + 1.0
    return y[:, None]


# ----------------------------------------------------------------------------- heads
def _fit_regression(Xf, Yf):
    Xf = Xf.copy()
    med = np.nanmedian(Xf, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(Xf)
    Xf[bad] = np.broadcast_to(med, Xf.shape)[bad]
    mu = Xf.mean(axis=0)
    sd = Xf.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (Xf - mu) / sd
    ym = Yf.mean(axis=0)
    A = Z.T @ Z + R.HEAD_PARAMS["ridge"]["alpha"] * np.eye(Z.shape[1])
    W = np.linalg.solve(A, Z.T @ (Yf - ym))
    return {"kind": "regression", "med": med, "mu": mu, "sd": sd, "W": W, "ym": ym}


def _fit_barrier(Xf, yf):
    from sklearn.linear_model import LogisticRegression

    Xf = Xf.copy()
    med = np.nanmedian(Xf, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(Xf)
    Xf[bad] = np.broadcast_to(med, Xf.shape)[bad]
    mu = Xf.mean(axis=0)
    sd = Xf.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (Xf - mu) / sd
    p = R.HEAD_PARAMS["ridge"]["barrier"]
    clf = LogisticRegression(C=p["C"], solver=p["solver"], max_iter=p["max_iter"], random_state=p["random_state"])
    clf.fit(Z, yf.astype(int))
    prior = np.bincount(yf.astype(int), minlength=3) / yf.size
    return {"kind": "barrier", "med": med, "mu": mu, "sd": sd, "clf": clf, "prior": prior,
            "coef": clf.coef_.copy(), "intercept": clf.intercept_.copy(), "classes": clf.classes_.copy()}


def _predict(model, Xv):
    Xv = Xv.copy()
    bad = ~np.isfinite(Xv)
    Xv[bad] = np.broadcast_to(model["med"], Xv.shape)[bad]
    Z = (Xv - model["mu"]) / model["sd"]
    if model["kind"] == "regression":
        return Z @ model["W"] + model["ym"]
    return R._full_proba(model["clf"], Z)


def _model_digest(model) -> str:
    h = hashlib.sha256()
    if model["kind"] == "regression":
        h.update(model["W"].tobytes())
        h.update(model["ym"].tobytes())
    else:
        h.update(model["coef"].tobytes())
        h.update(model["intercept"].tobytes())
        h.update(model["classes"].astype("int64").tobytes())
    h.update(model["mu"].tobytes())
    h.update(model["sd"].tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- run
def run_weekly_closure(plan_path, train_feature_files, train_targets_file, val_feature_files, val_targets_file,
                       out_dir, *, k_primary: int = 24, validation_year: int = 2024) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    plan = R.load_plan(Path(plan_path))
    population = list(plan["population"])
    contract = seal_contract(plan, k_primary, validation_year, out_dir)       # sealed before any validation read
    protocol = build_protocol(validation_year, contract["contract_sha256"])
    validation_weeks = tuple(w for w in protocol.weeks() if w.split is EvaluationSplit.VALIDATION)
    test_weeks = tuple(w for w in protocol.weeks() if w.split is EvaluationSplit.TEST)
    firewall = BusinessObjectiveFirewall.create([f"test:{_iso(w.start)}" for w in test_weeks])

    Xt, ids_t, tdf_t, _, dig_t, bound_t = _load_train(train_feature_files, train_targets_file, population)
    validation_first_read_utc = _now()
    Xv, ids_v, tdf_v, dig_v, bound_v = R.load_validation(val_feature_files, val_targets_file, population)
    X = np.vstack([Xt, Xv])
    del Xt, Xv
    import pandas as pd

    tdf = pd.concat([tdf_t, tdf_v], ignore_index=True)
    row_ids = np.concatenate([ids_t, ids_v])
    ts = (tdf["t_decision_utc"].astype("int64") // 10**9).to_numpy()
    order = np.argsort(ts, kind="stable")
    X, tdf, row_ids, ts = X[order], tdf.iloc[order].reset_index(drop=True), row_ids[order], ts[order]
    if len(set(row_ids.tolist())) != len(row_ids):
        raise WeeklyClosureError("DUPLICATE_ROW_IDS across TRAIN and VALIDATION inputs")
    col = {name: i for i, name in enumerate(population)}
    record_ids = np.array([f"{int(r):012d}" for r in row_ids])
    rid_index = {r: i for i, r in enumerate(record_ids)}
    event_times = [dt.datetime.fromtimestamp(int(t), UTC) for t in ts]
    Y = {fam: _target_matrix(tdf, fam) for fam in FAMILIES}
    finite = {fam: np.all(np.isfinite(Y[fam]), axis=1) for fam in FAMILIES}
    regression_all_finite = finite["short"] & finite["long"]

    # as-of candidate rows per family (independent of the set)
    def rows_for(fam):
        purge = dt.timedelta(hours=FAMILIES[fam][2])

        def provider(week: WeekSpec):
            lo = np.searchsorted(ts, int(week.fit_start.timestamp()), side="left")
            hi = np.searchsorted(ts, int((week.cutoff - purge).timestamp()), side="right")
            out = []
            for i in range(lo, hi):
                if not finite[fam][i]:
                    continue
                et = event_times[i]
                out.append(AsOfRow(record_id=str(record_ids[i]), event_time=et, available_time=et,
                                   target_available_time=et + purge,
                                   row_digest=hashlib.sha256(f"{record_ids[i]}|{int(ts[i])}".encode()).hexdigest()))
            return out
        return provider

    supports = {fam: SupportSpec(input_lookback=dt.timedelta(0), target_horizon=dt.timedelta(hours=purge),
                                 maximum_holding_support=dt.timedelta(0), inner_validation_weeks=1)
                for fam, (_, _, purge) in FAMILIES.items()}
    contract_score = financial_task_contract()
    counters = {"fits_performed": 0, "weeks_restored": 0, "fit_seconds": 0.0}
    set_records = []
    for set_id, spec in contract["candidate_sets"].items():
        ledger_path = out_dir / f"weekly_ledger_{set_id.replace(':', '_').replace('/', '_')}.json"
        barrier_path = out_dir / f"weekly_barrier_{set_id.replace(':', '_').replace('/', '_')}.json"
        models_path = out_dir / f"weekly_models_{set_id.replace(':', '_').replace('/', '_')}.json"
        ledger = WeeklyScoreLedger.from_json(ledger_path.read_text()) if ledger_path.is_file() else WeeklyScoreLedger(validation_weeks, contract_score)
        barrier_ledger = json.loads(barrier_path.read_text()) if barrier_path.is_file() else {"schema": "fs_close_weekly_barrier_ledger.v1", "weeks": {}}
        model_store = json.loads(models_path.read_text()) if models_path.is_file() else {"schema": "fs_close_weekly_models.v1", "weeks": {}}
        scored_keys = {k for k in ledger._scores}
        week_units: dict[str, dict] = {}
        for fam, (kind, _targets, purge_h) in FAMILIES.items():
            feats = spec["families"][fam]
            idx = [col[f] for f in feats]
            models: dict[str, dict] = {}
            terminal = {w.start for w in validation_weeks
                        if (_iso(w.start) + "|validation") in scored_keys and (fam in (model_store["weeks"].get(_iso(w.start), {})))
                        and (kind == "regression" or _iso(w.start) in barrier_ledger["weeks"])}

            def trainer(request: TrainingRequest, fam=fam, idx=idx, kind=kind, models=models, terminal=terminal):
                wk = _iso(request.week.start)
                stored = model_store["weeks"].get(wk, {}).get(fam)
                if request.week.start in terminal and stored and stored["data_identity"] == request.data_identity:
                    counters["weeks_restored"] += 0   # restored terminal week: no refit
                    return TrainingResult(family=request.family, week_start=request.week.start, cutoff=request.week.cutoff,
                                          seed=request.seed, data_identity=request.data_identity,
                                          procedure_digest=request.procedure_digest, update_mode=request.update_mode,
                                          cost_identity=request.cost_identity, parent_digest=None,
                                          model_digest=stored["model_digest"], execution_backend=request.required_backend)
                fit_idx = np.array([rid_index[r.record_id] for r in request.window.fit_rows], dtype="int64")
                if fit_idx.size < 100:
                    raise WeeklyClosureError(f"TOO_FEW_FIT_ROWS {fit_idx.size}")
                Xf = X[np.ix_(fit_idx, idx)]
                Yf = Y[fam][fit_idx]
                t0 = time.time()
                model = _fit_regression(Xf, Yf) if kind == "regression" else _fit_barrier(Xf, Yf[:, 0])
                elapsed = time.time() - t0
                counters["fits_performed"] += 1
                counters["fit_seconds"] += elapsed
                digest = _model_digest(model)
                models[wk] = model
                model_store["weeks"].setdefault(wk, {})[fam] = {
                    "model_digest": digest, "data_identity": request.data_identity, "fit_rows": int(fit_idx.size),
                    "fit_population_digest": request.window.fit_population_digest, "fit_start": _iso(request.window.fit_start),
                    "fit_end": _iso(request.window.fit_end), "fit_max_event_time": _iso(max(r.event_time for r in request.window.fit_rows)),
                    "fit_seconds": elapsed, "n_features": len(idx)}
                return TrainingResult(family=request.family, week_start=request.week.start, cutoff=request.week.cutoff,
                                      seed=request.seed, data_identity=request.data_identity, procedure_digest=request.procedure_digest,
                                      update_mode=request.update_mode, cost_identity=request.cost_identity, parent_digest=None,
                                      model_digest=digest, execution_backend=request.required_backend)

            run = run_weekly_training(protocol=protocol, split=EvaluationSplit.VALIDATION, families=(TaskFamily.FORECAST,),
                                      support=supports[fam], rows_for_week=rows_for(fam), trainer=trainer, firewall=firewall,
                                      seed=R.SEED, cost_identity=f"{kind}:{set_id}:{fam}:k{len(idx)}", required_backend="cpu_numpy_sklearn")
            for disp in run.dispositions:
                wk = _iso(disp.week.start)
                unit = week_units.setdefault(wk, {"start": wk, "end": _iso(disp.week.end), "cutoff": _iso(disp.week.cutoff),
                                                  "fit_start": _iso(disp.week.fit_start), "ordinal": disp.week.ordinal, "families": {}})
                if disp.status is TrainingStatus.COMPLETED:
                    stored = model_store["weeks"][wk][fam]
                    unit["families"][fam] = dict(stored) | {"status": "COMPLETED", "model": models.get(wk)}
                else:
                    unit["families"][fam] = {"status": "FAILED", "reason": disp.reason, "model_digest": "", "fit_population_digest": "",
                                             "fit_rows": 0, "fit_end": wk, "fit_max_event_time": wk}
        # score every week not yet terminal in the ledger
        for w in validation_weeks:
            wk = _iso(w.start)
            key = wk + "|validation"
            unit = week_units[wk]
            if key in ledger._scores and wk in barrier_ledger["weeks"]:
                counters["weeks_restored"] += 1
                continue
            lo = np.searchsorted(ts, int(w.start.timestamp()), side="left")
            hi = np.searchsorted(ts, int(w.end.timestamp()), side="left")
            rows = np.arange(lo, hi)
            if key not in ledger._scores:
                fams_ok = all(unit["families"].get(f, {}).get("status") == "COMPLETED" for f in ("short", "long"))
                origins_idx = rows[regression_all_finite[rows]]
                if not fams_ok or origins_idx.size == 0:
                    reason = "forecast family training failed" if not fams_ok else "no scored row with all 12 targets finite"
                    ledger.record_terminal(w, DispositionStatus.FAILED, reason)
                else:
                    origins = tuple(str(r) for r in record_ids[origins_idx])
                    horizons = []
                    for fam, ff in (("short", ForecastFamily.SHORT), ("long", ForecastFamily.LONG)):
                        model = unit["families"][fam]["model"]
                        feats = spec["families"][fam]
                        pred = _predict(model, X[np.ix_(origins_idx, [col[f] for f in feats])])
                        for j, label in enumerate(HORIZON_LABEL[fam]):
                            target = tuple(float(v) for v in Y[fam][origins_idx, j])
                            horizons.append(HorizonForecast(family=ff, horizon=label, origins=origins, target=target,
                                                            prediction=tuple(float(v) for v in pred[:, j]),
                                                            naive=tuple(0.0 for _ in origins), unit="log_return", scale="raw_log_return",
                                                            model_digest=unit["families"][fam]["model_digest"],
                                                            dataset_id=f"eurusd_1h.validation_{validation_year}", week=w,
                                                            task=FINANCIAL_TASK, population=f"set:{set_id}"))
                    try:
                        ledger.submit(WeekRelease(week=w, horizons=tuple(horizons)), firewall=firewall)
                    except ScoreRefusal as exc:
                        ledger.record_terminal(w, DispositionStatus.FAILED, f"score refused: {exc}")
            if wk not in barrier_ledger["weeks"]:
                entry = {}
                for fam in ("barrier_s6", "barrier_l144"):
                    u = unit["families"].get(fam, {})
                    bidx = rows[finite[fam][rows]]
                    if u.get("status") != "COMPLETED" or bidx.size == 0:
                        entry[fam] = {"status": "FAILED", "reason": u.get("reason", "no supported barrier row"), "n_rows": int(bidx.size),
                                      "naive_kind": "fit_prior", "naive_n_rows": int(bidx.size), "naive_log_loss": float("nan")}
                        continue
                    model = u["model"]
                    feats = spec["families"][fam]
                    proba = _predict(model, X[np.ix_(bidx, [col[f] for f in feats])])
                    yb = Y[fam][bidx, 0]
                    metrics = {m: v for m, v, _, _ in R.barrier_metrics(yb, proba, model["prior"])}
                    naive = {nk: nv for _, _, nk, nv in R.barrier_metrics(yb, proba, model["prior"])}
                    entry[fam] = {"status": "MEASURED", "n_rows": int(bidx.size), "naive_kind": "fit_prior", "naive_n_rows": int(bidx.size),
                                  "log_loss": metrics["log_loss"], "brier": metrics["brier"],
                                  "naive_log_loss": naive["fit_prior"], "naive_brier": [v for m, v, _, _ in R.barrier_metrics(yb, proba, model["prior"]) if m == "brier"][0],
                                  "skill_log_loss": R.skill(metrics["log_loss"], naive["fit_prior"]),
                                  "model_digest": u["model_digest"], "rows_sha256": hashlib.sha256(row_ids[bidx].astype("int64").tobytes()).hexdigest()}
                    entry[fam]["naive_brier"] = [nv for m, v, nk, nv in R.barrier_metrics(yb, proba, model["prior"]) if m == "brier"][0]
                    entry[fam]["skill_brier"] = R.skill(entry[fam]["brier"], entry[fam]["naive_brier"])
                entry["digest"] = _digest(entry)
                barrier_ledger["weeks"][wk] = entry
            # persist after every week so a restart resumes at the next week
            _write(ledger_path, ledger.to_json())
            _write(barrier_path, json.dumps(barrier_ledger, indent=1, sort_keys=True))
            _write(models_path, json.dumps(model_store, indent=1, sort_keys=True))
        close = ledger.close()
        set_records.append(_aggregate_set(set_id, spec, ledger, barrier_ledger, week_units, close))
    winner = _choose_winner(set_records)
    record = {
        "schema": CLOSURE_SCHEMA, "evaluation_mode": BUSINESS_MODE, "update_mode": UPDATE_MODE.value,
        "validation_year": validation_year, "k_primary": k_primary, "seed": R.SEED,
        "procedure_digest": protocol.procedure_digest, "contract": contract,
        "weeks": [{"ordinal": w.ordinal, "start": _iso(w.start), "end": _iso(w.end), "cutoff": _iso(w.cutoff), "fit_start": _iso(w.fit_start),
                   "retrain_due": w.retrain_due} for w in validation_weeks],
        "weeks_in_validation_year": len(validation_weeks), "test_weeks_sealed": len(test_weeks), "test_weeks_opened": 0,
        "firewall_phase": firewall.phase.value, "train_bound": bound_t, "validation_bound": bound_v,
        "validation_first_read_utc": validation_first_read_utc, "validation_read_count": 1,
        "inputs_sha256": {**dig_t, **dig_v}, "plan_sha256": R.sha256_file(Path(plan_path)), "code_sha256": R.sha256_file(Path(__file__)),
        "sets": [{k: v for k, v in s.items()} for s in set_records],
        "winner": winner, "strategy_eligible": bool(winner) and winner.get("beats_naive_every_forecast_horizon", False),
        "aggregate_rule": AGGREGATE_RULE,
        "fits_performed": counters["fits_performed"], "weeks_restored": counters["weeks_restored"], "fit_seconds": counters["fit_seconds"],
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "wall_seconds": time.time() - started, "finished_utc": _now(),
    }
    _write(out_dir / "closure_record.json", json.dumps(record, indent=1, sort_keys=True, default=str))
    return record


def _write(path: Path, text: str) -> None:
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _aggregate_set(set_id, spec, ledger: WeeklyScoreLedger, barrier_ledger, week_units, close) -> dict:
    per_h: dict[str, dict] = {}
    weeks_out = []
    for score in close.dispositions:
        unit = week_units[score.week_start]
        weeks_out.append({"start": unit["start"], "end": unit["end"], "cutoff": unit["cutoff"], "fit_start": unit["fit_start"],
                          "disposition": score.disposition, "reason": score.reason, "strategy_eligible": score.strategy_eligible,
                          "families": {f: {k: v for k, v in u.items() if k != "model"} | ({} if f.startswith("barrier") else {})
                                       | (barrier_ledger["weeks"].get(unit["start"], {}).get(f, {}) if f.startswith("barrier") else {})
                                       for f, u in unit["families"].items()},
                          "horizon_scores": [{"family": h.family, "horizon": h.horizon, "sample_count": h.sample_count, "model_mae": h.model_mae,
                                              "naive_mae": h.naive_mae, "model_mse": h.model_mse, "naive_mse": h.naive_mse, "skill_mae": h.skill_mae,
                                              "beats_primary_naive": h.beats_primary_naive} for h in score.horizons]})
        for h in score.horizons:
            d = per_h.setdefault(f"{h.family}:{h.horizon}", {"model_mae": [], "naive_mae": [], "skill_mae": [], "model_mse": [], "naive_mse": []})
            d["model_mae"].append(h.model_mae)
            d["naive_mae"].append(h.naive_mae)
            d["model_mse"].append(h.model_mse)
            d["naive_mse"].append(h.naive_mse)
            d["skill_mae"].append(h.skill_mae if h.skill_mae is not None else float("nan"))
    aggregates = {}
    for key, d in per_h.items():
        sk = np.array(d["skill_mae"], dtype="float64")
        aggregates[key] = {"weeks": len(d["model_mae"]), "mean_model_mae": float(np.mean(d["model_mae"])), "mean_naive_mae": float(np.mean(d["naive_mae"])),
                           "mean_model_mse": float(np.mean(d["model_mse"])), "mean_naive_mse": float(np.mean(d["naive_mse"])),
                           "mean_weekly_skill_mae": float(np.nanmean(sk)), "std_weekly_skill_mae": float(np.nanstd(sk)),
                           "weeks_beating_naive": int(np.sum(sk > 0)),
                           "beats_naive_on_aggregate": bool(np.mean(d["model_mae"]) < np.mean(d["naive_mae"]))}
    for fam in ("barrier_s6", "barrier_l144"):
        vals = [w[fam] for w in barrier_ledger["weeks"].values() if w.get(fam, {}).get("status") == "MEASURED"]
        if vals:
            sk = np.array([v["skill_log_loss"] for v in vals], dtype="float64")
            aggregates[fam] = {"weeks": len(vals), "mean_log_loss": float(np.mean([v["log_loss"] for v in vals])),
                               "mean_naive_log_loss": float(np.mean([v["naive_log_loss"] for v in vals])),
                               "mean_brier": float(np.mean([v["brier"] for v in vals])), "mean_naive_brier": float(np.mean([v["naive_brier"] for v in vals])),
                               "mean_weekly_skill_log_loss": float(np.nanmean(sk)), "std_weekly_skill_log_loss": float(np.nanstd(sk)),
                               "weeks_beating_naive": int(np.sum(sk > 0))}
        else:
            aggregates[fam] = {"weeks": 0}
    cell_scores = [v["mean_weekly_skill_mae"] for k, v in aggregates.items() if ":" in k] + \
                  [v["mean_weekly_skill_log_loss"] for k, v in aggregates.items() if k.startswith("barrier") and v.get("weeks")]
    forecast_cells = [v for k, v in aggregates.items() if ":" in k]
    fit_seconds = sum(u.get("fit_seconds", 0.0) for w in week_units.values() for u in w["families"].values())
    return {"set_id": set_id, "set_kind": spec["set_kind"], "k": spec["k"], "set_sha256": spec["set_sha256"], "frozen_sha256": spec["frozen_sha256"],
            "n_features": max(len(v) for v in spec["families"].values()),
            "weeks_expected": close.expected_weeks, "weeks_completed": sum(1 for s in close.dispositions if s.disposition == "COMPLETED"),
            "eligible_weeks": close.eligible_weeks, "origin_rows": close.origin_rows, "independence_status": close.independence_status,
            "aggregates": aggregates, "cells_scored": len(cell_scores),
            "score": float(np.mean(cell_scores)) if len(cell_scores) == 14 else float("nan"),
            "beats_naive_every_forecast_horizon": bool(forecast_cells) and all(v["beats_naive_on_aggregate"] for v in forecast_cells),
            "fit_seconds_total": float(fit_seconds), "ledger_digest": json.loads(ledger.to_json())["ledger_digest"], "weeks": weeks_out}


def _choose_winner(set_records) -> dict | None:
    order = {k: i for i, k in enumerate(CANDIDATE_KINDS)}
    complete = [s for s in set_records if s["cells_scored"] == 14 and s["weeks_completed"] == s["weeks_expected"]]
    if not complete:
        return None
    best = sorted(complete, key=lambda s: (-s["score"], s["n_features"], s["fit_seconds_total"], order.get(s["set_kind"], 9)))[0]
    return {"set_id": best["set_id"], "set_kind": best["set_kind"], "set_sha256": best["set_sha256"], "frozen_sha256": best["frozen_sha256"],
            "score": best["score"], "k": best["k"], "n_features": best["n_features"], "eligible_weeks": best["eligible_weeks"],
            "weeks": best["weeks_expected"], "beats_naive_every_forecast_horizon": best["beats_naive_every_forecast_horizon"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--val-features", nargs="+", required=True)
    ap.add_argument("--val-targets", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k-primary", type=int, default=24)
    ap.add_argument("--validation-year", type=int, default=2024)
    a = ap.parse_args(argv)
    rec = run_weekly_closure(Path(a.plan), a.features, a.targets, a.val_features, a.val_targets, Path(a.out_dir),
                             k_primary=a.k_primary, validation_year=a.validation_year)
    print(json.dumps({"winner": rec["winner"], "strategy_eligible": rec["strategy_eligible"], "fits": rec["fits_performed"],
                      "weeks": rec["weeks_in_validation_year"], "peak_rss_bytes": rec["peak_rss_bytes"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
