#!/usr/bin/env python3
"""RP27: the E1 task loader — a structured contract that the training consumes, and one window
enumerator from which every count, tensor, mask, origin id/time and denominator derives.

Contract (`TaskContract`): ordered feature names, target names, metadata names, control names and the
expected dtype per role; an unexpected column is REFUSED unless listed in `ignore` (explicit exclusion);
a column is never included because of its dtype. Timestamps are parsed, never coerced to numbers.
The target's own history may be a feature only when `target_history_as_feature` is declared (household
primary pilot: yes, to equal the information of persistence / seasonal-naive); the future of the target
never enters an input: inputs of the window with origin t are rows t - W + 1 .. t, the target is row
t + h, enforced by identity of the rows, not by column name.

Grid (`fixed_grid`): the declared step; a window whose W + h rows are not exactly consecutive on the grid
(gap, duplicate, disorder, DST-ambiguous row) is WITHDRAWN (policy `withdraw`), or, under a declared causal
policy `mask_ffill` (last observation carried forward with a mask channel; never a future value), kept with
its mask recorded. Time is never compressed by deleting rows.

Masks: an input row is valid when every feature is finite; a target is valid when finite and, for the
electricity family, when the client is ACTIVE under the declared causal rule: active from the first row
where the client has a finite non-zero value observed at or before the window origin (a proxy for the
service start, declared as such), never-active clients get state NEVER_ACTIVE, and a real zero after
activation is a measurement. Nothing is trained on a label without an objective.

Scaling: `fit_scaler(train_windows)` only; validation/test changes cannot alter it nor any earlier window.
Evaluation: every arm scores the SAME origins/targets/denominators (`EvalSet`); if an arm cannot emit a
prediction for an origin, coverage is reported and the common policy applies (origin dropped for ALL arms).

    python tools/df_e1_loader.py --self-check
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


class ContractRefusal(ValueError):
    pass


@dataclass
class TaskContract:
    family: str
    timestamp: str
    ts_format: str
    step_seconds: int
    features: list
    targets: list
    metadata: list = field(default_factory=list)
    controls: list = field(default_factory=list)
    ignore: list = field(default_factory=list)
    target_history_as_feature: bool = False
    dtype: str = "float64"
    activation_rule: str = "none"                 # "none" | "first_finite_nonzero_at_or_before_origin"
    gap_policy: str = "withdraw"                  # "withdraw" | "mask_ffill"
    ambiguous_labels: list = field(default_factory=list)   # timestamp labels declared AMBIGUOUS_SUPPORT (e.g. DST hours)
    horizon: int = 1
    window: int = 24
    splits: dict = field(default_factory=lambda: {"train": 0.7, "validation": 0.15, "test": 0.15})
    #: RP35: what kind of task this contract describes. A forecast needs a strictly positive horizon;
    #: `h = 0` means the target row IS inside the window, which is RECONSTRUCTION — another task, whose
    #: trivial solution (copy the last input) says nothing about forecasting. Declaring it is the only
    #: way to get it, and a forecasting contract refuses it.
    task_kind: str = "forecast"                   # "forecast" | "reconstruction"

    def validate(self) -> "TaskContract":
        """Every domain the enumerator later assumes, checked before it runs (RP35)."""
        def integral(value, name, minimum):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ContractRefusal(f"{name} must be an integer, not {value!r}")
            if int(value) < minimum:
                raise ContractRefusal(f"{name} must be >= {minimum}, not {value}")
            return int(value)
        if self.task_kind not in ("forecast", "reconstruction"):
            raise ContractRefusal(f"unknown task kind {self.task_kind!r}")
        integral(self.window, "window", 1)
        integral(self.step_seconds, "step_seconds", 1)
        h = integral(self.horizon, "horizon", 0)
        if self.task_kind == "forecast" and h < 1:
            raise ContractRefusal("a forecasting contract needs horizon >= 1: with h = 0 the target row is inside "
                                  "the window and copying the last input would score perfectly; declare "
                                  "task_kind='reconstruction' if that is the task you mean")
        if self.task_kind == "reconstruction" and h != 0:
            raise ContractRefusal("a reconstruction contract has horizon 0")
        if not self.targets:
            raise ContractRefusal("a contract declares at least one target")
        for name, group in (("features", self.features), ("targets", self.targets), ("metadata", self.metadata),
                            ("controls", self.controls), ("ignore", self.ignore)):
            if not isinstance(group, list) or any(not isinstance(x, str) or not x for x in group):
                raise ContractRefusal(f"{name} must be a list of column names")
        if not isinstance(self.splits, dict) or not self.splits:
            raise ContractRefusal("splits must be a non-empty object")
        total = 0.0
        for name, frac in self.splits.items():
            if isinstance(frac, bool) or not isinstance(frac, (int, float)) or not (0 < float(frac) <= 1):
                raise ContractRefusal(f"split {name!r}: a fraction in (0, 1], not {frac!r}")
            total += float(frac)
        if total > 1 + 1e-9:
            raise ContractRefusal(f"the splits add to {total}, more than the data")
        if self.gap_policy == "mask_ffill":
            raise ContractRefusal("gap policy 'mask_ffill' is REFUSED until a consumer exists: the loader would fill "
                                  "values causally but no model in this programme takes the input mask, so the fill "
                                  "would enter the tensor unannounced. Use 'withdraw'.")
        if self.gap_policy != "withdraw":
            raise ContractRefusal(f"unknown gap policy {self.gap_policy!r}")
        if self.activation_rule not in ("none", "first_finite_nonzero_at_or_before_origin"):
            raise ContractRefusal(f"unknown activation rule {self.activation_rule!r}")
        if not isinstance(self.target_history_as_feature, bool):
            raise ContractRefusal("target_history_as_feature must be a boolean")
        return self

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(dataclasses.asdict(self), sort_keys=True).encode()).hexdigest()

    def input_columns(self) -> list:
        cols = list(self.features)
        if self.target_history_as_feature:
            cols += [t for t in self.targets if t not in cols]
        return cols


def resolve(frame: pd.DataFrame, c: TaskContract) -> dict:
    """Bind the frame's columns to the contract's roles; refuse extras and missing roles; parse time."""
    c.validate()
    cols = list(frame.columns)
    declared = [c.timestamp] + c.features + c.targets + c.metadata + c.controls + c.ignore
    missing = [x for x in [c.timestamp] + c.features + c.targets + c.metadata + c.controls if x not in cols]
    if missing:
        raise ContractRefusal(f"columns declared by the contract are absent: {missing}")
    extra = [x for x in cols if x not in declared]
    if extra:
        raise ContractRefusal(f"columns not foreseen by the contract (declare a role or list them in `ignore`): {extra}")
    dup = [x for x in set(declared) if declared.count(x) > 1]
    if dup:
        raise ContractRefusal(f"a column has more than one role: {dup}")
    if not pd.api.types.is_string_dtype(frame[c.timestamp]) and not pd.api.types.is_datetime64_any_dtype(frame[c.timestamp]):
        raise ContractRefusal("the timestamp column must be text or datetime, never numeric")
    ts = pd.to_datetime(frame[c.timestamp], format=c.ts_format) if not pd.api.types.is_datetime64_any_dtype(frame[c.timestamp]) else frame[c.timestamp]
    for col in c.features + c.targets:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ContractRefusal(f"{col}: a feature/target must be numeric (dtype {frame[col].dtype})")
    return {"timestamps": ts.reset_index(drop=True), "inputs": frame[c.input_columns()].to_numpy(dtype=c.dtype),
            "targets": frame[c.targets].to_numpy(dtype=c.dtype), "metadata": frame[c.metadata].reset_index(drop=True),
            "controls": frame[c.controls].to_numpy(dtype=c.dtype) if c.controls else None, "input_columns": c.input_columns()}


def grid_ok(ts: pd.Series, step: int) -> np.ndarray:
    """Row i is 'grid-ok' when its label is exactly step seconds after row i-1 (row 0 is ok)."""
    d = ts.diff().dt.total_seconds().to_numpy()
    ok = np.ones(ts.size, dtype=bool)
    ok[1:] = d[1:] == step
    return ok


def ambiguous_rows(ts: pd.Series, ambiguous: set) -> np.ndarray:
    """Rows whose label is declared AMBIGUOUS_SUPPORT (e.g. the DST hours): any window containing one is withdrawn."""
    if not ambiguous:
        return np.zeros(ts.size, dtype=bool)
    lab = ts.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    return np.isin(lab, list(ambiguous))


def activation(targets: np.ndarray, rule: str) -> np.ndarray:
    """Per target column: rows at or after the first finite non-zero value (a declared proxy of service
    start); a never-active column is all False. Under rule 'none' every row is active."""
    n, k = targets.shape
    if rule == "none":
        return np.ones((n, k), dtype=bool)
    if rule != "first_finite_nonzero_at_or_before_origin":
        raise ContractRefusal(f"unknown activation rule {rule!r}")
    act = np.zeros((n, k), dtype=bool)
    for j in range(k):
        nz = np.flatnonzero(np.isfinite(targets[:, j]) & (targets[:, j] != 0))
        if nz.size:
            act[nz[0]:, j] = True
    return act


def split_edges(n: int, splits: dict) -> dict:
    edges, cur = {}, 0
    for name, frac in splits.items():
        edges[name] = [cur, cur + int(n * frac)]
        cur = edges[name][1]
    last = list(splits)[-1]
    edges[last][1] = n
    return edges


def enumerate_windows(resolved: dict, c: TaskContract) -> dict:
    """THE enumerator: for every split, the admissible origins with ids, times, support and masks."""
    c.validate()
    ts = resolved["timestamps"]
    X, Y = resolved["inputs"], resolved["targets"]
    n, W, h = X.shape[0], int(c.window), int(c.horizon)
    ok = grid_ok(ts, c.step_seconds)
    amb = ambiguous_rows(ts, set(c.ambiguous_labels))
    csum_amb = np.concatenate([[0], np.cumsum(amb)])
    fin_in = np.isfinite(X).all(axis=1)
    act = activation(Y, c.activation_rule)
    fin_t = np.isfinite(Y)
    csum_grid = np.concatenate([[0], np.cumsum(ok)])
    csum_fin = np.concatenate([[0], np.cumsum(fin_in)])
    edges = split_edges(n, c.splits)
    purge = W + h
    out = {"contract_sha256": c.sha256(), "window": W, "horizon": h, "purge": purge, "splits": {}, "grid_rows_not_ok": int((~ok).sum()), "ambiguous_rows": int(amb.sum()),
           "input_rows_non_finite": int((~fin_in).sum()), "never_active_targets": [c.targets[j] for j in range(Y.shape[1]) if not act[:, j].any()],
           "policy": {"gap": c.gap_policy, "activation": c.activation_rule, "ambiguous_labels": len(c.ambiguous_labels)}}
    for name, (lo, hi) in edges.items():
        end = hi - (purge if name != list(edges)[-1] else h)
        origins = np.arange(max(lo, W - 1), max(end, 0))
        if origins.size == 0:
            out["splits"][name] = {"origins": 0, "admissible": 0}
            continue
        # support rows t-W+1 .. t+h must all be grid-consecutive: rows t-W+2 .. t+h each 'ok' relative to their predecessor
        span_ok = ((csum_grid[origins + h + 1] - csum_grid[origins - W + 2]) == (W + h - 1)) & ((csum_amb[origins + h + 1] - csum_amb[origins - W + 1]) == 0)
        inputs_finite = (csum_fin[origins + 1] - csum_fin[origins - W + 1]) == W
        admissible = span_ok & inputs_finite                               # the only implemented policy (RP35)
        mask_used = np.zeros(origins.size, dtype=bool)
        tgt_valid = fin_t[origins + h] & act[origins, :]                     # activity judged AT THE ORIGIN, never with future rows
        rows = origins[admissible]
        out["splits"][name] = {"range": [int(lo), int(hi)], "origins": int(origins.size), "admissible": int(admissible.sum()),
                               "withdrawn_grid": int((~span_ok).sum()), "withdrawn_non_finite_inputs": int((span_ok & ~inputs_finite).sum()) if c.gap_policy == "withdraw" else 0,
                               "masked_inputs": int(mask_used.sum()),
                               "targets_valid_per_column": {c.targets[j]: int(tgt_valid[admissible, j].sum()) for j in range(Y.shape[1])},
                               "targets_valid_any": int(tgt_valid[admissible].any(axis=1).sum()), "targets_valid_all": int(tgt_valid[admissible].all(axis=1).sum()),
                               "origin_ids": rows.tolist(), "origin_times": ts.iloc[rows].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                               "target_ids": (rows + h).tolist(), "target_mask": tgt_valid[admissible].tolist(),
                               "support_physical_seconds": int(rows.size) * c.step_seconds}
    return out


def build_tensors(resolved: dict, enum: dict, split: str, c: TaskContract, scaler: dict | None) -> dict:
    """The tensor the model receives, for one split, from the enumerator's origins only."""
    X, Y = resolved["inputs"], resolved["targets"]
    s = enum["splits"][split]
    rows = np.asarray(s.get("origin_ids") or [], dtype=int)
    W, h = enum["window"], enum["horizon"]
    if rows.size == 0:
        return {"X": np.zeros((0, W, X.shape[1])), "y": np.zeros((0, Y.shape[1])), "mask_y": np.zeros((0, Y.shape[1]), dtype=bool), "origins": rows}
    Xw = np.stack([X[t - W + 1:t + 1] for t in rows])
    y = Y[rows + h]
    if scaler is not None:
        Xw = (Xw - scaler["mean"]) / scaler["sd"]
    return {"X": Xw, "y": y, "mask_y": np.asarray(s["target_mask"], dtype=bool), "origins": rows, "target_ids": rows + h}


def fit_scaler(train_tensor: dict, *, grain: str = "windows") -> dict:
    """Train-only per-feature mean and SD, over a DECLARED grain (RP35).

    `windows` weighs each row once per window that contains it, so interior rows count W times; `rows`
    weighs each distinct row once. The two give different moments on overlapping windows, so the grain
    is part of the scaler's identity and travels with it. Neither reads validation or test.
    """
    if grain not in ("windows", "rows"):
        raise ContractRefusal(f"unknown scaler grain {grain!r}")
    X, origins, W = train_tensor["X"], np.asarray(train_tensor.get("origins", []), dtype=np.int64), int(train_tensor["X"].shape[1])
    if grain == "windows" or origins.size == 0:
        flat = X.reshape(-1, X.shape[2])
        n_unique = None
    else:
        rows = np.unique((origins[:, None] - W + 1 + np.arange(W)[None, :]).ravel())
        index = {int(r): k for k, r in enumerate(rows)}
        flat = np.empty((rows.size, X.shape[2]), dtype=X.dtype)
        for i, o in enumerate(origins):
            for k, r in enumerate(range(int(o) - W + 1, int(o) + 1)):
                flat[index[r]] = X[i, k]
        n_unique = int(rows.size)
    mean, sd = flat.mean(axis=0), flat.std(axis=0)
    return {"mean": mean, "sd": np.where(sd > 0, sd, 1.0), "fitted_on": "train windows only", "grain": grain,
            "n_rows": int(flat.shape[0]), "unique_rows": n_unique,
            "grain_note": "windows: each row counted once per window containing it; rows: each distinct row once"}


def eval_set(enum: dict, split: str) -> dict:
    """The common evaluation set: origins/targets/mask every arm must score; a denominator per target from
    the train seasonal naive is computed by the caller with the same origins."""
    s = enum["splits"][split]
    return {"origins": list(s.get("origin_ids") or []), "target_ids": list(s.get("target_ids") or []), "mask": s.get("target_mask") or [],
            "rule": "every arm scores exactly these; an origin an arm cannot score is dropped for ALL arms and reported as coverage"}


def household_contract(window: int = 60, horizon: int = 1, history: bool = True) -> TaskContract:
    return TaskContract(family="uci_235", timestamp="timestamp_label", ts_format="%d/%m/%Y %H:%M:%S", step_seconds=60,
                        features=["Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"],
                        targets=["Global_active_power"], metadata=[], controls=[], ignore=[], target_history_as_feature=history,
                        activation_rule="none", gap_policy="withdraw", horizon=horizon, window=window)


def electricity_contract(clients: list, window: int = 96, horizon: int = 1, ambiguous: list | None = None) -> TaskContract:
    return TaskContract(family="uci_321", timestamp="timestamp_label", ts_format="%Y-%m-%d %H:%M:%S", step_seconds=900,
                        features=[], targets=list(clients), metadata=[], controls=[], target_history_as_feature=True,
                        activation_rule="first_finite_nonzero_at_or_before_origin", gap_policy="withdraw", ambiguous_labels=list(ambiguous or []),
                        horizon=horizon, window=window)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        c = household_contract(window=4, horizon=1)
        ts = pd.date_range("2007-01-01 00:00", periods=40, freq="min")
        df = pd.DataFrame({"timestamp_label": ts.strftime("%d/%m/%Y %H:%M:%S"), **{k: np.arange(40, dtype=float) + i for i, k in enumerate(c.features + c.targets)}})
        r = resolve(df, c)
        e = enumerate_windows(r, c)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in e["splits"].items()}, indent=1))
