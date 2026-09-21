#!/usr/bin/env python3
"""BENCHMARK-CONTRACTS: a versioned benchmark contract per task, and the comparability it decides.

The owner's rule of 2026-09-21 (PROGRAM_METRICS_CONTRACT_v1, "Adicion obligatoria"): two results are
not comparable because both columns are called MAE. Before a new scientific training, the task's
contract is fixed — target and its construction, resolution, horizon, splits, transformations and
the scaler's fit population, the metric's exact formula and aggregation, the naive baseline, the
reference protocol — and comparability is DECIDED FROM THOSE FIELDS, before any score is read.

Two lanes, no cross-ranking:

  REPRODUCTION                the published task and protocol, including target, transform and the
                              paper's own metric, which is reported first; MAE_z/skill are added
  MATCHED_DOMAIN_COMPARISON   our task, with a literature method RE-EXECUTED on the same rows, scale
                              and population; results of the re-execution are compared, never the
                              paper's numbers
  NOT_COMPARABLE              anything else — with the exact field(s) that differ and the matched
                              run that would resolve it. Design and mechanical tests are allowed;
                              presenting a run as a comparable benchmark is not.

An affine re-expression of a preserved score is valid only when the transform and its parameters
are known and the target, evaluation population and horizon are identical; it never resolves a
difference of target, horizon or split.

This module also carries the refusal the real entry points call: `require(design)` raises before any
scientific training when the design names no contract, so the check lives in the runner and not
only in a document.

    python tools/df_benchmark_contract.py registry --out REGISTRY.json
    python tools/df_benchmark_contract.py decide --ours A.json --theirs B.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

SCHEMA = "benchmark_contract.v1"
MODES = ("REPRODUCTION", "MATCHED_DOMAIN_COMPARISON", "NOT_COMPARABLE")
# the fields whose difference alone makes two results non-comparable
IDENTITY_FIELDS = ("dataset_id", "target", "target_construction", "resolution_seconds", "horizon_steps",
                   "split_rule", "missing_policy", "target_transform", "scaler_fit_population",
                   "metric_formula", "metric_aggregation")


class ContractRefusal(SystemExit):
    """A scientific training without a benchmark contract does not start."""


@dataclass
class BenchmarkContract:
    """What a result must carry to be compared with anything at all."""
    task_id: str
    dataset_id: str
    source: dict                                    # primary source: doi/url, section/table, code revision, or OURS
    target: str
    target_construction: str                        # e.g. "minute-averaged active power, kW, as published"
    resolution_seconds: int
    horizon_steps: int
    horizon_seconds: int
    input_window_steps: int | None
    split_rule: str                                 # dates or proportions, and what is reserved
    missing_policy: str
    target_transform: str                           # "none" | "zscore_train" | "log1p" | "minmax_train" ...
    scaler_fit_population: str                      # "train rows" | "train windows" | "whole series" ...
    metric_formula: str
    metric_scale: str                               # "kW" | "z (train sigma)" | "percent" ...
    metric_aggregation: str                         # "mean over evaluation rows and horizon steps" ...
    naive_baseline: str | None                      # "persistence at the horizon on identical rows" | None
    reference_method: str | None
    tuning_budget: str | None
    checkpoint_selection: str | None
    replicas: str | None
    version: str = "1"
    notes: list = field(default_factory=list)

    def sha256(self) -> str:
        body = {k: v for k, v in asdict(self).items() if k != "notes"}
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()

    def identity(self) -> dict:
        return {k: getattr(self, k) for k in IDENTITY_FIELDS}


def decide(ours: BenchmarkContract, theirs: BenchmarkContract, *, reexecuted_on_our_rows: bool = False) -> dict:
    """Comparability from the FIELDS, before any number is read.

    `reexecuted_on_our_rows`: the reference METHOD was run by us under `ours` (same rows, scale,
    population). That opens the MATCHED_DOMAIN_COMPARISON lane; a published score never does.
    """
    differ = [k for k in IDENTITY_FIELDS if ours.identity()[k] != theirs.identity()[k]]
    if not differ:
        mode = "REPRODUCTION"
        why = "every identity field matches the published protocol; the paper's metric is reported first"
    elif reexecuted_on_our_rows:
        mode = "MATCHED_DOMAIN_COMPARISON"
        why = ("the reference method was re-executed under our contract; its published numbers are "
               "not used, only the re-execution's")
    else:
        mode = "NOT_COMPARABLE"
        why = f"identity fields differ: {differ}; a published score under another protocol is not a comparator"
    return {"mode": mode, "why": why, "fields_that_differ": differ,
            "ours_sha256": ours.sha256(), "theirs_sha256": theirs.sha256(),
            "resolution": (None if mode != "NOT_COMPARABLE" else
                           "a re-execution of the reference method under OUR contract on the same rows "
                           "(MATCHED_DOMAIN_COMPARISON), or an exact reproduction of THEIR protocol "
                           "(REPRODUCTION); never their published number in our comparison column")}


def affine_reexpression(value: float, *, from_transform: dict, to_transform: dict,
                        same_target: bool, same_population: bool, same_horizon: bool) -> dict:
    """Re-express a preserved MAE between two AFFINE scales with known parameters — and refuse otherwise.

    MAE is equivariant under y -> a*y + b only in |a|, so MAE_to = MAE_from * |a_to / a_from| where
    each transform is y_t = (y - mean)/sd (a = 1/sd) or identity (a = 1). Anything non-affine
    (log1p), an unknown parameter, or a different target/population/horizon is refused.
    """
    if not (same_target and same_population and same_horizon):
        return {"ok": False, "why": "a conversion of scale never resolves a difference of target, "
                                    "evaluation population or horizon"}
    def scale(t):
        kind = t.get("kind")
        if kind == "identity":
            return 1.0
        if kind == "zscore":
            sd = t.get("sd")
            if sd is None or not (sd > 0):
                return None
            return 1.0/float(sd)
        return None                                   # log1p, minmax without params, unknown: not affine or unknown
    a_from, a_to = scale(from_transform), scale(to_transform)
    if a_from is None or a_to is None:
        return {"ok": False, "why": "the transform is not affine or its parameters are unknown"}
    return {"ok": True, "value": float(value)*abs(a_to/a_from),
            "formula": "MAE_to = MAE_from * |a_to / a_from| with a = 1/sd for a z-score and 1 for identity"}


def require(design: dict, *, purpose: str = "scientific training") -> dict:
    """The refusal a real entry point calls BEFORE any fit: no contract, no run."""
    contract = design.get("benchmark_contract")
    if not isinstance(contract, dict) or contract.get("schema") != SCHEMA:
        raise ContractRefusal(f"REFUSED: {purpose} without a benchmark contract (design.benchmark_contract "
                              f"with schema {SCHEMA}); comparability is decided from its fields before any score")
    missing = [k for k in ("task_id", "dataset_id", "target", "horizon_steps", "split_rule",
                           "target_transform", "scaler_fit_population", "metric_formula",
                           "naive_baseline", "comparability") if k not in contract]
    if missing:
        raise ContractRefusal(f"REFUSED: the benchmark contract lacks {missing}")
    if contract["comparability"].get("mode") not in MODES:
        raise ContractRefusal("REFUSED: the contract's comparability mode is not one of "
                              f"{MODES}; it is decided from fields, not from a score")
    return contract


# --- the registry: every source read at its origin, and every task of ours --------------------------

def household_ours() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="uci_235.W60_h60.DEV_28d_7d", dataset_id="public.uci.235.individual_household_electric_power_consumption",
        source={"kind": "OURS", "design_sha256": "143abb57d97daa07e3f5228eadf4e3f1a0deb5fe30eb95ca065663f76ff888a7",
                "contract_sha256": "9f4c972043be998150cd09dafaceaf05ed5e9d4e6c9efef4263a8c29d6d53027",
                "task_sheet": "docs/tres_temas_entrevista/program_v3/E1_TASKS.json"},
        target="Global_active_power", target_construction="minute-averaged active power, kW, as published; not aggregated",
        resolution_seconds=60, horizon_steps=60, horizon_seconds=3600, input_window_steps=60,
        split_rule="DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train "
                   "span; family test rows 1763969..2075259 never read",
        missing_policy="rows with a non-finite input withdrawn from the enumeration; nothing imputed",
        target_transform="zscore_train", scaler_fit_population="train windows (each row weighted by the windows containing it)",
        metric_formula="MAE = mean |yhat - y| over the common evaluation set, in kW; MAE_z = MAE / sd_train",
        metric_scale="kW (and z with sd_train 0.9125164391265214)", metric_aggregation="mean over 10020 evaluation origins, one horizon",
        naive_baseline="persistence at the horizon, y(t), on identical origins",
        reference_method=None, tuning_budget="none: declared settings", checkpoint_selection="best validation, patience 3, restored",
        replicas="3 paired seeds; development, not confirmation")


def gasparin_2019() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="IHEPC.15min.day_ahead_96.MIMO", dataset_id="public.uci.235.individual_household_electric_power_consumption",
        source={"kind": "PRIMARY", "citation": "Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series "
                "Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060",
                "read_at": "arXiv PDF v1, sections 7.1-7.4, Tables 3, 4, 5", "code": "NOT PUBLIC (no repository named in the paper or found)"},
        target="Global_active_power", target_construction="resampled to 15-minute values from the 1-minute series (Sec. 7.2)",
        resolution_seconds=900, horizon_steps=96, horizon_seconds=86400, input_window_steps=384,
        split_rule="test = last year of measurements (35 040 samples); remaining data split train/validation keeping "
                   "aside one month every five (Sec. 7.2, Table 3: train 103 301)",
        missing_policy="missing values (~1.25%) reconstructed with the mean power of the same time slot across years",
        target_transform="none stated for IHEPC (standard normalisation stated for GEFCom loads)", scaler_fit_population="not stated for IHEPC",
        metric_formula="RMSE and MAE averaged over N test pairs and the n_O=96 steps (Sec. 7.1); NRMSE% = RMSE/(ymax-ymin)*100 with train max/min; R2",
        metric_scale="kW", metric_aggregation="mean over test pairs and over the 96 horizon steps; mean +- sd of 10 repeated trainings",
        naive_baseline=None,
        reference_method="Table 5 (kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, "
                         "GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53 (RMSE/MAE)",
        tuning_budget="grid search, best configurations in Table 4 (L, n_H, lambda, dropout; TCN k=2, M=32)",
        checkpoint_selection="best configuration on validation, then evaluated once on test", replicas="10 repeated trainings",
        notes=["no naive baseline is reported", "the target is a 15-minute resample and the task is 96-step day-ahead: "
               "neither the target construction nor the horizon matches our 60-minute task"])


def saad_saoud_2022() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="IHEPC.multires.one_step.SWT", dataset_id="public.uci.235.individual_household_electric_power_consumption",
        source={"kind": "PRIMARY", "citation": "Saad Saoud, AlMarzouqi, Hussein (2022), Cascaded Deep Hybrid Models for "
                "Multistep Household Energy Consumption Forecasting, arXiv:2207.02589v2", "read_at": "arXiv PDF v2, Sec. 2-3, Tables 1-3, Fig. 6",
                "code": "NOT PUBLIC"},
        target="Global_active_power", target_construction="minutely, hourly, daily and weekly series of global active power (aggregation rule not stated)",
        resolution_seconds=3600, horizon_steps=1, horizon_seconds=3600, input_window_steps=None,
        split_rule="first three years train, remaining year test (Sec. 3, following Mocanu 2016 / Marino 2016)",
        missing_policy="not stated", target_transform="SWT (db1, 3 levels) sub-bands normalised to [0, 1]",
        scaler_fit_population="not stated; the SWT is computed on the whole series before any split (Sec. 2.2)",
        metric_formula="RMSE, MAE, MAPE on the test set (Table 1)", metric_scale="kW", metric_aggregation="one value per resolution; no repeats stated",
        naive_baseline=None,
        reference_method="Table 1 hourly (RMSE/MAE kW): CNN-LSTM [Kim&Cho re-executed] 0.5957/0.3317, Transformer-SWT 0.4183/0.2637; "
                         "minutely: CNN-LSTM 0.6114/0.3493, Transformer-SWT 0.3929/0.1911",
        tuning_budget="RMSProp lr 0.001, batch 32, 100 epochs", checkpoint_selection="not stated", replicas="not stated",
        notes=["the wavelet transform is computed over the whole series before the split: the class this repository's "
               "causal battery names full_series_dwt_as_time_row and detects; CAUSALITY_UNVERIFIED by construction",
               "the 'hourly' row is a one-step forecast of an HOURLY-aggregated target, not the minute-averaged power 60 minutes ahead"])


def vaygan_2021() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="IHEPC.minute.time_pooling", dataset_id="public.uci.235.individual_household_electric_power_consumption",
        source={"kind": "PRIMARY", "citation": "Vaygan, Rajabi, Estebsari (2021), Short-Term Load Forecasting Using Time Pooling "
                "Deep Recurrent Neural Network, arXiv:2109.12498", "read_at": "arXiv PDF v1, Sec. III-IV, Table I", "code": "NOT PUBLIC"},
        target="Global_active_power", target_construction="1-minute global active power, kW",
        resolution_seconds=60, horizon_steps=None, horizon_seconds=None, input_window_steps=None,
        split_rule="weeks of 10 080 minutes pooled into N=720 half-day groups, M=14; train/test 67/23 within the pools",
        missing_policy="missing data completed by averaging over available data", target_transform="not stated",
        scaler_fit_population="not stated", metric_formula="RMSE and MAE over N test samples (eq. 16-17)", metric_scale="kW",
        metric_aggregation="one value", naive_baseline=None,
        reference_method="Table I (RMSE/MAE): SVR 0.96/0.77, ARIMA 0.81/0.75, RNN 0.75/0.55, DRNN 0.39/0.20, TPRNN 0.37/0.19",
        tuning_budget="not stated", checkpoint_selection="not stated", replicas="not stated",
        notes=["the forecast horizon is never declared", "train and test are drawn from pooled groups inside the same weeks"])


def kim_cho_2019() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="IHEPC.CNN-LSTM.multires", dataset_id="public.uci.235.individual_household_electric_power_consumption",
        source={"kind": "PRIMARY_PAYWALLED", "citation": "Kim & Cho (2019), Predicting residential energy consumption using CNN-LSTM "
                "neural networks, Energy 182:72-81, doi:10.1016/j.energy.2019.05.230", "read_at": "abstract only (publisher page); "
                "numbers seen only as re-executed by Saad Saoud 2022 Table 1", "code": "NOT PUBLIC"},
        target="Global_active_power", target_construction="not verifiable from the abstract", resolution_seconds=None,
        horizon_steps=None, horizon_seconds=None, input_window_steps=None, split_rule="not verifiable from the abstract",
        missing_policy="not verifiable", target_transform="not verifiable", scaler_fit_population="not verifiable",
        metric_formula="RMSE (abstract); values not on the public page", metric_scale="unknown", metric_aggregation="unknown",
        naive_baseline=None, reference_method=None, tuning_budget=None, checkpoint_selection=None, replicas=None,
        notes=["a primary source whose protocol fields are behind a paywall is recorded, not used"])


def fx_eurusd_1h_ours() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="fx.eurusd.1h.FIN-LOSS-OPT", dataset_id="financial_files:market_data/forex/g10/eurusd/1h.parquet",
        source={"kind": "OURS_DESIGNED", "coverage": "129 873 rows, 2005-01-03 01:00 .. 2025-12-31 16:00, time column 'datetime'",
                "holdout": "deny_from 2025-01-01 (policy); reserve never read", "state": "DESIGNED, NOT_STARTED"},
        target="close", target_construction="hourly bar close, as served; horizons declared by the task before scoring",
        resolution_seconds=3600, horizon_steps=None, horizon_seconds=None, input_window_steps=None,
        split_rule="weekly walk-forward folds; retraining weekly; reserve from 2025-01-01 unread",
        missing_policy="non-finite bars withdrawn; weekend gaps kept as gaps (no fill)",
        target_transform="zscore_train (per fold)", scaler_fit_population="train rows of the fold, shared by every arm",
        metric_formula="MAE_z = mean|yhat-y|/sd_train per horizon and fold; RMSE_z; skill = 1 - MAE_z_model/MAE_z_naive",
        metric_scale="z (fold train sigma); original units secondarily", metric_aggregation="per horizon, fold, seed and arm; paired",
        naive_baseline="persistence at the horizon on identical rows", reference_method=None,
        tuning_budget="equal, declared per loss family; defaults arm explicit", checkpoint_selection="common monitor, restore best",
        replicas="paired seeds within host blocks",
        notes=["no literature reference bound yet: NOT_COMPARABLE until a matched re-execution exists"])


def rl_weekly_ours() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="e3.weekly_long_flat", dataset_id="financial_files (asset declared per campaign)",
        source={"kind": "OURS", "contract": "docs/tres_temas_entrevista/program_v3/13C_E3_WEEKLY_CYCLE_CONTRACT_2026_09_18.md"},
        target="net weekly return with capital and costs explicit", target_construction="weeks Monday 00:00 UTC .. Sunday 23:00 UTC",
        resolution_seconds=3600, horizon_steps=None, horizon_seconds=None, input_window_steps=None,
        split_rule="weekly cycle: cutoff <= fit_start < fit_end <= release <= first_decision", missing_policy="per environment contract",
        target_transform="none", scaler_fit_population="n/a",
        metric_formula="net return, drawdown, Sharpe (weekly periodicity stated), turnover, exposure, observed steps",
        metric_scale="return units with capital and costs", metric_aggregation="per week and asset; uncertainty across weeks/regimes",
        naive_baseline="trivial policies: flat, buy-and-hold, random with the same turnover; competent non-modular control",
        reference_method=None, tuning_budget=None, checkpoint_selection=None, replicas="weeks and regimes",
        notes=["RL keeps its own reward/return/risk and execution contract; no forecast MAE is invented for the agent"])


def registry() -> dict:
    ours = household_ours()
    sources = {"gasparin_2019": gasparin_2019(), "saad_saoud_2022": saad_saoud_2022(),
               "vaygan_2021": vaygan_2021(), "kim_cho_2019": kim_cho_2019()}
    decisions = {name: decide(ours, c) for name, c in sources.items()}
    return {"schema": "benchmark_contract_registry.v1", "contract_schema": SCHEMA,
            "ours": {"household_W60_h60": asdict(ours), "fx_eurusd_1h": asdict(fx_eurusd_1h_ours()),
                     "rl_weekly": asdict(rl_weekly_ours())},
            "literature": {k: asdict(v) for k, v in sources.items()},
            "decisions_against_household_W60_h60": decisions,
            "reading": "every literature entry above was read at its source and its fields transcribed; every decision "
                       "is NOT_COMPARABLE because identity fields differ, and the resolution named is a re-execution "
                       "under our contract or an exact reproduction of theirs — never their number in our column"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["registry", "decide"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ours", type=Path)
    ap.add_argument("--theirs", type=Path)
    a = ap.parse_args(argv)
    if a.command == "registry":
        doc = registry()
        if a.out:
            a.out.write_text(json.dumps(doc, indent=1, default=str))
        print(json.dumps({k: v["mode"] for k, v in doc["decisions_against_household_W60_h60"].items()}, indent=1))
        return 0
    ours = BenchmarkContract(**json.loads(a.ours.read_text()))
    theirs = BenchmarkContract(**json.loads(a.theirs.read_text()))
    print(json.dumps(decide(ours, theirs), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
