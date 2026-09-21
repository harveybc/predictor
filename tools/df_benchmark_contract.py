#!/usr/bin/env python3
"""BENCHMARK-CONTRACTS (RP67): a typed benchmark contract bound to what the runner actually consumes.

Musashi's probes against the previous version (4ef9f71) showed four holes, each closed here:

  * changing `metric_scale` to USD or `horizon_seconds` to 72 h still returned REPRODUCTION —
    now both are identity fields, and physical time must be consistent (horizon_seconds =
    horizon_steps * resolution_seconds) or the contract is refused outright;
  * a bare `reexecuted_on_our_rows=True` promoted a mismatched protocol — the boolean is gone; the
    MATCHED lane needs REFERENCE EVIDENCE: a closed reference run under the same contract digest,
    and until it exists the state is PLANNED_REFERENCE, never a comparable score;
  * `require()` accepted null fields and a self-declared mode — fields are typed with finite domains,
    the mode is recomputed from the fields and must agree with the declaration;
  * the factorial validator accepted a foreign target/horizon with a recomputed outer digest — the
    contract carries its own canonical digest, and `bind()` checks it against the RUNTIME
    preparation (DATA.json: panel digest, target column, horizon, window, evaluation population,
    scaler parameters); a stale inner digest and a consistently re-digested foreign task both refuse.

Deliberate contrasts (input or model varied under one estimand) are DECLARED in `varying_factors`
and are not silently ignored: two contracts that declare the same contrast are comparable within it.

Lanes: REPRODUCTION (their protocol, their metric first) / MATCHED_DOMAIN_COMPARISON (their method
re-executed under our contract — with evidence) / NOT_COMPARABLE (with the differing fields and the
resolution). Comparator states: PLANNED_REFERENCE (protocol ready, no result) / VERIFIED_COMPARATOR
(a closed run under this contract digest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import math
from dataclasses import dataclass, field, asdict, fields, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "benchmark_contract.v2"
MODES = ("REPRODUCTION", "MATCHED_DOMAIN_COMPARISON", "NOT_COMPARABLE")
COMPARATOR_STATES = ("PLANNED_REFERENCE", "LOCALLY_CHECKED_REFERENCE", "VERIFIED_COMPARATOR", "NONE")
TRANSFORMS = ("none", "zscore_train", "log1p", "minmax_train", "swt_whole_series", "unknown")
SCALES = ("native", "z_train", "log1p", "minmax", "percent", "unknown")
# a difference in any of these, outside a declared contrast, makes two results non-comparable
IDENTITY_FIELDS = ("dataset_id", "target", "target_construction", "resolution_seconds", "horizon_steps",
                   "horizon_seconds", "split_rule", "missing_policy", "target_transform",
                   "scaler_fit_population", "metric_formula", "metric_scale", "metric_aggregation",
                   "permitted_inputs")
REQUIRED_NONNULL = ("task_id", "dataset_id", "target", "target_construction", "resolution_seconds",
                    "horizon_steps", "horizon_seconds", "split_rule", "missing_policy", "target_transform",
                    "scaler_fit_population", "metric_formula", "metric_scale", "metric_aggregation",
                    "naive_baseline", "permitted_inputs")


class ContractRefusal(SystemExit):
    """A scientific training without a valid, bound benchmark contract does not start."""


@dataclass
class BenchmarkContract:
    task_id: str
    dataset_id: str
    source: dict
    target: str
    target_construction: str
    resolution_seconds: int
    horizon_steps: int
    horizon_seconds: int
    input_window_steps: int | None
    split_rule: str
    missing_policy: str
    target_transform: str
    scaler_fit_population: str
    metric_formula: str
    metric_scale: str
    metric_aggregation: str
    naive_baseline: str
    permitted_inputs: str
    reference_method: str | None = None
    tuning_budget: str | None = None
    checkpoint_selection: str | None = None
    replicas: str | None = None
    varying_factors: tuple = ()                     # a DECLARED contrast under one estimand
    estimand: str | None = None
    version: str = "2"
    notes: list = field(default_factory=list)

    # --- validity ------------------------------------------------------------------------------------
    def validate(self) -> list:
        """Every reason this contract cannot be used, or an empty list."""
        p = []
        for name in REQUIRED_NONNULL:
            v = getattr(self, name)
            if v is None or (isinstance(v, str) and not v.strip()):
                p.append(f"{name} is null or empty")
        for name in ("resolution_seconds", "horizon_steps", "horizon_seconds"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                p.append(f"{name} must be a positive integer, not {v!r}")
        if self.input_window_steps is not None and (not isinstance(self.input_window_steps, int) or self.input_window_steps <= 0):
            p.append("input_window_steps must be a positive integer or None")
        if all(isinstance(getattr(self, n), int) and getattr(self, n) > 0 for n in ("resolution_seconds", "horizon_steps", "horizon_seconds")):
            if self.horizon_seconds != self.horizon_steps*self.resolution_seconds:
                p.append(f"physical time is inconsistent: horizon_seconds {self.horizon_seconds} != "
                         f"horizon_steps {self.horizon_steps} x resolution_seconds {self.resolution_seconds}")
        if self.target_transform not in TRANSFORMS:
            p.append(f"target_transform {self.target_transform!r} is not one of {TRANSFORMS}")
        if self.metric_scale not in SCALES:
            p.append(f"metric_scale {self.metric_scale!r} is not one of {SCALES}")
        if self.target_transform == "zscore_train" and self.metric_scale not in ("z_train", "native"):
            p.append("a z-score transform reports in z_train or in native units after inversion; nothing else")
        if self.varying_factors and not self.estimand:
            p.append("a declared contrast needs an estimand")
        for f in self.varying_factors:
            if f not in IDENTITY_FIELDS:
                p.append(f"varying factor {f!r} is not an identity field")
        return p

    def sha256(self) -> str:
        body = {k: v for k, v in asdict(self).items() if k not in ("notes",)}
        body["varying_factors"] = sorted(self.varying_factors)
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()

    def identity(self) -> dict:
        return {k: getattr(self, k) for k in IDENTITY_FIELDS}

    def to_design_block(self, *, comparability: dict) -> dict:
        problems = self.validate()
        if problems:
            raise ContractRefusal(f"REFUSED: the contract is invalid: {problems}")
        body = asdict(self)
        body["varying_factors"] = sorted(self.varying_factors)
        body.update(schema=SCHEMA, contract_sha256=self.sha256(), comparability=comparability)
        return body

    @classmethod
    def from_block(cls, block: dict) -> "BenchmarkContract":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in block.items() if k in names}
        if "varying_factors" in kw:
            kw["varying_factors"] = tuple(kw["varying_factors"] or ())
        return cls(**kw)


# --- comparability, decided from fields ---------------------------------------------------------------

def reference_evidence(ours: BenchmarkContract, reference_run: Path | None, *, reference_arm: str | None = None,
                       warehouse=None, seeds: tuple | None = None) -> dict:
    """What a MATCHED lane needs (RP74/A2): a CLOSED reference run whose sealed design carries THIS contract digest,
    whose reference ARM is named, whose forecast population is COMPLETE (every declared seed), and whose every
    forecast is anchored by the accepted artifact chain and scored independently (tools/df_closure_table.py). A
    contract hash plus any COMPLETED unit is preparation, not a comparator; without a warehouse read the state is
    LOCALLY_CHECKED_REFERENCE, never VERIFIED_COMPARATOR."""
    if reference_run is None:
        return {"state": "PLANNED_REFERENCE", "why": "no reference run root was given"}
    root = Path(reference_run)
    design_path, receipts_path = root/"DESIGN.json", root/"TERMINAL_RECEIPTS.json"
    if not design_path.is_file() or not receipts_path.is_file():
        return {"state": "PLANNED_REFERENCE", "why": f"{root} holds no sealed design with accepted terminals"}
    design = json.loads(design_path.read_text())
    block = design.get("benchmark_contract") or {}
    if block.get("contract_sha256") != ours.sha256():
        return {"state": "PLANNED_REFERENCE", "why": f"the reference run's contract {str(block.get('contract_sha256'))[:12]} is not ours {ours.sha256()[:12]}"}
    if not isinstance(design.get("design_sha256"), str) or len(design["design_sha256"]) != 64:
        return {"state": "PLANNED_REFERENCE", "why": "the reference run has no sealed design digest"}
    cells = [c for c in design.get("cells") or [] if isinstance(c, dict) and c.get("arm")]
    arms = sorted({c["arm"] for c in cells})
    if reference_arm is None or reference_arm not in arms:
        return {"state": "PLANNED_REFERENCE", "why": f"the reference ARM must be named and registered in the design; registered arms: {arms}"}
    units = [c["cell_id"] for c in cells if c["arm"] == reference_arm]
    want_seeds = sorted(set(seeds) if seeds else {c["seed"] for c in cells if c["arm"] == reference_arm})
    have_seeds = sorted({c["seed"] for c in cells if c["arm"] == reference_arm})
    receipts = (json.loads(receipts_path.read_text()) or {}).get("units") or {}
    missing = [u for u in units if u not in receipts]
    if missing or have_seeds != want_seeds:
        return {"state": "PLANNED_REFERENCE", "why": f"the reference population is incomplete: units without accepted terminal {missing}; "
                                                  f"seeds declared {want_seeds}, present {have_seeds}"}
    T = _module("df_closure_table")
    ver = T.verify_run(root, label=root.name, registry=registry(), warehouse=warehouse)
    if ver["design_identity"].get("recomputes") is False:
        return {"state": "PLANNED_REFERENCE", "why": "the reference design's digest does not recompute from its content (relabeled or edited design)"}
    rows = [r for r in ver["rows"] if r["unit"] in units]
    # RP83: the accepted payload's own tags must name THIS arm for every reference cell (a correct forecast is not its label)
    for r in rows:
        if "arm" not in ((r.get("warehouse") or {}).get("accepted_tags_checked") or []) and warehouse is not None:
            return {"state": "PLANNED_REFERENCE", "why": f"{r['unit']}: the accepted terminal carries no arm tag to bind the reference identity"}
    n_expected = (ours.source or {}).get("evaluation_origins")
    bad = [f"{r['unit']}: {r['problems'] or 'custody ' + r['custody']['class']}" for r in rows
           if r["problems"] or r["model_error"] is None or (n_expected is not None and r["n_evaluated"] != n_expected)]
    if len(rows) != len(units) or bad:
        return {"state": "PLANNED_REFERENCE", "why": f"the reference forecasts do not verify: {bad or 'rows missing'}", "units": units}
    if warehouse is None or any(r["custody"]["class"] != "ACCEPTED_ARTIFACT_CHAIN" or not r.get("verified") for r in rows):
        return {"state": "LOCALLY_CHECKED_REFERENCE", "why": "arrays, labels, population and metrics check locally; the accepted artifact "
                                                          "chain was not read from the warehouse, so this is not a verified comparator yet",
                "run": str(root), "design_sha256": design["design_sha256"], "units": units,
                "derived_mae_z": {r["unit"]: r["model_error_z"] for r in rows}}
    return {"state": "VERIFIED_COMPARATOR", "run": str(root), "design_sha256": design["design_sha256"], "reference_arm": reference_arm,
            "units": units, "seeds": have_seeds, "n_evaluated": rows[0]["n_evaluated"],
            "derived_mae_z": {r["unit"]: r["model_error_z"] for r in rows},
            "custody": "ACCEPTED_ARTIFACT_CHAIN for every forecast; metrics derived independently from the arrays"}


def _module(name: str):
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def decide(ours: BenchmarkContract, theirs: BenchmarkContract, *, reference_run: Path | None = None,
           reference_arm: str | None = None, warehouse=None) -> dict:
    """Comparability from the FIELDS, before any number is read. A boolean never opens a lane."""
    for c, who in ((ours, "ours"), (theirs, "theirs")):
        bad = c.validate()
        if bad:
            return {"mode": "NOT_COMPARABLE", "why": f"the {who} contract is invalid: {bad}",
                    "fields_that_differ": [], "comparator_state": "NONE",
                    "resolution": "repair the contract; nothing is comparable to an invalid contract"}
    # an 'unknown' placeholder is a recorded gap, never a match: two unknowns are not the same protocol
    unknown = [k for c in (ours, theirs) for k, v in c.identity().items()
               if isinstance(v, str) and v.strip().lower().startswith("unknown")]
    if unknown:
        return {"mode": "NOT_COMPARABLE", "why": f"unknown identity fields cannot match as proof: {sorted(set(unknown))}",
                "fields_that_differ": sorted(set(unknown)), "declared_contrast": [], "comparator_state": "NONE",
                "reference_evidence": {"state": "NONE", "why": "an unknown protocol has no reference lane; read the source first"},
                "ours_sha256": ours.sha256(), "theirs_sha256": theirs.sha256(),
                "resolution": "read the primary source (or its code) and fill the field; a placeholder is not a protocol"}
    declared = set(ours.varying_factors) & set(theirs.varying_factors)
    if ours.varying_factors and theirs.varying_factors and ours.estimand != theirs.estimand:
        declared = set()
    differ = [k for k in IDENTITY_FIELDS if ours.identity()[k] != theirs.identity()[k] and k not in declared]
    evidence = reference_evidence(ours, reference_run, reference_arm=reference_arm, warehouse=warehouse)
    if not differ:
        mode, why = "REPRODUCTION", ("every identity field matches" + (f" (contrast declared on {sorted(declared)})" if declared else "")
                                     + "; the paper's metric is reported first")
        state = evidence["state"] if reference_run else "NONE"
    elif evidence["state"] == "VERIFIED_COMPARATOR":
        mode, why = "MATCHED_DOMAIN_COMPARISON", ("the reference METHOD was re-executed under our contract and closed "
                                                  f"({evidence['run']}); its published numbers are not used")
        state = "VERIFIED_COMPARATOR"
    else:
        mode = "NOT_COMPARABLE"
        why = (f"identity fields differ: {differ}; " +
               ("a reference re-execution is planned but has no closed run yet" if reference_run else
                "a published score under another protocol is not a comparator"))
        state = evidence["state"] if reference_run else "NONE"
    return {"mode": mode, "why": why, "fields_that_differ": differ, "declared_contrast": sorted(declared),
            "comparator_state": state, "reference_evidence": evidence,
            "ours_sha256": ours.sha256(), "theirs_sha256": theirs.sha256(),
            "resolution": (None if mode != "NOT_COMPARABLE" else
                           "a re-execution of the reference method under OUR contract, CLOSED under this contract "
                           "digest (MATCHED_DOMAIN_COMPARISON), or an exact reproduction of THEIR protocol "
                           "(REPRODUCTION); never their published number in our comparison column")}


def affine_reexpression(value: float, *, from_transform: dict, to_transform: dict,
                        same_target: bool, same_population: bool, same_horizon: bool) -> dict:
    """MAE between two AFFINE scales with known, valid parameters — and refusal otherwise."""
    if not (same_target and same_population and same_horizon):
        return {"ok": False, "why": "a conversion of scale never resolves a difference of target, evaluation population or horizon"}
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return {"ok": False, "why": f"the value {value!r} is not a finite non-negative error"}

    def scale(t):
        kind = t.get("kind")
        if kind == "identity":
            return 1.0
        if kind == "zscore":
            sd = t.get("sd")
            if not isinstance(sd, (int, float)) or not math.isfinite(sd) or sd <= 0:
                return None
            return 1.0/float(sd)
        return None
    a_from, a_to = scale(from_transform), scale(to_transform)
    if a_from is None or a_to is None:
        return {"ok": False, "why": "the transform is not affine or its parameters are unknown or invalid (sigma must be finite and > 0)"}
    return {"ok": True, "value": float(value)*abs(a_to/a_from),
            "formula": "MAE_to = MAE_from * |a_to / a_from| with a = 1/sd for a z-score and 1 for identity"}


# --- the refusal the real entry points call, bound to the runtime preparation --------------------------

def planned_reference(ours: BenchmarkContract, *, why: str) -> dict:
    """The decision for a task with NO literature contract yet: nothing is comparable, a reference is planned."""
    bad = ours.validate()
    if bad:
        raise ContractRefusal(f"REFUSED: the contract is invalid: {bad}")
    return {"mode": "NOT_COMPARABLE", "why": why, "fields_that_differ": [], "declared_contrast": [],
            "comparator_state": "PLANNED_REFERENCE", "reference_evidence": {"state": "PLANNED_REFERENCE", "why": why},
            "ours_sha256": ours.sha256(), "theirs_sha256": None,
            "resolution": "a re-execution of a reference method under THIS contract, CLOSED under this digest "
                          "(MATCHED_DOMAIN_COMPARISON); never a published number from another protocol"}


def require(design: dict, *, purpose: str = "scientific training") -> BenchmarkContract:
    """No contract, no run — and a contract whose fields, digest or mode do not hold is no contract."""
    block = design.get("benchmark_contract")
    if not isinstance(block, dict) or block.get("schema") != SCHEMA:
        raise ContractRefusal(f"REFUSED: {purpose} without a benchmark contract (design.benchmark_contract "
                              f"with schema {SCHEMA}); comparability is decided from its fields before any score")
    try:
        contract = BenchmarkContract.from_block(block)
    except TypeError as exc:
        raise ContractRefusal(f"REFUSED: the benchmark contract is malformed: {exc}")
    problems = contract.validate()
    if problems:
        raise ContractRefusal(f"REFUSED: the benchmark contract is invalid: {problems}")
    if block.get("contract_sha256") != contract.sha256():
        raise ContractRefusal("REFUSED: the contract's own digest does not recompute from its fields (stale or edited)")
    comp = block.get("comparability") or {}
    if comp.get("mode") not in MODES or comp.get("comparator_state") not in COMPARATOR_STATES:
        raise ContractRefusal("REFUSED: comparability must carry a mode and a comparator_state decided from fields")
    if comp.get("ours_sha256") != contract.sha256():
        raise ContractRefusal("REFUSED: the comparability decision was taken for another contract")
    return contract


def bind(contract: BenchmarkContract, data_json: dict, *, purpose: str = "scientific training") -> dict:
    """The contract against the RUNTIME preparation: what the runner will actually consume."""
    p = []
    inputs = data_json.get("input_columns") or []
    j = data_json.get("target_channel")
    actual_target = inputs[j] if isinstance(j, int) and 0 <= j < len(inputs) else None
    if actual_target != contract.target:
        p.append(f"target: contract {contract.target!r}, prepared data {actual_target!r}")
    if int(data_json.get("horizon", -1)) != contract.horizon_steps:
        p.append(f"horizon_steps: contract {contract.horizon_steps}, prepared data {data_json.get('horizon')}")
    if contract.input_window_steps is not None and int(data_json.get("window", -1)) != contract.input_window_steps:
        p.append(f"input_window_steps: contract {contract.input_window_steps}, prepared data {data_json.get('window')}")
    src = contract.source or {}
    if src.get("panel_sha256") and src["panel_sha256"] != data_json.get("panel_sha256"):
        p.append("dataset identity: the contract names other panel bytes than the prepared data")
    if contract.target_transform == "zscore_train":
        scaler = data_json.get("scaler") or {}
        sd = (scaler.get("sd") or [None]*(len(inputs)))[j] if isinstance(j, int) else None
        if not isinstance(sd, (int, float)) or not math.isfinite(sd) or sd <= 0:
            p.append("scaler: the prepared data carries no finite positive sd for the target")
        if src.get("sd_train") is not None and sd is not None and abs(float(src["sd_train"])-float(sd)) > 1e-9:
            p.append(f"scaler identity: contract sd_train {src['sd_train']} != prepared {sd}")
    en = (data_json.get("enumerator") or {}).get("validation") or {}
    if src.get("evaluation_origins") is not None and en.get("admissible") != src["evaluation_origins"]:
        p.append(f"evaluation population: contract {src['evaluation_origins']}, prepared {en.get('admissible')}")
    if p:
        raise ContractRefusal(f"REFUSED: {purpose}: the benchmark contract does not bind to the prepared data: {p}")
    return {"bound": True, "target": actual_target, "horizon_steps": contract.horizon_steps,
            "panel_sha256": data_json.get("panel_sha256"), "evaluation_origins": en.get("admissible")}


# --- the registry ----------------------------------------------------------------------------------------

def household_ours() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="uci_235.W60_h60.DEV_28d_7d", dataset_id="public.uci.235.individual_household_electric_power_consumption",
        source={"kind": "OURS", "design_sha256": "143abb57d97daa07e3f5228eadf4e3f1a0deb5fe30eb95ca065663f76ff888a7",
                "panel_sha256": "b3192c0bcb117b2ee120a906dbcfb9550cd907abff74fea9bc2b1aa320ebc8db",
                "sd_train": 0.9125164391265214, "evaluation_origins": 10020,
                "task_sheet": "docs/tres_temas_entrevista/program_v3/E1_TASKS.json"},
        target="Global_active_power", target_construction="minute-averaged active power, kW, as published; not aggregated",
        resolution_seconds=60, horizon_steps=60, horizon_seconds=3600, input_window_steps=60,
        split_rule="DEV slice rows 1412361..1462761 (28 d train / 7 d validation) inside the family's train span; family test never read",
        missing_policy="rows with a non-finite input withdrawn from the enumeration; nothing imputed",
        target_transform="zscore_train", scaler_fit_population="train windows (each row weighted by the windows containing it)",
        metric_formula="MAE = mean |yhat - y| over the common evaluation set; MAE_z = MAE / sd_train",
        metric_scale="z_train", metric_aggregation="mean over 10020 evaluation origins, one horizon",
        naive_baseline="persistence at the horizon, y(t), on identical origins",
        permitted_inputs="the 7 declared channels (6 features + the target's own history), window 60",
        reference_method=None, tuning_budget="none: declared settings",
        checkpoint_selection="best validation MAE, patience 3 epochs, restored", replicas="3 paired seeds; development")


def gasparin_2019() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="IHEPC.15min.day_ahead_96.MIMO", dataset_id="public.uci.235.individual_household_electric_power_consumption",
        source={"kind": "PRIMARY", "citation": "Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series "
                "Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060",
                "read_at": "arXiv PDF v1, sections 7.1-7.4, Tables 3, 4, 5", "code": "NOT PUBLIC"},
        target="Global_active_power", target_construction="resampled to 15-minute values from the 1-minute series (Sec. 7.2)",
        resolution_seconds=900, horizon_steps=96, horizon_seconds=86400, input_window_steps=384,
        split_rule="test = last year (Table 3: 35 040 samples); remaining data train/validation keeping one month every five",
        missing_policy="missing values (~1.25%) reconstructed with the mean power of the same time slot across years (split side unstated)",
        target_transform="unknown", scaler_fit_population="not stated for IHEPC",
        metric_formula="RMSE and MAE averaged over N test pairs and the 96 steps (Sec. 7.1); NRMSE% with train max/min; R2",
        metric_scale="native", metric_aggregation="mean over test pairs and 96 steps; mean +- sd of 10 repeated trainings",
        naive_baseline="NONE REPORTED", permitted_inputs="historical load only (Sec. 7.2)",
        reference_method="Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, "
                         "LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53",
        tuning_budget="grid search; Table 4 IHEPC column: GRU L=1 n_H=50 lambda=0.0005 dropout=0; LSTM L=1 n_H=20 lambda=0.001; "
                      "TCN L=8, k=2, M=32, lambda=0.005, dropout=0.1 (TCN has NO n_H entry)",
        checkpoint_selection="best configuration on validation, one evaluation on test", replicas="10 repeated trainings",
        notes=["no naive baseline reported", "Table 3 gives sample counts, not an exact window enumeration",
               "optimizer, seeds, exact dates and code are not stated: limitations, not facts to invent"])


def saad_saoud_2022() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="IHEPC.multires.one_step.SWT", dataset_id="public.uci.235.individual_household_electric_power_consumption",
        source={"kind": "PRIMARY", "citation": "Saad Saoud, AlMarzouqi, Hussein (2022), Cascaded Deep Hybrid Models for "
                "Multistep Household Energy Consumption Forecasting, arXiv:2207.02589v2", "read_at": "arXiv PDF v2, Sec. 2-3, Tables 1-3",
                "code": "NOT PUBLIC"},
        target="Global_active_power", target_construction="hourly series of global active power (aggregation rule not stated)",
        resolution_seconds=3600, horizon_steps=1, horizon_seconds=3600, input_window_steps=None,
        split_rule="first three years train, remaining year test (Sec. 3)", missing_policy="not stated",
        target_transform="swt_whole_series", scaler_fit_population="not stated; the SWT is computed on the whole series before any split",
        metric_formula="RMSE, MAE, MAPE on the test set (Table 1)", metric_scale="native", metric_aggregation="one value per resolution",
        naive_baseline="NONE REPORTED", permitted_inputs="the target's own SWT sub-bands",
        reference_method="Table 1 hourly (RMSE/MAE kW): CNN-LSTM 0.5957/0.3317, Transformer-SWT 0.4183/0.2637",
        tuning_budget="RMSProp lr 0.001, batch 32, 100 epochs", checkpoint_selection="not stated", replicas="not stated",
        notes=["whole-series wavelet before the split: the causal battery's full_series_dwt_as_time_row class; CAUSALITY_UNVERIFIED"])


def vaygan_2021() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="IHEPC.minute.time_pooling", dataset_id="public.uci.235.individual_household_electric_power_consumption",
        source={"kind": "PRIMARY", "citation": "Vaygan, Rajabi, Estebsari (2021), Short-Term Load Forecasting Using Time Pooling "
                "Deep Recurrent Neural Network, arXiv:2109.12498", "read_at": "arXiv PDF v1, Sec. III-IV, Table I", "code": "NOT PUBLIC"},
        target="Global_active_power", target_construction="1-minute global active power, kW",
        resolution_seconds=60, horizon_steps=1, horizon_seconds=60, input_window_steps=None,
        split_rule="weeks of 10 080 minutes pooled into N=720 half-day groups, M=14; train/test 67/23 within the pools",
        missing_policy="missing data completed by averaging over available data", target_transform="unknown",
        scaler_fit_population="not stated", metric_formula="RMSE and MAE over N test samples (eq. 16-17)", metric_scale="native",
        metric_aggregation="one value", naive_baseline="NONE REPORTED", permitted_inputs="the target's own history",
        reference_method="Table I (RMSE/MAE): SVR 0.96/0.77, ARIMA 0.81/0.75, RNN 0.75/0.55, DRNN 0.39/0.20, TPRNN 0.37/0.19",
        tuning_budget="not stated", checkpoint_selection="not stated", replicas="not stated",
        notes=["the forecast horizon is never declared: horizon_steps=1 is an ASSUMPTION recorded as such",
               "train and test are drawn from pooled groups inside the same weeks"])


def kim_cho_2019() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="IHEPC.CNN-LSTM.multires", dataset_id="public.uci.235.individual_household_electric_power_consumption",
        source={"kind": "PRIMARY_PAYWALLED", "citation": "Kim & Cho (2019), Predicting residential energy consumption using CNN-LSTM "
                "neural networks, Energy 182:72-81, doi:10.1016/j.energy.2019.05.230", "read_at": "abstract only", "code": "NOT PUBLIC"},
        target="Global_active_power", target_construction="unknown (paywalled)", resolution_seconds=60, horizon_steps=1,
        horizon_seconds=60, input_window_steps=None, split_rule="unknown (paywalled)", missing_policy="unknown",
        target_transform="unknown", scaler_fit_population="unknown", metric_formula="RMSE (abstract); values not public",
        metric_scale="unknown", metric_aggregation="unknown", naive_baseline="NONE REPORTED", permitted_inputs="unknown",
        notes=["every 'unknown' here is a placeholder that can NEVER match another unknown as proof: the identity "
               "comparison treats it as different by construction (see decide())", "recorded, not used"])


def fx_eurusd_1h_ours() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="fx.eurusd.1h.FIN-LOSS-OPT.h6h", dataset_id="financial_files:market_data/forex/g10/eurusd/1h.parquet",
        source={"kind": "OURS_DESIGNED", "coverage": "129 873 rows, 2005-01-03 01:00 .. 2025-12-31 16:00, time column 'datetime'",
                "holdout": "deny_from 2025-01-01; reserve never read", "state": "DESIGNED, NOT_STARTED"},
        target="close", target_construction="hourly bar close as served; the target of an origin is the bar whose timestamp is "
                                             "origin + h hours ELAPSED (tools/df_fin_task.map_targets); an origin without that exact "
                                             "bar is excluded and counted",
        resolution_seconds=3600, horizon_steps=6, horizon_seconds=21600, input_window_steps=60,
        split_rule="weekly walk-forward folds (Monday 00:00 UTC .. Sunday 23:00 UTC); DEV = the last 26 weeks before 2025-01-01",
        missing_policy="non-finite bars withdrawn; weekend/holiday gaps kept; an origin whose elapsed-time target bar is absent is excluded with reason",
        target_transform="zscore_train", scaler_fit_population="train rows of the fold, shared by every arm of the fold",
        metric_formula="MAE_z = mean|yhat-y|/sd_train per horizon and fold; RMSE_z; skill = 1 - MAE_z_model/MAE_z_naive",
        metric_scale="z_train", metric_aggregation="per horizon, fold, seed and arm; paired by origin",
        naive_baseline="persistence: the origin's close, on identical origins", permitted_inputs="OHLCV of the served bars (5 channels), window 60",
        reference_method=None, tuning_budget="equal enumerated candidates per loss family", checkpoint_selection="common monitor, restore best",
        replicas="paired seeds within host blocks", notes=["no literature reference bound: NOT_COMPARABLE until a matched re-execution exists"])


def rl_weekly_ours() -> BenchmarkContract:
    return BenchmarkContract(
        task_id="e3.weekly_long_flat", dataset_id="financial_files (asset declared per campaign)",
        source={"kind": "OURS", "contract": "docs/tres_temas_entrevista/program_v3/13C_E3_WEEKLY_CYCLE_CONTRACT_2026_09_18.md"},
        target="net weekly return with capital and costs explicit", target_construction="weeks Monday 00:00 UTC .. Sunday 23:00 UTC",
        resolution_seconds=3600, horizon_steps=168, horizon_seconds=604800, input_window_steps=None,
        split_rule="weekly cycle: cutoff <= fit_start < fit_end <= release <= first_decision", missing_policy="per environment contract",
        target_transform="none", scaler_fit_population="n/a",
        metric_formula="net return, drawdown, Sharpe (weekly periodicity stated), turnover, exposure, observed steps",
        metric_scale="native", metric_aggregation="per week and asset; uncertainty across weeks/regimes",
        naive_baseline="trivial policies: flat, buy-and-hold, random with the same turnover; a competent non-modular control",
        permitted_inputs="the environment's observation contract at decision time",
        notes=["RL keeps its own reward/return/risk and execution contract; no forecast MAE is invented for the agent"])


def registry() -> dict:
    ours = household_ours()
    sources = {"gasparin_2019": gasparin_2019(), "saad_saoud_2022": saad_saoud_2022(),
               "vaygan_2021": vaygan_2021(), "kim_cho_2019": kim_cho_2019()}
    decisions = {name: decide(ours, c) for name, c in sources.items()}
    return {"schema": "benchmark_contract_registry.v2", "contract_schema": SCHEMA,
            "ours": {"household_W60_h60": asdict(ours), "fx_eurusd_1h": asdict(fx_eurusd_1h_ours()),
                     "rl_weekly": asdict(rl_weekly_ours())},
            "validity": {k: v.validate() for k, v in {"household_W60_h60": ours, "fx_eurusd_1h": fx_eurusd_1h_ours(),
                                                       "rl_weekly": rl_weekly_ours(), **sources}.items()},
            "literature": {k: asdict(v) for k, v in sources.items()},
            "decisions_against_household_W60_h60": decisions,
            "reading": "every decision is NOT_COMPARABLE because identity fields differ; the MATCHED lane opens only with a "
                       "closed reference run under our contract digest (VERIFIED_COMPARATOR), never with a flag"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["registry", "decide"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ours", type=Path)
    ap.add_argument("--theirs", type=Path)
    ap.add_argument("--reference-run", type=Path)
    a = ap.parse_args(argv)
    if a.command == "registry":
        doc = registry()
        if a.out:
            a.out.write_text(json.dumps(doc, indent=1, default=str))
        print(json.dumps({k: v["mode"] for k, v in doc["decisions_against_household_W60_h60"].items()}, indent=1))
        return 0
    ours = BenchmarkContract.from_block(json.loads(a.ours.read_text()))
    theirs = BenchmarkContract.from_block(json.loads(a.theirs.read_text()))
    print(json.dumps(decide(ours, theirs, reference_run=a.reference_run), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
