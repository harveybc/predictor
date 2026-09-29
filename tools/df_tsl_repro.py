#!/usr/bin/env python3
"""RB02: the matched Weather (then Traffic) reference reproduction of TimeFilter (Hu et al., ICML 2025) on the OFFICIAL processed
TSL benchmarks already registered in the `sota_benchmarks` lake.

This tool does NOT introduce a second model, loader, loss or scorer. It is the dataset-parameterised sibling of
`df_sota_repro.py` (whose module constants freeze it to Electricity) and it reuses that module's author bridge verbatim:
the author's own argparse read by AST from `run.py`, the author's shell script expanded into argv, the author's
`Dataset_Custom`, the author's `Exp_Long_Term_Forecast`, the author's optimizer/loss/early stopping.

Blocks
  seal        the source-and-protocol LOCK for ONE dataset and ONE input length, from the author's script and parser, the
              registered lake receipt and the published tables — sealed BEFORE any byte of the benchmark is read
  characterize the author's `Dataset_Custom` on the delivered bytes: column order, the target column the loader moves last,
              missing values, split borders, window counts per horizon and the training scaler's identity
  receipt     build the `tsl_literature_metrics.v1` governed terminal body for one scored cell
  validate    the PRODUCER gate: a receipt without protocol identity, without scaler identity, without evaluation-population
              identity, with a nonfinite metric, or whose elapsed horizon contradicts its step horizon and the dataset's own
              sample interval, is REFUSED here. The general-purpose warehouse is not a scientific verifier; this is
  pilot       TRAIN-ONLY bounded cost measurement: K optimizer steps of the author's own training loop on the real train
              loader, timed, with the device asserted from INSIDE the child and the whole-cgroup peak recorded. It reads no
              validation batch, evaluates no test window and produces NO metric of any kind
  plan        the frozen full-reproduction design: every cell, its measured unit cost and the total, from a pilot record

Clocks. Horizons 96/192/336/720 are STEPS. Weather steps are 600 s; Electricity and Traffic steps are 3600 s. The same step
count is NOT the same elapsed horizon and this module refuses to emit a receipt that says it is.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import df_sota_repro as S                                            # noqa: E402  the author bridge, reused, never re-typed

SCHEMA = "df_tsl_repro_design.v1"
CONTRACT_PATH = REPO / "docs/contracts/tsl_literature_metrics.v1.json"
AUTHOR_REPO = S.AUTHOR_REPO
PINNED_COMMIT = S.PINNED_COMMIT
LAKE = "sota_benchmarks"
ROLE = "benchmark"

#: TimeFilter, arXiv:2501.13041 (ICML 2025). Table 8 = full results at fixed L = 96; Table 9 = L searched in {192,336,512,720};
#: Table 7 = standard deviation over the paper's runs of the FOUR-HORIZON AVERAGE; Table 6 = the per-dataset hyperparameters.
#: Read 2026-09-28 from the arXiv HTML of v2. Electricity is carried here ONLY as a cross-check of the reading against the
#: values `df_sota_repro.PAPER` has held since RP92; Electricity itself stays owned by that module.
PAPER = {
    "citation": "Hu, Y., Zhang, G., Liu, P., Lan, D., Li, N., Cheng, D., Dai, T., Xia, S.-T., Pan, S. TimeFilter: Patch-Specific "
                "Spatial-Temporal Graph Filtration for Time Series Forecasting. ICML 2025; arXiv:2501.13041. Author code "
                "github.com/TROUBADOUR000/TimeFilter",
    "read_at": "arXiv HTML v2, Tables 6, 7, 8, 9; read 2026-09-28",
    "weather": {
        "L96": {"table": "Table 8 (L = 96, full results)",
                "per_horizon": {"96": {"mse": 0.153, "mae": 0.199}, "192": {"mse": 0.202, "mae": 0.246},
                                "336": {"mse": 0.260, "mae": 0.289}, "720": {"mse": 0.342, "mae": 0.341}},
                "average": {"mse": 0.239, "mae": 0.269}},
        "Lsearched": {"table": "Table 9 (L searched in {192, 336, 512, 720}; the author script offers L = 720 only for Weather)",
                      "per_horizon": {"96": {"mse": 0.141, "mae": 0.193}, "192": {"mse": 0.184, "mae": 0.234},
                                      "336": {"mse": 0.234, "mae": 0.276}, "720": {"mse": 0.305, "mae": 0.327}},
                      "average": {"mse": 0.216, "mae": 0.258}},
        "std_of_average": {"table": "Table 7 (Weather 0.239+-0.006 / 0.269+-0.004)", "mse": 0.006, "mae": 0.004},
        "table6": {"patch_len": 48, "e_layers": 2, "learning_rate": 5e-4, "d_model": 128, "d_ff": 256, "train_epochs": 10,
                   "batch_size": 32, "reading": "Table 6 lists patch length, encoder layers, lr, d_model, d_ff and epochs; it "
                                                "does not state dropout, patience, lradj, n_heads, alpha, top_p or pos"}},
    "traffic": {
        "L96": {"table": "Table 8 (L = 96, full results)",
                "per_horizon": {"96": {"mse": 0.375, "mae": 0.251}, "192": {"mse": 0.395, "mae": 0.262},
                                "336": {"mse": 0.414, "mae": 0.271}, "720": {"mse": 0.445, "mae": 0.289}},
                "average": {"mse": 0.407, "mae": 0.268}},
        "Lsearched": {"table": "Table 9 (L searched in {192, 336, 512, 720}; the author script offers L = 720 only for Traffic)",
                      "per_horizon": {"96": {"mse": 0.332, "mae": 0.238}, "192": {"mse": 0.348, "mae": 0.246},
                                      "336": {"mse": 0.365, "mae": 0.256}, "720": {"mse": 0.396, "mae": 0.275}},
                      "average": {"mse": 0.360, "mae": 0.254}},
        "std_of_average": {"table": "Table 7 (Traffic 0.407+-0.008 / 0.268+-0.004)", "mse": 0.008, "mae": 0.004},
        "table6": {"patch_len": 96, "e_layers": 3, "learning_rate": 1e-3, "d_model": 512, "d_ff": 2048, "train_epochs": 30,
                   "batch_size": 16, "reading": "Table 6 agrees with scripts/Traffic.sh on every field it states; the script "
                                                "additionally sets dropout 0.3, top_p 0.0 and pos 0, none of which the paper states"}},
    "electricity_crosscheck": {"L96": {"96": {"mse": 0.133, "mae": 0.230}, "192": {"mse": 0.154, "mae": 0.248},
                                       "336": {"mse": 0.162, "mae": 0.261}, "720": {"mse": 0.184, "mae": 0.284}},
                               "agrees_with_df_sota_repro": True},
    "reported_precision": 0.0005,
}

#: The author script each dataset is driven by, and the ONE input length each of its two blocks offers.
SCRIPTS = {"weather": {"script": "scripts/Weather.sh", "data_name": "weather.csv", "L96": 96, "Lsearched": 720},
           "traffic": {"script": "scripts/Traffic.sh", "data_name": "traffic.csv", "L96": 96, "Lsearched": 720},
           "electricity": {"script": "scripts/ECL.sh", "data_name": "electricity.csv", "L96": 96, "Lsearched": 512}}

PLACEHOLDERS = {"", "fixture", "unknown", "UNKNOWN", "none", "None", "null", "NA", "n/a", "todo", "TODO", "tbd", "TBD", "-"}
_HEX = set("0123456789abcdef")


class TslRefusal(SystemExit):
    """A reproduction whose identity cannot be shown does not emit a receipt."""


def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


def dataset_facts(dataset: str) -> dict:
    """The registered resource's own facts, from the receipt contract — never re-derived from a number in prose."""
    d = contract()["datasets"]
    if dataset not in d:
        raise TslRefusal(f"REFUSED: {dataset!r} is not a registered resource of the {LAKE} lake: {sorted(d)}")
    return {**d[dataset], "dataset": dataset}


def is_digest(value, length: int = 64) -> bool:
    v = str(value or "")
    return len(v) == length and set(v.lower()) <= _HEX


# --- the lock and the sealed design ---------------------------------------------------------------------------------------

def horizon_seconds(dataset: str, horizon_steps: int) -> int:
    """The ONLY place elapsed horizon is computed: the registered resource's own sample interval times the step count."""
    return int(horizon_steps) * int(dataset_facts(dataset)["step_seconds"])


def borders(rows: int, seq_len: int, horizon: int) -> dict:
    """The author's `Dataset_Custom.__read_data__` arithmetic, from the registered row count alone (no byte is read)."""
    n = int(rows)
    num_train, num_test = int(n * 0.7), int(n * 0.2)
    num_vali = n - num_train - num_test
    b1 = [0, num_train - seq_len, n - num_test - seq_len]
    b2 = [num_train, num_train + num_vali, n]
    win = {flag: max(0, (b2[i] - b1[i]) - seq_len - horizon + 1) for i, flag in enumerate(("train", "vali", "test"))}
    return {"rows": {"train": num_train, "vali": num_vali, "test": num_test}, "border1s": b1, "border2s": b2, "windows": win}


def agreement(dataset: str, protocol: str) -> dict:
    """The agreement criterion is built from THIS dataset's own Table 7 dispersion. Electricity's margin is not Weather's."""
    std = PAPER[dataset]["std_of_average"]
    if std.get("mse") is None or std.get("mae") is None:
        return {"state": "NO_MARGIN_EXISTS", "reading": std.get("reading", "the published dispersion for this dataset was not read"),
                "rule": "no agreement class may be assigned to this dataset until its own published row and dispersion are pinned"}
    return {"rule": "per horizon and for the four-horizon average (formed WITHIN each seed first), the seed mean of the replicated "
                    "metric is in OPERATIONAL_AGREEMENT with the published value when |mean - published| <= 2 x std_paper + 0.0005; "
                    "OPERATIONAL_PARTIAL when <= 3 x std_paper + 0.0005; OUTSIDE_OPERATIONAL_MARGIN otherwise. std_paper is THIS "
                    "dataset's Table 7 standard deviation of the four-horizon average, borrowed per horizon as a predeclared "
                    "operational margin - it is not a published per-horizon error bar and not statistical equivalence; our own seed "
                    "dispersion is reported beside it and never replaces the criterion",
            "std_paper": {"mse": std["mse"], "mae": std["mae"]}, "source": std["table"],
            "rounding": PAPER["reported_precision"], "k_agree": 2.0, "k_partial": 3.0, "protocol": protocol}


def paper_code_disagreements(dataset: str, seq_len: int, cells: list) -> list:
    """Every place the published table and the executable configuration differ, with how it was resolved. Written into the lock
    BEFORE any score can exist, because a disagreement resolved after a number is seen is not a resolution."""
    out = []
    if dataset == "weather":
        flags = {c["effective_args"]["is_training"] for c in cells}
        out.append({"id": "WEATHER-IS-TRAINING-2",
                    "disagreement": f"scripts/Weather.sh drives its L = 96 block with `--is_training {sorted(flags)[0]}`, while "
                                    "ECL.sh and Traffic.sh use `--is_training 1`",
                    "resolution": "NO EFFECT, resolved by reading run.py: `args.is_training` is read exactly once, at run.py:153, "
                                  "as the truth value of `if args.is_training:`. 2 and 1 select the identical train-then-test "
                                  "branch. The `is_training=` keyword inside the model/exp/layers is a separate local boolean for "
                                  "train vs eval mode and never receives this argument",
                    "verified_by": "grep of every `is_training` occurrence in the pinned clone; the argv value is carried into the "
                                   "cell unchanged rather than normalised, so the lock reproduces the author's own command"})
        out.append({"id": "WEATHER-TABLE6-SILENT-DEFAULTS",
                    "disagreement": "Table 6 states patch 48, e_layers 2, lr 5e-4, d_model 128, d_ff 256, epochs 10 for Weather and "
                                    "the script agrees on every one of them; the script additionally sets dropout 0.3, and leaves "
                                    "patience (3), lradj (cosine), n_heads (4), alpha (0.1), top_p (0.5), pos (1), use_norm (1), "
                                    "label_len (48), factor (3), d_layers (1) at run.py defaults. None of those is stated in the paper",
                    "resolution": "the EXECUTABLE configuration governs and is pinned verbatim from the author's script plus run.py's "
                                  "own parser defaults. The paper is recorded as the published reference, not as the configuration",
                    "verified_by": "effective_args of every cell are the author's parser applied to the author's argv"})
        out.append({"id": "WEATHER-FREQ-H-ON-A-TEN-MINUTE-SERIES",
                    "disagreement": "Weather is sampled every 600 s, yet the script passes no `--freq`, so the loader builds its "
                                    "calendar marks with run.py's default freq 'h'",
                    "resolution": "NO EFFECT on the model: `Exp_Long_Term_Forecast` calls `self.model(batch_x, self.masks, "
                                  "is_training=...)` and never passes `batch_x_mark`/`batch_y_mark` to TimeFilter, so the marks are "
                                  "built by the loader and discarded. It is kept at the author's value rather than 'corrected', "
                                  "because changing it would be a deviation from the pinned recipe for no numerical gain",
                    "verified_by": "exp/exp_long_term_forecasting.py lines 80, 128, 189: the model call takes (batch_x, masks, is_training)"})
        out.append({"id": "WEATHER-TABLE9-SEARCH-SPACE",
                    "disagreement": "Table 9 says the input length is searched in {192, 336, 512, 720}; scripts/Weather.sh offers "
                                    "exactly one long-horizon block, L = 720 (patch 144, d_ff 128, dropout 0.6, lr 1e-4, epochs 10)",
                    "resolution": "only L = 96 (Table 8) is sealed by THIS design. The L-searched protocol is recorded as available "
                                  "at L = 720 alone and is NOT claimed to reproduce a search the released code does not contain",
                    "verified_by": "script expansion: the two blocks of Weather.sh carry seq_len 96 and 720"})
        out.append({"id": "WEATHER-SEED",
                    "disagreement": "the paper reports a standard deviation over runs; run.py hard-codes fix_seed = 2021 and offers "
                                    "no CLI path to vary it",
                    "resolution": "the same three calls run.py makes (random.seed, torch.manual_seed, np.random.seed) are made with "
                                  "each of our seeds. This is declared as an operational patch with no other effect",
                    "verified_by": "df_sota_repro.fix_seeds and main_like_run_py, unchanged"})
    elif dataset == "traffic":
        out.append({"id": "TRAFFIC-TABLE6-SILENT-DEFAULTS",
                    "disagreement": "Table 6 states patch 96, e_layers 3, lr 1e-3, d_model 512, d_ff 2048, epochs 30 for Traffic and "
                                    "the script agrees on every one of them; the script additionally sets dropout 0.3, top_p 0.0 and "
                                    "pos 0, and leaves patience (3), lradj (cosine), n_heads (4), alpha (0.1), label_len (48), factor "
                                    "(3), d_layers (1) at run.py defaults. None of those is stated in the paper",
                    "resolution": "the EXECUTABLE configuration governs and is pinned verbatim from the author's script plus run.py's "
                                  "own parser defaults",
                    "verified_by": "effective_args of every cell are the author's parser applied to the author's argv"})
        out.append({"id": "TRAFFIC-LONG-HORIZON-BLOCK-SPLIT",
                    "disagreement": "Traffic.sh's long-horizon section is written as two blocks that differ only in dropout "
                                    "(0.5 for pred_len 96, 0.4 for 192/336/720) and both use L = 720, patch 720; Table 9 announces a "
                                    "search over {192, 336, 512, 720}",
                    "resolution": "only L = 96 (Table 8) is sealed by a Traffic L96 design; the L-searched protocol exists at L = 720 "
                                  "alone and no search is claimed",
                    "verified_by": "script expansion: the two long-horizon blocks both carry seq_len 720"})
        out.append({"id": "TRAFFIC-CLOCK",
                    "disagreement": "Traffic and Weather are compared side by side in the same published table with the same step "
                                    "counts 96/192/336/720",
                    "resolution": "Traffic steps are 3600 s and Weather steps are 600 s, so Traffic's 96-step horizon is 96 h and "
                                  "Weather's is 16 h. Both clocks are stored per receipt and the producer refuses a receipt whose "
                                  "elapsed horizon contradicts its own dataset's interval",
                    "verified_by": "tools/test_tsl_producer_contract.py::test_the_wrong_clock_is_refused"})
        out.append({"id": "TRAFFIC-SEED",
                    "disagreement": "the paper reports a standard deviation over runs; run.py hard-codes fix_seed = 2021",
                    "resolution": "the same three calls run.py makes, with each of our seeds; declared as an operational patch",
                    "verified_by": "df_sota_repro.fix_seeds and main_like_run_py, unchanged"})
    else:
        out.append({"id": f"{dataset.upper()}-NOT-PINNED",
                    "disagreement": "the published row for this dataset was not read for this delivery",
                    "resolution": "NO_SEAL: this dataset is PLANNED. No comparison, class or margin may be produced for it",
                    "verified_by": "PAPER[dataset] carries an explicit not-read note instead of numbers"})
    return out


def seal(*, dataset: str = "weather", seq_len: int = 96, seeds=(2021, 2022, 2023), horizons=(96, 192, 336, 720),
         protocol: str = "L96") -> dict:
    """The lock and the design for ONE dataset, sealed BEFORE any delivery: nothing in it depends on data or results."""
    facts = dataset_facts(dataset)
    spec = SCRIPTS[dataset]
    if protocol not in ("L96", "Lsearched"):
        raise TslRefusal("REFUSED: protocol must be L96 (Table 8) or Lsearched (Table 9)")
    if seq_len != spec[protocol]:
        raise TslRefusal(f"REFUSED: the author script offers L = {spec[protocol]} for {dataset} under {protocol}, not {seq_len}")
    git = S.author_git()
    if not git["pinned_matches"] or not git["clean"]:
        raise TslRefusal(f"REFUSED: the author clone is not at the pinned revision or not clean: {git}")
    parser, fix_seed = S.author_parser()
    script = [c for c in S.script_cells(script=spec["script"]) if c["seq_len"] == seq_len and c["pred_len"] in horizons]
    if sorted(c["pred_len"] for c in script) != sorted(horizons):
        raise TslRefusal(f"REFUSED: {spec['script']} has no L = {seq_len} invocation for every horizon {horizons}: "
                         f"{sorted(c['pred_len'] for c in script)}")
    cells = []
    for c in sorted(script, key=lambda c: c["pred_len"]):
        args = parser.parse_args(c["argv"])
        if args.data_path != spec["data_name"]:
            raise TslRefusal(f"REFUSED: {spec['script']} reads {args.data_path!r}, not the registered {spec['data_name']!r}")
        if int(args.enc_in) != int(facts["channels"]) or int(args.c_out) != int(facts["channels"]):
            raise TslRefusal(f"REFUSED: the script declares enc_in {args.enc_in} / c_out {args.c_out} against a registered "
                             f"{facts['channels']}-channel resource")
        effective = {k: v for k, v in sorted(vars(args).items())}
        for seed in seeds:
            cells.append({"cell_id": f"{dataset}_L{seq_len}_h{c['pred_len']}_s{seed}", "arm": "TimeFilter", "dataset": dataset,
                          "protocol": protocol, "seq_len": seq_len, "horizon_steps": c["pred_len"],
                          "horizon_seconds": horizon_seconds(dataset, c["pred_len"]), "seed": int(seed), "argv": c["argv"],
                          "effective_args": effective, "setting": S.setting_of(args),
                          "configuration_sha256": S.sha_obj({"argv": c["argv"], "effective": effective})})
    published = PAPER[dataset].get(protocol) if isinstance(PAPER[dataset].get(protocol), dict) else None
    lock = {
        "paper": PAPER["citation"], "paper_read_at": PAPER["read_at"], "dataset": dataset, "protocol": protocol,
        "published": published,
        "published_is_not_current_sota": "a published-recipe reference reproduces ONE named paper's own reported configuration. It is "
                                         "not a claim that this recipe is the current best result on this benchmark",
        "source": {"repository": "https://github.com/TROUBADOUR000/TimeFilter", "revision": PINNED_COMMIT, "git": git,
                   "script": spec["script"], "files_sha256": S.source_digests(),
                   "requirements_txt": (AUTHOR_REPO / "requirements.txt").read_text().split()},
        "official_input": {"lake": LAKE, "resource": facts["resource"], "sha256": facts["sha256"], "rows": facts["rows"],
                           "channels": facts["channels"], "step_seconds": facts["step_seconds"],
                           "provenance": "thuml/Time-Series-Library revision 2b66e59ee19dac8f6f19fb5d4997f289fdfea357, registered and "
                                         "byte-verified in the sota_benchmarks lake (docs/TSL_BENCHMARK_LAKE.md); this design does "
                                         "not download, re-register or re-adopt it",
                           "delivery_kind": "whole-resource AS_IS; availability UNDECLARED; every date range refused; NOT authorized "
                                            "as point-in-time or live financial data"},
        "targets": {"features": "M", "rule": "every channel is an input and a target; the loader moves the column named by --target "
                                             "(run.py default 'OT') to the last position and keeps the file's order otherwise",
                    "columns_pinned_by": "the characterize block, on the delivered bytes, records the exact ordered column list and "
                                         "the presence of the target column; a seal alone does not assert it"},
        "partitions": {"rule": "Dataset_Custom: num_train = int(0.7 n), num_test = int(0.2 n), num_vali = n - train - test; borders "
                               "[0, train), [train - L, train + vali), [n - test - L, n); scored windows per split = "
                               "(border2 - border1) - L - T + 1",
                       "per_horizon": {str(h): borders(facts["rows"], seq_len, h) for h in horizons},
                       "paper_table5_convention": "the paper's dataset-size column counts rows_of_segment - L + 1 and does not "
                                                  "subtract the horizon; both conventions are recorded so they are never confused"},
        "preprocessing": {"scaler": "sklearn StandardScaler fit on the TRAIN rows [0, int(0.7 n)) of every channel and applied to all "
                                    "rows (scale=True). Fitted per (dataset, L, T); never reused across datasets",
                          "inverse": "none: --inverse is False, so every metric below lives in the normalized target space",
                          "time_marks": "timeF features at run.py's default freq 'h'; built by the loader and never passed to the model",
                          "missing_values": "recorded by the characterize block from the delivered bytes; no imputation exists in the "
                                            "author path, so a missing value would be a refusal, not a policy"},
        "training": {"optimizer": "torch.optim.Adam(lr = learning_rate) with torch's default betas/eps/weight_decay",
                     "loss": "nn.MSELoss() on the normalized targets + 0.05 x the MoE routing loss (the coefficient is hard-coded in "
                             "the author's train())",
                     "lr_schedule": "utils.tools.adjust_learning_rate with lradj 'cosine': lr_e = lr/2 (1 + cos(e / train_epochs pi))",
                     "batch": "batch_size from the script; train loader shuffle=True, drop_last=False",
                     "epochs": "train_epochs from the script (run.py default 10 when the block does not set it)",
                     "early_stop": "utils.tools.EarlyStopping(patience from run.py's default 3, delta 0) on the VALIDATION MSE, which "
                                   "excludes the MoE term; the test loss the author prints each epoch is logging only",
                     "checkpoint": "the lowest-validation-loss epoch's state_dict (checkpoint.pth), reloaded before test()",
                     "amp": False},
        "evaluation": {"scorer": "utils.metrics.metric(preds, trues) on the concatenated float32 test predictions",
                       "metric_space": "the training-scaler normalized target space (no inverse transform)",
                       "reduction": "MSE = mean (yhat - y)^2 and MAE = mean |yhat - y| over EVERY test window x forecast step x target "
                                    "channel element - not an unweighted average of batch means",
                       "test_loader": "shuffle=False, drop_last=False, batch_size from the script",
                       "paired_naive": "persistence of the last observed value of each window, repeated over every step and channel, on "
                                       "EXACTLY the same windows the model is scored on",
                       "aggregation": "one mean per horizon; the four-horizon average is formed within a seed first, then across seeds"},
        "clock": {"step_seconds": facts["step_seconds"],
                  "horizons": {str(h): {"steps": h, "seconds": horizon_seconds(dataset, h),
                                        "elapsed": f"{horizon_seconds(dataset, h) / 3600:g} h"} for h in horizons},
                  "warning": "Weather's 96 steps are 16 h; Electricity's and Traffic's 96 steps are 96 h. A comparison that labels "
                             "the same step count as the same elapsed horizon is invalid"},
        "seeds": {"author": f"run.py fixes random/torch/numpy to {fix_seed} and offers no CLI path to vary it; the paper reports a "
                            "standard deviation over runs", "ours": [int(s) for s in seeds],
                  "how": "the same three calls with each seed"},
        "environment_author": {"torch": "2.3.1", "numpy": "1.26.4", "pandas": "2.2.3", "scikit_learn": "1.5.2",
                               "gpu": "NVIDIA A100 40GB (paper A.3)"},
        "environment_ours": S.environment(),
        "operational_patches": [
            {"what": "import shims for sktime.datasets and patoolib", "why": "author modules import them at module level for the "
             "UEA/M4 paths, which the long-term forecasting path never calls", "effect": "none: the shims refuse any call",
             "where": str(S.SHIMS)},
            {"what": "run.py's __main__ replicated by df_sota_repro.main_like_run_py", "why": "run.py hard-codes one seed",
             "effect": "none beyond the seed value"},
            {"what": "np.Inf restored as an alias of np.inf before the author modules import", "why": "NumPy 2 removed the alias that "
             "utils/tools.py EarlyStopping uses as its initial best", "effect": "none: the same float object"},
            {"what": "exp_long_term_forecasting.metric wrapped to capture the arrays the author scores", "why": "the author's test() "
             "saves no arrays", "effect": "none: the author's return value is passed through"}],
        "paper_code_disagreements": paper_code_disagreements(dataset, seq_len, cells),
        "agreement": agreement(dataset, protocol),
        "comparability": {"class": "MATCHED_PUBLISHED_RECIPE_PENDING_EXECUTION",
                          "reading": "the class becomes a comparison only when the cells have run under this exact digest. Until then "
                                     "no published value may be placed beside a measured one",
                          "never": "a missing published comparability is never cured by rescaling an incompatible task"},
        "receipt_contract": {"schema": contract()["schema"], "sha256": S.sha_obj(contract())},
    }
    design = {"schema": SCHEMA, "task": f"{dataset}_official_tsl", "dataset": dataset,
              "purpose": f"RB02 matched reference reproduction of TimeFilter on the official processed {dataset} benchmark",
              "sealed_at": S.now_iso(), "seq_len": seq_len, "seeds": [int(s) for s in seeds], "horizons": list(horizons),
              "protocol": protocol, "cells": cells, "lock": lock,
              "source_data": {"lake": LAKE, "resource": facts["resource"], "sha256": facts["sha256"]}}
    design["design_sha256"] = S.sha_obj({k: v for k, v in design.items() if k != "design_sha256"})
    design["lock"]["protocol_sha256"] = S.sha_obj(design["lock"])
    return design


def validate_design(design: dict) -> None:
    body = {k: v for k, v in design.items() if k != "design_sha256"}
    lock = {k: v for k, v in body["lock"].items() if k != "protocol_sha256"}
    body = {**body, "lock": lock}
    if design.get("schema") != SCHEMA or design.get("design_sha256") != S.sha_obj(body):
        raise TslRefusal("REFUSED: the design digest does not recompute from its content (relabeled or edited design)")


def protocol_sha256(design: dict) -> str:
    return design["lock"]["protocol_sha256"]


# --- the delivered bytes: the author's own loader is the only reader ------------------------------------------------------

def characterize(design: dict, data_path: Path) -> dict:
    """The author's `Dataset_Custom` on the delivered bytes: ordered columns, the target column, missing values, borders,
    window counts and the training scaler's identity per (L, T). Nothing here is a model result."""
    import pandas as pd
    validate_design(design)
    facts = dataset_facts(design["dataset"])
    got = S.sha_file(data_path)
    if got != facts["sha256"]:
        raise TslRefusal(f"REFUSED: these are not the registered bytes of {facts['resource']} ({got[:12]} != {facts['sha256'][:12]})")
    S.author_env()
    import importlib
    from types import SimpleNamespace
    DL = importlib.import_module("data_provider.data_loader")
    raw = pd.read_csv(data_path)
    cols = list(raw.columns)
    target = "OT"
    if target not in cols:
        raise TslRefusal(f"REFUSED: the author loader removes and re-appends the column named by --target {target!r}; the delivered "
                         f"resource has no such column (last columns: {cols[-3:]})")
    if cols[0] != "date":
        raise TslRefusal(f"REFUSED: the author loader expects the first column to be 'date', found {cols[0]!r}")
    missing = int(raw.isna().sum().sum())
    reordered = ["date"] + [c for c in cols if c not in ("date", target)] + [target]
    out = {"schema": "df_tsl_characterization.v1", "design_sha256": design["design_sha256"],
           "protocol_sha256": protocol_sha256(design), "dataset": design["dataset"], "resource": facts["resource"],
           "file_sha256": got, "bytes": data_path.stat().st_size,
           "columns": {"count_including_date": len(cols), "file_order": cols, "loader_order": reordered,
                       "target_argument": target, "target_moved_last": reordered[-1] == target,
                       "reordering_is_a_noop": cols == reordered},
           "rows": int(len(raw)), "rows_match_registered": int(len(raw)) == int(facts["rows"]),
           "missing_values": missing, "first_label": str(raw.iloc[0, 0]), "last_label": str(raw.iloc[-1, 0]),
           "step_seconds_registered": facts["step_seconds"], "sets": {}}
    if missing:
        raise TslRefusal(f"REFUSED: the delivered resource has {missing} missing values and the author path has no imputation; a "
                         "missing-value policy would be a deviation, not a default")
    if not out["rows_match_registered"]:
        raise TslRefusal(f"REFUSED: {out['rows']} rows read against {facts['rows']} registered")
    ns = SimpleNamespace(augmentation_ratio=0)
    arrays = {}
    for h in design["horizons"]:
        key = f"L{design['seq_len']}_h{h}"
        sets, expected = {}, borders(facts["rows"], design["seq_len"], h)
        for flag in ("train", "val", "test"):
            ds = DL.Dataset_Custom(ns, str(data_path.parent), flag=flag, size=[design["seq_len"], 48, h], features="M",
                                   data_path=data_path.name, target=target, scale=True, timeenc=1, freq="h")
            name = {"train": "train", "val": "vali", "test": "test"}[flag]
            sets[name] = {"windows": len(ds), "rows": int(ds.data_x.shape[0]), "channels": int(ds.data_x.shape[1]),
                          "windows_expected_from_registered_rows": expected["windows"][name],
                          "agrees": len(ds) == expected["windows"][name]}
            if not sets[name]["agrees"]:
                raise TslRefusal(f"REFUSED: {key}/{name}: the loader yields {len(ds)} windows, the sealed arithmetic "
                                 f"{expected['windows'][name]}")
            if flag == "train":
                mean = np.asarray(ds.scaler.mean_, dtype=np.float64)
                scale = np.asarray(ds.scaler.scale_, dtype=np.float64)
                arrays[f"{key}_scaler_mean"], arrays[f"{key}_scaler_scale"] = mean, scale
                sets["scaler_sha256"] = S.sha_array(np.concatenate([mean, scale]))
                sets["scaler_fit_population_sha256"] = S.sha_obj(
                    {"resource": facts["resource"], "file_sha256": got, "rows": [0, expected["rows"]["train"]],
                     "channels": int(ds.data_x.shape[1]), "estimator": "sklearn.preprocessing.StandardScaler",
                     "fit_on": "train rows only"})
        sets["elements_test"] = int(sets["test"]["windows"]) * int(h) * int(sets["test"]["channels"])
        sets["horizon_steps"], sets["horizon_seconds"] = int(h), horizon_seconds(design["dataset"], h)
        out["sets"][key] = sets
    out["scaler_identities_differ_per_horizon"] = len({v["scaler_sha256"] for v in out["sets"].values()}) > 1
    return out, arrays


# --- the producer path: the receipt and the gate that refuses an unidentified one -----------------------------------------

def build_receipt(*, design: dict, characterization: dict, horizon: int, seed: int, metrics: dict, naive: dict,
                  population: dict, model_commit: str, scorer_sha256: str, campaign_key: str, campaign_sha256: str,
                  unit_id: str, delivery_id: str, availability_contract_sha256: str, costs: dict, started_at: str,
                  finished_at: str, comparison_class: str, metric_dtype: str = "float32",
                  classification: str = "NON_GOVERNING") -> dict:
    """One `tsl_literature_metrics.v1` governed terminal for one scored cell. Every identity comes from an artifact."""
    validate_design(design)
    C = contract()
    facts = dataset_facts(design["dataset"])
    key = f"L{design['seq_len']}_h{horizon}"
    sets = characterization["sets"][key]
    cell = next(c for c in design["cells"] if c["horizon_steps"] == horizon and c["seed"] == seed)
    tags = {
        "metric_contract": C["schema"], "dataset": design["dataset"], "resource": facts["resource"],
        "dataset_sha256": facts["sha256"], "model": cell["arm"], "model_commit": model_commit,
        "configuration_sha256": cell["configuration_sha256"], "protocol_sha256": protocol_sha256(design),
        "reference_source": design["lock"]["paper"], "reference_table": (design["lock"]["published"] or {}).get("table", ""),
        "comparison_class": comparison_class, "input_window_steps": str(design["seq_len"]),
        "horizon_steps": str(horizon), "horizon_seconds": str(horizon_seconds(design["dataset"], horizon)),
        "seed": str(seed), "scaler_sha256": sets["scaler_sha256"],
        "scaler_fit_population_sha256": sets["scaler_fit_population_sha256"],
        "evaluation_population_sha256": population["sha256"], "windows": str(population["windows"]),
        "target_channels": str(population["target_channels"]), "elements": str(population["elements"]),
        "metric_scale": "z_train", "metric_reduction": C["reduction"], "metric_dtype": metric_dtype,
        "scorer_sha256": scorer_sha256,
    }
    rows = [("sota.test.mse_normalized", metrics["mse"], "z^2"), ("sota.test.mae_normalized", metrics["mae"], "z"),
            ("sota.test.naive_mse_normalized", naive["mse"], "z^2"), ("sota.test.naive_mae_normalized", naive["mae"], "z")]
    body = {"schema": "governed_terminal.v1", "campaign_sha256": campaign_sha256, "campaign_key": campaign_key,
            "unit_id": unit_id, "generation": 1, "actor": "satoshi", "project": "predictor",
            "classification": classification, "status": "COMPLETED", "reason": None, "started_at": started_at,
            "finished_at": finished_at, "terminal_lake": "olap_cube", "config_sha256": design["design_sha256"],
            "code_identity": {"kind": "git_commit", "value": model_commit}, "costs": costs, "tags": tags,
            "synthetic_spec_sha256": None, "deliveries": [delivery_id],
            "metrics": [{"metric": m, "value": float(v), "split": "test", "horizon": int(horizon), "unit": u,
                         "std_dev": None, "min_value": None, "max_value": None} for m, v, u in rows],
            "verified_datasets": [{"delivery_id": delivery_id, "lake_id": LAKE, "resource_id": facts["resource"],
                                   "role": ROLE, "sha256": facts["sha256"], "bytes": characterization["bytes"],
                                   "source_sha256": None, "range_from": None, "range_to": None, "delivery_kind": "AS_IS",
                                   "time_column": "date", "availability_contract_sha256": availability_contract_sha256,
                                   "state": "VERIFIED_TRANSFER"}],
            "artifacts": []}
    body["terminal_sha256"] = S.sha_obj(body)
    return body


def validate_receipt(body: dict) -> dict:
    """THE PRODUCER GATE. The deployed warehouse accepts the governed metric schema and rejects a nonfinite value; it does not
    and should not decide science. Everything below is refused HERE, before anything is sent anywhere.

    Returns the checks that passed; raises TslRefusal naming every failure."""
    C = contract()
    bad = []
    tags = body.get("tags") or {}
    if tags.get("metric_contract") != C["schema"]:
        bad.append(f"CONTRACT: tags.metric_contract is {tags.get('metric_contract')!r}, not {C['schema']!r}")
    for field in C["required_context_tags"]:
        value = tags.get(field)
        if value is None:
            bad.append(f"MISSING_TAG: {field}")
        elif str(value).strip() in PLACEHOLDERS:
            bad.append(f"PLACEHOLDER_TAG: {field} = {value!r} is not an identity")
    for field in ("protocol_sha256", "configuration_sha256", "dataset_sha256", "scaler_sha256",
                  "scaler_fit_population_sha256", "evaluation_population_sha256", "scorer_sha256"):
        if field in tags and not is_digest(tags.get(field)):
            bad.append(f"NOT_A_DIGEST: {field} = {str(tags.get(field))[:24]!r} is not a 64-hex digest")
    dataset = tags.get("dataset")
    if dataset in C["datasets"]:
        step = int(C["datasets"][dataset]["step_seconds"])
        try:
            steps, seconds = int(tags["horizon_steps"]), int(tags["horizon_seconds"])
        except (KeyError, TypeError, ValueError):
            steps = seconds = None
        if steps is not None and seconds != steps * step:
            bad.append(f"CLOCK: {dataset} steps are {step} s, so {steps} steps are {steps * step} s, not {seconds} s")
        if tags.get("resource") != C["datasets"][dataset]["resource"]:
            bad.append(f"RESOURCE: {tags.get('resource')!r} is not the registered resource of {dataset}")
        if tags.get("dataset_sha256") != C["datasets"][dataset]["sha256"]:
            bad.append(f"DATASET_BYTES: dataset_sha256 is not the registered digest of {dataset}")
    elif dataset is not None:
        bad.append(f"DATASET: {dataset!r} is not a registered resource of the {LAKE} lake")
    try:
        windows, channels, steps_i, elements = (int(tags["windows"]), int(tags["target_channels"]),
                                                int(tags["horizon_steps"]), int(tags["elements"]))
        if elements != windows * steps_i * channels:
            bad.append(f"POPULATION: elements {elements} != windows {windows} x steps {steps_i} x channels {channels}")
    except (KeyError, TypeError, ValueError):
        bad.append("POPULATION: windows, target_channels, horizon_steps and elements must all be integers")
    if tags.get("metric_reduction") != C["reduction"]:
        bad.append("REDUCTION: metric_reduction is not the contract's element-wise mean")
    if tags.get("metric_scale") != "z_train":
        bad.append(f"SCALE: metric_scale {tags.get('metric_scale')!r} is not the normalized target space these names carry")
    units = {**{k: v["unit"] for k, v in C["primary_metrics"].items()}, **{k: v["unit"] for k, v in C["paired_baselines"].items()}}
    seen = set()
    rows = body.get("metrics") or []
    if not rows:
        bad.append("METRICS: a receipt with no metric row is not a result")
    for m in rows:
        name, value = m.get("metric"), m.get("value")
        seen.add(name)
        if name in units and m.get("unit") != units[name]:
            bad.append(f"UNIT: {name} must carry {units[name]!r}, not {m.get('unit')!r}")
        try:
            v = float(value)
        except (TypeError, ValueError):
            bad.append(f"NONFINITE: {name} value {value!r} is not a number")
            continue
        if not np.isfinite(v):
            bad.append(f"NONFINITE: {name} = {value!r}")
        if m.get("split") != C["split"]:
            bad.append(f"SPLIT: {name} is reported on {m.get('split')!r}, not {C['split']!r}")
        if str(m.get("horizon")) != str(tags.get("horizon_steps")):
            bad.append(f"HORIZON: {name} carries horizon {m.get('horizon')} against tags.horizon_steps {tags.get('horizon_steps')}")
    for required in (set(C["primary_metrics"]) | set(C["paired_baselines"])):
        if required not in seen:
            bad.append(f"MISSING_METRIC: {required} — the model's error and the paired naive on the same rows are one receipt")
    if body.get("terminal_sha256") and body["terminal_sha256"] != S.sha_obj({k: v for k, v in body.items() if k != "terminal_sha256"}):
        bad.append("TERMINAL_DIGEST: terminal_sha256 does not recompute from the body")
    if bad:
        raise TslRefusal("REFUSED by the producer contract:\n  - " + "\n  - ".join(bad))
    return {"contract": C["schema"], "dataset": dataset, "horizon_steps": tags["horizon_steps"],
            "horizon_seconds": tags["horizon_seconds"], "protocol_sha256": tags["protocol_sha256"],
            "scaler_sha256": tags["scaler_sha256"], "evaluation_population_sha256": tags["evaluation_population_sha256"],
            "metric_rows": len(rows), "checks": ["contract", "required_tags", "digests", "clock", "resource_bytes",
                                                 "population_arithmetic", "reduction", "scale", "units", "finite",
                                                 "split", "horizon", "paired_naive_present", "terminal_digest"]}


# --- the TRAIN-only cost pilot -------------------------------------------------------------------------------------------

def train_only_pilot(design: dict, *, data_path: Path, work: Path, horizon: int, seed: int | None = None, steps: int = 30,
                     gpu: int = 0, require_gpu_uuid: str | None = None, dataloader_workers: int | None = 0) -> dict:
    """K optimizer steps of the AUTHOR's own training loop, timed. It builds the train loader only, reads no validation batch,
    evaluates no test window and returns NO metric. Its output is a cost, and a cost is not a result."""
    validate_design(design)
    cell = next(c for c in design["cells"] if c["horizon_steps"] == horizon and (seed is None or c["seed"] == seed))
    facts = dataset_facts(design["dataset"])
    if S.sha_file(data_path) != facts["sha256"]:
        raise TslRefusal("REFUSED: the pilot would read bytes that are not the registered resource")
    work.mkdir(parents=True, exist_ok=True)
    S.author_env()
    import torch
    device_check = S.assert_child_device(require_gpu_uuid, gpu)
    S.fix_seeds(cell["seed"])
    args = S.build_args(cell["argv"], data_dir=data_path.parent, data_name=data_path.name, checkpoints=work / "checkpoints",
                        gpu=gpu, dataloader_workers=dataloader_workers)
    import importlib
    Exp = importlib.import_module("exp.exp_long_term_forecasting").Exp_Long_Term_Forecast
    t0 = time.time()
    exp = Exp(args)
    built = time.time() - t0
    n_params = int(sum(p.numel() for p in exp.model.parameters()))
    t0 = time.time()
    train_data, train_loader = exp._get_data(flag="train")
    loader_seconds = time.time() - t0
    optim, criterion = exp._select_optimizer(), exp._select_criterion()
    exp.model.train()
    per_step, done = [], 0
    ru0, wall0, cpu0 = resource.getrusage(resource.RUSAGE_SELF), time.time(), time.process_time()
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        t = time.time()
        optim.zero_grad()
        batch_x = batch_x.float().to(exp.device)
        outputs, moe_loss = exp.model(batch_x, exp.masks, is_training=True)
        f_dim = -1 if args.features == "MS" else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].float().to(exp.device)
        loss = criterion(outputs, batch_y) + 0.05 * moe_loss
        loss.backward()
        optim.step()
        if str(exp.device).startswith("cuda"):
            torch.cuda.synchronize()
        per_step.append(time.time() - t)
        done += 1
        if done >= steps:
            break
    wall, cpu = time.time() - wall0, time.process_time() - cpu0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    batches = len(train_loader)
    med = float(np.median(per_step[1:] or per_step))
    out = {"schema": "df_tsl_train_pilot.v1", "kind": "TRAIN_ONLY_COST_PILOT",
           "reading": "an optimizer-step cost and a memory footprint. NO validation loss, NO test score, NO metric, NO model "
                      "decision. It does not shorten, shrink or alter the sealed recipe in any way",
           "design_sha256": design["design_sha256"], "protocol_sha256": protocol_sha256(design), "dataset": design["dataset"],
           "cell_id": cell["cell_id"], "horizon_steps": horizon, "horizon_seconds": horizon_seconds(design["dataset"], horizon),
           "seed": cell["seed"], "seq_len": design["seq_len"], "batch_size": int(args.batch_size),
           "train_epochs_sealed": int(args.train_epochs), "patience_sealed": int(args.patience),
           "n_parameters": n_params, "steps_timed": done, "train_batches_per_epoch": batches,
           "train_windows": len(train_data),
           "seconds_per_train_step_median": med, "seconds_per_train_step_mean": float(np.mean(per_step)),
           "seconds_per_train_step_p90": float(np.percentile(per_step, 90)),
           "model_build_seconds": built, "train_loader_build_seconds": loader_seconds,
           "train_epoch_seconds_projected": med * batches,
           "wall_seconds": wall, "cpu_seconds": cpu,
           "maxrss_bytes_self": ru1.ru_maxrss * 1024, "maxrss_bytes_children": 0,
           "user_cpu_seconds": ru1.ru_utime - ru0.ru_utime, "system_cpu_seconds": ru1.ru_stime - ru0.ru_stime,
           "cgroup": S._cgroup_memory(), "device": str(exp.device), "device_uuid_measured_inside_child": S.actual_device_uuid(gpu),
           "device_assertion": device_check,
           "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()) if str(exp.device).startswith("cuda") else 0,
           "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()) if str(exp.device).startswith("cuda") else 0,
           "gpu_state_after": S.gpu_state(), "environment": S.environment(), "host": socket.gethostname(),
           "boot_id": (Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                       if Path("/proc/sys/kernel/random/boot_id").is_file() else None),
           "pid": os.getpid(), "started_at": S.now_iso(),
           "reservation": {k: os.environ.get(k) for k in ("CRISPDM_RESERVATION_ID", "CRISPDM_RESERVATION_BYTES",
                                                          "CRISPDM_JOB_NAME", "CRISPDM_SLICE", "INVOCATION_ID")},
           "file_sha256": facts["sha256"]}
    out["record_sha256"] = S.sha_obj(out)
    return out


def full_reproduction_cost(design: dict, pilot: dict, *, epoch_overhead_factor: float = 1.0) -> dict:
    """The frozen full-reproduction design's measured cost: every sealed cell, priced from the pilot's MEASURED per-step cost
    and the sealed batch counts. It is an upper bound: early stopping can only shorten it."""
    validate_design(design)
    if pilot["design_sha256"] != design["design_sha256"]:
        raise TslRefusal("REFUSED: this pilot record was not measured under this design")
    facts = dataset_facts(design["dataset"])
    step_s = float(pilot["seconds_per_train_step_median"])
    cells = []
    for c in design["cells"]:
        b = borders(facts["rows"], design["seq_len"], c["horizon_steps"])
        bs = int(c["effective_args"]["batch_size"])
        train_b = -(-b["windows"]["train"] // bs)
        vali_b, test_b = -(-b["windows"]["vali"] // bs), -(-b["windows"]["test"] // bs)
        epochs = int(c["effective_args"]["train_epochs"])
        # the author's train() runs one train pass plus a validation pass plus a logging test pass per epoch; the pilot measures
        # only the train pass, so the evaluation passes are priced from the pilot's own step cost as forward-only work
        eval_s = step_s / 3.0 * (vali_b + test_b)
        per_epoch = step_s * train_b + eval_s
        cells.append({"cell_id": c["cell_id"], "horizon_steps": c["horizon_steps"], "seed": c["seed"],
                      "train_batches": train_b, "vali_batches": vali_b, "test_batches": test_b, "epochs_max": epochs,
                      "epoch_seconds_projected": per_epoch * epoch_overhead_factor,
                      "cell_seconds_max": per_epoch * epochs * epoch_overhead_factor})
    total = sum(c["cell_seconds_max"] for c in cells)
    out = {"schema": "df_tsl_full_cost.v1", "design_sha256": design["design_sha256"],
           "protocol_sha256": protocol_sha256(design), "dataset": design["dataset"],
           "measured_from": {"pilot_record_sha256": pilot["record_sha256"], "host": pilot["host"],
                             "device_uuid": pilot["device_uuid_measured_inside_child"],
                             "seconds_per_train_step_median": step_s, "steps_timed": pilot["steps_timed"],
                             "cgroup_peak_bytes": (pilot.get("cgroup") or {}).get("memory.peak"),
                             "peak_gpu_allocated_bytes": pilot["peak_gpu_allocated_bytes"]},
           "cells": cells, "cells_total": len(cells), "seconds_max": total, "hours_max": total / 3600.0,
           "evaluation_pass_pricing": "the validation and logging-test forward passes are priced at one third of a measured "
                                      "train step per batch; the pilot measured the train pass only, so this part of the total is "
                                      "DERIVED, not measured, and is named as such",
           "reading": "upper bound: every cell runs its full train_epochs. EarlyStopping (patience 3 on the validation MSE) can "
                      "only shorten it. The host-memory ceiling of the author's evaluation path is a separate admission question "
                      "and is not answered by this number"}
    out["cost_sha256"] = S.sha_obj(out)
    return out


# --- CLI -----------------------------------------------------------------------------------------------------------------

def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["seal", "characterize", "pilot", "plan", "validate-receipt"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dataset", default="weather")
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--protocol", default="L96")
    ap.add_argument("--seeds", type=int, nargs="*", default=[2021, 2022, 2023])
    ap.add_argument("--horizons", type=int, nargs="*", default=[96, 192, 336, 720])
    ap.add_argument("--data-path", type=Path, default=None)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dataloader-workers", type=int, default=0)
    ap.add_argument("--require-gpu-uuid", default=os.environ.get(S.REQUIRED_GPU_ENV) or None)
    ap.add_argument("--receipt", type=Path, default=None)
    ap.add_argument("--pilot", type=Path, default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    design_path = root / f"DESIGN.{a.dataset}.{a.protocol}.json"

    if a.command == "seal":
        design = seal(dataset=a.dataset, seq_len=a.seq_len, seeds=tuple(a.seeds), horizons=tuple(a.horizons), protocol=a.protocol)
        print(json.dumps({"design_sha256": design["design_sha256"], "protocol_sha256": protocol_sha256(design),
                          "cells": len(design["cells"]), "path": str(_write(design_path, design))}, indent=1))
        return 0
    if a.command == "validate-receipt":
        print(json.dumps(validate_receipt(json.loads(Path(a.receipt).read_text())), indent=1))
        return 0
    design = json.loads(design_path.read_text())
    if a.command == "characterize":
        record, arrays = characterize(design, Path(a.data_path))
        np.savez(root / f"SCALERS.{a.dataset}.npz", **arrays)
        record["scaler_archive_sha256"] = S.sha_file(root / f"SCALERS.{a.dataset}.npz")
        print(json.dumps({"path": str(_write(root / f"CHARACTERIZATION.{a.dataset}.json", record)),
                          "missing_values": record["missing_values"], "rows": record["rows"],
                          "sets": {k: v["test"]["windows"] for k, v in record["sets"].items()}}, indent=1))
        return 0
    if a.command == "pilot":
        rec = train_only_pilot(design, data_path=Path(a.data_path), work=root / "work", horizon=a.horizon, seed=a.seed,
                               steps=a.steps, gpu=a.gpu, require_gpu_uuid=a.require_gpu_uuid,
                               dataloader_workers=a.dataloader_workers)
        print(json.dumps({"path": str(_write(root / f"TRAIN_PILOT.{a.dataset}.h{a.horizon}.json", rec)),
                          "seconds_per_train_step_median": rec["seconds_per_train_step_median"],
                          "device": rec["device"], "device_uuid": rec["device_uuid_measured_inside_child"],
                          "cgroup_peak": (rec.get("cgroup") or {}).get("memory.peak"),
                          "peak_gpu_allocated_bytes": rec["peak_gpu_allocated_bytes"]}, indent=1))
        return 0
    if a.command == "plan":
        cost = full_reproduction_cost(design, json.loads(Path(a.pilot).read_text()))
        print(json.dumps({"path": str(_write(root / f"FULL_COST.{a.dataset}.{a.protocol}.json", cost)),
                          "cells": cost["cells_total"], "hours_max": cost["hours_max"]}, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
