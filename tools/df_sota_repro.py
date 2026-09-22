#!/usr/bin/env python3
"""RP92-RP96 (SOTA-first): faithful, governed reproduction of the selected reference — TimeFilter (Hu et al., ICML 2025) on the
OFFICIAL processed ECL benchmark (Time-Series-Library electricity.csv, 26 304 hourly rows x 321 clients) — with the AUTHOR's
model, loader, loss, optimizer, early stopping and scorer, executed from the pinned clone; this tool only transports, seals,
accounts and verifies. Nothing here re-implements a layer, a loss, a split or a metric.

Blocks
  seal      the source-and-protocol LOCK (author revision + file digests, the ECL.sh cells with EVERY effective default read
            from run.py's own parser, dataset provenance and digest, partitions, training/validation/checkpoint rules, metric
            reductions, seeds, environment, operational patches, the numerical-agreement criterion) and the sealed design
            (cells = horizons x seeds), BEFORE any data is read
  prepare   governed delivery of the official file (registered before reading), the author's Dataset_Custom borders and
            scaler (fit on the train rows only) recorded as the preparation's own evidence, prepare terminal with digests
  preflight bounded: K optimizer steps and a few validation forwards of the author's loop on the real train/vali loaders,
            timing, peak memory and parameters -> a full allocation for every cell; no test score is produced
  child     one cell: seeds set as run.py sets them, args parsed by run.py's parser from the script's argv, the author's
            Exp_Long_Term_Forecast.train() then .test() (metric captured by wrapping the author's `metric` symbol),
            predictions + checkpoint + record as artifacts of its own governed unit
  close     RP96: every cell's arrays and checkpoint bytes bound to the accepted chain, targets re-derived from the delivered
            file by the author's loader, metrics recomputed with the author's function and independently in float64, naive
            on identical windows, fresh-process checkpoint reload through the author's test(), the frozen agreement criterion,
            and the RP97 table

Operational patches (declared, no mathematical effect — verified by tests):
  * import shims for `sktime.datasets` and `patoolib` (author modules import them for the UEA/M4 paths, never called here;
    the shims REFUSE any call);
  * run.py's `__main__` body is replicated by `main_like_run_py` (seed fixing, parser, post-parse logic, Exp, train, test):
    run.py hard-codes fix_seed 2021 and has no path to vary it; our seeds use the same three calls;
  * `exp_long_term_forecasting.metric` is wrapped to CAPTURE the arrays the author scores; the author's value is returned.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import platform
import resource
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
HOME = Path.home()
SCHEMA = "df_sota_repro_design.v1"
AUTHOR_REPO = HOME / ".local/state/crispdm-data-foundation/sota_sources/TimeFilter"
PINNED_COMMIT = "dffde87e4fff0fdeeebbacde03dc1e432e15b3a1"
SHIMS = HERE / "sota_shims"
LAKE, RESOURCE, ROLE = "sota_benchmarks", "thuml_tsl_electricity/electricity.csv", "benchmark"
FILE_SHA256 = "7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad"
STORE_RECEIPT = HOME / ".local/state/crispdm-data-foundation/sota_benchmarks_v1/BUILD_RECEIPT.json"
SOURCE_FILES = ("run.py", "exp/exp_basic.py", "exp/exp_long_term_forecasting.py", "models/TimeFilter.py", "layers/TimeFilter_layers.py",
                "layers/Embed.py", "layers/StandardNorm.py", "data_provider/data_factory.py", "data_provider/data_loader.py",
                "utils/tools.py", "utils/metrics.py", "utils/timefeatures.py", "scripts/ECL.sh", "requirements.txt")
#: TimeFilter, arXiv:2501.13041 (ICML 2025), Table 8 (L = 96, full results) and Table 9 (L searched in {192,336,512,720});
#: Table 7: three runs, std of the four-horizon average 0.005 (MSE) / 0.006 (MAE); values are printed to three decimals
PAPER = {
    "citation": "Hu, Y. et al. TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting. ICML 2025 "
                "(PMLR v267); arXiv:2501.13041. Author code github.com/TROUBADOUR000/TimeFilter",
    "L96": {"table": "Table 8 (L = 96)", "per_horizon": {"96": {"mse": 0.133, "mae": 0.230}, "192": {"mse": 0.154, "mae": 0.248},
                                                        "336": {"mse": 0.162, "mae": 0.261}, "720": {"mse": 0.184, "mae": 0.284}},
            "average": {"mse": 0.158, "mae": 0.256}, "std_of_average_three_runs": {"mse": 0.005, "mae": 0.006}},
    "Lsearched": {"table": "Table 9 (L searched in {192, 336, 512, 720}; the script offers L = 512 only)",
                  "per_horizon": {"96": {"mse": 0.126, "mae": 0.220}, "192": {"mse": 0.143, "mae": 0.237},
                                  "336": {"mse": 0.153, "mae": 0.252}, "720": {"mse": 0.177, "mae": 0.275}},
                  "average": {"mse": 0.150, "mae": 0.246}},
    "reported_precision": 0.0005, "runs_reported": 3,
}
#: frozen BEFORE any test score is read (RP92/RP96): reported precision + the paper's own run-to-run dispersion
AGREEMENT = {
    "rule": "per horizon and for the four-horizon average, the three-seed mean of the replicated metric agrees with the published "
            "value when |mean - published| <= 2 x std_paper + 0.0005 (rounding half-unit); PARTIAL when <= 3 x std_paper + 0.0005; "
            "DISAGREEMENT otherwise. std_paper is the paper's std over its three runs of the four-horizon average (Table 7), applied "
            "per horizon as the only dispersion the paper reports; the replicated seed dispersion is reported beside it and never "
            "replaces the criterion",
    "std_paper": {"mse": 0.005, "mae": 0.006}, "rounding": 0.0005, "k_agree": 2.0, "k_partial": 3.0,
    "replay": {"device": "cpu", "atol": 1e-4, "rtol": 1e-4,
               "reading": "a GPU-trained float32 checkpoint reloaded in a fresh process through the author's test() on CPU: kernel-level "
                          "differences are expected around 1e-6..1e-5 in normalized units; the metric recomputed from the replayed "
                          "predictions must be within 1e-5 of the stored one; both maxima are always reported"},
    "metric_recompute": {"author_float32": "bitwise equal to the record (same arrays, same function)", "independent_float64": "|delta| <= 1e-6"},
}


class SotaRefusal(SystemExit):
    """A reproduction that cannot be shown faithful does not report a number."""


# --- small helpers --------------------------------------------------------------------------------------------------------

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def sha_array(a: np.ndarray) -> str:
    """Digest of the array bytes, streamed in chunks along the first axis (a 4 GB array never gets a second copy)."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    step = max(1, (64 << 20) // max(1, a[0:1].nbytes)) if a.ndim else 1
    for i in range(0, a.shape[0] if a.ndim else 1, step):
        h.update(memoryview(a[i:i + step]).cast("B"))
    return h.hexdigest()


def float64_metrics(preds: np.ndarray, trues: np.ndarray, chunk: int = 256) -> dict:
    """MAE and MSE accumulated in float64 over chunks of windows: an independent reduction with no full-size temporaries."""
    n, ab, sq = 0, 0.0, 0.0
    for i in range(0, preds.shape[0], chunk):
        d = preds[i:i + chunk].astype(np.float64) - trues[i:i + chunk].astype(np.float64)
        ab += float(np.abs(d).sum()); sq += float((d * d).sum()); n += d.size
    return {"mae": ab / n, "mse": sq / n}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def author_git() -> dict:
    def run(*argv):
        return subprocess.run(["git", "-C", str(AUTHOR_REPO), *argv], capture_output=True, text=True, timeout=60)
    head = run("rev-parse", "HEAD").stdout.strip()
    # byte-code caches are not source: a clone is clean when every SOURCE entry is unchanged and untracked
    dirty = "\n".join(l for l in run("status", "--porcelain").stdout.strip().splitlines() if "__pycache__" not in l and not l.endswith(".pyc"))
    return {"path": str(AUTHOR_REPO), "head": head, "pinned": PINNED_COMMIT, "pinned_matches": head == PINNED_COMMIT,
            "clean": not dirty, "dirty_entries": dirty.splitlines()[:10]}


def source_digests(repo: Path = AUTHOR_REPO) -> dict:
    return {f: sha_file(repo / f) for f in SOURCE_FILES if (repo / f).is_file()}


# --- the author's own parser and script (read, never re-typed) --------------------------------------------------------------

def author_parser(repo: Path = AUTHOR_REPO) -> tuple:
    """run.py's argparse, extracted by AST from its `__main__` block: every `parser.add_argument(...)` statement is executed
    verbatim on a fresh parser, so the EFFECTIVE defaults are the author's file, not a transcription. Returns (parser, fix_seed)."""
    tree = ast.parse((repo / "run.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and getattr(n.test.left, "id", None) == "__name__")
    fix_seed = None
    parser = argparse.ArgumentParser(description="PatchTST")
    ns = {"parser": parser, "argparse": argparse}
    for node in main.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "fix_seed" for t in node.targets):
            fix_seed = ast.literal_eval(node.value)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, "attr", None) == "add_argument":
            exec(compile(ast.Expression(node.value), "run.py", "eval"), ns)
    return parser, fix_seed


def script_cells(repo: Path = AUTHOR_REPO, script: str = "scripts/ECL.sh") -> list:
    """The `python -u run.py ...` invocations of the author's ECL script, loops expanded, variables substituted: one argv each."""
    text = (repo / script).read_text()
    env, loop, block, out = {}, None, None, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("export "):
            continue
        if "=" in line and not line.startswith("--") and not line.startswith("python") and block is None and not line.startswith("for "):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
            continue
        if line.startswith("for ") and " in " in line:
            var, values = line[4:].split(" in ", 1)
            loop = (var.strip(), values.strip().split())
            continue
        if line == "do" or line == "done":
            if line == "done":
                loop = None
            continue
        if line.startswith("python") and "run.py" in line:
            block = [line.rstrip("\\").strip()]
            if not raw.rstrip().endswith("\\"):
                out.append((dict(env), loop, " ".join(block))); block = None
            continue
        if block is not None:
            block.append(line.rstrip("\\").strip())
            if not raw.rstrip().endswith("\\"):
                out.append((dict(env), loop, " ".join(block))); block = None
    cells = []
    for env_, loop_, cmd in out:
        values = [(loop_[0], v) for v in loop_[1]] if loop_ else [(None, None)]
        for var, val in values:
            e = dict(env_)
            if var:
                e[var] = val
            s = cmd
            for k in sorted(e, key=len, reverse=True):
                s = s.replace(f"${k}", e[k])
            s = s.replace("'", "")
            argv = shlex.split(s)
            i = argv.index("run.py")
            cells.append({"argv": argv[i + 1:], "seq_len": int(e.get("seq_len", 0)), "pred_len": int(e.get("pred_len", 0)), "model": e.get("model_name")})
    return cells


def setting_of(args) -> str:
    return "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
        args.task_name, args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len, args.d_model,
        args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed, args.distil, args.des, 0)


# --- the sealed design and lock ---------------------------------------------------------------------------------------------

def environment() -> dict:
    out = {"host": socket.gethostname(), "python": platform.python_version(), "platform": platform.platform()}
    for m in ("torch", "numpy", "pandas", "sklearn", "scipy"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception as exc:                                    # noqa: BLE001
            out[m] = f"MISSING ({type(exc).__name__})"
    try:
        import torch
        out["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
        out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        out["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    except Exception:                                               # noqa: BLE001
        pass
    out["gpu_devices"] = gpu_state()                                   # physical identity (UUID) and thermal state, by nvidia-smi
    out["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return out


def gpu_state() -> list:
    """Every GPU nvidia-smi sees: UUID, name, temperature, utilization, memory — the physical device identity an execution
    record must carry (a CUDA index is an assumption; a UUID is a device)."""
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except Exception:                                               # noqa: BLE001
        return []
    rows = []
    for line in proc.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 7:
            rows.append({"index": int(parts[0]), "uuid": parts[1], "name": parts[2], "temperature_c": float(parts[3]), "utilization_pct": float(parts[4]),
                         "memory_used_mib": float(parts[5]), "memory_total_mib": float(parts[6])})
    return rows


def seal(*, seq_len: int = 96, seeds=(2021, 2022, 2023), horizons=(96, 192, 336, 720), protocol: str = "L96") -> dict:
    """The lock and the design, sealed BEFORE any delivery: nothing in it depends on data or results."""
    git = author_git()
    if not git["pinned_matches"] or not git["clean"]:
        raise SotaRefusal(f"REFUSED: the author clone is not at the pinned revision or not clean: {git}")
    parser, fix_seed = author_parser()
    cells_script = [c for c in script_cells() if c["seq_len"] == seq_len and c["pred_len"] in horizons]
    if sorted(c["pred_len"] for c in cells_script) != sorted(horizons):
        raise SotaRefusal(f"REFUSED: the author script has no L={seq_len} invocation for every horizon {horizons}: {[c['pred_len'] for c in cells_script]}")
    B = _module("df_benchmark_contract")
    receipt = json.loads(STORE_RECEIPT.read_text())["results"][0]
    if receipt["sha256"] != FILE_SHA256 or not receipt["digest_matches_lfs_oid"]:
        raise SotaRefusal("REFUSED: the benchmark store receipt does not carry the official digest")
    cells, contracts = [], {}
    for c in sorted(cells_script, key=lambda c: c["pred_len"]):
        args = parser.parse_args(c["argv"])
        effective = {k: v for k, v in sorted(vars(args).items())}
        contract = B.ecl321_official_tsl_ours(horizon_steps=c["pred_len"], input_window_steps=seq_len)
        contracts[str(c["pred_len"])] = contract.to_design_block(comparability={**B.decide(contract, contract), "comparator_state": "PLANNED_REFERENCE"})
        for seed in seeds:
            cells.append({"cell_id": f"L{seq_len}_h{c['pred_len']}_s{seed}", "arm": "TimeFilter", "protocol": protocol, "seq_len": seq_len,
                          "horizon": c["pred_len"], "seed": int(seed), "argv": c["argv"], "effective_args": effective, "setting": setting_of(args),
                          "contract_sha256": contract.sha256()})
    n = receipt["rows"]
    train, test = int(0.7 * n), int(0.2 * n)
    lock = {
        "paper": PAPER["citation"], "protocol": protocol, "published": PAPER["L96" if protocol == "L96" else "Lsearched"],
        "source": {"repository": "https://github.com/TROUBADOUR000/TimeFilter", "revision": PINNED_COMMIT, "git": git, "files_sha256": source_digests(),
                   "requirements_txt": (AUTHOR_REPO / "requirements.txt").read_text().split()},
        "dataset": {"lake": LAKE, "resource": RESOURCE, "sha256": FILE_SHA256, "bytes": receipt["bytes"], "rows": n, "columns": receipt["columns"],
                    "channels": receipt["channels"], "provenance": receipt["provenance"], "first_label": receipt["first_label"], "last_label": receipt["last_label"],
                    "missing_values": receipt["missing_values"], "column_order": "the file's own order: date, 0..319, OT (features M: every channel is input and target)"},
        "partitions": {"rule": "Dataset_Custom: num_train = int(0.7 n), num_test = int(0.2 n), num_vali = n - train - test; borders "
                               "[0, train), [train - L, train + vali), [n - test - L, n); windows = rows - L - T + 1 per split",
                       "rows": {"train": train, "vali": n - train - test, "test": test},
                       "windows_per_split": {str(h): {"train": train - seq_len - h + 1, "vali": (n - train - test) + seq_len - seq_len - h + 1,
                                                      "test": test + seq_len - seq_len - h + 1} for h in horizons},
                       # the paper's Table 5 "dataset size" convention counts rows_of_segment - L + 1 (pred_len not subtracted); the loader's
                       # __len__ above is what is scored; both are recorded so the two never get confused
                       "paper_table5_convention_rows_minus_L_plus_1": {"train": train - seq_len + 1, "vali": (n - train - test) + 1, "test": test + 1}},
        "preprocessing": {"scaler": "sklearn StandardScaler fit on the TRAIN rows [0, train) of every channel, applied to all rows (scale=True)",
                          "inverse": "none: metrics in the normalized space (--inverse False)", "time_marks": "timeF features, freq h; consumed by the loader, unused by TimeFilter",
                          "missing": "none in the file"},
        "training": {"optimizer": "torch.optim.Adam(lr = learning_rate), default betas/eps/weight_decay", "loss": "nn.MSELoss() + 0.05 x MoE routing loss (hard-coded alpha in train())",
                     "lr_schedule": "adjust_learning_rate lradj=cosine: lr_e = lr/2 (1 + cos(e / train_epochs pi)) set after epoch e", "batch": "batch_size from the script; train loader shuffle=True, drop_last=False",
                     "epochs": "train_epochs from the script; EarlyStopping(patience from run.py default, delta 0) on the validation MSE (no MoE term)",
                     "checkpoint": "the lowest-validation-loss epoch's state_dict (checkpoint.pth), reloaded before test()", "amp": False,
                     "test_loss_printed_each_epoch": "the author's train() evaluates the test loss every epoch for logging only; early stopping uses the validation loss"},
        "evaluation": {"scorer": "utils.metrics.metric(preds, trues) on the concatenated float32 test predictions: MAE = mean|.|, MSE = mean(.)^2 over every window x step x channel",
                       "test_loader": "shuffle=False, drop_last=False, batch_size from the script", "aggregation": "one mean per horizon; the paper averages the four horizons"},
        "seeds": {"author": f"run.py fixes random/torch/numpy seeds to {fix_seed} and offers no CLI path to vary it; the paper reports three runs",
                  "ours": list(int(s) for s in seeds), "how": "the same three calls (random.seed, torch.manual_seed, np.random.seed) with each seed"},
        "environment_author": {"torch": "2.3.1", "numpy": "1.26.4", "pandas": "2.2.3", "scikit_learn": "1.5.2", "gpu": "NVIDIA A100 40GB (paper A.3)"},
        "environment_ours": environment(),
        "operational_patches": [
            {"what": "import shims for sktime.datasets and patoolib", "why": "author modules import them at module level for UEA/M4 paths not on the forecasting path",
             "effect": "none: the shims refuse any call; verified by tests", "where": str(SHIMS)},
            {"what": "run.py __main__ replicated by main_like_run_py", "why": "run.py hard-codes fix_seed 2021; the paper reports three runs", "effect": "none beyond the seed value"},
            {"what": "exp_long_term_forecasting.metric wrapped to capture preds/trues", "why": "the author's test() saves no arrays (np.save lines commented out)",
             "effect": "none: the author's return value is passed through; the captured arrays are the ones the author scores"},
            {"what": "np.Inf restored as an alias of np.inf before importing the author modules", "why": "NumPy 2 removed the alias; utils/tools.py EarlyStopping initialises val_loss_min = np.Inf",
             "effect": "none: the same float object the author's NumPy named; verified by tests"}],
        "paper_code_disagreements": [
            "paper A.3 / Table 6 give lr 1e-3, e_layers 2, d_model 512, d_ff 512, patch 32, epochs 15 for Electricity and batch 16; the script adds dropout 0.5 (h96) / 0.4 (h192-720) and leaves patience (3), lradj (cosine), n_heads (4), alpha (0.1), top_p (0.5), pos (1), use_norm (1) at run.py defaults: not stated in the paper",
            "Table 9 says L searched in {192, 336, 512, 720}; the script offers only L = 512 (patch 128, top_p 0.0) for the long-horizon runs",
            "requirements.txt omits sktime and patool which the code imports",
            "the paper reports the std over three runs; run.py has one fixed seed (2021)"],
        "agreement": AGREEMENT,
        "replay_rule": AGREEMENT["replay"],
    }
    design = {"schema": SCHEMA, "task": "ecl321_official_tsl", "purpose": "SOTA-first reproduction of TimeFilter on the official processed ECL (RP90-RP97)",
              "sealed_at": now_iso(), "seq_len": seq_len, "seeds": [int(s) for s in seeds], "horizons": list(horizons), "protocol": protocol,
              "benchmark_contract": contracts[str(horizons[0])], "contracts": contracts, "cells": cells, "lock": lock,
              "source_data": {"lake": LAKE, "resource": RESOURCE, "sha256": FILE_SHA256}}
    design["design_sha256"] = sha_obj({k: v for k, v in design.items() if k != "design_sha256"})
    return design


def validate(design: dict) -> None:
    body = {k: v for k, v in design.items() if k != "design_sha256"}
    if design.get("schema") != SCHEMA or design.get("design_sha256") != sha_obj(body):
        raise SotaRefusal("REFUSED: the design digest does not recompute from its content (relabeled or edited design)")


def code_drift(design: dict) -> dict:
    """The author files now vs the sealed digests: a changed author file is a protocol deviation, never silent."""
    sealed = design["lock"]["source"]["files_sha256"]
    now = source_digests()
    return {f: {"sealed": sealed.get(f), "now": now.get(f)} for f in sorted(set(sealed) | set(now)) if sealed.get(f) != now.get(f)}


# --- the author environment ---------------------------------------------------------------------------------------------------

def author_env() -> None:
    """The author's package on the path FIRST (its modules are top-level: exp, models, layers, data_provider, utils), the shims after."""
    sys.dont_write_bytecode = True                              # never litter the pinned clone with __pycache__ (its cleanliness is part of the lock)
    for p in (str(SHIMS), str(AUTHOR_REPO)):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(SHIMS))
    sys.path.insert(0, str(AUTHOR_REPO))
    # operational patch (declared in the lock): NumPy 2 removed the alias `np.Inf`, which the author's EarlyStopping uses as
    # its initial "best" (utils/tools.py); the alias is restored to the same object it always named. No arithmetic changes.
    if not hasattr(np, "Inf"):
        np.Inf = np.inf


def fix_seeds(seed: int) -> None:
    """Exactly run.py's three calls."""
    import random
    import torch
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)


def build_args(argv: list, *, data_dir: Path, data_name: str, checkpoints: Path, gpu: int = 0, use_gpu: bool | None = None):
    """run.py's parse and post-parse logic, verbatim in effect; the data root and checkpoint folder are the only transport values."""
    import torch
    parser, _ = author_parser()
    args = parser.parse_args(list(argv))
    args.root_path, args.data_path, args.checkpoints, args.gpu = str(data_dir), data_name, str(checkpoints), int(gpu)
    if use_gpu is not None:
        args.use_gpu = bool(use_gpu)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(i) for i in args.devices.split(",")]
        args.gpu = args.device_ids[0]
    return args


class Captured:
    """The arrays the author's test() hands to `metric`, and the author's own returned value."""
    def __init__(self):
        self.preds = self.trues = self.value = None


def capture_metric(exp_module, captured: Captured):
    original = exp_module.metric

    def wrapped(preds, trues):
        captured.preds, captured.trues = preds, trues
        captured.value = original(preds, trues)
        return captured.value
    exp_module.metric = wrapped
    return original


def main_like_run_py(argv: list, *, seed: int, data_dir: Path, data_name: str, work: Path, gpu: int = 0, use_gpu: bool | None = None,
                     log: Path | None = None, train: bool = True) -> dict:
    """run.py's `__main__` for one invocation: seeds, parse, Exp, train, test — the author's code path, in the author's cwd layout."""
    author_env()
    fix_seeds(seed)
    args = build_args(argv, data_dir=data_dir, data_name=data_name, checkpoints=work / "checkpoints", gpu=gpu, use_gpu=use_gpu)
    exp_module = importlib.import_module("exp.exp_long_term_forecasting")
    Exp = exp_module.Exp_Long_Term_Forecast
    captured = Captured()
    original = capture_metric(exp_module, captured)
    work.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(work)                                             # the author's test() writes ./results/ and result_long_term_forecast.txt in cwd
    t0, c0 = time.time(), time.process_time()
    try:
        with open(log or (work / "author_stdout.log"), "a") as fh, contextlib.redirect_stdout(fh):
            print("Args in experiment:"); print(vars(args))
            exp = Exp(args)
            setting = setting_of(args)
            if train:
                print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
                exp.train(setting)
            print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
            exp.test(setting, test=0 if train else 1)
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        os.chdir(cwd)
        exp_module.metric = original
    mae, mse = float(captured.value[0]), float(captured.value[1])
    return {"args": vars(args), "setting": setting, "preds": captured.preds, "trues": captured.trues, "author_metric": {"mae": mae, "mse": mse},
            "wall_seconds": time.time() - t0, "cpu_seconds": time.process_time() - c0, "checkpoint": work / "checkpoints" / setting / "checkpoint.pth",
            "n_parameters": int(sum(p.numel() for p in exp.model.parameters())), "device": str(exp.device)}


def parse_author_log(text: str) -> dict:
    epochs, stops = [], []
    for line in text.splitlines():
        if line.startswith("Epoch:") and "Train Loss" in line:
            parts = line.replace("|", " ").replace(",", " ").split()
            try:
                epochs.append({"epoch": int(parts[1]), "steps": int(parts[3]), "train_loss": float(parts[6]), "vali_loss": float(parts[9]), "test_loss": float(parts[12])})
            except (ValueError, IndexError):
                pass
        if "EarlyStopping counter" in line or line.strip() == "Early stopping":
            stops.append(line.strip())
    best = min(epochs, key=lambda e: e["vali_loss"]) if epochs else None
    return {"epochs_run": len(epochs), "per_epoch": epochs, "best_epoch_by_vali": best["epoch"] if best else None,
            "early_stopped": any(s == "Early stopping" for s in stops), "stop_lines": stops[-3:]}


# --- preparation: the delivered file and the author's borders/scaler as the preparation's own evidence -------------------------

def governance_modules():
    return _module("df_e1_governed"), _module("df_utility_run")


def acquire(a, design: dict, unit: str) -> dict:
    G, _ = governance_modules()
    return G.acquire(run_id=a.run_id or f"sota-{design['design_sha256'][:8]}", root=a.root, lake=a.lake, resource=a.resource, unit_id=unit, role=ROLE,
                     gov_url=a.gov_url, api_key_file=a.api_key_file, design_sha256=design["design_sha256"], expect_sha256=design["source_data"]["sha256"])


def delivered_file(root: Path, design: dict, unit: str | None) -> Path:
    """The bytes delivered to `unit`; with unit None, any unit's delivery on this host whose bytes are the sealed file."""
    G, _ = governance_modules()
    if unit is None:
        doc = json.loads((Path(root) / "DELIVERIES.json").read_text())
        for u, d in (doc.get("units") or {}).items():
            if d.get("sha256") == design["source_data"]["sha256"] and Path(d["path"]).is_file():
                unit = u; break
        if unit is None:
            raise SotaRefusal("REFUSED: no delivery of the sealed file on this host")
    path = Path(G.require_delivery(root, design, unit)["delivery"]["path"])
    if sha_file(path) != design["source_data"]["sha256"]:
        raise SotaRefusal("REFUSED: the delivered bytes are not the sealed official file")
    return path


def bench_data(design: dict, data_path: Path) -> tuple:
    """The author's Dataset_Custom for every (L, T): borders, scaler fit on the train rows, window counts — recorded as the
    preparation's evidence (BENCH_DATA.npz + .json) that the closure binds to the accepted prepare terminal."""
    author_env()
    from types import SimpleNamespace
    DL = importlib.import_module("data_provider.data_loader")
    arrays, record = {}, {"schema": "df_sota_bench_data.v1", "design_sha256": design["design_sha256"], "file_sha256": sha_file(data_path), "sets": {}}
    ns = SimpleNamespace(augmentation_ratio=0)
    for h in design["horizons"]:
        key = f"L{design['seq_len']}_h{h}"
        sets = {}
        for flag in ("train", "val", "test"):
            ds = DL.Dataset_Custom(ns, str(data_path.parent), flag=flag, size=[design["seq_len"], 48, h], features="M", data_path=data_path.name,
                                   target="OT", scale=True, timeenc=1, freq="h")
            sets[flag] = {"windows": len(ds), "rows": int(ds.data_x.shape[0]), "channels": int(ds.data_x.shape[1])}
            if flag == "train":
                arrays[f"{key}_scaler_mean"] = np.asarray(ds.scaler.mean_, dtype=np.float64)
                arrays[f"{key}_scaler_scale"] = np.asarray(ds.scaler.scale_, dtype=np.float64)
                sets["scaler_sha256"] = sha_array(np.concatenate([arrays[f"{key}_scaler_mean"], arrays[f"{key}_scaler_scale"]]))
        record["sets"][key] = sets
    return arrays, record


def run_prepare(a, design: dict) -> dict:
    validate(design)
    G, U = governance_modules()
    root = Path(a.root)
    started = U._z(U.now_iso())
    acquire(a, design, "prepare")
    path = delivered_file(root, design, "prepare")
    try:
        arrays, record = bench_data(design, path)
    except BaseException as exc:
        G.report_failed(root, "prepare", f"prepare refused: {str(exc)[:200]}", gov_url=a.gov_url, api_key_file=a.api_key_file)
        raise
    np.savez(root / "BENCH_DATA.npz", **arrays)
    record["data_sha256"] = sha_file(root / "BENCH_DATA.npz")
    (root / "BENCH_DATA.json").write_text(json.dumps(record, indent=1))
    terminal = U._terminal(status="COMPLETED", reason=None, cost={"wall_seconds": 0.0, "cpu_seconds": time.process_time()},
                           metrics=[U._metric("sota.prepare.rows", float(design["lock"]["dataset"]["rows"]), "rows")],
                           started=started, finished=U._z(U.now_iso()),
                           tags={"purpose": "SOTA_REPRODUCTION", "classification": "NON_GOVERNING", "phase": "REPRODUCTION", "unit": "prepare",
                                 "role": "PREPARATION", "design_sha256": design["design_sha256"], "lake": a.lake, "resource": a.resource})
    terminal["artifacts"] = [{"role": r, "sha256": sha_file(root / f), "bytes": (root / f).stat().st_size} for r, f in (("data", "BENCH_DATA.npz"), ("record", "BENCH_DATA.json"))]
    (root / "TERMINALS").mkdir(exist_ok=True)
    (root / "TERMINALS" / "prepare.json").write_text(json.dumps(terminal, indent=1, default=str))
    reported = G.report_terminal(root, "prepare", terminal, gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(root / "outbox"), started_at=started)
    if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
        raise SotaRefusal(f"REFUSED: the prepare terminal was not accepted: {reported['flushed']['failures']}")
    return record


# --- a cell -----------------------------------------------------------------------------------------------------------------

def run_cell(design: dict, cell: dict, *, data_path: Path, folder: Path, gpu: int = 0, use_gpu: bool | None = None) -> dict:
    """The author's run for one cell; artifacts written under `folder`. Nothing scientific is decided here."""
    folder.mkdir(parents=True, exist_ok=True)
    work = folder / "work"
    gpu_before = gpu_state()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:                                           # noqa: BLE001
        pass
    res = main_like_run_py(cell["argv"], seed=cell["seed"], data_dir=data_path.parent, data_name=data_path.name, work=work, gpu=gpu, use_gpu=use_gpu,
                           log=folder / "author_stdout.log")
    preds, trues = np.asarray(res["preds"], dtype=np.float32), np.asarray(res["trues"], dtype=np.float32)
    np.savez(folder / "arrays.npz", pred=preds)                    # uncompressed by the owner's decision (float32 outputs compress < 13 %)
    trues_sha = sha_array(trues)
    ckpt = folder / "checkpoint.pth"
    shutil.copy2(res["checkpoint"], ckpt)
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    peak_gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            peak_gpu = int(torch.cuda.max_memory_allocated())
    except Exception:                                           # noqa: BLE001
        pass
    # independent float64 metric on the same arrays (the author's is float32 by construction), chunked: no full-size temporaries
    f64 = float64_metrics(preds, trues)
    training = parse_author_log((folder / "author_stdout.log").read_text())
    record = {"schema": "df_sota_cell_record.v1", "cell": {k: cell[k] for k in ("cell_id", "arm", "protocol", "seq_len", "horizon", "seed")},
              "design_sha256": design["design_sha256"], "setting": res["setting"], "effective_args": {k: v for k, v in sorted(res["args"].items())},
              "author_metric_float32": res["author_metric"], "independent_metric_float64": f64,
              "shapes": {"pred": list(preds.shape), "true": list(trues.shape)}, "dtype": str(preds.dtype),
              "arrays_sha256": sha_file(folder / "arrays.npz"), "pred_sha256": sha_array(preds), "true_sha256": trues_sha, "checkpoint_sha256": sha_file(ckpt),
              "checkpoint_bytes": ckpt.stat().st_size, "n_parameters": res["n_parameters"], "device": res["device"], "training": training,
              "cost": {"wall_seconds": res["wall_seconds"], "cpu_seconds": ru1.ru_utime + ru1.ru_stime - (ru0.ru_utime + ru0.ru_stime),
                       "peak_rss_bytes": int(ru1.ru_maxrss) * 1024, "peak_gpu_allocated_bytes": peak_gpu, "host": socket.gethostname(),
                       "gpu_before": gpu_before, "gpu_after": gpu_state()},
              "environment": environment(), "source_files_sha256": source_digests(), "source_drift": code_drift(design),
              "data_file_sha256": sha_file(data_path), "at": now_iso()}
    (folder / "cell.json").write_text(json.dumps(record, indent=1, default=str))
    return record


def child(a, design: dict) -> dict:
    validate(design)
    G, U = governance_modules()
    root = Path(a.root)
    cell = next((c for c in design["cells"] if c["cell_id"] == a.unit), None)
    if cell is None:
        raise SotaRefusal(f"REFUSED: {a.unit} is not a cell of the sealed design")
    if code_drift(design):
        raise SotaRefusal(f"REFUSED: the author files differ from the sealed digests: {code_drift(design)}")
    started = U._z(U.now_iso())
    acquire(a, design, a.unit)
    path = delivered_file(root, design, a.unit)
    folder = root / "attempts" / a.unit
    try:
        record = run_cell(design, cell, data_path=path, folder=folder, gpu=a.gpu, use_gpu=None if not a.cpu else False)
    except BaseException as exc:
        (folder).mkdir(parents=True, exist_ok=True)
        (folder / "FAILED.json").write_text(json.dumps({"at": now_iso(), "error": f"{type(exc).__name__}: {str(exc)[:600]}"}))
        G.report_failed(root, a.unit, f"cell failed: {type(exc).__name__}: {str(exc)[:160]}", gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(root / "outbox"))
        raise
    metrics = [U._metric("sota.test.mse_normalized", record["author_metric_float32"]["mse"], "z", split="test", horizon=cell["horizon"]),
               U._metric("sota.test.mae_normalized", record["author_metric_float32"]["mae"], "z", split="test", horizon=cell["horizon"]),
               U._metric("sota.train.epochs_run", float(record["training"]["epochs_run"]), "epochs"),
               U._metric("sota.train.best_epoch", float(record["training"]["best_epoch_by_vali"] or 0), "epoch")]
    terminal = U._terminal(status="COMPLETED", reason=None, cost={"wall_seconds": record["cost"]["wall_seconds"], "cpu_seconds": record["cost"]["cpu_seconds"]},
                           metrics=metrics, started=started, finished=U._z(U.now_iso()),
                           tags={"purpose": "SOTA_REPRODUCTION", "classification": "NON_GOVERNING", "phase": "REPRODUCTION", "unit": a.unit, "role": "forecast",
                                 "arm": cell["arm"], "horizon": str(cell["horizon"]), "seq_len": str(cell["seq_len"]), "seed": str(cell["seed"]),
                                 "design_sha256": design["design_sha256"], "lake": a.lake, "resource": a.resource, "host": socket.gethostname()})
    terminal["artifacts"] = [{"role": r, "sha256": sha_file(folder / f), "bytes": (folder / f).stat().st_size}
                             for r, f in (("predictions", "arrays.npz"), ("checkpoint", "checkpoint.pth"), ("record", "cell.json"))]
    (root / "TERMINALS").mkdir(exist_ok=True)
    (root / "TERMINALS" / f"{a.unit}.json").write_text(json.dumps(terminal, indent=1, default=str))
    reported = G.report_terminal(root, a.unit, terminal, gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(root / "outbox"), started_at=started)
    if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
        raise SotaRefusal(f"REFUSED: the terminal of {a.unit} was not accepted: {reported['flushed']['failures']}")
    return record


def execute(a, design: dict) -> list:
    """The cells of this host, one after another (one GPU per host), each in its own process under this accounted scope."""
    validate(design)
    units = [c["cell_id"] for c in design["cells"] if (not a.seeds or c["seed"] in a.seeds) and (not a.horizons or c["horizon"] in a.horizons)]
    if a.units:
        units = [u for u in units if u in a.units]
    root = Path(a.root)
    out = []
    for unit in units:
        folder = root / "attempts" / unit
        receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text()).get("units", {}) if (root / "TERMINAL_RECEIPTS.json").is_file() else {}
        if unit in receipts and (folder / "cell.json").is_file():
            out.append({"unit": unit, "ok": True, "reused": True}); continue
        argv = [sys.executable, str(Path(__file__).resolve()), "child", "--root", str(root), "--unit", unit, "--gov-url", a.gov_url, "--api-key-file", str(a.api_key_file),
                "--lake", a.lake, "--resource", a.resource, "--gpu", str(a.gpu)] + (["--cpu"] if a.cpu else []) + (["--run-id", a.run_id] if a.run_id else [])
        t0 = time.time()
        proc = subprocess.run(argv, capture_output=True, text=True)
        (folder).mkdir(parents=True, exist_ok=True)
        (folder / "child_stderr.log").write_text(proc.stderr[-20000:])
        out.append({"unit": unit, "ok": proc.returncode == 0, "returncode": proc.returncode, "wall_seconds": time.time() - t0, "tail": proc.stderr[-400:]})
        print(json.dumps(out[-1]), flush=True)
    (root / f"EXECUTE.{socket.gethostname()}.json").write_text(json.dumps(out, indent=1))
    return out


# --- merge: a worker's cells into the coordinator root (artifacts verified by digest against the worker's own terminals) ------------

def merge(root: Path, source: Path) -> dict:
    """Copy a worker's attempts, terminals and receipts into the coordinator root. Nothing is trusted from the copy itself: each
    unit's artifacts must hash to the digests of the worker's terminal file, the design digests must be identical, and the
    receipts merge unit by unit (a unit already present with another receipt is a problem, never overwritten)."""
    root, source = Path(root), Path(source)
    d0, d1 = json.loads((root / "DESIGN.json").read_text()), json.loads((source / "DESIGN.json").read_text())
    out = {"from": str(source), "units": {}, "problems": []}
    if d0.get("design_sha256") != d1.get("design_sha256"):
        out["problems"].append("the worker root was sealed under another design digest"); return out
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text()) if (root / "TERMINAL_RECEIPTS.json").is_file() else {"units": {}}
    src_receipts = (json.loads((source / "TERMINAL_RECEIPTS.json").read_text()) if (source / "TERMINAL_RECEIPTS.json").is_file() else {}).get("units") or {}
    for folder in sorted((source / "attempts").glob("*")):
        unit = folder.name
        if not (folder / "cell.json").is_file():
            continue
        terminal_path = source / "TERMINALS" / f"{unit}.json"
        if not terminal_path.is_file() or unit not in src_receipts:
            out["problems"].append(f"{unit}: no terminal or receipt on the worker; not merged"); continue
        terminal = json.loads(terminal_path.read_text())
        acc = {a["role"]: a["sha256"] for a in terminal.get("artifacts") or []}
        digests = {r: sha_file(folder / f) for r, f in (("predictions", "arrays.npz"), ("checkpoint", "checkpoint.pth"), ("record", "cell.json")) if (folder / f).is_file()}
        if any(acc.get(r) != digests.get(r) for r in ("predictions", "checkpoint", "record")):
            out["problems"].append(f"{unit}: the worker's artifacts do not hash to its terminal; not merged"); continue
        if unit in receipts["units"] and receipts["units"][unit] != src_receipts[unit]:
            out["problems"].append(f"{unit}: already present under another receipt; not overwritten"); continue
        dest = root / "attempts" / unit
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(folder, dest, ignore=shutil.ignore_patterns("work", "replay_work"))
        (root / "TERMINALS").mkdir(exist_ok=True)
        shutil.copy2(terminal_path, root / "TERMINALS" / f"{unit}.json")
        receipts["units"][unit] = src_receipts[unit]
        out["units"][unit] = {"merged": True, "artifacts": digests, "host": (json.loads((folder / "cell.json").read_text()).get("cost") or {}).get("host")}
    # units without an attempts folder (prepare, preflights): terminal + receipt travel too; the preparation evidence with them
    for unit, receipt in src_receipts.items():
        terminal_path = source / "TERMINALS" / f"{unit}.json"
        if unit in out["units"] or not terminal_path.is_file() or (source / "attempts" / unit).is_dir():
            continue
        if unit in receipts["units"] and receipts["units"][unit] != receipt:
            out["problems"].append(f"{unit}: already present under another receipt; not overwritten"); continue
        (root / "TERMINALS").mkdir(exist_ok=True)
        shutil.copy2(terminal_path, root / "TERMINALS" / f"{unit}.json")
        receipts["units"][unit] = receipt
        out["units"][unit] = {"merged": True, "artifacts": None}
    for name in ("BENCH_DATA.npz", "BENCH_DATA.json"):
        if (source / name).is_file() and not (root / name).is_file():
            shutil.copy2(source / name, root / name)
    (root / "TERMINAL_RECEIPTS.json").write_text(json.dumps(receipts, indent=1))
    for extra in sorted(source.glob("PREFLIGHT*.json")) + sorted(source.glob("EXECUTE.*.json")):
        shutil.copy2(extra, root / extra.name)
    (root / f"MERGE.{source.name}.{int(time.time())}.json").write_text(json.dumps(out, indent=1))
    return out


# --- preflight (RP94): bounded, no test score ---------------------------------------------------------------------------------

def preflight(design: dict, *, data_path: Path, work: Path, steps: int = 20, horizons=None, gpu: int = 0, use_gpu: bool | None = None) -> dict:
    """K optimizer steps of the author's training step on the real train loader and a few validation forwards: seconds per step,
    peak memory, parameters -> per-cell and full allocation. The test loader is never scored (its length is counted only)."""
    author_env()
    import torch
    exp_module = importlib.import_module("exp.exp_long_term_forecasting")
    out = {"schema": "df_sota_preflight.v1", "design_sha256": design["design_sha256"], "host": socket.gethostname(), "steps_timed": steps, "cells": {}, "environment": environment()}
    for h in (horizons or design["horizons"]):
        cell = next(c for c in design["cells"] if c["horizon"] == h)
        fix_seeds(cell["seed"])
        args = build_args(cell["argv"], data_dir=data_path.parent, data_name=data_path.name, checkpoints=work / "ckpt", gpu=gpu, use_gpu=use_gpu)
        with contextlib.redirect_stdout(io.StringIO()):
            exp = exp_module.Exp_Long_Term_Forecast(args)
            train_data, train_loader = exp._get_data(flag="train")
            vali_data, vali_loader = exp._get_data(flag="val")
            test_data, test_loader = exp._get_data(flag="test")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        model_optim, criterion = exp._select_optimizer(), exp._select_criterion()
        exp.model.train()
        times = []
        it = iter(train_loader)
        for i in range(steps):
            batch_x, batch_y, _, _ = next(it)
            t0 = time.perf_counter()
            model_optim.zero_grad()
            batch_x = batch_x.float().to(exp.device); batch_y = batch_y.float().to(exp.device)
            outputs, moe_loss = exp.model(batch_x, exp.masks, is_training=True)
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]; batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y) + 0.05 * moe_loss
            loss.backward(); model_optim.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        exp.model.eval()
        vt = []
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(vali_loader):
                if i >= max(3, steps // 4):
                    break
                t0 = time.perf_counter()
                exp.model(batch_x.float().to(exp.device), exp.masks, is_training=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                vt.append(time.perf_counter() - t0)
        warm = times[max(1, len(times) // 4):]
        step_s, fwd_s = float(np.median(warm)), float(np.median(vt))
        n_train, n_vali, n_test = len(train_loader), len(vali_loader), len(test_loader)
        epoch_s = n_train * step_s + (n_vali + n_test) * fwd_s                   # the author evaluates vali AND test each epoch
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ru = resource.getrusage(resource.RUSAGE_SELF)
        out["cells"][f"L{design['seq_len']}_h{h}"] = {
            "n_parameters": int(sum(p.numel() for p in exp.model.parameters())), "batches": {"train": n_train, "vali": n_vali, "test": n_test},
            "windows": {"train": len(train_data), "vali": len(vali_data), "test": len(test_data)},
            "seconds_per_train_step_median": step_s, "seconds_per_eval_batch_median": fwd_s, "epoch_seconds_projected": epoch_s,
            "max_epochs": int(args.train_epochs), "cell_seconds_projected_max": epoch_s * int(args.train_epochs) + n_test * fwd_s,
            "peak_gpu_allocated_bytes": peak, "peak_rss_bytes": int(ru.ru_maxrss) * 1024, "device": str(exp.device), "batch_size": int(args.batch_size)}
        del exp, model_optim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    per_seed = sum(c["cell_seconds_projected_max"] for c in out["cells"].values())
    out["allocation"] = {"cells_total": len(design["cells"]), "seconds_per_seed_all_horizons_max": per_seed, "seconds_all_cells_max": per_seed * len(design["seeds"]),
                         "hours_all_cells_max": per_seed * len(design["seeds"]) / 3600.0,
                         "reading": "upper bound: every cell runs its full train_epochs; early stopping (patience 3 on validation MSE) can only shorten it"}
    return out


def run_preflight(a, design: dict) -> dict:
    validate(design)
    G, U = governance_modules()
    root = Path(a.root)
    unit = f"preflight_{socket.gethostname()}"
    started = U._z(U.now_iso())
    acquire(a, design, unit)
    path = delivered_file(root, design, unit)
    try:
        doc = preflight(design, data_path=path, work=root / "preflight_work", steps=a.steps, horizons=a.horizons or None, gpu=a.gpu, use_gpu=None if not a.cpu else False)
    except BaseException as exc:
        G.report_failed(root, unit, f"preflight failed: {type(exc).__name__}: {str(exc)[:160]}", gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(root / "outbox"))
        raise
    (root / f"PREFLIGHT.{socket.gethostname()}.json").write_text(json.dumps(doc, indent=1, default=str))
    ru = resource.getrusage(resource.RUSAGE_SELF)
    terminal = U._terminal(status="COMPLETED", reason=None, cost={"wall_seconds": 0.0, "cpu_seconds": ru.ru_utime + ru.ru_stime},
                           metrics=[U._metric("sota.preflight.hours_all_cells_max", doc["allocation"]["hours_all_cells_max"], "hours")],
                           started=started, finished=U._z(U.now_iso()),
                           tags={"purpose": "SOTA_REPRODUCTION", "classification": "NON_GOVERNING", "phase": "PREFLIGHT", "unit": unit, "role": "PREFLIGHT",
                                 "design_sha256": design["design_sha256"], "host": socket.gethostname()})
    terminal["artifacts"] = [{"role": "record", "sha256": sha_file(root / f"PREFLIGHT.{socket.gethostname()}.json"), "bytes": (root / f"PREFLIGHT.{socket.gethostname()}.json").stat().st_size}]
    (root / "TERMINALS").mkdir(exist_ok=True)
    (root / "TERMINALS" / f"{unit}.json").write_text(json.dumps(terminal, indent=1, default=str))
    reported = G.report_terminal(root, unit, terminal, gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(root / "outbox"), started_at=started)
    if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
        raise SotaRefusal(f"REFUSED: the preflight terminal was not accepted: {reported['flushed']['failures']}")
    return doc


# --- RP96: verification and the RP97 table ----------------------------------------------------------------------------------------

def replay_code_sha256() -> str:
    """The digest of the code that PERFORMS a replay (the replay driver and the author-path functions it calls), so that a
    cached replay is invalidated by a change of that path and not by unrelated edits of this file."""
    import inspect
    parts = [inspect.getsource(f) for f in (replay_cell, main_like_run_py, build_args, author_env, fix_seeds, capture_metric, setting_of, author_parser)]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def replay_cell(root: Path, design: dict, unit: str, *, data_path: Path, device: str = "cpu") -> dict:
    """Fresh process: the author's test(test=1) reloads the checkpoint through the author's own path and scores; the captured
    predictions are compared with the stored ones under the frozen replay rule."""
    cell = next(c for c in design["cells"] if c["cell_id"] == unit)
    code = f"""
import json, sys, importlib.util, shutil, numpy as np
from pathlib import Path
spec = importlib.util.spec_from_file_location("df_sota_repro", {str(Path(__file__).resolve())!r}); M = importlib.util.module_from_spec(spec); sys.modules["df_sota_repro"] = M; spec.loader.exec_module(M)
root = Path({str(root)!r}); unit = {unit!r}; design = json.loads((root/"DESIGN.json").read_text()); cell = next(c for c in design["cells"] if c["cell_id"] == unit)
folder = root/"attempts"/unit; work = folder/"replay_work"; shutil.rmtree(work, ignore_errors=True)
(work/"checkpoints"/cell["setting"]).mkdir(parents=True); shutil.copy2(folder/"checkpoint.pth", work/"checkpoints"/cell["setting"]/"checkpoint.pth")
import torch, functools
if {device == "cpu"!r}:
    # operational patch (declared in REPORT.verification.replay_patch): the author's test(test=1) calls torch.load(path) with no
    # map_location, so a checkpoint saved from CUDA cannot be read on a CPU-only replay; placement only, no arithmetic changes
    _load = torch.load
    torch.load = functools.partial(_load, map_location=torch.device("cpu"))
res = M.main_like_run_py(cell["argv"], seed=cell["seed"], data_dir=Path({str(data_path.parent)!r}), data_name={data_path.name!r}, work=work, gpu=0, use_gpu={device != "cpu"}, train=False)
with np.load(folder/"arrays.npz") as z: stored = z["pred"]
rep = np.asarray(res["preds"], dtype=np.float32)
rule = design["lock"]["replay_rule"]
d = np.abs(rep.astype(np.float64) - stored.astype(np.float64))
print(json.dumps({{"unit": unit, "device": res["device"], "max_abs_prediction_difference": float(d.max()), "allclose_rule": bool(np.allclose(rep, stored, atol=rule["atol"], rtol=rule["rtol"])),
                   "replayed_author_metric": res["author_metric"], "true_sha256_replayed": M.sha_array(np.asarray(res["trues"], dtype=np.float32)), "shape": list(rep.shape)}}))
shutil.rmtree(work, ignore_errors=True)
"""
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=7200, env=env)
    if proc.returncode:
        return {"unit": unit, "allclose_rule": False, "error": proc.stderr[-800:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def naive_and_trues(design: dict, cell: dict, data_path: Path) -> dict:
    """The author's test loader (same args, same borders, same scaler): the targets re-derived, and persistence on identical windows."""
    author_env()
    import torch
    from types import SimpleNamespace
    DF = importlib.import_module("data_provider.data_factory")
    args = build_args(cell["argv"], data_dir=data_path.parent, data_name=data_path.name, checkpoints=Path("/nonexistent"), gpu=0, use_gpu=False)
    args.augmentation_ratio = 0
    test_data, test_loader = DF.data_provider(args, "test")
    trues, sq, ab = [], 0.0, 0.0
    n = 0
    for batch_x, batch_y, _, _ in test_loader:
        y = batch_y[:, -args.pred_len:, :].float()
        last = batch_x[:, -1:, :].float().expand(-1, args.pred_len, -1)
        d = (last - y).double()
        sq += float((d * d).sum()); ab += float(d.abs().sum()); n += d.numel()
        trues.append(y.numpy().astype(np.float32))
    trues = np.concatenate(trues, axis=0)
    return {"true_sha256": sha_array(trues), "n_elements": n, "windows": int(trues.shape[0]), "naive": {"mse": sq / n, "mae": ab / n},
            "naive_definition": "persistence: the window's last observed value repeated over the horizon, every channel, same test windows, normalized space"}


def metrics_vault(preds: np.ndarray, test_loader, *, pred_len: int, chunk: int = 128, max_lag: int = 168) -> dict:
    """Every metric a future analysis could want from ONE cell's predictions, so the raw arrays can be deleted afterwards
    (owner's decision 2026-09-22). Everything is accumulated in float64 over chunks of windows: no full-size temporary.
    Space: normalized (train-standardized) units, the space the paper reports in.

    Contents: global MSE/MAE/RMSE/MAPE/MSPE/RSE/R2/CORR/bias; per horizon step (T rows) and per channel (C rows) MSE, MAE,
    bias, residual sd, naive persistence and 24 h seasonal naive (where the window supports it), skill; residual moments
    (mean, var, skew, kurtosis), histogram, approximate quantiles, tail fractions, entropy; residual autocorrelation along the
    window index (channel-mean residual per step, lags 1..max_lag; per-channel lags 1/24/168 at the first and last steps);
    error growth by step; prediction/true correlation and mutual information (64 x 64 histogram); bias of the mean level."""
    W, T, C = preds.shape
    sums = {k: np.zeros((T, C)) for k in ("ab", "sq", "r", "nab", "nsq", "sab", "ssq")}
    sums["y"] = np.zeros((T, C)); sums["y2"] = np.zeros((T, C)); sums["p"] = np.zeros((T, C)); sums["p2"] = np.zeros((T, C)); sums["py"] = np.zeros((T, C))
    r3 = r4 = 0.0; n_mape = 0; mape = 0.0; mspe = 0.0
    hist_edges = np.linspace(-20.0, 20.0, 2001); hist = np.zeros(2000)
    tail = {"1": 0, "2": 0, "3": 0}
    joint = np.zeros((64, 64)); jedges = np.linspace(-6.0, 6.0, 65)
    step_series = np.zeros((W, T)); ch_series_first = np.zeros((W, C)); ch_series_last = np.zeros((W, C))
    w0 = 0
    for batch_x, batch_y, _, _ in test_loader:
        y = batch_y[:, -pred_len:, :].numpy().astype(np.float64); x = batch_x.numpy().astype(np.float64)
        b = y.shape[0]; p = preds[w0:w0 + b].astype(np.float64); d = p - y
        last = x[:, -1:, :]; naive = np.broadcast_to(last, y.shape)
        L = x.shape[1]
        seas = np.stack([x[:, L - 24 + (k % 24), :] for k in range(T)], axis=1) if L >= 24 else naive
        sums["ab"] += np.abs(d).sum(0); sums["sq"] += (d * d).sum(0); sums["r"] += d.sum(0)
        sums["nab"] += np.abs(naive - y).sum(0); sums["nsq"] += ((naive - y) ** 2).sum(0)
        sums["sab"] += np.abs(seas - y).sum(0); sums["ssq"] += ((seas - y) ** 2).sum(0)
        sums["y"] += y.sum(0); sums["y2"] += (y * y).sum(0); sums["p"] += p.sum(0); sums["p2"] += (p * p).sum(0); sums["py"] += (p * y).sum(0)
        r3 += float((d ** 3).sum()); r4 += float((d ** 4).sum())
        nz = np.abs(y) > 1e-8; n_mape += int(nz.sum()); mape += float(np.abs(d[nz] / y[nz]).sum()); mspe += float(((d[nz] / y[nz]) ** 2).sum())
        hist += np.histogram(d, bins=hist_edges)[0]
        for t_ in ("1", "2", "3"):
            tail[t_] += int((np.abs(d) > float(t_)).sum())
        joint += np.histogram2d(np.clip(p.ravel(), -6, 6), np.clip(y.ravel(), -6, 6), bins=[jedges, jedges])[0]
        step_series[w0:w0 + b] = d.mean(axis=2); ch_series_first[w0:w0 + b] = d[:, 0, :]; ch_series_last[w0:w0 + b] = d[:, -1, :]
        w0 += b
    n = W * T * C
    mse, mae = float(sums["sq"].sum() / n), float(sums["ab"].sum() / n)
    ymean = float(sums["y"].sum() / n); sst = float(sums["y2"].sum() - n * ymean ** 2)
    pmean = float(sums["p"].sum() / n)
    cov = float(sums["py"].sum() - n * pmean * ymean); vp = float(sums["p2"].sum() - n * pmean ** 2); vy = sst
    rmean = float(sums["r"].sum() / n); rvar = mse - rmean ** 2
    skew = (r3 / n - 3 * rmean * rvar - rmean ** 3) / (rvar ** 1.5) if rvar > 0 else None
    kurt = (r4 / n - 4 * rmean * r3 / n + 6 * rmean ** 2 * mse - 3 * rmean ** 4) / (rvar ** 2) if rvar > 0 else None
    cdf = np.cumsum(hist) / max(1.0, hist.sum()); centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    quantiles = {str(q): float(centers[min(len(centers) - 1, int(np.searchsorted(cdf, q)))]) for q in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999)}
    ph = hist / max(1.0, hist.sum()); entropy_bits = float(-(ph[ph > 0] * np.log2(ph[ph > 0])).sum())
    pj = joint / max(1.0, joint.sum()); pa, pb = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
    nzj = pj > 0; mi_bits = float((pj[nzj] * np.log2(pj[nzj] / (pa @ pb)[nzj])).sum())
    def acf(series: np.ndarray, lags: int) -> list:
        s_ = series - series.mean(axis=0, keepdims=True); den = (s_ * s_).sum(axis=0)
        out = []
        for lag in range(1, lags + 1):
            num = (s_[lag:] * s_[:-lag]).sum(axis=0)
            out.append(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan))
        return out
    lags = min(max_lag, W - 2)
    step_acf = np.array(acf(step_series, lags))                                    # [lags, T]
    first_acf = {str(l): acf(ch_series_first, l)[-1].tolist() for l in (1, 24, 168) if l < W - 1}
    last_acf = {str(l): acf(ch_series_last, l)[-1].tolist() for l in (1, 24, 168) if l < W - 1}
    per_step = {"mse": (sums["sq"].sum(1) / (W * C)).tolist(), "mae": (sums["ab"].sum(1) / (W * C)).tolist(), "bias": (sums["r"].sum(1) / (W * C)).tolist(),
                "naive_mse": (sums["nsq"].sum(1) / (W * C)).tolist(), "naive_mae": (sums["nab"].sum(1) / (W * C)).tolist(),
                "seasonal24_mse": (sums["ssq"].sum(1) / (W * C)).tolist(), "seasonal24_mae": (sums["sab"].sum(1) / (W * C)).tolist()}
    per_step["skill_mae_vs_naive"] = [1 - a / b if b > 0 else None for a, b in zip(per_step["mae"], per_step["naive_mae"])]
    per_step["error_growth_mae_over_step1"] = [a / per_step["mae"][0] if per_step["mae"][0] > 0 else None for a in per_step["mae"]]
    per_channel = {"mse": (sums["sq"].sum(0) / (W * T)).tolist(), "mae": (sums["ab"].sum(0) / (W * T)).tolist(), "bias": (sums["r"].sum(0) / (W * T)).tolist(),
                   "naive_mse": (sums["nsq"].sum(0) / (W * T)).tolist(), "naive_mae": (sums["nab"].sum(0) / (W * T)).tolist(),
                   "seasonal24_mae": (sums["sab"].sum(0) / (W * T)).tolist(),
                   "r2": [float(1 - sq / (y2 - yy ** 2 / (W * T))) if (y2 - yy ** 2 / (W * T)) > 0 else None for sq, y2, yy in zip(sums["sq"].sum(0), sums["y2"].sum(0), sums["y"].sum(0))],
                   "corr_pred_true": [float((py - pp * yy / (W * T)) / math.sqrt(max(1e-300, (p2 - pp ** 2 / (W * T)) * (y2 - yy ** 2 / (W * T)))))
                                      for py, pp, yy, p2, y2 in zip(sums["py"].sum(0), sums["p"].sum(0), sums["y"].sum(0), sums["p2"].sum(0), sums["y2"].sum(0))]}
    per_channel["skill_mae_vs_naive"] = [1 - a / b if b > 0 else None for a, b in zip(per_channel["mae"], per_channel["naive_mae"])]
    def clean(o):
        """Undefined is None, never a number and never a JSON NaN token."""
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    return clean({"schema": "df_sota_metrics_vault.v2", "space": "normalized (train-standardized) units; author reduction = mean over windows x steps x channels",
            "population": {"windows": W, "steps": T, "channels": C, "elements": n},
            "estimators_and_limitations": {
                "global/per_step/per_channel errors": "exact float64 sums over the full population; MAPE/MSPE exclude |true| <= 1e-8 (population reported)",
                "naive": "persistence = last input value repeated; seasonal24 = the input value 24 h before each step (wraps within the 96-step window for steps > 24: an approximation past step 24)",
                "residual moments": "exact float64 power sums (mean, var, skew, kurtosis); kurtosis is raw (not excess)",
                "quantiles": "APPROXIMATE: read from a 2000-bin histogram on [-20, 20] (bin width 0.02); residuals outside the range are not binned "
                             "(counts sum vs population reported); not a full exact distribution",
                "entropy": "of the same clipped 2000-bin histogram, bits; depends on the bin width",
                "mutual_information": "APPROXIMATE: 64 x 64 joint histogram of clipped values on [-6, 6]; plug-in estimator, biased upward for small populations",
                "autocorrelation": "sample ACF along consecutive test windows (one hour apart) of the CHANNEL-MEAN residual per step (lags 1..max_lag) and of "
                                   "each channel's residual at the first and last steps (lags 1, 24, 168); no PACF, no spectral estimate; no confidence bands "
                                   "(the windows overlap by construction, so the usual 1/sqrt(W) band does not apply)",
                "r2/corr": "per channel and global, exact sums; None when the true variance is zero",
                "uncertainty": "none per estimator: one cell is one sample; seed dispersion is reported at the table level",
                "independent_check": "global MSE/MAE equal the author's metric() within 1e-6 and the float64 reduction (closure); per-step and per-channel "
                                     "means average back to the global values (tests)"},
            "global": {"mse": mse, "mae": mae, "rmse": math.sqrt(mse), "mape": mape / max(1, n_mape), "mspe": mspe / max(1, n_mape), "mape_population": n_mape,
                       "rse": math.sqrt(float(sums["sq"].sum()) / sst) if sst > 0 else None, "r2": 1 - float(sums["sq"].sum()) / sst if sst > 0 else None,
                       "corr_pred_true": cov / math.sqrt(vp * vy) if vp > 0 and vy > 0 else None, "bias": rmean, "true_mean": ymean, "pred_mean": pmean,
                       "true_var": vy / n, "pred_var": vp / n, "naive_mse": float(sums["nsq"].sum() / n), "naive_mae": float(sums["nab"].sum() / n),
                       "seasonal24_mse": float(sums["ssq"].sum() / n), "seasonal24_mae": float(sums["sab"].sum() / n),
                       "skill_mae_vs_naive": 1 - mae / float(sums["nab"].sum() / n), "skill_mse_vs_naive": 1 - mse / float(sums["nsq"].sum() / n),
                       "mase_vs_naive": mae / float(sums["nab"].sum() / n), "mutual_information_bits_pred_true_64x64": mi_bits},
            "residuals": {"mean": rmean, "var": rvar, "sd": math.sqrt(max(0.0, rvar)), "skewness": skew, "kurtosis": kurt, "entropy_bits_2000bins_[-20,20]": entropy_bits,
                          "quantiles": quantiles, "fraction_abs_gt": {k: v / n for k, v in tail.items()}, "histogram": {"edges": hist_edges.tolist(), "counts": hist.astype(int).tolist()}},
            "per_step": per_step, "per_channel": per_channel,
            "autocorrelation": {"reading": "residual autocorrelation along consecutive test windows (one hour apart)",
                                "channel_mean_residual_per_step": {"lags": list(range(1, lags + 1)), "acf": step_acf.T.tolist()},
                                "per_channel_first_step": first_acf, "per_channel_last_step": last_acf},
            "joint_histogram_pred_true": {"edges": jedges.tolist(), "counts": joint.astype(int).tolist()}})


def verify_sota_run(root: Path, *, warehouse=None, data_path: Path | None = None, replay: bool = True, replay_device: str = "cpu",
                    replay_units: list | None = None) -> dict:
    root = Path(root)
    design = json.loads((root / "DESIGN.json").read_text())
    validate(design)
    B = _module("df_benchmark_contract")
    receipts = (json.loads((root / "TERMINAL_RECEIPTS.json").read_text()) or {}).get("units") or {} if (root / "TERMINAL_RECEIPTS.json").is_file() else {}
    problems = []
    drift = code_drift(design)
    # preparation custody: BENCH_DATA anchored by the accepted prepare terminal (data + record)
    prep = {"class": "PREPARATION_LOCAL_ONLY", "why": "warehouse not read"}
    if not (root / "BENCH_DATA.npz").is_file() or not (root / "BENCH_DATA.json").is_file():
        prep = {"class": "PREPARATION_MISSING", "why": "no BENCH_DATA"}
    else:
        rec = json.loads((root / "BENCH_DATA.json").read_text())
        if rec.get("data_sha256") != sha_file(root / "BENCH_DATA.npz") or rec.get("design_sha256") != design["design_sha256"] or rec.get("file_sha256") != design["source_data"]["sha256"]:
            prep = {"class": "PREPARATION_CHANGED", "why": "BENCH_DATA differs from its record, its design or the sealed file digest"}
        elif warehouse is not None and "prepare" in receipts:
            row = ((warehouse(receipts["prepare"]["campaign_sha256"]) or {}).get("current") or {}).get("prepare") or {}
            acc = {x.get("role"): x.get("sha256") for x in row.get("artifacts") or []}
            if row.get("terminal_sha256") == receipts["prepare"].get("terminal_sha256") and row.get("status") == "COMPLETED" \
                    and acc.get("data") == sha_file(root / "BENCH_DATA.npz") and acc.get("record") == sha_file(root / "BENCH_DATA.json") \
                    and row.get("config_sha256") in (None, design["design_sha256"]):
                prep = {"class": "PREPARATION_ACCEPTED_ARTIFACT", "why": "the accepted prepare terminal carries the digests of BENCH_DATA and its record"}
            else:
                prep = {"class": "PREPARATION_NOT_ACCEPTED", "why": "the accepted prepare terminal does not anchor these bytes"}
    if prep["class"] not in ("PREPARATION_ACCEPTED_ARTIFACT", "PREPARATION_LOCAL_ONLY"):
        problems.append(f"preparation custody {prep['class']}: {prep['why']}")
    if data_path is not None and sha_file(data_path) != design["source_data"]["sha256"]:
        problems.append("the data file given for re-derivation is not the sealed official file")
    disp = B.disposition(design)
    rows, cache = [], {}
    for cell in design["cells"]:
        unit, folder, p_ = cell["cell_id"], root / "attempts" / cell["cell_id"], []
        if not (folder / "cell.json").is_file() or not (folder / "arrays.npz").is_file() or not (folder / "checkpoint.pth").is_file():
            rows.append({"unit": unit, "cell": cell, "verified": False, "status": "MISSING", "problems": [f"{unit}: a registered cell has no record, arrays or checkpoint — missing, not absent"]}); continue
        record = json.loads((folder / "cell.json").read_text())
        if record.get("design_sha256") != design["design_sha256"] or any(record["cell"].get(k) != cell[k] for k in ("cell_id", "horizon", "seed", "seq_len", "arm")):
            p_.append(f"{unit}: the record's identity contradicts the design cell")
        if record.get("effective_args") != cell["effective_args"] | {k: record["effective_args"].get(k) for k in ("root_path", "data_path", "checkpoints", "gpu", "use_gpu", "device_ids", "devices")}:
            diff = sorted(k for k in set(cell["effective_args"]) | set(record.get("effective_args") or {}) if k not in ("root_path", "data_path", "checkpoints", "gpu", "use_gpu", "device_ids", "devices")
                          and cell["effective_args"].get(k) != (record.get("effective_args") or {}).get(k))
            if diff:
                p_.append(f"{unit}: the effective arguments the cell ran with differ from the sealed ones: {diff}")
        if record.get("source_drift"):
            p_.append(f"{unit}: the cell ran with author files that differ from the sealed digests: {sorted(record['source_drift'])}")
        arrays_sha, ckpt_sha, record_sha = sha_file(folder / "arrays.npz"), sha_file(folder / "checkpoint.pth"), sha_file(folder / "cell.json")
        if arrays_sha != record.get("arrays_sha256"):
            p_.append(f"{unit}: the arrays on disk are not the record's arrays: CHANGED ARRAYS")
        if ckpt_sha != record.get("checkpoint_sha256"):
            p_.append(f"{unit}: the checkpoint bytes on disk are not the record's checkpoint: CHANGED CHECKPOINT")
        custody = {"class": "UNCHECKED"}
        receipt = receipts.get(unit)
        if receipt is None:
            p_.append(f"{unit}: no accepted terminal receipt")
        elif warehouse is not None:
            row = ((warehouse(receipt["campaign_sha256"]) or {}).get("current") or {}).get(unit)
            if not row:
                p_.append(f"{unit}: the warehouse holds NO terminal for this unit"); custody = {"class": "NO_ACCEPTED_TERMINAL"}
            else:
                acc = {x.get("role"): x.get("sha256") for x in row.get("artifacts") or []}
                if row.get("terminal_sha256") != receipt.get("terminal_sha256") or row.get("status") != "COMPLETED":
                    p_.append(f"{unit}: the accepted terminal digest/status disagrees with the receipt")
                if row.get("config_sha256") and row.get("config_sha256") != design["design_sha256"]:
                    p_.append(f"{unit}: reported under another configuration")
                tags = row.get("tags") or row.get("tags_json") or {}
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except Exception:
                        tags = {}
                for key, want in (("horizon", cell["horizon"]), ("seed", cell["seed"]), ("seq_len", cell["seq_len"]), ("arm", cell["arm"])):
                    if key in tags and str(tags[key]) != str(want):
                        p_.append(f"{unit}: the accepted terminal's {key} is {tags[key]!r}, the design says {want!r}: IDENTITY")
                if acc.get("predictions") == arrays_sha and acc.get("checkpoint") == ckpt_sha and acc.get("record") == record_sha and not p_:
                    custody = {"class": "ACCEPTED_ARTIFACT_CHAIN"}
                else:
                    for role, sha in (("predictions", arrays_sha), ("checkpoint", ckpt_sha), ("record", record_sha)):
                        if acc.get(role) != sha:
                            p_.append(f"{unit}: the {role} on disk is not the accepted {role} artifact")
                    custody = {"class": "UNANCHORED"}
        # metrics recomputed from the arrays; targets re-derived by the author's loader when the file is available
        with np.load(folder / "arrays.npz") as z:
            preds = z["pred"]
        if not np.isfinite(preds).all():
            p_.append(f"{unit}: non-finite predictions")
        derived = None
        if data_path is not None and not p_:
            key = (cell["seq_len"], cell["horizon"])
            if key not in cache:
                cache[key] = naive_and_trues(design, cell, data_path)
            derived = cache[key]
            if derived["true_sha256"] != record.get("true_sha256"):
                p_.append(f"{unit}: the targets re-derived from the delivered file by the author's loader are not the ones the cell scored: TARGETS")
            if list(preds.shape) != record["shapes"]["pred"] or preds.shape[0] != derived["windows"]:
                p_.append(f"{unit}: prediction shape {list(preds.shape)} is not the test population {derived['windows']} windows: POPULATION")
        recomputed = None
        if data_path is not None and not p_:
            author_env()
            MET = importlib.import_module("utils.metrics")
            # the author's function on the author's arrays (float32) must reproduce the record bitwise; float64 independently
            args = build_args(cell["argv"], data_dir=data_path.parent, data_name=data_path.name, checkpoints=Path("/nonexistent"), gpu=0, use_gpu=False)
            args.augmentation_ratio = 0
            DF = importlib.import_module("data_provider.data_factory")
            _, test_loader = DF.data_provider(args, "test")
            trues = np.concatenate([b[1][:, -args.pred_len:, :].float().numpy().astype(np.float32) for b in test_loader], axis=0)
            mae32, mse32 = MET.metric(preds, trues)[:2]
            recomputed = {"author_float32": {"mae": float(mae32), "mse": float(mse32)}, "independent_float64": float64_metrics(preds, trues)}
            del trues
            vault_path = folder / "METRICS_VAULT.json"
            existing = json.loads(vault_path.read_text()) if vault_path.is_file() else {}
            if existing.get("pred_sha256") != record.get("pred_sha256") or existing.get("schema") != "df_sota_metrics_vault.v2":
                _, test_loader = DF.data_provider(args, "test")
                vault = metrics_vault(preds, test_loader, pred_len=args.pred_len)
                vault.update(unit=unit, pred_sha256=record.get("pred_sha256"), design_sha256=design["design_sha256"], at=now_iso())
                vault_path.write_text(json.dumps(vault, default=str))
            recomputed["metrics_vault_sha256"] = sha_file(vault_path)
            if abs(float(mae32) - record["author_metric_float32"]["mae"]) > 0 or abs(float(mse32) - record["author_metric_float32"]["mse"]) > 0:
                p_.append(f"{unit}: the author's metric recomputed from the arrays is not the record's: METRIC")
            if abs(recomputed["independent_float64"]["mae"] - record["author_metric_float32"]["mae"]) > 1e-6 or abs(recomputed["independent_float64"]["mse"] - record["author_metric_float32"]["mse"]) > 1e-6:
                p_.append(f"{unit}: the independent float64 metric differs from the author's by more than 1e-6: REDUCTION")
        rows.append({"unit": unit, "cell": {k: cell[k] for k in ("cell_id", "arm", "protocol", "seq_len", "horizon", "seed")}, "custody": custody,
                     "author_metric_float32": record["author_metric_float32"], "recomputed": recomputed, "derived": derived,
                     "training": record.get("training"), "cost": record.get("cost"), "n_parameters": record.get("n_parameters"), "device": record.get("device"),
                     "verified": not p_ and custody["class"] == "ACCEPTED_ARTIFACT_CHAIN" and prep["class"] == "PREPARATION_ACCEPTED_ARTIFACT" and not problems and data_path is not None,
                     "problems": p_, "task_id": disp["task_id"], "disposition": disp["disposition"]})
    replays = {}
    if replay and data_path is not None:
        prior = json.loads((root / "REPLAYS.json").read_text()) if (root / "REPLAYS.json").is_file() else {}
        for r in rows:
            if r["problems"] or r.get("status") == "MISSING":
                continue
            unit = r["unit"]; folder = root / "attempts" / unit
            if replay_units is not None and unit not in replay_units:
                r["replay"] = {"skipped": True, "why": "REPLAY_PENDING: not replayed on this host (memory placement); the row stays UNVERIFIED until replayed"}
                r["verified"] = False
                continue
            ident = {"checkpoint_sha256": sha_file(folder / "checkpoint.pth"), "arrays_sha256": sha_file(folder / "arrays.npz"), "design_sha256": design["design_sha256"],
                     "replay_code_sha256": replay_code_sha256(), "author_files": source_digests(), "device": replay_device}
            cached = prior.get(unit) or {}
            if cached.get("identity") == ident and "allclose_rule" in cached:
                replays[unit] = {**cached, "adopted_from": "REPLAYS.json (identical checkpoint, arrays, design, author files and replay code, hashed now)"}
            else:
                replays[unit] = {**replay_cell(root, design, unit, data_path=data_path, device=replay_device), "identity": ident}
            rep = replays[unit]
            if not rep.get("allclose_rule"):
                r["problems"].append(f"{unit}: fresh-process reload through the author's test() exceeds the frozen replay rule ({rep.get('error') or rep.get('max_abs_prediction_difference')})")
            elif rep.get("true_sha256_replayed") != r["derived"]["true_sha256"]:
                r["problems"].append(f"{unit}: the replay's targets differ from the derived targets")
            elif abs(rep["replayed_author_metric"]["mae"] - r["author_metric_float32"]["mae"]) > 1e-5 or abs(rep["replayed_author_metric"]["mse"] - r["author_metric_float32"]["mse"]) > 1e-5:
                r["problems"].append(f"{unit}: the metric of the replayed predictions differs from the stored one by more than 1e-5")
            r["replay"] = {k: v for k, v in rep.items() if k != "identity"}
            r["verified"] = r["verified"] and not r["problems"]
        (root / "REPLAYS.json").write_text(json.dumps({**prior, **replays}, indent=1, default=str))
    elif replay:
        for r in rows:
            if not r["problems"] and r.get("status") != "MISSING":
                r["replay"] = {"skipped": True, "why": "REPLAY_PENDING: no data file on this host"}; r["verified"] = False
    return {"design_sha256": design["design_sha256"], "disposition": disp, "preparation_custody": prep, "source_drift_now": drift, "rows": rows,
            "replay_patch": {"what": "torch.load defaults to map_location=cpu inside the CPU replay process", "why": "the author's test(test=1) loads the checkpoint "
                             "without map_location; a CUDA-trained checkpoint is otherwise unreadable on CPU", "effect": "tensor placement only; the frozen replay rule "
                             "(CPU, atol/rtol 1e-4, metric within 1e-5) is unchanged"},
            "problems": problems + [q for r in rows for q in r["problems"]] + ([f"author files drifted from the sealed digests: {sorted(drift)}"] if drift else []),
            "verified_units": sorted(r["unit"] for r in rows if r["verified"]), "unverified_units": sorted(r["unit"] for r in rows if not r["verified"])}


def rows_for_table(root: Path, ver: dict, design: dict, *, label: str) -> list:
    """The verification in the closure table's row schema (tools/df_closure_table): published normalized metrics FIRST
    (MSE in `published_metric_value`, MAE as the model error), naive persistence on identical windows, one authority."""
    pub = design["lock"]["published"]["per_horizon"]
    out = []
    for r in ver["rows"]:
        c = r["cell"]; h = str(c["horizon"])
        m = r.get("author_metric_float32") or {}
        nv = (r.get("derived") or {}).get("naive") or {}
        mae, naive_mae = m.get("mae"), nv.get("mae")
        skill = (None if mae is None or naive_mae in (None, 0) else 1 - mae / naive_mae)
        out.append({"run": label, "unit": r["unit"], "role": "forecast", "arm": c["arm"], "seed": str(c["seed"]), "task_id": r["task_id"], "disposition": r["disposition"],
                    "task_horizon_split": f"{r['task_id']} | h={c['horizon']} steps ({c['horizon']*3600} s) | L={c['seq_len']} | chronological 7/1/2, test split",
                    "metric_and_scale": "MSE and MAE over every test window x step x channel in the normalized (train-standardized) space; published metric first",
                    "published_metric_value": m.get("mse"), "published_metric": "mse", "model_mse_z": m.get("mse"), "naive_mse_z": nv.get("mse"),
                    "model_error": mae, "model_error_z": mae, "naive_error": naive_mae, "naive_error_z": naive_mae,
                    "naive_definition": (r.get("derived") or {}).get("naive_definition", "persistence on identical windows (not derived: no data file)"),
                    "model_population": (r.get("derived") or {}).get("n_elements"), "naive_population": (r.get("derived") or {}).get("n_elements"),
                    "model_horizon": c["horizon"], "naive_horizon": c["horizon"], "model_scale": "z_train", "naive_scale": "z_train",
                    "target": "all 321 channels", "n_evaluated": (r.get("derived") or {}).get("windows"), "horizon_steps": c["horizon"],
                    "skill_vs_naive": {"value": skill, "status": "DEFINED" if skill is not None else "UNDEFINED", "formula": "1 - MAE_model/MAE_naive"},
                    "literature_value_and_source": {"status": "REPRODUCTION", "comparator_state": "PLANNED_REFERENCE" if not r["verified"] else "VERIFIED_COMPARATOR",
                                                    "source": PAPER["citation"], "published_value": pub.get(h),
                                                    "published_or_reproduced": "PUBLISHED value; this row IS the reproduction under the author protocol",
                                                    "placed_in_comparison_column": False, "why_not": "the reproduction is compared by the frozen agreement rule in SOTA_TABLE, not by a column of published numbers",
                                                    "planned_matched_comparison": "SOTA_TABLE.json: per-horizon agreement under the frozen rule"},
                    "comparability_status": "REPRODUCTION", "scope": design.get("purpose"),
                    "binding": {"level": "TERMINAL_ARTIFACT" if (r.get("custody") or {}).get("class") == "ACCEPTED_ARTIFACT_CHAIN" else "NOT_BOUND"},
                    "warehouse": {"checked": (r.get("custody") or {}).get("class") not in (None, "UNCHECKED")}, "custody": r.get("custody") or {"class": "UNCHECKED"},
                    "record_score_checked": r.get("recomputed") is not None, "problems": list(r["problems"]), "verified": bool(r["verified"]),
                    "preserved_with_qualified_scope": False, "precision_note": "author float32 metric; float64 recomputation beside it in REPORT.json",
                    "replay": r.get("replay"), "training": r.get("training"), "cost": r.get("cost")})
    return out


def agreement(published: dict, values: list, metric: str, rule: dict = AGREEMENT) -> dict:
    if not values:
        return {"status": "NO_MEASUREMENT", "mean": None, "sd_ddof1": None, "difference": None}
    mean = float(np.mean(values)); sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
    p = float(published[metric]); diff = mean - p
    s = rule["std_paper"][metric]; tol_a = rule["k_agree"] * s + rule["rounding"]; tol_p = rule["k_partial"] * s + rule["rounding"]
    status = "NUMERICAL_AGREEMENT" if abs(diff) <= tol_a else ("PARTIAL_AGREEMENT" if abs(diff) <= tol_p else "DISAGREEMENT")
    return {"status": status, "mean": mean, "sd_ddof1": sd, "values": [float(v) for v in values], "published": p, "difference": diff,
            "tolerance_agree": tol_a, "tolerance_partial": tol_p, "n_seeds": len(values)}


def table(design: dict, ver: dict) -> dict:
    """RP97: dataset/protocol, model/revision, horizon, published, replicated, difference, matched naive, seed dispersion, training
    completion, cost, agreement — normalized official metrics first; nothing unverified enters a mean."""
    pub = design["lock"]["published"]
    rows, avg_pool = [], {"mse": [], "mae": []}
    complete = True
    for h in design["horizons"]:
        cells = [r for r in ver["rows"] if r["cell"]["horizon"] == h]
        ok = [r for r in cells if r["verified"]]
        expected = [c["cell_id"] for c in design["cells"] if c["horizon"] == h]
        row = {"dataset_protocol": f"ECL official processed (TSL), L={design['seq_len']}, T={h}, split 7/1/2, normalized space", "model_revision": f"TimeFilter @ {PINNED_COMMIT[:12]}",
               "horizon": h, "published": pub["per_horizon"][str(h)], "seeds_expected": expected, "seeds_verified": [r["unit"] for r in ok],
               "verified_all": len(ok) == len(expected) and bool(expected)}
        for m in ("mse", "mae"):
            vals = [r["author_metric_float32"][m] for r in ok]
            row[m] = agreement(pub["per_horizon"][str(h)], vals, m)
            if row["verified_all"]:
                avg_pool[m].append(row[m]["mean"])
        complete = complete and row["verified_all"]
        row["matched_naive"] = (ok[0]["derived"]["naive"] if ok and ok[0].get("derived") else None)
        row["training_completion"] = [{"unit": r["unit"], "epochs_run": (r.get("training") or {}).get("epochs_run"), "best_epoch": (r.get("training") or {}).get("best_epoch_by_vali"),
                                       "early_stopped": (r.get("training") or {}).get("early_stopped"), "max_epochs": (design["cells"][0]["effective_args"].get("train_epochs"))} for r in ok]
        row["cost"] = {"wall_seconds_sum": sum((r.get("cost") or {}).get("wall_seconds") or 0 for r in ok), "cpu_seconds_sum": sum((r.get("cost") or {}).get("cpu_seconds") or 0 for r in ok),
                       "peak_gpu_bytes_max": max([((r.get("cost") or {}).get("peak_gpu_allocated_bytes") or 0) for r in ok] or [0]), "hosts": sorted({(r.get("cost") or {}).get("host") for r in ok})}
        row["replay_max_abs_difference"] = max([((r.get("replay") or {}).get("max_abs_prediction_difference") or 0.0) for r in ok] or [0.0]) if ok else None
        row["problems"] = [q for r in cells for q in r["problems"]]
        rows.append(row)
    average = {m: (agreement(pub["average"], avg_pool[m], m) if complete and len(avg_pool[m]) == len(design["horizons"]) else
                   {"status": "NOT_COMPUTED", "why": "not every horizon is verified with all seeds; no average is claimed"}) for m in ("mse", "mae")}
    fidelity = {"source_hashes": "sealed == now" if not ver["source_drift_now"] else f"DRIFT {sorted(ver['source_drift_now'])}",
                "preparation": ver["preparation_custody"]["class"], "environment": "declared divergence from the author's (torch/numpy/pandas/sklearn/GPU differ; recorded in the lock)",
                "operational_patches": [p["what"] for p in design["lock"]["operational_patches"]],
                "verdict": ("FAITHFUL_WITH_DECLARED_ENVIRONMENT_DIVERGENCE" if not ver["source_drift_now"] and ver["preparation_custody"]["class"] == "PREPARATION_ACCEPTED_ARTIFACT" and not ver["problems"]
                            else "NOT_ESTABLISHED: " + "; ".join(ver["problems"][:3]))}
    return {"schema": "df_sota_table.v1", "at": now_iso(), "design_sha256": design["design_sha256"], "paper": PAPER["citation"], "protocol": design["protocol"],
            "published_table": pub["table"], "rows": rows, "average_over_horizons": average, "complete": complete, "protocol_fidelity": fidelity,
            "agreement_rule": AGREEMENT["rule"], "disposition": ver["disposition"], "unexecuted": [c["cell_id"] for c in design["cells"] if c["cell_id"] not in ver["verified_units"]],
            "verified_units": ver["verified_units"], "problems": ver["problems"]}


def markdown(t: dict) -> str:
    lines = [f"# SOTA reproduction table — {t['paper'][:60]}…", "", f"Protocol {t['protocol']} ({t['published_table']}); design {t['design_sha256'][:12]}; "
             f"protocol fidelity: **{t['protocol_fidelity']['verdict']}**; complete: {t['complete']}", "",
             "| dataset / protocol | model @ revision | T | published MSE / MAE | replicated mean MSE / MAE (sd, n) | difference | matched naive MSE / MAE | training | cost | replay max|Δ| | agreement |",
             "|---|---|---:|---|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        f = lambda m: (f"{r[m]['mean']:.4f}" if r[m].get("mean") is not None else "—")
        sd = lambda m: (f"{r[m]['sd_ddof1']:.4f}" if r[m].get("sd_ddof1") is not None else "—")
        dd = lambda m: (f"{r[m]['difference']:+.4f}" if r[m].get("difference") is not None else "—")
        nv = r.get("matched_naive") or {}
        tr = "; ".join(f"{x['unit'].rsplit('_s', 1)[-1]}:{x['epochs_run']}ep best{x['best_epoch']}{'*' if x['early_stopped'] else ''}" for x in r["training_completion"]) or "—"
        lines.append(f"| {r['dataset_protocol']} | {r['model_revision']} | {r['horizon']} | {r['published']['mse']:.3f} / {r['published']['mae']:.3f} | "
                     f"{f('mse')} / {f('mae')} (sd {sd('mse')} / {sd('mae')}, n={r['mse'].get('n_seeds', 0)}) | {dd('mse')} / {dd('mae')} | "
                     f"{nv.get('mse', float('nan')):.4f} / {nv.get('mae', float('nan')):.4f} | {tr} | {r['cost']['wall_seconds_sum']/3600:.2f} h wall, {r['cost']['peak_gpu_bytes_max']/2**30:.1f} GiB peak | "
                     f"{(r['replay_max_abs_difference'] if r['replay_max_abs_difference'] is not None else float('nan')):.2e} | {r['mse']['status']} / {r['mae']['status']} |")
    a = t["average_over_horizons"]
    lines += ["", f"Average over the four horizons: MSE {a['mse'].get('mean') if a['mse'].get('mean') is None else round(a['mse']['mean'], 4)} vs published {t['rows'][0]['published'] and ''}"
              f"{'' if a['mse'].get('published') is None else a['mse']['published']} ({a['mse']['status']}); MAE {a['mae'].get('mean') if a['mae'].get('mean') is None else round(a['mae']['mean'], 4)} "
              f"vs {'' if a['mae'].get('published') is None else a['mae']['published']} ({a['mae']['status']}).",
              "", f"Unexecuted or unverified cells: {t['unexecuted'] or 'none'}.", f"Agreement rule (frozen before any test score): {t['agreement_rule']}",
              "", "* = early stopped. Metrics are the author's `metric()` on the concatenated float32 test predictions in the normalized space (no inversion); "
              "the paper prints three decimals. Training completion per seed: epochs run and the checkpointed epoch. Naive = persistence on identical windows."]
    return "\n".join(lines) + "\n"


def close(a, design: dict) -> dict:
    validate(design)
    root = Path(a.root)
    C = _module("df_mod_e0_close")
    token = Path(a.warehouse_token_file).read_text().strip().strip('"').strip("'") if getattr(a, "warehouse_token_file", None) else None
    warehouse = (lambda campaign: C.warehouse_terminals(a.warehouse_url, token, campaign)) if token else None
    data_path = Path(a.data_path) if getattr(a, "data_path", None) else (delivered_file(root, design, None) if (root / "DELIVERIES.json").is_file() else None)
    ver = verify_sota_run(root, warehouse=warehouse, data_path=data_path, replay=not getattr(a, "skip_replay", False), replay_device=getattr(a, "replay_device", "cpu"),
                          replay_units=getattr(a, "replay_units", None))
    if warehouse is None:
        ver["problems"].append("closure without a warehouse read: no accepted custody, nothing is verified")
        for r in ver["rows"]:
            r["verified"] = False
        ver["verified_units"] = []
    t = table(design, ver)
    report = {"schema": "df_sota_report.v1", "design_sha256": design["design_sha256"], "verification": ver, "table": t,
              "verified": bool(t["complete"] and not ver["problems"]), "problems": ver["problems"]}
    (root / "REPORT.json").write_text(json.dumps(report, indent=1, default=str))
    (root / "SOTA_TABLE.json").write_text(json.dumps(t, indent=1, default=str))
    (root / "SOTA_TABLE.md").write_text(markdown(t))
    print(json.dumps({"verified": report["verified"], "complete": t["complete"], "verified_units": ver["verified_units"], "problems": ver["problems"][:5],
                      "agreement": {str(r["horizon"]): (r["mse"]["status"], r["mae"]["status"]) for r in t["rows"]}}, indent=1))
    return report


# --- CLI ---------------------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["seal", "prepare", "preflight", "execute", "child", "close", "lock", "merge"])
    ap.add_argument("--from", dest="source", type=Path, default=None, help="merge: a worker's copy of this root")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--unit"); ap.add_argument("--run-id")
    ap.add_argument("--seq-len", type=int, default=96); ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--horizons", type=int, nargs="*", default=None); ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=20); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gov-url", default="http://127.0.0.1:5055"); ap.add_argument("--api-key-file", type=Path)
    ap.add_argument("--lake", default=LAKE); ap.add_argument("--resource", default=RESOURCE)
    ap.add_argument("--warehouse-url", default="http://127.0.0.1:5057"); ap.add_argument("--warehouse-token-file", type=Path)
    ap.add_argument("--data-path", type=Path, default=None); ap.add_argument("--skip-replay", action="store_true"); ap.add_argument("--replay-device", default="cpu")
    ap.add_argument("--replay-units", nargs="*", default=None, help="close: replay only these units on this host; the others stay UNVERIFIED (REPLAY_PENDING)")
    a = ap.parse_args(argv)
    if a.command == "seal":
        a.root.mkdir(parents=True, exist_ok=True)
        if (a.root / "DESIGN.json").is_file():
            raise SotaRefusal("REFUSED: this root already holds a sealed design; a design is never written over")
        design = seal(seq_len=a.seq_len, seeds=tuple(a.seeds or (2021, 2022, 2023)), horizons=tuple(a.horizons or (96, 192, 336, 720)))
        (a.root / "DESIGN.json").write_text(json.dumps(design, indent=1, default=str))
        print(json.dumps({"design_sha256": design["design_sha256"], "cells": len(design["cells"]), "author_head": design["lock"]["source"]["git"]["head"]}))
        return 0
    design = json.loads((a.root / "DESIGN.json").read_text())
    if a.command == "lock":
        print(json.dumps(design["lock"], indent=1, default=str)); return 0
    if a.command == "merge":
        out = merge(a.root, a.source); print(json.dumps(out, indent=1)); return 0 if not out["problems"] else 1
    if a.command == "prepare":
        run_prepare(a, design); return 0
    if a.command == "preflight":
        doc = run_preflight(a, design); print(json.dumps(doc["allocation"], indent=1)); return 0
    if a.command == "child":
        child(a, design); return 0
    if a.command == "execute":
        out = execute(a, design); return 0 if all(o["ok"] for o in out) else 1
    report = close(a, design)
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
