#!/usr/bin/env python3
"""RP142: the matched ECL adapter for the modular intervention, against the accepted protocol-A reference.

This is NOT the household E1 pilot with another dataset name. That pilot predicts ONE channel at ONE offset; the reference
evaluates `horizon x 321` channels on the author's own windows. Everything below is enumerated so the contrast is matched
before anything is fitted.

Dataset and task identity
  Resource      the official processed ECL benchmark (TSL), 321 client load channels, hourly.
  Channel roles all 321 channels carry the same role (client load) and are BOTH inputs and targets: the author's
                multivariate setting (`features='M'`). Channel ORDER is the author's CSV column order after dropping
                'date', and is never re-sorted by us: order identity is asserted against the reference's own loader.
  Split         the author's 7/1/2 boundaries, computed by his `Dataset_Custom`, not re-derived here.
  Scaler        the author's StandardScaler, fitted on the TRAIN rows only, applied by his loader.
  Window        seq_len 96, label_len 48, pred_len in {96, 192, 336, 720}: the accepted protocol-A geometry.
  Target        y[b, t, c] for t in 0..pred_len-1 over all 321 channels, in the author's normalized space, taken from the
                author's loader so that windows, steps and channels are his by construction.
  Reduction     the author's float32 MSE/MAE over windows x steps x channels, reused from df_sota_repro.

Tensor shapes, module by module (B = batch, W = 96, P = 321, H = pred_len)
  input x                      [B, W, P]
  branch select (per group g)  [B, W, |g|]
  detector g{g}_det1/_det2     [B, W, 16]      2 residual dilated causal Conv1D blocks, kernel 3, dilation 1, elu
  integrator g{g}_int*         [B, W, 16]      dilations (2, 4, 8, 16), same block, receptive field 65 <= W
  adapter g{g}_adapt           [B, W, 16]
  fusion (sequence)            [B, W, 16*G]
  core_conv                    [B, W, 16]      Conv1D 16, kernel 3, causal, elu
  core_last                    [B, 16]
  head                         [B, H*P]        Dense, the INCREMENT over the last observation
  reshape                      [B, H, P]
  persistence skip             [B, H, P]       x[:, -1, :] broadcast over H and added: the approved head rule, generalised
                                               from one step to H steps and changed in no other way

Regimes
  R0/R1/R2 are the approved ones and are applied by df_e1_regimes to this model: the detector layers are the same
  `g{g}_det*` names, so the pre-trained weights transfer by name and the freeze/update proofs work unchanged.

Causality
  The auto-encoder consumes outer TRAIN windows only. Its internal validation is a chronological tail of the TRAIN origins,
  purged from its training origins by at least W + H. The outer validation may select the downstream checkpoint and nothing
  else; the outer test selects nothing and is never read during any fit. `causality_report` proves this over row identities
  and by future perturbation, before a fit is allowed to start.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SEQ_LEN = 96
LABEL_LEN = 48
CHANNELS = 321
FUSION = "sequence"
ARCH = "B"
DEFAULT_HORIZONS = (96, 192, 336, 720)


def _load(name: str):
    import sys
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sota():
    return _load("df_sota_repro")


def _e0():
    return _load("df_mod_e0")


def _regimes():
    return _load("df_e1_regimes")


# --- the author's data, used as the author's ---------------------------------------------------------------------------------

def author_datasets(data_path: Path, *, pred_len: int, seq_len: int = SEQ_LEN, label_len: int = LABEL_LEN) -> dict:
    """The author's own Dataset objects for the three splits, built through his data_factory with the sealed arguments. The
    split boundaries, the scaler fit population, the window construction and the channel order are therefore his."""
    S = _sota()
    S.author_env()
    # the arguments are the REFERENCE's own, taken from a sealed protocol-A cell of this horizon rather than hand-written here:
    # a matched task cannot resolve its split, scaler or window geometry differently from the reference it is matched to
    design = S.seal(seq_len=seq_len, seeds=(2021,), horizons=(pred_len,), protocol="A")
    cell = design["cells"][0]
    args = S.build_args(cell["argv"], data_dir=Path(data_path).parent, data_name=Path(data_path).name,
                        checkpoints=Path("/nonexistent"), gpu=0, use_gpu=False)
    assert int(args.pred_len) == int(pred_len) and int(args.seq_len) == int(seq_len)
    args.augmentation_ratio = 0
    DF = importlib.import_module("data_provider.data_factory")
    out = {"args": args, "splits": {}}
    for flag in ("train", "val", "test"):
        ds, loader = DF.data_provider(args, flag)
        out["splits"][flag] = {"dataset": ds, "loader": loader, "n_windows": len(ds)}
    return out


def channel_order_digest(data_path: Path) -> dict:
    """The identity of the channel ordering we consume: the author's column order after 'date', hashed. A matched task must
    reproduce this exactly; a re-sorted or subset column list is a different task."""
    import pandas as pd
    df = pd.read_csv(data_path, nrows=1)
    cols = [c for c in df.columns if c != "date"]
    return {"n_channels": len(cols), "first": cols[:3], "last": cols[-3:],
            "sha256": hashlib.sha256("\\n".join(cols).encode()).hexdigest()}


def target_identity(data_path: Path, *, pred_len: int, n_windows: int | None = None) -> dict:
    """RP142: the targets this adapter will learn against, generated INDEPENDENTLY of the reference's stored arrays and then
    compared with the reference's own. Same windows, same steps, same channels, same normalized space."""
    S = _sota()
    d = author_datasets(data_path, pred_len=pred_len)
    ds = d["splits"]["test"]["dataset"]
    total = len(ds)
    take = total if n_windows is None else min(total, int(n_windows))
    h = hashlib.sha256()
    shapes = set()
    for i in range(take):
        _x, y, _xm, _ym = ds[i]
        y = np.asarray(y, dtype=np.float32)[-pred_len:, :]
        shapes.add(y.shape)
        h.update(np.ascontiguousarray(y).tobytes())
    return {"pred_len": pred_len, "test_windows_total": total, "windows_hashed": take,
            "target_shape_per_window": sorted(shapes)[0] if shapes else None,
            "targets_sha256": h.hexdigest(),
            "scope": "targets read from the author's own test Dataset in his order; the digest covers the first `windows_hashed` windows"}


# --- the model: the approved modular architecture with a full [B, H, P] readout -----------------------------------------------

def build_ecl_modular(assignment: list, *, pred_len: int, seq_len: int = SEQ_LEN, seed: int = 1, arch: str = ARCH,
                      fusion: str = FUSION):
    """The approved modular model, with the ONE change the matched task requires: the readout emits `pred_len x channels`
    instead of one step, and the persistence skip broadcasts the last observation over the horizon. Detector, integrator,
    adapter, fusion and core are the approved ones, with the approved layer names, so the regimes apply unchanged."""
    E = _e0()
    tf = E._tf()
    p = len(assignment)
    tf.keras.utils.set_random_seed(int(seed))
    inp = tf.keras.Input(shape=(seq_len, p), name="x")
    groups = sorted(set(assignment))
    branches = []
    for g in groups:
        idx = [k for k in range(p) if assignment[k] == g]
        sub = tf.keras.layers.Lambda(lambda t, idx=idx: tf.gather(t, idx, axis=2), name=f"g{g}_select")(inp)
        branches.append(E.branch_extractor(tf, sub, f"g{g}", arch))
    if fusion != "sequence":
        raise ValueError("the matched ECL adapter is declared with the sequence fusion")
    joint = tf.keras.layers.Concatenate(axis=2, name="fusion_seq")(branches) if len(branches) > 1 else branches[0]
    core = tf.keras.layers.Conv1D(16, 3, padding="causal", activation=E.ACTIVATION, name="core_conv")(joint)
    read = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="core_last")(core)
    flat = tf.keras.layers.Dense(int(pred_len) * p, name="head")(read)
    delta = tf.keras.layers.Reshape((int(pred_len), p), name="head_reshape")(flat)
    last = tf.keras.layers.Lambda(lambda t: tf.repeat(t[:, -1:, :], int(pred_len), axis=1), name="last_observation")(inp)
    out = tf.keras.layers.Add(name="persistence_skip")([last, delta])
    return tf.keras.Model(inp, out, name=f"ecl_modular_{arch}_{fusion}_h{pred_len}")


def model_shape_report(model, *, pred_len: int, seq_len: int = SEQ_LEN, channels: int = CHANNELS) -> dict:
    """Every declared shape, read from the built graph rather than from the document."""
    E = _e0()
    layers = {l.name: tuple(l.output.shape) for l in model.layers}
    return {"input": layers.get("x"), "output": tuple(model.output.shape),
            "output_is_full_horizon_by_channels": tuple(model.output.shape)[1:] == (pred_len, channels),
            "detector_layers": _regimes().detector_layer_names(model),
            "receptive_field_samples": E.RECEPTIVE_FIELD,
            "receptive_field_within_window": E.RECEPTIVE_FIELD <= seq_len,
            "params": E.count_params(model),
            "layers": layers}


# --- causality: proved over row identities, not asserted ----------------------------------------------------------------------

def split_origins(data_path: Path, *, pred_len: int, seq_len: int = SEQ_LEN) -> dict:
    """The row support of each split's windows in the author's own indexing. A window at origin `o` reads rows
    [o, o + seq_len) and predicts rows [o + seq_len, o + seq_len + pred_len).

    NOTE, declared rather than hidden: in the author's protocol the val and test splits begin `seq_len` rows BEFORE their own
    boundary, so their windows' INPUT CONTEXT deliberately reaches back into the preceding split. That is the published
    protocol and this adapter does not change it. The causal statement that matters is therefore made over TARGET support,
    which is what a fit can leak through, and that is what the checks below use."""
    import pandas as pd
    d = author_datasets(data_path, pred_len=pred_len, seq_len=seq_len)
    # the author does not keep border1 on the object, so it is RE-DERIVED here with his own formulas and then checked against
    # the length of each split's own array: an independent derivation that must agree with his objects, not a guess
    n_rows = int(pd.read_csv(data_path, usecols=[0]).shape[0])
    num_train, num_test = int(n_rows * 0.7), int(n_rows * 0.2)
    num_vali = n_rows - num_train - num_test
    border1s = {"train": 0, "val": num_train - seq_len, "test": n_rows - num_test - seq_len}
    border2s = {"train": num_train, "val": num_train + num_vali, "test": n_rows}
    out = {}
    for flag, s_ in d["splits"].items():
        ds = s_["dataset"]
        n = len(ds)
        b1, b2 = border1s[flag], border2s[flag]
        rows_in_split = len(getattr(ds, "data_x", []))
        if rows_in_split != (b2 - b1):
            raise ValueError(f"the re-derived {flag} boundary {b1}:{b2} does not match the author's own slice of {rows_in_split} rows")
        out[flag] = {"n_windows": n, "rows": [b1, b2], "first_input_row": b1, "origins": list(range(b1, b1 + n)),
                     "first_target_row": b1 + seq_len, "last_target_row": b1 + n - 1 + seq_len + pred_len - 1,
                     "boundary_source": "re-derived with the author's 7/2 formulas and checked against his own split array length"}
    return out


def _target_rows(origins, *, seq_len: int, pred_len: int) -> set:
    """Every raw row any of these windows PREDICTS."""
    rows = set()
    for o in origins:
        rows.update(range(o + seq_len, o + seq_len + pred_len))
    return rows


def ae_partition(train_origins: list, *, seq_len: int, pred_len: int, internal_validation_fraction: float = 0.2) -> dict:
    """RP142/RP36: the auto-encoder's own split is a chronological tail of the TRAIN origins, purged by seq_len + pred_len.
    The outer validation split is never touched here."""
    tr = np.asarray(sorted(train_origins), dtype=np.int64)
    cut = max(1, int(round(tr.size * (1.0 - float(internal_validation_fraction)))))
    purge = int(seq_len) + int(pred_len)
    ae_train = tr[:max(1, cut - purge)]
    ae_val = tr[cut:]
    return {"ae_train_origins": ae_train.tolist(), "ae_validation_origins": ae_val.tolist(), "purge": purge,
            "internal_validation_fraction": float(internal_validation_fraction),
            "rule": "a chronological tail of the outer TRAIN origins, separated from the AE's training origins by seq_len + pred_len"}


def causality_report(data_path: Path, *, pred_len: int, seq_len: int = SEQ_LEN,
                     internal_validation_fraction: float = 0.2, perturb_windows: int = 3) -> dict:
    """The proofs RP142 requires, executed over row identities: membership, disjointness of TARGET support, the purge, and that
    a window's tensors do not move when rows beyond its own support are perturbed. Absence of a label is never accepted as
    evidence of TRAIN membership."""
    origins = split_origins(data_path, pred_len=pred_len, seq_len=seq_len)
    part = ae_partition(origins["train"]["origins"], seq_len=seq_len, pred_len=pred_len,
                        internal_validation_fraction=internal_validation_fraction)
    tr, va, te = (set(origins[k]["origins"]) for k in ("train", "val", "test"))
    ae_tr, ae_va = set(part["ae_train_origins"]), set(part["ae_validation_origins"])
    T = lambda o: _target_rows(o, seq_len=seq_len, pred_len=pred_len)
    t_tr, t_va, t_te = T(tr), T(va), T(te)
    t_ae_tr, t_ae_va = T(ae_tr), T(ae_va)
    checks = {
        "ae_validation_inside_outer_train": bool(ae_va) and ae_va <= tr,
        "ae_validation_targets_inside_outer_train_targets": bool(t_ae_va) and t_ae_va <= t_tr,
        "ae_validation_disjoint_from_outer_validation": not (ae_va & va) and not (t_ae_va & t_va),
        "ae_validation_disjoint_from_outer_test": not (ae_va & te) and not (t_ae_va & t_te),
        "ae_training_disjoint_from_ae_validation": bool(ae_tr) and not (ae_tr & ae_va) and not (t_ae_tr & t_ae_va),
        "ae_purge_respected": (min(ae_va) - max(ae_tr)) >= part["purge"] if (ae_tr and ae_va) else False,
        "outer_target_support_disjoint": not (t_tr & t_va) and not (t_tr & t_te) and not (t_va & t_te),
    }
    checks["future_perturbation_leaves_windows_unchanged"] = _future_perturbation_check(
        data_path, pred_len=pred_len, seq_len=seq_len, n=perturb_windows)
    out = {"schema": "df_ecl_modular_causality.v2", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "pred_len": pred_len, "seq_len": seq_len,
           "origins": {k: {kk: vv for kk, vv in v.items() if kk != "origins"} for k, v in origins.items()},
           "ae_partition": {k: v for k, v in part.items() if not k.endswith("_origins")},
           "ae_train_n": len(ae_tr), "ae_validation_n": len(ae_va), "checks": checks,
           "pass": all(bool(v) for v in checks.values()),
           "authors_input_context_overlap": ("declared: the author's val and test splits start seq_len rows before their own "
                                             "boundary, so their INPUT context reaches into the preceding split. That is his "
                                             "published protocol; the disjointness proved here is of TARGET support, which is "
                                             "what a fit could leak through"),
           "selection_rule": ("the outer validation may select the downstream forecasting checkpoint and nothing else; the outer "
                              "test selects nothing and is read by no fit")}
    return out


def _future_perturbation_check(data_path: Path, *, pred_len: int, seq_len: int, n: int = 3) -> bool:
    """A window must not move when rows after its own support change. Executed on the author's Dataset by perturbing the
    scaled array in place beyond each window's last target row."""
    d = author_datasets(data_path, pred_len=pred_len, seq_len=seq_len)
    ds = d["splits"]["train"]["dataset"]
    data = getattr(ds, "data_x", None)
    if data is None or len(ds) <= n + 2:
        return False
    rng = np.random.default_rng(0)
    for i in range(min(n, len(ds) - 1)):
        x0, y0, _a, _b = ds[i]
        end = i + seq_len + pred_len
        if end >= len(data):
            continue
        keep = np.array(data[end:], copy=True)
        data[end:] = data[end:] + rng.normal(size=np.shape(data[end:])).astype(data.dtype) * 10.0
        x1, y1, _c, _d = ds[i]
        data[end:] = keep
        if not (np.array_equal(np.asarray(x0), np.asarray(x1)) and np.array_equal(np.asarray(y0), np.asarray(y1))):
            return False
    return True


# --- the sealed design of the contrast ----------------------------------------------------------------------------------------

def seal_contrast(data_path: Path, *, pred_len: int, seeds=(2021, 2022, 2023), regimes=("R0", "R1", "R2"),
                  internal_validation_fraction: float = 0.2, max_updates: int | None = None) -> dict:
    """The factorial, its budgets and every resolved argument, frozen BEFORE any outer-validation model selection."""
    E = _e0()
    assignment = [0] * CHANNELS
    model = build_ecl_modular(assignment, pred_len=pred_len, seed=int(seeds[0]))
    shapes = model_shape_report(model, pred_len=pred_len)
    design = {
        "schema": "df_ecl_modular_contrast.v1", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference": {"model": "TimeFilter", "revision": _sota().PINNED_COMMIT, "protocol": "A (Table 8, L=96)",
                      "status": "the contrast is matched to the ACCEPTED protocol-A reference; protocol B is not a dependency of it"},
        "task": {"dataset": "official processed ECL (TSL), 321 channels, hourly", "features": "M",
                 "seq_len": SEQ_LEN, "label_len": LABEL_LEN, "pred_len": int(pred_len), "channels": CHANNELS,
                 "target": "all 321 channels over the full horizon, in the author's normalized space",
                 "split": "the author's 7/1/2 boundaries", "scaler": "the author's StandardScaler fitted on TRAIN rows only",
                 "reduction": "the author's float32 MSE/MAE over windows x steps x channels"},
        "channel_order": channel_order_digest(data_path),
        "architecture": {"family": "the approved modular model", "arch": ARCH, "fusion": FUSION,
                         "grouping": "one branch over all 321 channels; grouping is held constant across the regimes",
                         "detector": "2 residual dilated causal Conv1D blocks, 16 filters, kernel 3, dilation 1, elu",
                         "integrator_dilations": list(E.INTEGRATOR_DILATIONS), "receptive_field": E.RECEPTIVE_FIELD,
                         "core": "Conv1D 16, kernel 3, causal, elu; read = last timestep",
                         "readout": f"Dense({pred_len} x {CHANNELS}) reshaped, added to the last observation broadcast over the horizon",
                         "shapes": {k: list(v) if isinstance(v, tuple) else v for k, v in shapes.items() if k != "layers"}},
        "regimes": {r: _regimes().REGIMES[r] for r in regimes},
        "factorial": {"seeds": list(seeds), "regimes": list(regimes), "cells": [f"{r}_s{s}" for r in regimes for s in seeds],
                      "pairing": "one auto-encoder per seed; R1 and R2 of a seed consume that seed's detector, R0 shares the seed's initial checkpoint",
                      "no_best_seed_choice": "every seed is reported; no seed is selected by its score"},
        "pretraining": {"objective": "masked reconstruction of the preprocessed input; the loss counts masked positions only",
                        "data": "outer TRAIN windows only",
                        "internal_validation": ae_partition([0, 1], seq_len=SEQ_LEN, pred_len=pred_len,
                                                            internal_validation_fraction=internal_validation_fraction)["rule"],
                        "internal_validation_fraction": internal_validation_fraction,
                        "decoder": "a separate per-branch 1x1 Conv1D stack, saved apart, never connected at inference",
                        "diagnostic_only": "reconstruction error is never a result of the task"},
        "optimisation": {"loss": "MSE on the normalized target", "optimizer": "Adam",
                         "checkpoint_rule": "the outer validation selects the downstream forecasting checkpoint and nothing else",
                         "max_updates": max_updates,
                         "cost_readings": ["the downstream fit alone", "the auto-encoder plus the downstream fit",
                                           "an equal-total-cost reading in which R0 receives the AE's budget as extra updates"]},
        "exposure": {"outer_test": "NO_ACCESS during any fit or selection",
                     "public_test_scoring": "not performed by the pilot and not used to choose anything"},
    }
    # RP152 (Musashi F3): the identity covers the SCIENCE and not the clock. `at` is operational and is excluded by name, so
    # two seals of the same design share one digest while any scientific change moves it.
    design["identity_covers"] = ["reference", "task", "channel_order", "architecture", "regimes", "factorial",
                                 "pretraining", "optimisation", "exposure"]
    design["design_sha256"] = hashlib.sha256(
        json.dumps({k: design[k] for k in design["identity_covers"]}, sort_keys=True, default=str).encode()).hexdigest()
    return design


# --- the TRAIN-ONLY pilot -----------------------------------------------------------------------------------------------------

def _windows_class(tf):
    """Keras 3 requires a PyDataset; the class is built lazily so importing this module never imports the framework."""

    class _TrainWindows(tf.keras.utils.PyDataset):
        """A sequence over the author's own Dataset, restricted to the given origins of the TRAIN split. Nothing outside TRAIN
        is reachable from here: the origins are the only index this object will serve."""

        def __init__(self, dataset, local_indices, *, seq_len, pred_len, batch, seed, masked=None, fixed_masks=False,
                     complete=False, **kw):
            super().__init__(**kw)
            self.ds, self.idx = dataset, np.asarray(local_indices, dtype=np.int64)
            self.seq_len, self.pred_len, self.batch, self.masked = seq_len, pred_len, int(batch), masked
            self.seed, self.fixed_masks, self.complete = int(seed), bool(fixed_masks), bool(complete)
            self.rng = np.random.default_rng(int(seed))

        def __len__(self):
            # RP152: a scoring population must lose no window. `complete` yields the uneven last batch instead of dropping it.
            if self.complete:
                return max(1, -(-len(self.idx) // self.batch))
            return max(1, len(self.idx) // self.batch)

        def __getitem__(self, b):
            sel = self.idx[b * self.batch:(b + 1) * self.batch]
            xs, ys = [], []
            for i in sel:
                x, y, _a, _c = self.ds[int(i)]
                xs.append(np.asarray(x, dtype=np.float32))
                ys.append(np.asarray(y, dtype=np.float32)[-self.pred_len:, :])
            x = np.stack(xs); y = np.stack(ys)
            if self.masked is None:
                return x, y
            if self.fixed_masks:
                # RP146: a validation mask must be a property of the WINDOW, not of the order in which it was read. The mask of
                # origin o is drawn from a generator seeded by (seed, o), so the same batch is byte-identical on every access.
                mask = np.stack([np.random.default_rng([self.seed, int(i)]).random(x.shape[1:]) for i in sel])
                mask = (mask < float(self.masked)).astype(np.float32)
            else:
                mask = (self.rng.random(x.shape) < float(self.masked)).astype(np.float32)
            return x * (1.0 - mask), np.concatenate([x, mask], axis=2)

    return _TrainWindows


class _Unused:
    def __init__(self, dataset, local_indices, *, seq_len: int, pred_len: int, batch: int, seed: int, masked=None):
        self.ds, self.idx = dataset, np.asarray(local_indices, dtype=np.int64)
        self.seq_len, self.pred_len, self.batch, self.masked = seq_len, pred_len, int(batch), masked
        self.rng = np.random.default_rng(int(seed))

    def __len__(self):
        return max(1, len(self.idx) // self.batch)

    def __getitem__(self, b):
        sel = self.idx[b * self.batch:(b + 1) * self.batch]
        xs, ys = [], []
        for i in sel:
            x, y, _a, _c = self.ds[int(i)]
            xs.append(np.asarray(x, dtype=np.float32))
            ys.append(np.asarray(y, dtype=np.float32)[-self.pred_len:, :])
        x = np.stack(xs); y = np.stack(ys)
        if self.masked is None:
            return x, y
        mask = (self.rng.random(x.shape) < float(self.masked)).astype(np.float32)
        return x * (1.0 - mask), np.concatenate([x, mask], axis=2)


def _masked_mse(tf, p: int):
    def loss(y_true, y_pred):
        x, m = y_true[..., :p], y_true[..., p:]
        return tf.reduce_sum(m * tf.square(x - y_pred)) / (tf.reduce_sum(m) + 1e-8)
    return loss


def train_only_pilot(data_path: Path, out_dir: Path, *, pred_len: int = 96, seed: int = 2021, batch: int = 32,
                     cpu_budget_seconds: float = 1800.0, wall_budget_seconds: float = 1800.0,
                     probe_steps: int = 5, internal_validation_fraction: float = 0.2) -> dict:
    """RP142's authorised pilot: TRAIN ONLY. It measures what an auto-encoder step and an R0 step cost on this host, PRESCRIBES
    the step counts from that measurement and the declared budget, runs them, and reports the regime diagnostics. It computes no
    test score, selects nothing on any validation, and makes no claim about H1."""
    import os as _os
    import resource
    E, RG = _e0(), _regimes()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cpu_now = lambda: sum(getattr(resource.getrusage(resource.RUSAGE_SELF), k) for k in ("ru_utime", "ru_stime"))
    started_wall, started_cpu = time.time(), cpu_now()
    causality = causality_report(data_path, pred_len=pred_len, perturb_windows=2)
    if not causality["pass"]:
        raise RuntimeError(f"REFUSED: the causality proofs did not pass: {causality['checks']}")
    design = seal_contrast(data_path, pred_len=pred_len)
    d = author_datasets(data_path, pred_len=pred_len)
    train_ds = d["splits"]["train"]["dataset"]
    origins = split_origins(data_path, pred_len=pred_len)
    part = ae_partition(origins["train"]["origins"], seq_len=SEQ_LEN, pred_len=pred_len,
                        internal_validation_fraction=internal_validation_fraction)
    base = origins["train"]["rows"][0]
    ae_tr_local = [o - base for o in part["ae_train_origins"]]
    ae_va_local = [o - base for o in part["ae_validation_origins"]]
    tr_local = [o - base for o in origins["train"]["origins"]]
    assignment = [0] * CHANNELS
    tf = E._tf()
    tf.keras.utils.set_random_seed(int(seed))
    out = {"schema": "df_ecl_modular_pilot.v1", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": _os.uname().nodename, "pred_len": pred_len, "seed": seed, "batch": batch,
           "budget": {"cpu_seconds": cpu_budget_seconds, "wall_seconds": wall_budget_seconds},
           "design_sha256": design["design_sha256"], "causality": causality["checks"],
           "exposure": "TRAIN ONLY: no validation selection and no test access in this pilot"}
    ae, dec_names = RG.build_autoencoder(assignment, SEQ_LEN, CHANNELS, arch=ARCH, seed=seed, mask_ratio=0.25)
    ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=_masked_mse(tf, CHANNELS))
    W = _windows_class(tf)
    ae_seq = W(train_ds, ae_tr_local, seq_len=SEQ_LEN, pred_len=pred_len, batch=batch, seed=seed, masked=0.25)
    ae.fit(ae_seq, epochs=1, steps_per_epoch=1, verbose=0)          # warm-up: the first call also traces the graph
    t0 = time.time()
    ae.fit(ae_seq, epochs=1, steps_per_epoch=probe_steps, verbose=0)
    ae_s_per_step = (time.time() - t0) / probe_steps
    model = build_ecl_modular(assignment, pred_len=pred_len, seed=seed)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    fit_seq = W(train_ds, tr_local, seq_len=SEQ_LEN, pred_len=pred_len, batch=batch, seed=seed)
    model.fit(fit_seq, epochs=1, steps_per_epoch=1, verbose=0)      # warm-up, for the same reason
    t0 = time.time()
    model.fit(fit_seq, epochs=1, steps_per_epoch=probe_steps, verbose=0)
    r0_s_per_step = (time.time() - t0) / probe_steps
    spent_wall = time.time() - started_wall
    remaining = max(0.0, min(wall_budget_seconds, cpu_budget_seconds) - spent_wall - 150.0)
    ae_steps = max(probe_steps, int(remaining * 0.40 / max(ae_s_per_step, 1e-6)))
    r0_steps = max(probe_steps, int(remaining * 0.40 / max(r0_s_per_step, 1e-6)))
    out["cost_probe"] = {"probe_steps": probe_steps, "ae_seconds_per_step": ae_s_per_step, "r0_seconds_per_step": r0_s_per_step,
                         "wall_spent_before_prescription": spent_wall, "wall_available_after_reserve": remaining,
                         "warm_up": "one untimed step precedes each timing so graph tracing is not charged to the per-step cost"}
    out["prescribed"] = {"ae_steps_requested": ae_steps, "r0_steps_requested": r0_steps,
                         "rule": "prescribed from the measured cost and the declared budget BEFORE the fits, never adjusted afterwards"}
    va_seq = W(train_ds, ae_va_local, seq_len=SEQ_LEN, pred_len=pred_len, batch=batch, seed=0, masked=0.25, fixed_masks=True)
    # a finite sequence yields at most len(seq) batches per epoch, so the prescribed step count is executed as whole passes:
    # asking for more steps than the sequence holds silently ran ONE pass in the first attempt, which is recorded in the return
    ae_epochs = max(1, math.ceil(ae_steps / max(1, len(ae_seq))))
    ae_steps = ae_epochs * len(ae_seq)
    t0 = time.time()
    hist = ae.fit(ae_seq, epochs=ae_epochs, steps_per_epoch=len(ae_seq), validation_data=va_seq,
                  validation_steps=min(10, len(va_seq)), verbose=0)
    out["autoencoder"] = {"steps": ae_steps, "epochs": ae_epochs, "steps_per_epoch": len(ae_seq),
                          "wall_seconds": time.time() - t0,
                          "loss": [float(v) for v in hist.history.get("loss", [])],
                          "internal_validation_loss": [float(v) for v in hist.history.get("val_loss", [])],
                          "validation_origins": {"n": len(ae_va_local), "inside_outer_train": True, "purge": part["purge"]},
                          "reading": "masked reconstruction; a diagnostic, never a result of the task"}
    det_names = RG.detector_layer_names(model)
    npz = out_dir / f"detector_seed{seed}.npz"
    np.savez(npz, **{f"{n}__{i}": np.asarray(w) for n in det_names for i, w in enumerate(ae.get_layer(n).get_weights())})
    out["pretrained_detector"] = {"path": str(npz), "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                                  "layers": det_names, "decoder_layers": dec_names,
                                  "decoder_never_connected_at_inference": True}
    r0_epochs = max(1, math.ceil(r0_steps / max(1, len(fit_seq))))
    r0_steps = r0_epochs * len(fit_seq)
    t0 = time.time()
    h0 = model.fit(fit_seq, epochs=r0_epochs, steps_per_epoch=len(fit_seq), verbose=0)
    out["R0"] = {"steps": r0_steps, "epochs": r0_epochs, "steps_per_epoch": len(fit_seq),
                 "wall_seconds": time.time() - t0, "loss": [float(v) for v in h0.history.get("loss", [])]}
    xb, yb = fit_seq[0]
    r0_detector_after_fit = RG.weights_digest(model, det_names)
    r1 = RG.apply_regime(model, "R1", npz)
    g1 = RG.gradient_report(model, xb[:4], yb[:4])
    d1 = RG.weights_digest(model, det_names)
    d1_step = _observed_step(tf, model, xb[:4], yb[:4], det_names, RG)
    r2 = RG.apply_regime(model, "R2", npz)
    g2 = RG.gradient_report(model, xb[:4], yb[:4])
    d2 = RG.weights_digest(model, det_names)
    d2_step = _observed_step(tf, model, xb[:4], yb[:4], det_names, RG)
    donor_check = donor_source_equality(npz, ae, det_names)
    step_check = {"R1_detector_unchanged_after_a_step": d1_step["unchanged"],
                  "R2_detector_changed_after_a_step": not d2_step["unchanged"],
                  "R1_max_abs_weight_move": d1_step["max_abs_move"], "R2_max_abs_weight_move": d2_step["max_abs_move"]}
    out["regimes"] = {
        "R0_detector_digest_after_its_own_fit": r0_detector_after_fit,
        "R1": {"imported": r1["imported"], "frozen_layers": len(r1["frozen"]),
               "detector_receives_gradient": g1["detector_receives_gradient"]},
        "R2": {"imported": r2["imported"], "trainable_layers": len(r2["trainable"]),
               "detector_receives_gradient": g2["detector_receives_gradient"]},
        "R1_and_R2_import_the_same_bytes": d1 == d2,
        "donor_source_equality": donor_check,
        "observed_optimizer_step": step_check,
        "proved": ("the import is checked against the donor's own bytes, and the freeze and the update are read from an OBSERVED "
                   "optimizer step, not from a flag and not from the existence of a gradient"),
    }
    out["measured_cost"] = {"cpu_seconds": cpu_now() - started_cpu, "wall_seconds": time.time() - started_wall}
    out["measured_cost"]["within_cpu_budget"] = out["measured_cost"]["cpu_seconds"] <= cpu_budget_seconds
    out["measured_cost"]["within_wall_budget"] = out["measured_cost"]["wall_seconds"] <= wall_budget_seconds
    out["claims"] = {"H1": "none: this pilot measures cost and proves the path; it says nothing about pre-training's effect",
                     "public_test": "not scored", "model_selection": "none performed"}
    (out_dir / "PILOT.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def _observed_step(tf, model, xb, yb, det_names, RG) -> dict:
    """RP146: run ONE real optimizer step and read the detector's weights before and after. A gradient that exists is not an
    update; only the weights after a step say whether the regime held."""
    before = [np.array(w, copy=True) for n in det_names for w in model.get_layer(n).get_weights()]
    opt = tf.keras.optimizers.Adam(1e-2)
    variables = list(model.trainable_variables)
    loss_fn = tf.keras.losses.MeanSquaredError()
    with tf.GradientTape() as tape:
        loss = loss_fn(tf.constant(yb, dtype=tf.float32), model(tf.constant(xb, dtype=tf.float32), training=True))
    grads = tape.gradient(loss, variables)
    pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
    if pairs:
        opt.apply_gradients(pairs)
    after = [np.asarray(w) for n in det_names for w in model.get_layer(n).get_weights()]
    moves = [float(np.max(np.abs(a - b))) for a, b in zip(after, before)] or [0.0]
    return {"unchanged": all(m == 0.0 for m in moves), "max_abs_move": max(moves), "loss_on_batch": float(loss),
            "trainable_variables_stepped": len(pairs)}


def donor_source_equality(npz_path: Path, donor_model, det_names: list) -> dict:
    """RP146: the imported bytes must be the DONOR's bytes. Comparing the two regimes to each other accepts the same wrong
    donor twice, so the check is against the saved file and against the auto-encoder that wrote it."""
    z = np.load(npz_path)
    equal, compared = True, []
    for n in det_names:
        ws = donor_model.get_layer(n).get_weights()
        for i, w in enumerate(ws):
            key = f"{n}__{i}"
            if key not in z.files:
                equal = False; compared.append({"layer": key, "present_in_file": False}); continue
            same = bool(np.array_equal(np.asarray(z[key]), np.asarray(w)))
            equal = equal and same
            compared.append({"layer": key, "equal_to_donor": same})
    return {"donor_file": str(npz_path), "donor_file_sha256": hashlib.sha256(Path(npz_path).read_bytes()).hexdigest(),
            "layers_compared": len(compared), "all_equal_to_donor": equal, "detail": compared[:6]}


def delivered_batch_oracle(data_path: Path, *, pred_len: int, seq_len: int = SEQ_LEN, batch: int = 4,
                           at_boundaries: bool = True, n_windows: int = 4) -> dict:
    """RP146: check the batches the model is actually FED against an independent oracle built from the raw CSV and the author's
    own scaler statistics, rather than by calling his Dataset a second time. Covers the first and last windows of the TRAIN
    split, the channel order and the metric reduction."""
    import pandas as pd
    E = _e0()
    tf = E._tf()
    d = author_datasets(data_path, pred_len=pred_len, seq_len=seq_len)
    train_ds = d["splits"]["train"]["dataset"]
    origins = split_origins(data_path, pred_len=pred_len, seq_len=seq_len)
    base, n = origins["train"]["rows"][0], origins["train"]["n_windows"]
    local = ([0, 1, n - 2, n - 1] if at_boundaries else list(range(min(n_windows, n))))
    local = [i for i in local if 0 <= i < n][:max(1, n_windows)]
    W = _windows_class(tf)
    seq = W(train_ds, local, seq_len=seq_len, pred_len=pred_len, batch=batch, seed=0)
    xb, yb = seq[0]
    # the independent side: the raw CSV, the author's train-only standardisation, and plain row slicing
    df = pd.read_csv(data_path)
    cols = [c for c in df.columns if c != "date"]
    raw = df[cols].to_numpy(dtype=np.float64)
    n_rows = raw.shape[0]
    num_train = int(n_rows * 0.7)
    mu, sd = raw[:num_train].mean(axis=0), raw[:num_train].std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    scaled = (raw - mu) / sd
    ok_x, ok_y, worst_x, worst_y = True, True, 0.0, 0.0
    for k, i in enumerate(local[:xb.shape[0]]):
        o = base + i
        ex = scaled[o:o + seq_len, :]
        ey = scaled[o + seq_len:o + seq_len + pred_len, :]
        dx = float(np.max(np.abs(np.asarray(xb[k], dtype=np.float64) - ex)))
        dy = float(np.max(np.abs(np.asarray(yb[k], dtype=np.float64) - ey)))
        worst_x, worst_y = max(worst_x, dx), max(worst_y, dy)
        ok_x, ok_y = ok_x and dx <= 1e-4, ok_y and dy <= 1e-4
    # the reduction, recomputed independently on a fabricated prediction
    pred = np.asarray(yb, dtype=np.float64) + 0.5
    mae = float(np.abs(pred - np.asarray(yb, dtype=np.float64)).mean())
    mse = float(((pred - np.asarray(yb, dtype=np.float64)) ** 2).mean())
    return {"schema": "df_ecl_modular_batch_oracle.v1", "pred_len": pred_len, "origins_checked": [base + i for i in local],
            "delivered_shapes": {"x": list(xb.shape), "y": list(yb.shape)},
            "inputs_match_independent_oracle": ok_x, "targets_match_independent_oracle": ok_y,
            "max_abs_input_difference": worst_x, "max_abs_target_difference": worst_y,
            "channel_count_delivered": int(xb.shape[2]), "channel_order": channel_order_digest(data_path),
            "reduction_check": {"fabricated_offset": 0.5, "mae": mae, "mse": mse,
                                "mae_equals_offset": abs(mae - 0.5) < 1e-9, "mse_equals_offset_squared": abs(mse - 0.25) < 1e-9},
            "scope": ("the oracle reads the raw CSV, standardises with the author's train-only statistics and slices rows; it "
                      "never calls his Dataset, so a batching or channel-order defect in the delivered windows would show")}



class AllocationError(RuntimeError):
    """A declared budget that cannot fit even a minimum viable run. Raised BEFORE anything is fitted."""


def regime_checks(cells: dict, seeds) -> dict:
    """RP152/RP153 (Musashi F2 and integration finding 5): a verdict over a population that must actually be there, computed
    only from TYPED evidence. `all()` over an empty set is True and `not v.get(k)` turns a MISSING measurement into a positive
    finding; both made an incomplete run look like a passing contrast. A check without typed evidence is None, a cell nobody
    asked for is an error rather than extra support, and every state is named."""
    seeds = list(seeds)
    cells = cells or {}
    expected = [f"AE_s{s}" for s in seeds] + [f"{r}_s{s}" for s in seeds for r in ("R0", "R1", "R2")]
    missing = [c for c in expected if c not in cells]
    unexpected = sorted(set(cells) - set(expected))
    typed = lambda cell, key: isinstance((cell or {}).get(key), bool)
    fit_names = [f"{r}_s{s}" for s in seeds for r in ("R0", "R1", "R2") if f"{r}_s{s}" in cells]
    without_evidence = sorted(n for n in fit_names if not typed(cells[n], "detector_unchanged_by_the_fit"))
    out = {"expected_cells": expected, "present_cells": sorted(cells), "missing_cells": missing,
           "unexpected_cells": unexpected, "cells_without_evidence": without_evidence}
    r1 = [cells[n] for n in fit_names if n.startswith("R1_")]
    r2 = [cells[n] for n in fit_names if n.startswith("R2_")]
    out["R1_detector_unchanged_by_its_fit"] = (all(c["detector_unchanged_by_the_fit"] for c in r1)
                                               if r1 and all(typed(c, "detector_unchanged_by_the_fit") for c in r1) else None)
    out["R2_detector_changed_by_its_fit"] = (all(c["detector_unchanged_by_the_fit"] is False for c in r2)
                                             if r2 and all(typed(c, "detector_unchanged_by_the_fit") for c in r2) else None)
    pairs = [(f"R1_s{s}", f"R2_s{s}") for s in seeds if f"R1_s{s}" in cells and f"R2_s{s}" in cells]
    out["R1_and_R2_share_the_donor_per_seed"] = (all(cells[a].get("donor") and cells[a].get("donor") == cells[b].get("donor")
                                                     for a, b in pairs) if pairs else None)
    fits = [cells[n] for n in fit_names]
    def _same(key):
        """True when every fit cell carries a typed value for `key` and they all agree; None when any is missing."""
        values = [c.get(key) for c in fits]
        if not fits or any(v is None or isinstance(v, bool) or not isinstance(v, int) for v in values):
            return None
        return len(set(values)) == 1

    out["same_update_allowance"] = _same("steps")
    out["same_observed_updates"] = _same("observed_updates")
    out["complete"] = not missing and not unexpected and not without_evidence
    out["verdict"] = ("COMPLETE" if out["complete"] else
                      ("UNEXPECTED_CELLS" if unexpected else
                       ("INCOMPLETE_POPULATION" if missing else "INCOMPLETE_EVIDENCE")))
    out["reading"] = ("every check is computed from typed evidence in cells that exist; a missing population, a cell nobody "
                      "asked for and an untyped measurement are each named, and none of them becomes a green verdict")
    return out

def run_contrast(data_path: Path, out_dir: Path, *, pred_len: int = 96, seeds=(2021, 2022, 2023), batch: int = 32,
                 cpu_budget_seconds: float = 14400.0, wall_budget_seconds: float = 28800.0,
                 internal_validation_fraction: float = 0.2, probe_steps: int = 5,
                 limit_train_windows: int | None = None, limit_validation_windows: int | None = None,
                 max_epochs: int | None = None, validation_monitor_batches: int = 20) -> dict:
    """RP146/RP152: the development R0/R1/R2 contrast on the matched ECL task.

    Corrected after Musashi's interim review. The declared checkpoint contract is now EXECUTED: each regime fits under a
    best-validation checkpoint callback, the selected epoch is recorded, the selected weights are restored and the restored
    model is saved with its digest, so the comparison is replayable. Update counts are read from the optimizer's own
    iteration counter rather than multiplied out. CPU and wall are tracked apart, an allocation that cannot fit a minimum
    viable run is REFUSED before anything is fitted, and every completed cell is persisted immediately so a hard limit leaves
    a durable partial result. The validation reading is a declared MONITOR and is named as not being the reference metric.
    """
    import os as _os
    import resource
    E, RG = _e0(), _regimes()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cpu_now = lambda: sum(getattr(resource.getrusage(resource.RUSAGE_SELF), k) for k in ("ru_utime", "ru_stime"))
    t_wall0, t_cpu0 = time.time(), cpu_now()
    causality = causality_report(data_path, pred_len=pred_len, perturb_windows=2)
    if not causality["pass"]:
        raise RuntimeError(f"REFUSED: causality proofs did not pass: {causality['checks']}")
    oracle = delivered_batch_oracle(data_path, pred_len=pred_len, n_windows=4)
    if not (oracle["inputs_match_independent_oracle"] and oracle["targets_match_independent_oracle"]):
        raise RuntimeError("REFUSED: the delivered batches do not match the independent oracle")
    design = seal_contrast(data_path, pred_len=pred_len, seeds=seeds, internal_validation_fraction=internal_validation_fraction)
    d = author_datasets(data_path, pred_len=pred_len)
    train_ds, val_ds = d["splits"]["train"]["dataset"], d["splits"]["val"]["dataset"]
    origins = split_origins(data_path, pred_len=pred_len)
    part = ae_partition(origins["train"]["origins"], seq_len=SEQ_LEN, pred_len=pred_len,
                        internal_validation_fraction=internal_validation_fraction)
    base = origins["train"]["rows"][0]
    tr_local = [o - base for o in origins["train"]["origins"]]
    ae_tr_local = [o - base for o in part["ae_train_origins"]]
    ae_va_local = [o - base for o in part["ae_validation_origins"]]
    val_local = list(range(origins["val"]["n_windows"]))
    if limit_train_windows:
        tr_local, ae_tr_local = tr_local[:int(limit_train_windows)], ae_tr_local[:int(limit_train_windows)]
        ae_va_local = ae_va_local[:max(batch, int(limit_train_windows) // 4)]
    if limit_validation_windows:
        val_local = val_local[:int(limit_validation_windows)]
    assignment = [0] * CHANNELS
    tf = E._tf()
    W = _windows_class(tf)
    out = {"schema": "df_ecl_modular_contrast_run.v2", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": _os.uname().nodename, "design_sha256": design["design_sha256"], "pred_len": pred_len,
           "seeds": list(seeds), "batch": batch,
           "budget": {"cpu_seconds": cpu_budget_seconds, "wall_seconds": wall_budget_seconds},
           "causality": causality["checks"],
           "batch_oracle": {k: oracle[k] for k in ("inputs_match_independent_oracle", "targets_match_independent_oracle",
                                                   "max_abs_input_difference", "max_abs_target_difference")},
           "exposure": "the outer test is never read; the outer validation is read only as the declared checkpoint monitor",
           "cells": {}, "cost_probe": {}}
    persist = lambda: (out_dir / "CONTRAST.json").write_text(json.dumps(out, indent=1, default=str))
    # --- one cost probe, then one update allowance used by every regime of every seed
    tf.keras.utils.set_random_seed(int(seeds[0]))
    probe_model = build_ecl_modular(assignment, pred_len=pred_len, seed=int(seeds[0]))
    probe_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    probe_seq = W(train_ds, tr_local, seq_len=SEQ_LEN, pred_len=pred_len, batch=batch, seed=int(seeds[0]), complete=True)
    probe_model.fit(probe_seq, epochs=1, steps_per_epoch=1, verbose=0)
    t0 = time.time(); probe_model.fit(probe_seq, epochs=1, steps_per_epoch=min(probe_steps, len(probe_seq)), verbose=0)
    fit_s = (time.time() - t0) / max(1, min(probe_steps, len(probe_seq)))
    ae_probe, _dn = RG.build_autoencoder(assignment, SEQ_LEN, CHANNELS, arch=ARCH, seed=int(seeds[0]), mask_ratio=0.25)
    ae_probe.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=_masked_mse(tf, CHANNELS))
    ae_seq0 = W(train_ds, ae_tr_local, seq_len=SEQ_LEN, pred_len=pred_len, batch=batch, seed=int(seeds[0]), masked=0.25,
                complete=True)
    ae_probe.fit(ae_seq0, epochs=1, steps_per_epoch=1, verbose=0)
    t0 = time.time(); ae_probe.fit(ae_seq0, epochs=1, steps_per_epoch=min(probe_steps, len(ae_seq0)), verbose=0)
    ae_s = (time.time() - t0) / max(1, min(probe_steps, len(ae_seq0)))
    spent_wall, spent_cpu = time.time() - t_wall0, cpu_now() - t_cpu0
    reserve = 60.0 if limit_train_windows else 600.0
    usable_wall = wall_budget_seconds - spent_wall - reserve
    usable_cpu = cpu_budget_seconds - spent_cpu - reserve
    usable = min(usable_wall, usable_cpu)
    n_seeds = len(seeds)
    minimum = n_seeds * (ae_s * len(ae_seq0) + 3 * fit_s * len(probe_seq))     # one epoch of each, per seed
    if usable <= 0 or usable < minimum:
        raise AllocationError(
            f"REFUSED: the declared allocation cannot fit a minimum viable run. usable {usable:.1f} s "
            f"(wall {usable_wall:.1f}, cpu {usable_cpu:.1f}) against a one-epoch minimum of {minimum:.1f} s; "
            f"nothing was fitted and no partial result was written")
    per_seed = usable / n_seeds
    ae_epochs = max(1, int((per_seed * 0.25) / max(1e-6, ae_s * len(ae_seq0))))
    fit_epochs = max(1, int((per_seed * 0.70 / 3) / max(1e-6, fit_s * len(probe_seq))))
    if max_epochs:
        ae_epochs, fit_epochs = min(ae_epochs, int(max_epochs)), min(fit_epochs, int(max_epochs))
    out["cost_probe"] = {"fit_seconds_per_step": fit_s, "ae_seconds_per_step": ae_s,
                         "steps_per_epoch_fit": len(probe_seq), "steps_per_epoch_ae": len(ae_seq0),
                         "wall_spent_before_prescription": spent_wall, "cpu_spent_before_prescription": spent_cpu,
                         "usable_wall": usable_wall, "usable_cpu": usable_cpu, "one_epoch_minimum": minimum}
    out["prescribed"] = {"ae_epochs": ae_epochs, "fit_epochs_per_regime": fit_epochs, "identical_across_regimes": True,
                         "rule": "one measured probe, one allowance; every regime of every seed receives the same one"}
    out["populations"] = {
        "train_windows": len(tr_local), "ae_train_windows": len(ae_tr_local), "ae_validation_windows": len(ae_va_local),
        "validation_windows_available": len(val_local),
        "validation_monitor_windows": min(len(val_local), validation_monitor_batches * batch),
        "scoring_population": "NOT_SCORED_IN_THIS_RUN",
        "reading": ("the validation reading is a declared MONITOR used only for checkpoint selection; the reference metric "
                    "over a full population is computed separately from retained or replayed predictions, not here")}
    persist()
    for seed in seeds:
        tf.keras.utils.set_random_seed(int(seed))
        seed_dir = out_dir / f"seed{seed}"; seed_dir.mkdir(parents=True, exist_ok=True)
        init = build_ecl_modular(assignment, pred_len=pred_len, seed=int(seed))
        init_path = seed_dir / "initial.weights.h5"
        init.save_weights(str(init_path))
        det_names = RG.detector_layer_names(init)
        ae, _dec = RG.build_autoencoder(assignment, SEQ_LEN, CHANNELS, arch=ARCH, seed=int(seed), mask_ratio=0.25)
        ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=_masked_mse(tf, CHANNELS))
        ae_tr = W(train_ds, ae_tr_local, seq_len=SEQ_LEN, pred_len=pred_len, batch=batch, seed=int(seed), masked=0.25,
                  complete=True)
        ae_va = W(train_ds, ae_va_local, seq_len=SEQ_LEN, pred_len=pred_len, batch=batch, seed=0, masked=0.25,
                  fixed_masks=True, complete=True)
        t0, c0 = time.time(), cpu_now()
        ae_hist = ae.fit(ae_tr, epochs=ae_epochs, steps_per_epoch=len(ae_tr), validation_data=ae_va,
                         validation_steps=len(ae_va), verbose=0)
        donor = seed_dir / "detector.npz"
        np.savez(donor, **{f"{n}__{i}": np.asarray(w) for n in det_names for i, w in enumerate(ae.get_layer(n).get_weights())})
        out["cells"][f"AE_s{seed}"] = {
            "epochs": ae_epochs, "steps": ae_epochs * len(ae_tr),
            "observed_updates": int(np.asarray(ae.optimizer.iterations)),
            "wall_seconds": time.time() - t0, "cpu_seconds": cpu_now() - c0,
            "masked_loss_first_last": [float(ae_hist.history["loss"][0]), float(ae_hist.history["loss"][-1])],
            "fixed_inner_validation_first_last": [float(ae_hist.history["val_loss"][0]), float(ae_hist.history["val_loss"][-1])],
            "donor_sha256": hashlib.sha256(donor.read_bytes()).hexdigest(),
            "donor_equals_autoencoder": donor_source_equality(donor, ae, det_names)["all_equal_to_donor"]}
        persist()
        for regime in ("R0", "R1", "R2"):
            model = build_ecl_modular(assignment, pred_len=pred_len, seed=int(seed))
            model.load_weights(str(init_path))
            info = RG.apply_regime(model, regime, None if regime == "R0" else donor)
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
            fit_seq = W(train_ds, tr_local, seq_len=SEQ_LEN, pred_len=pred_len, batch=batch, seed=int(seed), complete=True)
            monitor_batches = min(len(val_local) // batch or 1, validation_monitor_batches)
            va_seq = W(val_ds, val_local[:monitor_batches * batch], seq_len=SEQ_LEN, pred_len=pred_len, batch=batch,
                       seed=0, complete=True)
            before_digest = RG.weights_digest(model, det_names)
            before_updates = int(np.asarray(model.optimizer.iterations)) if model.optimizer.built else 0
            best_path = seed_dir / f"{regime}_best.weights.h5"
            checkpoint = tf.keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True,
                                                            save_weights_only=True, mode="min", verbose=0)
            t0, c0 = time.time(), cpu_now()
            h = model.fit(fit_seq, epochs=fit_epochs, steps_per_epoch=len(fit_seq), validation_data=va_seq,
                          validation_steps=len(va_seq), callbacks=[checkpoint], verbose=0)
            wall, cpu = time.time() - t0, cpu_now() - c0
            observed = int(np.asarray(model.optimizer.iterations)) - before_updates
            after_digest = RG.weights_digest(model, det_names)
            val_curve = [float(v) for v in h.history.get("val_loss", [])]
            selected_epoch = int(np.argmin(val_curve)) + 1 if val_curve else fit_epochs
            restored, restored_loss = False, None
            if best_path.is_file():
                model.load_weights(str(best_path))
                restored_loss = float(model.evaluate(va_seq, verbose=0))
                restored = abs(restored_loss - min(val_curve)) <= 1e-5 if val_curve else False
            model_path = seed_dir / f"{regime}_selected.weights.h5"
            model.save_weights(str(model_path))
            out["cells"][f"{regime}_s{seed}"] = {
                "regime": regime, "seed": seed, "epochs": fit_epochs, "steps": fit_epochs * len(fit_seq),
                "observed_updates": observed, "steps_equal_observed": observed == fit_epochs * len(fit_seq),
                "wall_seconds": wall, "cpu_seconds": cpu,
                "train_loss_first_last": [float(h.history["loss"][0]), float(h.history["loss"][-1])],
                "validation_monitor_curve": val_curve,
                "validation_monitor_loss_is_not_the_reference_metric": True,
                "selected_epoch": selected_epoch, "selected_validation_monitor_loss": (min(val_curve) if val_curve else None),
                "restored_validation_monitor_loss": restored_loss, "restored_matches_selection": restored,
                "model_path": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "detector_digest_before": before_digest, "detector_digest_after": after_digest,
                "detector_unchanged_by_the_fit": before_digest == after_digest,
                "imported": info.get("imported"), "frozen_layers": len(info.get("frozen") or []),
                "donor": (None if regime == "R0" else hashlib.sha256(donor.read_bytes()).hexdigest())}
            persist()
            if cpu_now() - t_cpu0 > cpu_budget_seconds or time.time() - t_wall0 > wall_budget_seconds:
                out["stopped_early"] = f"the declared allocation was reached after {regime}_s{seed}; the cells above are durable"
                break
        if out.get("stopped_early"):
            break
    out["regime_checks"] = regime_checks(out["cells"], seeds)
    out["measured_cost"] = {"cpu_seconds": cpu_now() - t_cpu0, "wall_seconds": time.time() - t_wall0}
    out["measured_cost"]["within_cpu_budget"] = out["measured_cost"]["cpu_seconds"] <= cpu_budget_seconds
    out["measured_cost"]["within_wall_budget"] = out["measured_cost"]["wall_seconds"] <= wall_budget_seconds
    out["claims"] = {"scope": "DEVELOPMENT contrast: no test score, no recipe selected from any score, no H1 claim",
                     "cost_readings": ["the downstream fit alone", "the auto-encoder plus the downstream fit"],
                     "equal_total_cost": "NOT CLAIMED: an equal update allowance is not equal total cost evidence"}
    persist()
    return out


# --- RP155: fresh-process replay, full-population scoring and content reconciliation -------------------------------------------

def label_disjoint_origins(n_windows: int, monitor_windows: int, *, seq_len: int = SEQ_LEN, pred_len: int) -> dict:
    """RP156 (Musashi finding 1): which validation origins share NO target row with the selection monitor.

    A window at origin o predicts rows [o + seq_len, o + seq_len + pred_len). Different origin indices do not prove untouched
    target support: with pred_len hourly steps the first pred_len - 1 origins after the monitor still predict rows the monitor
    predicted. The disjoint set is computed here, not assumed."""
    if monitor_windows <= 0:
        return {"monitor_windows": 0, "first_disjoint_origin": 0, "disjoint_origins": list(range(n_windows)),
                "overlapping_origins": [], "rule": "no monitor, so every origin is label-disjoint"}
    last_monitor_target = (monitor_windows - 1) + seq_len + pred_len - 1
    first_disjoint = max(0, last_monitor_target - seq_len + 1)
    disjoint = [o for o in range(n_windows) if o >= first_disjoint]
    overlapping = [o for o in range(monitor_windows, n_windows) if o < first_disjoint]
    return {"monitor_windows": monitor_windows, "last_monitor_target_row": last_monitor_target,
            "first_disjoint_origin": first_disjoint, "disjoint_origins": disjoint,
            "overlapping_origins": overlapping, "n_disjoint": len(disjoint), "n_overlapping_after_monitor": len(overlapping),
            "rule": ("an origin is label-disjoint when its FIRST target row is after the monitor's last target row; sharing an "
                     "input context is not sharing a label")}


def score_cell(data_path: Path, run_dir: Path, cell: str, *, pred_len: int = 96, batch: int = 32,
               monitor_windows: int | None = None, work_dir: Path | None = None) -> dict:
    """Score ONE finished cell from its saved selected weights, in a FRESH process.

    RP156: the checkpoint identity is verified BEFORE the model is loaded, the populations are computed rather than assumed,
    and BOTH reductions are reported: the author's float32 metric through the existing bounded exact reducer, and a float64
    diagnostic, with their difference. Neither stands in for the other."""
    S, E, RG = _sota(), _e0(), _regimes()
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "CONTRAST.json").read_text())["cells"][cell]
    weights = Path(record["model_path"])
    if not weights.is_file():
        raise FileNotFoundError(f"REFUSED: {cell} declares {weights} and it is not on disk")
    on_disk = hashlib.sha256(weights.read_bytes()).hexdigest()
    if on_disk != record["model_sha256"]:
        raise ValueError(f"REFUSED: {cell}'s checkpoint hashes to {on_disk[:12]} and the record declares "
                         f"{str(record['model_sha256'])[:12]}; nothing is loaded or scored from it")
    seed = int(record["seed"])
    tf = E._tf()
    tf.keras.utils.set_random_seed(seed)
    model = build_ecl_modular([0] * CHANNELS, pred_len=pred_len, seed=seed)
    model.load_weights(str(weights))
    d = author_datasets(data_path, pred_len=pred_len)
    val_ds = d["splits"]["val"]["dataset"]
    n = len(val_ds)
    W = _windows_class(tf)
    monitor = int(monitor_windows or 0)
    support = label_disjoint_origins(n, monitor, pred_len=pred_len)
    groups = {"complete_validation": list(range(n)), "label_disjoint_from_selection": support["disjoint_origins"]}
    work = Path(work_dir or (run_dir / "scoring_work")); work.mkdir(parents=True, exist_ok=True)
    out = {"cell": cell, "seed": seed, "regime": record["regime"], "selected_epoch": record["selected_epoch"],
           "model_path": str(weights), "model_sha256_recorded": record["model_sha256"], "model_sha256_on_disk": on_disk,
           "model_identity_reconciled": True, "pred_len": pred_len, "target_support": {k: v for k, v in support.items()
                                                                                       if not k.endswith("_origins")},
           "populations": {}}
    for name, idx in groups.items():
        seq = W(val_ds, idx, seq_len=SEQ_LEN, pred_len=pred_len, batch=batch, seed=0, complete=True)
        pred_path, true_path, naive_path = (work / f"{cell}.{name}.{w}.npy" for w in ("pred", "true", "naive"))
        writers = {}
        rows = 0
        for b in range(len(seq)):
            x, y = seq[b]
            p_ = np.asarray(model.predict(x, verbose=0), dtype=np.float32)
            t_ = np.asarray(y, dtype=np.float32)
            nv = np.repeat(np.asarray(x, dtype=np.float32)[:, -1:, :], pred_len, axis=1)
            for path, block in ((pred_path, p_), (true_path, t_), (naive_path, nv)):
                if path not in writers:
                    from numpy.lib.format import open_memmap
                    writers[path] = open_memmap(path, mode="w+", dtype=np.float32,
                                                shape=(len(idx), pred_len, CHANNELS))
                writers[path][rows:rows + block.shape[0]] = block
            rows += p_.shape[0]
        for handle in writers.values():
            handle.flush()
        del writers
        author = S.author_metric_exact(S.StoredArray(pred_path), S.StoredArray(true_path))
        author_naive = S.author_metric_exact(S.StoredArray(naive_path), S.StoredArray(true_path))
        f64 = S.float64_metrics_files(pred_path, true_path, (len(idx), pred_len, CHANNELS))
        entry = {"windows": len(idx), "elements": int(len(idx) * pred_len * CHANNELS),
                 "author_float32": {"mae": author["mae"], "mse": author["mse"], "route": author.get("route")},
                 "matched_persistence_author_float32": {"mae": author_naive["mae"], "mse": author_naive["mse"]},
                 "skill_mae_vs_persistence": (1.0 - author["mae"] / author_naive["mae"]) if author_naive["mae"] else None}
        if f64:
            entry["independent_float64"] = {"mae": f64["mae"], "mse": f64["mse"]}
            entry["float64_minus_float32"] = {"mae": f64["mae"] - author["mae"], "mse": f64["mse"] - author["mse"]}
        out["populations"][name] = entry
        for path in (pred_path, true_path, naive_path):
            path.unlink(missing_ok=True)
    out["scope"] = ("scored on the outer VALIDATION split. `complete_validation` includes the checkpoint-selection monitor and "
                    "is therefore not an independent estimate. `label_disjoint_from_selection` shares NO target row with the "
                    "monitor; it is still adjacent in time and disjoint labels do not establish statistical independence. The "
                    "outer TEST split was not read")
    return out



#: RP158 (Musashi finding 1): the canonical scientific fields of a contrast design are defined by THIS SCHEMA, not by a list
#: the design carries about itself. An object cannot decide which of its own fields matter to its identity.
CANONICAL_DESIGN_FIELDS = ("reference", "task", "channel_order", "architecture", "regimes", "factorial",
                           "pretraining", "optimisation", "exposure")


class DesignError(RuntimeError):
    """A design that cannot authenticate itself against the run it claims to describe. Raised BEFORE any child is dispatched."""


def authenticate_design(contrast: dict, design: dict) -> dict:
    """RP157 (Musashi finding 1): a design is accepted only when it authenticates against the run's own record.

    Its digest is RECOMPUTED from its canonical scientific content, that recomputation must equal what the design claims, and
    that in turn must equal the digest the run recorded when it produced these cells. An empty factorial, or one that does not
    cover the cells the run holds, is refused. Comparing two caller-supplied strings would not be this."""
    if not isinstance(design, dict) or not isinstance(design.get("factorial"), dict):
        raise DesignError("REFUSED: a design with no factorial cannot describe this run")
    cells = list(design["factorial"].get("cells") or [])
    if not cells:
        raise DesignError("REFUSED: the design declares an empty factorial; an empty population is not a complete contrast")
    absent = [f for f in CANONICAL_DESIGN_FIELDS if f not in design]
    if absent:
        raise DesignError(f"REFUSED: the design omits canonical scientific field(s) {', '.join(absent)}; a partial object "
                          f"cannot authenticate as this design")
    covers = list(design.get("identity_covers") or ())
    if covers and sorted(covers) != sorted(CANONICAL_DESIGN_FIELDS):
        raise DesignError(f"REFUSED: the design declares its identity covers {sorted(covers)}, and this schema's canonical "
                          f"fields are {sorted(CANONICAL_DESIGN_FIELDS)}; an object does not choose what identifies it")
    recomputed = hashlib.sha256(
        json.dumps({k: design[k] for k in CANONICAL_DESIGN_FIELDS}, sort_keys=True, default=str).encode()).hexdigest()
    if recomputed != design.get("design_sha256"):
        raise DesignError(f"REFUSED: the design's canonical content hashes to {recomputed[:12]} and it claims "
                          f"{str(design.get('design_sha256'))[:12]}")
    if design.get("design_sha256") != contrast.get("design_sha256"):
        raise DesignError(f"REFUSED: the design digest {str(design.get('design_sha256'))[:12]} is not the one this run "
                          f"recorded when it produced these cells, {str(contrast.get('design_sha256'))[:12]}")
    held = {c for c in (contrast.get("cells") or {}) if not c.startswith("AE_")}
    declared = set(cells)
    if not held <= declared:
        raise DesignError(f"REFUSED: the run holds cells the design does not declare: {sorted(held - declared)[:4]}")
    if not declared <= held | {c for c in (contrast.get("cells") or {})}:
        raise DesignError(f"REFUSED: the design declares cells this run does not hold: {sorted(declared - held)[:4]}")
    declared_pred_len = (design.get("task") or {}).get("pred_len")
    run_pred_len = contrast.get("pred_len")
    if run_pred_len is not None and int(declared_pred_len or 0) != int(run_pred_len):
        raise DesignError(f"REFUSED: the design declares horizon {declared_pred_len!r} and this run produced its cells at "
                          f"{run_pred_len!r}; the task argument is bound to dispatch, not asserted beside it")
    declared_seeds = sorted(int(x) for x in (design.get("factorial") or {}).get("seeds") or [])
    run_seeds = sorted(int(x) for x in (contrast.get("seeds") or []))
    if declared_seeds and run_seeds and declared_seeds != run_seeds:
        raise DesignError(f"REFUSED: the design declares seeds {declared_seeds} and the run recorded {run_seeds}")
    return {"design_sha256": design["design_sha256"], "cells": sorted(declared), "pred_len": int(declared_pred_len),
            "canonical_fields": list(CANONICAL_DESIGN_FIELDS), "seeds": declared_seeds or run_seeds,
            "authenticated": ("the digest was recomputed over this SCHEMA's canonical fields, equals what the design claims, "
                              "equals the digest the run recorded, and its task arguments match the dispatch")}


def validate_children(children: dict, *, expected, pred_len: int, populations, reductions=("author_float32",),
                      expected_windows: dict | None = None, checkpoints: dict | None = None) -> dict:
    """RP157 (Musashi finding 2): bind every returned child to the cell it was asked for before ANY aggregate exists.

    A child must name its own cell, the seed and regime that cell encodes, the horizon the design declares, every expected
    population with a consistent window count, and typed finite numbers in every declared reduction. A silently missing
    population would otherwise average one regime over fewer seeds than the others."""
    problems, windows = [], {}
    expected_windows = expected_windows or {}
    for cell in expected:
        record = children.get(cell)
        if record is None:
            problems.append(f"{cell}: the design declares this cell and no child returned it")
            continue
        if record.get("cell") != cell:
            problems.append(f"{cell}: the child names cell {record.get('cell')!r}")
        regime, _, seed = cell.partition("_s")
        if str(record.get("regime")) != regime:
            problems.append(f"{cell}: the child declares regime {record.get('regime')!r}, not {regime!r}")
        if str(record.get("seed")) != seed:
            problems.append(f"{cell}: the child declares seed {record.get('seed')!r}, not {seed!r}")
        if int(record.get("pred_len") or 0) != int(pred_len):
            problems.append(f"{cell}: the child declares horizon {record.get('pred_len')!r}, not {pred_len}")
        reconciled = record.get("model_identity_reconciled")
        if reconciled is not True:
            problems.append(f"{cell}: the child's checkpoint identity is {reconciled!r}, which is not the boolean True; "
                            f"a truthy string is not a reconciliation")
        expected_digest = (checkpoints or {}).get(cell)
        if expected_digest is not None and record.get("model_sha256_on_disk") != expected_digest:
            problems.append(f"{cell}: the child scored a checkpoint hashing to "
                            f"{str(record.get('model_sha256_on_disk'))[:12]}, and the retained record declares "
                            f"{str(expected_digest)[:12]}")
        held = record.get("populations") or {}
        for population in populations:
            entry = held.get(population)
            if not isinstance(entry, dict):
                problems.append(f"{cell}: the child omits the {population} population")
                continue
            count = entry.get("windows")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                problems.append(f"{cell}: the {population} population declares an untyped window count {count!r}")
                continue
            # RP158 (finding 2): the expected count comes from the validated task and support CONTRACT, never from whatever
            # the first sibling happened to report. Nine children agreeing on 1 window is nine children being wrong together.
            contract = expected_windows.get(population)
            if contract is not None and count != contract:
                problems.append(f"{cell}: the {population} population reports {count} windows and the support contract "
                                f"declares {contract}")
            # RP158 (Musashi, medium): validating a field only WHEN it is an integer means an absent or string element count
            # passes. The count is what ties the metric to the array it was computed over, so it is mandatory and typed.
            elements = entry.get("elements")
            if isinstance(elements, bool) or not isinstance(elements, int):
                problems.append(f"{cell}: the {population} population declares an untyped element count {elements!r}; "
                                f"the count that ties a metric to its array is mandatory, not optional")
            elif elements != count * pred_len * CHANNELS:
                problems.append(f"{cell}: the {population} population declares {elements} elements, which is not "
                                f"{count} x {pred_len} x {CHANNELS}")
            seen = windows.setdefault(population, count)
            if seen != count:
                problems.append(f"{cell}: the {population} population has {count} windows and another cell has {seen}")
            baseline = entry.get("matched_persistence_author_float32")
            if not isinstance(baseline, dict):
                problems.append(f"{cell}: the {population} population omits its matched persistence baseline")
            else:
                for metric in ("mae", "mse"):
                    value = baseline.get(metric)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        problems.append(f"{cell}: the matched persistence {metric} is {value!r}, not a finite number")
                    elif float(value) < 0:
                        # RP158: MAE and MSE are means of absolute and squared errors. A negative one is not a weak result;
                        # it is a number that cannot have come from the definition it claims.
                        problems.append(f"{cell}: the matched persistence {metric} is {value!r}, and a mean of "
                                        f"{'absolute' if metric == 'mae' else 'squared'} errors is never negative")
            # RP158: the skill is DERIVED from two numbers in this same record. A finite value that does not follow from them
            # is not a weak skill; it is a contradiction, and `None` is only correct when the baseline is exactly zero.
            skill = entry.get("skill_mae_vs_persistence")
            observed = (entry.get(reductions[0]) or {}).get("mae") if isinstance(entry.get(reductions[0]), dict) else None
            base_mae = baseline.get("mae") if isinstance(baseline, dict) else None
            usable = (isinstance(base_mae, (int, float)) and not isinstance(base_mae, bool) and math.isfinite(float(base_mae))
                      and isinstance(observed, (int, float)) and not isinstance(observed, bool)
                      and math.isfinite(float(observed)))
            if skill is None:
                if usable and float(base_mae) != 0.0:
                    problems.append(f"{cell}: the {population} skill is null while its baseline MAE is {base_mae!r}; "
                                    f"an undefined skill is only correct against a zero baseline")
            elif isinstance(skill, bool) or not isinstance(skill, (int, float)) or not math.isfinite(float(skill)):
                problems.append(f"{cell}: the derived skill is {skill!r}, not a finite number")
            elif usable:
                if float(base_mae) == 0.0:
                    problems.append(f"{cell}: the {population} skill is {skill!r} against a zero baseline; that quotient "
                                    f"is undefined and must be reported as null, not as a number")
                else:
                    derived = 1.0 - float(observed) / float(base_mae)
                    if abs(derived - float(skill)) > 1e-9:
                        problems.append(f"{cell}: the {population} skill is {skill!r}, and the MAE {observed!r} against the "
                                        f"baseline {base_mae!r} derives {derived!r}; a stored derived value that does not "
                                        f"follow from its own record is a contradiction, not a rounding")
            for reduction in reductions:
                values = entry.get(reduction)
                if not isinstance(values, dict):
                    problems.append(f"{cell}: the {population} population omits the {reduction} reduction")
                    continue
                for metric in ("mae", "mse"):
                    value = values.get(metric)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        problems.append(f"{cell}: {reduction}.{metric} is {value!r}, which is not a finite number")
                    elif float(value) < 0:
                        problems.append(f"{cell}: {reduction}.{metric} is {value!r}, and a mean of "
                                        f"{'absolute' if metric == 'mae' else 'squared'} errors is never negative")
    unexpected = sorted(set(children) - set(expected))
    if unexpected:
        problems.append(f"children returned for cells nobody asked for: {unexpected[:4]}")
    return {"bound": not problems, "problems": problems, "windows_per_population": windows,
            "reading": ("every child is bound to its expected cell, seed, regime, horizon, populations and typed finite "
                        "reductions before any aggregate is formed")}


def score_contrast(data_path: Path, run_dir: Path, *, pred_len: int = 96, batch: int = 32, python: str | None = None,
                   design: dict | None = None) -> dict:
    """RP156: replay every cell of the REGISTERED design in a separate process. The expected population comes from the design,
    not from whatever survives in the run directory, and no summary is produced while a cell is missing or an identity fails."""
    import subprocess
    import sys
    run_dir = Path(run_dir)
    contrast = json.loads((run_dir / "CONTRAST.json").read_text())
    monitor = (contrast.get("populations") or {}).get("validation_monitor_windows") or 0
    # RP158: a retained design beside the run is the authority. A supplied one is accepted only if it authenticates, and a
    # re-derivation is accepted only if it reproduces the digest the run recorded: neither is a fresh seal trusted on sight.
    retained = run_dir / "DESIGN.json"
    if design is not None:
        registered, source = design, "supplied_and_authenticated"
    elif retained.is_file():
        registered, source = json.loads(retained.read_text()), "retained_beside_the_run"
    else:
        registered, source = seal_contrast(data_path, pred_len=pred_len, seeds=tuple(contrast["seeds"])), "re_derived_and_matched"
    authenticated = authenticate_design(contrast, registered)       # refuses before a single child is dispatched
    authenticated["design_source"] = source
    # RP158 continuation: a run whose design was only ever re-derived cannot be closed again without the datasets, which makes
    # every later closure depend on data that may be gone. The AUTHENTICATED design is retained beside the run the first time,
    # so the next closure reads it instead of rebuilding it. An existing file is never overwritten.
    if not retained.is_file():
        retained.write_text(json.dumps(registered, indent=1, default=str))
        authenticated["design_retained_now"] = True
    expected = authenticated["cells"]
    pred_len = authenticated["pred_len"]                            # the dispatch uses the AUTHENTICATED horizon
    # the expected window counts come from the task and support contract, not from whatever a child reports
    probe = author_datasets(data_path, pred_len=pred_len)
    n_validation = len(probe["splits"]["val"]["dataset"])
    support = label_disjoint_origins(n_validation, monitor, pred_len=pred_len)
    expected_windows = {"complete_validation": n_validation,
                        "label_disjoint_from_selection": support["n_disjoint"]}
    checkpoints = {c: (contrast["cells"].get(c) or {}).get("model_sha256") for c in expected}
    out = {"schema": "df_ecl_modular_scoring.v3", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "run_dir": str(run_dir), "design_sha256": contrast["design_sha256"],
           "registered_design_sha256": registered["design_sha256"], "monitor_windows": monitor,
           "expected_cells": expected, "fresh_process_per_cell": True, "cells": {}, "problems": []}
    for cell in expected:
        if cell not in contrast["cells"]:
            out["problems"].append(f"{cell}: the registered design declares it and the run holds no such cell")
            continue
        code = (f"import json,sys,importlib.util,pathlib;"
                f"spec=importlib.util.spec_from_file_location('m',{str(Path(__file__).resolve())!r});"
                f"m=importlib.util.module_from_spec(spec);sys.modules['m']=m;spec.loader.exec_module(m);"
                f"print(json.dumps(m.score_cell(pathlib.Path({str(data_path)!r}),pathlib.Path({str(run_dir)!r}),{cell!r},"
                f"pred_len={pred_len},batch={batch},monitor_windows={monitor})))")
        proc = subprocess.run([python or sys.executable, "-c", code], capture_output=True, text=True,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "3"})
        line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        if proc.returncode != 0 or not line:
            out["problems"].append(f"{cell}: the replay process refused or failed ({proc.returncode}): {proc.stderr.strip()[-220:]}")
            continue
        out["cells"][cell] = json.loads(line[-1])
    return close_contrast(out, run_dir, expected=expected, pred_len=pred_len,
                          expected_windows=expected_windows, checkpoints=checkpoints, authenticated=authenticated)


POPULATIONS = ("complete_validation", "label_disjoint_from_selection")


def close_contrast(out: dict, run_dir: Path, *, expected, pred_len: int, expected_windows: dict,
                   checkpoints: dict, authenticated: dict, write_to: str = "SCORING.json") -> dict:
    """RP158 (Musashi): the closure over RETAINED child records. It validates, reconciles and aggregates, and it runs no
    inference and starts no process. Whatever produced the children -- a dispatch minutes ago or a run from last week read
    back off disk -- the rules applied here are the same ones, in one place, so a re-closure cannot drift from a first one."""
    run_dir = Path(run_dir)
    binding = validate_children(out["cells"], expected=expected, pred_len=pred_len, populations=POPULATIONS,
                                reductions=("author_float32", "independent_float64"),
                                expected_windows=expected_windows, checkpoints=checkpoints)
    out["problems"] += binding["problems"]
    identity_failures = [c for c, r in out["cells"].items() if not r.get("model_identity_reconciled")]
    out["design_authentication"] = authenticated
    out["child_binding"] = {k: v for k, v in binding.items() if k != "problems"}
    out["child_binding"]["expected_windows_from_contract"] = expected_windows
    out["reconciliation"] = {"cells_expected": len(expected), "cells_scored": len(out["cells"]),
                             "identity_failures": identity_failures,
                             "complete": (len(out["cells"]) == len(expected) and not identity_failures
                                          and binding["bound"] and not out["problems"])}
    if not out["reconciliation"]["complete"]:
        # RP156 (finding 2): an incomplete or identity-failing population produces DIAGNOSTICS, never a regime summary
        out["by_regime"] = None
        out["status"] = "INCOMPLETE_EVIDENCE"
        out["diagnostics"] = {c: r["populations"] for c, r in out["cells"].items()}
        out["reading"] = ("a summary over a population that is missing a cell, or that includes a checkpoint whose identity "
                          "failed, would not be the declared contrast; the per-cell readings above are diagnostics")
        (run_dir / write_to).write_text(json.dumps(out, indent=1, default=str))
        return out
    out["status"] = "COMPLETE"
    by_regime = {}
    for cell, r in out["cells"].items():
        by_regime.setdefault(r["regime"], []).append(r)
    out["by_regime"] = {}
    for regime, rows in sorted(by_regime.items()):
        for population in POPULATIONS:
            picked = [row["populations"][population] for row in rows if population in row["populations"]]
            if not picked:
                continue
            block = {"seeds": len(picked),
                     "matched_persistence_author_float32_mae": float(np.mean([v["matched_persistence_author_float32"]["mae"] for v in picked]))}
            for reduction in ("author_float32", "independent_float64"):
                values = [v[reduction] for v in picked if reduction in v]
                if not values:
                    continue
                block[reduction] = {
                    "mae_mean": float(np.mean([v["mae"] for v in values])),
                    "mae_sd": (float(np.std([v["mae"] for v in values], ddof=1)) if len(values) > 1 else None),
                    "mse_mean": float(np.mean([v["mse"] for v in values])),
                    "mse_sd": (float(np.std([v["mse"] for v in values], ddof=1)) if len(values) > 1 else None)}
            out["by_regime"].setdefault(regime, {})[population] = block
    out["claims"] = {"population": "the outer VALIDATION split; the outer TEST split was not read",
                     "reductions": "the author's float32 metric and an independent float64 diagnostic are reported separately",
                     "selection_caveat": ("`complete_validation` includes the selection monitor; "
                                          "`label_disjoint_from_selection` shares no target row with it, which is not the same "
                                          "as statistical independence"),
                     "H1": "none: this is a development contrast and no hypothesis is decided here"}
    (run_dir / write_to).write_text(json.dumps(out, indent=1, default=str))
    return out


def contract_windows_from_run(contrast: dict, pred_len: int) -> dict:
    """The expected population, derived from the ACCEPTED run record alone.

    `validation_windows_available` and `validation_monitor_windows` were written when the run was prepared and are part of the
    evidence already accepted. Everything else follows from them by the same pure rule the scoring used, so no dataset is
    rebuilt and nothing is asked of the file being checked."""
    populations = contrast.get("populations") or {}
    available = populations.get("validation_windows_available")
    monitor = populations.get("validation_monitor_windows")
    for name, value in (("validation_windows_available", available), ("validation_monitor_windows", monitor)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DesignError(f"the run record declares an untyped {name} {value!r}; the accepted population is undeclared "
                              f"and this closure will not invent it")
    support = label_disjoint_origins(available, monitor, pred_len=pred_len)
    return {"complete_validation": available, "label_disjoint_from_selection": support["n_disjoint"]}


def close_retained_run(run_dir: Path, *, design: dict | None = None) -> dict:
    """Close a run from what was retained beside it: CONTRAST.json, DESIGN.json and the children inside SCORING.json.

    Nothing here loads a model, reads the market data or starts a process. The expected window counts are NOT recomputed --
    they are read from the contract the scoring recorded, because re-deriving them would mean rebuilding the datasets, and a
    closure that quietly rebuilds its own expectations can always agree with itself."""
    run_dir = Path(run_dir)
    contrast = json.loads((run_dir / "CONTRAST.json").read_text())
    scoring_path = run_dir / "SCORING.json"
    if not scoring_path.is_file():
        raise DesignError(f"{run_dir}: no retained SCORING.json; there are no child records to close over")
    retained = json.loads(scoring_path.read_text())
    registered = design
    if registered is None:
        design_path = run_dir / "DESIGN.json"
        if not design_path.is_file():
            raise DesignError(f"{run_dir}: no retained DESIGN.json and none supplied; the expected cells are undeclared")
        registered = json.loads(design_path.read_text())
    authenticated = authenticate_design(contrast, registered)
    # CL12 (Musashi finding 4): the expected counts are DERIVED from the accepted run record, not read out of the file whose
    # children they are meant to check. Rewriting SCORING.json and every child coherently used to pass, because the only
    # authority consulted lived inside the rewritten file.
    expected_windows = contract_windows_from_run(contrast, authenticated["pred_len"])
    self_reported = ((retained.get("child_binding") or {}).get("expected_windows_from_contract"))
    if self_reported and self_reported != expected_windows:
        raise DesignError(
            f"{run_dir}: the retained scoring declares the population counts {self_reported} and the accepted run record "
            f"derives {expected_windows}; a scoring file does not get to restate the population it was checked against")
    out = {"schema": "df_ecl_modular_scoring.v3", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "run_dir": str(run_dir), "design_sha256": contrast["design_sha256"],
           "registered_design_sha256": registered["design_sha256"],
           "monitor_windows": (contrast.get("populations") or {}).get("validation_monitor_windows") or 0,
           "expected_cells": authenticated["cells"], "fresh_process_per_cell": False,
           "source": "RETAINED_RECORDS_NO_INFERENCE",
           # CL12: what each part of this closure rests on, kept apart. Local consistency is not authentication.
           "authority": {
               "design": "AUTHENTICATED: the retained design reproduces the digest the run recorded",
               "population": ("AUTHENTICATED_FROM_RUN_RECORD: the counts are derived from the accepted CONTRAST.json through "
                              "label_disjoint_origins, and the scoring file's own restatement is only cross-checked"),
               "checkpoint_identity": "AUTHENTICATED_FROM_RUN_RECORD: each child's scored checkpoint digest must be the cell's",
               "metric_values": ("SELF_REPORTED_BY_THE_CHILD_RECORDS: a coherent rewrite of a metric and its baseline cannot "
                                 "be detected here. Detecting that needs custody over the child records themselves, which "
                                 "this closure does not have and does not claim"),
           },
           "cells": copy.deepcopy(retained.get("cells") or {}), "problems": []}
    checkpoints = {c: (contrast["cells"].get(c) or {}).get("model_sha256") for c in authenticated["cells"]}
    # a re-closure never overwrites the record it read: the retained scoring stays exactly as it was written
    return close_contrast(out, run_dir, expected=authenticated["cells"], pred_len=authenticated["pred_len"],
                          expected_windows=expected_windows, checkpoints=checkpoints, authenticated=authenticated,
                          write_to="CLOSURE.json")
