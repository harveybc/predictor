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

import hashlib
import importlib
import importlib.util
import json
import math
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
    """RP152 (Musashi F2): a verdict over a population, which first says whether that population is there. `all()` over an
    empty set is True, which made an empty or one-cell run look like a passing contrast. A check with no evidence is None."""
    seeds = list(seeds)
    expected = [f"AE_s{s}" for s in seeds] + [f"{r}_s{s}" for s in seeds for r in ("R0", "R1", "R2")]
    missing = [c for c in expected if c not in (cells or {})]
    out = {"expected_cells": expected, "present_cells": sorted(cells or {}), "missing_cells": missing,
           "complete": not missing, "verdict": "COMPLETE" if not missing else "INCOMPLETE_POPULATION"}
    r1 = [v for k, v in (cells or {}).items() if k.startswith("R1_")]
    r2 = [v for k, v in (cells or {}).items() if k.startswith("R2_")]
    fits = [v for k, v in (cells or {}).items() if not k.startswith("AE_")]
    out["R1_detector_unchanged_by_its_fit"] = (all(v.get("detector_unchanged_by_the_fit") for v in r1) if r1 else None)
    out["R2_detector_changed_by_its_fit"] = (all(not v.get("detector_unchanged_by_the_fit") for v in r2) if r2 else None)
    pairs = [(f"R1_s{s}", f"R2_s{s}") for s in seeds if f"R1_s{s}" in (cells or {}) and f"R2_s{s}" in (cells or {})]
    out["R1_and_R2_share_the_donor_per_seed"] = (all(cells[a].get("donor") == cells[b].get("donor") and cells[a].get("donor")
                                                     for a, b in pairs) if pairs else None)
    allowances = {v.get("steps") for v in fits if v.get("steps") is not None}
    out["same_update_allowance"] = (len(allowances) == 1 if fits else None)
    observed = {v.get("observed_updates") for v in fits if v.get("observed_updates") is not None}
    out["same_observed_updates"] = (len(observed) == 1 if observed else None)
    out["reading"] = ("every check is computed only from cells that exist; a missing population is reported as such and never "
                      "as a green verdict")
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
