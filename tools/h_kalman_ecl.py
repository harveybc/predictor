"""Lane H: ECL testbed for the three Kalman arms (M04 v2 NPZ, L = 24, H = 1..24, 321 channels). DEVELOPMENT, CPU.

Inputs (read-only): M04's ``data_ecl_l24_h24_v1`` NPZ files. The zip members are stored uncompressed, so they are read
through ``np.memmap`` (no 2.9 GB materialisation). The train-standardized row panel is reconstructed exactly from the
windows and targets (contiguous rows 0 .. 21043; the test rows from 21044 are never present in the NPZ files).

Learners are CHANNEL-INDEPENDENT and pooled across the 321 channels: one sample per (origin, channel), features = the
channel's own blocks at lags 0..23, target = the SEASONAL RESIDUAL r_h = y(t+h) - y(t+h-24) for h = 1..24; the forecast
is y(t+h-24) + r_hat_h, so a zero residual IS the seasonal naive. Arms (blocks per channel):
  A  [z]                         B  [z, level, innov, zinnov, logvar]        C  [level, innov, zinnov, logvar]
  C_EWMA [ewma level, ewma innov] (comparable causal bandwidth)   B_PERMUTED / B_NOISE (equal capacity, B's Kalman blocks
  destroyed)   C_SMOOTHER_NONCAUSAL (rejection control, never eligible).
Kalman: local level per channel, parameters fitted on TRAIN rows [0, 18412) only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pipe = _load("h_kalman_pipeline")
kf = pipe.kf
arms_lib = pipe.arms_lib


# ---------------------------------------------------------------------------------------------- reading
def _member(path, key):
    """(offset, shape, dtype) of an uncompressed member ``key``.npy of an .npz file."""
    path = str(path)
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(f"{key}.npy")
        if info.compress_type != 0:
            raise ValueError(f"{key} is compressed")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
        start = info.header_offset + 30 + int.from_bytes(local[26:28], "little") + int.from_bytes(local[28:30], "little")
        f.seek(start)
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(f)
        if fortran:
            raise ValueError("fortran order not supported")
        return f.tell(), shape, dtype


def npz_memmap(path, key):
    """Read-only memmap of an uncompressed member (small members and tests)."""
    off, shape, dtype = _member(path, key)
    return np.memmap(str(path), dtype=dtype, mode="r", offset=off, shape=shape)


def iter_member(path, key, chunk=512):
    """Sequential chunks of the leading axis, read with plain reads; the page cache of each chunk is dropped
    (posix_fadvise DONTNEED) so a large member does not inflate the job's cgroup charge."""
    off, shape, dtype = _member(path, key)
    row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    fd = os.open(str(path), os.O_RDONLY)
    try:
        for i in range(0, shape[0], chunk):
            n = min(chunk, shape[0] - i)
            buf = os.pread(fd, n * row_bytes, off + i * row_bytes)
            if len(buf) != n * row_bytes:
                raise ValueError("short read")
            arr = np.frombuffer(buf, dtype=dtype).reshape((n,) + tuple(shape[1:]))
            try:
                os.posix_fadvise(fd, off + i * row_bytes, n * row_bytes, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
            yield i, arr
    finally:
        os.close(fd)


def _origins(path):
    with np.load(path, allow_pickle=False) as z:
        ids = z["row_ids"].astype(str).tolist()
    return np.array([int(r.split(":")[1][3:]) for r in ids], dtype=np.int64)


def reconstruct_rows(train_path, val_path, horizons=24, window=24):
    """Exact train-standardized rows from windows (rows o-23..o) and targets (rows o+1..o+H); NaN where not covered.
    Every cell is written from every window and every target that contains it, and every overlap must agree bitwise."""
    origins = {"train": _origins(train_path), "validation": _origins(val_path)}
    end = int(origins["validation"].max()) + horizons + 1
    C = _member(train_path, "windows")[1][2]
    rows = np.full((end, C), np.nan)

    def put(r, vals):
        prev = rows[r]
        seen = ~np.isnan(prev)
        if not np.array_equal(prev[seen], vals[seen]):
            raise ValueError("overlapping windows/targets disagree")
        rows[r] = vals

    for split, path in (("train", train_path), ("validation", val_path)):
        o = origins[split]
        for i, W in iter_member(path, "windows"):
            oo = o[i:i + len(W)]
            for j in range(window):
                put(oo - window + 1 + j, W[:, j, :].astype(np.float64))
        for i, T in iter_member(path, "targets"):
            oo = o[i:i + len(T)]
            for k in range(horizons):
                put(oo + k + 1, T[:, k, :].astype(np.float64))
    return rows, origins


def panel(rows, origins, train_rows=(0, 18412), horizons=24, window=24):
    return {"Z": rows, "origins": origins, "train_rows": list(train_rows), "H": horizons, "L": window,
            "C": rows.shape[1], "rows_sha256": hashlib.sha256(np.nan_to_num(rows, nan=-1e300).astype("<f8").tobytes()).hexdigest()}


# ---------------------------------------------------------------------------------------------- targets
def residual_targets(Z, o, H=24):
    return np.stack([Z[o + h] - Z[o + h - 24] for h in range(1, H + 1)], axis=1)       # (N, H, C)


def seasonal_naive(Z, o, H=24):
    return np.stack([Z[o + h - 24] for h in range(1, H + 1)], axis=1)


def persistence_naive(Z, o, H=24):
    return np.repeat(Z[o][:, None, :], H, axis=1)


# ---------------------------------------------------------------------------------------------- features
def channel_features(blocks, o, lags=24):
    """(len(o) * C, len(blocks) * lags): row index = origin_index * C + channel; columns block-major, lag-minor."""
    C = blocks[0].shape[1]
    cols = []
    for M in blocks:
        for l in range(lags):
            cols.append(M[o - l])                             # (n, C)
    X = np.stack(cols, axis=2)                                 # (n, C, P)
    return X.reshape(len(o) * C, len(cols))


# ---------------------------------------------------------------------------------------------- pooled ridge
def pooled_ridge(feats, tgt, o_train, alphas=(1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7), block=256, holdout_frac=0.2, purge=48):
    """Ridge over very many rows, accumulated block by block. Standardisation: TRAIN column statistics. Alpha chosen on
    a chronological inner hold-out of the train origins (purged), then refit on all train origins (incl. the purge)."""
    n = len(o_train)
    cut = int(n * (1 - holdout_frac))
    inner, hold = o_train[:max(cut - purge, 1)], o_train[cut:]
    # pass 1: column statistics over all train rows
    s = s2 = None
    cnt = 0
    for i in range(0, n, block):
        X = feats(o_train[i:i + block])
        s = X.sum(0) if s is None else s + X.sum(0)
        s2 = (X * X).sum(0) if s2 is None else s2 + (X * X).sum(0)
        cnt += len(X)
    mu = s / cnt
    sd = np.sqrt(np.maximum(s2 / cnt - mu * mu, 0.0))
    sd[sd == 0] = 1.0

    def gram(orig):
        G = R = ysum = None
        m = 0
        for i in range(0, len(orig), block):
            oo = orig[i:i + block]
            A = (feats(oo) - mu) / sd
            Y = tgt(oo)
            G = A.T @ A if G is None else G + A.T @ A
            R = A.T @ Y if R is None else R + A.T @ Y
            ysum = Y.sum(0) if ysum is None else ysum + Y.sum(0)
            m += len(A)
        return G, R, ysum, m

    G, R, ys, m = gram(inner)
    ym = ys / m
    # centring the target: R - colsum(A)^T ym ; colsum(A) over the inner rows
    asum = None
    for i in range(0, len(inner), block):
        a = ((feats(inner[i:i + block]) - mu) / sd).sum(0)
        asum = a if asum is None else asum + a
    Rc = R - np.outer(asum, ym)
    Gc = G - np.outer(asum, asum) / m
    w, V = np.linalg.eigh(Gc)
    VtR = V.T @ Rc
    abs_err = {a: 0.0 for a in alphas}
    cnt = 0
    amean = asum / m
    for i in range(0, len(hold), block):
        oo = hold[i:i + block]
        AV = ((feats(oo) - mu) / sd - amean) @ V
        Y = tgt(oo)
        for a in alphas:
            P = AV @ (VtR / (w[:, None] + a)) + ym
            abs_err[a] += float(np.abs(P - Y).sum())
        cnt += Y.size
    inner_mae = {a: abs_err[a] / cnt for a in alphas}
    best = min(alphas, key=lambda a: (inner_mae[a], a))
    G, R, ys, m = gram(o_train)
    ym = ys / m
    asum = None
    for i in range(0, n, block):
        a = ((feats(o_train[i:i + block]) - mu) / sd).sum(0)
        asum = a if asum is None else asum + a
    amean = asum / m
    Gc = G - np.outer(asum, asum) / m
    Rc = R - np.outer(asum, ym)
    W = np.linalg.solve(Gc + best * np.eye(len(Gc)), Rc)
    # predictions use (x - mu)/sd - amean as the centred design; with standardisation on all train rows amean ~ 0
    return {"alpha": best, "inner_holdout_MAE": {str(a): v for a, v in inner_mae.items()}, "W": W, "mu": mu + amean * sd,
            "sd": sd, "ym": ym}


def ridge_predict(fit, feats, o, block=256):
    out = []
    for i in range(0, len(o), block):
        out.append(((feats(o[i:i + block]) - fit["mu"]) / fit["sd"]) @ fit["W"] + fit["ym"])
    return np.concatenate(out)


# ---------------------------------------------------------------------------------------------- Kalman + arms
def kalman_blocks(d, variant):
    Z = d["Z"]
    t0, t1 = d["train_rows"]
    nan_rows = np.where(np.isnan(Z[t0:t1]).any(axis=1))[0]
    t1_fit = t0 + (int(nan_rows[0]) if len(nan_rows) else t1 - t0)
    names = [f"ch{j}" for j in range(d["C"])]
    spec = kf.default_spec(kf.LOCAL_LEVEL, **variant.get("local_level", {}))
    art = kf.fit(spec, Z[t0:t1_fit], {"dataset_id": "ecl_m04_v2", "role": "TRAIN", "row_range": [t0, t1_fit],
                                       "column_ids": names, "units": "z_train"})
    if art["status"] != "FITTED":
        raise kf.OperatorAbstain(art["abstain_reason"])
    out = kf.transform_batch(art, Z)
    return {"artifact": art, "output": out, "fit_rows": [t0, t1_fit]}


def arm_blocks(d, kal):
    Z = d["Z"]
    o = kal["output"]
    lv, inn, zin = o.arrays["level"], o.arrays["innov"], o.arrays["zinnov"]
    with np.errstate(divide="ignore", invalid="ignore"):
        lvar = np.log(o.arrays["state_var"])
    K = [lv, inn, zin, lvar]
    n_tr = d["train_rows"][1]
    stackK = np.concatenate(K, axis=1)
    P = kf.permutation_control(np.nan_to_num(stackK), [n_tr], seed=20261001)
    N = kf.noise_control(np.nan_to_num(stackK), n_tr, seed=20261002)
    C = d["C"]
    split = lambda M: [M[:, i * C:(i + 1) * C] for i in range(4)]
    e = kf.ewma_comparable(kal["artifact"], Z)
    arms = {"A": {"blocks": [Z], "eligible": True}, "B": {"blocks": [Z] + K, "eligible": True},
            "C": {"blocks": K, "eligible": True}, "C_EWMA": {"blocks": [e["level"], e["innov"]], "eligible": True},
            "B_PERMUTED": {"blocks": [Z] + split(P), "eligible": True}, "B_NOISE": {"blocks": [Z] + split(N), "eligible": True}}
    if not np.isnan(Z).any():
        sm = kf.smoother_control(kal["artifact"], Z)
        arms["C_SMOOTHER_NONCAUSAL"] = {"blocks": [sm.arrays["level"]], "eligible": False}
    else:
        arms["C_SMOOTHER_NONCAUSAL"] = {"blocks": [lv], "eligible": False, "note": "smoother needs complete rows; placeholder"}
    return arms


def score(Z, o, pred_res, H=24):
    """Per-horizon MAE/MSE (mean over channels) of y_hat = seasonal + residual, beside the seasonal, persistence and
    zero-residual naives on the same rows."""
    Y = np.stack([Z[o + h] for h in range(1, H + 1)], axis=1)
    S = seasonal_naive(Z, o, H)
    Pn = persistence_naive(Z, o, H)
    yhat = S + pred_res
    rows = []
    for k in range(H):
        e, es, ep = yhat[:, k] - Y[:, k], S[:, k] - Y[:, k], Pn[:, k] - Y[:, k]
        rows.append({"horizon": k + 1, "model_MAE": float(np.abs(e).mean()), "model_MSE": float((e ** 2).mean()),
                     "seasonal_MAE": float(np.abs(es).mean()), "seasonal_MSE": float((es ** 2).mean()),
                     "persistence_MAE": float(np.abs(ep).mean()), "persistence_MSE": float((ep ** 2).mean())})
    return rows, np.abs(yhat - Y).mean(axis=2)                # per-origin per-horizon MAE (N, H) for pairing
