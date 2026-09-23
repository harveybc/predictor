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
    "rule": "per horizon and for the four-horizon average (formed within each seed first), the three-seed mean of the replicated metric is in "
            "OPERATIONAL_AGREEMENT with the published value when |mean - published| <= 2 x std_paper + 0.0005 (rounding half-unit); "
            "OPERATIONAL_PARTIAL when <= 3 x std_paper + 0.0005; OUTSIDE_OPERATIONAL_MARGIN otherwise. std_paper is the paper's std over its "
            "three runs of the FOUR-HORIZON AVERAGE (Table 7), borrowed per horizon as a predeclared operational margin — not a published "
            "per-horizon error bar, not statistical equivalence; the replicated seed dispersion is reported beside it and never replaces the criterion",
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

def _dontneed(fd: int, offset: int, length: int) -> None:
    """Tell the kernel the pages of [offset, offset+length) of this file are not needed again (they stay on disk): a bounded
    evaluation must not let file-backed pages pile up until the host's memory-pressure killer acts (gamma, 2026-09-22)."""
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        pos = 0
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block); pos += len(block)
            if pos % (1 << 28) < (1 << 22):
                _dontneed(fh.fileno(), 0, pos)
        _dontneed(fh.fileno(), 0, pos)
    return h.hexdigest()


def _npy_header_len(path: Path) -> int:
    """Offset of the array bytes inside a .npy file (public API: the memmap's own offset)."""
    m = np.load(path, mmap_mode="r"); off = int(m.offset); del m
    return off


def stream_npz(dest: Path, member: str, src_npy: Path) -> dict:
    """Write an UNCOMPRESSED .npz holding `member` from an existing .npy file, streaming in 64 MiB pieces with the page cache
    released behind both files; returns the sha256 of the member's array bytes (header excluded) and of the whole npz."""
    import zipfile
    header_len = _npy_header_len(src_npy)
    body_hash = hashlib.sha256()
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf, open(src_npy, "rb") as src:
        with zf.open(member + ".npy", "w", force_zip64=True) as out:
            src.seek(0); written = 0
            while True:
                block = src.read(1 << 26)
                if not block:
                    break
                out.write(block)
                if written + len(block) > header_len:
                    start = max(0, header_len - written)
                    body_hash.update(block[start:])
                written += len(block)
                _dontneed(src.fileno(), 0, written)
                try:
                    zf.fp.flush(); _dontneed(zf.fp.fileno(), 0, zf.fp.tell())
                except (AttributeError, OSError):
                    pass
    return {"array_sha256": body_hash.hexdigest(), "npz_sha256": sha_file(dest)}


def float64_metrics_files(preds_npy: Path, trues_npy: Path, shape: tuple, chunk_windows: int = 64) -> dict:
    """MAE/MSE in float64 read directly from two .npy files window-chunk by window-chunk, releasing the page cache behind."""
    W = shape[0]; per_window = int(np.prod(shape[1:])); item = 4
    hp, ht = _npy_header_len(preds_npy), _npy_header_len(trues_npy)
    n, ab, sq = 0, 0.0, 0.0
    with open(preds_npy, "rb") as fp, open(trues_npy, "rb") as ft:
        for w0 in range(0, W, chunk_windows):
            cnt = min(chunk_windows, W - w0) * per_window
            fp.seek(hp + w0 * per_window * item); ft.seek(ht + w0 * per_window * item)
            a = np.frombuffer(fp.read(cnt * item), dtype=np.float32).astype(np.float64); b = np.frombuffer(ft.read(cnt * item), dtype=np.float32).astype(np.float64)
            d = a - b; ab += float(np.abs(d).sum()); sq += float((d * d).sum()); n += d.size
            _dontneed(fp.fileno(), 0, fp.tell()); _dontneed(ft.fileno(), 0, ft.tell())
    return {"mae": ab / n, "mse": sq / n}


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
    row = int(np.prod(preds.shape[1:])) * 8 if len(preds.shape) > 1 else 8
    chunk = max(1, min(chunk, (96 << 20) // row))                        # <= ~96 MB of float64 per operand per chunk
    for i in range(0, preds.shape[0], chunk):
        d = np.asarray(preds[i:i + chunk]).astype(np.float64) - np.asarray(trues[i:i + chunk]).astype(np.float64)
        ab += float(np.abs(d).sum()); sq += float((d * d).sum()); n += d.size
    return {"mae": ab / n, "mse": sq / n}


class StoredArray:
    """A stored array read chunk by chunk along its first axis straight from its file — a .npy, or a ZIP_STORED member of an
    .npz — with the pages released behind each read. RP101 applied to the closure: a T=720 cell's predictions are 4.2 GB and
    the closure (finite check, catalog, float64 reduction, replay comparison) must never hold a whole array, let alone a float64
    copy of two. `a[w0:w1]` returns an ordinary in-memory chunk; `memmap()` is for the author's own reduction within budget."""

    def __init__(self, path: Path, member: str | None = None):
        import struct
        self.path = Path(path); self.member = member
        self.compressed = False; self._zip_name = None; self._stream = None; self._stream_row = 0
        base = 0
        if member is not None:
            import zipfile
            with zipfile.ZipFile(self.path) as zf:
                names = zf.namelist()
                self._zip_name = member if member in names else member + ".npy"
                info = zf.getinfo(self._zip_name)
            if info.compress_type != zipfile.ZIP_STORED:
                # the cells written before the owner's no-compression decision (savez_compressed): the member is decompressed as a
                # forward-only stream, row chunk by row chunk; never inflated whole
                self.compressed = True
            else:
                with open(self.path, "rb") as fh:
                    fh.seek(info.header_offset); lh = fh.read(30)
                    if lh[:4] != b"PK\x03\x04":
                        raise SotaRefusal(f"REFUSED: {self.path.name}:{member} has no local file header at its recorded offset")
                    name_len, extra_len = struct.unpack("<HH", lh[26:30])
                base = info.header_offset + 30 + name_len + extra_len
        if self.compressed:
            fh = self._open_stream()
            shape, fortran, dtype = self._read_header(fh)
            self.offset = None
            self._stream, self._stream_row = fh, 0
        else:
            with open(self.path, "rb") as fh:
                fh.seek(base)
                shape, fortran, dtype = self._read_header(fh)
                self.offset = fh.tell()
        if fortran or dtype.hasobject:
            raise SotaRefusal(f"REFUSED: {self.path.name} is not a C-ordered numeric array")
        self.shape, self.dtype = tuple(int(x) for x in shape), np.dtype(dtype)
        self.ndim = len(self.shape); self.size = int(np.prod(self.shape)) if self.shape else 1
        self.row_nbytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize if self.ndim else self.dtype.itemsize
        self.nbytes = self.size * self.dtype.itemsize
        self._fd = None

    @staticmethod
    def _read_header(fh):
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(fh)
        if version == (2, 0):
            return np.lib.format.read_array_header_2_0(fh)
        return np.lib.format._read_array_header(fh, version)

    def _open_stream(self):
        import zipfile
        self._zf = zipfile.ZipFile(self.path)
        return self._zf.open(self._zip_name, "r")

    def _read_compressed(self, w0: int, n: int) -> bytes:
        """Rows [w0, w0+n) of a compressed member: forward-only; a slice behind the cursor reopens the stream."""
        if self._stream is None or w0 < self._stream_row:
            self.close()
            fh = self._open_stream(); self._read_header(fh); self._stream, self._stream_row = fh, 0
        while self._stream_row < w0:                                  # skip forward in bounded pieces
            skip = min(w0 - self._stream_row, max(1, (64 << 20) // max(1, self.row_nbytes)))
            got = self._stream.read(skip * self.row_nbytes)
            if len(got) != skip * self.row_nbytes:
                raise SotaRefusal(f"REFUSED: {self.path.name}:{self.member} ended before window {w0} (short member)")
            self._stream_row += skip
        buf = self._stream.read(n * self.row_nbytes)
        if len(buf) != n * self.row_nbytes:
            raise SotaRefusal(f"REFUSED: {self.path.name}:{self.member} ended before window {w0 + n} (short member)")
        self._stream_row += n
        return buf

    def _fileno(self) -> int:
        if self._fd is None:
            self._fd = os.open(self.path, os.O_RDONLY)
        return self._fd

    def close(self):
        if self._fd is not None:
            os.close(self._fd); self._fd = None
        if self._stream is not None:
            try:
                self._stream.close(); self._zf.close()
            except Exception:
                pass
            self._stream = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return self.shape[0] if self.ndim else 1

    def __getitem__(self, s):
        if isinstance(s, int):
            if s < 0:
                s += len(self)
            return self[s:s + 1][0]
        if not isinstance(s, slice) or s.step not in (None, 1):
            raise TypeError("StoredArray supports contiguous slices along the first axis")
        w0, w1, _ = s.indices(len(self))
        n = max(0, w1 - w0)
        if self.compressed:
            return np.frombuffer(self._read_compressed(w0, n), dtype=self.dtype).reshape((n,) + self.shape[1:])
        fd = self._fileno(); off = self.offset + w0 * self.row_nbytes; length = n * self.row_nbytes
        buf = bytearray(length); view = memoryview(buf); got = 0
        while got < length:
            k = os.preadv(fd, [view[got:]], off + got)
            if k <= 0:
                raise SotaRefusal(f"REFUSED: {self.path.name} ended before window {w1} (short file)")
            got += k
        _dontneed(fd, off, length)
        return np.frombuffer(buf, dtype=self.dtype).reshape((n,) + self.shape[1:])

    def chunk_windows(self, target_bytes: int = 96 << 20) -> int:
        return max(1, target_bytes // max(1, self.row_nbytes))

    def all_finite(self) -> bool:
        step = self.chunk_windows()
        return all(bool(np.isfinite(self[w0:w0 + step]).all()) for w0 in range(0, len(self), step))

    def memmap(self) -> np.memmap:
        if self.compressed:
            raise SotaRefusal(f"REFUSED: {self.path.name}:{self.member} is compressed and cannot be memory-mapped")
        return np.memmap(self.path, dtype=self.dtype, mode="r", offset=self.offset, shape=self.shape)

    def load(self) -> np.ndarray:
        return self[0:len(self)]


AUTHOR_SCORER_ROUTE = "df_sota_author_metric_exact.v1"
_PAIRWISE_LEAF = 1 << 22                                        # elements per leaf reduced by numpy itself (16 MB of float32)


def _pairwise_f32(leaf_sum, lo: int, n: int, leaf: int = _PAIRWISE_LEAF):
    """numpy's pairwise summation tree (numpy/core/src/umath/loops_utils.h.src `pairwise_sum`): below `leaf` the range is reduced
    by numpy's own inner loop (which applies the same tree), above it the range splits at n//2 rounded down to a multiple of 8
    and the two partial sums are added once in float32. Verified bit-equal to np.add.reduce for n = 7 … 6.3e6 and leaves 128 …
    2^20 (numpy 2.5.1) and on a 3-D contiguous array reduced with axis=None."""
    if n <= leaf:
        return np.float32(leaf_sum(lo, n))
    n2 = n // 2; n2 -= n2 % 8
    return np.float32(_pairwise_f32(leaf_sum, lo, n2, leaf) + _pairwise_f32(leaf_sum, lo + n2, n - n2, leaf))


def author_metric_exact(preds, trues, *, leaf: int = _PAIRWISE_LEAF) -> dict:
    """The author's float32 MAE/MSE (utils.metrics: np.mean(np.abs(true - pred)), np.mean((true - pred) ** 2) on C-contiguous
    float32 arrays) reproduced BIT FOR BIT through a bounded route: the element-wise differences are formed in float32 chunk by
    chunk (element-wise ops are exact per element), numpy's pairwise summation tree is replicated over the flattened element index
    space with leaves reduced by numpy, and the mean is float32(sum) / float32(N) as numpy's `_mean` does. RP109: a mean of chunk
    means is NOT this; this is the same arithmetic in the same order. `preds`/`trues` are arrays or StoredArray-like objects of
    the same (W, T, C) shape read by contiguous window slices."""
    shape = tuple(preds.shape)
    if shape != tuple(trues.shape) or len(shape) < 1:
        raise SotaRefusal(f"REFUSED: author metric on shapes {shape} vs {tuple(trues.shape)}")
    re_ = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    N = int(shape[0]) * re_
    cache = {"w0": None, "w1": None, "d_abs": None, "d_sq": None}

    def rows(w0, w1):
        if cache["w0"] != w0 or cache["w1"] != w1:
            a = np.asarray(preds[w0:w1], dtype=np.float32); b = np.asarray(trues[w0:w1], dtype=np.float32)
            d = (b - a).reshape(-1)                                       # true - pred, float32, element-wise exact
            cache.update(w0=w0, w1=w1, d_abs=np.abs(d), d_sq=d * d)
        return cache["d_abs"], cache["d_sq"]

    def leaf_sum(kind):
        def f(lo, n):
            w0, w1 = lo // re_, (lo + n - 1) // re_ + 1
            d_abs, d_sq = rows(w0, w1)
            off = lo - w0 * re_
            return np.add.reduce((d_abs if kind == "abs" else d_sq)[off:off + n], dtype=np.float32)
        return f
    if N == 0:
        raise SotaRefusal("REFUSED: author metric on an empty population")
    s_abs = _pairwise_f32(leaf_sum("abs"), 0, N, leaf)
    cache.update(w0=None)                                              # second pass reads again (the leaf order is identical)
    s_sq = _pairwise_f32(leaf_sum("sq"), 0, N, leaf)
    n32 = np.float32(N)
    return {"mae": float(np.float32(s_abs / n32)), "mse": float(np.float32(s_sq / n32)), "route": AUTHOR_SCORER_ROUTE, "elements": N, "leaf": leaf,
            "dtype": "float32", "reduction": "numpy pairwise tree replicated over the flattened C-order element index; mean = float32(sum)/float32(N)"}


def all_finite(a, step: int = 128) -> bool:
    """Finite check over chunks of windows for any array-like (memmap, StoredArray, ndarray)."""
    return all(bool(np.isfinite(np.asarray(a[w0:w0 + step])).all()) for w0 in range(0, a.shape[0], step))


def compare_predictions(rep, stored, *, atol: float, rtol: float, step: int = 64) -> dict:
    """The frozen replay rule evaluated chunk by chunk in float64 (|rep - stored| <= atol + rtol*|stored| everywhere), with the
    maximum absolute difference and the exact-equality count; no full-size temporary."""
    max_abs, n_exact, close, n_total = 0.0, 0, True, int(np.prod(stored.shape))
    for w0 in range(0, stored.shape[0], step):
        a = np.asarray(rep[w0:w0 + step], dtype=np.float32).astype(np.float64); b = np.asarray(stored[w0:w0 + step], dtype=np.float32).astype(np.float64)
        d = np.abs(a - b); max_abs = max(max_abs, float(d.max()) if d.size else 0.0); n_exact += int((a == b).sum())
        close = close and bool((d <= atol + rtol * np.abs(b)).all())
    return {"max_abs_prediction_difference": max_abs, "allclose_rule": close, "exact_equal_elements": n_exact, "elements": n_total,
            "exact_equal_fraction": (n_exact / n_total) if n_total else None}


MALLOC_TUNABLES_DEFAULT = "glibc.malloc.mmap_threshold=1048576:glibc.malloc.trim_threshold=2097152"


def child_env(malloc_tunables: str | None = MALLOC_TUNABLES_DEFAULT, **extra) -> dict:
    """The environment of a cell or replay process. RP102 (T=720 on WORKER_B): the author's validation moves every batch's outputs
    and targets to the host to score them; with glibc's dynamic mmap threshold those 15 MB buffers are carved from the heap and
    the freed pages stay resident, so a 720-step validation over 404 batches grew the process by ~5 GB and pinned it at the
    slice ceiling (throttled, GPU idle). Fixing the thresholds returns each freed buffer to the OS: host allocator only, no
    arithmetic, no batch, no schedule is touched (declared in the record as an operational patch)."""
    env = {**os.environ, **{k: str(v) for k, v in extra.items()}}
    if malloc_tunables and malloc_tunables.lower() != "none":
        env["GLIBC_TUNABLES"] = malloc_tunables
    return env


def host_allocator_patch() -> list:
    t = os.environ.get("GLIBC_TUNABLES")
    return ([{"what": f"GLIBC_TUNABLES={t} (host malloc thresholds fixed)", "effect": "none on arithmetic: freed host buffers of the author's "
              "validation/evaluation return to the OS instead of staying resident; host memory only"}] if t else [])


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


def actual_device_uuid(index: int = 0) -> str | None:
    """The UUID of the CUDA device a process actually computes on. nvidia-smi lists every GPU of the host in its own order, so
    `gpu_state()[cuda_index]` is not the device when CUDA_VISIBLE_DEVICES restricts the process (gamma: the RTX 5090 is
    nvidia-smi index 1 and cuda:0 of the cell process). torch's own device properties first, then the visibility mask, then
    nvidia-smi's order as a last resort."""
    try:
        import torch
        if torch.cuda.is_available():
            u = getattr(torch.cuda.get_device_properties(index), "uuid", None)
            if u:
                u = str(u)
                return u if u.startswith("GPU-") else f"GPU-{u}"
    except Exception:                                               # noqa: BLE001
        pass
    vis = [x.strip() for x in (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if x.strip()]
    if vis and all(x.startswith("GPU-") for x in vis):
        return vis[index] if index < len(vis) else vis[0]
    g = gpu_state()
    return (g[index] if index < len(g) else (g[0] if g else {})).get("uuid") if g else None


def trained_device_of(record: dict) -> str | None:
    """The device UUID a cell was trained on, from its record: the recorded actual UUID (records from 2026-09-22 on), else the
    visibility mask the record's environment carried, else nvidia-smi's first GPU of the training host (older records)."""
    if not str(record.get("device", "")).startswith("cuda"):
        return None
    if record.get("device_uuid"):
        return record["device_uuid"]
    vis = [x.strip() for x in str((record.get("environment") or {}).get("cuda_visible_devices") or "").split(",") if x.strip()]
    if vis and all(x.startswith("GPU-") for x in vis):
        return vis[0]
    # older records: the author's Exp overwrites CUDA_VISIBLE_DEVICES with its own index after the CUDA context exists, so the
    # recorded mask is not evidence; the device whose memory the process occupied between gpu_before and gpu_after is
    cost = record.get("cost") or {}
    before = {g.get("uuid"): g for g in cost.get("gpu_before") or []}
    after = {g.get("uuid"): g for g in cost.get("gpu_after") or []}
    growth = {u: float(after[u].get("memory_used_mib") or 0) - float(before.get(u, {}).get("memory_used_mib") or 0) for u in after}
    if growth and max(growth.values()) >= 256:
        return max(growth, key=growth.get)
    return next(iter(before)) if before else None


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


def build_args(argv: list, *, data_dir: Path, data_name: str, checkpoints: Path, gpu: int = 0, use_gpu: bool | None = None, dataloader_workers: int | None = None):
    """run.py's parse and post-parse logic, verbatim in effect; the data root and checkpoint folder are the only transport values.
    `dataloader_workers` (operational, declared): how many DataLoader processes fetch batches — the sampler's order and every
    batch's content are produced in the main process and do not depend on it (tested); it changes host memory only."""
    import torch
    parser, _ = author_parser()
    args = parser.parse_args(list(argv))
    args.root_path, args.data_path, args.checkpoints, args.gpu = str(data_dir), data_name, str(checkpoints), int(gpu)
    if dataloader_workers is not None:
        args.num_workers = int(dataloader_workers)
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
                     log: Path | None = None, train: bool = True, bounded: bool = False, author_metric_budget_bytes: int | None = None,
                     dataloader_workers: int | None = None) -> dict:
    """run.py's `__main__` for one invocation: seeds, parse, Exp, train, test — the author's code path, in the author's cwd layout."""
    author_env()
    fix_seeds(seed)
    args = build_args(argv, data_dir=data_dir, data_name=data_name, checkpoints=work / "checkpoints", gpu=gpu, use_gpu=use_gpu, dataloader_workers=dataloader_workers)
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
            bounded_out = None
            if bounded:
                bounded_out = bounded_test(exp, setting, work, author_metric_budget_bytes=author_metric_budget_bytes)
            else:
                exp.test(setting, test=0 if train else 1)
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        os.chdir(cwd)
        exp_module.metric = original
    base = {"args": vars(args), "setting": setting, "wall_seconds": time.time() - t0, "cpu_seconds": time.process_time() - c0,
            "operational_patches": ([{"what": f"DataLoader num_workers {dataloader_workers} instead of the author default", "effect": "none: sampler order and batch contents are produced in the main process (tested); host memory only"}] if dataloader_workers is not None else []) + host_allocator_patch(),
            "checkpoint": work / "checkpoints" / setting / "checkpoint.pth", "n_parameters": int(sum(p.numel() for p in exp.model.parameters())), "device": str(exp.device)}
    if bounded_out is not None:
        return {**base, "preds": bounded_out["preds"], "trues": bounded_out["trues"], "author_metric": bounded_out["author_metric"],
                "author_metric_state": bounded_out["author_metric_state"], "independent_metric_float64": bounded_out["independent_metric_float64"],
                "bounded": {k: bounded_out[k] for k in ("finalized", "adapter", "preds_path", "trues_path", "shape")}}
    mae, mse = float(captured.value[0]), float(captured.value[1])
    return {**base, "preds": captured.preds, "trues": captured.trues, "author_metric": {"mae": mae, "mse": mse}, "author_metric_state": "EXECUTED: the author's test()"}


BOUNDED_ADAPTER_VERSION = "df_sota_bounded_eval.v1"


def bounded_test(exp, setting: str, work: Path, *, author_metric_budget_bytes: int | None = None) -> dict:
    """RP101: the author's test() with its accumulation replaced by disk-backed buffers — same model, weights, loader, batches,
    order, dtype and slicing as `Exp_Long_Term_Forecast.test()`; predictions and targets stream into float32 .npy memmaps
    under `work` instead of Python lists (the author keeps three full lists and concatenates them: ~4x the arrays in RAM),
    inputs are not retained (they are unused by the author's scorer), targets are hash-streamed. The author's `metric()`
    (float32, full arrays) is then evaluated on the memmaps when its temporaries fit `author_metric_budget_bytes`; the
    streaming float64 reduction is always computed. Nothing about the forward pass changes; the adapter's source digest is
    recorded with every cell that used it (an adapter is not an unmodified author implementation)."""
    import torch
    from numpy.lib.format import open_memmap
    test_data, test_loader = exp._get_data(flag="test")
    n_windows = len(test_data)
    checkpoint = Path(exp.args.checkpoints) / setting / "checkpoint.pth"
    if not checkpoint.is_file():
        raise SotaRefusal(f"REFUSED: no checkpoint to evaluate at {checkpoint}")
    exp.model.load_state_dict(torch.load(checkpoint, map_location=exp.device))     # the author's test(test=1) reload, on the same device
    preds_path, trues_path = work / "bounded_preds.npy", work / "bounded_trues.npy"
    preds = trues = None
    fp = ft = None; header_off = None
    true_hash = hashlib.sha256()
    f_dim = -1 if exp.args.features == "MS" else 0
    n_seen, batches = 0, []
    exp.model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            outputs, _ = exp.model(batch_x, exp.masks, is_training=False)
            outputs = outputs[:, -exp.args.pred_len:, :]
            batch_y = batch_y[:, -exp.args.pred_len:, :].to(exp.device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            outputs = outputs[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:]
            if fp is None:
                shape = (n_windows, outputs.shape[1], outputs.shape[2])
                # the files are created with their .npy headers (the memmap only writes the header), then filled through plain file
                # handles so that every written page can be released from the cache at once; nothing is held in RAM
                m = open_memmap(preds_path, mode="w+", dtype=np.float32, shape=shape); header_off = m.offset; del m
                m = open_memmap(trues_path, mode="w+", dtype=np.float32, shape=shape); del m
                fp = open(preds_path, "r+b"); ft = open(trues_path, "r+b"); fp.seek(header_off); ft.seek(header_off)
            b = outputs.shape[0]
            if n_seen + b > n_windows:
                raise SotaRefusal(f"REFUSED: the loader yielded more windows ({n_seen + b}+) than the dataset declares ({n_windows})")
            ob = np.ascontiguousarray(outputs.astype(np.float32, copy=False)); yb = np.ascontiguousarray(batch_y.astype(np.float32, copy=False))
            fp.write(memoryview(ob).cast("B")); ft.write(memoryview(yb).cast("B"))
            true_hash.update(memoryview(yb).cast("B"))
            batches.append(b); n_seen += b
            if len(batches) % 16 == 0:
                fp.flush(); ft.flush(); _dontneed(fp.fileno(), 0, fp.tell()); _dontneed(ft.fileno(), 0, ft.tell())
    if fp is not None:
        fp.flush(); ft.flush(); _dontneed(fp.fileno(), 0, fp.tell()); _dontneed(ft.fileno(), 0, ft.tell()); fp.close(); ft.close()
    if n_seen != n_windows:
        raise SotaRefusal(f"REFUSED: the loader yielded {n_seen} windows, the dataset declares {n_windows}: INCOMPLETE POPULATION")
    preds = np.load(preds_path, mmap_mode="r"); trues = np.load(trues_path, mmap_mode="r")
    finalized = {"windows": n_seen, "batches": len(batches), "batch_sizes": {"first": batches[0], "last": batches[-1], "distinct": sorted(set(batches))},
                 "true_sha256": true_hash.hexdigest(), "preds_bytes": int(preds.nbytes)}
    # RP109: the author's float32 reduction through the exact bounded route (same arithmetic, same order, ~64 MB of temporaries);
    # the author's own function runs beside it only when its full-size temporaries fit, as parity evidence for this cell
    need = 2 * int(preds.nbytes) + int(preds.nbytes)
    exact = author_metric_exact(StoredArray(preds_path), StoredArray(trues_path))
    author_metric = {"mae": exact["mae"], "mse": exact["mse"]}
    author_metric_state = f"EXECUTED: {AUTHOR_SCORER_ROUTE} (numpy pairwise float32 replica over the whole population, bit-equal to utils.metrics.metric by construction and by test)"
    author_scorer_parity = None
    if author_metric_budget_bytes is None or need <= author_metric_budget_bytes:
        MET = importlib.import_module("utils.metrics")
        mae, mse, rmse, mape, mspe = MET.metric(np.asarray(preds), np.asarray(trues))
        author_scorer_parity = {"author_function": {"mae": float(mae), "mse": float(mse)}, "bit_equal": bool(float(mae) == exact["mae"] and float(mse) == exact["mse"])}
        if not author_scorer_parity["bit_equal"]:
            raise SotaRefusal(f"REFUSED: the exact bounded route disagrees with utils.metrics.metric on this cell: {author_scorer_parity}")
        author_metric_state += "; utils.metrics.metric executed on the memmapped arrays as well: bit-equal"
    else:
        author_scorer_parity = {"author_function": None, "why": f"utils.metrics.metric not executed here: needs ~{need} bytes of temporaries, budget {author_metric_budget_bytes}"}
    f64 = float64_metrics_files(preds_path, trues_path, preds.shape)
    return {"preds": preds, "trues": trues, "preds_path": preds_path, "trues_path": trues_path, "finalized": finalized, "shape": list(preds.shape),
            "author_metric": author_metric, "author_metric_state": author_metric_state, "author_scorer_parity": author_scorer_parity, "author_metric_route": exact,
            "independent_metric_float64": f64,
            "adapter": {"version": BOUNDED_ADAPTER_VERSION, "source_sha256": hashlib.sha256(__import__("inspect").getsource(bounded_test).encode()).hexdigest(),
                        "author_test_untouched": True, "replaces": "Exp_Long_Term_Forecast.test(): list accumulation + np.concatenate + result file; forward pass identical"}}


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

def run_cell(design: dict, cell: dict, *, data_path: Path, folder: Path, gpu: int = 0, use_gpu: bool | None = None, bounded: bool = False,
             author_metric_budget_bytes: int | None = None, dataloader_workers: int | None = None) -> dict:
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
                           log=folder / "author_stdout.log", bounded=bounded, author_metric_budget_bytes=author_metric_budget_bytes, dataloader_workers=dataloader_workers)
    if res.get("bounded"):
        # the artifact is streamed from the prediction file (uncompressed npz, the owner's decision); nothing is materialized in RAM
        shape = tuple(res["bounded"]["shape"])
        streamed = stream_npz(folder / "arrays.npz", "pred", Path(res["bounded"]["preds_path"]))
        pred_sha, trues_sha, true_shape, pred_shape = streamed["array_sha256"], res["bounded"]["finalized"]["true_sha256"], list(shape), list(shape)
        preds = None
        for tmp in (res["bounded"]["preds_path"], res["bounded"]["trues_path"]):
            try:
                Path(tmp).unlink()
            except OSError:
                pass
    else:
        preds, trues = np.asarray(res["preds"], dtype=np.float32), np.asarray(res["trues"], dtype=np.float32)
        np.savez(folder / "arrays.npz", pred=preds)                    # uncompressed by the owner's decision (float32 outputs compress < 13 %)
        trues_sha = sha_array(trues); true_shape = list(trues.shape); pred_sha = sha_array(preds); pred_shape = list(preds.shape)
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
    f64 = res["independent_metric_float64"] if res.get("bounded") else float64_metrics(preds, trues)
    dtype_name = "float32"
    training = parse_author_log((folder / "author_stdout.log").read_text())
    record = {"schema": "df_sota_cell_record.v1", "cell": {k: cell[k] for k in ("cell_id", "arm", "protocol", "seq_len", "horizon", "seed")},
              "design_sha256": design["design_sha256"], "setting": res["setting"], "effective_args": {k: v for k, v in sorted(res["args"].items())},
              "author_metric_float32": res["author_metric"], "author_metric_state": res.get("author_metric_state"), "independent_metric_float64": f64,
              "evaluation_path": ({"bounded_adapter": res["bounded"]["adapter"], "finalized": res["bounded"]["finalized"]} if res.get("bounded") else {"author_test": True}),
              "operational_patches": res.get("operational_patches", []),
              "shapes": {"pred": pred_shape, "true": true_shape}, "dtype": dtype_name,
              "arrays_sha256": sha_file(folder / "arrays.npz"), "pred_sha256": pred_sha, "true_sha256": trues_sha, "checkpoint_sha256": sha_file(ckpt),
              "checkpoint_bytes": ckpt.stat().st_size, "n_parameters": res["n_parameters"], "device": res["device"],
              "device_uuid": (actual_device_uuid(int(str(res["device"]).split(":")[-1]) if ":" in str(res["device"]) else 0) if str(res["device"]).startswith("cuda") else None),
              "training": training,
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
        record = run_cell(design, cell, data_path=path, folder=folder, gpu=a.gpu, use_gpu=None if not a.cpu else False, bounded=bool(getattr(a, "bounded", False)),
                          author_metric_budget_bytes=(int(a.author_metric_budget_gib * 2 ** 30) if getattr(a, "author_metric_budget_gib", None) else None),
                          dataloader_workers=getattr(a, "dataloader_workers", None))
    except BaseException as exc:
        (folder).mkdir(parents=True, exist_ok=True)
        (folder / "FAILED.json").write_text(json.dumps({"at": now_iso(), "error": f"{type(exc).__name__}: {str(exc)[:600]}"}))
        G.report_failed(root, a.unit, f"cell failed: {type(exc).__name__}: {str(exc)[:160]}", gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(root / "outbox"))
        raise
    return report_unit(a, design, cell, record, started=started)


def report_unit(a, design: dict, cell: dict, record: dict, *, started=None) -> dict:
    """The unit's COMPLETED terminal from its record: metrics on the record's basis (the author's float32 reduction when it was
    executed, else the labelled float64 reduction), the three artifacts by digest, identity tags."""
    G, U = governance_modules()
    root = Path(a.root); folder = root / "attempts" / cell["cell_id"]
    started = started or U._z(U.now_iso())
    metric, basis = metric_of(record)
    if metric is None:
        raise SotaRefusal(f"REFUSED: the record of {cell['cell_id']} carries no metric")
    metrics = [U._metric("sota.test.mse_normalized", metric["mse"], "z", split="test", horizon=cell["horizon"]),
               U._metric("sota.test.mae_normalized", metric["mae"], "z", split="test", horizon=cell["horizon"]),
               U._metric("sota.train.epochs_run", float(record["training"]["epochs_run"]), "epochs"),
               U._metric("sota.train.best_epoch", float(record["training"]["best_epoch_by_vali"] or 0), "epoch")]
    terminal = U._terminal(status="COMPLETED", reason=None, cost={"wall_seconds": record["cost"]["wall_seconds"], "cpu_seconds": record["cost"]["cpu_seconds"]},
                           metrics=metrics, started=started, finished=U._z(U.now_iso()),
                           tags={"purpose": "SOTA_REPRODUCTION", "classification": "NON_GOVERNING", "phase": "REPRODUCTION", "unit": cell["cell_id"], "role": "forecast",
                                 "arm": cell["arm"], "horizon": str(cell["horizon"]), "seq_len": str(cell["seq_len"]), "seed": str(cell["seed"]),
                                 "design_sha256": design["design_sha256"], "lake": a.lake, "resource": a.resource, "host": socket.gethostname(), "metric_basis": basis[:40]})
    terminal["artifacts"] = [{"role": r, "sha256": sha_file(folder / f), "bytes": (folder / f).stat().st_size}
                             for r, f in (("predictions", "arrays.npz"), ("checkpoint", "checkpoint.pth"), ("record", "cell.json"))]
    (root / "TERMINALS").mkdir(exist_ok=True)
    (root / "TERMINALS" / f"{cell['cell_id']}.json").write_text(json.dumps(terminal, indent=1, default=str))
    reported = G.report_terminal(root, cell["cell_id"], terminal, gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(root / "outbox"), started_at=started)
    if reported["flushed"]["pending"] or reported["flushed"]["failures"]:
        raise SotaRefusal(f"REFUSED: the terminal of {cell['cell_id']} was not accepted: {reported['flushed']['failures']}")
    return record


def thermal_guard(*, cool_below_c: float = 70.0, max_wait_s: int = 1800, poll_s: int = 30) -> dict:
    """Between cells: wait until every visible GPU is below `cool_below_c` (a pause between cells changes nothing in the recipe);
    the device's own target/slowdown policy governs DURING a cell (the driver throttles clocks at its target), and every cell record
    keeps the temperatures before/after. Returns what was observed and how long it waited."""
    waited, samples = 0, []
    while True:
        g = gpu_state(); temps = [x.get("temperature_c") for x in g if x.get("temperature_c") is not None]
        samples.append({"at": now_iso(), "temperatures_c": temps})
        if not temps or max(temps) < cool_below_c or waited >= max_wait_s:
            return {"waited_seconds": waited, "samples": samples[-5:], "cool_below_c": cool_below_c, "proceeded_hot": bool(temps and max(temps) >= cool_below_c)}
        time.sleep(poll_s); waited += poll_s


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
        guard = thermal_guard() if not a.cpu else {"skipped": "cpu"}
        argv = [sys.executable, str(Path(__file__).resolve()), "child", "--root", str(root), "--unit", unit, "--gov-url", a.gov_url, "--api-key-file", str(a.api_key_file),
                "--lake", a.lake, "--resource", a.resource, "--gpu", str(a.gpu)] + (["--cpu"] if a.cpu else []) + (["--run-id", a.run_id] if a.run_id else []) \
               + (["--bounded"] if getattr(a, "bounded", False) else []) + (["--author-metric-budget-gib", str(a.author_metric_budget_gib)] if getattr(a, "author_metric_budget_gib", None) else []) \
               + (["--dataloader-workers", str(a.dataloader_workers)] if getattr(a, "dataloader_workers", None) is not None else [])
        t0 = time.time()
        proc = subprocess.run(argv, capture_output=True, text=True, env=child_env(getattr(a, "malloc_tunables", MALLOC_TUNABLES_DEFAULT)))
        (folder).mkdir(parents=True, exist_ok=True)
        (folder / "child_stderr.log").write_text(proc.stderr[-20000:])
        out.append({"unit": unit, "ok": proc.returncode == 0, "returncode": proc.returncode, "wall_seconds": time.time() - t0, "tail": proc.stderr[-400:], "thermal_guard": guard,
                    "gpu_after": gpu_state()})
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
    # the worker's replay history travels as a dated record (RP100: never an input to a closure's verification; the table cites it
    # as same-device history when the coordinator closes on another device)
    for extra in sorted(source.glob("REPLAYS*.json")):
        shutil.copy2(extra, root / f"REPLAYS_HISTORY.{source.name}.{extra.name}")
        out.setdefault("replay_history_files", []).append(f"REPLAYS_HISTORY.{source.name}.{extra.name}")
    (root / f"MERGE.{source.name}.{int(time.time())}.json").write_text(json.dumps(out, indent=1))
    return out


# --- versioned failed attempts: a retry is a NEW governed unit, the failed one stays on record --------------------------------------

def retire_attempt(root: Path, unit: str, *, reason: str) -> dict:
    """After a FAILED terminal, move the attempt folder, its delivery and its receipt into versioned history so that the next
    execution of the same cell registers a fresh campaign/delivery instead of reusing the closed one. Nothing is deleted."""
    root = Path(root); stamp = f"failed.{int(time.time())}"
    out = {"unit": unit, "stamp": stamp, "reason": reason, "moved": {}}
    folder = root / "attempts" / unit
    if folder.exists():
        dest = root / "attempts" / f"{unit}.{stamp}"
        folder.rename(dest); out["moved"]["attempt"] = str(dest)
    for name, key in (("DELIVERIES.json", "units"), ("TERMINAL_RECEIPTS.json", "units")):
        path = root / name
        if path.is_file():
            doc = json.loads(path.read_text())
            entry = (doc.get(key) or {}).pop(unit, None)
            if entry is not None:
                doc.setdefault("failed_attempts", []).append({"unit": unit, "stamp": stamp, "reason": reason, "retired_at": now_iso(), "entry": entry})
                write_atomic(path, json.dumps(doc, indent=1, default=str)); out["moved"][name] = True
    term = root / "TERMINALS" / f"{unit}.json"
    if term.is_file():
        term.rename(root / "TERMINALS" / f"{unit}.{stamp}.json"); out["moved"]["terminal"] = True
    write_atomic(root / f"RETIRED.{unit}.{stamp}.json", json.dumps(out, indent=1))
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


# --- RP100: route-level diagnostic of a CPU/GPU discrepancy (bounded; no training, no changed top-p) -------------------------------

def _cgroup_memory() -> dict:
    """memory.current / memory.peak of THIS process's cgroup (file-backed pages included), when readable."""
    out = {}
    try:
        cg = Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
        base = Path("/sys/fs/cgroup") / cg.lstrip("/")
        for name in ("memory.current", "memory.peak", "memory.max", "memory.high"):
            f = base / name
            if f.is_file():
                v = f.read_text().strip(); out[name] = int(v) if v.isdigit() else v
        stat = base / "memory.stat"
        if stat.is_file():
            kv = dict(line.split() for line in stat.read_text().splitlines() if " " in line)
            out["anon"] = int(kv.get("anon", 0)); out["file"] = int(kv.get("file", 0))
        out["cgroup"] = str(base)
    except Exception as exc:                                        # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:120]
    return out


def route_trace(root: Path, design: dict, unit: str, *, data_path: Path, windows: list | None = None, n_windows: int = 3, gpu: int = 0) -> dict:
    """Locate WHERE a CPU-vs-GPU prediction discrepancy arises for a cell: the batches holding the most discrepant windows are
    forwarded on both devices with IDENTICAL batch composition, capturing every graph block's top-p routing decision (the
    gating probabilities, their cumulative sums, the distance of the cut to the top-p threshold and the resulting expert mask)
    and the tensors before and after the route; routes and tensors are compared. Evaluation mode: the gating noise is off, so
    a difference is numerical (kernel) or a routing flip at a threshold, never stochastic. Bounded: a few batches, no training."""
    import torch
    author_env()
    cell = next(c for c in design["cells"] if c["cell_id"] == unit)
    folder = root / "attempts" / unit
    with np.load(folder / "arrays.npz") as z:
        stored = z["pred"]
    fix_seeds(cell["seed"])
    args = build_args(cell["argv"], data_dir=data_path.parent, data_name=data_path.name, checkpoints=Path("/nonexistent"), gpu=gpu, use_gpu=False)
    args.augmentation_ratio = 0
    DF = importlib.import_module("data_provider.data_factory")
    test_data, test_loader = DF.data_provider(args, "test")
    bs = int(args.batch_size)
    exp_module = importlib.import_module("exp.exp_long_term_forecasting")
    sd = torch.load(folder / "checkpoint.pth", map_location="cpu")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    exps = {}
    for dev in devices:
        a2 = build_args(cell["argv"], data_dir=data_path.parent, data_name=data_path.name, checkpoints=Path("/nonexistent"), gpu=gpu, use_gpu=(dev == "cuda"))
        with contextlib.redirect_stdout(io.StringIO()):
            e = exp_module.Exp_Long_Term_Forecast(a2)
        e.model.load_state_dict(sd); e.model.eval(); exps[dev] = e
    # 1. which windows: a CPU pass over all batches (bounded to the test set) unless given
    L = importlib.import_module("layers.TimeFilter_layers")
    if windows is None:
        diffs = np.zeros(stored.shape[0])
        with torch.no_grad():
            for i, (bx, by, _, _) in enumerate(test_loader):
                out, _ = exps["cpu"].model(bx.float(), exps["cpu"].masks, is_training=False)
                out = out[:, -args.pred_len:, :].numpy()
                diffs[i * bs:i * bs + out.shape[0]] = np.abs(out.astype(np.float64) - stored[i * bs:i * bs + out.shape[0]].astype(np.float64)).max(axis=(1, 2))
        windows = [int(w) for w in np.argsort(diffs)[::-1][:n_windows]]
        cpu_diff_by_window = {str(w): float(diffs[w]) for w in windows}
    else:
        cpu_diff_by_window = None
    batches = sorted({w // bs for w in windows})
    # 2. hooks on every gating module and graph learner: capture routes and pre/post tensors
    captures = {}
    def hook_gate(dev, block_id):
        def h(module, inputs, output):
            x = inputs[0].detach().float().cpu()
            logits = module.softmax(module.gate(inputs[0])).detach().float().cpu()
            sorted_probs, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > module.top_p
            thr = mask.long().argmax(dim=-1)
            # the cut position per token and how close the cumulative probability at the cut is to top_p
            dist = (cum - module.top_p).abs().min(dim=-1).values
            captures.setdefault(dev, {})[f"gate_block{block_id}"] = {"top_p_mask": output[0].detach().float().cpu(), "probs": logits, "cut_index": thr, "distance_to_threshold": dist,
                                                                    "route_input_sha": hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16]}
        return h
    def hook_learner(dev, block_id):
        def h(module, inputs, output):
            captures.setdefault(dev, {})[f"learner_block{block_id}"] = {"adj_out": output[0].detach().float().cpu() if isinstance(output, tuple) else output.detach().float().cpu()}
        return h
    handles = []
    for dev, e in exps.items():
        for bi, block in enumerate(e.model.backbone.blocks if hasattr(e.model, "backbone") and hasattr(e.model.backbone, "blocks") else []):
            gate = block.gnn.graph_learner.mask_moe
            handles.append(gate.register_forward_hook(hook_gate(dev, bi)))
            handles.append(block.gnn.graph_learner.register_forward_hook(hook_learner(dev, bi)))
    report = {"schema": "df_sota_route_trace.v1", "unit": unit, "devices": devices, "windows": windows, "batches": batches, "batch_size": bs,
              "cpu_max_abs_diff_by_window_vs_stored": cpu_diff_by_window, "mode": "eval (is_training=False): noisy gating OFF; a difference is numerical or a threshold flip",
              "gpu": gpu_state(), "batches_traced": []}
    try:
        with torch.no_grad():
            for bi_ in batches:
                # identical batch composition on both devices: the same contiguous windows of the author's test loader
                idx = list(range(bi_ * bs, min(len(test_data), (bi_ + 1) * bs)))
                items = [test_data[i] for i in idx]
                bx = torch.tensor(np.stack([it[0] for it in items])).float()
                outs = {}
                for dev, e in exps.items():
                    captures.pop(dev, None)
                    out, _ = e.model(bx.to(e.device), e.masks, is_training=False)
                    outs[dev] = out[:, -args.pred_len:, :].detach().float().cpu().numpy()
                entry = {"batch": bi_, "windows": idx, "pred_max_abs_diff_cpu_vs_stored": float(np.abs(outs["cpu"].astype(np.float64) - stored[idx].astype(np.float64)).max())}
                if "cuda" in outs:
                    entry["pred_max_abs_diff_gpu_vs_stored"] = float(np.abs(outs["cuda"].astype(np.float64) - stored[idx].astype(np.float64)).max())
                    entry["pred_max_abs_diff_cpu_vs_gpu"] = float(np.abs(outs["cpu"].astype(np.float64) - outs["cuda"].astype(np.float64)).max())
                    blocks = {}
                    for key in sorted(captures.get("cpu", {})):
                        c, g = captures["cpu"][key], captures["cuda"].get(key)
                        if g is None:
                            continue
                        if key.startswith("gate"):
                            flips = int((c["top_p_mask"] != g["top_p_mask"]).sum())
                            blocks[key] = {"route_flips": flips, "routes_total": int(c["top_p_mask"].numel()),
                                           "cut_index_changes": int((c["cut_index"] != g["cut_index"]).sum()), "tokens": int(c["cut_index"].numel()),
                                           "max_abs_prob_diff": float((c["probs"] - g["probs"]).abs().max()),
                                           "min_distance_to_threshold_cpu": float(c["distance_to_threshold"].min()), "min_distance_to_threshold_gpu": float(g["distance_to_threshold"].min()),
                                           "route_input_equal": c["route_input_sha"] == g["route_input_sha"]}
                        else:
                            blocks[key] = {"max_abs_adj_diff_post_route": float((c["adj_out"] - g["adj_out"]).abs().max())}
                    entry["blocks"] = blocks
                    flips = sum(b.get("route_flips", 0) for b in blocks.values())
                    entry["classification"] = ("ROUTING_FLIP_AT_THRESHOLD (numerical difference crossed a top-p cut)" if flips > 0 else
                                               ("NUMERICAL_ONLY (identical routes, kernel-level tensor differences)" if entry["pred_max_abs_diff_cpu_vs_gpu"] > 0 else "IDENTICAL"))
                report["batches_traced"].append(entry)
            # 3. repeatability under identical batching: the same batch twice on each device
            if batches:
                bi_ = batches[0]; idx = list(range(bi_ * bs, min(len(test_data), (bi_ + 1) * bs)))
                bx = torch.tensor(np.stack([test_data[i][0] for i in idx])).float()
                rep = {}
                for dev, e in exps.items():
                    o1, _ = e.model(bx.to(e.device), e.masks, is_training=False); o2, _ = e.model(bx.to(e.device), e.masks, is_training=False)
                    rep[dev] = float((o1 - o2).abs().max())
                report["same_device_same_batch_repeat_max_abs_diff"] = rep
    finally:
        for h in handles:
            h.remove()
    return report


# --- RP101: memory/disk/thermal profile of the bounded evaluation path ------------------------------------------------------------

def profile_eval(design: dict, *, data_path: Path, work: Path, horizon: int, checkpoint: Path | None, gpu: int = 0, use_gpu: bool | None = None,
                 author_metric_budget_bytes: int | None = None) -> dict:
    """Run the bounded evaluation for one horizon (with a given checkpoint, or an UNTRAINED model for memory only) and measure:
    cgroup memory current/peak including file-backed pages, RSS, disk used under `work`, GPU VRAM peak, GPU temperature before/after,
    elapsed time. No result is stored when the model is untrained."""
    import torch, shutil
    cell = next(c for c in design["cells"] if c["horizon"] == horizon)
    author_env(); fix_seeds(cell["seed"])
    work.mkdir(parents=True, exist_ok=True)
    args = build_args(cell["argv"], data_dir=data_path.parent, data_name=data_path.name, checkpoints=work / "checkpoints", gpu=gpu, use_gpu=use_gpu)
    exp_module = importlib.import_module("exp.exp_long_term_forecasting")
    before = {"cgroup": _cgroup_memory(), "gpu": gpu_state(), "disk_free_bytes": shutil.disk_usage(work).free, "at": now_iso()}
    with contextlib.redirect_stdout(io.StringIO()):
        exp = exp_module.Exp_Long_Term_Forecast(args)
    setting = setting_of(args)
    (work / "checkpoints" / setting).mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        shutil.copy2(checkpoint, work / "checkpoints" / setting / "checkpoint.pth"); trained = True
    else:
        torch.save(exp.model.state_dict(), work / "checkpoints" / setting / "checkpoint.pth"); trained = False
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = bounded_test(exp, setting, work, author_metric_budget_bytes=author_metric_budget_bytes)
    elapsed = time.time() - t0
    disk_used = sum(f.stat().st_size for f in work.rglob("*") if f.is_file())
    ru = resource.getrusage(resource.RUSAGE_SELF)
    after = {"cgroup": _cgroup_memory(), "gpu": gpu_state(), "at": now_iso()}
    prof = {"schema": "df_sota_eval_profile.v1", "horizon": horizon, "trained_checkpoint": trained, "device": str(exp.device), "windows": out["finalized"]["windows"],
            "preds_bytes": out["finalized"]["preds_bytes"], "elapsed_seconds": elapsed, "peak_rss_bytes": int(ru.ru_maxrss) * 1024,
            "cgroup_before": before["cgroup"], "cgroup_after": after["cgroup"], "gpu_before": before["gpu"], "gpu_after": after["gpu"],
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "disk_used_under_work_bytes": disk_used, "author_metric_state": out["author_metric_state"], "author_metric": out["author_metric"],
            "independent_metric_float64": out["independent_metric_float64"] if trained else "not stored (untrained model)"}
    for tmp in (out["preds_path"], out["trues_path"]):
        try:
            Path(tmp).unlink()
        except OSError:
            pass
    return prof


# --- RP103: authorized deletion of prediction arrays after verified metrics ---------------------------------------------------------

def _readers_of(path: Path) -> list:
    """Processes holding or transferring the file: fuser on the path plus any rsync/scp/cp command line naming it."""
    out = []
    try:
        r = subprocess.run(["fuser", str(path)], capture_output=True, text=True, timeout=20)
        if r.stdout.strip():
            out.append(f"fuser: {r.stdout.strip()}")
    except Exception:                                               # noqa: BLE001
        pass
    try:
        r = subprocess.run(["pgrep", "-fa", f"(rsync|scp|cp|tar) .*{path.name}"], capture_output=True, text=True, timeout=20)
        for line in r.stdout.strip().splitlines():
            if str(path.parent.name) in line or str(path) in line:
                out.append(f"transfer: {line[:120]}")
    except Exception:                                               # noqa: BLE001
        pass
    return out


def inventory_predictions(roots: list) -> list:
    inv = []
    for root in roots:
        root = Path(root)
        for f in sorted(root.glob("attempts/*/arrays.npz")):
            st = f.stat(); vfs = os.statvfs(f)
            inv.append({"path": str(f), "root": str(root), "unit": f.parent.name, "bytes": st.st_size, "sha256": sha_file(f), "filesystem_id": f"{os.stat(f).st_dev}",
                        "readers": _readers_of(f)})
    return inv


def deletion_gate(root: Path, unit: str) -> dict:
    """Every condition the retention document names, checked from persisted evidence: a closure REPORT whose row for the unit is
    verified (custody accepted, catalog recomputed and read back, independent check) or historically verified; the catalog on disk
    with the reported digest; cross-device failure diagnostics preserved when the replay history holds a failure; no reader."""
    root = Path(root); folder = root / "attempts" / unit
    reasons = []
    report_path = root / "REPORT.json"
    if not report_path.is_file():
        return {"pass": False, "reasons": ["no closure REPORT.json"]}
    report = json.loads(report_path.read_text())
    row = next((r for r in report["verification"]["rows"] if r["unit"] == unit), None)
    if row is None:
        reasons.append("the unit has no row in the closure report")
    else:
        if not row.get("verified"):
            reasons.append(f"the closure did not verify the unit: {row.get('problems') or row.get('status')}")
        if (row.get("custody") or {}).get("class") != "ACCEPTED_ARTIFACT_CHAIN":
            reasons.append("custody is not the accepted artifact chain")
        rc = row.get("recomputed") or {}
        if not (rc.get("metrics_vault_recomputed") and rc.get("metrics_vault_read_back_equal")):
            reasons.append("the metric catalog was not recomputed and read back at closure")
        vp = folder / "METRICS_VAULT.json"
        if not vp.is_file() or sha_file(vp) != rc.get("metrics_vault_sha256"):
            reasons.append("the catalog on disk is not the one the closure reported")
        else:
            v = json.loads(vp.read_text())
            if v.get("schema") != VAULT_SCHEMA:
                reasons.append(f"catalog schema {v.get('schema')} is not {VAULT_SCHEMA}")
            ic = v.get("independent_check") or {}
            if not ic or abs((ic.get("global_vs_author") or {}).get("mae", 1)) > 1e-6:
                reasons.append("the catalog's independent check is absent or failed")
            if any(c.get("state") == "UNDEFINED" for k, c in (v.get("catalog") or {}).items() if k in ("errors", "matched_baselines")):
                reasons.append("required errors or baselines are UNDEFINED")
    if any(not e.get("allclose_rule") for e in replay_history(root, unit)) and not (folder / "ROUTE_TRACE.json").is_file():
        reasons.append("a replay failure is recorded (this root's or a merged worker's history) and its route-level diagnostic is not preserved")
    if not (folder / "cell.json").is_file() or not (folder / "checkpoint.pth").is_file():
        reasons.append("record or checkpoint missing (they are preserved, never deleted)")
    arrays = folder / "arrays.npz"
    if not arrays.is_file():
        reasons.append("no prediction array to delete")
    else:
        readers = _readers_of(arrays)
        if readers:
            reasons.append(f"active readers/transfers: {readers}")
        rec = json.loads((folder / "cell.json").read_text()) if (folder / "cell.json").is_file() else {}
        if sha_file(arrays) != rec.get("arrays_sha256"):
            reasons.append("the arrays on disk are not the record's arrays")
    return {"pass": not reasons, "reasons": reasons, "report_sha256": sha_file(report_path)}


def delete_predictions(root: Path, units: list, *, extra_roots: list | None = None, dry_run: bool = False) -> dict:
    """The owner's authorized deletion (2026-09-22): inventory every copy under the given roots, pass the gate per unit, delete the
    inventoried arrays (including the final copy), record per-path receipts with the bytes the filesystem actually reclaimed, and
    leave a PREDICTIONS_DELETED.json beside each record so the closure reports the dated historical verification."""
    root = Path(root); roots = [root] + [Path(x) for x in (extra_roots or [])]
    receipt = {"schema": "df_sota_deletion_receipt.v1", "at": now_iso(), "host": socket.gethostname(), "root": str(root), "dry_run": dry_run, "units": {}, "paths": []}
    inv = inventory_predictions(roots)
    for unit in units:
        gate = deletion_gate(root, unit)
        copies = [x for x in inv if x["unit"] == unit]
        entry = {"gate": gate, "copies_inventoried": copies, "deleted": []}
        if gate["pass"] and not dry_run:
            rec = json.loads((root / "attempts" / unit / "cell.json").read_text())
            for c in copies:
                path = Path(c["path"])
                if c["readers"]:
                    entry["deleted"].append({"path": c["path"], "deleted": False, "why": f"readers {c['readers']}"}); continue
                before = os.statvfs(path); free_before = before.f_bavail * before.f_frsize
                path.unlink()
                after = os.statvfs(path.parent); free_after = after.f_bavail * after.f_frsize
                entry["deleted"].append({"path": c["path"], "deleted": True, "bytes": c["bytes"], "sha256": c["sha256"], "filesystem_id": c["filesystem_id"],
                                         "reclaimed_bytes_fs_delta": free_after - free_before, "at": now_iso()})
            marker = {"schema": "df_sota_predictions_deleted.v1", "unit": unit, "deleted_at": now_iso(), "arrays_sha256": rec.get("arrays_sha256"), "pred_sha256": rec.get("pred_sha256"),
                      "closure_report_sha256": gate["report_sha256"], "verified_at_deletion": True, "authorization": "owner 2026-09-22 (retention document); Musashi RP98-RP105 RP103",
                      "paths_deleted": [d["path"] for d in entry["deleted"] if d["deleted"]], "reading": "metrics verified before authorized deletion; the arrays' hash records their former identity and cannot reconstruct them; no current replay is possible"}
            write_atomic(root / "attempts" / unit / "PREDICTIONS_DELETED.json", json.dumps(marker, indent=1))
            entry["marker"] = marker
        receipt["units"][unit] = entry
        receipt["paths"] += entry["deleted"]
    receipt["reclaimed_bytes_by_filesystem"] = {}
    for d in receipt["paths"]:
        if d.get("deleted"):
            receipt["reclaimed_bytes_by_filesystem"][d["filesystem_id"]] = receipt["reclaimed_bytes_by_filesystem"].get(d["filesystem_id"], 0) + d["reclaimed_bytes_fs_delta"]
    write_atomic(root / f"DELETION_RECEIPT.{int(time.time())}.json", json.dumps(receipt, indent=1, default=str))
    return receipt


# --- RP96: verification and the RP97 table ----------------------------------------------------------------------------------------

def replay_code_sha256() -> str:
    """The digest of the code that PERFORMS a replay (the replay driver and the author-path functions it calls), so that a
    cached replay is invalidated by a change of that path and not by unrelated edits of this file."""
    import inspect
    parts = [inspect.getsource(f) for f in (replay_cell, main_like_run_py, build_args, author_env, fix_seeds, capture_metric, setting_of, author_parser)]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def replay_cell(root: Path, design: dict, unit: str, *, data_path: Path, device: str = "cpu", author_metric_budget: int | None = None,
                malloc_tunables: str | None = MALLOC_TUNABLES_DEFAULT) -> dict:
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
res = M.main_like_run_py(cell["argv"], seed=cell["seed"], data_dir=Path({str(data_path.parent)!r}), data_name={data_path.name!r}, work=work, gpu=0, use_gpu={device != "cpu"}, train=False,
                         bounded=True, author_metric_budget_bytes={author_metric_budget!r})
stored = M.StoredArray(folder/"arrays.npz", "pred")
rep = res["preds"]                                     # the adapter's file-backed float32 predictions
rule = design["lock"]["replay_rule"]
shape_equal = tuple(rep.shape) == tuple(stored.shape)
finite = M.all_finite(rep) and stored.all_finite()
if shape_equal and finite:
    cmp_ = M.compare_predictions(rep, stored, atol=rule["atol"], rtol=rule["rtol"])
    max_abs, n_exact, n_total, allclose = cmp_["max_abs_prediction_difference"], cmp_["exact_equal_elements"], cmp_["elements"], cmp_["allclose_rule"]
else:
    max_abs, n_exact, n_total, allclose = None, 0, int(np.prod(rep.shape)), False
stored.close()
gpu = M.gpu_state(); dev_uuid = None; dev_name = "cpu"
if str(res["device"]).startswith("cuda"):
    idx = int(str(res["device"]).split(":")[-1]) if ":" in str(res["device"]) else 0
    dev_uuid = M.actual_device_uuid(idx)
    dev_name = next((g.get("name") for g in gpu if g.get("uuid") == dev_uuid), None) or (torch.cuda.get_device_name(idx) if torch.cuda.is_available() else None)
print(json.dumps({{"unit": unit, "device": res["device"], "device_uuid": dev_uuid, "device_name": dev_name,
                   "max_abs_prediction_difference": max_abs, "allclose_rule": allclose, "finite": finite, "shape_equal": shape_equal,
                   "exact_equal_elements": n_exact, "elements": n_total, "exact_equal_fraction": (n_exact / n_total) if n_total else None,
                   "replayed_author_metric": res["author_metric"], "replayed_author_metric_state": res.get("author_metric_state"), "replayed_metric_float64": res.get("independent_metric_float64"),
                   "true_sha256_replayed": res["bounded"]["finalized"]["true_sha256"], "shape": list(rep.shape), "operational_patches": res.get("operational_patches", []),
                   "rule": {{"atol": rule["atol"], "rtol": rule["rtol"]}}}}))
for tmp in (res["bounded"]["preds_path"], res["bounded"]["trues_path"]):
    Path(tmp).unlink(missing_ok=True)
shutil.rmtree(work, ignore_errors=True)
"""
    env = child_env(malloc_tunables, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"))
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=7200, env=env)
    if proc.returncode:
        return {"unit": unit, "allclose_rule": False, "finite": None, "shape_equal": None, "device": device, "error": proc.stderr[-800:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def naive_and_trues(design: dict, cell: dict, data_path: Path, work_dir: Path | None = None) -> dict:
    """The author's test loader (same args, same borders, same scaler): the targets re-derived, and persistence on identical windows.
    The targets are hashed as they stream; with `work_dir` they are also written to a .npy file (header first, filled through a
    file handle, pages released) so the closure can re-read them chunk by chunk instead of holding 4 GB of them."""
    author_env()
    import torch
    from numpy.lib.format import open_memmap
    DF = importlib.import_module("data_provider.data_factory")
    args = build_args(cell["argv"], data_dir=data_path.parent, data_name=data_path.name, checkpoints=Path("/nonexistent"), gpu=0, use_gpu=False)
    args.augmentation_ratio = 0
    test_data, test_loader = DF.data_provider(args, "test")
    n_windows = (len(test_loader) * test_loader.batch_size) if getattr(test_loader, "drop_last", False) else len(test_data)
    h = hashlib.sha256(); sq, ab, n, w = 0.0, 0.0, 0, 0
    path = (Path(work_dir) / f"trues_L{cell['seq_len']}_T{cell['horizon']}.npy") if work_dir is not None else None
    ft = None; header_off = None
    for batch_x, batch_y, _, _ in test_loader:
        y = batch_y[:, -args.pred_len:, :].float()
        last = batch_x[:, -1:, :].float().expand(-1, args.pred_len, -1)
        d = (last - y).double()
        sq += float((d * d).sum()); ab += float(d.abs().sum()); n += d.numel()
        yb = np.ascontiguousarray(y.numpy().astype(np.float32, copy=False))
        h.update(memoryview(yb).cast("B"))
        if path is not None:
            if ft is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                m = open_memmap(path, mode="w+", dtype=np.float32, shape=(n_windows, yb.shape[1], yb.shape[2])); header_off = m.offset; del m
                ft = open(path, "r+b"); ft.seek(header_off)
            if w + yb.shape[0] > n_windows:
                ft.close(); path.unlink(missing_ok=True)
                raise SotaRefusal(f"REFUSED: the loader yielded more windows ({w + yb.shape[0]}+) than it declares ({n_windows})")
            ft.write(memoryview(yb).cast("B"))
            if (w // max(1, yb.shape[0])) % 16 == 0:
                ft.flush(); _dontneed(ft.fileno(), 0, ft.tell())
        w += yb.shape[0]
    if ft is not None:
        ft.flush(); _dontneed(ft.fileno(), 0, ft.tell()); ft.close()
        if w != n_windows:
            path.unlink(missing_ok=True)
            raise SotaRefusal(f"REFUSED: the loader yielded {w} windows, it declares {n_windows}: INCOMPLETE POPULATION")
    return {"true_sha256": h.hexdigest(), "n_elements": n, "windows": int(w), "naive": {"mse": sq / n, "mae": ab / n}, "trues_path": (str(path) if path else None),
            "naive_definition": "persistence: the window's last observed value repeated over the horizon, every channel, same test windows, normalized space"}


class VaultRefusal(SotaRefusal):
    """A metric catalog is published only from a complete, finite, correctly shaped, ordered population."""


VAULT_SCHEMA = "df_sota_metrics_vault.v3"
VAULT_BLOCK_WINDOWS = 168                      # one week of hourly test windows per time block


def _corr(py, pp, yy, p2, y2, n):
    """Pearson correlation from sums; None (UNDEFINED) unless BOTH variances are positive."""
    vp, vy = p2 - pp * pp / n, y2 - yy * yy / n
    if vp <= 0 or vy <= 0:
        return None
    return float((py - pp * yy / n) / math.sqrt(vp * vy))


def metrics_vault(preds: np.ndarray, test_loader, *, pred_len: int, chunk: int = 128, max_lag: int = 168, identity: dict | None = None) -> dict:
    """The finite metric catalog of ONE cell (owner's decision 2026-09-22; retention document: required errors, matched baselines,
    per-horizon/channel/time-block summaries, residual/information diagnostics), so the raw arrays can be deleted afterwards.

    Population discipline (Musashi RP97 #3): the loader must yield EXACTLY the predictions' windows, in order, each batch of
    shape (b, pred_len, C) and finite; the population is the COUNT CONSUMED, and the catalog refuses (typed) otherwise. Every
    estimator carries a state — DONE / APPROXIMATE / UNDEFINED / NOT_APPLICABLE — with its parameters, excluded counts and
    reason. Undefined is None, never zero. Space: normalized (train-standardized) units, the space the paper reports in."""
    if preds.ndim != 3:
        raise VaultRefusal(f"REFUSED: predictions must be (windows, steps, channels); got {preds.shape}")
    W, T, C = preds.shape
    if T != pred_len:
        raise VaultRefusal(f"REFUSED: predictions have {T} steps, the cell's horizon is {pred_len}")
    if W == 0:
        raise VaultRefusal("REFUSED: no prediction windows")
    if not all_finite(preds):
        raise VaultRefusal("REFUSED: non-finite predictions")
    Z = lambda: np.zeros((T, C))
    S = {k: Z() for k in ("ab", "sq", "r", "nab", "nsq", "sab", "ssq", "y", "y2", "p", "p2", "py")}
    r3 = r4 = 0.0; n_pct = 0; mape = 0.0; mspe = 0.0
    hist_edges = np.linspace(-20.0, 20.0, 2001); hist = np.zeros(2000); n_outside = 0
    tail = {"1": 0, "2": 0, "3": 0}
    joint = np.zeros((64, 64)); jedges = np.linspace(-6.0, 6.0, 65); n_joint_clipped = 0
    step_series = np.zeros((W, T)); ch_first = np.zeros((W, C)); ch_last = np.zeros((W, C))
    per_window = {k: np.zeros(W) for k in ("mae", "mse", "naive_mae", "naive_mse", "seasonal24_mae", "seasonal24_mse")}
    true_hash = hashlib.sha256()
    w0 = 0; seasonal_supported = None; L_in = None
    for batch in test_loader:
        batch_x, batch_y = batch[0], batch[1]
        x = np.asarray(batch_x, dtype=np.float64) if not hasattr(batch_x, "numpy") else batch_x.numpy().astype(np.float64)
        y_all = np.asarray(batch_y) if not hasattr(batch_y, "numpy") else batch_y.numpy()
        if y_all.ndim != 3 or y_all.shape[2] != C or y_all.shape[1] < pred_len or x.ndim != 3 or x.shape[2] != C:
            raise VaultRefusal(f"REFUSED: a target batch of shape {tuple(y_all.shape)} / input {tuple(x.shape)} does not match predictions {(W, T, C)}")
        y32 = y_all[:, -pred_len:, :].astype(np.float32)
        b = y32.shape[0]
        if w0 + b > W:
            raise VaultRefusal(f"REFUSED: the loader yields more windows ({w0 + b}+) than the predictions ({W}): EXTRA ROWS")
        if not np.isfinite(y32).all() or not np.isfinite(x).all():
            raise VaultRefusal("REFUSED: non-finite targets or inputs")
        true_hash.update(memoryview(np.ascontiguousarray(y32)).cast("B"))
        y = y32.astype(np.float64); p = preds[w0:w0 + b].astype(np.float64); d = p - y
        L_in = x.shape[1]
        naive = np.broadcast_to(x[:, -1:, :], y.shape)
        if L_in >= 24:
            seasonal_supported = True
            seas = np.stack([x[:, L_in - 24 + (k % 24), :] for k in range(T)], axis=1)
        else:
            seasonal_supported = False; seas = naive
        S["ab"] += np.abs(d).sum(0); S["sq"] += (d * d).sum(0); S["r"] += d.sum(0)
        nd, sd_ = naive - y, seas - y
        S["nab"] += np.abs(nd).sum(0); S["nsq"] += (nd * nd).sum(0); S["sab"] += np.abs(sd_).sum(0); S["ssq"] += (sd_ * sd_).sum(0)
        S["y"] += y.sum(0); S["y2"] += (y * y).sum(0); S["p"] += p.sum(0); S["p2"] += (p * p).sum(0); S["py"] += (p * y).sum(0)
        r3 += float((d ** 3).sum()); r4 += float((d ** 4).sum())
        nz = np.abs(y) > 1e-8; n_pct += int(nz.sum()); mape += float(np.abs(d[nz] / y[nz]).sum()); mspe += float(((d[nz] / y[nz]) ** 2).sum())
        h, _ = np.histogram(d, bins=hist_edges); hist += h; n_outside += int(d.size - h.sum())
        for t_ in tail:
            tail[t_] += int((np.abs(d) > float(t_)).sum())
        pr, yr = p.ravel(), y.ravel(); n_joint_clipped += int(((np.abs(pr) > 6) | (np.abs(yr) > 6)).sum())
        joint += np.histogram2d(np.clip(pr, -6, 6), np.clip(yr, -6, 6), bins=[jedges, jedges])[0]
        step_series[w0:w0 + b] = d.mean(axis=2); ch_first[w0:w0 + b] = d[:, 0, :]; ch_last[w0:w0 + b] = d[:, -1, :]
        per_window["mae"][w0:w0 + b] = np.abs(d).mean(axis=(1, 2)); per_window["mse"][w0:w0 + b] = (d * d).mean(axis=(1, 2))
        per_window["naive_mae"][w0:w0 + b] = np.abs(nd).mean(axis=(1, 2)); per_window["naive_mse"][w0:w0 + b] = (nd * nd).mean(axis=(1, 2))
        per_window["seasonal24_mae"][w0:w0 + b] = np.abs(sd_).mean(axis=(1, 2)); per_window["seasonal24_mse"][w0:w0 + b] = (sd_ * sd_).mean(axis=(1, 2))
        w0 += b
    if w0 != W:
        raise VaultRefusal(f"REFUSED: the loader consumed {w0} windows, the predictions have {W}: INCOMPLETE POPULATION")
    n = W * T * C
    ab_all, sq_all = float(S["ab"].sum()), float(S["sq"].sum())
    mse, mae = sq_all / n, ab_all / n
    ymean, pmean = float(S["y"].sum()) / n, float(S["p"].sum()) / n
    sst = float(S["y2"].sum()) - n * ymean ** 2; ssp = float(S["p2"].sum()) - n * pmean ** 2
    rmean = float(S["r"].sum()) / n; rvar = mse - rmean ** 2
    skew = (r3 / n - 3 * rmean * rvar - rmean ** 3) / (rvar ** 1.5) if rvar > 0 else None
    kurt = (r4 / n - 4 * rmean * r3 / n + 6 * rmean ** 2 * mse - 3 * rmean ** 4) / (rvar ** 2) if rvar > 0 else None
    hist_total = float(hist.sum())
    cdf = np.cumsum(hist) / hist_total if hist_total > 0 else None; centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    quantiles = ({str(q): float(centers[min(len(centers) - 1, int(np.searchsorted(cdf, q)))]) for q in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999)}
                 if cdf is not None else None)
    ph = hist / hist_total if hist_total > 0 else None
    entropy_bits = float(-(ph[ph > 0] * np.log2(ph[ph > 0])).sum()) if ph is not None else None
    pj = joint / max(1.0, joint.sum()); pa, pb = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True); nzj = pj > 0
    mi_bits = float((pj[nzj] * np.log2(pj[nzj] / (pa @ pb)[nzj])).sum())
    naive_mae, naive_mse = float(S["nab"].sum()) / n, float(S["nsq"].sum()) / n
    seas_mae, seas_mse = float(S["sab"].sum()) / n, float(S["ssq"].sum()) / n
    def ratio(a, b_):
        return (a / b_) if b_ > 0 else None
    # --- autocorrelation along consecutive windows (support: lag < W - 1) ----------------------------------------------------
    def acf_series(series: np.ndarray, lags: int) -> list:
        s_ = series - series.mean(axis=0, keepdims=True); den = (s_ * s_).sum(axis=0)
        out = []
        for lag in range(1, lags + 1):
            num = (s_[lag:] * s_[:-lag]).sum(axis=0)
            out.append([float(num[i] / den[i]) if den[i] > 0 else None for i in range(series.shape[1])])
        return out
    lags = max(0, min(max_lag, W - 2))
    step_acf = acf_series(step_series, lags) if lags > 0 else []
    ch_acf = {}
    for where, arr in (("first_step", ch_first), ("last_step", ch_last)):
        ch_acf[where] = {str(l): (acf_series(arr, l)[-1] if l < W - 1 else None) for l in (1, 24, 168)}
    # --- time blocks: weekly blocks of consecutive windows ---------------------------------------------------------------------
    blocks = []
    for b0 in range(0, W, VAULT_BLOCK_WINDOWS):
        b1 = min(W, b0 + VAULT_BLOCK_WINDOWS)
        blocks.append({"windows": [b0, b1], "n": b1 - b0, **{k: float(per_window[k][b0:b1].mean()) for k in per_window}})
    per_step = {"mse": (S["sq"].sum(1) / (W * C)).tolist(), "mae": (S["ab"].sum(1) / (W * C)).tolist(), "bias": (S["r"].sum(1) / (W * C)).tolist(),
                "naive_mse": (S["nsq"].sum(1) / (W * C)).tolist(), "naive_mae": (S["nab"].sum(1) / (W * C)).tolist(),
                "seasonal24_mse": (S["ssq"].sum(1) / (W * C)).tolist(), "seasonal24_mae": (S["sab"].sum(1) / (W * C)).tolist()}
    per_step["mae_relative_to_test_persistence"] = [ratio(a, b_) for a, b_ in zip(per_step["mae"], per_step["naive_mae"])]
    per_step["error_growth_mae_over_step1"] = [ratio(a, per_step["mae"][0]) for a in per_step["mae"]]
    WT = W * T
    per_channel = {"mse": (S["sq"].sum(0) / WT).tolist(), "mae": (S["ab"].sum(0) / WT).tolist(), "bias": (S["r"].sum(0) / WT).tolist(),
                   "naive_mse": (S["nsq"].sum(0) / WT).tolist(), "naive_mae": (S["nab"].sum(0) / WT).tolist(), "seasonal24_mae": (S["sab"].sum(0) / WT).tolist(),
                   "r2": [float(1 - sq / (y2 - yy ** 2 / WT)) if (y2 - yy ** 2 / WT) > 0 else None for sq, y2, yy in zip(S["sq"].sum(0), S["y2"].sum(0), S["y"].sum(0))],
                   "corr_pred_true": [_corr(py, pp, yy, p2, y2, WT) for py, pp, yy, p2, y2 in zip(S["py"].sum(0), S["p"].sum(0), S["y"].sum(0), S["p2"].sum(0), S["y2"].sum(0))]}
    per_channel["mae_relative_to_test_persistence"] = [ratio(a, b_) for a, b_ in zip(per_channel["mae"], per_channel["naive_mae"])]
    n_corr_undefined = sum(1 for v in per_channel["corr_pred_true"] if v is None)
    catalog = {
        "errors": {"state": "DONE", "estimators": ["mse", "mae", "rmse", "bias"], "population": n, "reduction": "exact float64 sums over every window x step x channel"},
        "percentage_errors_zspace": {"state": "DONE" if n_pct == n else ("APPROXIMATE" if n_pct > 0 else "UNDEFINED"), "estimators": ["mape_zspace", "mspe_zspace"],
                                     "excluded_elements_abs_true_le_1e-8": n - n_pct, "note": "ratios in the centered z-space, NOT physical percentage errors; diagnostic only"},
        "matched_baselines": {"state": "DONE", "persistence": "last input value repeated over the horizon, same windows",
                              "seasonal24": ("DONE for steps <= 24; APPROXIMATE beyond (the value 24 h before the step wraps inside the input window)" if seasonal_supported else "NOT_APPLICABLE: input window shorter than 24"),
                              "mae_relative_to_test_persistence": "MAE / persistence MAE on the SAME test windows (not MASE: the denominator is not the training-set naive scale)"},
        "residual_moments": {"state": "DONE" if rvar > 0 else "UNDEFINED", "estimators": ["mean", "var", "sd", "skewness", "kurtosis_raw"], "reason": None if rvar > 0 else "zero residual variance"},
        "quantiles": {"state": "APPROXIMATE" if hist_total > 0 else "UNDEFINED", "parameters": {"bins": 2000, "range": [-20.0, 20.0], "bin_width": 0.02},
                      "excluded_outside_range": n_outside, "note": "read from the histogram; not an exact order statistic"},
        "entropy": {"state": "APPROXIMATE" if hist_total > 0 else "UNDEFINED", "parameters": {"bins": 2000, "range": [-20.0, 20.0]}, "excluded_outside_range": n_outside},
        "mutual_information": {"state": "APPROXIMATE", "parameters": {"bins": [64, 64], "range": [-6.0, 6.0], "estimator": "plug-in on the joint histogram"}, "clipped_elements": n_joint_clipped},
        "correlation_r2": {"state": "DONE" if n_corr_undefined < C else "UNDEFINED", "undefined_channels": n_corr_undefined, "rule": "Pearson needs positive variance in BOTH series; R2 needs positive true variance"},
        "autocorrelation": {"state": "DONE" if lags > 0 else "NOT_APPLICABLE", "lags": lags, "population_windows": W, "rule": "lag L needs W > L + 1; consecutive windows overlap by construction, so no 1/sqrt(W) band is claimed; no PACF, no spectrum",
                            "per_channel_lags_not_applicable": [l for l in (1, 24, 168) if not l < W - 1]},
        "time_blocks": {"state": "DONE", "block_windows": VAULT_BLOCK_WINDOWS, "n_blocks": len(blocks), "last_block_partial": (W % VAULT_BLOCK_WINDOWS) != 0},
        "per_window_series": {"state": "DONE", "length": W, "use": "paired contrasts between seeds and against the matched baselines, per window"},
        "paired_contrasts": {"state": "NOT_APPLICABLE_IN_ONE_CELL", "note": "seed-to-seed and model-vs-baseline paired contrasts are computed at closure from the per-window series of the cells"},
    }
    vault = {"schema": VAULT_SCHEMA, "space": "normalized (train-standardized) units; author reduction = mean over windows x steps x channels",
             "identity": {**(identity or {}), "true_sha256_consumed": true_hash.hexdigest(), "shape": [W, T, C], "input_window": L_in, "row_order": "the author test loader's order (shuffle=False), consumed in full",
                          "metric_implementation_sha256": hashlib.sha256(__import__("inspect").getsource(metrics_vault).encode()).hexdigest(), "numeric": "float64 accumulation of float32 inputs"},
             "population": {"windows": W, "steps": T, "channels": C, "elements": n, "consumed_windows": w0},
             "catalog": catalog,
             "global": {"mse": mse, "mae": mae, "rmse": math.sqrt(mse), "bias": rmean, "mape_zspace": (mape / n_pct) if n_pct else None, "mspe_zspace": (mspe / n_pct) if n_pct else None,
                        "rse": math.sqrt(sq_all / sst) if sst > 0 else None, "r2": 1 - sq_all / sst if sst > 0 else None,
                        "corr_pred_true": _corr(float(S["py"].sum()), float(S["p"].sum()), float(S["y"].sum()), float(S["p2"].sum()), float(S["y2"].sum()), n),
                        "true_mean": ymean, "pred_mean": pmean, "true_var": sst / n, "pred_var": ssp / n,
                        "naive_mse": naive_mse, "naive_mae": naive_mae, "seasonal24_mse": seas_mse, "seasonal24_mae": seas_mae,
                        "skill_mae_vs_naive": (1 - mae / naive_mae) if naive_mae > 0 else None, "skill_mse_vs_naive": (1 - mse / naive_mse) if naive_mse > 0 else None,
                        "mae_relative_to_test_persistence": ratio(mae, naive_mae), "mae_relative_to_seasonal24": ratio(mae, seas_mae),
                        "mutual_information_bits_pred_true_64x64": mi_bits},
             "residuals": {"mean": rmean, "var": rvar, "sd": math.sqrt(max(0.0, rvar)), "skewness": skew, "kurtosis_raw": kurt, "entropy_bits": entropy_bits,
                           "quantiles": quantiles, "fraction_abs_gt": {k: v / n for k, v in tail.items()},
                           "histogram": {"edges": hist_edges.tolist(), "counts": hist.astype(int).tolist(), "outside_range": n_outside}},
             "per_step": per_step, "per_channel": per_channel,
             "time_blocks": blocks, "per_window": {k: v.tolist() for k, v in per_window.items()},
             "autocorrelation": {"channel_mean_residual_per_step": {"lags": list(range(1, lags + 1)), "acf_by_lag": step_acf}, "per_channel": ch_acf},
             "joint_histogram_pred_true": {"edges": jedges.tolist(), "counts": joint.astype(int).tolist(), "clipped_elements": n_joint_clipped}}
    def clean(o):
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    return clean(vault)


def vault_equal(a: dict, b: dict) -> bool:
    """Two catalogs are the same measurement when everything but their timestamps and closure notes agree."""
    strip = lambda v: {k: x for k, x in v.items() if k not in ("at", "independent_check")}
    return json.dumps(strip(a), sort_keys=True) == json.dumps(strip(b), sort_keys=True)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def verify_sota_run(root: Path, *, warehouse=None, data_path: Path | None = None, replay: bool = True, replay_device: str = "cpu",
                    replay_units: list | None = None, author_metric_budget: int | None = None) -> dict:
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
        deleted = folder / "PREDICTIONS_DELETED.json"
        if not (folder / "arrays.npz").is_file() and deleted.is_file() and (folder / "cell.json").is_file():
            # RP103: the arrays were deleted under the owner's authorization AFTER their metrics were verified; the record and the
            # deletion receipt say when and under which closure. Not a current replay, not a missing artifact.
            dd = json.loads(deleted.read_text()); record = json.loads((folder / "cell.json").read_text())
            ok_receipt = dd.get("arrays_sha256") == record.get("arrays_sha256") and dd.get("closure_report_sha256") and dd.get("deleted_at")
            basis_metric, basis = metric_of(record)
            rows.append({"unit": unit, "cell": {k: cell[k] for k in ("cell_id", "arm", "protocol", "seq_len", "horizon", "seed")}, "verified": False,
                         "verified_historically": bool(ok_receipt and dd.get("verified_at_deletion")), "status": "METRICS_VERIFIED_BEFORE_AUTHORIZED_DELETION" if ok_receipt else "DELETED_WITHOUT_VALID_RECEIPT",
                         "author_metric_float32": basis_metric, "metric_basis": basis, "custody": {"class": "HISTORICAL: arrays deleted under authorization; digests retained"},
                         "deletion": dd, "training": record.get("training"), "cost": record.get("cost"), "n_parameters": record.get("n_parameters"), "device": record.get("device"),
                         "device_uuid": trained_device_of(record),
                         "replay": {"skipped": True, "why": "predictions deleted under authorization: no current replay is possible; historical verification dated in the receipt"},
                         "problems": ([] if ok_receipt else [f"{unit}: predictions absent and the deletion receipt does not bind to the record: DELETED_WITHOUT_VALID_RECEIPT"]),
                         "task_id": disp["task_id"], "disposition": disp["disposition"]})
            continue
        if not (folder / "cell.json").is_file() or not (folder / "arrays.npz").is_file() or not (folder / "checkpoint.pth").is_file():
            rows.append({"unit": unit, "cell": cell, "verified": False, "status": "MISSING", "problems": [f"{unit}: a registered cell has no record, arrays or checkpoint — missing, not absent"]}); continue
        record = json.loads((folder / "cell.json").read_text())
        if record.get("design_sha256") != design["design_sha256"] or any(record["cell"].get(k) != cell[k] for k in ("cell_id", "horizon", "seed", "seq_len", "arm")):
            p_.append(f"{unit}: the record's identity contradicts the design cell")
        transport = ("root_path", "data_path", "checkpoints", "gpu", "use_gpu", "device_ids", "devices", "num_workers")
        if record.get("effective_args") != cell["effective_args"] | {k: record["effective_args"].get(k) for k in transport}:
            diff = sorted(k for k in set(cell["effective_args"]) | set(record.get("effective_args") or {}) if k not in transport
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
        try:
            preds = StoredArray(folder / "arrays.npz", "pred")             # chunked reads, never a whole 4 GB array (RP101 at closure)
        except (SotaRefusal, KeyError, OSError, ValueError) as exc:
            p_.append(f"{unit}: the predictions cannot be streamed from arrays.npz: {str(exc)[:120]}")
            rows.append({"unit": unit, "cell": {k: cell[k] for k in ("cell_id", "arm", "protocol", "seq_len", "horizon", "seed")}, "custody": custody, "status": "UNREADABLE",
                         "author_metric_float32": metric_of(record)[0], "metric_basis": metric_of(record)[1], "recomputed": None, "derived": None,
                         "problems": p_, "verified": False}); continue
        if not preds.all_finite():
            p_.append(f"{unit}: non-finite predictions")
        derived = None
        if data_path is not None and not p_:
            key = (cell["seq_len"], cell["horizon"])
            if key not in cache:
                cache[key] = naive_and_trues(design, cell, data_path, work_dir=root / "closure_work")
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
            # the targets were streamed to a file by the author's loader above (digest checked against the record): re-read chunked
            trues = StoredArray(derived["trues_path"])
            recomputed = {"independent_float64": float64_metrics(preds, trues), "targets_source": "closure_work: streamed by the author's test loader, digest = record"}
            exact = author_metric_exact(preds, trues)
            recomputed["author_float32"] = {"mae": exact["mae"], "mse": exact["mse"]}
            recomputed["author_float32_route"] = AUTHOR_SCORER_ROUTE
            need = (5 if preds.compressed else 3) * int(preds.nbytes)          # a compressed member is inflated whole for the author's function
            if author_metric_budget is None or need <= author_metric_budget:
                a32 = preds.load() if preds.compressed else preds.memmap()
                mae32, mse32 = MET.metric(a32, trues.memmap())[:2]; del a32
                recomputed["author_scorer_parity"] = {"author_function": {"mae": float(mae32), "mse": float(mse32)}, "bit_equal": bool(float(mae32) == exact["mae"] and float(mse32) == exact["mse"])}
                if not recomputed["author_scorer_parity"]["bit_equal"]:
                    p_.append(f"{unit}: the exact bounded route and utils.metrics.metric disagree at closure: SCORER_ROUTE")
            else:
                recomputed["author_scorer_parity"] = {"author_function": None, "why": f"utils.metrics.metric not executed at closure (needs ~{need} bytes of temporaries)"}
            trues.close(); del trues
            vault_path = folder / "METRICS_VAULT.json"
            # RP99 (Musashi RP97 #1): a persisted vault is never trusted from disk — the catalog is RECOMPUTED from the accepted
            # arrays through the author loader and compared; a differing persisted candidate is preserved with a disposition and
            # the recomputed successor is published atomically; the digest reported is the successor's
            _, test_loader = DF.data_provider(args, "test")
            identity = {"unit": unit, "design_sha256": design["design_sha256"], "file_sha256": design["source_data"]["sha256"], "pred_sha256": record.get("pred_sha256"),
                        "true_sha256_record": record.get("true_sha256"), "arrays_sha256": arrays_sha, "checkpoint_sha256": ckpt_sha, "record_sha256": record_sha,
                        "scaler_sha256": ((json.loads((root / "BENCH_DATA.json").read_text()).get("sets") or {}).get(f"L{cell['seq_len']}_h{cell['horizon']}") or {}).get("scaler_sha256")
                        if (root / "BENCH_DATA.json").is_file() else None, "horizon": cell["horizon"], "seq_len": cell["seq_len"], "seed": cell["seed"]}
            try:
                fresh = metrics_vault(preds, test_loader, pred_len=args.pred_len, identity=identity)
            except VaultRefusal as exc:
                p_.append(f"{unit}: metric catalog refused: {str(exc)[:160]}"); fresh = None
            if fresh is not None:
                if fresh["identity"]["true_sha256_consumed"] != record.get("true_sha256"):
                    p_.append(f"{unit}: the catalog's consumed targets are not the record's targets: TARGETS")
                basis_metric, basis = metric_of(record)
                fresh["independent_check"] = {"record_metric": basis_metric, "record_metric_basis": basis,
                                              "global_vs_author": {"mae": fresh["global"]["mae"] - basis_metric["mae"], "mse": fresh["global"]["mse"] - basis_metric["mse"]},
                                              "per_step_mean_vs_global": {"mae": float(np.mean(fresh["per_step"]["mae"])) - fresh["global"]["mae"], "mse": float(np.mean(fresh["per_step"]["mse"])) - fresh["global"]["mse"]},
                                              "per_channel_mean_vs_global": {"mae": float(np.mean(fresh["per_channel"]["mae"])) - fresh["global"]["mae"]},
                                              "naive_vs_closure": {"mae": fresh["global"]["naive_mae"] - (derived or {}).get("naive", {}).get("mae", float("nan"))},
                                              "rule": "|global - author float32| <= 1e-6; per-step and per-channel means reproduce the global within 1e-9; naive equals the closure's within 1e-6"}
                ic = fresh["independent_check"]
                if abs(ic["global_vs_author"]["mae"]) > 1e-6 or abs(ic["global_vs_author"]["mse"]) > 1e-6 or abs(ic["per_step_mean_vs_global"]["mae"]) > 1e-9 \
                        or abs(ic["per_channel_mean_vs_global"]["mae"]) > 1e-9 or abs(ic["naive_vs_closure"]["mae"]) > 1e-6:
                    p_.append(f"{unit}: the recomputed catalog fails its independent check: {ic}"[:220])
                fresh["at"] = now_iso()
                persisted = json.loads(vault_path.read_text()) if vault_path.is_file() else None
                if persisted is not None and not vault_equal(persisted, fresh):
                    if persisted.get("schema") != VAULT_SCHEMA:
                        superseded = folder / f"METRICS_VAULT.superseded.{persisted.get('schema', 'unknown')}.{int(time.time())}.json"
                        write_atomic(superseded, json.dumps({"disposition": f"SUPERSEDED: catalog of schema {persisted.get('schema')} replaced by {VAULT_SCHEMA} recomputed from the accepted arrays",
                                                             "at": now_iso(), "candidate": persisted}, default=str))
                        recomputed["metrics_vault_superseded"] = superseded.name
                    else:
                        rejected = folder / f"METRICS_VAULT.rejected.{int(time.time())}.json"
                        write_atomic(rejected, json.dumps({"disposition": "REJECTED: persisted catalog differs from the one recomputed from the accepted arrays at closure",
                                                           "at": now_iso(), "candidate": persisted}, default=str))
                        p_.append(f"{unit}: the persisted metric catalog differs from the recomputed one: VAULT_CHANGED (candidate preserved as {rejected.name})")
                if persisted is None or not vault_equal(persisted, fresh):
                    write_atomic(vault_path, json.dumps(fresh, default=str))
                recomputed["metrics_vault_sha256"] = sha_file(vault_path)
                recomputed["metrics_vault_recomputed"] = True
                recomputed["metrics_vault_read_back_equal"] = vault_equal(json.loads(vault_path.read_text()), fresh)
                if not recomputed["metrics_vault_read_back_equal"]:
                    p_.append(f"{unit}: the published catalog does not read back equal to the recomputed one")
            basis_metric, basis = metric_of(record)
            if basis_metric is None:
                p_.append(f"{unit}: the record carries no metric at all")
            if record.get("author_metric_float32") and recomputed.get("author_float32") is not None:
                if abs(recomputed["author_float32"]["mae"] - record["author_metric_float32"]["mae"]) > 0 or abs(recomputed["author_float32"]["mse"] - record["author_metric_float32"]["mse"]) > 0:
                    p_.append(f"{unit}: the author's metric recomputed from the arrays is not the record's: METRIC")
            if basis_metric is not None and (abs(recomputed["independent_float64"]["mae"] - basis_metric["mae"]) > 1e-6 or abs(recomputed["independent_float64"]["mse"] - basis_metric["mse"]) > 1e-6):
                p_.append(f"{unit}: the independent float64 metric differs from the record's ({basis}) by more than 1e-6: REDUCTION")
            recomputed["metric_basis"] = basis
        basis_metric, basis = metric_of(record)
        if not record.get("author_metric_float32") and recomputed and recomputed.get("author_float32"):
            # RP109: the record carries only the float64 reduction (the author's function did not fit at training time); the
            # closure recomputed the author's float32 reduction from the accepted arrays through the exact bounded route
            basis_metric, basis = recomputed["author_float32"], f"author_float32 (recomputed at closure by {AUTHOR_SCORER_ROUTE}; the record carries the float64 reduction only)"
            recomputed["metric_basis"] = basis
        rows.append({"unit": unit, "cell": {k: cell[k] for k in ("cell_id", "arm", "protocol", "seq_len", "horizon", "seed")}, "custody": custody,
                     "author_metric_float32": basis_metric, "metric_basis": basis, "author_metric_state": record.get("author_metric_state"),
                     "recomputed": recomputed, "derived": derived,
                     "training": record.get("training"), "cost": record.get("cost"), "n_parameters": record.get("n_parameters"), "device": record.get("device"),
                     "device_uuid": trained_device_of(record), "environment": record.get("environment"),
                     "verified": not p_ and custody["class"] == "ACCEPTED_ARTIFACT_CHAIN" and prep["class"] == "PREPARATION_ACCEPTED_ARTIFACT" and not problems and data_path is not None,
                     "problems": p_, "task_id": disp["task_id"], "disposition": disp["disposition"]})
    replays = {}
    if replay and data_path is not None:
        # RP100 (Musashi RP97 #2): a cached replay record is NEVER an input — every closure re-runs the fresh-process reload for the
        # selected units on the selected device and validates the OUTPUT (finite, shape, count, full pointwise comparison, metric
        # reductions, targets). REPLAYS.json is a history keyed by unit and device, written for the record only.
        history = json.loads((root / "REPLAYS.json").read_text()) if (root / "REPLAYS.json").is_file() else {}
        trained_device = lambda r_: trained_device_of(r_)
        for r in rows:
            if r["problems"] or r.get("status") in ("MISSING", "METRICS_VERIFIED_BEFORE_AUTHORIZED_DELETION", "DELETED_WITHOUT_VALID_RECEIPT"):
                continue
            unit = r["unit"]; folder = root / "attempts" / unit
            if replay_units is not None and unit not in replay_units:
                r["replay"] = {"skipped": True, "why": "REPLAY_PENDING: not replayed on this host in this closure; the row stays UNVERIFIED until replayed"}
                r["verified"] = False
                continue
            ident = {"checkpoint_sha256": sha_file(folder / "checkpoint.pth"), "arrays_sha256": sha_file(folder / "arrays.npz"), "design_sha256": design["design_sha256"],
                     "replay_code_sha256": replay_code_sha256(), "author_files": source_digests(), "device_requested": replay_device}
            rep = replay_cell(root, design, unit, data_path=data_path, device=replay_device, author_metric_budget=author_metric_budget)
            rep["identity"] = ident
            rep["path"] = "fresh process: the author's checkpoint reload and forward pass through the bounded adapter (df_sota_bounded_eval.v1); the reload line is the author's test(test=1) line"
            rep["environment"] = environment()
            # the four properties are kept apart: which device actually replayed, and whether it is the device that trained the cell
            actual = rep.get("device_uuid")
            same_device = bool(actual and trained_device(r) and actual == trained_device(r)) or (rep.get("device") == "cpu" and str(r.get("device", "")).startswith("cpu"))
            rep["property"] = "same_device_repeatability" if same_device else "cross_device_portability"
            rep["trained_on_device_uuid"] = trained_device(r)
            ok = bool(rep.get("allclose_rule")) and rep.get("finite") is True and rep.get("shape_equal") is True
            if not ok:
                r["problems"].append(f"{unit}: fresh-process reload ({rep['property']}, {rep.get('device')}) fails the frozen replay rule (atol/rtol 1e-4): "
                                     f"max|delta| {rep.get('max_abs_prediction_difference')}, finite {rep.get('finite')}, shape_equal {rep.get('shape_equal')}, error {str(rep.get('error') or '')[-160:]}")
            elif rep.get("true_sha256_replayed") != r["derived"]["true_sha256"]:
                r["problems"].append(f"{unit}: the replay's targets differ from the derived targets")
            elif (rep.get("replayed_author_metric") or rep.get("replayed_metric_float64")) is None:
                r["problems"].append(f"{unit}: the replay produced no metric")
            elif abs((rep.get("replayed_author_metric") or rep["replayed_metric_float64"])["mae"] - r["author_metric_float32"]["mae"]) > 1e-5 or \
                    abs((rep.get("replayed_author_metric") or rep["replayed_metric_float64"])["mse"] - r["author_metric_float32"]["mse"]) > 1e-5:
                r["problems"].append(f"{unit}: the metric of the replayed predictions differs from the stored one by more than 1e-5")
            r["replay"] = {k: v for k, v in rep.items() if k not in ("identity", "environment")}
            r["same_device_repeatability"] = ("PASS" if ok and same_device else ("FAIL" if same_device else "NOT_TESTED_ON_THIS_DEVICE"))
            r["cross_device_portability"] = ("PASS" if ok and not same_device else ("FAIL" if not same_device else "NOT_TESTED_ON_THIS_DEVICE"))
            r["verified"] = r["verified"] and not r["problems"]
            history.setdefault(unit, {})[f"{rep.get('device')}:{actual or 'cpu'}@{now_iso()}"] = rep
            replays[unit] = rep
        write_atomic(root / "REPLAYS.json", json.dumps(history, indent=1, default=str))
    elif replay:
        for r in rows:
            if not r["problems"] and r.get("status") not in ("MISSING", "METRICS_VERIFIED_BEFORE_AUTHORIZED_DELETION", "DELETED_WITHOUT_VALID_RECEIPT"):
                r["replay"] = {"skipped": True, "why": "REPLAY_PENDING: no data file on this host"}; r["verified"] = False
    shutil.rmtree(root / "closure_work", ignore_errors=True)          # derived targets: re-streamed by the author's loader at every closure
    return {"design_sha256": design["design_sha256"], "disposition": disp, "preparation_custody": prep, "source_drift_now": drift, "rows": rows,
            "replay_patch": {"what": "torch.load defaults to map_location=cpu inside the CPU replay process", "why": "the author's test(test=1) loads the checkpoint "
                             "without map_location; a CUDA-trained checkpoint is otherwise unreadable on CPU", "effect": "tensor placement only; the frozen replay rule "
                             "(CPU, atol/rtol 1e-4, metric within 1e-5) is unchanged"},
            "problems": problems + [q for r in rows for q in r["problems"]] + ([f"author files drifted from the sealed digests: {sorted(drift)}"] if drift else []),
            "verified_units": sorted(r["unit"] for r in rows if r["verified"]), "unverified_units": sorted(r["unit"] for r in rows if not r["verified"]),
            "historically_verified_units": sorted(r["unit"] for r in rows if r.get("verified_historically"))}


def replay_history(root: Path, unit: str) -> list:
    """Every recorded replay of the unit on this root and on merged worker roots (REPLAYS_HISTORY.*), in both the RP96 flat form
    and the RP100 keyed form, labelled HISTORICAL_RECORD_ONLY: a closure verifies with fresh replays, never with these."""
    root = Path(root); out = []
    for f in sorted(root.glob("REPLAYS*.json")):
        try:
            h = json.loads(f.read_text())
        except Exception:
            continue
        entries = []
        if isinstance(h.get("cells"), dict) and unit in h["cells"]:            # RP96 REPLAYS_GPU.json: {cells: {unit: rep}}
            entries.append((f.name, h["cells"][unit]))
        elif isinstance(h.get(unit), dict):
            v = h[unit]
            if "allclose_rule" in v:                                          # RP96 flat: {unit: rep}
                entries.append((f.name, v))
            else:                                                             # RP100 keyed: {unit: {device@time: rep}}
                entries.extend((f"{f.name}#{k}", e) for k, e in v.items() if isinstance(e, dict))
        for name, e in entries:
            out.append({"source": name, "device": e.get("device"), "device_uuid": e.get("device_uuid"), "property": e.get("property"),
                        "allclose_rule": e.get("allclose_rule"), "max_abs_prediction_difference": e.get("max_abs_prediction_difference"),
                        "at": e.get("at") or e.get("replayed_at"), "scope": "HISTORICAL_RECORD_ONLY"})
    return out


def metric_of(record: dict) -> tuple:
    """(metric dict, basis): the author's float32 reduction when it was executed; otherwise the independent float64 reduction,
    labelled as such (the difference between the two was <= 2.1e-8 on every cell where both exist; never assumed bitwise)."""
    if record.get("author_metric_float32"):
        return record["author_metric_float32"], "author_float32"
    return record.get("independent_metric_float64"), "independent_float64 (author float32 reduction NOT executed within the memory budget)"


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
    status = "OPERATIONAL_AGREEMENT" if abs(diff) <= tol_a else ("OPERATIONAL_PARTIAL" if abs(diff) <= tol_p else "OUTSIDE_OPERATIONAL_MARGIN")
    return {"status": status, "mean": mean, "sd_ddof1": sd, "values": [float(v) for v in values], "published": p, "difference": diff,
            "tolerance_agree": tol_a, "tolerance_partial": tol_p, "n_seeds": len(values),
            "scope": "predeclared OPERATIONAL margin (2 x the paper's Table 7 four-horizon-average SD + rounding), borrowed as a heuristic for every "
                     "horizon; NOT a published per-horizon error bar and NOT statistical equivalence; the measured seed dispersion is reported beside it"}


def table(design: dict, ver: dict, *, root: Path | None = None) -> dict:
    """RP97: dataset/protocol, model/revision, horizon, published, replicated, difference, matched naive, seed dispersion, training
    completion, cost, agreement — normalized official metrics first; nothing unverified enters a mean."""
    pub = design["lock"]["published"]
    rows = []
    complete = True
    by_seed = {}                                                    # seed -> horizon -> verified metrics (for the within-seed average)
    for h in design["horizons"]:
        cells = [r for r in ver["rows"] if r["cell"]["horizon"] == h]
        ok = [r for r in cells if r["verified"] or r.get("verified_historically")]
        expected = [c["cell_id"] for c in design["cells"] if c["horizon"] == h]
        row = {"dataset_protocol": f"ECL official processed (TSL), L={design['seq_len']}, T={h}, split 7/1/2, normalized space", "model_revision": f"TimeFilter @ {PINNED_COMMIT[:12]}",
               "horizon": h, "published": pub["per_horizon"][str(h)], "seeds_expected": expected, "seeds_verified": [r["unit"] for r in ok],
               "verified_all": len(ok) == len(expected) and bool(expected)}
        for m in ("mse", "mae"):
            vals = [r["author_metric_float32"][m] for r in ok]
            row[m] = agreement(pub["per_horizon"][str(h)], vals, m)
        for r in ok:
            by_seed.setdefault(r["cell"]["seed"], {})[h] = r["author_metric_float32"]
        row["historically_verified_seeds"] = [r["unit"] for r in ok if r.get("verified_historically")]
        row["metric_basis"] = sorted({r.get("metric_basis", "author_float32") for r in ok})
        row["per_seed"] = {str(r["cell"]["seed"]): {"mse": r["author_metric_float32"]["mse"], "mae": r["author_metric_float32"]["mae"],
                                                    "difference_mse": r["author_metric_float32"]["mse"] - pub["per_horizon"][str(h)]["mse"],
                                                    "difference_mae": r["author_metric_float32"]["mae"] - pub["per_horizon"][str(h)]["mae"]} for r in ok}
        row["scopes"] = {"recipe_fidelity": "sealed author files, arguments and preparation bound (closure)" if ok else None,
                         "same_device_repeatability": sorted({r.get("same_device_repeatability", "NOT_TESTED_ON_THIS_DEVICE") for r in ok}),
                         "cross_device_portability": sorted({r.get("cross_device_portability", "NOT_TESTED_ON_THIS_DEVICE") for r in ok}),
                         "published_score_agreement": {"mse": row["mse"]["status"], "mae": row["mae"]["status"]},
                         "replay_history": ({r["unit"]: replay_history(root, r["unit"]) for r in ok} if root is not None else None)}
        complete = complete and row["verified_all"]
        row["matched_naive"] = (ok[0]["derived"]["naive"] if ok and ok[0].get("derived") else None)
        row["training_completion"] = [{"unit": r["unit"], "epochs_run": (r.get("training") or {}).get("epochs_run"), "best_epoch": (r.get("training") or {}).get("best_epoch_by_vali"),
                                       "early_stopped": (r.get("training") or {}).get("early_stopped"), "max_epochs": (design["cells"][0]["effective_args"].get("train_epochs"))} for r in ok]
        row["cost"] = {"wall_seconds_sum": sum((r.get("cost") or {}).get("wall_seconds") or 0 for r in ok), "cpu_seconds_sum": sum((r.get("cost") or {}).get("cpu_seconds") or 0 for r in ok),
                       "peak_gpu_bytes_max": max([((r.get("cost") or {}).get("peak_gpu_allocated_bytes") or 0) for r in ok] or [0]), "hosts": sorted({(r.get("cost") or {}).get("host") for r in ok})}
        row["replay_max_abs_difference"] = max([((r.get("replay") or {}).get("max_abs_prediction_difference") or 0.0) for r in ok] or [0.0]) if ok else None
        row["problems"] = [q for r in cells for q in r["problems"]]
        # executed cells that did NOT verify are shown with their values and the check that failed — never pooled into a mean
        row["executed_unverified"] = [{"unit": r["unit"], "author_metric_float32": r.get("author_metric_float32"),
                                       "custody": (r.get("custody") or {}).get("class"), "why": (r["problems"][:1] or [(r.get("replay") or {}).get("why") or "no replay"])[0][:160],
                                       "replay_max_abs_difference": (r.get("replay") or {}).get("max_abs_prediction_difference"),
                                       "replayed_author_metric": (r.get("replay") or {}).get("replayed_author_metric")}
                                      for r in cells if not r["verified"] and r.get("author_metric_float32")]
        rows.append(row)
    # RP103 (Musashi RP97 #5): the four-horizon average is formed WITHIN each matched seed first (a seed must have every horizon
    # verified), then its dispersion is across seeds; between-horizon spread can never masquerade as seed spread
    seeds_complete = [sd for sd, hs in sorted(by_seed.items()) if all(h in hs for h in design["horizons"])]
    seed_averages = {m: [float(np.mean([by_seed[sd][h][m] for h in design["horizons"]])) for sd in seeds_complete] for m in ("mse", "mae")}
    average = {m: ({**agreement(pub["average"], seed_averages[m], m), "seeds": seeds_complete, "grain": "average over the four horizons within each seed, then across seeds"}
                   if seeds_complete and len(seeds_complete) == len(design["seeds"]) else
                   {"status": "NOT_COMPUTED", "why": f"a full four-horizon average needs every horizon verified for every seed; complete seeds: {seeds_complete}",
                    "seed_averages_available": {str(sd): {m: float(np.mean([by_seed[sd][h][m] for h in design["horizons"]]))} for sd in seeds_complete}})
               for m in ("mse", "mae")}
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
    unv = [(r["horizon"], u) for r in t["rows"] for u in r.get("executed_unverified") or []]
    if unv:
        lines += ["", "## Executed cells NOT verified by the closure (values shown, never pooled into a mean)", "",
                  "| T | unit | MSE / MAE (author float32) | custody | replayed MSE / MAE | replay max|Δ| | check that failed |", "|---:|---|---|---|---|---|---|"]
        for h, u in unv:
            m = u.get("author_metric_float32") or {}; rm = u.get("replayed_author_metric") or {}
            lines.append(f"| {h} | {u['unit']} | {m.get('mse', float('nan')):.5f} / {m.get('mae', float('nan')):.5f} | {u.get('custody')} | "
                         f"{(rm.get('mse') if rm else float('nan')):.8f} / {(rm.get('mae') if rm else float('nan')):.8f} | "
                         f"{(u.get('replay_max_abs_difference') if u.get('replay_max_abs_difference') is not None else float('nan')):.2e} | {u.get('why')} |")
    a = t["average_over_horizons"]
    lines += ["", f"Average over the four horizons: MSE {a['mse'].get('mean') if a['mse'].get('mean') is None else round(a['mse']['mean'], 4)} vs published {t['rows'][0]['published'] and ''}"
              f"{'' if a['mse'].get('published') is None else a['mse']['published']} ({a['mse']['status']}); MAE {a['mae'].get('mean') if a['mae'].get('mean') is None else round(a['mae']['mean'], 4)} "
              f"vs {'' if a['mae'].get('published') is None else a['mae']['published']} ({a['mae']['status']}).",
              "", f"Unexecuted or unverified cells: {t['unexecuted'] or 'none'}.", f"Agreement rule (frozen before any test score): {t['agreement_rule']}",
              "", "* = early stopped. Metrics are the author's `metric()` on the concatenated float32 test predictions in the normalized space (no inversion); "
              "the paper prints three decimals. Training completion per seed: epochs run and the checkpointed epoch. Naive = persistence on identical windows."]
    return "\n".join(lines) + "\n"


def paired_contrasts(root: Path, design: dict, ver: dict) -> dict:
    """RP99/RP103: the paired contrasts the current design needs, computed from the persisted per-window series of the catalogs
    (so they survive the deletion of the arrays): per horizon, seed-to-seed paired differences of per-window MAE/MSE (mean, sd,
    sign counts, n windows), and per cell the model-vs-seasonal-24 and model-vs-persistence paired differences per window."""
    out = {"schema": "df_sota_paired_contrasts.v1", "grain": "per test window (one hour apart), paired by window index", "horizons": {}}
    vaults = {}
    for r in ver["rows"]:
        vp = root / "attempts" / r["unit"] / "METRICS_VAULT.json"
        if (r["verified"] or r.get("verified_historically")) and vp.is_file():
            v = json.loads(vp.read_text())
            if v.get("schema") == VAULT_SCHEMA and v.get("per_window"):
                vaults[r["unit"]] = (r["cell"], v)
    for h in design["horizons"]:
        cells = {u: (c, v) for u, (c, v) in vaults.items() if c["horizon"] == h}
        entry = {"cells": sorted(cells), "seed_pairs": {}, "vs_baselines": {}}
        for u, (c, v) in cells.items():
            pw = v["per_window"]; m = np.asarray(pw["mae"]); s24 = np.asarray(pw["seasonal24_mae"]); nv = np.asarray(pw["naive_mae"])
            entry["vs_baselines"][u] = {"n_windows": int(m.size),
                                        "model_minus_seasonal24_mae": {"mean": float((m - s24).mean()), "sd_ddof1": float((m - s24).std(ddof=1)) if m.size > 1 else None, "windows_model_better": int((m < s24).sum())},
                                        "model_minus_persistence_mae": {"mean": float((m - nv).mean()), "sd_ddof1": float((m - nv).std(ddof=1)) if m.size > 1 else None, "windows_model_better": int((m < nv).sum())}}
        units = sorted(cells)
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                a, b = np.asarray(cells[units[i]][1]["per_window"]["mae"]), np.asarray(cells[units[j]][1]["per_window"]["mae"])
                am, bm = np.asarray(cells[units[i]][1]["per_window"]["mse"]), np.asarray(cells[units[j]][1]["per_window"]["mse"])
                if a.size != b.size:
                    entry["seed_pairs"][f"{units[i]} - {units[j]}"] = {"status": "NOT_APPLICABLE: different populations"}; continue
                d, dm = a - b, am - bm
                entry["seed_pairs"][f"{units[i]} - {units[j]}"] = {"n_windows": int(d.size), "mae": {"mean": float(d.mean()), "sd_ddof1": float(d.std(ddof=1)), "first_better": int((d < 0).sum()), "second_better": int((d > 0).sum())},
                                                                   "mse": {"mean": float(dm.mean()), "sd_ddof1": float(dm.std(ddof=1)), "first_better": int((dm < 0).sum()), "second_better": int((dm > 0).sum())}}
        entry["state"] = "DONE" if len(cells) >= 2 else ("PARTIAL: one cell, baseline contrasts only" if cells else "NOT_APPLICABLE: no verified cell")
        out["horizons"][str(h)] = entry
    return out


def close(a, design: dict) -> dict:
    validate(design)
    root = Path(a.root)
    C = _module("df_mod_e0_close")
    token = Path(a.warehouse_token_file).read_text().strip().strip('"').strip("'") if getattr(a, "warehouse_token_file", None) else None
    warehouse = (lambda campaign: C.warehouse_terminals(a.warehouse_url, token, campaign)) if token else None
    data_path = Path(a.data_path) if getattr(a, "data_path", None) else (delivered_file(root, design, None) if (root / "DELIVERIES.json").is_file() else None)
    ver = verify_sota_run(root, warehouse=warehouse, data_path=data_path, replay=not getattr(a, "skip_replay", False), replay_device=getattr(a, "replay_device", "cpu"),
                          replay_units=getattr(a, "replay_units", None),
                          author_metric_budget=(int(a.author_metric_budget_gib * 2 ** 30) if getattr(a, "author_metric_budget_gib", None) else None))
    if warehouse is None:
        ver["problems"].append("closure without a warehouse read: no accepted custody, nothing is verified")
        for r in ver["rows"]:
            r["verified"] = False
        ver["verified_units"] = []
    t = table(design, ver, root=a.root)
    contrasts = paired_contrasts(root, design, ver)
    write_atomic(root / "PAIRED_CONTRASTS.json", json.dumps(contrasts, indent=1, default=str))
    report = {"schema": "df_sota_report.v2", "design_sha256": design["design_sha256"], "verification": ver, "table": t, "paired_contrasts": contrasts,
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
    ap.add_argument("command", choices=["seal", "prepare", "preflight", "execute", "child", "close", "lock", "merge", "route-trace", "profile-eval", "delete-predictions", "retire-attempt", "report"])
    ap.add_argument("--reason", default=None)
    ap.add_argument("--extra-roots", nargs="*", default=None, help="delete-predictions: other roots holding copies of the same cells (staging copies)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=None, help="profile-eval: a trained checkpoint (else an untrained model, memory only)")
    ap.add_argument("--horizon", type=int, default=None, help="profile-eval: which horizon's cell arguments")
    ap.add_argument("--windows", type=int, nargs="*", default=None, help="route-trace: window indices (else the most discrepant on CPU)")
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
    ap.add_argument("--bounded", action="store_true", help="child/execute: evaluate through the disk-backed bounded adapter (RP101) instead of the author's test()")
    ap.add_argument("--author-metric-budget-gib", type=float, default=None, help="bounded: run the author's float32 metric() only if its temporaries fit this budget")
    ap.add_argument("--dataloader-workers", type=int, default=None, help="operational: DataLoader worker processes (author default 1); batch order/content unchanged, host memory only")
    ap.add_argument("--malloc-tunables", default=MALLOC_TUNABLES_DEFAULT, help="operational: GLIBC_TUNABLES of cell and replay processes ('none' to leave the host allocator alone); host memory only")
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
    if a.command == "report":
        # a unit whose record and artifacts exist but whose terminal was never sent (a crash after run_cell): report it from the record
        cell = next((c for c in design["cells"] if c["cell_id"] == a.unit), None)
        if cell is None or not (a.root / "attempts" / a.unit / "cell.json").is_file():
            raise SotaRefusal("REFUSED: report needs a registered cell with its record on disk")
        record = json.loads((a.root / "attempts" / a.unit / "cell.json").read_text())
        report_unit(a, design, cell, record); print(json.dumps({"unit": a.unit, "reported": True, "metric_basis": metric_of(record)[1]})); return 0
    if a.command == "retire-attempt":
        if not (a.unit and a.reason):
            raise SotaRefusal("REFUSED: retire-attempt names its unit and its reason")
        print(json.dumps(retire_attempt(a.root, a.unit, reason=a.reason), indent=1)); return 0
    if a.command == "delete-predictions":
        rec = delete_predictions(a.root, a.units or [c["cell_id"] for c in design["cells"]], extra_roots=a.extra_roots, dry_run=a.dry_run)
        print(json.dumps({u: {"gate": e["gate"]["pass"], "reasons": e["gate"]["reasons"], "deleted": [d for d in e["deleted"] if d.get("deleted")].__len__(), "copies": len(e["copies_inventoried"])}
                          for u, e in rec["units"].items()}, indent=1)); print(json.dumps({"reclaimed_bytes_by_filesystem": rec["reclaimed_bytes_by_filesystem"]}))
        return 0 if all(e["gate"]["pass"] for e in rec["units"].values()) else 1
    if a.command == "route-trace":
        path = Path(a.data_path) if a.data_path else delivered_file(a.root, design, None)
        rep = route_trace(a.root, design, a.unit, data_path=path, windows=a.windows, gpu=a.gpu)
        (a.root / "attempts" / a.unit / "ROUTE_TRACE.json").write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps({k: rep[k] for k in ("windows", "cpu_max_abs_diff_by_window_vs_stored", "same_device_same_batch_repeat_max_abs_diff")}, default=str))
        for b in rep["batches_traced"]:
            print(json.dumps({k: b.get(k) for k in ("batch", "pred_max_abs_diff_cpu_vs_gpu", "pred_max_abs_diff_gpu_vs_stored", "classification", "blocks")}, default=str)[:1200])
        return 0
    if a.command == "profile-eval":
        path = Path(a.data_path) if a.data_path else delivered_file(a.root, design, None)
        prof = profile_eval(design, data_path=path, work=a.root / "profile_work", horizon=a.horizon, checkpoint=a.checkpoint, gpu=a.gpu, use_gpu=None if not a.cpu else False,
                            author_metric_budget_bytes=(int(a.author_metric_budget_gib * 2 ** 30) if a.author_metric_budget_gib else None))
        (a.root / f"EVAL_PROFILE.h{a.horizon}.{socket.gethostname()}.json").write_text(json.dumps(prof, indent=1, default=str))
        print(json.dumps({k: prof[k] for k in ("horizon", "trained_checkpoint", "device", "windows", "preds_bytes", "elapsed_seconds", "peak_rss_bytes", "peak_gpu_allocated_bytes", "disk_used_under_work_bytes", "author_metric_state")}, default=str))
        print(json.dumps({"cgroup_after": prof["cgroup_after"], "gpu_after_temp": [g.get("temperature_c") for g in prof["gpu_after"]]}, default=str))
        return 0
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
