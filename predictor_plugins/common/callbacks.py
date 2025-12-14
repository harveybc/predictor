"""Reusable callbacks shared by predictor plugins."""
from __future__ import annotations
import gc
import os
import time
from dataclasses import dataclass
from typing import Any
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
import tensorflow.keras.backend as K

class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.patience_counter = self.wait if self.wait > 0 else 0
        print(f"DEBUG: ReduceLROnPlateau patience counter: {self.patience_counter}")

class EarlyStoppingWithPatienceCounter(EarlyStopping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.patience_counter = self.wait if self.wait > 0 else 0
        print(f"DEBUG: EarlyStopping patience counter: {self.patience_counter}")

class ClearMemoryCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        gc.collect()


class MemoryUsageLogger(Callback):
    """Append RSS/VmHWM to a log file each epoch.

    Enabled by passing a path via config key `memory_log_file`.
    """
    def __init__(self, file_path: str, flush_every: int = 1, tag: str | None = None):
        super().__init__()
        self.file_path = file_path
        self.flush_every = max(1, int(flush_every))
        self.tag = tag or ""
        self._epoch_count = 0
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("ts,epoch,tag,VmRSS_kB,VmHWM_kB\n")

    @staticmethod
    def _read_status_kb(key: str) -> int | None:
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(key + ":"):
                        parts = line.split()
                        return int(parts[1])  # kB
        except Exception:
            return None
        return None

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_count += 1
        rss = self._read_status_kb("VmRSS")
        hwm = self._read_status_kb("VmHWM")
        ts = time.time()
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(f"{ts:.3f},{epoch},{self.tag},{rss if rss is not None else ''},{hwm if hwm is not None else ''}\n")
            if (self._epoch_count % self.flush_every) == 0:
                f.flush()


@dataclass(frozen=True)
class ResourceSnapshot:
    ts: float
    rss_kb: int | None
    hwm_kb: int | None
    gpu_current_bytes: int | None
    gpu_peak_bytes: int | None
    gc_counts: tuple[int, int, int] | None


def _read_proc_status_kb(key: str) -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    return int(parts[1])  # kB
    except Exception:
        return None
    return None


def _read_gpu_mem_bytes() -> tuple[int | None, int | None]:
    """Best-effort GPU memory snapshot.

    Uses TF's get_memory_info when available; returns (current, peak) in bytes.
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return (None, None)
        info = tf.config.experimental.get_memory_info("GPU:0")  # type: ignore[attr-defined]
        cur = int(info.get("current")) if isinstance(info, dict) and info.get("current") is not None else None
        peak = int(info.get("peak")) if isinstance(info, dict) and info.get("peak") is not None else None
        return (cur, peak)
    except Exception:
        return (None, None)


def capture_resource_snapshot(*, include_gpu: bool = True, include_gc: bool = True) -> ResourceSnapshot:
    ts = time.time()
    rss_kb = _read_proc_status_kb("VmRSS")
    hwm_kb = _read_proc_status_kb("VmHWM")
    gpu_current_bytes = None
    gpu_peak_bytes = None
    if include_gpu:
        gpu_current_bytes, gpu_peak_bytes = _read_gpu_mem_bytes()
    gc_counts = None
    if include_gc:
        try:
            gc_counts = gc.get_count()
        except Exception:
            gc_counts = None
    return ResourceSnapshot(
        ts=ts,
        rss_kb=rss_kb,
        hwm_kb=hwm_kb,
        gpu_current_bytes=gpu_current_bytes,
        gpu_peak_bytes=gpu_peak_bytes,
        gc_counts=gc_counts,
    )


class ResourceUsageLogger(Callback):
    """Append RSS/HWM (+ optional GPU + optional GC counts) each epoch.

    Designed for long GA runs where the kernel OOM killer provides no Python traceback.
    """
    def __init__(
        self,
        file_path: str,
        *,
        tag: str | None = None,
        flush_every: int = 1,
        include_gpu: bool = True,
        include_gc: bool = False,
    ):
        super().__init__()
        self.file_path = file_path
        self.tag = tag or ""
        self.flush_every = max(1, int(flush_every))
        self.include_gpu = bool(include_gpu)
        self.include_gc = bool(include_gc)
        self._epoch_count = 0
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(
                    "ts,epoch,tag,VmRSS_kB,VmHWM_kB,gpu_current_B,gpu_peak_B,gc0,gc1,gc2\n"
                )

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_count += 1
        snap = capture_resource_snapshot(include_gpu=self.include_gpu, include_gc=self.include_gc)
        gc0 = gc1 = gc2 = ""
        if snap.gc_counts is not None:
            gc0, gc1, gc2 = snap.gc_counts
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(
                f"{snap.ts:.3f},{epoch},{self.tag},"
                f"{snap.rss_kb if snap.rss_kb is not None else ''},"
                f"{snap.hwm_kb if snap.hwm_kb is not None else ''},"
                f"{snap.gpu_current_bytes if snap.gpu_current_bytes is not None else ''},"
                f"{snap.gpu_peak_bytes if snap.gpu_peak_bytes is not None else ''},"
                f"{gc0},{gc1},{gc2}\n"
            )
            if (self._epoch_count % self.flush_every) == 0:
                f.flush()


class ResourceGuard(Callback):
    """Abort training before the OS kills the process.

    Set `max_rss_gb` (or `max_rss_mb`) in config to enable.
    """
    def __init__(
        self,
        *,
        max_rss_mb: int | None = None,
        max_rss_gb: float | None = None,
        include_gpu: bool = True,
        print_every: int = 1,
    ):
        super().__init__()
        if max_rss_mb is None and max_rss_gb is None:
            raise ValueError("ResourceGuard requires max_rss_mb or max_rss_gb")
        self.max_rss_kb = int(max_rss_mb * 1024) if max_rss_mb is not None else int(float(max_rss_gb) * 1024 * 1024)
        self.include_gpu = bool(include_gpu)
        self.print_every = max(1, int(print_every))
        self._epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_count += 1
        rss_kb = _read_proc_status_kb("VmRSS")
        if (self._epoch_count % self.print_every) == 0:
            gpu_cur, gpu_peak = _read_gpu_mem_bytes() if self.include_gpu else (None, None)
            print(
                f"[RESOURCE] epoch={epoch} VmRSS_kB={rss_kb} limit_kB={self.max_rss_kb} "
                f"gpu_current_B={gpu_cur} gpu_peak_B={gpu_peak}"
            )
        if rss_kb is not None and rss_kb >= self.max_rss_kb:
            raise RuntimeError(
                f"ResourceGuard abort: VmRSS {rss_kb/1024/1024:.2f} GB exceeded limit {self.max_rss_kb/1024/1024:.2f} GB"
            )

__all__ = [
    'ReduceLROnPlateauWithCounter',
    'EarlyStoppingWithPatienceCounter',
    'ClearMemoryCallback',
    'MemoryUsageLogger',
    'ResourceUsageLogger',
    'ResourceGuard',
    'capture_resource_snapshot',
    'ResourceSnapshot',
]
