"""Reusable callbacks shared by predictor plugins."""
from __future__ import annotations
import gc
import os
import time
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
    def __init__(self, file_path: str, flush_every: int = 1):
        super().__init__()
        self.file_path = file_path
        self.flush_every = max(1, int(flush_every))
        self._epoch_count = 0
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("ts,epoch,VmRSS_kB,VmHWM_kB\n")

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
            f.write(f"{ts:.3f},{epoch},{rss if rss is not None else ''},{hwm if hwm is not None else ''}\n")
            if (self._epoch_count % self.flush_every) == 0:
                f.flush()

__all__ = [
    'ReduceLROnPlateauWithCounter', 'EarlyStoppingWithPatienceCounter', 'ClearMemoryCallback', 'MemoryUsageLogger'
]
