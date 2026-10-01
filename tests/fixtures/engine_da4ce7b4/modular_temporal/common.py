"""Validation, canonical serialization, hashing, and temporal-grid helpers."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _copy(value):
    return json.loads(_json(value))


def _digest(value):
    return sha256(_json(value).encode("utf-8")).hexdigest()


def _file_hash(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def weights_hash(model):
    """Ordered shape/dtype/byte identity, independent of trainability and names."""
    h = sha256()
    for weight in model.get_weights():
        h.update(_json([list(weight.shape), str(weight.dtype)]).encode("ascii"))
        h.update(np.ascontiguousarray(weight).tobytes())
    return h.hexdigest()


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _partition(grid, steps):
    _positive_int(steps, "output_steps")
    if steps > len(grid) or len(grid) % steps:
        raise ValueError("Temporal reduction requires exact divisibility")
    return tuple(grid[len(grid) // steps - 1::len(grid) // steps])


def _keys(value, allowed, label):
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f"Invalid {label}: expected keys in {sorted(allowed)}")
