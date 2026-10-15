"""Independent reviewer control RP03: masked loss (no gradient from invalid labels;
a NaN smuggled past the mask is caught, never averaged away).

CPU only, one tiny synthetic batch through the REAL model and loss. Exit 0 iff
all controls hold.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from tools.i6dmo import model as M  # noqa: E402  (flags before TF)
from tools.i6dmo import train as T  # noqa: E402
import tensorflow as tf  # noqa: E402

results = {}


def check(name, ok, detail):
    results[name] = {"ok": bool(ok), "detail": detail}


M.enable_determinism()
rng = np.random.default_rng(20261014)
features = [f"f{i}" for i in range(19)]
bundle = M.build_dmo(M.model_config(features))
model = bundle.forecast_model
x = tf.constant(rng.normal(size=(64, 13, 19)).astype("float32"))
mask_np = rng.random((64, 6)) > 0.4
mask_np[:, 0] = True
mask_np[0, 1:] = False
y_valid = rng.normal(size=(64, 6))


def grads_for(y_np):
    with tf.GradientTape() as tape:
        pred = tf.concat(model(x, training=True), axis=1)
        loss, _, _ = T.masked_normalized_loss(tf.constant(y_np, tf.float32), pred,
                                              tf.constant(mask_np))
    return float(loss), [g.numpy() for g in tape.gradient(loss, model.trainable_variables)]

# --- control 1: invalid labels contribute NOTHING — NaN vs 1e6 behind the mask,
# loss and every gradient tensor bit-identical ---
y_nan = np.where(mask_np, y_valid, np.nan)
y_huge = np.where(mask_np, y_valid, 1e6)
loss_a, grads_a = grads_for(y_nan)
loss_b, grads_b = grads_for(y_huge)
same = loss_a == loss_b and all(np.array_equal(a, b) for a, b in zip(grads_a, grads_b))
finite = math.isfinite(loss_a) and all(np.isfinite(g).all() for g in grads_a)
check("masked_labels_no_gradient", same and finite,
      f"loss {loss_a!r} == {loss_b!r}: {loss_a == loss_b}; all gradient tensors bit-identical: "
      f"{same}; finite: {finite}")

# --- control 2 (reviewer mutation): NaN smuggled past the mask (mask True) makes the
# training loss non-finite, which the fit loop types as NONFINITE_LOSS ---
y_smuggled = y_nan.copy()
j_row = int(np.flatnonzero(mask_np[:, 2])[3])
y_smuggled[j_row, 2] = np.nan  # mask stays True
loss_s, _, _ = T.masked_normalized_loss(tf.constant(y_smuggled, tf.float32),
                                        tf.concat(model(x, training=True), axis=1),
                                        tf.constant(mask_np))
check("smuggled_nan_not_averaged", not math.isfinite(float(loss_s)),
      f"loss with smuggled NaN = {float(loss_s)!r} (non-finite is caught by the "
      "NONFINITE_LOSS guard in fit_dmo)")

# --- control 3: the frozen-weight metrics path types the same smuggle ---
try:
    T.masked_split_metrics(y_smuggled, np.zeros_like(y_smuggled), mask_np)
    check("metrics_smuggle_typed", False, "masked_split_metrics accepted a NaN valid label")
except T.TrainDisposition as exc:
    check("metrics_smuggle_typed", exc.code == "NONFINITE_LABEL", f"typed {exc.code}")

# --- control 4: the bundle loader refuses a NaN valid label at load (fail closed) ---
sys.path.insert(0, str(ROOT / "tests" / "i6dmo"))
from tests.i6dmo import dmo_model_fixtures as fx  # noqa: E402
work = Path(sys.argv[1])
root = work / "bundle"
fx.write_bundle(root)
import hashlib
import json as _json
split_path = root / "fit.npz"
with np.load(split_path, allow_pickle=False) as z:
    arrays = {k: z[k].copy() for k in z.files}
r = int(np.flatnonzero(arrays["mask"][:, 1])[0])
arrays["target_price"][r, 1] = np.nan  # mask stays True
np.savez(split_path, **arrays)
manifest = _json.loads((root / "BUNDLE.json").read_text())
manifest["splits"]["fit"]["sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
(root / "BUNDLE.json").write_text(_json.dumps(manifest, sort_keys=True))
from tools.i6dmo import design as D  # noqa: E402
effective = D.effective_config(D.validate_design(fx.load_design_doc()))
try:
    T.load_data_bundle(root, effective)
    check("loader_smuggle_typed", False, "load_data_bundle accepted a NaN valid label")
except T.TrainDisposition as exc:
    check("loader_smuggle_typed", exc.code == "NONFINITE_LABEL", f"typed {exc.code}")

ok = all(v["ok"] for v in results.values())
print(json.dumps({"probe": "RP03_masked_loss", "ok": ok, "controls": results}, indent=1))
sys.exit(0 if ok else 1)
