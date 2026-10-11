"""I6-N audit: grid enumeration + tiny Keras alignment probes (CPU, synthetic only)."""
import json, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from fractions import Fraction
H = [2,3,4,5,6,12,18,24,36,48,60,72]
M = [1,2,4,6]; KS = [1,2,3,4,8,9]
out = {"grid": {}}
for h in H:
    dx = sorted({h//k for k in KS if h % k == 0} | {1}, reverse=True)
    excluded_k = [k for k in KS if h % k]
    rows = [(m*h, d, 1 + m*h//d) for m in M for d in dx]
    n1 = (2*h, Fraction(h,2), (1 + Fraction(2*h)/Fraction(h,2)) if h % 2 == 0 else None)
    out["grid"][h] = dict(delta_x=dx, k_inadmissible=excluded_k, configs=len(rows),
                          n_min=min(r[2] for r in rows), n_max=max(r[2] for r in rows),
                          n_odd=sum(r[2] % 2 for r in rows), n1_anchor_integer=h % 2 == 0,
                          absolute_L_24_168_reachable={L: L in {m*h for m in M} for L in (24, 168)})
import numpy as np, tensorflow as tf
K = tf.keras
tf.keras.utils.set_random_seed(0)
def stage(x, kind, k):
    return K.layers.Conv1D(4, k, strides=2, padding=kind)(x)
def probe(n, stages, kind, k=2):
    inp = K.Input((n, 3)); x = inp; shapes = [n]
    try:
        for _ in range(stages):
            x = stage(x, kind, k); shapes.append(int(x.shape[1]))
    except Exception as e:
        return dict(error=type(e).__name__, shapes=shapes)
    if shapes[-1] < 1:
        return dict(shapes=shapes, error='EMPTY_TIME_AXIS')
    m = K.Model(inp, x)
    a = np.random.RandomState(1).randn(2, n, 3).astype("float32")
    b = a.copy(); b[:, -1, :] += 10.
    ya, yb = m(a).numpy(), m(b).numpy()
    last_reads_t = bool(np.abs(ya[:, -1] - yb[:, -1]).max() > 1e-6)
    return dict(shapes=shapes, last_output_reads_t=last_reads_t)
res = {}
for n in (5, 9, 55, 433, 28, 2, 3):
    for kind in ("causal", "valid", "same"):
        for st in (1, 2, 3):
            res[f"n{n}_{kind}_k2_s{st}"] = probe(n, st, kind, 2)
        res[f"n{n}_{kind}_k3_s1"] = probe(n, 1, kind, 3)
out["stride2"] = res
# LSTM readout: return_sequences=True + last == return_sequences=False
inp = K.Input((9, 3)); l = K.layers.LSTM(8, return_sequences=True)
seq = l(inp); m1 = K.Model(inp, seq[:, -1, :])
l2 = K.layers.LSTM(8, return_sequences=False); m2 = K.Model(inp, l2(inp)); l2.set_weights(l.get_weights())
x = np.random.RandomState(2).randn(4, 9, 3).astype("float32")
out["lstm_last_vs_nosequence_maxabs"] = float(np.abs(m1(x).numpy() - m2(x).numpy()).max())
# ClockView position code: position of origin t depends on n
def pos_t(n, spacing): return (n - 1) * spacing
out["clockview_origin_position_hours"] = {"h72_L144_dx36_n5": pos_t(5, 36), "h72_L432_dx36_n13": pos_t(13, 36),
                                          "h72_L432_dx8_n55": pos_t(55, 8), "h72_L432_dx1_n433": pos_t(433, 1)}
print(json.dumps(out, indent=1, default=str))
