"""I6-N plan audit ADDENDUM, alignment lane: known-weight receptive-field controls.

CPU only. Reads NO market data (no TRAIN/VALIDATION/TEST rows). Builds small Keras
graphs from the audited worktree's own code and from the routes the proposal text
specifies, assigns KNOWN weights, feeds impulses, and reports which input positions
reach the last (head-read) position. Every measured set is printed beside an
independent analytic expectation, and positive/negative controls validate the
instrument. Output is deterministic JSON on stdout (no timings, no absolute paths).

Usage:
    python alignment_known_weights.py --repo <audited worktree root> --original-out <probes/cde/out.json>
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.dont_write_bytecode = True  # never write __pycache__ into the audited read-only worktree

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--original-out", required=True)
args = ap.parse_args()
REPO = os.path.abspath(args.repo)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

K = tf.keras
OUT = {"schema": "i6n.addendum.alignment.known_weights.v1"}


def rel(path):
    return os.path.relpath(os.path.abspath(path), REPO)


def file_ids(path):
    data = open(path, "rb").read()
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    return {"path": rel(path), "sha256": hashlib.sha256(data).hexdigest(), "git_blob": blob}


# ---------------------------------------------------------------- 0. environment and loaded code
from predictor_plugins.modular_temporal import components as C  # noqa: E402
from predictor_plugins.modular_temporal import common as CM  # noqa: E402
from predictor_plugins.modular_temporal import layers as LY  # noqa: E402
from tools import i6e_masked_model as I6E  # noqa: E402
from tools import i6m_model as I6M  # noqa: E402
from tools import fs4_temporal_predictor as FS4  # noqa: E402

OUT["environment"] = {
    "python": platform.python_version(), "tensorflow": tf.__version__,
    "keras": K.__version__ if hasattr(K, "__version__") else None, "numpy": np.__version__,
    "visible_gpus": len(tf.config.list_physical_devices("GPU")),
}
OUT["loaded_modules"] = {m.__name__: file_ids(m.__file__) for m in (C, CM, LY, I6E, I6M, FS4)}
OUT["loaded_from_audited_repo"] = all(not rel(m.__file__).startswith("..") for m in (C, CM, LY, I6E, I6M, FS4))
ORIG = os.path.abspath(args.original_out)
OUT["original_probe_output"] = {"sha256": hashlib.sha256(open(ORIG, "rb").read()).hexdigest()}

# ---------------------------------------------------------------- 1. grid (proposal:76-77, 115-118)
M_MULT, K_DIV = (1, 2, 4, 6), (1, 2, 3, 4, 8, 9)


def grid_for(h):
    rows = []
    for m in M_MULT:
        L = m * h
        dxs = sorted({h // k for k in K_DIV if h % k == 0} | {1}, reverse=True)
        for dx in dxs:
            rows.append({"m": m, "L": L, "delta_x": dx, "n": 1 + L // dx})
    return rows


GRID = {}
for h in (4, 72):
    rows = grid_for(h)
    ns = sorted({r["n"] for r in rows})
    GRID[h] = ns
    OUT.setdefault("grid", {})[str(h)] = {
        "N1_anchor": {"L": 2 * h, "delta_x": h // 2, "n": 1 + (2 * h) // (h // 2)},
        "N2_rows": len(rows),
        "N2_distinct_n": ns,
        "N2_even_n": [n for n in ns if n % 2 == 0],
        "N2_odd_n": [n for n in ns if n % 2 == 1],
        "k_inadmissible_non_integer_hours": [k for k in K_DIV if h % k],
        "rows": rows,
    }
ALL_N = sorted(set(GRID[4]) | set(GRID[72]))

# ---------------------------------------------------------------- 2. analytic receptive fields
def analytic_layer(n, conv, k, s):
    """Input index sets read by each output of one layer (indices < 0 or >= n are padding)."""
    if conv == "stride1_causal":
        return [list(range(i - k + 1, i + 1)) for i in range(n)]
    if conv == "keras_causal":
        out = (n + k - 1 - k) // s + 1
        return [list(range(i * s - (k - 1), i * s + 1)) for i in range(out)]
    if conv == "keras_valid":
        out = (n - k) // s + 1 if n >= k else 0
        return [list(range(i * s, i * s + k)) for i in range(out)]
    if conv == "keras_same":
        out = -(-n // s)
        pad_left = max((out - 1) * s + k - n, 0) // 2
        return [list(range(i * s - pad_left, i * s - pad_left + k)) for i in range(out)]
    if conv == "fs4_grouped":
        out = n // s
        return [list(range(i * s + s - 1 - (k - 1), i * s + s)) for i in range(out)]
    if conv == "origin_anchored":
        r = (-n) % s
        out = (n + r) // s
        return [list(range(i * s - r, i * s - r + s)) for i in range(out)]
    raise ValueError(conv)


def analytic_rf(n, stages):
    """Receptive field of the LAST output after a stack of (conv, k, s) stages."""
    sets = [{i} for i in range(n)]
    length = n
    for conv, k, s in stages:
        reads = analytic_layer(length, conv, k, s)
        sets = [set().union(*[sets[j] for j in r if 0 <= j < length]) if r else set() for r in reads]
        length = len(sets)
        if length == 0:
            return [], 0
    return sorted(sets[-1]), length


def out_len(n, conv, k, s):
    return len(analytic_layer(n, conv, k, s))


# ---------------------------------------------------------------- 3. measured receptive fields (known weights)
def make_layer(conv, k, s, length, kernel=None):
    """One single-channel layer with KNOWN weights: all-ones by default, or an explicit kernel."""
    if conv == "fs4_grouped":
        layer = FS4._grouped_layer_class()(1, 1, k, s, None)
        layer.build((None, length, 1))
        w = np.ones((1, k, 1), "float32") if kernel is None else np.asarray(kernel, "float32").reshape(1, k, 1)
        layer.set_weights([w, np.zeros((1, 1), "float32")])
        return [layer]
    layers = []
    if conv == "origin_anchored":
        r = (-length) % s
        if r:
            layers.append(K.layers.ZeroPadding1D((r, 0)))
        conv_layer = K.layers.Conv1D(1, s, strides=s, padding="valid", use_bias=False)
        k = s
    else:
        padding = {"stride1_causal": "causal", "keras_causal": "causal", "keras_valid": "valid",
                   "keras_same": "same"}[conv]
        conv_layer = K.layers.Conv1D(1, k, strides=s, padding=padding, use_bias=False)
    conv_layer.build((None, length + (len(layers) and (-length) % s), 1))
    w = np.ones((k, 1, 1), "float32") if kernel is None else np.asarray(kernel, "float32").reshape(k, 1, 1)
    conv_layer.set_weights([w])
    layers.append(conv_layer)
    return layers


def measured_rf(n, stages, kernels=None):
    x = tf.constant(np.eye(n, dtype="float32")[:, :, None])  # batch j = impulse at position j
    length = n
    for idx, (conv, k, s) in enumerate(stages):
        for layer in make_layer(conv, k, s, length, None if kernels is None else kernels[idx]):
            x = layer(x)
        length = int(x.shape[1])
        if length == 0:
            return [], 0
    last = x.numpy()[:, -1, 0]
    return [int(j) for j in np.nonzero(last != 0.0)[0]], length


def check(n, stages, kernels=None):
    exp, exp_len = analytic_rf(n, stages)
    if kernels is None:
        mea, mea_len = measured_rf(n, stages)
    else:
        mea, mea_len = measured_rf(n, stages, kernels)
    return {"n": n, "stages": [f"{c}:k{k}s{s}" for c, k, s in stages], "out_len": mea_len,
            "measured_last_rf": mea, "analytic_last_rf": exp, "match": mea == exp and mea_len == exp_len,
            "reads_t": (n - 1) in mea, "rf_size": len(mea)}


# 3a. instrument validation
ctrl = {}
ctrl["positive_stride1_causal_k3_n5_reads_t"] = check(5, [("stride1_causal", 3, 1)])
ctrl["positive_origin_anchored_n55_3stages"] = check(55, [("origin_anchored", 2, 2)] * 3)
x = tf.constant(np.eye(5, dtype="float32")[:, :, None])
crop = K.layers.Cropping1D((0, 1))(x)  # constructed to exclude t by design
cl = K.layers.Conv1D(1, 3, padding="causal", use_bias=False)
cl.build((None, 4, 1))
cl.set_weights([np.ones((3, 1, 1), "float32")])
ctrl["negative_crop_last_then_causal_k3_n5"] = {
    "measured_last_rf": [int(j) for j in np.nonzero(cl(crop).numpy()[:, -1, 0] != 0)[0]],
    "analytic_last_rf": [1, 2, 3], "reads_t": False}
ctrl["negative_crop_last_then_causal_k3_n5"]["match"] = (
    ctrl["negative_crop_last_then_causal_k3_n5"]["measured_last_rf"] == [1, 2, 3])
# weights-not-wiring demonstration (owner point 5b): one-hot kernels per tap
taps = {}
for q in range(3):
    kern = [[1.0 if i == q else 0.0 for i in range(3)]]
    taps[f"stride1_causal_k3_n5_onehot_tap{q}"] = check(5, [("stride1_causal", 3, 1)], kern)
taps["union_of_onehot_taps"] = sorted(set().union(*[set(v["measured_last_rf"]) for v in taps.values()]))
taps["all_ones"] = ctrl["positive_stride1_causal_k3_n5_reads_t"]["measured_last_rf"]
taps["note"] = ("tap0 (oldest) alone does NOT read t although t is wired: a no-change result can be a "
                "property of weights; structural RF = union over one-hot taps = all-ones RF")
ctrl["weights_vs_wiring"] = taps
# inherited GroupedCausalConv1D (tools/fs4_temporal_predictor.py:253-297), docstring: "the LAST output
# always covers the newest input row". FS4 used it on 24 -> 12 -> 6 (even lengths).
ctrl["inherited_fs4_grouped_k3s2"] = {str(n): check(n, [("fs4_grouped", 3, 2)]) for n in (24, 12, 6, 5, 9, 55)}
ctrl["inherited_fs4_grouped_k3s2_fs4_chain_24_12_6"] = check(24, [("fs4_grouped", 3, 1), ("fs4_grouped", 3, 2),
                                                                 ("fs4_grouped", 3, 2)])
OUT["instrument_controls"] = ctrl
OUT["instrument_valid"] = (ctrl["positive_stride1_causal_k3_n5_reads_t"]["match"]
                          and ctrl["positive_stride1_causal_k3_n5_reads_t"]["reads_t"]
                          and ctrl["positive_origin_anchored_n55_3stages"]["reads_t"]
                          and ctrl["negative_crop_last_then_causal_k3_n5"]["match"]
                          and taps["union_of_onehot_taps"] == taps["all_ones"])

# 3b. reproduce the ORIGINAL synthetic probe configurations with known weights
orig = json.load(open(ORIG))["stride2"]
cmp_rows, agree, disagree = [], 0, []
for key, val in sorted(orig.items()):
    n_s, kind, k_s, st_s = key.split("_")
    n, k, st = int(n_s[1:]), int(k_s[1:]), int(st_s[1:])
    conv = {"causal": "keras_causal", "valid": "keras_valid", "same": "keras_same"}[kind]
    exp, exp_len = analytic_rf(n, [(conv, k, 2)] * st)
    if "last_output_reads_t" not in val:
        known = {"error_or_empty": True, "analytic_out_len": exp_len}
        same = exp_len == 0
    else:
        mea, _ = measured_rf(n, [(conv, k, 2)] * st)
        known = {"known_weight_reads_t": (n - 1) in mea, "analytic_reads_t": (n - 1) in exp}
        same = known["known_weight_reads_t"] == val["last_output_reads_t"] == known["analytic_reads_t"]
    agree += bool(same)
    if not same:
        disagree.append({"key": key, "original": val, "known": known})
OUT["original_probe_vs_known_weights"] = {"configs": len(orig), "agree": agree, "disagree": disagree}

# 3c. routes x grid n (proposal:84-86, 122-130). Conventions are the candidate padding choices.
CONVS = [("keras_causal", 2), ("keras_causal", 3), ("keras_valid", 2), ("keras_same", 2),
         ("keras_same", 3), ("fs4_grouped", 3), ("origin_anchored", 2)]


def schedule(n, conv, k, stages, mode, rule):
    """Proposal:128-130. mode: none | progressive | final_only.
    rule 'out>=2': a factor-2 stage is applied only if its OUTPUT keeps >=2 positions;
    rule 'in>=2': applied whenever its INPUT has >=2 positions (literal alternative reading)."""
    plan, length = [], n
    for i in range(stages):
        reduce = mode == "progressive" or (mode == "final_only" and i == stages - 1)
        if reduce:
            ok = out_len(length, conv, k, 2) >= 2 if rule == "out>=2" else length >= 2
            if ok and out_len(length, conv, k, 2) >= 1:
                plan.append((conv, k, 2))
                length = out_len(length, conv, k, 2)
                continue
        plan.append(("stride1_causal", 3, 1))
    return plan


routes = {}
summary = {}
for h in (4, 72):
    for conv, k in CONVS:
        for mode in ("progressive", "final_only"):
            for rule in ("out>=2", "in>=2"):
                tag = f"h{h}|{conv}:k{k}|{mode}|{rule}"
                lost = []
                rows = []
                for n in GRID[h]:
                    for stages in (1, 2, 3):
                        plan = schedule(n, conv, k, stages, mode, rule)
                        r = check(n, plan)
                        r["stage_count"] = stages
                        r["reductions_applied"] = sum(1 for p in plan if p[2] == 2)
                        rows.append(r)
                        if not r["reads_t"]:
                            lost.append([n, stages])
                        if not r["match"]:
                            lost.append(["ANALYTIC_MISMATCH", n, stages])
                routes[tag] = rows
                summary[tag] = {"configs": len(rows), "t_lost": len(lost), "t_lost_at_[n,stages]": lost}
for h in (4, 72):
    rows = [dict(check(n, [("stride1_causal", 3, 1)] * s), stage_count=s) for n in GRID[h] for s in (1, 2, 3)]
    routes[f"h{h}|none"] = rows
    summary[f"h{h}|none"] = {"configs": len(rows), "t_lost": sum(not r["reads_t"] for r in rows),
                             "all_match": all(r["match"] for r in rows),
                             "rf_size_vs_n": sorted({(r["n"], r["stage_count"], r["rf_size"]) for r in rows})}
OUT["routes_summary"] = summary
OUT["routes_all_analytic_match"] = all(r["match"] for rows in routes.values() for r in rows)
OUT["routes_detail_digest"] = hashlib.sha256(json.dumps(routes, sort_keys=True).encode()).hexdigest()

# ---------------------------------------------------------------- 4. known-weight assignment for full graphs
def known_weights(model, scheme):
    """'ones': every kernel = 1. 'trig': kernel.flat[f] = sin(0.9 + 1.618 f) / sqrt(fan_in), a
    closed-form, full-rank, mixed-sign assignment (rank-1 or constant kernels are degenerate under
    LayerNormalization, which removes the channel mean and scale). Biases/beta 0, gamma 1 in both.
    Attention query/key kernels are zero in both schemes (uniform causal attention)."""
    for v in model.weights:
        path = getattr(v, "path", v.name)
        shape = tuple(v.shape)
        if path.endswith("gamma"):
            val = np.ones(shape)
        elif path.endswith("beta") or path.endswith("bias"):
            val = np.zeros(shape)
        elif ("/query/" in path or "/key/" in path) and path.endswith("kernel"):
            val = np.zeros(shape)
        elif scheme == "ones":
            val = np.ones(shape)
        else:
            fan_in = float(np.prod(shape[:-1])) if len(shape) > 1 else 1.0
            val = (np.sin(0.9 + 1.618 * np.arange(int(np.prod(shape)))) / math.sqrt(fan_in)).reshape(shape)
        v.assign(val.astype(v.dtype if isinstance(v.dtype, str) else "float32"))


def impulse_rf(model, n, channels, build_input, scheme):
    base = build_input(None)
    batch = np.stack([base] + [build_input(j) for j in range(n)]).astype("float32")
    y = model(batch, training=False).numpy()
    if y.ndim == 3:
        y = y[:, -1, :]  # the head reads the LAST position only
    y = y.reshape(len(batch), -1)
    moved = np.abs(y[1:] - y[:1]).max(axis=1)
    return [int(j) for j in np.nonzero(moved != 0.0)[0]]


# 4a. inherited residual temporal stage (components.py:317-341) and its guarded caller
res = {}
for n in (4, 5, 9, 10, 55, 433):
    entry = {}
    try:
        inp = K.Input((n, 3))
        model = K.Model(inp, C._residual_temporal_stage(inp, 3, 2, 3, "stage_0"))
        entry["builds"] = True
        entry["out_len"] = int(model.output_shape[1])
        for scheme in ("ones", "trig"):
            known_weights(model, scheme)

            def build(j, n=n):
                x = np.zeros((n, 3))
                if j is not None:
                    x[j, :] = 1.0
                return x
            entry[f"measured_last_rf_{scheme}"] = impulse_rf(model, n, 3, build, scheme)
        m = n // 2
        entry["analytic_last_rf"] = list(range(max(0, 2 * (m - 3)), 2 * m))
        entry["reads_t_trig"] = (n - 1) in entry["measured_last_rf_trig"]
        entry["trig_matches_analytic"] = entry["measured_last_rf_trig"] == entry["analytic_last_rf"]
    except Exception as exc:  # pragma: no cover - recorded
        entry = {"builds": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        C.transformer_conv(input_shape=(n, 4), time_grid=tuple(range(n)), output_steps=n // 2,
                           output_channels=4, name="tc",
                           params={"d_model": 8, "heads": 2, "blocks": 1, "ff_dim": 8,
                                   "stage_channels": [6, 5, 4], "time_factors": [2, 1, 1]})
        entry["transformer_conv_accepts"] = True
    except Exception as exc:
        tb = traceback.extract_tb(exc.__traceback__)[-1]
        entry["transformer_conv_accepts"] = False
        entry["transformer_conv_raises"] = f"{type(exc).__name__}: {exc}"
        entry["raised_at"] = f"{rel(tb.filename)}:{tb.lineno}"
    res[str(n)] = entry
OUT["inherited_residual_temporal_stage"] = res

# 4b. inherited i6m route (Transformer + causal Conv1D, stride 1) and N1 grid acceptance
acc = {}
for n in ALL_N:
    try:
        I6M.build_model(n, 1, 1)
        acc[str(n)] = "ACCEPTED"
    except ValueError as exc:
        acc[str(n)] = f"REFUSED: {exc}"
OUT["i6m_factory_acceptance_by_grid_n"] = acc
i6m_rf = {}
for h, n, dx in ((4, 5, 2), (72, 5, 36), (72, 55, 8)):
    model = I6M.build_model(n, 1, dx)
    for scheme in ("ones", "trig"):
        known_weights(model, scheme)

        def build(j, n=n):
            x = np.zeros((n, 3))
            x[:, 1] = 1.0  # source availability
            x[:, 2] = 1.0  # market presence
            x[:, 0] = 0.1 * np.arange(n) / n
            if j is not None:
                x[j, 0] += 1.0
            return x
        i6m_rf[f"h{h}_n{n}_dx{dx}_{scheme}"] = impulse_rf(model, n, 3, build, scheme)
OUT["i6m_route_last_rf"] = {"expected_all_positions": "0..n-1 (causal attention over every slot)",
                            "measured": i6m_rf,
                            "trig_equals_all_positions": {k: v == list(range(int(k.split("_")[1][1:])))
                                                          for k, v in i6m_rf.items() if k.endswith("trig")},
                            "ones_note": "all-ones kernels make every channel equal; LayerNormalization then "
                                         "outputs beta=0, so no position can move the head (degenerate instrument)"}

# 4c. N1 Conv1D-only route as written in proposal:85 (two causal Conv1D k=3 s=1, last state -> Dense(1))
n1 = {}
for n in (5, 9, 55, 433):
    inp = K.Input((n, 1))
    x = K.layers.Conv1D(16, 3, padding="causal")(inp)
    x = K.layers.Conv1D(8, 3, padding="causal")(x)
    out = K.layers.Dense(1)(x[:, -1, :])
    model = K.Model(inp, out)
    known_weights(model, "ones")

    def build(j, n=n):
        x = np.zeros((n, 1))
        if j is not None:
            x[j, 0] = 1.0
        return x
    rf = impulse_rf(model, n, 1, build, "ones")
    n1[str(n)] = {"measured_last_rf": rf, "analytic_last_rf": list(range(max(0, n - 5), n)),
                  "reads_t": (n - 1) in rf, "rf_size": len(rf), "history_slots_unread": n - len(rf)}
OUT["n1_conv1d_only_route"] = n1

# ---------------------------------------------------------------- 5. position encoding (AUD-16)
pe = {}
width = 32
for L in (8, 144, 432):
    for dx in (1, 8, 36):
        if L % dx:
            pe[f"L{L}_dx{dx}"] = "NOT_ON_GRID (L/delta_x not an integer)"
            continue
        n = 1 + L // dx
        z = tf.zeros((1, n, width))
        cv = I6E.ClockView("position", spacing_hours=float(dx))(z).numpy()[0]
        ordinal = LY.PositionalEncoding()(z).numpy()[0]
        pe[f"L{L}_dx{dx}"] = {"n": n, "t_position_hours_clockview": (n - 1) * dx,
                              "t_code_clockview_first6": [round(float(v), 6) for v in cv[-1, :6]],
                              "t_code_clockview_sha256": hashlib.sha256(np.round(cv[-1], 6).tobytes()).hexdigest()[:16],
                              "t_code_layers_ordinal_first6": [round(float(v), 6) for v in ordinal[-1, :6]]}
codes = {k: v["t_code_clockview_sha256"] for k, v in pe.items() if isinstance(v, dict)}
by_L = {}
for key, digest in codes.items():
    by_L.setdefault(key.split("_")[0], set()).add(digest)
rates = [1.0 / 10000.0 ** (2 * math.floor(b / 2) / width) for b in range(width)]
pe["t_code_depends_only_on_L"] = all(len(s) == 1 for s in by_L.values())
pe["t_code_distinct_across_L"] = len({next(iter(s)) for s in by_L.values()}) == len(by_L)
pe["bands_above_nyquist_(step>pi)_by_dx"] = {dx: sum(dx * r > math.pi for r in rates) for dx in (1, 2, 4, 7, 8, 36)}
pe["bands_wrapping_full_period_(step>2pi)_by_dx"] = {dx: sum(dx * r > 2 * math.pi for r in rates)
                                                     for dx in (1, 2, 4, 7, 8, 36)}
pe["origin_anchored_alternative_t_position_hours"] = 0
OUT["position_encoding"] = pe

# ---------------------------------------------------------------- 6. LSTM readout (AUD-17)
lstm_in = K.Input((9, 3))
seq_layer = K.layers.LSTM(4, return_sequences=True)
m_seq = K.Model(lstm_in, seq_layer(lstm_in)[:, -1, :])
last_layer = K.layers.LSTM(4, return_sequences=False)
m_last = K.Model(lstm_in, last_layer(lstm_in))
known_weights(m_seq, "trig")
last_layer.set_weights(seq_layer.get_weights())
xs = np.sin(1.0 + np.arange(2)[:, None, None] + 0.7 * np.arange(9)[None, :, None]
            + 0.3 * np.arange(3)[None, None, :]).astype("float32")
diff = float(np.abs(m_seq(xs).numpy() - m_last(xs).numpy()).max())
lstm_rf = impulse_rf(m_last, 9, 3, lambda j: (np.eye(9)[j][:, None] * np.ones((1, 3))) if j is not None
                     else np.zeros((9, 3)), "trig")
# placeholder handling: a gated zero slot vs a masked (skipped) slot
x_gap = xs.copy()
x_gap[:, 4, :] = 0.0
mask = np.ones((2, 9), bool)
mask[:, 4] = False
y_zero = last_layer(tf.constant(x_gap)).numpy()
y_masked = last_layer(tf.constant(x_gap), mask=tf.constant(mask)).numpy()
OUT["lstm"] = {"return_sequences_true_last_vs_false_maxabs": diff, "weights": "trig, identical in both",
               "last_state_rf_positions": lstm_rf,
               "zero_placeholder_vs_keras_mask_maxabs": float(np.abs(y_zero - y_masked).max()),
               "note": "a gated zero slot still advances the LSTM state; only an explicit mask skips it"}

# ---------------------------------------------------------------- 7. missing slots on the inherited route (AUD-09)
def i6m_input(n, nf, missing, placeholder, perturb_present=None):
    x = np.zeros((n, 2 * nf + 1), "float32")
    for f in range(nf):
        x[:, 2 * f] = 0.2 + 0.05 * f + 0.01 * np.arange(n)
        x[:, 2 * f + 1] = 1.0
    x[:, -1] = 1.0
    for j in missing:
        x[j, :-1] = placeholder
        x[j, -1] = 0.0
    if perturb_present is not None:
        x[perturb_present, 0] += 5.0
    return x


ms = {}
n, nf, dx, gaps = 5, 2, 36, (1, 2)
model = I6M.build_model(n, nf, dx)
for scheme in ("ones", "trig"):
    known_weights(model, scheme)
    y0 = model(i6m_input(n, nf, gaps, 0.0)[None]).numpy()
    y1 = model(i6m_input(n, nf, gaps, 7.5)[None]).numpy()
    yp = model(i6m_input(n, nf, (1,), 0.0, perturb_present=2)[None]).numpy()
    yb = model(i6m_input(n, nf, (1,), 0.0)[None]).numpy()
    ms[f"i6m_full_route_{scheme}"] = {
        "placeholder_perturbation_maxabs": float(np.abs(y1 - y0).max()),
        "positive_control_present_slot_perturbation_maxabs": float(np.abs(yp - yb).max()),
        "instrument_valid": bool(np.abs(yp - yb).max() > 0)}

# conv-only sub-route assembled from the SAME inherited pieces (clean, branch, gate, core convs; no attention)
inp = K.Input((n, 2 * nf + 1))
mask_t = I6E.ClockView("mask")(inp)
clean = I6E.ClockView("clean")(inp)
grid = tuple((i - n + 1) * dx for i in range(n))
parts = []
for f in range(nf):
    feat = I6E.ClockView("feature", index=f)(clean)
    br = C.causal_conv1d(input_shape=(n, 2), time_grid=grid, output_steps=n, name=f"b{f}",
                         params={"channels": 16, "kernel_size": 3})
    parts.append(I6E.ClockView("gate")([br.model(feat), mask_t]))
x = K.layers.Concatenate()(parts)
for ch in (24, 16, 8):
    r_ = K.layers.Conv1D(ch, 1)(x)
    u_ = K.layers.Conv1D(ch, 3, padding="causal", activation="gelu")(x)
    x = K.layers.LayerNormalization()(K.layers.Add()([r_, u_]))
    x = I6E.ClockView("gate")([x, mask_t])
conv_only = K.Model(inp, K.layers.Dense(1)(I6E.ClockView("last")(x)))
for scheme in ("ones", "trig"):
    known_weights(conv_only, scheme)
    y0 = conv_only(i6m_input(n, nf, gaps, 0.0)[None]).numpy()
    y1 = conv_only(i6m_input(n, nf, gaps, 7.5)[None]).numpy()
    yp = conv_only(i6m_input(n, nf, (1,), 0.0, perturb_present=2)[None]).numpy()
    yb = conv_only(i6m_input(n, nf, (1,), 0.0)[None]).numpy()
    ms[f"inherited_pieces_conv_only_{scheme}"] = {
        "placeholder_perturbation_maxabs": float(np.abs(y1 - y0).max()),
        "positive_control_present_slot_perturbation_maxabs": float(np.abs(yp - yb).max()),
        "instrument_valid": bool(np.abs(yp - yb).max() > 0)}
# negative instrument control: an UNGATED causal Conv1D route reading raw input (placeholder must leak)
inp = K.Input((n, 2 * nf + 1))
ung = K.Model(inp, K.layers.Dense(1)(K.layers.Conv1D(4, 3, padding="causal")(inp)[:, -1, :]))
known_weights(ung, "ones")
ms["negative_control_ungated_conv1d"] = {
    "placeholder_perturbation_maxabs": float(np.abs(ung(i6m_input(n, nf, (3,), 7.5)[None]).numpy()
                                                    - ung(i6m_input(n, nf, (3,), 0.0)[None]).numpy()).max())}
OUT["missing_slots"] = ms

print(json.dumps(OUT, indent=1, sort_keys=True, default=str))
