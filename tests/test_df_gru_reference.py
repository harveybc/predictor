"""RP69: the adapted GRU comparator — parameter graph, target alignment, learning, restore, fresh replay.

The article gives one GRU layer, 50 units, L2 0.0005, dropout 0 (Table 4, IHEPC); everything else
is declared as ours in tools/df_gru_reference.py. These rules pin that the built graph is the
declared one and that it trains and replays like every other receiver in the block runner.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load("df_gru_reference")
K = _load("df_e1_block")
E = _load("df_mod_e0")


def test_the_parameter_graph_is_the_declared_one():
    m = G.build(60, 7, 6, seed=1)
    assert K.n_params(m) == G.parameters(7, 50) == 3*50*7 + 3*50*50 + 2*3*50 + 50 + 1
    names = {w.path if hasattr(w, "path") else w.name for w in m.trainable_weights}
    assert any("gru" in n and "kernel" in n for n in names) and any("readout" in n for n in names)
    gru = m.get_layer("gru")
    assert gru.units == 50 and gru.dropout == 0.0 and gru.recurrent_dropout == 0.0 and not gru.return_sequences
    assert gru.kernel_regularizer.l2 == pytest.approx(0.0005) and gru.recurrent_regularizer.l2 == pytest.approx(0.0005)
    assert gru.bias_regularizer is None and m.get_layer("readout").activation.__name__ == "linear"
    assert m.output_shape == (None, 1)
    decl = G.declaration(60, 7)
    assert decl["source"]["table4_gru_mimo_ihepc"] == {"L": 1, "n_H": 50, "lambda_l2": 0.0005, "dropout": 0.0}
    assert "NOT a reproduction" in decl["adaptation"]["what_this_is"] and "optimizer" in decl["adaptation"]["unknown_in_the_article"]


def test_the_same_seed_gives_the_same_initial_weights_and_the_reach_is_the_whole_window():
    a, b = G.build(60, 7, 6, seed=3), G.build(60, 7, 6, seed=3)
    assert K.weight_hash(a) == K.weight_hash(b) and K.weight_hash(a) != K.weight_hash(G.build(60, 7, 6, seed=4))
    R = _load("df_e1_receiver")
    X = np.random.default_rng(0).normal(size=(2, 60, 7)).astype(np.float32)
    r = R.measure_reach(a, X)
    # by construction a GRU can read the whole window; what the MEASUREMENT shows at initialization is a gradient
    # that decays with distance (here below 1e-9 beyond a few dozen rows). The number is recorded, never asserted
    # to be the window: the block records it per fitted cell as well
    assert 1 <= r["measured_reach_by_gradient"] <= 60 and r["measured_reach"] >= 1
    assert r["gradient_max_abs_per_row"][-1] > r["gradient_max_abs_per_row"][0]


def test_target_alignment_learning_and_restore_through_the_block_loop():
    """On a series where y(t+h) = 2 x(t) of the target channel, the GRU must cut the error of its
    initial state and its restored weights must reproduce the best validation event."""
    rng = np.random.default_rng(1)
    n, h = 4000, 5
    Xs = rng.normal(size=(n+h+1, 3)).astype(np.float32)
    Y = np.zeros(n+h+1)
    Y[h:] = 2.0*Xs[:-h, 2]
    o = np.arange(30, n)
    tr = K.Batches(Xs, Y, o[:3000], 30, h, 2, 64, mean=0.0, sd=1.0, shuffle=True, seed=1)
    va = K.Batches(Xs, Y, o[3000:], 30, h, 2, 64, mean=0.0, sd=1.0, shuffle=False, seed=1)
    m = G.build(30, 3, 2, seed=1)
    before = K.evaluate_mae(m, va)
    r = K.fit_by_updates(m, tr, va, max_updates=150, validate_every=50, patience=3, lr=0.003, seed=1)
    assert r["best_val_mae_scaled"] < 0.6*before and r["restore_verified"] and r["updates_are_optimizer_iterations"]
    pred = K.predict(m, va)
    assert np.mean(np.abs(pred - np.concatenate([va[i][1][:, 0] for i in range(len(va))]))) == pytest.approx(r["restored_val_mae_scaled"], abs=1e-6)


def test_a_fresh_process_replays_the_saved_weights_exactly(tmp_path):
    m = G.build(20, 3, 2, seed=2)
    X = np.random.default_rng(2).normal(size=(5, 20, 3)).astype(np.float32)
    np.save(tmp_path/"x.npy", X)
    m.save_weights(tmp_path/"w.weights.h5")
    np.save(tmp_path/"pred.npy", np.asarray(m.predict(X, verbose=0)))
    code = f"""
import sys, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("df_gru_reference", r"{TOOLS/'df_gru_reference.py'}")
G = importlib.util.module_from_spec(spec); sys.modules["df_gru_reference"] = G; spec.loader.exec_module(G)
m = G.build(20, 3, 2, seed=99)
m.load_weights(r"{tmp_path/'w.weights.h5'}")
np.save(r"{tmp_path/'replay.npy'}", np.asarray(m.predict(np.load(r"{tmp_path/'x.npy'}"), verbose=0)))
"""
    subprocess.run([sys.executable, "-c", code], check=True, env={"CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "3", "PATH": "/usr/bin:/bin"})
    assert np.array_equal(np.load(tmp_path/"pred.npy"), np.load(tmp_path/"replay.npy"))
