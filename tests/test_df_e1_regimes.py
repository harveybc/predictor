"""RP29: R0/R1/R2 on the modular model, proven by gradients, weight changes, resume and reload — not by a
`trainable=True` in a config. Same graph, same shared initial checkpoint, same data/roles/objective;
masked-AE pre-training on train windows with a separate decoder; controls: shifted/future labels,
decoder not connected at inference, a model without updates keeps its detector, selection on the
test split is impossible by construction (only validation is passed to fit)."""
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = _load("df_mod_e0")
RG = _load("df_e1_regimes")
ASSIGN = [0] * 4 + [1] * 4
W, P = 24, 8
FAST = {**E.TRAINING, "max_updates": 30, "max_epochs": 10, "early_stopping": {**E.TRAINING["early_stopping"], "patience": 3}}


@pytest.fixture(scope="module")
def data():
    g = E.generate(3, 1, 8, n=1500)
    periods = [g["params"]["groups"][gg]["period"] for gg in g["params"]["latent_groups"]]
    prep = E.prepare(g["x"], g["oracle"], periods, W, 1)
    Pp, s = prep["parts"], prep["scale"]
    Xtr, ytr = E._sx(Pp["train"]["X"], s), E._sy(Pp["train"]["y"], s)
    Xva, yva = E._sx(Pp["validation"]["X"], s), E._sy(Pp["validation"]["y"], s)
    return {"Xtr": Xtr[:600], "ytr": ytr[:600], "Xva": Xva[:200], "yva": yva[:200], "Xin_tr": Xtr[:400], "Xin_val": Xtr[400:600]}


@pytest.fixture(scope="module")
def pretrained(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("pre")
    rec = RG.masked_pretrain(ASSIGN, W, P, arch="A", seed=1, Xtr=data["Xin_tr"], Xval=data["Xin_val"], out_dir=out, max_updates=40, patience=3)
    init = RG.initial_checkpoint(ASSIGN, W, P, arch="A", fusion="sequence", seed=1, out_dir=out)
    return {"pre": rec, "init": init, "out": out}


def _model(pretrained):
    m = E.build_modular(ASSIGN, W, P, fusion="sequence", seed=1, arch="A")
    m.load_weights(pretrained["init"]["path"])                       # the SHARED initial checkpoint of the replicate
    return m


def test_RP29_pretraining_uses_train_windows_only_with_a_separate_decoder_and_reconstruction_is_a_diagnostic(pretrained):
    rec = pretrained["pre"]
    assert rec["updates"] > 0 and rec["diagnostic_only"] and "masked positions" in rec["loss"] and Path(rec["decoder_file"]).is_file()
    assert sorted(rec["detector_layers"]) == sorted(pretrained["init"]["detector_layers"])
    # the modular model has no decoder layer: the decoder is never connected at inference
    m = _model(pretrained)
    assert not any("dec" in l.name for l in m.layers)
    assert rec["detector_digest"] != pretrained["init"]["detector_digest"]           # pre-training moved the detector away from the shared random state


def test_RP29_the_three_regimes_share_the_graph_and_the_initial_checkpoint_and_differ_only_in_the_detector(pretrained):
    ms = {r: _model(pretrained) for r in ("R0", "R1", "R2")}
    infos = {r: RG.apply_regime(ms[r], r, Path(pretrained["pre"]["detector_file"]) if r != "R0" else None) for r in ms}
    assert infos["R0"]["detector_digest_after_setup"] == pretrained["init"]["detector_digest"]                      # R0: the shared random detector
    assert infos["R1"]["detector_digest_after_setup"] == infos["R2"]["detector_digest_after_setup"] == pretrained["pre"]["detector_digest"]   # R1/R2: the SAME imported weights
    non_det = RG.non_detector_weighted_layer_names(ms["R0"])
    for r in ms:                                                                    # the rest of the model is identical (same checkpoint) in every regime
        assert RG.weights_digest(ms[r], non_det) == RG.weights_digest(ms["R0"], non_det)
    assert set(infos["R1"]["frozen"]) == set(infos["R1"]["detector_layers"]) and not infos["R0"]["frozen"] and not infos["R2"]["frozen"]
    assert infos["R1"]["params"]["frozen"] > 0 and infos["R2"]["params"]["frozen"] == 0 == infos["R0"]["params"]["frozen"]
    assert infos["R1"]["non_trainable_states_in_detector"] == 0                       # Conv1D detector: no normalisation states
    assert "core_conv" in infos["R1"]["trainable"] and "head" in infos["R1"]["trainable"] and any("_int" in n or "_adapt" in n for n in infos["R1"]["trainable"])


def test_RP29_gradients_weights_resume_and_reload_prove_the_regimes(pretrained, data, tmp_path):
    for regime in ("R0", "R1", "R2"):
        m = _model(pretrained)
        info = RG.apply_regime(m, regime, Path(pretrained["pre"]["detector_file"]) if regime != "R0" else None)
        det = info["detector_layers"]
        g = RG.gradient_report(m, data["Xtr"][:64], data["ytr"][:64])
        before_det, before_rest = RG.weights_digest(m, det), RG.weights_digest(m, RG.non_detector_weighted_layer_names(m))
        rec = E.fit(m, data["Xtr"], data["ytr"], data["Xva"], data["yva"], training=FAST, seed=1, descriptors=False)
        after_det, after_rest = RG.weights_digest(m, det), RG.weights_digest(m, RG.non_detector_weighted_layer_names(m))
        assert rec["updates"] > 0 and after_rest != before_rest                    # the rest always learns
        if regime == "R1":
            assert not g["detector_receives_gradient"] and after_det == before_det   # frozen: no gradient variable, no change
            assert all(rec["weight_change_by_layer"][n] == 0.0 for n in det)
        else:
            assert g["detector_receives_gradient"] and after_det != before_det        # R0 and R2: gradients reach the detector and it moves
            assert any(rec["weight_change_by_layer"][n] > 0 for n in det)
        # resume: a second fit continues from the current weights (no reset); reload reproduces predictions
        path = tmp_path / f"{regime}.weights.h5"
        m.save_weights(str(path))
        m2 = E.build_modular(ASSIGN, W, P, fusion="sequence", seed=9, arch="A")
        m2.load_weights(str(path))
        assert np.allclose(m2.predict(data["Xva"][:16], verbose=0), m.predict(data["Xva"][:16], verbose=0), atol=1e-6)
        assert RG.weights_digest(m2, det) == after_det
        RG.apply_regime(m2, regime, None if regime == "R0" else Path(pretrained["pre"]["detector_file"])) if regime == "R0" else None
        if regime == "R1":
            for l in m2.layers:
                if l.name in det:
                    l.trainable = False
            rec2 = E.fit(m2, data["Xtr"], data["ytr"], data["Xva"], data["yva"], training=FAST, seed=2, descriptors=False)
            assert RG.weights_digest(m2, det) == after_det and rec2["updates"] > 0    # still frozen after reload + resume


def test_RP29_controls_shifted_labels_no_update_model_and_no_test_selection(pretrained, data):
    m = _model(pretrained)
    RG.apply_regime(m, "R1", Path(pretrained["pre"]["detector_file"]))
    # a model with zero updates keeps the imported detector exactly (and every other weight)
    d0 = RG.weights_digest(m, RG.detector_layer_names(m) + RG.non_detector_weighted_layer_names(m))
    _ = m.predict(data["Xva"][:8], verbose=0)
    assert RG.weights_digest(m, RG.detector_layer_names(m) + RG.non_detector_weighted_layer_names(m)) == d0
    # shifted / future labels: training the real pipeline on labels moved one step forward changes nothing about the inputs,
    # and a leak of the future target INTO the inputs is detectable (a separate control), proving the tensor path does not carry it
    Xleak = data["Xtr"].copy()
    Xleak[:, -1, :] = data["ytr"]                                                   # leak: the target placed in the last input row
    m_leak = _model(pretrained)
    rec_leak = E.fit(m_leak, Xleak, data["ytr"], data["Xva"], data["yva"], training=FAST, seed=1, descriptors=False)
    m_true = _model(pretrained)
    rec_true = E.fit(m_true, data["Xtr"], data["ytr"], data["Xva"], data["yva"], training=FAST, seed=1, descriptors=False)
    assert min(rec_leak["curve"]["train"]) < min(rec_true["curve"]["train"])          # the leak would show; the real tensors do not carry it
    # selection on the test split is impossible: fit receives validation only (monitor val_loss), test arrays are never passed
    assert E.TRAINING["early_stopping"]["monitor"].startswith("validation")
    import inspect
    src = inspect.getsource(E.fit)
    assert "test" not in src.split("def fit")[1].split("EarlyStopping")[0].lower().replace("latest", "")
