#!/usr/bin/env python
"""Interface conformity tests for predictor plugins.

Checks:
- build_model creates outputs with expected names output_horizon_<H>.
- predict_with_uncertainty returns two lists of equal length (#horizons) with (N,1) arrays.
- Deterministic plugin (n_beats, base) uncertainty zeros shape.
"""
import numpy as np
import importlib
import types

PLUGIN_MODULES = [
    'predictor_plugins.predictor_plugin_ann',
    'predictor_plugins.predictor_plugin_cnn',
    'predictor_plugins.predictor_plugin_lstm',
    'predictor_plugins.predictor_plugin_transformer',
    'predictor_plugins.predictor_plugin_n_beats',
    'predictor_plugins.predictor_plugin_base',
]

HORIZONS = [1,2]
INPUT_SHAPE = (8,1)
N_SAMPLES = 4


def _synth_data():
    X = np.random.randn(N_SAMPLES, *INPUT_SHAPE).astype('float32')
    y = {f'output_horizon_{h}': np.random.randn(N_SAMPLES,1).astype('float32') for h in HORIZONS}
    return X,y


def test_plugin_interfaces():
    X,y = _synth_data()
    for module_name in PLUGIN_MODULES:
        mod = importlib.import_module(module_name)
        assert hasattr(mod,'Plugin'), f"{module_name} missing Plugin class"
        plugin = mod.Plugin({'predicted_horizons': HORIZONS})
        # Build
        plugin.build_model(INPUT_SHAPE, X, {'predicted_horizons': HORIZONS})
        assert hasattr(plugin,'output_names')
        assert plugin.output_names == [f'output_horizon_{h}' for h in HORIZONS]
        # Prepare train targets format: DL expects dict, RF expects dict; unify
        history, train_preds, train_unc, val_preds, val_unc = plugin.train(
            X, y, epochs=1, batch_size=2, threshold_error=0, x_val=X, y_val=y, config={'predicted_horizons':HORIZONS,'plotted_horizon':HORIZONS[0], 'mc_samples':3}
        )
        assert isinstance(train_preds, (list,tuple))
        assert isinstance(train_unc, (list,tuple))
        assert len(train_preds)==len(HORIZONS)==len(train_unc)
        for arr in train_preds:
            assert arr.shape[0]==N_SAMPLES
        for u in train_unc:
            assert u.shape[0]==N_SAMPLES
        # Predict
        preds, uncs = plugin.predict_with_uncertainty(X, mc_samples=3)
        assert len(preds)==len(uncs)==len(HORIZONS)
        for p in preds: assert p.shape[0]==N_SAMPLES and p.shape[1]==1
        for u in uncs: assert u.shape[0]==N_SAMPLES and u.shape[1]==1
