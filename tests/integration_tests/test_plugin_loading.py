"""Focused loading tests for the current predictor plugin architecture.

The active plugin surface is the `predictor.plugins` entry-point group backed
by the root `predictor_plugins` package (CNN and friends). The historical
encoder/decoder pair loading belongs to the feature-extractor project and is
not part of this repository.
"""
import pytest
from importlib.metadata import entry_points

from app.plugin_loader import load_plugin


def test_current_cnn_plugin_class_importable():
    from predictor_plugins.predictor_plugin_cnn import Plugin

    assert isinstance(Plugin.plugin_params, dict)
    assert len(Plugin.plugin_params) > 0


def test_load_plugin_from_installed_entry_points_or_skip():
    eps = entry_points().select(group='predictor.plugins')
    names = sorted({ep.name for ep in eps})
    if not names:
        pytest.skip(
            'predictor distribution not installed in this environment; '
            'predictor.plugins entry points unavailable'
        )
    plugin_class, required_params = load_plugin('predictor.plugins', names[0])
    assert isinstance(plugin_class.plugin_params, dict)
    assert required_params == list(plugin_class.plugin_params.keys())
