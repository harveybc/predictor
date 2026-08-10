import pytest
from unittest.mock import MagicMock, patch

from app.plugin_loader import load_plugin, get_plugin_params


class _FakePlugin:
    plugin_params = {'epochs': 10, 'batch_size': 256}


def _fake_entry_points(group_map):
    """Build a stand-in for importlib.metadata.entry_points() whose select()
    returns fake entry points for the configured groups."""
    eps = MagicMock()

    def select(group):
        entries = []
        for name, cls in group_map.get(group, {}).items():
            entry = MagicMock()
            entry.name = name
            entry.load.return_value = cls
            entries.append(entry)
        return entries

    eps.select.side_effect = select
    return eps


@patch('app.plugin_loader.entry_points')
def test_load_plugin_success(mock_entry_points):
    mock_entry_points.return_value = _fake_entry_points(
        {'predictor.plugins': {'cnn': _FakePlugin}}
    )
    plugin_class, required_params = load_plugin('predictor.plugins', 'cnn')
    assert plugin_class is _FakePlugin
    assert required_params == ['epochs', 'batch_size']


def test_load_plugin_missing_raises_import_error():
    with pytest.raises(ImportError) as excinfo:
        load_plugin('predictor.plugins', 'non_existent_plugin')
    assert 'Plugin non_existent_plugin not found in group predictor.plugins.' in str(excinfo.value)


@patch('app.plugin_loader.entry_points')
def test_get_plugin_params_success(mock_entry_points):
    mock_entry_points.return_value = _fake_entry_points(
        {'predictor.plugins': {'cnn': _FakePlugin}}
    )
    params = get_plugin_params('predictor.plugins', 'cnn')
    assert params == {'epochs': 10, 'batch_size': 256}


def test_get_plugin_params_missing_raises_import_error():
    with pytest.raises(ImportError) as excinfo:
        get_plugin_params('predictor.plugins', 'non_existent_plugin')
    assert 'Plugin non_existent_plugin not found in group predictor.plugins.' in str(excinfo.value)
