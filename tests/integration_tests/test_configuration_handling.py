import pytest
import json
from unittest.mock import patch, mock_open
from app import config_handler
from app.config import DEFAULT_VALUES


@pytest.fixture
def default_config():
    return DEFAULT_VALUES.copy()


def test_load_default_config(default_config):
    with patch('builtins.open', mock_open(read_data=json.dumps(default_config))):
        loaded_config = config_handler.load_config('config_in.json')
        assert loaded_config == default_config


# compose_config is patched to identity because it resolves plugin default
# parameters through installed entry points, which is environment-dependent;
# this test targets the save_config file contract only.
def test_save_config(default_config):
    m = mock_open()
    with patch('builtins.open', m), \
         patch('app.config_handler.compose_config', side_effect=lambda c: c):
        config_handler.save_config(default_config, 'config_out.json')
    m.assert_called_once_with('config_out.json', 'w')
    handle = m()
    written_content = "".join(call.args[0] for call in handle.write.call_args_list)
    assert json.loads(written_content) == default_config
