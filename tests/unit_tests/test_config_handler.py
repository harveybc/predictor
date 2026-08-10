import pytest
import json
from unittest.mock import patch, mock_open
from app.config_handler import (
    load_config,
    save_config,
    save_debug_info,
    load_remote_config,
    save_remote_config,
)

# Mock data for tests
mock_config = {
    'x_train_file': './test_input.csv',
    'predictor_plugin': 'cnn',
    'batch_size': 64,
    'epochs': 20
}

mock_debug_info = {
    'mean_squared_error_0': 0.01,
    'mean_absolute_error_0': 0.005
}

mock_remote_config = {
    'status': 'success',
    'config': mock_config
}


# Test loading configuration from a file
def test_load_config():
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
        config = load_config('config.json')
        assert config == mock_config


# Test saving configuration to a file.
# compose_config is patched to identity because it resolves plugin default
# parameters through installed entry points, which is environment-dependent;
# this test targets the save_config file contract only.
def test_save_config():
    with patch("builtins.open", mock_open()) as mocked_file, \
         patch("app.config_handler.compose_config", side_effect=lambda c: c):
        config, path = save_config(mock_config, 'config_out.json')
        handle = mocked_file()
        handle.write.assert_called()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        assert json.loads(written_content) == mock_config
        assert config == mock_config
        assert path == 'config_out.json'


# Test saving debug information to a file
def test_save_debug_info():
    with patch("builtins.open", mock_open()) as mocked_file:
        save_debug_info(mock_debug_info, 'debug_out.json')
        handle = mocked_file()
        handle.write.assert_called()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        assert json.loads(written_content) == mock_debug_info


# Test loading remote configuration (via the load_remote_config migration alias)
def test_load_remote_config():
    with patch('requests.get') as mocked_get:
        mocked_get.return_value.status_code = 200
        mocked_get.return_value.json.return_value = mock_remote_config
        config = load_remote_config('http://example.com/config', 'user', 'pass')
        assert config == mock_remote_config


# Test saving remote configuration (via the save_remote_config migration alias)
def test_save_remote_config():
    with patch('requests.post') as mocked_post, \
         patch("app.config_handler.compose_config", side_effect=lambda c: c):
        mocked_post.return_value.status_code = 200
        result = save_remote_config(mock_config, 'http://example.com/config', 'user', 'pass')
        assert result is True


# The renamed remote helpers must stay importable under their historical names.
def test_remote_migration_aliases_point_to_current_functions():
    from app import config_handler
    assert config_handler.load_remote_config is config_handler.remote_load_config
    assert config_handler.save_remote_config is config_handler.remote_save_config


# merge_config moved to app.config_merger; config_handler re-exports it so the
# historical import path keeps working.
def test_merge_config_alias_points_to_config_merger():
    from app.config_handler import merge_config as aliased
    from app.config_merger import merge_config as canonical
    assert aliased is canonical


if __name__ == "__main__":
    pytest.main()
