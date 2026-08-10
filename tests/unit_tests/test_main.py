import pytest
from argparse import Namespace
from unittest.mock import patch

from app.main import main


@patch('app.main.load_plugin', side_effect=ImportError('predictor plugin unavailable'))
@patch('app.main.parse_args')
def test_main_exits_when_predictor_plugin_fails_to_load(mock_parse_args, mock_load_plugin):
    """main() must exit with code 1 when the predictor plugin cannot be loaded."""
    mock_parse_args.return_value = (
        Namespace(remote_load_config=None, load_config=None, username=None, password=None),
        [],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    mock_load_plugin.assert_called_once()


@patch('app.main.parse_args')
def test_main_exits_when_local_config_load_fails(mock_parse_args):
    """main() must exit with code 1 when the requested local config cannot be read."""
    mock_parse_args.return_value = (
        Namespace(
            remote_load_config=None,
            load_config='/nonexistent/config_in.json',
            username=None,
            password=None,
        ),
        [],
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
