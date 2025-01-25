import sys
import json
from typing import Any, Dict

from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
    remote_log
)
from app.cli import parse_args
from app.data_processor import run_preprocessor_pipeline
from app.data_handler import load_csv
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args


def main() -> None:
    """
    The main entry point of the application.

    This function orchestrates the overall workflow:
    1. Parses command-line arguments.
    2. Loads default and file-based configurations.
    3. Merges configurations with CLI arguments.
    4. Loads and processes data using specified plugins.
    5. Executes the feature engineering pipeline.
    6. Saves configurations and logs data as specified.

    Raises:
        Exception: Propagates any exception that occurs during execution.
    """
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()
    cli_args: Dict[str, Any] = vars(args)

    print("Loading default configuration...")
    config: Dict[str, Any] = DEFAULT_VALUES.copy()

    file_config: Dict[str, Any] = {}
    # Remote configuration file loading
    if args.remote_load_config:
        file_config = remote_load_config(
            args.remote_load_config,
            args.username,
            args.password
        )
        print(f"Loaded remote config: {file_config}")

    # Local configuration file loading
    if args.load_config:
        file_config = load_config(args.load_config)
        print(f"Loaded local config: {file_config}")

    print("Merging configuration with CLI arguments and unknown args...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(
        base_config=config,
        plugin_params={},  # No plugin-specific parameters in the first pass
        file_config=file_config,
        cli_args=cli_args,
        unknown_args=unknown_args_dict
    )

    # Load data using data_handler with row limits
    print(f"Loading training data from {config['train_input_file']} with max rows: {config['max_steps_train']}...")
    training_data = load_csv(
        file_path=config['train_input_file'],
        headers=config.get('train_headers', False),
        max_rows=config['max_steps_train']
    )

    print(f"Loading validation data from {config['validation_input_file']} with max rows: {config['max_steps_test']}...")
    validation_data = load_csv(
        file_path=config['validation_input_file'],
        headers=config.get('validation_headers', False),
        max_rows=config['max_steps_test']
    )

    # Plugin loading and processing
    plugin_name: str = config.get('plugin', 'default_plugin')
    print(f"Loading plugin: {plugin_name}")
    plugin_class, _ = load_plugin('preprocessor.plugins', plugin_name)
    plugin = plugin_class()
    # Override plugin parameters with already configured params
    plugin.set_params(**config)
    plugin_params: Dict[str, Any] = getattr(plugin, 'plugin_params', {})

    print("Merging configuration with plugin-specific arguments...")
    config = merge_config(
        base_config=config,
        plugin_params=plugin_params,
        file_config=file_config,
        cli_args=cli_args,
        unknown_args=unknown_args_dict
    )

    print("Running the feature engineering pipeline...")
    run_preprocessor_pipeline(config, plugin, training_data, validation_data)

    # Save local configuration if specified
    if config.get('save_config'):
        save_config(config, config['save_config'])
        print(f"Configuration saved to {config['save_config']}.")

    # Save configuration remotely if specified
    if config.get('remote_save_config'):
        print(f"Remote saving configuration to {config['remote_save_config']}")
        remote_save_config(
            config,
            config['remote_save_config'],
            config['username'],
            config['password']
        )
        print("Remote configuration saved.")

    # Log data remotely if specified
    if config.get('remote_log'):
        print(f"Logging data remotely to {config['remote_log']}")
        remote_log(
            config,
            config['remote_log'],
            config['username'],
            config['password']
        )
        print("Data logged remotely.")


if __name__ == "__main__":
    main()
