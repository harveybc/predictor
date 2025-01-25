import sys
import json
import pandas as pd
from typing import Any, Dict

from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
    remote_log
)
from app.cli import parse_args
from app.data_processor import (
    process_data,
    load_and_evaluate_model,
    run_prediction_pipeline
)
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args


def main() -> None:
    """
    The main entry point of the Predictor application.

    This function orchestrates the overall workflow:
    1. Parses command-line arguments.
    2. Loads default and file-based configurations (remote and/or local).
    3. Merges configurations with CLI arguments and unknown arguments.
    4. Loads and configures the specified plugin.
    5. Depending on the configuration, either loads and evaluates an existing model
       or processes data through the prediction pipeline.
    6. Saves configurations locally and/or remotely as specified.

    The configuration merging process ensures that the application can dynamically
    adjust its behavior based on user inputs and configuration files, including
    limiting the number of rows read from training and validation datasets using
    `max_steps_train` and `max_steps_test`.

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
            config_url=args.remote_load_config,
            username=args.username,
            password=args.password
        )
        print(f"Loaded remote config: {file_config}")

    # Local configuration file loading
    if args.load_config:
        file_config = load_config(config_path=args.load_config)
        print(f"Loaded local config: {file_config}")

    # First pass: Merge config with CLI args and unknown args WITHOUT plugin-specific parameters
    print("Merging configuration with CLI arguments and unknown args (first pass, no plugin params)...")
    unknown_args_dict = process_unknown_args(unknown_args)
    # Provide empty plugin_params in this pass to exclude plugin-specific configurations
    config = merge_config(
        base_config=config,
        plugin_params={},  # No plugin-specific parameters in the first pass
        file_config=file_config,
        cli_args=cli_args,
        unknown_args=unknown_args_dict
    )

    # Determine the plugin to use; default to 'ann' if not specified via CLI or config
    if not cli_args.get('plugin'):
        cli_args['plugin'] = config.get('plugin', 'ann')

    plugin_name: str = cli_args['plugin']
    print(f"Loading plugin: {plugin_name}")
    plugin_class, _ = load_plugin(plugin_namespace='predictor.plugins', plugin_name=plugin_name)
    plugin = plugin_class()
    # Override plugin parameters with the merged configuration
    plugin.set_params(**config)

    # Second pass: Merge config with the plugin's parameters (if any)
    print("Merging configuration with CLI arguments and unknown args (second pass, with plugin params)...")
    # Pass plugin-specific parameters to ensure they are recognized and merged appropriately
    config = merge_config(
        base_config=config,
        plugin_params=plugin.plugin_params,
        file_config=file_config,
        cli_args=cli_args,
        unknown_args=unknown_args_dict
    )

    # Decide whether to load and evaluate an existing model or run the prediction pipeline
    if config.get('load_model'):
        print("Loading and evaluating model...")
        load_and_evaluate_model(config=config, plugin=plugin)
    else:
        print("Processing and running prediction pipeline...")
        run_prediction_pipeline(config=config, plugin=plugin)

    # Save the current configuration locally if a save path is specified
    if config.get('save_config'):
        save_config(config=config, save_path=config['save_config'])
        print(f"Configuration saved to {config['save_config']}.")

    # Save the current configuration remotely if a remote save endpoint is specified
    if config.get('remote_save_config'):
        print(f"Remote saving configuration to {config['remote_save_config']}")
        remote_save_config(
            config=config,
            remote_url=config['remote_save_config'],
            username=config.get('username'),
            password=config.get('password')
        )
        print("Remote configuration saved.")


if __name__ == "__main__":
    main()
