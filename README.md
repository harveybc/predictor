
# Predictor

## Description

The predictor project is a comprehensive tool for timeseries prediction, equipped with a robust plugin architecture. This project allows for both local and remote configuration handling, as well as replicability of experimental results. The system can be extended with custom plugins for various types of neural networks, including artificial neural networks (ANN), convolutional neural networks (CNN), long short-term memory networks (LSTM), and transformer-based models.

### Directory Structure

- `app/`: The main application package.
  - `cli.py`: Handles command-line argument parsing.
  - `config.py`: Stores default configuration values.
  - `config_handler.py`: Manages the loading, saving, and merging of configurations.
  - `config_merger.py`: Merges configurations from various sources.
  - `data_handler.py`: Handles the loading and saving of data.
  - `data_processor.py`: Processes input data and runs the prediction pipeline.
  - `main.py`: Main entry point for the application.
  - `plugin_loader.py`: Dynamically loads prediction plugins.
  - `plugins/`: Directory for prediction plugins.
    - `predictor_plugin_ann.py`: Predictor plugin using an artificial neural network.
    - `predictor_plugin_cnn.py`: Predictor plugin using a convolutional neural network.
    - `predictor_plugin_lstm.py`: Predictor plugin using long short-term memory networks.
    - `predictor_plugin_transformer.py`: Predictor plugin using transformer layers.

- `tests/`: Test modules for the application.
  - `acceptance`: User acceptance tests.
  - `system`: System tests.
  - `integration`: Integration tests.
  - `unit`: Unit tests.

- `README.md`: Overview and documentation for the project.
- `requirements.txt`: Lists Python package dependencies.
- `setup.py`: Script for packaging and installing the project.
- `set_env.bat`: Batch script for environment setup.
- `set_env.sh`: Shell script for environment setup.
- `.gitignore`: Specifies intentionally untracked files to ignore.
