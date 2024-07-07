
# Predictor

## Description

The predictor project is a comprehensive tool for timeseries prediction, equipped with a robust plugin architecture. This project allows for both local and remote configuration handling, as well as replicability of experimental results. The system can be extended with custom plugins for various types of neural networks, including artificial neural networks (ANN), convolutional neural networks (CNN), long short-term memory networks (LSTM), and transformer-based models.

## Installation Instructions

To install and set up the predictor application, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/harveybc/predictor.git
    cd predictor
    ```

2. **Create and Activate a Virtual Environment (Anaconda is required)**:

    - **Using `conda`**:
        ```bash
        conda create --name predictor-env python=3.9
        conda activate predictor-env
        ```

3. **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Build the Package**:
    ```bash
    python -m build
    ```

5. **Install the Package**:
    ```bash
    pip install .
    ```

6. **(Optional) Run the predictor**:
    - On Windows, run the following command to verify installation (it uses all default valuex, use predictor.bat --help for complete command line arguments description):
        ```bash
        predictor.bat tests\data\EURUSD_hour_2010_2020.csv
        ```

    - On Linux, run:
        ```bash
        sh predictor.sh tests\data\EURUSD_hour_2010_2020.csv
        ```

7. **(Optional) Run Tests**:
For pasing remote tests, requires an instance of [harveybc/data-logger](https://github.com/harveybc/data-logger)
    - On Windows, run the following command to run the tests:
        ```bash
        set_env.bat
        pytest
        ```

    - On Linux, run:
        ```bash
        sh ./set_env.sh
        pytest
        ```

8. **(Optional) Generate Documentation**:
    - Run the following command to generate code documentation in HTML format in the docs directory:
        ```bash
        pdoc --html -o docs app
        ```
9. **(Optional) Install Nvidia CUDA GPU support**:

Please read: [Readme - CUDA](https://github.com/harveybc/predictor/blob/master/README_CUDA.md)

## Usage

The application supports several command line arguments to control its behavior:

```
usage: predictor.bat tests\data\csv_sel_unb_norm_512.csv
```

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
