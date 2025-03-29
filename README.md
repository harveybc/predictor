
# Predictor

## Description

The predictor project is a comprehensive tool for timeseries prediction, equipped with a robust plugin architecture. This project allows for both local and remote configuration handling, as well as replicability of experimental results. The system can be extended with custom plugins for various types of neural networks, including artificial neural networks (ANN), convolutional neural networks (CNN), long short-term memory networks (LSTM), and transformer-based models. Examples of the aforementioned models are included alongside with historical EURUSD and other training data in the **examples** directory.

## Installation Instructions

To install and set up the predictor application, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/harveybc/predictor.git
    cd predictor
    ```

2. **Add the clonned directory to the Windows or Linux PYTHONPATH environment variable**:

In Windows a close of current command line promp may be required for the PYTHONPATH varible to be usable.
Confirm you added the directory to the PYTHONPATH with the following commands:

- On Windows, run:
    ```bash
    echo %PYTHONPATH%
    ```

- On Linux, run:
    ```bash
    echo $PYTHONPATH 
    ```
If the clonned repo directory appears in the PYTHONPATH, continue to the next step. 

3. **Create and Activate a Virtual Environment (Anaconda is required)**:

    - **Using `conda`**:
        ```bash
        conda create --name predictor-env python=3.9
        conda activate predictor-env
        ```

4. **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

5. **Build the Package**:
    ```bash
    python -m build
    ```

6. **Install the Package**:
    ```bash
    pip install .
    ```

7. **(Optional) Run the predictor**:
    - On Windows, run the following command to verify installation (it uses all default valuex, use predictor.bat --help for complete command line arguments description):
        ```bash
        predictor.bat --load_config examples\config\phase_1\phase_1_ann_6300_1h_config.json
        ```

    - On Linux, run:
        ```bash
        sh predictor.sh --load_config examples\config\phase_1\phase_1_ann_6300_1h_config.json
        ```

8. **(Optional) Run Tests**:
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

9. **(Optional) Generate Documentation**:
    - Run the following command to generate code documentation in HTML format in the docs directory:
        ```bash
        pdoc --html -o docs app
        ```
10. **(Optional) Install Nvidia CUDA GPU support**:

Please read: [Readme - CUDA](https://github.com/harveybc/predictor/blob/master/README_CUDA.md)

## Usage

Example config json files are located in examples\config, for a list of individual parameters to call via CLI or in a config json file, use: **predictor.bat --help**

After executing the prediction pipeline, the predictor will generate 4 files:
- **output_file**: csv file, predictions for the selected time_horizon **(see defaults in app\config.py)**
- **results_file**: csv file, aggregated results for the configured number of iterations of the training with the selected number of training epochs 
- **loss_plot_file**: png image, the plot of error vs epoch for training and validation in the last iteration 
- **model_plot_file**: png image, the plot of the used Keras model
 
The application supports several command line arguments to control its behavior for example:

```
usage: predictor.bat --load_config examples\config\phase_1\phase_1_ann_6300_1h_config.json --epochs 100 --iterations 5
```

There are many examples of config files in the **examples\config directory**, also training data of EURUSD and othertimeseries in **examples\data** and the results of the example config files are set to be on **examples\results**, there are some scripts to automate running sequential predictions in **examples\scripts**.


### Directory Structure

```
predictor/
│
├── app/                                 # Main application package
│   ├── __init__.py                     # Package initialization
│   ├── cli.py                          # Command-line interface handling
│   ├── config.py                       # Default configuration values
│   ├── config_handler.py               # Configuration management
│   ├── config_merger.py                # Configuration merging logic
│   ├── data_handler.py                 # Data loading and saving functions
│   ├── data_processor.py               # Core data processing pipeline
│   ├── main.py                         # Application entry point
│   ├── plugin_loader.py                # Dynamic plugin loading system
│   ├── reconstruction.py               # Data reconstruction utilities
│   └── plugins/                        # Prediction plugins directory
│       ├── predictor_plugin_ann.py     # Artificial Neural Network plugin
│       ├── predictor_plugin_cnn.py     # Convolutional Neural Network plugin
│       ├── predictor_plugin_lstm.py    # Long Short-Term Memory plugin
│       └── predictor_plugin_transformer.py # Transformer model plugin
│
├── tests/                              # Test suite directory
│   ├── __init__.py                    # Test package initialization
│   ├── conftest.py                    # pytest configuration
│   ├── acceptance_tests/              # User acceptance tests
│   ├── integration_tests/             # Integration test modules
│   ├── system_tests/                  # System-wide test cases
│   └── unit_tests/                    # Unit test modules
│
├── examples/                           # Example files directory
│   ├── config/                         # Example configuration files
│   ├── data/                           # Example training data
│   ├── results/                        # Example output results
│   └── scripts/                        # Example execution scripts
│       └── run_phase_1.bat                 # Phase 1 execution script
│
├── concatenate_csv.py                  # CSV file manipulation utility
├── setup.py                           # Package installation script
├── predictor.bat                      # Windows execution script
├── predictor.sh                       # Linux execution script
├── set_env.bat                        # Windows environment setup
├── set_env.sh                         # Linux environment setup
├── requirements.txt                    # Python dependencies
├── LICENSE.txt                        # Project license
└── prompt.txt                         # Project documentation
```

## Example of plugin model:
```mermaid
graph TD

    graph TD

    subgraph SP_Input ["Input Processing (Features Only)"]
        I[/"Input (ws, num_channels)"/] --> FS{"Split Features"};

        subgraph SP_Branches ["Feature Branches (Parallel)"]
             FS -- Feature 1 --> F1_FLAT["Flatten"] --> F1_DENSE["Dense x M"];
             FS -- ... --> F_DOTS["..."];
             FS -- Feature n --> Fn_FLAT["Flatten"] --> Fn_DENSE["Dense x M"];
        end

        F1_DENSE --> M{"Merge Concat Features"};
        F_DOTS --> M;
        Fn_DENSE --> M;
    end

    subgraph SP_Heads ["Output Heads (Parallel)"]

        subgraph Head1 ["Head for Horizon 1"]
            M --> H1_DENSE["Dense x K"]; %% Input DIRECTLY from Merged Features
            H1_DENSE --> H1_BAYES{"DenseFlipout (Bayesian)"};
            H1_DENSE --> H1_BIAS["Dense (Bias)"];
            H1_BAYES --> H1_ADD{"Add"};
            H1_BIAS --> H1_ADD;
            H1_ADD --> O1["Output H1"];
        end


         subgraph HeadN ["Head for Horizon N"]
            M --> HN_DENSE["Dense x K"]; %% Input DIRECTLY from Merged Features
            HN_DENSE --> HN_BAYES{"DenseFlipout (Bayesian)"};
            HN_DENSE --> HN_BIAS["Dense (Bias)"];
            HN_BAYES --> HN_ADD{"Add"};
            HN_BIAS --> HN_ADD;
            HN_ADD --> ON["Output HN"];
        end

    end

    O1 --> Z((Final Output List));
    ON --> Z;


    subgraph Legend
         NoteM["M = config['intermediate_layers']"];
         NoteK["K = config['intermediate']"];
         NoteNoFB["NOTE: Diagram simplified - Feedback loops not shown."]; %% Updated Note
    end

    %% Styling (Earth Tones - Simplified)
    style H1_BAYES,HN_BAYES fill:#556B2F,stroke:#333,color:#fff;
    style H1_BIAS,HN_BIAS fill:#4682B4,stroke:#333,color:#fff;
    style NoteM,NoteK,NoteNoFB fill:#8B4513,stroke:#333,stroke-dasharray:5 5,color:#fff; %% Updated Note ID

```