
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
%% Top Down layout

    %% Input Processing Subgraph
    subgraph "Input Processing (Features Only)"
        %% direction LR removed - Rely on default or TD from main graph
        I[/"Input (ws, num_channels)"/] --> FS{"Split Features"};

        subgraph "Feature Branches (Parallel)"
             %% direction TD removed - Rely on default or TD from main graph
             %% Layout branches Top-Down
             FS -- Feature 1 --> F1_FLAT["Flatten"] --> F1_DENSE["Dense x M"];
             FS -- ... --> F_DOTS["..."];
             FS -- Feature n --> Fn_FLAT["Flatten"] --> Fn_DENSE["Dense x M"];
        end

        %% Merging point
        F1_DENSE --> M{"Merge Concat Features"};
        F_DOTS --> M;
        Fn_DENSE --> M;
    end

    %% Output Heads Subgraph (Vertical Layout)
    subgraph "Output Heads (Parallel)"
        %% direction TD removed - Rely on default or TD from main graph
        %% Layout heads Top-Down

        %% Conceptual Link from Merged Features to all Heads
        M -- To Each Head --> HeadInput{Input to Heads};
        HeadInput -.-> Head1 subgraph;
        %% Dashed lines for clarity
        HeadInput -.-> HeadN subgraph;


        subgraph "Head for Horizon 1" id=Head1
            %% Control Action Feedback Path (from previous step's control output)
            LF1[/"self.local_feedback[0]"/] --> LF1_TILEFLAT["Tile/Flatten (Batch)"];

            %% Combine Merged Features (M) with Control Action Feedback via Concatenate
            M --> CONCAT1["Concatenate"];
            LF1_TILEFLAT --> CONCAT1;
            %% Concatenate control action feedback

            %% Head Processing Layers
            CONCAT1 --> H1_DENSE["Dense x K"];
            H1_DENSE --> H1_BAYES{"DenseFlipout (Bayesian)"};
            H1_DENSE --> H1_BIAS["Dense (Bias)"];
            H1_BAYES --> H1_ADD{"Add"};
            H1_BIAS --> H1_ADD;
            H1_ADD --> O1["Output H1"];
        end

        %% --- Other heads similar (...) ---

         subgraph "Head for Horizon N" id=HeadN
             %% Control Action Feedback Path (from previous step's control output)
            LFN[/"self.local_feedback[N-1]"/] --> LFN_TILEFLAT["Tile/Flatten (Batch)"];

            %% Combine Merged Features (M) with Control Action Feedback via Concatenate
            M --> CONCATN["Concatenate"];
            LFN_TILEFLAT --> CONCATN;
            %% Concatenate control action feedback

             %% Head Processing Layers
            CONCATN --> HN_DENSE["Dense x K"];
            HN_DENSE --> HN_BAYES{"DenseFlipout (Bayesian)"};
            HN_DENSE --> HN_BIAS["Dense (Bias)"];
            HN_BAYES --> HN_ADD{"Add"};
            HN_BIAS --> HN_ADD;
            HN_ADD --> ON["Output HN"];
        end
    end

    %% Loss Calculation Subgraph (Conceptual side process)
    subgraph "Loss Calculation per Head (Updates Feedback & Control Action Lists)"
       %% direction LR removed - Rely on default or TD from main graph
       %% Show loss as a separate flow
        subgraph LossHead1
             O1 --> Loss1["Global::composite_loss(...)"];
             Loss1 -- Updates --> LSE1[/"self.last_signed_error[0]"/];
             Loss1 -- Updates --> LSD1[/"self.last_stddev[0]"/];
             Loss1 -- Updates --> LMMD1[/"self.last_mmd[0]"/];
             Loss1 -- Updates --> LF1[/"self.local_feedback[0]"/]; %% Updated with ControlAction
        end
        subgraph LossHeadN
             ON --> LossN["Global::composite_loss(...)"];
             LossN -- Updates --> LSEN[/"self.last_signed_error[N-1]"/];
             LossN -- Updates --> LSDN[/"self.last_stddev[N-1]"/];
             LossN -- Updates --> LMMDN[/"self.last_mmd[N-1]"/];
             LossN -- Updates --> LFN[/"self.local_feedback[N-1]"/]; %% Updated with ControlAction
        end
    end


    %% Final outputs list (still conceptually gathered)
    O1 --> Z((Final Output List));
    %% Circle for final output aggregation
    ON --> Z;


    %% Legend Subgraph
    subgraph Legend
         NoteM["M = config['intermediate_layers']"];
         NoteK["K = config['intermediate']"];
         NoteListUpdate["Loss Updates: self.last_xxx (metrics) & self.local_feedback (control action)"];
         NoteInputFB["Head Input = Concat(Merged Features, Control Action Feedback)"];
    end

    %% Styling (Earth Tones)
    style H1_BAYES,HN_BAYES fill:#556B2F,stroke:#333,color:#fff;
    style H1_BIAS,HN_BIAS fill:#4682B4,stroke:#333,color:#fff;
    style LSE1,LSD1,LMMD1,LSEN,LSDN,LMMDN fill:#696969,stroke:#333,color:#fff;
    style LF1,LFN fill:#B8860B,stroke:#333,color:#fff; %% Style local_feedback nodes (DarkGoldenrod)
    style Loss1,LossN fill:#708090,stroke:#f00,stroke-dasharray:5 5,color:#fff;
    style NoteM,NoteK,NoteListUpdate,NoteInputFB fill:#8B4513,stroke:#333,stroke-dasharray:5 5,color:#fff;
    style CONCAT1,CONCATN fill:#D2B48C; %% Style concat node - Tan color

```