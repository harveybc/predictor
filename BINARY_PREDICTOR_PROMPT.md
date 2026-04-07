Please act as a senior software developer, machine learning expert and enemy of the mediocrity, one that hates medicre things like supposing things, guessing or improovising without reading the full involved code first, act as a professional developer instead.

# Predictor Repo: Add Binary Classification Predictor Plugins

## Context & Goal

The predictor repo currently trains **continuous regression** models that predict future price values at multiple horizons (output_horizon_1, output_horizon_5, etc.). These models are loaded by the prediction_provider (PP) repo for serving predictions via API.

We now need **binary classification** versions of each existing predictor plugin. These binary models will be loaded by PP's new `binary_entry_predictor`, `binary_exit_predictor`, and `binary_predictor` plugins (already created, inference-only, in the PP repo).

The binary models predict whether a trade's take-profit (TP) will be hit before its stop-loss (SL) within a weekly horizon — producing binary (0/1) signals instead of continuous price forecasts.

## What Already Exists in the Predictor Repo

### Current plugins (continuous/regression):
- `ann` — Dense layers (Bayesian, MC dropout)
- `cnn` — Conv1D + BiLSTM (Bayesian)
- `lstm` — BiLSTM (Bayesian)
- `transformer` — MultiHeadAttention + Conv1D (Bayesian)
- `n_beats` — N-BEATS blocks (deterministic)
- `tft` — Temporal Fusion Transformer (Bayesian)
- `tcn` — Temporal Convolutional Network (Bayesian)
- `mimo` — Multi-Input Multi-Output (Bayesian)

### Base classes (in predictor_plugins/common/):
- `BasePredictorPlugin` — core interface
- `BaseKerasPredictor` — Keras train loop, EarlyStopping, save/load
- `BaseBayesianKerasPredictor` — MC dropout uncertainty, KL annealing
- `BaseDeterministicKerasPredictor` — zero uncertainty

### Current output structure:
- `predicted_horizons = [1, 6, 24]` → output_names = ["output_horizon_1", "output_horizon_6", "output_horizon_24"]
- y_train / y_val are dicts: `{"output_horizon_1": array(N,1), ...}`
- Loss: Huber (regression)
- Output activation: linear (continuous values)

### Current target plugin:
- `default_target.py` computes targets by looking ahead h bars: `y[t] = baseline[t + h]`

## What Needs to Be Created

### 1. Binary predictor plugins — one per architecture, thin subclass:

Each binary plugin is a **thin subclass** of its regression counterpart. It inherits the full architecture (shared trunk with all layers) and only overrides `build_model()` to replace the multi-head regression output with a **single sigmoid head**.

The `signal_type` config parameter (`buy_entry`, `sell_entry`, `buy_exit`, `sell_exit`) selects which target column to train on. The same plugin file trains 4 separate models by invoking it 4 times with different `signal_type` configs.

**Files to create (8 total, one per architecture):**
- `predictor_plugin_binary_ann.py` — subclasses ANN Plugin
- `predictor_plugin_binary_cnn.py` — subclasses CNN Plugin
- `predictor_plugin_binary_lstm.py` — subclasses LSTM Plugin
- `predictor_plugin_binary_transformer.py` — subclasses Transformer Plugin
- `predictor_plugin_binary_n_beats.py` — subclasses N-BEATS Plugin
- `predictor_plugin_binary_tft.py` — subclasses TFT Plugin
- `predictor_plugin_binary_tcn.py` — subclasses TCN Plugin
- `predictor_plugin_binary_mimo.py` — subclasses MIMO Plugin

**What each binary plugin overrides:**
- `build_model()` — reuses parent's shared trunk, replaces the per-horizon output loop with a single `Dense(1, activation="sigmoid")` head
- `plugin_params` — adds `signal_type` (default: `"buy_entry"`), removes `predicted_horizons`
- Loss: `binary_crossentropy` instead of Huber
- Metrics: accuracy + AUC instead of MAE
- `self.output_names` — set to `["{signal_type}_binary"]` (e.g. `["buy_entry_binary"]`)

**What each binary plugin inherits unchanged:**
- All shared layers (Dense, Conv1D, LSTM, Attention, etc.)
- `train()` — works as-is since it accepts any `y_train` dict
- `save()` / `load()`
- `predict_with_uncertainty()` — MC dropout works identically for binary outputs
- Callbacks (early stopping, LR reduction, KL annealing)

### 2. Binary target plugin:
A new target plugin that reads the appropriate label column (based on `signal_type` from config) and returns the standard `y_train`/`y_val`/`y_test` dict format.

### 3. Register all in setup.py

---

## Required Model Outputs (CRITICAL — must match PP inference plugins)

### Each binary model has exactly 1 output:

**Buy entry model:**
```
buy_entry_binary — P(buy TP hit before buy SL within weekly horizon)
```

**Sell entry model:**
```
sell_entry_binary — P(sell TP hit before sell SL within weekly horizon)
```

**Buy exit model:**
```
buy_exit_binary — P(buy TP still reachable from current bar)
```

**Sell exit model:**
```
sell_exit_binary — P(sell TP still reachable from current bar)
```

For all models:
- **Output shape:** (batch, 1) float32 in [0, 1]
- **Loss:** binary_crossentropy
- **Final activation:** sigmoid
- **PP thresholds at 0.5** (configurable) to produce the 0/1 signal

### Exit model extra inputs at inference:
At PP inference time, exit models get 3 extra columns appended to the feature window: `[direction_feat, tp_distance_pips, sl_distance_pips]`. During training these are not present (the exit model learns purely from market features). PP handles appending them at inference.

### Confidence/uncertainty (for Bayesian variants):
PP extracts uncertainty via MC dropout at inference time:
- `confidence = max(0, 1 - 2*std)` across MC samples

Models with dropout layers naturally support this — no special training needed.

---

## Required Model Inputs

### Feature columns (from the normalized training data):
The model inputs are the **normalized feature columns** from the phase_1_b datasets:

| # | Column | Description |
|---|--------|-------------|
| 1 | `typical_price` | (HIGH + LOW + CLOSE) / 3, z-score normalized |
| 2 | `hod_sin` | sin(2π × hour / 24) |
| 3 | `hod_cos` | cos(2π × hour / 24) |
| 4 | `dow_sin` | sin(2π × day_of_week / 7) |
| 5 | `dow_cos` | cos(2π × day_of_week / 7) |
| 6 | `dom_sin` | sin(2π × (day_of_month - 1) / 31) |
| 7 | `dom_cos` | cos(2π × (day_of_month - 1) / 31) |
| 8 | `moy_sin` | sin(2π × (month - 1) / 12) |
| 9 | `moy_cos` | cos(2π × (month - 1) / 12) |
| 10 | `rolling_std_24` | 24-bar rolling std of typical_price |
| 11 | `rolling_ema_24` | 24-bar EMA of typical_price |
| 12 | `price_minus_ema` | typical_price − rolling_ema_24 |

**Total: 12 input features** (all z-score normalized using training set params).

### Input shape: `(batch, window_size, 12)`
- Default window: 64 bars (configurable)
- n_features: 12

---

## Training Data (Already Generated)

Labeled datasets are at `examples/data_downsampled/phase_1_b/`:

### Base files (un-normalized features + labels):
- `base_d1.csv` — 7,639 rows (train, 2005-06 → 2010-05)
- `base_d2.csv` — 1,879 rows (validation, 2010-05 → 2011-07)
- `base_d3.csv` — 1,889 rows (test, 2011-07 → 2012-10)
- `base_d4.csv` — 7,666 rows (train2, 2012-10 → 2017-09)
- `base_d5.csv` — 1,899 rows (validation2, 2017-09 → 2018-12)
- `base_d6.csv` — 2,112 rows (test2, 2018-12 → 2020-04)

### Normalized files (z-score normalized features + raw labels):
- `normalized_d1.csv` through `normalized_d6.csv` (same row counts)

### Normalization configs:
- `normalization_config_a.json` — z-score params computed from d1 → applied to d1, d2, d3
- `normalization_config_b.json` — z-score params computed from d4 → applied to d4, d5, d6

### Dataset roles:
- **d1**: train feature extractor (encoder/representation learning)
- **d2**: validate feature extractor
- **d3**: test feature extractor
- **d4**: train predictor (fine-tune or train head)
- **d5**: validate predictor
- **d6**: test predictor

### Columns in each CSV (base and normalized):
- `DATE_TIME` — datetime index
- 12 feature columns (see "Required Model Inputs" above)
- `buy_entry_label` — binary (0/1), ~18-21% positive rate
- `sell_entry_label` — binary (0/1), ~17-24% positive rate
- `buy_exit_label` — binary (0/1), ~20-25% positive rate
- `sell_exit_label` — binary (0/1), ~20-28% positive rate
- `bars_to_friday` — integer (horizon info, available as extra feature)

Labels are **not** normalized (kept as raw 0/1 in normalized files too).

### Label generation parameters (for reference):
- TP: 131.325 pipettes (tp_multiplier=5.15 × profit_threshold=25.50)
- SL: 93.33 pipettes (sl_multiplier=3.66 × profit_threshold=25.50)
- Worst-case costs: spread=30 pipettes, slippage=10, commission=$10/lot
- 4h EURUSD OHLC data, prediction_horizon=30 bars (≈1 week)
- Generated by `feature-eng/generate_phase1b_labels.py` using `feature-eng/app/plugins/oracle_labels.py`

---

## Binary Target Plugin Specification

A new `binary_target` plugin that maps a single label column to the y_train/y_val/y_test format. The `signal_type` parameter selects which column:

| signal_type | Target column | Output name |
|-------------|---------------|-------------|
| `buy_entry` | `buy_entry_label` | `buy_entry_binary` |
| `sell_entry` | `sell_entry_label` | `sell_entry_binary` |
| `buy_exit` | `buy_exit_label` | `buy_exit_binary` |
| `sell_exit` | `sell_exit_label` | `sell_exit_binary` |

### Target dict format:
```python
# For signal_type="buy_entry":
y_train = {"buy_entry_binary": array(N, 1)}

# For signal_type="sell_entry":
y_train = {"sell_entry_binary": array(N, 1)}

# etc.
```

Each model is trained on exactly one target column.

---

## Key Architectural Differences from Regression Plugins

| Aspect | Regression (current) | Binary (new) |
|--------|---------------------|--------------|
| Output activation | linear | **sigmoid** |
| Loss function | huber | **binary_crossentropy** |
| Output names | output_horizon_{h} | **{signal_type}_binary** (single head) |
| Output shape | (batch, 1) per horizon | **(batch, 1)** single output |
| Metrics | MAE, R² | **accuracy, AUC, precision, recall** |
| Target values | continuous future price | **0 or 1** |
| De-normalization | yes (mean/std) | **no** (already probabilities) |
| Number of output heads | len(predicted_horizons) | **1** |
| Models per architecture | 1 | **4** (buy_entry, sell_entry, buy_exit, sell_exit) |
| Input features | configurable | **12** (typical_price + seasonals + rolling) |

---

## Implementation Notes

- **Do NOT modify existing regression plugins** — create new `binary_` files alongside them
- Each binary plugin is a **thin subclass** (~40-60 lines) of its regression parent
- One plugin file per architecture (e.g. `predictor_plugin_binary_ann.py`)
- Override only `build_model()` — reuse the parent's shared trunk construction, replace the output loop
- `self.output_names` set to `[f"{signal_type}_binary"]` based on config
- The base `train()` method works unchanged since `y_train[output_name]` is just an array
- For Bayesian variants (ANN, CNN, LSTM, Transformer, TFT, TCN, MIMO): keep inheriting from parent which uses `BaseBayesianKerasPredictor` — MC dropout works identically for sigmoid outputs
- For deterministic variant (N-BEATS): inherit from parent which uses `BaseDeterministicKerasPredictor`
- Save format: same `.keras` + `_metadata.json` convention
- Metadata must include: `feature_columns`, `window_size`, `signal_type`, `model_type: "binary"`

### Example: binary_ann.py (illustrative skeleton)
```python
from .predictor_plugin_ann import Plugin as ANNPlugin
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC

class Plugin(ANNPlugin):
    plugin_params = {
        **ANNPlugin.plugin_params,
        "signal_type": "buy_entry",        # buy_entry | sell_entry | buy_exit | sell_exit
    }
    plugin_params.pop("predicted_horizons", None)  # not used for binary

    def build_model(self, input_shape, x_train, config):
        # Build ONLY the shared trunk from parent (do NOT call super().build_model)
        # Replicate parent's trunk layers, then add single binary head:
        # ... (parent trunk code: Input -> Flatten -> Dense layers -> trunk) ...
        
        signal_type = self.params.get("signal_type", "buy_entry")
        out_name = f"{signal_type}_binary"
        output = Dense(1, activation="sigmoid", name=out_name)(trunk)
        
        self.output_names = [out_name]
        self.model = Model(inputs=inputs, outputs=[output])
        self.model.compile(
            optimizer=...,
            loss={out_name: BinaryCrossentropy()},
            metrics={out_name: ["accuracy", AUC(name="auc")]},
        )
```

### Note on trunk reuse:
Since each parent's `build_model()` constructs both trunk AND output heads in one method, the binary subclass must **replicate the trunk construction** from the parent (copy the shared layers code) and then add its own single output. This is unavoidable because the parent doesn't expose the trunk as a separate method. The trunk code is typically 10-20 lines — small enough that copying is acceptable and cleaner than refactoring the parent.

## Verification

After training, each model must:
1. Load in PP's `binary_entry_predictor` / `binary_exit_predictor` plugins
2. Accept input shape `(1, window_size, 12)`
3. Output shape `(1, 1)` — single sigmoid probability
4. Output in [0, 1] range (sigmoid)
5. Work with MC dropout for uncertainty when called with `training=True`
