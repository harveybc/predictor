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

### 1. Binary predictor plugins (one for each existing architecture):
Each needs a `binary_` version:
- `binary_ann` — from predictor_plugin_ann.py
- `binary_cnn` — from predictor_plugin_cnn.py
- `binary_lstm` — from predictor_plugin_lstm.py
- `binary_transformer` — from predictor_plugin_transformer.py
- `binary_n_beats` — from predictor_plugin_n_beats.py
- `binary_tft` — from predictor_plugin_tft.py
- `binary_tcn` — from predictor_plugin_tcn.py
- `binary_mimo` — from predictor_plugin_mimo.py

### 2. Binary target plugin:
A new target plugin that produces binary labels instead of continuous future values.

### 3. Register all in setup.py

---

## Required Model Outputs (CRITICAL — must match PP inference plugins)

### Entry model outputs:
The model must have **exactly 2 output heads** with sigmoid activation:
```
buy_entry_binary  — P(buy TP hit before buy SL within weekly horizon)
sell_entry_binary — P(sell TP hit before sell SL within weekly horizon)
```
- **Output shape per head:** (batch, 1) float32 in [0, 1]
- **Output names:** `"buy_entry_binary"`, `"sell_entry_binary"`
- **Loss:** binary_crossentropy (per head)
- **Final activation:** sigmoid
- **PP thresholds at 0.5** (configurable) to produce the 0/1 signal

### Exit model outputs:
The model must have **exactly 1 output head** with sigmoid activation:
```
exit_binary — P(TP still reachable from current bar)
```
- **Output shape:** (batch, 1) float32 in [0, 1]
- **Output name:** `"exit_binary"`
- **Loss:** binary_crossentropy
- **Final activation:** sigmoid
- **Extra input features** appended to the feature window at inference: `[direction_feat, tp_distance_pips, sl_distance_pips]` (3 extra columns)

### Confidence/uncertainty (for Bayesian variants):
PP extracts uncertainty via MC dropout at inference time:
- `buy_confidence = max(0, 1 - 2*std)` across MC samples
- `sell_confidence = max(0, 1 - 2*std)` across MC samples
- `exit_confidence = max(0, 1 - 2*std)` across MC samples

So models with dropout layers will naturally support this — no special training needed.

---

## Required Model Inputs

Inputs are configurable (feature selection is applied upstream), but the labeled training data has:

### Feature columns (15 technical indicators from feature-eng):
RSI, MACD, MACD_Histogram, MACD_Signal, EMA, Stochastic_%K, Stochastic_%D, ADX, DI+, DI-, ATR, CCI, WilliamsR, Momentum, ROC

### OHLC columns:
OPEN, HIGH, LOW, CLOSE

### Input shape: `(batch, window_size, n_features)`
- Entry model default window: 64 bars
- Exit model default window: 32 bars
- n_features: configurable, depends on feature selection

---

## Training Data (Already Generated)

Labeled datasets are at `examples/data_downsampled/phase_1_b/`:
- `base_d1.csv` — 7,662 rows (train, 2005-06 → 2010-05)
- `base_d2.csv` — 1,902 rows (validation, 2010-05 → 2011-07)
- `base_d3.csv` — 1,912 rows (test, 2011-07 → 2012-10)
- `base_d4.csv` — 7,689 rows (train2, 2012-10 → 2017-09)
- `base_d5.csv` — 1,922 rows (validation2, 2017-09 → 2018-12)
- `base_d6.csv` — 2,135 rows (test2, 2018-12 → 2020-04)

Also includes `normalization_config_a.json` (mean/std) and `normalization_config_b.json` (min/max).

### Columns in the CSVs:
- `DATE_TIME` — index, datetime
- `OPEN`, `HIGH`, `LOW`, `CLOSE` — raw 4h OHLC prices
- `buy_entry_label` — binary (0/1), ~18-21% positive rate
- `sell_entry_label` — binary (0/1), ~17-24% positive rate
- `buy_exit_label` — binary (0/1), ~20-25% positive rate
- `sell_exit_label` — binary (0/1), ~20-28% positive rate
- `bars_to_friday` — integer (horizon info, available as feature)

### Label generation parameters (for reference):
- TP: 131.325 pipettes (tp_multiplier=5.15 × profit_threshold=25.50)
- SL: 93.33 pipettes (sl_multiplier=3.66 × profit_threshold=25.50)
- Worst-case costs: spread=30 pipettes, slippage=10, commission=$10/lot
- 4h EURUSD OHLC data, prediction_horizon=30 bars (≈1 week)
- Generated by `feature-eng/generate_phase1b_labels.py` using `feature-eng/app/plugins/oracle_labels.py`

---

## Binary Target Plugin Specification

A new `binary_target` plugin that maps the label columns to the y_train/y_val/y_test dict format the predictor pipeline expects:

### For entry training:
```python
y_train = {
    "buy_entry_binary": array(N, 1),   # from buy_entry_label column
    "sell_entry_binary": array(N, 1),   # from sell_entry_label column
}
```

### For exit training:
```python
y_train = {
    "exit_binary": array(N, 1),   # from buy_exit_label or sell_exit_label
}
```
The exit model is trained on interleaved buy/sell exit labels with direction as extra feature.

---

## Key Architectural Differences from Regression Plugins

| Aspect | Regression (current) | Binary (new) |
|--------|---------------------|--------------|
| Output activation | linear | **sigmoid** |
| Loss function | huber | **binary_crossentropy** |
| Output names | output_horizon_{h} | **buy_entry_binary, sell_entry_binary** (entry) or **exit_binary** (exit) |
| Output shape | (batch, 1) per horizon | (batch, 1) per signal |
| Metrics | MAE, R² | **accuracy, AUC, precision, recall** |
| Target values | continuous future price | **0 or 1** |
| De-normalization | yes (mean/std) | **no** (already probabilities) |
| Number of output heads | len(predicted_horizons) | **2 (entry) or 1 (exit)** |

---

## Implementation Notes

- **Do NOT modify existing regression plugins** — create new `binary_` files alongside them
- Each binary plugin inherits from the same base class as its regression counterpart
- Override `build_model()` to use sigmoid output + binary_crossentropy
- Override output_names to use the binary naming convention
- The `train()` method from BaseKerasPredictor should work with the binary target dict unchanged
- For Bayesian variants, keep DenseFlipout layers — MC dropout uncertainty works the same for binary outputs
- Save format: same `.keras` + `_metadata.json` convention
- Metadata must include: `feature_columns`, `window_size`, `model_type: "binary_entry"` or `"binary_exit"`

## Verification

After training, the models must:
1. Load in PP's `binary_entry_predictor` / `binary_exit_predictor` plugins
2. Accept input shape `(1, window_size, n_features)`
3. Output shape `(1, 2)` for entry or `(1, 1)` for exit
4. All outputs in [0, 1] range (sigmoid)
5. Work with MC dropout for uncertainty when called with `training=True`
