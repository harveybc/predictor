# Supervised residual donor pilot

## Purpose

The existing autoencoder donors reconstruct raw `z_train` windows. Frozen R1
with those donors reached approximately `0.24819 MAE_z`, effectively the
seasonal naive (`0.24797`). This pilot instead trains the complete per-feature
architecture to forecast the 24-hour seasonal residual on TRAIN only, then
exports its 321 branch encoders and temporal core.

## Frozen execution

- Input: governed ECL L24/H1..24 TRAIN NPZ only.
- Internal split: final 10% of TRAIN origins for early stopping; 24 origins
  purged before it; target timestamps must end before validation input support.
- Seed: `2021` only for the pilot.
- Loss/optimizer: MAE + AdamW, matching the current best loss family.
- Architecture: one causal Conv1D branch per input, sequence-preserving fusion,
  two-block Transformer core with positional encoding and temporal output
  `(6, 8)`, direct multi-horizon head.
- Outputs: 321 branch donors, one core donor, exact R1/R2/R3 configs,
  `PRETRAIN.json`, and reload parity for every donor.
- Prohibited: outer validation during pretraining, test/holdout access, a second
  seed before the pilot result, and heuristic-strategy evaluation unless the
  resulting predictor beats its same-row naive.

## Acceptance

1. `PRETRAIN.json.status == COMPLETE` and `outer_validation_used == false`.
2. All 322 donors load with `conditioning_contract == OPERATIONAL`.
3. R1/R2/R3 strict configs build without fallback.
4. The subsequent predictor report includes MAE_z, same-row seasonal naive,
   skill per horizon, seed, selected epoch, updates, and cost.
5. Repeat with at most two further seeds only after a strict one-seed
   improvement over the current `grouped32` engineering incumbent.

## Verification before launch

- Six new contract tests pass.
- A tiny real TensorFlow fit exported two branch donors plus core and rebuilt
  all three regimes.
- Integrated modular suite: 155 passed, two upstream deprecation warnings.
