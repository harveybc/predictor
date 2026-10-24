# SAC Engine Progress and Shared Support Note

The owner's instruction is to advance SAC independently of the expert-guided
lane's multi-resolution ZigZag demonstrations. Those demonstrations, their
weights and specialist labels are not dependencies or inputs of SAC.

## Implemented and Executed

- Native frozen-recipe observation bridge to the existing gym-fx engine.
  No call to the old rolling scaler, clipping or NaN substitution. No feedback
  field reaches the actor or critics. Engine business state is retained.
- Exact timestamp join: completed bar included, then action dispatched for
  the next open. Invalid windows, unexpected clocks and TEST bars are refused
  before reset, not interpolated or removed from the execution price stream.
- 27 focused CPU tests pass. The real engine integration uses four synthetic
  SAC updates and inspects the native replay's current/next tensors exactly.
- Real admitted-bar witness passes eight HOLD transitions with exact actor
  tensors and clocks. Zero financial fits, zero trades, no TEST read.
  Evidence: `docs/audits/evidence/shared_recipe_rereview_20261024/SAC_ENGINE_PROBE_SUPPORT.json`.

This verifies observation wiring, not the full economic rollout. The existing
backtrader path's annual financing/constant costs must not silently replace the
recipe's per-lot commission, side-specific swap, spread schedule and execution
rules. Musashi owns that mapping and verification, with no need to wait for the
expert label discussions. No new scientific campaign is running yet.

## Support Finding for Both Lanes

These are the exact accepted recipe masks, now joined to all admitted bars:

| Period | Admitted bars | Valid windows | No valid window |
| --- | ---: | ---: | ---: |
| TRAIN | 21809 | 19514 | 2295 |
| INNER_A | 3086 | 2589 | 497 |
| OBS_2024 | 6221 | 5600 | 621 |

The receipt lists nonfinite cells by feature and period. It separates source
cells from affected 13-slot windows; those counts are not interchangeable.
Warm-up accounts for some exclusions, but not all. In OBS_2024, `px.er_1` has
53 NaNs and `px.er_2` one. Direct comparison against the admitted close series
finds all 53 `px.er_1` NaNs on zero-close-change rows; its finite min and max are
both 1.0. This is consistent with an efficiency ratio's 0/0 case, not proof of
the producer formula until that code is traced. No source bar was missing in
this comparison. Do not reclassify the entire admitted price dataset as broken.

### Bounded Task for Satoshi

Please trace the producer of `px.er_1` and `px.er_2`, confirm the flat-price
semantics, and provide its path, revision and a minimal causal regression.
Determine whether retaining this constant-when-defined column was intentional.
Do not modify the accepted store or recipe in place, silently impute, remove a
column, repeat the full selection or launch a new fit for this diagnosis.
Any proposed mathematical convention/correction gets an explicit new identity
and an affected-stages assessment. This is separate from ZigZag label design.

Both lanes must preserve all price bars and business obligations while an input
window is unavailable. SAC cannot silently compress several engine bars into
one ordinary one-bar replay transition: discounting/reward aggregation must be
explicit and tested. Musashi will implement that part; Satoshi should state how
his existing HOLD-on-missing path reports coverage. No policy feedback is needed.

## Exact Next Work

1. Close no-observation transition and economic-accounting witnesses.
2. Pin native SAC optimizer, head and fixed-budget sparse evaluation settings;
   no expert buffer, hindsight action mask, rolling scaler or expensive ES loop.
3. Run the finite sequential SAC contrast on the primary GPU with restart-safe
   state, publication, TRAIN/VALIDATION metrics and checkpoint retention.
4. Keep the expert-guided lane independent on its assigned GPU. Compare only
   against explicitly matched supports and business accounting.
