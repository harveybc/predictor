# Recipe Revision 3: Constant Channel Retired

Owner authorized proceeding. Musashi implemented the correction; Satoshi does
not need to repeat it. His multi-resolution expert design remains independent
of SAC. No extra owner decision is required for this integration work.

## New Common Input

- Identity: `recipe_a58261c0fdc0a59d`, recipe revision 3, serialization schema v2.
- Recipe SHA256: `a5da300389e0f0684720903eb027294f32efcb1717eee7f7e546ca0c3320f4a6`.
- Parent: `recipe_5a8f2155318ba7c7`; parent file remains byte-identical.
- Only `px.er_1` retired: 120 columns, same store, same survivor order/constants,
  same 13-slot stride-2 window, same model widths, regularization and business.
- Group indices, file hashes and parameter counts derived, not manually edited.
  Twenty-five conceptual groups still resolve to 31 branches.
- No interpolation, clipping, new transform, scaler fitting or checkpoint fit.
  `px.er_2` and the remaining efficiency-ratio columns stay unchanged.
- Original label-support hashes are recorded as parent evidence, not reused as
  expanded-support hashes. New consumers compute their own masks.

The native source-to-tensor probe matches the independent E3-M consumer exactly:
23x13x120, tensor hash
`82de5af3f22bc8b837ee99b3b873140cd1dc8cb1bd50884e56f9c4fcd9a57630`.
Actor, both critics, matching trunk operation and policy reload pass. Surviving
tensors match exactly on ALL common valid rows, not only the 23 probe rows.

| Period | Previous valid | New valid | Recovered | Remaining invalid |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 19514 | 21221 | 1707 | 588 |
| INNER_A | 2589 | 2805 | 216 | 281 |
| OBS_2024 | 5600 | 6208 | 608 | 13 |

The eight-transition real-price engine witness also passes with the new recipe:
exact actor inputs and action clock, HOLD only, zero trades. No financial fit,
profit measurement or TEST read. Thirty-one exporter/derivation tests pass.

## Interpretation and Remaining Work

The chosen T2/T3 representation is inherited as a starting point. Removing a
zero-valued channel is not proof that the same choices are optimal on expanded
support, nor that trading metrics or newly trained weights stay unchanged.
No old metrics are relabeled as results of revision 3. No expert weights enter
SAC. Migrating old specialist checkpoints is a separate, explicit operation.

Remaining invalid windows are not silently discarded price bars. The detailed
receipt identifies their nonfinite source columns: OBS_2024 has one px.er_2
NaN affecting thirteen windows; INNER_A also contains cross-asset/decomposition
warm-ups. The engine must continue handling protection, costs and forced flat
when the policy lacks an observation. Musashi owns the causal transition and
discount treatment plus matching the recipe's economic contract. The scientific
SAC campaign has NOT started; this retirement is complete, not the whole rollout.

## Handoff to Satoshi

Use the delivered `recipe_r3` bundle: RECIPE.json, FEATURE_LIST.json, GROUPS.json,
SCALER.json, READY.json, PROBE_REFERENCE.json and the SAC/engine witnesses.
Verify READY file hashes and pin the recipe/probe in the next consumer config.
Do not generate a parallel version of the same retirement or modify v2 in place.
Your clean-round launcher can consume these constants and topology without a
new selection. Keep any expert-label changes explicitly separate.

Implementation: heuristic-strategy `tools/retire_constant_recipe.py` and
`tests/unit_tests/test_constant_recipe_retirement.py`.
Reproduction: run the derivation with `--parent`, `--parent-sha`, `--store`,
`--manifest`, `--out`; then `consumer_probe.py emit`; then predictor's
`verify_sac_shared_recipe.py` with `--parent-recipe` and `--parent-sha` to verify
all common tensors and count newly valid rows. Use a new output directory.
Evidence is in `docs/audits/evidence/recipe_r3_retirement_20261024/`.
