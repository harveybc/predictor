# RAW Long Matched Data/Design/Model Extension

Owner scope: Codex helper, 2026-10-15. Operations, reporting, campaign, CLI and
deployment remain with the coordinating agent. No scientific result is claimed.
Authority: `../RAW_PRIORITY_20261015.md`.

## Interface

- `tools/i6dmo/design.py`: `LONG_MATCHED_SCOPE`, `ALL_MATCHED_SCOPE`,
  `LONG_MATCHED_CONTRAST_ID`, `LONG_MATCHED_POPULATION_RULES`,
  `matched_arm_set(arms)`, `matched_population_rules(arm)`,
  `matched_market_time_design(short_design, arm, data_designs, *, calendar=None,
  runtime=None)`, `matched_arm(doc)`, `seal_matched_long(short_design_path,
  out_dir, *, calendar=None, runtime=None)`.
- `seal_matched_long` emits `DESIGN.mg1l.json`, `DESIGN.mg12m.json`,
  `EXPERIMENT.mg1l.json`, `EXPERIMENT.mg12m.json` and
  `MATCHED_POPULATION_RULES.json`. Idempotent writes refuse conflicting bytes.
  Default runtime pins: TensorFlow 2.21.0 / Keras 3.13.2. The validated original
  short document is shared protocol ancestry, not the long run identity.
- `tools/i6dmo/common_population.py`: `matched_data_design`,
  `validate_matched_data_design`, `build_common_population`,
  `write_common_population`, `load_common_population`, `arm_tensors`,
  `replay_against_input` support either exact pair, never mixed/partial/all-four
  arms. `ARMS` and `HORIZONS` retain their short values; `LONG_ARMS` and
  `LONG_HORIZONS` describe the separate long contrast.
- Long populations carry their own horizons, contrast/rules hashes, ordered
  rows, arm identities and exclusions. Long-specific report/seal keys are
  `mg12m_slots_subset_of_mg1l`, `reference_closed_mg12_rows`,
  `mg12m_rows_before_intersection`. Exclusion arrays use
  `input_reason_mg1l` / `input_reason_mg12m`, with long-only reason names
  `INPUT_INVALID_MG1L_ONLY` / `INPUT_INVALID_MG12M_ONLY`.
- `tools/i6dmo/model.py`: `MATCHED_ARMS` adds `mg1l -> long_mg1` and
  `mg12m -> long_mg12`; `DECLARED_CAPACITY`, `capacity_record` and
  `matched_capacity_delta(hourly_record, downsampled_record)` cover both pairs.
  Market family recognition uses rule AND slot count AND horizons. The
  145-slot probe accumulates participation across fixed signed perturbations,
  checks look-ahead on every perturbation, and never changes weights. Short
  probing is unchanged.
- `tools/i6dmo/train.py`: `_elapsed_view` recognizes long family AND horizons,
  not an MG12 prefix. Existing `load_matched_population`, `load_matched_data`,
  `matched_scales`, `_prepare_matched` and `verify_matched_identity` consume
  the long population through the same validation path, without bypasses.
- `market_grid.py` needs no change: its existing observed market-slot sampler
  accepts both 145x1 and 13x12 windows, includes current t, and retains actual
  UTC offsets and closure flags.

## Frozen Question

Contrast: `I6-DMO-LONG-MG1-VS-MG12-MATCHED-R0`.
`mg1l`: 145 slots, spacing 1; `mg12m`: 13 slots, spacing 12. Both span 144
open-market hours and predict exact elapsed 18/24/36/48/60/72-hour
targets, issued hourly. Both use 19 RAW features, static NONE cadence, identical
per-head business/maturity masks, common FIT/INNER/DEVELOPMENT intersections,
one FIT source-bar-union scaler, and common FIT naive loss scales.

Architecture: branch 4, core 8/4, head hidden 8, six outputs, dynamic physical
time encoding. Measured totals: 30,226 (`mg1l`) and 4,882 (`mg12m`). Only heads
differ: `(145-13)*4*8*6 = 25,344`. This is an input/readout recipe comparison,
not evidence that sampling alone causes any predictive gain.

## Traceability

Tests: `tests/i6dmo/matched_long/test_data_design_model.py`.

| Requirement | Executable Evidence |
|---|---|
| Explicit long seal, exact scope/runtime; preserve short bytes | `test_long_seals_and_old_identity_bytes` |
| Only two supported exact pairs; reject short heads in long | `test_exact_supported_pairs_only` |
| Published production calendar/default-runtime files replay byte-identically | `test_published_long_designs_replay_production_seal` |
| Common masks, FIT union scaler, exact t+h, UTC, maturity, restart | `test_long_intersection_scaler_maturity_and_restart` |
| Long family/capacity; same trainer population/scales | `test_long_model_capacity_and_hourly_elapsed_view` |
| Real untrained graphs, measured capacity, dynamic context | `test_long_real_graph_capacity_and_dynamic_context` |
| No-lookahead and dropped-slot negative controls | `test_long_causality_probe_rejects_lookahead_and_dropped_slots` |
| No future influence on FIT statistics/population | `test_long_future_perturbation_does_not_refit_scaler_or_fit_population` |
| Identity, wrong naive and post-load tamper refusals | `test_long_identity_and_postload_tampering_refused` |
| Missing hourly source remains a typed exclusion | `test_long_missing_bar_exclusion_is_typed_not_filled` |
| Original short counts and capacity hashes | `test_short_capacity_records_replay_frozen_snapshot` |

Initial red evidence: 2 failed, 2 setup errors, before implementation, for
missing `LONG_MATCHED_SCOPE` and refused `mg1l`. The actual 145-slot build
subsequently reproduced a dead-ReLU participation false negative in the old
probe, before the long probe extension. The first eight focused tests then
passed under CPU-only `crispdm-run -m 2G -t 300s`.

Final focused verification: **13 passed, 2 protobuf deprecation warnings** in
10.48 seconds (11 tests above plus existing
`test_committed_matched_designs_validate_and_reseal_identically` and
`test_closed_identities_are_byte_identical_and_still_their_own_families`).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 crispdm-run -q -m 2G -t 120s -n i6dmo-long-final-focused -- python -m pytest -q -p no:cacheprovider tests/i6dmo/matched_long/test_data_design_model.py tests/i6dmo/test_matched_population.py::test_committed_matched_designs_validate_and_reseal_identically tests/i6dmo/test_matched_population.py::test_closed_identities_are_byte_identical_and_still_their_own_families
```

A broader first run had 89 passed / 1 digest-rebuild failure because an owned
code file was edited between the fixture build and rebuild (code bytes are part
of that identity). The stable combined 100-test rerun reached its 300-second
timeout (exit 124), so it is **not** claimed as passing. No admission refusal
was retried, no memory limit was lowered, and no full-suite pass is claimed.

The five production-calendar long files in this directory were generated with
`D.seal_matched_long` and default TF 2.21.0 / Keras 3.13.2 pins. Design digests:
`mg1l=91cca314a55619e78f888d235f29afc0685398144d36ab977a91ee434b212b6d`,
`mg12m=b1ea9c8dd3648d726aaeebefe048a1ba544d3411bc307196d862ba1251bd6028`.
`POPULATION_SEAL.json` and the control protocol file belong to the parent;
neither was written by this helper. No private input path is recorded here.

The repository method's referenced canonical Spanish document is absent in
this checkout. Existing root method state was read; no coordinator-owned state
was edited. This return and the local state record preserve helper traceability.

## Boundaries

Only synthetic temporary fixtures were materialized. No fit, GPU use, service
change, deployment, real-data write or TEST read. Synthetic CPU checks do not
certify execution under the sealed scientific runtime or scientific eligibility.
Existing schemas, seals and results are retained.

One legacy assertion in
`test_matched_fit_path.py::test_sc07_real_sealed_builds_three_inputs_78_channels_and_2304_delta`
compares the ENTIRE expanded `DECLARED_CAPACITY` registry to the old short-only
snapshot. That assertion needs to compare its short projection; the new short
replay test checks that projection, original measured counts and capacity hashes.
Neither the old snapshot nor that legacy test was rewritten by this helper.

Next action: coordinator's independent actual-path review and operations
integration, then a separately admitted execution of the two newly sealed arms.
No dispatch authority or scientific fit is conferred by these code checks.
