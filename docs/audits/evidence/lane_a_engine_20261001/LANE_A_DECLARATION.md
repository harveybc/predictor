# Lane A — integrated engine declaration

Satoshi, successor technical lead · 2026-10-01 (orders ac125db9)

**Declared integrated revision.** The code tip is `3ecdb256`. Commits after it change only evidence and documentation.

- Base: owner-corrected package `da4ce7b4`.
- The engine ports, the facade, entry points, the flat/nested grammar and the remanifest tool are ported on top of it.
- M02's `56ef9728` and `3932d08c` are included.

## Tests (pinned versions: Python 3.12.13, TF 2.21.0, Keras 3.13.2; worker_b, CPU, crispdm-run 4G)

| Run | Count | Receipt |
|---|---|---|
| Checkout, envs/tensorflow — architecture_default, temporal, assembly, flat_parity_m04, legacy_boundaries, donor_remanifest, pretrain (M02), candidate_evaluator | 128 passed, 2 skipped (installed-only checks), 416.76 s | receipts_3ecdb256/suite3.log |
| Installed isolated venv (non-editable, no system/user site, tests run from a copy outside the checkout, same eight files, `M01_REQUIRE_INSTALLED=1`) | 129 passed, 1 xfailed, 452.29 s | receipts_3ecdb256/installed2/m01/pytest.log |
| Legacy boundaries, master dc72170e vs 3ecdb256 in the same venv | see below | receipts_3ecdb256/installed2/probe_diff.txt and the probe.json files |
| LTS b05d01d routes (broker, CSV, live selection) | baseline 21 passed; with the integrated predictor installed, 21 passed | receipts_3ecdb256/lts/ |
| Architecture proof, F=3 and F=7 (real Keras) | 4 passed at c0d7b07b; default-factor pin added in b2c99b8d, included in the 128/129 | arch_test.log, proof/ |

The xfail is a defect already present on master: `predictor_plugins/common` is not packaged, so legacy Keras plugins cannot import when the package is installed. It behaves the same on master and here.

Legacy boundaries, master vs integrated:

- **Entry points:** every legacy entry point is byte-identical. The only additions are `predictor.plugins: modular_temporal` and the four `modular.*` built-ins.
- **Configs:** 224 of 224 legacy configs resolve identically. The one new opt-in example config is the only one that differs.
- **Unchanged behaviour:** the prediction_provider direction route, the refusal of an unmigrated old config, and the legacy e2e results header, 90 metric labels and output file set are all identical.
- The shared OLAP outbox received 0 new envelopes during these runs.

## Engineering pilots (not scientific results)

Regenerated identities, in identities/LANE_A_IDENTITIES.json, built with the integrated engine on the real ECL channel order (321 channels; channel-order sha256 matches M04's manifest):

- **Candidate:** M04's pinned default point, corrected to the approved design (branch_steps 24, time factors [2,2,1]; both inside M04's bounds).
  - model config sha256 `4d5402a0d79be0c17d7560b7c6d9fdde9e6f56a5b32e2fff0a4ff7fdb7f67739`
  - M04 candidate identity `5c0e29e5feb0ad184e4440dc91c2e77a4b882cc252ee59419c456a02c7f9405e`
- **Donors:** 321 branch manifests, list sha256 `d46b98778df2a92ae78d03a8260f62d7a43d9049848778577d4681bd36da5cf6`.
  - Weight-free core manifest `d7b1ce0acb9b8970eb6977e3a6131b2e2085054c7cd5483088030a7c108c9616`.
  - The full core identity also binds the donor weight hashes M02 produces.
- **Fused materialization identity:** `ecfb808d668857538927104c1aaf4c0b689d08b2ada7170ee41eaf966defda8b`, row shape (24, 5136), float32, grid 1..24.
- **Size with 24 retained steps:**
  - 493,056 bytes per row.
  - Analytic: train (18,341 windows) 9,043,140,096 B; validation (2,609 windows) 1,286,383,104 B. That is twice the 12-step design's 4,521,570,048 / 643,191,552 B.
  - Measured on 512 real validation windows with R0 weights: 0.0180 s per row, peak RSS 1,286,475,776 B. The model build took 9.35 s.

## Scientific scores

None. Nothing in this lane fitted a model beyond synthetic test mechanics.

## Findings carried forward

- **M04:** its pinned default flat point (branch_steps 12, time factors [2,1,1]) is the superseded design. The integrated grammar refuses it, and a test pins that refusal.
- **Superseded donors:** 556c5f3e-architecture donors (12-step branches) are refused by `tools/modular_donor_remanifest.py`, not relabelled. Literal-params donors from da4ce7b4 are converted without touching their weights.
