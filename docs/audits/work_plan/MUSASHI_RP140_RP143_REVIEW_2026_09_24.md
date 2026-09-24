# RP140-RP143 scoped review and parallel news-track handoff

Target `07140d03`. Disposition: CHANGES_REQUIRED_COMPOSITION_AND_ML_GATES.
Read-only source review with delegated bounded CPU probes, followed by parent
source inspection. No original-array reduction, live warehouse revalidation,
training, model-weight download or production mutation by this review.
Frozen probe files/results: `../evidence/RP143_MUSASHI_REVIEW_2026_09_24/`.

## Scientific progress retained

Satoshi reports A's complete twelve-cell mean MSE0.161962/MAE0.259662 versus
0.158250/0.255750 published; stored values recompute to0.161961774/0.259662179.
Retained composition selects four report replays and eight IDENTITY_BOUND
histories, no METRIC_AND_SHAPE_BOUND selection. The first finding below does
not by itself invalidate that numerical mean. Accepted-content provenance
still needs the corrected checks; no claim of a newly audited full population.
UNKNOWN/INFERRED training UUID is correctly separate from repeatable predictions.
B's ten-of-twelve state and the TRAIN-only modular pilot are reported progress,
not newly observed live state in this document. No H1 scientific outcome yet.

## Findings

1. High: `_bind_replay_record` (`df_sota_repro.py`) calls metric-only evidence
   METRIC_AND_SHAPE_BOUND without checking shape or population; `composed_table`
   does not require stronger binding. A matching metric with shape[1,1,1] pools.
2. High: `_claim_for` admits reports with no `accepted_artifacts` because it
   checks only mismatches. Empty identity blocks with row_verified=true also
   bind, and a replay checkpoint without a root counterpart becomes IDENTITY_BOUND.
   Require both sides present and matching against current accepted evidence.
3. Medium, ML: `df_ecl_modular.target_identity` and its test read the author
   dataset directly, not delivered adapter batches. Their agreement cannot
   detect a batching/channel-order defect in `_TrainWindows`. Current targets
   are not demonstrated wrong; the required independent test is missing.
4. Medium, ML: AE validation `_TrainWindows` advances RNG on each access.
   Repeating the same batch changes20 mask positions in a bounded probe. The
   pilot's curve is therefore not loss on a fixed reconstruction task. No
   validation-based selection was claimed in that pilot; preserve that scope.
5. Medium, ML: reload parity accepts `or d1 == d2`, so loading the same incorrect
   donor into both regimes can pass. R1/R2 diagnostics measure gradients, not
   actual updates. Require donor-source equality, fresh reload prediction parity
   and post-optimizer-step weights for frozen/trainable regimes separately.

Orders [RP144-RP151](../../handoffs/MUSASHI_RP144_RP151_NEWS_AND_EXPERIMENTS_2026_09_24.md)
repair the affected acceptance path before the scientific modular contrast,
while B and the independent news-model integration continue. No wholesale reset.

## News-track implementation and review

New independent repo `news-signal`: strict broker-free typed adapter, pinned Laya
SDK, local checkpoint manifest, refusal-first timestamps/token/device handling,
fixture CLI, README, agent instructions, test design and submission draft.
Real weights, calibration, news collector, governance integration and broker
connections are explicitly NOT MEASURED/NOT DEPLOYED. A second review found
offline flags set after imports and unstructured extreme numeric/runtime errors;
both were corrected with negative tests. All model output remains shadow-only.
Alpaca/MT5 integration reuses LTS risk and execution, not a duplicate transport.

Resource observation: coordinator4070 has8188MiB VRAM with6485MiB free,46C and
about19GiB host RAM available at inspection; no CUDA compute process listed.
This supports testing it as resident inference host, not a measured Laya budget.
External5090 retains priority for heavy work. No GPU load launched by this update.

## Verification scope

- news-signal: fresh venv editable install, fixture CLI,26 CPU tests; no real SDK
  inference or live broker call. Wheel build checked separately.
- predictor: program validator PASS,34 documentary regression tests.
- RP143 probes: actual functions with disposable synthetic evidence/NumPy batches;
  SDK/framework fixtures do not establish trained-model quality.
- No owner credentials or capital decision needed for independent engineering.
  Missing feed/broker entitlements, if encountered, must be reported precisely.
