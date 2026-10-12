# I6N causal trace audit: addendum for Musashi

Base audit: `retsu/i6n-causal-trace-audit-20261012` @5a9858c1 (order 8956fced,
object commit cbbe1f16, H4 config fb70328d, sealed plan 1b83490b, champion
356e4acc). Audit run: one CPU process, 485 s, COMPLETE; 16 tests pass.
Assertions: 91 = 67 PROVEN_SAFE, 9 PROVEN_LEAK, 15 RISK_UNRESOLVED, 0 N/A.
Independent reviewer: IN PROGRESS at the time of this addendum (four fail-closed
mutations, rerun and diff); its REVIEW.md will be appended under this directory
when it lands. Nothing here changes the base audit's bytes.

This addendum applies Musashi's methodological clarification of 2026-10-12 and
separates the four categories the clarification requires.

## 1. Real input/target leakage from >t: NONE FOUND (PROVEN_SAFE)

Executed, exact-equality evidence (ALIGNMENT_CASES.jsonl, 28/28; MASK_PARITY.json):
- every window ends exactly at t and includes the observation at t; perturbing
  any source row after t leaves the input tensor byte-identical; perturbing the
  row at t changes the rightmost input; perturbing t+4h changes the H4 target
  and not the input; trajectory offsets are exactly +1,+2,+3,+4 h;
- feature scaler and target scale fitted on TRAIN are byte-identical under a
  large INNER sentinel and under later-fold perturbation; no weights, optimizer
  state or normalizers cross folds or candidates;
- market masks (weekend, forced-flat, purge, DST as elapsed instants) re-derived
  independently match the frozen support exactly on all four folds; the model
  and the H4 naive use exactly the same ordered (origin, target) identities in
  original price scale on all 27 local fits.
Consequence per clarification item 10: H72 continues automatically after H4.

## 2. Retrospective selection relative to the 2023 folds (FUTURE_RELATIVE_TO_2023_FOLDS)

Base finding C2 (four folds) is RECLASSIFIED from PROVEN_FUTURE_SELECTION_LEAK to
FUTURE_RELATIVE_TO_2023_FOLDS. Fact unchanged: the 20 RAW features are exactly
the union of the 14 EURUSD FS4 RAW winners (TEST_FREEZE e1449779), chosen on
VALIDATION weeks 2024-01-01..2024-12-30, i.e. after every 2023 fold cutoff.
Meaning: the 2023 fold scores are not an independent estimate; the fit, the
development optimization and the future TEST 2025 remain valid because TEST was
never read (verified: no row, metric or artifact dated 2025+ in the trace). The
features are hyperparameters chosen on VALIDATION. H4 and H72 are
DEVELOPMENT_TUNING, never advertised as expected out-of-sample performance.

## 3. In-sample bias of the ranking (labelled, not a leak)

Base findings E7 and D7 (four folds) are RELABELLED: the TRAIN half of the
per-fold fitness, 0.5*(MAE_TRAIN/naive_TRAIN + MAE_INNER/naive_INNER), is an
in-sample regularizer term; its scaler/target scale legitimately see TRAIN rows
after the scored TRAIN origin because that split is not an out-of-sample score.
It is not leakage toward INNER (scalers are TRAIN-only, proven) nor toward TEST.
The campaign fitness keeps its identity. Three numbers are reported separately:

| Fold | INNER n | INNER MAE | INNER naive | INNER skill | TRAIN skill (in-sample) |
|---|---|---|---|---|---|
| 2023Q1_DCA | 208 | 0.00151020 | 0.00150721 | +0.00198 (worse than naive) | -0.01783 |
| 2023Q2_DCA | 209 | 0.00127532 | 0.00129373 | -0.01423 | -0.00494 |
| 2023Q3_DCA | 368 | 0.00118409 | 0.00120707 | -0.01904 | -0.01971 |
| 2023Q4 | 355 | 0.00132592 | 0.00132963 | -0.00279 | -0.00813 |

Combined TRAIN/INNER fitness (campaign identity): -0.010586766321054819,
replayed exactly from retained sums and equal to the chain optimae.
INNER-only skill (unweighted mean of folds): -0.0085. Per-fold INNER skill
ranges from +0.0020 to -0.0190: the improvement is not uniform and one fold is
below naive. Recommendation retained for FUTURE campaigns only: DOIN ranking by
INNER-only; the TRAIN/INNER mean stays the level-1 early-stopping criterion.

## 4. Unresolved provenance (RISK_UNRESOLVED, never SAFE)

- A7: the six annual minute source archives are not present locally; source
  bytes verified only at the retained hourly layer.
- Seven of the 20 features cannot be reproduced from a producer in any sibling
  repo (PS1 producer code absent); price-derived features agree with the vendor
  layer at 61 %; dgs3mo.diff_5d / dgs7.logret_5d match their formulas at 86/81 %.
  STLFSI4 declared bound was loose; the +23 h as-of still precedes FRED's
  Thursday release. Action before TEST (clarification item 8): recover
  producers, formulas and clocks; do not drop features automatically.
- D5: the nominal NY calendar deviates from the retained bar clock in 18 weeks
  of 2023 (known DST session offset); no scored pair touches an unobserved bar.
- E3b: cross-island endpoint target centers differ at ~1e-20 (float reduction;
  no tolerance was declared, so left unresolved rather than passed).
- E9: the champion's row-level artifacts exist only on its island; metric parity
  is exact at sums level, row-level replay pending artifact replication.
- F0: local fits ran the evaluator at 9fbcf1ec (recovery-only diff vs cbbe1f16).

## Disposition

H4 result 356e4acc: DEVELOPMENT_TUNING (per clarification; equivalent to the
base audit's RETROSPECTIVE_DEVELOPMENT_ONLY). Not promotable as out-of-sample
performance; not rejected; campaign continues; H72 continues automatically.
TEST 2025 remains closed. Before any TEST or strategy use: resolve section 4.
