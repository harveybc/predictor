# FS-REP decision rule (frozen before any computation)

Lane FS-REP of `SATOSHI_FEATURE_SELECTION_CLOSURE_2026_10_05.md` §6. Frozen at the
commit that introduces this file; `tools/fs_rep_dispositions.py` implements it
literally and refuses to run if its embedded `RULE_VERSION` differs from the one
below. Changing the rule requires a new version here first.

`RULE_VERSION = fs_rep_rule.v3`

v3 (2026-10-05, Musashi correction order §1.4/§4): the output class is
`PROVISIONAL_EXTRACTIBILITY_DISPOSITION`. Every disposition is an
extractibility/probe statement that may prioritise RAW or a family for M4; it
does NOT claim downstream improvement of latents while trained-family G4 is
`NOT_APPLICABLE` (the real raw-vs-latent comparison is M4, after the
manifest). The column FS-CLOSE consumes is `PLUS_EXTRACTIBILITY_EVIDENCE`
(`candidate_decisions.csv`), formerly PLUS_REP. New disposition
`NOT_AVAILABLE_FOR_TRAIN`: when no terminal of a candidate admitted any
representation, raw included (all three FAILED), the aggregator may not default
to `RAW`; the row carries the cause (`NO_OBSERVED_TRAIN_VALUES` when the
receipts' reason text or the config declares it, else
`ALL_TERMINALS_FAILED_CAUSE_UNDECLARED`), representation and extractibility
cells are `NOT_APPLICABLE` with the FAILED receipt digests, and the only flag is
`TRAIN_SUPPORT_ABSENT` (never `RAW`, `NO_TRAINED_ADVANTAGE` or
`DECIDED_WITH_FAILED_FAMILIES`). The case where raw has support and only
trained families fail is unchanged (`RAW` with `DECIDED_WITH_FAILED_FAMILIES`).
Regression tests cover both. Candidate dispositions do not change the 366/137
denominators.

v2 (2026-10-05): G4 rewritten for the FS-CLOSE export, which carries
identity-only refits. No other clause changed.

Roles only. No host names, IPs or GPU identifiers are written by the tool.

## 0. What is decided

Extractibility decides **how to represent** a heavy candidate, not whether it
survives. One decision per heavy candidate (137 = features with
`in_extractibility_queue` in `coverage_reconciliation.json`):

| Decision | Meaning |
|---|---|
| `<trained family>` (`ae`, `dae`, `masked_temporal_ae`, `past_to_current_siamese`) | that trained encoder passed every gate and beat raw |
| `RAW` | raw/identity strictly dominates every trained family that was measured |
| `NO_TRAINED_ADVANTAGE` | no trained family passed the gates and raw does not strictly dominate either (trained ≈ raw within fold resolution); raw kept by parsimony and cost |
| `PENDING` | at least one of the three PS3-R terminals of the candidate is not yet terminal (neither COMPLETED nor FAILED with receipt) |
| `NOT_AVAILABLE_FOR_TRAIN` | all three terminals FAILED, so no representation (raw included) has TRAIN support; cause column filled; nothing here can train on it |

Reconstruction, ACF/spectrum/extremes/DTW, stability and effective dimension are
reported in every row and **never enter the decision**. Good reconstruction
without downstream utility does not select; poor reconstruction does not reject.

## 1. Evidence admitted

TRAIN-only. Each candidate has three PS3-R terminals (`ut_pilot_run.v1`,
`status == COMPLETED`, `results_sha256` verified against the file bytes):

| terminal role | families in the terminal | code identity | planned by |
|---|---|---|---|
| `baseline` | identity, random, ae, dae | feature-extractor `8987c57…` | `automation/baseline_batch_002_4090.tsv`, `automation/baseline_batch_001_003_5090.tsv` |
| `alt_mtae` | identity, random, masked_temporal_ae | feature-extractor `31add12…` | `automation/alternative_5090.tsv` |
| `alt_p2c` | identity, random, past_to_current_siamese | feature-extractor `31add12…` | `automation/alternative_5090.tsv` |

A terminal is admitted only if: seed 0; the five inner folds
`inner_2019..inner_2023`; `series_sha256` equals the PS2 batch digest of the
candidate's batch; the recipe matches the frozen one (window 168, latent 8,
`max_fit_windows 16384`, `max_val_windows 0`, `max_ref_windows 256`,
`probe_lags 0,1,2,23`, `ridge_alpha 1.0`, `max_epochs 200`, `patience 10`); the
row kinds and probe contract are complete (same checks as
`m2_readiness/ps3r_manifest_ingestor.py`: `Y_s` h0..5 mae+mse, `Y_l` h0..5
mae+mse, `Y_b` h0..1 log_loss+brier; one `probe_delta` per fold × trained family
× target × horizon; one `feature_summary`). Cap-measurement runs
(`--max_epochs 1`) and sealed older passes fail the recipe check and are not
admitted. Two COMPLETED copies of one planned cell with different
`results_sha256` make the cell `FAILED/CONTRADICTORY_TERMINAL`. A planned cell
whose directory holds a `FAILED*.json` receipt and no COMPLETED manifest is
`FAILED` with the receipt digest; a cell FAILED and later COMPLETED is COMPLETED.

The `identity` and `random` rows exist in all three terminals of a candidate.
Per the lane D1 decision they use identical probe rows (same
`train_row_ids_sha256`); the tool verifies that digest agrees across the three
terminals and flags `PROBE_ROWS_DIFFER` otherwise (decision still proceeds
within each terminal, because `probe_delta`/preservation are computed inside a
terminal, never across terminals). Raw metrics in the table come from the
`baseline` terminal when present, else from the first admitted alternative.

Receipt digest per row = `results_sha256` of the admitting terminal.

## 2. Cells and fold-level comparison

Business-target cells, `C` (14):
`(Y_s,h)` h=0..5 loss `mae`; `(Y_l,h)` h=0..5 loss `mae`; `(Y_b,h)` h=0..1 loss `log_loss`.

Within a terminal, for trained family F and cell c, the pilot writes per fold:

* `delta_probe_random_minus_trained` = L(random) − L(F)
* `preservation_raw_minus_trained` = L(identity) − L(F)

Differences of 1e-5/1e-6 are kept, never rounded. Their resolution is judged by
sign agreement across the five folds, not by magnitude:

* F **beats random** on c iff delta > 0 in ≥ 4 of 5 folds.
* F **beats raw** on c iff preservation > 0 in ≥ 4 of 5 folds.
* raw **beats F** on c iff preservation < 0 in ≥ 4 of 5 folds.
* otherwise the cell is **indistinguishable** for that pair.

Same-row naive per cell: regression probes carry `naive_zero_mae` and
`naive_train_mean_mae` on the same validation rows (`naive_n_val == n_val`);
barrier probes carry `prior_log_loss`. A representation **has skill** on c iff
its fold-mean loss is strictly below the fold-mean of *every* naive for that
cell (`skill_strict > 0`).

## 3. Gates for a trained family F (all required)

* **G1 learned** (Delta_probe vs random): F beats random on ≥ 7 of 14 cells.
* **G2 preserves/improves** (vs raw): F beats raw on ≥ 7 of 14 cells, and raw
  beats F on ≤ 2 cells.
* **G3 utility exists**: F has skill vs every same-row naive on ≥ 1 cell.
* **G4 incremental gain with refit**: applied to a trained family F only when
  the paired-refit table (`refit_input`, FS-CLOSE export `refit_gain_export.csv`
  or any CSV/JSON with `feature_id, family, refit_gain`, positive = better after
  refit) carries a finite row for `(feature, F)`; then `refit_gain > 0` is
  required. FS-CLOSE's export declares that trained-family refits are not
  materialised (latents are not available on the refit host), so it carries
  `family = identity` only: for every trained family G4 is `NOT_APPLICABLE` with
  `refit_gate_applied = false`, `refit_gain = NOT_AVAILABLE`, and the gate is
  not silently passed, its non-application is declared per row. The identity
  row carries the feature-level `refit_gain` (`refit_gate_applied = true` once
  its row has landed) and the candidate gets flag `RAW_REFIT_GAIN_POSITIVE` or
  `RAW_REFIT_GAIN_NONPOSITIVE`. That number says whether the feature helps a
  refit model; it does not compare representations, so it never changes the
  representation decision here. It is FS-CLOSE's survival input.

Winner among families that pass all gates: largest `(cells beating raw) −
(cells lost to raw)`; tie → lower total fit cost (sum of `fit_wall_seconds` over
the five folds); tie → fewer `params`; tie → fixed order ae, dae,
masked_temporal_ae, past_to_current_siamese.

If no family passes:

* `RAW` iff for **every** measured trained family F, raw beats F on ≥ 7 cells.
* `NO_TRAINED_ADVANTAGE` otherwise.

A family whose terminal is `FAILED` is excluded from the contest and listed in
`families_failed`; the decision proceeds once all three terminals are terminal.
If `families_failed` is non-empty the decision carries flag
`DECIDED_WITH_FAILED_FAMILIES`.

## 4. Flags (reported, never decisive)

* `NO_PROBE_SKILL_VS_NAIVE_ANY_FAMILY`: no representation, raw included, has
  skill on any cell. Passed to FS-CLOSE; the representation decision stands.
* `RAW_HAS_PROBE_SKILL` / `RAW_NO_PROBE_SKILL`.
* `RAW_REFIT_GAIN_POSITIVE` / `RAW_REFIT_GAIN_NONPOSITIVE`: identity refit gain
  from the FS-CLOSE export (feature-level; see G4).
* `UNSTABLE_LATENT(<family>)`: min linear CKA across reference-fold pairs < 0.5.
* `COLLAPSED_LATENT(<family>)`: `n_components_95 == 1` in every fold for a
  latent of dimension 8.
* `PROBE_ROWS_DIFFER`: `train_row_ids_sha256` differs across the candidate's
  terminals.
* `RECONSTRUCTION_WORSE_THAN_TRAIN_CONSTANT(<family>)`: fold-mean
  `mae_rel_train_constant >= 1` (diagnostic only).

## 5. Metric cells and their statuses

Every metric cell in `representation_dispositions.csv` is one of
`MEASURED`, `FAILED` (terminal failed; receipt digest of the FAILED file),
`NOT_APPLICABLE` (the pilot declares the metric undefined for the family, e.g.
reconstruction for identity/random/MTAE/P2C which export no decoder), or
`PENDING` (terminal not yet landed). Values are fold means over the five inner
folds unless the column name says otherwise; the long catalog carries mean, std,
min, max and `n_folds`.

## 6. Cost

Per family: `fit_wall_seconds_sum`, `updates_sum`, `params`,
`encode_latency_ms_mean`; per terminal: `wall_seconds`, `peak_rss_bytes`,
`cgroup_peak_bytes`. Cost breaks ties only (§3).

## 7. What this rule does not do

It does not read VALIDATION or TEST, does not read targets (only the probe
rows the pilot already wrote), does not select or drop a feature, does not
aggregate across seeds (one seed, 0, retained in every receipt) and does not
convert `PENDING` into any other state by extrapolation.
