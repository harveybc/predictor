# I6N causal traceability audit: RETURN

Order: `docs/handoffs/RETSU_I6N_CAUSAL_TRACE_AUDIT_2026_10_12.md` (sha256 `8956fced…`), branch
`codex/retsu-i6n-causal-audit-20261012` @ `f5fb48aa`. Audited object: predictor commit
`cbbe1f168c0a6cc189abbd1e8b297e6bd093f951` (the audit branch adds only the handoff on top of it),
H4 scientific config `fb70328d…` (recomputed: sha256 of the canonical `optimization_config`),
sealed plan `1b83490b…` (recomputed from the embedded `i6n_plan` with `plan_sha256`/`seal`
excluded; located inside every H4 island config and `config/MANIFEST.json`), provisional champion
genotype `356e4acc…` (located in the content-addressed archive store, body `1bc4ad6f…`).

Runner: `tools/i6n_causal_trace_audit.py run|status`, spec `SPEC.json` (tolerances and
availability bounds declared before the run), tests `tests/test_i6n_causal_trace_audit.py`
(16 passed). Executed once as the audit under
`CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 30m -n i6n-causal-audit -- …`: terminal state
`COMPLETE`, 485 s, 91 assertions: **67 PROVEN_SAFE, 9 PROVEN_LEAK, 15 RISK_UNRESOLVED, 0
NOT_APPLICABLE**. Disclosure: two earlier invocations of the same command terminated
`COMPLETE_WITH_PHASE_ERRORS` on defects of the runner itself (a spec key path; `nan != nan` in an
unperturbed window) and produced no verdict; their outputs were overwritten. No model was fitted,
no service touched, no TEST row read (external series are filtered on their date column before
any value is used; campaign inputs end 2024-12-31), nothing under a campaign root was written.
Host, island and user identifiers are redacted (`ISLAND_n`, `<campaign-root>`, `<data-root>`).
The independent reviewer rerun with mutated inputs required by the order was not performed by
this agent; the fail-closed behaviour is demonstrated only on synthetic data by the tests.

## Findings, by severity

### 1. PROVEN_LEAK: feature selection is not chronological (C1, C2.*; 4 assertions)

The 20 campaign features (`CLOCK.json`, parent `DESIGN.members`, I6-B manifest, all in the same
order) are exactly the union of the 14 EURUSD FS4 RAW winners in `TEST_FREEZE.json` (seal
recomputed; `freeze_sha256 e1449779…`). `WEEKLY_PLAN.json` scores that selection on VALIDATION
weeks 2024-01-01 .. 2024-12-30 (`validation_year 2024`, arm RAW). Every fold cutoff
(2023-03-31T21Z, 06-30T21Z, 09-29T21Z, 12-29T22Z) precedes 2024-12-30, so every feature used by
the 2023 folds was admitted with outcomes that lie after the fold. Classification:
`PROVEN_FUTURE_SELECTION_LEAK` for any claim that a 2023 fold score is an unbiased estimate. The
fit itself is not invalidated. `FEATURE_PROVENANCE.csv` has one row per feature and fold (80 rows)
with the winners that admitted it, the selection interval and the chronology state. This is the
**first causal break in chronological order**: it sits upstream of every fold, scaler, window,
fit and score in `TRACE_GRAPH.json` (edge `fs4:test_freeze -> features:20`).

### 2. PROVEN_LEAK: the in-sample TRAIN split enters the ranking (E7, D7.*; 5 assertions)

The plan rule `per_fold = 0.5*(sum_abs_error_TRAIN/naive_TRAIN + sum_abs_error_INNER/naive_INNER)`
was replayed exactly on all 47 archived candidates (E6: 47/47 exact). Half of every fold fitness
is the in-sample TRAIN ratio, and for that half the feature scaler and target scale consume TRAIN
rows later than the scored origin (D7: `max_fit_time` 2023-03-03T05Z vs first scored TRAIN origin
2019-04-01T09Z, and likewise for the other folds). This is not a crossing of the fold boundary
(INNER is clean, D6/D8), but the ranking is not an out-of-sample quantity, contrary to the
order's "no TRAIN metric enters ranking". For the champion: recorded fitness
`-0.010586766321054819` = replay (exact; E8); INNER-only aggregate `-0.0085199`; TRAIN-only
`-0.0126536`; the Q1 INNER half alone is **worse than naive** (`+0.0019817`).

### 3. RISK_UNRESOLVED (15)

- A7: the six annual minute archives named in `PREPARATION.json` (retained source bytes) are on
  no local root; price reconciliation remains a narrative claim. The campaign `INPUT.npz`
  bytes, their digests, and their byte-identity to the parent input `897caee1…` after the
  documented clock correction are proven (B1–B3).
- C3 (7 features): `fred.rates.dgs3mo.diff_5d` and `fred.rates.dgs7.logret_5d`: as-of lag 48 h
  identified but the closed formula reproduces only 86 % / 81 % of hours (< 0.95), so the
  support interval is not proven; `px.close_loc`, `px.log_co`, `px.rv5`, `rg.atr_ratio`,
  `fx.gbpusd.logret_1h`: the PS1 producer code is in no sibling repository and the independent
  vendor bars match the retained close only at a -5 h clock offset with 61 % agreement
  (`px.log_co` matches that same bar at 61 %: consistent with "bar ending at t", not proven).
- D5 (4 folds): the nominal NY calendar deviates from the retained bar clock in 18 of the 2023
  weeks (session end 1 h earlier in the data, Monday first bar 2 h late, Good Friday). This is
  the known DST/clock-policy mismatch. It changes which origins are eligible, not label
  causality: every scored (t, t+4h) pair uses observed bars 14400 s apart (D4).
- E3b: endpoint target centers recorded on the other islands differ from the local recomputation
  at the 1e-21 level (137/188; max |Δ| 1.06e-20). Cross-host float64 reduction; no tolerance was
  declared, so it stays unresolved. Scalers are digest-identical on all islands (E3: 0/188).
- E9: the champion's four fit directories (scores, paired-metric inputs, model) exist only on
  its island; row-level parity and forward mutation checks could not be run on the champion.
  Its receipts bind the same frozen support, scaler digests, stopping rule and plan (E2–E5).
- F0: the 27 retained local fits were produced by the evaluator at commit `9fbcf1ec` (digest
  `1fdd9e7b…`); `cbbe1f16` differs only in the interrupted-attempt recovery code.

### 4. PROVEN_SAFE (67), the checks that held under counterexample

Identity (A0–A6: plan seal, config digest on all four islands and the live config, fold
configs/frozen supports/inputs by digest); data (B1–B3); fold support re-derived by the campaign
code (D1) and by an independent eligibility implementation with exact origin-digest equality
(D3: 18695/208, 18183/209, 17860/368, 17878/355); cutoffs recomputed from the sealed calendar
(D2); disjoint TRAIN/INNER, purge, forced-flat, no weekend counted as market hours, elapsed-
instant targets, forced-flat counterexamples rejected (D4); scaler and target scale byte-identical
under INNER and later-fold sentinel perturbation, changed by the TRAIN control (D6); INNER windows
byte-identical under later-fold perturbation (D8); archive integrity and identity (E1, E2),
seeded identical initialization per candidate with optimizer iterations 0 and four distinct
checkpoints (E4), stopping on mean(TRAIN, INNER) only, no outer metric anywhere (E5); exact metric
replay (E6, E8, F2); model/naive/frozen-population row identity on all 27 local fits, original
price scale (F1); 28 alignment cases on 4 retained local models (endpoint and trajectory, Q1
with both DST transitions, Q4): rightmost input exactly t, grid right-anchored, target exactly
t+14400 s, trajectory heads +1,+2,+3,+4 h, source rows after t leave tensor and CPU output
byte-identical, row t changes only the rightmost slot and the output, target price changes only
the label (G1). 11 external features have a proven as-of availability: FRED daily levels and
DPRIME at source date + 48/49 h, Yahoo closes at + 24 h, STLFSI4 at + 23 h. Caveat on STLFSI4:
the declared bound (0 h) is the loose one; FRED publishes that weekly index the Thursday after
its Friday date, so +23 h is earlier than real publication. This is a declared-bound limitation
recorded here, not a change of the computed state.

## Minimum correction (order, section "First critical question")

Either (a) select features strictly before each 2023 fold cutoff and nest the selector inside
the fold, or (b) freeze one feature manifest from a period strictly earlier than all search folds
and rerun the search under a new identity; in both, TEST 2025+ stays closed. Additionally, for the
ranking to be an out-of-sample quantity, drop the TRAIN half of `per_fold` (or declare it a
regularizer, never an estimate), and declare publication-lag bounds per external source.

## The five verdicts

1. Tensor/target pipeline: **causally valid** on the audited code (PROVEN_SAFE, G1/D3/D4/D8);
   for the champion itself only by code and receipt identity (its tensors are not local: E9).
2. Preprocessing and transformations: the campaign-internal transformations (scaler, target
   scale, windows, difference transform) are **causally valid** (PROVEN_SAFE); the upstream 20
   features are **not fully proven**: 11 PROVEN_SAFE as-of, 7 RISK_UNRESOLVED (producer code
   absent), and one declared bound (STLFSI4) is looser than real publication.
3. Feature selection chronology: **PROVEN_LEAK** (future selection on VALIDATION 2024).
4. Model and naive rows and scale: **exactly paired** on all 27 local fits (PROVEN_SAFE);
   champion exact at the sums level, rows RISK_UNRESOLVED (not retained locally).
5. Disposition of the provisional H4 result: **RETROSPECTIVE_DEVELOPMENT_ONLY**. Not promotable,
   not usable for VALIDATION or strategy claims; not rejected (metric parity exact, fit valid).
   The live campaign may continue as engineering exploration.

## Files

`STATUS.json`, `FINDINGS.csv`, `TRACE_GRAPH.json` (64 nodes, 83 edges), `FEATURE_PROVENANCE.csv`,
`FEATURE_AVAILABILITY.json`, `ALIGNMENT_CASES.jsonl` (28), `MASK_PARITY.json`,
`METRIC_REPLAY.json` (47-candidate census, champion replay, chain optimae), `INPUTS_READ.json`,
`SPEC.json`, `MANIFEST.sha256` (outputs, runner, tests and every executable input by digest).
