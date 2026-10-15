# I6-DMO matched short control: MODEL lane (SC05, SC06, SC07, SC11 CPU)

Status: IMPLEMENTED_CPU_TESTED_NO_FIT. No scientific fit was run, no GPU was
used and no TEST row was read. The successor stays PREPARING until review.
Authority: `NEXT_CONTRASTS_PLAN_20261015.md` section 2 and SC05 to SC12.

## Recipe

Both arms use the existing small R0 recipe. Neither arm has a head, core or
branch change.

| Component | mg1 (25 slots, 1 h) | mg2m (13 slots, 2 h) |
|---|---|---|
| 19 causal Conv1D branches (4 filters, k3, s1), 16 parameters each | 304 | 304 |
| Weightless concat fusion | 0 | 0 |
| `DynamicPhysicalTimeEncoder` 1.0.0 (actual signed UTC offsets, plus 2 context channels) | 0 | 0 |
| Core Conv1D 8 then 4 (k3, s1) on 78 channels | 1880 + 100 = 1980 | 1880 + 100 = 1980 |
| Six heads: Flatten, Dense(8, relu), Dense(1) | 6 x 817 = 4902 | 6 x 433 = 2598 |
| Total trainable | 7186 | 4882 |

Per head, mg1 has 808 hidden + 9 output parameters (Flatten 100). mg2m has
424 + 9 (Flatten 52).

The head difference is exactly (25-13)·4·8·6 = 2304 parameters. Every other
component differs by 0. These numbers are measured on the real sealed builds
(`EXPERIMENT.mg1.json` design 64899668 / effective 894cda3b, and
`EXPERIMENT.mg2m.json` design d71d2b3c / effective ad949872, resealed on the
TF 2.21.0 / Keras 3.13.2 execution runtime; the superseded 9f0d086e / a3e073c7
seals measured the identical counts). They are recorded
in `CAPACITY_MATCHED.json`, and every matched receipt carries them
(`matched.capacity`, also written to `CAPACITY.json`).

This compares two practical input/readout recipes. It does not prove that
additional signal alone caused any difference.

## Guards

- **Capacity (SC07).** `model.capacity_record(bundle)` checks three things:
  - the build matches the analytic recipe count;
  - the build matches the literal `DECLARED_CAPACITY`;
  - the accounting identity holds.

  If any check fails, it raises `CAPACITY_UNREPORTED_CHANGE`. The trainer calls
  it before fitting. `matched_capacity_delta` refuses any mg1−mg2m difference
  other than heads = total = 2304.
- **Identity (SC06/SC10).** `design.matched_arm` from the data lane admits only
  `short_mg1_matched` and `short_mg2_matched`. The closed MG2/MG12 identities
  are refused as `MATCHED_IDENTITY_REQUIRED`, whether or not the run is
  scientific. A fit receipt can become a matched arm only if
  `identity.common_population_sha256` equals the pinned population and
  `identity.matched_arm` equals the arm. Anything else is `NOT_A_MATCHED_FIT`,
  including a cached full-population MG2 fit.
- **Time inputs.** The spacing bound comes from the sealed design: 1 h for mg1
  and 2 h for mg2m. The trainer checks it, and so does the encoder. Offsets
  must equal (slot instant − t)/3600 (`OFFSETS_NOT_ACTUAL_UTC`). These ladders
  are refused as `ORDINAL_OR_INVALID_OFFSETS`:
  - ordinal or positive ladders;
  - compressed ladders;
  - an mg1 ladder under mg2m.

## Fit path (SC06)

```
python -m tools.i6dmo.train --design matched_short/EXPERIMENT.<arm>.json --out <new unit dir> \
  --device {cpu,gpu} --common-population <population dir> \
  --common-population-digest <POPULATION_SEAL.population_digest> [--input-dir <retained input>] --json
```

For a matched fit, `--bundle` is omitted or names the population directory.
Arm tensors come only from `common_population.arm_tensors(pop, sealed_design,
split)`, and the model side re-verifies them. The fit then works as follows:

- **Data.** Rows and masks must equal the population's. Labels outside the
  common mask are NaN, so they never reach a loss, a monitor or a score.
- **Scaling.** Input z-scores use `pop.scaler`. The loss coordinate is
  delta/`pop.naive_loss_scale[h]`. The arm's exact FIT prices must reproduce
  that naive (`NAIVE_SCALE_MISMATCH`). Nothing is refit per arm: SCALES.json is
  byte-identical across the arms.
- **Monitor.** The monitor is 0.5·(FIT+INNER) on the common masks at the same
  weights. DEVELOPMENT is predicted only after the best weights are restored.
- **Training settings.** AdamW lr 1e-3, weight decay 1e-4, batch 256, at most
  200 epochs, patience 10, min_delta 1e-8, best weights restored, seed
  6102026. Determinism flags are set before TensorFlow is imported.
- **Refusals.**
  - `I6DMO_TEST_MAX_ROWS` is refused (`MATCHED_SUBSET_FORBIDDEN`).
  - An unsupported naive head (`None`) is refused (`HEAD_UNSUPPORTED_IN_FIT`).

## Report path (SC05)

`reporting.run_unit(spec, ...)` switches to the matched path when
`spec.common_population = {path, sha256}` is present. It also needs
`spec.matched = {arm, fit_unit, input: {path, sha256}}`. The path works as
follows:

1. **Load and verify.** It loads the population with source verification and
   verifies the fit `RECEIPT.json` through the terminal digest chain and the
   matched identity. It also checks the capacity disclosure.
2. **Rows, mask and prices.** The DEVELOPMENT predictions must carry exactly
   the common rows and head mask. The saved p(t) and p(t+h) must equal the
   accessor's values, and those are replayed by timestamp against INPUT.npz
   (`replay_against_input`).
3. **Scoring.** It scores with the audited C2, the manual pooled replay from
   the saved predictions and the disk naive replay. All three must agree.
4. **Results.** Each head has its own MAE, naive and (MAE_h−naive_h)/naive_h,
   where negative is better. The selection score is the equal-head mean of
   annual pooled relative MAE on the common DEVELOPMENT rows. A failed head
   makes it `INCOMPLETE`, so the aggregate never hides a failed head. Weekly
   diagnostics cover the observed common weeks; the population declares no
   empty weeks.

`reporting.decide_matched(mg1_metrics, mg2m_metrics)` applies the decision
rule. The lower score wins. If the scores are within 1e-8, the cheaper mg2m is
kept. If any head or arm failed, the result is `INCOMPLETE_COMPARISON` and
lists the missing evidence. The rule also refuses arms that disagree on the
population, rows, mask or counts.

## Evidence

- `tests/i6dmo/test_matched_fit_path.py` runs against the REAL
  `tools/i6dmo/common_population.py`. It uses a synthetic population built by
  `build_common_population` from sparse synthetic weeks, and designs sealed
  with the fixture calendar.
- Both arms go through the real trainer CLI. The report uses the real hook.
- One bounded CPU smoke ran on the real sealed population 4ab41bd8, under the
  superseded seals, before the reseal (the resealed population ba01f87a keeps
  identical splits, scaler and counts; the smoke was not repeated). It made no
  fit. Each arm's FIT/INNER/DEVELOPMENT tensors passed every model-side check,
  including the shared-naive cross-check on 16397 FIT rows. A 64-row forward
  pass ran for each arm. SCALES were identical across arms, the delta was
  2304, and peak memory was 814 MiB.
