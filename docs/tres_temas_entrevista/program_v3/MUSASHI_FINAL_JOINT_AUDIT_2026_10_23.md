# Final joint selection audit and correction order

## Verdict

**REVISE_BEFORE_SHARED_CONSUMPTION.** The measured T3 execution, artifact
custody and original-context replays are accepted within their recorded scope.
The shared representation is not yet accepted as a fold-clean, executable
contract for either downstream lane. These are reproducible defects, not a
request for new approval paperwork. No new policy fit or TEST scoring occurred.

Audited pins: heuristic-strategy `f57f8ef9ccf283409521a23ab1b43cb0b0298e45`,
upstream predictor `048549f047b4efc95ba3c4e7dd06837a907cbbab`, request
`73fe2c0286cc086ca0749ad011dff24c668cf74a`. Findings apply to these pins.
Musashi authored the earlier execution/replay tooling; those receipts are not
an independent certification of his own work. This review adds separate
source, mutation and real-data fit-scope checks.

## Findings

### A1 HIGH: learned upstream producers include the internal selection block

Source: predictor `tools/layer2_pool_h1.py:256`, `:278`, `:374`, `:493`, `:528`.
The authenticated layer-2 manifest fits Kalman, GMM, HMM and fractional
differencing on 2020-01-01 through 2024-01-01 exclusive. The C consumer's FIT
ends 2023-07-01; INNER_A is the remaining half of 2023. Six selected columns
depend on these fitted producers: `kf.lc.level_minus_close`, four `rg.gmm.*`
columns and `fd.lc.d0.3@diff`. HMM is also in the selection pool, although not
in the selected list.

**Real-data witness:** the retained GMM parameters and an unchanged native
refit both reproduce the stored GMM columns with maximum difference zero.
Changing only the 3,086 INNER rows used by that upstream fit, then repeating
the native fit with the same configuration, changes 87,048 earlier cells
(21,785 rows; maximum absolute delta 0.7357681693). Applying already-frozen
parameters instead leaves that prefix exactly unchanged. The existing
`perturbation_check` at line 433 tests only this latter, frozen-apply property;
it cannot establish fit isolation.

This demonstrates internal fit/selection contamination, **not** a read of
2024 or TEST by this fit. Fitting on all of 2020-2023 can be legitimate for a
subsequent 2024 prediction; it does not establish independent internal-fold
validation. The selected architecture's internal gain must be rechecked with
the declared fit isolation. No claim that every feature or measured metric
is numerically wrong follows from this witness.

**Correction:** partition-aware fit/apply for every learned producer and
data-dependent encoding decision, including the earlier selection folds.
Freeze parameters on that fold's allowed training rows, then apply to its
evaluation rows. Retain causal history without fitting on held-out rows.
Audit supervised derived features such as `act.p_hit` for out-of-fold training
predictions and label availability too; their producer/cross-fit provenance is
not certified by this GMM test. Do not silently remove useful feature families.
Produce an affected-descendant map and rerun dependent decisions only. Shared
contexts can make a contrast affected even if its named family is unchanged.

### A2 HIGH: the exporter accepts artifacts inconsistent with sealed C

Source: heuristic-strategy `tools/expert_guided_consumer_recipe.py:42-94`.
The delivered authentic recipe matches C, but the exporter does not enforce
that derivation. Seven independent mutations were accepted: changed scaler
mean, rolling scaler policy, changed extractor kind, changed model depth,
wrong store digest, different manifest identity, and a combination report
explicitly marked replay-failed. The last two retain the same recipe identity.
Two negative controls (changed result bytes and wrong column) are correctly
rejected. Hashing caller-supplied objects is not comparison with the already
authenticated producer chain.

**Correction:** use the existing receipt chain and externally pinned inputs;
verify scaler values and policy, store/list/manifest hashes, resolved group
indices/kinds/units, model configuration and replay success against actual C.
Reject malformed lengths, nonfinite constants and invalid standard deviations.
Cover operational semantics in an identity, either the recipe's identity or a
separately pinned execution contract. No signature system or new warehouse is
required. Add the seven retained mutations as rejection regressions while
preserving the valid baseline.

### A3 HIGH: the next consumer queue does not consume the selected recipe

Source: `tools/expert_guided_clean_round.py:156-173` and
`app/short_horizon/expert_guided/data.py:433-441,603-618`.
Executing the real command builder for `strict_fam_w13` selects the old groups
file (`33ac83c1...`), not C (`a9d878cf...`). It supplies no recipe or saved-scaler
argument. The store loader redecides stationarity and refits scaling; the
existing saved-scaler loader rejects the store loader type. A correct JSON
export alone therefore does not make both lanes representation-identical.
No new round has run, so this is caught before an incorrect consumption run.

**Correction:** both consumers must apply the same pinned tensor/preprocessing
contract and resolved groups. Establish a C-representation baseline before
the separately identified architecture contrasts. Verify exact tensor values,
order, valid-row masks, window endpoints, scaler and group identities on a
common probe; reject a permuted column, altered scaler or wrong groups before
training. SAC must receive neither expert labels nor expert-trained weights.

### A4 MEDIUM: transformation stages and temporal/business semantics need precision

Source: recipe exporter lines 28-35,72-76,93. Twenty-six selected store columns
already contain `@diff`, `@log` or `@asinh` transformations. C's additional
consumer differencing list is empty. The recipe infers transformations from
suffixes without saying they are already applied upstream. This is a
double-transformation risk for a new consumer, not a demonstrated double
transformation in C.

The phrase "observed rows (session gaps never compressed)" is ambiguous.
The actual 13-slot, stride-2 window counts observed rows; closures occupy no
slots. Its 24 observed-row intervals span 24 to 105 elapsed hours in this
store. Preserve this established behavior and the current tick; do not add
weekend interpolation or silently switch to a 24-clock-hour window.

**Correction:** distinguish upstream-applied and consumer-applied operations.
Add typed window/time semantics and bind the existing admitted calendar,
12-hour opening/forced-close offsets, costs, execution timing, leverage,
SL/TP and sizing rules. Reuse the effective execution contract; do not invent
new costs. Business prose alone is not a loadable, identity-covered contract.

## Accepted evidence and limits

- All 43 result files match the retained join, and the authentic C result,
  scaler, groups, admitted bars/calendar and upstream store hashes match.
- The native per-group decision reproduces the exact retained digest.
- The 3,882 ledger units are unique and error-free. Independent arm-aware
  checks find zero support/seed/fold discrepancies in 714 contrast groups,
  72 encoding groups and 120 rescue groups.
- The authentic recipe is regenerated byte-identically. Cross-group redundant
  columns remain disclosed; T3 is not a unique-information ablation.
- Earlier 516+12 checkpoint probes and control/C trading replays remain exact
  in their original CPU/thread contexts. They were not repeated in this CPU
  source audit and do not prove fit isolation.
- The positive 2024 seed remains activity-invalid; the reported negative mean
  is not concealed. This review does not retune on 2024. TEST remains closed.
- Full external publication-clock provenance, every fitted upstream producer,
  fresh warehouse read-back and broker cost accuracy are not certified here.

## Executable handoff to Satoshi

1. Implement A1 with fit-scope tests first. Perturb held-out inputs and labels,
   rebuild including fitting, and require unchanged training parameters and
   earlier training tensors. Keep the frozen-apply test as a separate check.
2. In parallel, fix A2 and A4; keep original artifacts immutable and write a
   new recipe identity. Return rejection tests plus an honest derivation test.
3. Return the fold-local producer identities and an actual changed-column /
   support / dependent-run map. Reuse only proven unaffected artifacts. Do not
   relabel the old internal gains as clean or repeat unrelated experiments.
4. Recompute the affected selection/T2/T3/C decisions with the same business,
   budgets and frozen decision rules. Cache fold-local features once and
   automate the dependent queue; do not keep an agent polling it.
5. Musashi integrates/tests the no-expert SAC consumer; Satoshi integrates the
   expert-guided consumer. Both must pass the same source-to-tensor probe
   before either new policy campaign. Expert targets stay training-only.
6. Return one correction report with pins and evidence. Musashi rechecks the
   affected findings; no new master-plan debate or user-run commands needed.

The completed execution is preserved as reproducible development evidence.
No RELEASE or new downstream policy campaign was issued during this audit.

## Reproduction and traceability

Script: `tools/audit_shared_consumer_recipe.py` in predictor.
Evidence: [PROBES.json](../../audits/evidence/SHARED_SELECTION_FINAL_20261023/PROBES.json).
Use the pinned checkouts above and the retained private data roots; placeholders
below are paths, not a request for the owner to run commands.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tools/audit_shared_consumer_recipe.py \
  --hs <pinned-heuristic-strategy> --predictor-source <pinned-feature-producer> \
  --pool-root <retained-pools> --admitted <retained-admitted-bars> --out <audit-output>
```

| Requirement | Executed evidence | Disposition |
| --- | --- | --- |
| exact artifact/decision reproduction | checks, honest emitter baseline | PASS |
| matched comparison supports | paired_support, 3882 unique units | PASS |
| fold-isolated learned inputs | learned_transform_witness | FAIL A1 |
| recipe rejects inconsistent inputs | emitter_mutations, two negative controls | FAIL A2 |
| common consumer of selected C | next_queue_consumer | NOT IMPLEMENTED A3 |
| explicit transformation/time/business contract | transform_boundary, observed_row_window, source | REVISE A4 |

Two diagnostic native GMM fits were used to reproduce the fit-scope witness;
these are not policy training. The existing exporter test passed in isolation
(one test, 13 deselected), showing why its present coverage misses A2.
