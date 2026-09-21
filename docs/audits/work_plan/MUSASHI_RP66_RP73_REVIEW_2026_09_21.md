# RP66-RP73: external code and ML review

Audited revision: `1880caa`. Disposition: **CHANGES_REQUIRED; preserve measurements**.
Satoshi's branch has now been pushed by Musashi. Both workers fetched that revision and have
an isolated, clean `predictor-rp73-reviewed` worktree; existing worker checkouts were not switched.
No service restarted, no financial campaign launched, no reserved data used.

## Findings, ordered by risk

### A1 - High: an unanchored record can still certify rewritten predictions

`tools/df_closure_table.py:178` falls back to the local record's arrays digest, without proving
that the accepted terminal commits to that record. At line 205, missing warehouse prediction
artifacts are not a failure. The production-shaped synthetic counterexample rewrites predictions
AND their local record: MAE changes **0.080478157 -> 0**, with the same warehouse terminal and
receipt. Both tables report verified with no problems. This is not a production data alteration.

The published historical table actually contains this weaker binding class (`RECORD_DIGEST`,
warehouse `artifact_rows=0`). It must not acquire end-to-end artifact custody merely from a
matching terminal digest. Preserve historical scores, but qualify the scope until an independent
canonical payload or accepted artifact chain supplies the missing anchor. Missing evidence is not
proof that a historical score is false. New block/financial closures must also compare bytes read
to accepted artifacts, not just compare two lists of old artifact descriptors.

### A2 - High: preparation alone still becomes a verified comparator

`tools/df_benchmark_contract.py:145`: a design containing only the matching contract hash and
one `prepare: COMPLETED` receipt returns **VERIFIED_COMPARATOR**, with no design digest, model,
predictions, complete forecast population or metric verification. Replacing the old boolean with
two files did not establish the claimed property. This API must consume verified reference
forecast evidence, not a status on any unit. The measured GRU is not thereby disproven; its
actual evidence must be distinguished from this permissive API.

### A3 - High: financial nonfinite features reach model inputs

`tools/df_fin_runner.py:163` excludes nonfinite rows only from scaler fitting. The window
enumeration at line 174 checks only `origin >= W-1`. A single NaN in synthetic `volume` retains
**60 affected train windows**, and the real `PairBatches` emits NaN to the model. The contract
says nonfinite inputs are withdrawn; that policy is not implemented for the consumed support.
Fix train/validation/test masks, common-arm populations and train-residual delta derivation.
Do not impute zero implicitly or let a later numerical failure masquerade as a loss comparison.

### A4 - High: financial uncertainty and replication do not implement the intended experiment

`tools/df_fin_runner.py:423` chooses the minimum-validation **cell**, including its seed, rather
than preserving paired seed replicates or selecting configurations across a declared seed
aggregate. A two-seed counterexample silently selects seed 1. This is a best-of-seeds policy,
not the intended paired loss/optimizer comparison; explicitly changing the estimand would be
another design, not a fix by renaming the output.

At line 399, two folds with block length two produce a nominal 95% interval **[0.1, 0.1]**
from fold effects **[-0.1, +0.3]**. There is one admissible block, not demonstrated certainty.
Return insufficient resampling support; do not fall back to iid certainty or claim that using
the word "block" establishes coverage. Nonconsecutive surviving folds must not become adjacent
weeks after silently dropping missing folds. No real financial experiment has used this runner.

### A5 - Medium: the update ceiling can be labelled uncensored

`tools/df_e1_block.py:604`: if patience expires on the last allowed update, early stopping wins
the branch. The real loop on a constant model at update 4/4 reports `STOPPED_ON_VALIDATION`,
contradicting its rule that reaching the ceiling is censoring. Record both triggers; budget
reached remains true irrespective of the winning branch. This does not establish that any of
the reported 18 cells has this boundary case; scan their saved events rather than retraining.

A separate suspicion about accumulated batch-loss logging did **not** reproduce in the installed
environment: batch losses 4, 0, 0, 0 were reported correctly. It is not a finding.

### A6 - Medium: cost projection counts validation twice

`tools/df_e1_block.py:675` divides total fit CPU (including validation) by updates; line 798
multiplies that inclusive number by the final update budget and adds validation again. Independent
arithmetic over retained pilots changes Q2's 127,579.07 s to **111,000.13 s**, retaining final
restore/loop overhead conservatively once. This is not a new measurement. Q2 remains far above
14,400 s, so the decision not to launch was correct. The 4.9 s/update figure is not evidence that
window gathering is the bottleneck. Profile gathering, forward/backward, validation and setup
separately before prescribing an optimization or reducing the scientific design.

## Measurements retained and independently checked

Musashi read all **18 new forecast arrays**, checked origins, labels, persistence, normalized
metrics, and hashes/sizes of arrays, records and weights against the terminal artifacts published
at the audited revision: **18/18, zero discrepancies**. This was not a new live warehouse query
and not a fresh-process replay of model weights. It does not extend the check to all 31 older rows.

Same 10,020 DEV origins; h60 minutes; common train sigma 0.9125164391265214. All values below
are MAE_z, not percentage error. Naive persistence is **0.6765599275** for every arm.

| Arm | Mean MAE_z | Sample SD (ddof=1) | Skill vs persistence |
|---|---:|---:|---:|
| Adapted GRU reference | 0.531273 | 0.001299 | +0.214744 |
| Modular, original inputs | 0.568045 | 0.003027 | +0.160392 |
| Modular, real calendar | 0.486819 | 0.009936 | +0.280449 |
| Modular, randomized calendar control | 0.552871 | 0.016420 | +0.182820 |
| Modular, 56 days | 0.560799 | 0.006639 | +0.171102 |
| Modular, 112 days | 0.557459 | 0.004023 | +0.176039 |

The return's smaller SDs use ddof=0; label this convention instead of mixing the two silently.
SD across three seeds is optimization variation, not sampling uncertainty over households/weeks.

Calendar versus its equal-capacity randomized control reduces mean error by **0.0660524 MAE_z**
(11.95% relative), with the same sign in all three pairs. This is the clearest new input-information
result. It does not establish that calendar-modular beats a calendar-GRU: that arm was not run.
GRU beats original-input modular by 0.0367724 MAE_z under the common training recipe. Increasing
history from 28 to 112 days yields a smaller 0.0105868 reduction, not proof that volume is exhausted.

The comparable literature-derived reference is the **adapted, locally rerun GRU**, not a published
number under another target/split. Published Gasparin Table 5 remains NOT_COMPARABLE. Do not
call this an exact paper reproduction or a general architecture ranking.

## ML decision and responsibility

The prescribed 200-update validation cadence with patience three translated to only **600 updates
without improvement**, versus roughly three passes over about 40,080 windows in the former epoch
recipe (about 1,881 updates at batch 64). My prior order required comparable update-based stopping
but did not adequately preserve this effective patience. That omission is ours to correct, not an
excuse to erase the results or to attribute the level change solely to architecture.

Use a prospectively documented DEV sensitivity tier with the SAME checkpoint cadence, 4,000-update
ceiling and patience ten events (2,000 updates without improvement, approximating the former three
passes). This choice is informed by already-seen DEV behavior; it is **not** an untouched confirmatory
design. It neither guarantees improvement nor proves convergence. Keep a common recipe across
arms and separate input changes from architecture. Never use final holdout to choose patience.

No financial choice of MAE/Huber or Adam/AdamW follows from the electricity result. The financial
question remains mandatory, with paired seeds, weekly folds, explicit time/availability contracts,
causal residual scaling, and sensitivity at 1e-5/1e-6 measured through the real path, not inferred
from `np.spacing(float64)` alone. Forecasting error still does not itself establish trading profit.

## Reproduction and scope

Evidence: `docs/audits/evidence/RP73_MUSASHI_REVIEW_2026_09_21/`.
Run `reproduce.py --repo <checkout> --output <scratch-result.json>` with the trading-stack interpreter.
`results.json` preserves synthetic counterexamples and the retained-pilot arithmetic.
`recheck_arrays.py` independently checks retained new results; `arrays.json` preserves the outcome.
All destructive mutations occur in temporary synthetic fixtures. No original evidence was changed.

Focused verification in `trading-stack`, CPU-only:

```bash
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=1 python -m pytest \
  tests/test_df_benchmark_contract.py tests/test_df_closure_table.py \
  tests/test_df_e1_block.py tests/test_fin_loss_opt_acceptance.py -q -k 'not FL08'
```

**85 passed, 2 deselected**, 122.26 s wall. The two FL08 disposable-stack tests were deliberately
not rerun; no live/disposable service was started for this review. There were 165 Keras NumPy
deprecation warnings. The adversarial probes above reproduce defects despite these existing tests
passing. The entire repository suite was not rerun or claimed green. Documentary plan check PASS
(`scientific_approval: false`); `git diff --check` clean.

Next executable order: `docs/handoffs/MUSASHI_RP74_RP81_2026_09_21.md`.
