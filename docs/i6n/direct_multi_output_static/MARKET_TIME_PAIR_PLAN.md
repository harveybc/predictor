# I6-DMO market-time pair: implementation and execution decision

Decision: Musashi, 2026-10-15, based on support-study return `98ffdf28`.
State: CLOSED / PAIR_NOT_ELIGIBLE, measured 2026-10-15 08:30 UTC. See
[closure](RETURN_REPORT_RECOVERY_20261015.md). Retain the following design as
the completed experiment's contract, not a current launch order. The next
accepted sequence is [NEXT_CONTRASTS_PLAN_20261015.md](NEXT_CONTRASTS_PLAN_20261015.md).
This was a fixed I6-DMO experiment, not a DOIN search or a claim of optimality.

Runtime qualification (2026-10-15): the closed MG2/MG12 results are RUNTIME_SEAL_NONCONFORMANT_OBSERVED_TF_2_21 (declared TF 2.18.0/Keras 3.9.0, receipts record TF 2.21.0); they cannot attest the declared runtime or serve as a runtime-matched donor/control. Numbers unchanged. Record: [RUNTIME_QUALIFICATION_MG2_MG12.json](RUNTIME_QUALIFICATION_MG2_MG12.json).

## 1. Question and limits

Can one small static R0 short/long pair, using real market-time input slots,
beat the naive for each usable physical horizon and qualify for the existing
business strategy? Support, predictive accuracy and trading utility are three
different results. Additional usable origins do not prove improved accuracy.

The support study found FIT/INNER/VALIDATION origins of 16496/715/4772 for MG2,
13174/546/3804 for MG12, and 13123/546/3760 shared origins. These are reference
counts for unchanged input/calendar identities, not quotas to force by filling
gaps. Any difference must be reconciled and retained before fitting.

## 2. Frozen experimental scope

| Field | Short | Long |
|---|---|---|
| Rule | MG2_MARKET_EXACT | MG12_MARKET_EXACT |
| Real input observations | 13 | 13 |
| Spacing in expected open-market hours | 2 | 12 |
| History in open-market hours | 24 | 144 |
| Output horizons in elapsed hours | 2,3,4,5,6,12 | 18,24,36,48,60,72 |

Issue at each supported hourly origin, with the last completed observed tick
t included. Derive slots from the versioned expected market calendar, not from
row counts. Require the exact real observation at each slot. Retain unplanned
open-session gaps as exclusions; no smoothing, interpolation, ASOF substitution,
synthetic weekend prices or changed target timestamps. Elapsed history varies
across closures and DST; never label it fixed 24/144 elapsed hours.

Keep the same 19 RAW features, STLFSI4 exclusion, seed 6102026, one repetition,
four-year TRAIN 2020-2023, last eight TRAIN weeks as inner, purged label maturity,
and full 2024 development validation. Target and naive remain unmodified.
Fit scalers and loss scales only on FIT. No within-year retraining. All
availability/provenance assumptions stay visible; this decision does not
resolve them or reclassify 2024 as independent out-of-sample evidence.

## 3. Small model with explicit physical time

Keep one causal Conv1D per feature (4 filters, kernel 3, stride 1), concatenate
on filters, core causal Conv1D 8 then 4 (kernels 3, strides 1), six independent
Flatten -> Dense(8,relu) -> Dense(1,linear) heads. No Transformer, LSTM,
uncertainty sampling, pretraining or capacity search in this identity.

The only necessary model extension is a serializable dynamic time encoder
immediately after branch fusion. Its per-window inputs are the actual signed
UTC offsets from t in hours and the adjacent-slot closure flags. Replace the
fixed-offset positional code with the same deterministic sinusoidal rule
evaluated on ACTUAL offsets, not ordinal slot indices. Then append two context
channels before the core: adjacent elapsed-hours/24 and closure-crossed (0/1).
The first slot has adjacent delta 0 and closure flag 0. The last offset is
exactly 0. With 19 branches the core sees 76 encoded value channels plus these
two context channels. The factor 24 is a fixed unit conversion, never fitted
using validation. Seal formula, dtype, shapes and layer version in the design.

No fixed per-batch time grid may replace these per-example inputs. Model
metadata must distinguish market-slot order from actual wall-clock time. Use
the existing plugin/facade boundaries; do not change historical plugin defaults.
No hourly or lagged encoder donor is compatible by shape alone.

Keep AdamW lr 0.001, weight decay 0.0001, batch 256, max epochs 200, patience
10, min_delta 1e-8, equal-head masked normalized MAE and restored best weights.
Monitor the mean of end-of-epoch FIT and INNER at identical weights. Retain
the already-reviewed CPU/GPU numerical policy without broadening tolerances.

## 4. Eligibility, comparisons and automatic progression

Apply audited C1/C2 per-output business masks: elapsed t+h strictly before
force-flat and inside its split, Thursday entry cutoff, entry/management
separation, one position and no weekend carry. Known past input history can
cross a closed market; prediction targets cannot bypass the business mask.

Each horizon reports original-price MAE, naive for that exact horizon on the
same rows, and (MAE_h-naive_h)/naive_h, negative better. Retain annual pooled
sums/counts, every weekly diagnostic, S_h, U_h and exclusions. Never compare
different populations through a shared naive or substitute an aggregate for
failed heads. Primary accuracy uses each family's declared available support.
Also score both families on shared origins and per-head masks before pairing;
pair-eligibility is checked on exactly the predictions the strategy could use.

Finite sequence: seal/review -> MG2 fit/report -> MG12 fit/report -> pair
eligibility -> validation strategy if eligible -> publication/mirror closure.
Complete the long measurement even if short eligibility fails: it answers a
separate question. Failure of short eligibility prevents trading, not the long
fit. A technical failure produces a typed terminal and dependency disposition,
not repeated retries, a hidden alternative design or a false COMPLETE.

The strategy requires the established minimum valid long heads and all other
frozen rules. Do not fabricate missing/failed short or long heads, relax the
naive gate, or use rejected predictions to obtain a trading result. If the
pair cannot qualify, close PAIR_NOT_ELIGIBLE with results and reasons.

This is a new treatment with a different time encoder and training population.
DS2 versus MG2 is descriptive, not an isolated downsampling effect, even on a
common evaluation subset. The prepared hourly control remains a later bounded
contrast, not a prerequisite or automatically launched extra experiment.

## 5. Confirmatory TEST and costs

TEST 2025 remains closed. Require reproducible producer and point-in-time
evidence for every unresolved source before final confirmation. Release time
alone does not establish which historical value was available if the source
can revise observations. Retain vintage values/intervals or equivalent
value-level provenance, source timestamps, content digests and replay rules.
Use an existing governed credential only through its configured secret file;
official unauthenticated evidence, where available, does not require inventing
a credential gate. Never retrieve, print or commit secrets to this repository.

The versioned instrument configuration `f2443141` may support a clearly
ASSUMED_COSTS_DEVELOPMENT_ONLY validation backtest if the pair qualifies.
It does not establish broker/live performance. Calendar assumptions remain
ASSUMED_NOT_BROKER. Cost/provenance recovery runs independently, not as a
reason to idle a permitted development fit. Do not relax TEST readiness.

## 6. Resources, evidence and acceptance

Preferred first fit: 5090, one heavy GPU process on its host; no parallel
5070 Ti. Short precedes long. Route to the 4090 only after fresh admission
if the preferred host is unhealthy or cannot admit it. Reuse existing leases,
bounded services, durable outbox, replica checks and status/ETA commands.
No change to DOIN repositories, services, consensus or old chains.

Before fits, independent CPU/GPU engineering checks must cover:

- current-t inclusion, future perturbation and exact elapsed target timestamps;
- weekend/DST offsets versus ordinary weeks, and unplanned gaps not compressed;
- actual per-example time/closure inputs reach the model under controlled weights;
- chronology, loss masks, same-row per-horizon naive and split/purge boundaries;
- serialization/reload of ALL inputs, compatible identity checks and old DS2 replay;
- whole campaign recovery, exclusive claims, publication idempotency, and
  progression to long even after a scientific short failure.

Use test-led implementation and a separate bounded reviewer. Existing passing
tests do not certify the new model. Seal designs and runtime pins before any
scientific fit. First full fit is its own cost measurement, not a discarded
pilot. Later reservation is 1.25 times its measured full-cgroup peak if it fits;
do not lower margins to evade admission. No CPU fallback for scientific fits.

Report real stage and unit denominators and ETA (UNMEASURED until timed), OLAP
write/readback, two verified physical copies, and an indexed numeric paragraph
per stage. Regenerate M1-M8 without inventing percentage improvements. Agents
finish after installing finite automation and checking one actual start;
they must not poll logs, retrain completed identities or await chat approvals.
