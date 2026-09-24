# RP144–RP151: B's twelve cells trained and closing, A's composition repaired without losing the mean, the doctoral contrast running

Orders [RP144–RP151](../../handoffs/MUSASHI_RP144_RP151_NEWS_AND_EXPERIMENTS_2026_09_24.md) at `d4bb3353`, after the
[RP140–RP143 review](MUSASHI_RP140_RP143_REVIEW_2026_09_24.md). Branch `satoshi/rp132-rp134-20260923`. What is running is
reported first, then what was measured, then the lanes I did not start and why.

## 1. Live ownership, checked before anything else

| job | host | state at 03:45Z | allocation |
|---|---|---|---|
| RP135 L512 fit queue | WORKER_B, external 5090 | **finished**: twelve of twelve cells trained, accepted receipts, service inactive | 14,011 CPU s of 24,000 consumed by the fits |
| RP145 closure of B | WORKER_B, external 5090 | **running**, 1,131 CPU s so far, on the H720 catalogs | the remaining 9,989 CPU s of the same allocation |
| RP146 development contrast | WORKER_A | **running** since 03:40Z, seed 2021 in progress | a NEW 14,400 CPU s / 8 h allocation, declared before it started |

The expired Hermes task was not unblocked and no duplicate was launched. The last return's "ten of twelve" is superseded by
the live reading above, which is why the first thing this round did was look.

## 2. Protocol B: twelve cells trained, recorded metrics

From the campaign's own accepted terminals, author float32:

| horizon | seed 2021 | seed 2022 | seed 2023 | mean MSE |
|---|---|---|---|---|
| 96 | 0.125551 | 0.125849 | 0.126375 | 0.125925 |
| 192 | 0.144051 | 0.142953 | 0.143947 | 0.143650 |
| 336 | 0.152274 | 0.152304 | 0.153511 | 0.152696 |
| 720 | 0.179536 | recorded | recorded | pending the closure |

These are **recorded training terminals, not accepted science**: the closure that produces the official reduction, the
same-row naive, the seed dispersion and the replay is the job still running. Table 9 publishes 0.126 / 0.220 at H96 and its
per-horizon searched lookback stays unresolved, so B is the released L512 recipe and not a claim of exact Table 9 identity.

## 3. Protocol A: the composition attack closed, and the mean survives on real links

All five of the review's counterexamples are repaired and are now tests.

| counterexample | before | after |
|---|---|---|
| metric-only evidence at shape [1, 1, 1] | METRIC_AND_SHAPE_BOUND, pooled | METRIC_WITHOUT_POPULATION, contributes nothing |
| a replay naming a checkpoint this root cannot answer for | IDENTITY_BOUND | IDENTITY_UNVERIFIABLE |
| a report row carrying no accepted-artifact digest | bound, because only mismatches were checked | NO_ACCEPTED_IDENTITY unless a link is present on both sides |
| a binding object with an empty identity block and a green flag | bound | NO_COMPARISON |
| the same wrong donor imported into both regimes | reload parity passed | checked against the donor file's own bytes |

Two corrections went the other way and are worth stating, because a repair that is too strict destroys evidence just as
surely as one that is too lax. The eight deleted cells' rows carry no `accepted_artifacts` at all: they are HISTORICAL_BOUND
rows, and the identities they do carry are their own replay's and their regeneration record's, which this root can answer
for. And the metric catalog and the regeneration ACCEPTANCE are derived objects that each closure and each acceptance round
rewrites, so binding on them produced false refusals between roots holding the same cell. Identity keys are now the cell's
immutable artifacts only.

Under the repaired binder the corrected closure is additive and the numbers are unchanged:

| horizon | mean MSE / MAE | pooled | status |
|---|---|---|---|
| 96 | 0.135496 / 0.232884 | 3/3 | OPERATIONAL_AGREEMENT |
| 192 | 0.157645 / 0.252163 | 3/3 | OPERATIONAL_AGREEMENT |
| 336 | 0.164480 / 0.262985 | 3/3 | OPERATIONAL_AGREEMENT |
| 720 | 0.190226 / 0.290617 | 3/3 | OPERATIONAL_AGREEMENT |
| four-horizon mean | **0.161962 / 0.259662** | 12/12 | against a published 0.158250 / 0.255750 |

Fifteen bindings are IDENTITY_BOUND and three are METRIC_AND_SHAPE_BOUND with the population now actually checked; there are
no refusals. Device attribution is unchanged and stays UNKNOWN or INFERRED_GPU_MEMORY: bit-identical replay does not recover
a training UUID.

## 4. The doctoral contrast: three gates repaired, then sealed and started

- **An independent oracle now checks the batches the model is fed.** It rebuilds inputs and targets from the raw CSV with the
  author's train-only statistics and slices rows, never calling his Dataset a second time, at the TRAIN split's first and last
  windows and at H96 and H720, and it recomputes the metric reduction on a fabricated offset. The previous check called the
  same dataset twice and could not have seen a batching or channel-order defect.
- **A validation mask is a property of the window, not of the read order.** The mask of origin `o` is drawn from a generator
  seeded by `(seed, o)`, so a fixed validation batch is byte-identical on repeated access while the training stream keeps
  drawing fresh masks. Musashi measured 20 positions moving between two reads.
- **The regime proof is an observed optimizer step.** `reload_parity ... or d1 == d2` is gone; the import is compared against
  the donor file's own bytes and against the auto-encoder that wrote it, a different donor fails, and one real step shows R1's
  detector moving by exactly 0.0 and R2's by more. A gradient that exists is not an update.

With those passing, the first **development** contrast is sealed and running: H96, three paired seeds, R0/R1/R2, the full
321-channel target, the author's split and normalisation, one auto-encoder per seed whose donor bytes R1 and R2 share, one
measured cost probe fixing one update allowance that every regime receives, the outer test never read and the outer
validation read only as the declared checkpoint rule. **No score from it is reported here and no H1 claim is made.**

## 5. Lanes advanced but not completed, stated plainly

- **RP147, real Laya.** The `news-signal` repository was cloned and read. Its own status says real weights, calibration and
  latency are NOT measured. The pinned identities are recorded: M5PHET at `bcd65b78c12e53e85adc9f2328cb0c0faccb94e8` and the
  Laya SDK at `1e28ac20c0896b1c37a744cd11f740eb98f8b178`. I did **not** start the checkpoint download and the bounded
  coordinator pilot: two heavy jobs already hold both workers, and the coordinator is the machine the owner is using. That is
  the named next runnable item, not a blocked one.
- **RP150, the typed contract.** Verified for real, not from a README. In an isolated environment the pinned M5PHET installs
  and its classification result carries `schema m5phet.classification.v1`, `uncertainty UNCALIBRATED_CLASS_PROBABILITIES`,
  `execution_authorized: false`, the three mandatory identity digests and a canonical `result_sha256`. Four refusals were
  executed: an uppercase digest, a distribution that does not sum to one, a label that is not maximal, and an incomplete task
  population. This verifies the contract, not any trained model.
- **RP148 and RP149, the news stream and the brokers: not started this round.** No feed or broker call was made and no
  entitlement was tested, so there is nothing to report as progress and nothing that could be mistaken for a fill.
- **RP150, financial.** The producer-contract investigation published last round stands; nothing new was read and the reserve
  was not touched.

## 6. Tests and cost

Suite **150 passed** on WORKER_A across the reproduction and ECL modules, plus 14 ECL tests on the real governed delivery.
Measured this round: about 600 CPU s of tooling tests on WORKER_A, 0.95 CPU s of read-only composition on WORKER_B, and a few
seconds for the isolated contract check on the coordinator. The two long jobs report their own budgets above.

## 7. Review request

(a) The two directions of the binding repair: the four refusals, and the two cases where a stricter rule was destroying real
evidence. (b) Whether the identity keys are the right set now that the catalog and the regeneration acceptance are excluded.
(c) The independent batch oracle and whether it is independent enough. (d) The contrast's sealed budget rule, one probe and
one allowance for every regime. (e) The lanes I did not start and whether that ordering is right.
