# RAW direct multi-output: current execution priority

Owner correction, 2026-10-15. Executor: Musashi, not Satoshi. This order
supersedes the R2-before-long sequence in NEXT_CONTRASTS_PLAN_20261015.md.
It does not rewrite scientific identities or launch a fit by itself.

## 1. Measured state and missing evidence

| Work | Actual evidence | Disposition |
|---|---|---|
| RAW exact-DS2 short | Six heads worse than their own naive | Complete, negative |
| RAW market-time MG2/MG12 pair | Both families measured; no eligible long head | Complete, PAIR_NOT_ELIGIBLE; strategy correctly not run |
| Matched RAW short MG1/MG2m | Common FIT/INNER/DEVELOPMENT, MG2_SELECTED | Complete; MG2m beats its naive only at 4/6/12 h |
| Matched RAW long downsampled/hourly | No matched hourly long arm retained | Not complete; immediate design/implementation gap |
| Complete eligible RAW pair in strategy | No eligible pair | Not executed, not evidence of trading failure |
| Branch R2 | 19/19 donors, R0 and diagnostic probe complete; R2 import failed before fitting | Deferred; not a prerequisite for RAW |

The historical MG2/MG12 receipts declared TF 2.18 but recorded TF 2.21.
Retain them with their qualification; they are not a runtime-matched control
for a new experiment. A new matched question is not a seed repeat of an old
fit. Reuse only when every required identity and population agrees.

## 2. Finite progression

1. Retain completed downsampled short/long and matched short evidence. Do not
   repeat these fits merely to populate a GPU.
2. Define, implement and review ONLY the missing RAW long hourly-versus-
   downsampled control. Both arms must have the same physical context,
   horizons, common FIT/INNER/DEVELOPMENT masks, FIT-only scaler, optimizer,
   runtime, seed and stopping rule. Their readout parameter-count difference
   must be disclosed, not attributed solely to sampling. Preserve the current
   long-family physical horizons 18/24/36/48/60/72 h; do not resurrect the
   superseded 120-hour heads silently.
3. Execute the two sealed RAW arms and report each full DEVELOPMENT-2024
   horizon against naive for that horizon on exactly its own scored rows.
   Negative results still close the experiment. Never select missing heads,
   fill them with price(t), average away ineligibility or read TEST to tune.
4. Evaluate the complete short/long pair against the frozen eligibility
   policy. If eligible, run the prediction-based strategy on DEVELOPMENT,
   recording model/config/feature hashes, initial capital 10000, trade count,
   costs, profit, drawdown and Sharpe. If not, close PAIR_NOT_ELIGIBLE with
   explicit per-head reasons; do not force a misleading trading run.
5. TEST remains closed until the full procedure and source-availability gates
   pass. TEST is not a hyperparameter-selection split. Do not substitute a
   DEVELOPMENT backtest for a confirmatory TEST result.
6. Only after the RAW block closes: consider separately declared capacity,
   cadence and branch-pretraining contrasts. R2/core AE/DOIN/RL/NEAT cannot
   become implicit dependencies of the missing RAW result.

## 3. Non-negotiable experiment and automation rules

Use 19 retained RAW features, no target denoising, no input reconstruction,
no smoothing/interpolation. Include current t in each window; targets are
exact elapsed t+h and issuance remains hourly. Open-market input sampling
is not a relabelled elapsed-time target. Retain actual offsets and closure
flags, per-head business masks, label maturity, forced-flat/reopening/entry
rules and the same masks for model and naive. The initial cadence is NONE;
later retraining comparisons are separate experiments, not hot changes.

Relative error is (MAE_h - naive_h)/naive_h; negative is better. Stopping uses
the declared mean of end-of-epoch FIT/INNER at identical weights, patience 10,
min_delta 1e-8, best restoration. One seed, no unnecessary repetitions.

Reuse the existing finite campaign queue, admission, workers, terminal
verification, outbox, warehouse readback, physical mirrors and automatic
close. Missing long-family factory support must be implemented and tested;
never monkeypatch short-family globals or disguise a long run as a short arm.
Status must name this gap until it exists; no RUNNING label or fabricated ETA.
Prefer the 5090; one heavy job on its host, no concurrent 5070 Ti. Independent
work may use the 4090 after fresh admission. Reserve coordinator memory.

Do not erase or retry the failed R2 import in place. Its code was actually
executed and produced a typed failure. Keep the first failure and all donor
bytes; when R2 is resumed later, repair and test cross-host donor staging,
verify terminal AND weight hashes before GPU launch, then use an explicitly
reviewed recovery/adoption path without changing a closed receipt.

All metrics, exclusions, costs and dispositions go through the existing OLAP
outbox/readback. Code/configs/summaries/hashes belong in Git; large artifacts
and databases remain on physical storage with authenticated mirrors.
