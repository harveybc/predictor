# Audit of Retsu PS4 and PS3-R return

Observed independently on 2026-10-03 after Retsu's return. This audit narrows
claims; it does not create a feature-selection decision.

## PS4 transform profiles: accepted in their declared scope

- Integrated source: `retsu/ps4-transform-profile-20261003@3026381c`.
- Population: ten emitted causal transform features by five inner TRAIN folds,
  50 feature-fold units and 1,050 metric rows.
- Every unit has the same 21 declared information/compression metrics. All
  output identities are unique, every completed value is finite, and all rows
  carry one source-digest set and profiler digest
  `68bc686ac8bf5a4679372332c857b869c50ac63228ddc343847fb913596c7f9f`.
- The profiler digest equals the bytes of `tools/df_profile_information.py` in
  the audited tree. The three reported artifact hashes and every retained input
  hash were independently recomputed and match.
- The focused suite passes 10/10. Its temporal mutation test changes rows after
  a fold cutoff and proves that the fold output is unchanged, then changes a
  row inside TRAIN and proves that the output changes.

Disposition: `ACCEPTED_PS4_TRANSFORM_SUBPOPULATION`. This closes the pending
profile state only for these ten emitted transforms. It is not PS4 coverage of
the 366 candidates and is not feature selection.

## Dragon PS3-R cell: authentic measurement, not a selection verdict

Cell `batch_002/fred.stress.vixcls.logret_5d`, seed 0, completed on the RTX 4090.
The retained results hash is
`ec16a107f0ecc2e5c86002e15a2397e214e2d7c0a9df1145048f678e14546f8a`;
the file has 441 rows: 20 fold-family rows, 280 probes, 140 paired deltas and
one summary. The run used five inner TRAIN folds, `window=168`, timestamp-based
seasonal context (`calendar_dim=6`), identity/random/AE/DAE controls, restored
best checkpoints and seed 0. The log proves TensorFlow created the RTX 4090 and
loaded cuDNN. The incomplete CPU attempt is retained in a distinct directory.

Measured interpretation is mixed. Against the random encoder, AE has a tiny
mean improvement on Y_s and Y_l but wins only 17/30 fold-horizon cells in each;
DAE wins 13/30 on Y_s and 9/30 on Y_l. Both trained encoders lose the Y_b
log-loss comparison in 10/10 cells. Reconstruction is excellent, but that does
not establish useful target representation. The cell therefore remains one
PS3-R evidence row and cannot select or reject the feature by itself.

Disposition: `ACCEPTED_PS3R_CELL_MIXED_UTILITY`.

## Gamma admission: stale retry instruction withdrawn

No model child ran for `fred.rates.dgs30.level`. The 8,000 MiB waiter was
cancelled and the one 7,936 MiB request was correctly rejected as
`CAP_LOWERED_AFTER_REFUSAL`. Waiting 3,600 seconds and submitting the same lower
cap would evade the recorded refusal rather than add memory evidence. That
retry is withdrawn. The current `fred.rates.dprime.logret_5d` request is also
only queued at 8,000 MiB with `HOST_HEADROOM`; it is not GPU work.

Heavy 8,000 MiB PS3-R cells move to dragon, which has admitted the real
workload. Gamma's external 5090 receives the already measured 6,670 MiB
alternative-extractor lane, one job at a time. The two gamma GPUs remain
non-concurrent because they share host RAM.

## Program state

- Denominator remains 366 candidate features.
- PS3-C `NOT_IDENTIFIED` remains neutral.
- No final selected/rejected/pending manifest exists.
- M1 weekly business evaluation and M2 feature selection remain co-critical.
- ARCH, R0/R1/R2 on the selected manifest, H-CORE, Dense/NEAT, real-data RL and
  strategy evaluation remain downstream and are not authorized by this return.
