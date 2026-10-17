# Recovered MG2m And Automatic Best Results

## Correct Reference

The owner explicitly identified **matched MG2m**, campaign
`i6dmo-6004b711281c`, code `796ec8f4`. Its checkpoint, predictions, scalers,
training history and configuration are retained. The earlier replay used the
unmatched MG2 from `24f5a6cc`: that was a reference-selection error, not recovery
of this MG2m. No new training was needed for this recovery.

Every value below was recomputed from authenticated saved prediction arrays.
Negative means better than the naive for that horizon on exactly the same rows.
The later fresh R0 was already trained before this recovery; it is not an
additional repetition. Its predictions agree at the displayed precision, but
its weights are not bitwise identical to the original 4090 fit.

| Horizon | Original MG1 hourly | Original MG2m | Retained later fresh R0 |
|---|---:|---:|---:|
| 2h | +1.909251% | +0.004425% | +0.004425% |
| 3h | +2.880215% | +0.008874% | +0.008874% |
| 4h | +4.654819% | -0.016506% | -0.016506% |
| 5h | +4.865553% | +0.003669% | +0.003669% |
| 6h | +5.028680% | -0.009498% | -0.009498% |
| 12h | +6.587571% | -0.037392% | -0.037392% |

MG2m remains the best recovered **complete static short recipe** by mean
native-development relative error (-0.007738%). The retained fresh R0 belongs
to that same recipe, with microscopic numerical differences. Other runs have
better individual heads; they are not one combined trained model. No retained
complete annual short recipe was found with all six original-population errors
negative. Favorable individual weeks/months remain PERIOD results, not annual
performance. These are development comparisons, not TEST or profitability.

## Exact Recipe

| Setting | Original matched MG2m |
|---|---|
| Input | 19 selected RAW EURUSD features; 13 observations every 2 open-market hours, including t; 24 open-market hours of history |
| Branch per feature | Causal Conv1D, 4 filters, kernel 3, stride 1, ReLU |
| Fusion and time context | Concatenate; original dynamic physical-time encoding and closure context |
| Core | Causal Conv1D 8 then 4 filters, kernels 3/3, strides 1/1 |
| Each of six heads | Flatten 52 -> Dense 8 ReLU -> Dense 1 linear |
| Parameters / seed | 4882 / 6102026 |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001; batch 256 |
| Early stopping | 0.5*(FIT loss + INNER loss), equal-head masked MAE normalized by FIT-only naive; same end-of-epoch weights |
| Patience / threshold / limit | 10 / absolute 1e-8 / 200 epochs |
| Actual training | 12 epochs, best 2, stopped by patience; 54.584 s; original GPU 4090 |
| Data roles | TRAIN 2020-2023, purged INNER tail of eight weeks; DEVELOPMENT 2024; TEST 2025 closed |
| Runtime | TensorFlow 2.21.0 / Keras 3.13.2 |

The later change to stopping on actual TRAIN/VALIDATION 2024, private temporal
heads, additional outputs and removal of the sinusoidal value encoding are
**different experiments**, not an unchanged replay of this reference.

[Complete machine-readable reference](MG2M_REFERENCE.json) retains model layer
configuration, features, data/config/scaler/source-code hashes, actual monitor,
determinism flags, original weights and all six MAE/naive/counts. Original profile:
`145c3a4b51cde718383b52bcf2a863ee5df091e728bc0e48131c9e732ceaf53b`.
Later fresh R0: `ca34287dc113d5ba8ae06eb159ca753c98f993d7d350b0c824417ea89d108f38`.

## Continuous Record

`tools.i6dmo.best_registry` runs as a separate CPU-only systemd timer every
120 seconds, including after reboot. It does not train, import TensorFlow,
modify DOIN, read TEST or change scientific campaign configurations.

- `REFERENCE.json`: explicitly pinned original MG2m, never substituted by name.
- `profiles/<identity>.json`: full model/data/training/checkpoint provenance.
- `LEADERBOARD.md` and `.json`: best per horizon and best complete recipe.
- `BEST.json` and `events/`: strict improvements with previous/new identities;
  a changed comparison population starts a new cohort, never a fake improvement.
- `STATUS.json`: compute, quarantine, delivery/readback and mirror state.
- `PROVENANCE_READBACK.json`: verified full-profile/event tags from the OLAP.

Original-population results remain intact. A separate annual comparison uses
the intersection of actual eligible rows per horizon and rejects conflicting
prices. Its numbers can differ from the original table and are explicitly
labelled COMMON_SUPPORT. Incomplete cadences cannot win an annual ranking.

Deployment acceptance: more than 350 authenticated profiles recovered; initial
14 champion/cohort records published with exact metric and full-provenance
readback, zero pending/mismatch. Local archive and second-host copies verified
file by file. Bounded CPU sidecar peaked at about 195 MiB under a 768 MiB cap;
26 focused CPU tests passed. A second tick was idempotent. Source scientific
campaign continues under its original runtime; no new fit or TEST read here.

The live register is deliberately outside Git; snapshots here identify the
reference, not a claim that this Markdown updates itself. Its private spec
provides storage and warehouse destinations without committing credentials or
host identities. [Operation and status commands](README.md).
