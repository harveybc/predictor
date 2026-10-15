# MG2/MG12 measured and closed

Closure: 2026-10-15 08:30 UTC. Runtime `a169fcba`, branch
`codex/i6dmo-report-recovery-20261015`, campaign `i6dmo-9e1ebad89ff2`.
`CLOSE.json`: CLOSED / PAIR_NOT_ELIGIBLE. TEST was not opened.

The source campaign `i6dmo-72809b6c88a6` fitted MG2 successfully (54.2345 s,
13 epochs, best epoch 3, peak 2742071296 bytes), then short.report failed
with UNKNOWN_DESIGN_FIELD. That report still called the elapsed design loader.
The failure blocked MG12. The source fit, predictions, checkpoint and failed
report are retained. Its idle timers were disabled after successor deployment.

## Recovery and verification

The default report path now authenticates the market bundle and rebuilds
validation through the same market calendar table, with expected weeks and
exact elapsed targets. New regression tests reproduce the production error
for both families; corrected entry tests use real sealed designs and data-lane
bundles, with no arrays-loader override and no training. Verification: 27
report/deployment/identity/TEST-closure checks and 6 completed-fit recovery
checks pass. They include changed-design/data rejection, corrupted source
rejection and a fit counter showing only the original short and successor long.

The successor imports the authenticated short fit under its original identity
`8ca83a7b6d4be5c9e9f48ab72edd6cfb4015d3cf8ce29c203329e6194e5c9913`
and code pin `24f5a6cc`. It is marked ADOPT_ONLY. New reports and the long fit
carry `a169fcba`; no short retraining occurred. All scientific design, data,
seed, model and training files match. The explicit reuse receipt records
allowed reporting/orchestration edits and the original terminal digest.

## MG2 short: development 2024

Original price USD/EUR; each naive uses precisely the model's rows and horizon.
Relative error is `(MAE_h-naive_h)/naive_h`, negative better.

| Horizon h | n | Model MAE | Naive for h | Relative error |
|---|---:|---:|---:|---:|
| 2 | 4770 | 0.000746830980 | 0.000746639413 | +0.025657% |
| 3 | 4714 | 0.000928480155 | 0.000927952906 | +0.056818% |
| 4 | 4663 | 0.001079576743 | 0.001079245121 | +0.030727% |
| 5 | 4607 | 0.001225189846 | 0.001224562622 | +0.051220% |
| 6 | 4556 | 0.001361444465 | 0.001362120281 | -0.049615% |
| 12 | 4237 | 0.001992701489 | 0.001994559830 | -0.093170% |

C2/manual/disk replay: AGREE for all heads, zero failed prediction rows.
The 6 h and 12 h heads strictly beat their naive; the others do not. These
are small development improvements from one minimal static R0 model. The
19-feature set used 2024; no independent holdout or trading claim follows.
54 reports (annual plus expected weekly segments), 2006 metric rows delivered
with exact readback; zero pending/mismatch; both mirrors VERIFIED.

## MG12 long: development 2024

| Horizon h | n | Model MAE | Naive for h | Relative error |
|---|---:|---:|---:|---:|
| 18 | 3786 | 0.003120902937 | 0.002515266772 | +24.078407% |
| 24 | 3493 | 0.003776758189 | 0.002949375895 | +28.052792% |
| 36 | 2910 | 0.005112759691 | 0.003665948454 | +39.466219% |
| 48 | 2331 | 0.005983700784 | 0.004392569713 | +36.223240% |
| 60 | 1750 | 0.006688512471 | 0.004740862857 | +41.082176% |
| 72 | 1166 | 0.007236858118 | 0.005137349914 | +40.867534% |

MG12 completed on the preferred 5090, initial attempt, cap 4 GiB and the
original 3 GiB host reserve. The initial estimate is based on the same 13x19
inputs, six heads and model capacity as the measured short with fewer rows;
it was not a claimed measured long peak. No previous long fit or admission
refusal existed. Actual result: 124.5457 s fit, 37 epochs, best epoch 27,
patience 10, restored best weights; measured cgroup peak 2662621184 bytes.
Long report took 10.9050 s. C2/manual/disk replay AGREE for all six heads.

## Pair disposition, publication and remaining engineering

Two short heads and zero long heads beat their own naive. The frozen policy
requires at least one eligible short and two eligible long heads. The pinned
pair hook executed and returned PAIR_NOT_ELIGIBLE; strategy.validation is
abstained with that disposition. No trades, TEST rows or additional fits.

An additional real integration defect was found: worker preconditions required
strategy OHLC pins before the pure pair eligibility step. Those pins are
absent. The negative decision does not read OHLC bars. A bounded operational
recovery verified the upstream report digests and sealed thresholds, proved
zero eligible long heads, and executed the ORIGINAL pinned pair entrypoint
through normal resource admission, queue claim, governed subprocess and stored
terminal verification. It did not fabricate missing inputs, change the unit
identity or mark a result complete without execution. The source and proof
are retained in private operational evidence and NEGATIVE_PAIR_PREFLIGHT.json.
The ordinary worker gate still needs the CPU-only correction in the handoff.

Final publication: 110 reports / 3952 metric rows, exact readback, zero pending
or mismatch. This includes the executed pair disposition. Both physical
mirrors verify 58 files, including CLOSE.json and closure-input proof snapshots.
The close receipt's publication/mirror hashes match those retained snapshots.

The same immutable runtime commit was staged on all executing roles. No second
heavy slot was enabled on the preferred host. Mutable state/SQLite/locks/private
configuration remain on ext4, large immutable archives on the configured NTFS
volume. Timers used 30 s worker/tick and 60 s publication operational drop-ins;
after closure all five campaign timers were disabled. ETA for this measurement
is DONE. No agent or scientific child remains running for this campaign.

M1-M8 remain 66/78/78/48/22/24/28/15. Source/calendar/broker assumptions
remain unresolved for confirmation. These measurements finish this minimal
static R0 baseline, not the architecture's broader validation program.
This result does not isolate the effect of market sampling versus DS2 because
the population and physical-time encoding also changed.

## Runtime qualification added 2026-10-15

The retained short and long designs declare TensorFlow 2.18.0 / Keras 3.9.0,
but both executed fit receipts record TensorFlow 2.21.0. This is a confirmed
runtime-seal nonconformance, not an input/target look-ahead finding. The
observed predictions/errors and original closed receipts remain retained;
they do not certify execution under their declared runtime. Do not reuse
them as runtime-compatible donors or claim a runtime-matched comparison.
No retroactive reseal or automatic refit is authorized. The new matched
control has separately sealed runtime identities. See the
[follow-up audit](matched_short/MUSASHI_AUDIT_20261015.md).
Machine-readable record, label `RUNTIME_SEAL_NONCONFORMANT_OBSERVED_TF_2_21`, citing both design seals, both
receipt hashes and the observed versions: [RUNTIME_QUALIFICATION_MG2_MG12.json](RUNTIME_QUALIFICATION_MG2_MG12.json).
