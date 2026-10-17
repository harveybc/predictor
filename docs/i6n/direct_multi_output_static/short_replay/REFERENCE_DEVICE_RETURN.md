# Original Training Reproduced On The Reference Device

Campaign `i6dmo-short-reference-device-20261016`, frozen runtime `a2c7fe6d`.
The original 5090 reproduced the complete historical training exactly: all 13
epoch weight digests, FIT/INNER monitors, updates, patience state, best epoch 3
and final learned weights. Source model.py and train.py remain byte-identical
to `24f5a6cc`. Effective config, source bundle, feature manifest, scalers and
initial weights match. No seed search, checkpoint tuning or model change.

| Horizon | MAE | Naive for horizon | Relative error | Historical parity |
|---|---:|---:|---:|---|
| 2h | 0.000746830980 | 0.000746639413 | +0.025657% | Exact |
| 3h | 0.000928480155 | 0.000927952906 | +0.056818% | Exact |
| 4h | 0.001079576743 | 0.001079245121 | +0.030727% | Exact |
| 5h | 0.001225189846 | 0.001224562622 | +0.051220% | Exact |
| 6h | 0.001361444465 | 0.001362120281 | -0.049615% | Exact |
| 12h | 0.001992701489 | 0.001994559830 | -0.093170% | Exact |

Relative=(MAE_h-naive_h)/naive_h. USD per EUR, exactly paired original rows.
DEVELOPMENT 2024 only; TEST remained closed. Two heads, not all six, beat naive.

## Finding And Correction

The previous replay was made on a 4090 rather than the historical 5090. That
was a failure to preserve the execution device for a training-reproduction
question, not evidence that the unchanged model/trainer source was damaged.
On the secondary device the same initialization took a different update
trajectory and selected epoch 5. Returning to the reference device/environment
recovered the exact original trajectory, weights and scores without changing
any scientific setting. This establishes the execution-profile dependence in
these measured runs, not the identity of a particular low-level kernel.

Both installed environments report TF2.21.0, Keras3.13.2, NumPy2.4.3,
cuDNN9.20.0.48 and cuBLAS12.9.1.4. Equal package versions and seeds alone were
not sufficient to justify expecting identical cross-device training.

The replay launcher now rejects a missing reference-device declaration or a
non-reference launch role, then checks the actual GPU name and visible UUID
before importing the trainer. Cross-device results remain separately retained
evidence and are not overwritten, hidden or promoted as exact reproduction.

## Verification

The comparison tool found no first differing epoch. All six receipt identity
checks and all six training checks pass. Independently, np.array_equal with
equal_nan=True confirms every stored array (horizons_hours, mask, origin_price,
predicted_price, row_id, target_price) in FIT, INNER and VALIDATION is exactly
equal to the historical predictions. This check did not execute another fit.

The trajectory comparison is reproducible with:

```bash
python -B -m tools.i6dmo.replay_comparison \
  --reference /path/to/historical/train \
  --candidate /path/to/reproduction/train \
  --out /path/to/new-diagnostics/TRAJECTORY.json
```

The inference/array check uses the matching split files in both prediction
directories, never TEST. Historical weight hash:
`95b42ea65472b3a5e8b442ecf54e5c5ce4a50b72b4b63431443639e01837763b`.

## Delivery And Operations

Fit: 54.2571 s, 13 epochs, best3, patience10, min_delta1e-8, max200.
Cgroup peak 3,483,820,032 B under 4 GiB. This was one necessary diagnostic fit,
the third execution counting the original and the secondary-device replay.
No further repetition was launched.

Main result: 73 metric rows, one report read back exactly, no pending/mismatch;
22 files verified in each physical mirror. The separate trajectory report has
13 boolean metrics, exact readback and three diagnostic JSON files verified
on two hosts. SQLite/WAL remain on ext4; large artifacts use configured storage.

The existing monthly fit was allowed to finish. No training process was killed.
The reference worker's ExecStopPost resumed the main queue automatically; its
timer is active and its next fit was observed active. Diagnostic timers are
closed; the independent recovery timer was canceled after successful recovery.

Result digest:
`38b587451a4abbd339d19ebe3cceb2948abc58edf31ea490e81074b7c4f9475a`.
Trajectory digest:
`bcb65250f9b478ebb872a07352a06c8e1ff5b1c1127102636f21a8a7919b9b97`.

The new device/identity/comparison regressions and the existing replay tests
pass: 36 tests. No DOIN, feature selection, masks, TEST or main-campaign
scientific identity was changed. The historical runtime-seal mismatch remains
qualified as before; it was not silently repaired by resealing old evidence.
