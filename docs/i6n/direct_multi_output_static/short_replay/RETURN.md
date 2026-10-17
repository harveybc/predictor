# Original Short Replay: Closed

Follow-up: [reference-device reproduction](REFERENCE_DEVICE_RETURN.md) closed
with exact historical trajectory, weights and scores. The secondary-device
replay below remains retained as measured, not overwritten.

One R0 fit, runtime ee40787e, actual TF2.21.0/Keras3.13.2, secondary 4090.
The original model.py, train.py and all eight historical trainer code pins,
effective config, source bundle, scalers and initial weight hash match the
historical short run. Original train_unit entrypoint; no new stopping rule or
extra model forward before training. Only short heads 2/3/4/5/6/12h.
Historical execution was on a 5090; cross-device exact training was not promised.

New fit: 15 epochs, best 5, stopped by mean FIT/INNER patience 10, min_delta
1e-8, max200; 66.2009 seconds, cgroup peak 2,799,689,728 B under 4 GiB.
Historical fit: 13 epochs, best3, stopped by patience. Same seed 6102026.

| Horizon | N | New MAE | Naive for horizon | Historical relative | New relative |
|---|---:|---:|---:|---:|---:|
| 2h | 4770 | 0.000748444551 | 0.000746639413 | +0.025657% | +0.241768% |
| 3h | 4714 | 0.000929363886 | 0.000927952906 | +0.056818% | +0.152053% |
| 4h | 4663 | 0.001081426830 | 0.001079245121 | +0.030727% | +0.202151% |
| 5h | 4607 | 0.001224270896 | 0.001224562622 | +0.051220% | -0.023823% |
| 6h | 4556 | 0.001364192671 | 0.001362120281 | -0.049615% | +0.152144% |
| 12h | 4237 | 0.001997726014 | 0.001994559830 | -0.093170% | +0.158741% |

Relative=(MAE_h-naive_h)/naive_h, original USD/EUR scale. Rows, masks and each
horizon's naive equal the historical control. The new training returns near
naive, unlike the twelve-output fits, but does not reproduce the historical
6h/12h improvements. Only 5h beats naive in this run. DEVELOPMENT 2024, TEST
closed; no full-pair strategy run or profitability claim.

## Read-Only Diagnostic

The initial weights match exactly. Already after epoch1, trained weights differ:
historical/new mean monitor 0.9417280953199912/0.9417280954116383; at epoch2,
0.9421109840400597/0.9420467014827973. This localizes divergence after updates,
not an input/config/initialization mismatch. Hardware/backend arithmetic is
a candidate explanation, not an isolated proven cause.

Then `replay_control.py` loaded the ORIGINAL historical learned weights on
the same secondary GPU and evaluated only the original 2024 validation rows.
NO fit was executed. The old improvements reproduce: 6h -0.0496149972%, 12h
-0.0931704846%; maximum MAE difference over all heads 4.619525248439338e-14.
The historical saved model and current metric/data path therefore reproduce
the prior scores to this measured precision. New training does not reproduce
its exact trajectory or selected checkpoint. A same-class controlled training
check would be a separate next decision; it was not silently launched.

## Delivery

Main replay: one 73-metric report read back exactly, 22 files verified in each
physical mirror, owned timers closed automatically. Saved-control inference:
separate 25-metric report read back exactly, three JSON diagnostic files mirrored
with identical checksums; no change to the closed fit's immutable manifest.
The first publisher invocation failed before publication because PYTHONPATH
was missing; setting the checkout path corrected it, without another fit.
Outbox SQLite remains on ext4; metadata copies are on configured storage.

Result digest: `3ae978c19b4ac9612e1d74f4b486178c0610de8f1c5be1acf31040904a839147`.
Inference JSON digest: `9ae41148c2d5790f593c3232a4a303f9ad92d422537953900880f883038aee1a`.
The main preferred GPU continuation was not changed or stopped.
