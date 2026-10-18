# Predict Then Update

The binding specification is [PROCEDIMIENTO_CANONICO.md](PROCEDIMIENTO_CANONICO.md).
The former same-week campaign is stopped and retained as superseded evidence.
Its trained models and predictions are NOT donors for this campaign. Its input
preparation is reused unchanged: no new feature selection, sampling, masks,
model architecture, stopping formula or hyperparameter search.

The initial donors for both families are their final usable VALIDATION-2024
checkpoints (week 2024-12-22). Their .keras files, tensor hashes, scalers and
configurations are authenticated. Unsupported boundary weeks carry that donor
forward without inventing scores or running an empty stopping monitor.

## Automation

```bash
python -m tools.i6dmo.prequential_test prepare --spec <private-config.json>
python -m tools.i6dmo.prequential_test launch --spec <private-config.json>
python -m tools.i6dmo.prequential_test status --spec <campaign-root>/RUN.private.json
```

The private configuration names the authenticated existing prepared data and
the retained pair, deployment roles, disk roots and warehouse EnvironmentFile.
No secrets, host identities or raw data belong in git. The shared deployment,
bounded timers, durable outbox, readback, physical mirrors and pooled annual
reporting are reused. One GPU unit handles one family/week. A family cannot
advance without its preceding checkpoint. The other family's independent ready
unit can still proceed. No agent polls logs to advance the campaign.

Each unit seals PREDICTED.npz and PREDICTION.json before reading OBSERVED.npz.
The prediction reader has no target or label mask. SCORE.json is then written
before FIT is loaded. The unchanged trainer warm-starts from that inference
donor, uses the completed week as its original mean-monitor stopping input,
and saves a next-week checkpoint. The input scaler used for inference belongs
to the donor; the refit scaler belongs to its four-year TRAIN window. Saved
predictions cannot be overwritten by a refit or a resumed unit.

Artifacts include both the pre-update prediction and post-update checkpoint,
separate TEST and OBSERVED-monitor metrics, donor/tensor/scaler identities and
the original per-horizon naive and paired counts. Qualification is
PREQUENTIAL_TEST_PREDICT_THEN_UPDATE. This describes online adaptation using
already-observed outcomes, not selection of a current week's checkpoint with
that same week's outcomes. Current-vintage FRED and assumed calendar limitations
remain declared; no new source certification is claimed.
