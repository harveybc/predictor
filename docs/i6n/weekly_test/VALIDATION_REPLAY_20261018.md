# Saved weekly VALIDATION checkpoint replay

Owner request: reload the saved best models and reproduce VALIDATION with the
original weekly adjustments, before attributing the TEST deterioration to overfit.

## Scope and acceptance

Load all 51 retained weekly checkpoints per family, not one final checkpoint
over earlier weeks. Each checkpoint is the output of its original four-year
weekly fit. Authenticate archives, tensor weights, configuration, scales,
prepared inputs and stored predictions. Recompute predictions with the same
inference function and batch partition; recompute each head's own paired naive
and MAE in price units. Report exact equality and the maximum difference;
never increase a tolerance to obtain acceptance. No training is needed to
replay already retained weekly adjustments; fits_executed must remain zero.

As a separate, non-governing diagnostic, apply each checkpoint to the following
available VALIDATION week using its own scalers. Require that the donor's
observed week ended before any scored origin. This diagnostic changes the
checkpoint selection/evaluation relationship on the same year; it is not a
rerun of the warm-start TEST campaign and must not be described as one.

No TEST reads, no new feature selection, no architecture/stopping-rule changes,
no DOIN changes, no edits to active runtimes. Keep the 4090 sweep intact. Use
one admitted process on the preferred GPU, bounded by the existing runtime cap.

## Tests before implementation

VR01: exact replay, a changed prediction fails exactness, zero-pair heads stay null.
VR02: aggregation pools sums/counts and retains different per-head naives.
VR03: applying a donor before its availability is refused.
VR04: a changed archive/input fails authentication before inference.
VR05: status reports partial work/ETA; immutable terminal identities survive resume.
GPU acceptance: authenticate all 102 archives, compare all retained predictions,
save the annual table and the separately labelled next-week diagnostic. Keep
the result even if exact equality fails. Publish metrics into the governed
warehouse and retain results on two physical machines.

## Small warm-start confirmation

After replay, use the first retained weekly checkpoint in each family as donor
and execute the next two consecutive VALIDATION weeks (four updates total).
Reuse prequential execute_week and GpuBackend with explicit VALIDATION metadata;
their default TEST behavior remains unchanged. Seal the prediction before the
observed-label reader, score, then warm-start the four-year update for the next
week. Retain patience10, min_delta1e-8, max200, mean TRAIN/observed-week loss,
the donor's inference scalers, and the original training scalers and masks.
This is a four-fit development diagnostic, not an annual reproduction or TEST.

Acceptance adds a synthetic test of VALIDATION labels/files and default TEST
regressions before launch. Saved inference replay still performs zero fits;
the separate warm-start pilot must report its own completed fit count.
