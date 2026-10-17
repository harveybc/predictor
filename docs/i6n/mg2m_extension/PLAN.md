# MG2m Extension And Window Study

Owner approved 2026-10-17. Parent: matched MG2m, code 796ec8f4, campaign
i6dmo-6004b711281c, seed 6102026. DEVELOPMENT 2024, TEST closed. No DOIN,
pretraining, feature selection, new transforms or cadence changes.

Sequence: plus18 (2/3/4/5/6/12/18 h); single18 if 18 fails naive or the short
mean deteriorates; short window7, control13, window25, all at MG2 spacing.
The two extra windows cover 12/48 open-market hours, including t. Window
comparisons use the same intersection of available input origins in all three
windows, with inherited per-head business masks. Reuse the original control13
only if every split array and scaler is identical; otherwise one necessary
matched-population control fit, never a silent reuse on different training rows.

E01: authenticate all original receipt artifacts and recover original MATCHED
population, union-of-arms FIT-only input scaler, masks and targets. The plus18
arm preserves the first six columns, origins, inputs and loss scales exactly.
E02: 18h targets are exact elapsed t+18, subject to original C1 business and
split maturity; no intersection-all-heads filter or weekend padding.
E03: legacy train.py remains byte-identical. Model changes only the declared
family catalogue/lookup; layer builders and temporal encoder stay unchanged.
E04: plus18 has 5315 parameters, same initial shared/short-head tensors and
short forward as the original six-head model; compare original initial digest.
E05: retain exact original mean FIT/INNER stopping, AdamW, batch256, patience10,
min_delta1e-8, max200. Save epoch2 checkpoint and score as diagnostic, not a
replacement selection rule. No new fit of the six-head reference for plus18.
E06: own cached naive per split/horizon/support; recompute from saved prediction
arrays before publishing, fail on any mismatch or nonfinite prediction.
E07: one heavy GPU child per host, governed 4GiB memory cap, no CPU fallback,
no 5070Ti parallel slot, no automatic fit repetition after interruption.
E08: finite timer-driven queue; per-cell status/ETA; governed OLAP outbox and
readback; authenticated copies on two hosts; automatic close and timer shutdown.
E09: full config, seed, source/code/input/weight identities, per-head history
and intentional differences from the parent retained. Register new results
in the existing best-result ledger without changing the pinned reference.

Acceptance descends from E01-E09 to CPU mutation/contract tests, then actual
read-only preparation and pre-fit GPU structural/initialization checks.
Implementation ascends to single fits, metric replay and publication/mirrors.
Review no scientific conclusion before completed fits and report readback.
