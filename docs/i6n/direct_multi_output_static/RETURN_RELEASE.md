# I6-DMO corrected return: exact-DS2 short release (order section E, post-audit)

Date: 2026-10-15. This supersedes the audited `RETURN.md` of `fdaae9d2`
(REVISE_BEFORE_CONTINUE). Release pin `2c03792e` on `satoshi/i6dmo-static-20261014`; reviewer
verdict ACCEPT_FOR_RELEASE with addendum `6a688a67` (base review on frozen `22691757`).

## 1. Commits and identities

- Release pin `2c03792e` (contains the audit `98d83fcf` and correction lanes: data `cdb21385`,
  model `e8968d67`..`2c03792e`, strategy `3268079d` + `722e5479`, ops `ba47cb73`..`93078a0f` plus
  the lane-4 audit corrections). Planning `2a440265`. heuristic-strategy pin `d40fd3b0`.
- Design `541ce40a` (unchanged); FINAL model effective config
  `287e4f11e4ef385234becf02d9ed31864d8a3a58fc0b4c7f3e4c6a765a76e097`.
- Exact short bundle `5f3d6782...0122` (BUNDLE.json `489ebb8d...d603`, rule `DS2_EXACT`, rows
  fit/inner/validation 14,279 / 624 / 4,122; row-id digests reproduce the pre-ruling exact build).
  Input `ef22ddb5...`; manifest `2ff34fa2`; calendar `782603fd` (ASSUMED, not broker).
- Refused for this identity: the ASOF diagnostic bundles `3e42c3f5` (short) and `e88f72aa` (long),
  and the superseded `5221615d`. The ASOF arm remains retained under its own identities, and the
  first ASOF fit remains HELD (AUDIT_HOLD, publication timer only).

## 2. The scientific fit (justified repetition under a new identity)

Campaign `i6dmo-a9b15cff6f4b`, root on the relocated filesystem that meets the 20 GiB disk gate.
`short.fit` ran on the **preferred role's RTX 5090** (preflight PASS, admission under the measured
class cap 1.25 x 2431946752 = 3039933440 B):

- COMPLETE, scientific=true, identity `2cf66f7696...`, trainer effective `287e4f11`.
- 28 epochs, best epoch 18, stop=patience; fit 94.27 s, wall 105.0 s; cgroup peak 2,651,291,648 B
  (within cap; the ASOF diagnostic's 2.43 GB peak was a valid class estimate).
- After completion no compute process remains on the GPU host. This differs from the held ASOF
  diagnostic (32 epochs/32.2 s): different sampling treatment and more data exclusions, so the two
  runs are not comparable repetitions.

`short.report` COMPLETED through the durable timers (PAIRED_METRICS_AGREE: C2, manual and disk
replay concur on identical rows). Validation year 2024, original price USD/EUR, relative error
(MAE_h-naive_h)/naive_h, negative is better:

| h | n | MAE | naive | relative | eligible (MAE<naive) |
|---|---|---|---|---|---|
| 2 | 4,121 | 0.00076460 | 0.00074753 | +0.0228 | no |
| 3 | 4,065 | 0.00097452 | 0.00093548 | +0.0417 | no |
| 4 | 4,014 | 0.00114524 | 0.00109250 | +0.0483 | no |
| 5 | 3,958 | 0.00132825 | 0.00124917 | +0.0633 | no |
| 6 | 3,907 | 0.00149393 | 0.00139809 | +0.0686 | no |
| 12 | 3,587 | 0.00231534 | 0.00211348 | +0.0955 | no |

**No head is predictively eligible: this ONE minimal static R0 exact-DS2 short model does not
beat the per-horizon naive on 2024 development.** The negative result is scoped to exactly that
identity. It says nothing about other predictors, modular extraction, R2 donor transfer, other
retraining policies or the heuristic strategy, which remain unmeasured here. The strategy is not
run on ineligible heads and TEST is not opened to rescue them.

**Causal attribution withdrawn.** No claim is made that earlier, more optimistic numbers were
caused by ASOF sampling, by TRAIN appearing in a fitness, or by the 2024 feature selection: this
run is not a controlled ablation of any of those factors. ASOF sampling is not inherently future
leakage - the specific violation was the unbounded substitution of the sealed exact-input
treatment; TRAIN inside a fitness was a declared regularizer, not an independent validation
score; and the 19-feature selection still used 2024, so this measurement remains development,
not an untouched holdout or a fully resolved point-in-time dataset.

Closure counts from the generated state (reports are not interchangeable with metric rows):
**55 delivered reports, 2,011 metric rows, zero pending, zero mismatch**, exact readback on all,
and **28 files verified on each of the two physical mirrors**. No fit is running anywhere, and
the campaign timers cannot resolve design decisions by polling: the long family and the pair
await rulings/artifacts, not time. Publication delivered through the
outbox with exact readback under the Musashi-installed warehouse EnvironmentFile; mirrors to the
local and second-host paths. The long chain holds under `WAIT_DESIGN_RULING`
(LONG_EXACT_EMPTY_SUPPORT) and the pair stage stays gated (sealed execution policy; the
instrument configuration is not yet in the effective config, a typed wait).

Status: `python -m tools.i6dmo status --root "$I6DMO_ROOT" --level campaign --json` (unit, stage,
publication, fleet levels available). Expected validation units: 5 (1 COMPLETE, 1 advancing,
3 design-held). ETA: per family; the short family is measured by this fit; the long family is
UNMEASURED and a short fit is never its ETA.

## 3. DMO traceability and the honest DMO10 story

`TRACEABILITY.json` (20 entries, every pointer verified, commit-pinned) plus
`audit_20261014/GPU_REPLAY.json`. CPU acceptance: the full `tests/i6dmo` suite passes
(283 at the release pin). GPU-backed engineering evidence on the admitted 5090: DMO01
(causality) and DMO04 (masking) replays PASS.

**DMO10** is reported exactly as adjudicated (Musashi clarification `3f10878c`):

- Outcome label: **NUMERICALLY_EQUIVALENT** under the sealed, separately versioned monitor-replay
  protocol — never BITWISE_IDENTICAL, never plain PASS. Partial-sum relative deltas at closure:
  max 4.60e-8 per head (floor declared prospectively as 1e-9 + 5e-5*|recorded|, monitor comparison
  only; reload and predictions stay bitwise).
- The old bitwise witness **keeps its FAIL** (row-level 96/96 rows, max relative 1.211e-4 on a
  near-zero row), so the GPU pytest selection remains red by design.
- Decision impact (negative control, no cross-context invariance claim): the divergence could not
  change the best epoch or the patience stop for the fixture unit (relative uncertainty 1.82e-5).
  **Near-tie caveat:** a run whose monitored values tie within the floor could have its stopping
  decision affected; the record reports this per unit rather than assuming it away.
- Bounded isolation diagnostic: flags-on vs flags-off fresh GPU children produce byte-identical
  predictions — **DOES_NOT_SUPPORT_AUTOTUNE_HYPOTHESIS**; the stale-path explanation for the
  earlier identical rounds is recorded as a hypothesis, per the reviewer's wording items.
- GPU coverage gaps named by the reviewer stand: only DMO01/04 have GPU replays; the other DMO
  properties are CPU-tested engineering acceptance, and no scientific claim is derived from tests.

## 4. DOIN scope (confirmed once at release)

The POO-v2 runtime clones on all three hosts match the audited pins exactly and are clean:
doin-node `393049dd`, doin-core `9f9d3f75`, doin-plugins `2fc0523e`. All retired POO-v2 services
are stopped/disabled and the successor timers inactive; archive timers and warehouse transports
stay deliberately running. No DOIN source, config, chain or consensus change anywhere in this
campaign, and the DMO10 numerical-equivalence tolerance applies ONLY to the sealed monitor-replay
protocol — DOIN work-proof and chain rules are completely outside it.

## 5. OLAP, mirrors, M1-M8

- Publication is installed for the new root (the publish timer loads the existing warehouse
  EnvironmentFile; no token read or printed). The retained ASOF cost report (5 metrics) was
  DELIVERED with exact readback during the audit. The new root's facts deliver via the outbox
  with exact readback as its terminals arrive; the VALIDATION_DELIVERY gate seals only on full
  delivery plus VERIFIED mirrors.
- Mirrors: local and second-host paths configured on filesystems passing the disk gate; verified
  by re-hashing on every publish pass.
- M1-M8: statuses and percentages untouched this round (last change was the M6 text note of
  `fdaae9d2`, no percentage moved).

## 6. Limitations

- The 19-feature set was selected using 2024: outer VALIDATION is development, not a holdout.
- FRED clocks (DGS3MO, DGS7) remain UNVERIFIED — missing artifact: ALFRED per-observation
  `realtime_start` 2019-2024 or H.15 release timestamps; no governed ALFRED adapter exists —
  so confirmatory TEST stays closed (`PROVENANCE_UNRESOLVED`).
- The calendar is ASSUMED, not a verified broker calendar.
- The long family cannot fit under exact sampling (0 rows; every 144h/12h ladder crosses a >=47h
  weekend closure): held as `WAIT_DESIGN_RULING` for the plan owner; pair and finals stay gated.
- Strategy costs are NOT_BROKER_VERIFIED simulation values; the pair's sealed execution policy
  awaits the versioned instrument configuration.
- DMO10's GPU evaluation identity is numerical equivalence under a declared floor, not bitwise;
  the near-tie decision-impact caveat above applies to future runs.
