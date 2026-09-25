# M5PHET round 6, 2026-09-25 — governed reads, and the calendar clock measured

## Governed access (WP15's second half)
data-gov's console script was broken because heuristic_strategy owns the top-level `app` package in anaconda's
site-packages; the honest repair was renaming data-gov's package to `data_gov` (branch
`satoshi/wp15-govern-20260925` @ d07695b), with a checkout shim so the documented `python -m app.main` still works.
Credentials were ISSUED BY data-gov's own code (`scripts/issue_service_key.py`: the key is verified against the
configuration it writes before the file is created), stored only under `~/.config/m5phet/` with mode 600, and served by
`systemd --user data-gov-m5phet.service` on a loopback port of its own. M5PHET master 785c6ee routes a governed
resource through the client, records a delivery receipt (lake, resource, delivered-bytes digest, identity, time) and
**refuses by name when data-gov refuses — never falling back to the ungoverned copy on disk**. Verified on the owner's
own instance: `e1_household_successor_v3` → 0.9631181359291077 kW with `profile: GOVERNED`.

## The calendar clock, measured (WP22)
The join that would have given an observed publication clock cannot run on this machine: the consensus archive spans
2011-01→2021-04 and the only archive with observed instants spans 2024-12→2026-05 — zero common days, join rate 0 of
54,275, with 27.2 % of rows clearing economy and release name and failing only on the date
(`EVENT_STUDY_DATA_STATUS.md` has the numbers and the three feeds that would unlock it).

What could be fixed was: the archive's wall clock is not UTC. `calendar_clock.py` measures it from the observed
announcement archive's own conventions, requires a cluster spanning zones with different daylight-saving calendars
(America/New_York and Australia/Sydney) so the agreement cannot be one convention's artefact, and names every
contradicting series. Result (`wp22_measured_calendar_clock.json`, 2,177 estimates): a **fixed UTC−05:00 year-round
until 2018-01**, then **America/New_York local time** from 2018-03 — which corrects the agent's own earlier eyeball
reading of "UTC−5 to 2017, UTC−4 after", and is the version in the artifacts.

Re-running the whole chain from where the releases actually happened (`wp22_projections_localized.json`):
- the two cells whose intervals excluded zero under the UTC misreading — NFP h+60 and CPI h+60 log return, both large
  and negative — **stop excluding zero**: they were reading the market four to five hours before the release;
- two new cells exclude zero at h+240 log return (CPI +7.24e-04, NFP +2.53e-04; the NFP one on 49 fitting and 10
  held-out events, which is thin);
- realized-volatility betas shrink by one to two orders of magnitude; all 20 intervals contain zero;
- the projection beats the naive sign-mean out of sample on **23 of 40** triples (was 15);
- superposition: ADDITIVE_HOLDS on 8 of 10, INTERACTIONS_IMPROVE on 2 (Crude Oil × Initial Jobless Claims, released at
  the same instant);
- **identification stays NOT_IDENTIFIED**, decided by the code, now for `ASSUMED_PUBLICATION_CLOCK` **and**
  `PLACEBO_FAILED on all 40` triples — the one cell that passed before was passing on the artefact.

No study was registered from the localized run: a placebo that fails everywhere is not something to serve.

NO_NEW_MEASUREMENT of any market claim. `execution_authorized: false` throughout.

## The first closure table (WP18 step 7 + WP13), and the first measured interval (WP07)

Full evidence: `../M5PHET_WP18_STEP7_20260925/`. One seal for every stage — the last 20 % of the household DEV slice,
population = the forecast origins whose whole 197-row window and whole 60-row horizon lie inside it, 9,824 rows, seal
`33820b552ddf`, protocol `d0ebd9a4bc75`, labels `REALISED_OUTCOME`; same epochs, same patience, deterministic ops,
refits byte-identical.

| stage | MAE (kW, h+60) | naive `last_value` | skill | rank |
|---|---|---|---|---|
| baseline_hand (predictor's own household configuration, one block of seven columns, window 60) | 0.537108 | 0.599327 | 0.103813 | 1 |
| quantile_hand (WP07's quantile head, same seal) | 0.538670 | 0.599327 | 0.101208 | 2 |
| laya_chosen (window 197, two groups, both dense) | 0.557190 | 0.599327 | 0.070307 | 3 |
| searched (WP06 stage 5) | — | — | — | DOES NOT EXIST |

**Laya's configuration lost to the hand baseline.** Nothing was adjusted; literature value `NOT_CARRIED` because no
literature number has been read into a report on this machine.

WP07: the quantile bundle (`[0.05, 0.5, 0.95]`, pinball loss, monotone by construction) answers `interval` and its
**measured coverage on the sealed holdout is 0.8781 against a nominal 0.90** — slightly too narrow, misses asymmetric
(7.35 % below, 4.85 % above), mean width 2.455 kW. A level with no fitted pair is refused `CONFIDENCE_LEVEL_NOT_FITTED`;
point bundles keep refusing `interval` with the old wording. The bundle lives in its own directory and is NOT in the
owner's live workbench: two bundles serving the same target make every sentence that names only `target_variable`
ambiguous, and the provider correctly refuses that.

**Calibration is structurally blocked, and that is the finding:** 10 of 13 decision records linked, every group
`NO_NEW_MEASUREMENT` (23-29 short of 30) **and** `NO_BEST_RANKED_OPTION` — the rank-1 stage has no decision records
because a person wrote it. The plan's WP23 now requires every stage entering a closure table to carry decision records
for the same question set, or to say it cannot be used for calibration.

GPU note: the fits ran on omega's RTX 4070 (16 seconds each), not the preferred 5090, because the worker's only
TensorFlow environment cannot open its GPU libraries (`Cannot dlopen some GPU libraries`, CUDA-13 wheels beside cu12);
building a CUDA TF environment there while a ~900-item Laya measurement held that GPU was not worth it for a 16-second
fit. Nothing on the worker was modified.
