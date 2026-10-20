# TIMING_PROXY_SELECTION v4 on the admitted source: result and correction table

Answers Musashi's v3 decision REVISE_BEFORE_BUSINESS_USE (predictor `codex/timing-selection-v3-audit-20261019`
@f342d0d5, `SATOSHI_CORRECTION_ORDER.md` items 1-8). v4 has a NEW identity and output directory; v3 is
kept as retrospective lake-proxy evidence (`docs/timing_selection/eurusd_h1_v3/RETROSPECTIVE_SCOPE.json`).
This is a scoped return for review of the changed findings.

## Result

v4 rebuilds the selection on the admitted EURUSD H1 source. Bars are UTC bar starts with availability at
start + 1 h; there are 0 missing required operating hours. Before any selection, fitting or scoring, one
shared eligibility mask removes every origin that the business rules cannot trade:

- an origin less than 12 h after the session reopens;
- a label window that ends within 12 h of the close;
- a window that spans a closure.

The mask comes from the admitted closure intervals. It removes 3,925 of 18,698 TRAIN origins, 650 of 3,111
(2023H1), 650 of 3,086 (2023H2) and 1,350 of 6,221 (2024), counted by reason in `TARGETS.json:eligibility`.

On the remaining TRAIN rows, a dependence-aware screen gates the MRMR K=12 union of the T1 task: a
Kruskal-Wallis statistic against a blocked circular-shift null, 4,999 shifts of at least 168 rows, then BH
at q = 0.1. 34 of 93 candidates pass; the iid screen passed 44. **Four inputs are selected:**
RAW_BAR_LOG_RETURN, cal.utc_session_us, hurst_spectrum_width_w200, px.close_loc_2.

Classification:

- **Ranking is above chance.** On identical eligible rows (INNER_B 2,436, 2024 4,871), per-direction
  PR-AUC is 0.35–0.41 against base rates of 0.20–0.22, for the four as for the admitted-core 15 and the
  v3 six.
- **Probability quality does not beat the constant TRAIN-prior classifier.** Every neural ensemble's
  log-loss is 0.011–0.045 worse than the prior. Gradient boosting matches it (−0.0048 to +0.0032 in
  INNER_B/2024, and up to +0.027 in INNER_A, where it is chosen).
- **Calibration:** ECE is 0.077–0.120.
- **The four versus the 15.** Against the 15, the four have lower log-loss in 2024 on m1
  (−0.0082 [−0.0169, −0.0001]) and m2 (−0.0117 [−0.0206, −0.0034]). They are indistinguishable in INNER_B
  and on m3.

Strategy (executable fill at b+1, 4/day cap, TP / SL 2σ): no consistent profit. Sharpe ranges from −3.2
to +0.7 across sets, arms and periods. Every paired Sharpe-difference interval includes zero, and DSR is at
most 0.042.

These are development comparisons (2023 joint development, 2024 previously observed), not independent
holdout claims. They describe two feature sets, plus the v3 six as a comparison candidate, under a shared
recipe and context. m1 also receives four shared time encodings, and model capacity scales with the
feature count. The negative financial result is reported as such.

## Identities

| Object | Identity |
|---|---|
| Admitted source / receipt / closures | `eurusd.parquet` c80847bb…, ADMISSION.json ee426b52…, CLOSURE_INTERVALS.json 71078e8f… |
| Bar-source adapter (clock lane) | heuristic-strategy `satoshi/clock-identity-20261019` @ce9a3dc, `bar_source.py` fab22a4d…, `clock_identity.py` 0bd00052…, `data.py` 28d50d25… |
| Expert | ZigZag v2 (abbfaa6) Depth 3 / Deviation 40.7 / Backstep 0, module fc24c6f2…, confirmation `confirmation_bars_fast` |
| Code that ran (byte-identical on the worker) | predictor 7ef9cb26, heuristic-strategy 1e0893c, causal-inference a379a67 |
| Manifest | TIMING_PROXY_SELECTION `ca5d75c28b6da80e2c832546445aca989e44b6f37e9b57b91276b54b2e7e6163`, status `TIMING_PROXY_SELECTION_SEALED_PENDING_INDEPENDENT_DECISION` |
| Eligibility digest / evaluation-input identity / freeze | `19835c884ed594e6…` / `e251de5de2d8d0b7…` / `ea8919e71cc868e8…` |
| Replay payload | 320 files, 515 MB (135 models, 46 npz, 62 parquet, 1 database); full copy on the secondary worker and on the coordinator, `sha256sum -c` ALL_OK on both |

## Finding → correction → executable test / evidence → result

| Finding | Correction | Test / evidence | Result |
|---|---|---|---|
| V3-01 population ignored business hours | One eligibility function (`tools/timing_eligibility.py`): 12 h after reopen, 12 h before close, k-window inside one session, per-target label support. Applied to every target before any stage; the evaluator consumes the mask by digest and refuses a finite label on an ineligible origin | `tools/test_timing_eligibility.py` (7: both DST regimes, weekend window, short holiday session, label usability, inside-closure, digest); `test_admitted_source_end_to_end…`; TE13, TE14; counts by reason × period | CLOSED: identical eligible population for every set, arm and baseline |
| V3-02 lake clock and gaps | Rebuilt on the admitted UTC source through the clock lane's `bar_source` (pins, receipt binding, no bypass); calendar features named by zone (`cal.utc_*`, `cal.london_hod_*` with DST); closures separated from unexpected gaps (`px.hours_since_session_open`, `px.unexpected_gap_hours`); pool, fractals (plugin, windows 48/96/200), labels, quality, pairs and selection regenerated | `test_utc_clock_names_zones…`; the admitted end-to-end test incl. a refused receipt; `PERTURBATION.json` + `PERTURBATION_FRACTAL_V4.json` all passed | CLOSED for this source; the inventory's `hod_*` / `dow_*` are UTC encodings of the availability instant (the admitted-core bundle's convention) |
| V3-03 backup omitted the replay payload | Snapshots now carry every file of the run; v3 backup repaired without fits | v3: 242 files / 450 MB, ALL_OK on the worker; v4: 320 files / 515 MB, ALL_OK on the worker and on the coordinator | CLOSED |
| V3-04 no probability baseline | Constant TRAIN-prior classifier, per-direction PR base rates, Brier, ECE and no-trade on the same rows | TE15; `eval/classification_baselines.csv`; `summary.csv` | REPORTED: ranking above chance; log-loss not better than the prior |
| V3-05 FDR on dependent rows | Blocked circular-shift permutation null (4,999 shifts ≥ 168 rows, resolution 2e-4), then the same BH and recipe; effect ε², support, fold stability | `test_blocked_permutation_screen…` (size on autocorrelated noise, power on a planted signal); `REPORT.json:screen_gate` | CLOSED: 34 / 93 pass (iid: 44) |
| V3-06 freeze did not bind the strategy | Freeze and reconfirm bound to the full evaluation-input identity plus per-shard population and scaler; drift refused before any model load | TE12 (cap, expert, label; reconfirm sentinel) | CLOSED; v4 reconfirm passed the check |
| V3-07 first-day return | Returns measured from initial capital | TE11 | CLOSED |
| V3-08 naming / coverage and wording | Finality on T1/T2/T3 at all cuts (`t1_t2_t3_all_cuts`); legacy blocks renamed honestly; "two feature sets under a shared recipe and context" in the seal | `TARGETS.json:finality` (all declared-final rows unchanged); `eval/SEAL.json:comparison` | CLOSED |
| Publication | Local SQLite idempotent readback | `OLAP_PUBLICATION.json` (inserted 1/93/1,892, then 0; exact) | Local DONE; central warehouse NOT_REACHABLE (no endpoint or token configured) |

## Limits

- The single 2024 read is development reconfirmation.
- Only 2023H1 chooses; the regularization rule replaced the primary cell in most shards (`eval/FROZEN.json`).
- The m3 seeds are identical.
- TP / SL was not optimized, by order.
- Inside the admitted gate the elapsed-hours family carries little signal. `RAW_ELAPSED_HOURS` and
  `px.unexpected_gap_hours` were in the union and failed the screen.
- The causal stage's halves agree on 20 / 36 verdicts.
