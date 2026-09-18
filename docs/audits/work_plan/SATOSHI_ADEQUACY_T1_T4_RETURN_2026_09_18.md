# Return: adequacy T1–T4 — corrected estimands and geometry, test exposure and identity, second cost pilot (budget-limited), independent evidence

Order: `docs/handoffs/MUSASHI_ADEQUACY_S_REVIEW_AND_T1_T4_2026_09_18.md`, over the review of `bfe2545`.
Executed without pausing. No reserve, financial scoring, RL training, GPU, architecture search or
warehouse maintenance. Prior designs, pilot records and costs preserved; S1–S3 statements corrected
beside the historical claims (12D). Execution status: **BUDGET_LIMITED** — the corrected full
factorial was not launched because its projection with headroom exceeds the remaining ceiling.

**PRE** (reviewer's reproducer at `bfe2545`, `docs/audits/evidence/d3_k5_20260917/T1_T2_PRE_POST.txt`):
`past_noise_only_old_error 3.0` vs `conditional_oracle_error 8.9e-16`; W4 train 790–1558 /
validation 1563–1659 vs W256 train 286–1054 / validation 1311–1407; CNN 2 layers (RF 7) at W4 vs 8
(RF 511) at W256. **POST** (reproducer unchanged): the module's oracle now gives `8.9e-16` for the
past-noise case; W4 and W256 share validation 1311–1407, test 1664–2047, cutoff 1054, purge 257;
the reproducer's `conv_dilations(w)` still shows the S1 variable-depth configuration, kept as
separate evidence — the factorial uses `conv_dilations_fixed()` (8 layers, RF 511) at every W.

## T1 — estimands and comparison geometry (tests first: 3 rules)

* **Conditional oracle** for `observed_increment`: `2cos(2π/P)·clean[t] − clean[t−1] − observed[t]`;
  residual = next noise only. Tested: current-only noise → error 0; future-only noise → error = that
  noise; zero noise → zero error; non-finite values refuse; label identity. The S1 oracle is kept as
  `oracle_old` in every cell's arrays (history), and the S1 adequacy criterion is not reused.
  Expected Gaussian noise-only MAE (σ√(2/π), declared generator) is reported apart from the realised
  oracle MAE; neither is a samplewise lower bound (pilot: 0.262 expected vs 0.2135 realised).
* **Shared decision rows**: for every W, model and L the validation rows (1311–1406) and test rows
  (1664–2047) are identical; purge = max W + h = 257 (separation for the largest allowed support);
  training histories are nested and end at one cutoff (1054), starting L rows before. The
  preparation asserts the geometry identity and refuses otherwise; the verifier re-checks it from
  the arrays' row ids. (L = 1 024 still does not fit n = 2 048 with W = 256: staged proposal.)
* **Fixed CNN graph across W**: dilations 1…128, receptive field 511 ≥ 256, causal padding,
  effective support = min(W, RF) reported; the same parameter count at every W (tested). The S1
  variable-depth configuration is retained as `causal_conv1d_variable_depth` — separate evidence,
  not in the factorial, no budget. Ridge's parameter count varies with W: declared limitation.
  No context-only causal claim across architectures is made.

## T2 — exposure, identity, diagnosis (tests first: 3 rules)

* **Exposure ledger corrected** in the design: the S3 cost pilot scored the test rows of three
  cells — a disclosed DEVELOPMENT diagnostic, not an untouched confirmation. New cost pilots have
  **no test access**: the test accessor raises `TestAccessDenied` while the pilot path succeeds;
  the runner refuses a pilot record that carries a test loss; pilot terminals publish validation
  metrics only (tested). Any proposed model/context is selected from inner validation only;
  test tables are descriptive.
* **Digest binding**: the consumed clean/observed arrays are re-digested with the bank's own
  scheme (`sha256({dtype, shape} + C bytes)`) and compared with the frozen metadata before
  anything else; one changed byte under unchanged metadata refuses before training (tested through
  `run_cell`; no `cell.json` written). Finite values and geometry validated before fitting.
* **Diagnosis**: stop reason (EARLY_STOPPING / UPDATE_BUDGET / EPOCH_BUDGET / CLOSED_FORM), restored
  checkpoint epoch, budget vs early-stopping declaration, deterministic prediction and
  **prediction parity after reload** recorded per cell; `OPTIMIZATION_FAILURE` only on hard
  evidence (non-finite loss, or no update / no weight change); a flat trend is the flag
  `NO_EARLY_PROGRESS`, never a failure verdict; every diagnosis carries the note "heuristic fit
  class, not a scientific adequacy verdict". Tests exercise the real learners and the production
  child path.

## T3 — cost pilot and the full governed diagnostic: BUDGET_LIMITED

Design v2 sealed (`T1_ADEQUACY_DESIGN_V2.json`, `ad14cc17…`; frozen write-once in the run root):
2 units × 3 tasks × 3 models × 4 contexts × 3 lengths = **216 cells, single model seed** (provisional
by declaration). Run `adequacy-v2` (`T3_ADEQUACY_V2_{REPORT,PLAN,VERIFY,DESIGN_FROZEN}.json`):
cost-pilot campaign registered before any child; **six** governed children at W = 4 and W = 256
(L = 768, 200-update ceiling, no test access): optimizer work separated from startup/evaluation
overhead (conv 0.048 / 0.074 s per update, LSTM 0.030 / 0.061, overhead 1.2–1.5 s per child,
ridge 0.15 s); reconciled; **verified** (arrays = record = parent = **live warehouse**, 6 units).
Projection (per-update cost interpolated in W, every NN cell at its full allowance, overhead per
child — a projection, not an exact need): **11 695 s; with 25 % headroom 14 619 s vs 14 355 s
remaining (14 400 − 45 spent) — does not fit by 264 s (1.8 %)**. As ordered: no dispatch, no cell
removed, `PLAN.json` with measured costs. CPU of this order: 45 s (the S3 pilot's 31 s recorded
separately). Trade-off proposal (`T3_ADEQUACY_V2_TRADEOFF.json`): A as sealed with a 14 664 s
ceiling (recommended: 14 700 s); B `max_epochs` 160 → 11 751 s with headroom (training-rule change,
re-seal before any outcome); D `max_epochs` 150 → 11 035 s. Nothing launched pending review.

## T4 — independent evidence and understandable results

`tools/df_adequacy_verify.py`: losses (model, baseline, conditional oracle) recomputed from the
stored arrays for every part the role holds; row ids checked against the frozen boundaries and the
shared decision-row identity; skill recomputed; parent report and **live** warehouse content
compared (labelled live vs retained); `verify` handles pilots without a test split. Result on
`adequacy-v2`: all verified, parent equal, warehouse equal (6/6). Task-wise curves across W and L
cannot be reported: no adequacy cell ran. 12D carries the corrections beside the historical
claims and the T results; 12C the updated matrix; 09_ADOPCION the paragraph. RL/weekly-business
design (12E) remains required and **not** executed.

**Diagnosis asked** (model inadequacy vs context limitation vs noise vs optimisation): not
separable yet — no factorial cell ran. Measured so far (descriptive, validation only): at a 200-update
ceiling both neural learners are UNDERFIT by the heuristic (one stopped by early stopping), ridge
skill 0.08 (W4) → 0.12 (W256) on the observed task, conditional-oracle MAE 0.2135 vs expected
0.262. None of this is an adequacy verdict.

## Closure

**Suites** (trading-stack, `crispdm-run`, CPU; `tests/test_d3_*.py tests/test_df_*.py tests/test_olap_*.py olap/store/tests`): **1290 passed, 6 skipped**, 669 s. Targeted: adequacy 16 · adequacy run 3. Skips: the store suite's
own skip; `systemd-run`-gated children present here. **Commits**: `987b4b3` (merge) · `b05a3a3`
(T1–T2) · `a5f1f2c` (T3 runner) · `e15ee7b` (design v2) · the pilot-verification commit · the closing
commit that names this return. Workers synced. Pending: the empty-envelope disposition (next
needed window).

Ending: **`ADEQUACY_MATCHED_CONTEXT_CORRECTED_ORACLE_REVIEW`** — execution **BUDGET_LIMITED**
(corrected design and pilots delivered; the 216-cell factorial awaits the ceiling/trade-off decision).
