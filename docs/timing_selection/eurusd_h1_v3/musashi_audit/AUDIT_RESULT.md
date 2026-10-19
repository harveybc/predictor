# Independent Audit: Timing Selection v3

Auditor: Musashi. Date: 2026-10-19. Predictor object: commit
`53d2bfb79e5363b77cc04e654320f5b0ca7e5c3b`, manifest
`b4da1012c53fd21003ec36f1d0c9f0e79354db1661444013168f86592fc2cced`.
Strategy code: `53d3b4eff83a3241c05dc85eb05f5dbcdb4254a7`.

## Decision

REVISE_BEFORE_BUSINESS_USE. Retain v3 as retrospective lake-based proxy research.
Do not admit this manifest as business-ready observations or replace the current
15-input SAC campaign with it. The six variables remain useful candidates for a
corrected-source comparison; this audit does not prove they are useless.

This is a substantive source/population decision, not a demand for signatures,
new credentials, perfect broker metadata or another general literature review.
The admitted root is already available: the next step is a bounded rebuild on
it with the business mask. No approval from the owner is needed to make those
corrections within this selection task. No new RL experiment is authorized here.

## Findings

### P1 / V3-01: Classifier population does not enforce business hours

`heuristic-strategy/tools/short_horizon_timing_eval.py:161` accepts a row when its
48-bar union window and label are finite and its period is declared. No operational
calendar mask enters `common_rows`, the TRAIN fit, the monitor or classifier metrics.
Execution filters later applied in `strategy` do not remove those training examples.

Independent counts against the sealed adapter's OWN calendar, with the completed
origin bar / next-open decision convention:

| Period | Scored origins | Outside operational origin interval |
| --- | ---: | ---: |
| TRAIN 2020-2022 | 18612 | 3708 |
| INNER_A 2023H1 | 2440 | 512 |
| INNER_B 2023H2 | 2952 | 588 |
| 2024 | 6236 | 1264 |

Additionally, 307 TRAIN and104 2024 k=2 windows cross calendar segments. These
are separate overlapping diagnostics, not counts to add together. The endpoint
checks in PROBE_RESULT use bar labels, not a final prescribed availability mask.

Correction: one shared calendar/availability function must determine eligible
origins and label use before selection, classifier fitting, monitoring and metrics.
Preserve all causal historical inputs including the prior week; never invent
weekend bars. Count exclusions by reason and apply the identical population to
both sets and their baselines. Do not substitute a favorable year or drop bad
weeks. Maintain per-target eligibility, including usable future support; do not
confuse label confirmation time with the time an online observation becomes known.

### P1 / V3-02: Source-clock and gap artifacts remain in the deciding population

The source is still the incomplete lake, not the repaired admitted core. The
loader treats its label as UTC (`tools/timing_pivot_targets.py:101`). Our witness
reproduces the clock problem and strengthens it with a prospective NY-local
interpretation, without changing the source:

| Mapping | Matched bars | Exact bid-close matches | p95 absolute close difference |
| --- | ---: | ---: | ---: |
| Label already UTC | 29149 | 102 | 0.004650 |
| Fixed +4 h | 30152 | 17775 | 0.001490 |
| Fixed +5 h | 30107 | 12676 | 0.001670 |
| Candidate NY local -> UTC | 30236 | 28291 | 0.000190 |

The last mapping is strong supporting evidence, not a complete provenance/clock
certificate. Under that mapping,818 admitted bars in2023 are absent from the
lake. The elapsed-time column spans720 transitions that skip admitted observations
across the available span; the probe retains counts and dates. Genuine closures
and source gaps must not share an unexplained missingness signal.

Correction: rebuild directly on admitted UTC bar-start / +1h availability and its
accepted closure intervals. This avoids requiring a speculative repair of the
old lake first. Regenerate pool, fractals, targets and source-dependent stages;
input changes invalidate target-independent content too. Local-hour cosine is
not intrinsically wrong: name the zone and DST explicitly. Do not delete elapsed
hours blindly; distinguish real closures from unexpected missing bars.

### P2 / V3-03: The claimed replay copy omits the main replay payload

`tools/timing_selection.py:905` lists metadata and ledger files, but no model,
prediction archive, feature/target matrix or phase-2 warehouse. The32-file receipt
has zero model files and zero npz files. FROZEN references18 selected model files.
All six replay inputs listed in PROBE_RESULT are outside that checksum inventory.

The separate-worker receipt can be true for its32 files while failing to establish
an independently recoverable experiment. This is not evidence that the models
were lost: the originals exist and their18 selected hashes checked correctly.
Extend the inventory and physical copy now, including the evaluated models,
predictions, scalers, feature/label matrices, ledgers and analytic databases. Keep
large payloads off Git. No retraining is needed to repair the copy.

### P2 / V3-04: Improvement over15 is not improvement over a probability baseline

We authenticated retained selected-model hashes and predicted rows/labels. A
constant class prior fitted ONLY on TRAIN is
`[0.5719428326, 0.2126584999, 0.2153986675]`.

| 2024 ensemble probabilities | Log-loss, lower is better |
| --- | ---: |
| Constant TRAIN prior | 0.983350 |
| Six, m1 Conv1D | 0.995996 |
| Six, m2 GRU | 1.003718 |
| Six, m3 HGB | 0.990320 |

All12 evaluated set/arm/period ensembles have greater log-loss than this prior.
This does not refute their above-base-rate ranking PR-AUC or the relative neural
advantage of six versus15. It limits the phrase "classification usefulness is
established": probability quality has not beaten this cheap baseline.
Report the prior, directional PR baselines, calibration and no-trade separately.
These are development comparisons, not independent holdout significance claims.

### P2 / V3-05: FDR gate uses independence-based p-values on dependent rows

`tools/timing_selection.py:248` computes ordinary Kruskal-Wallis p-values, then
BH; `:646` uses that pass/fail to remove features from the selected union. Targets
overlap and rolling inputs are serially dependent. The separate HAC/PCMCI report
does not recalibrate these p-values. SciPy explicitly defines this Kruskal test
for independent samples. BH alone cannot repair miscalibrated input p-values.

This shows an unsupported FDR calibration, not that all six are false positives.
Use a declared dependence-aware screen on TRAIN (for example an appropriate
blocked null with adequate p-value resolution), then the same multiplicity rule.
Retain effect sizes, support and temporal stability. Do not invent a new hard
causal-link requirement or reject every NO_LINK feature.

Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html

### P2 / V3-06: Frozen evaluation does not reject a changed strategy contract

`heuristic-strategy/tools/short_horizon_timing_eval.py:629` validates the freeze's
self-hash and model bytes, but reads the current FEATURE_SETS strategy again.
Our temporary-copy negative probe changes ONLY cap4 to cap3 and preserves data,
freeze and model bytes. Reconfirm reaches model loading without refusing the
changed cap. The sentinel stops before inference and before writing any scores.

This is a demonstrated guard gap, not an assertion that Satoshi mixed the real
v3 artifacts; our real comparisons matched. Before v4, bind freeze/reconfirm to
the full evaluation-input identity including expert, strategy, population and
scalers. Refuse drift before loading a model, and test cap, expert and label
changes. Compare staged shard identities before reusing them.

### P3 / V3-07: First-day return counterexample, with no observed2024 impact

`heuristic-strategy/tools/short_horizon_timing_eval.py:498` starts daily returns
from the first daily close, whereas execution accounting starts from initial
capital. With100 initial and closes110/121, the helper returns0/10%, not10/10%.
Fix this helper and add a nonzero-first-day regression.

Crucially, all18 real2024 replays begin with first daily close10000, equal to
initial capital. Thus this defect does NOT change their actual2024 series. Do
not invalidate or rerun all fits because of this edge case. Recompute affected
derived uncertainty/DSR only when a period's first-day equity differs from initial.

### P3 / V3-08: Evidence scope and comparison interpretation

The block named t1_t3_all_cuts actually skips T3 with check_t2=False. We closed
the factual coverage gap by replaying T1/T2/T3 at ALL five effective pre2025 cuts:
no declared-final row changed. Correct the name/coverage report, not the labels.

The m1 model also receives four shared origin time encodings (`build_dmo_class`,
`:250`), in addition to the named feature set, and capacity varies with feature
count. Describe the result as two feature sets under a shared recipe/context,
not a pure six-input experiment with matched parameter counts. No extra fit is
required just to correct that wording.

## Confirmed Strengths And Limits

- All24 published checksum entries match; canonical manifest hash matches.
-20 predictor tests pass, including the real v2 expert finality tests.
-19 strategy tests pass;3 tiny-fit cases deliberately excluded from this audit.
- Independent finality replay covers all5 effective cuts, all3 targets, truncation
  and future perturbation. This supports those cases, not every possible future.
-18 real2024 strategy replays from retained predictions match net profit within
  1.14e-13 USD, Sharpe within4.45e-16 and entries exactly. No model fit or GPU use.
-18 selected model file hashes and common evaluated rows/labels checked. Neural
  forward passes were not rerun; do not claim a fresh weight-to-prediction replay.
- No additional input repainting was demonstrated by these tests. Real UTC
  availability and business eligibility remain subject to V3-01/02.
- Mixed CI is correctly labelled predictive association, with neutral NO_LINK.
  Its50 full records include two history members. Halves cover48 non-history
  candidates plus three controls, explaining the51 denominator. We inspected
  its code/evidence, did not rerun PCMCI.
-2023 is joint development and2024 previously observed. Keep those roles and
  dates. Weak trading does not prove that no useful RL feature set exists.

## Minimal Next Work

See SATOSHI_CORRECTION_ORDER.md. No owner intervention is needed for the listed
available inputs. Do not optimize TP/SL, change expert parameters or launch new
RL merely to make this audit pass. Existing active SAC regularization continues.

Evidence: PROBE_RESULT.json and probe_v3.py beside this file. The final probe was
bounded to1GiB RAM, one CPU and180s; actual peak350.1MiB and5.8s. The preceding
graph index was not memory-bounded and caused the separately recorded coordinator OOM;
that operational failure belongs to this auditor, not to Satoshi's selection.

Audit persistence:80 measurements in one governed warehouse report,1/1 delivered
with readback, zero pending/mismatch. The closed SQLite backup was copied to the
second physical machine and compared by content checksum. See PERSISTENCE.json.
