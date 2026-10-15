# I6-DMO acceptance and CLI implementation contract

Design gates only. No assertion below is a passing test until evidence exists.
Implement unit tests before governing runtime changes. Use the portable
test-led/data-centric method because this checkout lacks its canonical method
document. Keep the new initiative's method state separate from historic campaigns.

| ID | Required falsification / positive control |
|---|---|
| DMO01 | Six named outputs are 2/3/4/5/6/12 h, with independent weights and one shared backbone. |
| DMO02 | 24-hour span gives 13 exact timestamps; t is present; t+1 perturbation changes no input. |
| DMO03 | Input spacing 2 allows exact h3/h5 targets; hourly origins are not thinned. |
| DMO04 | Per-head mask retains other valid heads; invalid NaN cannot enter loss. |
| DMO05 | Weekend, force-flat, Thursday entry versus management, DST and split boundaries use C1. |
| DMO06 | Future/outer data perturbation leaves preprocessing, early stopping and trained state unchanged. |
| DMO07 | Exactly one validation fit; weekly reports never call fit; two output families are separate units. |
| DMO08 | Each original-price MAE and naive use identical row IDs and physical h; independent manual recomputation agrees. |
| DMO09 | Unequal weekly counts aggregate via sums/counts; absent weeks/horizons are explicit. |
| DMO10 | Same-weight fit/inner monitoring arithmetic, patience, best-weight restore and save/load predictions agree. |
| DMO11 | Completed identity is reused; changed config/data/code cannot adopt it; interrupted units remain visible. |
| DMO12 | Outbox retry/restart duplicates no OLAP facts and never retrains; delivery closes only after exact readback. |
| DMO13 | TEST cannot be opened by short init/run/deploy; final freeze/provenance/pair receipts are required. |
| DMO14 | Masks and chronological TP/SL consumption distinguish reversed hit order; no current-price head filling. |
| DMO15 | Force-flat obtains an actual fill before cutoff; stop() invents none; one position per asset. |
| DMO16 | Trial corpus uses four calendar years and all annual validation origins; no silent sample/epoch shortcuts. |
| DMO17 | No new scientific GPU child on the preferred host while another heavy child holds its host lease. |
| DMO18 | TRAIN-only normalization/loss scales; 19-feature identity; STLFSI4 aliases and unavailable revisions rejected. |
| DMO19 | Status gives real task progress/age, measured ETA or UNMEASURED; stale/failed is not COMPLETE. |
| DMO20 | Static, quarterly, monthly, weekly have distinct identities; full TEST reads cannot influence selection. |

## Required scripts (to implement, not existing runnable commands)

Use package `tools.i6dmo` with small modules `contracts`, `data`, `model`,
`training`, `metrics`, `campaign`, `deployment`, `publication`, `reporting`.
Reuse existing helpers rather than a new distributed optimizer or warehouse.

Required public CLI, all commands supporting `--json` where output is structured:

```bash
python -m tools.i6dmo validate-design --design docs/i6n/direct_multi_output_static/EXPERIMENT.short.json
python -m tools.i6dmo preflight --design <design> --deployment <private-spec>
python -m tools.i6dmo init --design <design> --deployment <private-spec> --root <new-root>
python -m tools.i6dmo deploy --root <new-root> --deployment <private-spec>
python -m tools.i6dmo tick --root <new-root>
python -m tools.i6dmo worker --root <new-root> --host-role <role> --once
python -m tools.i6dmo status --root <new-root> --level campaign --json
python -m tools.i6dmo status --root <new-root> --level unit --unit <id> --json
python -m tools.i6dmo status --root <new-root> --level publication --json
python -m tools.i6dmo close --root <new-root>
```

Contract details:

- `validate-design` is pure: no fit, network, TEST, or credential read.
- `preflight` resolves pinned paths, event/availability clocks, real dataset
  spans, GPU visibility, host/slice memory and disk, units and stopped old
  timers; emits PRECHECK.json, not a fabricated successful launch.
- A private deployment spec maps role -> SSH alias/GPU UUID/data roots/venv/
  warehouse EnvironmentFile. Write neither its values nor secrets into Git.
- `init` resolves design to a sealed effective config and enumerates all
  expected work. No 2025 TEST array is materialized here.
- `deploy` installs bounded user services/timers with exclusive claims and
  reboot recovery; refuse old campaign resurrection. It is idempotent and
  reports actual service and child start. No interactive approval per stage.
- `tick` is finite, atomically advances state and schedules the next eligible
  stage. No waiting chat/agent or endless foreground polling.
- `worker --once` runs exactly one claimed subprocess and persists terminal,
  best checkpoint, timings, costs, per-head masks/counts and prediction lineage.
- `status` is a bounded read of generated state, not new training or a log tail.
- `close` checks complete expected populations, scientific and delivery gates,
  immutable result identities, and verified physical mirrors before closure.

Design long and strategy interfaces during this work, but do not fit the long
model until short closure. The top-level pipeline advances to long without
another chat approval. Hyperparameter search and cadence experiments are not
implicit stages of this fixed-model baseline.

## Traceability and closure

For each ID record code commit, test name, PRE/POST result and evidence digest
in TRACEABILITY.json. A separate reviewer runs temporal negative controls and
manual MAE/naive replay, with no repeated scientific training.

Publish `STATUS.json`, `PRECHECK.json`, `EFFECTIVE_CONFIG.json`, `RESULT.json`,
`PUBLICATION.json`, `TRACEABILITY.json`, `RESULTS_INDEX.md`, `RETURN.md` and
generated M1-M8 plot. Failures and exclusions are first-class metrics. Unit
progress may count epochs/batches; stage progress counts declared terminals.
Keep performance estimates separate from engineering completion percentages.


## Market-time pair addendum (order 50d20d01, ops scope)

The market-time campaign (`mode = MARKET_TIME_PAIR`) runs one finite graph and has **no TEST
stage**: `short.fit -> short.report -> long.fit -> long.report -> pair.eligibility ->
strategy.validation (only on a qualified pair) -> publication/mirror closure`.

| ID | Required falsification / positive control | Test |
|---|---|---|
| MTP01 | The graph is exactly those six units with the declared dependencies, both designs sealed at init, and no unit is TEST-open. | `test_market_time_graph_is_finite_and_has_no_test_unit` |
| MTP02 | The long family is fitted and reported even when the short family is scientifically NOT_ELIGIBLE. | `test_long_family_runs_even_when_short_is_not_eligible` |
| MTP03 | A disqualified pair closes `PAIR_NOT_ELIGIBLE` with reasons; the strategy never runs and burns no attempt. | `test_disqualified_pair_closes_with_a_typed_disposition_and_no_strategy_run` |
| MTP04 | A qualified pair runs the validation strategy. | `test_qualified_pair_runs_the_validation_strategy` |
| MTP05 | No TEST unit is enumerable (`TEST_NOT_ENUMERABLE_IN_MARKET_TIME_PAIR`) or claimable even if injected into the queue. | `test_no_test_unit_can_be_enumerated_or_claimed` |
| MTP06 | An elapsed-grid (DS2/ASOF) terminal is refused for a market-time identity even with matching shapes and horizons. | `test_an_elapsed_grid_terminal_is_never_adopted_by_shape_or_horizon_match` |
| MTP07 | A technical failure is a typed terminal with no retry and no false COMPLETE. | `test_a_technical_failure_is_a_typed_terminal_without_retries` |

All seven are in `tests/i6dmo/test_ops_market_time_pair.py`. PRE: collection error (the market-time
mode did not exist). POST: 7 passed; whole suite 326 passed.
