# I6-N audit addendum: strategy lane (I6N-AUD-02, owner point 6b)

Date: 2026-10-11. This addendum covers only the strategy part of the audit.
It does not repeat the full audit, and the original verdict
REVISE_BEFORE_IMPLEMENTATION stands. No file in the parent audit directory was
modified, no model was trained, and no service, timer, GPU, broker or campaign
was touched. No market data was read. Every price below is synthetic.

All compute ran on the coordinator, on CPU (`CUDA_VISIBLE_DEVICES=""`), one
job at a time under `crispdm-run -q -m 1G -t 10m -n i6n-add-strategy`. The
two worker roles (worker_a, worker_b) were not inspected (see Open items).
`$GH` is the parent directory of the sibling checkouts and `$CONDA_BASE` is
the conda base prefix.

## How to reproduce

`bash run_strategy_addendum.sh` regenerates every `*.out` in this directory.
The last complete run was 2026-10-11 12:16:04Z to 12:16:24Z, with exit 0.
`MANIFEST.json` gives the hash of every input, script and output.

## 1. Reconciling `probes/a/strategy_counterexample.out` (sha256 d9e1399f...)

Cause, from the session transcript of the original sub-auditor (timestamps
UTC) and from the scratch-copy mtimes (scratch copies are byte-identical to
the retained files):

1. At 11:46:06 revision 0 of the script failed at import with a dataclass
   `sys.modules` error, so it produced no output.
2. At 11:46:12 the script was patched with `sed` (revision 1) and run as
   `... strategy_counterexample.py ... 2>&1 | tee strategy_counterexample.out`.
   That run produced the full 20 lines. Its last line was
   `calculate_entry_geometry signature has no short argument: False`.
3. At 11:46:20 the last print was replaced (revision 2, the file now retained
   as `probes/a/strategy_counterexample.py`, sha256 15970757...). The script
   was run again as `... 2>&1 | tail -4 | tee -a strategy_counterexample.out`.
   This **appended** that run's last four lines.

The retained `.out` is therefore one full revision-1 run (lines 1-20)
followed by the last four lines of a revision-2 run (lines 21-24). The
apparent "repeated blocks" are lines 17-19 printed a second time. Line 20
comes from revision 1, which no longer exists. `reconcile_original_out.out`
proves this mechanically:

- lines 1-19 of the retained `.out` equal lines 1-19 of a clean rerun;
- lines 21-24 equal lines 17-20 of the clean rerun;
- the retained `.py` contains 0 occurrences of the line-20 text.

The revision-1 line was also **wrong as a check**. It tested whether the
substring "short" appeared in `co_varnames`, which includes local variables,
and `calculate_entry_geometry` has a local named `short_signal`. So it printed
False even though the function takes no short-family argument (the parameters
are `current_price`, `long_horizon_predictions`, `params`).

One more reproducibility gap: the retained stub is named
`stub_trading_contracts.py`, but the probe imports `trading_contracts`. The
clean rerun therefore restores it under its import name as
`stub/trading_contracts.py` (byte-identical, sha256 9e208a30...). It runs with
the interpreter of the original run (conda base, Python 3.12.7). The result
is `original_probe_clean_rerun.out` (20 lines, sha256 be7eb63d...), which
matches exactly what the retained script prints. Every substantive result in
the original output reproduces.

## 2. Inspected source compared with the code an executor would load

### (i) Inspected source

Source: `inspect_sources_and_executors.out`.

- **Checkout state.** The heuristic-strategy checkout is at HEAD `79977b2` on
  branch `docs/agent-onboarding-20260816`, which is an ancestor of
  origin/master `5c87a25`. The original audit wrote "checkout = origin/master
  5c87a25"; that is not the checked-out commit.
- **The code is still the same.** Both files are byte-identical at `79977b2`,
  at master `a213b65` and at origin/master `5c87a25`:
  - policy `app/policies/prediction_entry_exit.py`: blob `0940589`, sha256
    `16394e2f...`;
  - plugin `app/plugins/plugin_long_short_predictions.py`: blob `3a19b3f`,
    sha256 `1a835fee...`.
  
  Both files are clean in the worktree. The policy blob is the same at all 18
  local and remote-tracking refs. Two other plugin blobs exist only on
  research branches: `4e3618e3` (strategy-support, continuous-synthetic) and
  `7802dd3f` (s07).

### (ii) Effective resolution per environment

Source: `resolve_effective_plugin.out`. Each case ran under `python -I`, with
an optional repository root prepended to `sys.path`, as `python -m` or
PYTHONPATH from that root would do.

| Interpreter | sys.path root | `heuristic_strategy.plugins:ls_pred_strategy` resolves to |
| --- | --- | --- |
| conda base 3.12.7 | none, or the lts root | none: the **2025-03-14 site-packages copy** `$CONDA_BASE/.../app/plugins/plugin_long_short_predictions.py`, sha256 `3e1ebec1...`, installed from an older separate clone, with no `app/policies` (lts root: ImportError `app.plugins`) |
| conda base | heuristic-strategy checkout | ImportError `trading_contracts`: the inspected code cannot load in this env |
| conda base | predictor I6-N worktree `269c2d13` | ImportError: predictor's own `app` package shadows it |
| trading-stack 3.12.13 (env of the lts/agent-multi user units inspected) | none / predictor / lts | no entry point; `plugins_strategy.heuristic_strategy` resolves to the editable lts copy (sha256 `954318e8...`, lts `12bce5f`, blob `b8aee0dd`) |
| trading-stack | heuristic-strategy checkout | the **inspected** plugin `1a835fee...`, whose `should_early_close` comes from the inspected policy `16394e2f...` |
| venvs feature-eng, feature-extractor, store-hosts | any | nothing loadable (no entry point, or ImportError `backtrader`/`trading_contracts`) |

### Which executor would consume I6-N predictions

None today. The I6-N plan is an uncommitted draft and names no strategy
executor and no pin. The places that could load a strategy, as found by
`inspect_sources_and_executors.out` and the resolver:

- **predictor (audited worktree `269c2d13`).**
  - `app/config.py` sets `use_strategy: False` and names
    `heuristic_strategy.plugins:ls_pred_strategy`, with no version.
  - The only loader is in `app/data_processor copy.py`, which nothing imports.
  - `app/optimizer_ga.py` imports a top-level `heuristic_strategy` module,
    which no environment provides, and nothing imports `optimizer_ga`.
- **lts.**
  - The copied logic is used only by `app/heartbeat.py`.
  - `start_heartbeat()` has no caller outside tests, and no user unit refers to
    heuristic code.
  - `heuristic_strategy` is not in lts's `plugins_strategy` entry-point group.
- **agent-multi and gym-fx.** Their `strategy.plugins` (`default_strategy`,
  `direct_*_sltp`, `shared_execution_envelope`) are execution overlays, not
  this forecast strategy.

So the defects below sit in the **intended** plugin (the inspected blobs),
not in a running pipeline. Which code an eventual executor loads would depend
on the interpreter and on `sys.path`. On this host the same entry-point name
gives the legacy copy, the inspected copy or an ImportError.

## 3. Counterexample v2

Script: `counterexample_v2.py`, which takes the module path as a parameter.
The inspected plugin's own `HeuristicStrategy.next()` runs unmodified on a
stub that replaces only backtrader's clock, close line, broker and order
calls.

**Construction**, checked in section [A] of every output:

- Long position: entry 1.10000, TP 1.10100, SL 1.09800.
- Short position: entry 1.10000, TP 1.09900, SL 1.10200.
- The long family is flat and never crosses either barrier.
- S1 (long position), short path in hours:
  2h=1.10120 (≥TP), 3h=1.10060, 4h=1.10040, 5h=1.10020, 6h=1.10000,
  12h=1.09750 (≤SL).
- S2 is S1 in reverse time order.

The script asserts that S1 has its first TP crossing at 2 h and its first SL
crossing at 12 h, that S2 has the reverse, and that S2 is a permutation of
S1. It applies the same checks to the short-position mirror.

Owner-expected decisions under request 3.1: S1 means no early close, and S2
means early close.

| Check | Inspected policy | Inspected plugin (`next()`) | Base site-packages copy | lts copy |
| --- | --- | --- | --- | --- |
| [B] S1 vs S2, buy and sell, every variant | identical for A-G, conforms for none | identical for A-G, conforms for none | identical, does not conform | identical for A-G, conforms for none |
| [B] TP used by prediction exit | no TP argument | compares only with `current_sl` | only `current_sl` | no TP argument |
| [C] 200 within-family permutations | one decision per variant | one decision per variant | one decision | one decision per variant |
| [D1] short family changed, long fixed | entry unchanged (conforms) | unchanged (conforms) | **changes** (TP 1.10072 to 1.10540, SL 1.09980 to 1.08800) | unchanged (conforms) |
| [D2] one finite long, five NaN | enters long | enters long | enters long | NaN-first: hold; NaN-last: enters long (**order-dependent**) |
| [D3] longs permuted | unchanged (conforms: 3.1 entry uses min/max) | unchanged | unchanged | unchanged |
| [E] 12 horizon keys | `_path()` sorts lexicographically: ranks `[6,7,1,8,2,9,3,10,4,5,11,12]` (hour keys) and `[1,10,11,12,2,...]` (index keys) | integer loop gives numeric **position** order; columns renamed positionally, no hours | same loop and positional rename | plain lists, order is the caller's |
| [F] missing head | NaN dropped by `_finite_path`; filled current price vs NaN changes variant E (close / hold) | **absent** column gets the current price (1.09850) and the last column is never read; E: close vs NaN: hold | NaN in the first slot hides the whole min (close=False vs True) | NaN is order-dependent (A, C); `None` raises TypeError |
| [G] `horizon_hours` / `target_timestamp` / `eligible_until` in source | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

## 4. Classification for each sub-claim of I6N-AUD-02

Classes: DEMONSTRATED_DEFECT, RISK, OWNER_REQUIREMENT,
METHODOLOGICAL_ALTERNATIVE, RETRACTED. The owner requirements come from
request §3.1 (sha256 `2e060f7d...`) and BUSINESS_MARKET_WEEK_CONTRACT (MWC,
sha256 `604f9480...`).

| # | Original sub-claim | Verdict | Class | Evidence | Minimal correction |
| --- | --- | --- | --- | --- | --- |
| S1 | Early close uses min/max without time order and never looks at TP | **Confirmed** | DEMONSTRATED_DEFECT against OWNER_REQUIREMENT 3.1 (chronological SL-before-TP) | `cx_v2_hs_policy_checkout.out` and `cx_v2_hs_plugin_checkout.out` [A][B][C]: S1 and S2 give the same decision for all 7 variants, buy and sell; no TP argument | One chronological first-crossing function over both families, ordered by `horizon_hours`, with declared rules for equality, ties, no crossing and missing heads. Acceptance: S1 holds and S2 closes, buy and sell |
| S2 | "heuristic-strategy checkout = origin/master 5c87a25" | **Retracted as worded** (the substance stands) | RETRACTED | `inspect_sources_and_executors.out`: HEAD `79977b2`; blobs `0940589` and `3a19b3f` identical at `79977b2`, `a213b65` and `5c87a25` | Cite the code by blob id and sha256, not by branch |
| S3 | No pin anywhere in predictor or lts | **Confirmed, qualified**: no executor exists yet; resolution depends on interpreter and `sys.path` | OWNER_REQUIREMENT (3.1: identify the pin of the plugin destined for production) | `resolve_effective_plugin.out`: the same entry point gives the legacy copy, the inspected copy or an ImportError; predictor `use_strategy: False` with no version | The strategy node pins the blob or sha256, loads by explicit path or a versioned dist in a declared env, and records the module sha256 in run identity |
| S4 | Installed site-packages copy differs and uses short+long for entry | **Confirmed as fact, qualified as to relevance** | DEMONSTRATED_DEFECT on that copy; RISK for I6-N (no I6-N executor loads it; it is what base-env resolution returns without the checkout on `sys.path`) | `cx_v2_legacy_plugin_base_site_packages.out` [D1]; resolver rows for base | Exclude the base-env install from any I6-N run (pin, as in S3); do not reinstall or delete it under this order |
| S5 | Positional `Prediction_h_{i+1}` mapping; no index→hours→timestamp map | **Confirmed** | DEMONSTRATED_DEFECT against OWNER_REQUIREMENT 3.1 (map required and persisted) | [E] and [G]: 0 occurrences of the map vocabulary; positional rename | The adapter emits and consumes `horizon_hours`, `target_timestamp`, `eligible_until`, `available` and `reason` for each head, and persists them with config and model |
| S6 | Lexicographic key sort breaks at 10 or more horizons (policy:214) | **Qualified** | RISK: the ordering defect is demonstrated, but today it changes no decision, because every consumer is permutation-invariant ([C]) and no I6-N executor uses the `trading_contracts` policy route | `cx_v2_hs_policy_checkout.out` [E] | Order by declared `horizon_hours`, never by key text; the acceptance test is in [E] |
| S7 | Missing column filled with the current price (plugin:302-305) | **Qualified** | DEMONSTRATED_DEFECT against OWNER_REQUIREMENT MWC:127 on the direct `pred_file` route; RISK on the `evaluate_candidate` route, whose positional rename makes columns contiguous, so a missing upstream head shifts positions silently instead | `cx_v2_hs_plugin_checkout.out` [F]: absent `Prediction_h_2` becomes 1.09850, `_6` is never read, and variant E closes where the NaN case holds | No default value: a missing head is unavailable, carries a reason code and is excluded |
| S8 | No availability masks | **Qualified** | DEMONSTRATED_DEFECT for the explicit mask contract (MWC: the adapter must consume masks). The inspected policy does drop non-finite values (`_finite_path`), an implicit mask; the legacy and lts copies do not (NaN is order-dependent, `None` raises) | [F], [G], [D2] | Consume the explicit mask; a NaN or `None` never reaches min/max |
| S9 | No two-long abstention | **Retracted as a defect** | RETRACTED. The two-long minimum is an owner-proposed rule to audit (MWC:150-151, request:207), not yet a requirement. The fact stays as a RISK: every copy enters with one finite long ([D2]) | [D2] in all four outputs | The owner decides the rule; until then the strategy node must declare its partial-availability rule explicitly |
| S10 | Entry uses only the long family | **Conforms** in the inspected policy, the inspected plugin and the lts copy; violated only by the legacy copy (S4) | (no defect on the inspected code) | [D1] | None for the inspected code |
| S11 | Entry is order-blind across long horizons (original probe line) | **Not a defect**: 3.1 entry is defined by the min and max of the longs | RETRACTED (any reading of it as a defect) | [D3] | None |
| S12 | No calendar in the plugin | **Qualified** | OWNER_REQUIREMENT that lands on the common environment or executor (MWC §4: liquidation "debe vivir en el componente comun del entorno/ejecutor"), not on the plugin; no such executor exists yet | MWC §4; the plugin has no calendar code | Put calendar liquidation and the Thursday entry cutoff in the executor, and keep the plugin calendar-agnostic |
| S13 | `strategy_counterexample.out` reconciles with the retained script | **Corrected** (evidence defect) | RETRACTED (the retained output as evidence of the retained script) | `reconcile_original_out.out`, `original_probe_clean_rerun.out` | Use the clean rerun and v2 outputs |

**Effect on the verdict:** none. The core of I6N-AUD-02 holds on the inspected
blobs: early close is not chronological and there is no horizon map, mask
contract or pin. Severity stays CRITICAL for any use of the U_h strategy, and
REVISE_BEFORE_IMPLEMENTATION stands. This addendum does not order the repair
itself. It states the minimal correction and the acceptance tests above.

## Open items

- **Worker roles not inspected.** worker_a and worker_b were not inspected:
  the standing rule forbids sweeps over ssh. Their interpreters' resolution of
  the strategy is UNVERIFIED. No I6-N executor exists there either, since the
  plan is a draft.
- **Tie rules not owner-ratified.** Exact-equality, tie and no-crossing rules
  for the chronological rule are still the owner's to ratify. S1 and S2 avoid
  them on purpose.
- **Short-lived processes outside `crispdm-run`.** A few read-only Python
  one-liners (a transcript parse, a version print, the `sed`-style edits of
  this addendum's own scripts) ran outside `crispdm-run`. Every retained
  Python output was produced under `crispdm-run` by
  `run_strategy_addendum.sh`. The two bash steps
  (`inspect_sources_and_executors.sh`, `reconcile_original_out.sh`) use only
  git, sha256sum, grep and diff.
