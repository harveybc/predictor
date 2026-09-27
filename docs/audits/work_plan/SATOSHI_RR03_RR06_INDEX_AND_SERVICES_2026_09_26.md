# RR03 / RR06 — the dispatch index reconciled, and the services the reboot actually restarted

**Date:** 2026-09-26
**Author:** Satoshi III (Mujuro Utsutsu), successor technical lead
**Orders:** RR03 and RR06 of `docs/handoffs/SATOSHI_POST_REBOOT_ORDERS_2026_09_26.md` (`0d6f6f7e`)
**Continues:** the DR02 index at `b790cd12`, against the RR01 restart manifest at `0ee09998` (§4 tips, §5 per-lane state)
**Verdict for the whole document:** `NO_NEW_MEASUREMENT`. Nothing here fits a model, and no number in it is new science.

---

## 0. The two headlines

**Nine of twelve index rows changed.** Eight of the eleven DR02 rows moved and one lane that had no row got one;
`FIN-LOSS-OPT/E3`, `calendar` and `news` are byte-identical to `b790cd12` because no artifact of theirs moved.
Every moved row carries its previous `observed_state` and `observed_at` unchanged in a new `observation_history`
field. Nothing was replaced and no second scheduler was built.

**What the two restarted services are actually running.** Both came up with the machine at **17:41:04 local**,
eight seconds after boot, with `NRestarts 0`:

| service | runs from | identity of that code | start |
|---|---|---|---|
| `lts-alpaca-model-runner.service` | the **primary live checkout** (`WorkingDirectory`) | HEAD `12bce5f2c4fe…` on the MT5-unknown-outcome branch, which the checkout moved to at **06:15:11** and committed at **06:32:00** — hours *before* the process started | 2026-09-26T22:41:04Z |
| `m5phet-chat.service` | **no checkout at all**: a non-editable pip copy in its own venv | `m5phet 0.1.0` installed **2026-09-25 13:59:02** from the chat worktree at `f532ee5`, which is **not** M5PHET master `8a6d1a3` | 2026-09-26T22:41:04Z |

---

## 1. RR06 first, because it is what expired

### 1.1 My claim, withdrawn

I had said the Alpaca model runner still held pre-checkout-change code resident because it had been running for
68 hours. **That is false as of 17:40:56 local and is withdrawn.** The process that held the old code no longer
exists. The current process started at **17:41:03/04**, after the checkout had already moved and committed, so it
imported the *new* code.

I checked this rather than assumed it. A static transitive import closure from `app.alpaca_model_runner` over the
checkout's own packages is **10 modules**, and `git hash-object` of every one of them equals
`rev-parse HEAD:<path>` — `all_match_HEAD: true`, zero uncommitted files under `app/` or `examples/configs/`, and
the newest mtime anywhere in `app/` is 06:28:22, before the process started. The closure contains
**`app/broker_refusal.py`**, which did not exist at `9090f49`. So the changed broker path is resident, which is
exactly what the expired claim denied.

**What does not change:** the F3 finding itself stands and is now sharper. The unit carries **no commit pin** —
`WorkingDirectory` points at the mutable checkout and `Restart=always` — so a restart adopts whatever branch that
checkout happens to be on. The reboot did not fix that defect; it *demonstrated* it.

### 1.2 The two services, as asked: checkout, module and package identities, start time

**`lts-alpaca-model-runner.service`** — start `2026-09-26T22:41:04Z` (`ExecMainStartTimestamp`; `ps lstart`
17:41:03), `NRestarts 0`. Runtime checkout `/home/harveybc/Documents/GitHub/lts` read from `/proc/<pid>/cwd`, the
primary live checkout, HEAD `12bce5f2c4fe2ab0e6a7433d7208d9e05a9cffaa`. Interpreter the trading-stack Python
3.12 from `/proc/<pid>/exe`. Entrypoint `python -m app.alpaca_model_runner --config …alpaca_spy_model_runner_v1.json`.
Resident extension modules from site-packages: `pydantic_core`, `charset_normalizer` and one mypyc-compiled module —
**no numpy, no pandas**; 41.5 MB current, 42.6 MB peak. One non-socket descriptor: the model-execution SQLite store.
Two of its own dependencies are **editable installs in mutable checkouts** — `prediction_provider_mechanics` 0.1.0
at revision `316c315` and `trading_contracts` 0.1.0 at `3d531f6`, both with zero tracked files differing from HEAD —
plus `requests` 2.34.2, `pydantic` 2.13.4, `certifi` 2026.6.17.

**`m5phet-chat.service`** — start `2026-09-26T22:41:04Z`, `NRestarts 0`. **It does not run from a checkout.** Its
working directory is the home directory and its code is a non-editable pip copy inside its own venv; there is no
editable `.pth` hook in that venv. Its provenance is therefore a wheel's `direct_url.json`: `m5phet 0.1.0`
installed 2026-09-25 13:59:02 from the chat worktree, whose HEAD is `f532ee5` (2026-09-25 13:59:00). **The running
chat is not master** — master is `8a6d1a3` of 2026-09-26 — and it will keep running the 2026-09-25 snapshot across
restarts until the venv is reinstalled. That is the *opposite* failure mode to the Alpaca runner's, and both are
now written into the index. Nine provider wheels were installed at nine different times, one of them
(`news_signal` 0.2.0) at 2026-09-26 04:00:39, *after* the app snapshot, so the venv is not a single coherent point
in time and must not be described as one. Resident scientific stack, read from the live process's mappings and not
merely from the installed list: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, pyarrow 25.0.1,
statsmodels 0.15.0, pydantic 2.13.5, fastapi 0.141.1, starlette 1.7.0, uvicorn 0.53.0; 255.7 MB current,
303.3 MB peak.

**Method, and its limits.** Read-only throughout: unit properties, `/proc/<pid>/{cwd,exe,cmdline,stat,maps,fd}`,
`dist-info` and `direct_url.json`, the import closure against committed blobs, and the lanes' own retained stores.
**No process environment was read, no `EnvironmentFile` was opened, no credential was read or printed, and no
broker call of any kind was made.** The closure is an *import* closure: anything reached by subprocess is outside
it — the watchdog, for instance, imports only the standard library and calls out by subprocess.

### 1.3 The no-new-order / read-only configuration: **partially in force, and the index was wrong**

The DR02 row said the Alpaca lane was "LIVE, READ-ONLY SINCE 2026-09-25 16:03Z". That conflated two paths.

- **Observer path — read-only by configuration.** `orders.enabled: false` in the paper-lab config;
  `adapter_version lts.alpaca.paper.readonly.v1`; `orders_submitted 0`; latest preflight complete, account ACTIVE,
  environment paper, `trading_blocked false`, `equity_market_open false`, `open_orders 1`, `open_positions 1`,
  `protected_execution_eligible false`, six crypto quotes, `missing_cells []`, 16,268 complete sessions.
- **Model-runner path — NOT read-only.** Its heartbeat writes `read_only: false` literally, and `AlpacaL1Profile`
  *refuses to load* a profile in which `orders.enabled` is not `true`, which the SPY L1 profile sets. The runner
  therefore cannot even start with orders disabled. What bounds it are bounds, not a prohibition:
  `quantity_ceiling 1`, `max_orders_per_day 4`, `max_risk_fraction_at_stop 5e-05`,
  `gross_notional_fraction_max 0.01`, `margin_fraction_max 0.01`, `daily_loss_budget_fraction 0.001`,
  `max_concurrent_positions 1`, an issuer allowlist of one, and the L0 risk layer.
- **What is actually true is empirical, not configured.** In the model-execution store the latest `l1_effects` and
  `l1_effect_contracts` rows are both `2026-09-25T16:03:56Z`; `execution_report_receipts` and `lifecycle_events`
  `2026-09-25T16:04:57Z`; `live_model_inferences` `2026-09-26T00:00:03Z` — all *before* the reboot — while
  `l1_broker_facts` keeps growing to 68,094 rows with the latest at `2026-09-27T01:45:43Z`. **No order effect has
  been created since before the reboot.**
- **How this was checked:** configuration, source and retained rows. **No broker mutation of any kind was used to
  test it**, and no read-only venue call was made either.

The chat service's equivalent posture is verified absent rather than configured: its venv contains no broker
package, it holds exactly one non-socket descriptor — its conversation store — and it opens no venue. That store's
mtime is 2026-09-25 06:06: **preserved and unread.** I did not open it and nothing in this work wrote to it.

### 1.4 The pinned deployment: prepared, rehearsed, reversible, **not activated**

In `~/.local/state/lts/pinned-deploys/`, outside every repository and every checkout:

| role | tree | commit | files | export verified against the commit |
|---|---|---|---|---|
| candidate | `candidate-12bce5f2c4fe/tree` | `12bce5f2c4fe2ab0e6a7433d7208d9e05a9cffaa` | 379 | bit-exact |
| rollback | `rollback-9090f49ae53b/tree` | `9090f49ae53bca047c4f3f39848fe8d457eab77d` | 367 | bit-exact |

The candidate is the code the resident runner actually imported; the rollback is the code the timers executed until
06:15:11 today. Each pin carries `MANIFEST.json` (commit, tree sha, per-file sha256, interpreter, dependency pin),
`PIP_FREEZE.txt` with local paths redacted, and its rehearsal record; `VERIFY.py` and `ROLLBACK.md` sit beside them.
Both trees are **mode-immutable** (`a-w`, proved by two refused writes — an overwrite and a new file). That is mode
immutability, not `chattr +i`, which needs root; the owner can still lift it.

**Rehearsal**, run under `crispdm-run -m 2G -t 900 -n rr0306` with `CUDA_VISIBLE_DEVICES=''`: integrity `INTACT`
before; the three entrypoint modules imported *from the pinned tree* and resolving to it; the watchdog AST-parsed;
`app.alpaca_paper_cli --help` rc 0; **75 tests passed** in `tests/unit/test_broker_refusal_recovery.py` in the
candidate pin and `ABSENT_IN_THIS_PIN` in the rollback pin, because that commit predates the file; integrity
`INTACT` after, so the rehearsal wrote nothing into the tree. Aggregate rc 0 for both pins. No venue contacted, no
credential read.

**Rollback** is written and not executed, and it records what it cannot restore: `9090f49` does not contain
`app/broker_refusal.py`, so rolling back *removes* the refusal-recovery behaviour the resident runner is loading.
Neither direction is a no-op.

**Activation** exists only as text: `candidate-12bce5f2c4fe/ACTIVATION.not-installed.conf` is the drop-in that
*would* pin the unit. **No drop-in was installed, no `daemon-reload` was run, the live LTS checkout was not
switched, nothing was merged beneath the five-minute timers, no service was restarted, and the new MT5 code was not
adopted** — it is present in the candidate tree only because a commit exports whole, and no MT5 unit exists on this
host.

**What the preparation revealed.** All four lts entrypoints resolve into the mutable checkout — the runner by
`WorkingDirectory` with `Restart=always`, the observer and shadow by absolute `ExecStart` path, the watchdog by
absolute path — and **not one carries a commit pin**. Measured against the 18 files that differ between the two
pins, the changed files inside each import closure are: model runner **3**
(`app/alpaca_l1.py`, `app/alpaca_paper_lab.py`, `app/broker_refusal.py`), observer **2**
(`app/alpaca_paper_lab.py`, `app/broker_refusal.py`), watchdog **0**, shadow **0**.

### 1.5 Said plainly

**Evidence of an active service is not evidence of correct model decisions or of correct fills.** The runner is
alive, monitoring, and declaring a `short` with `probability_up 0.41917197764782277` off a last closed bar of
`2026-09-25T04:00:00Z` while the equity market is closed. That is a process fact. It says nothing about whether the
signal is right, whether acting on a day-old bar is sound, or whether a fill would be correct.

---

## 2. RR03 — the index, corrected row by row

Every correction below is read from an artifact, and each is written **into the row** so no reader has to diff two
files to learn that a state changed. The index's `row_contract` gained two fields: `observation_history` and
`rr03_correction`.

| row | what moved |
|---|---|
| **`ADMISSION`** (added) | the follow-on is **implemented**; its **operational recovery is still being checked**; it is **not** verified |
| `M4` | verifiability at `4b009c35` is **`DELIVERED_FOR_EXTERNAL_REVIEW`** |
| `H-CORE` | `c2d4388b` **already materialized** the prefix output; the remaining dataset/delivery/consumer requirements are named; the resolution report carries its caveat |
| `E1-Q2-CONTEXT` | v1 **historical**, v2 **sealed and not executed**, the twelve cells **cannot be relabelled** into authority |
| `MT5-demo` | `d631f4a` **published, not demonstrated deployed**; unknown outcomes keep their disposition |
| `A/B`, `R0/R1/R2-ECL` | **stay closed**, re-confirmed without reactivating anything |
| `M5PHET-5F-CHAT` | product-lane restatement plus the restarted-service accounting |
| `Alpaca-paper` | the expired claim corrected, the read-only wording corrected, the pinned deployment recorded |
| `FIN-LOSS-OPT/E3`, `calendar`, `news` | **unchanged on purpose** — no artifact of theirs moved |

### 2.1 Admission: implemented, recovery still being checked, **not verified**

The eleven-row index had no row for the lane that gates the others, while RR01's own manifest carries two. I added
**one** row, so the file's one-row-per-lane rule still holds, flagged exactly as the existing eleventh row is
flagged. The gate at `ee30935a` is deployed and closes double admission **at entry**. The follow-on at `f66ee25d`
deployed the launcher to both worker roles with remote dispatch proven to reserve, four of eight fit runners
refusing a bare invocation and the other four deferred with their pinning seal named. **Its two pytest selections
are not passed:** both attempts ended at 16:27:49 when the session died, their pipes did not survive, and they are
`INTERRUPTED_RESULT_NOT_RETAINED` — **1,997.051 CPU s over 1,477.71 s wall**, spent, not recovered, not re-budgeted
and not reset because a process died. The day also established the gate's limit: admission is an entry gate with
**no monitor after it**, so the two already-admitted scopes ran to the end while the desktop was killed. Next work
is the re-run of those two selections **on an admitted worker**, plus the post-admission monitor, lease identity
fields and lease-body retention, validated on simulated `/proc` and clocks and never by exhausting the desktop.

### 2.2 M4 verifiability is delivered for external review

The row now separates two things that were being read as one. **Execution** at `b8865058` + body `0de54534`:
`NO_NEW_MEASUREMENT`, 0 of 3024 units, neither ADVANCES nor DOES_NOT_ADVANCE, because all 16 Holm cells read
`UNDETERMINED_NO_OBSERVATION` and Holm is a step-down over p-values that do not exist. **Verifiability** at
agent-multi `4b009c35`: **`DELIVERED_FOR_EXTERNAL_REVIEW`** — six repairs, PRE reproduced the auditor's p-values
through the public path, 153 tests. Neither still unwritten nor approved. **The auditor owns the verdict.** What I
own is supplementary evidence: freezing the day review's probes as PRE and turning them into tests of the real
public path, positives and negatives under DEVELOPMENT fixtures including the full entrypoint, and correcting the
"zero arrays" sentence to what was actually checked with the declared disjointness exception resolved by name. The
external design-review record the gate requires is the auditor's to write and **no form of it may be manufactured
here.** M4 CONFIRMATION remains NOT AUTHORIZED; no unit may be fitted.

### 2.3 The prefix output is already materialized — so the row asks for what is left, not for a second fit

`PREFIX_OUTPUT_MATERIALIZATION.json` at `c2d4388b` is state `VERIFIED` with
`reasons_it_would_be_inadmissible: []`, reproduced from a clean pinned checkout at `ed27bb89` whose git status is
empty. The cut is at the fusion (23 layers inside, 0 downstream layers found inside); five stages are manifested —
preprocessing `LEARNED`, groups `CONSTANT_BY_DESIGN`, detector `LEARNED`, adapter `LEARNED`, fusion
`CONFIGURATION` — each carrying learned state, clock, rows, shape and version; the arrays are
`prefix_output_train.npy` `(40080, 60, 24)` float32 and `prefix_output_validation.npy` `(10020, 60, 24)` float32
under store version `dr05.prefix.fd0fde9b2a913257`. Four proofs `VERIFIED`: direct-against-cache (64 origins per
split, 92,160 elements each, element-by-element **and** byte-identical, max |Δ| **0.0**), reload in a fresh
interpreter (`all_digests_match true`), mutation of weights and state (four perturbations, each moved the output
and the bound identity, weight cases restored bit for bit, nothing written to a run root), and causality (measured
reach **5** equals declared reach 5; a future panel row leaves the earlier window untouched while moving the window
containing it). Window invariance measured over 1,393 positions, largest absolute discrepancy 0.0. The reserve was
not opened: the test split is `REFUSED_UNMATERIALIZED_BY_DESIGN` and its **absence** is what is verified.

So the row's previous next-work — "materialize the frozen prefix OUTPUT" — is **retired, not repeated**. What
remains, named from that return's own `what_this_does_not_deliver`:

- **dataset** — one panel, one donor, two splits, float32; no second panel or family; the clock's timezone is
  `UNKNOWN` because the producer never declared one, carried and not resolved, with delivery availability
  `UNDECLARED`;
- **delivery** — the donor cell `R1_s1` is **named, not chosen on evidence**: which learned state
  `MOD-FROZEN-PREFIX` should freeze is a design decision the module still owes, and the store's version binds the
  donor, so a different donor is a different version;
- **consumer** — nothing downstream of the fusion yet reads the store; row-addressable reads are licensed only at
  or beyond the measured reach of 5; and **no H-CORE comparison is delivered**, because the core is downstream of
  the cut and nothing here fits one.

None of that requires fitting a model. **No duplicate fit is scheduled.**

### 2.4 The three-seed resolution report is an input for review, not authority for 69 fits

`RESOLUTION.json` at `4168ebdc` (`458d22cb` is its ancestor) is `training_performed: false`, `problems: []`, and
its verdict is `UNANSWERABLE_BY_THIS_PROTOCOL_AT_THIS_SEED_COUNT`. The row now carries the caveat the order
requires, with both numbers:

- **69 = 3 arms × `n_per_arm` 23**, sized at α 0.05, power 0.80, on σ = **0.011363459833619234 kW**, the pooled
  within-arm SD of R0/R1/R2 at df 6 — and sized with **`paired: false`**. That is **independent-arm** dispersion.
- **The paired-contrast dispersion is a different quantity, and here it is larger.** The realized
  `sd_of_differences` for R1−R0 is **0.016644 kW** against √2·σ = **0.016069 kW**, and the report's own
  `seed_main_effect` finds no shared seed effect (F 0.0148, p 0.9853), so pairing buys nothing. Sizing a successor
  on the paired dispersion would require **more** seeds than 69, not fewer. The report's own estimator band already
  spans 0.031106 kW (unpaired, σ at pooled df) to 0.049762 kW (paired), and it governs by the most favourable of
  the three so the ruling cannot be blamed on a pessimistic test.
- **Censored optimization does not by itself prove the observed seed variance is a lower bound.** 8 of the 15
  retained cells stopped on `UPDATE_BUDGET`; the within-arm pooled slope is **+1.1481697636690154e-05 kW per
  update** with r 0.6346 — in these cells *more* updates went with a *worse* error, so lifting a short arm to the
  common ceiling would move its mean **up**. The report's own caveat states the update counts are themselves
  outcomes of early stopping, that the relation is observational, and that **only a budget-matched re-run** measures
  the causal effect of budget. Censoring could inflate or deflate the observed dispersion; neither direction is
  established.
- **Estimand.** `equal_updates`, declared before the comparison, and explicitly **not** equal cost. Under it the
  retained run's unequal consumed updates (11,762 / 11,762 / 10,270) invalidate the comparison; under
  `recipe_under_early_stopping` they do not.

**No blanket restart on an unreviewed power calculation.** Any successor citing 69 fits must first say which
variance it sized on and must justify its budget as a treatment rather than inherit a censored one.

### 2.5 Q2: two generations, and relabelling is not a seal

v1 is **historical** — 12 fits, 1,012.7 CPU s, retained for the record, not relaunched, not a source of authority.
`Q2_CONTEXT_DEEP` is the **sealed successor and, as a governing measurement, not executed**: its seal declares six
arms at three seeds = **18** units, while what exists on disk is **twelve attempt directories from an ungoverned
run**, 0 verified, in an unanchored table, with the block's own closure **refusing** because no unit holds an
accepted terminal receipt. The two W1440 full-depth arms have no cell and no attempt directory: six units were
never admitted. **The old twelve cells cannot acquire governing authority or a new seal by relabelling** — twelve
ungoverned attempt directories are not twelve results of the sealed successor, and renaming them under its seal
would be forgery of custody. Any table reporting 12 where the design says 18 is reporting a different thing and
must say so. Next: DR01's reservation first, then prove delivery → consumed bytes → terminal → live warehouse on
one small bounded mechanical unit **before** any further scientific fit. Neither generation is relaunched.

### 2.6 MT5 is published, not demonstrated deployed

`d631f4a` carries the corrections and **nothing shows them running**: no MT5 unit is installed on this host, the
local default bridge database still does not exist, and the new code is **not adopted**. The live watchdog reading
of `2026-09-27T01:41:55Z`: `execution_enabled true`, `read_only false`, `command_counts` succeeded **48** / failed
**5**, heartbeat received `2026-09-08T22:28:55Z` and now **1,566,806.998 s = 18.13 days** stale — that stale
heartbeat still says `connected true` and `trade_allowed true` — terminal build 6140, latest snapshot
`orders_total 0` / `positions_total 0` / `symbols_total 8` also from 2026-09-08, and `exposure_reconciliation`
`all_authorized true` over **zero** positions. **No per-order acknowledgement, retcode, ticket or fill is retained
on this host**, only the two aggregate counters. Therefore **the unknown outcomes keep their exposure and budget
disposition**: five daily entry slots stay reclaimed and held, `BUDGET_RELEASING_STATES == {'failed'}`, deny by
default. An aggregate 48/5 from a remote status endpoint is not per-order reconciliation, and a stale heartbeat
that still says `trade_allowed` is not evidence of a stopped bridge.

### 2.7 The two closed lanes, and the two maps

`A/B` and `R0/R1/R2-ECL` **stay closed**. I re-read the live host to earn the re-observation rather than assert it:
no crispdm scope, no `crispdm-batch.slice` cgroup, no rp135 or ECL unit, no `df_*` process, zero admission leases
held. Their numbers are unchanged. **Re-observation is not reactivation** and creates no dispatchable work.

The two maps stay **linked and distinct**: M5PHET is the product lane; the doctoral modules and the financial
evaluation are the scientific and business lane. They are linked by the rows that appear in both — the calendar
lane is the causal family's data dependency in one map and E3's reference input in the other — and never merged
into one ranking. A defect found in one M5PHET family is a row in that family and holds none of the others.
`R0/R1/R2-ECL` is measured on the ECL panel and is **not** the doctoral `MOD-CORE-PRETRAIN` contrast on the
household panel; different datasets, different units, never pooled.

### 2.8 Units and source transformations now live in the index

A new `metrics_units_and_source_transformations` block states, in the index itself: kW as the household panel's
physical unit and the inverse-scaler transformation that produces it; the dimensionless persistence-scaled error
with a worked example and the rule that kW and scaled quantities are never combined; that A/B's MSE/MAE are in the
published recipe's normalized units and comparable to that reference and to nothing in kW; that the two panels'
kW numbers are never pooled (household matched persistence 0.6173720558882235 kW over 10,020 rows at horizon 60;
ECL matched persistence 0.868283 kW over 1,802 windows); `resolution_kW` with its estimator band and its scaled
equivalent reported separately; macro-F1, skill and ECE for classification; and CPU s versus wall s, GiB as binary
multiples, staleness in seconds converted to days only for readability, and the fact that **no monetary unit
appears in any row of this index**.

---

## 3. Costs, admissions and what I did not touch

| | |
|---|---|
| admissions taken | **3**, all `rr0306`, each at a 2 GiB cap, each `ADMISSION_ADMITTED → LEASE_ARMED → LEASE_RELEASED` within 1–2 s |
| leases held now | **0** |
| refusals | none: the coordinator travel hold is absent (`coordinator-travel-hold.released-20260923`), and no request was re-asked at a lower cap |
| GPU | none used. `CUDA_VISIBLE_DEVICES=''` throughout. No GPU work was eligible in this assignment, so no GPU is idle for a bad reason — it is idle because nothing here needs one |
| host at the time of work | 30 GiB total, 23 GiB available, memory pressure some/full avg10 **0.00** |
| heavy compute on the coordinator | none. The only executed code was file hashing, imports, `--help` and 75 offline tests |
| services started, stopped, restarted, enabled or reloaded | **none** |
| broker calls | **none**, mutating or read-only |
| credentials | none read, none printed, none written. No `EnvironmentFile`, no `/proc/<pid>/environ` |
| the live `lts` checkout | not switched, not committed to, not staged; only read-only git queries with `--no-optional-locks` |
| owner files | nothing cleaned, staged or committed. `git add` by name only, never `-A` |
| host names, addresses, tokens, account identifiers | none written into any repository; roles only |
| model fits | **zero**. `NO_NEW_MEASUREMENT` |

---

## 4. What is not done, refused, or not measured

- The admission follow-on's **two pytest selections are still not re-run**; until they are, that row stays
  `IMPLEMENTED / OPERATIONAL RECOVERY STILL BEING CHECKED`, and they belong on an admitted worker, not here.
- The pinned deployment is **not activated**, and a real activation still needs an owner window, adoption tests,
  and a decision about the two editable dependencies the pin does **not** freeze.
- `m5phet-chat` is running a 2026-09-25 snapshot. **I did not reinstall the venv and did not restart the service**;
  RR06 forbids both. Whether the product lane wants master resident is a decision, not a repair.
- The M4 verdict, the disposition review, the confirmatory candidate and now the resolution report are all
  **waiting on the auditor**. No signature of his exists anywhere in this work, in any form.
- No quality claim, news entitlement or identified causal effect was invented to make anything look complete.
- Nothing here measures whether any model decision or any fill is correct. A live service is a process fact.

---

## 5. Next action, concrete and owned

1. Re-run **only** the two interrupted pytest selections, in bounded sequential shards, on an admitted worker,
   retaining exit code, revision, selection and resources.
2. Settle `MOD-FROZEN-PREFIX`'s **donor-selection criterion** as a written design decision, then build one consumer
   that reads the materialized prefix store row-addressably at or beyond reach 5 — CPU, no reserve, no fit.
3. Leave the pinned deployment as prepared until the owner grants a window; do not let a restart decide which
   commit trades.

— Satoshi III (Mujuro Utsutsu), successor technical lead, 2026-09-26
