# Return: hosts published, D2 reporting corrected, production transition blocked by the runtime

Orders: `docs/handoffs/MUSASHI_TO_SATOSHI_PUBLICAR_Y_DESPLEGAR_HOSTS_2026_09_14.md` and
`docs/handoffs/MUSASHI_TO_SATOSHI_HOSTS_NOW_AND_D2_REPORT_CORRECTIONS_2026_09_14.md`
(commit `71faa4e`). Work plan: `docs/integracion_workplan_2026_09_10/09_ADOPCION_DATA_LAKE_DATA_WAREHOUSE_2026_09_14.md`.

States used below: `PUBLISHED`, `PROVEN_DISPOSABLE`, `DEPLOYED`, `PROVEN_PRODUCTION`,
`BLOCKED(<what>, <who can unblock>, <minimum action>)`.

## 1. The two repositories

| | URL | default branch | commit | visibility |
|---|---|---|---|---|
| lake host | https://github.com/harveybc/data-lake | `master` | `1ba23ed` | **private** — see §5 |
| warehouse host | https://github.com/harveybc/data-warehouse | `master` | `6f16565` | **private** — see §5 |

Both carry code, tests, README, AGENTS.md, example configurations and
`docs/IMPLEMENTATION_STATE.md` (requirements, acceptance scenarios, test matrix, stage
state). Local history was pushed as it stands; no commit was rewritten to fake a timeline.

Providers, from their own repositories:

| provider | repository | revision | how it installs |
|---|---|---|---|
| `financial-data-store` | financial-data | `d9be1b368` on the default branch | `pip install "git+https://github.com/harveybc/financial-data.git#subdirectory=store"` |
| `predictor-olap-store` | predictor | `6d5c9ed` on `satoshi/olap-store-provider-20260914`, PR #44 into `master` | `pip install "git+https://github.com/harveybc/predictor.git@satoshi/olap-store-provider-20260914#subdirectory=olap/store"` |

The predictor provider could not be pushed to `master` directly (the runtime refused a push
to the default branch), so it comes as a pull request and the pinned public revision is the
declared installable origin meanwhile, exactly as the order allows.

## 2. Installed from GitHub, in a clean environment — `PROVEN_DISPOSABLE`

A virtual environment **without** system site packages, nothing from a sibling checkout on
`sys.path`, four distributions installed from their GitHub URLs:

```
datalake.backends      [('memory_store', 'data-lake-service'), ('financial_files', 'financial-data-store')]
datawarehouse.backends [('sqlite_store', 'data-warehouse-service'), ('predictor_olap', 'predictor-olap-store')]
```

* published test suites against the installed code: **23** (lake, including the three that
  build and install a fixture provider into a throwaway venv) and **18** (warehouse);
* route parity against the hosts that are running today: **8 of 8** data routes identical in
  status, bytes and every header the kernel reads; **11 of 11** identical in status and
  body, including the repeated report and the repeated terminal;
* the whole Flow v3 campaign through both new hosts with the governance configuration
  untouched: same availability contract `139f3adc…`, same dataset digest `5b4cabe4…`,
  reconciliation exact.

**Two real defects were found by these runs, not by inspection:**

1. the packaged OLAP provider was a copy of *my branch*, not of the plugin the warehouse
   runs: it answered `/api/v1/discover` with row counts where the deployed plugin answers
   `row_count_status: NOT_SCANNED` and lists the `gov_metric_current` view. The package now
   pins the deployed revision `a7a86e9` (`ded98356…`) and the test checks the pin against
   Git, because that revision was reachable only from the runtime worktree and belonged to
   no branch; it is now published as `runtime/warehouse-deployed-20260914`;
2. the lake host, started from the runtime configuration alone, inventoried **16,346**
   resources where the service inventories **5,275**: `include_globs` lived in the legacy
   application's `app/config.py`. Those domain defaults now belong to the provider, with a
   test comparing them field by field against the legacy application.

## 3. Operator console — `PROVEN_DISPOSABLE`

Both hosts serve an AdminLTE console with vendored assets: inventory and provider identity;
resource metadata (coverage plus the producer contract, or a plain statement that the
resource declares none) or relation schema; a bounded read-only query whose result table is
rendered and escaped; and a configuration page that writes a **pending** file atomically.
A pending file is not an authorization and not a deployment; secrets are redacted and a
redacted value cannot be saved back. 10 + 10 tests, plus browser acceptance at 1440×900 and
390×844 over 6 and 8 pages: every asset served by the host itself, no failed or external
request, zero horizontal overflow. Receipts and PNGs in `docs/console/` of each repository.

## 4. Transition validation — done up to the point the runtime allows

Prepared and verified before any swap:

* window with nothing in flight: 12 of 12 governed deliveries verified, 3 of 3 terminals
  COMPLETED, last governance event 07:03Z;
* backups: `accounting.before.db`, `gov_tables.before.sql`, `cube_counts.before.txt`;
* runtime configurations for both hosts derived from the live ones, with the same store and
  resource IDs, the same data root, cuts and spool, the same holdout and the same database;
* dry run on spare ports against the real data: inventory **identical** to the live host,
  5,275/5,275 resources with the same sizes and time columns; the warehouse candidate
  answers identically except one additive `transport` field in `describe`, which is the
  host stating how it is reached (declared, not hidden);
* service environment built **from the published packages** in `~/.venvs/store-hosts`, each
  service to run in its own transient scope with `MemoryMax=2G`, `MemorySwapMax=0`, one BLAS
  thread; rollback = stop the new process and start the previous command, unchanged.

**`BLOCKED(stopping :5056/:5057 and starting the replacements, the owner or an operator,
run the two commands below in a window with no delivery in flight)`** — the runtime
classifier refuses process control over running services from this agent, as it refused the
restart script in the previous round. Everything before that step is done.

Start (token and password from `musashi-n3-20260914T063541Z/505{6,7}.environment.json`,
never written into a repository):

```
D=~/.local/state/crispdm-data-foundation/satoshi-store-hosts-20260914T1215Z
# lake  (replaces the financial-data/lake app on :5056)
cd "$D" && DATA_GOV_LAKE_TOKEN=<token> PYTHONUNBUFFERED=1 setsid nohup \
  crispdm-run -m 2G -t 720h -n datalake5056 -- ~/.venvs/store-hosts/bin/python \
  -m data_lake_service.main --load_config "$D/5056.host.json" >> "$D/5056.log" 2>&1 &
# warehouse (replaces predictor/olap/lake on :5057)
cd "$D" && DATA_GOV_LAKE_TOKEN=<token> PGHOST=127.0.0.1 PGPORT=5432 PGUSER=metabase \
  PGPASSWORD=<password> PGDATABASE=predictor_olap PYTHONUNBUFFERED=1 setsid nohup \
  crispdm-run -m 2G -t 720h -n datawarehouse5057 -- ~/.venvs/store-hosts/bin/python \
  -m data_warehouse_service.main --load_config "$D/5057.host.json" >> "$D/5057.log" 2>&1 &
```

Rollback: stop the new process and start the previous command from its worktree
(`musashi-n3-financial-data-20260914T063541Z/lake` and
`musashi-n3-predictor-20260914T063541Z/olap/lake`, `python -m app.main --load_config
musashi-n3-20260914T063541Z/505{6,7}.runtime.json`). Nothing to restore in the data: both
hosts read and write the same files and the same database. Triggers: any route answering
differently from the recorded parity, a failed health check, a governed micro-run that does
not reconcile, or a write path refusing where the previous adapter accepted.

data-gov on :5055 is not restarted: its configuration does not change, because both stores
keep their `base_url`.

**Declared limit of the production micro-run:** with data-gov's configuration frozen, the
only HTTP lake a governed campaign can consume is `financial_files`, and **no financial
resource has a producer-derived contract yet** (`resource_contracts: {}` in the running
service). That deficit predates these hosts and is not caused by them; until it is closed,
a governed *delivery* through the lake host in production is impossible, while the governed
*terminal* through the warehouse host is not.

## 5. Public visibility — `BLOCKED(the owner, one command)`

The order decided public. The runtime refuses to create or convert public surface
(`gh repo create --public`, `gh repo edit --visibility public`), so both repositories were
created private with their full content. One command each makes them public:

```
gh repo edit harveybc/data-lake --visibility public --accept-visibility-change-consequences
gh repo edit harveybc/data-warehouse --visibility public --accept-visibility-change-consequences
```

## 6. D2 reporting — B1, B2, B3

**B1 — totals from the records.** `tools/df_d2_summary.py` counts by `arm_role` and
decision, distinguishing rows, regimes and unique methods, on the grain
`subject_kind + subject + operator_params + regime`:

| | published | successor |
|---|---|---|
| CANDIDATE `LAB_CALIBRATED` | 51 | **47** |
| CANDIDATE `REGIME_LIMITED` | 7 | **6** |
| `SNR_CALIBRATED_FOR_REGIME` | 39 | 39 |
| rows changed | — | 138 |
| passes lost | — | **5 candidates + 2 identity controls**, 0 gained |

Six tests, including one that reproduces the mistake: on the coarse `(subject, regime)` join
the same records appear to lose far more passes. That is how "48 plus 6" and "seven lose a
pass" survived; both are withdrawn, and the earlier reports are kept with their prose
explicitly superseded. Table: `d2_support_r1/r3/D2_OPERATOR_TOTALS.{json,md}`.

**B2 — the comparison enforces its population.** Tests first: PRE 12 of 17 failing, POST 17
of 17. A role that omits a selected unit, a missing selected fact, a duplicated fact, two
files for one role, a replay of another design or a fact with no historical row now make the
result INCONCLUSIVE with named reasons, and `changed` is published only when the verdict is
MEASURED. Rows outside the subset are carried untouched and counted apart from missing
selected facts. A replayed non-estimate, a missing fact or a non-finite value replaces the
published estimate with NOT_IDENTIFIABLE instead of leaving the old value standing.

Re-comparison of the conserved replays, registered as an **additive governed diagnostic** on
the live services (campaign `aef32e87…`, terminal COMPLETED, 42 metrics, 2 artifacts,
reconciliation exact; the cube moves from 3 to 4 governed terminals):

| denominator | value |
|---|---|
| selected units | 32 |
| expected facts per role | 572 |
| roles | 3 |
| expected cells / compared cells | 1,716 / 1,716 |
| regimes | 16 |
| substituted facts | 639 estimates + **153 explicit non-estimates** |
| decisions compared / changed | **288 / 0** |

The 153 non-estimates were previously left as published values; substituting them properly
does not change a single decision. Per estimator: four exact (0.0 dB), `ar_residual` inside
the tolerance at 1.918e-13 dB, `local_level_kalman` outside in 83 of 264 cells.

**B3 — attribution demoted, diagnostic proposed.** The CPU-only explanation is withdrawn as
a result: one of three hosts differs and *why* is not measured; matching version strings do
not isolate the processor, and library build identity, runtime dispatch and the optimiser's
convergence path were never recorded. The claim that the observed deviations bound future
error is removed. `docs/integracion_workplan_2026_09_10/10_DIAGNOSTICO_KALMAN_NO_REPRODUCIBLE_2026_09_14.md`
proposes the narrow diagnostic with its interventions, controls, budget and limits, and what
a deterministic replacement would have to prove beyond an iteration cap. Nothing executed;
AT9 stays open under its original criterion.

## 7. State of the services

`:5055/:5056/:5057` healthy and unchanged; loader `active`, `NRestarts=0`; no throwaway
database; the cube holds 4 governed terminals (3 before, plus this round's diagnostic) and
its history intact.

**One side effect, declared:** while preparing the provider I checked out another branch in
the financial-data working tree, which is the lake's `root_path`. For a few minutes two
census files were absent from disk; no campaign was running and no delivery was in flight.
The tree was restored to `satoshi/c122-c145-20260912` and the inventory is 5,275 again,
identical to the candidate host's.

Stop: `HOSTS_PUBLISHED_PRIVATE_PARITY_AND_CONSOLE_PROVEN_D2_REPORTING_CORRECTED_TRANSITION_BLOCKED_BY_RUNTIME`.
