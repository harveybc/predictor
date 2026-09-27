# RR04 — the governed chain now delivers bytes to a terminal in the live warehouse; no programme task is READY

**Date:** 2026-09-26
**Author:** Satoshi III (Mujuro Utsutsu), successor technical lead
**Continues:** RR04 of the post-reboot orders (`0d6f6f7e`), on top of the RR01 restart manifest
(`0ee09998`). It does not replace either, and it creates no allocation, reserve access, real-capital
authority or M4 approval.
**Verdict in one line:** **the governed path works end to end** — two bounded mechanical units and one
negative control, with a receipt at every hop and both terminals read back out of the live warehouse
under the worker's own identity — and **no task in the doctoral/business programme has all four of its
own gates satisfied**, so nothing was dispatched and the exact smallest missing prerequisite is named
below. `NO_NEW_MEASUREMENT` for the science.

Evidence: `docs/audits/evidence/RR04_GOVERNED_DELIVERY_20260926/`
(`GOVERNED_CHAIN.json`, `WAREHOUSE_VERIFY.json`, `CLOSURE_TABLE.json` / `.md`,
`READINESS_GATES.json`, `AVAILABILITY_SCOPE_DERIVED.json`, `ADMISSION_LEDGER_WORKER.jsonl`,
`unit1/`, `unit2/`).

---

## 1. Job 1 — the credential, resolved from a documented reference, not hunted for

**Nothing was issued, rotated, read, copied or printed.** No secret was searched for.

The order offered `data-gov scripts/issue_service_key.py` at `d07695b` as the means. I read it, and
then read two facts that made issuing both unnecessary and useless:

1. **The concrete reference already exists and is documented.** Predictor
   `docs/handoffs/MUSASHI_WORKER_ACTIVATION_AND_R1_R6_COMPLETION_2026_09_15.md` §1 states, in its own
   words, that dedicated worker credentials were installed privately on the two remote workers and
   that *"the worker key file is `~/work/gov/worker.key` on each worker. Use it explicitly via the
   existing client's `api_key_file` / CLI key-file option. Do not keep selecting the old
   `predictor.key`."* I verified by existence check — not by reading — that **both workers hold that
   file**, and that the matching service principals are resident in the effective governance
   configuration. That is the documented credential reference the DR02 row was pointing at; the
   dependency was never a missing secret, it was a runner invoked on a host that had no key file
   configured.
2. **Issuing into the live configuration would not have taken effect.** `data-gov` builds its access
   plugin once at start-up and every request authenticates against that in-memory instance
   (`web_plugins/default_web.py:260` → `access_plugins/default_access.py:49`). A new or rotated
   digest in the configuration file becomes effective only on a restart — which the same handoff
   confirms happened twice, deliberately, in September. RR02 and RR06 forbid me to restart a
   production service, so **issuing a key would have produced a credential that authenticates
   nothing.** Rotating the shared `predictor` principal would additionally have invalidated whatever
   still holds it.

So the resolution is: **run the governed path on a worker, with that worker's own key file, over the
existing governance tunnel.** That is also exactly where RR04 says the placement belongs.

## 2. Job 1 — the chain, hop by hop, with the receipt of each hop

Runner: `tools/governed_run.py`, unchanged — the decision-bearing Flow-v3 client, not a new one.
Launcher: `$HOME/.local/bin/crispdm-run` on the worker; every unit took an atomic reservation.
Checkout: a fresh worktree at the published commit `0ee0999868e0f5438a0e2fb34ac769116caf38c9`, clean
and untracked-clean, so `strict_code_identity` returns a governing commit.
Unit: the committed toy daily ANN, 2 epochs over 300 steps, on the downsampled example CSVs — a
mechanical unit, declared `--classification NON_GOVERNING`, not a scientific fit.

| hop | what had to be true | receipt |
|---|---|---|
| **delivery** | the campaign is registered before anything opens, and every input role is requested separately and verified | campaign `e0d552de…` (u1) and `c9b60e72…` (u2), each `stored: true`; six deliveries per unit with their own `delivery_id`, `source_sha256`, delivered `sha256`, byte count, availability-contract digest `5a521473…` and `availability_use` `OFFLINE_DAY_GRANULAR` |
| **reader** | the run reads the delivered bytes, not the checkout's copies | the generated `governed_config.json` names `~/.cache/data-gov/predictor_examples/<sha256>.csv` for all six input roles; the checkout's own `examples/data_downsampled/...` paths appear nowhere in it |
| **consumed bytes** | what was read is what was delivered | each file the config names was re-hashed on the worker: on-disk `sha256` == delivered `sha256` == source `sha256`, six for six, `ALL_CONSUMED_BYTES_MATCH_DELIVERY: True` |
| **terminal** | the terminal is written `O_EXCL`, sent, and not left pending | u1 `FAILED` / `PREDICTOR_EXIT_1`, u2 `COMPLETED` with 90 metrics and 6 artifact digests; both `outbox_flush {sent: 1, pending: 0, failures: {}}`, both `terminal_pending: false` |
| **live warehouse** | the terminal exists in the cube and accounting agrees with it | read back with `GET /api/v1/query` on lake `olap_cube` by the same worker identity: `gov_terminal` rows `ddf81e8b…` (FAILED) and `e0c1a2b2…` (COMPLETED), actor = **the worker's own service principal**, 12 `gov_terminal_dataset` rows, 90 `gov_terminal_metric` rows, 6 `gov_terminal_artifact` rows; `reconciliation` `accounting_only: []`, `lake_only: []`, `missing_units: []` |

**Cache reuse is measured, not assumed.** Unit 1 transferred three resources (`VERIFIED_TRANSFER`)
and served three from the verified cache (`VERIFIED_CACHE`); unit 2 reused all six.

**Negative control, so the chain is not decoration.** A third unit, identical except that `--gov-url`
pointed at a closed loopback port, refused: `data-gov unreachable … Connection refused`, exit 1, and
its output namespace was created **empty** — zero prediction, results, uncertainty or model files. The
reader does not open data when governance is unreachable.

**The one defect the smoke exposed, and it is in the committed config, not in the chain.** Unit 1
failed inside predictor with `ColumnRoleError: this run declares no column_roles`. The toy config
predates `app/column_roles.py`. Unit 2 declared `--column_roles_migration
LEGACY_ALL_COLUMNS_ARE_FEATURES` deliberately and on the record — an allowed extra flag, since
`refuse_governed_overrides` guards inputs, outputs and the config and nothing else — and it is in the
execution spec, the terminal and the closure table's comparability reason. The FAILED terminal of
unit 1 is kept: a refusal that reaches the warehouse is part of what had to be proved.

**Measured full-process memory, on the correct basis.** The launcher's sampler records the scope
cgroup's peak at release (`observed_tree_peak_bytes`, `peak_scope: "cgroup"`), never one process's
RSS: **0.390 GiB** (u1), **0.596 GiB** (u2), **0.011 GiB** (negative control), all under a 3 GiB cap.
The cap was not chosen to pass a gate: 3 GiB is the smallest cap for which a retained cgroup peak of a
TensorFlow-importing scope existed on this fleet (2.9 G under a 3 G cap, DR01 attempt 1). The 0.596
GiB now measured is the basis recorded for this unit's successors.

**Historical custody is untouched.** No `NON_GOVERNING` attempt was relabelled, no old cost
re-budgeted, no terminal back-dated. The twelve Q2 cells gain nothing from this: their closure still
reads zero verified, and it should.

## 3. Job 1 — the closure table, generated from artifacts

`docs/audits/evidence/RR04_GOVERNED_DELIVERY_20260926/CLOSURE_TABLE.md`, produced by the new
`tools/df_governed_unit_closure.py`: **19 rows, 6 verified, 0 with problems.** Disposition
`MECHANICAL_TRANSPORT_ONLY` on every row.

| disposition | unit · horizon · split | metric & scale | binding | warehouse | model error | naive (same rows, n) | skill | reference | comparability | verified |
|---|---|---|---|---|---:|---:|---:|---|---|---|
| MECHANICAL_TRANSPORT_ONLY | u2 · H9 · test | MAE, target units of `test_CLOSE` | TERMINAL_ARTIFACT | ACCEPTED | 2.62221 | 0.00425323 (n=244) | −615.522 | NOT_CARRIED | NOT_COMPARABLE | YES |
| MECHANICAL_TRANSPORT_ONLY | u2 · H12 · test | " | TERMINAL_ARTIFACT | ACCEPTED | 2.83471 | 0.00487739 (244) | −580.194 | NOT_CARRIED | NOT_COMPARABLE | YES |
| MECHANICAL_TRANSPORT_ONLY | u2 · H15 · test | " | TERMINAL_ARTIFACT | ACCEPTED | 2.83808 | 0.00540014 (244) | −524.556 | NOT_CARRIED | NOT_COMPARABLE | YES |
| MECHANICAL_TRANSPORT_ONLY | u2 · H18 · test | " | TERMINAL_ARTIFACT | ACCEPTED | 2.81790 | 0.00589038 (244) | −477.390 | NOT_CARRIED | NOT_COMPARABLE | YES |
| MECHANICAL_TRANSPORT_ONLY | u2 · H21 · test | " | TERMINAL_ARTIFACT | ACCEPTED | 2.87813 | 0.00635462 (244) | −451.919 | NOT_CARRIED | NOT_COMPARABLE | YES |
| MECHANICAL_TRANSPORT_ONLY | u2 · H24 · test | " | TERMINAL_ARTIFACT | ACCEPTED | 3.00947 | 0.00687312 (244) | −436.860 | NOT_CARRIED | NOT_COMPARABLE | YES |
| MECHANICAL_TRANSPORT_ONLY | u1 · no forecast rows | null (absent) | NOT_BOUND | ACCEPTED | null (absent) | null (absent) | null (absent) | NOT_CARRIED | NOT_COMPARABLE | NO |
| MECHANICAL_TRANSPORT_ONLY | u2 · H9–H24 · train, validation (12 rows) | MAE, target units as reported | RECORD_DIGEST | ACCEPTED | null (absent) | null (absent) | null (absent) | NOT_CARRIED | NOT_COMPARABLE | NO |

Read it for what it is. **`verified: YES` means the predictions digest binds to the accepted terminal,
the terminal is in the live warehouse, and the error was recomputed here in float64 and agreed.** It
does **not** mean the number is usable: the skill is between **−437 and −616**, which is what two
epochs of an untrained network against persistence should look like, and every row carries
`NOT_COMPARABLE` with its reason. `NOT_CARRIED` for the reference is honest: no published value
applies to a downsampled toy slice. A `null` is an absent measurement — the twelve train/validation
rows carry a warehouse value whose rows this unit did not keep, so the recomputation is absent, and
the generator says so instead of printing a zero.

**The generator is tested where a badge-reader would pass.** `tests/test_df_governed_unit_closure.py`,
**10 passed**: forging one prediction to be perfect moves the recomputed error *and* breaks the digest
binding *and* raises a problem; deleting the warehouse rows drops every row to `NOT_IN_WAREHOUSE` and
unverified; a `COMPLETED` unit whose rows were removed is **refused**, not emitted empty; a zero naive
leaves skill `null` rather than infinite; rows missing any of prediction, target or base are dropped
from **both** errors so the two populations cannot differ.

## 4. Job 2 — nothing is READY, and this is the exact smallest missing prerequisite

Full per-lane assessment with derivations: `READINESS_GATES.json`. Summary of the eight
doctoral/business lanes of `research_dispatch_index.v2`:

| lane | governance | design | resource | budget | verdict |
|---|---|---|---|---|---|
| A/B | — | — | — | — | closed, `NOTHING TO DISPATCH`, do not reactivate |
| R0/R1/R2-ECL | — | — | — | — | closed, 9/9 scored, do not reactivate |
| E1-Q2-CONTEXT | **FAILED** | sealed | **FAILED** | **FAILED** | not ready |
| MOD-FROZEN-PREFIX | **FAILED** | partial | ok | undeclared | not ready |
| MOD-CORE-PRETRAIN | n/a | ok | ok | **FAILED** (23 seeds/arm) | not ready |
| M4 | — | — | — | not authorized | excluded by this order |
| FIN-LOSS-OPT | **FAILED** | sealed v4 | ok | pilot only | not ready |
| calendar | **FAILED** | — | — | — | owner entitlement decision |

**The smallest missing prerequisite, and it blocks two lanes at once:**

> **No store serves the `public_panels` lake.** Governance at 5055 routes `public_panels` to
> `http://127.0.0.1:5059`; **nothing listens on 5059**; the user unit
> `crispdm-data-lake-public-panels.service` exists but reads `UnitFileState=disabled`,
> `ActiveState=inactive`, **`Result=success`** — a clean, deliberate stop, not a crash, superseded on
> 2026-09-22 when that deployment directory's successor was brought up as the `sota_benchmarks` store
> on 5060 instead. A governed `discover` of `public_panels`, made through the worker's own identity
> over the tunnel, returns **HTTP 200 with 0 resources**. The panel bytes are on this host, at
> `public_panels_c126_v2/uci_235_individual_household_power/panel.parquet`.

So the chain proved in §2 cannot carry any E1 household-panel unit: **Q2 successor v2 and
MOD-FROZEN-PREFIX's materialization have no deliverable input.** Removing it means either running a
store for `public_panels` again or repointing that entry in the 5055 configuration at a deployed
store. Both are an operator apply plus a restart of a production service, which RR02 and RR06 forbid
me to perform — and the unit is **disabled by a deliberate act**, which I will not override on my own
authority. **Named, and I stop here.**

**The second prerequisite, independent of the first**, for FIN-LOSS-OPT — and here I correct my own
DR02 row rather than repeat it:

- The deployed financial store carries **`resource_contracts: {}`** in its start-up parameters, so the
  sealed cost pilot's *ranged* download of the EURUSD 1h resource stays HTTP 422 *"resource
  availability contract required"*.
- **The DR02 row was wrong to call this reachable through `resource_registration.py`.** That module's
  own docstring says *"Registering a resource does not open it … a resource with no availability
  contract stays closed"*, and the contract is read from `self.params` at start-up
  (`financial-data lake/inventory_plugins/fs_inventory.py:476`). It cannot be registered hot; the
  operator-configuration route writes a *pending* file that an operator applies with a restart.
- **And it is not merely operator metadata.** I ran the repository's own derivation tool rather than
  asserting anything: `financial-data _scripts/derive_availability_scope.py` assessed **420 families
  and instantiated 0**, every one missing `available_from_ts_col`, `min_latency_minutes`,
  `revision_policy`, `timezone` and `license_scope` (`AVAILABILITY_SCOPE_DERIVED.json`, scope digest
  `f9343a9a…`). For the EURUSD resource in particular: the parquet carries **one** time column
  (`datetime`, tz-aware UTC, 129,873 rows, hourly); the resampler at
  `_scripts/workers/stage13_*_light_worker.py:125` uses pandas defaults, so the bar label is
  `WINDOW_START` **by source evidence**; and `provenance.json` states source, description,
  `acquired_at` and file digests and **nothing** about the provider's publication lag.
  `availability_scope()` requires `completion_lag_max` and refuses to default it. A truthful ranged
  contract therefore **cannot be authored from source evidence**, and inventing the lag is precisely
  what is forbidden.
- **One owner question, because it is a genuinely missing external fact:** the HistData FX archive's
  bar-label convention and maximum publication lag, or a substitute source carrying an observed
  availability instant.
- Closing the stale `satoshi-fin-cost-pilot-20260921-prepare-data` campaign at its boundary is also
  not available to me: data-gov refuses a terminal from any actor but the campaign's own
  (`web_plugins/default_web.py:667`), and that campaign was opened under a different identity from
  the one this return resolved. It stays open, visibly, rather than being closed by a forged actor.

**What I executed instead of a dispatch** — the work that removes the prerequisite as far as my
authority reaches: the governed chain itself (§2), which was the binding constraint on the whole
scientific lane; the availability derivation above, which turned "operator metadata" into a measured
420/0 verdict and one named owner question; and the closure-table generator with its ten tests. No
duplicate experiment was created to keep a GPU busy, and the external RTX 5090 stayed idle because no
eligible GPU work exists — its host ran only the three CPU-bounded smoke scopes.

## 5. Q2 successor v2, reconciled — and not relaunched

Neither the refused W1440 attempt nor v1 was relaunched. Reconciliation against the three things the
order names:

- **Runner identity.** The governed path is now available to Q2 under a worker's own principal at a
  published commit, with campaign, delivery, terminal and reconciliation receipts — *once its panel is
  deliverable*. Until then every Q2 unit remains `NON_GOVERNING` by construction, and the E1 helper
  additionally hard-codes `classification: "NON_GOVERNING"` in its campaign
  (`tools/df_e1_governed.py`), so a successor that intends to govern must say so explicitly rather
  than inherit it.
- **Actual full-process memory — the prior basis was wrong and is corrected here.** The 7.4 GiB in
  `ASYMMETRIC_READING.json` and the `measured_pilot_peak_bytes` of 8,458,399,744 in
  `MEMORY_GATE.jsonl` are a **resident set**. Q2 ran through `df_memory_gated_run` and **never took a
  `crispdm-run` scope, so no `memory.peak` exists for any Q2 cell.** A cap for a W1440 successor
  cannot be declared from those numbers. The correct next step is one bounded pilot under
  `crispdm-run` that measures the scope's cgroup peak — and that pilot is itself blocked by the panel.
- **Residual budget, and it does not fit.** v1 spent 1,012.7 CPU s for 12 fits; the deep block spent
  522.833 CPU s for 12 of 18 units. At the measured 4.165 CPU s per train update and 2,000–4,000
  updates per cell, the six W1440 cells cost of the order of **50,000–100,000 CPU s and 15–30 h
  wall**. No allocation grants it. Today the launcher would refuse the recorded 9,532,141,568-byte
  request on every host available: measured `MemAvailable` is 6.5–7.1 GiB on one worker and 6.7 GiB on
  the other, against a 3 GiB desktop reserve, leaving 3.4–4.1 GiB for new work.
- **The estimand limit, carried honestly.** The executed 600-update design **cannot answer a
  converged-accuracy question.** Its twelve cells stay `CENSORED_BY_BUDGET`, read under
  `ASYMMETRIC_READING.json`'s asymmetric rule — a long-context arm that lands worse is
  `CONFOUNDED_WITH_BUDGET` and identifies nothing. Either that limited estimand is carried as such, or
  an adequately costed successor is prepared for review; I changed no window, depth, cell count or
  optimization to make it fit, and I declare no scientific dimension or budget change here.

## 6. What I did not do, deliberately

- **No secret searched for, read, copied, issued or rotated**, and no authorization invented. No key
  material of any kind is in this branch.
- **No production service started, stopped, restarted or reconfigured**, and no deliberately disabled
  unit enabled. No operator config file outside the checkouts was written.
- **No heavy compute on the coordinator.** It ran three bounded inspections (1–2 GiB caps, seconds
  each) through `crispdm-run`; all model work was on a worker.
- **No relaunch** of the refused W1440 attempt, of Q2 v1, of the A/B queue or of the nine-cell
  contrast. **No duplicate experiment to occupy a GPU.**
- **No historical custody changed**, no cost re-budgeted, no `NON_GOVERNING` result promoted, no
  closure `null` written as a zero.
- **No holdout read, no broker call, no trading mutation, no M4 unit fitted, no auditor signature.**
- **No hostname, IP, token or account identifier written.** The worker service principal's name
  embeds a host name, so the warehouse verification document is committed with that value redacted to
  `<WORKER_A_SERVICE_PRINCIPAL>`; the unredacted value stays in the live warehouse and the operator
  configuration, outside every checkout. One `financial-data` script I cite by path also has the
  coordinator's host name inside its own filename, so it is cited as
  `_scripts/workers/stage13_*_light_worker.py:125` — locatable by glob, and one more entry for the
  host-name inventory in that repository rather than a new leak in this one.

## 7. Next action, owned

1. **Ask the owner one question and wait**: whether `public_panels` should be served again (its unit
   and its bytes both exist) or its entry repointed — and, separately, the HistData bar-label and
   publication-lag facts. Both are decisions, not code.
2. The moment `public_panels` delivers, the first unit to run is **not** a Q2 arm: it is one bounded
   `crispdm-run` pilot of a W1440 cell to measure the scope's cgroup peak, so a cap can be declared
   from a measurement instead of a resident set.
3. Then, and only with an allocation, an explicitly justified and pre-declared Q2 successor — or the
   600-update block carried as the limited estimand it is.

— Satoshi
