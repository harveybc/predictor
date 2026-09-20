# Satoshi — return of RP33–RP40: governed E1 validation and the weekly runtime

Order: [MUSASHI_PROGRAM_RP33_RP40](../../handoffs/MUSASHI_PROGRAM_RP33_RP40_2026_09_19.md), dictum
[MUSASHI_RP25_RP32_REVIEW](MUSASHI_RP25_RP32_REVIEW_2026_09_19.md) (F1–F7). Base `60daac9`. Ending state:
`RP33_RP40_E1_GOVERNED_VALIDATION_AND_WEEKLY_RUNTIME_READY_FOR_REVIEW`.

**One block is incomplete and it is the first one.** The adoption step of RP33 — writing the data-gov
configuration and restarting that one service — was refused by the execution environment's permission
layer, not by data-gov, not by a missing owner authorisation and not by anything scientific. Everything
that does not depend on it was completed; the successor pilot is sealed and **was not executed**, because
the order forbids scientific training while the governance prerequisite is open. The exact operation and
the evidence that it is ready are in
[`RP33/ADOPTION_BLOCKED.json`](../evidence/d3_k5_20260917/RP33/ADOPTION_BLOCKED.json).

## PRE

The dictum's reproduction script was run against the reviewed revision before any repair, and its report is
**byte-identical** to the reviewer's own: same accepted forgeries, same 2 785 mask positions moving the
auto-encoder's validation loss from 0.259784 to 0.326498 with the weights untouched, same 53-of-60 input
rows that change nothing, same donor accepted with `updates` 1100 → 0
([`RP33_PRE/RP32_REVIEW_REPRODUCED_PRE.json`](../evidence/d3_k5_20260917/RP33_PRE/RP32_REVIEW_REPRODUCED_PRE.json)).

## RP33 — the data, registered (rehearsed, not adopted)

Inventory first ([`RP33/INVENTORY_BEFORE.json`](../evidence/d3_k5_20260917/RP33/INVENTORY_BEFORE.json)): five
services active with no restarts, four lakes, none serving the public panels. The proposed change is one lake
entry and one policy inside the configuration data-gov already loads — no new backend, no new service, no copy
of the bytes, no census. What it publishes is bounded on purpose: the household and electricity panels only;
the panels declared `untimed`, because their timestamp labels are data and not evidence of when a row became
available; no availability block, so every delivery carries `UNDECLARED`; and a holdout that refuses every
date range, so only the whole resource is deliverable, AS_IS. UNKNOWN stays UNKNOWN.

Rehearsed on a disposable data-gov plus a disposable DuckDB cube, with the same procedure the store adoptions
use ([`RP33/REHEARSAL.json`](../evidence/d3_k5_20260917/RP33/REHEARSAL.json)): campaign registered before any
read, both panels delivered whole, delivered digests equal to the characterised ones, availability UNDECLARED
in the delivery, and the date range refused at campaign registration by the holdout policy. The change is
refused by the tool itself unless the four previous lakes and every other key are byte-identical, and the
backup plus the restore command are written before anything is touched.

**Blocked:** `python tools/df_public_lake_adopt.py adopt --state-dir …`, which writes `5055.runtime.json`
(backup taken) and restarts `crispdm-data-gov.service`. The environment refused it as a shared-resource
change. I did not route around it: nothing was written to the production configuration and no service was
started, stopped or restarted by me.

## RP34 — the pilot preserved, its closure repaired

**Frozen first**: 131 original files with digests, sizes and times
([`RP34/PILOT_FREEZE.json`](../evidence/d3_k5_20260917/RP34/PILOT_FREEZE.json)); re-verified at the end,
unchanged.

The closure is new (`tools/df_e1_close.py`) and reports **four facts apart**, because they can hold apart:
metrics (recomputed from the unit's arrays, which must BE the prepared evaluation set — origins, labels and
denominator element by element, every value finite, no bool arrays, nothing empty); inference (the saved
weights reloaded in a **fresh process**, the windows regathered from DATA, predictions reproduced within a
declared tolerance); regime (judged on digests recomputed from the reloaded weights, with R1/R2's imported
detector required to BE that seed's auto-encoder bytes); and governance (whether a delivery, campaign and
accepted terminal existed **when the unit ran**). The population and the task come from the sealed design and
the prepared DATA, never from the record being judged.

Result on the preserved run, **without retraining anything**
([`RP34/E1_PILOT_CLOSE_REPAIRED.json`](../evidence/d3_k5_20260917/RP34/E1_PILOT_CLOSE_REPAIRED.json)):
15 of 15 units verified, 11 inferences replayed, 14 regimes verified, and **all 15 HISTORICAL_UNGOVERNED**.

The dictum's counterexamples now fail at the **full closing entry point**, on copies, with the originals
untouched (`tests/test_df_e1_close.py`, 17 rules): MAE forged to 999, MASE NaN, a bool metric, empty arrays,
foreign design/data/cell id, origins shifted by 123, missing weights and job, a foreign denominator, labels
from another horizon, updates past the ceiling, a relabelled regime, a transplanted unit and a missing one
named by id, a stranger attempt reported rather than absorbed, and a DESIGN forged with a zeroed digest **and**
with a recomputed one.

**Retrospective import.** The warehouse envelope did not have a way to say "this ran before its governance",
so it was extended additively: a `provenance` block with mode, `executed_at`, `imported_at`,
`governed_delivery_at_execution` and a reason, four additive columns, NULL on every row loaded before (NULL
means UNSTATED, never "prospective"), and refusals for every contradiction — a retrospective import that claims
a delivery, one stamped before the execution it imports, a confirmatory class, a missing or undeclared field
(`tests/test_retrospective_import.py`). Built from the closure, the import carries 15 units and 40 metric rows
and was loaded into a disposable cube twice: idempotent, and every row retrospective
([`RP34/RETROSPECTIVE_IMPORT_ENVELOPE.json`](../evidence/d3_k5_20260917/RP34/RETROSPECTIVE_IMPORT_ENVELOPE.json),
[rehearsal](../evidence/d3_k5_20260917/RP34/RETROSPECTIVE_IMPORT_REHEARSAL.json)). The deployed warehouse build
predates the block, so adopting it is the same guarded operation as the lake registration.

**F7, the donor's equivalence**, now compares every recorded fact plus the closure's own checks and replay
identity, not a nine-field list: the 1100 → 0 donor is excluded with `fields: ["updates"]`, and so are changed
parameters, curves, denominators, assignments, a different check outcome and a different attempt.

## RP35 — a contract that refuses what it cannot do

Domains are validated before the enumerator runs: window and step positive integers, horizon a non-negative
integer and **strictly positive for a forecast**, no booleans, split fractions in (0, 1] adding to at most the
data, roles as column names, known activation rule. `h = 0` is reconstruction — another task, which must be
declared as such, and whose trivial solution (copy the last input) is shown in the test so nobody mistakes it
for forecasting skill. `mask_ffill` is **refused** until a consumer exists: the loader would fill values
causally but no model here takes the input mask, and an unannounced fill in the tensor is worse than a refusal.
The scaler declares its **grain** (windows, where an interior row counts W times, or unique rows) and the test
shows the two give different moments. Labels are bound by identity: with every value equal but one, the label
is right because it is that row.

## RP36 — a criterion that is a criterion

The mask is now derived from the **window's own identity**: the validation bank is fixed (invariant to epoch,
batch order, resume and repeated evaluation) while the training noise still varies per epoch, as declared. The
dictum's own measurement, repeated on the real runner path: **zero mask positions change** and the loss moves
by less than 1e-9 with the weights untouched. Pre-training's internal validation is now a **purged tail of the
train origins** — disjoint from the DEV validation and never the test — with the bank's digest recorded per
unit, and the restored checkpoint reproduces the minimum it was chosen by. Reaching the update ceiling is
reported as **censoring** with the slope of the last third, the duration and an adequacy criterion; the phrase
"not truncated because the best epoch is not the last" is gone.

## RP37 — support, reach and capacity, kept apart

Measured, not asserted (`tools/df_e1_receiver.py`): perturbing each input row and taking the gradient per row
gives the reach of each core, and both agree with the declaration — **7 samples for the local core, 60 for the
dilated one**, whose dilations (1, 2, 4, 8, 16) are derived from W and not searched. The detector and adapter
of ARCH-A are preserved; the core is never pre-trained here.

Bounded DEV diagnostics with known answers, at two budgets:

| task (R² on DEV validation) | local core (reach 7) | dilated core (reach 60) | linear, whole window | linear, last 7 |
|---|---|---|---|---|
| distant lag (50 samples back), 400 updates | −0.57 | 0.78 | 0.9999 | −0.88 |
| distant lag, 1 600 updates | −0.71 | **0.93** | 0.9999 | −0.88 |
| near lag | solved | solved | solved | solved |
| unpredictable | not solved | not solved | not solved | not solved |

That is the distinction the order asked for: the local receiver fails at any budget because the information is
outside its reach; the full receiver's 0.78 was **budget**, not capacity, and 1 600 updates show it. The cost
of the dilated core is 1.77× the parameters and 1.26× the seconds per update, measured on the same windows.

## RP38 — the successor, sealed and not executed

Sealed design `143abb57…`
([`RP38/E1_PILOT_DESIGN_V2_SEALED_NOT_EXECUTED.json`](../evidence/d3_k5_20260917/RP38/E1_PILOT_DESIGN_V2_SEALED_NOT_EXECUTED.json)):
the same panel, target and horizon (W60_h60), the dilated core whose measured reach covers the declared
context, three paired seeds with R0/R1/R2 on one graph, `data_access: GOVERNED_DELIVERY`, and controls named by
the **information they use** — persistence (one sample), seasonal naive (a day back, outside the window, a
higher-information reference), ridge on the whole window, and ridge on the reach only, which is the control
that matches a local receiver's information.

The governed route is not a promise: it is executed end to end against a disposable data-gov plus a disposable
DuckDB cube (`tests/test_df_e1_governed_route.py`, 5 rules). One campaign and one delivery **per unit**, because
that is the shape data-gov completes; the first unit transfers and the next is served from the verified cache,
so transfer and reuse are measured; preparation refuses without a delivery; an absent host, an unserved
resource and bytes that changed after the confirmation all stop new work; and a unit's terminal reaches the
accounting through the outbox with the campaign reconciling clean.

Budget sheet from measurements, not from a promise
([`RP38/BUDGET_SHEET.json`](../evidence/d3_k5_20260917/RP38/BUDGET_SHEET.json)): **2 353 CPU s projected,
2 941 s with 25 % headroom**, from the per-update seconds measured for both cores on the same DEV windows. The
sealed run still measures its own cost pilot first.

## RP39 — the controller, the model and the real broker

`tools/e3_weekly_runtime.py` drives the deployed gym-fx environment with its broker plugin. The controller
**runs the selected model's own inference**; an action produced elsewhere is admitted only through an identity
check against that selection, and a foreign model's action never reaches the broker. Equity, commissions,
position and pending orders are the environment's own; an order in flight reserves its cash and carries its
side until the fill is observed. Zero latency is refused at construction; a non-finite price or equity refuses
the decision; short is refused and closing is explicit.

15 rules pass (`tests/test_e3_weekly_runtime.py`), including a fixture of **opposite models** that discriminates
(the same episode goes flat under the fallback and long when the model is released), a week change carrying a
position with continuous equity and commissions, a real gap with the broker's entry at the decision bar's next
open, insufficient cash, NaN and infinity, and five mutants **of the real path** that fail the same acceptance:
filling at the decision bar's close, no release check, flat turned into short, a controller that forgets its
outstanding order, and an identity check removed. Offline software: no venue, no account, no RL training, and
this is **not** a completed E3.

## RP40 — closure

**Tests.** `pytest tests docs/tres_temas_entrevista/program_v3/test_check_plan.py olap/store/tests --continue-on-collection-errors`:
**2 300 passed, 40 skipped, 3 failed, 8 errors** in 32 min 24 s
([`RP40_FULL_SUITE_SUMMARY.txt`](../evidence/d3_k5_20260917/RP40_FULL_SUITE_SUMMARY.txt)). The 3 failures
(`tests/integration_tests/test_configuration_handling.py`) and the 8 collection errors are the stale legacy
tests AGENTS.md documents; they are unchanged by this round and touch none of its code. The skips are
environment-gated and named in the summary: two storage rules of the retrospective import need `duckdb_engine`,
which this interpreter does not carry, and the same storage path is exercised instead against a disposable
warehouse service, whose report is the evidence. Every RP33–RP39 rule passes: closure 17, loader 30 (13 of them
new domain rules), pre-training 5, receiver and pilot 5, governed route 5, weekly runtime 15, weekly controller
8, retrospective import 9, effects and composition 26.

**Budget.** Systemd ledger on omega since this order
([`RP40_CPU_LEDGER_omega.json`](../evidence/d3_k5_20260917/RP40_CPU_LEDGER_omega.json)): **3 973 CPU s** over 65
transient units (the PRE reproduction, the lake rehearsals, the repaired closure with its fresh-process replays,
the retrospective import and its cube rehearsal, the receiver diagnostics at two budgets, the governed route,
the runtime tests, the full suite and this closure) of the 14 400 s authorised, leaving **10 427 s** unused. No
new training ran on the real panel: the only fits this round are the bounded receiver diagnostics RP37
authorises and the tests' own tiny fixtures.

**Hosts.** All work ran on omega (load 0.5, 20 GiB available at close, tree clean). The workers were not used:
the only distributable work of this round would have been the successor pilot's seeds, and that pilot did not
run. Workers are synced to the closing commit after this table.

**Defects of my own this round**, all fixed and in the history: the closure's first version compared parameter
counts against an unconfigured reload, so R1 units were refused for a regime the replay had not applied; the
closure's early refusals returned before the verdict was assembled; a block replacement dropped
`require_delivery` from the governed module; the first governed campaign shape used one campaign for every unit,
which data-gov cannot complete; the first rehearsal wrote its report inside the repository and dirtied the
checkout the code identity requires; and three RP37 mutants were too weak on their first run and were corrected.

Plan and state updated;
`check_plan` passes 21/21 with `scientific_approval` false, the five fronts, six proposals, weekly E3 and
H-CORE unchanged.

## Request

One review of RP33–RP40: the bounded lake contract and its rehearsal, the repaired closure and what it now
refuses, the retrospective import's provenance, the loader's domains, the fixed validation bank, the receiver's
measured reach against its learned capacity, the sealed successor with its governed route and budget, and the
weekly runtime on the real broker.
