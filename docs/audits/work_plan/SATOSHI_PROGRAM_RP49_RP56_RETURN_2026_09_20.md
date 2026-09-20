# Satoshi — return of RP49–RP56: the successor E1 ran, closed and reconciled

Order: [MUSASHI_PROGRAM_RP49_RP56](../../handoffs/MUSASHI_PROGRAM_RP49_RP56_2026_09_20.md), dictum
[MUSASHI_RP41_RP48_REVIEW](MUSASHI_RP41_RP48_REVIEW_2026_09_20.md). Base `f83d635`. Ending state:
`RP49_RP56_EXECUTED_E1_SUCCESSOR_RUN_CLOSED_AND_GOVERNED_READY_FOR_REVIEW`.

**What happened.** The external provider was repaired and rehearsed, the public-panel resource was
adopted in production, and the sealed E1 successor then ran end to end on that route: sixteen units,
all closed, fifteen cells `VERIFIED_AND_GOVERNED`, the warehouse matching the client's receipts by
content, 2 535.5 CPU seconds of an 11 000-second cap. Its result is negative and narrow: modular
pretraining did not help, and a competent linear control beats all three regime means.

**Five defects of my own appeared only because the route was finally executed for real**, not
mocked. Every one is repaired, with a mutant that reopens it. I record them as findings against my
own work, in the order they bit.

## PRE

The dictum's reproducer was run on the reviewed revision before any edit; its outputs are
field-for-field the reviewer's
([`RP49_PRE/`](../evidence/d3_k5_20260917/RP49_PRE/)): the first child refusing its own result, the
runner leaving seven of eight campaigns open while declaring all governed, local declarations passing
as GOVERNED, the missing rehearsal, the broker's false ALREADY_LONG, and the noise-floor reference.

## RP49 — the provider that actually serves the route

The deployed external provider could not serve these two archives: it parsed `available_time_column`
unconditionally ("unparseable time column") and refused a whole resource under a holdout. The
correction was made **in financial-data**, not by silently substituting the embedded kernel:
`governed_download` now has an untimed branch that returns the whole archive AS_IS with an
UNDECLARED scope and `UNKNOWN` availability, and raises for any range — before a holdout date as
well as after, so no accidental date lets an earlier range through
(`store/tests/test_archive_retrospective_delivery.py`, 4 rules).

The rehearsal runs installed provider → external host → data-gov → client → a disposable DuckDB
warehouse, and is bound to the bytes that serve it (deployed configuration with the token excluded,
panel digests, provider and serving-tool digests):
[`RP49/REHEARSAL_EXTERNAL_HOST.json`](../evidence/d3_k5_20260917/RP49/REHEARSAL_EXTERNAL_HOST.json) —
`route_through: EXTERNAL_HOST`, both panels delivered (`VERIFIED_TRANSFER`, 10 890 295 bytes,
`b3192c0b…`), availability `UNKNOWN`/`UNDECLARED`, ranges refused, cache re-verified, both units of
the campaign closed including a probe that deliberately fails, and the terminals read back from the
warehouse **by content** through the canonical reader.

## RP50 — the first child of a fresh run

`run_isolated` asked the closure for a verdict before writing the parent record, so the first child
of a fresh run was refused while a copy that already had one was accepted. Phases are explicit now
(`FRESH`, `RESUME`), the same verifier serves fresh, resume and closure, and a provisional record is
never read as a score. `tests/test_df_e1_first_child.py` drives the **real** subprocess and verifier
on a small but genuine AE and fit: weights written, predictions stored, the update counter running,
survival across reload and resume, refusals with cost for a failed child, an incomplete result and an
altered score, and a worker that operates from its delivery alone with the design's panel path
removed.

## RP51 — the population, and receipts produced by the real route

Every unit of the sealed design has a state at all times (`NOT_STARTED`, `RUNNING`, `CLOSED`,
`BLOCKED`, `PENDING`), and a registered unit that did not close is **named**, never absorbed into a
total. Receipts are persisted from the client's own answer when the service accepts a terminal —
never synthesised from a summary. The produced receipts are published
([`RP51/DELIVERIES.json`](../evidence/d3_k5_20260917/RP51/DELIVERIES.json),
[`RP51/TERMINAL_RECEIPTS.json`](../evidence/d3_k5_20260917/RP51/TERMINAL_RECEIPTS.json)), so the
positive of RP52 is now the product of the real route and no longer skips.

## RP52 — the authoritative closure and the chronology

GOVERNED is a set of facts read from the receipts' **contents**: campaign, design, actor, resource,
delivery, a valid terminal state with the digest the service returned, and a complete reconciliation,
with the delivery bound to the recorded **start of the work**, not merely to a later acceptance.
Arbitrary local files do not grant it. The negative cases (absent campaign, another unit, impossible
state, absent payload, absent lists, delivery after the work, altered terminal) all refuse.

## RP53 — RL: the terminal without a fill

A terminal verdict with no fill now releases or updates the pending state from the broker's own
events by `order_ref`, never by elapsed time or an assumed position; the decided quantity reaches the
broker through the environment's execution plugin; the minimum executable latency is 2 bars,
measured. None of this is profitability or a trained RL policy.

## RP54 — the ML reference and the diagnostic's provenance

The noise floor is developed analytically and checked numerically: copy reference vs the Bayes
predictor on the noisy input vs the latent oracle, with units and assumptions stated, amplitude/power
SNR and dB labelled apart, and finite-error separated from theory
([`RP54/INNOVATION_WITH_REFERENCES_dragon.json`](../evidence/d3_k5_20260917/RP54/INNOVATION_WITH_REFERENCES_dragon.json)).
No data or training was changed to meet a retrospective threshold, and the evidence of independent
information does not prove that every short receiver fails on real data.

## RP55 — the adoption, and then the run

The whole-run rehearsal ran three disposable services, a declared synthetic panel and **real
training**, with a positive, a genuine failure (a 48 MiB task-memory ceiling produced
`COST_PILOT_FAILED: pilot_ae RESOURCE_EXCEEDED` with the population named) and a verified restoration
that left production untouched
([`RP55/FULL_RUN_REHEARSAL.json`](../evidence/d3_k5_20260917/RP55/FULL_RUN_REHEARSAL.json)).
The adoption then succeeded in production with backup, verified rollback path and effective identity
checked — not merely a 200 — and only the one service it needed
([`RP55/ADOPTION_RECEIPT.json`](../evidence/d3_k5_20260917/RP55/ADOPTION_RECEIPT.json), `adopted: true`).

### The run

`satoshi-e1-successor-20260920`, design `143abb57…` (the successor re-seals to the identical digest),
host omega, panel delivered per unit from `public_panels`.

| | |
|---|---|
| units declared / closed | 16 / 16 |
| cells `VERIFIED_AND_GOVERNED` | 15 / 15 |
| closure verdict | `ALL_VERIFIED` (fresh-process replay, 512 windows per unit, tol 1e-5) |
| warehouse by content | 16 / 16 match the client's receipts, no stranger |
| CPU | 2 535.504 s of an 11 000 s cap |

The numeric ML sheet — windows and overlap, volume per split, missing and exclusions, parameters and
activations, reach **measured** not asserted, target, denominator, early stopping, observed budget and
the censoring criterion — is
[`RP55/E1_SUCCESSOR_ML_SHEET.md`](../evidence/d3_k5_20260917/RP55/E1_SUCCESSOR_ML_SHEET.md). Its
quantities come from the sealed design and the prepared DATA, both of which predate the first fit;
the document was written afterwards and says so.

**Result (validation MASE, train-only denominator 0.61626):**

| regime | n | mean | sd |
|---|---|---|---|
| R0 | 3 | 0.8875 | 0.0127 |
| R1 (frozen detector) | 3 | 0.9034 | 0.0175 |
| R2 (fine-tuned) | 3 | 0.8968 | 0.0236 |
| **linear ridge control** | — | **0.8852** | — |
| persistence / seasonal daily | — | 1.0018 / 1.1873 | — |

Paired: R1 − R0 **+0.0159**, R2 − R0 **+0.0093**, R2 − R1 −0.0066. AE cost 137.156 s total, 22.86 s
amortised per consuming fit. **Convergence is not declared**: four of the nine fits stopped at the
update ceiling and are `CENSORED_BY_BUDGET`, so their comparison is a lower bound. Three seeds on one
task are development evidence — not H1, not equivalence, not a confirmatory reserve.

### Five defects of mine, found by executing the route

1. **Concurrent receipt writes erased units.** `DELIVERIES.json` and `TERMINAL_RECEIPTS.json` were
   read-modify-written, so three parallel children overwrote each other and the run stopped with
   `REFUSED: unit 'R0_s2' has no delivery` after seven children. Both writes are merged under a lock
   and replaced atomically (`tests/test_df_e1_receipt_concurrency.py`, real concurrent processes).
2. **One shared terminal spool.** Each child flushed the others' envelopes through its own campaign's
   client and raced on the same file (`FileNotFoundError` mid-send). Each unit reports through its own
   spool; a shared spool is flushed by one process at a time and an envelope another flusher already
   delivered is named, not counted as a failure.
3. **`prepare` rebuilt on every resume**, piling up four envelopes the service rightly refused 409
   against the generation it had already accepted. It now reuses the accepted terminal, like the cells.
4. **The chronology compared text, not instants.** `…19:01:29Z` and `…19:01:29+00:00` are the same
   instant; as strings the second sorts first, so ten fully governed units were demoted to
   `HISTORICAL_UNGOVERNED` on a spelling. Both are parsed now; a real inversion in either direction is
   still refused.
5. **The table would have described the aborted attempt.** A report is written once, so the completed
   resume landed as `REPORT.<epoch>.json` beside the stopped run's `REPORT.json`, which the closure
   preferred. It now takes the report of the run that ended without stopping, refuses when two claim
   it, and records the choice in `RESULTS.json`.

The damaged root was repaired by `tools/df_e1_recover.py`, which copies what the **service already
holds** — campaign, verified delivery with the bytes checked on disk, and an accepted terminal only
when the campaign reconciles live — and **never invents a terminal**: `ae_s3` stayed the runner's
debt until the runner reported it
([`RP55/E1_RECEIPT_RECOVERY.json`](../evidence/d3_k5_20260917/RP55/E1_RECEIPT_RECOVERY.json)).

## RP56 — closure and faithful state

Files → parent → accounting → warehouse, by content and by population, rehearsals and failures
included: [`RP56_WAREHOUSE_CONTENT_CHECK.json`](../evidence/d3_k5_20260917/RP56_WAREHOUSE_CONTENT_CHECK.json)
(16/16 `MATCHES_THE_CLIENTS_RECEIPT`, no problems, no stranger). This round's five guards each die
under their own mutation: [`RP56_MUTANTS_POST.json`](../evidence/d3_k5_20260917/RP56_MUTANTS_POST.json)
(5 killed of 5; the first M14 mutation survived because it did not reproduce the defect it named, and
was corrected).

`PROJECT_METHOD_STATE.json` now distinguishes IMPLEMENTED, EXECUTED, VERIFIED and
EXTERNALLY_REVIEWED, and MOD-E1 is **VERIFIED, not externally reviewed** — nothing in this round has
been seen by a reviewer. `09_ADOPCION` records that the adopted resource was consumed by a real run.

### One operation denied, recorded exactly

Adjudicating the four duplicated `prepare` envelopes in the shared outbox was refused by the
execution environment's auto-mode classifier as *Logging/Audit Tampering*. It was not evaded and not
worked around. The refusal is the environment's, not data-gov's and not the owner's; it does not
excuse any defect, and it blocks nothing: the envelopes remain visible in `pending/` with their
failure sidecars, and the run is complete without them. **This is the one thing I will ask for.**

### PRE / POST

[`RP49_RP56_PRE_POST.txt`](../evidence/d3_k5_20260917/RP49_RP56_PRE_POST.txt) sets the reviewer's
frozen outputs against this code: local claims fall to `HISTORICAL_UNGOVERNED` with their reasons
named, the failing runner leaves two registered units both closed and six named `NOT_STARTED`
instead of eight registered with seven open and `all_units_governed: true`, an empty report no longer
passes as a total, and the unit the services really accepted closes `GOVERNED`. Two things are
reported rather than dressed up: the reviewer's script **no longer runs unmodified** (the population
now needs the sealed design, so `_governed_summary` takes it — two lines marked `# ADAPTED`, nothing
else touched), and the `fresh_child_verification_order` probe is **unchanged**, because it calls the
closure directly and cannot see a repair that is about the order `run_isolated` does things in; what
shows that repair is `tests/test_df_e1_first_child.py`. 131 historical originals: 0 changed.

## Costs and scope

The full suite, its command, its environment and every skip explained:
[`RP56_FULL_SUITE_SUMMARY.txt`](../evidence/d3_k5_20260917/RP56_FULL_SUITE_SUMMARY.txt) —
**3 failed, 2 289 passed, 39 skipped, 8 collection errors**, where the 3 failures and 8 errors are
the repository's documented stale legacy suite (AGENTS.md) and no work of this round touches them.
Two of my own changes broke rules that had to be corrected rather than excused: the eight adoption
rules that copied a configuration which is now adopted (they build the pre-adoption configuration
explicitly, and a new rule pins the state the host is in), and the RP32 rule that *characterised*
the outbox race, which now states the guarantee the repair gives.

Cost per host: omega **6 676.35 CPU seconds** of the 14 400 ceiling
([`RP56_CPU_LEDGER_omega.json`](../evidence/d3_k5_20260917/RP56_CPU_LEDGER_omega.json)). The two
workers' cost is **UNMEASURED and declared so**
([`RP56_CPU_LEDGER_workers.txt`](../evidence/d3_k5_20260917/RP56_CPU_LEDGER_workers.txt)): their
jobs did not run inside an accounted scope, so no figure exists and I will not estimate one. That
gap is mine, and the fix — routing worker jobs through the same wrapper — is not done in this round.
Backlog at the close: the four duplicated `prepare` envelopes, nothing else pending anywhere.

No GPU, no
live, no venue, no scientific RL training and no confirmatory reserve; E0 was not repeated and the
historical pilot was not retrained; no healthy service was restarted for an unrelated block and no
served checkout was changed. Synthetic fixtures are declared as such wherever they appear.

H-CORE stays after E1 and a verified frozen prefix. Forecasting and the weekly RL cycle remain
obligatory, the five fronts and thirteen signal steps are not replaced by this pilot, and the index,
Metabase and terminology fronts stay where they are.

One review is requested of this round. Nothing here is approved in advance.
