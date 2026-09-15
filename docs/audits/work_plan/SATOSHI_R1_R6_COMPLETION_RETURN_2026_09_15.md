# Return: the completion order of 2026-09-15

Order: `docs/handoffs/MUSASHI_WORKER_ACTIVATION_AND_R1_R6_COMPLETION_2026_09_15.md` (`d171de2`).
Counts are mine and await independent verification. No production service was restarted by me,
no cube history altered, no real archive policy installed, no GPU used.

## Revisions

| repository | branch | head |
|---|---|---|
| predictor | `satoshi/r1-r6-20260914` | `5c1d418` |
| agent-multi | `satoshi/moltbook-answer-every-reply-20260914` | `e3c63bca` |
| feature-eng | `satoshi/r1-r6-20260914` | `df7ea5b` |
| preprocessor | `satoshi/r1-r6-20260914` | `ad1a313` |
| feature-extractor | `satoshi/r1-r6-20260914` | `4ba9595` |
| financial-data | `satoshi/ethusdt-4h-contract-successor-20260914` | `d0f81490a` |

## Requirement → test → evidence

| § | requirement | where it is tested | evidence |
|---|---|---|---|
| 2 | probe completes the outcome, no open campaigns | the probe refuses on `missing_units` | `worker_identity_20260915/unit{1,2}-{gamma,dragon}.json` |
| 2 | one governed delivery per worker, own key | run on each machine | actor `satoshi-gamma` / `satoshi-dragon` in `governed_deliveries.actor` |
| 2 | accounting actor matches the worker | read from data-gov's store, not asserted | same |
| 2 | cache reuse in a second unit | second unit per worker | `VERIFIED_CACHE`, same digest, nothing transferred |
| 3 | observed steps and updates published | `rl_pipeline._observed_work` from the model | `REPLAY_LOG_PROD{12,13}.json` |
| 3 | requested / observed / duration distinct | `test_governed_replay_custody.py` (8 rules) | prod-13: 64 requested, 256 observed, 4.4 s |
| 3 | the governed file is really read | `test_governed_input_is_consumed.py` (6 rules) | digest, row ids and values vs delivered bytes; planted decoy |
| 3 | flat vs nested regression retained | first rule of that file | — |
| 3 | one corrected production replay | `doin-offline-replay-prod-13` | COMPLETED, reconciliation empty, fresh directory |
| 4 | disposable provider → host → data-gov → receipt → warehouse | run on gamma | `archive_route_20260915/R4_ROUTE.json` |
| 4 | UNKNOWN survives in persisted evidence | contract digest in two stores | `62965feb…` in `governed_deliveries` and `gov_terminal_dataset` |
| 4 | ranges refused for the declared reason | same run | HTTP 422, refusal naming the retrospective archive |
| 4 | existing synthetic contracts keep working | same run | plain resource delivered 200 |
| 5 | caller, schema, design, evidence, producer | inventory | `SATOSHI_P1LR_DECISION_GATE_INVENTORY_2026_09_15.md` |
| 5 | do not fabricate | nothing produced | gate left refusing, units left failed |
| 6 | rights research with dated source and scope | primary sources read | `SATOSHI_MARKET_DATA_RIGHTS_RESEARCH_2026_09_15.md` |
| 6 | capability tokens deferred | not implemented | the present distinction is documented below |

Suites actually run this round: agent-multi 17 (custody 8, consumption 6, outbox 3),
predictor 81, feature-eng 44, feature-extractor 46, preprocessor 40.

## The three findings that matter

**A budget was being reported as work done, and the separation caught it immediately.** The
first receipt with real counters reported **10,240** observed environment steps against a
configured **64** — the agent plugin resolves its parameters from the top level of the
configuration and the runner wrote the budget only under `training`, so the plugin used its own
default of 10,000. The old receipt would have said "64" and looked correct. After the fix,
prod-13 observes 256 steps and 10 updates with wall time down from 15.8 s to 4.4 s.

**256 against a 64 budget is still a factor of four and is NOT explained here.** The resolved
configuration names no agent, environment or pipeline plugin, so defaults decide, and the
relation between a `learn()` budget and counted environment steps depends on that plugin. No
interpretation was written into the receipt.

**UNKNOWN survives by reference, and the limit of that is reported.** The archive contract's
digest appears in both the accounting and the cube, and it can only come from a contract whose
lag is the string `UNKNOWN`. But the cube stores the **digest**, not the lag: reading the cube
alone, nobody can tell that a delivery was a retrospective archive without resolving it. That
is a gap in the persisted schema, stated rather than smoothed over.

## Attribution today: what data-gov knows and what the store knows

Documented as §6 asks, without redesigning anything. **data-gov** records the actor, the
campaign, the unit, the resource, the delivered digest and the verification state per delivery:
that is where "who ran what" lives, and it is now per machine. **The store hosts** authenticate
with a single shared service token issued to data-gov; their own logs therefore identify
*data-gov*, not the experiment. Nothing in this round depends on closing that gap, per-delivery
capability tokens remain a deferred proposal, and no concrete attribution failure has been
observed — if one appears it will be submitted on its own, not smuggled into a closure.

## What remains open

| item | owner | exact state |
|---|---|---|
| the factor of four between requested and observed steps | Satoshi | needs the selected plugins named and their step semantics measured |
| the cube cannot say "archive" without resolving a digest | Satoshi / schema | persisted evidence carries the digest only |
| the 16 screen records and the replica proof for `p1lr-decision@202` | the owner of that front | producer commands named in the inventory; nothing fabricated |
| verbatim clauses of the Product Terms of Use | Satoshi | the page renders client-side; a paraphrase is not a clause |
| per-delivery capability tokens | deferred | architectural proposal, not a dependency |

R1–R6 is **not** declared complete: the four items above are named, with their owners, and the
first two are mine.
