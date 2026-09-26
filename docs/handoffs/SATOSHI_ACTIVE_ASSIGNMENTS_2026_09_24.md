# Start here: the one dispatch index, and the lanes it carries

**Superseded as a state document, 2026-09-26.** State now lives in ONE place, one row
per lane: [`EXPERIMENT_EXECUTION_QUEUE.json`](../tres_temas_entrevista/program_v3/EXPERIMENT_EXECUTION_QUEUE.json)
(`research_dispatch_index.v2`), summarized in
[`SATOSHI_DR02_DISPATCH_INDEX_2026_09_26.md`](../audits/work_plan/SATOSHI_DR02_DISPATCH_INDEX_2026_09_26.md).
Nobody should have to infer a lane's state from prose again, this index included: where
the table below and a row disagree, **the row governs**. Everything under
"Current assignments (2026-09-24 snapshot, history)" is a dated photograph.

## The lanes, at a glance

Eleven lanes have a current row: the ten DR02 names, plus `E1-Q2-CONTEXT`, which was
added because it holds the only live compute in the programme.

| lane | real dependency, from artifacts | next eligible work |
|---|---|---|
| A/B | none outstanding | nothing to dispatch. **Do not reactivate**: the twelve-cell queue finished |
| R0/R1/R2-ECL | none | nothing to dispatch. **Do not reactivate**: closed 9 of 9 on 2026-09-26T01:28:52Z |
| E1-Q2-CONTEXT | two: the shared admission guard itself (the gate logs `crispdm_run_would_accept_the_cap` false while the request fits available memory plus margin), and a data-gov service key for any cell to be governed — the issuing script is named, not waited on | DR01 atomic reservation; smoke delivery → terminal on one small unit. The attempt closed at its own boundary; six of eighteen units were never admitted |
| H-CORE | MOD-FROZEN-PREFIX's own deliverable — the materialized prefix **output** with real parity. Not the scrambled-label floor, which is withdrawn | materialize that output and its four parity/causality tests. CPU, no reserve |
| M4 | six named verifier repairs, and one record the executor must not write | freeze the audit's probes as PRE, turn them into tests of the real public path, then repair. **Not authorized to run CONFIRMATION** |
| FIN-LOSS-OPT/E3 | a registered availability contract for the EURUSD 1h resource; BUSINESS-CONTRACT executed. **H-CORE is not a blocker of this lane as a whole** | register the resource and re-attempt the sealed cost pilot; prepare the weekly reference and protocol |
| calendar | a consensus source that also carries an observed publication instant — an owner entitlement decision, named | prospective point-in-time capture; extend absence/placebo reporting. UNKNOWN stays UNKNOWN |
| M5PHET five families + chat | one per family, and none of them is another family | six items that may run at once, one per family plus the chat router repair; three bookkeeping contradictions to fix |
| news | a licensed feed entitlement the package does not hold | the pinned-weight smoke **with a retained receipt**, which also settles a README-vs-state contradiction |
| MT5 demo | a readable copy of the off-host bridge store, then an operator attestation | dry-run the migration against that copy; reconcile "never placed an order" with the remote 48/5 |
| Alpaca paper | nothing for monitoring; for progress, the commit-pinned deployment F3 requires | that deployment with rehearsal, window and rollback; adoption tests; a zero-network shadow sink |

Five rows could not establish a **remaining cost** from artifacts and say so by name:
news, MT5 demo, Alpaca paper, M5PHET and calendar.


**2026-09-26 successor:** [DR01-DR08](SATOSHI_DAY_REVIEW_CONTINUATION_2026_09_26.md)
and the [independent day review](../audits/work_plan/MUSASHI_DAY_REVIEW_2026_09_26.md).
The dated snapshots below are history, not a current dispatch manifest. Reconcile
the published supplements and live leases before launching. M5PHET remains a
parallel product lane, not a replacement for the scientific/financial programme.
Do not launch new heavy coordinator work until shared memory admission is repaired;
independent admitted worker/application work continues. No M4 confirmation approval
is granted by the day review.

**Owner clarification:** the public M5PHET interface must match Laya's experience:
data/context and natural-language questions in, task-specific structured answers
out. This is required across the five families, not a later optional frontend.
See the updated [current delivery order](SATOSHI_M5PHET_CLASSIFICATION_FIRST_2026_09_24.md).
Do not stop or duplicate the real Laya pilot while adding this interface.

**Current priority after `e36c1c6d`: [classification-first and external providers](SATOSHI_M5PHET_CLASSIFICATION_FIRST_2026_09_24.md).**
Deliver real Laya relevance classification through M5PHET with native parity.
Implement other family adapters in parallel in their existing owner repos.
RP158 retained numbers are consistent; remaining validator maintenance is CPU-only,
without another checkpoint pass and without holding the application lane.

**Prior `3c25ecab`: [RP157 review and bounded completion](../audits/work_plan/MUSASHI_RP157_REVIEW_2026_09_24.md).**
Retained metrics are consistent; finish canonical authority and expected-population
validation without another routine model replay. Laya setup is reported underway;
collector/calendar/broker CPU lanes proceed during installation.

**Latest `b3a064d5`: [RP156 review and execution priorities](../audits/work_plan/MUSASHI_RP156_REVIEW_2026_09_24.md).**
Retain corrected development scores; repair remaining acceptance without routine
re-inference. Laya, news, calendar and broker implementation are still due in parallel.

**Prior return `ebe35fb3`: [RP155 review and application continuation](SATOSHI_RP155_REVIEW_AND_EXECUTION_2026_09_24.md).**
The contrast has now finished training and has reported scores; its scoring closure
requires a successor, not retraining. RP147-RP150 remain unstarted in that return.
The earlier in-progress snapshots below are preserved history, not current dispatch.

Owner requested a consolidated reminder on 2026-09-24. This is the front page
for the existing orders, not another experiment or replacement work plan.
Satoshi implements and dispatches; Musashi reviews. No routine permission to
continue is needed within the existing assignments and budgets.

## Evidence and freshness

Checked fetched predictor branch `satoshi/rp132-rp134-20260923` at `b75097d3`.
The original RP144-RP151 orders (`d4bb3353`), their continuation (`32a5467d`),
optional integration orders (`d46d3363`) and prior review (`af8ca197`) are
ancestors of that branch. They were not replaced by the M5PHET work. The newest
review `032aaa52` was NOT yet an ancestor when checked: incorporate its scoped
orders together with this index, without overwriting newer work.

Sources read at `b75097d3`:

- `docs/audits/work_plan/SATOSHI_RP144_RP151_RETURN_2026_09_24.md`.
- `docs/audits/work_plan/SATOSHI_M5PHET_STATUS_2026_09_24.md`, including updates.
- `docs/audits/evidence/d3_k5_20260917/RP145/CLOSURE_RECONCILIATION.md`.
- The owner's latest report of corrected contrast progress, 6,138/14,400 CPU s.
- Fetched news-signal still at `d9b22e4`; no newer remote implementation observed.

These are published/report snapshots, NOT a fresh reading of running processes,
accepted warehouse content, account entitlements or unpublished worktrees. Check
those before dispatch; absence from a return is not proof no local work exists.

## Current assignments (2026-09-24 snapshot, history)

Preserved for its detail, not for dispatch. Read the row first.

| Order / lane | Last evidenced state | Remaining action, without repeating completed work |
|---|---|---|
| RP144, protocol A | Corrected 12-cell aggregate reported; MSE/MAE 0.161962/0.259662 retained | Preserve the qualified device provenance and existing accepted evidence. No retraining; only outstanding audit reconciliation if identified. |
| RP145, protocol B | Closure REPORTS 12/12 verified and bit-exact, MSE/MAE 0.150307/0.246397; 16,167/24,000 CPU s | Deliver/reuse closure evidence for independent review, not another fit queue. Table 9 searched-lookback ambiguity stays explicit. The old 10/12 queue is historical, not dispatch authority. |
| RP146, doctoral R0/R1/R2 | Corrected successor reported on second seed; first seed has saved/restored checkpoints and observed updates | Inspect actual lease, child, completed cells and residual allocation. Continue only missing work in that successor, then fresh-process replay, full-population scoring, same-row naive and content reconciliation. Preserve superseded diagnostic; no duplicate or automatic budget reset. |
| RP147, real Laya | SDK/contract/source work exists; no real-weight pilot delivery in inspected returns | Complete the pinned SDK/checkpoint pilot and direct-SDK vs M5PHET provider/consumer parity. Reuse upstream, measure memory/latency, preserve uncalibrated status. Existing 600 CPU s / 900 wall s smoke allocation applies after setup. |
| RP148, news collection/shadow | Not started in inspected return; real feed entitlement not verified there | Implement and test receipt clocks, dedup/revisions, persistent queues, stale/missing input behavior now. Check only documented entitlement references. Real prospective capture needs the actual entitlement; resident classification depends on RP147. No broker mutation in shadow. |
| RP149, MT5 demo and Alpaca paper | Not started in inspected return; no actual broker acknowledgement/fill evidence | Implement policy/risk/execution interfaces and negative/recovery tests independently of weights. Actual canaries require the existing mandate, confirmed demo/paper account and prerequisites. Report each broker separately; never bypass risk or substitute real-capital credentials. |
| RP150, finance | Historical resource availability evidence unresolved; matched financial reference absent | Continue producer/source investigation and reference preparation without private-reserve reads. UNKNOWN is not zero lag. Retain FIN-LOSS-OPT (MAE/Huber x Adam/AdamW) as mandatory but do not fit through an unmet data prerequisite. News/calendar work need not wait for this resource. |
| RP150, calendar/domain adapters | CAL01-CAL12 and real remaining family engines not delivered in inspected status | Inventory the existing economic dataset and vintages; implement causal as-of transforms and their negative tests. Thin adapters over existing engines, not invented generic models. Later scientific ablations retain their own design/data/budget gates. |
| M5PHET runtime + INT | d9ffc4d repaired cases reviewed; three specific gaps remain. Real governance/DOIN round trips not delivered | Apply the latest scoped review, complete real provider integration in parallel, then optional governance and DOIN round trips by dependency. Core/apps remain usable without the stack. Do not claim supported combinations alone establish model capability. |
| RP151, consolidated return | Existing returns describe partial progress, not completion of all lanes | One status row for EVERY lane above, including completed, running, ready, dependency-held and externally unavailable work; evidence and next action per row. Do not say all work is done when only M5PHET tests finished. |

## Immediate sequence and parallel ownership

1. Read current services/leases, per-cell records, outboxes, worktrees and remaining
   budgets. Do not replay the stale queue. Incorporate newer legitimate work before
   deciding what is absent. Refresh the execution queue by evidence, not filenames.
2. Keep the corrected admitted contrast moving. B's reported completion is not a
   reason to rerun B. External RTX 5090 is first choice for eligible NEW GPU work;
   do not migrate sealed running cells or evict a valid experiment. The bounded
   Laya residency pilot may use its already authorized admitted host candidate.
3. Split CPU/code lanes using available Hermes/subagents when useful: framework
   boundary repairs; real Laya adapter/pilot; collector/calendar; broker integration;
   financial source/reference. Disjoint worktrees, one integrator, no secrets in
   prompts. Model downloads/admission and CPU work can proceed independently of
   a GPU lease. Do not wait for all five ML engines to deliver the first useful app.
4. Apply the existing [d9ffc4d review](../audits/work_plan/MUSASHI_M5PHET_D9FFC4D_REVIEW_2026_09_24.md)
   as a narrow continuation. Respect dependency holds on the affected output path,
   not a global experimentation hold. There is no new compute budget in this index.
5. Report measurable progress, not just test counts: experiments with declared
   metric/scale/error/same-row naive/literature/comparability; real Laya identity,
   memory/latency/output parity; feeds and brokers with their actual evidence.
   Where no new measurement exists, say so. Never turn an unobserved job into a
   finished one or an unavailable entitlement into permission to invent access.

Detailed specifications remain in
[RP144-RP151](MUSASHI_RP144_RP151_NEWS_AND_EXPERIMENTS_2026_09_24.md),
[its continuation](SATOSHI_RP144_RP151_CONTINUATION_2026_09_24.md), and
[M5PHET implementation](SATOSHI_M5PHET_IMPLEMENTATION_2026_09_24.md).
Deferred master-plan stages (including later core pretraining and financial/RL
confirmatory work) remain in the plan; this index does not authorize launching
them ahead of their prerequisites or manufacture a new allocation.
