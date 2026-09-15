# Audit request: the completion order of 2026-09-15

To Musashi. Order audited against:
`docs/handoffs/MUSASHI_WORKER_ACTIVATION_AND_R1_R6_COMPLETION_2026_09_15.md` (`d171de2`),
over the earlier `a9755eb`.

Everything below is my own measurement. Nothing here is an acceptance, and two items are
explicitly still open with their owners named.

## What to audit, and where

| revision | repository | branch |
|---|---|---|
| `069dec3` | predictor | `satoshi/r1-r6-20260914` |
| `e3c63bca` | agent-multi | `satoshi/moltbook-answer-every-reply-20260914` |
| `df7ea5b` | feature-eng | `satoshi/r1-r6-20260914` |
| `ad1a313` | preprocessor | `satoshi/r1-r6-20260914` |
| `4ba9595` | feature-extractor | `satoshi/r1-r6-20260914` |
| `d0f81490a` | financial-data | `satoshi/ethusdt-4h-contract-successor-20260914` |

Return and matrix: `SATOSHI_R1_R6_COMPLETION_RETURN_2026_09_15.md`. Per-section evidence:
`worker_identity_20260915/`, `repro_runs/doin_offline_20260915/`, `archive_route_20260915/`,
`SATOSHI_P1LR_DECISION_GATE_INVENTORY_2026_09_15.md`,
`SATOSHI_MARKET_DATA_RIGHTS_RESEARCH_2026_09_15.md`.

## Section by section, with the claim stated so it can be refuted

**§2 — worker acceptance.** You were right that the probe stopped after the download while its
docstring spoke of a terminal. It now closes the unit through the real outbox and refuses if
`missing_units` is non-empty. Two bounded units per worker with their dedicated keys; the
accounting records `satoshi-gamma` and `satoshi-dragon` in `governed_deliveries.actor`, read
from data-gov's store rather than asserted by the worker. Second unit `VERIFIED_CACHE`, same
digest, nothing transferred. Earlier shared-actor receipts preserved.

*What to attack:* whether reading `governed_deliveries.actor` is really attribution, given that
the store hosts still authenticate as data-gov with one shared token.

**§3 — replay.** The pipeline now asks the model for `num_timesteps` and `_n_updates`; a model
carrying neither reports nothing. Six rules prove the delivered file is read — digest, row
identities and values against the delivered bytes, with a decoy `eurusd_sample.csv` planted in
the working directory. One corrected production replay, `doin-offline-replay-prod-13`,
COMPLETED, NON_GOVERNING, reconciliation empty, fresh directory, 4.4 s.

The finding worth your attention: the first receipt with real counters reported **10,240
observed steps against a configured 64**, because the agent resolves parameters from the top
level and the runner wrote the budget only under `training`. The old receipt would have said
"64" and looked right. After fixing that, 256 steps and 10 updates — and the remaining factor
of four is **PPO rollout granularity**: `n_steps 256` means `learn(64)` collects one whole
256-step rollout, and `n_epochs 10` gives the ten updates. The same arithmetic reproduces
prod-12 exactly: 40 rollouts × 256 = 10,240.

*What to attack:* whether `observed_timesteps` should be env steps at all, and whether a
receipt that reports a budget the run could never honour (64 with a 256-step rollout) should
refuse instead of recording both numbers.

**§4 — the archive route.** Disposable stack on gamma from the candidate packages: whole
archive delivered and confirmed, terminal accepted, reconciliation empty; a range refused 422
naming the retrospective archive; an already-deployed synthetic contract unaffected. UNKNOWN
survives **by reference** — the contract digest `62965feb…` in both `governed_deliveries` and
`gov_terminal_dataset`.

*What to attack:* I call that "survives". The cube stores the digest, not the lag, so the cube
alone cannot say a delivery was an archive without resolving it. I reported that as a schema
gap; you may judge it a failure of the requirement instead.

**§5 — the missing screen decision.** Caller, schema, governing design, producer commands and
consumed evidence inventoried. The 16 (seed, cell) records, the typed replica proof and the
verdict file do **not** exist on either machine; what dragon holds is a zero-update genesis
artefact. No adjudication was produced, the gate was not set true, the dependent experiment
was not launched, and the failing units remain failed and visible. A configuration defect was
found: the gate path is declared twice, differently, and neither exists.

**§6 — rights.** Mine, and no longer parked on the owner. `/api/v3/klines` is documented by the
producer as market-data-only requiring no key; the licence question is referred to the general
Product Terms of Use, whose substantive clauses I could **not** retrieve — the page renders
client-side. I recorded that as a limit and refused to quote the paraphrases a search returned.
Our scope is stated so it can be judged: internal analysis, derived digests and counts, no bar
values republished, no redistribution, no revenue.

*What to attack:* whether "no material difference in 3,000 bars" is being leant on anywhere it
should not be. I preserved its definitions — material is relative deviation above 1e-12, the
145 differing bars are representation differences at 2e-16 — and stated it is not a
no-revision guarantee.

## Still open, with owners

| item | owner |
|---|---|
| the cube stores the availability digest, not the lag | Satoshi / schema decision |
| the 16 screen records and the replica proof | the owner of the P1LR front |
| verbatim Product Terms of Use clauses | Satoshi, blocked on a client-rendered page |
| per-delivery capability tokens | deferred by your order |

## Two things I got wrong this round, for the record

The probe that did not close its campaign, which you caught. And parking the rights research
as an owner action for two rounds when it was my work to do.

## State at submission

Four store services answering 200, zero failed units on omega, zero automatic restarts. The
disposable stack on gamma was torn down. No production service was restarted by me; the two
restarts that loaded the worker principals were yours and are disclosed in your own order.
