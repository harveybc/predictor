# Return: operational reconciliation, the first financial contract, and adoption per consumer

Orders: `docs/handoffs/MUSASHI_TO_SATOSHI_CONTRACTS_AND_CONSUMER_ADOPTION_2026_09_14.md`
(A1–A5), over Musashi's acceptance record
`docs/handoffs/MUSASHI_STORE_HOSTS_PRODUCTION_ACCEPTANCE_2026_09_14.md`. Work plan:
`docs/integracion_workplan_2026_09_10/09_ADOPCION_DATA_LAKE_DATA_WAREHOUSE_2026_09_14.md`.

Nothing here restarts a service, checks out a branch in the live data root, touches a
financial resource without a contract, or issues a scientific approval. Every campaign below
is **NON_GOVERNING**.

## A1. What changed since my previous return — verified, not assumed

| my previous return said | the fact now | how it was checked |
|---|---|---|
| repositories published **private**; visibility pending the owner | **public**, both | Musashi's record; the providers install from public URLs |
| `predictor-olap-store` pending as PR #44 | **merged**, master `19d1edc` | the merge commit is an ancestor of master |
| production transition **blocked by the runtime** | **done by Musashi**: the hosts serve :5056 and :5057, a synthetic instance serves :5058 | `/api/v1/host` on each port names the installed distribution and version |
| services started by hand, no supervision | four persistent user services, `enabled`, `Restart=on-failure`, `MemoryMax=2 GiB`, `NRestarts=0`, linger on | `systemctl --user show` per unit |
| loader "active" | **advancing**: heartbeat published at the moment of the check, age ≈ 0 s, `healthy: true`, `process_fresh: true`, pending 0, loaded 43, 16 dead letters all adjudicated (from 09-12, not new) | the loader's own `HEARTBEAT.json`, plus `df_fact_load_receipt` |
| inventory 5,275 resources | **5,275**, unchanged | `/api/v1/discover` on :5056 |

Two details worth recording rather than smoothing over:

* `/api/v1/lakes` lists three stores, not four. `governance_smoke` is absent **because its
  policy grants only `download` to the `predictor` principal and no `discover`** — the
  catalogue shows what the caller may discover. The store is configured and reachable; the
  listing is correct.
* The warehouse host adds one field to `describe` that the previous adapter did not send
  (`transport: http`). It is the host stating how it is reached, it is additive, and it is
  declared here rather than hidden behind an "identical" claim.

Old blockers are closed: publication, the PR and the transition are facts, not pendings.
The blockers that remain are named in §5.

## A2. First financial availability contract — candidate, with its evidence

Resource: `market_data/crypto/spot_top50/ethusdt/4h.parquet`, the raw input of the lineage
the census already traced (`ETH_H4_SUCCESSOR_TEMPORAL_CONTRACT.v3` binds its successor to
this file by exact value match). No new census was started.

Measured from the bytes — 18,337 rows: both time columns are `datetime64[ms, **UTC**]` in
the physical schema; `open_time` is the bar open and `close_time` its last millisecond
(modal difference 4 h − 1 ms); strictly increasing, no duplicates; every row on the nominal
4 h grid; 8 gaps; 21 truncated bars, one of which closes at its own open; and the file's
digest equals the digest its producer pinned.

Candidate (`contract_sha256` `998e3f80…`):

```json
{"event_time_column": "open_time", "available_time_column": "close_time",
 "timezone": "UTC", "time_unit": null, "frequency": "4h",
 "availability": {"label": "WINDOW_END", "completion_lag_max": "0s",
                  "timezone_evidence": "PRODUCER_STATEMENT",
                  "use_class": "OFFLINE_DAY_GRANULAR"}}
```

Declared unknowns, not estimated: **provider delivery latency UNOBSERVED** (which is why the
use class is offline and not live equivalent), **revision policy UNKNOWN** (one acquisition
cannot show restatements), **usage rights UNKNOWN** (the source is named; no licence is
recorded in the repository).

Eleven tests over bytes ran **before** any installation — range extremes on completion, the
day a bar belongs to, truncated bars, the schema's time zone, intrabar refusal, disjoint
partitions whose union is the whole range, a future tail that leaves an earlier delivery
byte-identical, and a revised past bar that changes the delivery identity.

Delivered as **financial-data PR #1**, branch `satoshi/ethusdt-4h-contract-candidate-20260914`
(`1b4a23431`). **Not installed**: it changes what a governed campaign may consume.

## A3. Adoption per consumer, against the deployed services

`tools/verify_consumer_adoption.py` drives **each consumer's own governed wrapper** — the
entry point the repository really uses — through data-gov into the deployed warehouse, with
four cases each. Receipts: `docs/audits/evidence/repro_runs/adoption_20260914/`.

| consumer | governed delivery | success | stale refused | failure with cost | retry sends nothing | scope |
|---|---|---|---|---|---|---|
| preprocessor | VERIFIED_TRANSFER, contract `5a521473…` | **COMPLETED** | REFUSED | FAILED | yes | **full** |
| predictor | 6 inputs verified (transfer + cache) | **COMPLETED** | REFUSED | FAILED | yes | **full** |
| feature-eng | VERIFIED | refused by its own pipeline | REFUSED | FAILED | yes | transport only |
| feature-extractor | 6 inputs verified | refused by its own pipeline | REFUSED | FAILED | yes | transport only |
| agent-multi / DOIN | — | — | — | — | — | not started |

Cube effect of these checks: 12 governed terminals, 114 metrics, 28 datasets, 24 artifacts,
all additive; every campaign reconciled with no missing unit and no record on only one side;
no duplicate row on retry.

Two findings came out of running the real entry points rather than describing them:

1. **preprocessor** consumes predictor's eligibility gate, which refuses a run that declares
   no partitions. A governed mechanical replay has to say what it is —
   `execution_purpose: ARCHIVAL_REPLAY_NON_AUTHORITATIVE` — and then it runs. A wrapper
   alone would not have shown this.
2. **predictor's** wrapper could only register GOVERNING campaigns, so a bounded mechanical
   check had no honest classification. `--classification NON_GOVERNING` is now explicit in
   the campaign and in the receipt.

What is missing for the two transport-only consumers, exactly: no lake that data-gov serves
carries the columns they need — their fixtures live inside their own repositories. Next
action: publish those fixtures as a governed resource with a derived contract (the same
procedure as A2, on synthetic data, so it needs no producer research). Until then their
adoption is transport, and it is reported as transport.

`agent-multi`/DOIN remain not started: the `doin_governed_result.v1` contract and its fixture
do not exist yet, and implementing them belongs to those repositories.

## A4. Reproducibility and causality of the integration

Covered by what ran, in the layers the order asks for:

| requirement | test | result |
|---|---|---|
| input identity and configuration reach the run | each consumer's receipt records resource, sha256, verification state and contract identity | 4/4 consumers |
| a stale output cannot be passed off as fresh | the stale case | REFUSED, 4/4 |
| a failure is recorded with its cost, not hidden | the failure case | FAILED with `wall_seconds`, 4/4 |
| retry does not duplicate an outcome | flush after each consumer | `sent: 0`, cube unchanged, 4/4 |
| future data cannot change what was already delivered | `test_ethusdt_4h_contract.py::test_appending_future_bars_leaves_an_earlier_cut_byte_identical` | passes |
| partitions are disjoint and cover the range | same suite | passes |
| a deliberately non-causal control must fail | the intrabar range refusal and the range-extreme exclusion | both refuse |

Scope of this evidence, declared: it covers transport, identity, refusals and the delivery
rule of one candidate contract. It does not claim the absence of leakage in any model, nor
that any dataset is correct.

## A5. Documentation, with D2 untouched

The corrected D2 counts stand exactly as they are: **47 + 6**, 39 SNR calibrations, five
candidate losses plus two control losses, 138 rows changed. The 288-decision check remains a
finite diagnostic, not a tolerance. The Kalman diagnostic proposal
(`10_DIAGNOSTICO_KALMAN_NO_REPRODUCIBLE_2026_09_14.md`) stands with its separated factors,
its prior budget and its criterion fixed before measuring; no new Kalman replay was run and
no tolerance was widened. D3 remains outside this order.

## 6. Blockers, each with an owner and a next action

| blocker | owner | next action |
|---|---|---|
| the candidate contract is not installed | review (Musashi), then a deliberate window | accept or reject financial-data PR #1; on acceptance, add the entry as a **pending** configuration and activate it in a window |
| feature-eng and feature-extractor adopt transport only | Satoshi | publish their synthetic fixtures as a governed resource with a derived contract, then re-run the same four cases |
| agent-multi / DOIN not started | Satoshi, in those repositories | define `doin_governed_result.v1` and a fixture; offline replay only |
| usage rights and revision policy of the ETH resource | the data owner | a licence statement and a second acquisition would close both |

Services at the end: `:5055/:5056/:5057/:5058` healthy, all four units `active` with zero
restarts, loader advancing, no throwaway database, cube at 20 governed terminals with its
history intact.

Stop: `FIRST_FINANCIAL_CONTRACT_CANDIDATE_READY_TWO_CONSUMERS_ADOPTED_TWO_TRANSPORT_ONLY`.
