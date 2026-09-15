# Return: S1–S4

Order: `docs/handoffs/MUSASHI_TO_SATOSHI_COUNTERS_ARCHIVE_AND_TERMS_2026_09_15.md` (`61c8d51`),
over the review of `2574890`. Counts are mine and await independent verification. No historical
receipt, terminal digest or cube row was altered; `prod-12` and `prod-13` were not re-run.

## Revisions

| repository | branch | head |
|---|---|---|
| predictor | `satoshi/r1-r6-20260914` | `6705e27` (+ this document) |
| agent-multi | `satoshi/s1-compute-contract-20260915` | `74eeba44` |
| data-gov | `satoshi/store-hosts-e2e-20260914` | `01482f0` |
| data-lake | `satoshi/s2-availability-contract-20260915` | `7ac4ace` |
| data-warehouse | `satoshi/s2-availability-contract-20260915` | `6320897` |
| financial-data | `satoshi/s2-availability-contract-20260915` | `609bfdce1` |

## An incident I caused, first, because it is still open

While tearing down the disposable stack I ran `pkill -f "data_warehouse_service.main"`. The
pattern matched the **production** warehouse host as well, and
`crispdm-data-warehouse-olap.service` (:5057) stopped. I tried to restart it and the execution
environment refused the action (*Modify Shared Resources*). The command an operator can run is:

    systemctl --user start crispdm-data-warehouse-olap.service

`:5055`, `:5056` and `:5058` answered 200 throughout. Terminals reported while :5057 is down
are held by the durable outbox, which is what it is for, so nothing is lost — but the outage is
real, it was mine, and it was not part of any test. Afterwards I stopped disposable services by
PID from `STACK.json` only.

## S1 — the compute contract

**The correction.** `_n_updates` is not "gradient updates". Measured on the installed SB3 2.9.0:
four rollouts at two epochs is **8 epochs and 16 optimizer calls**; the ratio is the minibatch
count and grows as the batch shrinks. Optimizer calls are now counted by instrumenting the real
optimizer, and the instrumentation is removed afterwards. An algorithm whose counter has no
measured meaning reports the raw delta, `sb3_n_updates_meaning: UNKNOWN`, and claims nothing.

**Two traps, both measured rather than assumed.** `learn()` resets a resumed model's step
counter by default, so 64 → 64: the naive delta reports a real training call as **zero work**.
It is detected, not assumed. And a run stopped inside its first rollout reports zero rollouts,
zero epochs and zero optimizer calls while its transitions are still counted — a partial
rollout has not trained.

**The ceiling.** A target and a cap are separate fields. `n_steps 256` against a cap of 64 is
refused with `model.num_timesteps` still 0; a target above the cap is refused; the cap is never
widened.

**Tests:** 21 rules (`test_compute_contract.py` 13, `test_compute_contract_publication.py` 8),
plus the retained governed suites — 38 green together, CPU only.

**The successor:** `doin-offline-replay-successor-1`, NON_GOVERNING, from a clean worktree so
`code_identity` is `ddbff4d6…` with **no `-dirty`**. Declared mechanically: 1 env, `n_steps 64`,
`batch 64`, `n_epochs 1`, target 64, cap 64, evaluation cap 384. Observed **64 transitions, 1
rollout, 1 epoch, 1 optimizer call, 384 evaluation transitions**, cap respected, 3.81 s,
terminal COMPLETED, reconciliation empty, input `VERIFIED_CACHE`. **21 counters persisted in
the production cube.** Process success is separate from acceptance of the resource contract.

Evidence: `docs/audits/evidence/repro_runs/compute_contract_20260915/`.

## S2 — the archive contract, resolvable after the producer is gone

I chose to add the dimension: the existing store retained a **digest only**, which is the gap I
reported in R4. `gov_availability_contract` retains the canonical bytes keyed by their own
digest — recomputed on write, so nothing can be filed under a digest a delivery already trusts
— and `use_class` and `completion_lag_max` are read **out of** those bytes, not accepted
beside them. The lag is TEXT: `UNKNOWN` is stored as `UNKNOWN`, and no path produces a zero.
`gov_delivery_availability` resolves a delivery or says `UNRESOLVED`.

The contracts travel **beside** the terminal, never inside it: a terminal's identity is the
digest of its own body, and adding a field there would change every receipt ever written.

**Proved on a disposable PostgreSQL stack**, in this order: the archive delivered (200) and its
unit closed; range, point-in-time and live-tail each refused 422 naming the retrospective
archive and each closed as a `REFUSED` terminal; a plain resource delivered 200 as a
regression. Then the lake host was **killed** (confirmed dead) and **its configuration file
deleted**. Only then did a fresh reader — a new provider over the same database, with no
data-gov, no lake, no stack — answer `ARCHIVE_RETROSPECTIVE` / `UNKNOWN` and **re-verify the
digest itself** by hashing the retained bytes. A delivery nobody recorded answers
`NO_SUCH_DELIVERY`; a reference nobody retained answers `UNRESOLVED` with null lag.

**A refusal I did not predict:** with a holdout declared, the whole archive is refused 403 for
the holdout — its publication time was never observed, so it cannot be shown to lie before the
boundary. That is the incompatible-holdout case, arrived at by measurement. It is also why
there are two phases: an earlier single-phase run had the archive refused and would have
reported a delivery that never happened.

**The seam that was missing was found by running it.** The first execution resolved
**UNRESOLVED** for both deliveries, which is how I found that the lake host published the
digest and not the bytes.

**Tests:** 10 dimension rules, plus data-gov 167, data-lake 33, data-warehouse 28,
financial-data store 22 — all green after the change.

Migration additive and idempotent, and **not deployed**: the package declares its divergence in
`PENDING_REVIEW` instead of moving its pin, so nothing here can be read as running in
production. Historical references stay `UNRESOLVED`; no backfill, because nothing retained
supports one.

Evidence: `docs/audits/evidence/archive_contract_20260915/`.

## S3 — the terms, read

The document retrieved is the **ADGM Binance Global Terms of Use**, effective **21 July 2026**,
sha256 `bf487971…` — identical to the URL's own basename, so the bytes are the ones the review
named. Its identity is recorded; it is not republished.

Three findings change the answer rather than confirm it:

* it governs **"your use of your Binance Account"**, and our reads used no account and no key;
* its effective date **post-dates** the 2026-05-01 acquisition, so it is not the edition that
  governed it, and I have not obtained the one that did;
* clause **14.1.2(b)** routes API access to *"separate API terms"* this document does not
  contain. Two locators for those answered **HTTP 202 with a zero-byte body** — an
  interstitial, not a document. Measured, not asserted.

Clause **27** is quoted in full; clause **1.4** places these Terms *below* Product Terms in
precedence; clause **33.11** disclaims the accuracy and currency of market data, which
independently supports keeping our revision policy at UNKNOWN.

**Withdrawn:** the inference that internal, no-revenue use is necessarily outside the
restrictions. Clause 27 is a grant with a narrow scope, and not selling something establishes
nothing.

**Checked instead of repeated:** the claim that the published repositories carry no market
rows is **false as stated**. `harveybc/financial-data` is PUBLIC and commits **47,233** OHLC bar
values across six CSVs, one family of which appears under a heading "Binance OHLCV". What is
true is narrower and I say only that: the **governed lake resource is not published** — under
`market_data/` only `.gitkeep`, `.json` and `.md` are tracked, zero `.csv` and zero `.parquet`.
Nothing was deleted or republished. Whether those committed rows are within whatever terms
applied when they were acquired is the one question here that is genuinely the owner's.

## S4 — P1LR closed out, not launched

**The canonical path, and both declared ones are wrong.** The tool writes no verdict file of
its own: `_emit()` prints to stdout and writes only with `--output`. So the gate path is an
operator convention and must follow the design's own naming, inside the contract's replica
root `~/.local/share/agent-multi/p1lr_v2_collections_20260815`, beside the manifest and the
replica proof. The unit default points at a **v3** tree; the per-instance override at a
differently-hashed v2 tree with an unidentified filename. Recorded as a candidate; not applied.

**Repeated attempts are happening now.** On dragon, `p1lr-idle-guard.timer` is **active** at
15-minute intervals with `Persistent=true`, the service is `failed`, and its journal matches a
failure **130 times in 24 hours**. The disposition is **dormant** — mask the timer, and
deliberately do **not** `reset-failed`, because the failed state is evidence. Not applied: S4
forbids configuration changes under this order.

**Costed plan, not a launch.** 16 `(seed, cell)` records at one pass-equivalent each —
`epoch_timesteps 20000`, one phase-1 and one phase-2 epoch — is **640,000 training transitions**
exactly. Wall-clock and memory are deliberately **not** estimated: the honest way to get them is
one bounded pilot cell measured with the S1 compute contract, which is the first costed step,
not a guess.

Document: `SATOSHI_P1LR_DORMANT_DISPOSITION_S4_2026_09_15.md`.

## Open, with owners

| item | owner | state |
|---|---|---|
| `crispdm-data-warehouse-olap.service` (:5057) is down | operator | one command, named above; caused by me |
| the availability-contract migration | Musashi / production review | proved on disposable PostgreSQL; not deployed |
| the separate Binance API terms | Satoshi | both locators answered 202 with an empty body |
| which edition governed the 2026-05-01 acquisition | Satoshi | unresolved |
| 47,233 committed bar values in a public repository | owner | a rights question, not an engineering one |
| the P1LR dormant disposition | operator on dragon | commands named; not applied, per the order |
| the 16 screen records and the replica proof | owner of that front | costed; deferred by this order |
| per-delivery capability tokens | deferred | unchanged |

No full-suite claim is made from a focal run: the suites named above are the ones I executed.
