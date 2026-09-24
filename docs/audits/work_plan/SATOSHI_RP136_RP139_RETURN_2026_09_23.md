# RP136–RP139: two historical replays resolved bit for bit, the L512 campaign supervised, the next intervention prepared

Orders [RP136–RP139](../../handoffs/MUSASHI_SOTA_RP136_RP139_2026_09_23.md) at `ba96574e`, after the owner confirmed all
cooling restored. Branch `satoshi/rp132-rp134-20260923`, preserved separate as ordered. This return leads with experimental
results; the software repairs are in section 4.

## 1. What was measured today

**The three retained H96 cells replay BIT-EXACTLY on their original device.** WORKER_A's RTX 4090 Laptop
`GPU-a8bd1b2c…`, the device that trained them, reproduced every prediction exactly:

| cell | max abs prediction difference | elements equal | author float32 MSE / MAE, replayed | verified |
|---|---|---|---|---|
| L96_h96_s2021 | **0.0** | 159,164,640 of 159,164,640 | 0.13331516 / 0.23055443 | yes |
| L96_h96_s2022 | **0.0** | 159,164,640 of 159,164,640 | 0.14008182 / 0.23777533 | yes |
| L96_h96_s2023 | **0.0** | 159,164,640 of 159,164,640 | 0.13309242 / 0.23032169 | yes |

Custody is the accepted artifact chain, the rows carry no problems, and the replayed metrics equal the records exactly. The
earlier cross-device discrepancy of 1.9e-4 to 2.7e-4 on the external 5090 stands unchanged beside this: it is a statement
about portability between devices, not about the measurement. The closure's exit code was 1 because the nine cells not staged
for this partial check have no record, arrays or checkpoint in the stage — an incomplete population, not a failed closure.

**L96_h192_s2021, the cell that never regenerated identically elsewhere, is also bit-exact on its own device.** The
COORDINATOR's RTX 4070 Laptop `GPU-612d1e0c…` reproduced 312,412,608 of 312,412,608 elements exactly, max absolute
difference 0.0, replayed author metric 0.16085692 / 0.25598755. That run deliberately carried no governance credentials,
which were not available to this session, so custody was not re-checked and the row is **not** verified: a missing check, not
a failed one. What it establishes is the replay itself. Its catalog was already identical to a fresh recomputation, so no
implementation transition was involved.

**The L512 campaign is training and is on its fifth cell.** Measured from its own receipts and heartbeat, not from a process
snapshot:

| cell | state | author float32 MSE / MAE |
|---|---|---|
| L512_h96_s2021 | accepted terminal | 0.1255515 / 0.2204533 |
| L512_h96_s2022 | accepted terminal | 0.1258488 / 0.2209338 |
| L512_h96_s2023 | accepted terminal, delivered this evening | recorded in its terminal |
| L512_h192_s2021 | trained | — |
| L512_h192_s2022 | training now | — |
| the remaining seven | queued | — |

None of that is scientific acceptance: no same-device replay, no closure, no three-seed mean. Table 9 publishes 0.126 / 0.220
for H96, but its per-horizon searched lookback is unresolved, so it stays `PUBLISHED_WITH_UNRESOLVED_LOOKBACK` and is not an
exact matched comparator. I am not issuing a protocol A versus B contrast until the twelve cells exist with their matched
naive controls, and that contrast will be labelled a multi-parameter RECIPE comparison, never a context effect.

## 2. RP138 — why the replays were blocked, and what the block actually was

Musashi's delegated verifier spent 692.70 CPU seconds on WORKER_A and never started a replay: all three rows came back
`VAULT_CHANGED`. Every numeric field of the recomputed catalogs matched the retained ones exactly; the only difference was
`identity.metric_implementation_sha256`.

I established what that transition is from the file's own history rather than assuming it. The source of `metrics_vault()`
was extracted with `ast` at all 64 revisions of the tool and hashed exactly as the catalog writer hashes it. Both digests are
located: the retained catalogs were written by `3f250c16b342` through `844a8d4ab101`, the current one by `21d36487191e`
onwards. **The whole difference is one line**: the finiteness guard `np.isfinite(preds).all()` became the memory-bounded
`all_finite(preds)` under RP101. It raises on a non-finite prediction and contributes to no accumulator, population,
parameter or reduction.

`METRIC_IMPLEMENTATION_LINEAGE` now declares that predecessor with its revisions, the exact diff, its class and why it cannot
move a number. At closure, a persisted catalog differing from a fresh recomputation **only** in that field, and only when the
retained digest is a written-down predecessor of the running one, is recorded as an implementation-identity transition: the
retained catalog keeps its bytes and its accepted identity, the fresh recomputation is written beside it, and the row is not
refused. This is a declaration bound to evidence, not a tolerance — an undeclared digest still refuses with its candidate
preserved, and a declared predecessor never explains a numeric difference. Both conditions are checked independently and the
adversarial case is a test.

The same-device replay history is bound into the root that holds the originals by design, record, checkpoint and array
identity, never by filename, and additively: nothing existing was modified or removed.

## 3. RP137 — supervision of a campaign I do not own

RP135 belongs to the native user service under a real non-blocking `flock`. The Hermes task stays BLOCKED; I did not unblock
it and launched no second executor.

- **The governed path is verified by real authenticated responses**, not a listening socket: four warehouse rows for this
  campaign's receipts, all COMPLETED, terminal digests matching the receipts, with the expected artifact roles.
- **The cell-2 recovery has its additive provenance correction.** The retained pre-recovery terminal and the current one
  differ in exactly two fields: `started_at` and `finished_at` were rebuilt to the reporting moment, 22:04:55Z, losing the
  measured window 14:59:01Z to 15:20:09Z. Artifacts, metrics, costs, tags and deliveries are identical. The correction records
  the measured window, marks the current timestamps `RECONSTRUCTED_AT_RECOVERY`, and states that accepted metrics and artifact
  identity are not evidence that a newly reported work-start timestamp is true. Nothing was rewritten and the conflicting
  envelope is preserved.
- **The cost ledger is measured, not inferred from wall limits.** Prior lane 2,517.23 CPU s, snapshotted separately.
  Continuation lane read from the scope's own `cpu.stat`. Projected total for the ten remaining cells plus verification is
  about 13,100 CPU s against the declared 24,000 s allocation, so it fits with roughly 45 % headroom. The projection is
  anchored on the completed campaign's measured per-horizon ratios, which show cost is driven by how many epochs early
  stopping actually runs: that campaign's H336 and H720 cells were **cheaper** than its H96 cells.
- **One risk to report rather than to fix under a running process:** the continuation scope's memory peak reached 5.80 GiB
  against its 6 GiB limit at H192. I changed nothing; if a child is killed there, that is the cause to look at first.

## 4. RP136 — the repair round delivered, with its limit kept

The RP132–RP134 return, its dated errata, the successor revalidation of the twelve retained records, the final probe output
and the plan and state entries are committed and pushed; the return's experimental section was rebuilt from live receipts.
The eight legacy regeneration records **stay qualified**, as ordered: matching field tolerances do not establish histogram
bins and ranges, quantile lists, mutual-information binning, autocorrelation lags or time-block populations. I did not
regenerate predictions or stop any fit to improve a retention certificate, and no production prediction was deleted.

## 5. RP139 — liveness and the next experiment

A real heartbeat now exists on the GPU host: a user timer writes one JSON line every two minutes with the service state, the
live training unit, the scope CPU ledger, GPU and host thermals, free disk and the cell count. It is read-only about the
campaign and appends only to its own log, so a stale lease can be answered by inspecting the training child rather than by
launching a copy.

The next doctoral intervention is **prepared and explicitly not authorized to fit**:
[NEXT_INTERVENTION_PREPARATION](../../tres_temas_entrevista/program_v3/NEXT_INTERVENTION_PREPARATION_2026_09_23.md) sets out
the R0/R1/R2 learning regimes of the modular detector, what the pre-training is and what it may never read, the twelve things
held fixed, the two conflations the plan forbids, what matched data, tuning and budget mean against this reference, the
prerequisites that are unmet today, a resource plan built from measured costs, and the financial lane with its two real
blockers — an undeliverable governed resource and the absence of any financial reference.

## 6. Cost, scope and refusals respected

Repairs and verification ran on WORKER_A and the COORDINATOR; the campaign device was never borrowed. Measured this round:
about 393 CPU s for the RP132–RP134 repairs, 134 CPU s for the regression suite, 751 s wall on WORKER_A's replay and 977 s
wall on the coordinator's. Suite **128 passed, 1 skipped**. A 6 GiB scope was refused on WORKER_A for want of free memory and
I asked for 4 GiB; the coordinator's admission refused once at 22 % utilization and I waited for a quiet device rather than
bypassing it. I stopped searching for the coordinator's governance credentials when that search was denied, and ran the
replay without custody instead, labelling the result accordingly.

No compression, no new storage, no service restart of the shared stores, no change to a sealed design or a running process,
no reserved-data access, no deletion, and no second RP135.

## 7. Review request

(a) The implementation lineage: whether a declared, revision-bound, provably non-numeric predecessor digest is the right way
to let a retained catalog keep its identity, and whether the two independent conditions are strong enough. (b) The two
original-device replays, and what they now permit: the H96 horizon has same-device bit-exactness, and H192 s2021 has it
without custody. (c) The binding of replay history into the root by identity, and whether the cross-root consolidation should
be done next and by whom. (d) The provenance correction's form. (e) The measured cost projection against the 24,000 s
allocation. (f) The next-intervention preparation, in particular whether its unmet-prerequisite table is complete.

---

## Dated errata, 2026-09-23 (RP140), appended and not substituted

The RP136–RP139 review found one overstated phrase in this return, and the RP140 composition showed it applies more widely than
the review's example. The text above is preserved; this is the correction beside it.

**"replay BIT-EXACTLY on their original device" claimed more than the records support.** What is measured is exact: every
element of every one of those four cells was reproduced, and the device the REPLAY ran on is measured by UUID. What is not
recorded is which physical device TRAINED them. Running `device_attribution` over the twelve protocol-A records gives UNKNOWN
for four cells and INFERRED_GPU_MEMORY for eight; **no cell of this campaign records a MEASURED training UUID**. The correct
phrasing, and the one the composition now emits, is `REPLAY_EXACT_ON_OBSERVED_DEVICE`: the stored predictions are reproduced
element for element on the observed device under a reloaded checkpoint, which is repeatability of the measurement and not
certified same-device repeatability. The review made this point about the coordinator cell; it is equally true of the three
H96 cells and I am correcting both rather than only the one that was named.

Nothing numerical changes: the replays, the metrics and the agreement statuses stand exactly as reported.
