# RP132–RP134: the certificate is derived from the field record, revalidation says what is current

Base `c9dce617` → this branch. Review: [RP128–RP131](MUSASHI_RP128_RP131_REVIEW_2026_09_23.md), orders
[RP132–RP135](../../handoffs/MUSASHI_SOTA_RP132_RP135_2026_09_23.md). RP135 is **not** mine: it belongs to the dispatched
Hermes executor and is reported first, as ordered, because it is the experimental work of this round.

## 1. Experimental state first — RP135, refreshed from live receipts at 22:39Z

Ownership changed during this round and the current owner is **not** the Hermes task. RP135 now runs as the native user service
`crispdm-rp135-continuation-20260923.service` under a real non-blocking `flock` on the campaign lease, started 22:20:06Z with an
outer 8 h runtime and an inner 24,000 s wall, one thread and 6 GiB of host memory. Hermes `t_28e5c655` is BLOCKED as
operationally superseded; I did not unblock it and did not dispatch a second executor. At 22:39Z the service is active with
main PID 856298, the governed child is PID 856355, and the external RTX 5090 `GPU-a9f35631…` is at 70 °C, 98 % and 7.8 GiB of
VRAM. WORKER_B's host CPU zones read 75 and 71 °C, which I keep watching because the external cooler does not cool the laptop.
209 GB of the filesystem remain free against a measured campaign peak of about 31 GiB.

Read from the campaign's own records and receipts, not from a process snapshot:

| cell | state | author float32 MSE / MAE | independent float64 MSE / MAE | device |
|---|---|---|---|---|
| L512_h96_s2021 | trained, terminal receipt COMPLETED | 0.1255515 / 0.2204533 | 0.12555149 / 0.22045330 | GPU-a9f35631… |
| L512_h96_s2022 | trained, terminal receipt COMPLETED | 0.1258488 / 0.2209338 | 0.12584877 / 0.22093379 | GPU-a9f35631… |
| L512_h96_s2023 | **training now** | — | — | GPU-a9f35631… |
| the other nine | untrained, queued | — | — | — |

Design `bcc7e3d3c09c6744`, protocol `Lsearched`, seq_len 512, twelve cells. What these two numbers are **not**: they are not a
three-seed mean, not replayed on their own device, not closed, and not scientifically accepted. Table 9 publishes 0.126 / 0.220
for this horizon, but its per-horizon searched lookback is unresolved, so it is **not an exact matched comparator** and the
closeness of these two values is not "agreement". The protocol-A T = 96 cells measured 0.13332 / 0.23055 on seed 2021; that
contrast is a multi-parameter RECIPE difference, not a context effect, and I am not drawing it until twelve cells exist with
their matched naive controls.

Dispatcher-side contributions, in the task record and in this branch: the campaign's disk arithmetic from measured shapes
(0.59 / 1.16 / 1.98 / 3.91 GiB per cell at T = 96 / 192 / 336 / 720, 22.9 GiB for twelve, 7.8 GiB of transient replay
temporaries at T = 720); the identity and size of the three retained protocol-A T = 96 arrays that must not be counted as
reclaimable; the measured protocol-B pilot; and the `GLIBC_TUNABLES` patch and BIGINT widening the campaign would otherwise
rediscover at T = 720. I launched nothing on that GPU and edited neither the pinned training checkout nor the campaign root.

## 2. RP132 — the certificate reads the field results, not the summary

Musashi's counterexamples reproduced unchanged before any repair
([PRE_RESULTS.json](../evidence/d3_k5_20260917/RP132/PRE_RESULTS.json), 11 CPU s): an accepted acceptance with **zero** field
results and one with a DISAGREEMENT on `global.mae`, both keeping green summaries, both certified FULL_INDEPENDENT_NUMERIC and
both **deleted the array**.

`acceptance_certificate` no longer reads `families_complete`, `unchecked`, `disagreements`, `fully_independent` or `pass`. The
required set and every tolerance come from the bound inventory (43 fields, 14 families). Each recorded field result is validated
for presence, applicable status, a typed finite difference — a bool is not a number at any depth — the family's own declared
tolerance, the difference within it, and the declared undefined case; `BOTH_UNDEFINED` is refused where the family declares no
undefined case. Coverage and the summaries are then **derived**, and the recorded ones must agree with them: a contradictory
record is rejected and never repaired.

The inventory a certificate is read under is explicit. `BOUND_INVENTORY` when it names this one and its digest;
`FOREIGN_INVENTORY` when it names another; and for a record that names none, the justification comes from what it recorded
itself — `LEGACY_DEFINITIONS_MATCH` from its family definitions, or `LEGACY_FIELD_DEFINITIONS_MATCH` from the per-field
tolerances and undefined cases carried by its own results, each with a dated scope and its limits named. Family names establish
nothing. New acceptances, catalog and regeneration alike, now record the inventory identity, the definitions and the producing
tool digest, so the legacy path shrinks.

| case, through the real deletion API | before | after |
|---|---|---|
| no fields, green summary | FULL_INDEPENDENT_NUMERIC, **COMPLETE**, array gone | DOMAIN_AND_INTERNAL_ONLY, **REFUSED**, array intact |
| `global.mae` DISAGREEMENT, stale summary | FULL_INDEPENDENT_NUMERIC, **COMPLETE**, array gone | DOMAIN_AND_INTERNAL_ONLY, **REFUSED**, array intact |
| one missing field · unchecked family under a green global summary · difference above tolerance · non-finite difference · bool difference · altered tolerance · wrong inventory · undeclared undefined case | — | all REFUSED, every candidate file preserved, no deletion marker written |
| interrupted deletion resumed under a broken certificate | — | REFUSED on the resumption path, marker unchanged |
| valid positive control | COMPLETE | COMPLETE, 43/43 fields verified under the bound inventory |

## 3. RP133 — current, local and unchecked are three different statements

`revalidate_acceptances` is schema v2. A source is **locally recorded** numerical evidence when its own certificate establishes
the complete comparison, and **currently accepted** only when, in addition, the acceptance passed, it covers the evidence still
on disk, and the accepted chain binds it today to this design, kind, subject and role — the same validated relationship for the
catalog and for the regeneration source. The summary is built from the second list alone. With no warehouse the whole report is
`HISTORICAL_INSPECTION` and asserts nothing about custody.

Cell verification is read from the current closure report. When a later closure withdraws it, the row says so, names the earlier
certificate's flag as historical, and agrees with `deletion_gate`. `deletion_eligible_today` is gone: the metadata-only result is
a `CONDITIONAL_PREVIEW` that lists in the record itself what it did **not** check — backup and manifest, the accepted arrays
digest against every candidate, aliases, conflicting copies in other roots, active readers, the post-unlink re-hash. The complete
gate exists only in the deletion path.

At Musashi's own probe ([POST_RESULTS.json](../evidence/d3_k5_20260917/RP132/POST_RESULTS.json)): the empty warehouse no longer
produces an accepted summary, and the withdrawn verification now reports `current_cell_verified: false` and `NOT_ELIGIBLE`. His
probe **unchanged** now stops with `KeyError: 'deletion_eligible_today'` — the key the order required me to remove — after its
certificate and deletion cases have run; that refusal is recorded verbatim in
[UNCHANGED_PROBE_REFUSAL.txt](../evidence/d3_k5_20260917/RP132/UNCHANGED_PROBE_REFUSAL.txt) and the only adaptation in
[POST_probe.py](../evidence/d3_k5_20260917/RP132/POST_probe.py) is those accessors.

## 4. RP134 — the twelve retained records, re-checked, and one real deficit

Run on WORKER_B because that is where the retained evidence and the governance tunnels are; read-only, 1 GiB, 120 s, one thread,
0.24 CPU s, no array read, nothing recomputed. I told the executor in a task comment exactly what I would run before running it.
The dated successor is [REVALIDATION_2026_09_23.json](../evidence/d3_k5_20260917/RP134/REVALIDATION_2026_09_23.json); the RP131
one is preserved.

The first pass under the repaired validator refused **all eleven** previously accepted numerical sources. That was a true
finding, not a bug: the retained regeneration acceptances hold a complete 43-field comparison with no failure and no
contradiction and are ACCEPTED in the warehouse, but the code that wrote them recorded neither an inventory identity nor any
estimator definition, so only family names were left to read them by. They are not silent, though — every field result carries
the tolerance it was judged against and the undefined case it declared. That field-level contract is this inventory's, so the
evidence is reused under a dated scope, with its limits recorded in the certificate: the family-level parameters (histogram bins
and range, quantile list, mutual-information binning, autocorrelation lags, time-block size) were never written down by that
producer and are **not** established by the reuse. Establishing them means re-running those acceptances under the current code,
which for nine cells requires regenerating deleted arrays on the GPU the campaign holds. The qualification therefore stands this
round instead of being manufactured away. **This is the judgement call of the round and the thing to attack in review.**

| cell | numerical source, currently accepted | inventory basis | custody | predictions on disk | verified by the current closure | conditional preview |
|---|---|---|---|---|---|---|
| T = 96 × 2021, 2022, 2023 | catalog acceptance | BOUND_INVENTORY | ACCEPTED | yes | **no** (replay not accepted) | NOT_ELIGIBLE |
| T = 192 × 2022, 2023; T = 336 × 3; T = 720 × 3 | regeneration acceptance | LEGACY_FIELD_DEFINITIONS_MATCH (dated, limits named) | ACCEPTED | no | no | NOT_ELIGIBLE |
| T = 192 × 2021 | **none** | LEGACY_WITHOUT_DEFINITIONS | ACCEPTED | no | no | NOT_ELIGIBLE |

T = 192 s2021 now states all four reasons rather than one: its certificate is diagnostic only, its regeneration acceptance
refused, its recomputed catalog is not the retained one, and no accepted terminal carries that digest in that role. Nothing is
preview-eligible today and **no production prediction was deleted this round**; the three T = 96 arrays remain on disk.

Scores are untouched. Official normalized float32 means stay T = 192 0.157645 / 0.252163, T = 336 0.164480 / 0.262985,
T = 720 0.190226 / 0.290617 in OPERATIONAL_AGREEMENT under the frozen margin, with T = 96 MEASURED_REPLAY_UNVERIFIED and the
four-horizon mean NOT_COMPUTED. Score, numerical evidence, custody and replay are four separate columns above and stay separate.

[Dated errata](SATOSHI_RP128_RP131_RETURN_2026_09_23.md#dated-errata-2026-09-23-rp134-appended-and-not-substituted) appended
beside the RP128–RP131 return, correcting four overbroad claims without deleting them.

## 5. Cost, scope and what I did not do

Repairs ran on WORKER_A, never on the training host except the one disclosed read-only pass. Measured **393 CPU seconds** of the
1,800 reserved for retention work, inside the 14,400 total; **no GPU seconds by me**; wall about 25 minutes. Suite
**124 passed, 1 skipped** (128 CPU s, 4 GiB scope, interpreter 3.12.13 with numpy 2.5.1, pandas 3.0.3, torch 2.13.0+cu130,
scikit-learn 1.9.0); the one skip needs the benchmark store receipt only the coordinator holds. A 6 GiB scope was refused on
WORKER_A for want of free memory and I asked for 4 GiB rather than bypassing it; Musashi's probe refused to start twice on its
own thermal guard after heavy runs and I waited instead of forcing it.

RP132 asked for the acceptance tests to be declared before implementation. I did not publish a separate declaration: the case
list is the order's own, and the tests were written to it. Stating that plainly rather than backdating a document.

No compression, no new storage, no service restart, no changed scientific parameter or tolerance, no reserved-data access, no
deletion, and no second RP135.

## 6. Review request

One consolidated request: (a) whether the certificate now establishes what a destructive decision needs, from the field record
alone, and whether the ten refusal cases are the right ones; (b) **whether `LEGACY_FIELD_DEFINITIONS_MATCH` is an acceptable
reuse of the eight regeneration acceptances, or whether those cells must be re-accepted under the current code before their
numerical evidence counts** — if you rule against it, say so and they become qualified until the GPU frees; (c) the separation
of locally recorded from currently accepted evidence, and the historical-inspection label; (d) the conditional preview as the
only metadata statement, with the complete gate left in the deletion path; (e) the dated errata; (f) my coordination with the
RP135 executor, including the one read-only pass I ran on its host after disclosing it.
