# Q2 household — the restoration verified, the governed chain proved, and one of my numbers corrected

**Satoshi, successor technical lead. 2026-09-29.**
Branch `satoshi/q2-household-governed-20260929`, own worktree.
Read first: §3 of `MUSASHI_CAMPAIGN_STATUS_TRIAGE_2026_09_29.md` at `50d06e50`;
`SATOSHI_RB01_RB03_RESOURCES_AND_ENDPOINT_2026_09_28.md` at `956dc43c`; the RB03 section of
`SATOSHI_RESUME_AND_BENCHMARK_ORDERS_2026_09_28.md`.

## 0. Lead

The restoration **verifies on all six of the triage's checks**. A **real governed unit now completes
for this lane** — two of them, one on the coordinator and one on the secondary worker, each with its
own campaign, its own `VERIFIED_TRANSFER` delivery, the consumed bytes re-digested inside the child,
an accepted terminal, and the same terminal digest read back out of the live warehouse. The bounded
pilot's **measured whole-cgroup peak is 1 463 877 632 B (1.363 GiB)** against a cap of 6.0 GiB
declared before the run — and that number is a **floor for the W1440 data stage**, not a cap for any
cell.

Nothing was started, stopped, enabled, disabled or restarted. No credential was printed, copied,
logged or written to a second file. No fit ran, no historical cell was touched, no allocation was
reset, no reserve was opened. The round's own closure table is **`NO_NEW_MEASUREMENT`** with every
error cell `null`, because this round scored nothing — which is the honest outcome, not a gap.

## 1. What I did not do

* The refused W1440 attempt was **not relaunched**; v1 was **not relaunched**. The six missing cells
  stay missing.
* The twelve historical cells were **not refitted, not re-bound to v2**, and no allocation was reset.
  They stay `NOT_BOUND_TO_A_SEAL`, custody `UNCHECKED`, closure `FAILED`, `HISTORICAL_DEV_ONLY`.
* No scientific dimension, budget, window, depth, cell count or optimization was changed. No
  successor is proposed here.
* No holdout access. The household holdout `2006-12-16` was touched only to confirm that a range
  crossing it is still **refused**.
* The preferred RTX 5090 host was read once, to confirm the standing ruling, and given no work. I did
  not re-measure it to argue otherwise.
* The **old registry-restore command was never used.** See the correction in §2.5.

## 2. Job 1 — the restoration verified on the triage's own six checks

All evidence in `docs/audits/evidence/Q2_HOUSEHOLD_20260929/`:
`RESTORATION_RECEIPT.json`, `PREFLIGHT.json`, `REHEARSAL.json`, `VERIFY_CONTENT.json`.

| # | the triage's check | verdict |
|---|---|---|
| 1 | preflight refreshed | **VERIFIED** — exit 0, every check green under the refreshed precondition |
| 2 | only the named lake serving | **VERIFIED** |
| 3 | actual delivered content, not discovery | **VERIFIED** — 10 890 295 B, `b3192c0b…c8db`, byte-exact |
| 4 | contract refusals still hold | **VERIFIED** — ranged request HTTP **422**, holdout-spanning range HTTP **422** |
| 5 | registry not replaced, other stores untouched | **VERIFIED** |
| 6 | receipt and rollback retained | **VERIFIED** |

### 2.1 The preflight had to be refreshed, and the one check that failed was the right one

Run unchanged, the shipped preflight **fails on exactly one check: `port_free`**, with
`"something already serves it"`. That is correct behaviour, not a defect. Before the restoration the
only admissible state of that port was *free*, because anything bound to it was an unknown process
and `apply` had to refuse.

Once the named unit is up the admissible state is the opposite one, and the check that carries the
same weight is that the listener on the port **is that unit's own main process**. The refreshed
precondition (`preflight --already-serving`) is **strictly harder** to satisfy than the old one: it
resolves the single loopback listener's pid against the unit's `MainPID` instead of counting sockets.
Under it, **exit 0 and every check green**, including the sixteen that were already green:
store id, root, the household resource registered **and** declared `untimed`, both panel digests
against the 2026-09-20 adoption binding, the serving and the *installed* provider digests, and the
registry publishing this lake at this port with this holdout.

`UnitFileState=disabled` is **recorded, not asserted and not changed**. Persistence across a reboot
is the owner's separate explicit choice.

The flag never starts, stops, enables or disables anything.

### 2.2 The rehearsal identity, refreshed today, matches the live serving unit exactly

`rehearse --port 5071 --already-serving` — **exit 0**. The deployed interpreter, module and
configuration on a throwaway loopback port, from a copy differing **only** in `web_port`
(`config_differs_only_in: ["web_port"]`, `backend_settings_identical: true`); never registered,
never enabled, and **stopped with its port freed**.

Against the live unit, the rehearsal's identity agrees on every field: `store_id` `public_panels`,
`backend.settings_sha256` `5287ba27…`, `backend.source_identity` (distribution, module, version and
source commit), the seven capabilities, and the delivered bytes digest. So what the live unit serves
and what the deployed command serves are the same thing, proved rather than assumed.

### 2.3 Only the named lake is serving, and it is still not the SOTA lake

The household port answers with `store_id: public_panels` and lists exactly its **two** registered
resources. The other store answers with a **different lake id**, `sota_benchmarks`. The household
resource is served by the named lake and by nothing else, and the two inventories do not overlap at
all. **Substitution stays REFUTED.**

> **One correction, in the order's own favour and beyond my 2026-09-28 note.** On 2026-09-28 I
> recorded that the SOTA store serves *three* resources, correcting an earlier *one*. Measured today
> it serves **nine** — three TSL time-series and six classification corpora. The corpus keeps growing
> additively and still contains nothing of the household panel.

### 2.4 Delivered content, because discovery is not evidence

`http_lake.discover()` swallows *"lake unreachable"* and returns an **empty list**, so a consumer
that preflights on discovery alone believes the lake is healthy and is then refused at its first
byte. `verify --content` therefore streams the resource from the **live** endpoint and digests what
actually arrived:

| | |
|---|---|
| HTTP | **200** |
| bytes | **10 890 295** |
| sha256 of the received body | **`b3192c0bcb117b2ee120a906dbcfb9550cd907abff74fea9bc2b1aa320ebc8db`** |
| byte-exact against the 2026-09-20 adoption binding | **yes** |
| `X-Content-SHA256` header | the same digest |
| `X-Availability-Contract-SHA256` | **`93d945cf…e2e2e`** — the digest the 119 retained deliveries carry |

**No data contract moved.** Availability is still `UNDECLARED` / `UNKNOWN`, as it was.

### 2.5 The registry was not replaced — and the old restore line is worse than I recorded

The live registry digest is `91f1e627…` and it is **identical before and after** the content read.
It publishes **six** lakes, `public_panels` and `sota_benchmarks` among them, with the household
lake's three access policies (`predictor`, and one principal per worker) all carrying
`deny_from: 2006-12-16`.

Every other store unit is in exactly its prior state, and their start timestamps **all predate the
restoration** (2026-09-28) while the household unit's is 2026-09-29 14:30:05. Nothing else was
restarted.

> **Correction to my own 2026-09-28 record.** I wrote that the adoption receipt's documented restore
> line "would today delete a healthy registered lake" — the SOTA one. Measured: the backup it copies
> is the **pre-adoption** registry. It carries **neither** `public_panels` **nor** `sota_benchmarks`,
> and none of the three household access policies. Applying that line today would delete **two**
> registered lakes — **including the household lake itself** — and restart the governance kernel every
> other consumer depends on. It is worse than recorded, and it was never used.

### 2.6 Receipt and rollback

`RESTORATION_RECEIPT.json` is the receipt. The rollback is
`tools/df_public_panels_restore.py rollback` — a `systemctl --user stop` of that one unit and nothing
else, asserting `port_free`, `registry_unchanged` and `other_units_unchanged`. It was **exercised as
a no-op on 2026-09-28** (`ROLLBACK_NOOP.json` at `956dc43c`), so the undo path is proven to run and
to change nothing else. **It was not run now**: running it would stop the lake the owner restored.
Verify, do not re-do.

## 3. Job 2 — the governed chain, proved end to end

`docs/audits/evidence/Q2_HOUSEHOLD_20260929/GOVERNED_CHAIN.json`. Two bounded units, each with its
own campaign and its own delivery, both under `crispdm-run`, both through the client path the Q2
block itself uses (`tools/df_e1_governed.py` → `tools/governed_run.py:GovHttp`). The credential was
reconciled by giving the existing client its `--api-key-file`; it was never printed, copied, logged,
written to a second file, issued or rotated.

| | `probe` (coordinator) | `pilot` (`WORKER_A`) |
|---|---|---|
| delivery | `VERIFIED_TRANSFER` | `VERIFIED_TRANSFER` |
| served from cache | **`false`** | **`false`** |
| availability contract | `93d945cf…` | `93d945cf…` |
| sha256 re-verified **inside the child** | `b3192c0b…` ✓ | `b3192c0b…` ✓ |
| terminal | `COMPLETED`, accepted, `sent 1 / pending 0` | `COMPLETED`, accepted, `sent 1 / pending 0` |
| reconciliation | empty on all three lists | empty on all three lists |
| warehouse row | 1 terminal, **3** metric rows, 1 artifact row | 1 terminal, **4** metric rows, 1 artifact row |
| warehouse digest = receipt digest | **yes** (`bda47cf8…`) | **yes** (`b852b550…`) |

**The warm-cache question is settled by measurement, not by assumption.** Both deliveries are
`cached: false` — the bytes were re-streamed and re-verified from the lake although **twelve** warm
caches of the identical digest sit on the coordinator's disk. So this had to actually pass, and it
did; the 119 retained deliveries were not allowed to stand in for it.

The `pilot` unit additionally proves the **remote** path: from the secondary worker, over the
governance forwarding, through the kernel's `http_lake` proxy, to the restored lake — from a clean
worktree at the published commit `4d120967`, which `strict_code_identity` requires.

The reader is `require_delivery`, which refuses a unit that has no delivery **of its own** and
re-digests the delivered file before returning it. The child then re-digests the consumed bytes a
second time, independently, before doing any work.

## 4. Job 3 — the successor reconciled, and one of my numbers corrected

`docs/audits/evidence/Q2_HOUSEHOLD_20260929/MEMORY_CORRECTION.json`.

### 4.1 What 7.4 G actually is — and where the order's premise needs correcting too

The circulating line is in `docs/tres_temas_entrevista/program_v3/EXPERIMENT_EXECUTION_QUEUE.json`:
*"7.4 GiB resident set — declared in ASYMMETRIC_READING.json from measurement, not estimate."*

**That line is wrong, and the order's characterisation of it is wrong in the opposite direction.**
The retained journal says, verbatim:

> `crispdm-q2deep-<epoch>-<pid>.scope: Consumed 21min 32.106s CPU time over 23min 9.019s wall clock time, 7.4G memory peak.`

So 7.4 G **is** a cgroup peak — systemd's own `memory.peak` line — not a resident set. The order
states it is a resident set and not a cgroup peak; on that one point the order is mistaken, and I say
so rather than adopt a convenient premise. **The order's substantive conclusion is nevertheless
right, for two reasons that do not depend on that premise:**

1. it is the peak of the **memory-gated wrapper scope** that hosted the pilot driver and its parallel
   children, **taken at an `oomd` kill** — a kill-time watermark of a multi-child tree, not any
   single cell's footprint; and
2. the number that has actually been used as *the measured pilot peak*, and that the memory gate
   sized its `required_bytes` from, is `measured_pilot_peak_bytes = 8 458 399 744` B (**7.877 GiB**),
   which comes from `resource.getrusage(RUSAGE_SELF).ru_maxrss` inside one cell child
   (`tools/df_e1_block.py:run_cell`) — **one process's resident set**. The second such figure is
   10 279 276 544 B (9.573 GiB).

So the queue line conflates two different numbers *and* mislabels the one it names.
**Neither authorizes a successor cap.**

### 4.2 No per-cell cgroup peak exists for any Q2 cell — and the repair is stranded on one branch

`tools/df_e1_block.py:run_units` launches every unit with a bare
`subprocess.run([sys.executable, …, "child", …])`. It never invokes the launcher, so **a cell child
takes no scope of its own** and inherits the driver's cgroup. Verified: no cell-scoped `memory.peak`
was ever created.

And a finding I owe the record, which the order did not have:

* the repair — `run_cell` recording `cost.cgroup_memory_peak_bytes` and labelling the RSS figure with
  its basis — exists in commit **`84bcd605`**, and `84bcd605` is contained by **exactly one branch**:
  `satoshi/q2-context-deep-arms-20260926`. Not `master`, not `956dc43c`, not `50d06e50`, not
  `f40fae93`, not `07846a7d`, not `0ae37a68`. The household block runner is **not on `master` at
  all**. Every revision any lane is working from records only `peak_rss_bytes`, so a successor cut
  from mainline would reproduce the same unusable figure.
* **Successor v2 (`27ed712d`) is sealed on that same lone branch and reachable from no other.**
* even on that branch the repair is not sufficient: the child still takes no scope, and
  `LIMITS["parallel_children"]` is **3**, so `cgroup_memory_peak_bytes` read from a cell child returns
  the peak of the **shared driver scope charging up to three concurrent cells**. It is a per-batch
  figure, not a per-cell one.

A successor that must report a per-cell cgroup peak therefore needs **both** that instrument **and** a
change `run_units` does not have — each cell child in its own scope, or `parallel_children` pinned to
1. I am not proposing it: a budget or dimension change belongs in an explicitly justified successor
declared before any score.

### 4.3 The bounded pilot, and what it measured

Not an arm. One bounded pilot, one child, `parallel = 1`, on `WORKER_A`, with the cap **declared
before the run and never lowered**.

| | |
|---|---|
| declared cap (as the kernel holds it) | 6 442 450 944 B = **6.0 GiB** |
| **measured whole-cgroup peak** | **1 463 877 632 B = 1.3633 GiB** |
| provenance | `memory.peak` of this job's own scope, **read from inside the child** |
| resident set of the same child | 1 473 036 288 B = 1.3719 GiB |
| cap never shrunk to evade a refusal | yes |

The peak is read from inside the child on purpose. `memory.peak` is a **kernel high-watermark**, not
a sample and not a value the launcher writes back on release — so it is a **measurement even for a
short child**, where a *sampled* tree peak would be only a floor, and it is not lost when a release
never succeeds. That is the exact failure mode RB01 §3.7 recorded: shard 1 and shard 2 failed
identically and only one kept its number.

The walk, from inside the child: 53 MB at entry → 55 MB after the bytes were verified → 366 MB after
the parquet read → 483 MB after the channel matrix → **1 464 MB after the window gather**.

**And a small result that matters for how these numbers are used:** the child's resident set came out
**9.16 MB above** the whole-scope charged peak. So RSS is not a conservative *lower* bound on a cgroup
peak either. The two quantities are simply not interchangeable **in either direction**, and a cap may
be sized only from the cgroup figure.

### 4.4 The footprint arithmetic a successor can be costed from

| | |
|---|---|
| panel rows / numeric channels | 2 075 259 / 7 |
| channel matrix, float32 | 58 107 252 B |
| usable origins at W1440 / h60 | 2 073 759 |
| bytes per W1440 window | 40 320 |
| windows this pilot materialised | 20 000 (bounded by the declared `max_windows`) |
| window tensor at that count | 806 400 000 B |
| if every origin were materialised at once | 83 613 962 880 B = **77.9 GiB** |

The block never materialises the whole population — `Batches` gathers per batch — which is precisely
why **no data-stage figure can be a cell's cap**.

### 4.5 Residual budget: `NO_NEW_MEASUREMENT`

The measured data-stage floor is **1 463 877 632 B**. The model, gradient, optimizer-slot, Keras-graph
and per-batch-gather terms **were not built and are therefore not measured**. A null is not small.
**No successor cap is declared from this round.** The estimand of the block this lane serves remains
**`MATCHED_BUDGET_DIFFERENCE`, never `CONVERGED_ACCURACY`**: every cell is censored at 600 observed
updates, and a fixed-600-update design cannot answer a converged-accuracy question.

## 5. Closure table

`docs/audits/evidence/Q2_HOUSEHOLD_20260929/CLOSURE_TABLE.{json,md}`, generated from this round's own
artifacts by `tools/df_q2_household_closure.py` (**9/9 generator tests pass**, and they fail on the
ways that matter: a null rendered as a zero, a null peak read as small, an estimand silently becoming
`CONVERGED_ACCURACY`, custody claimed without an accepted terminal or a warehouse row, a warehouse
digest disagreeing with the receipt).

| unit | estimand | model error | paired naive, same rows | skill | reference (source) | comparability | scope peak | custody |
|---|---|---|---|---|---|---|---|---|
| `probe` | `MATCHED_BUDGET_DIFFERENCE` | `null` | `null` | `null` | `NOT_CARRIED` | `NOT_COMPARABLE` | 384 962 560 B | accepted terminal + warehouse row |
| `pilot` | `MATCHED_BUDGET_DIFFERENCE` | `null` | `null` | `null` | `NOT_CARRIED` | `NOT_COMPARABLE` | 1 463 877 632 B | accepted terminal + warehouse row |

**Round verdict: `NO_NEW_MEASUREMENT`, 0 scored rows.** Every `null` is an **absent measurement,
never a numeric zero**: these units fit no model, hold no checkpoint and score no evaluation
population, so there are no rows for a naive to be paired on and no ratio for a skill.
`NOT_COMPARABLE` because a chain and a memory footprint are not an error — and because the block's
own estimand is a matched-budget difference, which no converged-training publication shares.

## 6. Report, in the §6 shape of `M5PHET/docs/WORK_PLAN_2026_09_24.md`

```
Q2 HOUSEHOLD — restoration verified, governed chain proved, memory number corrected
repo/branch/tip: predictor / satoshi/q2-household-governed-20260929, own worktree, cut at 956dc43c
files: tools/df_public_panels_restore.py   (preflight --already-serving, verify --content)
       tools/df_q2_household_governed.py   (new: the two governed units and the inside-child peak)
       tools/df_q2_household_closure.py    (new: this round's closure table, from artifacts)
       tests/test_df_q2_household_closure.py (new: 9 tests)
       docs/audits/work_plan/SATOSHI_Q2_HOUSEHOLD_GOVERNED_2026_09_29.md
       docs/audits/evidence/Q2_HOUSEHOLD_20260929/{RESTORATION_RECEIPT,PREFLIGHT,REHEARSAL,
           VERIFY_CONTENT,GOVERNED_CHAIN,MEMORY_CORRECTION,CLOSURE_TABLE}.json + CLOSURE_TABLE.md
suites: tests/test_df_q2_household_closure.py 9 passed
acceptance:
  restoration        ALL SIX TRIAGE CHECKS VERIFIED
                     preflight --already-serving exit 0, every check green
                     rehearse --already-serving exit 0, identity matches the live unit field for field
                     verify --content exit 0: HTTP 200, 10 890 295 B, b3192c0b…c8db byte-exact,
                       availability contract 93d945cf… unchanged
                     refusals hold: ranged 422, holdout-spanning 422
                     registry 91f1e627… unchanged, six lakes, other five store units untouched
                     receipt retained; rollback proven as a no-op 2026-09-28 and deliberately NOT run
  governed chain     2/2 units COMPLETED and accepted (sent 1 / pending 0, reconciliation empty)
                     both VERIFIED_TRANSFER with cached=false — a warm cache did NOT skip the lake
                     bytes re-digested inside each child; both terminal digests read back out of the
                       live warehouse and both agree with their receipts
  pilot cgroup peak  1 463 877 632 B = 1.3633 GiB, cap 6.0 GiB declared before the run,
                     memory.peak of the job's own scope read from INSIDE the child
  closure            NO_NEW_MEASUREMENT, 0 scored rows, every error cell null with its reason
what is NOT done / refused / not measured:
  - no service started, stopped, enabled, disabled or restarted; the restoration was the owner's
  - no successor cap declared: the model, gradient, optimizer-slot and graph terms were NOT BUILT
    and are therefore NOT MEASURED; 1.3633 GiB is a data-stage FLOOR, not a cell's cap
  - the W1440 refused attempt and v1 NOT relaunched; the six missing cells stay missing
  - the twelve historical cells NOT refitted, NOT re-bound to v2, no allocation reset;
    NOT_BOUND_TO_A_SEAL, custody UNCHECKED, 0 verified rows, closure FAILED
  - no holdout or reserve opened; the holdout was probed only to confirm a range is still refused
  - the preferred RTX 5090 host read ONCE to confirm the standing ruling, given no work, not
    re-measured to argue otherwise
  - no credential printed, copied, logged, written to a second file, issued or rotated
  - fresh-process bitwise replay of any historical cell: UNMEASURED, and not attempted
  - two corrections carried: the queue's "7.4 GiB resident set" is a kill-time cgroup peak of a
    multi-child wrapper scope (and the order's own reading of that number is corrected too); the
    2026-09-20 restore line would remove TWO registered lakes, not one
  - a finding nobody had: the cgroup-peak instrument AND successor v2's seal exist on exactly ONE
    branch and on no other, master included — so no mainline successor can report a per-cell peak
```

## 7. The single next decision, and it is not mine

An adequately costed successor for the six missing W1440 cells needs three things this round did not
produce and did not pretend to: a **per-cell** cgroup peak (which requires both the `84bcd605`
instrument and a `run_units` that gives each child its own scope), a **reachable** v2 seal (today it
lives on one branch), and a cost basis that includes the model, gradient and optimizer terms. Until
those exist, **`NO_PROGRAMME_TASK_IS_READY` for a long-window fit**, and saying so is the honest
outcome rather than a gap to be filled with a number.
