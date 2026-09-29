# Traffic's training footprint, measured — and the parity the untrained proof could not give

Satoshi III (Mujuro Utsutsu), successor technical lead. 2026-09-29 (America/Bogota).
Branch `satoshi/traffic-train-pilot-20260929`, own worktree off `8d461628`.
Order: row **B: Traffic** of `docs/handoffs/SATOSHI_Q2_RESOURCE_SUCCESSOR_2026_09_29.md`
at `04f02555`, and QRM02 of the same document.
Design `6cba7e20aa359986…815285575` · protocol `322262e8d78abd…3ba520`.

**The one-line result.** Traffic's TRAIN-only footprint at the sealed recipe is
**1.724 GiB** whole-cgroup at h96 and **2.030 GiB** at h720, inside the
authorized **8 GiB** cap, at **0.123 s** and **0.153 s** per optimizer step. The
bounded evaluator's **3.600 GiB** from `8d461628` is not this number and never
was: it carried no optimizer, no gradients and no optimizer slots. And the
reducer parity that delivery proved on an **untrained** model now holds
**bitwise on trained outputs** — MAE and MSE identical to the last float32 bit
over the complete h96 validation population of the weights this pilot's own
optimizer produced.

**What this is not.** No Traffic cell was trained to anything. No test window
was opened by any child in this delivery. No accuracy number here is a result:
the two metric values that appear are the two sides of one bitwise comparison
over a model with **32 optimizer updates**, and a 32-update model's error is not
a result. **`NO_NEW_MEASUREMENT` of Traffic model quality.** Nothing here
authorizes a Traffic campaign.

---

## 0. Stated before dispatch

Declared before the first child was submitted, and kept.

| | |
|---|---|
| worktree | `/home/harveybc/Documents/GitHub/.worktrees/predictor-traffic-train-20260929` |
| source tip | `8d461628` on `satoshi/traffic-bounded-evaluator-20260929` (reconciled in §1) |
| first command | `python -m pytest tools/test_tsl_train_pilot.py` under `crispdm-run -m 3G`, on the coordinator, before any GPU child |
| acceptance test | a record whose verdict is `MEASURED_WITHIN_CAP`: every one of the 13 required stages present in ONE cell scope, the optimizer's slots **measured** non-zero after the first step, the test split **not opened**, and the whole-cgroup peak at or under the cap the child was admitted under. A missing peak yields `UNDETERMINED`, never success |
| budget | **stages** per child: 5 warmup + 25 steady optimizer steps, one train step at each distinct final-batch shape, one checkpoint write + one reload, one **complete** validation pass; **host cap** 8 GiB `MemoryMax` per child inside `crispdm-batch.slice` (ceiling 14 GiB); **wall** 40 min per child; **device** WORKER_A's RTX 4090, 16 376 MiB, asserted by physical UUID from inside the child; **sequential**, never two children at once |
| remaining allocation at that moment | zero live leases · `MemAvailable` 21.00 GiB · desktop reserve 3 GiB · **host free for new 18.00 GiB** · slice ceiling 14 GiB, slice current 4.90 GiB, **room under the slice ceiling 9.10 GiB**; an 8 GiB request fits **both** gates, so **no new allocation was invented and no cap was reduced** |
| actual blocker | the **native-witness** child (the author's own unchunked `test()` beside the bounded path) needs a **12 GiB** cap. 12 GiB > 9.10 GiB of slice room, so it is `QUEUED / SLICE_AGGREGATE_BUDGET`. It was **not** run and the cap was **not** lowered. The exact additional request is in §6 |

The 8 GiB cap is not a number I chose here: it is the cap already sealed in §6
of `8d461628` ("*declared cap 8 GiB each, sequential*"). I reconciled that
authority rather than writing a new one.

**Lane A's launch path was not used and not needed.** This lane went through the
existing `crispdm-run` atomic admission on the admitted worker, which is the
independently verified runner `8d461628` already used. No shared-runner file was
touched.

---

## 1. Reconciling `8d461628`, and the one thing in it that is now wrong

Read in full. Its substance stands, and this delivery depends on three of its
results: the term-by-term split of the author's evaluation path, the sealed
reducer's bitwise equality with `utils.metrics.metric`, and the measured
**3.600 GiB** whole-cgroup peak of the bounded path at h720. Its own statements
of what it did **not** establish are also carried forward unchanged (§5 below).

Two corrections, both mine:

**(a) The 3.600 GiB figure must never be read as a training footprint, and this
delivery proves the gap is real rather than asserting it.** That child built the
model and reloaded a checkpoint. It built **no optimizer**, held **no
gradients**, instantiated **no optimizer slots** and wrote **no checkpoint of
its own work**. Measured here on the same device, the three terms it was missing
are **29.2 MB of gradients, 58.3 MB of optimizer slots and a 29.2 MB
checkpoint** at h96 — small in isolation, and the reason the *whole* training
footprint is nonetheless 1.724 GiB is the train and validation loaders'
resident arrays, which that child also did not hold. The peak of a thing is not
the sum of the terms you can name; it is measured or it is unknown.

**(b) `df_tsl_repro.full_reproduction_cost` says the epoch budget is an upper
bound because "EarlyStopping (patience 3 on the validation MSE) can only shorten
it". At the pinned author commit that is false.** `utils/tools.py`
`EarlyStopping.__call__` calls `save_checkpoint` unconditionally and its entire
scoring-and-counter body is **commented out**, so `early_stop` is never set. The
consequence is not cosmetic: **every cell runs all 30 epochs**, the 30-epoch
budget is the **schedule** and not a ceiling, and the checkpoint is written
**once per epoch**. `early_stopping_state()` reads that from the pinned source
and a test asserts it, so no later projection can quietly reintroduce the
shortening. The sealed module was **not** changed; the projection that replaces
it is in §4.

`git diff 8d461628 -- tools/df_sota_repro.py tools/df_tsl_repro.py
tools/df_tsl_execute.py tools/df_tsl_bounded_eval.py docs/contracts/` is
**empty**. The sealed modules were called, never edited — including the bounded
evaluator delivered at the source tip.

---

## 2. The training footprint, measured

Both children: WORKER_A (the admitted secondary worker), its RTX 4090
`GPU-a8bd1b2c-…6780f9` asserted **inside the child** by physical UUID, python
3.12.13, torch 2.13.0+cu130, numpy 2.5.1. The registered Traffic bytes
(`cb06463d…0dab0b16`) were **re-verified by sha256 inside every child** before
the loader saw them, and `code_drift` against the sealed author digests was
empty in every child.

The recipe was **not** altered. Batch 16, float32, Adam at lr 1e-3, `lradj`
cosine, `d_model` 512, `e_layers` 3, `patch_len` 96, `enc_in` 862, the author's
own `Dataset_Custom` loader at `num_workers` 1, 30 sealed epochs, patience 3.
**The only thing bounded was the number of steps**, and the record says so in
that field.

| | **h96** | **h720** |
|---|---:|---:|
| elapsed horizon (Traffic's own hourly clock) | **96 h = 4 d** | **720 h = 30 d** |
| **whole-cgroup peak, in-child `memory.peak`** | **1 851 158 528 B = 1.724 GiB** | **2 179 903 488 B = 2.030 GiB** |
| declared cap (passed once, never shrunk) | 8 589 934 592 B = 8 GiB | 8 589 934 592 B = 8 GiB |
| headroom | 6.276 GiB | 5.970 GiB |
| **steady step, median of 25** | **0.123025 s** | **0.153422 s** |
| steady step p90 / min / max | 0.123145 / 0.122814 / 0.123467 | 0.154150 / 0.152746 / 0.155775 |
| first step (warmup, reported apart) | 0.51729 s = **4.2×** the steady step | 0.521493 s = **3.4×** |
| parameters | 7 306 748 | 7 626 860 |
| parameter bytes | 29 226 992 | 30 507 440 |
| gradient bytes after the first backward | 29 164 928 | 30 445 376 |
| **optimizer slot bytes, MEASURED after the first step** | **58 330 040** | **60 890 936** |
| attention-mask bytes (`_get_mask`, on the device) | 8 916 528 | 8 916 528 |
| peak GPU allocated / reserved | 4.816 / **5.365 GiB** | 4.884 / **5.514 GiB** |
| complete validation pass | 104 batches / 6.07 s | 65 batches / 11.71 s |
| validation seconds per batch | 0.058385 | 0.180086 |
| checkpoint bytes / write / reload | 29 248 117 B / 0.0248 s / 0.0238 s | 30 528 565 B / 0.0272 s / 0.0248 s |
| child wall / CPU | 15.41 s / 15.20 s | 22.62 s / 22.27 s |
| verdict | **`MEASURED_WITHIN_CAP`** | **`MEASURED_WITHIN_CAP`** |
| record digest | `ea143639…809755` | `06966c3d…4f78e6b` |

**The optimizer slots were measured, and the arithmetic closes.** Adam allocates
`exp_avg` and `exp_avg_sq` lazily on the first `step()`. At h96 the measured slot
total is **58 330 040 B**, which is exactly `2 × 29 164 928` (two slots per
gradient-bearing parameter) `+ 46 × 4` (the per-tensor `step` scalars). It is
read from the optimizer's own state, not derived from the parameter count,
because a derivation cannot tell an instantiated slot from an absent one — and
`pilot_verdict` **refuses** a record whose slot total is zero. A footprint
measured before `step()` is not a training footprint, and that record would have
said `REFUSED_NO_OPTIMIZER_STATE`.

**A small discrepancy, named rather than rounded away.** Parameter bytes exceed
gradient bytes by **62 064 B** at h96, i.e. **15 516 parameters receive no
gradient** and therefore carry no Adam slot. That is 46 parameter tensors with
state out of a larger module list. I have not chased which submodule they belong
to; it is named, not explained.

### Where the memory goes: the stage peaks in one cell scope

The kernel's own monotone high-water for the job's whole transient scope, read
at each stage boundary from inside the child. Because it is a kernel high-water
and not a sample, a stage shorter than the launcher's 5 s sampler is still
measured — the two numbers are different quantities and are never interchanged.

| stage | h96 | h720 | what had happened |
|---|---:|---:|---|
| entry | 0.480 | 0.479 | interpreter, torch, CUDA context |
| after model build | 0.541 | 0.540 | the model and the 8.9 MB mask tensor on the device |
| **after train loader** | **1.028** | **1.026** | **the author's own train split, resident. +0.49 GiB — the largest single jump** |
| after validation loader | 1.178 | 1.179 | the validation split beside it |
| after optimizer build | 1.178 | 1.179 | Adam holds nothing until the first step |
| **after the first optimizer step** | **1.713** | **1.908** | **gradients, activations, the pinned staging buffers and the slots, all at once** |
| after 5 warmup steps | 1.722 | 1.953 | allocator growth settling |
| after 25 steady steps | 1.724 | 1.955 | +1.7 MB over the whole steady run |
| after both final-batch shapes | 1.724 | 1.955 | unchanged |
| after checkpoint write | 1.724 | 1.955 | unchanged |
| after checkpoint reload | 1.724 | 1.955 | unchanged |
| **after the complete validation pass** | 1.724 | **2.030** | h720's `.detach().cpu()` of `[16, 720, 862]` blocks: **+0.075 GiB**. At h96 it cost nothing measurable |
| exit | **1.724** | **2.030** | |

Two readings worth keeping. First, **the resident cost of training Traffic is
mostly the data, not the model**: 1.18 of 1.72 GiB at h96 is there before a
single gradient exists. Second, **the peak is reached in the first optimizer
step**, not accumulated over the run — 25 further steps added 1.7 MB. That is
what makes a 30-step pilot a defensible instrument for the per-step terms, and
it is also exactly why it is **not** a defensible instrument for anything
reachable only after many epochs (§4).

### The largest final-batch shape: measured, and it did not bind

The author's `data_provider` sets `drop_last=False` for **every** flag, so every
split ends on a partial batch the steady loop never sees. A new shape can force
fresh allocations while the steady blocks are still cached, which is why it was
measured rather than reasoned about. Over the splits this pilot opens, the final
batches are **9 rows** (train) and **13 rows** (validation); the largest is 13,
and both are **smaller** than the steady 16.

| final shape | h96 train step | h720 train step | cgroup peak after |
|---|---:|---:|---|
| 9 rows (train's own last batch) | 0.0699 s | 0.0845 s | unchanged |
| 13 rows (validation's, the largest) | 0.0980 s | 0.1213 s | unchanged |

**So the final batch is cheaper, and the binding shape is the steady full
batch.** That is the measurement, and I am reporting that the risk did not
materialize rather than implying it did. The shapes were reached by
materializing the split's own last windows through torch's own `default_collate`
— the **shape** is what is priced, and iterating a 756-batch epoch to arrive at
it is not part of a bounded pilot. The record says so.

### The replay, because one execution on one host is not a measurement twice

`8d461628` §10 item 2 named its own single-execution weakness. Both children ran
**twice**, in fresh processes, under fresh admission. (The first pair also
carried a wrong descriptive string in my own record — it called the checkpoint's
weights "untrained" when they hold 32 optimizer updates — which is why the pair
was rerun; the first records are retained as `*.attempt1.json`.)

| | first execution | second execution | agreement |
|---|---:|---:|---|
| h96 whole-cgroup peak | 1 976 864 768 B | 1 851 158 528 B | **6.8 % apart** |
| h720 whole-cgroup peak | 2 179 960 832 B | 2 179 903 488 B | 0.003 % apart |
| h96 steady step | 0.122965 s | 0.123025 s | 0.05 % |
| h720 steady step | 0.152988 s | 0.153422 s | 0.28 % |

**The per-step cost replays; the whole-cgroup peak does not replay to the
byte.** A 6.8 % spread at h96 is real and it is the honest reading: `memory.peak`
counts file-backed pages, so page-cache state differs between executions. The
peaks in §2 are the **second** execution's, and the spread is the reason the
projection in §4 carries explicit headroom instead of pretending the peak is a
constant.

### The transfer that was forbidden, and the number that shows why

Weather's TRAIN-only pilot measured **0.010006 s** per step at a **1.365 GiB**
whole-cgroup peak. Traffic's h96 step is **0.123025 s** — **12.3 times** as
expensive — and its peak is **1.26 times** Weather's. **862 channels against 42
at `d_model` 512 against 128 is a different model, and pricing Traffic from
Weather's pilot would have under-costed the twelve cells by an order of
magnitude.** `refuse_foreign_pilot` now refuses that transfer in code: a pilot
whose `design_sha256` or `dataset` is not this design's cannot price it, and a
test exercises the refusal with a Weather record against the Traffic design. The
superseded `df_tsl_train_pilot.v1` schema is refused by the same gate, with its
reason recorded: it measured no slots, no final-batch shape, no checkpoint and
no validation, so **its peak is a floor for a training cell, never the cell's
footprint**.

---

## 3. Bounded-evaluator parity, on TRAINED outputs

**The gap, stated precisely.** `8d461628` proved `author_metric_exact` bit-equal
to the author's own `utils.metrics.metric` over the complete real h96 test
population — of an **untrained** model. An untrained field can be near-constant,
and two reductions of a near-constant array can agree for a reason that has
nothing to do with either reduction. That is the difference the order named, and
it matters.

**What ran.** One child, 8 GiB, on the same worker and device. It rebuilt the
author's model from the sealed argv, wrote **the h96 pilot's own checkpoint** to
the author's own checkpoint path and reloaded it with `load_state_dict` —
exactly as `train()` ends and `test(test=1)` begins — with the file's sha256
checked against the pilot record before and after transport. It then streamed
the author's own **validation** loader through the trained model and ran **both**
reductions over the complete population.

**The test split was not opened.** `_get_data` is wrapped in this child too and
refuses `flag="test"`; the record carries `splits_opened: ["val"]`. Parity of a
reduction is a property of the arithmetic and of the array, not of which split
the array came from, and a cost pilot has no business reading held-out data.

| | value |
|---|---|
| weights | the pilot's own checkpoint `6664470c…297fe3`, **32 optimizer updates**, selected on nothing |
| population | **1 661 × 96 × 862 = 137 451 072** float32 elements, the complete sealed validation population, 104 batches, sizes 16 … 13 |
| the author's own `utils.metrics.metric`, unchunked | MAE `0.4809713661670685` · MSE `0.6688601970672607` |
| the sealed bounded route `df_sota_author_metric_exact.v2` | MAE `0.4809713661670685` · MSE `0.6688601970672607` |
| **bitwise equal** | **yes, both. Absolute difference exactly 0.0** |
| the independent float64 reduction, beside and never instead | MAE `0.48097134762646543` · MSE `0.6688602273876249` |
| non-degeneracy of the trained field | std **0.9769**, min −13.750, max 40.957, **5 084 608 distinct values** in a 64-window sample |
| measured whole-cgroup peak | 3 496 890 368 B = **3.257 GiB**, under the 8 GiB cap |
| forward pass over the population | 6.41 s · arrays 549 804 288 B each, deleted after their digests |
| **verdict** | **`BIT_EQUAL_ON_TRAINED_OUTPUTS`** |
| record digest | `449adb33…41d5ea` |

**The non-degeneracy line is the point of the exercise.** A prediction field
with 5.08 million distinct values and a standard deviation near 1 is not a
constant, so the bitwise agreement above is an agreement about the two
reductions and not an artifact of a degenerate array. Both sides are other
people's code: `utils.metrics.metric` is the author's and `author_metric_exact`
is the sealed route. **This delivery's own module reduces nothing**, and a test
asserts that against its source.

**What this does NOT close, and the words are chosen.** Parity of the bounded
**writer's row placement**, and parity of the **model's outputs through the
author's own unchunked `test()`** on the **test** population, remain exactly
where `8d461628` left them: **NOT MEASURED**. The native-witness child that
would settle both needs a 12 GiB cap and was not admitted (§0, §6). h336 and
h720 trained parity are also **NOT MEASURED**: at h720 the author's unchunked
function needs `3 × 2.398 = 7.19 GiB` of temporaries on top of `4.80 GiB` of
mapped arrays, about 12 GiB, which the same slice budget does not admit. No cap
was lowered to reach any of them.

---

## 4. The projection, with explicit headroom — and it is a bracket

Priced from the two **measured** pilots and the sealed batch counts. No term is
taken from another dataset.

| cell (per seed) | elapsed horizon | train batches | vali batches | epoch seconds | **cell hours** | per-step cost class |
|---|---|---:|---:|---:|---:|---|
| h96 | 96 h = 4 d | 756 | 104 | 111.60 | **0.930** | **MEASURED at this horizon** |
| h192 | 192 h = 8 d | 750 | 98 | 110.16 | 0.918 | DERIVED at h96's rate — a **floor** |
| h336 | 336 h = 14 d | 741 | 89 | 108.00 | 0.900 | DERIVED at h96's rate — a **floor** |
| h720 | 720 h = 30 d | 717 | 65 | 153.25 | **1.277** | **MEASURED at this horizon** |

**The twelve cells' training cost is 12.08 – 15.02 GPU-hours** on this device.
The lower end prices h192 and h336 at h96's measured rate; the upper end prices
them at h720's. The bracket is the reportable quantity; the 12.08 h point is
**not** a best estimate, and the record labels it so. The bracket brackets the
truth only if the per-step cost is monotone in the prediction length — the two
measured points are consistent with that and **nothing here proves it**, so it
is carried as a stated assumption, and h192 and h336 remain **unmeasured**.

**Every cell runs all 30 epochs** (§1b). This is a schedule, not a ceiling that
early stopping shortens.

**One term inside each epoch is DERIVED, and it is named.** The author's
`train()` runs a per-epoch logging pass over the **test** loader. This pilot does
not open that split, so its **time** is priced from the measured validation
forward rate times the sealed test batch count (12.49 s per epoch at h96, 31.51 s
at h720) and its **memory was not measured at all**. It is listed under
`unpriced_terms` in the record rather than folded into the peak.

**Memory, as a projection with explicit headroom:**

| | |
|---|---|
| worst **measured** training peak | 2 179 903 488 B = **2.030 GiB** (h720) |
| headroom fraction, stated | **0.25** |
| **projected cap** | 2 724 879 360 B = **2.538 GiB** |
| class | **PROJECTION WITH EXPLICIT HEADROOM** |
| what it is **not** | a proof that the training path cannot exceed it |

A 30-step pilot on one host in two executions does not bound a 30-epoch run: a
longer run reaches allocator and fragmentation states a short one does not, and
the 6.8 % replay spread at h96 (§2) is direct evidence that this peak is not a
constant. The horizons no pilot covered, any state reachable only after many
epochs, and the per-epoch logging test pass are all listed as **not measured** in
the record. **A campaign request would still declare 8 GiB**, which the
measurements show is ample rather than tight — that is the point of measuring.

**The whole cost of one Traffic reproduction, assembled from both sides:**

| side | figure | class |
|---|---|---|
| training, 12 cells | **12.08 – 15.02 GPU-hours** | MEASURED endpoints, DERIVED bracket |
| training host memory | **2.030 GiB** worst measured, 2.538 GiB projected at 25 % headroom | measured / projection |
| training device memory | **5.514 GiB** reserved, worst measured | measured |
| evaluation, 12 cells | ~0.18 GPU-hours | DERIVED in `8d461628`, offered as a bound |
| evaluation host memory | **3.600 GiB** at h720 through the bounded path | measured in `8d461628` |
| evaluation transient disk | **12.925 GiB** once, at h720, sequential | measured in `8d461628` |

---

## 5. What is VERIFIED, what is measured, and what is not measured

| claim | class |
|---|---|
| Traffic's TRAIN-only whole-cgroup peak is 1.724 GiB (h96) and 2.030 GiB (h720) with the optimizer's slots live, inside an 8 GiB cap | **measured**, two executions each, in-child `memory.peak`, one host, one device; `NOT_INDEPENDENTLY_VERIFIED` |
| the peak replays to 0.003 % at h720 and to 6.8 % at h96 | **measured**; the h96 spread is reported, not smoothed |
| the steady per-step cost is 0.123025 s (h96) and 0.153422 s (h720) | **measured**, 25 steady steps after 5 warmup, warmup reported apart |
| the optimizer's slots were instantiated and are 58 330 040 B (h96) / 60 890 936 B (h720) | **VERIFIED** — read from the optimizer's own state, and the arithmetic `2 × grad + 46 × 4` closes exactly |
| every required stage — model build, both loaders, optimizer build, first step, warmup, steady, both final-batch shapes, checkpoint write, checkpoint reload, complete validation pass — ran in ONE cell scope | **VERIFIED** by the record's own stage list; a missing stage yields `REFUSED_INCOMPLETE_STAGE_COVERAGE` |
| no child in this delivery opened the test split | **VERIFIED** — `_get_data` is wrapped and refuses `flag="test"`; every record carries `splits_opened` |
| the largest final-batch shape over the opened splits is 13 rows, smaller than the steady 16, and it did not raise the peak | **measured**; the shapes were materialized from the split's own last windows |
| the checkpoint write and reload cost 0.025 s and 0.024 s and moved the cgroup peak by nothing measurable | **measured** |
| the complete validation pass costs 6.07 s (h96) and 11.71 s (h720), and at h720 it raises the peak by 0.075 GiB | **measured**; its loss **value** was deliberately not recorded |
| early stopping cannot fire at the pinned author commit, so 30 epochs is the schedule and the checkpoint is written once per epoch | **VERIFIED** as a reading of `utils/tools.py` at the pinned commit, asserted by test |
| `author_metric_exact` is bit-equal to `utils.metrics.metric` on the complete h96 validation population of **TRAINED** outputs, and that field is not degenerate | **VERIFIED, bitwise**, MAE and MSE, absolute difference exactly 0.0, on a field with 5 084 608 distinct sampled values |
| the bounded **writer's** row placement, and the author's own unchunked `test()` over the **TEST** population, on trained weights | **NOT MEASURED.** The 12 GiB native-witness child is `QUEUED / SLICE_AGGREGATE_BUDGET`; no cap was lowered |
| trained-output parity at h192, h336, h720 | **NOT MEASURED.** At h720 the author's own function needs ~12 GiB, which the slice budget does not admit |
| the twelve cells' training cost | **12.08 – 15.02 GPU-hours**: MEASURED at two horizons, **DERIVED bracket** at the other two, under a stated monotonicity assumption |
| the 2.538 GiB projected cap | **PROJECTION WITH EXPLICIT HEADROOM.** Not a proven maximum. A 30-step pilot does not bound a 30-epoch run |
| the per-epoch logging pass over the test loader | **time DERIVED** from the measured validation rate and the sealed batch count; **memory UNMEASURED**; the split was never opened |
| any Traffic model quality number | **`NO_NEW_MEASUREMENT`.** No cell was trained. The two metric values in §3 are the two sides of one parity comparison over a 32-update model |
| Weather's training footprint as a bound on Traffic's | **REFUTED as a method.** Traffic's step is 12.3× Weather's; the transfer is now refused in code and by test |

---

## 6. The one exact additional request

I am **not** requesting a Traffic campaign and nothing here authorizes one. One
measurement remains blocked, and rather than lower a cap to reach it I state the
request exactly:

> **One child, declared cap 12 GiB, wall 40 minutes, sequential, on WORKER_A's
> RTX 4090, under design `6cba7e20…85575` / protocol `322262e8…ba520`: the
> native-witness parity of the bounded writer at h96 on the pilot's own trained
> checkpoint — the author's unchunked `test()` and the bounded path in one child
> from one on-disk checkpoint, compared on the sha256 of the arrays and on the
> metric values bitwise.**

**Why it does not fit today, with the numbers.** `crispdm-batch.slice` has a
15 032 385 536 B (14 GiB) ceiling. With **zero live leases**, its
`memory.current` still reads 4.9 – 5.1 GiB — residual page cache charged to the
slice by earlier children — so the room for a new cap is **9.04 GiB**. A 12 GiB
request is refused as `QUEUED / SLICE_AGGREGATE_BUDGET`. The host itself is not
the constraint: 17.86 GiB was free for new work at the last reading.

**The exact thing being asked for**, either one:

1. raise `crispdm-batch.slice` `MemoryMax` from 15 032 385 536 B to
   19 327 352 832 B (18 GiB) for the duration of that single child, or
2. authorization to submit the 12 GiB child when the slice's residual charge
   falls below 2 GiB, whichever comes first.

Both are **service-level changes I did not make and am not authorized to
make**. I changed no cap, ceiling, cache, swap, oomd setting or kernel
parameter, and I did not resubmit under a different slice or a different job
name, because asking again under another name is the same evasion by another
route.

---

## 7. Not disturbed, for the record

**Nothing was displaced.** Every admission reading in this delivery showed
**zero live leases** on the admitted worker, so no other lane's work was queued
behind mine and none of mine displaced anyone. Fresh aggregate admission was
taken **per child** through `crispdm-run -q`, the launcher's own atomic
reservation, and every child was **sequential** — never two at once. **No
declared cap was ever shrunk**, before or after a refusal.

**The preferred RTX 5090 host was not used.** Its host-memory restriction is not
lifted by this order and I did not treat it as lifted.

**The coordinator stayed light.** Its only load was the pytest suite under a
3 GiB cap, about one second of CPU, before any GPU child.

No service was started, stopped or restarted. No host was rebooted, no driver
reloaded, no broker touched. No cap, cache, swap, oomd setting or kernel
parameter was changed. No committed sample output and no live checkout was
altered. The sealed modules and the bounded evaluator delivered at the source
tip were **called, never changed**. No process outside my own children's own
scopes was signalled, and every child ran to completion. Host names, addresses,
tokens and account identifiers appear nowhere in this document or in the
committed code.

---

## 8. Artifacts

Committed on `satoshi/traffic-train-pilot-20260929`:

- `tools/df_tsl_train_pilot.py` — the TRAIN-only cost pilot (`df_tsl_train_pilot.v2`),
  the trained-output reducer parity (`df_tsl_trained_reducer_parity.v1`), the
  batch geometry, the hourly clock, and the projection with its bracket and
  explicit headroom. It introduces no model, loader, loss, scorer, reduction,
  recipe or margin, and it defines no reduction of its own.
- `tools/test_tsl_train_pilot.py` — **44 tests** of the generators: the final-batch
  arithmetic under the author's own `drop_last=False`; that Traffic's 96 steps are
  96 hours where Weather's are 16; the verdict boundaries (the cap boundary is
  inclusive, an unread peak is `UNDETERMINED` and never success, a zero-slot
  record is refused, a missing stage is refused, test access refuses the whole
  record whatever else is true); the refusal of a foreign dataset's pilot and of
  the superseded v1 schema; that the projection labels measured against derived,
  brackets the unmeasured horizons between measured rates, never claims early
  stopping shortens what cannot stop early, and never turns an unread peak into
  zero; and, against this module's own source, that it opens no test split,
  writes no cap, records no validation loss value and reduces nothing.
- `docs/audits/work_plan/SATOSHI_TRAFFIC_TRAIN_PILOT_2026_09_29.md` — this document.

Operator-retained, not committed (they carry host detail):
`~/.local/state/crispdm-data-foundation/tsl_traffic_train_pilot_20260929/` —
`TRAIN_PILOT.traffic.h96.json` (`ea143639…`),
`TRAIN_PILOT.traffic.h720.json` (`06966c3d…`), both first executions retained as
`*.attempt1.json`, `TRAINED_PARITY.traffic.h96.json` (`449adb33…`),
`TRAIN_PROJECTION.traffic.L96.json` (`c5a5610b…`),
`GEOMETRY.traffic.h96.json`, `GEOMETRY.traffic.h720.json`, and the design and
characterization copied from the bounded-evaluator lane unchanged.

Suites, at this tip: **`tools/test_tsl_train_pilot.py` 44 passed ·
`tools/test_tsl_bounded_eval.py` 21 passed · `tools/test_tsl_execution.py`
23 passed · `tools/test_tsl_producer_contract.py` 22 passed — 110 passed**, with
the production warehouse provider on the path. The new suite also passes on the
worker against the deployed copy of the module, whose sha256 equals the
committed file's.

---

## 9. What the auditor should attack first

1. **The projection's bracket rests on a monotonicity assumption.** h192 and
   h336 were never run. If the per-step cost is not monotone in the prediction
   length, the bracket does not bracket, and the honest answer would be two
   measured points and two holes. Two more 30-step children would close it for
   about 40 seconds of GPU time; I did not run them because the authorization
   named h96 and h720.
2. **The h96 peak replayed 6.8 % apart.** `memory.peak` counts reclaimable
   file-backed pages, so the reported peak is conservative — and also means a
   smaller number is achievable under pressure and this one is not a floor. Two
   executions are not a distribution.
3. **The trained-output parity was proved on the validation split, not the test
   split.** The reduction's equality is arithmetic and population-independent,
   which is why the split does not matter to *that* claim — but the writer's row
   placement and the author's own `test()` are still unproven on trained
   weights, and no sentence in §3 should be read as covering them.
4. **32 optimizer updates is "trained" only in the sense that every weight
   moved.** It is not a converged model, it was selected on nothing, and its
   error is not a result. If the auditor thinks parity on a 32-update model is
   still too close to parity on an untrained one, the objection is legitimate
   and the answer is a longer run, which is a campaign and is not authorized.
5. **`CRISPDM_RESERVATION_BYTES` is still not exported by the launcher**, so each
   child compared itself against the cgroup's own `memory.max` rather than
   against the reservation. The same integer by construction, and the record
   names which source it used — but it is a fallback and should be checked, as
   `8d461628` already said.

— Satoshi III (Mujuro Utsutsu), successor technical lead, 2026-09-29.
