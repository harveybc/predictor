# Traffic's long horizons: an implementation footprint, and the bounded path that removes it

Satoshi III (Mujuro Utsutsu), successor technical lead. 2026-09-29 (America/Bogota).
Branch `satoshi/traffic-bounded-evaluator-20260929`, own worktree off `f40fae93`.
Specification: §2 of `MUSASHI_CAMPAIGN_STATUS_TRIAGE_2026_09_29.md` at `50d06e50`.

**The correction, stated first and without hedging.** I told the owner that
Traffic's two long horizons are impossible on this fleet. The auditor said that
sentence confuses an implementation footprint with a model lower bound, and he
is right. Every term in the figure I quoted is a property of one way of running
`Exp_Long_Term_Forecast.test()`. Not one of them is a property of the model. The
model's own weights are **0.027 GiB**. The figure I called impossible — I quoted
37 GiB; re-derived here it is **33.102 GiB** — is **1 216 times** the model it
was supposed to be about. Measured, the same evaluation on the same complete
population through the bounded path this repository already contains peaks at
**3.600 GiB**, inside a **4 GiB** cap, on the worker I said could not hold it.

**What this delivery is.** The term-by-term re-derivation, the demonstration
that the bounded evaluator this repository already contains applies to Traffic,
its parity against the author's own evaluator proved bitwise on Traffic
populations that fit, and the measured whole-cgroup peak of the bounded path at
full size. **What it is not.** It is not a Traffic campaign, not an increased
machine cap, not new hardware and not an altered recipe. No Traffic cell was
trained. No Traffic number is a scientific result: every metric value in this
document is one side of a numerical parity comparison between two reductions of
identical arrays produced by an **untrained** model, and an untrained model's
error is not a result.

---

## 1. The memory derivation, term by term, and what each term is a property of

`tools/df_tsl_execute.py::eval_path_memory_derivation` at `f40fae93` prices the
author's own unchunked evaluation: three float32 Python lists, three
`np.concatenate` copies, then `utils.metrics.metric` on the whole arrays. Its
terms are re-derived here from **Traffic's own characterized populations**, not
from the numbers in the status. The characterization was produced by the sealed
`df_tsl_repro.characterize` on the registered Traffic bytes
(`cb06463d…0dab0b16`, 17 544 rows, 862 channels); its test-window counts —
**3 413 / 3 317 / 3 173 / 2 789** — reconcile exactly with the sealed plan's
arithmetic, so the element counts below are the author's loader's, not mine.

**h720, the binding horizon** (2 789 windows × 720 steps × 862 channels =
1 730 964 960 float32 elements = 6.448 GiB per array):

| term | GiB | what it is a property of |
|---|---:|---|
| `accumulated_lists` — the `preds`, `trues` and `inputs` per-batch blocks | 13.756 | **EVALUATION IMPLEMENTATION** — Python lists that are never freed until the loop ends |
| `concatenate_stage_peak` — inputs and preds already copied while the `trues` list is still alive and its copy is being written | 20.205 | **EVALUATION IMPLEMENTATION** — the cost of making a second copy of everything |
| `retained_after_concatenate` — the three concatenated arrays | 13.756 | **EVALUATION IMPLEMENTATION** — including `inputs`, which the scorer never reads |
| `metric_temporaries` — MAPE/MSPE build `(true-pred)`, divide by `true`, then abs/square | 19.345 | **EVALUATION IMPLEMENTATION** — three full-size float32 temporaries for two metrics the producer contract does not store |
| **scoring stage = retained + temporaries** | **33.102** | **EVALUATION IMPLEMENTATION** (the binding maximum) |
| model weights, the author's own model counted from the sealed argv: 7 306 748 parameters | **0.027** | **MODEL** |
| activations, optimizer state, training host and device memory | **UNMEASURED** | **MODEL** — named unmeasured, never derived. Null is not small |

All four horizons, arrays only (`baseline_bytes = 0`), against the **14 GiB**
`crispdm-batch.slice` ceiling on the admitted worker:

| horizon | elapsed | windows | author unchunked | fits 14 GiB | **bounded path** | fits | disk |
|---|---|---:|---:|:--:|---:|:--:|---:|
| h96 | **96 h = 4 d** | 3 413 | 6.313 GiB | yes | **0.178 GiB** | yes | 2.104 GiB |
| h192 | **192 h = 8 d** | 3 317 | 11.248 GiB | yes | **0.355 GiB** | yes | 4.090 GiB |
| h336 | **336 h = 14 d** | 3 173 | 18.096 GiB | **no** | **0.621 GiB** | yes | 6.847 GiB |
| h720 | **720 h = 30 d** | 2 789 | 33.102 GiB | **no** | **1.332 GiB** | yes | 12.897 GiB |

Traffic's step is 3 600 s. Its 96 steps are **96 hours**, where Weather's 96
steps are 16. The same step count is not the same elapsed horizon, and a test
asserts the two clocks differ so no later document can report one as the other.

**The 22 and 37 GiB in the status are not reproduced.** The term-by-term
re-derivation gives **18.096** and **33.102 GiB** of arrays; adding the only
measured baseline on record — the 1.364 GiB whole-cgroup peak of the Weather
TRAIN-only pilot, which is Weather's and not Traffic's — gives **19.46** and
**34.47 GiB**. I cannot reconstruct a baseline that produces 22 and 37 from
these terms, and I am not going to invent one. The two figures should be read as
**superseded by the derivation in this section**, which is the same function
applied to Traffic's own characterized populations. The *direction* of the
earlier claim survives — the author's unchunked path at h336 and h720 does not
fit any cap on this fleet — but the magnitudes do not, and neither did the
conclusion drawn from them.

**Where the derivation is known to be wrong, and by how much.** Gate one at
`f40fae93` measured Weather h720 at **4.547 GiB** against a **4.204 GiB**
derivation: 8 % optimistic overall, and wrong in its term structure — the
accumulate-and-concatenate stage cost *more* than derived and the metric's
temporaries *less* (two full-size operands live at once, not three). That
correction is carried here rather than quietly dropped: a derivation is a
derivation, which is why §4 measures instead.

## 2. The bounded path already exists, and it is not a chunked average

The order asked whether the existing bounded ECL scoring and reduction
implementation applies to Traffic. **It does, unmodified.** Two objects,
already in this repository, already exercised by the Electricity work:

- **`df_sota_repro.bounded_test`** (`df_sota_bounded_eval.v1`). The author's own
  test loader, model, batches, order, slicing and dtype, with the three
  accumulating Python lists and the three `np.concatenate` copies replaced by
  two ordered float32 `.npy` files written block by block with the written pages
  released behind (`posix_fadvise(DONTNEED)` every 16 batches). `inputs` — which
  the author's scorer never reads — is not retained. It reloads the checkpoint
  from disk exactly as the author's `test(test=1)` does.
- **`df_sota_repro.author_metric_exact`** (`df_sota_author_metric_exact.v2`).
  The author's float32 MAE/MSE reproduced **bit for bit** through a bounded
  route: element-wise differences formed in float32 chunk by chunk (element-wise
  operations are exact per element), **numpy's own pairwise summation tree
  replayed over the same flattened C-order element index space** with leaves
  reduced by numpy itself, and the mean taken as numpy's own
  `float32(float64(sum)/int64(N))`.

`tools/df_tsl_bounded_eval.py` adds **no** reduction, **no** chunked mean, **no**
downcast and **no** sub-sampling. It is a TSL-shaped driver that imports and
calls the two objects above; a test asserts against this module's own source
that the only `np.mean` it contains is the counter-example described below.

**The trap, met head on.** An arbitrary chunked average is *not* float32-identical
to the unchunked reduction, and I did not assume otherwise. The reducer-parity
battery runs the author's own `utils.metrics.metric` beside the bounded route on
populations that fit, and computes **an average of per-chunk means** beside both,
so the difference is demonstrated rather than asserted:

| population (W, T, C) | elements | denominator exact in float32 | author fn vs bounded route | mean-of-chunk-means, MAE difference |
|---|---:|:--:|:--:|---:|
| 16 × 96 × 862 | 1 324 032 | yes | **bitwise equal** | 1.30e-08 |
| 101 × 336 × 862 | 29 252 832 | yes | **bitwise equal** | **1.69e-05** |
| 35 × 721 × 863 | 21 777 805 | **no** | **bitwise equal** | 0.0 |
| 61 × 720 × 862 | 37 859 040 | yes | **bitwise equal** | 5.20e-07 |

**4 of 4 bitwise equal; maximum absolute difference exactly 0.0**, numpy 2.5.1.
The third row is the case that catches a naive reducer: 21 777 805 is odd and
above 2²⁴, so it is *not* exactly representable in float32 and the divisor must
be promoted through int64 — the v1 route got this wrong and was superseded.
The fourth row uses Traffic's own h720 window geometry, so pairwise leaf
boundaries fall **inside** windows rather than on them.

The last column is the point. Chunked averaging differs by up to **1.69e-05** on
MAE — and on one population it happened to agree exactly. **That coincidence is
reported rather than suppressed**, because it is exactly why a coincidental
agreement is not a proof, and why the bounded route is built to be the same
arithmetic in the same order rather than a mean that happens to be close.

## 3. Parity against the native evaluator, on Traffic populations that fit

Parity has three separable parts, and they are reported separately because they
are established by different evidence and at different sizes.

**(a) The reduction.** `author_metric_exact` against `utils.metrics.metric`
itself: **bitwise equal on 4 of 4 populations, maximum absolute difference
exactly 0.0** (§2). This is the part the trap is about, and it is the part that
does not depend on the population: the route replays numpy's own summation tree
over the same flattened element index space, so its equality is arithmetic, not
statistical. `bounded_test` also runs the author's own function beside the
bounded route whenever its temporaries fit the declared budget, and **refuses
the cell** if they disagree — so a silent drift is not reachable.

**That gate was exercised on Traffic itself, at full population.** The h96 probe
ran with a 5 GiB scorer budget, so `utils.metrics.metric` executed on the
memmapped arrays beside the bounded route over the complete
**3 413 × 96 × 862 = 282 432 576**-element test population:

| | MAE | MSE |
|---|---|---|
| the author's own `utils.metrics.metric`, unchunked | `0.9714772701263428` | `1.9507478475570679` |
| `author_metric_exact`, the bounded route | `0.9714772701263428` | `1.9507478475570679` |
| **bit-equal** | **yes** | **yes** |
| the independent float64 reduction, reported beside and never instead | `0.9714772367866366` | `1.950747929954006` |

The verdict of that probe is **`ADMISSIBLE`** — the only one of the two that is
not `_PARITY_INHERITED`. The numbers are an **untrained** model's error and are
not results; they appear here only as the two sides of the comparison.

**(b) The writer's row placement.** The bounded writer places each batch's block at a running row
offset inside one loop body, so its output is the author's `np.concatenate` of
the same blocks unless the placement is wrong. `target-pairing` establishes that
independently and cheaply: a **separate pass of the author's own test loader**
hashes the target windows as they stream, and that digest is compared to the one
the bounded writer produced. It is exactly the identity the sealed scored-cell
path already refuses a cell over (`REFUSED: the naive was computed on a different
target population than the model was scored on`). Predictions and targets are
written in the same loop body at the same offset, so a placement that is right
for one is right for the other.

**Status: NOT MEASURED.** The 4 GiB child was submitted and **queued** on the
admitted worker for the whole window and never admitted: three lanes were active
on it and the `crispdm-batch.slice` aggregate budget (14 GiB ceiling, 6.98 GiB
in use plus 4.07 GiB of unrealised reservations) left under 3 GiB free. The cap
was **not** lowered to fit — the launcher treats a lowered cap after a refusal as
terminal, and asking again under a different name would be the same evasion by
another route. The job is a one-command rerun when the worker frees.

**(c) The model's outputs under the author's own `test()`.** The strongest form of the proof runs the author's own unchunked
`test()` and the bounded path **in one child, from one on-disk checkpoint**, and
compares the sha256 of the arrays and the metric values bitwise. `--native-witness`
implements exactly that, and `bounded_probe` reports `parity.holds` from it.

**Status: NOT MEASURED.** The child's honest declared cap is 12 GiB at h96
(6.313 GiB of arrays derived, times the 1.082 correction gate one measured on
Weather, plus a baseline), and the worker's slice ceiling is 14 GiB whose
aggregate budget counts current slice usage including file cache. It was
submitted, **queued and never admitted** while three other lanes held the slice,
and the cap was not lowered to fit — the launcher treats a cap lowered after a
refusal as terminal (`CAP_LOWERED_AFTER_REFUSAL`), and asking again under another
name would be the same evasion by another route.

What *did* run instead is the 8 GiB probe above, which establishes (a) on the
real Traffic h96 population. What remains unestablished is narrower than it was:
that the author's `np.concatenate` of the same per-batch blocks is byte-identical
to the file the bounded writer produced, and that the author's `test()` drives
the same forward pass. Both are arrangements of the same loop body rather than
different arithmetic — but an argument is not a measurement, and it is labelled
one.

**What this means for the verdict.** It means h720's verdict is
`ADMISSIBLE_PARITY_INHERITED` and not `ADMISSIBLE`, and the record says so in
that word. Parity of the **reduction** is demonstrated bitwise (a); parity of the
**writer** and of the **model's outputs through the author's own `test()`** on
Traffic data is **not yet demonstrated**, and no sentence in this document should
be read as claiming it.

## 4. The full-size bounded path, measured

**h720 — 2 789 windows × 720 steps × 862 channels, the complete sealed test
population, the horizon I called impossible.** An untrained model, the whole-cgroup
high-water mark read from **inside** the child, on the admitted worker's RTX 4090
`GPU-a8bd1b2c-…6780f9`, asserted inside the child by physical UUID.

| | value |
|---|---|
| **measured whole-cgroup peak, in-child `memory.peak`** | **3 865 726 976 B = 3.600 GiB** |
| declared cap (passed once, never shrunk) | 4 294 967 296 B = 4 GiB |
| headroom | 429 240 320 B = 0.400 GiB |
| **author's unchunked path at the same horizon, derived** | **35 542 480 512 B = 33.102 GiB** |
| **verdict** | **`ADMISSIBLE_PARITY_INHERITED`** |
| scored population | 2 789 × 720 × 862 = **1 730 964 960** elements, float32, **complete**, equal to the sealed count |
| batches | 175, sizes 16 … 16, last 5 — the author's own test loader, `drop_last` not in effect |
| bounded evaluation wall / CPU | 101.8 s / 103.3 s |
| peak GPU allocated / reserved | 718 MiB / 802 MiB |
| disk high-water under the work directory | 13 878 248 501 B = **12.925 GiB** |
| record digest | `7abe795a…0500eb` |

The cgroup stage peaks say where the memory went, which one number cannot:

| stage | cgroup peak | what had happened |
|---|---:|---|
| entry | 0.479 GiB | interpreter, torch, CUDA context |
| after model build | 0.541 GiB | the model on the device |
| **after the bounded evaluation** | **2.829 GiB** | all 175 batches written, the whole population reduced, the float64 diagnostic taken |
| after digests | 3.600 GiB | **my own** sha256 pass over the 6.4 GiB prediction file, for the parity record |

**The last row is mine, not the bounded evaluator's.** The evaluation itself
peaked at **2.829 GiB**; the extra 0.771 GiB is the digest pass this probe adds
so that the writer's output can be identified. It is reported separately rather
than folded in, because a future scored cell that does not need the digest would
not pay it.

The derivation was **optimistic by 2.70×** here (1.332 GiB of derived resident
terms against 3.600 GiB measured). The difference is the baseline the derivation
deliberately carries as zero — interpreter, torch, CUDA context and the author
loader's own arrays — which is exactly why the measurement exists. The *shape*
of the derivation held: the peak did not grow with the **number of windows**.
Watched live while the arrays grew from 2.2 to 9.7 GiB on disk, the scope's
`MemoryPeak` stayed at **2.776–2.777 GiB**. (It does scale with one window's
size times the flush window — a constant of the horizon, not of the population —
which is why h720 costs more resident memory than h96 and why neither costs more
as the test set lengthens.)

**h96, measured beside it, with the author's own scorer running as a witness.**

| | value |
|---|---|
| **measured whole-cgroup peak** | **5 866 139 648 B = 5.463 GiB** |
| declared cap | 8 589 934 592 B = 8 GiB · headroom 2.537 GiB |
| **verdict** | **`ADMISSIBLE`** |
| scored population | 3 413 × 96 × 862 = **282 432 576** elements, complete; 214 batches, 16 … 16, last 5 |
| bounded evaluation wall | 26.8 s · peak GPU allocated 684 MiB |
| disk high-water | 2.1315 GiB (derived 2.104; the difference is the `.npy` headers, the checkpoint and the log) |
| record digest | `b6920e0c…306ce2` |

**That 5.463 GiB is not the bounded path's cost.** This probe deliberately ran
the author's own unchunked `utils.metrics.metric` on the memmapped arrays beside
the bounded route, as the parity witness of §3 — three full-size float32
temporaries over a 1.052 GiB array, on top of both mapped files. The bounded
route's own derived resident terms at h96 are **0.178 GiB**. h720, which could
not afford the witness, is the honest reading of the bounded path's own cost.

**h336, the other horizon I called impossible: NOT MEASURED.** Only h720, the
binding one, was probed — the same choice gate one made for Weather. h336 is
strictly smaller in every term (18.096 GiB against 33.102 GiB for the author's
path, 0.621 GiB against 1.332 GiB derived for the bounded one, 6.847 GiB against
12.897 GiB on disk), so the h720 measurement bounds it — but that is an
inference from a monotone derivation, not a measurement, and it is labelled one.

**A defect in my own record, named.** `filesystem_after_retention` in the h720
record reads 693 426 769 920 B free — essentially unchanged from the high-water
reading — although the probe had already unlinked both arrays. The space is not
returned while the probe still holds the `np.load(mmap_mode="r")` handles that
`bounded_test` returns; the inodes survive the unlink until the process exits.
Verified independently after the child exited: **707 274 272 768 B free**, i.e.
13.85 GB returned. The retention rule works; the *reading of it inside the same
process* does not, and should be taken after the handles are dropped.

## 5. The two prerequisites that are easy to forget

**Disk budget.** A memmap is not a proof of bounded resident memory and a
bounded evaluator that fills the disk has moved the failure rather than removed
it. The two arrays the bounded path writes are the *only* thing whose size
grows with the population:

| horizon | preds + trues on disk |
|---|---:|
| h96 | 2.104 GiB |
| h192 | 4.090 GiB |
| h336 | 6.847 GiB |
| h720 | **12.897 GiB** |

Run **sequentially with the retention rule below, the transient high-water is
12.897 GiB**, once, at h720 — not the 26 GiB sum, because no two cells' arrays
are alive together. Measured at h720: the filesystem held **660.1 GiB** free
before and **645.8 GiB** at the high-water, i.e. the evaluator consumed **2.1 %**
of the free space and returned it. The probe records the filesystem's own free
space before, at the high-water and after retention, so the budget is measured
rather than assumed — and §4 names the one place that reading is defective.

**Artifact-retention policy.** The arrays are deleted by the probe as soon as the
metrics, the complete element count and the sha256 digests that identify the
population are recorded. **What is retained is the record, not the arrays** —
consistent with the standing metrics-vault rule, which is what makes deleting
raw predictions legitimate: the digest still identifies the exact population any
later analysis would have to reproduce. `--retain-arrays` exists and is off; a
retained array is listed by path and byte count in the record when it is used.

## 6. What this authorizes, and the costed proposal

This is **not** a request to run a Traffic campaign, and nothing here authorizes
one. The deliverable the order asked for is the proven evaluator plus a costed
proposal, and the honest costing has a hole in it that I am not going to fill by
analogy:

- **The evaluation side is now priced at its binding horizon**: one bounded pass
  over Traffic's complete h720 test population costs **101.8 s** of wall on the
  admitted 4090, at a **3.600 GiB** whole-cgroup peak and a **12.925 GiB**
  transient disk high-water. h96 is measured too: **26.8 s**, 5.463 GiB with the
  author's own scorer running beside it as a parity witness, 2.13 GiB of disk.
  The two measured points are **sub-linear** in element count — 101.8 s / 26.8 s
  = 3.80 against an element ratio of 6.13, and h96's number is inflated because it
  also carries the author's witness scorer. h192 and h336 lie between them and are
  **NOT MEASURED**; interpolating linearly in elements puts the twelve cells'
  evaluation at roughly **0.18 GPU-hours**, which is a DERIVED number offered as a
  bound, not as a price.
- **The training side is not.** Traffic's per-step cost is **UNMEASURED**.
  Traffic's patch geometry gives 862 graph tokens against Weather's 42, at
  `d_model` 512 against 128, for 30 epochs against 10. The Weather pilot's
  median of 10 ms says nothing about it, and I did not price it by analogy.

The single request, therefore, is for the **TRAIN-only cost pilot** that would
make a campaign request possible — not for the campaign:

> **Traffic L = 96, TimeFilter, a TRAIN-only cost pilot of two 30-step children
> (h96 and h720) on the admitted worker's RTX 4090, under design
> `6cba7e20…85575` / protocol `322262e8…ba520`, declared cap 8 GiB each,
> sequential, no duplicate GPU executor.** It produces a measured per-step cost
> and no score. Until it has run, no Traffic campaign number exists to be
> costed, and I am not proposing one.

## 7. What is VERIFIED, what is measured, and what is not measured

| claim | class |
|---|---|
| Traffic's characterized test populations — 3 413 / 3 317 / 3 173 / 2 789 windows, 862 channels, 17 544 rows, 0 missing values — are the author's own loader's and reconcile with the sealed arithmetic | **VERIFIED** (the sealed `df_tsl_repro.characterize` refuses a disagreement; the registered sha256 was re-verified inside the child) |
| Every term in the author-path figure is a property of the evaluation implementation, not of the model | **VERIFIED** as a reading of `eval_path_memory_derivation`'s own source, term by term, each naming the line of `test()` it comes from |
| The author's unchunked path needs 18.096 GiB (h336) and 33.102 GiB (h720) of arrays | **DERIVED, not measured** — by the same sealed function, on Traffic's own characterized populations |
| The 22 and 37 GiB figures in the campaign status | **NOT REPRODUCED.** Superseded by the derivation above; no baseline I can evidence produces them |
| The author's model has 7 306 748 parameters = 0.027 GiB of float32 weights | **VERIFIED** (the author's own model instantiated from the sealed argv and counted) |
| Traffic activations, optimizer state and training memory | **UNMEASURED.** Named, never derived. Null is not small |
| `author_metric_exact` is bit-equal to `utils.metrics.metric` | **VERIFIED, bitwise, 4 of 4**, max absolute difference exactly 0.0, on numpy 2.5.1, including a population whose element count is not exactly representable in float32 and one with Traffic's h720 window geometry |
| An arbitrary chunked average is NOT bit-equal to that reduction | **MEASURED and reported**: it differed by up to 1.69e-05 on MAE, and on one of four populations it coincidentally agreed — which is why the coincidence is reported, not relied on |
| `author_metric_exact` is bit-equal to `utils.metrics.metric` on the **complete real Traffic h96 population** (282 432 576 elements, the author's own model's outputs) | **VERIFIED, bitwise**, both MAE and MSE; the probe's verdict is `ADMISSIBLE` and `bounded_test` would have refused the cell otherwise |
| The bounded writer's file equals the author's `np.concatenate` of the same per-batch blocks, and the author's own `test()` drives the same forward pass | **NOT MEASURED.** The 12 GiB native-witness child was submitted, queued and never admitted while other lanes held the worker's slice; no cap was lowered to fit. h720's verdict is `ADMISSIBLE_PARITY_INHERITED` for exactly this reason |
| The bounded path at Traffic h720 peaks at 3.600 GiB whole-cgroup over the complete 1 730 964 960-element population | **measured**, one execution, one host, in-child `memory.peak`; `NOT_INDEPENDENTLY_VERIFIED` and **not replayed in a fresh process** |
| The bounded path's resident peak does not grow with the **number of test windows** (it does scale with one window's size × the flush window, which is a constant of the horizon) | **measured** as a live observation on one horizon (the scope's `MemoryPeak` held at 2.776–2.777 GiB while the arrays grew from 2.2 to 9.7 GiB on disk); **derived** in general, and asserted by a test that doubles the window count and requires the resident terms to be unchanged while the disk term doubles |
| The disk high-water is 12.925 GiB at h720 and the arrays are deleted after their digests are recorded | **measured**; the in-process reading of the reclaim is defective and was corrected by an external reading (§4) |
| Any Traffic model quality number | **`NO_NEW_MEASUREMENT`** — no cell was trained. Every metric value in this document is one side of a parity comparison over an untrained model's output |
| Traffic per-step training cost | **UNMEASURED**; the Weather pilot does not bound it and nothing here prices it by analogy |
| The bounded evaluator would give the same answer as the author's path on a TRAINED Traffic cell | **UNMEASURED** — parity was established on the arithmetic and on untrained outputs; no trained cell exists to check |

## 8. Not disturbed, for the record

Another lane's job (`cb03pub400`, 6 GiB, the classification reference lane) was
live on the admitted worker throughout. **Gate admission queued behind it rather
than displacing it**, and the declared cap was passed once and never shrunk to
squeeze into the remaining slice room. Fresh aggregate admission was taken per
child. No service was started, stopped or restarted; the household lake
restoration under way elsewhere was not touched. No cap, cache, swap, oomd
setting or kernel parameter was changed. No driver was reloaded and no host was
rebooted. The preferred RTX 5090 host was **not used** — it holds about 2.8 GiB
available against its unreclaimable kernel slab, and is not usable. No
committed sample output and no live checkout was altered. The sealed modules
were **called, never changed**: `git diff f40fae93 -- tools/df_sota_repro.py
tools/df_tsl_repro.py tools/df_tsl_execute.py docs/contracts/` is empty.

## 9. Artifacts

Committed on `satoshi/traffic-bounded-evaluator-20260929`:

- `tools/df_tsl_bounded_eval.py` — the TSL driver: the term-by-term split, the
  characterization, the reducer-parity battery and the measured bounded probe.
  It introduces no model, loader, loss, scorer, reduction, recipe or margin.
- `tools/test_tsl_bounded_eval.py` — 18 tests of the generators.
- `docs/audits/work_plan/SATOSHI_TRAFFIC_BOUNDED_EVALUATOR_2026_09_29.md` — this
  document.

Operator-retained, not committed (they carry host detail):
`~/.local/state/crispdm-data-foundation/tsl_traffic_bounded_20260929/` —
`CHARACTERIZATION.traffic.json`, `MEMORY_SPLIT.traffic.L96.json`,
`REDUCER_PARITY.traffic.json`, `BOUNDED_PROBE.traffic.h*.json`, and the design
copied from the RB02 lane.

## 10. What the auditor should attack first

1. **The bounded path's parity is proved where the author's own function can
   run, and inherited where it cannot.** At h336 and h720 the author's unchunked
   function needs 18 and 33 GiB and was not executed, so parity at those
   horizons is labelled `ADMISSIBLE_PARITY_INHERITED`, never "in parity". The
   inheritance rests on the reduction being population-independent arithmetic,
   which is an argument, not a measurement at that size.
2. **A single measurement on a single host, one execution, no replay.** The
   whole-cgroup peaks below were not reproduced in a fresh process, and nothing
   here establishes that they would be identical on another host.
3. **The retired 22/37 GiB figures.** I could not reconstruct them and I say so.
   If a baseline exists that produces them, it should be produced, because two
   documents carrying two figures for one quantity is the defect, not the gap.
4. **`memory.peak` counts reclaimable file-backed pages.** That makes the
   reported peak conservative, but it also means a smaller number could be
   achieved under pressure and should not be quoted as the floor.

— Satoshi III (Mujuro Utsutsu), successor technical lead, 2026-09-29.
