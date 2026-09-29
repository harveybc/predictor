# QRM02 — TRAIN-only cost pilot for the six missing W1440 cells

**Status: DESIGN, revision 2. Not sealed, not dispatched.**
**Author:** Satoshi III (Mujuro Utsutsu), successor technical lead
**Date:** 2026-09-29
**Supersedes:** revision 1 at `8a31ba1f`, which was reviewed at `8fc61cf0` and found **not ready for
dispatch**. Three of its four findings were design errors of mine and are repaired here by name.
**Sealing condition:** sealed only against lane A's published runner commit, and only once §3's authority
exists. A cost measured through a driver that cannot isolate a cell is not a per-cell cost.

---

## 0. The four repairs, stated before anything else

| # | what revision 1 did wrong | repair |
|---|---|---|
| 1 | **Circular limits.** It set the device cap from "this pilot's own stage 1" and the stage budgets from "this pilot's own warmup". Those are not limits declared before the pilot runs. | §3 declares a **finite envelope from evidence outside this pilot**, and §4 puts a **predeclared bounded calibration phase inside it**. No implicit expansion mid-run. |
| 2 | **An assumption dressed as a stage.** It called the ragged final batch the place "where a tight cap fails". A ragged final batch is **smaller**, and nothing established it is the worst. | §5 **tests the actual full and final shapes** and reports the observed order. No shape is assumed worst. |
| 3 | **Peak attribution overstated.** It recorded a "host RAM cgroup peak per stage". `memory.peak` is a **cumulative high-watermark over the scope lifetime**; readings at stage boundaries are cumulative, not independent stage peaks. | §6 records both, under names that cannot be confused. |
| 4 | **The historical null used as more than a diagnostic.** | §1 keeps it as a diagnostic only and enumerates the successor's controls separately. |

---

## 1. The population, and what the historical null does and does not license

The six missing cells are three seeds each of **two** arms, both at window 1440:

| arm | what it is |
|---|---|
| `long_window_own_depth` | an arm of the primary factorial, at its own depth |
| `long_window_local_support_67` | **measured extra context of 67 raw samples** — the clamped core still reaches branch 5 + core 63 − 1. Explicitly **not** a null. |

**Both are costed separately.** A bound over one may cover the other only if it is **demonstrated on the
built models**; they differ in depth and reach, which are the terms that drive activation memory, so the
shared window length proves nothing.

The third 1440 arm, `long_window_crop60`, is the **exact-information null**. It already ran and reproduced
the baseline bitwise in 3 of 3 seeds. It is **retained as a diagnostic** and is **not refitted** for this
resource pilot. **It does not by itself authorize those cells as governed comparators** for the future
scientific successor. The successor's required controls and each one's evidence status are enumerated
**before any full fit**, not inferred from this equivalence.

---

## 2. What this pilot is

A **cost measurement of the training path with the model present**, per cell, in that cell's own scope. It
produces **no accuracy**, touches **no test split**, and selects nothing.

It exists because every figure offered so far as the cost of these cells measures something else:

| circulating figure | what it actually is |
|---|---|
| 7.4 G | the cgroup peak of a **killed multi-child wrapper scope**, at the kill |
| 8 458 399 744 B | `getrusage(RUSAGE_SELF)` **child RSS** |
| 1 463 877 632 B | the cgroup peak of one **20 000-window data materialization**, **with the model absent** |

**RSS and cgroup accounting measure different quantities**, and the one observed ordering between them
establishes no invariant. Each is recorded under its own name with the cgroup's identity, ancestry and
lifetime.

---

## 3. The envelope, declared before dispatch, from evidence outside this pilot

| item | value | basis, and why it is not circular |
|---|---|---|
| placement | secondary worker, one cell at a time | the preferred accelerator host is ineligible: ~3 GiB free of 14 with 4.93 GiB unreclaimable kernel slab |
| **host envelope** | **12 GiB per cell, ENFORCED** | the launcher sets cgroup `MemoryMax`; 12 GiB sits under the 14 GiB slice ceiling and inside the worker's ~21 GiB free. Chosen as a bound, not derived from the thing being measured |
| **device envelope** | **12 GiB, MONITORED ONLY, NOT ENFORCED** | the cgroup bounds host RAM and **not** VRAM. The 4090 carries 16 376 MiB. The child asserts `torch.cuda.max_memory_allocated` and `max_memory_reserved` against the envelope and **raises**; **CUDA allocation statistics are not an enforced GPU limit** and are never described as one |
| wall deadline | 1 800 s per cell, 4 800 s total worst case | a declared stopping rule, not a prediction |
| CPU deadline | 2 400 CPU s per cell | as above |
| **abort path** | host: the cgroup kills the child and the attempt is recorded **terminated**, never retried at a larger cap. device: the child raises and exits non-zero before allocating past the envelope. Either way the peak is read **before the scope is removed** and a missing peak is `UNKNOWN` | |

**If a calibration exceeds the envelope, the pilot stops and reports.** It never expands mid-run.

---

## 4. Predeclared bounded calibration, then a derived proposal

Inside the envelope, per architecture: **30 optimizer updates**, at the production batch policy, unchanged.
That count is fixed here, before dispatch.

From the calibration, and only from it, the **proposed successor's** costs are derived and published as a
proposal for review. The calibration's own numbers are measurements; the successor's are **projections with
explicit headroom**, never a claim that maximum memory has been proven.

An isolation fixture from lane A proves that **scope accounting works**. Its peak **does not size this
model** and is not used to.

---

## 5. Stages, and the shapes actually tested

Measured in the **same cell scope**, in this order:

1. **model build** — parameters instantiated, before any step;
2. **optimizer-slot initialization** — completed **before** any steady-state cost is attributed, so the
   moment buffers are resident when the steady regime is measured rather than appearing inside it;
3. **warmup** — where allocator behaviour differs from steady state;
4. **steady batches**;
5. **both the full batch shape and the final batch shape**, each measured, with **graph retracing and
   workspace allocation** observed rather than assumed absent. **The order between them is reported, not
   assumed** — revision 1 asserted the final batch was worst and had no evidence for it;
6. **checkpoint write and reload**;
7. **validation mechanics and their overlap with checkpointing, as production executes it** — on a
   **TRAIN-only fixture of the planned shape**. Mechanics, never a score.

The production batch policy is **unchanged**. This pilot measures what production would do; it does not
propose a cheaper way to do it.

---

## 6. How each number is recorded, so none can be misread

- **`host_cumulative_high_watermark_at_stage_boundary`** — `memory.peak` read at each boundary. **Cumulative
  over the scope lifetime**, monotonic, and labelled so.
- **`host_stage_peak_after_reset`** — `memory.peak` is writable on this kernel (7.0), so it is reset at each
  stage boundary and the following reading is the peak **since that reset**. Both series are kept: resetting
  changes what the number means, and losing the lifetime watermark would be a worse trade.
- **`device_allocated_peak` / `device_reserved_peak`** — each names the **framework and API**
  (`torch.cuda.max_memory_allocated`, `max_memory_reserved`) and its **reset basis**
  (`reset_peak_memory_stats` at the same boundaries). **Unavailable statistics are `UNKNOWN`** and are
  **never synthesized** from host memory or from a parameter count.
- **`rss_self_peak`** — recorded separately, never as a bound on the cgroup figure or vice versa.
- Plus, per cell: scope identity, the kernel limit in force, optimizer updates, CPU time, wall time, stage.

The peak is read **before the scope is removed**. An external supervisor retains the **termination status**
when a child fails. **A missing peak is `UNKNOWN` — never zero, and never success.**

---

## 7. Authority: there is none, and here is the request

**Reconciled, as the order requires me to do rather than delegate.** The dispatch index records, for this
lane: *"The six W1440 units have no approved allocation"*, and **`lease: NONE HELD`**. The only remainder
recorded anywhere belongs to another lane and is declared not a licence to spend. **I am not borrowing it.**

One further caution on the index's own `resource` row: it prices the six cells at *"2 000–4 000 updates per
cell at 4.165 CPU s per update, 15–30 h wall … at a 7.4 GiB resident set"*. That memory term is the
disqualified figure of §2 **and** it is called a resident set when the journal figure was a cgroup peak of a
killed wrapper. The **time** terms are not invalidated by that mislabel, but they inherit its provenance and
are treated as inputs to be re-measured, not as authority.

> **The request, concrete and bounded.** Two architectures × one calibration cell each, sequential, on the
> secondary worker's RTX 4090 by UUID. **Stages:** build, optimizer-slot initialization, warmup, steady,
> both batch shapes, checkpoint write and reload, validation mechanics. **Worst-case spend: 4 800 s wall and
> 4 800 CPU s total**, host envelope 12 GiB enforced, device envelope 12 GiB monitored.
> **Stopping rules:** any stage exceeding either envelope stops the pilot; the wall or CPU deadline stops it;
> a terminated child is recorded and **never retried at a larger cap**; and a missing peak stops the pilot
> rather than being written as a number.
> **This buys a costed successor proposal. It buys no fit, no accuracy and no training authorization.**

---

## 8. What this pilot may never do

- No test access, no accuracy, no selection.
- **No promotion of the twelve historical cells** through this or any new seal; they stay
  `NOT_BOUND_TO_A_SEAL`, custody unchecked.
- The estimand stays **`MATCHED_BUDGET_DIFFERENCE`**, never `CONVERGED_ACCURACY`.
- No scientific dimension changes — not a window, depth, cell count or the optimization — except in an
  explicitly justified successor declared **before any score**.

## 9. Blockers that remain

1. **Lane A's runner does not exist yet.** A per-cell number is impossible until a cell gets a fresh
   exclusive scope enclosing its complete process tree with its own reservation.
2. **Reachability, not master membership:** the instrument and the v2 seal must be **reproducibly reachable
   from the selected runner commit**.
3. **The allocation in §7 is requested, not held.**

**Until 1, 2 and 3 are discharged, `NO_PROGRAMME_TASK_IS_READY` for a long-window fit.** That is the status,
not a gap to fill with a number.

— Satoshi
