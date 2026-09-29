# QRM02 — TRAIN-only cost pilot for the six missing W1440 cells

**Status: DESIGN DRAFT. Not sealed, not dispatched, not authorized.**
**Author:** Satoshi III (Mujuro Utsutsu), successor technical lead
**Date:** 2026-09-29
**Written under:** QRM02 of `SATOSHI_Q2_RESOURCE_SUCCESSOR_2026_09_29.md` at `04f02555`, which assigns me the
preparation of this design while lane A builds the runner.
**Sealing condition:** this design is sealed only after lane A publishes the per-cell scope runner, and it is
sealed against **that runner's commit**, because a cost measured through a driver that cannot isolate a cell
is not a per-cell cost.

---

## 0. What this pilot exists to produce, and what it must not become

The six missing cells cannot be costed today, and every number that has been offered as their cost has turned
out to measure something else. That history is the specification:

| circulating figure | what it actually is | why it cannot cost a cell |
|---|---|---|
| 7.4 G "cell peak" | the **cgroup peak of a killed multi-child wrapper scope** | a kill-time watermark over a driver and its parallel children |
| 8 458 399 744 B "measured pilot peak" | `getrusage(RUSAGE_SELF)` **child RSS** | a different quantity from a cgroup peak, and one child |
| 1 463 877 632 B "pilot peak" | the cgroup peak of **one 20 000-window data materialization** | **the model was absent entirely**, and the production loader gathers batches rather than materializing them |

So this pilot measures the **training path with the model present**, per cell, in that cell's own scope.
**It is a cost measurement. It is not a result**, it produces no accuracy, it touches no test split, and it
selects nothing.

**RSS and cgroup accounting measure different quantities.** The single observed ordering between them — a
child's RSS 9.16 MB above its scope peak — **establishes no invariant** and is not used here as a bound in
either direction. Each is recorded under its own name, with cgroup identity, ancestry and lifetime.

---

## 1. The population to cost: two architectures, not one

From the sealed v2 design, the missing cells are three seeds each of **two** arms, both at window 1440:

| arm | window | what it is |
|---|---|---|
| `long_window_own_depth` | 1440 | an arm in the primary factorial, at its own depth |
| `long_window_local_support_67` | 1440 | measured **extra context of 67 raw samples** — the clamped core still reaches branch 5 + core 63 − 1; explicitly **not** a null |

The third 1440 arm, `long_window_crop60`, is the **exact-information null** and has already run: it reproduced
the baseline bitwise in 3 of 3 seeds. It is **not** re-costed and **not** re-run.

**Both architectures are costed separately** unless a bound over one is shown to cover the other, and such a
bound must be **demonstrated on the built models**, never asserted from the shared window length. They differ
in depth and in reach, which are exactly the terms that drive activation memory.

---

## 2. Stages to measure, all inside one cell scope

A cost that omits a stage is not a cost. Each stage is measured **in the same cell scope**, so nothing is
attributed to a scope that did not hold it:

1. **model build** — parameters instantiated, before any step;
2. **warmup** — the first steps, where allocator behaviour differs from steady state;
3. **steady batches** — the regime that dominates wall time;
4. **the largest final-batch shape** — the ragged last batch, which is where a tight cap fails and which a
   steady-state measurement never sees;
5. **forward, backward and optimizer steps that actually instantiate the optimizer slots** — a step that never
   materializes moment buffers under-reports the peak by the size of those buffers;
6. **checkpoint write and reload**;
7. **validation mechanics** on a **TRAIN-only fixture of the planned shape** — the mechanics, not a score.

Recorded per stage and per cell, each under its own name: **host RAM cgroup peak**, **GPU allocated** and
**GPU reserved** separately, scope identity, the kernel limit, optimizer updates, CPU time, wall time.
**The peak is read before the scope is removed.** An external supervisor retains the termination status when a
child fails. **A missing peak is `UNKNOWN` — never zero, and never success.**

---

## 3. Declared before dispatch

Per the order, these are stated **before** anything runs, and a run that would exceed them stops rather than
continues:

| item | value | source |
|---|---|---|
| host cap per cell | **to be set from lane A's isolation test**, not from any figure in §0 | A's published runner |
| device cap per cell | to be set from the built model's measured allocation at stage 1 | this pilot's own stage 1 |
| stage budgets | wall and CPU per stage, from the pilot's own warmup | this pilot |
| remaining allocation | **UNRECONCILED — see §5** | the dispatch index |

**No allocation is invented and no cap is reduced to pass admission.** If the reconciled authority is
insufficient, the deliverable is an **exact additional request** with its arithmetic, not a quiet resize.

---

## 4. What this pilot may never do

- **No test access and no accuracy selection.** Validation appears only as mechanics on a TRAIN-only fixture.
- **No promotion of the twelve historical cells** through this or any new seal. They stay
  `NOT_BOUND_TO_A_SEAL`, custody unchecked.
- **The estimand stays `MATCHED_BUDGET_DIFFERENCE`**, never `CONVERGED_ACCURACY`. Six hundred observed updates
  cannot answer a converged-accuracy question, and renaming it would be the failure this programme exists to
  refuse.
- **No scientific dimension changes** — not a window, not a depth, not a cell count, not the optimization —
  except in an explicitly justified successor **declared before any score**.
- The reported peak is a **peak with explicit headroom**, never a claim that maximum memory has been proven.

---

## 5. Blockers, named rather than worked around

1. **Lane A's runner does not exist yet.** Until a cell gets a fresh exclusive scope enclosing its complete
   process tree with its own reservation, a per-cell number cannot be produced. One sequential child in a
   reused driver scope is not sufficient.
2. **Reachability, not master membership.** The instrument and the v2 seal must be reachable **reproducibly
   from the selected runner commit**. Presence on master is not itself required, and the previous framing of
   this as a master-membership problem was wrong.
3. **The remaining allocation is unreconciled.** The Q2 lane's row carries no numeric remainder I can read,
   and the only remainder recorded anywhere belongs to another lane and is declared not a licence to spend.
   Reconciling it is a precondition of §3 and is mine, not another lane's.
4. **The preferred accelerator host is ineligible** and this design does not assume it returns: read today at
   about 3 GiB free of 14 with 4.93 GiB of unreclaimable kernel slab. The secondary worker is the placement.

**Until 1, 2 and 3 are discharged, `NO_PROGRAMME_TASK_IS_READY` for a long-window fit** — which is the honest
status, not a gap to fill with a number.

— Satoshi
