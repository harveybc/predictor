# Confirmatory design candidate — H1, H2, H3

**Status: CANDIDATE. Not sealed, not authorized, not executable.**
**Author:** Satoshi III (Mujuro Utsutsu), successor technical lead
**Date:** 2026-09-26
**Written under:** DR05 of `SATOSHI_DAY_REVIEW_CONTINUATION_2026_09_26.md`, which assigns the drafting of
this candidate to me and its review to Musashi.
**What this document is not:** authorship of a candidate is not the auditor's signature and is not
permission to execute. Nothing here may be run until the auditor has reviewed it, the implementation
corresponds to what is written, and a reviewed record binds that implementation. I do not write the
auditor's record and I have not written one.

---

## 0. Why this candidate exists, and what it must not repeat

Three defects in my own recent work are the reason this is being drafted rather than assumed:

1. **An estimand was never declared before arms were compared.** Two arms were given the same ceiling of
   4000 updates per seed, the same batch of 64 and the same patience of 3, and then consumed 11,762 and
   10,270 updates respectively because the monitor changed with the loss. I read the resulting difference as
   evidence about a recipe. Offering the same ceiling is not spending the same budget, and **which of those
   two things the estimand is about must be fixed in advance**, because the answer differs.
2. **A scrambled-label difference was used as a resolution floor.** It is not one. Changing the labels
   changes the task, so that difference estimates how much structure the labels carried, not the error
   distribution of the R0/R1/R2 contrast nor its minimum detectable difference. It is withdrawn here and
   nothing in this candidate rests on it.
3. **Two scales were compared without conversion.** The retained arithmetic is
   0.6015421144 − 0.5524983663 = 0.0490437480 kW, which divided by 0.6162615768 is **0.0795826803** in the
   scaled-error unit, not 0.049. A candidate that mixes kW with scaled error produces conclusions about
   neither.

A fourth constraint, structural rather than a defect: **there are three pairs per contrast, not nine
independent replicas.** Seed dispersion is a statement about optimization restarts on fixed data. It is not
the uncertainty of generalizing to new periods or new tasks, and it may not be substituted for it.

---

## 1. Hypotheses

Carried from design v4 (`SATOSHI_C87_C105_RETURN_PACKET_2026_09_12.md` §8) without redefinition:

| id | hypothesis | shape |
|---|---|---|
| **H1** | A1 against A0 | one directional contrast |
| **H2** | A2 against **both** A0 and A3 | two contrasts that must both hold |
| **H3** | descriptive, plus a harm rule | no significance claim; a rule that can only refuse |

A3 is generated from **training-fold moments only**. H3 asserts nothing and can only withhold: it exists so
a descriptive result cannot be promoted into a comparative one.

**H2 is a conjunction, not a disjunction.** A2 beating A0 while failing against A3 is a failure of H2, not a
partial success, and the reporting must not present it as one.

---

## 2. Estimands, declared before any comparison

This is the section whose absence caused the defect, so it is the one that binds hardest. **Each contrast
carries exactly one estimand and it is fixed here.**

| contrast | estimand | what is held fixed | what is allowed to vary |
|---|---|---|---|
| H1: A1 − A0 | **recipe under equal consumed updates** | consumed optimizer updates, batch size, data, splits, initial weights per seed | the architecture and loss under test |
| H2a: A2 − A0 | **recipe under equal consumed updates** | as above | as above |
| H2b: A2 − A3 | **recipe under equal consumed updates**, A3 built from training-fold moments only | as above, plus A3's moment source | as above |
| H3 | none — descriptive | — | — |

**Equal consumed updates, not equal offered ceiling.** Arms run until the declared update count is reached;
early stopping may **record** where it would have fired but may not **end** an arm, because ending on a
monitor makes consumption a function of the thing under test. If an arm cannot reach the declared count for
a reason outside the contrast, it is `CENSORED_BY_BUDGET` and its contrast reports an incomplete population
rather than a narrowed one.

Two consequences I accept in advance, because they are the price of a clean estimand:

- This candidate **cannot** answer "which recipe wins under early stopping". That is a different estimand,
  defensible, and it needs its own design. It is not smuggled in here.
- Fixing consumed updates means the winner may be worse under a practitioner's budget. Reporting must say so
  rather than implying operational superiority.

**Scale.** Every error is reported in **both** kW and the scaled unit, with the scaling denominator named and
its source cited, in every table, with no row carrying only one. No contrast is computed across the two.

---

## 3. Tasks

Drawn from the current programme and the retained ECL evidence. **The recovered Huber/AdamW v2 design is a
household antecedent only** (`satoshi/huber-design-recovery-20260926`, digest
`be2e776e5c64a8422a6411447a4cbeba6e5607c7158456f9a64f89a4244b6965`). It is not a universally winning recipe
and it is not a financial one. It enters as a named prior on hyperparameters for the household task and as
nothing at all for any other.

| task family | admitted here | condition |
|---|---|---|
| household electricity panel | yes | it is the only family with a retained temporal support that has been verified per split |
| ECL R0/R1/R2 | yes, as its own family with its own evidence | its contrasts are not gated by a household control |
| financial series | **no** | no governed forecasting result exists on that domain; every retained price table is `CAUSALITY_UNVERIFIED` with both lineages `UNBOUND` |

**Household electricity is never a substitute for financial validation.** A result in the household family
says nothing about the financial one, and this candidate does not carry one across. The financial lane
proceeds separately on its own data, contracts and risks, and needs a named financial reference before it can
carry any hypothesis at all.

**Task variability is part of the uncertainty.** A contrast measured on one task is a statement about that
task. Where more than one task is admitted, the contrast is reported per task **and** pooled, with the pooled
interval widened by between-task variance rather than by seed spread alone. Where only one task is admitted,
the candidate says the between-task component is `NOT_ESTIMABLE` and the contrast's scope is that task.

---

## 4. Partitions and reserves

| split | role | rule |
|---|---|---|
| train | fitting | `COMMON_INTERSECTION`: every arm trains on the **identical** origin set, element-wise, not merely the same count |
| validation | monitor recording and early-stopping observation | never a selection surface for the confirmatory comparison |
| **reserve (test)** | untouched | `NO_TEST_ACCESS` is a refusal, not a warning |

Non-negotiable properties:

- **The reserve is not opened by this candidate, for any purpose, including to build a frozen prefix.**
- The purge between splits is declared numerically in the design and **bound** to the artifact, not left to be
  discovered in a data audit. Today's measurement — declared 120, observed inclusive gap 121, with one row
  read by neither split — is the shape this must take.
- A preparation whose declared origin policy disagrees with the origins it holds is **refused before sealing
  and before fitting**, on both paths.
- Materialization is verified per split from the delivered bytes, with a split that was not materialized
  reported as `REFUSED_UNMATERIALIZED_BY_DESIGN` rather than as verified.
- **LOPO is required** where the programme's v4 requires it, and a family that cannot support it declares that
  rather than substituting a random split.

---

## 5. Multiplicity

- **Family:** all confirmatory contrasts of H1 and H2 form **one** family. H3 contributes no test.
- **Correction:** **Holm step-down**, declared here, before any p-value exists. Bonferroni is reported
  alongside as a non-governing cross-check, never as the governing correction.
- **Level:** the family-wise level and the bound levels are executable values in the sealed design, not prose.
- **H2's conjunction is not a multiplicity discount.** Both of its contrasts are members of the family and
  both are corrected; requiring both to hold does not buy back alpha.
- A rank belongs to **one** contest. A contrast may not appear in two families, and a result from one may not
  be re-ranked inside another.

**The timestamp is the whole value of this section.** Sealing the correction after seeing a result voids the
screen, regardless of which correction was chosen.

---

## 6. Precision

Precision is derived from **paired differences and the variability that actually limits generalization**, not
from seed spread:

1. **Pairing.** Each contrast is a set of paired differences over matched units — same task, same split, same
   seed, same consumed updates, differing only in the arm under test. The paired difference is the unit of
   analysis.
2. **Interval.** A paired t interval over the pairs, with the **t quantile executable** in the design, plus a
   row bootstrap over evaluation rows as a non-governing cross-check.
3. **What n is.** With three seeds there are **three pairs per contrast**. The interval is computed on three
   pairs and the document says three, everywhere. Nine is not written anywhere.
4. **Components that must appear separately, never merged:** seed-to-seed dispersion on identical
   configurations; temporal variability across evaluation periods; between-task variability. Each is reported
   with its own estimate or as `NOT_ESTIMABLE` with a reason.
5. **Minimum detectable difference.** Derived from the paired-difference dispersion at the declared level and
   the declared n, stated **before** execution as a number with its estimator named. It is **not** derived
   from a scrambled-label difference, from a naive reference, or from anything that changes the task.
6. **What this candidate does not claim.** It does not claim sufficient power. If the stated minimum
   detectable difference is larger than the effect the programme cares about, the honest reading is that the
   design at this n cannot settle it, and the candidate says how many pairs would be needed rather than
   proceeding and calling a null result evidence of absence.

Every result carries the programme's closure table, generated from artifacts and never typed: model error with
its scale, the naive reference **on the same rows**, the skill, a literature value with its source or
`NOT_CARRIED`, and comparability with `NOT_COMPARABLE` plus a reason. A closure `null` is an **absent
measurement**, never a numeric zero.

---

## 7. Budget

- **Aggregate, observed, enforced.** The budget is a campaign-level quantity enforced at the launcher, on
  **observed** consumption. A per-child limit is not a campaign budget; today's executable path had a child
  limit and an advisory pilot report that did not prevent fits, and that combination is what must not recur.
- **Footprint from the process tree or cgroup peak**, never from the main process RSS alone, and each pilot's
  footprint bound to its own evidence.
- **Admission is a held reservation**, not a check. This candidate is **not executable until shared atomic
  admission exists on the host**: two requests that fit alone but not together must not both be admitted.
  That repair is in flight under DR01 and is a precondition here.
- Declared per contrast before execution: total consumed updates, wall ceiling, memory ceiling, and the host.
  A terminated run is recorded with its unit, limit, attempt and cost lost, distinguishing a cgroup OOM from
  memory pressure. **A termination never becomes a retry with a larger cap.**
- The external RTX 5090 is first choice for new GPU-compatible work; other machines remain eligible under
  observed admission and cooling. **A free GPU does not license fabricating an experiment**, and the CPU/GPU
  or environment of a sealed experiment is not changed without declaring it and demonstrating equivalence.

---

## 8. Stopping

- **Arms** stop at the declared consumed-update count. Early stopping **records** and does not **end** (§2).
- **The screen** stops when every unit of the census has a complete record or a named refusal. There is no
  interim look and no optional stop: an interim decision would be a second, undeclared test.
- **Resumption** carries the complete schema — arms, lineage, state, disjointness — and a partially written
  record is never read as complete. An interrupted unit's bytes are preserved and set aside; the unit re-runs
  from the beginning rather than resuming mid-arm.
- **Verification before any verdict.** Raw authenticated evidence and numerical re-derivation precede any
  `VERIFIED`; a producer's summary never suffices; repeated seeds and incomplete populations refuse by name.
  A verdict of `DOES_NOT_ADVANCE` is a complete delivery, and so is `NO_NEW_MEASUREMENT`.

---

## 9. What this candidate leaves undetermined, deliberately

1. **The concrete n.** Three seeds give three pairs, which I judge insufficient for the effects the programme
   cares about. I have not set a larger n because that is a budget decision with a real cost, and the
   auditor should see the minimum detectable difference at n = 3 before anyone pays for more.
2. **Which ECL tasks are admitted.** I name the family; the specific tasks require the dispatch index that is
   being reconciled under DR02, and choosing them from a stale map is how a closed node got re-dispatched
   before.
3. **The financial reference.** Named as required and named as absent. Nothing here supplies it.
4. **Whether the early-stopping estimand also deserves a design.** I think it does. It is a separate
   candidate, not an amendment to this one.

---

## 10. Preconditions before this may be sealed

| # | precondition | owner |
|---|---|---|
| 1 | the auditor's review of this candidate | Musashi |
| 2 | shared atomic admission on the executing host (DR01) | me, in flight |
| 3 | the M4-class verifier repairs, so a fabricated summary, a repeated seed or an incomplete population cannot reach `VERIFIED` (DR04) | me, in flight |
| 4 | the reconciled dispatch index, so tasks are chosen from real dependencies (DR02) | me, in flight |
| 5 | a reviewed record binding the implementation that will run, not merely a recorded revision | Musashi |
| 6 | for the financial lane only: a named financial reference and contract | open |

A design sealed before 2, 3 and 4 would be a protocol that cannot be trusted to have run as written, which is
the failure this whole candidate exists to avoid.

— Satoshi
