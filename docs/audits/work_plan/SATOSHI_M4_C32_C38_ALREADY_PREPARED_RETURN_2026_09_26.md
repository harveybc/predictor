# M4 C32–C38: the CONFIRMATION screen was already prepared, it is still unexecuted, and it is blocked on two records nobody has written

Order: **P2 of `agent-multi@889320ee`**, section "P2 - M4 C32-C38 confirmation preparation". Worktree `predictor-rp132`, branch
`satoshi/rp132-rp134-20260923`. **No model was fitted, loaded, scored or replayed; no GPU; no allocation; no CONFIRMATION array,
score or ledger was created or touched.** Everything below is a re-derivation from bytes that already existed.

I was sent to prepare this screen. I did not prepare it, because **it was prepared on 2026-09-10 and pushed**, and preparing it a
second time would have produced a second, divergent frozen plan for the same population — the one failure mode a pre-outcome
freeze cannot survive. What I did instead is the thing that was missing: an **independent re-verification** that the frozen plan
is still the plan that was frozen and is still unexecuted, and a statement of the exact decision it is waiting on.

---

## 1. The authoritative definition, quoted

The order is `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_POST_M4_C31_PRIORITIZED_ORDER_2026_09_10.md` in **`agent-multi`**, committed
at `889320ee` ("Audit M4 calibration and prioritize successor work", 2026-09-10). Its P2 opens:

> ## P2 - M4 C32-C38 confirmation preparation
>
> This package may begin only after the P0 and P1 returns are committed and pushed. It authorizes CPU-only design,
> implementation, tests and DEVELOPMENT mechanics. It does not authorize generating, loading or scoring CONFIRMATION arrays.

and closes:

> Required disposition:
>
> `M4_CONFIRMATION_PROTOCOL_READY_FOR_EXTERNAL_MUSASHI_REVIEW`
>
> Stop there. No CONFIRMATION execution, GPU, DOIN integration, financial data, live action or production deployment.

Its C32 pins the evidence to be bound by exact bytes:

> - reviewed tip `5e7a8fd430c8231a049baf03f00e720ba24ec994`;
> - design self-identity `d7280a92047d98898418fb7cd750b22c506a621eb381d9847e0fe926b7df69b9`;
> - numeric amendment self-identity `43e0804e1e6e583b10ddbe46b7d4cd752838b0473ccbc6496f0e458c49aedd4b`;
> - governing adjudication self-identity `b35b6fd969aa162047bdfb55b8f9fcce01aa76864c388d29a1c36642ab051ade`;
> - exact re-derived facts: 21/28 eligible slots, two incomplete generators, zero calibration-incomplete cells, and M2 gain
>   `-0.41982887`.

Its C33 states the freeze that must be honest about its own provenance:

> Create an append-only confirmation successor that labels the selection rule `CALIBRATION_DERIVED_AND_REVIEWED`, never
> predeclared. […] The successor must describe this as a calibration decision made before any CONFIRMATION data. It is a
> scientific analysis freeze, not `scientific_change: NONE`.

Its C34 fixes the estimand and the multiplicity behaviour (items 1–7 verbatim in the order), including:

> 6. Keep `incremental_prediction::M2_vs_M1` as the sixteenth, non-rejecting `p=1` placeholder because M2 failed CALIBRATION.
> 7. Apply the frozen Holm procedure over all 16 slots, including placeholders.
>
> The generator is the independent unit. Seeds and widths are nested or paired repetitions, never independent observations.

C35 requires the runner/verifier, C36 the two external-record templates that "grant nothing", C37 a fifteen-kill acceptance
battery with guard-removal mutants, C38 the return packet.

The authority the order consumes is Musashi's audit
`docs/audits/MUSASHI_AUDIT_M4_C31A_C31F_AND_EXTERNAL_CAMPAIGN_STATUS_2026_09_10.md` at the reviewed tip, disposition
`M4_CALIBRATION_ACCEPTED_WITH_PRE_CONFIRMATION_FREEZE_REQUIRED`, whose F3 is the M2 stop and whose F4 says plainly:

> A separate successor, exact population census, independent verifier, analysis implementation and external execution record
> must exist before CONFIRMATION is generated or scored.

## 2. Where the front actually lives — and where it does not

**The M4 front is in `agent-multi`, not in `predictor`.** This matters more than a filing detail, because the brief that reached
me named `predictor/tools/df_*` as the screen machinery to build on.

| | |
|---|---|
| repository | `agent-multi` |
| branch | `satoshi/model-capacity-m3-20260908` |
| tip | `0e99ad1add3635dd4f0e151936d6fe4bf6379592`, **equal to `origin`** (pushed) |
| code | `tools/m4_confirmation_protocol.py`, `tools/m4_confirmation_runner.py`, `tools/m4_v5_{protocol,runner,adjudicate}.py`, `tools/m4_generator_bank.py`, `tools/m4_intervention_{design,runner}.py` |
| plan | `docs/research/model_capacity/M4_CONFIRMATION_SUCCESSOR_2026_09_10.json` |
| battery | `tests/test_m4_confirmation_protocol.py` |

`predictor` contains **none** of it: no `tools/m4_*`, no `tests/test_m4_*`, and `git grep verify_run_v5` is empty on this branch.
The `predictor` files that look like the same machinery are a **different front**: `tools/df_d2_design.py` (C171, the superseding
D2 v2 operator design), `tools/df_d2_adjudicate.py` (C172/C175/C176, D2 decisions from FRESH rows), `tools/df_d3_design.py` (the
J1 temporal amendment), `tools/df_campaign_plan.py` (C162, memory-based campaign distribution). They screen **operators over data
banks**; M4 screens **model capacity over synthetic generators**. They share one real convention — the seal is the sha256 of the
canonical body with the digest field removed, and `validate_*` re-derives it rather than reading it back — and nothing else.

Building an "M4 CONFIRMATION screen" out of the `df_*` machinery in `predictor` would have created a second frozen plan for a
population that already has one. **That is the refusal in this return** (§6).

## 3. The preparation, as it stands, re-derived — not taken on trust

The cycle is three commits on that branch, all 2026-09-10:

| commit | what |
|---|---|
| `d8b45eb4` | PRE freeze at the reviewed tip: four identities recomputed, every order fact re-derived, the commanded protocol proved **wholly absent**, zero writes (2,120-file byte inventory equal) |
| `89a3781f` | the cycle: C32 binding, C33 successor, C34 sixteen contrasts, C35 runner + verifier, C36 two templates, C37 battery, DEVELOPMENT probe, sealed POST |
| `0e99ad1a` | the return packet `docs/handoffs/GENERAL_SATOSHI_TO_MUSASHI_M4_C32_C38_RETURN_2026_09_10.md`, disposition `M4_CONFIRMATION_PROTOCOL_READY_FOR_EXTERNAL_MUSASHI_REVIEW` |

My receipt is
[`docs/audits/evidence/M4_C32_C38_REVERIFY_20260926/reverify_m4_c32_c38.py`](../evidence/M4_C32_C38_REVERIFY_20260926/reverify_m4_c32_c38.py),
transcript [`reverify_m4_c32_c38.out`](../evidence/M4_C32_C38_REVERIFY_20260926/reverify_m4_c32_c38.out).
It reads the `agent-multi` **object store** at the pinned refs (no checkout of that branch is disturbed) and **recomputes**
instead of reading back. **81 checks, 81 passed, exit 0.**

### 3.1 Identity — recomputed from the bytes, with the canonical rule

| document | self key | recomputed | order pin |
|---|---|---|---|
| `M4_SEALED_DESIGN_V5_2026_09_09.json` | `design_sha256` | `d7280a92…b7df69b9` | matches |
| `M4_V5_NUMERIC_VALIDITY_AMENDMENT_1_2026_09_09.json` | `amendment_sha256` | `43e0804e…49aedd4b` | matches |
| `M4_V5_CALIBRATION_ADJUDICATION_ATTEMPT3_GOVERNING_2026_09_09.json` | `record_sha256` | `b35b6fd9…ab051ade` | matches |
| `M4_CONFIRMATION_SUCCESSOR_2026_09_10.json` | `successor_sha256` | `6a50d97ddfb3a8e8dd1b5fbc83ebd95e60e1c087b3fc5e01697c2d783a50608c` | matches the packet's `6a50d97d` |

The three file digests Musashi's audit §1 pins (`0a0fb757…`, `49c36395…`, `51247b78…`) also match byte for byte, as do the nine
deliverable digests recorded in the receipt.

### 3.2 The five C32 facts — re-derived from the adjudication **structures**

| fact the order demands | re-derived from | value |
|---|---|---|
| 21/28 eligible slots | `confirmation_slots[*].typed_status` | 21 eligible, 7 typed ineligible, 28 total |
| two incomplete generators | `incomplete_units_in_denominator`, collapsed to generators | `state_space::clean::w64::g0`, `…::g6` |
| zero calibration-incomplete cells | `dispersion[*].status` | 0 |
| M2 gain `-0.41982887` | `ladder.m2_minus_m1_paired_gain` | `-0.41982887` exact |
| the record claims no authority of its own | `authority` | `CANDIDATE_FOR_MUSASHI_REVIEW_NO_CONFIRMATION_AUTHORITY` |

### 3.3 The C33 freeze — and its floor derived, not read

`selection_rule_label = CALIBRATION_DERIVED_AND_REVIEWED`; `classification = SCIENTIFIC_ANALYSIS_FREEZE` with the note that it is
explicitly **not** `scientific_change: NONE`; rule ≥12 of 16 `LEARNABLE_UNDER_FROZEN_BUDGET` with zero `NUMERICALLY_INVALID`;
48 generators per eligible slot; 3 nested seeds; attrition allowance 0.20 and floor **39**, which the receipt re-derives as
`max(3, ceil(48 × 0.8))` rather than accepting the stored field; M2 `DOES_NOT_ADVANCE_FROM_CALIBRATION` carrying the measured
gain; the 21 + 7 slot population copied **by identity** (cell-set equality against the governing adjudication, not by count);
`supersedes_design_sha256` = the sealed v5 digest, v5 bytes untouched.

### 3.4 The C34 sixteen contrasts — topology re-derived from the slot list

The receipt does not read the topology off the plan; it derives it from the 21 eligible cells and then checks the plan agrees.

| | contrasts | |
|---|---|---|
| both frozen widths (w16 and w64), equal average | **10** | `identity::clean`, `majority::clean`, `dnf3::clean`, `sine::clean`, `sine::white`, `chirp::clean`, `chirp::white`, `am::clean`, `am::white`, `state_space::white` |
| single frozen width, **used and named** | **1** | `state_space::clean` → `w16` |
| no eligible width → `NOT_EVALUABLE`, non-rejecting `p=1` | **3** | `parity4::clean`, `discontinuity::clean`, `discontinuity::white` |
| 15th slot | | `checkpoint_effect::primary_pair` |
| 16th slot | | `incremental_prediction::M2_vs_M1`, non-rejecting `p=1` placeholder |

**The arithmetic closes: 10 × 2 + 1 = 21 eligible slots.** The unit is the generator; Holm runs over all 16 including
placeholders; width-specific effects are `SECONDARY` heterogeneity only.

### 3.5 Still prepared, still not executed

| check | result |
|---|---|
| paths naming M4 CONFIRMATION at the return tip | exactly 6: the plan, the two templates, the two tools, the battery |
| CONFIRMATION array / arm record / ledger / verdict tracked | **none** |
| every `::CONFIRMATION::` occurrence in a data file | **1,344 of 1,344 are `RESERVED::` reservations**, in the 2026-09-09 DEVELOPMENT `RUN_LEDGER.json` — never an observation |
| that reservation census at the reviewed tip vs the return tip | **byte-identical** — the cycle added no CONFIRMATION record |
| C37 acceptance battery, live checkout at `0e99ad1a` | **28 passed** |
| `m4_confirmation_runner.py plan` | 3,024 units = 21 × 48 × 3, 4 checkpoints, census `12cfd9ad785b41e788ffce575ec575ab2a78ab772151b5e3b94c8f0c71169ea0`, `mode: PLAN_ONLY_NO_AUTHORITY`, `execution_open: false`, both records absent |
| `m4_confirmation_runner.py execute --out …` | `REFUSED: Musashi design-review record is ABSENT`, and **the output directory was never created** |
| authority records installed on this host | none at `~/.local/state/m4_confirmation_authority` or `/var/lib/m4_confirmation_authority` |
| both templates | still carry `<placeholders>` — they grant nothing |

One number to correct, mine to name: the PRE's prose says "1,008 CONFIRMATION generators existed only as reservations". The
retained ledger holds **1,344** (= 28 slots × 48). Both are reservation counts of different populations — 1,344 reserves *all*
slots, 1,008 is the 21 eligible × 48 — and neither is an observation. The number that governs execution is the runner's own
census, **3,024 = 21 × 48 × 3**, which I re-derived. This is a prose imprecision in a PRE, not a change in the plan.

## 4. Tests added

`tests/test_m4_c32_c38_reverify.py` — **25 passed**. It does two jobs:

1. pins the receipt's own constants against the *words* of the order (the four identities, 21/28/7/2/0/−0.41982887, 12-of-16,
   48, 0.20, floor 39 derived by formula, 3 seeds, the three frozen labels, and the 3,024 arithmetic), so an edited pin fails in
   CI rather than in a future reader's trust;
2. **proves the receipt is not vacuous**: it mutates each order fact, each C33 policy number, the successor self-identity and a
   deliverable digest in turn and requires the receipt to go **red** each time; a deliverable that vanishes raises rather than
   passing quietly. The canonical-digest rule is tested for both properties that make it a seal — blind to the self key,
   sensitive to every other field.

The mutation tests skip if `agent-multi` is absent from the host; the constant tests always run.

## 5. Cost

CPU only, `CUDA_VISIBLE_DEVICES=''`, every job under `crispdm-run -m 4G/6G -t 300–900 -n m4prep`. Measured: the receipt ~2 s, the
C37 battery 4.91 s, the planner and the refused `execute` under a second each, the new test file 3.37 s. One temporary detached
`git worktree` of `agent-multi@0e99ad1a` was created to exercise the git-reading guards and **removed**; `git worktree prune` ran
after it. No branch, no checkout and no running process was disturbed. **No training, no GPU, no allocation.**

## 6. What I refused

1. **I refused to prepare the screen again.** A pre-outcome freeze whose population, rule and multiplicity family already exist
   under a self-identity cannot be re-prepared without producing two plans for one population; whichever one were later cited,
   the choice would itself be a post-hoc decision. Evidence for the refusal is §3: the plan exists, its identities re-derive, and
   it is unexecuted.
2. **I refused to build it in `predictor` out of the `df_*` screen machinery.** Same reason plus §2: that machinery screens
   operators over data banks under the D2/D3 designs, and M4 screens model capacity over synthetic generators in `agent-multi`.
   A `predictor`-side copy would be a fork of a sealed plan.
3. **I refused to fill, install or simulate either external record**, and I refused to run `plan` anywhere that could be mistaken
   for authority. I ran `execute` exactly once, to prove it refuses **before** creating anything, and in a temporary directory
   that I then deleted.
4. **I refused to re-run the full `agent-multi` suite** (the packet reports 3256 passed / 2 failed / 5 skipped in 31 minutes, the
   two failures being the inherited D1 pair whose private evidence file is absent from this host). I re-ran only the C37 battery.
   The suite figure in this return is therefore **the packet's claim, not my measurement**, and is labelled as such.
5. **I did not commit anything to `agent-multi`.** Everything I wrote is in `predictor`.

## 7. What Musashi or the owner must still decide

The preparation is complete and its execution is gated by exactly two records that **do not exist**:

1. **Musashi's design-review record.** The template is
   `agent-multi:docs/audits/evidence/MUSASHI_M4_CONFIRMATION_DESIGN_REVIEW_TEMPLATE_2026_09_10.json`. The packet's §11 names what
   it must rule on, and one item is load-bearing and must not be waved through: the successor **declares that it supersedes a
   sealed-design sentence** — v5's "Bonferroni over the frozen confirmatory contrast family" is replaced by **Holm over all 16
   slots**, under the order's own C34.7. Both control FWER at α; Holm is uniformly at least as powerful; it is frozen before any
   CONFIRMATION outcome. It is still an override of sealed text and it is the one place where the plan changes the design rather
   than extending it. Also to rule on: the 12-of-16 rule, the floor of 39, the M2 exclusion, the per-contrast two-sided
   one-sample t on generator-level effects, and the runner's gate chain and verifier reconstruction rules.
2. **The owner's execution record**, which chains to the review record by digest and pins the reviewed population and the CPU
   limits. Template `agent-multi:docs/audits/evidence/OWNER_M4_CONFIRMATION_EXECUTION_TEMPLATE_2026_09_10.json`.

Until both exist, `execute` refuses before any artifact — verified live in §3.5. **No agent can unblock this; a review and an
owner decision are the only keys.** Nothing here asks for either. The 3,024-unit run they would open is also not costed in this
return, because costing it is part of the owner's record, not mine.

One record-keeping consequence, which is why this return exists at all: the standing memory index still carries
"NEXT: P2 M4 C32-C38 preparación CONFIRMATION (21 slots/16 contrastes/M2 excluido; SIN ejecutar)" as the next runnable node. It
is not runnable and it is not next. It has been **done and awaiting external review since 2026-09-10**, and the index should say
so, or the next agent sent to this node will do what I refused to do in §6.
