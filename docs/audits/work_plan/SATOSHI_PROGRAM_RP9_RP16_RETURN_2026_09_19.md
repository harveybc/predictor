# Return RP9–RP16 — corrections, contract metrics, ARCH-A/B/C/0 comparison

Date: 2026-09-19. Executor: Satoshi. Order: [MUSASHI_MOD_E0_RP9_RP16_2026_09_18.md](../../handoffs/MUSASHI_MOD_E0_RP9_RP16_2026_09_18.md)
over the dictum [MUSASHI_RP1_RP8_REVIEW_2026_09_18.md](MUSASHI_RP1_RP8_REVIEW_2026_09_18.md).
Branch `satoshi/r1-r6-20260914`; commits `bd741f5`, `e1550ac`, `8c7d3fa`, `deec2ca`, `c9e1971` and the closing
commit named at the end. Evidence under `docs/audits/evidence/d3_k5_20260917/` (files `RP9_*` … `RP16_*`).
One external request, at the end. No reserve, no live, no financial scoring; H-CORE untouched (after E1).

## Per block: implemented · executed · verified locally · reconciled independently · reviewed · inference

| block | impl. | exec. | verified locally | reconciled independently | reviewed | inference |
|---|---|---|---|---|---|---|
| RP9 state + PRE | yes | yes | yes | n/a | no | master's state adopted; PRE identical to the dictum; originals unchanged |
| RP10 closure by population | yes | yes | yes (tests through the real CLI/API) | yes (historic pilot live: 69/69) | no | every counterexample refuses; 6/69, empty root, strangers, altered design, parent disagreement typed |
| RP11 task/weights/metrics | yes | yes | yes (fresh-process replays; 12/12 mutants killed) | yes (live) | no | historic pilot reverified without training: TOTAL |
| RP12 profiles + inference | yes | yes | yes | n/a | no | descriptor v2 leaves every consumed datum unchanged → reuse justified; H2/H3 reframed |
| RP13 contract metrics | yes | yes | yes | yes (backfill gen 3 equal in the cube) | no | trajectory of the historic run NO_MEDIDO, everything else backfilled |
| RP14 ARCH comparison | yes | yes (informative stage, 3 hosts) | yes (120 verified, 4 DX failed then re-run as a sealed successor) | yes (live 124/124 + DX 4/4) | no | no architecture wins; readout, not history, carries most of the sequence–summary gap |
| RP15 E1 + RL demand | yes | yes | yes | n/a | no | 4 public families; source deficit recorded; one owner decision escalated |
| RP16 closure | yes | yes | suite below | ledger + live closures | no | negatives and NO_MEDIDO visible |

## RP9 — state and PRE

`PROJECT_METHOD_STATE.json` had conflict markers after the merge; master's version (Musashi's) was taken and
`check_plan.py` passes (`documentary_coverage PASS`). MOD-E0-DEV stays EXECUTED (review with corrections),
MOD-ARCH-COMPARE / MOD-FROZEN-PREFIX / MOD-CORE-PRETRAIN keep their dependencies; H-CORE not launched.
The dictum's counterexamples were frozen as PRE by running Musashi's own `reproduce.py` on copies:
[RP9_PRE_REVIEW_COUNTEREXAMPLES.json](../evidence/d3_k5_20260917/RP9_PRE_REVIEW_COUNTEREXAMPLES.json) is
byte-equal to the dictum's `results.json` in every counterexample (labels = pred, denominator × 100,
rows + 100 000, MAE 999 999, weights text, empty root, 6/69, parent disagreement, NaN MEDIDO: all accepted
by the old code); 497 original files, digests before/after identical.

## RP10 — a closure bound to the population

`tools/df_mod_e0_close.py` (v2). Population derived from the sealed `DESIGN.json` (schema, self-digest, cells
equal to the design's own enumeration, dependencies members) plus the runner's pilot derivation and the
campaign registrations (`CAMPAIGNS.json`); `REPORT.json` is compared against the files (parent), never used
as a source. Per member: job equals the design's job on every consumed parameter (identity, arm, window,
training rule, design digest, run id, role, extractor path, architecture, donor kind, diagnostic); receipts
and digests; dependency graph (an H3 arm is `H3_ARM_WITHOUT_VERIFIED_DONOR` unless its declared donor is a
VERIFIED attempt). A stranger attempt, a duplicated or unexpected terminal, a design that does not seal
itself or a report of another run are typed refusals (exit 2). A missing or failed member makes the closure
PARTIAL (exit 1) with the denominator fixed by the design; `all_verified` is the conjunction of every
member's status and the parent comparison (never `all([])`). Effects are computed on VERIFIED cells only and
labelled with `verified_cells / of`.

Live closure: the registered population of each campaign is derived from accounting (reconciliation:
missing units ∪ units with a terminal; accounting and lake must agree) and compared with the design; per
unit the current-generation terminal in the warehouse must carry the local status, costs, output digest,
code identity, design identity, every metric row and every metric state. Local and live are separate
sections of `CLOSE.json`; neither stands for the other.

Tests through the real CLI/API on copies of a small real campaign (`tests/test_df_mod_e0_close.py`, 21):
empty root (refused), root without attempts (PARTIAL, 11 MISSING, effects empty), 6 of the population
(PARTIAL, denominator kept, effects labelled incomplete), omitted member, duplicated terminal, stranger
attempt and stranger in the report (refused), altered design (refused: self-digest), re-sealed design (every
job unbound), parent disagreement (`all_verified` false, `parent_equal` false), missing replicate and failed
attempt (arms `NO_DONOR`), receiver without a verifiable donor; CLI exit 0 only for TOTAL.

Historic pilot `mod-e0-dev-v2`: local closure TOTAL 69/69 ([RP11_MOD_E0_V2_CLOSE_LOCAL.json](../evidence/d3_k5_20260917/RP11_MOD_E0_V2_CLOSE_LOCAL.json)),
live closure all equal, population equal, 69 covered ([RP10_MOD_E0_V2_CLOSE_LIVE.json](../evidence/d3_k5_20260917/RP10_MOD_E0_V2_CLOSE_LIVE.json)).
Effects unchanged from RP5: H2 e(h) = −0.0089 / +0.0058 / −0.0001 / +0.0056, slope +0.0038; H3 d0 = −0.0227,
d1 = −0.0907, γ = −0.0680.

## RP11 — the task, the weights and the metrics, not their labels

Recomputed from the generator and the contract for every attempt: rows, targets, naive, oracle,
denominators (train seasonal-naive MAE), train-only scale, the ridge linear reference; schema, shape, dtype
and finiteness of every array; every score of every baseline and variable (MAE, MASE per variable and
means); profile labels, kept descriptors, ARI, the profile assignment and the exact predefined
redistribution k. `mase()` now refuses shape/dtype violations and returns states: MEDIDO, NO_APLICA (zero
denominator), NO_MEDIDO (non-finite or empty: never a mean), plus MSE/RMSE.

Weights: `df_mod_e0.py --replay` rebuilds the graph in a fresh CPU process, loads the saved weights,
regenerates the inputs, predicts every split and compares with the stored arrays; verifies that the saved
checkpoint reaches the recorded best validation loss; for an H3 arm loads the donor in its own graph and
compares every extractor weight and every adapter activation. Tolerance declared before the check
(|diff| ≤ 1e-5, float32 accumulation order); measured: historic pilot 69/69 |diff| = 0.0, restore ≤ 6.5e-8;
ARCH stage 120/120 |diff| ≤ 9.5e-7 across three hosts. The parity flag alone is never evidence. Checkpoints
are bound to jobs and donors (`_bind_donor`: the donor must be the declared dependency's attempt; digest
recorded). The update allowance is enforced inside an epoch (an allowance below one epoch was silently
exceeded before: 8 → 32 updates); `fit` re-evaluates the restored weights (`restore_verified`) also under a
budget stop; the ML01 future test now perturbs the real `causal_oracle` callable.

Mutants: [RP11_MUTANTS.json](../evidence/d3_k5_20260917/RP11_MUTANTS.json) — 12 guards mutated on a copy of
`tools/` (labels, denominator, rows, MAE, unreadable weights, prediction tolerance, frozen extractor,
non-finite-is-measured, parent, `all_verified` conjunction, update allowance, strangers): 12/12 accept the
altered copy under the mutant and refuse it intact (`MUTANT_KILLED_BY_BEHAVIOUR_TEST`).

## RP12 — profiles and the inference

Trend strength corrected to F_T = 1 − Var(R)/Var(T+R) on the declared centred moving-average decomposition
(tests: pure sine → F_T ≈ 0, F_S ≈ 1; pure ramp → F_T ≈ 1; constant → 0 by convention; white noise → both
small; mixtures → both high; short series and non-multiple lengths); the decomposition is declared a
train-batch characterisation, not a causal online operator. The executed descriptor (v1) is kept only for
the reanalysis: [RP12_PROFILE_REANALYSIS.json](../evidence/d3_k5_20260917/RP12_PROFILE_REANALYSIS.json) —
15/15 ordered partitions equal, kept masks equal, sizes equal, random redistributions equal, x digests equal
to the records, donors' assignments equal → reuse justified; nothing historic retouched.

13A errata fixed (P_B(h=3) = 60, W/P_B = 0.80, (W−1)/P_B = 0.783; rows [47, 2102) = 2055; the curve label is
the mse loss). H2 reframed as a descriptive difference between two grouping procedures (not equivalence);
H3 as a contrast between two complete procedures with a donor pre-trained for one of them.
[RP12_PRECISION_DESIGN.json](../evidence/d3_k5_20260917/RP12_PRECISION_DESIGN.json): half-widths and the
effect detectable at 80 % power per number of replicates for SD × 0.5 / 1 / 2; the pilot's own SD has a 95 %
interval spanning a factor ~4; no confirmatory size proposed.

## RP13 — the metrics contract

`tools/df_mod_e0_metrics.py`: D/Y per variable and split — bytes, dtype, endian, shape, mask, zlib 9 / lzma 6
lengths on raw float64 and on train-frozen 16-quantile symbols (reused from `df_profile_information`), H0,
lag-1 conditional redundancy (INCONCLUSIVE below 20 counts per state), planted SNR with the declared power
convention, planted lagged dependence, ACF/Welch/trend/seasonal, support. M/G — parameters total/trainable/
frozen apart, serialized weight bytes and raw float32 bytes, per-layer Frobenius, per matrix effective rank
(entropy, reused), stable rank, nuclear ratio and spectral norm (a zero matrix is NOT_DEFINED, not 0),
gradient norms on one fixed training mini-batch, adapter activation statistics on one fixed validation batch,
graph nodes/edges/density (spectral radius NO_APLICA for a DAG). Checkpoints at a geometric epoch schedule fixed
before any run, plus initial, final (pre-restore) and best; cost measured in the pilots (descriptors ≈ 0.4 s per
checkpoint, per-update cost projected net of them). Terminal rows carry `mod_e0.data.*` and `mod_e0.model.*`
with their states in the tags (168 rows per cell).

Backfill of the historic run from arrays and saved weights only: every data metric, and the model descriptors
of the saved (best) checkpoint including gradients and activations on the fixed batches; initial, scheduled
and final checkpoints NO_MEDIDO with the reason. Published as generation-3 successor terminals through the
outbox for the 69 units (8 776 rows), reconciled (missing 0) and equal in the warehouse:
[RP13_MOD_E0_V2_CLOSE_LIVE_BACKFILL.json](../evidence/d3_k5_20260917/RP13_MOD_E0_V2_CLOSE_LIVE_BACKFILL.json),
[receipt](../evidence/d3_k5_20260917/RP13_MOD_E0_V2_BACKFILL_RECEIPT.json).

## RP14 — ARCH-A / ARCH-B / ARCH-C / ARCH-0

Architectures in `df_mod_e0.build_modular(arch=…)`: A = two residual causal Conv1D blocks, identity
integrator (reference); B = TCN (the executed pilot's extractor, 9 608 parameters unchanged); C = Conv1D +
GRU(16) (GRU chosen before results: fewer parameters, one state); 0 = no learned extractor (raw group
channels). Common core/head (Conv1D(16,3) for sequence fusions, Dense(32) for summary fusions, Dense(p) +
persistence skip, ELU as a recorded DEV factor). Reach declared per architecture and fusion and TESTED by
perturbation (A: branch 5, sequence 7; B: 65/48; C: whole context; 0: 3 through the core, which sees the
planted lag τ = 3). Readout control: 2 × 2 fusion × readout (`sequence`, `sequence_gap`, `summary`,
`summary_last`) sharing one frozen donor. Bounded donor sensitivity (A, B; r = 1): donor trained with the
summary receiver. Diagnostic trend/event condition (drift 0.001/step, level shifts at t = 1300 and 2750).

Budget: the full factorial ([RP14_ARCH_DESIGN.json](../evidence/d3_k5_20260917/RP14_ARCH_DESIGN.json), 338
cells, 3 000 updates) projected 52 624 s CPU with 25 % headroom from the 12 cost pilots
([plan not launched](../evidence/d3_k5_20260917/RP14_ARCH_FULL_PLAN_NOT_LAUNCHED.json)) against the 14 400 s
ceiling with 3 753 s already spent. An informative stage was sealed BEFORE any score
([RP14_ARCH_STAGE_DESIGN.json](../evidence/d3_k5_20260917/RP14_ARCH_STAGE_DESIGN.json), sha `2059a3ab…`):
four architectures, H2 at h ∈ {0, 3}, H3 at r ∈ {0, 1}, readout controls at r = 1, donor sensitivity A/B,
DX, two replicates, allowance 1 100 updates (33 epochs; the pilot's best epochs: median 25, q90 34),
patience 8; 112 cells projected 5 650 s with headroom. Executed on three hosts under their own identities
(coordinator omega parallel 4; dragon WORKER_A parallel 6; gamma WORKER_B parallel 2, memory-aware
dealing 4/6/2), governance registered every unit before any child, workers executed only, the coordinator
reported the collected attempts ([REPORT.collected](../evidence/d3_k5_20260917/RP14_ARCH_STAGE_REPORT_COLLECTED.json):
112 cells, missing 0, accounting = lake). Incidents recorded: gamma's worker crashed twice (a thread race
in the module loader and a loader/dataclass interaction; fixed at `8c7d3fa`, resumed from its outcomes);
the four DX cells failed at start by a code defect (`KeyError 'profiles'`: the diagnostic hypothesis was
routed to the H3 arm map) and stand as FAILED terminals with their tracebacks; they were re-run as a sealed
DX-only successor ([design](../evidence/d3_k5_20260917/RP14_ARCH_STAGE_DX_DESIGN.json), inheriting the
stage's measured pilot costs by identity; [closure TOTAL 4/4, live equal](../evidence/d3_k5_20260917/RP14_ARCH_STAGE_DX_CLOSE.json)).

Closure of the stage: local [PARTIAL 120 VERIFIED + 4 FAILED (DX)](../evidence/d3_k5_20260917/RP14_ARCH_STAGE_CLOSE_LOCAL.json)
(108 cells + 12 pilots verified; replays |diff| ≤ 9.5e-7), live [all equal, population equal, 124 covered](../evidence/d3_k5_20260917/RP14_ARCH_STAGE_CLOSE_LIVE.json).
Effects and tables: [RP14_ARCH_STAGE_EFFECTS.json](../evidence/d3_k5_20260917/RP14_ARCH_STAGE_EFFECTS.json),
[RP14_ARCH_STAGE_TABLES.md](../evidence/d3_k5_20260917/RP14_ARCH_STAGE_TABLES.md) (per cell: MAE, MASE, naive,
linear, oracle, test MASE, denominator, reach, updates, stop, cost).

Receiver adequacy (H2 profiles at h = 3, r = 1, validation): every architecture beats the seasonal naive
(0.506 / 0.509): A 0.401 / 0.410, C 0.409 / 0.419, 0 0.408 / 0.421, B 0.437 / 0.443; the linear reference is
0.418 / 0.425 and the oracle 0.364 / 0.366. B (the pilot's TCN) is the only architecture that does not reach
the linear bar at this allowance.

| arch | e(h0), e(h3) | H2 slope [boot 95 %] | d_0 | d_1 [boot] | γ = d_1 − d_0 [boot] | ρ_1 last − pooled | donor Δ | cpu/cell |
|---|---|---|---|---|---|---|---|---|
| A | +0.0022, −0.0014 | −0.0012 [−0.0016, −0.0007] | −0.065 | −0.030 [−0.033, −0.028] | +0.035 [+0.033, +0.038] | −0.110 | +0.083 | 16.3 s |
| B | −0.0154, +0.0088 | +0.0080 [+0.0079, +0.0081] | −0.028 | −0.018 [−0.022, −0.015] | +0.010 [+0.004, +0.015] | −0.077 | +0.065 | 24.6 s |
| C | −0.0043, +0.0013 | +0.0019 [+0.0018, +0.0020] | −0.064 | −0.030 [−0.030, −0.030] | +0.034 [+0.026, +0.042] | −0.102 | n/a | 26.2 s |
| 0 | +0.0002, +0.0007 | +0.0001 [−0.0003, +0.0006] | −0.075 | −0.035 [−0.035, −0.034] | +0.040 [+0.032, +0.048] | −0.095 | n/a | 6.4 s |

Reading (descriptive, two replicates, bootstrap of two units is a formality, not precision): H2 — the
profile grouping is indistinguishable from a random redistribution in every architecture (|e| ≤ 0.015; the
slope is positive for B, C, ~0 for 0, slightly negative for A): no architecture makes the grouping matter on
this generator. H3 — the sequence-fusion arms beat the summary-fusion arms in every architecture (d < 0), but
γ is POSITIVE everywhere: the advantage is SMALLER with the planted lagged dependence (r = 1) than without,
the opposite of the H3 expectation; the readout control ρ_1 (last position − pooled, averaged over fusions)
is of the same size as d_0 and larger than d_1 in every architecture, so most of the sequence–summary gap
is the readout (last step vs pooling), not preserving history until fusion. The donor sensitivity is
large (Δ ≈ +0.06 to +0.08): with a donor trained for the summary receiver the sequence advantage shrinks by
more than d_1 itself, so the executed pilot's H3 number depended on the donor's receiver. ARCH-0 and ARCH-A
are as good as or better than the TCN on raw MASE at 1/4 and 2/3 of its cost. The test split agrees in sign
and size (`effects_test` in the JSON). None of this is confirmation; it bounds what E1 may expect from the
extractor regime and it says the readout must be a declared factor. DX: under trend + level shifts every
architecture beats the naive (0.500): A 0.404, C 0.413, 0 0.415 within the linear bar (0.412 + 0.03), B 0.451
not ([DX tables](../evidence/d3_k5_20260917/RP14_ARCH_STAGE_DX_TABLES.md)).

Not done, visible: the full factorial (338 cells) needs a new budget; three replicates and all four levels
were not run; the donor sensitivity covers A and B only; ELU is common by decision, not by comparison.

## RP15 — E1 and RL demand

[E1_FAMILIES.json](../../tres_temas_entrevista/program_v3/E1_FAMILIES.json) from the warehouse census
(715 datasets: 198 FINANCIAL pending licence evidence, 513 SYNTHETIC, 4 PUBLIC CC-BY-4.0): electricity load
diagrams (370, 15 min, 4 years) and household power (7, 1 min, 4 years) as DEV; Beijing air quality (144,
hourly) and appliances (28, 10 min) as reserve candidates not opened; cycles of a declared daily period AND
context + horizon blocks reported, adequacy for aperiodic processes by blocks plus the D1 diagnostics; the
source deficit is recorded as engineering, not hidden. [13B](../../tres_temas_entrevista/program_v3/13B_NEXT_STEPS_E1_E3_AND_LANES_2026_09_18.md)
rewritten: config → entry point → plugin → data for six RL configs and two forecasting configs, each
classified (HISTORICAL_CANDIDATE / RESOLVABLE / EXECUTABLE_VERIFIED_BY_INSPECTION), with the facts that no
weekly boundary, funding, latency or capital beyond `initial_cash` exists in code. [13C](../../tres_temas_entrevista/program_v3/13C_E3_WEEKLY_CYCLE_CONTRACT_2026_09_18.md):
the weekly cycle, information at decision time, costs, capital and sizing as an experiment contract with
traced values, declared values and OWNER_DECISION items. Escalated to the owner (one decision): asset
universe, real capital / maximum exposure, retraining-and-release schedule, funding model — with two options
and their effect; everything else proceeds under the declared values.

## RP16 — closure

CPU ledger (systemd accounting of every transient unit of this order on omega:
[RP16_CPU_LEDGER_omega.json](../evidence/d3_k5_20260917/RP16_CPU_LEDGER_omega.json); the workers' children
from their reports: dragon 812.8 s, gamma 212.1 s):

| item | CPU s |
|---|---|
| tests, replays, closures, backfill, reanalysis, mutants, pilots on omega (ledger, 40 units, before the final suite) | 6 936 |
| workers' children (dragon + gamma) | 1 025 |
| final suite and last closures (below) | see closing line |
| ceiling of the order | 14 400 |

Full suite: `SUITE_LINE`. Legacy collection errors in the stale integration tests are the ones AGENTS.md
documents; no test of this order is skipped.

State: [PROJECT_METHOD_STATE.json](../../tres_temas_entrevista/program_v3/PROJECT_METHOD_STATE.json)
(`rp9_rp16_blocks`, tasks MOD-E0-DEV / MOD-E0-CLOSURE / MOD-ARCH-COMPARE EXECUTED, MOD-E1 and
BUSINESS-CONTRACT DESIGNED, stage external_review, `check_plan` PASS). Workers synced to the closing commit.

## Request

One review of RP9–RP16: the closure v2 and its counterexamples, the descriptor reanalysis that justifies the
reuse, the backfill, and the informative ARCH stage with its reading (no winner; readout and donor as
declared factors for E1). Pending on the owner: the 13C business items.
