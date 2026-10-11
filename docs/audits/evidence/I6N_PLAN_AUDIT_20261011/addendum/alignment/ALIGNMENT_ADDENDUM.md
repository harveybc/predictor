# I6-N plan audit: addendum, alignment lane (owner point 5; I6N-AUD-06, -09, -16, -17)

Run date: 2026-10-11. This addendum does not modify the original dictamen (commit 4d3af09d).
The verdict stays REVISE_BEFORE_IMPLEMENTATION. No model was trained. No market data was read:
no TRAIN, VALIDATION or TEST rows, and nothing dated 2025 or later. The work ran on CPU only.

## Identified inputs

The audited object is a worktree of branch `codex/i6d-weekly-walk-forward-20261009` at HEAD
`269c2d132b21a55e59cd6faa8242b1ac74f0f7ae`, carrying an uncommitted draft. Below it is called `$AUDITED`.
Every code file the script imported came from `$AUDITED`, and its git blob equals the blob at HEAD
269c2d13 (`out.json` -> `loaded_modules`, cross-checked with `git rev-parse 269c2d13:<path>`):

| File (repo-relative) | sha256 | git blob |
|---|---|---|
| predictor_plugins/modular_temporal/components.py | 0d834e626722b76f949704b2d3f3c13fa5292d2631f8f764fa432911cf712272 | c91cb4ec |
| predictor_plugins/modular_temporal/common.py | c08aba769378faf61a7c304a720d2c35bd0099bf2010f10435401d63424b368b | 0cb8ec91 |
| predictor_plugins/modular_temporal/layers.py | 2ce7d4f394baf77fd0316bd2024e9179a84704ae7b2ee8a9e4c84c9a31be36a8 | c364403b |
| tools/i6e_masked_model.py | 7e401085b4a4111b805495f20bc3e13575a93fe111ac5c0c570156f4e6cd2519 | 6d09ee28 |
| tools/i6m_model.py | cc7fdf99ae1156d737a9fb5a4dd14f929849e662f701216b6382011a4d86e900 | 6ec61eed |
| tools/fs4_temporal_predictor.py | 2e35d9851be54b0a466efcfbd7e43a289fe97e44cbb9fd87038b8d14b295a896 | 1d966df7 |

Read but not imported:

| File | sha256 | Use |
|---|---|---|
| `$AUDITED`/docs/tres_temas_entrevista/program_v3/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md | d4341e72f4fc05531cfcffd5036c1968d522a105fa534d2add0ae2d10eb56487 | route text: lines 36-37, 76-101, 115-130 |
| `$AUDITED`/tools/i6e_clock_windows.py | d5d53f2d3dbe2212cb86fc476a822fa481c112bc1096f7089c8a49514e45ce01 | placeholder convention: lines 40-48 |
| `$AUDITED`/predictor_plugins/modular_temporal/assembly.py | 6b1601a2ec8cd59c5eb3954539716760479841382b9c44f5ff147a4b891d936c | existing random-weight probe: lines 33-61 |
| ../../probes/cde/out.json (original audit) | 42e7cdcbe4eea2b9636cc098901db03148b8ab26d9c8099f501857db18517b0b | original synthetic results, recompared |
| ../../probes/cde/check.py (original audit) | ea521a732ab4a3cbe949fd2cfd4b20d4570d7075631e98340f80476de415f637 | original configuration list |

Environment: Python 3.12.7, TensorFlow 2.18.0, Keras 3.9.0, NumPy 1.26.4, 0 visible GPUs.

## Command

Run from this directory, on the coordinator (CPU):

```bash
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 15m -n i6n-add-alignment -- \
  python3 alignment_known_weights.py --repo "$AUDITED" --original-out ../../probes/cde/out.json > out.json
```

- `out.json` is exactly the script's stdout. Nothing was appended or edited by hand.
- stderr carried only TensorFlow/CUDA plugin-registration warnings and was not retained.
- The script sets `sys.dont_write_bytecode`, so it writes nothing into `$AUDITED`.

| Output | sha256 |
|---|---|
| alignment_known_weights.py | 67c44afacf24a7ae591e88b43f9bdfeedff1b60b9ea97631bad680ba8317a21e |
| out.json | cdd25c09de2f0b01b176808d494159c751eac868f64e2ca33ede0ce631c8665b |

A second, independent run produced the same out.json sha256, so the output is deterministic.
`routes_detail_digest` = 918a2a35cc9cec0f41ce6ec98cf5e3c84d9152d030ada4eb686e863e582987a8.

## Instrument

**Known weights.** Single-channel convolution stacks use all-ones kernels, zero biases and linear
activations. Impulses are fed at every position. A position belongs to the receptive field (RF) of the
last, head-read output exactly when that output is nonzero. Inputs and weights are nonnegative, so
nothing cancels. Every measured set is printed beside an independent index-arithmetic expectation.
Result: all 2,349 route configurations match their expectation (`routes_all_analytic_match: true`).

**Graphs with LayerNorm.** Constant (all-ones) or rank-1 kernels make every channel equal, and
LayerNorm then outputs beta = 0. Such a graph is degenerate: nothing can move the head. The retained
`*_ones` entries show this, and their positive controls fail (`instrument_valid: false`). Conclusions
for these graphs therefore use a closed-form, full-rank assignment instead: kernel.flat[f] =
sin(0.9 + 1.618 f)/sqrt(fan_in), with biases and beta at 0, gamma at 1, and attention query/key kernels
at 0, which makes attention uniform over the causal window. A result counts only when its positive
control moves the output.

**Controls.**

- Positive: a stride-1 causal k=3 convolution reads t.
- Negative: a convolution after `Cropping1D` of the last slot does not read t.
- Weights versus wiring (owner point 5b): with a one-hot kernel on the oldest tap of a causal k=3
  convolution, the measured RF is {2}, which excludes t although t is wired. The union over one-hot
  taps, {2,3,4}, equals the all-ones RF.
- `instrument_valid: true`.

**Original synthetic probe recompared.** Of the 84 random-weight configurations in probes/cde/out.json,
84 agree with both the known-weight and the analytic result. Its stride conclusions were therefore not
weight artifacts. They describe Keras padding conventions applied to synthetic layers, not the
concrete route.

## Concrete routes and grid

**Grid.** n = 1 + L/delta_x, with L in {1,2,4,6}*h and delta_x in {h/k: k in {1,2,3,4,8,9}, h/k an integer} ∪ {1}.

- N1 anchor: n = 5 for both h=4 (L=8, delta_x=2) and h=72 (L=144, delta_x=36).
- N2 distinct n for h=4: {2,3,5,7,9,13,17,25}.
- N2 distinct n for h=72: {2,3,4,5,7,9,10,13,17,19,25,33,37,49,55,73,145,289,433}.
- Odd n dominate. Even n occur only at h=4: 2; h=72: 2, 4, 10.

**Inherited code that could be reused.** The proposal fixes no padding for reduction.

| Stage | Candidate code | Convention | Result on the grid (out.json) |
|---|---|---|---|
| Entry and core, no reduction (N1 both families, N2 "ninguna") | tools/i6m_model.py:19-46; tools/i6e_masked_model.py:69-103 | Conv1D k=3, stride 1, `padding="causal"`; head reads `[:, -1]` | t always read: i6m RF = all 0..n-1 (trig weights; n=5 at delta_x 2 and 36, n=55 at delta_x 8). The i6m factory refuses n=2, 289 and 433 (`3 <= samples <= 241`, i6m_model.py:14-17). |
| Conv1D-only N1 (proposal:85) | built as written | two causal k=3, stride 1 | Reads t. RF = 5 samples, so at n=5 all of it is read; at n=55 only the last 5 (50 unread); at n=433 only the last 5 (428 unread). |
| Factor-2 reducer | components.py:317-341 `_residual_temporal_stage` | `valid`, kernel = stride = factor | Builds without error on odd n and silently drops t: n=5 -> RF {0..3}, n=55 -> {48..53}, n=433 -> {426..431}. Even n (4, 10) reads t. |
| Same, through its caller | components.py:227-297 `transformer_conv` | exact divisibility | Odd n refused: `ValueError: Temporal reduction requires exact divisibility`, raised at common.py:47 (`_partition`), before the components.py:294 guard. It is t-preserving whenever it accepts n. |
| Strided causal reducer | tools/fs4_temporal_predictor.py:253-297 `GroupedCausalConv1D` | causal left pad; outputs at s-1, 2s-1, ... | n=24, 12, 6 and the FS4 chain 24->12->6 read t. n=5, 9, 55 do not (RF {1,2,3}, {5,6,7}, {51,52,53}). Its docstring (lines 257-258), "the LAST output always covers the newest input row", is false for odd lengths. |
| Not inherited, candidate choices | Keras `causal` k=2/3 stride 2 | left pad k-1 | Drops t whenever a stride-2 stage receives an even length. |
| | Keras `same` k=2/3 stride 2 | TF SAME | 0 losses on every grid n, stage count and rule. |
| | origin-anchored (left zero-pad to a multiple of 2, valid k = s = 2) | | 0 losses on every grid n, stage count and rule. |

**Schedules** (proposal:128-130):

- "progresiva": every stage reduces by 2 while the rule allows.
- "solo al final": only the last stage reduces.
- "ninguna": all stages have stride 1.
- 1-3 stages.
- Two readings of "solo si quedan al menos dos posiciones": `out>=2` (the reduced output keeps at least 2 positions) and `in>=2` (the input has at least 2 positions).

t is lost in the following counts of (n, stages) configurations, out of 24 for h=4 and 57 for h=72:

| Convention | Rule | Progressive h4 | Progressive h72 | Final only h4 | Final only h72 |
|---|---|---|---|---|---|
| keras_causal (k=2 or k=3) | out>=2 | 3 | 14 | 0 | 6 |
| keras_causal (k=2 or k=3) | in>=2 | 9 | 20 | 3 | 9 |
| keras_valid k=2 (= `_residual_temporal_stage`, factor 2, unguarded) | out>=2 | 18 | 47 | 18 | 45 |
| keras_valid k=2 (= `_residual_temporal_stage`, factor 2, unguarded) | in>=2 | 21 | 50 | 21 | 48 |
| fs4_grouped k=3 (inherited class) | out>=2 | 18 | 47 | 18 | 45 |
| fs4_grouped k=3 (inherited class) | in>=2 | 21 | 50 | 21 | 48 |
| keras_same k=2/3 | both | 0 | 0 | 0 | 0 |
| origin_anchored k=2 | both | 0 | 0 | 0 | 0 |
| none (stride 1) | | 0 | 0 | | |

## Position encoding (I6N-AUD-16)

ClockView (i6e_masked_model.py:40-47) places slot i at i*delta_x hours from the window start, so t
sits at L hours.

- t's 32-dim code depends only on L: it is identical across delta_x 1, 8 and 36 at fixed L
  (`t_code_depends_only_on_L: true`).
- It differs across L = 8, 144 and 432 (`t_code_distinct_across_L: true`).
- The ordinal code in layers.py:79-95 depends on n instead. At delta_x = 1 it equals ClockView.

Bands above Nyquist (per-sample phase step > pi), out of 32:

| delta_x | Above Nyquist | Wrap a full period (> 2 pi) |
|---|---|---|
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 4 | 2 | 0 |
| 7 | 4 | 2 |
| 8 | 4 | 2 |
| 36 | 10 | 8 |

## LSTM readout (I6N-AUD-17)

- With identical trig weights, LSTM(return_sequences=True)[:, -1] and return_sequences=False differ by
  a maximum of |diff| = 0.0.
- The last-state RF covers all 9 positions.
- A gated zero step and a Keras-masked (skipped) step differ by 0.00417. A zero placeholder therefore
  advances the LSTM state unless a mask is propagated.

## Missing slots on the inherited route (I6N-AUD-09)

Setup: i6m, n=5, 2 features, delta_x=36, slots 1 and 2 absent. Placeholder 0.0 is compared with 7.5,
written into every value and source channel.

| Route | Placeholder perturbation, max \|diff\| | Positive control (present slot +5) | Instrument valid |
|---|---|---|---|
| i6m full route, trig | 0.0 | 0.0633 | yes |
| Conv-only route built from the same inherited pieces (clean, branch causal_conv1d, gate, core convs; no attention), trig | 0.0 | 0.0749 | yes |
| Either route, all-ones | 0.0 | 0.0 | no (degenerate under LayerNorm) |
| Negative control: ungated causal Conv1D on raw input | 120.0 | | |

## Claims

Classes: DEMONSTRATED_DEFECT, RISK, OWNER_REQUIREMENT, METHODOLOGICAL_ALTERNATIVE, RETRACTED.

| ID | Claim | Class | Disposition | Minimal correction |
|---|---|---|---|---|
| AL-01 (AUD-06) | The proposal requires t coverage (proposal:36, 128-129) but fixes no padding, rounding or reading of "al menos dos posiciones" for reduction. | OWNER_REQUIREMENT | CONFIRMED | Seal one reduction convention and one reading of the rule. Make the known-weight RF table in this script an N0 acceptance test over every grid n and stage count. |
| AL-02 (AUD-06) | Keras `causal` stride-2 loses t on even stage inputs, and `valid`/phase-(s-1) conventions on odd inputs. On the N2 grid this affects 0-50 of 57 h=72 configurations, depending on the choice. | RISK | QUALIFIED: conditional on the padding choice. N1 and "ninguna" routes never lose t. | Choose `same` or origin-anchored padding (0 losses), or any convention that passes AL-01's test. |
| AL-03 (AUD-06) | The inherited `GroupedCausalConv1D` (fs4_temporal_predictor.py:253-297) does not cover the newest row for odd lengths, which contradicts its docstring. | DEMONSTRATED_DEFECT | CONFIRMED in inherited code. Latent: FS4 used only even lengths 24/12. Becomes live only if the class is reused on the N2 grid. | Do not reuse it for odd n, or fix its phase to anchor at the last row, and correct the docstring. |
| AL-04 (AUD-06) | "Existing `_residual_temporal_stage` raises on odd n (components.py:294-295)." | RETRACTED | The stage builds on odd n and silently drops t. Its caller `transformer_conv` refuses odd n at common.py:47, before line 294. | Do not call the stage outside `transformer_conv`'s divisibility guard. Corrected wording: the stage drops t on odd n; the caller refuses. |
| AL-05 (AUD-06) | The remediation "left-pad masked, then valid k = s = factor" is the fix. | METHODOLOGICAL_ALTERNATIVE | QUALIFIED: Keras `same` also gives 0 losses. | Either is acceptable once sealed and tested. |
| AL-06 (AUD-06 acceptance) | The original acceptance test, "perturbing the last slot changes the last reduced output", and the inherited `probe_alignment` (assembly.py:33-61) use random weights, so a no-change result can come from the weights (one-hot tap-0 control). | METHODOLOGICAL_ALTERNATIVE | QUALIFIED | Use known weights: all-ones or one-hot taps for linear conv stacks, and full-rank closed-form weights with a passing positive control for LayerNorm graphs. |
| AL-07 (new, support) | The Conv1D-only route without reduction reads only the last 1+2*(layers) samples: 5 of 55, or 5 of 433. Declared history L is not reached. | RISK | NEW. N1 (n=5) is unaffected. | Each route's declared temporal support (proposal:97) must state its RF. Reject or flag configurations whose RF is shorter than n. |
| AL-08 (AUD-16) | ClockView t-code varies with L and is invariant to delta_x at fixed L; the ordinal code varies with n. | RISK | CONFIRMED as fact. It is not a defect inside one fitted model, where the offset is constant. It matters only when weights move across L (proposal:97 "pesos importables", N4 transfer, R0-R2 donors). | Specify the encoding. If weights are imported across L, require identical L or an origin-anchored code. |
| AL-09 (AUD-16) | "Origin-anchored lag-hour encoding, with t invariant to L, is required." | METHODOLOGICAL_ALTERNATIVE | QUALIFIED. The owner asked for physical position (proposal:79, 99), and ClockView already uses hours. | Optional. Required only under AL-08's cross-L import condition. |
| AL-10 (AUD-16) | "Aliases for delta_x >= 7 h." | RETRACTED (wording) | Corrected: Nyquist is exceeded from delta_x = 4 (2 bands), and a full period wraps from delta_x = 7 (2 bands); at delta_x = 36, 10 bands exceed Nyquist and 8 wrap. Low bands stay monotone. No harm was measured. | Report as a property, not a defect. |
| AL-11 (AUD-16 order) | The Transformer/Conv1D order is contradictory between proposal:86 and :119. | RISK | QUALIFIED. N1 order is specified (proposal:89-92: 1x1 Conv1D -> Transformer -> two Conv1D). :119 lists N2 entry options. The open points are whether that entry Conv1D is temporal or 1x1, and how blocks = 0 is handled. | Specify the N2 entry Conv1D kernel and blocks = 0 handling. |
| AL-12 (AUD-17) | return_sequences=True followed by a last-state head is exactly equivalent to return_sequences=False (known weights, 0.0). | METHODOLOGICAL_ALTERNATIVE | QUALIFIED. Inert only given the specified head (proposal:92). It is not a gene, so no duplicate identity arises. | Either add a declared readout gene or state that the head reads the last state. Identity test only if it becomes a gene. |
| AL-13 (AUD-09) | The proposal does not state the missing-slot rule. Only "sin suavizado/interpolacion" (proposal:37) is given. | OWNER_REQUIREMENT | CONFIRMED (specification gap) | Declare that every new route reuses the inherited presence-channel, clean and gate convention (i6e_clock_windows.py:40-48; i6m_model.py:19-46), or an equivalent. |
| AL-14 (AUD-09) | "Conv1D reads zeros" implies placeholder values leak. | RETRACTED (as a leak) | On the inherited route, placeholder values never reach the output (0.0 with valid positive controls; the negative control leaks 120.0). Absence is encoded deliberately as gated zeros plus a source mask. | No change to the inherited route. Keep the retained test as acceptance. |
| AL-15 (AUD-09) | An LSTM adapter integrates zero placeholders. | RISK | QUALIFIED. No inherited LSTM exists. A gated zero step changes the state against a masked skip (0.00417). | For the N2 LSTM adapter, specify mask propagation or accept zero-step semantics explicitly. Add it to the placeholder test. |

The early-stopping criterion and the repetition counts are outside this lane and were not touched.
