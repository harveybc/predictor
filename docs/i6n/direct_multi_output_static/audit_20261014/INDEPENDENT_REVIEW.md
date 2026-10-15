# Independent review of the I6-DMO corrected candidate, 2026-10-14

Reviewer: independent (wrote none of the candidate). Mandate: release-sequence
step 2 of `docs/handoffs/MUSASHI_TO_SATOSHI_I6DMO_AUDIT_CORRECTIONS_20261014.md`,
plus the four verification points in the final section of
`docs/handoffs/MUSASHI_TO_SATOSHI_DMO10_NUMERICAL_REPLAY_20261014.md`
(sha256 `cba02e7e8efb2ebba5a1dbce887c32fa72715da40b775279aee96a971e1ed2f9`).

**Reviewed pin:** `satoshi/i6dmo-static-20261014` @
`0bcf7f6b71ab93aa33959c82cc75da5fd8d3f33c`, recorded at review start.
While this review ran, the branch advanced to `a33cfe70` (commits
`d11a76ad..a33cfe70`, among them "sealed protocol, proven scope, decision
impact, bounded isolation" and a NUMERICALLY_EQUIVALENT closure round). Those
commits are noted, NOT reviewed; every statement below is about `0bcf7f6b`.

Constraints honored: CPU only (`CUDA_VISIBLE_DEVICES=""`), every execution one
at a time under `crispdm-run -q -m 3G -t 600s -n i6dmo-review`, no scientific
fit, no TEST (2025+) rows, no services, no credentials, no DOIN changes,
retained artifacts read-only, bounded synthetic fixtures only. The candidate
branch was not edited; this review lives on its own branch.

## Verdict

**FIX_REQUIRED** — the data/final/strategy/ops corrections hold under every
control I ran, but the pin does NOT implement the revised DMO10
numerical-replay protocol the release conditions on. Blocking items:

1. **(a) No sealed numerical-equivalence protocol identity.** The pin's
   `MONITOR_TOLERANCE` (tools/i6dmo/train.py) is a relative-only
   `1e-5` per partial sum with a prose justification. Absent: the revised
   rtol `5e-5` / atol `1e-9`, the combined formula
   `abs(actual-reference) <= atol + rtol*abs(reference)` with a declared
   reference direction, scope, dtype, flags and runtime version sealed into a
   versioned protocol identity, and the `NUMERICALLY_EQUIVALENT` label
   (replay_monitor reports `TOLERATED`). Probe RP04 control I.
2. **(b) CPU replay is NOT strictly exact.** Tolerance activation
   (train.py:508) keys only on `TF_CUDNN_USE_AUTOTUNE == "0"`, and
   tools/i6dmo/model.py sets that flag at import on EVERY process, GPU or not.
   Executed counterexample (RP04 control B): a pure-CPU replay TOLERATED a
   5e-6 relative partial-sum divergence. The claimed GPU/cross-process scope is
   not enforced anywhere.
3. **(c) Monitor forward-tensor divergence is tolerated as pass/fail
   leniency.** Executed counterexample (RP04 control G): a doctored
   `batch0_predictions_hex` row with identical sums was TOLERATED
   (`_compare_witness`, train.py:640-652). The handoff forbids extending the
   tolerance to forward tensors.
4. **DMO10 itself is an unresolved FAIL declared a "library floor".** The
   pin's own GPU_REPLAY.json records `MONITOR_WITNESS_DIVERGENCE` on the
   retained witness with per-row relative deltas up to `1.211e-4` — above the
   current 1e-5 bound AND above the proposed 5e-5. The "declared library
   floor" is a declaration after excluding hypotheses, not the controlled
   isolation the handoff requires; the handoff explicitly rejects that
   inference. PASS, numeric equivalence and unresolved FAIL remain distinct;
   the pin is at unresolved FAIL.

The later, unreviewed commits (`95c4c714`, `a33cfe70`) claim to address items
1-3; if so, release-sequence step 2 must be repeated on that tip for the DMO10
scope before release. Everything NON-DMO10 below was verified at `0bcf7f6b`
and needs no repetition unless those files changed.

## 1. Audit probe replay (mandate step 1)

In-tree `audit_20261014/probes.py` at the pin, CPU, governed scope — all three
**NOT_REPRODUCED** (exit 1 = "all reproduced" is false):

| Probe | Result | Evidence |
|---|---|---|
| AUD-DMO-01 | NOT_REPRODUCED | `sampler_rule=DS2_EXACT`, `current_feature_staleness_hours=0.0` |
| AUD-DMO-02 | NOT_REPRODUCED | selected final TRAIN `[2021-01-01, 2025-01-01)` = required |
| AUD-DMO-03 | NOT_REPRODUCED | typed `FINAL_BLOCKED_GATE:TEST_READINESS_MISSING`, no placeholder |

**Probe-seam adjudication.** The candidate MODIFIED probes.py: AUD-DMO-03 now
probes `unit_runner.hook_module("final_test")` instead of
`strategy_adapter.run_unit`. I replayed the ORIGINAL audit probe bytes
(`probes_original_fdaae9d2.py`, sha256
`dfa94836650857f9485e805d54464cc3ca7ea055faaa54810c9901269ad9b0c6`) on the
pin: AUD-DMO-01/02 NOT_REPRODUCED, AUD-DMO-03 still REPRODUCED against
`strategy_adapter.run_unit` directly. I verified this is legitimate: at the
audited `fdaae9d2`, `unit_runner.HOOKS["final_test"]` was
`tools.i6dmo.strategy_adapter`; at the pin it is `tools.i6dmo.final`, the
worker resolves hooks only through `unit_runner.hook_module`
(tools/i6dmo/worker.py:611), and `tools.i6dmo.final.run_unit` is a real
guarded implementation (gates, FRED clocks, readiness-bound TEST bundle,
frozen-weights predictions, typed refusals). The retained refusal in
strategy_adapter is fail-closed defense for a stale registration.
Non-blocking: its disposition string still reads `FINAL_TEST_HOOK_NOT_PREPARED`,
which is now a misnomer; rename suggested.

## 2. Candidate suite and independent controls (mandate step 2)

Full candidate suite at the pin: `pytest tests/i6dmo` — **279 passed, 276 s**,
CPU, governed scope. My independent probes (all under `review_probes/`, each a
bounded synthetic execution; JSON verdicts quoted from their output):

- **RP01 alignment — PASS (6/6 controls).** Table-wide exactness: 6,961,948
  selected slot values, 0 not at their exact bar. A dropped bar is a COUNTED
  exclusion (`INPUT_SLOT_NOT_OBSERVED`), a NaN exact value is
  `INPUT_VALUE_NOT_FINITE`; on the same fixture the ASOF arm selects an older
  bar while the default selects nothing, and `asof_slot_index` is bound to the
  exact indexer. Load-time authentication: an ASOF-rule manifest and a
  tampered rule string both fail typed `SAMPLING_RULE_MISMATCH`.
- **RP02 naive pairing — PASS (3/3).** Paired metrics AGREE on identical row
  ids in original price (C2 vs manual vs disk replay). Reviewer mutations:
  one mutated row id → typed `NaiveReplayMismatch` ("origin or exact target
  bar absent"); retained prices rescaled by 1.0001 → typed
  `NaiveReplayMismatch` (naive C2 vs disk replay disagree).
- **RP03 masked loss — PASS (4/4).** Behind the mask, NaN vs 1e6 labels give
  a bit-identical loss AND bit-identical gradients on every trainable tensor
  (no contribution). A NaN smuggled past the mask: training loss goes
  non-finite (caught by `NONFINITE_LOSS`), `masked_split_metrics` and
  `load_data_bundle` both fail typed `NONFINITE_LABEL`.
- **RP04 DMO10 protocol — defects above, plus requirements that DO hold:**
  same-process CPU replay bitwise (A); over-bound 2e-5 refused (C); the
  retained witness magnitudes 1.8e-5 and 1.211e-4 KEEP their FAIL under the
  current bound (D); zero-vs-finite sum refused, though no zero/near-zero
  handling is declared (E); same-process corruption typed
  `MONITOR_SELF_CHECK_FAILED` (F); a record without determinism flags gets no
  tolerance, so old flagless receipts cannot inherit it (H); exported
  reload/serving equality exact in the named CPU context — in-session vs
  canonical rebuild vs full Keras load, zero divergent pairs (K).
- **RP05 decision impact — point (d).** Training knobs untouched in the sealed
  design (`max_epochs=200, patience=10, min_delta=1e-8`, rule
  `i6dmo.mean_monitor_stop.v1`); `MONITOR_TOLERANCE` is referenced by
  tools/i6dmo/train.py ONLY — nothing in naive pairing, eligibility, app/,
  olap/ or optimizer_plugins/. Near-tie negative control, reported honestly:
  with perturbations of 1.8e-5 and 1.211e-4 relative (the measured
  uncertainties), **best_epoch flips (3→2) and stop_epoch flips (13→12)** on
  a near-tie history. `min_delta=1e-8` is 3-4 orders of magnitude below the
  measured replay uncertainty, so checkpoint selection is NOT invariant
  across contexts and this review claims no such invariance; within one
  process the monitor is bit-stable, so each recorded selection is
  well-defined per run.
- **RP06 policy / final / traceability — PASS (5/5).** Execution-policy
  receipt seals deterministically from the versioned instrument config and
  the ACTUAL pinned imported policy (`min_valid_long_heads=2`, simulation
  assumptions stated, `periods_per_year=260`, costs `NOT_BROKER_VERIFIED`);
  one tampered byte → typed `EXECUTION_POLICY_TAMPERED`; no
  `STRATEGY_OWNER_RULINGS`/`WAIT_OWNER` gate remains and EXECUTION_POLICY is
  auto-produced. Bundle identity: superseded prefix → `SUPERSEDED_BUNDLE`,
  both ASOF diagnostic prefixes → `ASOF_DIAGNOSTIC_ARM`, ASOF rule →
  `BUNDLE_SAMPLING`, exact pin accepted. TRACEABILITY.json: `broken=[]`;
  spot-checked DMO01/02/10/12/20 — all code, evidence and test pointers
  resolve at the pin.
- **FINAL_REFIT through the real CLI on synthetic dates:** candidate test
  `test_final_refit_real_cli_trains_on_the_final_window_only` plus the rest of
  tests/i6dmo/test_dmo_final.py rerun in isolation — 17 passed. The typed
  final design fixes TRAIN `[2021-01-01, 2025-01-01)`, removes outer
  validation, refuses an undecided plan field, and `unit_runner` refuses a
  mismatched declared window (`FINAL_WINDOW_MISMATCH`).

## 3. Non-blocking observations

- `authenticate_sampling` verifies the sealed RULE NAME and the campaign's
  pinned bundle identities, not a load-time re-derivation of bundle arrays
  from retained input; a forged manifest claiming DS2_EXACT over filled
  arrays would pass load-time authentication. Mitigated by the pinned
  `bundle_sha256` allowlist/refusals; worth closing when convenient.
- The declared tolerance names no zero/near-zero handling; the relative-only
  bound happens to fail closed there (RP04 E).
- `strategy_adapter.run_unit` final_test disposition string misnomer (above).

## 4. Missing coverage (honest)

- **Everything GPU-only.** No GPU execution was permitted: I could not replay
  the GPU witness divergence, the GPU save/load identity, the TOLERATED path
  on real GPU records, or anything on the preferred/secondary GPU hosts.
  GPU_REPLAY.json was consumed as retained evidence, not reproduced.
- **Retained campaign artifacts.** The private campaign root (first fit's
  TRAINING.json/AUDIT_HOLD, real bundles) is not locally readable here; the
  near-tie control is synthetic and the real fit's monitored-gap profile was
  not checked.
- **Warehouse/publication.** No credentials, no services: the outbox
  LIMIT/readback fix is verified only by the candidate's tests
  (test_ops_dmo12_outbox, test_audit_warehouse_contract in the 279), not by a
  live readback.
- **FRED publication clocks** remain unresolved; I verified the guard refuses
  TEST without them, not the clocks.
- **Release step 1** (per-commit PRE-fail/POST-pass) was not re-executed
  commit by commit; the suite at the pin passes in full.
- **Release step 3** (fleet DOIN pin/timer confirmation) is outside this
  step-2 review; the pin's diff contains no optimizer_plugins or
  doin-data-predictor changes (also found by the prior audit).
- The post-pin commits `d11a76ad..a33cfe70` are unreviewed.

## 5. Reproduction

From a checkout of the reviewed pin, CPU, each command one at a time under the
governed runner:

```bash
CUDA_VISIBLE_DEVICES='' python docs/i6n/direct_multi_output_static/audit_20261014/probes.py
CUDA_VISIBLE_DEVICES='' python docs/i6n/direct_multi_output_static/audit_20261014/probes_original_fdaae9d2.py
CUDA_VISIBLE_DEVICES='' python -m pytest tests/i6dmo -q
for p in rp01_alignment rp02_naive_pairing rp03_masked_loss rp04_dmo10_protocol rp06_policy_final_traceability; do
  CUDA_VISIBLE_DEVICES='' python docs/i6n/direct_multi_output_static/audit_20261014/review_probes/$p.py <workdir>
done
CUDA_VISIBLE_DEVICES='' python docs/i6n/direct_multi_output_static/audit_20261014/review_probes/rp05_decision_impact.py
```

RP04 exits 0 when its observations (including the three confirmed defects)
reproduce; the defects are blocking regardless of that exit code.

---

# Addendum: a33cfe70

Repeat of the step-2 DMO10 scope (blocking items a-d) against the integration
tip `a33cfe70`, same constraints (CPU only, governed runner, no fit, no TEST,
retained GPU artifacts read as evidence only). The `0bcf7f6b..a33cfe70` diff
touches ONLY DMO10-scope files (train.py, replay_monitor.py, design.py,
dmo10_isolation.py, their tests, MODEL_RETURN/TRACEABILITY/GPU_REPLAY/
INTERFACE_MODEL and the handoff copy), so every non-DMO10 verdict of the main
review carries over unchanged.

## Verdict: ACCEPT_FOR_RELEASE (DMO10 scope at a33cfe70)

All four blocking items are closed, each verified by execution
(`review_probes/rp07_dmo10_v2_addendum.py`, 12/12 controls, exit 0; RP05
re-run green; test_dmo_train + test_dmo_model + test_dmo10_isolation: 40
passed on CPU):

- **(a)** `floor_protocol_identity()` seals `i6dmo.monitor_numeric_equivalence.v2`:
  the formula `abs(actual - reference) <= atol + rtol*abs(reference)` with
  reference = recorded fit-time value, rtol 5e-5 / atol 1e-9, scope, dtype
  float64, label NUMERICALLY_EQUIVALENT, required flags and runtime versions
  under a protocol sha256. A foreign identity is typed
  `MONITOR_PROTOCOL_MISMATCH`; a pre-v2 receipt (no protocol, no recording
  identity) KEEPS its FAIL at both retained magnitudes (1.82e-5, 1.211e-4 →
  `MONITOR_WITNESS_DIVERGENCE`), even though 1.82e-5 would sit inside the new
  floor — no inheritance.
- **(b)** Scope is PROVEN, not flag-inferred: executed outcomes — CPU-recorded
  record refused strict; same-process record refused strict; same-process
  corruption typed `MONITOR_SELF_CHECK_FAILED` (self-check additionally
  demands report status EXACT); GPU-recorded record replayed without a GPU is
  the separately typed `CROSS_DEVICE_NOT_COMPARABLE` (never graded equal or
  defective); only GPU-recorded + different process + GPU visible activates
  the floor (simulated in memory: NUMERICALLY_EQUIVALENT within floor;
  1.211e-4 still refused in scope). Boundary inclusive, atol covers near-zero,
  over-bound refused.
- **(c)** Forward tensors are recorded DIAGNOSTICS only: a diverging batch-0
  hex row with equal sums yields status EXACT for the sums comparison plus a
  `row_forward_diagnostics` report, never leniency and never a bitwise claim —
  replay_monitor labels it `EXACT_SUMS_WITH_FORWARD_DIAGNOSTICS`. Exported
  reload/serving equality re-verified exact in the named CPU context.
- **(d)** Training knobs untouched (`min_delta=1e-8, patience=10,
  max_epochs=200`); `MONITOR_TOLERANCE` still referenced only by train.py.
  `stopping_decision_impact` is recorded per unit with the note "never claims
  cross-context decision invariance"; executed: on a near-tie history it
  reports `could_change_best_epoch=True` — my main-review finding is carried
  faithfully, not contradicted. The GPU closure's `false/false` applies to
  that one actual history only.

## Adjudications

- **Retained GPU evidence** (GPU_REPLAY.json @ code 95c4c714, read-only):
  closure NUMERICALLY_EQUIVALENT with max partial-sum relative delta 4.597e-8
  (~1000x inside the floor); old witness FAIL kept; pytest selection
  deliberately red on GPU so it is never read as plain PASS. Honest.
- **Isolation verdict** DOES_NOT_SUPPORT_AUTOTUNE_HYPOTHESIS: consistent with
  the tool's verdict logic (flags-on/off fresh GPU children byte-identical ⇒
  the flag cannot be the mechanism) and honestly refutes the author's own
  earlier hypothesis. I could not re-run it on GPU; on this host it would be
  NOT_INFORMATIVE_ON_CPU by design.
- **Round 16 wording in MODEL_RETURN**: supported — it leads with the
  retraction of the round-15 "library floor" OVERCLAIM, keeps autotune a
  hypothesis, keeps the old FAIL, and claims no PASS.
- **Stale-path explanation** ("the earlier monitor fixes were not reaching the
  GPU path until this revision", GPU_REPLAY/TRACEABILITY note): NOT
  demonstrated by the retained evidence. The 0bcf7f6b→95c4c714 diff changes no
  forward-execution code (model.py untouched; train.py changes are
  protocol/scope/reporting; the end-to-end test already spawned flags-first
  subprocesses). The evidence shows the measurement moved (1.8e-5 → ≤4.6e-8);
  WHY it moved remains unestablished. Stated as fact, this is the same class
  of overclaim Round 16 retracted — reword as hypothesis. Non-blocking: the
  acceptance label does not rest on the explanation.
- **Wording inconsistency, non-blocking:** the sealed design's
  `monitor_comparison_floor.justification` still says "library floor ...
  (observed max 1.82e-5)" while Round 16 retracts that framing and the
  closure note implies 1.82e-5 was never a floor. Align the texts in the next
  design revision note; the bound itself was pre-approved prospectively.

## Addendum missing coverage

GPU-side: the closure replay, the isolation flags-on/off GPU comparison and
CROSS_DEVICE behavior on real GPU records were not reproduced here (CPU-only);
retained artifacts were consumed as evidence. The gpu_cross_process branch was
exercised via an in-memory `_gpu_visible` stub, not real hardware. The private
campaign root remains unreadable locally.
