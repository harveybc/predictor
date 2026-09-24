# RP156: corrected development signal retained; acceptance still incomplete

Candidate `b3a064d5cdba3fb4dd49087d0516ac4c300cda6b`. Scope: source, published
per-cell arithmetic, exact target-support calculation, and real scorer with
subprocess-result fixtures. No original checkpoint inference, live warehouse,
GPU occupancy or application deployment was independently exercised here.

## Remaining findings

1. **High: the claimed registered design is not authenticated or even compared.**
   `tools/df_ecl_modular.py:978` accepts an arbitrary supplied dictionary or calls
   seal_contrast using the local run's seeds. That is reconstruction, not a read
   of the registered design. The two design digests are reported but never checked.
   Changing the supplied design digest alone still yields COMPLETE. Declaring an
   empty factorial yields COMPLETE with 0 expected/0 scored and an empty summary.
   Require the retained accepted design/registration, validate its canonical
   contents and run binding, and refuse empty/altered populations before dispatch.
   Merely comparing caller-controlled digest strings is not the complete repair.

2. **High: complete cells can still contain incomplete or foreign scoring evidence.**
   At `:1002-1005`, completeness is count plus a truthy checkpoint flag; `:1022`
   and `:1028` silently omit missing populations/reductions. Removing the
   label-disjoint population from one R2 child returns COMPLETE and an R2 average
   over two seeds, against three in the other regimes. Returning horizon 192 in
   that child, or NaN in its MAE, also returns COMPLETE. Bind every returned child
   to its expected cell/seed/regime/horizon/population/reduction and require typed,
   finite metrics before ANY accepted aggregate. Diagnostics may remain available.

The published nine real rows all include the expected subset, horizon 96 and
finite metrics; their recorded design digests also agree. These counterexamples
show acceptance gaps, NOT that the reported scores have those injected defects.

## Repairs accepted at the tested scope

- The actual support function returns first disjoint origin 735, 1,802 windows,
  and 95 overlapping origins after the monitor. An independent set-of-target-rows
  check agrees. Legitimate input-context overlap is not being prohibited.
- A missing expected cell now returns INCOMPLETE_EVIDENCE with no regime summary.
- A child with identity=false now suppresses the regime summary. An independently
  altered temporary checkpoint is refused before the model framework is touched.
- Source and records now name author_float32 and independent_float64 separately.
  The scorer calls the existing author reducer. This review did not re-reduce the
  original arrays or recertify that reducer; its earlier acceptance remains scoped.

## Experimental reading

Recomputed from the published per-cell author-float32 metrics, not fresh inference:

| Regime | MAE mean | SD, three seeds | Same-row persistence MAE |
|---|---:|---:|---:|
| R0 | 0.371173809 | 0.001533646 | 0.868283093 |
| R1 | 0.374583503 | 0.000733318 | 0.868283093 |
| R2 | 0.368596435 | 0.000648887 | 0.868283093 |

R2 minus R0: -0.001711756, -0.001701564, -0.004318804. Mean change
**-0.002577374 MAE, 0.694385% relative improvement**. R1 is worse in all three
pairs. The directional development result survives removal of shared target
labels. Preserve it; do not discard or retrain because acceptance code needs work.
It does not decide H1, establish statistical independence, or imply trading profit.
The matching TimeFilter score on these origins remains undelivered; do not insert
its published test number into a validation comparison. The previous qualifications
about literal test-byte access, total post-fit costs and fresh-inference versus
prediction-parity evidence remain open, not silently accepted by this review.

Probe and results: [CPU probe](../evidence/RP156_MUSASHI/probe.py),
[observations](../evidence/RP156_MUSASHI/results.json). Eight isolated parent-scorer
cases, target-support oracle and pre-model checkpoint refusal were executed. The
child transport is a double; this is not nine new model replays or a whole suite.
Existing CPU tests `test_df_ecl_contrast_gates.py -k 'F2 or F5'`: 9 passed,
6 deselected; no training/data-dependent tests were run for this review.

## Execute now, in parallel

1. **Do not launch another training or routine nine-checkpoint inference loop.**
   First freeze these remaining failures and repair pure acceptance/aggregation.
   Revalidate the retained RP156 per-cell results against the accepted design and
   their actual bindings, then publish an additive closure. Re-infer only if a
   specific missing artifact/identity makes reuse invalid, under the existing
   allocation and with a measured reason. Do not regenerate an authority record
   from local scores. Preserve all earlier reports and their corrected scope.
2. Keep the existing positive cases AND test empty/foreign/re-digested designs,
   missing population/reduction, wrong seed/regime/horizon, duplicate cell, false
   or non-boolean identity and nonfinite metric. Expected outcomes must be defined
   before editing; no new scorer correction is complete with only its own happy
   path. Close over a fully enumerated contract, not a growing list of flags.
3. **RP147 is still executable and already assigned. Start its real admission,
   isolated pinned SDK/checkpoint setup and bounded pilot now**, independently
   of those CPU acceptance edits. Continue direct SDK -> M5PHET -> news-signal
   integration. Use the external 5090 first when eligible and the existing
   600 CPU s / 900 wall s smoke allowance after setup; do not evict running work.
   The next application update must report the actual attempted admission/setup/
   inference result, not another unexplained NOT STARTED. If a real dependency
   fails, report the exact measured object and continue the other assigned lane.
4. Delegate calendar/collector and broker interface tests to independent worktrees
   where useful. Do not wait for calibrated news probabilities to implement queues,
   timestamps, as-of joins and broker refusals. Paper/demo canaries still require
   the existing account/risk mandate; no real-money activation is authorized.
5. Finance remains gated for fitting on missing temporal evidence, not gated for
   producer investigation, alternative-source assessment or reference preparation.
   Nobody is to invent availability fields. State missing evidence, not a generic
   demand that the owner authorize a scientific fact.
6. Reconcile the full cost ledger and the still-pending matched reference. Keep
   the existing artifact policy: identify disposable scoring scratch explicitly,
   keep it isolated from retained files, and do not treat score_cell's unconditional
   unlink at :962 as independent acceptance of permanent scientific-artifact deletion.
   This is a policy/interface check, not a demand to archive every prediction.

All other details remain in the
[RP155 execution continuation](../../handoffs/SATOSHI_RP155_REVIEW_AND_EXECUTION_2026_09_24.md)
and [active assignment index](../../handoffs/SATOSHI_ACTIVE_ASSIGNMENTS_2026_09_24.md).
No new budget, model redesign, owner decision or global hold. A/B are not rerun.
Satoshi owns implementation/execution. Application progress and closure fixes are
parallel obligations, not successive permission rounds.
