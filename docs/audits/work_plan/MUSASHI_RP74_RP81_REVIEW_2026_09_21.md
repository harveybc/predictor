# Musashi review: RP74-RP81

Reviewed revision: `bd6fcf4`. Disposition: **CHANGES_REQUIRED; MEASUREMENTS_PRESERVED**.
Next orders: [RP82-RP89](../../handoffs/MUSASHI_RP82_RP89_2026_09_21.md).

## Findings, ordered by consequence

1. **High: the production block closure bypasses accepted artifact custody.**
   `tools/df_e1_block.py:1080` still implements its own checks rather than consuming
   the strict table verifier. On a private copy, replacing predictions with truth
   and updating the local record changes MAE_z from 0.539779 to 0 while `close()`
   remains verified. Original accepted payloads and receipts are unchanged. The
   strict table rejects this same copy. The financial closure at
   `tools/df_fin_runner.py:539` also remains a separate implementation; that
   observation is code inspection, not an executed financial closure attack.

2. **High: the normalized metric's denominator is not anchored.**
   `tools/df_closure_table.py:108` reads the local prepared scaler, but output
   artifact verification does not establish its identity. Multiplying only
   `DATA.npz`'s scaler_sd by ten changes a strictly verified MAE_z from
   **0.539779 to 0.053978**, with no problem reported and predictions, record,
   accepted payloads and receipts unchanged. Verifying predictions alone is not
   verifying the task, transformation, fit population or normalized metric.

3. **High: a reference can be renamed into a comparator that never trained.**
   `tools/df_benchmark_contract.py:147` checks a design digest's form, not its
   derivation and accepted identity. Renaming the reference arm to
   `UNTRAINED_REFERENCE`, without recomputing the old design digest, still yields
   `VERIFIED_COMPARATOR`. Bind the actual design, registered population and each
   cell's arm, seed and task; a correct forecast does not establish its label.

4. **High, before financial selection: the runtime mixes the declared populations.**
   `tools/df_fin_runner.py:466` pools candidates by loss, ignoring population and
   optimizer. With all paired seeds present, it selects an A default and a D
   delta arm, comparing **10 MAE against 16 Huber candidates**. This is not the
   v3 plan's B-only, equal LR search within each loss/optimizer combination.
   A repaired JSON registry has not repaired the executed scientific comparison.

5. **High, before financial inference: the interval and estimate use different weeks.**
   `tools/df_fin_runner.py:437`, with `[-100, NaN, 1, ..., 1]` (ten ones) and
   block_len=2, reports mean **-8.181818** over eleven observed weeks but interval
   **[1, 1]**, `RESAMPLED`. The isolated negative week cannot enter any complete
   block. Missingness must not silently change the resampled estimand. In addition,
   the selector calls block_len=2 whereas the reported dependent coverage exercises
   other block lengths. Coverage >=0.80 in a test does not establish nominal 0.95
   coverage. No financial campaign was run, so this does not invalidate financial
   measurements that do not yet exist.

6. **Medium, ML interpretation: tier comparisons confound patience with training environment.**
   The old and new shared arms have exactly equal training prefixes for seed 1.
   All eight seed-2/3 prefixes differ before stopping could act (maximum
   validation difference 0.034576 MAE_z); one GRU initialization also differs.
   Those seeds moved to workers. Do not attribute the aggregate tier change
   solely to patience or identify CPU hardware as the cause. Within the new
   block, each seed's arms share its host: retain those blocked contrasts.
   Across-seed SD here also includes host effects, not just initialization noise.

## Independent checks and what remains useful

Evidence: [counterexamples](../evidence/RP81_MUSASHI_REVIEW_2026_09_21/results.json),
[reproducer](../evidence/RP81_MUSASHI_REVIEW_2026_09_21/reproduce.py),
[fresh-process replays](../evidence/RP81_MUSASHI_REVIEW_2026_09_21/replay.json),
[replay runner](../evidence/RP81_MUSASHI_REVIEW_2026_09_21/replay.py).

All mutations were on disposable copies. Counterexamples used a warehouse
stand-in serving the fixed published accepted payloads, not a mutable new anchor.
There was **no fresh live-warehouse query**, training, service operation or final
holdout read in this review. These findings do not show that retained results
were actually altered.

For all **15 new cells**, independently checked arrays/weights/records against
published terminal artifact hashes, origin identities, generator truth, naive,
and scores. Rebuilt the actual model and reloaded each checkpoint in its own new
process. **15/15 pass** the pre-existing `allclose(atol=1e-6, rtol=1e-6)` rule.
Maximum absolute prediction difference is 1.366554e-6 in native units; maximum
MAE_z difference is 1.649473e-9. The combined relative/absolute rule, not an
absolute-only 1e-6 bound, is what passed. This establishes finite checkpoint
inference reproducibility, not identical cross-host training or a universal
resolution guarantee for future marginal financial improvements.

| Same DEV task and common train sigma | MAE_z | Naive MAE_z | Skill vs naive |
|---|---:|---:|---:|
| Modular, original inputs | 0.539087 | 0.676560 | 0.2032 |
| GRU adapted, original inputs | 0.528993 | 0.676560 | 0.2181 |
| Modular + calendar | 0.492654 | 0.676560 | 0.2718 |
| GRU + calendar | 0.479838 | 0.676560 | 0.2908 |

UCI235, minute target, h60, W60, same 10,020 DEV origins; sigma_train =
0.9125164391265214. Three seed/host blocks, seven of fifteen budget-censored.
Calendar improves both architectures in all three paired blocks; GRU is better
in all three pairs with either input set. This is useful development evidence
about information and architecture on this task, not financial utility or a
general winner. The interaction has mixed signs. The adapted GRU is our matched
re-execution derived from Gasparin, not an exact paper reproduction; the paper's
different task remains NOT_COMPARABLE. No published number substitutes for it.

Existing focal tests: **79 passed, 9 deselected** (benchmark, closure table,
financial acceptance excluding FL01/02/03/05/08). They pass despite the new
counterexamples. Full suite and service-stack tests were not rerun by Musashi.
Historical rows outside these fifteen retain their previously declared scope.

## Disposition

Preserve measurements and append corrected verification; no blanket restart.
Repair the real callers before new scientific selection. Continue independent
design and bounded diagnostic work under RP82-RP89, with no new owner approval
needed within that scope. Financial Huber/MAE and Adam/AdamW remains mandatory;
electricity does not select the financial recipe. Weekly RL and the modular
extractor/core hypotheses retain their separate work-plan dependencies.
