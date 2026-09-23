# RP128-RP131 independent review

Reviewed commit: `c9dce617a0badc480dc0519e8b94234c555f40f5`.
Disposition: **CHANGES_REQUIRED**, limited to certificate derivation and the
new revalidation report. Preserve numerical results and successful comparisons;
no retraining or blanket regeneration to repair these consumers. This is NOT a
global experimental hold: the existing published L512 recipe is authorized in
the independent RP135 lane, with original scientific and resource safeguards.

## Findings

### 1. High: certificate class is still derived from summaries, not fields

`tools/df_sota_repro.py:2265`, particularly `:2277`, checks family names and the
summary lists `unchecked`, `disagreements`, `families_complete`. It never reads
`independent_comparison.fields` or verifies each family's actual field coverage.
The deletion consumer therefore trusts a new summary-derived class rather than
the complete comparison RP128 required.

Executed on disposable actual-author CPU fixtures through the real deletion API,
with report digests/accepted fixture terminals and backup bytes consistent:

- Remove ALL field results, preserving the old summary. The certificate is
  FULL_INDEPENDENT_NUMERIC with zero recorded fields; deletion returns COMPLETE.
- Leave the full 43-field map but set `global.mae` to DISAGREEMENT, discrepancy
  1.0 against tolerance 1e-9, while retaining the old summary. The certificate
  remains FULL_INDEPENDENT_NUMERIC; deletion returns COMPLETE.
- The valid positive control has the full certificate and passes dry-run.

Both counterexamples remove the disposable array. These are internally
inconsistent accepted reports, not a claim that local hash edits can impersonate
the warehouse. The contract explicitly requires the destructive consumer to
validate the report's scientific content, not merely accepted storage of it.
Derive required fields and tolerances from the bound inventory, check actual
field results and coverage, and refuse contradictory/stale summaries.

### 2. Medium: revalidation overstates current acceptance and deletion readiness

`tools/df_sota_repro.py:2838` selects numerical acceptance by class alone,
regardless of whether the evidence is accepted by the warehouse. At `:2847`
the field named `cell_verified_by_current_closure` is copied from the earlier
catalog acceptance, not the current closure. Its deletion-readiness label at
`:2848` is consequently not the full current gate the docstring describes.

Executed with the actual `revalidate_acceptances` function:

- An empty warehouse response gives `accepted: false` in the nested details,
  but still lists the cell in `with_full_numeric_acceptance`.
- Change the current report's row to unverified/REPLAY_PENDING while preserving
  its earlier accepted catalog certificate. Revalidation reports current-cell
  verification true and deletion eligible true; the actual `deletion_gate`
  correctly returns false for that same current report.

The second case is a false readiness report, NOT an additional deletion bypass:
the actual deletion gate protects the array. A local numeric comparison may be
described as such without custody, but not promoted to currently accepted
evidence. A metadata-only revalidation may be a preview, not an assertion that
all destructive preconditions, backup, current custody and readers were checked.

## Progress preserved

The earlier missing-reference, foreign/missing-design and boolean examples now
have concrete repairs. Separating numerical catalog certification from cell
replay is the correct distinction; retain it. The new field-level defect does
not invalidate those numerical calculations or imply incorrect real MSE/MAE.

The submitted REVALIDATION snapshot lists eleven full numeric sources with
accepted-chain details and H192 s2021 without a full source, and no eligible
deletion. This review does not establish that those eleven real field maps are
inconsistent; the counterexamples demonstrate the checker could accept such
inconsistency. Recheck retained evidence, not arrays, after the fix.

Reported official normalized MSE/MAE means remain H192 0.157645/0.252163,
H336 0.164480/0.262985 and H720 0.190226/0.290617. These are previously measured
values, not fresh measurements by Musashi. Operational agreement remains a
predeclared margin, not exact equality or statistical equivalence.

The physical holds remain: H96 original-device replay on WORKER_A and H192 s2021
original-device inference on COORDINATOR require explicit cooling confirmation
and fresh admission. Do not repeat the failed 5090 inference, widen tolerances
or guarantee that original-device execution will be bit-identical. H96 arrays
remain retained; no verified four-horizon aggregate yet.

## Evidence and next action

[Probe](../evidence/RP131_MUSASHI_REVIEW_2026_09_23/probe.py),
[executed output](../evidence/RP131_MUSASHI_REVIEW_2026_09_23/PROBE_RESULTS.json),
and [verification scope](../evidence/RP131_MUSASHI_REVIEW_2026_09_23/VERIFICATION.md).
All probes use disposable files, a tiny actual-author CPU fixture and a stub
warehouse. WORKER_B, one thread, 3 GiB / 240 s hard scope. Endpoint host readings
were at most 50.5 C before and 62.8 C after the new probe, not a measured peak.
No GPU, real prediction deletion, service restart, new production measurement
or live warehouse query by this reviewer.

Next: [RP132-RP135](../../handoffs/MUSASHI_SOTA_RP132_RP135_2026_09_23.md).
RP132-RP134 are finite repairs to two evidence consumers, capped at 1,800 CPU
seconds and not prerequisites for RP135. Retention stays non-destructive.
RP135 executes the already mapped published L512 recipe on the admitted external
5090, within the standing total budget and without assuming future deletions.

## Orchestration correction after the owner's intervention

The preceding repair-only draft would have repeated a program-level mistake:
letting a defect in deletion authorization stop independent experiments. The
last rounds improved tooling, not the doctoral hypotheses. There is no missing
owner engineering permission. Original-device thermal holds remain local to
those devices and their historical replays. Neither hold blocks new eligible
external-5090 work. An actual new-run data/scoring/admission failure still does.
Protocol B is reference-building work already in the plan, not a new toy test,
and not an isolated context effect or an exact Table 9 claim.
