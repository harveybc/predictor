# External review of RP82-RP89

Inspected revision: `3389a25`. Disposition: CHANGES_REQUIRED.
New owner priority: SOTA-first; see RP90-RP97. This review does not authorize
another simplified-model or financial fit. Findings remain open unless fixed
and re-tested; the new work-plan policy alone does not repair runtime code.

## Executed findings

Evidence: [results.json](../evidence/RP89_MUSASHI_REVIEW_2026_09_21/results.json)
and [reproducer](../evidence/RP89_MUSASHI_REVIEW_2026_09_21/reproduce.py).
Faults were injected only in private copies or fabricated data, not live evidence.

1. High: actual block closure accepts a corrupt checkpoint when cached replay
   evidence remains. Replacing weights with `not a checkpoint` leaves verified
   true and claims the same checkpoint bytes. Hash the consumed file before
   cache reuse and bind the complete replay identity, not a claimed digest.
2. High: financial closure does not anchor the fold metadata that defines its
   required population. Marking a fold insufficient in FIN_DATA.json and deleting
   its arrays reduces four verified units to two with no problem reported.
   Derive required folds from accepted preparation, not mutable local metadata.
3. High: a cost-pilot schema refusal can be reported COMPLETED with exit zero;
   pending outbox also failed to affect the demonstrated exit. Acquisition
   failures must retain campaign identity and close without fabricated delivery.
4. High before any financial fit: the inspected HistData parser treats a fixed
   EST timestamp as UTC and drops volume, while the pilot expects OHLCV. A
   synthetic 00:00 source label becomes 00:00 UTC rather than 05:00 UTC under the
   [source specification](https://www.histdata.com/f-a-q/data-files-detailed-specification/).
   This demonstrates current parser behavior, NOT the historical producer of
   every retained file. Establish producer lineage before correcting real data;
   do not invent volume, observed arrival time or a zero availability lag merely
   to make the resource downloadable. Installing metadata alone is insufficient.

## Independent retained-measurement check

[Fresh-process replay](../evidence/RP89_MUSASHI_REVIEW_2026_09_21/replay.json):
six retained daily-context cells reproduced within the previously declared
combined atol/rtol 1e-6, with accepted preparation and artifact hashes checked.
Maximum absolute native prediction difference was about 1.414e-6; maximum MAE_z
change about 1.242e-9. This supports the narrow retained measurements, not the
soundness of the bypassed verifier or any new model-selection recommendation.
No fitting, no new financial bar values and no live warehouse re-query by this
audit. The owner now scopes these results as HISTORICAL_DEV_ONLY.

Focal verification: 84 passed, 10 deselected in benchmark/closure/financial
acceptance tests. This is not a full-suite pass; the executed counterexamples
above still expose defects despite those green tests. No live repair deployed.
