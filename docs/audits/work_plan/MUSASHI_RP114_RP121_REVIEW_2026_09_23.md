# RP114-RP121 independent review

Reviewed commit: `316c67af0af317c345c6959681135ba622770584`.
Disposition: **CHANGES_REQUIRED; preserve measurements; no benchmark refitting**.
The return's claim that only review and cooling remain is not accepted.

## Executed findings

### 1. High: plausible wrong estimator values still pass independent acceptance

`tools/df_sota_repro.py:2365` and `:2394`, in `accept_catalog`, do not compare
the catalog ACF or quantiles against independent values. ACF is checked only
against [-1, 1]; quantiles only for monotonicity. The independent ACF at `:2220`
also uses a different population: concatenated steps from sampled windows,
rather than consecutive origins at each fixed forecast step. Nullable metrics
are assigned a comparison difference of zero when either side is missing.

Executed through the real closure and catalog acceptance with `independent=True`,
using the actual author's tiny CPU fixture and a mutated catalog PRODUCER:

- Original first-lag ACF values about 0.907/0.943/0.933/0.937 became zero.
- All residual quantiles became zero (original median about 0.15).
- Correlation became null although the independent value was -0.1978629665.
- Acceptance still returned `pass: true`, no refusals, and correlation difference
  0.0. The valid control also passed.

The required condition is equality under each declared estimator's definition,
or a justified undefined/approximate state, not membership in its numeric domain.
This invalidates the blanket phrase "every estimator family agrees ... to 0.0".
It does NOT establish that a production catalog contains these wrong values.
It does not invalidate MSE/MAE merely because ACF acceptance is inadequate.

### 2. High: acceptance kind comes from the mutable registry, not the terminal

`tools/df_sota_repro.py:2898`, `accepted_artifact`, verifies COMPLETED status,
terminal digest and artifact digest, but reads the requested kind from the local
registry. It does not bind the accepted terminal's kind, subject, design or
artifact role to the requested closure/catalog/regeneration role.

Executed with the warehouse stub held immutable: a report is an accepted
`diagnostic_attachment` of kind `diagnostic`, subject `unrelated-subject`.
`expect_kind="closure"` initially refuses. Changing ONLY the local registry to
kind `closure`, role `closure_report`, subject `closure` makes it return
`accepted: true`. The warehouse payload is byte-for-byte unchanged.

A stored artifact is not necessarily an acceptance for the particular cell and
purpose being checked. Require that relationship from the authoritative terminal
and reject contradictory local hints. This is a provenance/type-boundary issue;
no claim is made that actual production evidence was relabeled this way.

### 3. High: the destructive API retains an acceptance bypass

`tools/df_sota_repro.py:1769` and `:1859` expose `require_acceptance=False`.
The public API forwards it directly to the deletion preflight. This contradicts
RP116's explicit requirement for mandatory prerequisites through API AND CLI.

Executed on a disposable array with valid local closure/catalog/backup but an
empty warehouse response: default call REFUSED and kept the file; the same call
with `require_acceptance=False` returned COMPLETE and removed it. The CLI's
default being safe does not close the public API. Tests should supply a faithful
accepted-chain fixture, not turn off a production deletion precondition.

## What is retained and what remains qualified

The prior wrong-report, fabricated-regeneration, bare-deletion and float32
counterexamples now have specific repairs and regression coverage. The retained
scorer correction and dated successor acceptances are useful progress. Do not
restart training or discard recorded scores because of these new findings.

The return reports official normalized MSE/MAE means:

| T | published | reproduced, reported | scope |
|---|---|---|---|
| 96 | 0.133 / 0.230 | 0.13550 / 0.23288 | measured, replay unverified; not pooled |
| 192 | 0.154 / 0.248 | 0.157645 / 0.252163 | reported operational agreement |
| 336 | 0.162 / 0.261 | 0.164480 / 0.262985 | reported operational agreement |
| 720 | 0.184 / 0.284 | 0.190226 / 0.290617 | reported operational agreement |

These are not fresh benchmark measurements by this review. Its scope is executed
software counterexamples, source inspection and the focused regression suite,
not another full production-array reduction or a new live warehouse query.
Operational agreement is not exact equality or statistical equivalence.

The return itself says only six cells had independent estimators recomputed
from real arrays (three H96 plus one of each other horizon); six had only domain
and internal checks. Even the former six do not have full numerical ACF/quantile
acceptance under the implementation audited here. Keep metric-level scopes.
Complete the finite catalog checks; do not hide that work behind the cooling hold.

T96 remains measured, with a failed cross-device pointwise criterion preserved.
The projected 66 seconds on WORKER_A addresses a replay step, not all acceptance
requirements, and does not retroactively measure the original training UUID.
Only the owner can confirm restored physical cooling; no elapsed date releases it.

## Correction to Musashi's previous review

**Erratum, 2026-09-23:** my RP113 review and RP120 instruction incorrectly used
patch length 24 for protocol A. The pinned `scripts/ECL.sh` uses **32**:
96/32 gives three patches per channel; 512/128 gives four. Satoshi's correction
is right. Tokens are 963 versus 1284, not four patches in both protocols. The
loader population correction remains right. The old text is preserved with a
dated erratum beside it. No experiment is invalidated by this documentary error.

The B pilot is resource evidence, not a completed B experiment or an exact
reproduction of unknown Table-9 lookback choices. No B factorial is ordered here.

## Evidence and next action

Executable probe and full numerical output:
[probe.py](../evidence/RP121_MUSASHI_REVIEW_2026_09_23/probe.py),
[PROBE_RESULTS.json](../evidence/RP121_MUSASHI_REVIEW_2026_09_23/PROBE_RESULTS.json).
The probe used a tiny actual-author CPU model, warehouse stub and disposable
files only, on WORKER_B with one thread and a 3 GiB / 240 s hard scope. Sampled
host temperatures before/after were at most 51.8/59.0 C; these are endpoints,
not a continuous peak measurement. No GPU, production unlink, service restart,
production database mutation or large coordinator computation by this reviewer.

Focused suite execution is recorded in
[VERIFICATION.md](../evidence/RP121_MUSASHI_REVIEW_2026_09_23/VERIFICATION.md).
Next orders: [RP122-RP127](../../handoffs/MUSASHI_SOTA_RP122_RP127_2026_09_23.md).
No additional engineering permission is requested from the owner.
