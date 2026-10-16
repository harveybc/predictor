# Short/Long Retained-Prediction Audit

## Verdict

Retained input/label/business-mask/scaler/metric checks PASS for both selected
recipes. **End-to-end absence of causal leakage is NOT certified.** Seven
source features have unverified availability, two producer records remain
unresolved, and upstream vintages/revisions, assumed calendar and retrospective
development selection remain qualifications. TEST remains closed.

Code `fa83954971299977c1779bb8899c28c188b11122`; result and status in
`DIAGNOSTIC_RESULT.json` / `DIAGNOSTIC_STATUS.json`. No fit, GPU job or TEST
access. Twenty focused tests pass, including seven adversarial temporal and
business mutations; two existing protobuf warnings. Before implementation,
the test import failed; a later fixture aliasing defect was corrected before
the final 20-test pass. These are focused tests, not a full-suite certificate.

## Business And Temporal Checks

Every retained row in FIT, INNER and DEVELOPMENT was checked against raw input
bytes and independently reconstructed rules, not just its stored checksums:

- All windows end exactly at t and include its observed value; no input slot
  is after t, invented or an expected closed-market hour. Slot ranks advance
  by 2/12 expected open-market hours, with actual UTC offsets and closure flags.
- Source features at those exact slots match the retained RAW matrix, and
  targets equal the retained price at exact elapsed t+h, never the next bar.
- Origin >= open+12 h; origin and target < close-12 h. The same head mask
  governs fitting and metrics. Entry also respects the separate entry cutoff.
- Labels stay inside their split and maturity limit. No TEST source bar is
  loaded. FIT-only scaler rows are exactly the union of both matched arms'
  FIT input slots; means/stds independently reproduce. FIT naive loss scales
  reproduce, separately from each DEVELOPMENT horizon's scoring naive.
- R0 and R2 prediction masks, origins and targets equal the frozen population.
  Independently summed price MAE, own-horizon naive and relative error match
  the authenticated report within 1e-12; counts agree exactly.

| Family | FIT origins / labels | INNER origins / labels | DEVELOPMENT origins / labels |
|---|---:|---:|---:|
| Short MG2m | 16,397 / 94,823 | 715 / 4,130 | 4,725 / 27,286 |
| Long MG12m | 12,566 / 50,957 | 546 / 2,226 | 3,424 / 13,864 |

These are union-of-head origins, not a claim every horizon has that many rows.
The audit covers 498,849 input slots and 193,286 supported labels per recipe
population (not new fits). Windows may use observed bars from the operational
buffers; closed hours do not count as observations. Calendar correctness is
conditional on the retained FX_NY_17 assumed calendar, not broker certification.

## What Explains The Discontinuity So Far

R0 boundary-head diagnostics on each head's own support:

| Head | FIT relative error | INNER relative error | DEVELOPMENT relative error | Mean absolute forecast movement / DEVELOPMENT naive MAE |
|---|---:|---:|---:|---:|
| 12 h | -0.088464% | +0.128988% | -0.037392% | 2.155405% |
| 18 h | -13.990395% | +3.395511% | +16.195553% | 56.887681% |

The short forecast stays very close to persistence, not a demonstrated strong
trading signal. The long model makes much larger movements; its FIT advantage
does not carry to INNER or DEVELOPMENT. Movement correlation on DEVELOPMENT
is +0.029046 at 12 h versus -0.030387 at 18 h. This is a generalization-gap
signature, not proof of its cause or a theorem about predictability.

On exactly the same 3,406 DEVELOPMENT origins, R0 is -0.081043% at 12 h and
+16.066649% at 18 h, each against its OWN recomputed horizon naive. The gap
therefore is not solely the different origin population. Horizon, spacing,
context, head objectives, scaler values and fitted weights still differ; no
single-factor attribution follows. R2 exhibits the same qualitative pattern.

The effective configurations have identical training settings, features,
branch/core/head sizes and seed. Only model horizons and market-grid context
differ. Targets are trained as price delta divided by the FIT naive scale,
then converted back to price, not as log1p targets. This audit checks that
price-domain reporting path, not a new cross-device forward replay.

## Next Action

No cadence or capacity campaign was launched. Complete upstream availability
and producer revalidation before calling the short result causally certified.
The proposed small 18 h context/resolution bridge is described in
`DIAGNOSTIC_PLAN.md`; its three new treatments still need sealed execution
contracts and common support. Preserve all old campaigns and scientific
results; do not relabel them as independent holdouts or trading evidence.

Publication: 228 governed reports delivered with exact OLAP readback, zero
pending/mismatch. CPU audit/publication: 11.336 s, cgroup peak 742.8 MiB,
zero swap. Small result/status copies and verification digests are recorded
in `DIAGNOSTIC_MIRRORS.json`; database binaries are not committed.
