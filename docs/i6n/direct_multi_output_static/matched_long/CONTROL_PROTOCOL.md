# RAW long control: frozen scope and acceptance

This implements RAW_PRIORITY_20261015.md. It is not R2, a DOIN search or a
claim that window=2*h is optimal. Two fixed R0 recipes answer one missing
question about practical input/readout sampling under the business masks.

| Item | mg12m reference (first admission) | mg1l hourly control |
|---|---|---|
| Context | 144 expected open-market hours | identical |
| Real observations, current t included | 13 | 145 |
| Spacing | 12 expected open-market hours | 1 expected open-market hour |
| Exact elapsed heads | 18,24,36,48,60,72 h | identical |
| Issuance | each supported hourly origin | identical |
| Feature values | same retained 19 RAW, no STLFSI4 | identical |
| Branch | causal Conv1D 4, kernel3, stride1 | identical |
| Fusion/core | feature-axis concat, actual-time encoding; causal Conv1D 8 then4 | identical |
| Heads | each Flatten, Dense8, Dense1 | identical recipe, longer Flatten |

No smoothing, interpolation, input reconstruction or target cleaning. Windows
can cross market closures using observed past inputs and explicit elapsed
offsets/closure flags. Targets cannot cross the operational forced-flat limit
or split/label-availability boundaries. Input sampling never changes t+h.

The hourly readouts have 25344 additional parameters: 132*4*8*6. This is
declared capacity, not an isolated causal attribution of sampling alone.
No prior MG12 fit has this common-population and corrected runtime identity;
the two fresh matched fits answer a new controlled question, not a seed repeat.

TRAIN 2020-2023, last eight weeks INNER with label maturity purging; full
DEVELOPMENT 2024; TEST 2025 never loaded. Intersect valid (origin,horizon) pairs
for FIT/INNER/DEVELOPMENT before fitting. Both arms use the same FIT-only
feature statistics over the union of their common-FIT input-source rows and
the same FIT-only naive loss scale per horizon. Preserve all exclusions.

Static, seed6102026, AdamW .001, weight decay .0001, batch256, max200 epochs,
patience10, min_delta1e-8, restore best. Monitor the same end-of-epoch mean of
FIT/INNER at identical weights. Runtime TF2.21.0/Keras3.13.2 is verified, not
resealed after a fit. No unnecessary repetitions or new hyperparameter sweep.

Select the lower equal-head mean annual relative MAE on frozen DEVELOPMENT
support; ties within1e-8 keep cheaper mg12m. Each head still reports its own
naive, price-unit MAE and (MAE_h-naive_h)/naive_h. Never use a mean to make an
ineligible head eligible. One seed is a recipe decision, not statistical proof.

Finite sequence: seal -> common population -> review/preflight -> two fits
and reports -> paired comparison -> authenticated short/long eligibility ->
strategy if eligible -> outbox/readback -> physical mirrors -> automatic close.
The existing frozen pair policy requires at least one eligible short and two
eligible long heads; it does not require all twelve to beat naive. A technical
failure is not a scientific negative. No TEST/tuning, R2 or new Satoshi work.

## Requirement-to-evidence map

| ID | Required evidence before release |
|---|---|
| LC01 | New arm IDs/schema/mode; old short seals and replay unchanged |
| LC02 | 145/13 slots, actual t included; no >t perturbation changes inputs |
| LC03 | Six exact long elapsed heads, no next-observed-row replacement |
| LC04 | Common masks/scaler and mature labels; DEVELOPMENT perturbation cannot change FIT scaler |
| LC05 | Declared parameter counts/delta, compatible import-free R0 models |
| LC06 | Real GPU fit/report and saved-prediction/manual naive replay agree |
| LC07 | Two-arm decision refuses forged support, mixed arms or missing heads |
| LC08 | Negative pair does not read bars/trade; positive fixture uses existing strategy |
| LC09 | Restart/lease/admission no duplicate fit; failure retained; no busy agent polling |
| LC10 | Per-host RAM/VRAM/disk, one heavy preferred job, no auxiliary parallel GPU |
| LC11 | Every numeric result/disposition/cost in OLAP with readback and two mirrors |
| LC12 | Timer-owned progression/close; honest phase-specific status/ETA |

Engineering tests use synthetic data, not TEST. Real-data preparation is
TRAIN/DEVELOPMENT-only. Failed preflight cannot authorize a fit.
