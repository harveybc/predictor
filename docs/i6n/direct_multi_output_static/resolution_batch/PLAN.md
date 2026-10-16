# Resolution and h72 batch

Owner directive: 2026-10-16. This replaces the next-experiment hold after the
closed context bridge, not its immutable design or results.

## Frozen Questions And Order

1. Compare 13 observed input ticks at spacing 2 versus 4 OPEN-MARKET hours,
   separately for direct multi-output short (2,3,4,5,6,12 elapsed hours) and
   long (18,24,36,48,60,72 elapsed hours). Spans are 24 versus 48 open hours,
   not equal-time windows. Every hourly origin includes its current tick.
2. Select long-family input spacing by the equal-head mean DEVELOPMENT
   `(MAE_h-naive_h)/naive_h`, negative better; ties within 1e-8 retain DS2.
   This is development selection, not an independent generalization claim.
3. At that spacing, fit single-output h72 with 6,9,13,18,24 ticks, in ascending
   window order. At each length compare an affine multivariate linear model,
   an explicitly adapted DLinear (kernel 3), and the existing minimal Conv1D.
   The 13-tick Conv1D versus the chosen DMO h72 isolates sharing of the model
   across heads on the same h72 rows and fixed scaler. Other windows also
   change parameter count, which must be reported, never hidden.

Exactly four DMO fits plus fifteen h72 fits; one seed 6102026, no repeats,
no DOIN, R2, autoencoder, retraining, feature reselection, strategy or TEST.
Linear controls test lower complexity; no premise that longer horizons need
larger networks. DLinear adaptation is not a reproduction of its paper.
Reference: https://github.com/cure-lab/LTSF-Linear (AAAI 2023).
The moving average is internal to the optional DLinear model's observed
window; it neither smooths the source data nor alters targets. RAW Conv1D
and linear arms never smooth. All forecast inputs are <= origin.

## Business And Evidence

Reuse the selected 19 RAW features, versioned assumed FX calendar, 12-hour
post-open/pre-close operational buffers, entry policy, masks and label purge.
TRAIN 2020-2023, last eight weeks INNER; all supported 2024 DEVELOPMENT;
TEST 2025 remains closed. No closed-market synthetic observations.
Short arms share intersection support; long arms share support across all
planned spacings/windows so DMO and single-head h72 compare identical pairs.
FIT input scaling uses the deduplicated union of actual FIT source bars;
per-horizon loss scales use only FIT naive. DEVELOPMENT never stops epochs.
Early stopping reuses train.fit_dmo: equal-head masked normalized MAE,
M=0.5*(FIT+INNER), patience 10, absolute min_delta 1e-8, maximum 200 epochs,
restore lowest monitored epoch. AdamW .001/.0001, batch 256.

## Automation And Tests

Sequential fits only on the 5090; no shared-RAM parallel 5070 Ti job.
One finite systemd GPU timer owns the next pending fit, one bounded CPU
timer owns status, synchronization, OLAP exact readback and physical mirrors.
Existing admission tool governs CPU/RAM before actual execution. A completed
cell is authenticated and never retrained to recover delivery. Interrupted
fits are typed and are not silently repeated. Admission refusals retry before
a fit, never with a lower cap. No agent monitoring loop is part of execution.
Status at batch and cell levels includes heartbeat, terminal counts and
measured-cost ETA; initial estimates are explicitly provisional.

Required tests before release: configuration/order bounds; conditional spacing
selection; equal support/scales; original-scale per-head naive; origin and
target alignment; no-future and split purge; linear affine behavior; DLinear
decomposition; serialization/reload; early-stop reuse; corrupted terminals;
restart/adoption; publication retry without training; resource wait and owned
timer shutdown. Real population and a bounded GPU engineering check precede
scientific deployment. Data/model binaries remain private; scripts, small
results and digest manifests are versioned.
