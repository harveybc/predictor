# Exact short-recipe extension to all horizons

Owner order 2026-10-16. This bounded diagnostic runs on the available secondary
GPU while the independent 407-cell continuation remains untouched.

## Frozen Contract

AH01: start from the actual best historical MG2 short fit at `24f5a6cc`, not
the new continuation model. Keep its 19 ordered RAW features, 13 observed
market-time slots spaced by 2 open-market hours including t, original
physical-time encoder (including its sinusoidal term), branch Conv1D(4,3),
core Conv1D(8,3)->Conv1D(4,3), and Flatten->Dense(8,relu)->Dense(1) per head.
Only append [18,24,36,48,60,72] after [2,3,4,5,6,12]. Parameters rise from
4,882 to 7,480 solely because six identical head structures are appended.

AH02: authenticate the original short bundle, model config, scales, training
receipt and source files. Preserve every original FIT/INNER/VALIDATION row,
input value, offset, closure flag, short label and short mask. Add long targets
only where the existing C1 calendar/business/split eligibility permits the
exact observed t+h. No re-intersection with a long or alternative-window cohort.
Do not read TEST. Preserve original input scales and six target scales exactly;
new target scales are each horizon's own naive on the original FIT rows.

AH03: keep seed 6102026, AdamW lr .001/decay .0001, batch 256, max 200 epochs,
patience 10, delta 1e-8 and the original mean FIT/INNER stopping procedure.
This is an explicit historical-recipe exception to the continuation's newer
TRAIN/VALIDATION protocol, not a change to the running continuation. Reuse the
original fit as control; do not retrain it. No import of learned weights: R0.

AH04: original shared/short initial tensors must match the extended model,
and initial short forwards must agree bitwise on the same hardware. Reuse the
original model builder and trainer, not another reimplementation. Same actual
TF 2.21.0/Keras 3.13.2 as the historical receipt; the old seal discrepancy
remains explicitly disclosed. The owner requested the secondary 4090, so
cross-device bitwise equivalence of scientific outcomes is not claimed.

AH05: one admitted GPU fit, no automatic rerun after an interrupted STARTED.
Keep immutable terminals, periodic machine-readable status, durable outbox,
exact warehouse readback and two physical copies. Finalize automatically.
No DOIN, feature selection, denoising, pretraining or live strategy changes.

AH06: report model MAE, its own horizon naive, N and relative error
(MAE_h-naive_h)/naive_h at every horizon in original USD/EUR units. Add a
12h/18h common-origin diagnostic, without replacing the horizon-specific
full business supports. All results remain DEVELOPMENT 2024, TEST closed.

## Test And Execution Sequence

1. Pure mutation tests: changed input/short label/short mask/scales rejected;
   missing/future/cross-split long labels remain masked, no row is dropped.
2. CPU model test: only horizon list differs; shared/short initial tensors and
   forwards equal, 7,480 parameters, original structural functions unchanged.
3. Authenticate real data and original retained receipt; prepare frozen data.
4. Deploy immutable source and data to secondary, health/admission gate, one fit.
5. Automatic sync, per-horizon report, OLAP/readback, mirror, close. A failure
   is a typed terminal, never an excuse to modify the recipe or repeat a fit.

The added heads necessarily change the multi-task loss and shared gradients;
the old short scores are not guaranteed to survive. A gap between horizons is
not itself proof of a code defect or impossibility. This experiment tests the
specific shared-representation explanation without asserting either outcome.
