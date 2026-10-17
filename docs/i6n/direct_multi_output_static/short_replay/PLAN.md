# Original Short Experiment Replay

Owner clarification 2026-10-16 supersedes the unlaunched short-only ANY_HORIZON
prototype. Reproduce the promising original short experiment, not another
stopping variant: horizons 2/3/4/5/6/12h, original mean FIT/INNER patience 10,
delta 1e-8, max 200, seed 6102026, 19 RAW features, 13 MG2 observations including
t, unchanged normalization, branch/core/head recipe and original time encoding.

Source starts directly at historical `24f5a6cc`. The ENTIRE model.py and
train.py files and every original receipt code pin must match exactly.
Call the original train.train_unit entrypoint on the authenticated original
source bundle and original sealed design. Do not use the modified 12-output
fit wrapper, AnyHorizonMonitorStopper, imported learned weights or a newer
continuation architecture. No extra model-building/forward before the fit.

Preparation independently checks that all short input values, labels, masks,
cohorts and scales project exactly from the completed 12-head mean-stop
control and match the original short bundle. The scientific run consumes
the copied ORIGINAL bundle, not the independently rebuilt projection.
Original source artifacts are not overwritten. One new R0 fit, no retries.

Use the idle secondary 4090 as requested for this investigation. The historical
run used the preferred 5090; report this hardware difference and do not promise
cross-device bitwise predictions. Actual TF2.21/Keras3.13.2 match the original
observed runtime; the original seal mismatch remains a disclosed limitation.
This is DEVELOPMENT 2024; TEST stays closed and the preferred GPU continuation
is untouched. Publish each horizon's MAE and own paired naive, verify warehouse
readback and two physical copies, then close the owned timers automatically.

Acceptance tests before deployment: reject any non-horizon model/config/data
change; exact file-level historical trainer/model proof; six-output model
shape and original initial-forward identity; guarded masks; only original
trainer entrypoint called. The pre-clarification prototype completed 27 CPU
checks but no scientific fit. It is not deployed by this order.
