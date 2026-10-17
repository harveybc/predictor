# MG2m Extension Launch

## Subsequent Completion And Single-Head Repair

The original graph closed with four completed fits, one pre-fit failure,
five exact report readbacks and 66 authenticated files per mirror. The +18h
fit took 168.681 s, 39 epochs, selected29: final relative error for 18h
+15.172428%, versus -0.238899% at the predeclared diagnostic epoch2.
The diagnostic is retained separately and is not the selected final result.

The conditional single18 failed before any optimizer update because the
adapter used heads instead of the actual head_models bundle field. The
separate one-cell successor is pinned to a433f4e8. It reuses the same prepared
arrays, scalers and untrained +18h head initialization; no completed fit is
repeated. Its real GPU preflight and epoch7/371 updates were observed.

Single-output Keras calls also return a tensor where the old multi-output
code expected a list. The repair normalizes only that output container in
prediction, monitoring and the training concat; existing list/tuple outputs
pass through unchanged. Exact source-diff tests reject any other change to
the trainer. Loss, optimizer, monitor formula and stopping parameters remain
unchanged. The original graph stays on its original frozen runtime, including
the CPU follower. The failed evidence is not overwritten or reclassified.
Fifteen focused CPU tests now pass for the study; the best-result registry has
27 focused tests including fixed-epoch exclusion from selected champions.

Original launch receipt follows; generated STATUS remains authoritative.

Runtime 3f126cf82fca842ffa30a56ba6d8d049283011e8, campaign
i6dmo-mg2m-extension-20261017. The +18h scientific fit was observed at epoch2,
130 updates, on the secondary 4090. The preferred 5090 remains on its existing
continuation; no job was stopped and no auxiliary shared-RAM slot was enabled.

The actual GPU preflight passed: original initial-weight digest reproduced,
all shared and original-head tensors equal, original short initial forward
bitwise equal, 5315 parameters for plus18. The only structural-name exceptions
are Keras-generated InputLayer/Concatenate UIDs, independently witnessed.
Layer configs, units, kernels, weights and forward equality remain enforced.
The legacy trainer is byte-identical to 796ec8f4; its builders/encoder are
AST-identical except the declared-family validator/catalogue extension.

The original matched population and its union-of-arms FIT-only scaler were
recovered, not replaced with a fresh single-arm scaler. +18h retains original
FIT head counts 16397/16219/16041/15863/15685/14618; its extra head has 13556.
Every original target/mask/input and short loss scale was checked. New targets
are exact elapsed t+18 under the original C1/split rules, not market-row offsets.

For the 7/13/25 window comparison, 203 FIT, 24 INNER and 113 DEVELOPMENT origins
are absent under the common input support. All three windows therefore use
that same intersection and original per-head masks; the original full-support
control cannot be reused. One necessary matched-support 13-slot control is
scheduled, not a repeat of the +18 experiment. Windows remain raw, causal and
origin-inclusive, with actual timestamps and consistent slot metadata.

The finite five-node queue advances automatically: plus18, conditional
single18, window7, matched window13, window25. single18 is needed if +18 loses
to its naive or the short-head mean deteriorates versus the pinned parent.
Its head starts from the same untrained +18h initial head, not its trained
weights. Epoch2 is saved as diagnosis only; the unchanged original stopping
rule chooses the final checkpoint. Every new identity has one fit; interruptions
become typed terminals without automatic training repeats.

CPU verification: 12 tests pass; PRE had eight missing-module failures.
Actual pre-fit preparation caught scalar metadata/NaN-equality handling and
window-slot metadata issues before any fit; the retired preparation is retained.
Two GPU structural preflights refused generated layer-name differences before
fits; the witnesses showed unchanged weights and naming-only differences.
The corrected structural proof passed before scientific launch. No new
measurement or eligibility is inferred merely from starting.

The CPU follower collects complete immutable terminals, checks every saved
artifact hash, publishes per-horizon metrics with its own naive/readback and
maintains two physical copies. The existing best-result sidecar now watches
this campaign too. TEST remains closed; no feature selection, DOIN consensus,
pretraining or cadence setting was changed.

Status, one read:

```bash
python -B -m tools.i6dmo.mg2m_study status \
  --spec /var/tmp/i6dmo/mg2m-extension-20261017/RUN.private.json
```

The generated STATUS and RESULTS.md supersede this launch snapshot. All timers
close themselves after terminals, publication and verified mirrors; agents do
not poll logs. The first fit's epoch2 elapsed time was 10.39 seconds; that is
not an end-to-end ETA. Status exposes a measured current-fit ceiling and the
actual remaining graph, without claiming the epoch ceiling is a forecast.
