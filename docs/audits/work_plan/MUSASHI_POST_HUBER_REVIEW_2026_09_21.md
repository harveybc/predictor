# Post-Huber review: corrections before the next measurement

Reviewed revision: `4ef9f71`. Disposition: `CORRECTIONS_REQUIRED_BOUNDED_NEXT_EXECUTION_CONDITIONAL`.
No new owner decision. No production write, service restart, training, reserve read or historical
artifact edit was performed in this review. Satoshi's preparation is preserved, not discarded.

## Findings, ordered by consequence

### 1. High: benchmark checks do not establish task or scale compatibility

`tools/df_benchmark_contract.py:44`, `:88`, `:138`; `tools/df_e1_huber.py:88`.
Changing metric_scale to USD, or horizon_seconds to 72 hours, still returns REPRODUCTION.
`reexecuted_on_our_rows=True` promotes mismatched protocols without reference-run evidence.
`require()` accepts null task fields and a self-declared mode. In the real factorial validator,
changing the target and horizon inside the benchmark block and recomputing the outer design
digest passes: the block is not bound to what the runner actually consumes. Its own digest was
not even repaired in that probe. Presence of a contract is not validation of that contract.

Required: typed fields, finite domains, physical-time consistency, canonical digest, dataset and
row identities, scaler parameters, actual execution/task binding. A planned comparator is not
an executed comparator. Separate protocol readiness from verified result comparability, otherwise
the first legitimate reproduction becomes circularly impossible. Deliberately varied factors
must be declared contrasts, not silently ignored fields or forbidden ML experimentation.

### 2. High: the table can report changed/missing predictions as verified

`tools/df_closure_table.py:118`, `:155`, `:206`.
The warehouse call checks a terminal digest against the receipt; it does not link the arrays
being scored to that terminal's artifacts or validate the complete expected forecast population.
On fabricated files, changing predictions reduced MAE from 2 to 0 with the same receipt, both
tables with zero problems and a matching warehouse badge. A missing warehouse terminal, NaN
predictions, and a missing forecast file also yield no problems. The last produces an empty
NO_NEW_MEASUREMENT table despite a declared forecast unit. Model/naive scale/population/horizon
validation only checks optional keys that the producer itself does not emit.

This demonstrates an inadequate new verifier, NOT altered historical results. Repair the chain
and reverify preserved arrays, not training. Derive forecast roles from the registered design;
do not classify missing arrays as a preparation unit from a filename or absence of data.

### 3. High: phase-2 context and volume controls still change more than stated

`tools/df_e1_phase2_design.py:168`, `:190`, `:215`; `tools/df_e1_pilot.py:373`.
The clamped long-window model reaches **67** raw samples, not 60: branch reach 5 plus core reach
63 minus 1. Measured on the real builder with clamped dilations: nonzero gradients at all seven
extra positions, earliest lag 66. A long window therefore sees extra information even in the
declared null. Its left padding also differs from the short-window model. Use an explicit common
raw-input crop for an exact information-null, plus a separate unrestricted local-support arm if
useful; test graph, gradients, perturbations and paired weights.

Volume validation every epoch changes stopping opportunities. In a gap-free extension, 28/56/112
days imply 627/1257/2517 updates per epoch at batch 64; 4000 updates supplies approximately
6.4/3.2/1.6 epochs. Patience 3 then has very different opportunity to operate. Measure actual
enumerated counts; fix validation cadence in observed updates. Budget termination remains censored
even if the best checkpoint was earlier, not only if it is near the ceiling.

Also, volume_counts calls the 50,400-row train-plus-validation slice a train span; train is
40,320 scheduled minutes before window/purge exclusions. Counts must derive from row identities.
Refitting scalers per volume arm changes preprocessing and the normalized denominator. For the
primary volume-only contrast use common permitted training data to fix input/target scaling and
one evaluation sigma; a refit-scaler variant is an explicitly separate factor.

### 4. High: financial targets and small-error Huber candidates are not executable as declared

`tools/df_fin_loss_opt_design.py:47`, `:76`, `:102`, `:136`.
The design equates row steps to elapsed hours while retaining weekend gaps. A synthetic Friday
23:00 origin has its sixth next weekday bar Monday 05:00, **54 hours**, not six. The residual
calibrator likewise subtracts array offsets without timestamp/availability identities. Retrospective
file coverage alone does not establish intrabar finality, timezone or historical availability.

On a nonconstant train target with positive sigma but zero median persistence residual, the
calibrator declares MEASURED with four zero Huber deltas. Rounding deltas to six decimals also
destroys the small-error regime this work is intended to study. Insufficient/nonfinite residual
populations, zero scale and flat training need explicit dispositions, not zero-threshold fits.
The 12-fit budget has no exact enumerated candidates across optimizer, LR, decay and delta;
the large receiver's width and input set are still unspecified. Design placeholders are not a
frozen executable scientific population.

### 5. Medium: the reference is not fully specified; the TCN table was mistranscribed

Independently checked the [primary PDF](https://arxiv.org/pdf/1907.09207), sections 7.1-7.4 and
visually inspected Table 4. TCN has no n_H entry: the config's n_H=50 is not supported by that
column. Its width is specified by M=32, with kernel 2 and L=8. Table 3 labels sample counts, not
an exact enumeration of admissible input/output windows; no code demonstrates that the config's
35040/103301 must equal window counts. The paper leaves implementation details unresolved, so
the packet's "only fully specified" description contradicts its own documented adaptations.
Fix the transcription and distinguish article facts, explicit adaptations and unknowns. Do not
claim exact numeric reproduction from counts alone. The adapted matched-task method can run
without pretending to recover an unspecified original implementation.

### 6. Medium: five xfails cannot turn green when the runner appears

`tests/test_fin_loss_opt_acceptance.py:104`, `:132`, `:150`, `:184`, `:198`.
Each body unconditionally raises NotImplementedError. Installing a runner changes nothing;
`strict=True` does not connect that body to the implementation. The equal-budget test also compares
one scalar to itself. Replace placeholders with actual entry-point calls and independent expected
outputs; retain the PRE proving absent behavior, then make the real acceptance tests pass.

## Evidence and scope

- [Reproducer](../evidence/POST_HUBER_REVIEW_2026_09_21/reproduce.py) and
  [measured output](../evidence/POST_HUBER_REVIEW_2026_09_21/results.json).
- CPU-only execution in trading-stack; real model build and gradient, no fit. Synthetic fixtures
  only for corrupt/missing files; source run DESIGN was read only for the real-validator probe.
- Reran all five new test modules: **61 passed, 5 xfailed** in 6.01 s. Their green status coexists
  with the demonstrated findings. Full suite NOT rerun; Satoshi's full-suite count remains attributed.
- The paper PDF was fetched for primary-source verification; no publisher text or PDF is committed.
- No independent live warehouse reconciliation in this review. A stand-in deliberately returning
  the same receipt isolates the table's missing array-to-terminal check; it does not simulate a
  successful production audit.

## Owner-facing result status

NO_NEW_MEASUREMENT. The previously measured Musashi factorial remains preserved. On its fixed
household DEV target/horizon, the previously verified MAE_z is 0.542536 for MAE+AdamW versus
0.676560 for persistence, skill 19.81%; Huber+AdamW is 0.572569, skill 15.37%. These are prior
development results, not new verification of every historic row. Literature comparator:
NOT_COMPARABLE / not yet measured on this contract. No incompatible paper score is placed beside
these numbers. The next orders prioritize generating that matched reference, not another table
that calls an unrelated published number a comparator.

## Disposition

Accept preparation and corrections as work performed; do not accept the new verifier guarantees
or phase-2/financial design as ready unchanged. Preserve all measurements. Execute
[RP66-RP73](../../handoffs/MUSASHI_POST_HUBER_RP66_RP73_2026_09_21.md), including conditional
bounded development training once the concrete acceptance conditions pass. No additional owner
permission is needed for those conditions, and no production reconfiguration is required here.
