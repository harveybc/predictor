# Musashi: R0/R2 launch audit and dispatch correction

Campaign `i6dmo-9d3c198c77b1`, live root ending `-b`, immutable runtime
`c34b1e48`. Published integration `813ef019` was checked against origin.
The alternate root without `-b` has no DEPLOY.json and is retained, not deleted.
No fit, donor, objective, loss, scientific seal or model weight was edited.

## Actual state, 2026-10-15 19:08 UTC

Superseding observation at 19:14 UTC: all 19 donors, R0.fit/R0.report and
the diagnostic probe completed. R2.fit FAILED with R2_DONOR_TERMINAL_DIGEST
before training (training=null in its receipt); R2.report is blocked. All 19
referenced terminals match their hashes on the coordinator; zero of 19 exist
at the referenced paths on the preferred worker. SshSystemdExecutor.start
copies the attempt folder but not the external donor files. Thus the import
guard rejected missing transport input correctly. This is not a measured
negative R2 predictor. The comparison cannot establish an R0/R2 winner.

Owner correction following this finding: R2 is deferred behind the unfinished
RAW short/long sampling/strategy block. Preserve donors, R0 and failed attempt.
Do not hot-retry, bypass verification or repeat pretraining. The active order
is [RAW_PRIORITY_20261015.md](../RAW_PRIORITY_20261015.md), executed by Musashi.
The following 19:08 snapshot is historical, not a current RUNNING claim.

The finite plan has 25 units: 19 per-feature pretrainings, two predictor fits,
two reports, one diagnostic probe and one comparison. All 19 pretrainings
completed; R0 was RUNNING on the required preferred class, the probe was
RUNNING, and R2 awaited the same class. No TEST unit is enumerable. Seed
6102026 and the ruled-contract file digest match the effective configuration.
The two predictor fits are pinned to rtx5090; other GPUs may train independent
donors but cannot substitute for that controlled predictor-fit class.

The previously reported four-minute ETA described model-training seconds,
not end-to-end completion. At 19:02 UTC only 8/19 donors had completed after
roughly 20 minutes: workers waited 120 s after short units, tick waited 300 s
and publication 600 s. Typical recorded donor fit time was 5-12 s, with
additional launch/supervision overhead. Do not omit this scheduling cost.

## Operational correction actually applied

Campaign-scoped timer drop-ins now specify GPU worker idle interval 5 s,
coordinator worker 10 s, tick 10 s and publisher 60 s, with 1 s accuracy.
Boot activation is retained at 30 s. A systemd user daemon-reload applied the
settings; no running service/child was stopped, restarted or manually launched.
The runtime pin, sealed deployment JSON, per-job caps/reserves and sole-heavy
GPU slot on the preferred host did not change. The auxiliary GPU was not
enabled. Existing locks, claims, admission and same-GPU-class rules still apply.

After the correction, preferred and secondary simultaneously executed
different donor units; progress advanced from 8 to 19 completed donors. R0
then started automatically. Effective timer properties were read back. This
is an operational cadence amendment, not a new scientific treatment or a
claim that every GPU must be busy during CPU-only dependencies.

## Scientific qualifications and residuals

1. **Pretraining early stopping differs from the owner's global preference.**
   The sealed registered O1 objective stops on purged INNER reconstruction
   loss only, not the owner's requested mean of end-of-epoch FIT and INNER.
   Predictor fine-tuning uses the declared mean in both R0 and R2. Keep the
   current predeclared campaign unchanged and report this deviation; do not
   describe it as an implementation of mean-based stopping at every level.
   The result measures this specific pretraining-and-selection pipeline.
   Implement the owner's mean monitor as a versioned future objective, not a
   silent change to O1 1.0.0 or a mid-campaign checkpoint-selection rule.
2. **INNER is consulted in two stages, not numerically weighted twice.**
   R2 pretraining selection uses unlabeled input windows, and fine-tuning
   subsequently uses the inner predictive loss. This does not by itself
   demonstrate target leakage or invalidate a declared development comparison.
   Nor does it establish an independent holdout. Feature selection already
   used DEVELOPMENT 2024; TEST remains closed under the existing source gates.
3. **Do not claim denoising from reconstruction.** O1 reconstructs the input
   through a small branch and decoder; its representation can encode identity.
   Reconstruction is diagnostic. The downstream per-horizon R0/R2 errors on
   identical common masks and their own naives determine predictive benefit.
4. **Automatic donor custody is not live warehouse authentication.** The R1
   verifier checks the custody-protected outbox readback record and both mirrors.
   Its live warehouse re-query is optional, as R1_DONOR_TRUST.md explicitly says.
   Satoshi's summary must distinguish that from mandatory live readback on
   every adoption. The two-host rewrite limit remains, without signature claims.
   This campaign has no external prior roots and accepts its own authenticated
   pretraining terminals. Do not infer cryptographic authenticity from hashes.
5. **Pretraining peak is not measured.** run_pretrain explicitly writes
   cgroup_peak_bytes=None. The conservative provisional cap is retained; do
   not lower it from a different engineering run's 1.7-GB observation. Add
   whole-scope peak capture and publication in the next release, including
   failure paths. A sampled or TF allocator peak is not the full cgroup peak.
6. **Keep ETA scopes separate.** Remaining fit seconds omit launch, idle and
   host serialization. In particular, completed pretraining rows with
   score_seconds=0 cannot measure the cost of a future probe/predictor report.
   Report UNMEASURED rather than zero for an unmeasured phase.

The old 2.18-declared/2.21-observed baseline remains qualified, not resealed.
The current sealed runtime is TF 2.21.0 / Keras 3.13.2. The undeployed earlier
root and original producer record are evidence; neither requires deletion to
continue. No restart or credential action is established as necessary here.

## Bounded independent CPU checks

At published source 813ef019, the donor-trust suite plus the ruled-objective,
strict-decision, R0-versus-R2 identity and atomic-import rollback checks passed:
33 passed, two protobuf deprecation warnings, 123.48 s, under a 2-GiB/180-s
governed cap with GPUs hidden. These checks are not a rerun of all 808 tests
or a new scientific fit. Exact command:

```bash
crispdm-run -m 2G -t 180s -n r2-audit-contract-trust -- env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/i6dmo/test_r1_donor_trust.py tests/i6dmo/test_r2_design.py::test_the_committed_ruled_contract_carries_the_lead_ruling_and_is_executable tests/i6dmo/test_r2_design.py::test_decision_integrity_is_strict_by_default tests/i6dmo/test_r2_design.py::test_two_plain_r0_fits_are_never_r0_vs_r2_even_for_engineering tests/i6dmo/test_r2_design.py::test_import_rolls_back_every_branch_when_setting_weights_fails
```
