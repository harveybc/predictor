# Bounded re-check at 796ec8f4 (Retsu, 2026-10-15)

The review worktree was fast-forwarded to `796ec8f4`, which already contains the review commit
`aab9807f`. This pass re-runs the original reproductions for D1, D2, D3, D5 and D6, and verifies
the reseal equality independently. It is not a new broad audit.

Constraints held throughout:

- No fit ran.
- No TEST rows were read.
- Nothing was deployed and no systemctl call was made; systemctl was injected where needed.
- Everything ran on the coordinator CPU under `crispdm-run`. There was no ssh and no GPU, and the
  GPU replay was not repeated.

Outputs: `out/recheck_796ec8f4.json` and `out/equality_796ec8f4.json`.

**Overall: ACCEPT_FOR_DEPLOYMENT.** The deployment gate is unchanged: `matched-short preflight`
must report PASS on both GPU roles right before the timers are enabled. That means the remote
workdirs are at the runtime pin and `population-tf2210` is replicated under a declared
`experiment_storage_root`. I did not re-probe the GPU hosts in this pass. If deployment proceeds
before that, the code now fails closed into waits and does not burn the contrast (D5).

## Verdicts

| Defect | Verdict | Evidence (`out/recheck_796ec8f4.json`) |
|---|---|---|
| D1, fake-procedure donor | VERIFIED | Root A (fake fit/report entrypoints, real designs and pins) was driven to CLOSE. B, with `--prior-root A`, adopted nothing and recorded `PRIOR_REJECTED: DONOR_FIT_PROCEDURE_DIFFERS`; 0 attempts. `prepare` refused with `INCOMPATIBLE_DONOR … ran other entry points` |
| D1, CPU donor (`fit_device` cpu) | VERIFIED | Worker: `DONOR_FIT_PROCEDURE_DIFFERS`; prepare: `engineering/CPU campaign` |
| D1, engineering donor | VERIFIED | Same two refusals |
| D1, donor whose fit receipt is non-scientific (override) | VERIFIED | Worker and prepare both refuse with `DONOR_FIT_NOT_SCIENTIFIC:mg1.fit scientific=False … overrides=True` |
| D1, receipt edited after storage | VERIFIED | Worker: `PRIOR_REJECTED ARTIFACT_DIGEST:RECEIPT` (the receipt is a digest-bound terminal artifact). `prepare --prior-root` ACCEPTED it, because `_check_donor` does not verify artifact digests; the worker's refusal is what governs |
| D1, a root with real entrypoints into which another campaign's terminals were copied by hand | **REFUTED (residual R1, NON_BLOCKING)** | The worker ADOPTED `mg1.fit` and prepare ACCEPTED. The lane's fake child writes a digest-bound `RECEIPT.json` claiming `scientific: true, device: gpu` (marked `fake: true`), so the forged root passes every donor check |
| D2, receipt of `train --device cpu` | VERIFIED | `scientific: false, device: cpu` (exit 3, stopped at `POPULATION_DIGEST_MISMATCH`) |
| D2, `I6DMO_TEST_*` in the role environment, the injected env or the process env | VERIFIED | Each is `WAIT_RESOURCE TEST_OVERRIDE_IN_SCIENTIFIC_ENVIRONMENT:<key>`, with 0 attempt directories |
| D2, campaign report and decision | VERIFIED | A report with `require_scientific_fit: true` on a cpu, non-scientific receipt gives `NON_SCIENTIFIC_FIT`. `decide_matched(require_scientific=True)` refuses with `NON_SCIENTIFIC_FIT`; with scientific GPU inputs it decides. The worker sets the flag from the sealed effective config |
| D2, direct calls (default lenient) | Behaves as documented | The report gives `MATCHED_PAIRED_METRICS_AGREE`; `decide_matched` gives a decision. See the ruling below |
| D3 | VERIFIED | A file edit plus a stored-string edit after a verified load is now refused (`ARM_ARRAYS_MISMATCH`); untampered arms are still served (16,397 FIT rows) |
| D5, remote pre-fit refusals | VERIFIED | Preferred had a wrong population replica (`POPULATION_PIN_MISMATCH`); secondary had CUDA hidden (`NO_GPU_VISIBLE`). Each attempt became INTERRUPTED with `INFRA_PREFIT:<code>@<role>`; the unit stayed `WAIT_RESOURCE`, never FAILED. No comparison terminal and no CLOSE. At 1801 s status shows a typed `INFRA_PREFIT_BLOCKED` |
| D5, failure after training started | VERIFIED | `NONFINITE_LOSS` with `fit_seconds` > 0 gives FAILED (`TYPED:NONFINITE_LOSS`). Even an infrastructure code with `fit_seconds` > 0 gives FAILED (`TYPED:NO_GPU_VISIBLE`). The same code with no training gives `WAIT_RESOURCE` |
| D6 | VERIFIED | Both `status --help` and `matched-short status --help` now say `mg1 \| mg2m` |
| D7 | Recorded as a limitation (accepted) | — |
| Reseal equality | VERIFIED | `out/equality_796ec8f4.json`; see the next section |
| Superseded seals | VERIFIED | The old seals `4786951a…` and `6e8689d4…` are refused with `MATCHED_IDENTITY_SUPERSEDED` |
| Runbook on the reseal | VERIFIED | `matched-short prepare` against the real `population-tf2210` and real input exits 0; `validate` exits 0 with `common_population_sha256 = ba01f87a…` |
| Matched tests | VERIFIED | The four matched files give 85 passed |

### Reseal equality

I compared the old and new populations directly with numpy and json, on the real arrays.

- **Arrays.** All 13 npz files have equal content: dtype, shape and bytes, with NaN treated as
  equal.
- **Shared statistics.** Splits, scaler, naive loss scales, per-head counts and the report are all
  identical.
- **Identities.** The calendar, source, splits and TEST identities are identical.
- **What differs in the `POPULATION.json` body.** Only `identities.arms.*` (the experiment design
  and seal digests), `identities.code`, `identity_digests.arms` and the top-level digest.
- **Designs.** Old and new matched designs differ only in `seal_sha256` and
  `sealed_identities.runtime` (now TF 2.21.0 / Keras 3.13.2). The new population's arm identities
  bind the committed `b6fd008d…` and `416c1156…` seals.
- **Verified loads.** Both populations still load: new `ba01f87a…`, old `4ab41bd8…`.

This agrees with `POPULATION_RESEAL_EQUALITY.json`.

## Residuals (all NON_BLOCKING)

- **R1, forged donor root.** Adoption trusts a donor root's own `EFFECTIVE_CONFIG.json` and a
  receipt the child wrote. Hand-assembling a donor from another campaign's terminals passes. The
  lane fake writes `scientific: true, device: gpu` by default, which makes that trivial.
  Suggested hardening:
  - require the donor receipt's trainer identity to bind this campaign: `common_population_sha256`
    equal to the pin, `matched_arm`, `weights_sha256` and capacity;
  - refuse `fake`;
  - make the fake default to `scientific: false`.

  This is not reachable by ordinary misuse. A real fake, CPU, engineering or override donor is
  refused.
- **R2, churn and arm starvation under a persistent infrastructure refusal.** Over three rounds,
  `mg1.fit` relaunched a child on both roles each round, while `mg2m.fit` stayed QUEUED and was
  never tried. The cause is that `_pick` returns the first eligible unit in plan order. It becomes
  visible as `INFRA_PREFIT_BLOCKED` after 1800 s. Each refused relaunch is a full trainer start
  that stops before loading data.
- **R3, `require_scientific` defaults to off for direct calls** (see the ruling).

## Ruling on `require_scientific`

**Adequate for deploying this contrast. Make it strict by default before any other consumer
exists.**

The only production path is worker → `unit_runner` → `reporting.run_unit` /
`matched_runner.run_unit`. That path sets the flag from the sealed effective config: strict unless
`engineering_cpu_fixture`. Both the report and the decision enforce it, as verified above.
`METRICS` also records `scientific_fit`, `fit_device` and `require_scientific_fit`, so a lenient
decision is visible afterwards.

Leaving the default lenient is still fail-open for any new caller: an R2 comparison, a manual
recomputation or a future CLI. My ruling is therefore:

- default `require_scientific=True` in `run_matched_unit` and `decide_matched`;
- engineering and fixture callers opt out explicitly;
- do this before R2. It is not a blocker for this deployment.

## Commands and exit codes

Every command was run from the review worktree, with `<review>` =
`docs/i6n/direct_multi_output_static/matched_short/review`.

| Step | Command | Exit |
|---|---|---|
| fast-forward | `git merge --ff-only 796ec8f4` | 0 |
| recheck (D1/D2/D3/D5/D6) | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 2400s -n i6dmo-review-recheck -- python <review>/r_recheck_796ec8f4.py` | 0 |
| reseal equality | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 1200s -n i6dmo-review-equality -- python <review>/r_equality_796ec8f4.py` | 0 |
| prepare (real reseal) | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 600s -n i6dmo-review-recheck-prep -- python -m tools.i6dmo matched-short prepare --deployment $SCRATCH/recheck/private/deployment.json --input-dir $INPUT --out $SCRATCH/recheck/private/prepared.json` | 0 |
| validate | `… python -m tools.i6dmo matched-short validate --prepared $SCRATCH/recheck/private/prepared.json` | 0 |
| matched tests | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 1400s -n i6dmo-review-recheck-tests -- python -m pytest -q -p no:cacheprovider tests/i6dmo/test_matched_population.py tests/i6dmo/test_matched_fit_path.py tests/i6dmo/test_ops_matched_runner.py tests/i6dmo/test_ops_matched_real_chain.py` | 0 (85 passed) |

The first recheck run exited 1 because of a bug in my own probe: its report spec lacked
`code_commit`. A second run was vacuous for D1, because I had driven donor A with the old runtime
pin in the facts and so it never produced terminals. Both were fixed in the probe, and the
recorded output comes from the final run.
