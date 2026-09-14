# Return packet — C166–C184, D2 reattestation

**Date:** 2026-09-13
**Order:** `MUSASHI_TO_GENERAL_SATOSHI_C166_C184_D2_REATTESTATION_ORDER_2026_09_13.md`
**Audit answered:** `MUSASHI_AUDIT_C146_C165_MEMORY_CAUSALITY_2026_09_13.md` (`ACCEPT_D0_D1_MECHANICS_REVISE_D2_BEFORE_D3`)
**Stopping at:** `D2_CURRENT_API_FRESH_CONFIRMATION_READY_FOR_MUSASHI_REVIEW`

CPU only. No GPU, models, feature selection, D3–D5, RL, DOIN, live or venue. Nothing here grants eligibility or opens D3.

---

## 1. My own faults, first (counts from the final tip)

1. **My PRE dry run had two harness defects.**
   - C166.3 used a sine perturbation of training rows 31–59. It was too weak to move the wavelet detail MAD, so the fit-mode bypass did not show up. I replaced it with rescaling, the same construction the battery's mutant uses.
   - The `load_receipts` directory was still writable, although its files were read-only. I made the directory read-only.

   The committed PRE (`323a772`) is the re-run.
2. **The deliberate C151(b) out-of-memory test ran by default in the focal battery.**
   - Each run produces a contained cgroup kill, which the desktop reports as "program closed".
   - It ran twice in my C165 POST runs (06:57, 07:03). It ran once more at 12:47 in an independent battery run under the base anaconda interpreter; I did not start that run.
   - The owner could not tell which agent caused the alert. The test is now skipped unless an explicit flag is set.
3. **My memory hook had a false positive.** It split on `|` inside a quoted string and blocked a read-only diagnostic over ssh. It now splits only on operators outside quotes, tested on 10 cases.
4. **The merged tip first failed 2 of 921 focal tests.** Four agents had built the placement dispatcher, identity/coverage v2, guard removal and the D2 lab code in parallel.
   - **Retired name:** `df_d2_design` and two tests spelled the retired name outside the naming record. The mapping is now derived from `df_operators.NAMING_DECISIONS`.
   - **Seed realization:** a D2 worker test asserted one fresh unit was not identifiable. The fresh seeds derive from the design digest, which binds the lab code digests, so merging other code changed the realization. The test now asserts the typing that holds for any realization.

   Both were fixed in `65a45e9`.
5. **My first provisional timing design was refused by the worker.** I had left budget and roles as placeholders. It was rebuilt with real values; it is used for timing only and never governs.
6. **Two agent CLIs refuse to write under `~/.local`.** These are the structural mutation harness and the battery. Their custody evidence is produced into scratch and then copied read-only into state roots.
7. **I broke the dispatcher's unit naming when I added `{role}` binding (`e1ae37b`).**
   - **Cause:** the start script took its unit name from the role-bound job, whose argv hash differs from the template's.
   - **First real dispatch (C172 reanalysis):** every shard unit started correctly, but under a name the dispatcher never polled. It recorded LAUNCH_NOT_FOUND and requeued jobs whose units were running.
   - **Effect:** WORKER_A exceeded its cap, running 14 shards at once against a cap of 10, with no memory pressure (18 GiB free). Every shard ran exactly once, on one role.
   - **Containment:** I stopped the dispatcher and let the units finish as independent services. The run is reconciled against physical unit states and shard roots in `d2_dispatch_c172_v1_reconciliation/RECONCILIATION.json` (`1f574512…`), not against the broken receipts.
   - **Fix:** `734bd55`, with a regression test that launches through the dispatcher. My first role-binding test compared the template's identity with itself and could not catch the bug.
8. **My D2 worker invalidated a root for units no arm could run on.**
   - **What happened:** the first C172 reanalysis (design `fd68248b`) wrote 9 `ROOT_INVALIDATED` markers. They cover every MCAR unit of the C137 bank: sinusoid, steps and multivariate, white, 10 dB, 2048 rows, seeds 11–13.
   - **Cause:** each unit's longest complete TRAIN stretch is 16–43 rows, under the sealed 50-row missing-data rule. Every arm was REFUSED with that one reason, controls included. The worker then counted the unevaluated oracle as an invalidation. C177 audits the units really evaluated, and the design types a refused arm as abstention.
   - **Effect:** no wavelet audit failed and no oracle went undetected. Left uncorrected, it would have invalidated the fresh reserve too, since the design gives these three regimes 30 seeds each.
   - **Fix:** `535420d`. A unit refused whole by the declared rule is `unit_evaluable: false`, and its unevaluated oracle does not invalidate the root. An oracle that is refused, fails or goes undetected on an evaluated unit still invalidates the whole root. Both cases are tested.
   - **Reseal:** the design was resealed as `D2V2_C171_R2_2026_09_13` before any fresh unit existed. Every field is identical except the worker code digest; the reseal script refuses otherwise.
   - **Reanalysis:** it was re-run in full under the resealed design. The first run is kept, read-only, as superseded evidence with its reconciliation.
9. **My loader extension for the D2 tables failed its first two throwaway loads.**
   - **First:** the generated DDL put the `value_text` CHECK on every table with `status` and `value`; the D2 unit tables have no `value_text`, so `CREATE TABLE df_fact_d2_unit_denoising` failed. Fixed in `5e10534`.
   - **Second:** `validate_row` read `row["value_text"]` the same way and raised `KeyError` on the first D2 row. Fixed in `87d8f90`, then all 352,540 reanalysis and comparison rows validated with zero refusals before the next attempt.
   - Both throwaway databases were dropped by the loader's own `finally`; nothing reached the real cube.
10. **I aimed the third throwaway load at a receipt directory I had made read-only in the PRE.** `load_receipts` holds the C164 receipts the order preserves, so the receipt write would have failed after the whole load. I stopped it with SIGINT (the `finally` dropped its database) and moved the C178 receipts to a new directory, `load_receipts_c178_v1`.
11. **The first workers' fresh dispatcher reserved 2 GiB per shard against an observed peak of 123 MB**, which held WORKER_A to 6 shards and WORKER_B to 2. I stopped that dispatcher, let its 8 shards finish as independent units (reconciled in `d2_dispatch_c174_workers_reconciliation`), and relaunched the other 30 shards at 0.8 GiB per shard: 16 on WORKER_A, 5–6 on WORKER_B. No memory pressure followed.
   - At 17:47 the COORDINATOR, 2 shards at a time, would have been the tail (about 19:50), while also hosting the real OLAP load. I created its dispatcher's stop file (checked before every placement), confirmed on the three roles that none of shards 44–47 had started, and dispatched those four to WORKER_A under `d2_dispatch_c174_workers_tail`. The COORDINATOR ran shards 38–43. Its dispatcher kept watching 42–43 after the stop, wrote COMPLETED receipts for all six, released their units, and ended with `DISPATCH_PROGRESS.1.json` (`stopped: true`, 6 COMPLETED, 4 NOT_STARTED = shards 44–47). No shard ran on two roles.
12. **Owner-ordered data-gov work shares this branch.** On the owner's direct request (outside this order) the predictor side of the data-gov flow v2 (`olap/lake` write_metrics, `tools/governed_run.py`, its tests and docs) was committed on `satoshi/c166-c184-20260913`. None of it is a D2 code file, and none of it ran against a D2 root. Its end-to-end check has not run. The POST lists those files separately in acceptance test 10.
13. **My real D0–D2 load added a second, cell-identical v1 coverage matrix under the C164 run id.**
   - **What happened:** the loader recomputes the superseded v1 coverage matrix on every load, and its run id (`c140_17e79fa3…`) was built from the contracts alone. `df_coverage.py` changed in C170, so the rows carried a new `code_sha256`, and therefore new row identities. The receipt says it: 220,347 offered, 220,347 inserted, 0 already present. Nothing was deduplicated silently, but a query by that run id now counts every cell twice.
   - **Measured in the cube:** the two matrices (`code_sha256` `31e3376d…` from C164 and `b4c9157c…` from C170) agree on all 220,347 cells, with the same state, and neither holds a cell the other lacks. The additive cube keeps both; a consumer tells them apart by `code_sha256`.
   - **Fix:** `39082c2`. The v1 and v2 coverage run ids now bind the first 12 hex of the coverage code digest, so a matrix computed by other code gets its own run id. A test holds it.
14. **My review agents caused two contained OOM kills on the COORDINATOR, and the owner saw the desktop alert.**
    - **When and what:** at 17:17 and 17:22, two probe processes were killed at their 2 GiB caps inside `crispdm-batch.slice`, with anon-rss of 2,089,164 kB and 2,083,644 kB (1.99 GiB each).
    - **Why:** they came from subagents reviewing the owner-ordered data-gov work. Those subagents reproduced "the lake loads the whole file" against large real financial files instead of small synthetic ones.
    - **Scope:** no host OOM, no memguard action, no D2 process affected, and no kill since.
    - **Rule now in force:** a memory finding is reproduced only with inputs sized under its cap, and the killed scope's name in `journalctl -k` identifies the source before anyone guesses.
15. **The `tables` step of my adjudication driver was never dispatched.** Its CLI accepted `tables` but fell through to `merge` with no work directory, and crashed before writing anything. Merge had already completed, and nothing was lost. Fixed in `ccff52e`, then re-run.
16. **Three statements I wrote without measuring, corrected before this packet:**
    - the number of untouched rows in the D0–D2 load (1,361,548, where the receipt says 1,567,040);
    - "90 fresh MCAR units refused whole" (the shard roots say 64);
    - a finishing time of 19:05–19:15 for the fresh run, which the per-shard estimate moved to 19:57; it ended at 19:58.

    Every figure in this packet is read from a receipt, a manifest, a root or the cube.
17. **My first POST failed 3 of 18 checks: two were my harness, one is a real cross-host difference.**
    - **AT1, harness error:** the scan counted string literals containing "guard" as switches. In `df_causal_battery.py` those are the probe module's name, its file name twice and one evidence key (table in section 3). Identifiers containing "guard", environment reads and `global` statements are 0 in all three modules, and the 26 behaviour tests pass.
    - **COUNTS, harness error:** it compared `d2_fresh_tape_c173_v1/SEED_TAPE.json` across roles, but only the COORDINATOR holds that root. The workers read the tape inside the reserve, and that copy is identical on the three roles (`29487904…`, equal to the sealed tape root file).
    - **AT9, measured difference, not corrected.** A fresh unit computed on WORKER_B (`sinusoid/ar1/0 dB`, seed `…522883311201724505`) was re-run on the COORDINATOR. Of its 583 numbers:

      | numbers | where | difference |
      |---|---|---|
      | 565 | everywhere | exactly equal |
      | 5 | denoising RESIDUAL `residual_excess_acf1` (ewma ×2, trailing_mean, trailing_median ×2) | 1–2 ulps (1.1e-16 to 2.2e-16), within tolerance |
      | 3 | SNR `ar_residual`, calibration | 8.9e-16, within tolerance |
      | 4 | SNR `local_level_kalman`, calibration (estimate, error, \|error\|, CI lower) | about 1.3e-10, within the relative tolerance |
      | 6 | SNR `local_level_kalman`: confirmation estimate, error, \|error\| (7.7e-6 dB), CI lower (1.0e-6 dB), CI upper (0.006 dB); calibration CI upper (2.6e-4 dB) | beyond the declared tolerance |

      - **Same role:** re-run on WORKER_B itself, all 583 numbers are exactly equal.
      - **Cause, measured not assumed:** the WORKER_A unit (Intel i9-13900HX) re-derives exactly on the COORDINATOR (AMD Ryzen 7 7435HS). The differences appear only between the Ryzen 9 8940HX and the Ryzen 7 7435HS, under the same numpy 2.5.1 and scipy 1.18.0. This is consistent with ulp-level floating-point differences between CPU microarchitectures, amplified by the Kalman likelihood optimizer into the 6 larger ones. It is an inference from these measurements; I did not instrument the optimizer.
      - **Decision effect: none.** That estimator's decision in that regime is SNR_NOT_IDENTIFIABLE (26 identifiable seeds of 27). Its mean |error| is 21.7 dB (CI 14.4–29.1) against the 1 dB threshold, with coverage 0.0.
      - **Not done:** I did not widen the declared tolerance after seeing this, and I did not re-check the other 3,971 units across hosts.
    - **Harness fixes and POST v2 (`35cb46c`):** AT1 now counts identifiers and lists strings; COUNTS reads the reserve's tape copy. AT9 keeps its criterion. The new AT9b re-runs each picked unit on the role that produced it and requires exact equality. The first POST output is committed unchanged (`8070cdd`).

18. **My C178 loader change broke a C164 test, and I only found it in the final-tip battery.**
    - **What broke:** extending `df_load_d0_d2 --table-dir` to the D2 grains changed its refusal message from "none of the C164 tables" to "none of the C164 or D2 tables".
    - **Why it went unseen:** after the change I ran only `test_df_load_d0_d2.py` and `test_c139_data_foundation_olap.py`, not `test_c164_runtime_grains.py`, which matched the old message.
    - **What surfaced it:** the full focal battery at `a27c3e6` reported 925 passed, 1 failed, 5 skipped. Four of the skips were PostgreSQL tests, because I had not passed the database credentials.
    - **Fix:** the refusal is unchanged in behaviour; the test now matches the new message. The battery was re-run at the final tip with the credentials (section 9).
    - **Effect on the evidence:** this fix touches a test only, not `tools/`. The POST ran on code identical to the final tip's.

## 2. PRE and POST

**PRE** `c166_c184_pre_2026_09_13.py` (`323a772`), frozen at the audit commit before any edit: **9 of 9 counterexamples reproduced**, facts hold, identities unchanged.

| item | reproduced |
|---|---|
| 1 source_rederive | public switch off: a mutated, re-digested matrix is accepted |
| 2 availability | public switch off: a row available after its decision is transformed |
| 3 fit_mode_enforcement | public switch off: trailing_haar_threshold fits EXPANDING, and outputs at t≤30 move when training rows 31–59 change |
| 4 exhaustive_cuts | public switch off: a one-sample leak passes seven cuts |
| 5 block identity | three ADF blocks give 3 estimate rows with 1 identity |
| 6 coverage | REFUSED maps to FAILED, and an inapplicable cell to NOT_RUN |
| 7 gate | a C137 LAB_CALIBRATED decision for the retired `wavelet_haar_atrous`, from old code, counts as a present D2 stage |
| 8 PCA parity | exact-equality test fails under numpy 1.26.4 with MKL, passes under numpy 2.5.1 with OpenBLAS |
| — | 18 preserved roots read-only; live CPU RAM and VRAM inventory by role; WORKER_B GPU1 without a handle |

**POST** `c166_c184_post_2026_09_13.py`, run from the final tip in new processes under `crispdm-run`. The first run is committed as it ran (`c166_c184_post_2026_09_13.out`: 18 checks, 15 corrected; AT1 and COUNTS failed on harness errors, AT9 on a measured difference; fault 17). The harness was fixed in `35cb46c`, and POST v2 ran into `c166_c184_post_2026_09_13.v2.out`: **19 checks, 18 corrected, 1 not corrected (AT9)**, 424 s, no OOM kill.

| check | order item | v2 result |
|---|---|---|
| BASE.tip | — | tip clean, no tracked edits |
| AT1 | C166.1–4, acceptance 1 | CORRECTED: 0 identifiers containing "guard", 0 environment reads and 0 `global` statements in the three productive modules. The battery holds 4 string literals (probe module name ×2, its file name, one evidence key). 26 behaviour tests pass: assigning tables, setting variables or passing `oracle_mode` lifts no check |
| AT2 | C168, acceptance 2 | CORRECTED: 17 of 17 structural mutants detected, each in its own process against verified bytes |
| AT3 | acceptance 3 | CORRECTED: the full battery passes and detects every negative control; a sampled battery exits nonzero with `MECHANICS_SAMPLE`; a leaking stand-in invalidates the whole D2 root |
| AT4 | C166.5, acceptance 4 | CORRECTED: three ADF and three KPSS blocks give 3 distinct identities each, `block_identity` bound; the cube holds 257,884 v2 estimates |
| AT5 | C166.6, acceptance 5 | CORRECTED: REFUSED, FAILED, NOT_APPLICABLE and NOT_RUN are distinct in code and in the cube's coverage v2 |
| AT6 | C166.8, acceptance 6 | CORRECTED: parity under tolerance `a55dc46f` on the six custody reports plus two new ones (both interpreters); one portable digest `c057e9ea…` |
| AT7 | C166.7, acceptance 7 | CORRECTED: a C137 decision fails the D2 gate both as a v1 record and with a stale v2 binding; the gate refuses the 14 fresh passing subjects |
| COUNTS | C173–C174 | CORRECTED: design, tape and reserve identical on the three roles; 3,972 of 3,972 fresh terminals COMPLETED (492 / 1,821 / 1,659); 513 historical terminals; no marker; design and tape verify against current code |
| UNSEEN | C173 | CORRECTED: a new scan of the C128–C163 seeds shares no seed or derived seed with the tape and reproduces the sealed scan digest |
| C172 | C172 | CORRECTED: the comparison re-derives for the two families with flips; units and truth equal; counts kept apart |
| AT8 | C175–C176, acceptance 8 | CORRECTED: calibration, historical, time-grained, tape-less and duplicated rows are refused; decisions for two families re-derive byte-equal in new processes |
| AT9 | acceptance 9 | **NOT CORRECTED**: two fresh units regenerate from their seeds and bound contracts. The WORKER_A unit re-derives exactly on the COORDINATOR. The WORKER_B unit exceeds the declared tolerance in 6 `local_level_kalman` SNR numbers (at most 0.006 dB); fault 17 |
| AT9b | acceptance 9, added in v2 | CORRECTED: both units re-run on the role that produced them, 583 of 583 numbers exactly equal each (Intel i9-13900HX and AMD Ryzen 9 8940HX) |
| C178 | C178 | CORRECTED: both throwaway loads idempotent and dropped; both real loads additive with history unchanged, 0 refused, no silent dedup; historical and fresh apart by mode; loader never restarted |
| C179 | C179 | CORRECTED: the submission grants nothing and the reviewer authority holds no D2 record |
| AT10 | acceptance 10 | CORRECTED: no forbidden path changed since the order; every dispatch launch hid the GPUs; no GPU compute process in a crispdm cgroup on any role |
| C180 | C180 | CORRECTED: see section 8 |
| PRESERVED | C166 | CORRECTED: every root and document the PRE froze is byte-identical |

## 3. Causal boundary without switches (C167–C168)

**No switch in production** (`1c810dc`):
- **Removed:** `GUARDS`, `_refuse_if` and `_guard` are gone from `df_snapshot.py` and `df_operators.py`. Every causal check is an unconditional `if bad: raise` on every productive path.
- **Read-only fit modes:** `KIND_FIT_MODES` is read-only, so a kind's fit modes cannot be widened in place.
- **Battery switches removed:** `df_causal_battery.py` no longer has `BATTERY_GUARDS`, `guard_disabled`, the in-process guard mutations or the seven-cut table.
- **AST scan** outside docstrings. It counts identifiers containing "guard", environment reads, `global` statements, module flags, flag tables read by conditions and variadic public parameters. String literals containing "guard" are listed, not counted:

| file | base `323a772` | tip: identifiers containing "guard" | tip: string literals containing "guard" |
|---|---|---|---|
| `df_snapshot.py` | 9 | 0 | none |
| `df_operators.py` | 19 | 0 | none |
| `df_causal_battery.py` | 33 | 0 | 4: the probe module's name `"df_guard_probes"` (its import), its file name twice (in the code digest map), and the summary key `"guard_mutation_evidence"`, whose value says guard mutations are not produced by the battery |

The first version of this table (from the development scan) said 0 for the battery without separating identifiers from strings. The POST's scan counted the 4 strings as switches and failed (fault 17). None of them is assigned, read by a condition or used to lift a check.

- **Behaviour tests:** every refusal still fires after any of these:
  - assigning `GUARDS` / `BATTERY_GUARDS` attributes to all-False on every module;
  - setting environment variables;
  - passing extra keyword arguments, which raise `TypeError`;
  - writing into `KIND_FIT_MODES`, which raises `TypeError`.
- **One boolean parameter remains, `oracle_mode`,** on `transform_batch`, `transform_batch_components` and `probe_transform`. It only opens the declared non-causal negative-control kind, and a test shows no causal check reads it.

**Official battery vs mechanics sample:**
- **PASS:** a row is PASS only when its class ran the full declared cut list.
- **Sampled runs:** a sampled run (`--mechanics-sample K`) labels rows `MECHANICS_SAMPLE`, sets `battery_scope: MECHANICS_SAMPLE` and `all_pass: false`, and exits nonzero.
- **Agent development run:** 1,062 rows, 252 PASS per operator class, 10 negative controls detected, 16 snapshot refusals, 28 fit modes; 168 s, 125 MB.
- **Custody run** (`causal_battery_c167_v2`, summary `26fa9654…`, run `c167_118e4017…`): scope FULL, `all_pass: true`, no failures. It holds 1,062 `df_fact_causal_test` rows plus 3 naming decisions:
  - PREFIX_ALL_T, BATCH_STEP_CHUNK_RESTART, SUFFIX_ADVERSARIAL and REFERENCE_EQUALITY: 252 PASS each;
  - FIT_MODE: 28 PASS;
  - SNAPSHOT_REFUSAL: 16 PASS;
  - NEGATIVE_CONTROL: 10 DETECTED.

  The run took 163.5 s, with a peak of 125 MB. It was produced into scratch and copied read-only into custody, because the battery CLI refuses `~/.local`. It was loaded into the cube with the real D0–D2 load.

**Structural mutations** (`tools/df_structural_mutation.py`, harness probes in `tools/df_guard_probes.py`):
- **Isolation:** each of the 17 checks is removed from a COPY of the minimal source by an AST transformation, in its own temporary directory, and run in a new process against both the intact source and the mutant.
- **Evidence process:** it never imports a tools module, and every child is verified to have loaded exactly the original or the mutant bytes.
- **Digests:** each row records sha256 of the original, the mutant and the probe.
- **Agent development run:** 17 of 17 detected (intact REFUSED, mutant ACCEPTED). The `artifact_bound` mutant raises `TypeError` instead of accepting.
- **Wavelet case:** the mutant fits EXPANDING, and outputs at t≤30 move when training rows 31–59 are rescaled.
- **Seven-cut mutant:** it accepts the one-sample leak.
- **Three probes rebuilt:** so that only their own check can refuse — matrix_digest forges and re-seals the digest; dataset_binding and stream_binding use the same bytes under two contracts.
- **Custody run** (`c168_structural_mutations_v1`, summary `228d299f…`, run `c168_5960239d…`): 17 declared, 17 run, 17 detected, `not_detected: []`, `complete: true`. The evidence process imported no tools module. Each child ran under `crispdm-run -m 1G -t 5m`; the largest child peak was 111 MB and the harness peak 33 MB, in 30 s. The 17 rows were loaded into the cube as `df_fact_causal_test`.

Tests at the C167–C168 tip: 591 passed.

## 4. Identity and coverage (C169–C170)

**Block identity** (`e70a9fe`):
- **Required fields:** every ADF/KPSS estimate must bind, before the library is called, the block offset (exact/start/middle/end), the absolute and partition ranges, the finite-run universe, the length and lag used, the policy digest and the variant. The planner refuses an estimate without them.
- **New table:** `df_fact_resource_estimate_v2` makes these explicit. v1 is untouched.
- **Recompute from the C162 roots, read-only, without re-running ADF/KPSS:**
  - v1: 257,884 rows, 255,786 distinct (2,098 collapsed: 1,049 ADF, 1,049 KPSS);
  - v2: 257,884 rows, 257,884 distinct, 0 unresolved;
  - 14,392 child estimates recomputed with no mismatch.

**PCA and eigen descriptors:**
- **Tolerances:** declared before the POST, with hash `a55dc46f` (absolute 1e-12, relative 1e-10). The raw value is kept, and a 9-significant-digit canonical value is stored for portable digests. The numpy version and BLAS backend are recorded per row.
- **Fixture run:** the same fixture ran on all three roles under both interpreters (numpy 1.26.4 with MKL, numpy 2.5.1 with OpenBLAS), and `c169_parity_reports/COMPARISON.json` (`2334e4d8`) records the comparison:
  - parity true across 6 reports;
  - 384 rows compared, 336 under tolerance;
  - one portable digest `c057e9ea`.
- **Largest raw difference:** effective rank 1.01e-13, then PC1 shares 1.27e-14.
- **Claim withdrawn:** byte equality for these values is no longer claimed. Exact equality is kept for every other metric of the parity fixture. This statement covers that fixture only. D2 unit rows re-derived on another CPU differ by ulps, and one optimizer-based SNR estimator differs by more (fault 17).

**Coverage v2:**
- **States:** RESULT, INCONCLUSIVE, UNAVAILABLE, NOT_APPLICABLE, NOT_RUN, REFUSED, FAILED, RESOURCE_EXCEEDED, UNCERTAIN.
- **Applicability:** declared per dataset, variable type, partition, metric and policy, never inferred from a missing row.
- **Precedence:** specified and tested over all orderings. RESULT together with FAILED, RESOURCE_EXCEEDED or UNCERTAIN gives UNCERTAIN; otherwise RESULT > INCONCLUSIVE > REFUSED > RESOURCE_EXCEEDED > FAILED > UNCERTAIN > UNAVAILABLE > NOT_RUN.
- **REJECTED:** counted as a reached decision (RESULT), with the raw status counts kept.
- **Migration:** additive, with v1 preserved, a per-cell v1→v2 map, and totals recounted member by member.

**Deliberate OOM test:** the C151(b) test is now skipped unless `CRISPDM_ALLOW_DELIBERATE_OOM_TEST=1`.

Tests at the merged tip: 88 passed, 1 skipped (that OOM test).

## 5. D2 design v2, historical reanalysis, fresh reserve (C171–C173)

**Design sealed before any result (C171).** `d2_design_c171_v2/D2_DESIGN_V2.json`, id `D2V2_C171_R2_2026_09_13`, sha `3577c154…` (supersedes `fd68248b…`, whose only difference is the worker digest, fault 8).
- 171 regimes (family × perturbation × declared SNR × length × missingness), never pooled over a field.
- 15 arms: 12 CANDIDATEs (the C137 specs with LAB_CALIBRATED + REGIME_LIMITED share ≥ 0.30 across regimes), IDENTITY_RAW_CONTROL (`identity`), PREVIOUSLY_REJECTED_CONTROL (`trailing_median` window 21, share 0.094), NON_CAUSAL_ORACLE_CONTROL (`centered_mean_oracle` window 5; detected, never competing).
- Seeds per regime from a paired one-sided noncentral-t power calculation with Bonferroni per operator family, 80% power, minimum 10, maximum 30: 123 regimes POWERED, 48 UNDERPOWERED, 3,972 fresh units.
- C137 (`C137_DISPERSION.json`, `692d0529…`) used only for dispersion and candidate choice; parameter selection is `HISTORICAL_DEVELOPMENT_ONLY`; the confirmation partition governs; retuning forbidden.
- Budget and roles are real values (task 2 GiB, host 12 GiB, wall 1,800 s; COORDINATOR ≤ 25% of free memory, 2 shards; WORKER_A 10; WORKER_B 6 with GPU1 quarantined).

**Historical reanalysis (C172), label `HISTORICAL_MIGRATION_REANALYSIS_NON_CONFIRMATORY`, grants nothing.**
- Bank: the intact C137 bank (513 units), read-only; every unit's content digest equals C137's (`truth_content_equal: true`, `units_equal: true`).
- Run: current code, public API only, 18 shards as independent systemd user services under the batch slice (WORKER_A 11, WORKER_B 7), 513 of 513 terminals COMPLETED, 0 invalidation markers, outputs re-hashed against their terminals at collection. Roots `d2_reanalysis_c172_v2/<ROLE>/shard_NN`; comparison `d2_reanalysis_c172_v2_comparison/C172_COMPARISON.json`; loader tables (`tables/`, 340,549 denoising rows, 8,229 SNR rows, 3,762 comparison rows, run `d2v2_hist_34330cf0…`).
- The first run of this step (`d2_reanalysis_c172_v1`, design `fd68248b…`) is kept as superseded evidence with its reconciliation (fault 7 and fault 8).

| comparison vs C137 (`c137_cc9cca24…`) | value |
|---|---|
| operator × regime rows | 3,762 (22 C137 specs × 171) |
| rows on both sides | 2,565 (the 15 arms of the design) |
| rows historical only | 1,197 = 7 specs not kept by the design: `trailing_hampel` ×2, `fir_sinc_lowpass` ×2, `causal_decomposition` ×2, one `ewma` |
| same decision on both sides | 2,563 |
| **flips** | **2**, both `local_linear_trend_kalman` LAB_REJECTED → LAB_CALIBRATED (multiband, white, −5 dB; sinusoid, student_t, 20 dB), primary cause TEMPORAL_MODE |
| name changes | `wavelet_haar_atrous` → `trailing_haar_threshold`, 342 rows, mapped through the naming record |
| outputs differ, decision unchanged | 681 rows |

Where the outputs differ and why:
- `trailing_haar_threshold` 320 of 342 rows, `local_level_kalman` 120 of 171, `local_linear_trend_kalman` 133 of 171: these kinds now fit in `FROZEN_PREVIOUS_PARTITION` mode (C167), so confirmation outputs no longer see calibration rows. Median relative difference 1.7–3.9%, maximum 26–31%.
- `trailing_mean` 110 rows: at most 1.9e-16 relative (summation order), no decision effect.
- `ewma`, `trailing_median`, `identity`, `centered_mean_oracle`: identical.

Counts kept apart, as ordered, and not at the same grain: C137 RESULT 13,251 / REFUSED 675 / INCONCLUSIVE 10,026 (operator runs and metric rows); reanalysis RESULT 8,967 / REFUSED 528 / INCONCLUSIVE 21,854 (unit × variable × arm, plus INCONCLUSIVE metric rows). None is FAILED. The 528 REFUSED arm results, counted from the reanalysis rows, are all typed refusals of the current code:

| count | reason |
|---|---|
| 225 | declared missing-data rule, fit stretch under 50 rows: the 9 MCAR units of the bank (fault 8), every arm |
| 200 | `ABSTAIN: MLE_AT_UPPER_BOUND` of the Kalman kinds, mostly in noise-free and 10–20 dB regimes |
| 49 | `ABSTAIN: ZERO_DETAIL_MAD` of `trailing_haar_threshold` (detail noise not identifiable) |
| 36 | `ABSTAIN: MLE_NOT_IDENTIFIED` (level/slope noise) |
| 18 | `ABSTAIN: CONSTANT_TRAIN_SERIES` |

**Fresh seed tape and reserve (C173).**
- Tape `d2_fresh_tape_c173_v1/SEED_TAPE.json` (`230a0a44…`), sealed before any unit existed: master entropy 32 bytes from `/dev/urandom` recorded in the tape; derivation `int(sha256(canonical{master_entropy, design_sha256, regime_key, index, counter=0})[:15 hex])`; prior scan of `synthetic_bank_c128_v1`, `lab_evaluation_c137_v1`, `lab_delay_cost_c137_v2`, `snr_calibration_c163_v1` plus the code constants: 132 prior seeds (`b4e39c45…`), zero collisions of seeds or derived seeds.
- Reserve `d2_fresh_reserve_c173_v1`: 3,972 units generated with the reviewed generator, each re-verified by regeneration, each with clean, noise, observed, missing mask, events, sealed contract and chronological 60/20/20 partitions; `ROOT_MANIFEST.json` `91810902…`; 31,778 files, 322,847,290 bytes, identical by digest on the three roles.

## 6. Distributed execution by memory (C174)

Memory-aware placement built for the owner's rule:
- a live inventory of CPU RAM, batch-slice headroom and per-GPU VRAM, re-read before every launch;
- quarantined GPUs are never scheduled, and GPU placement stays gated off for this order;
- jobs too large for any role are split.

**How the work was placed** (every job is one shard of units, run as a systemd user service inside `crispdm-batch.slice`; each unit is its own child process with a hard memory limit; GPUs hidden with `CUDA_VISIBLE_DEVICES=`):

| run | dispatch root | roles and concurrency | units | terminals | markers |
|---|---|---|---|---|---|
| C172 reanalysis v1 (superseded, faults 7–8) | `d2_dispatch_c172_v1` + reconciliation | WORKER_A 14 shards (cap 10 exceeded, fault 7), WORKER_B 4 | 513 | 513 COMPLETED | 9 (fault 8) |
| C172 reanalysis v2 | `d2_dispatch_c172_v2` | WORKER_A 11 shards, WORKER_B 7 | 513 | 513 COMPLETED | 0 |
| C174 fresh, COORDINATOR | `d2_dispatch_c174_coordinator` (stop file at 17:47 after 6 launches; ended 18:06, success) | COORDINATOR only, 2 at a time, ≤ 25% of its free memory; shards 38–43 | 492 | 492 COMPLETED, 6 receipts COMPLETED | 0 |
| C174 fresh, workers first dispatcher (stopped, fault 11) | `d2_dispatch_c174_workers` + `d2_dispatch_c174_workers_reconciliation` | shards 00–07: WORKER_A 6 (00, 01, 02, 03, 05, 07), WORKER_B 2 (04, 06); finished as independent units at 18:18, every unit exit 0, released after recording | 664 | 664 COMPLETED | 0 |
| C174 fresh, workers | `d2_dispatch_c174_workers_v2` (16:44–19:58, success) | shards 08–37: WORKER_A 12, WORKER_B 18; at most 10 shards of this dispatcher on WORKER_A (16 units with the stopped dispatcher's 6) and 5–6 on WORKER_B | 2,488 | 2,488 COMPLETED, 30 receipts COMPLETED | 0 |
| C174 fresh, tail moved from the COORDINATOR | `d2_dispatch_c174_workers_tail` (17:48–19:26, success, final receipt) | shards 44–47 on WORKER_A, 4 at a time | 328 | 328 COMPLETED, 4 receipts COMPLETED | 0 |

Why the COORDINATOR ran its own shards: the placement rule never puts a job on the COORDINATOR while a worker merely waits for this dispatcher's own jobs, because other agents work on that host. Left alone it would have stayed idle for the whole run. Restricting shards 38–47 to the COORDINATOR honours "three machines" without weakening that rule.

**Custody of the fresh roots.** After every dispatch unit ended, each role's roots were copied to the COORDINATOR with `rsync --delete`, then a second `rsync --dry-run --checksum --itemize-changes` pass reported 0 differences:

| role | shards | terminals | checksum differences | partial files | markers |
|---|---|---|---|---|---|
| COORDINATOR | 6 | 492 | local | 0 | 0 |
| WORKER_A | 22 | 1,821 | 0 | 0 | 0 |
| WORKER_B | 20 | 1,659 | 0 | 0 | 0 |
| total | 48 | 3,972 of 3,972 | | | |

The copy (`d2_fresh_c174_v1`, 3.6 GB) was then made read-only before collection. No dispatch unit, unit scope or failed unit remained on any role.

**One confirmation root.** `d2_fresh_c174_v1_collected` hard-links every terminal and re-hashed output of the 48 shard roots:
- run `d2v2_fresh_1fd0e710bc0689b3bf9d7162`, mode `FRESH_CONFIRMATION`, design `3577c154…`;
- 3,972 terminals, all COMPLETED, no invalidation marker;
- `RUN_MANIFEST.json` sha256 `90942903f9efddf1…`;
- all 48 shards were created under one lab code digest, `1924dbdfebd9…`;
- after the run, the current `lab_code_sha256` is still `1924dbdfebd9…`, every per-file digest equals the resealed design's, and the design validates against current code. The only commit touching a D2 code file after `734bd55` is `535420d`, the worker fix that preceded the reseal.

**Memory, measured, not assumed:**
- Observed peak per unit child: 121–124 MB on all three roles (planned estimate 918 MB, limit 1,843 MB).
- Wall per unit: median 55 s on the workers, 36 s on the COORDINATOR; CPU time equals wall (single thread).
- Batch slice peaks, cumulative since the slices started at 00:56 (`systemctl --user show crispdm-batch.slice -p MemoryPeak`):
  - COORDINATOR: 12.81 GiB of 14 GiB. This includes the D0–D2 load, whose process I sampled at up to 6.45 GiB resident, the probes of fault 14, and the COORDINATOR's own fresh units.
  - WORKER_A: 5.43 GiB of 14 GiB.
  - WORKER_B: 2.41 GiB of 8 GiB.
- Memguard logged no action on any role all day.
- The four watchers polled every 2–3 minutes and would alert below 4 GiB free on a worker or 5 GiB on the COORDINATOR; none fired.
- The only OOM kills of the day are the two contained ones of fault 14; none touched a D2 process.

## 7. Adjudications (C175–C177)

**Only fresh confirmation rows decide.** From the collected root, 1,434,898 rows were kept: the confirmation partition, plus the COST branch. 1,229,858 calibration and train rows were excluded as non-governing. `check_rows` refuses any historical, calibration, time-grained, tape-less or duplicated row before a statistic is computed. Each family was decided in its own capped process (regimes never cross families), 82 s in total. The result is `DECISIONS.jsonl` (`f4958c88…`), 3,591 rows = 171 regimes × (15 arms + 6 SNR estimators). Every row is `externally_reviewed: false`.

**Denoising (C176), 2,565 operator × regime decisions:**

| arm role | LAB_CALIBRATED | REGIME_LIMITED | LAB_REJECTED | NOT_IDENTIFIABLE | UNDERPOWERED | CONTROL_NOT_AN_ARM |
|---|---|---|---|---|---|---|
| CANDIDATE (12 specs) | 51 | 7 | 816 | 307 | 871 | 0 |
| IDENTITY_RAW_CONTROL | 28 | 0 | 134 | 9 | 0 | 0 |
| PREVIOUSLY_REJECTED_CONTROL | 2 | 0 | 158 | 11 | 0 | 0 |
| NON_CAUSAL_ORACLE_CONTROL | 0 | 0 | 0 | 0 | 0 | 171 |

Candidates that reached LAB_CALIBRATED or REGIME_LIMITED, by number of regimes:

| spec | LAB_CALIBRATED | REGIME_LIMITED |
|---|---|---|
| `local_level_kalman` | 11 | 0 |
| `local_linear_trend_kalman` | 10 | 1 |
| `ewma` α=0.5 | 8 | 0 |
| `butterworth2_lowpass` cutoff 0.25 | 7 | 4 |
| `trailing_haar_threshold` levels 2, k 3 | 7 | 0 |
| `trailing_haar_threshold` levels 3, k 3 | 2 | 0 |
| `butterworth2_lowpass` cutoff 0.1 | 1 | 1 |
| `trailing_median` window 5 | 1 | 1 |
| `ewma` α=0.3, `trailing_mean` 5, `trailing_mean` 9, `trailing_median` 9 | 1 each | 0 |

Where the 58 candidate passes fall (counts of regimes):

| grouping | counts |
|---|---|
| family | bumps 14, trend_linear 12, sinusoid 11, multiband 10, multivariate 6, steps 3, motif 2 |
| declared SNR | 5 dB 27, 20 dB 15, 10 dB 11, 0 dB 3, −5 dB 2 |
| perturbation | white 41, contaminated 7, student_t 7, heteroscedastic 3 |
| missingness | none 58 |

No candidate passes with missing data, and none passes in general. Every decision holds for its own regime, and no average between perturbations was used.

How to read the controls and the power states:
- **Identity control:** its 28 LAB_CALIBRATED are all noise-free regimes (`inf` dB), where the rule only tests false-positive distortion and raw data passes by construction. In noisy regimes it is LAB_REJECTED 133 times (no improvement) and NOT_IDENTIFIABLE 9 times.
- **Previously rejected control** (`trailing_median` window 21, chosen as the lowest C137 share, 0.094): it passes in `trend_linear/white/20` and `trend_linear/white/inf`, the same two regimes where C137 also rated it LAB_CALIBRATED, and is rejected or not identifiable in the other 169.
- **Oracle:** CONTROL_NOT_AN_ARM in all 171 regimes; it never competes.
- **Power is per spec, not per regime.** The 871 UNDERPOWERED are candidates whose own power entry is UNDERPOWERED:
  - 680: the C137 effect does not exceed the null;
  - 131: 30 seeds do not reach 80% power;
  - 60: no historical dispersion.
  The 29 LAB_CALIBRATED inside regimes the design labels UNDERPOWERED are all controls (28 identity, 1 rejected control); no candidate was calibrated in an underpowered regime.
- **MCAR regimes** (fault 8): every candidate is NOT_IDENTIFIABLE, every SNR estimator SNR_NOT_IDENTIFIABLE, the oracle a control.

**SNR (C175), 1,026 estimator × regime decisions** (CI95 upper bound of mean |error| ≤ 1 dB, coverage ≥ 90%, NOT_IDENTIFIABLE ≤ 10%, per seed, no cancelling of signed errors):

| estimator | CALIBRATED_FOR_REGIME | REGIME_LIMITED | REJECTED | NOT_IDENTIFIABLE |
|---|---|---|---|---|
| `mad_first_difference` | 16 | 36 | 61 | 58 |
| `spectral_floor` | 16 | 56 | 27 | 72 |
| `local_level_kalman` | 5 | 27 | 25 | 114 |
| `trailing_median_residual` | 2 | 8 | 50 | 111 |
| `ar_residual` | 0 | 44 | 66 | 61 |
| `wavelet_mad` | 0 | 0 | 0 | 171 (TRAIN aggregate only; no confirmation estimate by contract) |

No estimator is general. On real data the only label stays `MODEL_CONDITIONAL_SNR_ESTIMATE`.

**Wavelet audit and oracle (C177)**, counted from every `result.json` and `wavelet_audit.json` of the 48 sealed shard roots:

| quantity | count |
|---|---|
| fresh units | 3,972 (COORDINATOR 492, WORKER_A 1,821, WORKER_B 1,659) |
| root invalidation flags or markers | 0 (also 0 across the 18 reanalysis roots) |
| units not evaluated (whole unit refused by the declared missing-data rule, fault 8) | 64: `multivariate/white/10/mcar` 30 of 30, `sinusoid/white/10/mcar` 17 of 30, `steps/white/10/mcar` 17 of 30 |
| evaluated units | 3,908 |
| oracle outcome on evaluated units | DETECTED 3,908, NOT_DETECTED 0, NOT_EVALUATED 0 |
| `trailing_haar_threshold` arms that completed and were audited | 7,312; checks failed: 0 |
| `trailing_haar_threshold` arms that did not complete | 632 typed REFUSED rows: 504 `ABSTAIN: ZERO_DETAIL_MAD` (detail noise sigma not identifiable, in the noise-free and `null` regimes) and 128 missing-data refusals (2 arms × 64 units) |

The count closes: 3,908 evaluated units × 2 wavelet arms = 7,816; minus 504 abstentions = 7,312 audits, every one passing every check. The checks are prefix at every t, every level against a prefix-only reference, adversarial suffixes at warm-up, power-of-two and seeded cuts, batch/step/chunk/restart, no delay compensation moving output backwards, and `wavelet_mad` answering only as one TRAIN aggregate.

The 26 MCAR units that were evaluated still leave their regimes NOT_IDENTIFIABLE, because the valid seeds (13 of 30) do not reach the design's 30.

## 8. OLAP (C178), submission (C179), health (C180)

**Two loads, each rehearsed on a throwaway database first.** `df_load_d0_d2` holds every table it loads in memory at once, and the D2 unit rows alone are 3.9 GB of JSONL. So the D2 grains went through a separate loader that streams them in chunks, under the owner's memory rule:
1. `df_load_d0_d2` (one process under a 10 GiB cap, sampled at up to 6.45 GiB resident): the D0–D1 tables plus the v2 resource estimates (C169), coverage v2 and its v1→v2 map (C170), and the C167–C168 causal evidence. Receipts `load_receipts_c178_v1/throwaway_d0d2_v2grains_c178_v1.json` and `real_d0d2_v2grains_c178_v1.json`.
2. `c178_d2_load.py` (chunks of 100,000 rows): `df_fact_d2_unit_denoising`, `df_fact_d2_unit_snr`, `df_fact_d2_decision`, `df_fact_d2_historical_reanalysis`, historical reanalysis and fresh confirmation separate by `mode` and `run_id`. Receipts `throwaway_d2_c178_v1.json` and `real_d2_c178_v1.json`.

Rehearsal of the chunked loader on the reanalysis tables alone (`throwaway_d2_reanalysis_only_c178_v1.json`): 340,549 denoising, 8,229 SNR and 3,762 comparison rows; second pass inserted 0; 0 refused; offered = inserted + present + refused; no duplicate row identity in the input; database dropped; 229 s.

**Load 1, D0–D2 with the v2 grains:**

| receipt | result |
|---|---|
| throwaway (`throwaway_d0d2_v2grains_c178_v1.json`) | two loads into a new database; the second inserted nothing; 0 refused; database dropped; 1,557 s |
| real (`real_d0d2_v2grains_c178_v1.json`) | 34 historical cube tables counted before and after: unchanged; every table's offered = inserted + already present + refused; 0 refused; 798 s; no OOM kill |

What the real load wrote, table by table:

| table | offered | inserted | already present |
|---|---|---|---|
| `df_fact_resource_estimate_v2` | 257,884 | 257,884 | 0 |
| `df_fact_coverage_v2` | 633,189 | 633,189 | 0 |
| `df_fact_coverage_v1_v2_map` | 633,189 | 633,189 | 0 |
| `df_fact_causal_test` | 2,158 | 1,079 (C167 battery 1,062 + C168 mutations 17) | 1,079 (C156) |
| `df_fact_naming_isolation_decision` | 6 | 3 (C167) | 3 |
| `df_dim_run` | 15 | 4 (C167, C168, C140 recomputed, C170) | 11 |
| `df_fact_coverage` (v1) | 220,347 | 220,347 (fault 13: cell-identical recomputation, other `code_sha256`) | 0 |
| the other 16 tables (datasets, variables, profiles, pairs, groups, information, sampling, SNR C163, lab C137 runs, metrics and decisions, delay/cost, terminals, host receipts, incident, v1 estimates) | 1,567,040 | 0 | 1,567,040 |

Loader `crispdm-olap-loader`: ActiveState `active`, NRestarts `0`, active since 2026-09-12 13:39, before and after both loads.

**Load 2, the D2 grains** (historical reanalysis and fresh confirmation, streamed in 100,000-row chunks):

| receipt | result |
|---|---|
| throwaway (`throwaway_d2_c178_v1.json`) | new database, schema, every chunk loaded twice; second pass inserted 0 in all four tables; 0 refused; offered = inserted + present + refused; 0 duplicate row identities in the input; database dropped; 1,976 s; no OOM kill; loader untouched |
| real (`real_d2_c178_v1.json`, `e0b63c46…`) | every one of the 62 cube tables existing before the load counted before and after: none outside the D2 grains changed; all four tables inserted exactly their distinct rows (0 already present, 0 refused); no duplicate identity in the input; cube counts after the load equal the rows offered; 1,023 s; no OOM kill; loader untouched |

In the cube, historical reanalysis and fresh confirmation stay apart by `mode` and by run:
- unit rows keep the run id of the shard root that produced them (18 historical, 48 fresh);
- the comparison rows carry `d2v2_hist_34330cf0…`;
- the decisions carry the collected root's run `d2v2_fresh_1fd0e710…`.

| table | distinct rows offered |
|---|---|
| `df_fact_d2_unit_denoising` | 2,941,709 (HISTORICAL 340,549 + FRESH 2,601,160) |
| `df_fact_d2_unit_snr` | 71,825 (8,229 + 63,596) |
| `df_fact_d2_historical_reanalysis` | 3,762 |
| `df_fact_d2_decision` | 3,591 (fresh only) |

**The gate still refuses (C179).** `df_consumption_gate.py` was asked about every fresh subject that reached LAB_CALIBRATED or REGIME_LIMITED: 14 operator specs (`d2_submission_c179_v1/GATE_REFUSAL.json`). The reviewer authority holds 0 valid and 0 refused records. Every one of the 14 is refused, with D0, D1, D2, D3 and D4 missing, and `public_eligibility: NEVER_GRANTED_BY_THIS_GATE`.

**Submission (C179).** `d2_submission_c179_v1/D2_REVIEW_SUBMISSION.json` (`a59b5917…`):
- **Kind:** `SUBMISSION_FOR_EXTERNAL_REVIEW`, authority `NONE`. It creates no reviewer record and grants nothing. The reviewer authority still holds no D2 record, so `df_consumption_gate` refuses.
- **Code:** tip `1c9ca14`, tracked tree clean, per-file lab code digests, and the design validating against that tip.
- **Design, tape and reserve:** design `3577c154…` and what it supersedes; tape `230a0a44…`; fresh reserve manifest `91810902…` with a tree digest over its 31,778 files.
- **Historical side:** bank manifest, comparison, reanalysis tables, and a tree digest of the 18 reanalysis shard roots.
- **Fresh confirmation:** run `d2v2_fresh_1fd0e710…`, 3,972 terminals, 0 markers, decisions `f4958c88…`, adjudication summary.
- **Dispatch:** all six dispatch roots, each with its final receipt or stop-time progress file, and the two reconciliations.
- **OLAP:** both real load receipts (D0–D2 `ce6debba…`, D2 `e0b63c46…`), each with `historical_unchanged: true`.
- **Gate:** the refusal itself (`10ff4fe0…`).

**Health by role (C180)**, read by the POST after all work ended:

| role | memguard | batch slice cap | memory available | dispatch units left | failed units | GPUs responding | GPU handle errors | kernel GPU error lines, 24 h |
|---|---|---|---|---|---|---|---|---|
| COORDINATOR | active | 14 GiB | 19.7 GiB | 0 | 0 | 1 | 0 | 2 |
| WORKER_A | active | 14 GiB | 19.0 GiB | 0 | 0 | 1 | 0 | 0 |
| WORKER_B | active | 8 GiB | 10.3 GiB | 0 | 0 | 1 (GPU0) | 1 | 3 |

- **WORKER_B GPU1:** stays `QUARANTINED_NOT_SCHEDULABLE`; `nvidia-smi` still cannot obtain its handle. It was not used or repaired.
- **COORDINATOR:** its 2 kernel GPU error lines in 24 h are recorded, not acted on (this order uses no GPU).
- **logrotate:** remains the owner's task and did not block D2.

## 9. Branches, tips, digests

| repository | branch | final tip |
|---|---|---|
| predictor (this order) | `satoshi/c166-c184-20260913` | the commit that adds this packet; its parent is `cae5421` (test fix of fault 18), whose parent `a27c3e6` holds the POST v2 output. Nothing under `tools/` changed after `35cb46c`, the harness POST v2 ran on, and every count in this packet was read at or after `1c9ca14` |
| data-gov (owner-ordered, not this order) | `master` | `a307c17`, local, not pushed |
| financial-data `lake/` (owner-ordered, not this order) | `satoshi/c122-c145-20260912` | `1867d45d0`, local, not pushed |

Identities: design `3577c154…` (`D2V2_C171_R2_2026_09_13`), tape `230a0a44…`, lab code `1924dbdf…`.

Tests at the final tip `cae5421`:
- **D2 focal battery** (`tests/test_df_*.py`, `test_c139_data_foundation_olap.py`, `test_c164_runtime_grains.py`, the same command that gave 921 passed and 1 skipped at the merged tip): **930 passed, 1 skipped** in 360 s, run with the PostgreSQL credentials so the four database tests ran on throwaway databases. The skip is the deliberate contained-OOM test, gated behind `CRISPDM_ALLOW_DELIBERATE_OOM_TEST`. No OOM kill; no throwaway database left; loader untouched. The run before the fault-18 fix, at `a27c3e6` and without credentials, gave 925 passed, 1 failed, 5 skipped.
- **Owner-ordered data-gov side on this branch, not part of the order:** `tests/test_governed_run.py` plus `tests/test_governed_run_overrides.py`, 17 passed; `olap/lake/tests`, 14 passed and 1 skipped (the PostgreSQL write test, which needs a throwaway database URL).

Custody files under `~/.local/state/crispdm-data-foundation` (sha256, first 16 hex):

| file | sha256 |
|---|---|
| `d2_design_c171_v2/D2_DESIGN_V2.json` | `61872db71a4c9bcb` |
| `d2_design_c171_v2/SEAL_SUMMARY.json` | `955112bf7c9f652f` |
| `d2_design_c171_v2/C137_DISPERSION.json` | `692d0529f4115bab` |
| `d2_fresh_tape_c173_v1/SEED_TAPE.json` | `29487904365dab81` |
| `d2_fresh_reserve_c173_v1/ROOT_MANIFEST.json` | `918109021ff66f79` |
| `d2_fresh_c174_v1_collected/RUN_MANIFEST.json` | `90942903f9efddf1` |
| `d2_fresh_c174_v1_collected/DECISIONS.jsonl` | `f4958c88f8caa6b7` |
| `d2_fresh_c174_v1_collected/ADJUDICATION_SUMMARY.json` | `f5053162842da972` |
| `d2_fresh_c174_v1_collected_tables/TABLES_MANIFEST.json` | `16a152510fce6f52` |
| `d2_reanalysis_c172_v2_comparison/C172_COMPARISON.json` | `1dabc9b4327c58b1` |
| `d2_reanalysis_c172_v2_comparison/tables/TABLES_MANIFEST.json` | `87f41b8f5df0d417` |
| `d2_dispatch_c172_v1_reconciliation/RECONCILIATION.json` | `1f574512adce56b4` |
| `d2_dispatch_c174_workers_reconciliation/RECONCILIATION.json` | `ae555dfb6151eb13` |
| `d2_submission_c179_v1/GATE_REFUSAL.json` | `10ff4fe03fe6d9f3` |
| `d2_submission_c179_v1/D2_REVIEW_SUBMISSION.json` | `a59b5917d4996513` |
| `load_receipts_c178_v1/throwaway_d0d2_v2grains_c178_v1.json` | `1f6a7359530f04f1` |
| `load_receipts_c178_v1/real_d0d2_v2grains_c178_v1.json` | `ce6debbae639aff3` |
| `load_receipts_c178_v1/throwaway_d2_reanalysis_only_c178_v1.json` | `3db80175e8f8f728` |
| `load_receipts_c178_v1/throwaway_d2_c178_v1.json` | `a287aa263766dea7` |
| `load_receipts_c178_v1/real_d2_c178_v1.json` | `e0b63c46c9451ebd` |
| `c168_structural_mutations_v1/STRUCTURAL_MUTATION_SUMMARY.json` | `228d299fbc497884` |
| `causal_battery_c167_v2/CAUSAL_BATTERY_SUMMARY.json` | `26fa9654aa5bd418` |
| `c169_parity_reports/COMPARISON.json` | `2334e4d851a79402` |

## 10. What needs you

1. **Review of the fresh D2 evidence.** This is the only path to any D2 state. Nothing here is externally reviewed. The gate refuses all 14 subjects that reached LAB_CALIBRATED or REGIME_LIMITED, and D3 stays closed until your separate decision.
2. **The two historical flips.** In the C172 reanalysis, `local_linear_trend_kalman` moves from LAB_REJECTED to LAB_CALIBRATED in `multiband/white/−5 dB` and `sinusoid/student_t/20 dB`. The primary cause is TEMPORAL_MODE: the Kalman kinds now fit in `FROZEN_PREVIOUS_PARTITION`. Both belong to the non-confirmatory stratum and decide nothing.
3. **The worker change I made mid-order** (fault 8, `535420d`). A unit refused whole by the declared missing-data rule now abstains instead of invalidating the root, and I resealed the design before any fresh unit existed. You may rule that such a unit should still invalidate the root. Under that rule, the 64 fresh units refused whole would have invalidated the whole fresh root, and no D2 decision could have been adjudicated. The alternative would be a design that drops the three MCAR regimes, whose missing rate leaves no 50-row TRAIN stretch in most seeds. Under the rule as applied, those three regimes are NOT_IDENTIFIABLE, and the other 168 are unaffected.
4. **The previously rejected control passes in 2 regimes** (`trend_linear/white/20` and `inf`), the same two where C137 passed it. Whether a rejected control should be chosen from specs rejected in every regime is a design question for the next round.
5. **The duplicated v1 coverage matrix** (fault 13). Two cell-identical copies sit under one run id, told apart by `code_sha256`. The cube is additive, so I did not delete either. Tell me if you want a view that exposes only one.
6. **The UNDERPOWERED share.** 871 of 2,052 candidate × regime decisions are UNDERPOWERED. In 680 of them, the C137 effect does not exceed the null, so no seed count could power them. A future design could drop those specs from those regimes up front, rather than spending seeds on them.
7. **Cross-host reproducibility of the optimizer-based SNR estimator** (fault 17).
   - **What was found:** `local_level_kalman` SNR estimates re-derive exactly on the machine that produced them, but move by up to 0.006 dB between AMD Zen microarchitectures. In the one unit checked, 12 other numbers moved by at most 1.4e-10, eight of them by less than 1e-15.
   - **What was checked:** only one unit per worker was re-derived across hosts.
   - **Your call:**
     - declare a cross-host tolerance for optimizer-based estimators before the next confirmation;
     - require re-derivation on the producing role (what AT9b tests);
     - or pin the optimizer (fixed iterations and a deterministic linear-algebra path) and re-run.
   - **Current decisions:** the one affected decision is 13 dB or more from its threshold (mean |error| CI 14.4–29.1 dB against 1 dB). I have not measured the margin of the other 1,025 SNR decisions, nor re-derived their units on another host.
8. **Outside this order, at the owner's request:** the data-gov flow v2 (governed download with hash, metrics reported to the cube, lineage). Its proposal and open findings are in `data-gov/docs/05_PROPUESTA_MUSASHI_BETA_2026_09_13.md`. Its end-to-end check has not run, and the live services on 5055–5057 still run the old code.
