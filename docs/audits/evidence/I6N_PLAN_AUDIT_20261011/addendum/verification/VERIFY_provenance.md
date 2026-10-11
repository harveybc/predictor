# Adversarial verification: addendum lane `provenance` (I6N-AUD-01, owner points 1 and 2)

Verified on 2026-10-11 (UTC). Lane files are under `addendum/provenance/` in the audit worktree, which is uncommitted on top of 4d3af09d. I did not modify any lane file.

## 1. Rerun

The rerun followed COMMANDS.txt exactly: CPU, crispdm-run with 2G and 15m, `--draft` pointing at the I6-D weekly worktree, `--commit 269c2d13`, and INPUTS written to this scratch directory. It exited 0 with empty stderr and finished at run_utc 2026-10-11T12:14:22Z.

`diff` against the retained `provenance_check.out` shows four differing lines:
- line 1: run_utc. Expected.
- line 30: WEEKLY_PLAN built_utc. Expected and declared.
- line 134: the WEEKLY_PLAN file sha256. Expected and declared. The retained run had 9729e873, the rerun ae1cb625, and plan_sha256 ef7ffd58 recomputed OK both times.
- **line 57: the order of the `selection targets/horizons` list differs** (`Y_l_144h`/`Y_b_l144` swapped). The script runs `sorted(set(...), key=horizon)`, and the order of tied elements follows set iteration order, which depends on PYTHONHASHSEED. I reproduced this with a 4-element toy under seeds 1-4. The same applies to the h=6 tie.

This refutes the COMMANDS.txt sentence "A rerun reproduces every line of provenance_check.out except run_utc, the built_utc value and that file's sha256". The content is unchanged, but the retained output is not byte-reproducible. Minimal fix: use `key=lambda x: (x[1], x[0])`, or pin `PYTHONHASHSEED=0` in the command.

INPUTS.json differs only in the WEEKLY_PLAN sha.
- Rerun output sha256: c85b3857b4f68a93a5ef0cb1cfbe36080f525280e5ca398f5f4327bf3afed29f
- Rerun INPUTS.json sha256: 44d1e1bda07e197a2e2198a5ea3a3377d4d516fc1b8f38f1567a2bdd97e08fac

## 2. Hashes

| file | listed | recomputed | match |
|---|---|---|---|
| provenance_check.py | d7999aae… | d7999aae… | yes |
| provenance_check.out | d32292a5… | d32292a5… | yes |
| INPUTS.json | 383a6e9c… | 383a6e9c… | yes |
| 21 inputs (rerun) | as in INPUTS.json | identical | yes, except WEEKLY_PLAN (declared in-place rewrite) |

The draft proposal sha d4341e72 is unchanged.

**New observation.** `$STATE/.../wave2_successor/EURUSD/FRONTIER_SEAL.json` is also rewritten in place, by the same fs4-closure timer cycle; its mtime is 2026-10-11 12:15Z. Its sha (450ab297) did not change, but COMMANDS.txt names only WEEKLY_PLAN as rewritten. The consequence is that the file's mtime cannot date the frontier. See check 5b.

## 3. Classification review

| # | lane claim | lane class | verdict | note |
|---|---|---|---|---|
| 1 | 20 features = union of 14 FS4 EURUSD RAW winners, bound by digests | DEMONSTRATED_DEFECT | CONFIRMED fact, MISLABELLED | The evidence text itself says "not itself a defect". It should be read as a verification with no defect. In a results table, this label would make a provenance confirmation look like a defect. I re-derived the set equality independently (separate code, TEST_FREEZE winners vs parent DESIGN members): True, 20 members, 14 targets. |
| 2 | Draft and identities do not declare VALIDATION-2024 selection | DEMONSTRATED_DEFECT | CONFIRMED, with a small overclaim | The phrase "I6-B manifest has … no field that records the selection's provenance" is too strong. `freeze_sha256` is a provenance pointer, and claim 1 relies on it. What is missing is an explicit selection split/year field. The "only I6A:51 admits it" result comes from a narrow regex. A broader `git grep` at 269c2d13 over docs and tools also found no other declaration, so the claim stands once that scope is stated. |
| 3 | Severity overstated: TRAIN-only frontier of 2 per target, 14 binary VALIDATION choices, 12 features only via RANDOM_K, all winners below naive | RETRACTED (severity) | CONFIRMED | 24 of the 28 EURUSD sets are RANDOM_K, whose `control_methods` is RANDOM_K. Scores range from -0.0254 to -0.3090. The original FINDINGS text ("selected on VALIDATION 2024", CRITICAL/CONTRADICTION) is quoted fairly; what is withdrawn is the severity, not the fact. Weak point: the lane's evidence for "before any VALIDATION read" is FRONTIER_SEAL's self-declared `validation_read.count=0` in a file regenerated today. My independent check (5b) closes this gap. |
| 4 | 2024 VALIDATION results with these features are DEVELOPMENT_ONLY | DEMONSTRATED_DEFECT (declaration missing) | CONFIRMED | Both v1 and v2 have 53 segments from 2024-01-01 to 2025-01-01, re-derived. This is consistent with owner point 2: development use is allowed, and the only defect is the missing declaration. The scikit-learn quote is accurate but is not retained in any file. |
| 5 | Look-ahead in 2023 screens: features chosen with 2024 data and ranked on folds that include inner_2023 | RISK, with a "demonstrated" sub-part | CONFIRMED, with an attribution gap | The lane attributes inner_2023 to the git copy of `FINAL_SELECTION_MANIFEST.json` (7eca0c1d). That digest is **not referenced** by the frontier, the phase-3 closure or the consolidated candidates, so the retained link is a role annotation, not a digest chain. The claim is still true through a chain I verified (5c). The ranking-bias part is correctly left as RISK. |
| 6 | TRAIN-only manifest is optional | METHODOLOGICAL_ALTERNATIVE | CONFIRMED | This respects the owner's ruling: no automatic redo. No early-stopping or seed requirements are introduced anywhere in the lane. |
| 7 | TEST 2025 untouched in the listed files | DEMONSTRATED_DEFECT | CONFIRMED fact, MISLABELLED | This is a no-defect verification, as the lane admits. The scope is limited to INPUTS.json, as stated. Label time is computed as decision + h wall-clock hours. That assumption is consistent with WEEKLY_PLAN.support, where available_time equals decision time at the bar end. It is also consistent with the censoring pattern: the last finite Y_l_72h is at Fri 22:00, and t+72h is Mon 22:00, whereas a bar-count horizon would end earlier. The claim still depends on that convention and should say so. The original major "Frontera VALIDATION/TEST: the last week of 2024 has targets in 2025" was neither qualified nor retracted here. For this parquet, labels beyond 2024 are censored (non-finite). The original major should be explicitly qualified, either in this lane or in the calendar lane. |
| 8 | I6-E-M segment 52 [2024-12-30, 2025-01-01) overlaps FS4 sealed TEST week 52 | DEMONSTRATED_DEFECT | QUALIFIED | The overlap is demonstrated. Under the owner's definition (TEST is 2025 onward), however, scoring those two days is legitimate development. The harm only arises if a later FS4 TEST confirmation reports week 52 without excluding them. The overlap should be filed as a demonstrated protocol inconsistency (minor), with the consequence filed as a RISK, not as a defect of I6-E-M. The minimal correction (declare one boundary and annotate the FS4 freeze annex) is appropriate and re-runs nothing. |

No claim overrides an owner decision: early stopping, mandatory seeds and an automatic redo of the selection are all left alone.

## 4. Privacy

I grepped all four lane files for /home/, the username, the email domain, IPv4 addresses, UUIDs and GPU-UUIDs, token, password and secret, known host names, and the machine's hostname. There were 0 hits. Paths are `$STATE/...` or `repo@commit`. Clean.

## 5. Independent refutation attempts (all read-only, VALIDATION or metadata only, no TEST rows)

a. **Set equality**, recomputed with separate code: True (20 members, 14 targets).

b. **Was the frontier fixed before the first VALIDATION read?** In the FS4 task store (read-only), all 4,368 EURUSD tasks carry the payload `plan_sha256 = ef7ffd58`:
- stage 1 RAW: 1,456 tasks;
- stage 2 RANDOM_ENCODER: 1,456 tasks;
- stage 2 TRAINED_ENCODER: 1,456 tasks.

The EURUSD task set_ids equal FRONTIER_SEAL.frontier_set_ids, which is 28 sets. The first `started_at` is 2026-10-07T18:36:38.9Z, the same moment as the first VALIDATION read. The plan digest includes `populations.EURUSD.frontier_seal_sha256 = 5a0edbfb`. So the 28-set frontier was bound into the tasks that performed the first read. This **supports** claim 3 with evidence stronger than the self-declaration. The lane should retain this check.

c. **inner_2023 in the rankings.** The digest chain is:
- FRONTIER_SEAL.inputs.phase3_closure_sha256 = b1fed3cc;
- which equals `$STATE/canonical_20261003/fs_phase23/eurusd/PHASE_3_FILTER_COMPLETE.json` closure_sha256;
- whose phase2_closure_sha256 is d3aca4df;
- which equals PHASE_2_COMPLETE.closure_sha256;
- whose fold_ids include inner_2019 through inner_2023.

This confirms claim 5's fold statement. The lane should cite this chain in place of the unlinked git copy of FINAL_SELECTION_MANIFEST.

d. The **broader declaration grep** at 269c2d13 found nothing beyond I6A:51. This supports claim 2.

## 6. Required minimal corrections to the lane (none invalidate its conclusions)

1. Make the script output deterministic: tie-break the sort, or set PYTHONHASHSEED. Then rerun and refresh the .out sha, and correct the reproducibility sentence in COMMANDS.txt.
2. Note in COMMANDS.txt that FRONTIER_SEAL.json is also rewritten in place, and identify it by `seal_sha256`.
3. Relabel claims 1 and 7 as verifications with no defect, for example "CONFIRMED (no defect)" in the per-finding table. The vocabulary limitation is not a reason to print DEMONSTRATED_DEFECT.
4. Recast claim 8 as a demonstrated inconsistency plus a RISK.
5. Soften claim 2's "no field that records the selection's provenance" to "no field that declares the selection split/year; freeze_sha256 points to it".
6. Add checks 5b and 5c to the script: the task-store plan digest with the first started_at, and the PHASE_3 → PHASE_2 fold chain.
7. In this lane or the calendar lane, state the effect on the original major "last week of 2024 has targets in 2025". For this targets file the labels are censored, under the bar-end wall-clock convention.

## Process note

I ran several small read-only `python3` inspections of JSON and sqlite metadata directly, not under crispdm-run. These included the task store queries and the phase-2/3 records. The full rerun and the tie-order demonstration did go through crispdm-run. Nothing was written outside this scratch directory, and no service, timer or GPU was touched.
