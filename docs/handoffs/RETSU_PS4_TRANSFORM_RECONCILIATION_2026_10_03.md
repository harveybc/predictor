# Retsu: reconcile transform variants with measured feature profiles

Owner: Retsu. This is one bounded CPU/data task. Do not start a GPU job, fit a
model, change an existing campaign, or claim a selected feature. The goal is to
replace an ambiguous `PENDING_PROFILE` statement with a verifiable join between
variant identities, their emitted features, existing PS1 metric cells, and PS2
statuses. PS4 *expanded* profiles remain pending unless already evidenced.

## Start here, not from your previous branch

The canonical remote branch is `origin/satoshi/canonical-exec-20261003`. Your
earlier report named `eff930ae`, which was correct at dispatch but is now stale.
Commit `749ba6a8` is already integrated byte-identically as `292e13cc`; the
one-pair PS5 pilot is `d5d8d077`; the current execution record is `90866f7c`.
Do **not** cherry-pick `749ba6a8` again or edit the canonical worktree.

In the predictor repository, run these checks before editing anything:

```bash
git fetch origin satoshi/canonical-exec-20261003
git rev-parse origin/satoshi/canonical-exec-20261003
git merge-base --is-ancestor 292e13cc origin/satoshi/canonical-exec-20261003
git merge-base --is-ancestor d5d8d077 origin/satoshi/canonical-exec-20261003
git status --short
```

The second command must return `90866f7c0f5348a36d18fa3cea0516e3a85c29ce`
at the time of dispatch, and both `merge-base` commands must exit zero. If
origin has advanced, record the new tip and continue only if those two commits
remain ancestors. If any command fails, return the literal command and output;
do not repair by reset, force-push, or guessing a different base. Make a new
worktree and branch from the verified remote tip. Preserve existing dirty trees:

```bash
WT="$HOME/Documents/GitHub/.worktrees/predictor-retsu-ps4-transform-20261003"
test ! -e "$WT"
git worktree add -b retsu/ps4-transform-join-20261003 "$WT" origin/satoshi/canonical-exec-20261003
cd "$WT"
git rev-parse HEAD
```

If `test ! -e` or `worktree add` fails, stop and report. Do not delete a path
or reuse a branch to make the command pass.

## Exact inputs and denominators

All paths below are relative to the verified predictor worktree:

- `docs/audits/evidence/canonical_20261003/source_transform_coverage/transform_coverage.csv`:
  exactly **9 variant rows**, 5 `ADMISSIBLE_CAUSAL_COMPUTABILITY_ONLY`, 4
  `NOT_ADMISSIBLE`. This is a variant ledger, not a feature ledger.
- `docs/audits/evidence/canonical_20261003/laneA/batch_001/transform_variants.csv`:
  the measured prefix probes and original variant definitions.
- `docs/audits/evidence/canonical_20261003/laneA/batch_003/admissible_features.json`:
  emitted feature IDs. Its `transform` field names the variant that emitted each
  output. The five admissible variants emit **10** feature IDs in total: five
  wavelet levels, one multitaper band, one Hilbert amplitude, two STL outputs,
  and one Kalman deviation. The four rejected global/smoother variants must
  emit **zero admitted IDs**.
- `docs/audits/evidence/canonical_20261003/laneA/batch_003/profile_cells.csv`:
  existing PS1 cells. `tools/eurusd_ps/profile.py` at
  `feature-eng@1f1abdf` defines 11 metric keys. Do not rerun them just to
  make a new table.
- `docs/audits/evidence/canonical_20261003/laneB/batch_003/ps2_status.csv`:
  reversible target/horizon status, **not** a final selection.
- The batch 003 `digests.json` and PS2 manifest bind retained input bytes. Check
  every available SHA-256 before interpreting rows. A missing or changed input
  is `INCOMPLETE_EVIDENCE`, not zero or `NOT_APPLICABLE`.

The exact emitted IDs can be cross-checked against
`feature-eng@1f1abdf:tools/eurusd_ps/run_batch3.py`, function
`variant_features`. Do not infer variant identity solely from an abbreviated
column name if the structured `transform` field contradicts it.

## Deliverable and tests

Write an additive generator and tests under
`docs/audits/evidence/canonical_20261003/ps4_transform_join/`. Produce:

1. `transform_feature_join.csv`: one row per `(variant_id, emitted_feature_id,
   metric)` for admitted outputs, plus explicit zero-output rows for the four
   rejected variants. Include `variant_state`, `feature_admissibility`,
   `ps1_metric_state`, `ps1_metric_version`, `ps2_status_by_target_horizon`,
   `ps4_expanded_state`, and input digests. Do not turn an absent cell into zero.
2. `REPORT.json`: separate counts for **9 variants**, **10 emitted features**,
   and the expected **110 PS1 metric cells**. Count measured, failed, pending,
   and not-applicable cells separately. State whether PS1 profile coverage is
   complete; keep PS4 expanded profiling `PENDING` if no such evidence exists.
3. Tests written before the generator: exact 9/5/4 and 10 counts; one missing
   metric cell; duplicate feature ID; an admitted feature mapped to a rejected
   variant; altered digest; an unrecognized variant; one absent PS2 status.
   Every mutation must yield a named refusal or incomplete state, never a
   silently shortened denominator.

Use `csv.DictReader`, `json`, and `hashlib`, not text slicing of CSV/JSON.
Run only the focused tests under `crispdm-run -m 2G -t 120s`; if that wrapper
is unavailable on the chosen host, report it rather than silently running
unbounded. No global suite, no GPU, no model refits, no test/holdout read.
After writing the test and again after implementation, use:

```bash
crispdm-run -m 2G -t 120s -n retsu-ps4-transform-join -- python -m pytest -q docs/audits/evidence/canonical_20261003/ps4_transform_join/test_join_transform_profiles.py
git diff --check
```

The first run should fail for the intended missing behavior; retain that PRE
result. The final run must pass. Then commit only the new join directory and
push only `retsu/ps4-transform-join-20261003`.

## Acceptance and return

The return must contain the exact base and final commit SHA, remote branch and
push verification, focused test command and result, SHA-256 of each output,
the 9/10/110 denominators, and a compact table of measured/failed/pending/NA
PS1 cells. Say `NO_NEW_MODEL_MEASUREMENT`. Name PS4 and selection as pending.
Report any real contradiction between batch 003 metadata, profiles, and the
variant ledger as a finding; do not rewrite historical evidence to make it fit.

Do not edit `CURRENT_EXECUTION.md`, `STATUS.json`, progress PNGs, Gamma queues,
source credentials, or services. The coordinator integrates your branch after
review. In parallel, Gamma E/F continue; your task must not wait for them.
