# Independent RAW Long Release Review

**Verdict: REVISE. Do not launch the GPU arms from this snapshot.**

Reviewed 2026-10-15 against `a253a2f2b20708bc1e7a7530ac408563910a97ed`,
including the uncommitted scoped core, raw_pair, timer templates, new tests,
and, at the owner's subsequent request, raw_long_campaign and its tests.
Read RAW_PRIORITY_20261015, CONTROL_PROTOCOL, DATA_MODEL_RETURN and RAW_PAIR_HOOK.
No source fixes were made. This document is the only repository edit by this reviewer.

## Findings

### F1 - P1: Pair execution ignores contradictory C2 and incomplete short support

Locations: `tools/i6dmo/raw_pair.py:69`, `:82`, `:87`, `:262`.

`_report` checks matched labels and some numeric agreement, but never validates
the top-level paired-metrics schema/split/scale, `agreement.status`, C2 state
or equality with `child.lineage.c2`. It explicitly permits `n < n_declared`
and ignores `n_failed`/head state. The short selection is not revalidated by
`_comparison`, which only checks the long reports.

Counterexamples, using the existing test_raw_pair fixtures with rehashed
METRICS/CHILD/stored envelopes, without changing any production artifact:

- A short h2 with n=2, n_declared=3, n_failed=1, state=INCOMPLETE, a declared
  INCOMPLETE selection, C2 INCOMPLETE and agreement DISAGREE still yields
  `gates.pair_status=ELIGIBLE`; only absent execution pins stop it.
- With the real synthetic positive execution fixture, changing the short
  METRICS C2 to INCOMPLETE and agreement to DISAGREE still produces
  `COMPLETE / PASS / pair_status=PASS` and executes the existing backtest.

Minimal fix: validate original paired-metrics/C2 metadata against lineage,
require scoring agreement, and refuse incomplete or inconsistent support as
a technical failure before eligibility. Require zero failures and complete
declared support for any head admitted to trading. Do not equate complete
reporting with all six heads beating naive. Add both negative controls to
the actual positive and negative hook paths.

### F2 - P1: Launcher status and post-deploy return always raise

Location: `tools/i6dmo/raw_long_campaign.py:48`.

`status.compute(..., level="campaign")` already returns a `state` key.
`dict(state=..., **doc)` therefore raises
`TypeError: dict() got multiple values for keyword argument 'state'`.
Bounded probes reproduced this for CLOSED and RUNNING states. This affects
status, existing-campaign resume and the return after a successful deployment.
Consequently `main` cannot reach its CLOSED timer-disable branch. The existing
resume test replaces `_existing_status` entirely, hiding the error.

Minimal fix: merge the returned mapping and set/normalize state once. Exercise
the real function for both states and verify CLOSED disables only its launcher
timer through a mocked systemctl.

### F3 - P1: Successful long outcomes cannot automatically close

Locations: `tools/i6dmo/campaign.py:1203`, `:1618`;
`tools/i6dmo/raw_pair.py:328`.

Readiness exposes MG1_LONG_SELECTED, MG12_SELECTED or
MG12_RETAINED_WITHIN_TOLERANCE for comparison-only long campaigns, and PASS
after a successful raw pair. None is in AUTO_CLOSE_DISPOSITIONS.
Direct closure_problems probes returned
`DISPOSITION_NOT_AUTO_CLOSABLE:<outcome>` for all four, while
PAIR_NOT_ELIGIBLE passes this particular gate. Publication and mirrors cannot
remove this unconditional blocker. Successful science is instead eligible
for the generic technical-terminal handling, without a CLOSE receipt.

Minimal fix: introduce a mode-specific successful RAW-pair closure disposition
and support all declared long outcomes without making arbitrary technical
FAIL/PASS strings universally closable. Verify positive, negative, tie and
comparison-only closure with exact outbox readback and both mirrors stubbed;
verify failed/missing artifacts remain technical, not scientific negatives.

### F4 - P1: Runtime deployment does not enforce reviewed bytes on resume

Locations: `tools/i6dmo/raw_long_campaign.py:184`, `:190`, `:209`.

An existing runtime directory is accepted solely because Git HEAD matches.
A dirty tracked model/worker file or untracked import override leaves HEAD
unchanged. The release manifest is checked in the source checkout, not in the
actual executing worktrees. Mocked provisioning recorded only directory
existence and `git rev-parse HEAD` checks on the existing-runtime path; there
was no runtime manifest or cleanliness check. Data rsync checksums do not
authenticate runtime code. Additionally an existing PLAN takes the deploy
branch before verify_release, with no verification of this launch config's
association with that retained campaign.

Minimal fix: authenticate the frozen review manifest and import environment
on every executing role before enabling workers, including existing worktrees.
Reject dirty/mismatched runtime bytes rather than resetting them. Bind resume
to the original prepared config, mode, code and deployment; preserve fit
identities and never initialize a replacement campaign implicitly. Add dirty
same-HEAD and wrong-retained-campaign refusal tests with remote calls mocked.

### F5 - P2: Interrupted preparation cannot adopt its completed population

Location: `tools/i6dmo/raw_long_campaign.py:140`.

The population-exists/PREPARED-missing recovery branch calls
`CP.load_common_population` without its required keyword-only expected_digest.
Binding this actual call against the function signature raises
`TypeError: missing a required keyword-only argument: 'expected_digest'`.
A crash after staging.rename and before PREPARED writes deterministically
enters this branch on every subsequent autonomous retry.

Minimal fix: durably retain the expected population digest before promotion,
then supply and validate it on adoption, including arm seals and source pins.
Add a restart test at the rename/PREPARED boundary; do not invent a new trusted
digest merely because an orphan directory's own JSON declares one.

### F6 - P2: Long comparison trusts cached score and horizon declarations

Locations: `tools/i6dmo/reporting.py:629`, `:645`, `:649`, `:657`.

The new long check validates selection.heads, not the actual per_horizon keys
or recomputed score. With genuine literal capacity records and matching
population metadata, deleting mg1l's actual h72 row still returns
LOWER_SCORE / mg1l, with h72=null. Changing only mg12m's cached selection score
from -0.01 to -0.9 reverses a comparison against mg1l=-0.02, despite unchanged
per-head values. Both outcomes were reproduced without fitting.

The later raw-pair verifier catches these particular long inconsistencies;
that limits trading exposure but does not make the official comparison or
its published decision valid, and comparison-only campaigns lack that check.

Minimal fix: require the exact six real rows, nonempty common support identities,
per-head complete support/count consistency and finite own-naive errors, then
recompute selection_score and compare it with declarations. Reuse that same
validation in the official comparison and raw-pair verifier.

## Additional Integration Gaps

- `campaign.py:471` uses MATCHED_STAGES, but `:698` assigns raw.pair the new
  RAW_PAIR_ELIGIBILITY_AND_STRATEGY stage, absent from that list. A real
  six-unit factory probe found raw.pair absent from all reported stage totals.
  Add the stage for this mode without changing old short identities.
- The launcher retry timer only self-disables on CLOSE. Its existing-PLAN
  branch redeploys every invocation, including after campaign technical-terminal
  finalization; this can re-enable timers the campaign deliberately stopped.
  Distinguish preflight resource retries from retained technical failure, and
  do not repeatedly redeploy an already-dispatched or finalized campaign.
- strategy_inputs is still helper-owned/in progress and is **not certified**
  here. At inspection, its receipt uses status=PREPARED and outputs, while
  launcher `:157` requires VERIFIED and `:160` reads artifacts. Freeze one
  interface and test actual helper output through launcher preparation. Bind
  the reconciliation to this INPUT digest, inherited calendar and approved
  historical archive digest, not just an asserted status and bar-file hash.
  The reported fresh download/exact-old-SHA match was not independently rerun;
  no network was used. The instrument-envelope fix at launcher `:163` uses
  the existing loader/converter and is consistent with that adapter API.

## Verified Scope

- Exact supported arm pairs; MG1 family recognition includes slots AND physical
  heads. 145/13 observations span 144 market hours, include actual t, retain UTC
  offsets/closure flags, and leave targets at exact elapsed 18/24/36/48/60/72 h.
  Synthetic missing-bar and future-perturbation controls pass.
- Common FIT/INNER/DEVELOPMENT head masks, maturity purge, FIT-source-union
  scaler and FIT-only per-head naive scales pass focused replay/perturbation
  checks. No DEVELOPMENT-dependent scaler refit was observed.
- Literal capacities are 30,226 and 4,882, delta 25,344 in readouts. Actual
  untrained graphs and negative causal probes pass, not merely a registry
  equality test. No fit was executed.
- Old short seals and closed identities replay unchanged. Long factory probing
  produces MATCHED_LONG_CONTRAST and six units: two fits, two reports,
  comparison, raw.pair. Long arm view designs reach unit_runner correctly.
- The original retained mg2m.report authenticated at identity
  dda1e2270c8eaff01ad5469cb0c64f47d659dd85a8640dfc4338bc3d93bdd1e0.
  METRICS and lineage C2 agree and are COMPLETE; agreement=AGREE; all head
  states COMPLETE; eligible heads 4/6/12. A Path.open tripwire prohibited NPZ
  reads. prediction_bytes_verified remained false, as documented for negatives.
- Negative pair tripwires, own-naive replay, positive synthetic execution,
  digest tampering, missing-artifact technical failure and TEST-window refusal
  pass the existing focused tests. These do not eliminate F1/F6.
- Confirmatory provenance still refuses with PROVENANCE_UNRESOLVED. Preserve
  current source-age/clock qualifications: a slot at t does not certify feature
  publication freshness. The FRED addendum says 339/1,587 NOT covered and
  1,248 covered for each series, not the reverse, and does not prove look-ahead.
  No TEST row, prediction array from the retained run or real strategy bar was read.

## Execution Evidence

All executed checks used CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE=1,
bounded CPU threads, and `crispdm-run -q -m 2G -t 180s`. No admission refusal
was retried at a lower cap. No agent, GPU job, fit, service change or network
operation was started by this review. Remote/systemctl deployment probes were
mocked; temporary synthetic artifacts were discarded.

| Check | Observed result |
| --- | --- |
| pytest --noconftest -q -p no:cacheprovider test_raw_pair.py test_raw_long_operations.py matched_long/test_data_design_model.py | 43 passed, 2 protobuf deprecation warnings, 14.99 s |
| Two existing test_matched_population.py seal/closed-identity checks | 2 passed, 1.05 s |
| test_raw_long_launcher.py, same pytest options | 3 passed, 0.19 s; its status path is mocked |
| Independent core adversarial probe | Exit 0; F1 partial short, F3 and F6 reproduced; original short metadata authenticated |
| Launcher/factory probe | F2/F5 reproduced; real six-unit graph and omitted stage verified; then harness KeyError for fixture missing coordinator; not called a passing script |
| Corrected final mocked-runtime/positive-C2/provenance probe | Exit 0; F4, COMPLETE/PASS despite bad short C2, and closed TEST provenance gate reproduced |

One initial python-stdin invocation through the governor returned without
running its script; it is not evidence. The subsequent python -c probes above
executed and produced the stated results. This is **48 focused pytest passes**,
not a full-suite certification. The supplied earlier 13/32 claims are not
added to this count. The earlier broad run timed out, not passed. The legacy
short-only capacity snapshot assertion needs a short projection after adding
the long registry entries; old evidence must not be rewritten to conceal that.

## Snapshot And Release Conditions

Core files were stable across this review. Principal SHA256s:

| File under tools/i6dmo | SHA256 |
| --- | --- |
| campaign.py | 562f0d2e94819379398f3c38c5465968fb8ebe7e8ba9f1adbdbf91a6b2235a11 |
| reporting.py | 1ee220bdfa6b2e435701b4516d08d76488070391b2612e489f92484ba8423e73 |
| raw_pair.py | bbfef6b75dd5ab58ce89aeb5307f91204e88e1a0d1b2a0fc035490d4224b405c |
| raw_long_campaign.py | 53272a7b04d52d57f5a16b6ddc9c0ead8872c33818c3bd8b878070f4c1a23c7d |

This is not RELEASE_ACCEPTANCE.json and grants no execution authority. Correct
the findings, freeze the completed source-input contract, and obtain focused
independent re-review of changed bytes plus the complete positive/negative
factory-to-publication/mirror/CLOSE path before GPU admission. Stop this review
here; no campaign polling or automatic re-review is implied.

## Release Addendum - 2026-10-15

**Current verdict: ACCEPT for the reviewed code and finite execution procedure
at the hashes below (ACCEPT_FOR_EXECUTION).** This supersedes the initial
REVISE for that source snapshot. It is not a claim of scientific performance,
successful production admission, a full-suite pass or permission to open TEST.

This requested re-review covered the F1-F6 changes, stage accounting, finalized
resume behavior, completed strategy-input helper and a newly exposed OLAP
emission error. No implementation was edited by this reviewer. The independent
regression file was expanded and this addendum was appended.

### Findings Resolved

| Finding | Independent recheck |
| --- | --- |
| F1 | The shared original-metrics/C2 validator rejects missing/incomplete C2, disagreement, changed child C2 and incomplete support before prediction/strategy reads. The positive-path tripwire passes. |
| F2 | The real _existing_status function merges state once; CLOSED and running-state controls pass without replacing the function. |
| F3 | Positive, negative and tied comparisons pass through the real long factory, queue terminal storage, unit_runner hooks, comparison, raw.pair, OLAP outbox/readback, mirrors and automatic CLOSE. Only COMPLETE/PASS raw.pair maps to RAW_PAIR_VALIDATION_COMPLETE; the long dispositions are mode guarded. |
| F4 | Same-HEAD dirty runtime refusal, isolated import/source attestation and manifest checks pass. Four actual temporary-Git tests pass, including ignored .pyc and a committed byte/manifest mismatch. Resume tests refuse changed config/population/raw-pair/deployment/source-spec identities and attest before init-before-deploy recovery. Already-deployed and technically finalized campaigns are not redeployed. |
| F5 | Recovery uses the durable POPULATION_BUILD expected digest, not an orphan's self-declared value. The checkpoint/input/design-pin fixture reaches verified loading. Dirty-source prepare is refused before materialization. |
| F6 | The official comparison refuses a missing actual head, forged cached score and missing support identity, using literal capacity records and valid controls. |

**Additional P1 found and resolved, F7:** raw_pair appended per-head eligibility
as Python bool. The real OLAP normalizer refuses that with
`ValueError: value must be a number or null`. The initial finite chains all
reached this REPORT_REJECTED boundary; those were not CLOSE-logic failures.
The implementation now converts only eligible values to int, retaining their
bool unit. Both the direct real-normalizer regression and all three finite
factory-to-CLOSE scenarios pass. No test-side normalization bypass was used.

### Finite Fixture Evidence

The positive and tie fixtures execute the existing pinned strategy on synthetic
arrays/bars; the negative fixture has prediction/bar/calendar/policy/backtest
tripwires. Long comparison and RAW-pair hooks run through unit_runner and the
actual worker terminal validation/storage/enqueue path. Fit terminals are
explicitly synthetic seeded evidence: no trainer or fit is executed.

Each scenario verifies six declared units, no TEST units, complete stage
membership, zero report rejections, exact governed metric readback, automatic
CLOSE, immutable closure-input proof, matching CLOSE bytes in two real temporary
mirror directories, mocked campaign-only timer stopping, and idempotent publish
re-entry without changing CLOSE. The warehouse is in-memory and systemctl is
mocked. This proves the finite plumbing, not physical-host/network availability
or actual trained-model eligibility.

The strategy-input helper was reviewed at its completed fixed-file interface.
It verifies the supplied archive before ZIP parsing, uses observed minute OHLC
and the inherited EU clock, refuses missing/mismatched retained 2024 closes,
retains calendar bytes and uncertainties, binds source/input/helper/output
hashes, refuses TEST timestamps before retained price loading, and publishes
without overwriting existing evidence. The launcher now consumes PREPARED /
outputs and binds the receipt to the original PREPARATION archive SHA, INPUT,
calendar and generating helper code. No new blocking defect was found in this
reviewed source path. The operator/helper's 6243/6243 exact reconciliation is
reported real-data evidence; this reviewer did not independently reread or
regenerate those private production bars.

### Final Bounded Verification

One final combined invocation on the hash-bound source below:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
crispdm-run -q -m 2G -t 180s -n i6dmo-final-hash-bound-review -- \
python -m pytest --noconftest -q -p no:cacheprovider --tb=short \
  tests/i6dmo/test_raw_long_review_regressions.py \
  tests/i6dmo/test_strategy_inputs.py \
  tests/i6dmo/test_raw_runtime_attestation.py \
  tests/i6dmo/test_raw_long_launcher.py
```

**76 passed, 2 protobuf deprecation warnings, 10.63 seconds, exit 0.** This
includes 32 independent regressions and 44 source/runtime/launcher checks.
The critical changed-file hashes were unchanged across this final run. An
earlier launcher snapshot changed during the preceding split checks; its
additional package-version/PYTHONHOME hardening was read and the combined run
above was repeated on the final launcher hash. No earlier red run, harness
failure, timeout or separately reported parent suite is counted as this pass.

### Accepted Bytes

SHA256, paths relative to the checkout. Changes to these bytes invalidate this
code acceptance until reviewed; the parent release manifest must match them.

```text
ec49003bd3a52c709070ffbe931a169b0e289d97fc0cf6c15fa46ec0747ca83e  tools/i6dmo/design.py
1d162c106f8cccb2aeea2bf81c1a97726d36ce12988673d1775575fc456b880b  tools/i6dmo/common_population.py
84492d7c4bcabbef362ac67125e0c93c84df9ba9f1811633c1544f7cff159347  tools/i6dmo/model.py
4587b312728880314489481a4234f2a79b6f7100d8da279d411bed2ea44f7cbd  tools/i6dmo/train.py
80816be2b4fd517722227d3099c8f0d5d63f7e853687126dfcd02862956bfb60  tools/i6dmo/campaign.py
daca1b7b708cf67a7c60d5f34a68a2af5dc63c4c6bd5e0e296ebff8309c8ad92  tools/i6dmo/matched_runner.py
9c49b45e101ce8346b9610fbbe36d54f9f24d0b3aad0d48f4c9a2587281a4576  tools/i6dmo/reporting.py
95b251c8d419bef622206f2b1d1d6be79dc42d0de0a184adf98589af29f27a7c  tools/i6dmo/worker.py
115637efe345523035a5c33fa1627c9f21ca805c428f70605218d43f4e0d8c23  tools/i6dmo/unit_runner.py
59d31e56def89bff1d19ca2e4292badc27e720c8e315695b201f773c831c3296  tools/i6dmo/cli.py
dfd345d80756656c29a84371bc9c146db29ad63d5c9d39d28d8fff5231798107  tools/i6dmo/deploy.py
1bbb7377630167d39ce173e608f5a07a47b0f5b834ff0822ff2f384d41fbe3e3  tools/i6dmo/raw_pair.py
366caf4d1005037319f9ed8563154b72f0cbc8ce9df2430f5d16ff6b67c212cd  tools/i6dmo/raw_long_campaign.py
59c0a36b80b7af37c80fd644e99a221fc5c219ba3c2444e54811d5b0c2816504  tools/i6dmo/strategy_inputs.py
a9f83fb2ba3318bdd99f7ed32a4020a2ee5dec6feeba32b9fa340f76141b88ef  tools/i6dmo/systemd/i6dmo-publish.timer.in
96a299f3ddc9af29d3f3497a08e260630e29cc207852d14284e58c73445bdfd3  tools/i6dmo/systemd/i6dmo-tick.timer.in
4ef5c7e37daf1487f302cdf16cbb45b3a5d428ced887424a014e032eb6155697  tools/i6dmo/systemd/i6dmo-worker.timer.in
d1a6cafb51166717775237bda822997ea6e1a058fcb9e42a5204a78b27921b7e  tests/i6dmo/test_raw_long_review_regressions.py
```

### Execution Conditions

Preserve the first engineering population as NON_GOVERNING. Commit the reviewed
source without changing these bytes, then generate the governing population in
the new v2 root under that frozen commit and retain its generation/digest pins.
Do not adopt the engineering population as proof of final-commit generation.
Use the existing release manifest, per-role runtime/source attestation and fresh
RAM/VRAM/disk/admission gates before enabling scientific work. The 6-GiB fit
caps and coordinator reserve are not relaxed by CPU-test success.

Retain the original short report and closed evidence unchanged. Source-age,
FRED publication-clock and assumed-calendar limitations remain; TEST stays
closed. One seed and synthetic backtests do not establish predictive or trading
performance. No GPU, production fit, service operation, network request, TEST
read or automatic monitoring was performed by this reviewer. Review ends here.
