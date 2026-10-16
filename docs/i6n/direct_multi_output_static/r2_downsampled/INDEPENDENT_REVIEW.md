# Independent Downsampled R0/R2 Release Review

Date: 2026-10-15. Verdict: **ACCEPT_FOR_RELEASE** for the scope and bytes below.
Reviewed implementation commit: `c36891f29ff25b733a7c7ed68fc276d8e6c7cca1`.
No confirmed P1 or other release-blocking implementation defect remains in
this bounded review. No implementation edits were made by this reviewer.

This is code/protocol acceptance, not a scientific result, a complete-suite
certification, or a claim that production runtime/custody gates have executed.
It can serve as the independent review reference for the prospective MG12m
long objective ruling. The release manifest, ruled contract, frozen execution
source, fresh preflight/admission and runtime/custody checks must still agree
before launch. No TEST access is authorized.

## Scope And Findings

Reviewed donor transport and worker/unit-runner integration; recovered-donor
factory adoption; short-to-long initialization, identity and resume gates;
recovery helper; selected-arm contracts; long branch mean monitor; and strict
R0/R2 comparison integration. Existing RAW closure mechanisms were not broadly
reaudited. There are no defect severities, implementation fixes or open P1s to
list against these reviewed bytes.

- RD01/RD06: legacy short contracts/objective remain exact. New long uses
  MG12m, 13 observations over 144 market hours and heads 18/24/36/48/60/72.
  Registered O1 identity/version/decoder remain unchanged; orchestration is
  separately versioned `i6dmo.branch_mean_fit_inner.v1`. Full FIT and purged
  INNER are evaluated at identical end-of-epoch weights, equally averaged,
  with patience 10, min_delta 1e-8 and best-weight restoration. Hard budgets
  fail without selecting a donor. Original short donors remain INNER-only.
- RD02/RD04/RD05: recovery revalidates retained source and both custody views,
  real donor arrays, publication readbacks and original identities. The
  factory adopts all 19 original queue identities and provenance as ADOPTED,
  not new fits. Its effective identity binds the recovery file, physical
  custody proof and explicit stored-terminal/donor-terminal/NPZ pins. Changed
  training-code identity requires a fresh R0, not asserted equivalence.
- RD03/RD07: an independent real-NPZ relocation/import replay removed the
  temporary coordinator-only paths and imported all 19 branches into a real
  built model. Branch values matched; core/head weights stayed unchanged and
  branches remained trainable. Terminal/NPZ mutation and parent-symlink
  substitutions were refused. The R0 unit received no donor binding.
- RD08: recovered short and new long enable strict report validation; the
  focused tests cover actual C2 metadata, score-cache contradictions, six-head
  completeness, import binding and same-GPU-class requirements. The decision
  is the six equal-head annual pooled relative MAEs, tolerance 1e-8, ties R0.
- RD09/RD10/RD12: independent tests used actual recovery verification, factory,
  adopted queue, initialization marker and sequence resume validation. An
  initialized-but-not-deployed campaign reached all three runtime attestation
  boundaries and one deployment boundary; a second invocation did not deploy
  again. Changed code, data pins, donor-recovery binding, physical proof and
  queue spec were refused before deployment. Owner sequence tests also cover
  finite positive/negative advancement, technical stop and closure checks.

The inspected initial sequence resume omission was corrected by its owner
before the frozen commit. It is not an outstanding finding against c36891f2.

## Executed Evidence

All reviewer replays used hidden CUDA, bounded thread counts, disabled pytest
plugin autoload, `--noconftest`, and `crispdm-run -q -m 2G -t 180s`.
No GPU operation, fit, network request, deployment or service management was
executed. Synthetic files were confined to temporary test roots.

The focused companion files were `test_r2_downsampled.py`,
`tests/test_r2_recovery.py`, `test_donor_transport.py`,
`test_r2_factory_profiles.py`, and `test_r2_sequence.py`: **82 passed** in the
combined bounded run. That run reported 86 passed and seven errors in the
independent fixture, not 93 passed. The borrowed helper-only capsule lacked
OPS scheduling metadata; its reusable-R0 target also conflicted with the
factory fixture's explicit fresh-R0 request. These were corrected only in
the independent temporary-fixture construction.

Final command, on unchanged reviewed implementation bytes:

```sh
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
crispdm-run -q -m 2G -t 180s -n i6dmo-r2-independent-handoff -- \
python -m pytest --noconftest -q -p no:cacheprovider --tb=short \
tests/i6dmo/test_r2_release_review.py
```

Result: **11 passed, two protobuf deprecation warnings, 10.56 seconds**.
All reviewer tool sessions have completed.

## Limits And Launch Qualifications

Production source recovery of 19 donors / 115 custody files / 115 readbacks,
including recomputed actual FIT scales and current-code target, is evidence
reported by the parent, not an independently executed production probe here.
Physical SSH custody remains the declared trusted-operator, unkeyed two-host
model; a synthetic proof in a test is not physical authentication.

The independent factory fixture bypasses canonical contract authority for its
opaque synthetic features/objective, but uses real recovery, adoption, stored
terminal checks and queues. Sequence source-freeze/preparation validation and
runtime/deployment boundaries are controlled in that fixture. Real donor
loading/import is executed without training. Companion contract tests cover
canonical short/long design validation separately.

No scientific full short-to-long fit/publication/mirror/CLOSE execution, remote
runtime provisioning or GPU numerical equivalence was performed by this
reviewer. RD11 production completion remains an execution gate, not a claim
of this report. No broad legacy suite pass is claimed. Existing source-age,
calendar, future-availability and business-mask qualifications remain in force;
this review does not qualify new data or authorize population regeneration.

## Reviewed File SHA256

The following working bytes were verified equal to the implementation commit.
Later documentation/release commits may reference them but must not substitute
changed implementation bytes without renewed review.

```text
38b2a36617f3a4cd4da07ad0d8b2decf81754287883fd8ab41b209bc760a9211  tools/i6dmo/donor_transport.py
d2b33eaedbcf9d8e51081a94ed806e17d32e2084afdb41a19fd84ead83768bf1  tools/i6dmo/worker.py
ab111c2d924f8dd83f1fad67e883fb0e6126e6196ae5f2244f1d633dadc058f7  tools/i6dmo/unit_runner.py
e4b1146e55ece3bec760ea05f3b68e921958c6c510916505021c03d613202fd0  tools/i6dmo/campaign.py
23013a467a559534a65bb4130de01b391629e7d5d6a82efeb07ae644628cac32  tools/i6dmo/r2_downsampled_campaign.py
9d40c53846137344a31f3260b56ea52fd593192cbd87d1b4e461835388bc4cc8  tools/i6dmo/r2_design.py
7c987d9601ebca4dd106ae494a8154495bcc9146e40aa97af93e1226bf8587e5  tools/i6dmo/r2_runner.py
e97dd0fd32ce87f6559ac1561fd49d57a68509d8a6052ebba9a07e33e875d85d  tools/i6dmo/model.py
82a68bc27b930f1eac300efd72415e71e8b010d6aa0fe11dee36e65e5b134545  tools/i6dmo/r2_recovery.py
9c49b45e101ce8346b9610fbbe36d54f9f24d0b3aad0d48f4c9a2587281a4576  tools/i6dmo/reporting.py
835efbeeba92407b78ff9b58e8d8035ff56b0480ca7c5b1b0cccfe0e5a95879f  tools/i6dmo/raw_long_campaign.py
```

## Production Storage-Root Qualification

Subsequent owner production-source probing reported successful adoption of all
19 donors followed by the storage gate refusing an absent
`EXPERIMENT_STORAGE_ROOT`. **Operational launch verdict: REVISE pending the
minimal sequencer fix and its focused regression.** This supersedes the
earlier absence of a known launch blocker; the preceding bounded test results
remain unchanged. The production probe was owner-executed, not reviewer-run.

P1-S1: `tools/i6dmo/campaign.py:540` calls environment-dependent
`storage_root()`. Baseline `tools/i6dmo/r2_downsampled_campaign.py:410` does not
establish the configured coordinator storage root for `run()`; its generated
service at line 462 also omits that environment. A valid source/deployment can
therefore fail initialization after donor adoption solely because the invoking
shell or service manager lacks the implicit variable. Minimal fix: derive the
root explicitly from the configured deployment in run/install, preserve the
existing storage gate, and prove operation with the inherited variable unset
or incorrect. No scientific negative or fit is implied by this refusal.

At qualification time the sequencer/test diff against c36891f2 was empty, so
no changed implementation bytes or fix regression are accepted by this
addendum. No further broad tests or automatic monitoring were started.

## S1 Fix Acceptance

Verdict: **S1 ACCEPT; ACCEPT_FOR_RELEASE restored**. This supersedes the
preceding S1 launch-blocking qualification. No confirmed P1 remains in this
review's scope. Baseline implementation remains c36891f2 except this reviewed
`tools/i6dmo/r2_downsampled_campaign.py` delta, SHA256:

`0bd97de308e3ac0159753fcb2137a8833e72e04ebb9263fc4060da2e68b972ae`

The minimal delta validates the configured absolute coordinator storage root,
scopes it around the existing sequence body, and restores the caller's prior
environment in `finally`, including on exceptions. Install validates the same
root before writing units and explicitly serializes it into the service
environment, quoting spaces and escaping systemd percent specifiers. The
service also enters through the scoped `run()` wrapper. Storage admission is
preserved; no scientific identities, fit procedures or decision rules change.

Independent replay used hidden CUDA, bounded threads, disabled pytest plugin
autoload and `crispdm-run -q -m 2G -t 180s`, selecting only these regressions
from `tests/i6dmo/test_r2_sequence.py`:

```text
run_scopes_coordinator_storage
run_refuses_missing_or_relative_coordinator_storage
install_service_explicitly_binds_storage
install_refuses_missing_or_relative_storage
```

Result: **10 passed, 29 deselected, 0.48 seconds**. These verify absent/wrong
inherited environment, success/exception restoration, missing/relative-root
refusal before initialization or unit writes, and service quoting including
spaces and percent characters. Service calls are intercepted; no actual
deployment, systemctl operation, network access, GPU operation or fit ran.
All reviewer tool sessions completed. No broad suite was rerun or certified.

The independent factory fixture now supplies the newly required explicit
coordinator root. Its earlier 11-case pass is baseline evidence; that entire
file was not replayed for this narrow ENV fix. Parent-reported production
adoption/storage success and its 39-test module pass remain parent evidence,
distinct from the 10 tests independently executed here.

Freeze the accepted delta and refresh the release manifest and prospective
long objective's review-document hash. All previously stated production
runtime, custody and scientific execution qualifications remain mandatory.

## Preflight Bytecode Fix Acceptance

Verdict: **ACCEPT_FOR_RELEASE** for the one-line preflight delta. The owner
reported a subsequent pre-fit production refusal: remote preflight created
ignored bytecode in a freshly attested runtime, and the later strict runtime
check correctly refused `RUNTIME_IMPORT_OVERRIDE`. This is a technical
predeployment failure, not a scientific result. The production reproduction
is owner evidence; no production probe or runtime cleanup was reviewer-run.

Reviewed only `preflight.probe_role()` changing remote Python arguments from
`[python, '-m', ...]` to `[python, '-B', '-m', ...]` and the three regressions
in `tests/i6dmo/test_probe_no_bytecode.py`. Argument order is correct; other
probe arguments, quoting, behavior and strict attestation are unchanged.

Accepted SHA256:

```text
8cabe4e0f0f694b28e65fb647f6f37059c2a1a9046a4889462393cda983bb44d  tools/i6dmo/preflight.py
5ac938816faad83176ea3676f4bbde4f5ec8107e90cfa5ded3efd7d6550d92dc  tests/i6dmo/test_probe_no_bytecode.py
```

Independent replay: hidden CUDA, bounded threads, disabled pytest plugin
autoload, `crispdm-run -q -m 2G -t 180s`, and:

```sh
python -m pytest --noconftest -q -p no:cacheprovider --tb=short \
tests/i6dmo/test_probe_no_bytecode.py
```

Result: **3 passed, 0.19 seconds**. Two cases check remote argv, strategy flag
and a filename containing spaces. The actual subprocess case uses the real
copied `probe_local` and common helper in a temporary module entrypoint,
removes inherited bytecode environment flags, verifies TensorFlow was not
imported, and verifies no pyc/cache directories or file byte/mtime changes.
SSH transport and external health/service commands are intercepted. This is
not a production SSH/GPU probe or a replay of the complete CLI import graph.

No new P1 remains in this minimal delta. No broad tests, fits, deployments,
service operations, runtime mutations or automatic monitoring were executed;
all reviewer tool sessions completed. Freeze this source, update the release
and review-document hashes, and use a new prospective work root/identity.
Do not hotpatch or relax attestation for the failed pre-fit attempt. Earlier
scope limits and all production runtime/custody/admission gates still apply.
