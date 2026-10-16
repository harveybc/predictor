# Context Bridge Independent Runtime Review

Date: 2026-10-16

Verdict: **ACCEPT_FOR_EXECUTION**, conditional on the final release hash
manifest binding the reviewed execution bytes and population pins below.
No remaining blockers were found in this final pass's two-fix scope.
This supersedes the previous blocked disposition; it is not deployment
confirmation or a new full audit.

Reviewed current commit:
`4e8b3cb11dbecba911dd883a559b866e9dff8f95`
(`fix: retry transient bridge activation and authenticate replication mounts`).
Earlier review covered `0aa7c4164c5701add5e2d9ef5a9c0bd120cb2c14` and
the narrow initialization/population-binding patch at
`3f9408df96e4dbd9e28eb312ae19a42d5cb4bd9b`.

## Final Two Fixes

1. **Transient activation recovery: resolved.** In
   `tools/i6dmo/context_bridge_campaign.py:322-355`, `retryable` and `main`
   distinguish transient activation, connectivity, mount and GPU-visibility
   failures from identity faults. A retryable run records WAIT_RESOURCE
   with a 60-second retry interval and does not disable its launcher timer.
   Identity failures retain the stopping path. Recovery reuses the existing
   campaign rather than authorizing a fresh fit. Invocations remain bounded;
   admission may continue waiting while infrastructure is unavailable.

2. **Replication destination admission: resolved for the declared filesystem
   policy.** In `tools/i6dmo/context_bridge_campaign.py:175-178,195-231`,
   replication is preceded by actual findmnt facts for both storage and
   input roots on every role. Coordinator archival storage requires the
   existing NTFS policy; state, worker storage and input roots require ext4.
   Missing, unwritable, read-only, low-space and disallowed-filesystem roots
   are refused. The nearest existing destination mount must match its
   checked root before provisioning. These are point-in-time filesystem
   checks, not disk/device identity attestation or protection against a
   mount changing after admission. The per-root free-space floor is 6 GiB;
   coordinator archival storage also retains its estimate/reserve gate.

The added tests cover transient-error classification versus identity faults
and an unexpected input-destination mount. This final static pass also
traced the actual main-handler branch and the gate's placement before
provisioning; it did not execute those paths.

## Exact Pins

All SHA256 values below were read or calculated during this pass.

- `tools/i6dmo/context_bridge_campaign.py`:
  `68d90fc5623d3f9863c44915638d0dde06a30ad62d63106531444c27588cab95`
- `POPULATION_ACCEPTANCE.json` file:
  `ea1ce3a8e90f3fe8400f1110fad7588ba9882bbb27ff46b012cbfe863f437e9a`
- Accepted population digest:
  `7b54f7ec60657f95f8b351318c0c7fdaedf2c2a786600d5715a6af93f23301e1`
- Accepted `POPULATION.json` file digest:
  `f23350ff671e1a771d993ab9a15e3e1fa6ac7b7fe038fb2d996e33a34975bf8e`
- Accepted retained input digest:
  `ef22ddb54921ee9f9c774a550731dcd522c9d712f2a1882f1bf4e6a35e3942b8`

The real population acceptance records PASS, zero fits, no TEST read and
FIT-only scaler replay over 20,915 source rows. Both arms have 13,123 FIT,
546 INNER and 3,762 DEVELOPMENT origins. DEVELOPMENT head counts for
18/24/36/48/60/72 hours are 3,747/3,453/2,868/2,296/1,722/1,142.
This pass inspected that acceptance artifact, not the private raw arrays.
Its assumed-calendar and retained-data qualifications remain in force.

## Scope And Release Conditions

Only the two remaining runtime fixes, their tests and the real population
acceptance were reviewed in this final pass. Earlier selection and
design/model audits were not reopened. The parent reports 80 focused tests
passing; this reviewer did not rerun them. The earlier broader-suite
timeout is still not a full-suite pass.

Before deployment, the final release manifest must bind the exact reviewed
execution-file hashes and accepted population digest. Documentation-only
commits may change HEAD without changing those execution bytes. This note
does not substitute for that manifest or fresh runtime admission, and it
does not authorize a changed scientific recipe, extra seeds, strategy
promotion or TEST access.

No tests, fits, GPU work, services, TEST access, waits or monitoring were
performed during this final pass. Only this review note was edited.
