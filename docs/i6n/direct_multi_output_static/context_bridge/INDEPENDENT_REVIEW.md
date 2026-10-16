# Context Bridge Independent Runtime Review

Date: 2026-10-16

Verdict: **BLOCKED_FOR_EXECUTION**. Two runtime blockers remain. Population
acceptance and a release hash manifest do not resolve these failure paths.

Reviewed baseline: `0aa7c4164c5701add5e2d9ef5a9c0bd120cb2c14`.
Also reviewed the narrow population-binding and actual-initialization patch at
`3f9408df96e4dbd9e28eb312ae19a42d5cb4bd9b`, as requested during this pass.
This is a bounded static review, not a deployment or scientific-result audit.

## Remaining Blockers

### P1 Activation failure disables its own recovery mechanism

References: `tools/i6dmo/context_bridge_campaign.py:245-253,318-322`.

`ensure_deployed` correctly raises when timer activation fails. However,
`main` handles that recoverable infrastructure error by recording FAILED and
disabling the launcher timer. One temporary systemd activation failure can
therefore leave a published campaign with missing worker/publication timers
and no automatic reconciliation. Calling `ensure_deployed` on each resume
does not help when the timer responsible for resuming has been disabled.
This does not itself duplicate a fit, but it breaks autonomous recovery.

Minimal fix: distinguish retryable pre-execution deployment failures from
identity violations and terminal scientific outcomes. Persist a typed wait
and bounded retry budget for the existing launcher timer; do not disarm it
on the first transient activation failure. Keep completed fits adopt-only
through the existing authenticated-terminal path.

Missing regression: exercise the real `main -> run -> ensure_deployed` error
path with mocked systemctl failure followed by success. Assert the first
failure does not disable recovery, the same campaign/queue is retained, and
no completed fit is launched again. The current activation test mocks
`deploy.deploy` and does not exercise launcher shutdown.

### P1 Remote write admission does not authenticate destination mounts

References: `tools/i6dmo/context_bridge_campaign.py:193-211` and
`tools/i6dmo/raw_long_campaign.py:238-245`.

The coordinator has a mount-qualified storage gate. The per-role remote
check only tests directory existence, access and a fixed 6 GiB of free
space at `experiment_storage_root`. An unavailable remote volume with a
writable underlying mountpoint directory can pass and receive population
replicas on the wrong filesystem. The separate `data_root` destination
receives the retained input replica without a pre-write mount/capacity
check. Later matched preflight runs after these writes.

Minimal fix: before remote mkdir/rsync, authenticate the actual filesystem
and expected mount target for both destination roots using the private
deployment contract. Check required replica bytes plus reserve on each
distinct destination filesystem. Preserve intentional ext4 worker scratch;
do not assume every worker uses the coordinator's archival filesystem.

Missing regression: simulate an absent remote mount whose underlying
directory is writable, and an insufficient-space input destination.
Neither case may reach replication. Existing coverage tests only refusal
by the coordinator's local storage gate.

## Changes Confirmed

- Campaign initialization now stages before atomic publication, binds its
  intent and preserves partial preparation evidence.
- Bridge deployment reads verify the sealed deployment digest; fits carry
  the sealed 5090 class and other roles lose their heavy GPU slots.
- Preparation refuses dirty source, binds generator file digests and
  verifies persisted population bytes. Release scope now includes the
  provisioning helper, market grid and storage adapter.
- Actual initial-weight hashes are read through the authenticated fit
  terminal's digest-pinned training artifact. The bridge decision refuses
  absent, malformed or unequal hashes and retains the control on a tie.
- Launch and resume compare the prepared population digest with release
  acceptance. These additions do not introduce a fit or TEST stage.

## Evidence And Limits

`POPULATION_ACCEPTANCE.json` records PASS for population digest
`7b54f7ec60657f95f8b351318c0c7fdaedf2c2a786600d5715a6af93f23301e1`:
13,123 FIT origins, 546 INNER origins and 3,762 DEVELOPMENT origins.
DEVELOPMENT head counts for 18/24/36/48/60/72 hours are
3,747/3,453/2,868/2,296/1,722/1,142. It records FIT-only scaler replay,
zero fits and no TEST read. This pass inspected that acceptance artifact;
it did not reopen private arrays or independently repeat its audit.

The parent reported 74 focused tests passing. The subsequent 75-test run
was announced, not independently executed or verified here. The broader
suite timed out at approximately 89 percent with no displayed failures;
that is not a full-suite pass. No tests, fits, GPU work, services or
monitoring were run during this re-review.

After the two blockers are addressed, final release acceptance must bind
the reviewed execution bytes and accepted population digest. Re-run the
targeted failure-path regressions before scientific deployment. This note
does not authorize additional seeds, strategy promotion or TEST access.
