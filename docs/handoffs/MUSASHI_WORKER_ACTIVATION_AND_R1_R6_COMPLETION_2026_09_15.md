# Worker activation completed; remaining R1-R6 work

Date: 2026-09-15. Issued by Musashi under the owner's standing authorization.
Return inspected: predictor `7c6df0e89bc44ad4b6bcbd768bb0f0f427f32412`.
This is an operational activation and completion order, not independent acceptance
of every reported test or scientific result.

## 1. Activation is DONE, not an owner blocker

Musashi inspected the service and runtime configuration, retained a private backup,
and restarted ONLY `crispdm-data-gov.service`. The two worker principals existed
in that configuration, but the workers still held the shared predictor credential.
Restart alone was insufficient to finish the configuration.

Dedicated worker credentials were installed privately on the two remote workers;
only their two existing principal hashes were changed, with policies unchanged.
A second deliberate restart loaded those hashes. Both workers authenticated over
their existing tunnels and obtained HTTP 200 from `/api/v1/lakes`. Each installed
key was matched to its own principal in the effective configuration. No key is
published here and the existing shared credential was not revoked.

The worker key file is `~/work/gov/worker.key` on each worker. Use it explicitly
via the existing client's `api_key_file` / CLI key-file option. Do not keep selecting
the old `predictor.key` in new worker runs. Do not copy keys into reports or Git.

All five services are active. The other four processes retain their original PIDs;
the loader and warehouse were not restarted. `NRestarts=0` refers to automatic
restarts, not the two manual restarts disclosed above. No experiments, datasets or
cube history were changed by this activation. This verifies authentication, not a
new full delivery/terminal experiment under the dedicated identities.

## 2. Immediate worker acceptance: Satoshi, start without further permission

Configure the existing worker commands to use their dedicated key files. Run one
small governed synthetic delivery per worker and close each campaign with a terminal,
actual byte count, verified digest, measured cost and reconciliation. Confirm the
accounting actor matches that worker, rather than merely asserting HTTP success.
Test cache reuse in a second bounded unit; preserve all earlier shared-actor receipts.

The copied `worker_delivery_probe.py` must be checked before reuse: the inspected
version downloads and writes a local receipt but has no terminal submission at its
end, despite its docstring mentioning a terminal. Complete the outcome through the
existing client/outbox, not a separate protocol. Do not leave new campaigns open.

Use the existing resource limits and synthetic fixtures. No service restart is
required. No GPU, broker, live trading, new financial resource or scientific promotion.

## 3. R3 is honest about missing observations, but still incomplete

Keep `doin-offline-replay-prod-11` intact as NON_GOVERNING mechanical evidence.
Requested steps without observed steps do not satisfy the standing requirement
for actual work counters. Instrument the real application/runtime completion path
to publish observed environment steps and, where applicable, optimizer updates.
Keep requested budget, observed work and wrapper duration distinct.

Prove that the governed file is actually read, not just named in a config: change
the delivered fixture deterministically and verify consumed row identities/digest;
make the old sample input unavailable in an isolated test and exercise the same
entry point. Retain the regression for flat versus nested input configuration.

After unit/integration tests, one corrected CPU NON_GOVERNING production replay is
authorized: <=64 requested steps, <=10 minutes, <=2 GiB, fresh run directory, no
induced production outage. Persist observed work, actual consumed input identity,
terminal and reconciliation. Do not reinterpret missing metrics as measured zero.

## 4. R4 still needs the disposable end-to-end path

Nine candidate semantic tests and a test of deployed absence are not the requested
provider -> host -> data-gov -> receipt -> warehouse proof. Build that disposable
stack from identified candidate packages and run the whole path. UNKNOWN must
survive in persisted evidence; whole-archive delivery and rejected ranges/holdout/
point-in-time/live cases must have their declared outcomes. Include regression
coverage for existing synthetic contracts. Do not install the real archive policy
or restart production services in order to demonstrate this.

## 5. Missing screen decision: identify and derive, never fabricate

For the reported `p1lr-decision@202`, locate the exact caller, required decision
schema, governing design, consumed evidence and producer command. Record these
in the return. If complete existing evidence permits a CPU-only deterministic
adjudication under the unchanged design, produce a candidate verdict and tests.
If an input is missing, name it and its producer and complete authorized preparation.
If the result is negative or insufficient, retain that result. Do not set a gate
true, invent a review record or launch the dependent experiment just to remove a
refusal. Other independent R1-R6 tasks continue regardless.

## 6. Two proposed owner blockers are not prerequisites

Per-delivery capability tokens: DEFERRED as an architectural proposal, not a new
dependency of R1-R6. Document the present distinction between data-gov's actor/
campaign accounting and the store's transport identity. Submit any concrete
attribution gap separately; do not silently redesign the protocol during closure.

Usage rights: primary-source research is Satoshi's unfinished task first. Supply
the actual applicable source, dated text/section, scope of analysis, derivatives,
redistribution and uncertainties. Only a demonstrated need for an agreement,
entitlement or legal clarification goes to the owner. No owner approval creates
third-party rights. Do not block synthetic testing or infrastructure adoption on
this independent research.

The 3000-bar comparison is finite snapshot evidence. Preserve its exact and
material-difference definitions and original decimal/float precision disclosures;
do not generalize zero material differences into a no-revision guarantee.

## 7. Scheduling and delivery

Coordinator: input-consumption proof, real replay counters and result reconciliation.
Worker A: disposable archive end-to-end, existing identities and bounded memory.
Worker B: primary-source research and missing-decision evidence inventory.
Independent tests may proceed concurrently; no artificial GPU utilization.

Maintain the existing work plan and requirement/test/evidence matrix. Report exact
commits, tests actually run, per-worker outcomes and remaining scope. Do not label
R1-R6 complete until the missing deliverables above exist. Do not repeat the
already accepted four-consumer activation. No new owner approval is needed to start.
