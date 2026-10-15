# Matched short contrast: launch receipt (2026-10-15)

**State: RUNNING.** The campaign is deployed. `mg1.fit` is running on the preferred RTX 5090 and
`mg2m.fit` is queued. Neither arm has completed, and the campaign is not CLOSED. This receipt names
roles only. The private copies of every output below are outside git, in `$PRIVATE/` (`deploy.json`,
`deploy-preview.json`, `START_CHECK.private.json`, `status-*.json`).

## Release and identity

| Item | Value |
|---|---|
| Release commit | `796ec8f48b0b1621fb9fa9e60e12c1c37079e51e` (staged-tree manifest `c1537e02…`, identical on all three roles) |
| Re-review | ACCEPT_FOR_DEPLOYMENT at `796ec8f4` (review commit `914734fb`) |
| Campaign | `i6dmo-6004b711281c`, mode `MATCHED_SHORT_CONTRAST`, effective config `c5790ca8fd61aa3c…` |
| Units | `mg1.fit`, `mg1.report`, `mg2m.fit`, `mg2m.report`, `comparison` |
| TEST units | none |
| Arm mg1 | seal `b6fd008d…`, design `64899668…` |
| Arm mg2m | seal `416c1156…`, design `d71d2b3c…` |
| Population | `ba01f87a14e1ceb39d1a5770d4d00f3803cd29748c64edf7254c1e6c57b249d1` |
| Runtime pin | TF 2.21.0 / Keras 3.13.2 |

## The GO sequence, as run

Every step ran from the staged tree, on CPU, on the coordinator.

| Step | Command | Result |
|---|---|---|
| 1 | `matched-short validate --prepared $PRIVATE/matched-short.prepared.json` | exit 0, VALID |
| 2 | `matched-short preflight --prepared … --out-dir $PRIVATE/precheck-go` | exit 0, **PASS**, no launch blockers. preferred: runtime, pins, GPU and admission all PASS (MemAvailable 10.09 GB; 4 GiB cap + 3 GiB reserve). secondary: PASS, routed (MemAvailable 22.7 GB) |
| 3 | `matched-short init --prepared … --root $ROOT --code-commit 796ec8f4…` | exit 0, campaign `i6dmo-6004b711281c` |
| 4 | `deploy --root $ROOT --deployment $PRIVATE/deployment.json --render-to $PRIVATE/units-preview` | exit 0, render only. 10 units, all carrying this campaign's id and `X-I6DMO-Campaign`. ExecStart runs only tick/worker/publish. No TEST or final unit |
| 5 | `deploy --root $ROOT --deployment $PRIVATE/deployment.json` | exit 0. 10 unit files written. These 5 timers are enabled and active: tick, publish, and the preferred, secondary and coordinator workers. Nothing waiting |

## Start check (the single check)

| Item | Observed |
|---|---|
| Child | `<cid>-mg1-fit-a01`, a transient user unit on the **preferred** role, ActiveState `active`, main PID present |
| Device | `rtx5090` (the role's heavy GPU) |
| Admission | role preferred, admitted, cap **4,294,967,296 B** (provisional, declared). Child `MemoryMax` = 4,294,967,296 B |
| Lease | durable, age 16.7 s at the check, expiring in 883 s (900 s TTL, renewed by the supervising worker) |
| Heavy GPU processes | **1** on the 5090 (TensorFlow reserved 30,288 MiB of device memory). **0** on the 5070 Ti |
| Progress at status | epoch 6 / 200 ceiling, best epoch 2, heartbeat age 1.6 s |

## Status (read once)

```bash
"$PY" -m tools.i6dmo matched-short status --root "$ROOT" --level campaign
"$PY" -m tools.i6dmo matched-short status --root "$ROOT" --level arm
"$PY" -m tools.i6dmo matched-short status --root "$ROOT" --level unit --unit mg1.fit
"$PY" -m tools.i6dmo status --root "$ROOT" --level campaign --probe-systemd --json
```

- **Campaign:** RUNNING. Counts: RUNNING 1, QUEUED 1, WAIT_DEPENDENCY 3. No failures.
- **Services:** tick, publish and all three workers had reported within 40 s.
- **Arm mg1:** RUNNING, 0/2, live child `mg1.fit` on preferred / rtx5090.
- **Arm mg2m:** WAITING, 0/2. `mg2m.fit` is QUEUED; the secondary role may take it only after its
  own fresh admission.

## ETA

**UNMEASURED.** No arm has completed a fit, so the campaign fit ETA is UNMEASURED for both arms.

`mg1.fit` reports only an in-flight upper bound: about 1,014 s at the declared 200-epoch ceiling.
Early stopping may end it sooner. It is not a completed-fit measurement, and it is never used as
mg2m's ETA.

## What happens next

The timers own every remaining transition. The first full fit of each arm is that arm's memory
measurement; later reservations are 1.25 × its measured full-cgroup peak. Nobody watches the fit.
At CLOSE, or at a typed technical terminal, the automation stops only this campaign's units and
writes `TIMERS_STOPPED.json`.

## Before R2 (non-blocking residuals, not addressed now)

- **R1:** harden against forged donors. Bind the donor receipt to campaign, population and
  weights, and make the fake runner non-scientific by default.
- **R2:** fix arm starvation under a persistent infrastructure refusal. `_pick` always takes the
  first eligible unit.
- **R3:** make `require_scientific` strict by default, before R2.
