# R0-versus-R2 contrast: launch receipt (2026-10-15)

**State: RUNNING.** Campaign `i6dmo-9d3c198c77b1` was deployed at 2026-10-15T18:41:16Z from
release `c34b1e48`. This receipt names roles only. Real paths, aliases and GPU identifiers are in
the private receipts outside git, under `$PRIVATE` (the R2 private directory on the coordinator's
ext4 `/var/tmp`).

## Release pin and staged trees

| Item | Value |
|---|---|
| Release commit | `c34b1e48c17d83d0f2eaa08cfc8daed4f1afb861` |
| Release tree | `1a464e013b53b7e4d0a9d6751e066490252cf2a7` |
| Staged-tree manifest | `092e0a516b0b7fafbef277581d60bd2873a1b940dcbfaea10484a688680de37d` (5,299 files) |

The manifest is the sha256 of the sorted per-file `sha256sum` list of `git archive c34b1e48`. The
tree was moved with `git archive` over the existing ssh trust to an ext4 `/var/tmp` stage on each
role, never `/home`.

| Role | Staged | Manifest identical | POPULATION.json sha256 | `common_population verify` (role's own python) |
|---|---|---|---|---|
| coordinator | yes | yes | matches the seal | VERIFIED `ba01f87a…` |
| preferred (RTX 5090) | yes | yes | matches the seal | VERIFIED `ba01f87a…` |
| secondary (RTX 4090 Laptop) | yes | yes | matches the seal | VERIFIED `ba01f87a…` |

The already replicated population was only re-verified, never re-copied. The verified counts after
intersection are unchanged from `DEPLOY_PREP.md`: FIT h2 16397 … h12 14618, INNER h2 715 … h12 635,
DEVELOPMENT h2 4724 … h12 4196.

## Private deployment spec

The spec is a new file, `$PRIVATE/deployment.json` (mode 0600, outside git, ext4). It was derived
from the matched spec, with these changes:

- every role's `workdir` points at the `c34b1e48` stage;
- the mirror targets are new and campaign-owned (`…/r2-contrast-20261015` locally and on the
  secondary host);
- `integration_commit` is `c34b1e48`.

The roles, caps and reserves are unchanged. Campaign root: `/var/tmp/i6dmo/campaigns/r2-contrast-20261015-b`.

## Prepare, validate, preflight

Each step was run from the staged coordinator tree on CPU, under `crispdm-run`.

| Step | Result |
|---|---|
| `r2 prepare` | `PREPARED`. Contract `0cad8817…`. R0 policy `FRESH_R0_FIT`. The `<out>.matched.json` companion was written. |
| `r2 validate` | `VALID`. mg2m design `416c1156…`, population `ba01f87a…`, runtime TF 2.21.0 / Keras 3.13.2. |
| `r2 preflight` (first run) | `PASS`, `launch_blockers: []` |
| `r2 preflight` (re-run immediately before `deploy`) | `PASS`, `launch_blockers: []` |

Preflight detail from the re-run before deploy:

| Role | Disk free | MemAvailable | Runtime | Pins | Admission at the initial cap |
|---|---:|---:|---|---|---|
| preferred (5090) | 75.8 GB | 9.87 GB | 2.21.0 / 3.13.2 PASS | PASS | PASS |
| secondary (4090 Laptop) | 533.7 GB | 23.5 GB | 2.21.0 / 3.13.2 PASS | PASS | PASS (`ROUTED_TO_NEXT_GPU_ROLE`) |
| coordinator (CPU only) | 36.7 GB | 18.7 GB | 2.18.0 / 3.9.0, INFO only | PASS | not a fit role |

## Init

**First attempt: refused. The root was kept as evidence and not reused.**

```
r2 init --root /var/tmp/i6dmo/campaigns/r2-contrast-20261015 --code-commit c34b1e48…
rc=2  {"cause": "STORAGE_GATE_STORAGE_ROOT_NOT_CONFIGURED: stage=init", "state": "REFUSED"}
```

The interactive shell had no `EXPERIMENT_STORAGE_ROOT`. The systemd user environment has it, and
its value equals the coordinator role's `experiment_storage_root` in the spec. `DEPLOY_PREP.md`
already requires that variable to be exported.

`init` writes the plan before the storage gate. As a result, the refused root holds
`EFFECTIVE_CONFIG.json`, `PLAN.json` and an unclaimed queue. Nothing about it is live:

- it has no `STATUS.json`, no deploy, no systemd unit and no claim;
- it carries the same campaign id, because the id is derived from the effective config, not the
  root;
- it was not redeployed or re-initialised.

Its removal is left to the coordinator.

**Second attempt:** `EXPERIMENT_STORAGE_ROOT` was exported from the systemd user environment, and
init ran into the NEW root `…/r2-contrast-20261015-b`:

```json
{
 "campaign_id": "i6dmo-9d3c198c77b1",
 "contract_binding": "SCIENTIFIC: design seal, design digest, population and runtime equal the ruled contract",
 "contract_seal_sha256": "0cad8817dffa2729164d55273a40814dcf794e348b95f9b02e1227ef641e4066",
 "effective_config_sha256": "0afedbf9830d1d4e8f069a63dd644689a142d118a699496dc388017c9e7fe7db",
 "fit_device": "gpu",
 "fit_gpu_class": "rtx5090",
 "mode": "R2_CONTRAST",
 "root": "<ROOT>",
 "units": "25 units: r2.pretrain.00..r2.pretrain.18, r0.fit, r2.fit, r0.report, r2.report, r2.probe, comparison"
}
```

What the sealed config says:

- `fit_gpu_class` is `rtx5090`, sealed to the `preferred` role. Both `r0.fit` and `r2.fit`
  carry `gpu_class` `rtx5090`.
- `fit_device` is `gpu`, `engineering_cpu_fixture` is false, `test_open` is false, and no
  TEST or final unit exists.
- Each admission class has a provisional cap of 3,584,604,160 B, with the basis recorded in
  `RUN_R2.md`.

## Deploy

`deploy --render-to` rendered 10 units, all prefixed `i6dmo-9d3c198c77b1-`: one tick, one
publish and three worker (preferred, secondary, coordinator) service/timer pairs. There is no TEST
unit. `WorkingDirectory` is the `c34b1e48` stage.

`deploy` returned `rc=0`. All five timers report `active` / `enabled`.

## Start check (one reading, 2026-10-15T18:41:30Z)

| Check | Reading |
|---|---|
| Real GPU child | `r2.pretrain.00` attempt 1 on preferred (`rtx5090`), executor `ssh-systemd`, child unit `active` with a main PID |
| Lease | renewed 13.6 s before the reading, expires in 886.4 s |
| Admission | `r2.pretrain.00` admitted at **3,584,604,160 B**, the pretraining class's provisional cap. The child's MemoryMax is the same value. |
| Heavy GPU processes | preferred host: **1** on the 5090; secondary host: 0 |
| 5070 Ti | **0** processes |
| C1 hold | `r0.fit` is `WAIT_RESOURCE` with cause `FIT_GPU_CLASS_HOLD:r0.fit needs rtx5090, role offers rtx4090`. The secondary's worker saw it first and held it; it was not failed and not relabelled. |

## Status (one reading at every level)

| Level | Reading |
|---|---|
| campaign | `WAITING_FOR_WORKER`; `r2.stage` BRANCH_PRETRAINING; pretraining 1/19 complete, none failed; decision null; `fit_gpu_class` rtx5090 with one hold (`r0.fit`); failures `[]` |
| stage | BRANCH_PRETRAINING 1/19 terminal, R0_R2_FITS 0/2, REPORTS 0/3, R2_DECISION 0/1 |
| regime (`R2_PRETRAIN`) | 1/19 complete |
| unit (`r2.pretrain.00`) | COMPLETE: `DONOR_WRITTEN`, seed 6102026, `tf_imported_before_flags` false, `device_class` rtx5090, fit 12.0 s (5,590 updates), wall 16.4 s; facts `loss_curve`, `reconstruction`, `compute_cost` |
| fleet | no running child at the reading |
| publication | PENDING, 3 expected, 0 delivered, 0 mismatch |
| `status --probe-systemd` | `WAITING_FOR_WORKER`, no service failure |

## ETA

| Part | ETA |
|---|---|
| Pretraining | MEASURED from one unit: 12.0 s per unit, about 216 s for the remaining 18 |
| Fits | UNMEASURED: no R0 or R2 fit has completed |
| Campaign end to end | UNMEASURED |

## Finding: the pretraining class has no measured peak

`r2_runner.run_pretrain` writes `costs.cgroup_peak_bytes = None`, and the attempt's `peak_bytes`
is NULL. The pretraining admission class therefore never acquires a measured peak, and every
pretraining unit keeps running under the provisional 3,584,604,160 B cap. That cap is an upper
bound: the reviewer's engineering pretraining peaked at 1.73 GB.

This is safe, but R2-17's "first unit is the measurement" is **not met for pretraining**. The fits
do record a peak through the trainer receipt. A fix belongs in a new release; nothing was changed
in this deployment.

## Commands (staged tree, `$PY` = coordinator python, `$ROOT` = the campaign root)

```bash
"$PY" -m tools.i6dmo r2 status --root "$ROOT" --level campaign
"$PY" -m tools.i6dmo r2 status --root "$ROOT" --level stage
"$PY" -m tools.i6dmo r2 status --root "$ROOT" --level arm --regime R2_PRETRAIN
"$PY" -m tools.i6dmo r2 status --root "$ROOT" --level unit --unit r2.pretrain.00
"$PY" -m tools.i6dmo r2 status --root "$ROOT" --level fleet
"$PY" -m tools.i6dmo status --root "$ROOT" --level campaign --probe-systemd --json
```

From here on, the timers own the campaign. Nothing is being monitored. At the automatic CLOSE, the
automation stops this campaign's own units and writes `TIMERS_STOPPED.json`.
