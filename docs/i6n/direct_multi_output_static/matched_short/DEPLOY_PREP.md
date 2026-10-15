# Matched short contrast: deployment preparation (2026-10-15)

**State: PREPARED, not deployed.** No campaign was initialised and nothing was deployed. No
timer or service was installed or enabled, and no fit ran. Everything below waits for GO.
This summary names roles only. The private receipt holds the real paths, and it lives
outside git: `$PRIVATE/DEPLOY_PREP_RECEIPT.private.json`.

## Release pin and staged trees

| Item | Value |
|---|---|
| Release commit | `796ec8f48b0b1621fb9fa9e60e12c1c37079e51e` |
| Release tree | `8e07e42cb83a2f5d642edd37f616749e2cf16ea0` |
| Staged-tree manifest | `c1537e02cb2f2040217788f7396827904be5f38f91088c277b99a95ecd6cc434` (5,257 files) |

The manifest is the sha256 of the sorted per-file `sha256sum` list of `git archive 796ec8f4`.

| Role | Staged location | Manifest identical |
|---|---|---|
| coordinator | Its ext4 `/var/tmp` scratch, not `/home` | yes |
| preferred (RTX 5090) | Its ext4 `/var/tmp` scratch | yes |
| secondary (RTX 4090 Laptop) | Its ext4 `/var/tmp` scratch | yes |

The trees were moved with `git archive` over the existing ssh trust. Nothing was pushed.

## Frozen population and retained input

The population is digest `ba01f87a14e1ceb39d1a5770d4d00f3803cd29748c64edf7254c1e6c57b249d1`,
with POPULATION.json sha256 matching the committed seal. The retained input is INPUT.npz
`ef22ddb5…`, which equals the sealed source on every role.

| Role | Population location | Verification on that host |
|---|---|---|
| coordinator | Configured experiment storage root | `common_population verify`: VERIFIED |
| preferred | Its own ext4 storage root, declared as `experiment_storage_root` in the private spec | rsync, then `common_population verify` with the role's own python: VERIFIED |
| secondary | Its own ext4 storage root, declared as `experiment_storage_root` in the private spec | rsync, then `common_population verify` with the role's own python: VERIFIED |

On all three roles, the verified counts after intersection were:

| Split | h2 | h3 | h4 | h5 | h6 | h12 |
|---|---:|---:|---:|---:|---:|---:|
| FIT | 16397 | 16219 | 16041 | 15863 | 15685 | 14618 |
| INNER | 715 | 707 | 699 | 691 | 683 | 635 |
| DEVELOPMENT | 4724 | 4671 | 4618 | 4565 | 4512 | 4196 |

The worker-side storage roots are local ext4. The coordinator's NTFS storage is not visible on
the workers, and none of its free space is counted for them.

## Prepare, validate and preflight (real run, coordinator)

- `matched-short prepare`: PREPARED. Seals: mg1 `b6fd008d…`, mg2m `416c1156…`. Warehouse
  EnvironmentFile `PRESENT_BY_PATH`.
- `matched-short validate`: VALID. Runtime pin TF 2.21.0 / Keras 3.13.2, taken from the sealed
  designs.
- `matched-short preflight`: **PASS**, no launch blockers.

| Role | Disk free (own fs) | MemAvailable | GPU free | Runtime | Pins | Admission at provisional cap |
|---|---:|---:|---:|---|---|---|
| preferred (5090) | 76.4 GB | 10.12 GB | 33.6 GB | 2.21.0 / 3.13.2 PASS | PASS | PASS (preferred; 4 GiB cap + 3 GiB reserve = 7.52 GB, leaving 2.6 GB headroom) |
| secondary (4090 Laptop) | 533.5 GB | 22.6 GB | 16.7 GB | 2.21.0 / 3.13.2 PASS | PASS | PASS (`ROUTED_TO_NEXT_GPU_ROLE`: takes the other arm only after its own admission) |
| coordinator (CPU only) | 28.3 GB on `/`; `/home` has about 13 GiB, unused | 18.1 GB | none | 2.18.0 / 3.9.0, INFO only | PASS | not a fit role |

On the preferred host, SUnreclaim was 1.40 GB, Slab 2.21 GB and MemTotal 15.3 GB at the probe.
No compute process was running on either heavy GPU, and none on the preferred host's second GPU.

The coordinator's runtime mismatch is informational. The coordinator runs only CPU
orchestration, the reports and the comparison. The runtime pin applies to the fit workers, and
worker admission enforces it at claim time.

## Provisional cap

**Cap: 4 GiB (4,294,967,296 B), the same for both arms.** It is the declared initial cap. Its
basis:

- The closed MG2 full-population 13-slot fit had a measured full-cgroup peak of
  2,651,291,648 B. 1.25 × that is 3.31 GB, below 4 GiB.
- The reviewer's 25-slot, 2-epoch engineering slice peaked at 2,250,780,672 B on 4,096 FIT rows.
  That is not a full-fit requirement.
- The full matched FIT tensors add only tens of MB: 16.4k rows × 25 slots × 19 features.

No 25-slot measured peak exists, and a 13-slot peak is never used as the 25-slot requirement:
each arm has its own resource class.

The first full fit of each arm is the measurement. Later reservations are 1.25 × that arm's
measured full-cgroup peak. A refused cap is never trimmed, and the desktop reserve is never
reduced.

**Placement.** The 5090 runs first, with one heavy job on its host. The 4090 takes the other arm
only if its own admission passes. The shared-host 5070 Ti is never a heavy slot.

## Commands for GO

Run these from the staged coordinator tree, on CPU, with `EXPERIMENT_STORAGE_ROOT` exported.
`$PRIVATE` is the private directory holding `deployment.json` and `matched-short.prepared.json`.

```bash
cd "$STAGED_TREE"            # the staged 796ec8f4 tree on the coordinator
export CUDA_VISIBLE_DEVICES=""
PY="$COORDINATOR_PYTHON"     # roles.coordinator.python in the private spec
ROOT=/var/tmp/i6dmo/campaigns/matched-short-20261015

crispdm-run -m 3G -t 600s -n i6dmo-matched-validate -- "$PY" -m tools.i6dmo matched-short validate \
  --prepared "$PRIVATE/matched-short.prepared.json"
crispdm-run -m 2G -t 900s -n i6dmo-matched-preflight -- "$PY" -m tools.i6dmo matched-short preflight \
  --prepared "$PRIVATE/matched-short.prepared.json" --out-dir "$PRIVATE/precheck-go"
crispdm-run -m 3G -t 600s -n i6dmo-matched-init -- "$PY" -m tools.i6dmo matched-short init \
  --prepared "$PRIVATE/matched-short.prepared.json" --root "$ROOT" \
  --code-commit 796ec8f48b0b1621fb9fa9e60e12c1c37079e51e
"$PY" -m tools.i6dmo deploy --root "$ROOT" --deployment "$PRIVATE/deployment.json" --render-to "$PRIVATE/units-preview"
"$PY" -m tools.i6dmo deploy --root "$ROOT" --deployment "$PRIVATE/deployment.json"   # enables this campaign's timers

"$PY" -m tools.i6dmo matched-short status --root "$ROOT" --level campaign
"$PY" -m tools.i6dmo matched-short status --root "$ROOT" --level arm
"$PY" -m tools.i6dmo matched-short status --root "$ROOT" --level unit --unit mg1.fit
"$PY" -m tools.i6dmo status --root "$ROOT" --level campaign --probe-systemd --json
```

Notes on these commands:

- `--code-commit` is required, because the staged tree is a `git archive` without `.git`.
- `init` re-validates before it seals.
- Preflight must be re-run, and must PASS, immediately before `deploy`.
- At CLOSE, or at a typed technical terminal, the automation stops only this campaign's own
  units and writes `TIMERS_STOPPED.json`.
