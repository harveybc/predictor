# M02 — staged pretraining with early stopping at every stage

Lane M02 of `docs/handoffs/SATOSHI_MODULAR_OPTIMIZATION_2026_09_30.md` (master `dc72170e`),
plan `docs/tres_temas_entrevista/program_v3/MODULAR_STACK_WORK_PLAN_2026_09_30.md`,
section "Pretraining and training sequence". Acceptance rows MS09, MS10, MS11.

**Every number below comes from a declared SYNTHETIC fixture. It is a component and
plumbing measurement, never a forecasting result (PLUMBING_NOT_A_RESULT).**

## Dispatch block

| Field | Value |
|---|---|
| worktree | `.worktrees/predictor-m02-pretraining-20260930` (coordinator) and the same path on worker_b |
| branch / source | `satoshi/m02-pretraining-20260930`, off `556c5f3e` (`codex/modular-stack-20260930`) |
| tested tip | `9844dafd` (71/71); pilot evidence produced at that tip |
| first command | `crispdm-run -q -W 7200 -m 3G -t 30m -n m02-suite -- env CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m pytest tests/test_modular_pretrain.py tests/test_modular_temporal.py tests/test_modular_candidate_evaluator.py` |
| acceptance test | `tests/test_modular_pretrain.py` (MS09/MS10/MS11 plus end-to-end with the real engine and evaluator) |
| budget | 3G cap and 30 min wall per child, CPU only, heartbeat every 30 s |
| blocker | memory admission only (Traffic leases). No M01 revision published, so the build is against `556c5f3e`. No M03 admissible-input declaration yet, so inputs are SYNTHETIC |

## §6 report

```
M02 — staged branch AE -> fixed-donor fusion -> core AE -> matched regimes
repo/branch/tip: predictor satoshi/m02-pretraining-20260930 9844dafd (source 556c5f3e)
files: tools/modular_pretrain.py, tools/modular_candidate_evaluator.py (shared early-stop loop),
       tests/test_modular_pretrain.py, docs/audits/evidence/m02_pretraining_20260930/SYNTHETIC_pilot_s7_e100/
suites: test_modular_pretrain + test_modular_temporal + test_modular_candidate_evaluator 71/71
        (worker_b, CPU, crispdm-run 3G; 131.5 s at 3e58892a, 133.0 s at 9844dafd)
acceptance: pilot exit 0, 131.6 s wall, cgroup peak 1.03 GiB of 3 GiB, 6 heartbeats, max gap 30.02 s
what is NOT done / refused / not measured: see the last section
```

### Stage-by-stage reconstruction (SYNTHETIC, seed 7, 3 features, hourly, window 24)

Early-stop rule for every stage: patience 4, min_delta 1e-4, monitor every epoch,
max_epochs 100, max_updates 20000, max_seconds 600, restore of the best monitored checkpoint.
The rule identity is `5828db24…` (sha256 of the rule), identical across all seven fits.
AE train: 1320 windows. AE internal validation: 307 windows, carved from the tail of the
forecast TRAIN split after a 24-row purge. The outer validation, test and holdout are never read.
No test rows exist in the fixture.

| Stage | Updates observed / selected | Epoch sel./done | Stop | Train MSE (ref., rel.) | Internal-val MSE (ref., rel.) | Reload max abs err |
|---|---|---|---|---|---|---|
| branch_0 AE | 1197 / 1113 | 53 / 57 | patience (no_improvement) | 0.00798 (0.985, 0.0081) | 0.00761 (1.069, 0.0071) | 0.0 |
| branch_1 AE | 1239 / 1155 | 55 / 59 | patience | 0.00792 (0.977, 0.0081) | 0.00811 (1.082, 0.0075) | 0.0 |
| branch_2 AE | 1302 / 1218 | 58 / 62 | patience | 0.00659 (0.897, 0.0073) | 0.00778 (1.464, 0.0053) | 0.0 |
| core AE (on fused 12×48) | 1071 / 987 | 47 / 51 | patience | 0.00829 (0.0694, 0.119); std-MSE 0.182 | 0.00916 (0.0944, 0.097); std-MSE 0.198 | 0.0 |

- The reference is a per-channel train-mean constant, which carries no information. Relative MSE is MSE divided by the reference MSE. Compression is lossy at every stage and no stage reached zero loss.
- The core keeps about 10–12 % of the fused variance as residual (relative MSE), or about 20 % after per-channel standardization.
- Branch AEs reconstruct their own single channel at 12 steps from 24 inputs. The core compresses 12×48 to 6×8.
- An earlier run capped at 30 epochs (`SYNTHETIC_pilot_20260930_s7`) stopped every AE on **budget** (max_epochs) while the AEs were still improving. With the limit raised to 100, all four AEs stopped on **patience**. The stop class keeps these two cases apart.

### Downstream utility with matched regimes (SYNTHETIC; real engine and real evaluator)

Horizons 1/3/6. Forecast train 1651 windows, outer validation 666. Same seed, same evaluator settings and same early-stop rule for every regime. The paired persistence MAE on the same rows is 0.80391.

| Regime | Val MAE | Skill vs persistence | Updates obs./sel. | Stop | Epoch sel./done |
|---|---|---|---|---|---|
| R0 fresh | 0.43321 | 0.461 | 546 / 442 | patience | 17 / 21 |
| R1 frozen donors | 0.47129 | 0.414 | 2600 / 2600 | max_epochs (budget) | 100 / 100 |
| R2 fine-tuned donors | 0.43749 | 0.456 | 338 / 234 | patience | 9 / 13 |

- R1 and R2 have identical initial weights (`51b89950…`). R0 differs (`ada031ee…`).
- The R1 core weights after the fit equal the donor digest; the R2 core weights differ. The end-to-end test asserts both.
- This is one seed on a fixture, so it says nothing about whether pretraining helps. Only the matched real-data design can answer that.

### Does a core donor refuse a mismatched upstream? Yes, and the test asserts it

`test_core_donor_binds_exact_upstream_and_refuses_mismatch`. `build_modular` rejects each case below with "Donor manifest mismatch" before any fit:

- (a) one branch switched to fresh R0 under the core donor;
- (b) a valid branch donor from a different pretraining run swapped in;
- (c) the other run's core donor placed over this run's branches.

The core's engine manifest binds the ordered branch manifests, the branch weight digests and the fusion identity. `core.provenance.json` also binds the branch donor file digests, the fused materialization digests and the `FUSION.json` digest.

## What changed, and why

- **MS09 (shared loop `fit_with_early_stopping`)**:
  - adds monitor cadence (`monitor_every`; validation also runs on the final permitted epoch, and patience counts monitor evaluations);
  - adds `stop_class` (`no_improvement` or `budget`);
  - hashes the weights at every epoch, re-hashes the restored weights and requires them to equal the selected checkpoint's digest;
  - adds `early_stop` rule identity: implementation `modular.early_stop.v2`, monitor, cadence, patience, min_delta and every hard limit. A change to any of these produces a new identity, so a variant cannot pass as the author's recipe. Batch size, optimizer and loss are not part of the identity;
  - adds an optional `progress` callback.
  - Backward compatible: the new settings have defaults, and the inherited evaluator tests pass unchanged.
  - The loop is applied to the branch AEs, the core AE and the forecast fit (`evaluate_candidate`).
  - Exact literature reproductions do not use this loop. They keep the author's recipe.
- **MS10**:
  - one AE per branch on train only;
  - the internal validation is a separate, purged chronological split;
  - reconstruction is reported for train and internal validation against a reference;
  - the encoder is exported, reloaded through `load_donor`, and checked for output parity and weight-hash equality;
  - the decoder is exported for replay.
- **MS11**:
  - the model is rebuilt with every branch in R1 from the exported files, and each branch's weights must equal the exported digest and be untrainable;
  - fused train and internal validation are materialized in bounded batches to memmapped `.npy`;
  - the core AE trains on the bytes re-read from disk after their digest is re-verified;
  - the core donor binds its upstream.
  - "Bounded" means bounded memory. The fusion contract forbids value-changing transforms, so the representation is not range-clipped. Its train channel statistics are recorded.
- **Time alignment**:
  - the right-edge grids are asserted to be 1..24, 2..24 step 2 and 4..24 step 4;
  - row i of the materialization equals the fusion of window i;
  - perturbing only the last observation moves only the last fused block and the last latent step.
- **Heartbeat**:
  - a daemon thread writes one fsynced JSON line at most `interval` seconds apart, with (0, 60] enforced;
  - each line carries stage, epoch/update progress, last completed checkpoint, RSS/HWM, cgroup current and peak, CPU seconds, and the ETA basis.
- **Input swap**: `pretrain_from_train_npz` reads the evaluator-format TRAIN NPZ only and refuses any other split. A `governed_resource` input requires the M03 manifest digest.
- **Labels**: every donor has a `*.provenance.json` carrying `SYNTHETIC FIXTURE - PLUMBING_NOT_A_RESULT`. The engine `.manifest.json` schema is strict and owned by M01, so it is untouched. The same label is in PRETRAIN.json, PILOT.json and `DONORS_SYNTHETIC.json`.

## Published donors (for M04 R1/R2 plumbing only)

Location: `docs/audits/evidence/m02_pretraining_20260930/SYNTHETIC_pilot_s7_e100/`. The index is `DONORS_SYNTHETIC.json`, which lists the stage, donor sha256, manifest digests and provenance digest for each donor.

- Donor files: `pretrain/branch_000..002.keras` and `pretrain/core.keras` (core donor `86e1b118…`).
- Fused width actually built: **48 channels** (3 branches × 16), shape 12×48 per row. Latent 6×8.

## ECL plan (input swap, waiting for the M03 declaration)

- ECL has 321 value columns including OT, and TRAIN rows [0, 18412).
- With the default of one branch per feature: 321 branch AEs, fused width 321 × 16 = **5136** channels on 12 steps, core projection 5136 → 64.
- Per-branch encoder ≈ 592 parameters and decoder ≈ 1.6 k.
- The fused materialization is about 18.4 k × 12 × 5136 × 4 B ≈ 4.5 GB on disk. It is memmapped, so RAM use stays bounded.
- The 321 AEs run sequentially, each with its own early stop. Cost is not yet measured: expect hours on CPU. A cost pilot on a subset of branches comes first.
- Not started, by order.

## Not done, refused or not measured

- No real data, so there is no scientific claim.
- No M01 facade revision has been consumed yet (none published).
- ECL donors: not started, waiting for the M03 declaration.
- Multi-seed and matched ablations: not run.
- No GPU was used.
- The 5090 host showed 12 GiB available during this lane but was not used, per the ineligibility order.
- RL early stopping is not this lane's.

Satoshi, successor technical lead — 2026-09-30
