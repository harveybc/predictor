# Worker venv for the unrun diagnostic

2026-09-30. Branch `satoshi/gpu-env-repair-20260930`. Preparation only. The diagnostic was not run, the two calibrations were not run, and no virtual environment was created on a worker.

Evidence: `docs/audits/evidence/GPU_ENV_REPAIR_20260930/WORKER_PREP.json` and `pins-worker-prep.txt`.

## What is still not retained

On the card that refused, UUID `GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9`, `libcuda.so.1` loads and TensorFlow registered zero devices. The missing soname, the driver version, and the CUDA and cuDNN versions of the build that refused are **NOT_RETAINED**. A CUDA 12.5 pin colliding with CUDA 13 sonames is **NOT_DEMONSTRATED**. This note does not promote that hypothesis.

## Pin

The preparation pin is `tensorflow==2.21.0`, `nvidia-cuda-runtime-cu12==12.5.82`, and `nvidia-cudnn-cu12==9.3.0.75`. The requirements file also names the sibling `nvidia-*-cu12` wheels from the same tested-build floor, at the versions already installed in the coordinator prefix. Forty-four wheels were re-hashed locally: **2 576 641 397** bytes, MEASURED, the same download total as the coordinator install. Nothing was re-downloaded to produce the pin. `pip` itself is not in that file.

The coordinator prefix, by filename and not by load, contains `libcudart.so.12` and does not contain `libcudart.so.13`. There is no `nvidia/cu13` directory. The wheel text `build_info.py` was read and not imported. It declares CUDA 12.5.1, cuDNN 9, and compute capabilities `sm_60`, `sm_70`, `sm_80`, `sm_89`, `compute_90`. That list is not a certificate for every card.

## Do not copy the prefix

`.repair-env` stays on the coordinator. It was not tested on a worker. Copying it would be invalid: `pyvenv.cfg` and the scripts record absolute paths of the machine that created them. The worker creates its own environment and uses a relative interpreter.

## Capability gate, then the venv

The preferred card is UUID `GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8`. A read-only query on 2026-09-30 reported compute capability **12.0**, 10 MiB used of 32 607 MiB, utilization 0. Capability 12.0 is outside the declared list, so this pin was **not** installed there. UUID `GPU-b77fc3ad-db77-b648-dc15-ec79b65e2519` was also 12.0 and was not a destination. UUID `GPU-612d1e0c-33de-d5cc-56eb-06c0ae424326` is excluded. It had 647 MiB used of 8 188 MiB and 29% utilization. It was queried and not used.

The retained refusal card reported compute capability 8.9, which is `sm_89`, with 14 MiB used of 16 376 MiB. No venv was created there either. The query placed no process.

The gate a later shell must apply, before `python3.12 -m venv`, is the same list. If the card's compute capability is not 6.0, 7.0, 8.0, 8.9, or 9.0, do not install this pin. Do not treat that refusal as evidence about sonames.

From the worktree root on the worker, only after that gate passes:

```bash
python3.12 -m venv .worker-diag-env
.worker-diag-env/bin/python -m pip install --no-cache-dir --require-hashes \
  -r docs/audits/evidence/GPU_ENV_REPAIR_20260930/pins-worker-prep.txt
```

The interpreter is `.worker-diag-env/bin/python`, relative to the worktree. It is not the coordinator prefix.

## Diagnostic, not run

**NOT_RUN.** No admission for it was used. The command below is the shape, not a launch. `--verify` starts a new child. Do not set `CUDA_VISIBLE_DEVICES` empty around it: an empty device list is the launcher's no-CUDA trial, not this diagnostic. The child is started with the declared UUID. The supervisor still does not import TensorFlow.

```bash
crispdm-run -m 6G -t 300s -n gpu-launcher-diag -- \
  .worker-diag-env/bin/python tools/df_placement_contract.py \
  --placement GPU \
  --device-uuid GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9 \
  --cpu-seconds 300 \
  --verify
```

That UUID is the retained refusal, and its measured capability is inside the declared list. It is not an instruction to run. The preferred UUID is not substituted into this command, because this pin does not fit the capability that was measured there. `-m 6G` is 6 GiB host RAM. `-t 300s` is wall time. `--cpu-seconds 300` is `RLIMIT_CPU` in the child. One fresh admission. Not `-q`.

The two calibrations stay unrun: 4 800 CPU seconds, 3 600 summed child wall seconds, and the 12 GiB envelope. The envelope was not reduced. 18 GiB was not requested.

## Occupancy

Checked, read-only, with an existing key and no new credential. Recorded above by UUID. The venv procedure itself is **WRITTEN_NOT_RUN**.
