# GPU environment repair — preparation only

2026-09-30. Branch `satoshi/gpu-env-repair-20260930`, tip `4d548a82`. This note prepares an isolated interpreter. It does not establish a GPU pilot, and it does not run the diagnostic that would.

Evidence: `docs/audits/evidence/GPU_ENV_REPAIR_20260930/REPAIR_PREP.json`.

## What the retained record establishes

The specific library TensorFlow failed to dlopen is **NOT_RETAINED**. No missing symbol is invented here.

The refusal JSON (`docs/audits/evidence/QRM_INTEGRATION_20260929/PLACEMENT_WORKER_REFUSED.json`) does not retain stderr, a `dlerror`, a TensorFlow `build_info` (compiled CUDA or cuDNN), or a driver version. The sentence that was kept is in `docs/audits/work_plan/SATOSHI_QRM_INTEGRATION_2026_09_29.md` §5.3, ellipsis included:

> Cannot dlopen some GPU libraries … Skipping registering GPU devices

Sibling JSON of that refusal (`PLACEMENT_COORDINATOR_GPU.json`, `PLACEMENT_REFUSAL_WRONG_UUID.json`, `PLACEMENT_REFUSAL_EMPTY_CVD.json`, `GOVERNED_ROUTE.json`, `CAPACITY.json`, `RUNNER_IDENTITY.json`) also does not retain that stderr. The telemetry self-checks under `docs/audits/evidence/qrm02_f3_20260929/` retain the version strings and the same refusal code, not the dlopen target.

What those records do establish, on the admitted worker:

| fact | retained value |
|---|---|
| placement | GPU, declared |
| device | UUID `GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9` verified through the driver |
| `libcuda.so.1` | loads (§5.3). The refusal JSON does not retain a dlerror |
| framework devices | `[]` |
| placement probe | `null` |
| interpreter named by §5.3 | Python 3.12.13 / TensorFlow 2.21.0, the anaconda env `tensorflow` on that host |
| library path | that interpreter's own wheels, both generations, both layouts (`nvidia/cu13/lib` first, then `nvidia/<pkg>/lib`) |
| sonames the contract recorded as answering | `libcublas.so.13`, `libcublasLt.so.13`, `libcudart.so.13`, `libcudnn.so.9`, `libcufft.so.12`, `libcurand.so.10`, `libcusolver.so.12`, `libcusparse.so.12`, `libnvJitLink.so.13` |
| code | `FRAMEWORK_REGISTERED_NO_DEVICE`, carried by `GPU_REQUEST_FELL_BACK_TO_CPU` |

`FRAMEWORK_REGISTERED_NO_DEVICE` names the failure. It is not the cause. The contract had already recorded every family as loadable. The CUDA 13 sonames are what that newest-first search recorded, not a measurement of which `dlopen` failed. TensorFlow's compiled CUDA and cuDNN versions were not retained, so this note does not say which soname was missing.

A day-earlier capacity census (`docs/audits/evidence/RB01_RB03_20260928/HOST_CAPACITY.json`, 2026-09-28) records the driver string `580.178.04` beside the same published UUID. That string is not the refusal's retained metadata and is not the compatibility constraint below.

## Compatibility constraint that was used

Source: the Linux GPU table at <https://www.tensorflow.org/install/source> (page updated 2026-03-12 UTC). Row `tensorflow-2.21.0`:

| Python | Compiler | Build tools | cuDNN | CUDA |
|---|---|---|---|---|
| 3.10–3.13 | Clang 18.1.8 | Bazel 7.4.1 | 9.3 | 12.5 |

That table has no driver column. It does not list the worker's driver, because the refusal did not retain a driver version.

The pip page <https://www.tensorflow.org/install/pip> states a Linux driver floor `>= 525.60.13`, next to CUDA Toolkit 12.3 and cuDNN SDK 8.9.7. That page is not the 2.21 tested-build row. Its toolkit pair is not CUDA 12.5 / cuDNN 9.3, so it was not the constraint.

The `tensorflow==2.21.0` distribution declares the userspace as the `and-cuda` extra, and only as `nvidia-*-cu12` ranges: lower bounds at the CUDA 12.5 / cuDNN 9.3 wheels (`nvidia-cuda-runtime-cu12>=12.5.82`, `nvidia-cudnn-cu12>=9.3.0.75`, and the sibling lower bounds), upper bounds excluding CUDA 13 (`nvidia-*-cu12<13` except cuDNN `<10` and NCCL `<3`). An unconstrained `tensorflow[and-cuda]==2.21.0` resolve on this date floated to `nvidia-cuda-runtime-cu12==12.9.79` and `nvidia-cudnn-cu12==9.27.0.42`. Those satisfy the ranges. They are not the tested-build pair, and they were not installed.

## Isolated environment

`.repair-env/` at the worktree root. Gitignored by `.repair-env/`. It is not the anaconda env, it was not activated against a GPU, and TensorFlow was not imported. No kernel driver was installed. Existing interpreters were not mutated.

The prefix was bootstrapped from the coordinator's base CPython **3.12.7** so the admitted worker's env `tensorflow` would not be used. The retained worker interpreter is **3.12.13**. The wheels are `cp312`. The patch levels are not the same, and no new CPython was installed to close that gap.

CUDA userspace is pinned at the extra's lower bounds, which are the tested-build floor, not the floated 12.9 set and not the cu13 tree that was first on the worker's path. On disk the layout is `nvidia/<pkg>/lib` only. Filename inventory, not a load: `libcudart.so.12` is present, `libcudart.so.13` is not, and there is no `nvidia/cu13/lib`.

| pin | role |
|---|---|
| `tensorflow==2.21.0` | the retained framework version, `cp312` manylinux wheel |
| `nvidia-cuda-runtime-cu12==12.5.82` | CUDA 12.5 runtime (`libcudart.so.12`) |
| `nvidia-cublas-cu12==12.5.3.2` | cuBLAS |
| `nvidia-cudnn-cu12==9.3.0.75` | cuDNN 9.3 (`libcudnn.so.9`) |
| `nvidia-cufft-cu12==11.2.3.61` | cuFFT (`libcufft.so.11`, the soname this wheel ships) |
| `nvidia-curand-cu12==10.3.6.82` | cuRAND |
| `nvidia-cusolver-cu12==11.6.3.83` | cuSOLVER (`libcusolver.so.11`) |
| `nvidia-cusparse-cu12==12.5.1.3` | cuSPARSE |
| `nvidia-cuda-cupti-cu12==12.5.82` | CUPTI |
| `nvidia-cuda-nvcc-cu12==12.5.82` | nvcc |
| `nvidia-cuda-nvrtc-cu12==12.5.82` | nvrtc |
| `nvidia-nccl-cu12==2.27.7` | NCCL |
| `nvidia-nvjitlink-cu12==12.5.82` | nvJitLink (`libnvJitLink.so.12`) |

The closure pip installed with those pins, exact versions: `absl-py==2.5.0`, `astunparse==1.6.3`, `certifi==2026.7.22`, `charset-normalizer==3.5.2`, `flatbuffers==25.12.19`, `gast==0.7.0`, `google-pasta==0.2.0`, `grpcio==1.84.0`, `h5py==3.14.0`, `idna==3.20`, `keras==3.15.1`, `libclang==18.1.1`, `markdown-it-py==4.2.0`, `mdurl==0.1.2`, `ml_dtypes==0.6.0`, `namex==0.1.0`, `numpy==2.5.3`, `opt_einsum==3.4.0`, `optree==0.20.0`, `packaging==26.3`, `protobuf==7.36.2`, `Pygments==2.21.0`, `requests==2.34.2`, `rich==15.0.0`, `setuptools==84.0.0`, `six==1.17.0`, `termcolor==3.3.0`, `typing_extensions==4.16.0`, `urllib3==2.8.0`, `wheel==0.48.0`, `wrapt==2.5.0`. The prefix's installer is `pip==26.2.1`, upgraded before the repair download and not part of the download total below.

## Sizes, measured

`pip download` wrote 44 wheels under `.repair-env/wheels`, then `pip install --no-index` installed them into the prefix. Cap 20 GiB = 21 474 836 480 B. The download is under the cap. The set is complete, not partial.

| measure | bytes | basis |
|---|---:|---|
| download | **2 576 641 397** | sum of `st_size` of the 44 wheels |
| installed | **5 135 683 136** | sum of `st_size` of regular files under `.repair-env/lib/python3.12/site-packages`. Symlinks not followed. The wheel directory is not included |
| prefix excluding the wheels | 5 135 701 564 | the same tree plus the small prefix outside site-packages |

Per-wheel sizes are in the JSON. The largest are `nvidia_cudnn_cu12` 577 163 502 B, `tensorflow` 572 611 111 B, `nvidia_cublas_cu12` 363 313 169 B, and `nvidia_nccl_cu12` 322 546 339 B.

## Smoke, written and not run

**NOT_RUN.** No diagnostic allocation was granted. The `-m 6G -t 300` shape is the earlier placement-verify command's shape, not an allocation for this repair. On a non-interactive shell the retained notes invoke the launcher as `<HOME>/.local/bin/crispdm-run`.

```
crispdm-run -m 6G -t 300 -n gpu-env-repair-diag -- \
    <WORKTREE>/.repair-env/bin/python tools/df_placement_contract.py \
    --placement GPU \
    --device-uuid GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9 \
    --verify
```

`tools/df_placement_contract.py --verify` calls `verify_or_refuse`, which calls `tools/df_tf_device_telemetry.py` `verify_declared_device`. That is the three facts: the driver UUID through `libcuda.cuDeviceGetUuid`, `tf.config.list_physical_devices("GPU")`, and one real op (`tf.device("/GPU:0")` plus a one-element tensor).

The integrated runner entry is `tools/df_e1_block.py` `run_units` (command `execute`). It refuses a bare subprocess, loads `tools/governed_run.py`, and calls the same `verify_or_refuse` before a cell. `execute` then continues into the cell. That is not this smoke, and it was not run.

## Rollback

Leave `.repair-env` in place. `.gitignore` covers it, and it must not be committed. To abandon it, from the worktree root:

```
rm -rf .repair-env
```

That deletes only this prefix. It does not touch the anaconda env, a driver, systemd, or any other interpreter.

## Not done

- The two calibrations were **not** run. 4 800 CPU s and 3 600 summed child wall s were not authorized.
- The 12 GiB device envelope was **not** reduced.
- The 18 GiB ceiling stays withdrawn. It was not moved.
- Existing interpreters were **not** mutated. The anaconda env `tensorflow` was not installed into and not activated.
- TensorFlow was not imported. `nvidia-smi` was not run. No operation was placed on a GPU. No kernel driver was installed.
- This preparation does not establish a GPU pilot.
