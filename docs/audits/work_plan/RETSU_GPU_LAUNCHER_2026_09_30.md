# GPU launcher — the check starts in a child

2026-09-30. Branch `satoshi/gpu-env-repair-20260930`. This note corrects the narration of the isolated prefix and of `--verify`. It does not run a GPU diagnostic, and it does not establish a GPU pilot.

Evidence: `docs/audits/evidence/GPU_LAUNCHER_20260930/PRE.json` and `POST.json`. The probe is `docs/audits/gpu_launcher_probe_20260930.py`. Both runs were one admission each, `crispdm-run -m 2G -t 120s`, `CUDA_VISIBLE_DEVICES` empty. No TensorFlow import. No CUDA library was mapped in the probe.

## An isolated environment was installed

An isolated environment was installed. That is not "no installation happened".

The measured bytes are the ones already recorded at `3f910fbf`, in `docs/audits/evidence/GPU_ENV_REPAIR_20260930/REPAIR_PREP.json` and `docs/audits/work_plan/RETSU_GPU_ENV_REPAIR_PREP_2026_09_30.md`:

| step | what it was | bytes |
|---|---|---:|
| download | 44 wheels into the prefix | 2 576 641 397 |
| isolated install | regular files under the prefix's `site-packages`, symlinks not followed | 5 135 683 136 |

Download and isolated installation both happened. They happened into `.repair-env` in this worktree, on the coordinator. The prefix was not repaired onto a worker, not deployed, and not GPU-tested. No existing interpreter was mutated.

Those are different acts from a diagnostic, from training, and from adoption. The diagnostic below was not run. Nothing was trained. The prefix was not adopted.

## What F1 showed, and what it did not

`tools/df_placement_contract.py` `main`, when `--verify` was set, wrote the prepared environment into `os.environ` of the same process and then verified there. On this glibc (`glibc 2.43`, from `CS_GNU_LIBC_VERSION`), `LD_LIBRARY_PATH` is read when a process starts. Assigning it afterwards does not change the search `dlopen` uses. Doing the assignment before the TensorFlow import does not fix that. The dynamic loader has already started with the process.

PRE, before that CLI was changed, with an innocent C library in a temporary directory and no CUDA loaded:

| process | return code | result |
|---|---:|---|
| child that assigns `LD_LIBRARY_PATH` and then loads by soname | 1 | the soname is not found |
| new process whose startup environment already contains the directory | 0 | the function returns 42 |

That is the smoke defect. It does not identify the library the admitted worker failed to dlopen. It does not prove that the new prefix repairs a GPU.

## What the entry point does now

The supervisor prepares a new mapping and starts a new process with that mapping. The child keeps the mapping it was started with and does the check. The supervisor does not write the mapping into its own `os.environ`, does not import TensorFlow, and does not dlopen a CUDA soname. Wheel directories are taken from the interpreter's own files. A soname that is not among those files is not confirmed by loading it in the supervisor.

`--verify` is that path. The child is `--verify-child`. If `--cpu-seconds` is set, that child installs `RLIMIT_CPU` on itself, soft and hard at the same integer, before `enforce_before_tensorflow` and before any TensorFlow import. A non-zero child is a refusal. The supervisor does not report it as success.

`--load-dir` / `--load-soname` is the same supervisor and a different child check: load one soname. It is how the mechanism was tested without CUDA. It is not the GPU diagnostic.

The integrated path was not the defect. `tools/governed_run.py` already passes `env` into `subprocess.run`. `tools/df_e1_block.py` `run_units` already builds the placement mapping and passes `env=child_env` to `df_cell_scope.supervise`, which passes it to `Popen`. The sealed replay passes `env=replay_env`. None of those call sites were changed.

POST, same cap, still no CUDA:

| check | result |
|---|---|
| in-process assignment of the same directory | return code 1; stdout is not 42 |
| public entry, fresh child, direct soname | supervisor return code 0; child stdout `42`; the supervisor did not import TensorFlow, did not map a CUDA library, and did not map the soname |
| `governed_run`'s `subprocess.run(..., env=env)` | child return code 0, result 42; the parent's `LD_LIBRARY_PATH` was unchanged; the source still passes `env` |
| `supervise` as `run_units` uses it | exit code 0; the child log is `42`; the parent's `LD_LIBRARY_PATH` was unchanged |
| public entry, soname whose `NEEDED` helper is not on the child's startup path | child return code 1 (`libretsu_helper.so` cannot be opened); supervisor return code 3, status `REFUSED`, not success |
| same soname with the helper present | child stdout `42`, so the refusal is the missing dependency |

`readelf -d` on that soname recorded `NEEDED` `libretsu_helper.so` and no `RPATH` or `RUNPATH`.

## The interpreter the unrun diagnostic would use

The prepared environment is on the coordinator: `.repair-env` in this worktree. The interpreter is the isolated prefix's `python3.12` (`.repair-env/bin/python3.12`). It has not been tested on the admitted worker. Do not transport the prefix. Copying a virtual environment whose scripts and `pyvenv.cfg` record absolute paths would be invalid; those paths are the machine that created the prefix, not a portable root.

The worker procedure is `docs/audits/work_plan/RETSU_GPU_WORKER_VENV_2026_09_30.md`. The worker creates `.worker-diag-env` with a relative interpreter after a compute-capability gate. That procedure was written and not run. The command in the next section is the coordinator-prefix shape. It is not the worker command.

The CUDA 12.5 pin is the tested-build floor used for that install (`tensorflow==2.21.0`, `nvidia-*-cu12` at the 12.5 / cuDNN 9.3 lower bounds). It is not a claim that the pin suits every GPU architecture. The installed wheel's `build_info.py`, read as text and not imported, declares CUDA 12.5.1, cuDNN 9, and compute capabilities `sm_60`, `sm_70`, `sm_80`, `sm_89`, `compute_90`. That list is not a certificate for every card.

## Diagnostic, not run

**NOT_RUN.** The owner did not approve running it. The numbers are a spend limit, not a measured footprint. One execution. One fresh admission: a single `crispdm-run`, which takes a new lease. Not `-q`. Not a reused scope. Not a second launch. No training: the child checks the three facts (driver UUID, TensorFlow's device list, one placed op). It does not fit.

From this worktree, so the relative interpreter is the isolated prefix's `python3.12`:

```
crispdm-run -m 6G -t 300s -n gpu-launcher-diag -- \
  .repair-env/bin/python3.12 tools/df_placement_contract.py \
  --placement GPU \
  --device-uuid GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9 \
  --cpu-seconds 300 \
  --verify
```

`-m 6G` is 6 GiB host RAM, IEC, as the launcher parses sizes. `-t 300s` is 300 wall seconds for that tree. `--cpu-seconds 300` is 300 CPU seconds, `RLIMIT_CPU` in the child. Do not export an empty `CUDA_VISIBLE_DEVICES` around this command. The child's startup environment carries the declared UUID. The supervisor still does not import TensorFlow and does not dlopen.

## Not in this lane

The two calibrations were not launched: 4 800 CPU seconds, 3 600 summed child wall seconds, and the 12 GiB envelope. The 12 GiB envelope was not reduced. 18 GiB was not requested.
