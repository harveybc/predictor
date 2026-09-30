# BANKING77 executable request

Status: **NOT_RUN**. Licence: **CC BY-NC 4.0**. Research only. This request is not commercial deployment approval. `trust_remote_code` stays the owner's decision. This lane did not set it and did not run the trial.

The weight tree and the corpus are on disk. Acquiring those bytes is not a score and not a security certification of the remote Python.

Evidence: `docs/audits/evidence/banking77_request_20260930/REQUEST.json` and `ACQUISITION.json`.

## Interpreter

Existing CPython **3.12.7**, manylinux x86_64, glibc 2.43. No other interpreter was installed. Requires-Python from the local wheel METADATA was evaluated at 3.12.7 and at 3.12.13. **No dependency's metadata requires `>=3.13`, and none excludes 3.12.7 or 3.12.13.**

cp312 pin list: **CLOSED**. Same 87 versions. 65 distributions were re-hashed locally and are reusable. 22 cp313-only files are not installable on cp312; the index metadata for the same version names a cp312-compatible file. Those 22 files were not downloaded. This phase did not remeasure them. A CPU-only torch index was not substituted. `torch==2.14.0` still depends on the CUDA 13 nvidia libraries. Those wheels were not installed here. The device for a later run is **not chosen**.

## Differences that are not version changes

Marker evaluation, 3.12.7 against 3.13.0, extras empty:

- `aiohttp==3.14.3` requires `typing_extensions>=4.4` when `python_version < "3.13"`. True on 3.12.7, false on 3.13.0. Already satisfied by `typing-extensions==4.16.0`.
- `aiosignal==1.4.0` requires `typing-extensions>=4.2` on the same marker. Same existing pin.

`triton==3.8.0` local METADATA says `>=3.10,<3.15` and the index says `<3.15,>=3.10`. The specifier is the same. It includes 3.12.

The 22 filename and size differences are in `REQUEST.json` under `wheel_differences`. Net byte delta, cp312 minus cp313: **381728**. No package was added, removed, or bumped. Reference versions kept: `mteb==2.9.0`, `sentence-transformers==5.1.2`, and every pin in section 4 of the closure review.

## Snapshot

Repository `jinaai/jina-embeddings-v5-text-small`, revision `46ed7da5b47e4bca710b756313fafaf4110c6bd1`. The tree downloaded by `tools/df_b77_acquire_20260930.py` is **1369721378** bytes, MEASURED, the same figure previously carried as NOT_PRESENT. Each file's sha256 or git blob matches the hub declaration for that revision, and the git oids match `REQUEST.json`. A mismatch would have deleted the file. The `REVISION` file is the sha plus a newline and is not part of the 1369721378.

Local directory, gitignored: `.b77-acquired-20260930/model/jina-embeddings-v5-text-small`. The owner is not asked to build it.

Three remote Python files were downloaded as bytes and not imported. Call-time `snapshot_download(allow_patterns=["adapters/*"])` runs when the path is not a local directory, and it passes no revision. Call-time `AutoTokenizer.from_pretrained` sets `trust_remote_code=True` in `modeling_jina_embeddings_v5.py` and in `custom_st.py`. **Absence of subprocess, eval, or exec is not a security certification.** This download is not one either.

The harness in `tools/df_b77_executable_request_20260930.py` still closes the unpinned branch on a synthetic local snapshot: inert bytes, offline, no import of the remote modules. A missing file raises a refusal and does not download. The execution module does not contact the hub. The synthetic fixture is **1349** bytes, MEASURED, and it is not the weight tree.

## Corpus

`REQUEST.json` names the task `Banking77Classification`. The pinned mteb 2.9.0 task metadata, read as text, names repository `mteb/banking77`, revision `0fd18e25b25c072e09e0d92ab615fda904d66300`. That is the revision acquired. It is not `Banking77Classification.v2`.

The revision's files are `.gitattributes`, `prepare_data.py`, `train.jsonl`, and `test.jsonl`. The rows are JSON lines, so a dataset script was not required. `prepare_data.py` was stored as bytes and **not executed**. There is no dev split.

| split | rows | labels | support | bytes | sha256 |
| --- | ---: | ---: | --- | ---: | --- |
| train | 10003 | 77 | 35 to 187 | 1245265 | `d411780d8c0e18e166f5664c6cfe90dc9de399d722aa7cde282e31a771323ea7` |
| test | 3080 | 77 | 40 | 365101 | `fb1b0043ded745b8767687084786e6dd0a5f0ce03243b6131992a1c7ae2c2595` |

Corpus hub bytes: **1612800**, MEASURED. Directory: `.b77-acquired-20260930/corpus/mteb-banking77`.

## Disk

| figure | bytes | label |
| --- | ---: | --- |
| 65 reusable wheels, re-hashed | 2284173652 | MEASURED, not remeasured this phase |
| 22 cp312 replacements, not downloaded | 946243017 | INDEX_METADATA, not remeasured this phase |
| cp312 wheelhouse | 3230416669 | MEASURED+INDEX_METADATA |
| prior cp313 closure, cited | 3230034941 | MEASURED |
| cp313-only wheels, not installable here | 945861289 | MEASURED (local size; content hash not recomputed; not reused) |
| weight tree | 1369721378 | MEASURED |
| hub-cache copy of that tree | 1369721378 | NOT_PRESENT |
| pip cache | none | NOT_PRESENT (`--no-cache-dir`) |
| installed site-packages | none | NOT_PRESENT |
| BANKING77 corpus | 1612800 | MEASURED |
| this phase, model plus corpus | 1371334178 | MEASURED, 80.856321 seconds |
| synthetic harness fixture | 1349 | MEASURED |

Unique payload if the cp312 wheelhouse and the weight tree each exist once: **4600138047**. Adding the measured corpus once: **4601750847**. Adding the hub-cache copy the command would make of the weight tree: **5969859425**, and that copy was not made. Also keeping the cp313-only wheels: **6915720714**. Installed expansion is not in those totals.

## Cost pilot, not run

**PROPOSAL.** No inference was timed. The download seconds above are transfer, not a forward pass.

The acquired train file has no separate dev split. The pilot is cut from train only, in file order, and it does not read `test.jsonl`. For each of the 77 labels, the first two occurrences are the probe and the next two are the dev slice. Every label has at least 35 train rows, so the slice fits: **154** probe rows and **154** dev rows. That subset is not `Banking77Classification` and not a score.

The reference protocol stays the one already pinned: ten experiments, eight train rows per label, scored on the full test split of **3080** rows. This lane did not re-execute that sampler. A timeout in the middle of that corpus is not a score. mteb 2.9.0 has no max-rows flag, which is why the pilot is a declared row selection rather than a shorter invocation of the official task.

## Device, not chosen

CPU does not need the ungranted GPU diagnostic. GPU stays blocked until the owner grants that diagnostic. This lane did not run it and did not import TensorFlow.

A read-only occupancy query, not a placement, reported:

| UUID | role | used / total MiB | util | compute |
| --- | --- | --- | ---: | --- |
| `GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8` | preferred | 10 / 32607 | 0% | 12.0 |
| `GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9` | retained refusal | 14 / 16376 | 7% | 8.9 |
| `GPU-612d1e0c-33de-d5cc-56eb-06c0ae424326` | excluded, not a destination | 647 / 8188 | 29% | not queried |

Compute capability 12.0 is outside the TensorFlow 2.21 wheel text (`sm_60`, `sm_70`, `sm_80`, `sm_89`, `compute_90`). The CUDA 12.5 preparation pin was not applied to the preferred card. That pin is not this request's torch stack. No inference cost was measured on CPU or on a GPU, so neither device is selected. The retained published evaluation time is not this run and does not choose a device.

The reference command below still refuses unless the owner sets the three gates and `OWNER_DEVICE`. `OWNER_DEVICE=gpu` refuses, does not launch the diagnostic, and does not export an empty `CUDA_VISIBLE_DEVICES` for that path. `OWNER_DEVICE=cpu` is the owner's choice, not this lane's. This lane set none of those variables.

## Where metrics go

The synthetic prep record, already written, is `docs/audits/evidence/banking77_request_20260930/SYNTHETIC_METRIC_RECORD.json`. It is constructed labels and scores. It is not a BANKING77 score and not the Jina model.

The real trial would write `docs/audits/evidence/banking77_request_20260930/METRIC_RECORD.json`. That file is **NOT_PRESENT**. Status of the trial: **NOT_RUN**.

## Command

The command refuses unless the owner exports `OWNER_DOWNLOAD_WHEELS=1`, `OWNER_RUN_RESEARCH=1`, `OWNER_TRUST_REMOTE_CODE=1`, and chooses `OWNER_DEVICE`. This lane exported none of them. The snapshot and the corpus are the trees already acquired. The command does not download them again. Whether the datasets loader reads the copied JSON lines offline was not measured.

```bash
set -euo pipefail

# Status: NOT_RUN. Licence: CC BY-NC 4.0. Research only. No commercial deployment approval.
# The snapshot and the corpus are already on disk. This command does not download them.
# It does not set OWNER_TRUST_REMOTE_CODE. Device is not chosen by this lane.
# prepare_data.py is not executed. A kill is not a score.
: "${OWNER_DOWNLOAD_WHEELS:?NOT_RUN: owner has not granted the wheel download}"
: "${OWNER_RUN_RESEARCH:?NOT_RUN: owner has not granted the research run}"
: "${OWNER_TRUST_REMOTE_CODE:?NOT_RUN: owner has not decided trust_remote_code}"
: "${OWNER_DEVICE:?NOT_RUN: owner has not chosen the device}"
test "$OWNER_DOWNLOAD_WHEELS" = "1"
test "$OWNER_RUN_RESEARCH" = "1"
test "$OWNER_TRUST_REMOTE_CODE" = "1"

if [ "$OWNER_DEVICE" = "gpu" ]; then
  printf '%s\n' "NOT_RUN: the GPU path stays blocked until the owner grants the diagnostic. This command does not launch that diagnostic and does not set CUDA_VISIBLE_DEVICES empty for it." >&2
  exit 2
fi
if [ "$OWNER_DEVICE" != "cpu" ]; then
  printf '%s\n' "NOT_RUN: device is not chosen" >&2
  exit 2
fi

SNAPSHOT=".b77-acquired-20260930/model/jina-embeddings-v5-text-small"
CORPUS=".b77-acquired-20260930/corpus/mteb-banking77"
RECORD="docs/audits/evidence/banking77_request_20260930/ACQUISITION.json"
PINS="docs/audits/evidence/banking77_request_20260930/pins-cp312.txt"
REV="46ed7da5b47e4bca710b756313fafaf4110c6bd1"
CORPUS_REV="0fd18e25b25c072e09e0d92ab615fda904d66300"
WHEELHOUSE=".b77-wheelhouse"
VENV=".b77-venv"
HUB=".b77-run-cache"
OUT=".b77-run-output"

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_CACHE="$HUB"
export CUDA_VISIBLE_DEVICES=

test -f "$PINS"
test -f "$RECORD"
test -d "$SNAPSHOT"
test -f "$SNAPSHOT/REVISION"
test "$(cat "$SNAPSHOT/REVISION")" = "$REV"
test -f "$SNAPSHOT/model.safetensors"
test -f "$SNAPSHOT/config.json"
test -d "$SNAPSHOT/adapters/classification"
test -d "$SNAPSHOT/adapters/clustering"
test -d "$SNAPSHOT/adapters/retrieval"
test -d "$SNAPSHOT/adapters/text-matching"
test -d "$CORPUS"
test -f "$CORPUS/REVISION"
test "$(cat "$CORPUS/REVISION")" = "$CORPUS_REV"
test -f "$CORPUS/train.jsonl"
test -f "$CORPUS/test.jsonl"

python3.12 tools/df_b77_acquire_20260930.py --check --dest .b77-acquired-20260930 --record "$RECORD"

# Existing CPython 3.12.7. Do not install another interpreter.
# Do not install cp313 wheels into it. Download uses that interpreter's pip
# and does not install into the base environment. Install is offline, into a new venv.
python3.12 -m venv "$VENV"
python3.12 -m pip download --dest "$WHEELHOUSE" --no-cache-dir --no-deps --only-binary=:all: --require-hashes -r "$PINS"
"$VENV/bin/python" -m pip install --no-index --no-cache-dir --no-deps --find-links "$WHEELHOUSE" --require-hashes -r "$PINS"

DEST="$HUB/models--jinaai--jina-embeddings-v5-text-small/snapshots/$REV"
mkdir -p "$DEST"
cp -a "$SNAPSHOT"/. "$DEST"/
mkdir -p "$HUB/models--jinaai--jina-embeddings-v5-text-small/refs"
printf '%s\n' "$REV" > "$HUB/models--jinaai--jina-embeddings-v5-text-small/refs/main"

# JSON lines only. prepare_data.py is not copied and not executed.
CDEST="$HUB/datasets--mteb--banking77/snapshots/$CORPUS_REV"
mkdir -p "$CDEST"
cp -a "$CORPUS/train.jsonl" "$CORPUS/test.jsonl" "$CORPUS/.gitattributes" "$CDEST"/
printf '%s\n' "$CORPUS_REV" > "$CDEST/REVISION"
mkdir -p "$HUB/datasets--mteb--banking77/refs"
printf '%s\n' "$CORPUS_REV" > "$HUB/datasets--mteb--banking77/refs/main"

# Reference protocol, not the 154/154 train pilot. mteb 2.9.0 --device accepts
# only an int CUDA index, so it is not passed. Empty CUDA_VISIBLE_DEVICES is
# reached only after the owner sets OWNER_DEVICE=cpu. The GPU branch exited
# above and did not export it. The pinned registry's loader_kwargs set
# trust_remote_code. That is the owner's decision, gated above.
# Offline mode makes a missing file a refusal, not a download.
# 8G and 30m are an unmeasured ceiling. A kill is not a score.
mkdir -p "$OUT"
crispdm-run -m 8G -t 30m -n b77-reference -- \
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HF_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 CUDA_VISIBLE_DEVICES= HF_HUB_CACHE="$HUB" \
    "$VENV/bin/python" -m mteb run \
      --model jinaai/jina-embeddings-v5-text-small \
      --model-revision "$REV" \
      --tasks Banking77Classification \
      --batch-size 8 \
      --no-co2-tracker \
      --output-folder "$OUT"
```

## Isolated install

1. Use the existing CPython 3.12.7. Do not install another interpreter. Do not download another CUDA or torch stack beyond the pinned wheels above.
2. `python3.12 -m venv` into `.b77-venv`. Do not install into the base environment.
3. Do not install the cp313 wheels into that venv.
4. `pip download` the pins file with `--require-hashes --no-deps --only-binary=:all: --no-cache-dir` using `python3.12`, so a cp313 file cannot satisfy the hash.
5. `pip install` into the venv with `--no-index --no-deps --no-cache-dir --require-hashes`.
6. Keep every reference version. Do not follow a newer index default. Do not switch torch to a CPU-only index.
7. The local snapshot is `.b77-acquired-20260930/model/jina-embeddings-v5-text-small`, revision `46ed7da5b47e4bca710b756313fafaf4110c6bd1`, with a `REVISION` file of that sha. The corpus is `.b77-acquired-20260930/corpus/mteb-banking77`. The command checks hashes and does not download either tree.
8. Copy the snapshot into an empty `.b77-run-cache`. Copy the corpus JSON lines beside it. Do not copy `prepare_data.py`. Export `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`. Unset hub tokens.
9. Do not choose the device in this lane. The GPU branch does not export an empty `CUDA_VISIBLE_DEVICES`.
10. The mteb registry loader sets `trust_remote_code`. Running that step is the owner's decision, not this lane's.

## What this lane did not do

- Did not import, execute, or score the remote model. Did not set `trust_remote_code`.
- Did not execute `prepare_data.py` or the three model Python files.
- Did not install the closure and did not install a Python 3.13 interpreter.
- Did not publish a BANKING77 score. Did not run the 154-row pilot. Did not measure inference.
- Did not run the GPU diagnostic or the calibrations.
