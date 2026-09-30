# BANKING77 executable request

Status: **NOT_RUN**. Licence: **CC BY-NC 4.0**. Research only. This request is not commercial deployment approval. `trust_remote_code` stays the owner's decision. This lane did not set it and did not run the trial.

This replaces the non-executable approval request in `RETSU_BANKING77_CLOSURE_REVIEW_2026_09_30.md` section 5. The pin versions do not change. The interpreter does.

Evidence: `docs/audits/evidence/banking77_request_20260930/REQUEST.json`.

## Interpreter

Existing CPython **3.12.7**, manylinux x86_64, glibc 2.43. No other interpreter was installed. Requires-Python from the local wheel METADATA was evaluated at 3.12.7 and at 3.12.13. **No dependency's metadata requires `>=3.13`, and none excludes 3.12.7 or 3.12.13.**

cp312 pin list: **CLOSED**. Same 87 versions. 65 distributions were re-hashed locally and are reusable. 22 cp313-only files are not installable on cp312; the index metadata for the same version names a cp312-compatible file. Those 22 files were not downloaded. A CPU-only torch index was not substituted. `torch==2.14.0` still depends on the CUDA 13 nvidia libraries. The device below is CPU anyway. GPU is not requested.

## Differences that are not version changes

Marker evaluation, 3.12.7 against 3.13.0, extras empty:

- `aiohttp==3.14.3` requires `typing_extensions>=4.4` when `python_version < "3.13"`. True on 3.12.7, false on 3.13.0. Already satisfied by `typing-extensions==4.16.0`.
- `aiosignal==1.4.0` requires `typing-extensions>=4.2` on the same marker. Same existing pin.

`triton==3.8.0` local METADATA says `>=3.10,<3.15` and the index says `<3.15,>=3.10`. The specifier is the same. It includes 3.12.

The 22 filename and size differences are in `REQUEST.json` under `wheel_differences`. Net byte delta, cp312 minus cp313: **381728**. No package was added, removed, or bumped. Reference versions kept: `mteb==2.9.0`, `sentence-transformers==5.1.2`, and every pin in section 4 of the closure review.

## Snapshot

Repository `jinaai/jina-embeddings-v5-text-small`, revision `46ed7da5b47e4bca710b756313fafaf4110c6bd1`. Fully materialised tree: **1369721378** bytes. Label: **NOT_PRESENT**. This command does not fetch that tree.

The files the closure review already fetched were re-hashed locally (MEASURED) and match that review. The weight files, tokenizer payload, vocabulary, merges, readme, and gitattributes were not fetched. Per-file hashes and the revision are in `REQUEST.json`.

Three remote Python files were read statically by that review. Call-time `snapshot_download(allow_patterns=["adapters/*"])` runs when the path is not a local directory, and it passes no revision. Call-time `AutoTokenizer.from_pretrained` sets `trust_remote_code=True` in `modeling_jina_embeddings_v5.py` and in `custom_st.py`. **Absence of subprocess, eval, or exec is not a security certification.**

The harness in `tools/df_b77_executable_request_20260930.py` closes that branch on a synthetic local snapshot: inert bytes in the same roles, offline, no import of the remote modules. A missing file raises a refusal and does not download. `snapshot_download` without the pinned revision cannot fetch a new one. The synthetic fixture is **1349** bytes, MEASURED, and it is not the weight tree.

## Disk

| figure | bytes | label |
| --- | ---: | --- |
| 65 reusable wheels, re-hashed | 2284173652 | MEASURED |
| 22 cp312 replacements, not downloaded | 946243017 | INDEX_METADATA |
| cp312 wheelhouse | 3230416669 | MEASURED+INDEX_METADATA |
| prior cp313 closure, cited | 3230034941 | MEASURED |
| cp313-only wheels, not installable here | 945861289 | MEASURED (local size; content hash not recomputed; not reused) |
| weight tree | 1369721378 | NOT_PRESENT |
| hub-cache copy of that tree | 1369721378 | NOT_PRESENT |
| pip cache | none | NOT_PRESENT (`--no-cache-dir`) |
| installed site-packages | none | NOT_PRESENT |
| BANKING77 corpus | none | NOT_PRESENT |
| synthetic harness fixture | 1349 | MEASURED |

Unique payload if the cp312 wheelhouse and the weight tree each exist once: **4600138047** (mixed MEASURED, INDEX_METADATA, NOT_PRESENT). Adding the hub-cache copy the command makes: **5969859425**. Also keeping the cp313-only wheels: **6915720714**. Installed expansion is not in those totals.

## Cost pilot

`crispdm-run -m 8G -t 30m -n b77-cost-pilot`. Device CPU. `CUDA_VISIBLE_DEVICES` empty. mteb 2.9.0's `--device` is an int CUDA index, so the command does not pass it; an empty device list is what makes that CLI select cpu. Batch size 8. mteb 2.9.0 has no max-rows flag; the wall clock is the cap. If the wrapper kills the process, that is not a score.

## Where metrics go

The synthetic prep record, already written, is `docs/audits/evidence/banking77_request_20260930/SYNTHETIC_METRIC_RECORD.json`. It is constructed labels and scores. It is not a BANKING77 score and not the Jina model.

The real trial would write `docs/audits/evidence/banking77_request_20260930/METRIC_RECORD.json`. That file is **NOT_PRESENT**. Status of the trial: **NOT_RUN**.

## Command

The command refuses unless the owner exports `OWNER_DOWNLOAD_WHEELS=1`, `OWNER_RUN_RESEARCH=1`, and `OWNER_TRUST_REMOTE_CODE=1`. This lane exported none of them.

```bash
set -euo pipefail

# Status: NOT_RUN. Licence: CC BY-NC 4.0. Research only. No commercial deployment approval.
# Run from the repository root. This command does not fetch the weight tree.
# A missing snapshot file is a refusal. GPU is not requested.
: "${OWNER_DOWNLOAD_WHEELS:?NOT_RUN: owner has not granted the wheel download}"
: "${OWNER_RUN_RESEARCH:?NOT_RUN: owner has not granted the research run}"
: "${OWNER_TRUST_REMOTE_CODE:?NOT_RUN: owner has not decided trust_remote_code}"
test "$OWNER_DOWNLOAD_WHEELS" = "1"
test "$OWNER_RUN_RESEARCH" = "1"
test "$OWNER_TRUST_REMOTE_CODE" = "1"

: "${B77_WHEELHOUSE:?}"
: "${B77_VENV:?}"
: "${B77_LOCAL_SNAPSHOT:?}"
: "${B77_HUB_CACHE:?}"
: "${B77_OUTPUT:?}"

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=
export HF_HUB_CACHE="$B77_HUB_CACHE"

PINS="docs/audits/evidence/banking77_request_20260930/pins-cp312.txt"
REV="46ed7da5b47e4bca710b756313fafaf4110c6bd1"
test -f "$PINS"
test -d "$B77_LOCAL_SNAPSHOT"
test -f "$B77_LOCAL_SNAPSHOT/REVISION"
test "$(cat "$B77_LOCAL_SNAPSHOT/REVISION")" = "$REV"
test -f "$B77_LOCAL_SNAPSHOT/model.safetensors"
test -f "$B77_LOCAL_SNAPSHOT/config.json"
test -d "$B77_LOCAL_SNAPSHOT/adapters/classification"
test -d "$B77_LOCAL_SNAPSHOT/adapters/clustering"
test -d "$B77_LOCAL_SNAPSHOT/adapters/retrieval"
test -d "$B77_LOCAL_SNAPSHOT/adapters/text-matching"

# Existing CPython 3.12.7. Do not install another interpreter.
# Do not install cp313 wheels into it. Download uses that interpreter's pip
# and does not install into the base environment. Install is offline, into a new venv.
python3.12 -m venv "$B77_VENV"
python3.12 -m pip download --dest "$B77_WHEELHOUSE" --no-cache-dir --no-deps --only-binary=:all: --require-hashes -r "$PINS"
"$B77_VENV/bin/python" -m pip install --no-index --no-cache-dir --no-deps --find-links "$B77_WHEELHOUSE" --require-hashes -r "$PINS"

DEST="$B77_HUB_CACHE/models--jinaai--jina-embeddings-v5-text-small/snapshots/$REV"
mkdir -p "$DEST"
cp -a "$B77_LOCAL_SNAPSHOT"/. "$DEST"/
mkdir -p "$B77_HUB_CACHE/models--jinaai--jina-embeddings-v5-text-small/refs"
printf '%s\n' "$REV" > "$B77_HUB_CACHE/models--jinaai--jina-embeddings-v5-text-small/refs/main"

# mteb 2.9.0 --device accepts only an int CUDA index, so it is not passed.
# An empty CUDA_VISIBLE_DEVICES makes that CLI select cpu. GPU is not requested.
# The pinned mteb registry's loader_kwargs set trust_remote_code. That is the
# owner's decision, gated above. This command adds no second override.
# Offline mode makes a missing file a refusal, not a download.
# Cost pilot: 8G memory, 30m wall, batch size 8. A kill is not a score.
mkdir -p "$B77_OUTPUT"
crispdm-run -m 8G -t 30m -n b77-cost-pilot -- \
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HF_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 CUDA_VISIBLE_DEVICES= HF_HUB_CACHE="$B77_HUB_CACHE" \
    "$B77_VENV/bin/python" -m mteb run \
      --model jinaai/jina-embeddings-v5-text-small \
      --model-revision "$REV" \
      --tasks Banking77Classification \
      --batch-size 8 \
      --no-co2-tracker \
      --output-folder "$B77_OUTPUT"
```

## Isolated install

1. Use the existing CPython 3.12.7. Do not install another interpreter. Do not download another CUDA or torch stack beyond the pinned wheels above.
2. `python3.12 -m venv` into an empty directory. Do not install into the base environment.
3. Do not install the cp313 wheels into that venv.
4. `pip download` the pins file with `--require-hashes --no-deps --only-binary=:all: --no-cache-dir` using `python3.12`, so a cp313 file cannot satisfy the hash.
5. `pip install` into the venv with `--no-index --no-deps --no-cache-dir --require-hashes`.
6. Keep every reference version. Do not follow a newer index default. Do not switch torch to a CPU-only index.
7. Point `B77_LOCAL_SNAPSHOT` at a complete local tree of revision `46ed7da5b47e4bca710b756313fafaf4110c6bd1` with a `REVISION` file of that sha. If it is absent, the command exits. It does not download the 1369721378 bytes.
8. Copy the snapshot into an empty `HF_HUB_CACHE`. Export `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`. Unset hub tokens.
9. Leave `CUDA_VISIBLE_DEVICES` empty. Do not request a GPU.
10. The mteb registry loader sets `trust_remote_code`. Running step 10 is the owner's decision, not this lane's.

## What this lane did not do

- Did not import, execute, or score the remote model. Did not set `trust_remote_code`.
- Did not call `snapshot_download` against the network. Did not download the weight tree.
- Did not install the closure and did not install a Python 3.13 interpreter.
- Did not publish a BANKING77 score.
