# BANKING77 closure review: the pinned source read, the closure priced, the run not started

Classification lane, for Retsu. 2026-09-30.
Order from Musashi, the same day: fetch and read the pinned remote Python without
executing it; inventory transitive revisions and the concrete risky operations;
price and pin the complete tier-1 dependency closure; write one approval request
for that revision and a research run. Do not perform the run.

Branch `satoshi/banking77-closure-review-20260930`, tip `c89a8ff4`. Nothing here
was committed or pushed.

The shipped one-shot wrapper remains what it was: 12 options, and 706/894 tokens
against head budgets 192/256. That is not impossibility at every budget. It is
not reopened.

**Licence: CC BY-NC 4.0. This review confers no commercial deployment approval.**
Research only. `trust_remote_code` stays false until the owner says otherwise.
The research run is **NOT_RUN**.

Evidence: `docs/audits/evidence/banking77_closure_20260930/CLOSURE.json`.

---

## 1. The source was fetched, and the revision that was hashed is the pin

Repository `jinaai/jina-embeddings-v5-text-small`, revision
**`46ed7da5b47e4bca710b756313fafaf4110c6bd1`**.

The revision endpoint's sha is that revision. Every fetched object returned the
same `x-repo-commit`. Git blob etags equal the tree oids. Five files also match
the digests pinned on 2026-09-29:

| file | bytes | sha256 | prior digest |
| --- | ---: | --- | --- |
| `custom_st.py` | 3,929 | `52e0931dca24ca1a4fe4f0a5165f1b77e85deac6bebbd3532cda4c09253a29db` | match |
| `modeling_jina_embeddings_v5.py` | 4,132 | `389836c791ae345c164108c23858522590be32477b370c94b218cb1b4f10c69c` | match |
| `config.json` | 991 | `1af1e1269488c83d8b2332e42099f0d2201d687fbe074d1ed096c6201f283546` | match |
| `config_sentence_transformers.json` | 276 | `e13a56778b7ba8561a3d540173e682ff92acb8457a24d567fe92a8388402fb14` | match |
| `adapters/classification/adapter_config.json` | 884 | `7c3f89c17197f40070fae65449fd013c6cf0ee5565ceb45e9d5e292aa8472563` | match |

The tree at that revision is **22 files, 1,369,721,378 bytes** if fully
materialised. That equals the prior whole-repository figure. **Three** of the
22 are Python. The weight files (`model.safetensors`, four
`adapter_model.safetensors`, `tokenizer.json`) were not downloaded.
`snapshot_download` was not called.

Fetched for review, and only these: the three Python files, `config.json` (it
holds `auto_map`), `config_sentence_transformers.json`, `modules.json`,
`generation_config.json`, `tokenizer_config.json`, and the four
`adapter_config.json` files. Digests of the files that had no prior pin are in
`CLOSURE.json`. They were read and passed through `ast.parse`. They were not
imported and not executed.

`auto_map` is:

- `AutoConfig` → `configuration_jina_embeddings_v5.JinaEmbeddingsV5Config`
- `AutoModel` → `modeling_jina_embeddings_v5.JinaEmbeddingsV5Model`

`modules.json` names a second entry point, `custom_st.Transformer`. There is no
`__init__.py` in the tree. The only local import is
`modeling_jina_embeddings_v5.py` line 12, `from .configuration_jina_embeddings_v5 import JinaEmbeddingsV5Config`.
That file was downloaded. It imports only `transformers.Qwen3Config`. `custom_st.py`
imports nothing local. No local file is missing.

---

## 2. Concrete operations, with file and line

Nothing at module level calls out. Importing either entry point would execute
the import statements (torch, transformers, peft, `huggingface_hub`) and define
classes. `snapshot_download` and `trust_remote_code=True` run only when
`from_pretrained` or `Transformer.__init__` is called. This lane did not call
them.

| operation | file:lines | when | what |
| --- | --- | --- | --- |
| `snapshot_download(repo_id=..., allow_patterns=["adapters/*"])` | `modeling_jina_embeddings_v5.py:37-40` | call-time, inside `from_pretrained` | only in the else branch, when `os.path.isdir(base_model.name_or_path)` is false (line 34). The library writes the adapter tree into the hub cache. This source has no `open()`. |
| `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` | `modeling_jina_embeddings_v5.py:57-60` | call-time, inside `from_pretrained` | hardcoded, not taken from the caller. This revision has no tokenizer Python file. `tokenizer_config.json` names `tokenizer_class` `Qwen2Tokenizer`. |
| `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` | `custom_st.py:44-47` | call-time, inside `Transformer.__init__` | hardcoded on the tokenizer only. `AutoConfig.from_pretrained` (32-34) and `AutoModel.from_pretrained` (41-43) do not set the flag. `model_args` is forwarded as `model_kwargs`, so a caller can still pass it. |
| `from_pretrained` / `load_adapter` | `modeling_jina_embeddings_v5.py:25-32, 47-55` | call-time | `JinaEmbeddingsV5Config`, `Qwen3Model`, `PeftConfig`, then `load_adapter` for every name in `task_names`. Each can fetch from the hub when given a repository id. |
| `os.path.isdir` / `os.path.join` | `modeling_jina_embeddings_v5.py:34-35, 41, 43` | call-time | path test and joins. Not a write. |

**Not present** in the three Python files, at import time or call time:
`subprocess`, `os.system`, `socket`, `urllib`, `requests`, `open`, `write`,
`pickle`, `torch.load`, `cloudpickle`, `eval`, `exec`.

Deserialization: the fetched Python does not unpickle and does not call
`torch.load`. Weight loading is delegated to `Qwen3Model.from_pretrained` and
`load_adapter`. The tree ships safetensors only. No `.bin`, `.pt`, or `.pkl`.
Those files were not opened. `cloudpickle` and `dill` appear later only as
packages in the pip closure (`joblib`, `datasets`), not as calls in this source.

`mteb` 2.9.0's own registry, read and not executed, is a separate fact. File
`mteb/models/model_implementations/jina_models.py` at tag 2.9.0, sha256
`dabfd27f4687e1803fcd7c5d4ca275a34874fb3a60f271d4c6c195afd4d8f09e` (matches the
prior digest). It names this model, pins this same revision, and sets
`loader_kwargs` `trust_remote_code=True`, with the `Classification` prompt
mapped to the `classification` adapter. That is mteb's code, not the model
repository's. It was not run.

---

## 3. Transitive revisions, not a vibe

From the revision itself:

| where | pin |
| --- | --- |
| model card, required floors | `transformers>=4.57.0`, `torch>=2.8.0`, `peft>=0.15.2` |
| `config_sentence_transformers.json` `__version__` | sentence-transformers **5.1.2**, transformers **4.57.0**, pytorch **2.8.0** |
| `config.json` and `generation_config.json` | `transformers_version` **4.51.0** (the writer of the file, older than the card's floor) |
| readme `base_model` | `Qwen/Qwen3-0.6B-Base` |
| mteb 2.9.0 `adapted_from` | `Qwen/Qwen3-0.6B` (not the same string) |
| four adapter configs | LoRA, `r` 32, `lora_alpha` 32, `revision: null`, **no peft version field**. `lora_dropout` is 0.05 on classification and 0.1 on the other three. `target_modules` are the same set. `base_model_name_or_path` is this repository, not a content revision. |

The tier-1 closure below **meets the floors and does not equal the recorded
export versions**, except sentence-transformers, which is exactly 5.1.2.
`transformers==4.57.6` satisfies `>=4.57.0` and sentence-transformers 5.1.2's
own `transformers<5`. `torch==2.14.0` satisfies `>=2.8.0` and is not 2.8.0.
`peft==0.21.1` satisfies `>=0.15.2`. A fresh environment of this closure is
therefore **not** the transformers 5.17 line a symbol check was previously run
against: sentence-transformers 5.1.2 will not accept transformers 5.

---

## 4. The closure, priced and pinned

Input, as ordered: `mteb==2.9.0`, `sentence-transformers==5.1.2`, `peft`,
`datasets`, `scikit-learn`.

Resolved for **CPython 3.13**, **manylinux_2_28 x86_64**, default package index,
not a CPU-only index. **87 wheels.** Every wheel was downloaded into scratch
(gitignored) and sha256-checked against the index. **87/87 match.** Total
**3,230,034,941 bytes** (under the 8 GiB cap of 8,589,934,592; the download was
not stopped).

That total is the closure. The prior **16,228,123** is not.

| direct package | prior bytes | this wheel, locally hashed | delta |
| --- | ---: | ---: | ---: |
| `mteb==2.9.0` | 5,138,022 | 5,124,917 | −13,105 |
| `sentence-transformers==5.1.2` | 539,845 | 488,009 | −51,836 |
| `peft==0.21.1` | 863,928 | 835,387 | −28,541 |
| `datasets==5.0.1` | 542,418 | 559,079 | +16,661 |
| `scikit-learn==1.9.1` | 9,143,910 | 9,121,732 | −22,178 |
| **five direct wheels** | **16,228,123** | **16,129,124** | **−98,999** |
| **full closure, 87 wheels** | not resolved then | **3,230,034,941** | — |

The five names and versions match the prior tier-1 resolution. The bytes do
not, even before dependencies. The closure is about 199 times the old
direct-wheel figure. Most of it is `torch==2.14.0` (554,619,993 bytes) plus the
CUDA 13 nvidia libraries that the default Linux wheel depends on
(`nvidia-cudnn-cu13==9.24.0.43` alone is 553,099,438). A CPU-only index would
be a different closure and was not substituted.

**Installed bytes: `NOT_MEASURED_NO_PYTHON_3_13`.** No Python 3.13 interpreter
was on the machine. The system interpreter is 3.14.4. The conda interpreters
found are 3.12.7 and 3.12.13. A marker resolver fetched a CPython 3.13 toolchain
in order to evaluate markers; that toolchain was removed and was not used to
create an environment. Python 3.12 was not substituted and called 3.13. Nothing
was installed into laya, conda, or a predictor venv. `sentence-transformers`
was not imported against the model.

The exact pin set, also in `CLOSURE.json`:

```
accelerate==1.15.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
annotated-types==0.8.0
anyio==4.15.1
attrs==26.1.0
certifi==2026.7.22
charset-normalizer==3.5.2
cloudpickle==3.1.2
cuda-bindings==13.4.3
cuda-pathfinder==1.8.2
cuda-toolkit==13.0.3.0
datasets==5.0.1
dill==0.4.1
filelock==4.0.7
frozenlist==1.8.0
fsspec==2026.6.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==0.36.2
idna==3.20
Jinja2==3.1.6
joblib==1.6.0
markdown-it-py==4.2.0
MarkupSafe==3.0.3
mdurl==0.1.2
mpmath==1.3.0
mteb==2.9.0
multidict==6.9.1
multiprocess==0.70.19
narwhals==2.26.0
networkx==3.7
numpy==2.5.3
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.24.0.43
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.30.7
nvidia-nvjitlink==13.4.92
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
packaging==26.3
pandas==3.0.6
peft==0.21.1
pillow==12.3.0
polars==1.44.2
polars-runtime-32==1.44.2
propcache==0.5.4
psutil==7.2.2
pyarrow==25.0.1
pydantic==2.13.5
pydantic_core==2.46.5
Pygments==2.21.0
python-dateutil==2.9.0.post0
pytrec-eval-terrier==0.5.10
PyYAML==6.0.3
regex==2026.9.29
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
scikit-learn==1.9.1
scipy==1.18.1
sentence-transformers==5.1.2
setuptools==84.0.0
six==1.17.0
sympy==1.14.0
threadpoolctl==3.7.0
tokenizers==0.22.2
torch==2.14.0
tqdm==4.70.1
transformers==4.57.6
triton==3.8.0
typing-extensions==4.16.0
typing-inspection==0.4.4
urllib3==2.8.0
xxhash==4.0.1
yarl==1.25.1
```

---

## 5. Approval request

This is a request, not a grant, and not a run.

Approve or refuse, as one decision, a **research** run of the identified code
at revision **`46ed7da5b47e4bca710b756313fafaf4110c6bd1`**, and only of that
revision, on the **exact pinned requirement set in §4** (87 pins, download
**3,230,034,941** bytes, installed bytes **`NOT_MEASURED_NO_PYTHON_3_13`**).

The concrete remote operations that run would execute, if and only if the owner
later sets `trust_remote_code` and calls `from_pretrained`, are:

1. `modeling_jina_embeddings_v5.py:37-40` — `snapshot_download` of `adapters/*`
   when the model path is not already a local directory. It writes the hub cache.
2. `modeling_jina_embeddings_v5.py:57-60` and `custom_st.py:44-47` —
   `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`.
3. The call-time hub loads named in §2 (`Qwen3Model.from_pretrained`,
   `PeftConfig.from_pretrained`, `load_adapter`, and the sentence-transformers
   `AutoModel.from_pretrained`). No shell, no `eval`/`exec`, no `pickle`, no
   `torch.load`, and no `open`/`write` in the fetched Python.

**Licence CC BY-NC 4.0. Research only. This review confers no commercial
deployment approval.** `trust_remote_code` remains **false** until the owner
says otherwise. Status of the research run: **NOT_RUN**.

Approving this request would still not approve acquiring the 1,369,721,378-byte
weight tree, choosing a device, or writing a warehouse row. Those stay outside
this request.

---

## 6. What is NOT done

- **NOT_RUN.** No BANKING77 score, no model load, no weights, no GPU, no
  warehouse row.
- Remote Python was not imported and not executed. `trust_remote_code` was not
  set. `snapshot_download` was not called.
- No environment was installed. Installed size is
  `NOT_MEASURED_NO_PYTHON_3_13`, not a 3.12 measurement wearing a 3.13 label.
- No commit, no push, no other worktree, no service, no port.

— Classification lane, for Retsu, 2026-09-30.
