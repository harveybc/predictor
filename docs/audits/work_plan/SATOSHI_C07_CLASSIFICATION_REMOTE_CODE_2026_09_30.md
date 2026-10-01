# C07 — Classification remote code reviewed, closure priced for Python 3.12, one approval request

Satoshi III, lane C07, 2026-09-30. Order: `docs/handoffs/SATOSHI_POST_CONSOLIDATION_2026_09_30.md`
§"Agent assignments" item 2, with the coordinator's two amendments of the same evening
(nothing above 1 GiB on the coordinator; installs on worker_b under 2 GiB; price the closure
for the Python 3.12 line the research run would actually use).

Branch `satoshi/c07-classification-remote-code-20260930` off master `dc72170e`.
Evidence: `docs/audits/evidence/C07_REMOTE_CODE_20260930/`.

**Status of the research run: NOT_RUN. `trust_remote_code` was set nowhere. No model weights
were downloaded. No repository Python was imported or executed. Licence CC BY-NC 4.0; this
review confers no commercial deployment approval.**

This lane builds on the prior closure review at `b5e540ea` (Retsu, Python 3.13 resolution,
installed bytes `NOT_MEASURED_NO_PYTHON_3_13`). What is new here: the full execution chain
through mteb → sentence-transformers → transformers read from the library sources at the pinned
versions, with the gate lines; six concrete risks with file:line; the closure resolved for
**CPython 3.12.13** in two index variants; **installed bytes actually measured** for the
CPU variant; and the one approval request.

---

## 1. What was fetched, and how it is known to be the pinned revision

Repository `jinaai/jina-embeddings-v5-text-small`, revision
**`46ed7da5b47e4bca710b756313fafaf4110c6bd1`** (hub `lastModified` 2026-02-17T12:27:50Z,
not gated, not private, hub tags `custom_code` and `license:cc-by-nc-4.0`). This is the revision
pinned by `NATIVE_PATH.json` (c89a8ff4) and by mteb 2.9.0's registry
(`jina_models.py:800`).

Fetched with `curl` only (no `huggingface_hub` on the coordinator; nothing imported). HTTP 200
on the tree listing, the revision endpoint and every one of the 14 fetched files. The tree
listing's sha256 is `c8be5f401537f21614d72f29ae7a9e06e3ae9b51c3200c9a5e7dd6149c7363eb`,
identical to the digest pinned on 2026-09-29. For every fetched file the git blob SHA-1 was
recomputed locally and equals the oid in the hub listing (14/14), so the bytes read are the
bytes of that revision. Per-file sha256, sizes, oids and the LFS sha256 of the un-fetched
weight files are in `MODEL_FILES.json`.

| | files | bytes |
|---|---:|---:|
| whole tree at the revision | 22 | 1,369,721,378 |
| fetched for review (3 .py, 10 .json, README, .gitattributes) | 14 | 36,030 |
| not fetched (model.safetensors, 4 adapter safetensors, tokenizer.json, vocab.json, merges.txt) | 8 | 1,369,685,348 |

The five previously pinned digests (`custom_st.py`, `modeling_jina_embeddings_v5.py`,
`config.json`, `config_sentence_transformers.json`, `adapters/classification/adapter_config.json`)
match again.

---

## 2. Inventory, file by file (full detail with line numbers in `REMOTE_CODE_INVENTORY.json`)

Three Python files exist in the tree. Nothing in any of them runs at import time except the
import statements and the class definitions.

**`configuration_jina_embeddings_v5.py`** (5 lines, sha256 `d67791f1…`): line 1
`from transformers import Qwen3Config`; line 4-5 a subclass with `model_type = "jina_embeddings_v5"`.
No network, no shell, no writes, no deserialization.

**`modeling_jina_embeddings_v5.py`** (114 lines, sha256 `389836c7…`). Imports (lines 1-12):
typing, `os`, `torch`, `torch.nn.functional`, `huggingface_hub.snapshot_download`,
`transformers.AutoTokenizer`, `transformers.modeling_utils.PreTrainedModel`,
`transformers.models.qwen3.Qwen3Model`, `peft.PeftMixedModel`, `peft.PeftConfig`, and the one
relative import `.configuration_jina_embeddings_v5` (present in the tree). Call-time, all inside
`from_pretrained` (21-61):

| lines | operation | class |
|---|---|---|
| 25-27 | `JinaEmbeddingsV5Config.from_pretrained(path)` | config read (hub if repo id) |
| 28-32 | `Qwen3Model.from_pretrained(path, config=…, dtype=kwargs.pop("dtype", torch.bfloat16))` | weight load, safetensors via transformers; **bfloat16 default on any device** |
| 34-41 | `if os.path.isdir(name_or_path): <local>/adapters` **else `snapshot_download(repo_id=…, allow_patterns=["adapters/*"])`** | **network + hub-cache write**, and **no `revision=`** |
| 47-55 | `PeftConfig.from_pretrained(adapters/retrieval)`, then `load_adapter` for all four `task_names` | adapter weight load, safetensors via peft |
| 57-60 | `AutoTokenizer.from_pretrained(path, trust_remote_code=True)` | **hard-coded grant** |
| 63-114 | `encode()`: tokenize, `set_adapter([task])`, forward under `no_grad`, last-token pooling, optional `truncate_dim`, L2-normalize | pure tensor code |

**`custom_st.py`** (106 lines, sha256 `52e0931d…`). Imports (1-6): typing, `torch`, `torch.nn`,
`torch.nn.functional`, `transformers.{AutoConfig, AutoModel, AutoTokenizer}`. No `os`, no
`huggingface_hub`. Call-time, inside `Transformer.__init__` (13-48): `AutoConfig.from_pretrained`
(32-34) and `AutoModel.from_pretrained` (41-43) **without** setting `trust_remote_code`
themselves — the flag arrives in `config_args`/`model_args` from sentence-transformers;
`AutoTokenizer.from_pretrained(..., trust_remote_code=True)` hard-coded (44-47);
`max_seq_length` defaults to `config.max_position_embeddings` = 32768 (48). `forward` (57-102)
resolves the task, `set_adapter(task)` (77), pools and normalizes. `load` (104-106) takes a single
path parameter. Line 35 dereferences `model_args` which defaults to `None` (a latent
`AttributeError` on direct construction; unreachable through sentence-transformers, which always
passes a dict).

**Not present in any of the three files:** `subprocess`, `os.system`, sockets, `urllib`,
`requests`, explicit `open()`/`write()`, `pickle`, `torch.load`, `cloudpickle`, `dill`, `eval`,
`exec`, `compile`, `__import__`, `importlib`, `ctypes`. The tree ships safetensors only (no
`.bin`, `.pt`, `.pkl`), and no `requirements.txt`.

**JSON files.** `config.json` lines 2-5 carry `auto_map` for `AutoConfig` and `AutoModel`, both
pointing at this tree's own modules (no `other_repo--Class` reference, so **no cross-repository
code and no second code revision**); `model_type` `jina_embeddings_v5` is not a transformers
library type. `modules.json` names `custom_st.Transformer` with kwargs `task`, `truncate_dim`.
`config_sentence_transformers.json` declares the export toolchain sentence-transformers 5.1.2 /
transformers 4.57.0 / torch 2.8.0 and the two prompts. `tokenizer_config.json` names the library
class `Qwen2Tokenizer`, has no `auto_map`, and carries the Qwen3 chat template (a Jinja template
rendered only by `apply_chat_template`, which nothing reviewed calls). The four
`adapter_config.json` are LoRA r=32/alpha=32, `revision: null`, `base_model_name_or_path` this
same repository; only `lora_dropout` differs (0.05 classification, 0.1 others).

### Is `trust_remote_code=True` required, and exactly what would execute

**Required: yes.** transformers 4.57.0 computes `has_remote_code = "auto_map" in config_dict and
"AutoConfig" in …` and `has_local_code = model_type in CONFIG_MAPPING`
(`configuration_auto.py:1333-1334`; the model analogue at `auto_factory.py:566-567`), then
`resolve_trust_remote_code` (`dynamic_module_utils.py:705-787`) raises `ValueError` at lines
781-785 when remote code exists, no library class exists and the flag is not true; with the flag
unset (`None`) it prompts on a TTY under a SIGALRM timeout (748-776) and raises without one
(777-779). **A local directory does not bypass this gate** — `has_local_code` is about the
registry, not the path.

The chain, read from the library sources at the pinned versions (digests in
`THIRD_PARTY_SOURCES.json`), that executes once the flag is true:

1. **mteb 2.9.0** `jina_models.py:778-781` registers the model with
   `loader_kwargs=dict(trust_remote_code=True, model_prompts={… "Classification": "classification" …})`
   and `revision="46ed7da5…"` (800). `JinaV5TextWrapper` (724-737) →
   `SentenceTransformerEncoderWrapper` (`sentence_transformer_wrapper.py:74-76`) →
   `SentenceTransformer(model, revision=revision, device=device, trust_remote_code=True, …)`.
   **So running the official evaluator with this registry entry IS setting
   `trust_remote_code=True`; it is not a separate switch we hold.**
2. **sentence-transformers 5.1.2** `_load_sbert_model` reads `modules.json` (2236-2245) and for
   `custom_st.Transformer` calls `_load_module_class_from_ref` (2251-2252). The ref is not
   `sentence_transformers.*`, so at **line 2138** — `if trust_remote_code or os.path.exists(model_name_or_path)`
   — it calls `transformers.get_class_from_dynamic_module("custom_st.Transformer", path, revision=…)`.
   That resolves `custom_st.py`, runs `check_imports` (imports torch and transformers), **copies the
   file into `HF_MODULES_CACHE/transformers_modules/<repo>/<commit>/`** (`dynamic_module_utils.py:429-463`),
   appends that cache to `sys.path` (87-96) and **imports it** (268-315). This is the first
   execution of repository code.
3. Because `custom_st.Transformer.load` takes one parameter and `__init__` has `model_args`/`config_args`,
   sentence-transformers takes its legacy branch (2257-2286): `Transformer._load_init_kwargs`
   (`models/Transformer.py:368-423`) injects `subfolder/token/revision/local_files_only/trust_remote_code`
   into `model_args`, `tokenizer_args` and `config_args` (394-412), then constructs
   `custom_st.Transformer(path, **kwargs)`.
4. `custom_st.py:32-34` → `AutoConfig.from_pretrained(..., trust_remote_code=True, revision=…)` →
   `configuration_auto.py:1345-1349` copies and **imports `configuration_jina_embeddings_v5.py`**
   (second file executed) and registers it.
5. `custom_st.py:41-43` → `AutoModel.from_pretrained(..., config=…, trust_remote_code=True)` →
   `auto_factory.py:585-596` copies and **imports `modeling_jina_embeddings_v5.py`** (third file;
   `check_imports` requires `huggingface_hub`, `transformers`, `peft`, `torch`), registers it and
   calls `JinaEmbeddingsV5Model.from_pretrained`: config, Qwen3 weights, **the adapters branch
   (local dir at 34-35, or the unpinned `snapshot_download` at 37-40)**, four `load_adapter`
   calls, and the hard-coded tokenizer grant (57-60).
6. `custom_st.py:44-47` the second hard-coded tokenizer grant (library `Qwen2Tokenizer` at this
   revision).
7. At scoring, `JinaV5TextWrapper.encode` (739-775) maps `Classification` → task
   `classification` (785) with prompt `"Document: "` (767-771), then `custom_st.Transformer.forward`
   with `set_adapter("classification")`.

With a **local snapshot directory** as the model path, sentence-transformers skips its own gate
(2138) but transformers still demands the flag at steps 4 and 5; the modeling file then takes the
local adapters branch (34-35) and `snapshot_download` is never reached — the only way the adapter
bytes stay at the reviewed revision without network.

### Concrete risks (file:line), or none

| id | where | what | handling in the request |
|---|---|---|---|
| R1 | `modeling_jina_embeddings_v5.py:37-40` | `snapshot_download` **without `revision=`**: loaded by repo id, the four adapters come from the repository's **current default branch**, not the pinned revision | load from a complete local snapshot, `HF_HUB_OFFLINE=1`, all 22 digests verified before the load |
| R2 | `modeling_jina_embeddings_v5.py:57-60`, `custom_st.py:44-47` | `trust_remote_code=True` hard-coded for the tokenizer; admits nothing extra at this revision (no tokenizer Python, library class) but is an unconditional grant | revision pin + local snapshot; a snapshot cannot change under the process |
| R3 | sentence-transformers 5.1.2 `SentenceTransformer.py:2138` | a local path bypasses sentence-transformers' own custom-module gate | recorded so a local-path load is never read as a no-remote-code load; the owner's decision is explicit either way |
| R4 | transformers `dynamic_module_utils.py:87-96, 429-463` | the loader **writes the repository's .py files under `HF_MODULES_CACHE`** (default `~/.cache/huggingface/modules`) and adds it to `sys.path`; persists after the environment is abandoned | `HF_HOME` inside the throwaway environment |
| R5 | `modeling_jina_embeddings_v5.py:31`, `config.json` `torch_dtype` | bfloat16 default regardless of device; a CPU bfloat16 run is a different numerics path from the (device-undeclared) published row | device and dtype declared beside every number; not a security item |
| R6 | `custom_st.py:35` | `model_args=None` dereference on direct construction | documented; unreachable via sentence-transformers |

No shell, `eval`/`exec`, pickle or `torch.load` was found in the reviewed files. R1 and R4 are the
two that change how the run must be set up; neither is a reason to stall.

---

## 3. Transitive code revisions

- Model code: the three files of this tree at `46ed7da5…`, nothing else. `auto_map` has no
  cross-repository (`--`) reference; no `code_revision`; the adapters' `revision` is `null`.
- Evaluator code that executes the above: mteb 2.9.0 (`jina_models.py` sha256 `dabfd27f…`,
  `sentence_transformer_wrapper.py` `46c3b18f…`), sentence-transformers 5.1.2
  (`SentenceTransformer.py` `21933a98…`, `models/Transformer.py` `03c1ebeb…`), transformers 4.57.0
  (`dynamic_module_utils.py` `4cd52ca9…`, `configuration_auto.py` `120b62cd…`,
  `auto_factory.py` `04c0b610…`). All read from the GitHub tags, HTTP 200, not run.
- Declared export toolchain in the repository: sentence-transformers 5.1.2 / transformers 4.57.0 /
  torch 2.8.0 (`config_sentence_transformers.json`), while `config.json` was written by
  transformers 4.51.0. mteb 2.9.0 `adapted_from` says `Qwen/Qwen3-0.6B`; the card says
  `Qwen/Qwen3-0.6B-Base`. Recorded, not reconciled.

---

## 4. The closure, resolved and pinned for CPython 3.12.13

Interpreter: the Python 3.12.13 line of the existing conda environments, borrowed only to
create throwaway venvs (`python -m venv`), one on the coordinator (from `trading-stack`, the
coordinator has no `tensorflow` env) and one on worker_b (from its `tensorflow` env). Nothing was
installed into any conda environment or shared interpreter; both venvs live under scratch and are
abandoned.

Direct pins, chosen to equal the repository's declared export toolchain where it declares one:
`mteb==2.9.0 sentence-transformers==5.1.2 transformers==4.57.0 torch==2.8.0 peft==0.21.1
datasets==5.0.1 scikit-learn==1.9.1`.

**Finding on the declared toolchain:** `transformers==4.57.0` is **yanked on PyPI**
("Error in the setup causing installation issues"). pip installs it when pinned exactly (with a
warning) and would never select it unpinned. The run can therefore use the exact declared
version, but the owner should know the reference toolchain's own transformers release was
withdrawn; `4.57.6` (Retsu's resolution) is the non-yanked successor on the same minor line.
Retained as a choice for the approval, not made here.

Two index variants were resolved as `pip install --dry-run --ignore-installed --report`:

| variant | index | how resolved | wheels | download bytes | largest items |
|---|---|---|---:|---:|---|
| **A — CPU** | `download.pytorch.org/whl/cpu` + PyPI | `pip install --dry-run --report` on the coordinator, CPython 3.12.13, x86_64; per-wheel size and sha256 from the indices (`CLOSURE_A_CPU_RESOLVED_PY312.json`, `PINS_A_CPU_PY312.txt`) | **68** | **414,220,011** | `torch==2.8.0+cpu` 183,914,597; `pyarrow` 50,102,437; `polars-runtime-32` 49,912,258; `scipy` 35,344,199; `numpy` 16,717,410; `transformers` 11,990,588 |
| **B — CUDA (default PyPI)** | PyPI only | **derived, not pip-resolved**: closure A with `torch==2.8.0+cpu` replaced by the PyPI `torch==2.8.0` cp312 manylinux wheel plus the exact-pinned platform dependencies in torch 2.8.0's own `Requires-Dist` (markers evaluated for linux/x86_64/cp312); every other pin identical to A; zero specifier conflicts against A's versions (`CLOSURE_B_PYPI_CUDA_DERIVED_PY312.json`, `PINS_B_PYPI_CUDA_PY312.txt`, `derive_closure_B.py`) | **83** | **4,136,690,400** | `torch==2.8.0` 887,913,624; `nvidia-cudnn-cu12==9.10.2.21` 706,758,467; `nvidia-cublas-cu12==12.8.4.1` 594,346,921; `nvidia-nccl-cu12==2.27.3` 322,364,134; `nvidia-cusparse-cu12` 288,216,466; `nvidia-cusparselt-cu12` 287,193,691; `nvidia-cusolver-cu12` 267,506,905; `nvidia-cufft-cu12` 193,118,695; `triton==3.4.0` 155,560,068 |

The two closures share 67 pins; B adds 14 `nvidia-*-cu12` wheels and `triton==3.4.0` (CUDA 12.8
line, as torch 2.8.0 pins it) in place of the `+cpu` torch. Retsu's 3.13 resolution carried
`torch==2.14.0` with CUDA 13 libraries (3,230,034,941 bytes); neither closure here is that one,
because torch is now pinned to the repository's declared 2.8.0.

Resolution A ran on the coordinator under `crispdm-run -m 1G` (exit 0). Resolution B was
attempted twice on the coordinator under the same cap and was **stopped both times** by the
admission monitor's `SUSTAINED_ABOVE_RESPOND` rule (`PRESSURE_STOP_SUSTAINED_ABOVE_RESPOND`,
records `c07-closure-1790811971-679626-b8cbc7` and `c07-closure-1790812283-686547-696988`,
observed tree peak 1.04-1.05 GB against the 1 GiB cap — pip stages the large CUDA wheels in
`/tmp`, which is tmpfs and charged to the job). No other process was signalled. B was then
resolved on worker_b. A pip dry-run of B was then started on worker_b under `crispdm-run -m 2G`; it was fetching the
CUDA wheels into `/tmp` (tmpfs, charged to the job) at roughly 1.4 MB/s and could not finish
inside its 20-minute limit or its cap, so this lane stopped its own job, deleted its temp files,
and derived B from index metadata instead (method in the table). B is therefore **a derived
closure with exact sizes and digests from PyPI, not a pip-verified resolution**; its 67 shared
pins are pip-verified through A, and its 16 torch-side pins are exact `==` pins from torch's
metadata. A pip resolution of B on a host with disk scratch and a larger admission would confirm
it; that was not done.

**Installed bytes, measured (closure A, CPU).** A real `pip install` of the 68 pins into the
throwaway venv on worker_b, under `crispdm-run -m 2G -t 25m`, exit 0, pip's temp directory on
the same disk: **venv 1,797,137,014 bytes** (site-packages 1,797,103,606; sum of every
installed file listed in the 69 RECORDs incl. pip 1,797,122,188). `pip freeze` equals the
dry-run pins exactly (68/68, zero differences), so the pip-verified pin set is the installed one.
Largest: `torch==2.8.0+cpu` 719,396,119; `polars-runtime-32` 180,197,199; `pyarrow` 161,152,777;
`scipy` 139,052,536; `transformers` 107,520,636; `mteb` 60,510,388; `scikit-learn` 46,469,490.
Per-distribution figures in `INSTALLED_A_CPU_PY312.json`.

**Where it was measured matters, and was corrected mid-lane.** The first install went into
`/tmp` on worker_b, which is tmpfs: the venv's 1,743,423,052 bytes were charged to the job as
unreclaimable shmem, the admission monitor stopped the job
(`c07-closure-install-1790813089-1242292-634e7b`), and the coordinator reported that this charge
was blocking other admissions on that host. On the coordinator's correction the lane removed
`/tmp/c07-closure` and its own pip temp dirs (**1,746,978,969 bytes of tmpfs freed**, plus
416,515,107 bytes of disk temp), recreated the venv **on disk** under its own
`~/.local/state/scratch/c07-closure/`, and measured there. Rule recorded for every host:
scratch goes to disk; `/tmp` only for files under 50 MB. A second leftover was found afterwards by the coordinator and confirmed mine: `/tmp/tmp3spjz0ia`
on worker_b, **827,588,608 bytes**, pip's download temp file of the CUDA cp312 torch wheel left
when this lane killed its own dry-run B (mtime 19:04:25 host-local, the kill; owner this lane's
user; zip first entry `functorch/`; no open handles). Deleted; **total tmpfs freed by this lane on
worker_b: 2,574,567,577 bytes**. After measurement the on-disk venv
(1,798,337,765 bytes) and the coordinator's 11,166,986-byte venv were removed; the user's shared
pip cache on worker_b now additionally holds the 68 downloaded wheels (a cache, left in place).

**Installed bytes for closure B (CUDA): `NOT_MEASURED`.** Not attempted: its 4.1 GB download
plus an install of the order of 6-7 GB does not fit a 2 GiB admission on either host, and B is
derived rather than pip-resolved (above). A bound: B's installed size is A's minus the 719 MB
`+cpu` torch plus the unpacked `torch==2.8.0` CUDA wheel and 14 `nvidia-*-cu12` + `triton`
wheels, whose compressed download alone is 3,906,384,986 bytes (B's 16 torch-side wheels; the 67 shared wheels are 230,305,414).

Model weights, separately: **1,369,721,378 bytes** for the whole tree (22 files, all four
adapters needed because of R1), of which 1,248,458,137 is the classification route alone; neither
figure is in the closure, and none of it was downloaded.

---

## 5. Licence

`README.md` front matter line 2: `license: cc-by-nc-4.0`; §License (line 311):
"jina-embeddings-v5-text-small is licensed under CC BY-NC 4.0. For commercial use, please
contact us." mteb 2.9.0 records the same (`jina_models.py:807`). The hub tags the revision
`license:cc-by-nc-4.0`.

**This is a research comparison against a published BANKING77 row. CC BY-NC 4.0 permits that
and forbids commercial use. Nothing in this lane, and nothing in an approval of the request below,
confers commercial deployment approval, trading use, or any entitlement beyond the research
run; the governed resources already refuse date ranges and commercial use by construction.**

---

## 6. Approval request (one decision; nothing here is granted by this document)

Approve or refuse, as one decision, a **research-only** comparison run of BANKING77 with:

**(a) Code revision.** `jinaai/jina-embeddings-v5-text-small` at
`46ed7da5b47e4bca710b756313fafaf4110c6bd1`, exactly the three Python files reviewed here
(sha256 `d67791f1…`, `389836c7…`, `52e0931d…`, full digests in `MODEL_FILES.json`), executed
through mteb 2.9.0 / sentence-transformers 5.1.2 / transformers 4.57.0 along the seven-step chain
of §2. **This requires `trust_remote_code=True`**, which mteb's registry sets by itself the
moment this model name is run; approving the run IS approving that flag for this revision and
no other.

**(b) Weights.** A one-time acquisition of the full tree, **1,369,721,378 bytes, 22 files**,
into a local snapshot directory, every file verified against the LFS sha256 / blob oid in
`MODEL_FILES.json` before any load; all four adapters are needed because of R1. Loaded as a
**local directory with `HF_HUB_OFFLINE=1`**, so `snapshot_download` (R1) is never reached and
no network is open during load or inference; a missing file must refuse, never download.

**(c) Environment.** A **fresh, third venv on disk** on worker_b, interpreter borrowed from its
existing CPython 3.12.13 (`python -m venv`; nothing installed into any conda environment),
`HF_HOME` inside the venv (R4), with **closure A (CPU)**: 68 pins in `PINS_A_CPU_PY312.txt`,
**download 414,220,011 bytes, installed 1,797,137,014 bytes** (measured). If the owner wants the
idle GPU on worker_b instead, **closure B (CUDA)**: 83 pins in `PINS_B_PYPI_CUDA_PY312.txt`,
download **4,136,690,400 bytes**, installed `NOT_MEASURED` (bounded in §4), and B must then be
pip-resolved once before use. The transformers pin is the repository's declared `4.57.0`, which
is **yanked on PyPI**; the owner may substitute the non-yanked `4.57.6` — a declared deviation
from the repository's export toolchain, to be printed beside every number.

**(d) The run.** `mteb run -m jinaai/jina-embeddings-v5-text-small --model-revision 46ed7da5…
-t Banking77Classification` (mteb 2.9.0 declared, per the option-b decision in NATIVE_PATH.json),
on the governed 3,080-row test population and the ten pinned 616-row training draws, device and
dtype **declared** (R5), under `crispdm-run` with `CUDA_VISIBLE_DEVICES` empty for CPU or the
single admissible GPU for CUDA. A **cost pilot on one training draw first**, its accuracy never
carried into a receipt, then the full ten-experiment protocol; a timeout mid-corpus is not a
score. Row identities re-hashed inside the child; the published target recomputed from its own
cells (0.914578, naive 0.012987) beside the result, with `in_training`/`held_out` tags per the
contract. Output: one classification receipt into the existing warehouse route, marked
research-only, `execution_authority: NONE`, `authorises_broker_deployment: false`.

**(e) What it costs.** Network: 1,369,721,378 (weights) + 414,220,011 (A) = **1,783,941,389
bytes**, or + 4,136,690,400 (B) = **5,506,411,778 bytes**. Disk: about 3.2 GB for A (weights +
venv + HF cache), 7-9 GB for B. Memory: mteb's own figure for this model is 1,137 MB resident
plus the Python stack; request **8 GiB** host admission for the CPU run (the prior lane's
`-m 8G -t 60m` figure, unmeasured here), or 8 GiB host + the model on the GPU for B. Time:
unmeasured; a CPU bfloat16 pass over 3,080 + 6,160 texts at a 32,768-token window set to the
real maximum sentence length is the pilot's job to measure. No GPU of the preferred external
host (quarantined) is requested.

**(f) What approval does NOT grant.** Commercial or trading use of the model or its outputs
(CC BY-NC 4.0); any other revision; any other `trust_remote_code` load; the full 7,600-row AG
News test; FOMC, MASSIVE or Financial PhraseBank; a warehouse row before the run has executed
under this exact configuration.

Refusing leaves the native path where c89a8ff4 left it:
`BANKING77_NATIVE_PATH_BLOCKED_ON_MODEL_AND_ENVIRONMENT_ACQUISITION`, with this review attached.

---

## 7. What was verified, what was not, and what is NOT done

Verified by this lane: the 14 fetched files are the pinned revision's bytes (blob SHA-1 = hub
oid, 14/14; tree digest = prior pin); the three Python files contain no shell, eval/exec, pickle,
torch.load or explicit writes (read, line by line); the exact gate and execution chain in mteb
2.9.0 / sentence-transformers 5.1.2 / transformers 4.57.0 (read from the tagged sources, with
digests); `transformers==4.57.0` yanked (pip's own warning, twice, on two hosts); closure A
resolved for 3.12.13 with per-wheel sizes and sha256 from the indices; closure A installed on worker_b with `pip freeze` equal to the pins (68/68) and its bytes measured on disk; closure B's torch-side pins read from torch 2.8.0's own index metadata with zero specifier conflicts against A.

Not verified: that the closure *runs* the remote code (no import, no load — by order); the
installed bytes of the CUDA variant (see §4); any accuracy; the published row's device.

NOT done: no `trust_remote_code=True` anywhere; no weights; no model load; no GPU; no BANKING77
score; no warehouse row; no change to any conda environment, service, or running job. Both
throwaway venvs and the pip temp directories created by this lane were removed; nothing that
was not this lane's was deleted.
