# RB02 return — matched Weather reference: the recipe pinned, the producer path proved by refusal, the cost measured

Satoshi III (Mujuro Utsutsu), successor technical lead. 2026-09-28 (America/Bogota).
Branch `satoshi/rb02-weather-matched-20260928`, based on
`musashi/reconcile-dispatch-20260928` at `45789070`.

**What this delivery is.** The Weather recipe of the already-selected reference
is pinned from the author's own files and the published tables, five
paper-versus-code disagreements are resolved *before* any score can exist, the
governed producer path is proved by refusal, and a TRAIN-only cost pilot was
executed on the admitted worker so that the frozen full reproduction carries a
measured price.

**What this delivery is not.** No Weather or Traffic model score exists. No
comparison against a published number was made, because no cell has run. The
pilot measures seconds and bytes, not accuracy: `NO_NEW_MEASUREMENT` of model
error on any dataset. A published-recipe reference reproduces one named paper's
own reported configuration; it is not a claim that this recipe is the current
best result on this benchmark.

---

## 1. The pinned Weather recipe

Sealed **before** a byte of the benchmark was read, from the author's own
`run.py` parser (read by AST, executed on a fresh parser so the effective
defaults are the author's file and not a transcription) and the author's own
`scripts/Weather.sh` (loops expanded, variables substituted).

| Field | Pinned value | Where it comes from |
|---|---|---|
| Reference | TimeFilter (Hu et al., ICML 2025), arXiv:2501.13041 | paper |
| Pinned source | `github.com/TROUBADOUR000/TimeFilter` at `dffde87e4fff0fdeeebbacde03dc1e432e15b3a1`, clone clean | author clone, verified at seal |
| Driving script | `scripts/Weather.sh`, first block (`seq_len=96`) | author repo |
| Official input | lake `sota_benchmarks`, resource `thuml_tsl_weather/weather.csv`, sha256 `34ee981d…2ba63`, 7 235 425 bytes, 52 696 rows × 21 variables + `date` | already-registered resource; **not** re-downloaded, re-registered or re-adopted |
| Sample interval | **600 s** (10 minutes); first label `2020-01-01 00:10:00`, last `2021-01-01 00:00:00` | registered contract, confirmed on the delivered bytes |
| Target columns | all 21 channels (`--features M`: every channel is input and target). `--target` is run.py's default `OT`; `OT` is present and already the last column, so the loader's remove-and-re-append is a **no-op** for Weather | characterization of the delivered bytes |
| Split boundaries | `Dataset_Custom`: train rows `[0, 36887)`, vali `[36791, 42157)`, test `[42061, 52696)` — i.e. `int(0.7 n) = 36887` / `n − train − test = 5270` / `int(0.2 n) = 10539` rows | author loader arithmetic, reproduced from the registered row count and then confirmed window-for-window by the loader itself |
| Lookback | 96 steps = **16 h** | script |
| Forecast vector | 96 / 192 / 336 / 720 steps = **16 h / 32 h / 56 h / 120 h** | script; elapsed computed only from the registered 600 s interval |
| Scored test windows | 10 444 / 10 348 / 10 204 / 9 820 | loader; equals the sealed arithmetic exactly |
| Scored elements | windows × steps × 21 = 21 055 104 / 41 723 136 / 71 999 424 / 148 478 400 | derived, asserted in the receipt |
| Training scaler | one `sklearn.StandardScaler` **fit on the train rows `[0, 36887)` only**, applied to every row; identity `9c315d4c…f740`, fit-population identity `5470e9d9…723`; identical across the four horizons because the train border does not depend on the horizon | characterization; archived as `SCALERS.weather.npz`, sha256 `54b3b936…d87a` |
| Missing-value handling | **none needed and none permitted**: the delivered resource has 0 missing values, and the author path contains no imputation, so a missing value is a refusal rather than a policy | characterization refuses on any non-zero count |
| Optimizer | `torch.optim.Adam(lr = 5e-4)`, torch default betas/eps/weight_decay | author `_select_optimizer` |
| Loss | `nn.MSELoss()` on the normalized targets **+ 0.05 × the MoE routing loss** (coefficient hard-coded in the author's `train()`) | author `train()` |
| LR schedule | `lradj = cosine`: `lr_e = lr/2 (1 + cos(e / train_epochs · π))` | run.py default + author `adjust_learning_rate` |
| Batch / epochs | 32 / 10 | script; `train_epochs` is run.py's default here and equals the paper's Table 6 value |
| Checkpoint / early stop | `EarlyStopping(patience = 3, delta = 0)` on the **validation MSE** (which excludes the MoE term); the lowest-validation-loss epoch's `state_dict` is reloaded before `test()`. The test loss the author prints each epoch is logging only | author `train()` + run.py defaults |
| Seeds | run.py hard-codes `fix_seed = 2021` and offers no CLI path to vary it; the paper reports a dispersion over runs, so our three seeds are 2021 / 2022 / 2023 set through **the same three calls** run.py makes | declared operational patch |
| Model geometry | `patch_len 48`, `d_model 128`, `d_ff 256`, `e_layers 2`, `d_layers 1`, `n_heads 4`, `factor 3`, `dropout 0.3`, `alpha 0.1`, `top_p 0.5`, `pos 1`, `use_norm 1`, `label_len 48`, `embed timeF`, `freq h`, `inverse False`, `use_amp False` | script + run.py defaults |

Identities of this delivery:

```text
design_sha256    dbb3e87f689a91e366207ebe2bd9bfb8bd81e165ab500d3b3eaa1fd7061d8c30
protocol_sha256  c9672ba4ea6399e1f667cbb1db11253b6c28096ec9efde60b19d1514a14a3df8
receipt contract tsl_literature_metrics.v1, sha256 d0b578a8ea08e7dc7f72f59bf8bdb0b2ab569173fe7c0cb9c608d3ed66ba985a
Traffic plan     design 6cba7e20aa359986482c0dbae7d7bc095b1951df9fdc4db10af5837815285575 / protocol 322262e8d78abd1fe0648d8e6f66c31c129d852a7cab97aad47c366311ba5a20
```

### The published row this recipe is measured against

TimeFilter, **Table 8** (full results at fixed L = 96), Weather row, and
**Table 7**, which gives the standard deviation of the four-horizon average
over the paper's runs:

| horizon (steps) | elapsed | published MSE | published MAE |
|---:|---:|---:|---:|
| 96 | 16 h | 0.153 | 0.199 |
| 192 | 32 h | 0.202 | 0.246 |
| 336 | 56 h | 0.260 | 0.289 |
| 720 | 120 h | 0.342 | 0.341 |
| **avg** | | **0.239 ± 0.006** | **0.269 ± 0.004** |

The margin is **Weather's own** Table 7 dispersion (0.006 / 0.004). Electricity's
is 0.005 / 0.006 and is not reused here; a test asserts the two differ.

The same arXiv reading was cross-checked against the Electricity row that
`tools/df_sota_repro.py` has carried since RP92 — 96 `0.133/0.230`, 192
`0.154/0.248`, 336 `0.162/0.261`, 720 `0.184/0.284` — and it agrees exactly.
Two independent readings of the same table, months apart, now agree, and a test
fails if they ever diverge.

### 2. The five paper-versus-code disagreements, resolved before any score

| id | disagreement | resolution |
|---|---|---|
| `WEATHER-IS-TRAINING-2` | `scripts/Weather.sh` drives its L = 96 block with `--is_training 2`, while `ECL.sh` and `Traffic.sh` use `1` | **NO EFFECT.** `args.is_training` is read in exactly two places in the whole pinned clone: `run.py:153`, as the truth value of `if args.is_training:`, and `utils/print_args.py`, which only formats it into the banner. Neither compares it to a number, so `2` and `1` select the identical train-then-test branch. The `is_training=` keyword inside the model/exp/layers is a separate local boolean for train-vs-eval mode and never receives this argument. The argv value is carried into every cell **verbatim** rather than normalised, so the lock reproduces the author's own command |
| `WEATHER-TABLE6-SILENT-DEFAULTS` | Table 6 states patch 48, e_layers 2, lr 5e-4, d_model 128, d_ff 256, epochs 10 and the script agrees on every one of them; the script additionally sets `dropout 0.3`, and patience, lradj, n_heads, alpha, top_p, pos, use_norm, label_len, factor, d_layers stay at run.py defaults. None of those is stated in the paper | the **executable** configuration governs and is pinned verbatim; the paper is recorded as the published reference, not as the configuration. Every cell's `effective_args` is the author's parser applied to the author's argv, and a test asserts each Table 6 field matches |
| `WEATHER-FREQ-H-ON-A-TEN-MINUTE-SERIES` | Weather is sampled every 600 s, yet the script passes no `--freq`, so the loader builds calendar marks at run.py's default `h` | **NO EFFECT on the model.** `Exp_Long_Term_Forecast` calls `self.model(batch_x, self.masks, is_training=…)` at lines 80, 128 and 189 and never passes `batch_x_mark`/`batch_y_mark` to TimeFilter, so the marks are built by the loader and discarded. Kept at the author's value rather than "corrected": changing it would deviate from the pinned recipe for no numerical gain |
| `WEATHER-TABLE9-SEARCH-SPACE` | Table 9 says the input length is searched in {192, 336, 512, 720}; `scripts/Weather.sh` contains exactly one long-horizon block, L = 720 | only **L = 96 (Table 8)** is sealed. The L-searched protocol is recorded as available at L = 720 alone, and no search the released code does not contain is claimed. `seal` refuses any `seq_len` the script does not offer |
| `WEATHER-SEED` | the paper reports a dispersion over runs; run.py hard-codes one seed | the same three calls with each of our three seeds, declared as an operational patch with no other effect |

Four declared operational patches carry over unchanged from the Electricity
runner and have no mathematical effect: the refusing `sktime`/`patoolib` import
shims, the replication of `run.py`'s `__main__` so the seed can vary,
`np.Inf` restored as an alias of `np.inf` for the author's `EarlyStopping`, and
the wrapper that captures the arrays the author's own `metric` scores.

### 3. Metric space, reduction and the two clocks

Bound per dataset in `docs/contracts/tsl_literature_metrics.v1.json` and
enforced by the producer:

- **Space.** `--inverse` is False, so the primary metrics live in the
  **training-scaler normalized target space**: `sota.test.mse_normalized`
  (`z^2`) and `sota.test.mae_normalized` (`z`). No inverse transform.
- **Reduction.** mean over **every** test window × forecast step × target
  channel element — not an unweighted average of batch means.
- **Paired naive.** persistence of each window's last observed value, repeated
  over every step and channel, on **exactly** the same windows, in the same
  receipt. A receipt without it is refused.
- **Scaler.** fit on the train rows only, per dataset. No cross-dataset reuse
  of a fitted scaler; the fit population carries its own digest.
- **Clocks.** Weather steps are 600 s, Electricity and Traffic steps are
  3600 s. Both the step horizon and the elapsed horizon are stored, and the
  producer refuses arithmetically — not by convention — any receipt whose
  `horizon_seconds` contradicts `horizon_steps × step_seconds` for its own
  dataset. Weather's 96 steps are 16 h; Traffic's 96 steps are 96 h. The same
  step count is **never** the same elapsed horizon.

No test split was read, no tuning of any kind was performed, and no
architecture, window or batch was changed to fit the machine.

### 4. The producer path, proved by refusal

`tools/test_tsl_warehouse_receipt.py` (pre-existing, re-run here: 2 passed)
proved the deployed DuckDB provider transports the governed metric schema and
rejects a nonfinite value. It also demonstrated — by filling every identity tag
with the literal string `fixture` and being accepted — that the general-purpose
warehouse does **not** check whether a run's protocol, scaler and evaluation
population are identified. That is correct for a generic warehouse and
unacceptable for a literature comparison, so the check now lives in the
producer: `df_tsl_repro.validate_receipt`.

`tools/test_tsl_producer_contract.py`: **22 tests, 22 passed, 0.70 s**, run with
the production warehouse's own installed provider against a **temporary**
DuckDB file only. The live cube was never opened.

| required by the order | proved by |
|---|---|
| a missing **protocol identity** is rejected | `test_missing_protocol_identity_is_refused` — absent, empty, `fixture`, and non-hex `protocol_sha256` all refused, each naming the field |
| a missing **scaler identity** is rejected | `test_missing_scaler_identity_is_refused` — both `scaler_sha256` and `scaler_fit_population_sha256`, absent and placeholder |
| a missing **population identity** is rejected | `test_missing_population_identity_is_refused` — absent `evaluation_population_sha256`, and an element count that does not reconcile with windows × steps × channels |
| a **nonfinite metric** is rejected | `test_nonfinite_metric_is_refused` — NaN, +inf and −inf in each of the four metric rows (12 cases) |
| the **typed receipt round-trips through a temporary store** | `test_accepted_receipts_round_trip_and_refused_ones_never_reach_the_store` — 4 horizons × 3 seeds = 12 terminals, 48 metric rows, tags/values/units/splits/horizons/resource compared field by field, idempotent on resubmission, and the per-dataset clock queryable back out of the stored tags |

It also refuses, in the same gate: a missing paired naive, the wrong clock, a
foreign resource or foreign dataset digest, a wrong unit, a wrong split, a
horizon that disagrees with its own tags, a tampered terminal digest, and —
explicitly — the whole `fixture` shape the warehouse alone accepts.

The design digest, the protocol digest, the configuration digest per horizon,
the scaler and scaler-fit-population digests and the evaluation-population
digest all come from artifacts, never from a literal.

### 5. The TRAIN-only cost pilot — measured, on the admitted worker

Placement followed the binding reading: the coordinator took only the seal, the
characterization and the tests; the **primary accelerator host was not used at
all** (about 4.5 GiB MemAvailable against a 4.4 GiB unreclaimable slab, plus
NVIDIA allocation errors earlier the same day — temperature and free VRAM do
not admit a run there); every GPU child went to the **secondary worker's cold
idle RTX 4090**.

Before any GPU work, inside the reservation: the device was admitted by
**physical UUID**, and a bounded allocate → 4096² matmul → free smoke ran and
returned a finite result. Retained as `GPU_ADMISSION_SMOKE.worker.json` on both
sides; the second, durably recorded run reads: admission `pass`, refusals `[]`,
`GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9` (RTX 4090 Laptop, 16 376 MiB, 14 MiB
used, 35 °C), **no competing compute application**, 16 362 MiB free VRAM,
18.34 GiB host MemAvailable, 662.5 GiB free disk; matmul 0.173 s, finite, peak
GPU allocation 209.8 MB falling back to 8.5 MB after `empty_cache`, whole-cgroup
peak 513.9 MB. The first, identical smoke before the pilots read 41 °C and
326 MiB used, matmul 0.202 s, whole-cgroup peak 530.8 MB.

Two bounded children, 30 optimizer steps of **the author's own training loop**
each, both refusing to start unless the CUDA device held *inside* the child is
the admitted one. No validation batch was read, no test window was evaluated,
no metric of any kind was produced.

| | h96 s2021 | h720 s2021 |
|---|---:|---:|
| parameters | 201 560 | 361 928 |
| train windows / batches per epoch (batch 32) | 36 696 / 1 147 | 36 072 / 1 128 |
| s / train step, median (steady state) | **0.010006** | **0.010663** |
| s / train step, p90 | 0.010193 | 0.010977 |
| s / train step, mean (includes the first, warm-up step) | 0.028350 | 0.027047 |
| projected train-pass seconds per epoch | 11.48 | 12.03 |
| CPU s in the timed region (user / system) | 0.830 (0.570 / 0.260) | 0.845 (0.573 / 0.272) |
| peak GPU allocated / reserved | 62.5 MB / 81.8 MB | 69.8 MB / 83.9 MB |
| **whole-cgroup peak, read in-child from `memory.peak`** | **1 465 098 240 B (1.364 GiB)** | **1 374 965 760 B (1.280 GiB)** |
| whole-cgroup peak, launcher sampler | 1 393 577 984 B | 1 374 965 760 B |
| declared cap (never shrunk) | 8 GiB (`MemoryMax`, `MemoryHigh` 7.2 GiB, swap 0) | 8 GiB |
| device measured **inside** the child | `GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9` | same |
| boot id of the executing host | `e7cb8a4f-2aab-4391-9b4b-7fc83f8906d0` | same |
| pid / scope | 190 992 / `crispdm-rb02-weather-train-pilot-h96-…scope` | 191 539 / `…h720-…scope` |
| record digest | `0dde8331…52dff` | `3cf90f8a…02fef` |

The in-child `memory.peak` is the kernel's high-water mark for the job's own
scope cgroup and is authoritative; the launcher's periodic sampler undershoots
on sub-second loads (it recorded 39 MB for the smoke whose in-child peak was
531 MB). Both are reported rather than the more convenient one.

**The first request was REFUSED and I waited rather than shrink it.** The
reservation system answered `SLICE_AGGREGATE_BUDGET — the observed aggregate
budget would be 14.07G against the crispdm-batch.slice ceiling 14.00G (in use
2.19G, unrealised reservations 3.89G)`, because another lane's admitted job
(`rb01-shard1-0ea5bff4`, 6 GiB cap, armed) was live on the same worker. The
8 GiB cap was **not** reduced; the job was queued under `-q` and started when
that lease released. Nothing was killed, no cap was raised, no cache or swap
was touched, no service was restarted.

### 6. The frozen full reproduction and its measured cost

`FULL_COST.weather.L96.json`, digest
`23e6418486bb570a8fe2e7c854e9b85adf99704b93d63194744a6a50de3b6b31`, priced from
the h96 pilot's measured median step on the admitted device and the sealed
batch counts:

| | value |
|---|---|
| cells | 12 (4 horizons × 3 seeds), `weather_L96_h{96,192,336,720}_s{2021,2022,2023}` |
| epochs per cell | 10 (sealed), `EarlyStopping` patience 3 — the number below is the **upper bound** |
| per-cell seconds (max) | 131.1 / 130.6 / 129.9 / 127.9 for h96 / h192 / h336 / h720 |
| **total, all 12 cells** | **1 558.3 s = 0.433 GPU-hours** |
| GPU memory | ≤ 70 MB allocated; the 4090's 16 GiB is not a constraint |
| host memory, training | ≤ 1.37 GiB whole-cgroup measured |

Two honesties about that number:

1. The **training** part is measured. The **evaluation** part is DERIVED: the
   pilot is TRAIN-only by order, so the validation and logging-test forward
   passes are priced at one third of a measured train step per batch. That
   share is named as derived inside the artifact.
2. The binding constraint on the Electricity campaign was never GPU time but
   **host memory of the author's unchunked evaluation path**, and that is not
   measured here. `Exp.test()` accumulates `preds`, `trues` and `inputs` as
   float32 lists, concatenates them, then `utils.metrics.metric` builds
   full-size temporaries for MAE, MSE, RMSE, MAPE and MSPE. For Weather this is
   small — h720 holds 148 478 400 prediction elements = 593.9 MB, and the
   derived peak is ≈ 4.4 GiB including the 1.3 GiB baseline the pilot measured
   — comfortably inside an 8 GiB cap and inside the worker's 19 GiB. It is
   still a **derivation**, so the frozen design's first gate is a bounded
   evaluation-path memory probe with an untrained model at h720 before any
   scored cell runs. That probe is not a TRAIN-only pilot and was therefore not
   executed under this order.

Contrast with Electricity, which is why Weather was the right next dataset:
RP94 measured ≥ 11.2 GiB at h336 and ≈ 20 GiB at h720 against a 14 GiB
batch-slice ceiling, and six of twelve ECL cells were named as a capacity
deficit. Weather's 21 channels remove that deficit entirely; all four horizons
are expected admissible on the secondary worker.

A side finding of the same code read: `utils.metrics.metric` computes MAPE and
MSPE by dividing by `true`, which in the z-normalized target space divides by
values arbitrarily close to zero and can return non-finite numbers. The author
prints only MSE and MAE, and this producer contract stores only MSE/MAE and
their paired naive, so nothing non-finite can reach a receipt — and if it were
ever added, `validate_receipt` refuses it.

### 7. Traffic, planned while Weather waits

Sealed as a **planning** artifact only (design
`6cba7e20…85575`, protocol `322262e8…ba520`): 12 cells, `scripts/Traffic.sh`
first block, L = 96, patch 96, `e_layers 3`, `d_model 512`, `d_ff 2048`,
`dropout 0.3`, `top_p 0.0`, `pos 0`, lr 1e-3, batch 16, **30 epochs**, 862
channels. Table 6 agrees with the script on every field it states; `dropout`,
`top_p` and `pos` are script-only, and four disagreements are recorded the same
way Weather's were. Published Table 8 row: 96 `0.375/0.251`, 192 `0.395/0.262`,
336 `0.414/0.271`, 720 `0.445/0.289`, average `0.407 ± 0.008` / `0.268 ± 0.004`
(Table 7) — Traffic's own margin, not Weather's and not Electricity's.

Clock: Traffic steps are **3600 s**, so its 96-step horizon is **96 h** where
Weather's is **16 h**. Split rows 12 280 / 1 756 / 3 508; test windows 3 413 /
3 317 / 3 173 / 2 789.

Two things must be measured before any Traffic cell is proposed, and neither is
inferable from the Weather pilot:

- **Per-step cost.** Traffic's patch geometry gives 862 graph tokens against
  Weather's 42, at `d_model 512` against 128, for 30 epochs against 10. The
  Weather median of 10 ms says nothing about it. Traffic needs its own
  TRAIN-only pilot on the same admitted device.
- **Evaluation-path memory.** Derived from the sealed populations: ≈ 8.8 GiB at
  h96, ≈ 14 GiB at h192, ≈ 22 GiB at h336 and ≈ 37 GiB at h720 against the
  14 GiB `crispdm-batch.slice` ceiling. On that derivation only **h96** is
  comfortably admissible, h192 is marginal, and **h336 and h720 are not
  admissible on any host of this fleet** under the author's unchunked scorer.
  That is a named capacity deficit, not a licence to chunk, downcast or
  sub-sample the author's metric population.

### 8. What is VERIFIED, what is measured-but-unverified, and what is not measured

| claim | class |
|---|---|
| The Weather recipe pinned in §1 is the author's own, from the pinned clean clone at `dffde87e…` and the author's own parser | **VERIFIED** (artifacts: design digest `dbb3e87f…`, `files_sha256` of 14 author files, clone clean at seal) |
| `--is_training 2` is numerically identical to `1` | **VERIFIED** (the only two reads of `args.is_training` in the whole clone are a truth test and a banner format; asserted by test) |
| The delivered bytes are the registered resource, 52 696 rows, 21 channels, `OT` present and last, 0 missing values, and the loader's window counts equal the sealed arithmetic for all four horizons | **VERIFIED** |
| The producer path refuses a missing protocol / scaler / population identity and a nonfinite metric, and an accepted typed receipt round-trips through a temporary store | **VERIFIED** (22/22 tests, production provider, disposable database) |
| Weather's published row and margin, and the agreement of this arXiv reading with the Electricity row pinned since RP92 | **VERIFIED as a reading of the paper**; asserted by test. It is not a verification of the paper's own numbers |
| Per-step training cost, GPU footprint and whole-cgroup peak on the admitted 4090 | **measured**, receipts retained, `NOT_INDEPENDENTLY_VERIFIED` (single host, single seed, 30 steps, not replayed) |
| The 0.433 GPU-hour total for the 12 Weather cells | **measured for the training pass, DERIVED for the evaluation passes** |
| Weather's and Traffic's evaluation-path host-memory peaks | **DERIVED from element counts, NOT measured** |
| Weather model error, MSE/MAE, agreement class against the published table | **`NO_NEW_MEASUREMENT`** — no cell has run. No comparison exists and none is implied |
| Traffic anything numerical | **`NO_NEW_MEASUREMENT`** |
| Traffic per-step cost | **UNMEASURED**; the Weather pilot does not bound it |

No paired naive was computed, because there is nothing to pair it with yet. The
receipt shape that will carry it is proved and refused-when-absent.

### 9. Three things the owner or auditor must decide, and two blockers I did not touch

**ONE costed allocation request.** The index carries no numeric allocation for
this lane (`remaining_cost: UNKNOWN new training cost`), and the only
remainder anywhere — 7 833 of A/B's 24 000 CPU s — belongs to another lane and
is explicitly "NOT a licence to spend". I therefore read this order's own
sentence "execute the smallest useful TRAIN-only cost pilot" as the pilot's
authorization, bounded it to two 30-step children on the admitted worker
plus two sub-second GPU smokes, and spent **1.675 CPU s inside the timed
regions**, whose wall was 0.875 s and 0.848 s, whole-cgroup peak 1.364 GiB. The children's full process wall, including interpreter and CUDA
initialisation, was not separately instrumented and is not claimed. That
interpretation is stated here rather than buried, so it can be corrected.

The request, exact and single:

> **Weather L = 96, TimeFilter, 12 cells (4 horizons × 3 seeds), on the
> secondary worker's RTX 4090 `GPU-a8bd1b2c-…6780f9`, under design
> `dbb3e87f…d8c30` / protocol `c9672ba4…4a3df8`. Measured upper bound 1 558 s
> = 0.433 GPU-hours plus a ≤ 5 min bounded evaluation-path memory probe;
> declared cap 8 GiB per cell, sequential, one cell at a time, no duplicate GPU
> executor. Request: 2 700 GPU-seconds (0.75 h) to cover the probe, the 12
> cells at their upper bound and one retry.**

Full Weather training is **not** authorized by registration, by this pilot, or
by this document. Traffic is not requested at all until its own TRAIN-only
pilot prices it.

**Second decision.** A new governed unit for a scored cell needs the data-gov
service key, which I do not hold and did not go looking for. The pilot read the
**already governed-delivered** Weather bytes from the adoption campaign's
verified cache (campaign `4ec0dd8e…a9b41`, delivery `aead9358ae4e492e909d3e4ee9446775`,
`VERIFIED_TRANSFER`, availability contract `975df3b6…44cdc`), transported to
the worker and **re-verified by sha256 inside the child**. That is a declared
transport of governed bytes, not a new governed unit, and it is labelled as
such. Scored cells must open their own campaign per unit, which needs the key
path from the operator.

**Third decision.** the templated governance-tunnel unit for the secondary worker (`crispdm-governance-tunnel@<worker>.service`) is in
`activating (auto-restart)`, `ExitOnForwardFailure` failing with status 255,
because an orphaned session from earlier today still holds the worker's
loopback forward ports while answering nothing. Clearing it means signalling
someone else's session, so **I did not touch it** — it belongs to RB01. For my
own work I opened an ad-hoc reverse forward on two unused loopback ports and
confirmed governance and the warehouse answer `200` from the worker, changing
no unit and leaving nothing behind.

**Not disturbed, for the record.** Another lane's job (`rb01-shard1-0ea5bff4`)
was live on the worker throughout; it was queued behind, never signalled. The
primary accelerator host was not used. No service was started, stopped or
restarted. No cache, swap, oomd setting or global cap was changed. No driver
was reloaded and no host was rebooted. No committed sample output, no live
checkout and no presentation work was altered. The live cube was read never and
written never; every store test used a temporary database file.

### Artifacts

Committed on `satoshi/rb02-weather-matched-20260928`:

- `tools/df_tsl_repro.py` — the dataset-parameterised sibling of
  `df_sota_repro.py`: seal, characterize, receipt, **producer gate**,
  TRAIN-only pilot, frozen cost. It introduces no second model, loader, loss or
  scorer and reuses the Electricity runner's author bridge verbatim.
- `tools/test_tsl_producer_contract.py` — 22 tests: the producer gate proved by
  refusal, the pinned recipe asserted against the author's own files and the
  published tables, and the temporary-store round trip.
- `docs/audits/work_plan/SATOSHI_RB02_WEATHER_MATCHED_2026_09_28.md` — this
  document.
- The dispatch index's `TSL-WEATHER-TRAFFIC` row, updated in place with its
  previous observation preserved in `observation_history`.

Operator-retained, not committed (they carry host detail):
`~/.local/state/crispdm-data-foundation/tsl_weather_rb02_20260928/` —
`DESIGN.weather.L96.json`, `DESIGN.traffic.L96.json`,
`CHARACTERIZATION.weather.json`, `SCALERS.weather.npz`,
`GPU_ADMISSION_SMOKE.worker.json`,
`TRAIN_PILOT.weather.h96.json`, `TRAIN_PILOT.weather.h720.json`,
`FULL_COST.weather.L96.json`; plus the admission store's retained leases
`rb02-gpu-smoke-…`, `rb02-weather-train-pilot-h96-…`,
`rb02-weather-train-pilot-h720-…` on the worker.

— Satoshi III (Mujuro Utsutsu), successor technical lead, 2026-09-28.
