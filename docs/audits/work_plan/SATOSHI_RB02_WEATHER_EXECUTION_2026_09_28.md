# RB02 execution — gate one measured, then the twelve Weather cells

Satoshi III (Mujuro Utsutsu), successor technical lead. 2026-09-28 (America/Bogota).
Branch `satoshi/rb02-weather-execution-20260928`, based on
`satoshi/rb02-weather-matched-20260928` at `a5317881`.

**What this delivery is.** The execution of the already-frozen Weather design
`dbb3e87f…d8c30` / protocol `c9672ba4…4a3df8` under the granted allocation of
2 700 GPU-seconds on the secondary worker's RTX 4090
`GPU-a8bd1b2c-…-fc3dfc6780f9`. Gate one — the bounded evaluation-path memory
probe that the matched delivery could only *derive* — was measured first, and
the twelve scored cells ran behind it. Nothing was re-pinned, nothing was
re-derived, and `tools/df_tsl_repro.py`'s sealed behaviour was called, never
changed (`git diff a5317881 -- tools/df_tsl_repro.py tools/df_sota_repro.py
tools/test_tsl_producer_contract.py docs/contracts/` is empty).

**What this delivery is not.** It is not a new governed unit: the bytes are the
adoption campaign's already governed-delivered ones, re-verified by sha256
inside every child and labelled a **declared transport of governed bytes**. It
is not statistical equivalence with the paper — the class below is a
*predeclared operational* one against Weather's own published dispersion. It is
not a claim of current best SOTA: a published-recipe reference reproduces one
named paper's own reported configuration. Traffic was not started, not priced by
analogy, and not touched.

---

## 1. Gate one: measured, against the figure that was only derived

The 4.4 GiB at h720 in the matched delivery was a **derivation** from element
counts, and the identical quantity is what left six Electricity cells dead. It
is now measured. The probe runs the **author's own unchunked `test()`** — three
float32 lists, three `np.concatenate` copies, then `utils.metrics.metric` on the
whole arrays — with an **untrained** model at h720 on the real 9 820-window test
population, reading the **whole-cgroup** high-water mark from **inside** the
child.

| | value |
|---|---|
| **measured whole-cgroup peak, in-child `memory.peak`** | **4 882 067 456 B = 4.547 GiB** |
| derived figure it replaces (this document's own term-by-term derivation) | 4 513 854 720 B = 4.204 GiB |
| measured / derived | **1.082 — the derivation was 8 % optimistic** |
| declared cap (never shrunk) | 8 589 934 592 B = 8 GiB |
| headroom | 3 707 867 136 B = 3.45 GiB |
| **verdict** | **ADMISSIBLE** |
| scored population | 9 820 × 720 × 21 = 148 478 400 elements, float32, complete |
| evaluation wall / CPU | 4.91 s / 5.06 s (3.75 user, 1.30 system) |
| peak GPU allocated / reserved | 23.6 MB / 52.4 MB |
| device asserted inside the child | `GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9` |
| record digest | `2b52f3f1…a3d92e` |

The cgroup stage peaks say *where* the memory went, which a single number
cannot. They are read at the moment the author's own scorer is entered and left,
through a measurement wrapper that passes the author's arrays and return value
through untouched — the same wrapper the lock already declares as an operational
patch with no arithmetic effect:

| stage | cgroup peak | what had happened |
|---|---:|---|
| entry | 514 MB | interpreter, torch, CUDA context |
| after model build | 567 MB | the model on the device |
| **before scoring** | **3 692 MB** | every batch accumulated and the three arrays concatenated |
| **after scoring** | **4 882 MB** | `utils.metrics.metric`'s full-size temporaries — +1 190 MB |

The derivation was low because it priced the concatenate stage at 1 861 MB and
the scoring stage at 1 267 + 1 782 MB; the machine spent **more** on the
accumulate-and-concatenate stage (3 692 MB) and **less** on the metric's
temporaries (+1 190 MB, i.e. two full-size operands live at once, not three).
The total came out 8 % above the derivation. Reported this way rather than as a
single ratio, because the error is in the term structure, not in the answer.

**Nothing in the execution path can make an oversized evaluation fit.** The
executor contains no chunking, no memmap adapter, no downcast and no
sub-sampling of the metric population; `tools/test_tsl_execution.py` asserts this
against the module's own source, and `probe_verdict` returns `UNDETERMINED` —
never admission — when either the peak or the cap cannot be read. Had the probe
exceeded the cap, the deliverable would have been the named capacity deficit.

Derived peaks for the other three horizons, for the record (h720 is the only one
measured, and it is the binding one): h96 1.835 GiB, h192 2.219 GiB,
h336 2.782 GiB.

## 2. The twelve cells

Sequential, one governed unit per cell through `crispdm-run`, 8 GiB declared cap
each, never two GPU children at once, the device asserted **inside** each child
by physical UUID, all three seeds through the same three calls `run.py` makes.
Every cell re-verified the registered bytes by sha256 **inside** the child
before the loader saw them, and every cell refused to start unless gate one's
record carried `ADMISSIBLE`.

| cell | MSE (z) | MAE (z) | naive MSE | naive MAE | skill MSE | skill MAE | best ep | wall s | cgroup peak GiB | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `weather_L96_h96_s2021` | 0.1530 | 0.1992 | 0.2591 | 0.2542 | 0.410 | 0.216 | 10 | 146 | 1.96 | 60 |
| `weather_L96_h96_s2022` | 0.1556 | 0.2025 | 0.2591 | 0.2542 | 0.400 | 0.204 | 9 | 144 | 1.95 | 60 |
| `weather_L96_h96_s2023` | 0.1587 | 0.2046 | 0.2591 | 0.2542 | 0.388 | 0.195 | 9 | 144 | 1.96 | 60 |
| `weather_L96_h192_s2021` | 0.2021 | 0.2464 | 0.3092 | 0.2917 | 0.346 | 0.155 | 6 | 147 | 2.42 | 61 |
| `weather_L96_h192_s2022` | 0.2037 | 0.2490 | 0.3092 | 0.2917 | 0.341 | 0.146 | 9 | 146 | 2.43 | 61 |
| `weather_L96_h192_s2023` | 0.2064 | 0.2493 | 0.3092 | 0.2917 | 0.333 | 0.146 | 9 | 146 | 2.42 | 61 |
| `weather_L96_h336_s2021` | 0.2595 | 0.2890 | 0.3764 | 0.3377 | 0.311 | 0.144 | 7 | 152 | 3.06 | 62 |
| `weather_L96_h336_s2022` | 0.2622 | 0.2920 | 0.3764 | 0.3377 | 0.304 | 0.135 | 8 | 162 | 3.07 | 62 |
| `weather_L96_h336_s2023` | 0.2634 | 0.2909 | 0.3764 | 0.3377 | 0.300 | 0.139 | 5 | 155 | 3.07 | 62 |
| `weather_L96_h720_s2021` | 0.3453 | 0.3442 | 0.4652 | 0.3935 | 0.258 | 0.125 | 5 | 168 | 4.79 | 68 |
| `weather_L96_h720_s2022` | 0.3453 | 0.3443 | 0.4652 | 0.3935 | 0.258 | 0.125 | 8 | 169 | 4.79 | 68 |
| `weather_L96_h720_s2023` | 0.3449 | 0.3441 | 0.4652 | 0.3935 | 0.258 | 0.125 | 10 | 170 | 4.79 | 68 |

`best ep` is the lowest-validation-MSE epoch, which is the checkpoint the author's
`train()` reloads before `test()`. **No cell early-stopped**; all twelve ran the
sealed 10-epoch budget. Two of them (`h96_s2021`, `h720_s2023`) had their best
validation epoch at epoch 10, so for those the **epoch budget, not convergence,
ended the run** — recorded per cell as `EPOCH_BUDGET_CEILING_BEST_AT_LAST_EPOCH`
against `RAN_THE_SEALED_EPOCH_BUDGET_BEST_BEFORE_THE_END` for the other ten. The
budget is the paper's own Table 6 value and was not extended.

## 3. The comparison, beside the published row

Metrics in the **normalized** target space (`--inverse` False), the mean over
**every** test window × forecast step × target channel element, with the paired
persistence naive on **exactly** the same rows — proved, not asserted, by the
digest of the target population, which the executor refuses to publish without.

| horizon | elapsed | model MSE (3-seed mean) | published MSE | class | model MAE | published MAE | class | naive MSE | skill MSE |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| 96 | 16 h | 0.1557 ± 0.0028 | 0.153 | OPERATIONAL_AGREEMENT | 0.2021 ± 0.0027 | 0.199 | OPERATIONAL_AGREEMENT | 0.2591 | 0.399 |
| 192 | 32 h | 0.2041 ± 0.0022 | 0.202 | OPERATIONAL_AGREEMENT | 0.2482 ± 0.0016 | 0.246 | OPERATIONAL_AGREEMENT | 0.3092 | 0.340 |
| 336 | 56 h | 0.2617 ± 0.0020 | 0.260 | OPERATIONAL_AGREEMENT | 0.2906 ± 0.0015 | 0.289 | OPERATIONAL_AGREEMENT | 0.3764 | 0.305 |
| 720 | 120 h | 0.3452 ± 0.0002 | 0.342 | OPERATIONAL_AGREEMENT | 0.3442 ± 0.0001 | 0.341 | OPERATIONAL_AGREEMENT | 0.4652 | 0.258 |
| **avg** | | **0.2417 ± 0.0017** | **0.239 ± 0.006** | **OPERATIONAL_AGREEMENT** | **0.2713 ± 0.0014** | **0.269 ± 0.004** | **OPERATIONAL_AGREEMENT** | | |

- **Margin.** 2 × Weather's own Table 7 dispersion + 0.0005 rounding =
  **0.0125 (MSE) / 0.0085 (MAE)**. Every difference is well inside it: MSE
  +0.0027 / +0.0021 / +0.0017 / +0.0032 per horizon and +0.0027 on the average;
  MAE +0.0031 / +0.0022 / +0.0016 / +0.0032 and +0.0023. **Electricity's margin
  (0.005 / 0.006) is not used anywhere here**; the executor reads
  `design["lock"]["agreement"]`, and a test fails if the two are ever confused.
- **Every replication is slightly WORSE than the published value** — all eight
  differences are positive. A systematic +1 to +2 % gap, not noise around the
  paper's number. That is what a matched-recipe reproduction on different
  hardware, a different torch and a seed the author's code cannot vary looks
  like; it is reported rather than averaged away.
- **The four-horizon average is formed WITHIN each seed first** (the paper's own
  quantity): 2021 → 0.23996 / 0.26971, 2022 → 0.24169 / 0.27193, 2023 →
  0.24336 / 0.27221. The ± beside it is the dispersion of those three numbers,
  not of the twelve cells; a test asserts the difference.
- **Seed dispersion collapses with the horizon**: ±0.0028 at h96 against
  ±0.0002 at h720. The dispersion is reported beside the class and never
  replaces it.
- **Skill against the paired naive** falls from 0.399 at 16 h to 0.258 at 120 h
  (MSE), and from 0.216 to 0.125 (MAE). The model beats persistence at every
  horizon, and the margin shrinks exactly as the horizon grows.
- **Comparability class: `MATCHED_PUBLISHED_RECIPE_EXECUTED`.** It is a
  predeclared operational class against this dataset's own published dispersion.
  It is not statistical equivalence, and a missing comparability is never cured
  by rescaling.

Two arithmetic checks that ran on every cell: the author's float32 reduction and
an independent chunked float64 reduction over the same arrays agree to
**3.2 × 10⁻⁸** at worst, and every prediction element is finite.

## 4. Governance: stated, not worked around

A new governed unit for a scored cell needs the data-gov service key. **This lane
does not hold it and did not go looking for it.** The adoption campaign
`tsl-extension-…-thuml_tsl_weather-lake-route`
(`4ec0dd8e…a9b41`) is closed and both of its units carry terminals, so there is
no open unit to report into either.

Every cell therefore reads the **already governed-delivered** bytes — delivery
`aead9358ae4e492e909d3e4ee9446775`, state `VERIFIED_TRANSFER`, availability
contract `975df3b6…44cdc`, availability `UNDECLARED` — and re-verifies them by
sha256 (`34ee981d…2ba63`, 7 235 425 bytes) **inside the child** before the
loader opens them. The result is labelled exactly as the previous lane labelled
its pilot:

> **`DECLARED_TRANSPORT_OF_GOVERNED_BYTES_NOT_A_NEW_GOVERNED_UNIT`**

carried in every cell record and in the closure's own evidence classes. No
authorization was invented, no secret was copied, and no runner was written that
avoids the dependency: the receipts are built and gated exactly as a governed
cell's would be, and they are retained rather than submitted. Opening real
governed units for these twelve cells needs the key path from the operator, and
nothing else.

**One governance weakness carried forward, not fixed here.** The deployed
warehouse accepts an identity tag whose value is the literal string `fixture`:
`tools/test_tsl_warehouse_receipt.py` fills *every* required context tag with it
and the store takes the receipt. The **producer-side** gate refuses it — which is
the right place for it, and
`test_the_fixture_identity_the_warehouse_still_accepts_is_refused_here` is the
standing record that it does — but **the warehouse itself would still accept
that shape from any other producer**. That is a warehouse-side defect and it is
still open.

## 5. Placement, discipline and cost

Everything went through `$HOME/.local/bin/crispdm-run` at its deployed revision;
nothing was reinstalled. The coordinator ran only orchestration, the seal-time
tests and the derivation. The **preferred RTX 5090 host was not used**, and the
reading taken before dispatch is why: 4.41 GiB MemAvailable against 4.41 GiB of
unreclaimable slab — the whole of its available memory is accounted for by slab
the kernel cannot reclaim, while the author's evaluation path needs 4.55 GiB of
mostly anonymous memory.

**The 8 GiB cap was never shrunk.** Gate one's request was admitted only after
**5 m 37 s queued** (`-q`) behind another lane's live 6 GiB job on the same
worker — 8 + 6 would have exceeded the 14 GiB `crispdm-batch.slice` ceiling. It
waited; it did not shrink. Nothing was killed or signalled, no cap was raised, no
cache, swap or oomd setting was touched, no service was restarted, no driver was
reloaded and no host was rebooted. The other lane's jobs ran throughout and were
left alone.

Before any GPU work, inside a reservation: admission by **physical UUID** plus a
bounded allocate → 4096² matmul → free smoke — admission `pass`, refusals `[]`,
**no competing compute application**, 16 362 MiB free VRAM, 39 °C, 18.0 GiB host
MemAvailable; matmul 0.177 s, finite (`GPU_ADMISSION_SMOKE.execution.json`).

| | value |
|---|---|
| grant | 2 700 GPU-seconds (0.75 h) for the probe, 12 cells and one retry |
| probe wall / CPU | 5.1 s / 5.1 s |
| twelve cells, wall total | **1 848.6 s** (0.514 h); CPU total 1 836.2 s |
| **probe + cells** | **1 853.7 s = 0.515 h — 68.7 % of the grant; no retry was needed** |
| frozen upper bound it was priced against | 1 558.3 s (0.433 h) |
| per-cell wall | 144–170 s, rising with the horizon |
| whole-cgroup peak per horizon (max over seeds) | 1.96 / 2.43 / 3.07 / **4.79 GiB**, all against an 8 GiB cap |
| peak GPU allocated | ≤ 68 MiB; the 4090's 16 GiB was never a constraint |
| GPU temperature after each cell | 45–49 °C |
| device, every cell | `GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9`, one boot id |

**The frozen estimate was 19 % low, and here is why.** It priced only the train
pass from the pilot's measured step and put the evaluation passes at one third of
a train step per batch. It did not carry interpreter and CUDA start-up, the final
unchunked `test()` the probe has now measured, the paired-naive CPU pass over the
test loader, or the population digests. Those are the 290 s. The measured number
now supersedes the derived one; the pilot's per-step cost itself held up.

Note that 1 853.7 s is **reserved child wall**, not pure GPU-busy time; the
GPU-attributable part is a subset of it. The grant was denominated in
GPU-seconds and the whole wall is charged against it, which is the conservative
reading.

## 6. What is VERIFIED, what is measured-but-unverified, and what is not measured

| claim | class |
|---|---|
| The author's unchunked evaluation path at h720 peaks at 4.547 GiB whole-cgroup, inside the 8 GiB cap | **measured**, in-child `memory.peak`, record `2b52f3f1…`; `NOT_INDEPENDENTLY_VERIFIED` (one host, one execution) |
| That measured peak is 8 % above the figure the matched delivery derived | **VERIFIED** as an arithmetic comparison of two retained artifacts |
| The twelve cells' MSE/MAE in the normalized space over the complete sealed populations | **measured**, receipts retained and gated; `NOT_INDEPENDENTLY_VERIFIED` (one execution per cell, one host, no replay in a fresh process) |
| The paired naive is on exactly the same rows as the model | **VERIFIED** by digest equality of the target population, refused otherwise |
| Every scored population is complete and finite, and equals the sealed windows × steps × channels | **VERIFIED** per cell |
| The author's float32 reduction and an independent float64 reduction agree to 3.2e-8 | **VERIFIED** per cell |
| The four horizons and the four-horizon average are in OPERATIONAL_AGREEMENT with Weather's published row under Weather's own margin | **measured and classified** under a predeclared rule; **not statistical equivalence**, and not a verification of the paper's numbers |
| The published row and the margin themselves | **VERIFIED as a reading of the paper**, asserted by test; not a claim of current best SOTA |
| Every cell ran the sealed recipe with no drift in the author's files | **VERIFIED** (`source_drift` empty on all twelve; clone clean at the pinned revision) |
| The receipts are governed units | **REFUTED by construction** — they are a `DECLARED_TRANSPORT_OF_GOVERNED_BYTES_NOT_A_NEW_GOVERNED_UNIT` and say so |
| Traffic, anything | **`NO_NEW_MEASUREMENT`** — not started, not priced, not touched |
| The eval-path peaks for h96/h192/h336 | **DERIVED, not measured**: only h720, the binding one, was probed |
| Whether these twelve results replay bit-for-bit in a fresh process | **UNMEASURED**; no replay was ordered or run |

## 7. Artifacts

Committed on `satoshi/rb02-weather-execution-20260928`:

- `tools/df_tsl_execute.py` — the probe, the scored cell and the closure. It
  adds no model, loader, loss, scorer, recipe or margin: the sealed
  `df_tsl_repro` provides the design, the receipt and the producer gate, and
  `df_sota_repro` provides the author bridge.
- `tools/test_tsl_execution.py` — 23 tests of the generators: the derivation
  against the documented figure, the probe's verdict at its boundaries, the
  source-level assertion that no chunking or downcast is reachable, Weather's
  margin against Electricity's, the class boundaries, the within-seed average,
  the tampered and foreign-design refusals, the declared transport's refusals,
  and the `fixture` shape the warehouse still accepts.
- `tools/rb02_weather_cells.sh` — the sequential driver: one cell at a time, one
  governed unit each, the cap passed once and never shrunk, the loop stopping on
  the first failure.
- `docs/audits/work_plan/SATOSHI_RB02_WEATHER_EXECUTION_2026_09_28.md` — this
  document.

Suites, re-run at this tip with the production warehouse provider on the path:
**`tools/test_tsl_producer_contract.py` 22 passed · `tools/test_tsl_execution.py`
23 passed — 45 passed**, and 23 passed again from the copy staged on the worker.

Operator-retained, not committed (they carry host detail):
`~/.local/state/crispdm-data-foundation/tsl_weather_rb02_20260928/` —
`EVAL_PROBE.weather.h720.json` (`2b52f3f1…`), `CELLS/*.json` (twelve records,
each carrying its gated receipt), `CLOSURE.weather.L96.json` (`b9561ffb…`) and
its `.md`, `TRANSPORT.weather.json` (`94848890…`),
`GPU_ADMISSION_SMOKE.execution.json`, plus the pre-existing design,
characterization, scalers, train pilots and frozen cost.

## 8. What the auditor should attack first

1. **The probe is a single measurement on a single host.** It admitted the path
   by 3.45 GiB of headroom, so the conclusion is not marginal — but the number
   itself has no replicate, and the cells' own h720 peak (4.79 GiB) came out
   above the probe's (4.55 GiB) because a trained cell carries the training
   baseline the untrained probe does not. The probe is therefore a **lower**
   bound on a cell's peak, not an upper one. It happened not to matter here.
2. **Every replication is worse than the published value, in the same
   direction, eight times out of eight.** Inside the predeclared margin, but
   the sign is systematic. If the margin were the paper's per-horizon error bar
   rather than a borrowed four-horizon-average dispersion, the reading might
   change — and the paper publishes no per-horizon bar.
3. **`MATCHED_PUBLISHED_RECIPE_EXECUTED` is my own class string.** It is defined
   in the sealed lock and it means what §3 says it means; it is not an
   equivalence test and no statistical claim should be read out of it.

— Satoshi III (Mujuro Utsutsu), successor technical lead, 2026-09-28.
