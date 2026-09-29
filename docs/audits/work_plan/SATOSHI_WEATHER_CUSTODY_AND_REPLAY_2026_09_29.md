# Weather: identity, replay, custody and the one-sided gap — four statuses, kept apart

Satoshi III (Mujuro Utsutsu), successor technical lead. 2026-09-29 (America/Bogota).
Branch `satoshi/weather-custody-and-replay-20260929`, own worktree off `f40fae93`.
Specification: `MUSASHI_CAMPAIGN_STATUS_TRIAGE_2026_09_29.md` §1 and §4 at `50d06e50`.

**Nothing was retrained.** The twelve retained checkpoints were reloaded, never refitted;
`git diff a5317881 -- tools/df_tsl_repro.py tools/df_sota_repro.py tools/test_tsl_producer_contract.py docs/contracts/`
is empty and so is `git diff f40fae93` over the executed lane's own files. This delivery adds
six files and changes none.

---

## 0. The four statuses, and what each one does NOT carry

They are published separately because three of them are green and one is not, and because a
green replay is the single easiest thing in this campaign to mistake for a green campaign.

| # | status | verdict | what it does NOT say |
|---|---|---|---|
| 1 | **Numerical agreement (identity)** | **IDENTITIES_RECONCILED** — recipe, selected weights, test population, scorer and naive all re-derived and equal, 12/12 cells | it says nothing about whether the numbers are right, only that the audited object is the object the return names |
| 2 | **Replay** | **BITWISE_REPRODUCED — 12 of 12** | an independently invoked evaluation reproduces the measurement; it does not verify the *training* that produced the weights, and it is not custody and not science |
| 3 | **Custody** | **UNCHANGED: `DECLARED_TRANSPORT_OF_GOVERNED_BYTES_NOT_A_NEW_GOVERNED_UNIT`** — no governed unit exists for any scored cell, and none was opened here | a later lane finding the credential does not make it authorization before fitting; the twelve are NOT promoted |
| 4 | **Scientific** | **MEASURED AND CLASSIFIED, NOT VERIFIED AS A RESULT** — `MATCHED_PUBLISHED_RECIPE_EXECUTED` stands; the one-sided gap is **not explained**, and the honest deliverable is a bounded candidate set | reproducing a number bitwise says nothing about whether the recipe was faithfully reproduced against the paper |

**Does the replay reproduce the twelve bitwise? Yes — 12 of 12**, on the same admitted device,
including the sha256 of the float32 prediction array, the sha256 of the target population, and
exact float equality of every reported MSE, MAE and paired naive value.

---

## 1. Job 1 — the five identities, re-derived rather than read from the claim

Every digest below was **recomputed inside the replay child** from the bytes on disk, before any
weight was loaded. A disagreement in any of them is a refusal, not a warning: `replay_cell`
raises before the model is built.

| identity | how it was re-derived | result |
|---|---|---|
| **Recipe — design** | the design body re-digested exactly as the sealer digests it (own digest field removed, `lock.protocol_sha256` removed, canonical separator-explicit JSON) | `dbb3e87f…d8c30` **recomputes** |
| **Recipe — protocol** | the lock re-digested without its own protocol field | `c9672ba4…4a3df8` **recomputes** |
| **Recipe — author code** | the fourteen pinned author files re-hashed from the clone in the scoring process | all fourteen equal; clone at `dffde87e…`, clean, `source_drift` empty |
| **Selected weights** | the sha256 of each `checkpoint.pth` on disk, then again after it was staged for the author's own loader | 12/12 equal to the record; **twelve distinct checkpoints**, so the seed does move the fit |
| **Selection *rule*** | the retained author log **re-parsed** by `parse_author_log`, and its best-validation epoch compared with the record's own `convergence` block rather than trusted as it | 12/12 agree on best epoch, epochs run and early-stopping state |
| **Test population** | the delivered CSV re-hashed in the child, then the target array rebuilt by the author's own loader and hashed as it streamed | `34ee981d…2ba63` (7 235 425 B) equal to the record, the registered lake digest and the delivery; target digest equal per cell |
| **Scorer** | `sha256(inspect.getsource(utils.metrics))` taken in the process that was about to score | `0a565c3f…a5df` equal, 12/12 |
| **Naive** | recomputed from the author's own test loader; the pairing **proved** by digest equality with the model's targets, refused otherwise | 12/12 equal to the record **bitwise**; one naive value per horizon, identical across seeds, as the design requires |

Three cross-checks on the retained artifacts themselves, all independent of the replay:

- **all twelve cell records re-digest to their own `record_sha256`**; the closure re-digests to
  `b9561ffb…`; the probe record re-digests to `2b52f3f1…`. Nothing was edited after the fact.
- **one scaler** and one scaler-fit population across all twelve cells; **one test-population
  digest per horizon**; **one naive value per horizon**.
- the lock's `environment_ours` block records the **sealing** host's environment (an RTX 4070
  laptop, no CUDA visible), **not** the environment that produced the numbers. The executing
  environment is correct in every per-cell record (RTX 4090, torch 2.13.0+cu130). This is a
  documentation defect in the lock, not a measurement defect — but a reader of the lock alone
  would draw the wrong conclusion about what hardware produced the twelve. **Named, not fixed
  here**: the design is sealed and I do not edit a sealed design to make it read better.

---

## 2. Job 2 — the independently invoked evaluation

The return labelled this **UNMEASURED**. It is now measured.

**What was run.** Twelve fresh processes, one per cell, each with its own interpreter, its own
CUDA context, its own loader and its own fresh aggregate admission. Each child re-verified the
five identities above, staged the retained checkpoint where the author's own
`test(setting, test=1)` looks for it, and let the author's code do the reload and the scoring —
the unchunked path: three float32 lists, three `np.concatenate` copies, then
`utils.metrics.metric` on the whole arrays. No chunking, no memmap adapter, no downcast, no
sub-sampling, no training. `tools/test_tsl_replay.py` asserts all of that against the module's
own source.

**The rule it is judged under is the one that already existed.** `df_sota_repro.AGREEMENT["replay"]`
was frozen before any Electricity or Weather score was read. It is read, not rewritten:
a test fails if any `atol=`/`rtol=` literal appears anywhere in the replay module. The rule's
declared device is `cpu`, i.e. a **cross-device** replay; the primary replay here ran on the
**same** admitted device, which is a **stricter** setting of the same rule, not a widened one.
Both registers are reported on their own lines and never merged: the **frozen tolerance**
(the metric recomputed from the replayed predictions within 1e-5 of the stored one) and the
strictly harder **bitwise** question (same array bytes, same reduced floats).

### 2.1 Result — same device (RTX 4090, the admitted physical UUID asserted inside every child)

| | |
|---|---|
| cells replayed | **12 of 12** |
| prediction array sha256 equal to the record | **12 of 12** |
| target population sha256 equal to the record | **12 of 12** |
| author float32 MSE and MAE **exactly** equal | **12 of 12** |
| paired naive MSE and MAE **exactly** equal | **12 of 12** |
| frozen-tolerance criterion met | **12 of 12** |
| **status** | **BITWISE_REPRODUCED** |

Every one of the twelve numbers the return published is therefore reproducible from the retained
checkpoint by a process that shares nothing with the original run but the retained bytes and the
host — not its interpreter, not its CUDA context, not its loader and not its reservation.

### 2.2 Result — cross device (CPU, the rule's own declared device)

The same twelve cells were replayed a second time on CPU, with the one operational patch the
Electricity lane already declares for this case (`torch.load` bound to `map_location=cpu`,
because the author's `test(test=1)` passes none and a CUDA-saved checkpoint is otherwise
unreadable — placement only).

- prediction arrays are **not** bitwise equal, which is exactly what a different kernel should do;
- the **metric** still agrees with the record to **≈3 × 10⁻⁸**, and the frozen tolerance is met
  by all twelve.

That number does real work in §5: **a complete change of compute device moves the reported metric
by 3 × 10⁻⁸, while the gap against the published row is 1.7 – 3.2 × 10⁻³ — five orders of
magnitude larger.** Hardware and kernel arithmetic cannot be the cause of the gap.

### 2.3 The reconciled resource budget

Nothing was priced by analogy and no cap was shrunk to fit.

| | value |
|---|---|
| declared cap per child | **8 GiB — the same cap the executed cells were admitted under, passed once, never reduced** |
| measured whole-cgroup peak, largest replay (h720, cuda) | **4.611 GiB**, read from inside the child |
| the probe's measured peak for the same path (untrained) | 4.547 GiB |
| the original scored cell's peak (trained, includes the training baseline) | 4.792 GiB |
| headroom against the cap | 3.39 GiB |
| every peak field in every record | **non-null**; a null peak would have meant *not measured*, never *small*, and no record here carries one |
| admission | one fresh aggregate reservation per child through the deployed `crispdm-run`; refusals **queued** (`-q`), never evaded |
| GPU | the secondary worker's RTX 4090, asserted by physical UUID inside every child; the preferred RTX 5090 host was **not used** |

The replay peak sits **between** the probe and the cell, which is the ordering the return
predicted and is now confirmed rather than assumed: the untrained probe is a lower bound on a
trained cell's peak, and an eval-only reload of a trained cell sits just above the probe.

**The sampled floor did not authorize anything.** The 8 GiB cap was not derived from these
replays; it is the cap the execution already declared, and it was re-used unchanged. No
measurement in this document is used to justify a tighter cap for anybody.

---

## 3. Job 3 — custody, and the chronology that governs it

### 3.1 The credential reference, reconciled through the existing client path

No secret was printed, copied, moved, logged or embedded. The credential was read by the code
that already reads it, handed straight to the existing client, and never left the process.
`tools/test_tsl_replay.py` asserts that the custody module's own output cannot contain the
credential's content, and that none of the three audit modules can even reach a call that would
register a campaign, take a delivery or submit a terminal.

| check | result |
|---|---|
| credential reference present at the path the deployed adopter already reads | **yes**, mode `0o600`, 33 bytes |
| when it came to exist | **2026-09-14T17:21:29Z** |
| read-only reconcile of the Weather adoption campaign through the existing client | **HTTP 200**, `missing_units: []`, `accounting_only: []`, `lake_only: []` |
| negative control — the same GET with **no** credential | **refused** (the 200 means the credential, not an open service) |
| anything mutated | **nothing**: no campaign registered, no unit opened, no delivery taken, no terminal submitted, no service started, stopped or restarted |

### 3.2 Evidence outside the execution return — recovered and inspected, not assumed absent

The triage requires that absence be established. It was looked for, in three places.

| where | what is there |
|---|---|
| the adoption record the return's own transport points at | the Weather route is `route_complete: true`, `campaign_closed: true`, 2 terminals sent, 0 pending, warehouse content matches |
| the data-gov service, live | the campaign reconciles with **no open unit** |
| the warehouse, read with the canonical reader | **both** adoption units present and terminal: `route-1` COMPLETED (1 metric row), `route-probe-that-fails` FAILED |

**So accepted campaign and terminal evidence for this campaign DOES exist, and it was recovered.**
It is evidence for the **delivery of the input bytes**. It is not evidence for any scored cell.

The decisive query is the other one: a search of the warehouse's `gov_terminal` for **each of the
twelve scored cell unit ids, across every campaign**, returns **0 rows**. Absence is therefore
**established by a read of the store**, not restated from the return's own claim.

### 3.3 The chronology, preserved

| when (UTC) | what | class |
|---|---|---|
| 2026-09-14T17:21:29Z | the service credential exists at the adopter's path | PRECONDITION |
| 2026-09-29T02:01:21Z | the Weather bytes are **governed-delivered** (`VERIFIED_TRANSFER`), both adoption units closed, both terminals in the warehouse | GOVERNED DELIVERY OF THE INPUT |
| 2026-09-29T03:48:32Z | gate one (the evaluation-path memory probe) | measurement |
| 2026-09-29T03:49:09Z | **the first scored cell starts fitting** | FITTING |
| 2026-09-29T04:20:48Z | the last scored cell finishes | FITTING |
| 2026-09-29T04:24:02Z | the execution return is committed, labelling the twelve a declared transport | reporting |
| 2026-09-29T04:47:40Z | a later lane commits the correction that the credential is reachable | reporting |

Two facts follow, and they must not be traded for each other.

1. **The input bytes WERE authorized before fitting.** The delivery and both terminals precede
   the first fit by one hour and forty-eight minutes. That part of the chain is sound and is
   now verified against the store rather than against the return.
2. **No scored cell was ever a governed unit**, before or after. The credential's reachability
   is a fact about the machine, not about the twelve.

**The return's stated reason is wrong; its result is right.** The return says the lane "holds no
data-gov service key". On the machine the credential existed fifteen days before the fits and was
used by Weather's *own* delivery an hour and forty-eight minutes before the first fit. The true
statement is that the lane **did not look for it** — which is what the return also says, in the
same paragraph, and the two cannot both be true. I am correcting the reason and leaving the
result untouched: no governed unit was opened for a scored cell either way.

### 3.4 What would change custody, and why it still would not promote the twelve

Opening a governed unit per scored cell and submitting its terminal would change the class.
**Done now, that is retrospective ingestion**: its registration clock would sit *after* the
fitting clock, and it would have to be labelled as such. It is not proof of authorization before
fitting, and it never becomes one. **The twelve are not promoted to verified-and-governed because
a later lane found a key.** This audit opens no unit for them and contains no code path that
could.

One governance weakness is carried forward unchanged and unfixed: the deployed warehouse still
accepts the literal string `fixture` as an identity tag from any producer. The producer-side gate
refuses it, and that standing test still passes; the warehouse-side defect is still open.

---

## 4. Job 4 — the four statuses in full

**1 · Numerical agreement (identity): `IDENTITIES_RECONCILED`.**
Scope: recipe, selected weights, test population, scorer and naive, each re-derived in the
scoring child, 12/12. It does **not** say the weights are the lowest-validation-loss epoch's
bytes — re-parsing the log proves which epoch the rule *selected*, and only retraining could prove
the saved bytes are that epoch's, which would destroy the audited object. That limit is recorded
in every replay record.

**2 · Replay: `BITWISE_REPRODUCED`, 12 of 12.**
Scope: a fresh process reloading the retained checkpoint through the author's own unchunked test
path reproduces the prediction array, the target population and every reported number exactly.
It does **not** verify the training, the recipe's faithfulness to the paper, or the custody class.
It converts the return's `NOT_INDEPENDENTLY_VERIFIED` on the *measurement* into an independent
verification of the measurement — and nothing else.

**3 · Custody: `DECLARED_TRANSPORT_OF_GOVERNED_BYTES_NOT_A_NEW_GOVERNED_UNIT`, unchanged.**
Scope: the input bytes were governed-delivered before fitting (verified against the store);
no scored cell is a governed unit anywhere in the warehouse; the credential is reachable and
authorized on the existing client path, and that was established without mutating anything.
It does **not** authorize the twelve, retrospectively or otherwise.

**4 · Scientific: measured and classified, not verified as a result.**
`MATCHED_PUBLISHED_RECIPE_EXECUTED` stands as the comparability class. The four horizons and the
four-horizon average remain in `OPERATIONAL_AGREEMENT` under Weather's own Table 7 margin — a
predeclared operational class, not statistical equivalence. **The one-sided gap is not explained.**
§5 gives the framing correction and a bounded candidate set; it gives no cause.

No status carries another. In particular: the replay being bitwise green does **not** make the
campaign verified, does **not** move custody, and does **not** settle §5.

---

## 5. Job 5 — the one-sided gap, diagnosed without touching the test

### 5.1 The framing correction, first

The return reports "eight times out of eight". **That is an inflated n and the argument must not
rest on it.**

- The eight positive differences are **four horizon means × two metrics**. MSE and MAE at one
  horizon are computed from **the same prediction array by the same function call**; they are not
  two observations.
- They are **not twelve per-seed comparisons** either: the class is defined on the three-seed
  mean, and a per-seed row carries a difference, not a class.
- At most **four** quasi-independent comparisons exist, and even those share one dataset, one
  scaler, one code revision, one library stack and one host, so four is generous.
- A two-sided sign test on four units gives **p = 0.125**. Under the most favourable independence
  assumption available, the one-sidedness is **not statistically established**.

The magnitude deserves the same discipline:

| | MSE | MAE |
|---|---:|---:|
| four-horizon average, replicated | 0.24167 | 0.27128 |
| published (Table 7) | 0.239 ± 0.006 | 0.269 ± 0.004 |
| gap | **+0.00267** | **+0.00228** |
| gap in units of the paper's own reported run-to-run std | **0.45 σ** | **0.57 σ** |
| our own three-seed std of the same quantity | 0.00170 | 0.00137 |

So the gap sits at roughly **half of the paper's own run-to-run dispersion**. The sign is
consistent; the size is ordinary. Anyone reporting this as a demonstrated systematic deficit is
over-reading four correlated comparisons.

### 5.2 Where the sign is and is not ordinary

The per-horizon picture is not uniform, and this is the sharpest thing in the diagnosis:

| horizon | gap MSE | our seed σ | gap/σ | gap MAE | our seed σ | gap/σ |
|---:|---:|---:|---:|---:|---:|---:|
| 96 | +0.00275 | 0.00283 | **1.0** | +0.00307 | 0.00271 | **1.1** |
| 192 | +0.00206 | 0.00216 | **1.0** | +0.00224 | 0.00158 | **1.4** |
| 336 | +0.00169 | 0.00204 | **0.8** | +0.00163 | 0.00148 | **1.1** |
| 720 | +0.00318 | 0.00021 | **15.4** | +0.00320 | 0.00006 | **52.0** |

At h96, h192 and h336 the gap is **about one seed-standard-deviation** — indistinguishable from
seed variation with three seeds. At **h720** our three seeds collapse onto each other (σ = 2×10⁻⁴
MSE, 6×10⁻⁵ MAE) and the gap is 15 σ / 52 σ **of our own dispersion**. Whatever produces the h720
gap is therefore **not our seed variation**. It could still be the paper's own run-to-run
variation, which at 0.006 is thirty times wider than ours at h720 — and that asymmetry is itself
a finding: the released code hard-codes one seed and offers no path to vary it, so the paper's
three runs can only have differed through training nondeterminism, not through seeding.

### 5.3 What was inspected, and what each inspection rules in or out

**Versions — RULED IN as a candidate, not demonstrated.** Every dependency is far newer than the
author's own pinned set: torch 2.3.1 → **2.13.0+cu130**, numpy 1.26.4 → **2.5.1**, pandas 2.2.3 →
**3.0.3**, scikit-learn 1.5.2 → **1.9.0**; the paper's A.3 hardware is an A100 40 GB, ours an RTX
4090 Laptop. Several major versions separate the training stack that produced the published row
from the one that produced ours.

**Hardware and kernel arithmetic — RULED OUT at this magnitude.** The cross-device CPU replay
(§2.2) changes the entire compute substrate and moves the reported metric by **3 × 10⁻⁸**. The
gap is **1.7 – 3.2 × 10⁻³**. Five orders of magnitude. Kernel-level differences in the *evaluation*
cannot produce this gap. (They can still differ in *training*, which this does not address — that
is the candidate in the previous paragraph, not this one.)

**dtype — RULED OUT.** The author's float32 reduction and an independent chunked float64
reduction over the same arrays agree to **3.2 × 10⁻⁸** on every cell. Reduction precision is five
orders too small to matter.

**Initialization and RNG — PARTLY RULED IN.** The twelve checkpoints are twelve distinct files, so
the seed genuinely moves the fit. But `run.py` seeds only `random`, `torch.manual_seed` and
`np.random.seed`; it does **not** call `torch.cuda.manual_seed_all` and does **not** set
`torch.backends.cudnn.deterministic`, so training is nondeterministic run-to-run even on identical
hardware. This is a live candidate at h96/h192/h336, where the gap is ~1 σ. It is **not** an
explanation at h720, where our own runs barely differ.

**Checkpoint selection — RULED IN with a sharp caveat, and one-directional.** The sealed budget is
the paper's own Table 6 value: 10 epochs, patience 3. No cell early-stopped; all twelve ran the
full budget. Reading the retained trajectories rather than the record's summary of them:

- the **training loss is still falling at epoch 10 in all twelve cells** — the fit had not converged
  when the budget ended;
- the **validation minimum is interior in 10 of 12** cells, and at the last epoch in 2
  (`h96_s2021`, `h720_s2023`), for which the budget rather than convergence ended the run;
- but the validation loss at epoch 10 exceeds its own minimum by **at most +0.0042** anywhere, so
  validation had flattened into noise rather than still improving.

Under-training can only make the error **worse**, so this is the one inspected mechanism whose sign
matches the observed sign. **The caveat is that it cannot be tested by simply running longer.** The
sealed schedule is `lradj cosine`, i.e. `lr_e = lr/2·(1 + cos(e / train_epochs · π))`: the epoch
budget is an *argument of the learning-rate trajectory*, not merely a stopping rule. Changing
`train_epochs` changes the recipe, it does not extend it. That is not authorized here, and anyone
who does it must report it as a different recipe rather than as this reference.

**Metric reduction — RULED OUT by measurement.** Reported in §5.4.

### 5.4 Metric reduction: the alternatives, measured, none adopted

A closed, predeclared set of reductions was evaluated on the **same replayed predictions**, before
any of them was computed, precisely so that none could be chosen after seeing it. The sealed
reduction remains the measurement; nothing here replaces it, and the diagnostics module cannot
even reach the agreement margin (a test asserts it).

Two denominators are kept apart, because confusing them is the easy mistake here: the **per-seed**
difference (one cell against the published value) and the **class quantity**, which is the
**three-seed mean**'s difference — the only one the agreement class is defined on. A variant that
looks large against one seed can be a rounding of the class quantity. The verdict below is taken
against the class quantity; a test asserts it.

Move in **MSE** relative to the sealed reduction, seed 2021, all four horizons:

| horizon | class gap (3-seed mean) | `batch_means_unweighted` | `drop_last_batch` | `per_channel_then_mean` | `per_step_then_mean` | largest move as a share of the class gap |
|---:|---:|---:|---:|---:|---:|---|
| 96 | +0.00275 | −2.08e−04 | +1.25e−04 | +8.3e−09 | +1.1e−08 | 7.6 % (MSE) / 5.5 % (MAE) |
| 192 | +0.00206 | −2.96e−04 | +1.78e−04 | −7.7e−09 | −8.1e−09 | 14.4 % (MSE) / 9.6 % (MAE) |
| 336 | +0.00169 | −6.46e−05 | +4.53e−04 | −2.2e−08 | −1.9e−08 | 26.9 % (MSE) / 19.1 % (MAE) |
| 720 | +0.00318 | +1.27e−04 | −8.90e−04 | −2.7e−08 | −3.0e−08 | 28.0 % (MSE) / 5.6 % (MAE) |

Readings, in order of how much they matter:

- **The two per-axis controls come out equal to the sealed reduction to ~10⁻⁸**, as they must on a
  rectangular population. They are in the set precisely so that a non-zero answer there would have
  meant a real defect. There is none.
- **No variant explains the gap at any horizon**: `NO_PREDECLARED_REDUCTION_EXPLAINS_THE_CLASS_GAP`,
  4 of 4. The largest single move anywhere is **28 % of the class gap**.
- **But 28 % is not nothing, and its sign matters.** `drop_last_batch` is the one variant with a
  real-world referent: older Time-Series-Library revisions built the *test* loader with
  `drop_last=True` (the pinned revision does not, unconditionally). At **h720** dropping the final
  28-window batch *lowers* MSE by 8.9e−4 — so a producer using that convention would publish a
  number 8.9e−4 below a full-population one, which is **the same direction as our gap** and up to
  28 % of it. At h96, h192 and h336 the sign is the **other** way and the convention would shrink
  our gap rather than explain it. That is exactly the shape of a **partial** explanation at the one
  horizon whose gap our own seed dispersion cannot account for, and it is the reason candidate C4
  in §5.5 stays live instead of being dismissed.
- **The table was produced twice, and the two agree.** The same-device set ran on the audited
  device and every record carries `predictions_are_the_audited_ones: true` — the diagnostic
  refuses to run at all on the GPU if the replayed prediction digest is not the audited one. A
  second set ran on CPU, where the predictions differ from the audited arrays at the 3 × 10⁻⁸
  level of §2.2 and every record honestly says `predictions_are_the_audited_ones: false`. **Every
  variant move above is identical between the two sets** to the precision printed, which is what a
  reduction-convention comparison should be: insensitive to the substrate, sensitive only to the
  convention.
- The same-device set **queued 8 GiB for over twenty minutes** behind another lane's live job and
  was admitted when that lane's budget freed. The cap was **not** shrunk to squeeze it in; the CPU
  set was run in the meantime on the coordinator so the question would be answered either way.

**Verdict: `NO_PREDECLARED_REDUCTION_EXPLAINS_THE_CLASS_GAP`, 4 of 4.** The reduction is not the
explanation. One variant with a real-world referent accounts for up to 28 % of it at one horizon,
in the right direction, and that is reported rather than rounded away.

### 5.5 The honest deliverable: a bounded candidate set and what would discriminate it

The gap is **not explained**. The candidates that survive inspection, with what would separate
them — none of which is authorized by this document:

| # | candidate | status after inspection | what would discriminate it |
|---|---|---|---|
| C1 | **Epoch budget ceiling / under-training** — train loss still falling at epoch 10 in 12/12 | **live, and the only candidate whose sign matches** | *not* "run it longer": with `lradj cosine` the budget is an argument of the LR schedule, so a longer run is a **different recipe**. The discriminating experiment is a `train_epochs = 20` fit on **one** horizon, read on the **validation** curve only, reported as a deviation and never as this reference. Requires explicit authorization; not done here |
| C2 | **Library-stack drift (torch 2.3.1 → 2.13, numpy 1.x → 2.x, A100 → 4090) acting through TRAINING** | **live** | rebuild the author's own pinned environment (`requirements.txt`: torch 2.3.1, numpy 1.26.4, pandas 2.2.3, sklearn 1.5.2) and refit one cell. Costly, and it is a new fit — it does not touch the twelve |
| C3 | **Training nondeterminism, i.e. the paper's three runs differ from ours by luck** | **live at h96/h192/h336 (gap ≈ 1 σ), dead at h720 (15–52 σ of our own σ)** | more seeds at h720 only. Our σ there is 2×10⁻⁴; if more seeds keep it that tight, C3 is finished at h720 |
| C4 | **The published row was produced by a configuration the released code no longer expresses** — an older Time-Series-Library test loader with `drop_last=True`, or unstated Table 6 defaults | **live, and now partially quantified**: at h720 that convention moves MSE the right way by 8.9e−4, **28 % of the class gap**; at h96/h192/h336 it moves it the wrong way | the pinned revision builds the test loader with `drop_last=False` unconditionally, so this is a claim about the paper's producer, not about our code. Only the authors' own artifacts or per-horizon error bars could settle it; the paper publishes neither |
| C5 | Hardware / kernel arithmetic in **evaluation** | **REFUTED** — 3×10⁻⁸ against a 10⁻³ gap | — |
| C6 | Reduction dtype | **REFUTED** — 3.2×10⁻⁸ | — |
| C7 | Metric reduction convention | **REFUTED** — §5.4 | — |
| C8 | Population, scaler, pairing, or a wrong checkpoint | **REFUTED** — §1, all re-derived and equal | — |

**What was not done, deliberately.** No tuning on the test set. No widening of the frozen
agreement tolerance — `std_paper` is still Weather's own 0.006/0.004 and the diagnostics module
is forbidden by test from touching it. No variant adopted because it read better. No extra seeds,
no extra epochs, no environment rebuild: each is a new fit and none was authorized.

---

## 6. Placement, discipline and cost

Everything heavy went through `$HOME/.local/bin/crispdm-run` at its deployed revision; nothing was
reinstalled. The coordinator ran only orchestration, the custody reconciliation and the test
suites, each of them capped.

- **Fresh aggregate admission per child**, 8 GiB declared, **never shrunk**. Children queued (`-q`)
  behind another lane's live job on the secondary worker and waited; nothing was killed, signalled,
  raised or evaded, and the other lane's jobs ran throughout and were left alone.
- **The preferred RTX 5090 host was not used**, per the order and the resource return's reading of
  its unreclaimable kernel slab. **No new first-hand reading of that host was taken**, so this
  document makes no claim about its current state.
- **No service was started, stopped, restarted, enabled or disabled.** The household lake
  restoration is not mine and was not touched.
- The audited lane was **read-only** throughout; every artifact produced here landed in a separate
  replay lane.

| | value |
|---|---|
| twelve same-device replays | 91.6 s of child wall in total (5.3–11.2 s per cell, rising with the horizon) |
| twelve cross-device (CPU) replays | 184.0 s of child wall in total (12.9–19.2 s per cell) |
| four reduction diagnostics, twice (same device and CPU) | 8 GiB declared each; the same-device set queued > 20 min behind another lane's job and was admitted unshrunk; largest peak 4.57 GiB |
| largest whole-cgroup peak, any child | 4.611 GiB against an 8 GiB cap |
| peak GPU allocated, any child | ≤ 68 MiB; the 4090's 16 GiB was never a constraint |
| retraining | **none** |

---

## 7. Artifacts

Committed on `satoshi/weather-custody-and-replay-20260929`:

- `tools/df_tsl_replay.py` — the identity reconciliation and the independently invoked
  evaluation. It introduces no model, loader, scorer, recipe, margin or tolerance: the sealed
  `df_tsl_repro` provides the design, `df_sota_repro` provides the author bridge and the frozen
  replay rule, and a test fails if any tolerance literal appears in this file.
- `tools/df_tsl_custody_20260929.py` — the credential reconciliation through the existing client
  path, the recovery of the governed evidence outside the return, and the chronology. Read-only by
  construction; it cannot reach a call that would open a unit.
- `tools/df_tsl_gap_diagnostics_20260929.py` — the predeclared reduction variants, with the
  class gap and the per-seed gap kept as two separate denominators. Labelled
  `DIAGNOSTIC_NOT_A_MEASUREMENT`; cannot touch the agreement margin (a test asserts it).
- `tools/rb02_weather_replay.sh` — the sequential driver: one child at a time, its own admission,
  the cap passed once and never reduced, the loop stopping on the first failure.
- `tools/test_tsl_replay.py` — 31 tests of the generators, written to fail on the specific lies
  this audit could most easily tell: merging bitwise with tolerance, letting a green status carry
  a red one, widening a frozen tolerance, promoting a retained receipt, concluding an absence it
  never looked for, adopting a flattering reduction, or trusting a record's own summary of its log.
- `docs/audits/work_plan/SATOSHI_WEATHER_CUSTODY_AND_REPLAY_2026_09_29.md` — this document.

Suites at this tip, with the production warehouse provider on the path:
**`tools/test_tsl_execution.py` 23 passed · `tools/test_tsl_producer_contract.py` 22 passed ·
`tools/test_tsl_replay.py` 31 passed — 76 passed.**
(Without the provider on the path, two of the producer-contract tests fail on
`ModuleNotFoundError: predictor_duckdb_store`; that is an environment fact, not a regression.)

Operator-retained, not committed (they carry host detail):
`~/.local/state/crispdm-data-foundation/tsl_weather_replay_20260929/` — `REPLAYS/*.cuda.json`
(twelve), `REPLAYS/*.cpu.json` (twelve), `DIAGNOSTICS/*.cuda.json` and `DIAGNOSTICS/*.cpu.json` (four each),
`REPLAY_CLOSURE.weather.L96.json`, `CUSTODY.weather.json`, `code/` (the staged audit tools, digest
matched against the committed ones), and the per-cell replay work directories. **The audited lane
was never written to**: every child read it and wrote only into the separate replay lane, and the
CPU diagnostic set read its four checkpoints through a **read-only shadow lane** whose
`checkpoint.pth` files were re-hashed against the records before use and matched exactly.

---

## 8. What the auditor should attack first

1. **The replay verifies the measurement, not the training.** Twelve bitwise reproductions prove
   that the twelve numbers follow deterministically from the twelve retained checkpoints and the
   sealed population. They prove nothing about how those checkpoints came to exist. The
   `IDENTITIES_RECONCILED` status explicitly excludes "the saved bytes are the best-validation
   epoch's", because only retraining could establish it and retraining would destroy the object.
2. **The custody correction cuts both ways.** I found that the credential existed and was in use
   for Weather's own delivery before the fits, which means the return's stated reason for not
   opening governed units was factually wrong about the machine. A reader could take that as a
   reason to ingest the twelve now. It is not. The registration clock would be after the fitting
   clock and that is all retrospective ingestion can ever be.
3. **C1 and C4 in §5.5 are the uncomfortable ones.** C1 (epoch-budget ceiling) is the only
   candidate whose sign matches the observed sign, and I did not test it because testing it means
   deviating from the sealed recipe. C4 (the published row came from a configuration the released
   code no longer expresses) is untestable from here and would, if true, mean the comparison is
   less matched than `MATCHED_PUBLISHED_RECIPE_EXECUTED` sounds. Neither is resolved.
4. **C4 acquired a number and it is not small.** If the published row came from a producer whose
   test loader dropped the final partial batch — as older Time-Series-Library revisions did — that
   convention alone accounts for **28 % of the class gap at h720, in the right direction**, and for
   nothing at the three shorter horizons. h720 is also the only horizon whose gap our own seed
   dispersion cannot explain. I cannot settle it from here, and I am not going to pretend the
   coincidence is uninteresting.
5. **The h720 asymmetry is the strongest single anomaly in the campaign** and I have not explained
   it: three seeds that agree to 6×10⁻⁵ on MAE, sitting 52 of their own standard deviations away
   from the published value, inside a margin borrowed from a four-horizon average. The margin
   holds. The reading is uncomfortable, and I would rather it be attacked than accepted.

— Satoshi III (Mujuro Utsutsu), successor technical lead, 2026-09-29.
