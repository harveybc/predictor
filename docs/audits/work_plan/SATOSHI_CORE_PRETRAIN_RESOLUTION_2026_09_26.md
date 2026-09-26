# MOD-CORE-PRETRAIN: the measured resolution, and the budget-matched re-contrast

**Authority.** The owner's grant of 2026-09-26. Written and signed by **Satoshi, successor technical
lead**, dated 2026-09-26. **Nothing here is written, implied or signed in Musashi's name**, and no
reviewer's name appears on it.

**What this answers.** The two prerequisites the block on `MOD-CORE-PRETRAIN` names in
[`SATOSHI_RP49_RP56_DISPOSITION_2026_09_26`](SATOSHI_RP49_RP56_DISPOSITION_2026_09_26.md) §9 and
[`SATOSHI_RP57_RP64_DISPOSITION_2026_09_26`](SATOSHI_RP57_RP64_DISPOSITION_2026_09_26.md) §6:

1. **a measured resolution for this protocol on this task** — which needs no new training, for the
   reason §2.1 gives rather than asserts;
2. **budget-matched arms with the monitor held fixed** — the monitor defect found in the code, named
   in both of its legs, repaired, proved by rules that fail against the pre-repair file, and the
   contrast re-run matched and published in §4, including the outcome the finding predicted and the
   one it did not.

**Headline.** The resolution of this protocol at 3 seeds per arm, derived from **seed-to-seed
dispersion among runs that share one configuration** and from the paired differences across the
retained replicas, is **0.031106 kW** (95% CI 0.020045 – 0.068498). The effect `MOD-CORE-PRETRAIN`
must resolve is **0.009805 kW**. The effect is **below** the resolution. **This protocol, as it was
run, cannot answer the module's question at 3 seeds.** It would take **23 seeds per arm** — 69 fits
for the three-arm design. The conclusion does not depend on a point estimate: even at the most
favourable end of sigma's own 95% interval the resolution is still 0.020045 kW, **2.04×** the effect.
That is not a maybe, and §7 does not soften it.

---

## 0. Retraction, stated before anything rests on it

An earlier reading — mine, in the two dispositions this document serves, and in the order that sent me
to write it — treated RP60's **scrambled-label difference** as this instrument's noise floor, and
concluded that the floor was five times the effect to be resolved. **That reasoning is withdrawn and
nothing below uses it.**

Shuffling the train labels destroys the signal. The difference it produces measures **how much
structure the labels carried** — a property of the task and the data — and not the seed-to-seed
dispersion of a fitted contrast. It is not a floor, it is not a resolution, and the difference between
two arms cannot be judged against it. **Also withdrawn**: mixing kW with persistence-scaled units
inside one comparison, and any module block derived from that mixture. The two scales appear side by
side throughout this document and are never subtracted from or divided by one another.

**What the ruling rests on instead**, and only on: the pooled within-arm standard deviation of the
module's own three arms — dispersion among runs that share one configuration, everything else held
fixed — together with the paired differences across the retained replicas, with the estimator, its
assumptions and its interval named in §2. This is not a repair applied to the prose: the retraction is
carried by the **artifact**. `RESOLUTION.json` reports the scrambled-label number under
`label_structure`, with `used_in_the_resolution: false` and `used_in_the_ruling: false`; the ruling
carries a `retraction` block; and a rule rebuilds the whole resolution **from the per-cell arrays
alone**, with the control's numbers nowhere in the computation, and requires the published value to
come back bit for bit. The field that carried the withdrawn framing, `dynamic_range`, no longer
exists, and a rule asserts its absence.

The scrambled-label measurement is kept, because it is a real measurement of something — §3 gives it
with its true scope.

## 1. What was read, and the rule that governed the reading

| | |
|---|---|
| audited as the specification | the two dispositions above, and their evidence under [`RP49_RP64_AUDIT_20260926/`](../evidence/RP49_RP64_AUDIT_20260926/) |
| measurements taken from | the retained run roots `e1_phase1_v1b` (RP62/RP63) and `e1_household_successor_v3` (RP49–RP56): `DATA.npz`, each cell's `arrays.npz`, `cell.json` and saved weights |
| one number READ rather than recomputed | RP60's full-scale scrambled-label fit. Its weights were not kept, so its MAE cannot be re-derived; it is labelled `READ_FROM_RETAINED_JSON_WEIGHTS_NOT_KEPT` and it enters no estimate |
| tools | [`tools/df_core_pretrain_resolution.py`](../../../tools/df_core_pretrain_resolution.py) · [`tools/df_core_pretrain_matched.py`](../../../tools/df_core_pretrain_matched.py) · repairs in [`tools/df_e1_phase1.py`](../../../tools/df_e1_phase1.py) and [`tools/df_closure_table.py`](../../../tools/df_closure_table.py) |
| rules | [`tests/test_df_core_pretrain_resolution.py`](../../../tests/test_df_core_pretrain_resolution.py) — **37 rules** |
| evidence | [`CORE_PRETRAIN_RESOLUTION_20260926/`](../evidence/CORE_PRETRAIN_RESOLUTION_20260926/) |

**The rule.** Every number below is **generated from artifacts** by the tools named above and is
reproduced in this document by a generator, not typed: the tables of §2, §4, §5 and §6 are the stdout
of a script reading `RESOLUTION.json`, `MATCHED_CONTRAST.json` and `CLOSURE_TABLE.json`. Where a
number could not be generated it is not here. Every arm mean is recomputed in float64 from the cells'
own stored predictions and labels, and each cell's own record is **checked against** the recomputation
rather than used in place of it — all 27 cells agree to 1·10⁻¹².

## 2. The resolution

### 2.1 Why this needs no training — and the reason, not the assertion

A resolution is a statement about the **dispersion of the instrument**, not about any new
configuration. Everything such a statement needs was already measured and kept:

- **seed-to-seed spread on identical configurations.** Six arms were each fitted at three seeds with
  everything else held fixed — the same prepared data by digest, the same origins, labels, scaler,
  batch, learning rate and ceiling. Those 18 cells are six independent estimates of exactly the
  dispersion a contrast has to beat.
- **a cross-runner replicate of one configuration.** Phase-1's `core_mse` **is** the successor run's
  `R0`: the same design, the same three seeds, and — recomputed here — the *identical* per-seed
  optimiser-update counts (3 762 / 4 000 / 2 508). Their per-seed MAEs differ by at most
  **2.52·10⁻⁶ kW**. So the runner contributes ~10⁻⁶ and the seed contributes ~10⁻², and the noise this
  instrument must beat is **seed noise, not implementation noise**. It also means those two arms are
  **one configuration**: pooling them as two would halve the apparent variance of a single
  configuration's seeds, and the tool counts them once. A rule pins that.
- **the paired differences to be resolved.** Already recorded across the retained replicas, and
  recomputed here from the arrays.

Nothing above is a new experiment. What was missing was the arithmetic that turns dispersion into a
threshold, and the honesty to publish the estimator and the interval with it.

### 2.2 The estimator, its assumptions, and what was tested rather than assumed

**Model.** For arm *a* and seed *s*, error `y_as = mu_a + e_as`, with `e` independent across seeds and
`Var(e) = sigma^2` shared by the arms that are pooled. **Estimator of sigma:** the pooled within-arm
standard deviation, `sqrt(sum SS / sum df)`. **Estimator of the resolution:** the minimum detectable
effect of a two-arm contrast,

    resolution = ( t_{1-alpha/2, df_test} + t_{power, df_test} ) * sqrt(2/n) * sigma ,
    alpha = 0.05 (two-sided), power = 0.80 .

**Interval:** the chi-square interval on sigma at its own df, carried through the same formula. The
interval is the interval of the *instrument*; it is wide at 6 df, and the width is part of the answer.

**Assumptions, stated.** Within-arm residuals normal; independent across seeds; one variance shared by
the arms pooled; the retained cells representative of the protocol's dispersion. The last is the
weakest: **5 of 9 phase-1 cells are `CENSORED_BY_BUDGET`**, so the observed spread is partly a
censored one, and a censored spread is a *lower* bound on the uncensored one — which makes the
resolution below conservative in the module's favour, not against it.

Three things a reader could reasonably dispute were **measured, not assumed**:

| question | measured |
|---|---|
| may the arms be pooled into one sigma? | Bartlett's test **refuses** pooling all five independent arms (χ²=13.61, **p=0.0086**) — `core_mae`'s seed spread is 0.00017 kW against `R2`'s 0.01452 — and **permits** pooling the module's own three (χ²=0.607, **p=0.7382**). The governing sigma is therefore pooled over `R0`, `R1`, `R2` **only**, on **6 df**, so every number below is the resolution of the module's own contrast and not of an average over arms it does not use |
| does seed-pairing reduce the dispersion? | **No.** A two-way additive fit over the five arms gives a seed main effect of **F = 0.0148 on (2, 8) df, p = 0.9853**. There is no shared seed effect, so pairing spends degrees of freedom for nothing — at n=3 a paired *t* has 2 df (critical value 4.303) against a two-sample *t*'s 4 df (2.776). The unpaired design is the *better* one here, and the resolution is quoted under it |
| is the ruling an artefact of a pessimistic test? | **No**, and it is quoted so that it cannot be: the resolution is a **band over three defensible estimators** and the **smallest** is the one ruled on |

### 2.3 The measured resolution

| | kW | persistence-scaled |
|---|--:|--:|
| **sigma** (pooled within-arm, `R0`/`R1`/`R2`, 6 df) | **0.011363460** | 0.018439 |
| sigma, 95% chi-square interval | 0.007322538 – 0.025023090 | |
| **resolution at n=3** (two-sample *t*, sigma at pooled df — the most favourable of the three) | **0.031106321** | **0.050476** |
| resolution, 95% interval | 0.020044708 – 0.068498176 | 0.032526 – 0.111149 |
| resolution under the two-sample *t* at its own df | 0.034488 | 0.055965 |
| resolution under the **paired** *t*, the test both rounds reported | 0.049762030 | 0.080748 |
| resolution at 50% power instead of 80% | 0.018634 | 0.030238 |

**The statement, in the form a resolution has to take.** *On the household minute-level active-power
task (window 60, horizon 60 steps, the run's own 10 020 evaluation origins), with this protocol and
**3 seeds per arm**, a difference between two arms' mean MAE smaller than **0.031106 kW** (95% CI
0.020045 – 0.068498 kW) is indistinguishable from the protocol's own seed-to-seed dispersion at
alpha = 0.05 and 80% power.* Under the paired test the two rounds actually reported, the same
threshold is **0.049762 kW**.

**What it is not**: not a bound on the task's achievable error; not a claim about any arm's accuracy;
not a statement about a single fit — it is about a **difference of arm means**; not transferable to
another task, split, horizon or seed count; and **not derived from the scrambled-label control** (§0).

## 3. The scrambled-label measurement, with its real scope

Kept, because it measures something real — and labelled for what it measures. **It is not a
resolution, not a noise floor, and no ruling here rests on it.**

| | kW |
|---|--:|
| untrained, before any optimiser update | 1.597428222 |
| fitted on 40 080 **scrambled** train labels, 4 000 updates | 0.601542114 |
| persistence naive, on the identical evaluation rows | 0.617372056 |
| the best arm this protocol has produced (`core_mae`, matched) | 0.497770213 |
| the span this measures | **0.103771902** |

**What it licenses.** One statement, and it is worth keeping: **a fit given scrambled labels already
beats the persistence naive** (0.601542 against 0.617372). The protocol reaches below the naive with
no true label information at all, which says something uncomfortable about what beating that naive
demonstrates. **What it does not license**: any threshold for a contrast between two arms.

Two further scope limits, recomputed here. The control's partner is **not budget-matched to it**
(the control ran 4 000 updates, `R0_s1` stopped at 3 762), and the control is **one seed**. It is a
single-seed, unmatched probe of label structure. Nothing more is claimed for it.

## 4. The estimand, declared before the comparison

**Offering the same ceiling does not imply the same updates consumed.** Three different estimands were
available, and a contrast that does not name one has no single interpretation:

| estimand | what each arm gets | what then differs |
|---|---|---|
| `recipe_under_early_stopping` | the stopping rule its own recipe implies — the budget is part of the treatment | updates, and cost |
| `equal_cost` | the same CPU seconds | updates |
| **`equal_updates`** — **declared here** | the same number of optimiser updates | cost |

**Why `equal_updates`.** The question `MOD-CORE-PRETRAIN` asks is whether a pretrained component
changes what *the same amount of optimisation* reaches, so the optimisation must be the same amount.
It was declared **before** the re-contrast ran, in the sealed design
`53b824e8a1ac1d9f092ad4629b2fd0757d4de06d6de0f002e3befcebc3f5b279`, whose `training.budget_match` is
`FIXED_UPDATES` and whose `factors_held` names `optimiser_updates`.

**What this does to the RP63 observation.** `core_mae` 11 762 / `tcn_mse` 11 762 / `core_mse` 10 270
**stands as an observation** and is recomputed here. What it *invalidates* depends on the estimand:
under `equal_updates` it invalidates the comparison; under `recipe_under_early_stopping` it does not —
those numbers are a valid estimate of *that* estimand. RP63's sealed design declared **neither**: its
`factors_held` lists `update_ceiling` and `patience` and **not** `optimiser_updates`, and its monitor
field is a sentence about each arm following its own loss. **The defect is the absent estimand, not
the numbers.** That is the sharper statement, and it is the one this document makes.

**`equal_updates` is not `equal_cost`**, and that is declared rather than discovered afterwards. At
12 000 updates per arm the measured CPU was `core_mae` 619.3 s, `core_mse` 566.3 s, `tcn_mse` 395.4 s.
A run matched on updates is **not** matched on cost, and none of §6 is a cost comparison.

## 5. The monitor defect: where it is, both of its legs, and the repair

### 5.1 Where it is

`tools/df_e1_phase1._fit`, one line, as it stood at `a84a913c`:

```python
loss = "mae" if arm.endswith("mae") else "mse"
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss)
...
es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(patience),
                                      restore_best_weights=True)
```

`val_loss` **is** the arm's own trained loss. The monitor was never a separate choice; it was the
recipe under test, wearing a fixed name.

### 5.2 Two legs, not one

**Leg 1 — the budget.** The stopping epoch followed the loss under test, so each arm consumed a
different number of optimiser updates from the same ceiling. Recomputed here, the whole 1 492-update
asymmetry is **one cell**, `core_mse_s3`, which stopped at 2 508 — and that cell is the **best** of its
arm (0.53801 against 0.55250 and 0.55028). The arm that lost was helped, not hurt, by its short seed,
which is why §6's direction could not be guessed from the spread alone.

**Leg 2 — the checkpoint, named nowhere in either disposition.** `restore_best_weights=True` restored
the argmin of `val_loss`, i.e. of a **different curve in each arm**. For the `*_mae` arms that curve
**is** the measure the comparison is judged on. So `core_mae` was permitted to select its checkpoint on
the judged measure and `core_mse` and `tcn_mse` were not — a selection advantage sitting on top of the
budget one, from the same line. §6 shows it is the leg that carries the effect.

### 5.3 The repair

- `monitor` and `budget_match` are now **required keyword arguments of `_fit` with no default** — the
  defect *was* a default, so a caller must name both.
- `resolve_protocol(design)` **refuses** a design that declares an arm-dependent monitor or no budget
  rule, by name, and names the repair in the refusal. The retained RP63 design is refused. The
  defective protocol stays reachable, but only by declaring `LEGACY_PER_ARM_MONITOR_UNMATCHED_BUDGET`,
  which then carries its own `defect` field: reproducing a defective run on purpose is legitimate;
  reaching it by silence is not.
- `budget_match="FIXED_UPDATES"` installs **no early stopping at all**: every cell runs exactly the
  ceiling and the monitor's only remaining job is to choose the checkpoint. Not an equal *ceiling*,
  which RP63 already had — an equal *count*.
- `budget_audit` / `require_budget_match` **refuse to report a contrast** whose arms consumed different
  budgets, reading the runner's own report shape and naming the spread.
- `monitor` and `budget_match` are part of the **sealed identity**, so a run under the repaired
  protocol cannot share a design digest with the defective one.

**Where the repair had and had not been done before.** `tools/df_e1_huber.py` — the post-RP63
factorial the disposition points to — already fixed the **monitor** (`val_mae` in every arm). It did
**not** fix the budget: it still early-stops with patience 3, so its arms can and do drift. The budget
leg was unrepaired everywhere in this repository until this round.

### 5.4 The rules, and the proof that they fail on the old behaviour

**37 rules pass.** The **10** rules of `TestTheMonitorRepair` were run against the pre-repair file
restored from `a84a913c` — **all 10 fail**, transcript retained at
[`PRE_REPAIR_BATTERY.txt`](../evidence/CORE_PRETRAIN_RESOLUTION_20260926/PRE_REPAIR_BATTERY.txt):

```
10 failed in 2.76s
```

They cannot pass on the old code by construction: each either passes `monitor=`/`budget_match=` (a
`TypeError` on the old signature), asserts a field the old record never carried, or — in the
behavioural rule — fits two arms with different losses and requires them to consume the **same** number
of updates, which the old loop cannot do because its stopping epoch follows the loss under test.

### 5.5 Two further defects, found by these rules and repaired here

**(a) the update count must be read before the checkpoint restore.** The behavioural rule failed on
the *repaired* code with `assert 6 == 18`. Keras 3's `save_weights` carries the optimizer's variables,
so restoring the argmin checkpoint **rewinds `optimizer.iterations`** to the value it held at that
epoch. Read afterwards, two arms that consumed the same budget and restored different epochs report
different counts and look mismatched when they are not — and the accounting invariant
`updates == optimizer.iterations`, which RP60 checked, would have silently inverted. The count is now
read before the restore, the restored value is recorded separately, and a rule pins the order. This
defect did not touch the retained runs: `EarlyStopping(restore_best_weights=True)` restores weights in
memory only, which is why every retained cell's two counts agree.

**(b) the closure table crashed on a root nothing governs.** `tools/df_closure_table.py` raised
`FileNotFoundError` on a run root with no `TERMINAL_RECEIPTS.json` and produced **no table at all** —
while its own stated rule is that a forecast unit without an accepted terminal is *a problem, not an
absence*. A crash reports neither the measurement nor the missing custody, and a table that omits a
measurement because its custody is weak hides it instead of qualifying it. Such a unit now yields its
**scored** row — model error, naive on the identical rows, skill, literature, comparability — with
custody `UNANCHORED_NO_TERMINAL`, its problem stated, and **no path to `verified`**. Four rules pin
that, including one that nothing in an ungoverned root may ever read `verified`.

## 6. The budget-matched re-contrast

**What ran.** Nine cells — `core_mae`, `core_mse`, `tcn_mse` at seeds 1, 2, 3 — on the **same prepared
DATA by digest**, the same origins, labels, scaler, batch, learning rate and seeds, under one
arm-independent monitor (`val_mae`) and **exactly 4 000 optimiser updates each**: 12 000 per arm,
matched by construction, `largest_total_minus_smallest = 0`. Every cell was replayed from its own saved
weights in a **fresh graph over all 10 020 evaluation rows**, and every replay reproduces the stored
predictions **bitwise** (max error 0.0). 1 580.96 CPU seconds, all on CPU. Estimand: `equal_updates`,
declared in §4 before the run.

**What it is not.** This round may not contact a service, so no campaign was registered and no terminal
reported. The run carries `custody: UNGOVERNED_LOCAL_RECONTRAST` in its own design, and every row of
its closure table reads `UNANCHORED_NO_TERMINAL` and `verified: NO`. It is a **diagnostic
re-measurement**, not a governed result. Stated here, not implied by an absent field.

### Closure table — every arm mean in this document

| run | arm | seeds | model error (kW) | model error (z) | naive, same rows (kW), n | h | skill vs naive | literature value & source | comparability | custody | verified |
|---|---|--:|--:|--:|--:|--:|--:|---|---|---|---|
| `e1_phase1_v1b` | `core_mae` | 3 | 0.497770213 | 0.545492 | 0.617372056, n=10020 | 60 | +0.193727 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNCHECKED | NO |
| `e1_phase1_v1b` | `core_mse` | 3 | 0.546930216 | 0.599365 | 0.617372056, n=10020 | 60 | +0.114099 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNCHECKED | NO |
| `e1_phase1_v1b` | `tcn_mse` | 3 | 0.535948087 | 0.587330 | 0.617372056, n=10020 | 60 | +0.131888 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNCHECKED | NO |
| `e1_household_successor_v3` | `R0` | 3 | 0.546929167 | 0.599364 | 0.617372056, n=10020 | 60 | +0.114101 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNCHECKED | NO |
| `e1_household_successor_v3` | `R1` | 3 | 0.556734156 | 0.610109 | 0.617372056, n=10020 | 60 | +0.098219 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNCHECKED | NO |
| `e1_household_successor_v3` | `R2` | 3 | 0.552660340 | 0.605644 | 0.617372056, n=10020 | 60 | +0.104818 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNCHECKED | NO |
| `e1_phase1_matched_v1` | `core_mae` | 3 | 0.497770213 | 0.545492 | 0.617372056, n=10020 | 60 | +0.193727 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNANCHORED_NO_TERMINAL | NO |
| `e1_phase1_matched_v1` | `core_mse` | 3 | 0.539266392 | 0.590966 | 0.617372056, n=10020 | 60 | +0.126513 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNANCHORED_NO_TERMINAL | NO |
| `e1_phase1_matched_v1` | `tcn_mse` | 3 | 0.522917583 | 0.573050 | 0.617372056, n=10020 | 60 | +0.152994 | NOT_COMPARABLE — see note | NOT_COMPARABLE | UNANCHORED_NO_TERMINAL | NO |

**Literature note, identical for every row above** (the registry decides it from identity fields, never from a score): source — Gasparin, Lukovic, Alippi (2019/2022), Deep Learning for Time Series Forecasting: The Electric Load Case, arXiv:1907.09207; CAAI Trans. Intell. Technol. 7(1), doi:10.1049/cit2.12060. Published values — Table 5 (RMSE/MAE kW): FNN 0.76/0.53, DFNN 0.75/0.53, TCN 0.76/0.54, ERNN-MIMO 0.79/0.56, LSTM-MIMO 0.75/0.53, GRU-MIMO 0.75/0.52, seq2seq-TF 0.78/0.57, seq2seq-SG 0.76/0.53. Status **NOT_COMPARABLE**, comparator state `NONE`; `placed_in_comparison_column: false`. Reason: unknown identity fields cannot match as proof: ['target_transform']. Planned matched comparison: read the primary source (or its code) and fill the field; a placeholder is not a protocol.

### The retained contrasts against the measured resolution

| contrast | effect (kW) | 95% CI (kW) | p | state against the resolution | seeds for this effect |
|---|--:|---|--:|---|--:|
| `R1-R0` | +0.009805 | [-0.031540, +0.051150] | 0.4149 | BELOW_THE_RESOLUTION | 23 |
| `R2-R0` | +0.005731 | [-0.031007, +0.042470] | 0.5712 | BELOW_THE_RESOLUTION | 63 |
| `R2-R1` | -0.004074 | [-0.065611, +0.057464] | 0.8026 | BELOW_THE_RESOLUTION | 124 |
| `tcn_mse-core_mse` | -0.010982 | [-0.038616, +0.016652] | 0.2294 | BELOW_THE_RESOLUTION | 18 |
| `core_mae-core_mse` | -0.049160 | [-0.068156, -0.030164] | 0.0080 | AT_THE_RESOLUTION_BOUNDARY | 3 |

### The matched re-contrast against the unmatched one

| contrast | unmatched effect (kW) | matched effect (kW) | change (kW) | matched 95% CI | matched p | state | erased? |
|---|--:|--:|--:|---|--:|---|---|
| `core_mae-core_mse` | -0.049160 | -0.041496 | +0.007664 | [-0.049390, -0.033602] | 0.0019 | AT_THE_RESOLUTION_BOUNDARY | **NO** |
| `tcn_mse-core_mse` | -0.010982 | -0.016349 | -0.005367 | [-0.037549, +0.004852] | 0.0801 | BELOW_THE_RESOLUTION | **NO** |
| `core_mae-tcn_mse` | — | -0.025147 | — | [-0.038471, -0.011824] | 0.0148 | BELOW_THE_RESOLUTION | n/a |

### Per-arm means and seed spread, unmatched against matched

| arm | mean unmatched (kW) | mean matched (kW) | change (kW) | sd unmatched (kW) | sd matched (kW) | updates unmatched | updates matched |
|---|--:|--:|--:|--:|--:|--:|--:|
| `core_mae` | 0.497770213 | 0.497770213 | +0.000000000 | 0.000170593 | 0.000170593 | 11762 | 12000 |
| `core_mse` | 0.546930216 | 0.539266392 | -0.007663824 | 0.007804003 | 0.003189395 | 10270 | 12000 |
| `tcn_mse` | 0.535948087 | 0.522917583 | -0.013030504 | 0.005011532 | 0.005344970 | 11762 | 12000 |

### Like for like: did the repair sharpen the instrument?

Only `core_mse`, `tcn_mse` were run under BOTH protocols, so only those two answer it.

| | sigma (kW) | df | resolution at n=3 (kW) | seeds for a flat 0.01 kW |
|---|--:|--:|--:|--:|
| unmatched (RP63) | 0.006558121 | 4 | 0.019906 | 8 |
| matched (this round) | 0.004401190 | 4 | 0.013359 | 5 |
| ratio | 0.671105 | | | |

### What the matched contrast says, plainly

**The 0.049 kW recipe advantage is not erased.** It shrinks from **0.049160** to **0.041496 kW**, by
**0.007664 kW — 15.6%** — and it keeps a 95% CI that excludes zero (p = 0.0019). The outcome the
finding predicted did **not** happen, and this document says so rather than reporting the prediction
as the result.

**Which leg carried it.** `core_mae` reproduces its retained per-seed values **bitwise** — its monitor
was already `val_mae`, and giving it the extra 238 updates it had been denied changed nothing, because
its restored checkpoint was already inside the budget. The whole 0.007664 kW came from `core_mse`
improving once it was allowed to select its checkpoint on the judged measure (0.546930 → 0.539266).
**Leg 2, the checkpoint, is the leg that carried the effect; leg 1, the budget, contributed nothing
measurable here.** The disposition named leg 1 and missed leg 2, and leg 2 is the one that mattered.

**A result that moved the other way.** The reference TCN block's advantage over our core did not
shrink, it **grew** — from 0.010982 to **0.016349 kW**, a change of −0.005367. Matching the protocol
made our core look *worse* against the reference block, not better. Its CI still spans zero
(p = 0.0801) so nothing is established, but the direction is on the record, and `tcn_mse` gained the
most of any arm from the repair (−0.013031 kW).

**And the observational prediction of §2.5 was wrong in both directions.** The within-arm relation
between updates and error was positive, which predicted that lifting the short arm to the ceiling
would push `core_mse`'s mean **up**. It went **down** by 0.007664 kW. An observational relation among
censored, early-stopped cells did not predict the causal effect of matching — which is precisely why
the re-run was required and is recorded here as a failed prediction of my own.

## 7. Ruling on `MOD-CORE-PRETRAIN`

**STILL BLOCKED** — and the module's own question is **UNANSWERABLE BY THIS PROTOCOL AT THIS SEED
COUNT**. The block's reason has changed: it is no longer the withdrawn scrambled-label argument (§0),
it is a measured resolution.

| the block's three prerequisites | state after this round |
|---|---|
| **(1) a measured resolution for this protocol on this task** | **DISCHARGED.** §2: 0.031106 kW at n=3 (95% CI 0.020045 – 0.068498), sigma 0.011363460 kW on 6 df, estimator and assumptions named, derived from within-arm dispersion alone |
| **(2) budget-matched arms with the monitor held fixed** | **DISCHARGED for the recipe and architecture contrast.** §4 declares the estimand, §5 repairs both legs of the defect with rules that fail on the old file, §6 publishes the matched result. **NOT discharged for the module's own `R0`/`R1`/`R2` arms**, which were never run under the repaired protocol |
| **(3) `MOD-FROZEN-PREFIX`** | **NOT DONE.** Untouched by this round, and it remains a prerequisite |

**The measurement that decides it.** `R1 − R0 = +0.009805 kW` (95% CI −0.031540 to +0.051150,
p = 0.4149); `R2 − R0 = +0.005731` (p = 0.5712); `R2 − R1 = −0.004074` (p = 0.8026). All three are
**below the resolution**, all three CIs contain zero, and the resolution exceeds the largest of them by
**3.17×** at the most favourable estimator and by **2.04×** at the most favourable end of sigma's own
interval. Running the module's contrast at 3 seeds would produce differences indistinguishable from the
protocol's own dispersion, in either direction.

**What it would take.** At the measured sigma, **23 seeds per arm** for the 0.009805 kW effect —
**69 fits** for the three-arm design — and 22 per arm for a flat 0.01 kW. At the retained per-cell cost
those 69 fits are roughly 5 CPU-hours of fitting plus the auto-encoders each `R1`/`R2` cell imports.
These are the numbers for the protocol **as it was run**. Sizing the re-measurement under the *repaired*
protocol requires a sigma measured on `R0`/`R1`/`R2` under it, which this round did not measure and does
not guess: §6's like-for-like ratio is measured on two other arms and is a **projection** if carried
across, and a projection is not a resolution.

**Three things this ruling does not say.** It does not say pretraining has no effect — only that this
instrument at this seed count cannot see one of this size. It does not say the effect is zero: the CI
spans −0.032 to +0.051 kW, and the measured sign is *worse with pretraining*, which the two dispositions
already reported. And it does not rule on `MOD-FROZEN-PREFIX`, `MOD-CONF` or `MOD-E3`.

## 8. Costs, resources, and the terminations ledger

CPU only, `CUDA_VISIBLE_DEVICES=''`, every job under `crispdm-run` inside `crispdm-batch.slice`, env
`trading-stack`. Total fitting: **1 580.96 CPU seconds** across 9 cells, the only training in this
round. Everything else — the resolution, the replays, the closure tables, the 37 rules — is inference,
arithmetic and subprocess testing, seconds each.

**Terminations: none.** No unit of this round was killed, by a cgroup OOM or by host pressure, and no
run was retried with a bigger cap.

| event | unit | limit | attempt | cost lost |
|---|---|---|---|---|
| **admission refusal** (not a termination) | the 37-rule battery | `-m 3G` requested; `crispdm-run` answered *"REFUSED 3G requested, only 2.6G free above the 3G host reserve"* | 1st | **none** — nothing had started |
| re-run of the same battery | the same battery | `-m 2G` | 2nd | — (passed) |

**On the guard, and on the instruction I was given.** Reading `MemAvailable` and then launching holds
**no reservation**: two jobs can each read the same free memory and both be admitted. The instruction
to read `MemAvailable` before each launch is that same defective pattern and I record it as such rather
than as a control I relied on. What actually kept this round from contending is that **every heavy job
ran strictly sequentially** — one launch at a time, each awaited to completion before the next — and
that the one refusal was answered by lowering the cap to 2G, never by raising it. Nine fits were run in
three batches of three for the same reason. No further heavy fit was launched after the correction
arrived, and none is pending.

**Nothing was touched.** No service started, stopped or restarted; no governance or warehouse host
contacted; no GPU; no allocation; no reserved split opened; no committed sample under
`examples/results/` overwritten; nothing under `docs/audits/evidence/d3_k5_20260917/` modified; no
hostname, address or secret written anywhere in this document or its evidence.

**Suites.** `37 passed` for this round's own battery. On a **clean checkout**, this battery beside the
sixteen neighbouring ones the two dispositions name — `test_df_closure_table`,
`test_df_benchmark_contract`, `test_rp49_rp64_audit`, `test_df_e1_huber`, `test_df_e1_seal`,
`test_df_e1_close`, `test_df_e1_pilot`, `test_df_e1_regimes`, `test_df_e1_loader`,
`test_stl_norm_contract`, `test_df_e1_chronology`, `test_df_e1_first_child`,
`test_df_e1_receipt_concurrency`, `test_df_e1_governed_route`, `test_df_e1_governing_report`,
`test_df_ecl_closure_boundaries` — **424 passed, 0 failed**
([`NEIGHBOUR_BATTERIES.txt`](../evidence/CORE_PRETRAIN_RESOLUTION_20260926/NEIGHBOUR_BATTERIES.txt)).
The same set on a **dirty** checkout fails 11 rules of `test_df_e1_governed_route`, all with
*"governing run requires a clean checkout"* — the guard doing its job, not a regression.

## 9. What this round refused

1. **I refused to deliver the resolution I was sent to deliver.** The order's premise — a
   scrambled-label difference as a noise floor five times the signal — is withdrawn in §0, before
   anything rests on it, and the retraction is carried by the artifact and pinned by rules, not only by
   this prose.
2. **I refused to report the outcome the finding predicted as the outcome.** The matched contrast does
   **not** erase the 0.049 kW advantage; it shrinks it by 15.6% and it stays significant. §6 says so in
   the sentence a reader will read first.
3. **I refused to let my own observational prediction stand unmarked when it failed.** §2.5 predicted
   the wrong direction and §6 says which prediction was mine and that it was wrong.
4. **I refused to size the re-measurement under the repaired protocol.** The like-for-like ratio is
   measured on two arms that are not the module's; carrying it across would be a projection wearing a
   resolution's clothes, and the field that would have held it says so instead.
5. **I refused to call the matched re-contrast governed.** It has no accepted terminal, its closure rows
   read `UNANCHORED_NO_TERMINAL`, and nothing in it reads `verified`.
6. **I refused to write anything in the reviewer's name.** This document stands under Satoshi's name,
   by the owner's grant of 2026-09-26, and says so in its first line.

---

**Signed.** Satoshi, successor technical lead — 2026-09-26, under the owner's grant of 2026-09-26.
