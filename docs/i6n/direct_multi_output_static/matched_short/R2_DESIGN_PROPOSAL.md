# I6-DMO R0 versus R2 on the selected MG2m recipe: design proposal

**Status: FROZEN_BEFORE_FIT_DESIGN_ONLY, objective UNDECIDED_REQUIRES_REVIEW.**

The sealed contract is `R2_CONTRAST_DESIGN.json`, built by `tools/i6dmo/r2_design.py`
(`seal`). Its fields come deterministically from the sealed matched artifacts.

Until a reviewer rules on the objective, this command refuses the contract with
`R2_OBJECTIVE_UNDECIDED`:

```
python -m tools.i6dmo validate-design --design matched_short/R2_CONTRAST_DESIGN.json --json
```

The same refusal comes from `python -m tools.i6dmo.r2_design validate --for-execution`.
Without `--for-execution`, it reports `PREPARING`.

Nothing in this document ran. There was no pretraining, no fit, no GPU job and
no read of TEST.

Authority:
- NEXT_CONTRASTS_PLAN_20261015 §3 (authoritative);
- MUSASHI_TO_SATOSHI_MATCHED_AUDIT_FOLLOWUP_20261015 §4;
- matched_short/MUSASHI_AUDIT_20261015.md.

## 1. What is fixed (not open to review here)

### Recipe

The closed campaign `i6dmo-6004b711281c` selected MG2m automatically
(`MG2_SELECTED`; COMPARISON.json `2ee85a12…`). MG2m means 13 slots, 2 expected
open-market hours apart, sealed design `d71d2b3c…`, effective `ad949872…`,
seal `416c1156…`.

Dev-2024 relative errors (MAE_h − naive_h)/naive_h for MG2m:

| h2 | h3 | h4 | h5 | h6 | h12 |
|---|---|---|---|---|---|
| +0.0044% | +0.0089% | −0.0165% | +0.0037% | −0.0095% | −0.0374% |

MG1 was worse on every head (+1.9% to +6.6%).

### Identical for R0 and R2

- **Population and inputs.** The common population is `ba01f87a…`. Both arms
  use the MG2m arm tensors from `common_population.arm_tensors(pop,
  EXPERIMENT.mg2m.json, split)`.
- **Scaler and masks.** Both arms use the shared FIT-only scaler (row digest
  `08f655df…`) and the common per-head masks.
- **Naive and seed.** Both arms use the FIT-only naive loss scales and seed
  6102026.
- **Architecture.** Both arms have 19 causal Conv1D branches (4 filters, k3,
  s1), concat fusion, `DynamicPhysicalTimeEncoder`, a core of 8 then 4 Conv1D
  layers, and six Flatten → Dense(8) → Dense(1) heads. That is 4882
  parameters, capacity `5b622fd0…`.
- **Training recipe.** AdamW lr 1e-3, weight decay 1e-4, batch 256, at most
  200 epochs, patience 10, min_delta 1e-8, best weights restored. The monitor is
  the mean of end-of-epoch FIT and INNER.
- **Targets.** RAW exact t+h prices. No reconstructed or denoised targets and
  no replacement input series.

### What R2 changes

R2 is the same network. Its 19 branch encoders start from pretrained weights,
imported by `tools.i6dmo.model.import_branch_donors` and kept trainable. The
whole network is then fine-tuned end to end.

Core and head initial weights stay bitwise equal to the R0 seed build; the test
asserts this.

### Out of scope

The contract refuses each of these with `R2_SCOPE` or `TEST_FORBIDDEN`:
- core pretraining;
- cadence changes;
- the long family;
- TEST;
- an R1 (frozen) arm;
- any objective or architecture search.

## 2. Donors: never accepted by shape alone

The Conv1D branch kernel is (3, 1, 4) whatever the slot count. Branches with
identical tensor shapes include:
- the MG1 matched branches (25 slots, 1 h);
- the closed full-population MG2 fit;
- the DS2 elapsed and hourly control branches.

An I6-B or FS4 donor differs in shape, but would be equally wrong if it did not.

`check_branch_donor` therefore requires and compares the full identity:
- schema;
- arm;
- sampling rule;
- slot count;
- spacing;
- feature name, index and order digest;
- branch model and conv layer names;
- branch parameters and shapes;
- shared-scaler identity (`scaler_identity_sha256`: exact float64 mean/std, feature order, row digest, n_rows);
- population digest;
- the ruled objective identity;
- per-tensor weight digests.

Every mismatch is a typed refusal (`R2_DONOR_*`). The import is atomic: all 19
donors are verified before any weight is set.

Explicitly excluded donors, from the repository's own evidence:
- **FS4 extractibility autoencoders.** `docs/audits/evidence/i6d_matched_control/README.md`:
  a `24→12→6` encoder with a `6×8` latent in HDF5, which is not a branch donor.
- **I6-B donors.** `tools/i6b_branch_pretraining.py`: `causal_conv1d` 2.0.0,
  shape 24×2 → 24×16, on the elapsed hourly grid.
- **Closed MG2/MG12 fit weights.** Different population. The runtime declaration
  was also nonconforming, per MUSASHI_AUDIT finding 1.
- **MG1 matched weights.**

## 3. Branch-pretraining objective: options (for review)

### Fixed regardless of the objective

- **Donor architecture.** Exactly the R0 branch (`tools.i6dmo.model:dmo_branch`
  1.0.0; input (13, 1), output (13, 4)). No new encoder architecture is inferred.
- **Starting weights.** Pretraining starts from the R0 seed-build branch weights.
- **One donor per feature.** Input is the feature's column of the identical MG2m
  slot windows, z-scored by the shared scaler.
- **Time inputs.** These are not branch inputs. The windows themselves carry the
  identical support, calendar and time identity.
- **Data.** FIT common origins only. Selection uses INNER common origins, purged
  of windows whose earliest slot precedes the INNER start; the purged count is
  recorded. DEVELOPMENT and TEST are never opened. No labels are used.
- **Early stopping.** On the INNER objective loss only, restoring the best epoch.
  - The I6-B precedent (`MeanTrainTailEarlyStopping`) monitored mean(train, inner
    tail). The coordinator ordered INNER only, so the contract freezes INNER only.
  - The consequence is that INNER influences R2 twice: once in pretraining
    selection and once in the fine-tune monitor. R0 sees INNER only once. This is
    disclosed, not hidden.

### Option table

The repository already contains these objectives as versioned components,
separate from the architecture (`predictor_plugins/modular_temporal/objectives.py`,
PS3-R/FS17).

| Option | What it is | In repo | Reconstruction | Parameters to rule |
|---|---|---|---|---|
| **O1** `autoencoder_reconstruction` 1.0.0 | Branch, then a learned decoder (`pretraining.build_decoder`/`build_autoencoder`; the decoder is discarded) reconstructs the branch's own z-scored input window. | Yes: `objectives.py:228`, `pretraining.py`. Exercised end to end with `save_donor` objective provenance in `tools/i6b_branch_pretraining.py`. | MEASURED | `decoder_channels` (registered default 32; I6-B used 16), `loss` (registered default mse) |
| **O2** `ts2vec_contrastive` 1.0.1 | Hierarchical instance and temporal contrastive loss on two cropped, masked views. Decoder-free. Yue et al., AAAI 2022; reference code @ b0088e14. | Yes: `objectives.py:114` | NOT_APPLICABLE | `alpha` 0.5, `mask_probability` 0.5, `min_overlap` 8, `temporal_unit` 0 |
| **O3** masked point reconstruction (Ti-MAE-style) | Mask about 75% of steps and reconstruct. | **No.** Only a clean-room description: `docs/audits/evidence/canonical_20261003/laneF/EXTRACTOR_FAMILY_DOSSIER.md` §2; the unofficial code is unlicensed. | MEASURED | It would need a new objective component and review |
| **O4** TimeSiam-style Siamese masked | Past/current windows, lineage embedding, cross-attention decoder. | **No.** Dossier §1: patch tokens, bidirectional. | MEASURED | Architecture mismatch with a per-step causal branch; out of scope |

A ruling must name a registered option. It must fix every declared parameter
and all fit settings: optimizer, learning_rate, weight_decay, batch_size,
max_epochs, patience, min_delta, seed_rule and restore. It must also carry the
reviewer and the sha256 of the ruling document. Anything less is refused:
- `R2_OBJECTIVE_UNKNOWN`
- `R2_OBJECTIVE_UNDECIDED`
- `R2_OBJECTIVE_UNREVIEWED`

`rule_objective` writes a new sealed contract. Every donor records the ruled
objective, and import refuses a donor whose objective differs.

### Recommendation: O1 `autoencoder_reconstruction`, registered defaults

Recommended ruling:
- `decoder_channels` 32, `loss` mse;
- fit settings equal to the R0 recipe: AdamW 1e-3/1e-4, batch 256, max 200,
  patience 10, min_delta 1e-8;
- seed `6102026 + feature_index`;
- restore the best INNER epoch.

Reasons:
1. It is the only branch objective already used end to end in this repository's
   donor path:
   - TRAIN-only data with inner early stopping;
   - restore-best;
   - `save_donor(objective=…)` provenance (I6-B).

   Choosing it adds no new method.
2. Its reconstruction is measurable and deterministic, which the plan's frozen
   reconstruction evaluation needs. O2 can only report NOT_APPLICABLE.
3. O2 fits this branch poorly:
   - The latent is only 4 channels × 13 steps.
   - Its declared deviations (crop by zeroing history; masking by zeroing inputs)
     collide with z-scored inputs, where 0 is the scaler mean, not "missing".
   - With 13 steps, `min_overlap` 8 leaves very few crop configurations.
4. O3 and O4 would mean inventing or porting an objective. The order forbids
   inferring an unreviewed objective.

Known weakness, stated in advance: a 4-channel ReLU causal encoder plus a decoder
can approach an identity code. A low reconstruction error is therefore expected
and proves nothing. Only the downstream R0-vs-R2 contrast decides utility; the
frozen predictive probe below is diagnostic only.

If the reviewer prefers O2, the contract accepts it unchanged. Reconstruction
then becomes NOT_APPLICABLE and the predictive probe carries the evaluation.

## 4. Frozen pretraining evaluations (specified before any fit)

These are published as separate OLAP facts. They never select R0 or R2 and never
gate execution.

- **Reconstruction.**
  - Per feature: `mae_z` and `mse_z` of the decoder output against the input, on
    purged INNER, with FIT reported separately.
  - Reference: the zero (scaler-mean) reconstruction on the same windows.
  - Decoder-free objectives report `NOT_APPLICABLE`.
  - Disclosure: reconstruction alone proves neither predictive utility nor noise
    removal.
- **Predictive probe.**
  - Input: the branches frozen twice, once pretrained and once at the R0 seed
    init, flattened to 19×13×4.
  - Fit: closed-form ridge per head, alpha 1.0, on the FIT common rows with the
    common masks, in delta/naive_loss_scale coordinates.
  - Evaluation: on all INNER common rows. Report per-head masked MAE as a ratio
    to the naive, plus the equal-head mean.
  - `probe_gain` = R0-init probe minus pretrained probe.
  - Report-only.

## 5. R0 reference: identity-agreement predicate

`tools.i6dmo.r2_design.r0_reuse_decision` returns `REUSE_R0` only when every
component agrees, and `FRESH_R0_REQUIRED` otherwise, naming each failed
component.

| Component | Required value |
|---|---|
| status | COMPLETED |
| scientific | true |
| overrides | `{}` |
| device | gpu |
| design_sha256 | `d71d2b3c…` |
| effective_config_sha256 | `ad949872…` |
| common_population_sha256 | `ba01f87a…` |
| matched_arm | mg2m |
| capacity_sha256 | `5b622fd0…` |
| scales_sha256 | equal to the R2 graph's own shared scales |
| scaler fit_row_digest | `08f655df…` |
| TensorFlow runtime | 2.21.0 |
| trainer `code_files_sha256` | equal to the R2 graph's pinned code |
| monitor identity | equal to current (procedure, partition, input materialization, forward) |
| R1 donor verifier | ACCEPT. This is the one common OPS verifier that will govern `campaign.donor_fit_receipt_problem`/`donor_problem`, `matched_runner._check_donor` and `worker._adoptable`; OPS is implementing it in parallel. |

The R0 report unit is always recomputed inside the R2 graph. The evaluation
identity therefore agrees by construction; only the fit can be reused.

**Predicted outcome at the current code:** `FRESH_R0_REQUIRED`. This commit
changes `tools/i6dmo/model.py` and `tools/i6dmo/design.py`, both trainer code
files, so the closed fit's code hashes (`796ec8f4`) cannot match. The R0 and R2
fits should run under one code pin. The measured MG2m fit took 54.6 s on the
preferred GPU (2.87 GB cgroup peak). That is a cost, not a reason to loosen the
predicate.

## 6. Decision rule and reporting

- **Score.** The equal-head mean of annual pooled relative MAE on the common
  DEVELOPMENT rows. Each head keeps its own naive; negative is better. Per-head
  numbers are primary.
- **Rule.** R2 is selected only if score_R2 < score_R0 − 1e-8. Otherwise R0 is
  retained, because it is cheaper (no pretraining). Any failed head or arm gives
  `INCOMPLETE_COMPARISON` with the missing evidence (`decide_r2`).
- **Separate OLAP facts, each with readback:**
  - pretraining loss curve, reconstruction, probe and compute cost;
  - R0 and R2 FIT/INNER diagnostics;
  - fit costs;
  - per-head DEVELOPMENT metrics;
  - the decision.
- **Limits.** One seed. DEVELOPMENT 2024 informed the feature set. No
  statistical-superiority, holdout or trading-eligibility claim.

## 7. Acceptance tests (requirement → observable; design-layer status)

| ID | Mandatory observable acceptance | Status |
|---|---|---|
| R2-01 | R0 and R2 use the identical population, MG2m tensors, shared scaler, masks, splits and seed; changed population/scaler/design digests are refused | DESIGN-TESTED (contract refusals) |
| R2-02 | R2 differs only in the 19 branch initial weights: core/head init bitwise R0, same capacity 4882, branches trainable, end-to-end fine-tune | DESIGN-TESTED (import) / runner pending |
| R2-03 | Shape-only donors refused: MG1, DS2/hourly/lagged, closed full-population, FS4, I6-B (by rule, slots, names, order, scaler, population, objective, digest) | DESIGN-TESTED |
| R2-04 | Pretraining reads FIT only and selects on purged INNER; DEVELOPMENT/TEST never opened (DEVELOPMENT perturbation leaves donors bitwise unchanged) | CONTRACT-TESTED / runner pending |
| R2-05 | RAW exact t+h targets; no reconstructed/denoised targets or replacement inputs; fine-tune input witness equal to R0's | CONTRACT-TESTED / runner pending |
| R2-06 | Undecided objective refuses execution; a ruled objective is registered, fully parameterized and reviewed; donors carry and match it | DESIGN-TESTED |
| R2-07 | Reconstruction/probe computed as frozen on INNER, published separately, never selecting, never claimed as utility/noise removal | CONTRACT-TESTED / runner pending |
| R2-08 | R0 reuse only when every predicate component agrees and the R1 verifier accepts; otherwise a fresh R0; never a full-population/MG1 fit; R0 report always recomputed | DESIGN-TESTED (predicate) / R1 verifier pending (OPS) |
| R2-09 | Equal-head DEVELOPMENT score; R2 only if lower by more than 1e-8; ties retain R0; failed heads give INCOMPLETE_COMPARISON | DESIGN-TESTED |
| R2-10 | Pretraining/fine-tune/R0 diagnostics and costs as separate OLAP facts with readback | runner pending |
| R2-11 | Donor and R2-model serialization reload bitwise on CPU; bounded GPU replay under the declared numerical policy | runner pending |
| R2-12 | No core pretraining, cadence, long family or TEST field accepted | DESIGN-TESTED |

## 8. Next steps (not done here)

1. Rule the objective. The reviewer writes a ruling document;
   `rule_objective` produces the new sealed contract.
2. Land the OPS common R1 donor verifier.
3. Then the OPS finite runner can be built: pretraining units, then the R0 fit
   (fresh unless the predicate passes), then the R2 fit, then both reports, then
   the comparison, then publication, mirror and close.
4. Until then the successor stays PREPARING.

## 9. Corrections and ruling after review 93bff41d (added; earlier text kept)

The independent review `review/R2_DESIGN_REVIEW_7488e899.md` (Retsu, ACCEPT_WITH_CONDITIONS)
found four defects (D1–D4). The technical lead then ruled the objective. This section records
the corrections. Sections 1–8 are kept as originally written, with one exception: the
`validate-design` command in the header was wrong (`python -m tools.i6dmo design …`) and is now
`python -m tools.i6dmo validate-design …`.

### 9.1 Correction: I6-B did NOT use O1 end to end (the review REFUTED this)

Section 3's claims that O1 was "exercised end to end … in `tools/i6b_branch_pretraining.py`" and
that it is "the only branch objective already used end to end" are wrong.

`tools/i6b_branch_pretraining.py` runs its own Keras loop, which differs from O1 in four ways:

| | I6-B loop | Registered O1 |
|---|---|---|
| Optimizer | Adam, no weight decay | AdamW |
| Decoder | 16 channels | 32 channels |
| Monitor | `MeanTrainTailEarlyStopping`, mean of running train loss and inner tail | validation loss |
| Objective identity | `branch_autoencoder_reconstruction` | `autoencoder_reconstruction` 1.0.0 |

What I6-B shares with O1 is only the decoder family (`branch_autoencoder` / `build_autoencoder`).

**Corrected statement:** O1 is an existing, registered, identity-bearing component
(`objectives.py:228`, defaults `{decoder_channels: 32, loss: mse}`). It has **no prior
end-to-end caller** in this repository. This R2 contrast is its first end-to-end donor use.

Ruling O1 still invents neither an objective nor an architecture:
- the branch is the R0 branch;
- the decoder is discarded.

The O2 version is 1.0.1, as the decorator declares; the `objectives.py` docstring now says so too.

### 9.2 The ruling

The ruling was registered with `rule_objective` (review reference `93bff41d`).

| Contract | File | Seal | Status |
|---|---|---|---|
| Base | `R2_CONTRAST_DESIGN.json` | `12ee51b7b6fc7c4ff0af97c557eabd3cb5b58317a292be30c752176840b638d3` | PREPARING. `validate-design` → `R2_OBJECTIVE_UNDECIDED` |
| Ruled | `R2_CONTRAST_DESIGN.ruled.json` | `0cad8817dffa2729164d55273a40814dcf794e348b95f9b02e1227ef641e4066` | `validate-design` PASS / READY_FOR_EXECUTION |

**Objective**

- O1 `autoencoder_reconstruction` 1.0.0.
- Registered identity sha256 `8f0fa272e6a64d368738ffc85ea4d1aedcbfbef712bd33b1d227e9dbe2a5b3e0`.
- `decoder_channels` 32, `loss` mse.

**Optimizer and stopping**

- AdamW, lr 1e-3, weight decay 1e-4, batch 256.
- `max_epochs` **100**: the registered loop's cap. **Disclosed: R0's fine-tune allows 200.**
- patience 10, min_delta 1e-8.
- Restore the best monitored epoch. Reaching max_epochs is an ordinary end.

**Budgets**

- `max_updates` 100000 and `max_seconds` 86400, both fixed explicitly.
- A `max_updates` or `max_seconds` stop is the typed failure `R2_PRETRAINING_BUDGET_STOP`:
  - no donor is selected;
  - the comparison closes `INCOMPLETE_COMPARISON`.

**Seed**

`keras.utils.set_random_seed(6102026 + feature_index)` is called immediately before each
autoencoder is built (`tools.i6dmo.model.pretrain_branch`). The applied seed is recorded in each
donor terminal and checked at import.

**Monitor**

Validation loss on the purged INNER windows, which are inputs only (no labels), exactly as the
registered loop computes it.

**Disclosures, carried in the contract**

- INNER inputs influence R2 twice and R0 once.
- DEVELOPMENT, the scoring split, is never touched.

### 9.3 Fixes D1–D4 and minor items (all CPU-tested; no pretraining on real data, no fit)

- **D1.** `_validate_objective` now proves the ruling is executable.
  - It requires every parameter and every fit setting, budgets included.
  - It requires the literal settings the registered loop actually runs (AdamW, validation-loss
    monitor, restore-best, the seed rule, typed budget stop).
  - It runs the registered validators (`objectives._settings` and the evaluator fit-loop
    `_settings`) and checks the registered identity sha.
  - It binds the review document bytes.
  - Example: `max_epochs` 200 is now refused (`R2_OBJECTIVE_NOT_EXECUTABLE`).
- **D2.** `pretrain_branch` binds the seed. The test checks that a tiny CPU branch pretraining
  replays bitwise even after the global RNG is perturbed.
- **D3 (blocker).** Donors are importable only through this campaign's own pretraining-unit
  stored terminals.
  - **Pipeline.**
    1. `write_branch_donor_terminal` writes the terminal.
    2. The OPS store authenticates it with a digest reference.
    3. `load_branch_donor` loads it.
    4. `import_branch_donors` imports it, taking `{feature: {"terminal": {path, sha256}}}`.
  - **Identity binds:**
    - campaign and unit;
    - applied seed;
    - code pin;
    - objective ruling seal;
    - learned-corpus digest from `branch_pretraining_corpus`: FIT windows, purged INNER selection
      rows, feature index, purge rule and scaler values;
    - objective;
    - donor artifact bytes;
    - terminal digest.
  - **A self-declared manifest never passes**, even with a perfect identity
    (`R2_DONOR_PROVENANCE_MISSING`).
  - **Trust boundary:** the OPS store's retained terminal reference. Unkeyed hashes cannot
    authenticate a fully rewritten store.
- **D4.** `decide_r2` is strict by default; only the literal `require_scientific=False` relaxes
  it. It requires:
  - scientific GPU fits on the same runtime and the same recorded GPU class (R2-18);
  - the import record binding all 19 donor terminal digests;
  - the R2 fit identity's `r2_import_sha256` to equal that record;
  - the R0 fit to carry no import binding.

  Two plain R0 fits are refused.
- **Minor.**
  - The import is transactional: a mid-way `set_weights` failure rolls every touched branch back
    (`R2_DONOR_IMPORT_ROLLED_BACK`; `…_ROLLBACK_FAILED` means rebuild from the seed).
  - The trainer records `resources.gpu_device_names`, and the matched report carries
    `fit_runtime` and `fit_gpu_class`.
  - The contract gained `shared.fit_pinning`.

### 9.4 Acceptance rows added (R2-13…R2-18, as the reviewer listed them)

| ID | Mandatory observable acceptance | Status |
|---|---|---|
| R2-13 | Each donor references this campaign's authenticated pretraining terminal plus the learned-corpus digest, ruling seal, applied seed and code pin; a forged or copied identity is refused | DESIGN-TESTED |
| R2-14 | Seed bound before the decoder is built; no wall-clock budget stop selects a donor; bitwise CPU replay | DESIGN-TESTED (tiny CPU fixture) |
| R2-15 | The ruled objective is executable by the registered component at validation | DESIGN-TESTED |
| R2-16 | `decide_r2` requires scientific GPU fits and the 19-terminal import record; refuses an import binding on R0 | DESIGN-TESTED |
| R2-17 | Automation (plan §4) for pretraining → R0 → R2 → reports → comparison → close: measured-peak admission class, no repeated completed unit, status at every level, a failed pretraining unit gives INCOMPLETE_COMPARISON | **runner pending (OPS)** |
| R2-18 | R0 and R2 fits on the same runtime and recorded GPU class | DESIGN-TESTED (decision); recording in trainer receipts |

### 9.5 Hooks for the future R2 fit path and OPS runner (not implemented here)

- **The R2 fit receipt.** It must record `identity.r2_import_sha256 = record["import_sha256"]`,
  and the runner must keep the import record.
- **The OPS runner.** It must:
  1. compute each feature's corpus with `branch_pretraining_corpus` from the authenticated
     population;
  2. call `pretrain_branch` in a pretraining unit;
  3. write `write_branch_donor_terminal`;
  4. retain the terminal digest;
  5. pass those references and the campaign provenance into `import_branch_donors`.
