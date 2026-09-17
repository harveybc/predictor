# Return: D3 M1–M6 — composition closed, causal utility harness, observed budgets, packaging, dispositions

Order: `docs/handoffs/MUSASHI_D3_L1_L6_REVIEW_AND_M1_M6_2026_09_17.md` (`3439d80`), over the
review of `ea555f5`. Executed without pausing between blocks. `NON_GOVERNING` throughout: no
utility on project data, no reserve scored, no GPU, no trading. The D3 campaigns were not
repeated; v1, v2, v3 and every receipt are untouched.

**PRE** (reviewer's reproducer `reproduce_d3_l_review_2026_09_17.py` at `ea555f5`):
`FUTURE_LABEL_AS_FEATURE ADVANCES 0.777`, `IGNORED_EMISSION True`, `T_DF1 9.71 vs 12.71`,
`CHANGED_UNMEASURED_OPERATOR verified=true`.
**POST**: the composite case prints `INHERITED_OPERATOR_CHANGED` (verified false); the harness
cases cannot be expressed any more — the API that took a precomputed array with `accepted=True`
no longer exists (the reproducer stops at `PROTO`); their substance lives as rules R5/R6/R8
(future label refused by record and by prefix check; `t_crit(0.975, 1) == 12.706` from scipy).

## M1 — composition closed, measurements conserved

Frozen red first (the reviewer's changed-parameter case and five more), then closed in
`_compose`, with the rule written into 07C as `inheritance_equivalence`:

| requirement | now |
|---|---|
| operator equivalence | same `params` **and** same `spec_sha256`, or a diff within the finite `allowed_spec_diff` the amendment declares (07C: **none**) → else `INHERITED_OPERATOR_CHANGED` |
| population identity | same units, variables and data contracts (bank: observed-signal digest; toys: delivered source bytes digest) → else `INHERITED_POPULATION_MISMATCH` |
| partition | inherited ∪ measured = the twelve, disjoint → else `INHERITANCE_PARTITION` |
| code dependency | an inherited row must carry the **source** freeze's design and code digests; a source that no longer verifies refuses the whole composition (`INHERITED_SOURCE_UNVERIFIED`) |

`tests/test_d3_matrix_verify.py` — **34 rules** (6 new: changed parameter, changed spec alone,
changed variable count, changed data contract, bad partition, unbound source row).
**v2 and v3 re-verified from preserved bytes** with the closed verifier: VERIFIED, 0 refusals,
figures identical to the earlier verified matrices (`MATRIX.verified.v3.json` beside the
others). The real scope is equivalent (all nine operators: params and spec digests equal between
the v2 and v3 freezes), so **no measurement is invalidated and none repeated**. `verify()` now
also exports the per-cell verdicts (`MATRIX.verified.*.cells.json`; v3: 6,390 cells — 6,259
accepted, 128 inconclusive, 3 refused) — the harness's eligibility record.

## M2 — temporality and real fit in the utility harness

`tools/df_utility_harness.py` rewritten (protocol `df_utility_protocol.v2`):

* representations are built **by the real operator, per block**: `fit` on that block's training
  prefix only, `transform` with a fresh state, contract-validated output; no precomputed arrays,
  no caches, no `accepted` flag;
* a feature row consumes only outputs **emitted at or before its decision instant**
  (`available_at[t]`); a late output aligns to the later row where it is first available — never
  cured by purge (R3);
* observation identity: ids strictly increasing, times non-decreasing, availability ≥ time;
  discordant → refused; gaps kept; rows paired **by id**, not by value (R4);
* eligibility from the verified matrix's cells for the exact `(unit, variable, operator,
  spec_sha256)` (R5); a control outside the record is never scored;
* **prefix-consistency check**: the series is cut at sampled decision rows and re-transformed;
  any moved output refuses the representation as non-causal *before scoring* — the reviewer's
  future-label control is refused even with a forged eligibility entry (R6);
* scaler and probe fit inside the training block only (R7); tail change leaves prefix features
  unchanged; restart by checkpoint reproduces features (CUSUM); a legitimate historical-lag
  representation is accepted and may advance.

## M3 — inference and experimental contract

* `scipy.stats.t.ppf` (R8: `t(0.975, 1) = 12.706`, low-df tails corrected by the family);
  blocks policy **all_or_insufficient** (no denominator shrinks silently); walk-forward blocks
  with purge `horizon + reach + window`.
* **Calibration record required for ADVANCES**: the protocol seals the false-advance rate
  measured under a null with diagnostic seeds; without it, or with a rate above
  `alpha_adjusted`, the result is descriptive (`INCONCLUSIVE_UNCALIBRATED`).
  Diagnostic (`utility_rehearsal_v3/CALIBRATION_DIAGNOSTIC.v2.json`, 200 sims each):
  exchangeable **white null 0/200** at margin 0 and 0.05 — the block-t interval is conservative
  under the level dependence the blocks share; **AR(1) φ=0.6: 33 %** at margin 0 (4 blocks), 46 %
  (8 blocks), 1 % at margin 0.05 — not false advances: that generator carries sign-run structure
  `delta_run_length` genuinely captures. The sealed calibration uses the exchangeable null; AR(1)
  stays a structured diagnostic (named so in `calibrate`). Dependence between blocks is not
  claimed away: it is what the calibration measures.
* `Protocol` validated before scoring: finite domains, target/model pairs, `n_blocks ≥ 3`,
  declared branches (an undeclared `augmented` request is refused), **family sealed as a tuple of
  contrast ids** (`comparisons = len(family)`, no loose integer), blocks policy, calibration
  schema.

`tests/test_df_utility_harness.py` — **20 rules bound to R1–R10**, on fabricated truth only.

## M4 — observed budgets and layered tests

* `run_isolated`: one contrast per child under `df_isolated_runner` (wall, CPU, memory ceilings
  enforced during the work; cost measured; result written only at the end). A deliberately slow
  control ends `RESOURCE_EXCEEDED` (`WALL_TIME_LIMIT`, cost 6.0 s wall / 6.0 s CPU) with **no
  partial score**; a contrast within budget completes with measured cost and a re-hashed output
  (R9, real systemd scope).
* **Governed entry point** `tools/df_utility_run.py`, rehearsed under data-gov on fabricated data:
  the real D3 battery on the fabricated series produces the eligibility record (v2 of the
  rehearsal showed what happens without an availability contract: every cell INCONCLUSIVE, every
  contrast refused — kept as evidence); calibration 200 sims white null (0 false advances) sealed;
  freeze write-once (protocol, family with the slow control declared, data and eligibility
  digests, budgets); isolated contrasts; campaign `utilreh-v4-utility-rehearsal` (SYNTHETIC,
  `bb95…`→`utilreh-v4` `…`), one terminal per contrast (deltas as metrics; `RESOURCE_EXCEEDED`
  with cost for the slow control), reconciled `missing_units: []`; envelope DEVELOPMENT
  `da341cc5…` tagged `UTILITY_HARNESS_REHEARSAL`, loaded. Outcomes: three contrasts
  `DOES_NOT_ADVANCE` (descriptive of that fabricated series; no claim about representations).
* Layers: unit (protocol, alignment, identity), integration (real operator + cells record),
  system (isolated runner budgets), acceptance-alpha (the governed rehearsal, receipts under
  `~/.local/state/crispdm-data-foundation/utility_rehearsal_v{2,3,4}/`).
* Holdout bound to the reserve identity (`campaign_sha256` or dataset digests) and the protocol,
  write-once in the state directory; a nameless reserve refuses (R10).

## M5 — packaging and dispositions

* `data-warehouse` `43edc44`: `build/` (10 files) and every `__pycache__` removed from the index
  (history preserved) and ignored; rule `test_packaging_clean_build.py`: no generated output
  tracked; a wheel from a clean `git archive` of HEAD is byte-identical to the source, installs
  `--no-deps` into a throwaway venv and the real entry point answers. **No production restart**
  (nothing at runtime changed). The adopter's clean export stays as a second defence.
* The incident stands in its chronology: `M5_WAREHOUSE_RESTART_CHRONOLOGY.txt` — 22 scheduled
  restarts of `crispdm-data-warehouse-olap` between 02:51:41 and 02:55:02 (−05) on 2026‑09‑17,
  before the restore from backup; the live `NRestarts 0` is a counter reset, not the history.
* Dispositions: the accidental envelope stays `ACCIDENTAL_OPERATIONAL_INGESTION` /
  `NOT_ADMISSIBLE_SCIENTIFIC` (accepted in principle by the review). Proved on production
  (`M5_STRICT_SELECTION.json`): the **strict mechanical selection** by identity
  (`df_d3_cube_select.mechanical_strict_sql`: `result_class = MECHANICAL` and no `NOT_ADMISSIBLE`
  disposition) returns `d3mech-v1`, `d3mech-v2` (with its `CORRECTIVE_RELATION`) and `d3mech-v3`
  and **not** `475b93fb…`; `gov_mechanical_evidence` is the broad **operational history** (51
  rows, everything not `CURRENT_SCIENTIFIC`, the accidental DEVELOPMENT ingestion included) and
  is documented as such. Result class, disposition and admissibility are three columns in every
  query; nothing was reclassified.

## M6 — closure

**Suites** (trading-stack, `crispdm-run`, candidate warehouse on the real service's path): D3
contract 40 · operators 37 · pipeline 15 · matrix 1 · verify 34 · probe 17 · twin 20 · delta 3 ·
ingestion 14 · utility harness 20 · dispatch · outbox disposition · `olap/store/tests` — **287
passed, 1 skipped**, 41.0 s; `data-warehouse`: **39 passed**. No exclusions.

**Commits (predictor)**: `5e4f349` M1 · `efa2987` M2–M4 · `707fe8f` `…` governed entry ·
`…` calibration null · this return. **data-warehouse**: `43edc44`.

Backlog, measured: terminal outbox 0 pending; OLAP outbox 0 pending, dead letters adjudicated,
`attention_required: false`.

Open, with owners: the AR(1)-structured diagnostic shows margin matters — the development pilot's
freeze must fix `margin`, `n_blocks` and the null with calibration support before any ADVANCES
(Satoshi, at the pilot's freeze); index-loss root cause, Metabase, terms: separate fronts.

Ending: **`D3_COMPOSITION_CLOSED_UTILITY_CAUSAL_HARNESS_READY_FOR_REVIEW`**. The next step is
the development utility pilot after this harness is reviewed; this order opened no confirmation.
