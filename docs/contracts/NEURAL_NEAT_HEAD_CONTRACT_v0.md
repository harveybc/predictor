# `neural_neat_head` contract v0

Satoshi lane S3, 2026-10-02. Branch base: `codex/modular-neat-integration-20261001` @ `22685545`.

**Status: SPECIFICATION ONLY. NOT IMPLEMENTED. NOT MEASURED.** This is a separate contract. It is not the hyperparameter NEAT already merged: `tools/modular_neat_policy.py` (`ModularNeatProposalPolicy`) evolves flat candidate parameters through `optimizer_plugins/neat_optimizer.py` and its module docstring says it does not evolve a neural head. This contract specifies a NEAT-evolved neural network that maps cached frozen latents to forecasts. Companion: `LATENT_CACHE_CONTRACT_v0.md`.

## 0. Coordinator rulings incorporated (2026-10-02, after a2d814b8)

- First scope is ETH: 83 features, 6 horizons, so `H*T` is tiny. ECL (321 channels x 24 horizons) is OUT OF SCOPE for v0. The latent is fused across branches, so a head applied per channel on that channel's slice of the latent is not available. ECL enters only after a later amendment that declares a target-subset decision (which targets, how many) and the genome-size consequences.
- Targets are stated per dataset in section 2.
- The 70/30 confirmation split is accepted as an addition; full-validation numbers are ALSO reported for comparability with every campaign (section 9).
- Budget caps stay PROPOSED, to be priced by lane D3 (agent a5441bce486e6493a) from its one-seed generation price; this document does not wait for it.

## 1. Scope and arms

| Arm | Producer | Head | Role |
|---|---|---|---|
| `B1C1-NEAT` | frozen branches and frozen core (R1/R1) | NEAT-evolved network (this contract) | subject |
| `B1C1-KERAS` | the same cached latents | Keras MLP, parameter-matched (section 8) | control on identical bytes |
| `B1C1-LINEAR` | the same cached latents | the engine's own head, `Flatten` + `Dense(H*T)` (`components.forecast`) | reference |
| `B2C2`, `R3` (end-to-end Keras) | trainable | engine head | adaptation arms; stay as they are; outside this contract and never fed from the cache |

Frozen and trainable producers are never mixed in one comparison cell. The cache refuses anything else (latent cache contract section 1).

## 2. Inputs

- Latent tensor: `[N, 6, 8]`, float32 little-endian, C order, layout time-major then channel (the encoder output `(6, 8)` for the default config: `branch_steps = window = 24`, core `time_factors [2,2,1]`, `stage_channels [32,16,8]`, so time 24 to 12 to 6 to 6; `components.transformer_conv`, `config.default_config`). Time index 0 is the oldest of the six right-edge steps and index 5 is the decision time t (`core_time_grid` is a right-edge grid, `config._normalize`, `common._partition`). Read only through a verified cache.
- Horizons: `config.horizons`, strictly increasing positive ints, bound in the cache grid. Output shape `[H, T]` per row.
- Target: identical to the supervised campaigns: the evaluator's `targets [N,H,T]` in `z_train` space (`tools/modular_candidate_evaluator.py`, scaler fitted on TRAIN rows only). Per dataset, as stated by the coordinator (not independently verified from campaign configs):
  - ETH / FX (the v0 scope): standardized CUMULATIVE log-returns `Y_h`. The zero-return forecast and the train-mean forecast are the strict-minimum references; both are reported.
  - ECL (later stage, out of scope in v0): z-scored LEVELS. There is no zero-return naive; the references are the seasonal-24 h naive (0.247966) and persistence (0.851406) on the same rows (coordinator-supplied values, UNVERIFIED here).
  - `target_residual` (in-window seasonal-naive terms) is supported through the residual-reference side-cache (cache contract section 2, `residual_reference`), so the head predicts the same residual target as the engine's `forecast_plus_seasonal_naive` output. `input_normalization` stays refused (cache contract 1.6).
- Rows: `train` rows are used for any fitting a head needs; `validation` rows for fitness, selection and early stopping. No other split exists.

## 3. Genome encoding

Feed-forward NEAT with explicit innovation numbers, deterministic evaluation order. The repository's genome cannot host node/connection genomes, so the head keeps a NEW genome class. Read in `optimizer_plugins/neat_optimizer.py`: (a) `NeatGene` is `(innovation, param_name, value)` and `NeatGenome.genes` maps innovation to one scalar hyperparameter; there is no node, connection, enabled flag or acyclicity notion; (b) `InnovationTracker.get_innovation(param_name)` assigns one innovation per parameter NAME, whereas connection innovations key on `(src, dst)` and node innovations on a split connection; (c) `compatibility_distance` normalizes value differences with `full_bounds[param_name]`, which has no meaning for connection weights; (d) `mutate_add_param`, `mutate_values` and `neat_crossover` draw from the module-global `random` (`random.seed(seed)` in `create_shared_population`), not from a per-run generator, which violates section 6's replay rule. Reusable only as ideas: the matching/disjoint-by-innovation crossover shape, fitness sharing by species size (`adjust_fitness`), and the greedy species assignment loop (`speciate`) once distance is redefined over `(src,dst)` innovations and weight deltas. They must be re-implemented with an injected `random.Random`.

- Nodes: input nodes `i_{t,c}` for `t in 0..5`, `c in 0..7` (48 inputs, index `t*8 + c`), one bias node, hidden nodes (innovation-numbered, each with an activation gene and a bias), output nodes `o_{h,k}` for `h in 0..H-1`, `k in 0..T-1`, linear (identity) activation.
- Connections: `(src, dst, weight, enabled, innovation)`; acyclic only (a mutation that would create a cycle is rejected, and recurrence is not part of v0); topological evaluation with ties broken by innovation id; float32 arithmetic with float32 accumulation in fixed order.
- Hidden activations, fixed list and order, recorded in the genome digest: `identity`, `relu`, `tanh`, `gelu_erf`. (Distinct from `neat_optimizer`'s 0..7 activation-index mapping, which was not read in full: UNVERIFIED.)
- Weights are evolved, not gradient-trained (primary variant). Mutation operators: perturb weights, replace weights, add connection, add node (split), toggle, change activation; crossover aligned by innovation.
- Variants, both pre-declared; the primary is `flat`:
  - `flat` (primary): all 48 latent values are visible to every hidden node. No time collapse occurs before the head; the first collapse is the head's own mixing of all (t,c) pairs, matching the engine head which does `Flatten` over the completed latent (`components.forecast`, `config.TEMPORAL_COLLAPSE` rule which forbids collapse in the latent, not in the head). Collapse in this contract occurs only inside the head and only as learned weighted sums over all six steps; no fixed pooling, last-step slice or average is applied before the genome sees the input.
  - `step_shared` (secondary, run only if `flat` is run): one genome with 8 channel inputs plus a step-position scalar `t/5`, evaluated at each of the six steps with shared weights, giving `m` per-step outputs; the six `m`-vectors are concatenated (no pooling) and mapped to `[H,T]` by a fixed-topology dense readout fitted by ridge regression on TRAIN rows only (closed form, regularization declared). Parameters of the readout count toward the budget. Its purpose is to test whether a time-equivariant structure helps; it is a different arm and carries its own verdict.
- Initial population: minimal genomes (inputs to outputs, a seeded sparse subset of connections, small weights). The declared control genome is the zero network (all outputs = bias 0), so generation zero always contains an explicit zero-forecast reference. (Zero forecast in `z_train` of returns is the zero-return naive, section 4.)
- Size caps: `max_hidden_nodes`, `max_enabled_connections` declared before the run; exceeding either rejects the offspring (never truncates). Output count is `H*T`; for ECL if `T` is the full channel count the minimal genome already has `48*H*T` possible connections, so an output-sharing design is an open question (section 12.1).

## 4. Fitness

For seed `s`, genome `g`: `MAE_z(g, s) = mean over validation rows, horizons and targets of |pred - target|` in `z_train` (evaluator `_metrics`: the same definition as the supervised campaigns' MAE). Fitness = mean over the 1 to 3 declared paired seeds (the cap used by the merged modular NEAT, `docs/handoffs/SATOSHI_MODULAR_NEAT_CONTINUATION_2026_10_02.md`). Lower is better. Seeds are paired: the NEAT run, the Keras control and the linear reference use the same seed list and the same cache bytes.

Same-row naive: next to every reported value (genome, champion, control), the persistence naive on the same validation rows (the cache `baseline` companion; the evaluator's `baseline_MAE`) and the zero-return forecast (all-zero prediction, i.e. forecast of zero change in the target as standardized), with `skill_MAE = 1 - MAE/baseline_MAE` (null if the denominator is zero, `_metrics`). A table row without its naive is invalid. Whether zero-return and persistence coincide for the chosen target depends on whether the target is a level or a return and is open question 12.2. A seasonal naive appears only if the supervised campaign it is compared with declares one (`config.target_residual` kinds `seasonal_naive`, `seasonal_naive_cumulative`).

A genome with nonfinite output on any row is assigned `+inf` fitness and is never selected. Complexity penalty: none in the fitness; parsimony is applied only as a tie-break (smaller enabled-connection count) at equal fitness to float32 equality.

## 5. Budget accounting

- Cost unit: multiply-add pairs, and wall seconds as a measured side column.
- Latent cost (one time, shared): `N_cache_rows * producer_forward_cost`, built once, charged once, reported beside both arms (cache contract section 5).
- Genome evaluation cost: `N_val * (E_enabled + hidden_nodes)` multiply-adds per genome per seed, plus `N_val*H*T` for outputs. Total NEAT cost: `sum over generations and genomes of evaluations * seeds`. Training rows are not used by `flat`, so only the `step_shared` readout fit adds `N_train * (m*6)^2`-order cost.
- Caps declared before the run (PROPOSED values, to be priced by lane D3's one-seed generation; do not freeze before that price): `population <= 64`, `generations <= 200`, `seeds in {1,2,3}`, `max_genome_evaluations = population * generations * seeds` and a total multiply-add cap; exceeding any cap defers the arm by name (`BUDGET_EXCEEDED_DEFERRED`, as `assembly.BudgetExceeded` does for configs), with no truncation. The values are a proposal; the first pricing run fixes them (a one-seed costing generation precedes any comparison, as the continuation handoff orders for the hyperparameter NEAT).
- The Keras control receives a matched budget: same total multiply-add cap counted as `epochs * N_train * 3 * params` (forward + backward estimate, labelled an estimate) and the same seeds. Both arms report spent versus cap.

## 6. Determinism and replay

- Seeds: a single integer per paired seed drives the Python `random.Random` and numpy `Generator` used by the evolution (the merged policy seeds `random.Random(seed)`, `tools/modular_neat_policy.py`). No wall-clock, set ordering or dict-hash-order dependence.
- State file `NEAT_HEAD_STATE.json`: generation index, RNG state, full population (genome JSON with innovation ids), innovation tracker, best genome, per-genome per-seed fitness, the cache `cache_id`, the head config digest, the code digest. Written to `<name>.tmp` and `os.replace`d (the pattern of `provenance._atomic`; the merged modular NEAT writes `NEAT_STATE.json` atomically the same way, `optimizer_plugins/modular_doin_optimizer.py`). On resume the state refuses to load if any digest differs. The state carries `state_sha256` over canonical JSON without that field.
- Replay: from `(cache_id, head config, seed list, code digest)` the champion genome and its validation MAE_z must reproduce exactly (bitwise within one environment; tolerance 1e-6 declared across environments). The champion genome is saved as JSON with its digest and a replay receipt recording per-row predictions' sha256.
- Duplicate genomes (same canonical topology and weights) are evaluated once and reused.

## 7. Forbidden

- Any test split, in any role, including selecting the champion or reporting a leaderboard.
- Fitting a scaler, mean, std or any statistic on validation rows (the scaler is bound to TRAIN and read from the cache).
- Reading raw windows, extra features or labels beyond the cached latents, the cached targets and the cached baseline.
- Mixing frozen and trainable producers in one cell, or refreshing the cache between arms.
- Re-selecting the cache, seeds, caps, delta or the success rule after seeing any result.
- Reporting an advantage that fails its same-row naive, or sending its forecasts to the heuristic strategy (the handoff rule: a forecast that fails its declared naive is never sent downstream).
- Calling the head a model-quality result before independent checkpoint replay.

## 8. Keras control head

Same input bytes (cache shards, flattened `[N,48]` for the `flat` variant, same row order), same targets and fitness definition, same early-stopping rule applied to validation (`modular_candidate_evaluator.fit_with_early_stopping` semantics: patience, `min_delta`, best weights restored). Architecture: `Flatten -> Dense(w, gelu) -> ... -> Dense(H*T)` with depth `d in {1,2}` and width `w` chosen so that trainable parameters equal the champion genome's parameter count within +-1 %: `params = enabled connections + hidden node biases + output biases` for the genome, `sum(in*out + out)` for the MLP. Because a genome's size is only known after evolution, the control is built after the NEAT champion is selected, and a second pre-registered control uses the engine linear head parameter count (`48*H*T + H*T`). Learning rate, weight decay and loss follow the supervised campaign defaults for the same dataset (`train.*` flat parameters of `tools/modular_search_space.py`; values to be copied from the governed campaign, UNVERIFIED here). Seeds as in section 4. If no integer width gets within +-1 %, the depth-2 split with a free second width resolves it, else the control is `PARAM_MATCH_UNREACHABLE` and the cell is not scored.

## 9. Comparison protocol and pre-registered success rule

Per paired seed `s`: `d_s = MAE_z(KERAS, s) - MAE_z(NEAT, s)` on the full validation set, both with the same cache. Selection optimism: both arms select on validation (NEAT by fitness, Keras by early stopping), so the full-validation number is optimistic for both. Therefore the confirmation split is carved before any run: the last 30 % of validation rows in time order, separated from the first 70 % by a purge of `max(window, max horizon)` rows; NEAT fitness and Keras early stopping use only the first 70 %; the champion of each arm is scored once on the confirmation 30 %. Both numbers are reported, and the FULL-validation numbers (all validation rows, no carve) are ALSO reported for every arm, seed and naive, for comparability with all supervised campaigns; they are labelled optimistic (selection and report on the same rows) and are never used by the rule. The rule below uses the confirmation numbers.

Pre-registered rule (`delta` fixed before the run; default `delta = max(1 % of MAE_z(KERAS), 2 * sd_seed(MAE_z(KERAS)))`, with the sd taken over seeds; with one seed use 1 %):

- `NEAT_HEAD_ADVANTAGE`: every paired `d_s > 0`, mean `d > delta`, and NEAT beats its same-row naive on the confirmation rows (`skill_MAE > 0`).
- `KERAS_HEAD_BETTER`: every `d_s < 0` and mean `d < -delta`.
- `EQUIVALENT_WITHIN_DELTA`: otherwise, including mixed signs.
- `NOT_BEATING_NAIVE`: neither arm has `skill_MAE > 0`; no head claim is made.
- `BUDGET_EXCEEDED_DEFERRED`, `CACHE_REFUSED`, `PARAM_MATCH_UNREACHABLE`, `INCOMPLETE_SEEDS` (any declared seed missing): no verdict.

With 1 to 3 seeds no significance test is valid; labels are descriptive and the report states it. A comparison against `B1C1-LINEAR` is reported with the same table and labels but does not decide the head question. NEAT-vs-random-search of hyperparameters is a different comparison (the matched proposal comparison in the handoff) and is not touched here.

## 10. Verdict table (required fields)

Per arm and seed: cache id, config digest, seed, population/generation/evaluation counts, spent multiply-adds and seconds versus cap, parameter counts, `MAE_z` selection, confirmation and full-validation, `baseline_MAE`, `skill_MAE` beside the zero-return and persistence naives on the same rows, champion digest, replay receipt hash. Cells that were not measured say `NO_NEW_MEASUREMENT` (standing closure-table rule).

## 11. Tests to write first (synthetic mechanics only)

`test_genome_acyclic_only`, `test_topological_order_deterministic`, `test_replay_same_seed_same_champion`, `test_state_roundtrip_resumes_same_next_population`, `test_state_refuses_changed_cache_id`, `test_atomic_state_write`, `test_zero_genome_is_in_generation_zero`, `test_nonfinite_output_gets_inf_fitness`, `test_fitness_requires_all_declared_seeds`, `test_more_than_three_seeds_refused`, `test_no_test_split_reachable`, `test_param_count_matches_between_genome_and_keras_control_within_1pct`, `test_both_heads_read_identical_cache_bytes` (hash of the arrays passed to each), `test_naive_beside_every_reported_value`, `test_budget_cap_defers_not_truncates`, `test_success_rule_labels` (table-driven over synthetic differences including mixed signs and an arm that fails its naive).

## 12. Open questions that reading could not settle

1. ECL target-subset decision and a genome sharing design, required before ECL is brought into scope (section 0).
2. The ETH/FX target definition and the ECL reference values come from the coordinator's ruling; re-read them from the actual campaign configs when the head is implemented.
3. Numeric budget caps: pending lane D3's priced one-seed generation.
4. No concrete ETH cache build has been made (window count, H = 6, T); the cache contract arithmetic uses F = 83.
5. Whether `input_normalization` encoders ever need support (they remain refused).
