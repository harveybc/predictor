# Satoshi continuation: modular regimes and NEAT

Resume from branch `codex/modular-neat-integration-20261001`. Do not recreate
the merge or repeat a candidate already present in a campaign queue.

## Delivered

- The modular engine and corrected durable DOIN campaign histories are merged.
- `component_regime_config` and `regime_matrix` produce explicit branch/core
  arms without mutating donor configuration.
- `modular_proposal_strategy=neat` runs the existing parameters-as-genes NEAT
  reproduction against the modular conditional grammar.
- A declared default candidate is always the control in generation zero.
- Candidates are valid before enqueue, paired-seed fitness must be complete,
  state is atomic and digest-bound, and duplicate identities are reused.
- The NEAT path permits one to three paired seeds, never more.

This is hyperparameter optimization. It is not a neural NEAT head and must not
be described as one. No new model-quality measurement was made.

## Next independent lanes

1. Review and integrate this branch without rewriting Satoshi's source branches.
2. Finish feature-selection dossiers and selected branch membership. Do not
   launch the main financial pretraining campaign from partial selection.
3. On an already governed TRAIN/validation task, price one generation before
   running a matched NEAT-vs-random proposal comparison. Same candidate count,
   seeds, evaluator, data rows and objective; one seed for costing, at most
   three only if the comparison requires it.
4. Specify the separate `neural_neat_head` contract over frozen rank-three
   B1-C1 latents. Compare against a Keras head on identical cached bytes. Keep
   B2-C2/R3 Keras as the end-to-end adaptation arms.
5. A latent cache is eligible only for frozen producers and must bind row order,
   scaler, feature order, branch/core donor bytes, fusion, grid and config.

## Verification

```bash
CUDA_VISIBLE_DEVICES='' TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_modular_temporal.py tests/test_modular_pretrain.py \
  tests/test_modular_doin_m04.py tests/test_modular_flat_parity_m04.py \
  tests/test_modular_evaluator_r3.py tests/test_modular_neat_policy.py
```

Report scientific metrics only after independent checkpoint replay, always with
the same-row naive. Do not send any forecast that fails its declared naive to
the heuristic strategy.
