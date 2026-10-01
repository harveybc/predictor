# Modular regimes and NEAT proposal integration

| Requirement | Structural evidence | Behavioral evidence | Status |
|---|---|---|---|
| Independent branch/core regimes | `component_regime_config`, `regime_matrix` | exact per-branch map, donor requirement, no input mutation | VERIFIED |
| B1 core-pretraining input | `pretrain_components` rebuilds branches R1 before fusion materialization | donor weights fixed and reload parity in staged-pretraining suite | VERIFIED |
| Modular NEAT candidates | `ModularNeatProposalPolicy` uses the existing parameters-as-genes reproduction | deterministic unique candidates all pass `validate_flat` and `from_flat` | VERIFIED |
| Durable generations | optimizer writes `NEAT_STATE.json` atomically with task/search identities | state round trip resumes the same next population | VERIFIED |
| Complete fitness | one mean objective requires every declared paired seed in `verified` state | missing fitness rejects before reproduction | VERIFIED |
| No unnecessary reruns | campaign candidate identity remains authoritative | repeated candidates are reused; integration performs at most the declared cells | VERIFIED |
| Neural NEAT head | separate future head plugin and frozen latent contract | no implementation or quality measurement | NOT_IMPLEMENTED |
| Scientific advantage | paired real-data comparison against random/DEAP | no new measurement in this change | NOT_MEASURED |

All tests use synthetic/fake objective values for mechanics. They make no model
quality, financial utility, or live-trading claim.
