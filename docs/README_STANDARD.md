# Repository README standard

Every project homepage must tell a new reader what is implemented, how to
try it, how to interpret its outputs and what remains unverified. A proposal,
an adapter, a tested integration and a production deployment are different
states. Link exact research snapshots when they are not on the default branch.

## Required coverage

| Topic | What a reader must be able to find |
|---|---|
| Purpose and status | Actual product, intended audience, maintained/research/legacy status |
| Ownership | What this repository does and which sibling owns adjacent work |
| Architecture | Main components, installed plugin groups and external dependencies |
| Requirements | Declared versus tested Python/runtime versions; separate environments |
| Installation | Commands for this directory, not an unrelated sibling package |
| Smallest example | Bounded local fixture; exact expected artifacts; no accidental live work |
| Agent use | A self-contained task prompt, relevant instructions, scope and evidence to return |
| Configuration | Meaning, precedence, example paths and activation behavior |
| Tests | Relevant commands, optional dependencies, known gaps and dated evidence |
| Outputs and reproducibility | Effective config, data/model identities, outcome and result location |
| Operations | State, backups, errors, resource limits and recovery where applicable |
| Limitations | Unsupported behavior and claims not established by tests |
| Related repositories | Working GitHub links, explicit private-access requirements |
| License and contribution | Actual license or its absence; dataset terms separate; useful issue format |

For an archival or interface-only repository, explain why a runtime section is
not applicable rather than inventing a runnable example. Documentation must
not command an agent to read a nonexistent AGENTS.md. For web-only agents,
provide GitHub URLs and commits rather than filesystem paths.

## Acceptance before publication

1. Read the corresponding code/config for every changed factual claim.
2. Check relative links and the real remote default branch.
3. Run changed examples/tests with disposable outputs and bounded resources.
4. Check screenshots against the actual UI; label fixture data as demonstration.
5. State test scope honestly: a README pass is not a new complete code audit.
6. Preserve contributor work, experiment artifacts and deployed services.

## 2026-09-14 follow-up

The 14-repository publication set was checked for agent-use sections. Missing
sections were added for predictor, financial-data, preprocessor, doin-plugins,
doin-domains, agent-multi, gym-fx, lts and prediction_provider. The existing
sections in feature-eng, feature-extractor, heuristic-strategy, synthetic-datagen
and timeseries-gan remain. The data-gov README was rewritten with executable
integration examples and a separately tested warehouse implementation.

Additional factual fixes: financial-data's separately packaged lake is no
longer described as having no tests/package; doin-domains distinguishes its
commented future entry points from registered plugins. Its interface test
passed (14 tests). This follow-up does not claim to have rerun every training
pipeline or full suite across all 14 repositories.
