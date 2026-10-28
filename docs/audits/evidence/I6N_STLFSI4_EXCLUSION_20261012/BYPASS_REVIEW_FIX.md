# H72 point-in-time bypass review and correction

Date: 2026-10-12

Scope: static and executable review of the clean 19-feature H72 successor. No
fit, TEST row, service, or running H4 artifact was changed.

## Findings and disposition

| Finding | PRE | Correction | POST |
|---|---|---|---|
| The sequence CLI could still generate the legacy H4/H72 configuration when the clean-successor flag was omitted. | Reproduced | The implicit legacy CLI path is refused before reading inputs or writing output. | Regression passes. |
| A V3 context could point at a re-labelled V1 20-feature cell. | Reproduced | The optimizer requires every clean-plan fold to use the V3 context and clean experiment; the evaluator independently requires a V2 cell, the exact 19-feature manifest, and the sealed availability contract before candidate construction. | Mutation is refused before fitting. |
| Seven retained development features have unresolved availability/consumption evidence. | Confirmed | The gate lists them explicitly and its promotion mode refuses them. Historical-cell TEST admission invokes that mode before the existing global TEST prohibition. | Exactly seven unresolved features are retained for development and blocked from TEST. |

## Verification

- Focused correction suite: `44 passed`.
- Expanded optimizer/evaluator/cell/generator suite: `130 passed`, two unrelated
  protobuf deprecation warnings.
- Read-only production-config preflight: first frozen fold resolves as a V2
  cell with exactly 19 features and without STLFSI4.
- `git diff --check`: clean.

The H72 successor remains a development campaign. The seven unresolved source
clocks remain a mandatory gate before TEST.
