# Adequacy S1-S4 review and T1-T4 continuation

Reviewed: predictor `bfe2545`, S1-S4 return, design/models/runner and recorded pilot verification.
Scope: descriptive model/context adequacy, not representation selection or trading.

## Findings, before choosing a resource option

1. **High: observed-increment oracle is not the conditional noise floor.**
   `tools/df_adequacy_models.py:task_series` uses next_clean - current_clean, but the target
   is next_observed - current_observed. With a known clean recurrence and observed current
   value, the informed predictor is next_clean - current_observed. Its residual is next
   noise, not next_noise - current_noise. The current oracle can overstate the unavoidable
   error and make the 1.1-times-oracle adequacy criterion too permissive.
   Independent deterministic probe: past noise 3, future noise 0 -> old oracle error 3,
   corrected conditional oracle error 8.9e-16. This is a unit-level mathematical example,
   not a new campaign or a claim that a finite sample loss is a universal lower bound.
2. **High: window effects are confounded with different training/validation dates.**
   `df_adequacy_design.boundaries` at L=768 gives W4 train [790,1558), validation [1563,1659);
   W256 train [286,1054), validation [1311,1407). Comparing them changes recency, phase and
   data realization as well as context. Equal rows across models at one W does not solve this.
   CNN depth also changes from 2 to 8 layers as W goes 4 to 256: it is not a pure context
   ablation. This can be a joint architecture/context experiment only when named as such.
3. **High for independence claims: the alleged untouched test was already scored.**
   The committed `S3_ADEQUACY_V1_VERIFY.json` has test rows 1664..2046 and test MAEs for
   all three cost-pilot children. Do not call those rows untouched or use test-best W to
   select the context while presenting its same error as independent validation.
4. **Reporting:** the verification record says ridge FITTED, not all three UNDERFIT.
   `exact_need_cpu_seconds` is an extrapolation, not an exact resource requirement.
   Software tests passed independently: **12 passed, 8 Keras deprecation warnings, 20.73 s**.
   The full 1283-test suite was not rerun; no fresh production warehouse query here.

Reproducer: `reproduce_adequacy_s_review_2026_09_18.py --repo <reviewed-checkout>`.
No historical campaign is invalidated wholesale. Preserve the pilot as measured cost and
descriptive outcomes, with the incorrect oracle interpretation and test-status statements corrected.

## Decision

Do NOT choose B: retaining only L=768 for neural learners removes their learning curves, a
central question the owner required. Prefer the complete 216-cell factorial after corrections.
Authorize a successor resource ceiling of **14400 aggregate CPU seconds**, counting its pilots,
failed attempts and all workers; record the prior pilot's 31 s separately in cumulative history.
This is a disclosed pre-execution resource amendment, not retrospective permission to exceed
the prior 7200-second limit. Execution remains conditional on T1/T2 and the cost check below.

## T1 - Correct the ML estimands and comparison geometry, tests first

Freeze the reproduced examples before implementation. Correct the observed oracle to use the
current observation and true past clean recurrence. Keep truth confined to its diagnostic role.
Separate expected Gaussian noise-only MAE (sigma*sqrt(2/pi), under the declared generator)
from its realized sample MAE; either may fluctuate and neither is a samplewise lower bound.
Test current-only noise, future-only noise, zero noise, nonfinite values and label identity.
Retain the old oracle arrays for history; do not reuse the old adequacy classification.

Use shared decision-row anchors across W and model at each L, with separation sufficient for
the largest allowed support. For L curves use nested training histories ending at the same
cutoff. Derive the geometry from consumed rows and label horizon, and assert equality of
train/validation/test ids where comparison requires it. Test validity before sealing.

For the main context ablation keep CNN graph/parameterization fixed across W, with enough
receptive field for W256; report padding and effective support. Retain the old variable-depth
configuration only as explicitly separate evidence, not another unbudgeted factorial. Ridge
parameter count necessarily varies with lag count; state this limitation. Do not claim a
context-only causal effect across architectures.

## T2 - Test exposure, input identity and diagnosis

Correct the historical exposure ledger: cost pilot observed test outcomes. It is permissible
to keep those rows for a disclosed DEVELOPMENT diagnostic, not an untouched confirmation.
Select any proposed model/context from inner validation only; test tables remain descriptive.
No repeated choices from the best test row; independent final confirmation remains future work.
New cost pilots must not score or publish test labels/losses: prove by making the test accessor
fail while the cost-pilot path still succeeds. Never use test scores for cost-model choices.

Bind consumed clean/observed arrays to the frozen design by recomputing their actual digest
with the bank's existing scheme, not copying UNIT metadata. Validate identities, geometry and
finite values before fitting. Change one array byte with unchanged metadata: refusal before
training is required. Reuse existing governed evidence paths; no new infrastructure service.

Keep FITTED/UNDERFIT heuristics distinct from scientific adequacy. Do not label optimization
failure from a single loss trend alone. Record optimizer iterations and stop reason, budget vs
early stopping, restored checkpoint and prediction parity. Tests must exercise actual learners
and the production child path; fixed fixtures alone do not establish model behavior.

## T3 - Cost pilot and full governed diagnostic

Freeze the corrected design, comparison table and metadata before any new outcomes. Preserve
two units, three tasks, three models, four contexts, three lengths (216 cells, single model seed)
if corrected geometry admits them. Single-seed neural results remain provisional; do not call
them robust architectural rankings. No tuning or additional models after outcomes.

Measure cost at representative short/long contexts under the corrected fixed CNN graph.
Separate startup/evaluation overhead from optimizer work. Report projection and assumptions,
not exact need. Require projected remaining total plus **25% headroom** to fit the remaining
14400-second ceiling before dispatch. If it fits, run ALL declared cells without another owner
permission. If not, deliver exact measured costs and a scientific trade-off proposal; do not
silently delete learning-curve cells. All children enforce remaining aggregate and local limits.

Distribute independent children across healthy workers under their own governed identities,
memory limits and a global CPU ledger. CPU only; do not occupy GPUs or restart services. Failed
cells are preserved as incomplete; no tuning-until-pass, dropping failed cells or reporting
partial completion as full. A bounded failure completes its disposition, not its scientific claim.

## T4 - Independent evidence and understandable results

Recompute model, baseline and corrected oracle losses from stored arrays; bind label/row
identities to consumed data. Reconcile parent, accounting and live warehouse by content.
Report task-wise curves across W and L, actual training lengths, context in periods, neural
updates and uncertainty limits. Distinguish model inadequacy, context limitations, noise and
optimization evidence; avoid diagnosing them solely from a test-error threshold.
Update 12C/12D/09_ADOPCION and the correction beside historical claims. Preserve prior designs,
pilot records and costs. RL/weekly-business design remains required, not marked executed.

Finish all blocks without requesting stepwise authorization. No reserves, financial scoring,
RL training, GPU, broad architecture search or unrelated warehouse maintenance. Ending:
`ADEQUACY_MATCHED_CONTEXT_CORRECTED_ORACLE_REVIEW`, with full/partial/budget-limited execution
reported explicitly. A negative result is acceptable; an unsupported adequacy claim is not.
