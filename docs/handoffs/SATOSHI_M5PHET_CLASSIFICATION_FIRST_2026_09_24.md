# Execute M5PHET's first real business slice; domain plugins in parallel

## Owner clarification: the same question-driven experience as Laya

M5PHET's public product is **data/context + natural-language questions -> typed
answers across the five ML families**. A plugin catalog, five example scripts,
a generated config, or requiring users to know tensor/plugin details does not
satisfy this requirement. Natural language is not a later optional convenience.
The explicit task API stays underneath and remains available to advanced callers.

Integrate M5PHET's `docs/NATURAL_LANGUAGE_INTERFACE.md` (NL01-NL08) from
`musashi/classification-first-20260924`. Complete the real Laya slice through
this question-first public path; reuse its native question handling, without
waiting for or interposing a generalized language interpreter. In parallel,
implement a question-to-forecast path that returns actual specialized-engine
results, not merely a TaskSpec. The same input/answer experience extends to the
other assigned family engines. Interpretation and validation may differ inside
providers; users should not have to assemble those internals themselves.

Preserve current pilot installations/jobs. No new repository, mandatory remote
language service, unbounded fitting or replacement of ongoing scientific work.
The detailed contract distinguishes valid profile defaults, meaningful ambiguity,
data availability, model readiness, and provider results from generated prose.
Use existing provider/client infrastructure, not another agent framework.

Owner instruction: do not stop the M5PHET lane behind doctoral closure. Satoshi
implements, Musashi audits. This supplements existing RP147-RP151 and INT orders;
it is not another research stage or a reset of any compute allocation.

## First deliverable, highest priority

**Interpreter deployment update:** reuse existing supported Hermes/OpenCode
language access first; the owner names inexpensive DeepSeek Flash as a candidate.
Resolve the actual configured model/endpoint and entitlement without exposing
secrets. A small local pretrained instruction model on an admitted coordinator
is an alternative behind the same contract, not a prerequisite or a new LLM
training project. Compare semantic task accuracy, refusals, cost and latency on
NL02-NL07. Disable tools/code execution for interpretation; validate structured
task proposals. Reuse retained tasks rather than reinterpreting every tick.
The existing Alpaca broker integration is not a language backend; inspect any
separate language installation by that name before assigning that role. Full
deployment constraints are in M5PHET's `NATURAL_LANGUAGE_INTERFACE.md`.

**English news relevance to EURUSD**, through the installed M5PHET provider
registry and real Laya, with direct SDK parity. Not three half-built classifiers,
not a buy/sell classifier and not another schema-only return. Existing event/tone
outputs may remain, but cannot delay this slice. A returned uncertainty field
does not imply domain calibration.

The implementation/acceptance contract is in M5PHET:
`docs/CLASSIFICATION_FIRST_DELIVERY.md`, branch
`musashi/classification-first-20260924` at published `f5ea67a`. Integrate it with the runtime branch
without losing either branch's changes or the owner's README edits. At review,
`origin/master@6165089` does not contain `d9ffc4d`'s runtime: release status and
method state must distinguish shipped master, implemented branch and real pilot.

CL01-CL07 require installed entry-point discovery, same-input real-SDK parity,
typed refusal, restart/batch identity, persisted shadow output and separately
labeled domain evaluation. The native comparison cannot call our wrapper.
Retain SDK numeric values without extra rounding/normalization. Use the pinned
SDK/checkpoint/config/device, record derived compatibility files if upstream
modifies a tokenizer during load, and do not silently upgrade main.

Refresh the already-running isolated setup on the workers, without duplicate
downloads/installers. Finish the existing 600 CPU s / 900 wall s pilot after
fresh admission, preferably on external5090. A functioning local recorded-news
example does not need feed credentials; real prospective collection does. No
claims of a live feed, model quality or trading profit from this smoke alone.
After a successful pilot, integrate with the existing news receipt/queue path,
document a working command and persist/replay actual structured results. No
broker action in the classification acceptance test.

## Other lanes: assign now, do not wait on the GPU

| Owner repo | Implementation order | Important boundary |
|---|---|---|
| news-signal | Laya provider and relevance application above | Existing `LayaBackend` -> real Registry, not builder-only integration |
| predictor + prediction_provider | Native forecast/config/checkpoint adapter; serving in its owner repo | Preserve branch/feature order, transforms, all target/horizon outputs and actual uncertainty method |
| feature-eng | Hierarchical-regime provider reusing `regime_analysis.py` behavior | Train-only scaler/PCA/tree; frozen new-point assignment; GMM/threshold labels are not a hierarchy |
| causal-inference | Repair packaging then a real economic-event provider using existing causal libraries | Preserve untracked local work; actual/consensus/vintage data and identified estimand, not return-derived surprise |
| agent-multi + gym-fx | Existing fitted policy adapter through native plugin, with action/state contract | LTS risk/execution downstream; DOIN is not this engine |
| Existing DOIN domain/plugin owners | Candidate/evaluator integration per supported family | Cross-cutting optimization; no new scheduler or implicit fits |

Use available Hermes/subagents in separate worktrees with a single integrator.
Assign classification its own worker; do not spend that worker's whole turn on
the numerical closure below. Keep independent CPU implementation moving while
weights install or the GPU is busy. Native-versus-wrapper parity is mandatory
for each family, not a reason to make all five prerequisites of classification.

The shared part is typed task validation, temporal availability, provider
lifecycle, evidence and optional integrations. Laya's question-conditioned text
encoder is NOT a universal temporal representation. Do not force all numeric
time series or causal tasks through text serialization. Replacing a neural head
requires a separate trained/evaluated variant; wrapping an existing full engine
does not. See the source-level reuse map in M5PHET's delivery contract.

Each adapter lives with the engine's owner, registers `m5phet.providers` and
delegates to native plugins. Keep distinct environments for conflicting `app`
packages. Declare actual supported parameter/output combinations; do not hide
branching/multi-horizon options or advertise unsupported joint uncertainty.
Local records and optional DuckDB remain available without data-gov or DOIN.
The optional governed path must never downgrade after a refusal.

## Discovery findings that must shape implementation

- `feature-eng@df7ea5b1`, `regime_analysis.py:143-176`: whole-frame scaler/PCA,
  sampled Ward tree, KNN assignment, then forward-return diagnostics. Reuse the
  algorithmic parts under temporal fits, not that exploratory full-data result.
- `causal-inference@1bd76694`: `setup.py` and loader still identify RL optimizer
  plugins. Causal research exists at root, but is not a working packaged service.
  The checkout is dirty. `causal_regime_analysis_v2.py`,
  `nfp_event_response_poc.py` and others are local untracked work; never overwrite,
  reset or blanket-stage them. Preserve a digest/source inventory before reuse.
- In the local NFP prototype, surprise is inferred from the outcome and timezone
  chosen by market response. Do not use either for the economic-calendar study.
  In the local EconML script, treatment is `bb_position`, not economic surprise;
  five-fold CV and averaged per-row CI endpoints do not establish the required
  temporal causal identification or uncertainty of an aggregate effect.
- `agent-multi@890fdc7a` registers PPO/DQN/SAC and gym-fx through native entry
  points. It is the chosen policy owner, not the older `rl-optimizer` template.

These are source observations, not claims that new engines have run. The full
causal task, interaction/compound response requirements and acceptance cases are
in the M5PHET contract. Repair the narrow causal package; do not launch a broad
new causal campaign before the actual data and identification contract exists.

## RP158 closure: bounded CPU maintenance, not a model lane

Reviewed `e36c1c6dd05340f7c3a1b53a575cad69625a5580`. Canonical fields are now
schema-defined, the authenticated horizon drives dispatch, and the new count,
naive-NaN and literal-True guards work in the reviewed cases. Existing focused
tests: **30 passed, 6 deselected**, CPU. Retained population metrics match RP157
exactly, so preserve R0 0.371174 / R1 0.374584 / R2 0.368596 and their development
scope. No independent model inference or live authority inspection in this audit.

One remaining **medium severity acceptance gap** in the already-assigned matrix:
`validate_children` only validates elements WHEN they are integers; missing or
string elements pass. Negative MAE and a finite but incorrect skill also pass.
The retained valid case and four one-factor mutations were executed against the
real validator, with expected counts/checkpoints supplied. See
`docs/audits/evidence/RP158_MUSASHI/{probe.py,results.json}`. This is not evidence
of incorrect real scores, nor a new need to train/replay.

Complete mandatory typed elements, nonnegative losses and derived-skill/zero-
baseline semantics. Validate exact populations and retained source identity in
the same path. Finish the existing completion matrix as a whole. The requested
pure closure is STILL absent: `score_contrast` launches subprocess inference
unconditionally. Extract/use validation and aggregation over retained child
records, with a test that fails if any inference subprocess is called. Satoshi's
reported extra nine-checkpoint pass was unnecessary for these repairs. Do not
repeat it again; do not label such a pass as no inference. Missing evidence
should be scoped and named, not silently recomputed or promoted.

## Return, without pausing at each step

Deliver real classification command/output, SDK equality counts, latency/memory,
domain quality scope, and a per-provider state with commits and native test
evidence. Include news/calendar/demo-paper/financial prerequisites from the
existing lane table; do not mark them done based on M5PHET contract tests.
The first response should show a usable classification path or an actual measured
setup failure plus independently completed CPU work, not another design-only
milestone. Existing risk, resource and held-out-data constraints stay in force;
no real capital, blind fitting, service restart or new training budget is granted.
