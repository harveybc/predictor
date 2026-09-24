# Classification delivery: independent review and parallel continuation

Candidates: news-signal `bb281cc`, M5PHET `e752ed1`, predictor `8d9eba36`.
Reviewer: Musashi. Implementation owner: Satoshi. Date: 2026-09-24.

## Measured result and limits

Independent reduction of retained direct-SDK and wrapper JSON confirms exact
decision fields, including nested key order, on the same 12 distinct input
identities, both first process and restart: **12/12 + 12/12**. No GPU was used
by this review. This is retained-output parity, not fresh hardware verification
or an accuracy guarantee. Latency/memory remain producer measurements.

Independent confusion counts reproduce 6/13 correct, macro-F1 0.333333,
unrelated recall 1/5. Always-related scores 7/13 accuracy and macro-F1 0.233333:
the model is worse on accuracy but better on macro-F1 than that trivial control.
These are 13 author-labelled rows, only 12 distinct inputs; one repeated item
and a revision are useful contract tests, not independent accuracy samples.
Do not equate relevance with direction, expected return or permission to trade.

## Findings (reproduced unless explicitly source-only)

1. **High: external identifiers escape the shadow directory.**
   `news-signal/src/news_signal/shadow.py:22` joins raw `event_id` to the path.
   Through `classify_event` with the real Registry/provider and a labelled
   fixture backend, `../escaped` gives SHADOW_ONLY and writes outside the store.
   The reproduction is confined to a temporary parent; no real data was changed.
2. **High: result identity and integrity are not enforced on reuse.**
   `shadow.py:38-49`: same news rejected before availability, then classified
   successfully at receipt time, yields DUPLICATE and retains only the refusal.
   Changing from relevance to triage also returns the old task's stored record.
   A tampered existing record is reused as DUPLICATE; replay reports corruption
   but still counts it as actionable (`shadow.py:78`). No actual trades occur.
3. **High: declared installation cannot import the application.**
   `news-signal/pyproject.toml:11` pins M5PHET `bcd65b78`, which has no
   `m5phet.runtime`. Importing `news_signal.application` against that exact
   source fails with ModuleNotFoundError. Passing tests used explicitly installed
   `e752ed1`, not the dependency that a normal installation would resolve.
4. **Medium: retained closure can accept a rewritten population.**
   `predictor/tools/df_ecl_modular.py:1287` trusts counts in SCORING.json itself.
   In the producer's positive retained-run fixture, changing those counts AND
   all nine child populations to one window/30,816 elements still gives COMPLETE
   with summaries, without changing DESIGN.json or CONTRAST.json. The pure path
   does avoid inference, but is not an independent population/custody check.
   This does not show that the actual nine scores were modified.
5. **Product gap, source-inspected:** `news_signal/provider.py:135` obtains the
   question from static TASKS; the public CLI accepts `--task`, not user question
   content. Real Laya integration is delivered; NL01 is not. A fixed EURUSD task
   is a useful first slice, not completion of the approved question-first API.

Evidence: [probe](../evidence/CL_FIRST_MUSASHI_20260924/probe.py) and
[results](../evidence/CL_FIRST_MUSASHI_20260924/results.json). Fixture findings
exercise production paths without real weights; parity/quality use retained
real-weight responses. No live service, account or current GPU lease was checked.

## Next orders: CL08-CL13, concurrent not serial

Read current leases, unpublished work, outboxes and remaining allocations first.
Do not duplicate Satoshi's jobs or replay old experiment queues. Incorporate these
orders with existing assignments, not instead of them. Use disjoint worktrees
and available Hermes/subagents for CPU lanes; one integrator owns releases.

### CL08: deployable and safe application (news-signal owner)

Freeze the four news counterexamples before repair. Pin the published runtime
actually used; install in an isolated clean environment with normal dependency
resolution, without editable sibling/PYTHONPATH substitution. Verify entry-point
discovery, CLI and fresh-process store recovery there.

Use opaque validated storage keys and root containment; test absolute/traversal
IDs and symlink escapes. Separate source/event/revision identity from evaluation
identity (task/question, model/settings, as-of/policy and attempt). Retries retain
history; only identical evaluations may be idempotent. Validate existing content
before reuse. Corrupt/refused/obsolete revisions cannot enter successful-event
counts. Test concurrent writers, interrupted publication and recovery through
the public application. No receipt may point to a different persisted result.

### CL09: real question-first classification (M5PHET + news-signal owner)

Implement NL01 now: data/context + user-authored natural-language question and
answer schema -> native Laya question-conditioned prediction -> typed receipt.
Keep the named EURUSD task as a preset. Do not replace every novel question with
that preset or restrict capability to task-name aliases. Validate supported output
shapes and encoded input limits before inference, and include actual question,
criteria, serialization and settings in identity. No extra LLM is necessary for
native Laya questions. Use the same question/options in the independent SDK witness.

Acceptance: two genuinely different questions over identical news; changing
question/criteria changes identity and the actual SDK input; exact wrapper/native
parity for each supported schema, fresh-process retrieval, malformed-schema
refusal, and no broker action. Unsupported requests fail explicitly. Provide the
public command and example, not only a test calling a private method.

### CL10: diagnose relevance quality, not just wrapper equality

Before tuning, produce a per-error table with news, gold/rationale, prediction,
class probabilities, actual serialized question/options and tokenizer output.
Check truncation/budgets for state AND question/options in the pinned SDK, option
ordering and upstream-recommended inference settings. Do not infer the cause
from 13 rows or describe uncalibrated confidence as probability of correctness.

Separate duplicate/revision robustness tests from the quality population. Keep
this smoke set as development; define a held-out, source-identified relevance
set with annotation rules and ambiguity/adjudication before selecting prompts.
Compare direct Laya and wrapped Laya, majority baseline, per-class precision/
recall/F1, confusion, abstention/coverage, latency and cost. Freeze diagnostic
variants and selection rule before measuring; do not train new weights by default.
Use remaining declared pilot allocation after fresh admission, 5090 preferred;
no automatic budget reset, speculative sweep or repeating identical parity runs.
Quality remains shadow-only until its separate business criterion is met.

### CL11: application lanes must advance in parallel

While CL08-10 run: implement prospective collector/receipt-clock tests, calendar
CAL01-CAL12, and demo/paper broker adapters in their existing owners. Real capture
requires the actual documented feed entitlement; do not search broadly for secrets.
Use actual/consensus/vintage/receipt metadata for calendar semantics, never price
reaction as a substitute for event surprise. Actual broker canaries require the
existing risk mandate and verified demo/paper accounts; no real-capital orders.

Begin the second question-first vertical slice: natural-language multi-horizon
forecast request -> validated typed task -> predictor/prediction_provider native
adapter -> forecasts/uncertainty with their actual scope. Reuse configured
Hermes/OpenCode interpreter access if available; record its exact model identity.
No generated arbitrary code, invented horizons/data columns or new LLM training.
Compare against the native engine on identical tensors, checkpoint and population.
Implement other adapters in parallel: hierarchy in feature-eng, causal calendar in
causal-inference (preserve existing exploratory files), policy in agent-multi/gym-fx.
DOIN remains optional transversal optimization. No new repository is required.

### CL12: finish retained closure on CPU, no checkpoint pass

Authenticate the retained population/score manifest against the already accepted
preparation/run/terminal identity, not a self-reported sibling field. Require every
population and its typed count; test consistent count+child rewrites, omitted
populations and coherent metric rewrites with the old authority unchanged. Keep
descriptive local consistency distinct from authenticated closure. Recover the
historical design only from retained evidence with matching recorded digest; do
not rebuild datasets or train/infer to obtain it. Missing evidence stays explicitly
missing and never holds CL09-11. Existing developmental scores stay preserved.

### CL13: consolidated result, not another partial all-done statement

Return an evidence/status/next-action row for CL08-12 and every prior active lane.
Lead with user-visible capability and measured result, then quality/coverage and
unstarted work. Record actual CPU/GPU costs and current dispatchable queue. Keep
independent admitted GPU experiments moving; an idle device is not permission to
repeat completed work or launch an uncosted campaign. No routine permission is
needed to continue the already authorized assignments.

RAG stays optional. Documentation retrieval may inform task interpretation;
separate point-in-time market retrieval may affect an answer through declared
model inputs. It is not forbidden to influence answers, but must not fabricate
measurements, treat retrieved instructions as authority or access future evidence.
MCP exposes bounded tools; skills supply procedures. Neither replaces the model,
schema validation, temporal checks or execution policy.

## Verification scope

- news-signal: 50 passed with candidate runtime explicitly installed in isolated
  system-site-packages venv, no GPU. The first uninstalled check gave 49 passed/
  1 entry-point failure; installation fixed that environment issue, not the bad pin.
- M5PHET: 138 passed, 1 skipped (DuckDB unavailable).
- ECL retained gates: 45 passed, 6 explicitly deselected data/model-dependent tests.
  Initial broader run: 46 passed, 5 failures from absent torch; no training ran.
- Probe: retained parity and confusion plus four application/store cases and the
  retained-population rewrite above. No full dependency-clean deployment, GPU
  re-inference, live warehouse reconciliation or trading validation claimed.
