# CL08-CL13 review: retain the question milestone, finish deployability

Candidates: news-signal `e40098b`, M5PHET `3f4a833`, feature-eng `1a0402e`,
predictor `1e9d994f`. Reviewer Musashi; implementation owner Satoshi.

## Result and accepted progress

Independent reduction of retained native/wrapper answers confirms **12/12 exact
distinct inputs for each of two user-authored questions**. The public `ask` path
exists. These are 13 rows including a duplicate, not 13 independent examples.
Rate-cut answers are 11 no_bearing, 1 more_likely, 1 less_likely. Economy answers
are 9 euro_area, 3 united_states, 1 elsewhere. Neither new task has a labelled
quality evaluation: two sensible examples do not establish factual accuracy, and
the frequency of no_bearing alone does not establish an inferential failure.

Previous traversal, refusal-to-success, different-task and tampered-byte cases
have regression coverage and pass under the proposed runtime. The pure ECL
closure now derives counts from the retained run instead of the scoring file;
the declared inability to authenticate metric values is an appropriate limit.
No new doctoral measurement, inference or training was needed by this review.

## Findings with executed counterexamples

**F1 / High: clean installation still rejects the new public feature.**
`news-signal/pyproject.toml:11` pins `e752ed1`; task-kind compatibility was added
only in `3f4a833` (`M5PHET/src/m5phet/runtime.py:370`). A newly created isolated
venv, normal pip resolution, no editable install and no PYTHONPATH substitution
imports correctly but returns REFUSED_WITH_INPUT for the user question, before
the fixture backend is called. With the proposed runtime the same request works.
The retained installation evidence tests absence of weights, not successful ask.

**F2 / High: a valid record under the wrong key is silently reused.**
`news_signal/shadow.py:103,133,150` checks the record's own content digest, not
the relation between requested evaluation, record identity and pathname. Copying
the intact record for question B onto question A's store path produces DUPLICATE
for A, returns B's stored digest, and replay reports zero integrity failures.
This is a fixture file substitution, not a claim that real records were changed.

**F3 / Medium: option text can still be silently truncated.**
`news_signal/backends.py:54` counts each option only AFTER slicing to 48 tokens.
Consequently fits=true does not imply the original option fits. A valid public
question with a long criterion passes; the real backend boundary calls its agent,
but the pinned SDK sequence drops the criterion's tail. The probe uses a named
deterministic tokenizer, the unmodified upstream sequence-builder functions
loaded from their AST, and no model. Source pinned and hashed in results:
[Laya common.py](https://github.com/NandhaKishorM/laya/blob/1e28ac20c0896b1c37a744cd11f740eb98f8b178/laya/common.py#L64).
This does not show truncation occurred in the measured short preset examples;
it limits the general claim that every silent cut is prevented.

**F4 / High, ML: publication and local receipt are confused in surprise.**
`feature-eng/app/economic_calendar.py:213` freezes consensus before local receipt
of the actual, not before its publication. Fixture: prior consensus 2.5; actual
2.9 published 13:30, received 13:45; updated consensus 2.9 published 13:34 and
received 13:35. The current output is surprise 0.0 rather than release surprise
0.4. A publication-based expectation and the trading system's delayed availability
are different boundaries; preserve both. This is not evidence of price-derived
surprise or corruption in an actual financial campaign.

**F5 / High, ML: availability refusal is not on the consuming path.**
`economic_calendar.py:245` returns unusable for UNKNOWN, but `add`/`surprise`
accept the same row carrying historical_availability=UNKNOWN and produce a
numeric surprise. Calling a separate helper in a test does not enforce the rule.

**F6 / Medium: a public calendar read can change history without its digest.**
`economic_calendar.py:123,142` exposes retained dictionaries. Mutating the actual
in a row returned by known_at changes the next view from 2.9 to 99, while the
vintage identity stays identical. No private state access is needed.

Executable evidence: `docs/audits/evidence/CL08_CL13_MUSASHI_20260924/`:
`probe.py`, `results.json`, and `declared_install.json`. All malformed writes
occurred under TemporaryDirectory. No current GPU, service, feed, warehouse or
account was inspected, and no real-weight replay or trade was executed.

## CL14-CL19: next delivery, with separate owners and dependency scope

Inspect active jobs, unpublished work and remaining allocations before dispatch.
Use available Hermes/subagents with disjoint worktrees; Satoshi integrates. Do
not close the round after only the repair lane while independent work remains.

| ID | Owner / lane | Required next deliverable |
|---|---|---|
| CL14 | news-signal + M5PHET release | Pin a published runtime supporting ask. Clean-install and test a successful installed entry-point ask with controlled backend, not only --print-question or missing-weight refusal. Then reuse existing real-weight parity evidence where code equivalence permits. |
| CL15 | news-signal persistence | Bind pathname/store key, source/evaluation identity and content before reuse or replay. Test intact wrong-record substitution, changed task/model/order and malformed JSON; quarantine without serving another task. Preserve every quarantine version. Include same-key concurrent writers, not only different event IDs. |
| CL16 | feature-eng calendar | Correct publication/receipt boundaries; enforce UNKNOWN at ingestion/consumption; make public reads immutable or detached. Add duplicate/tied sequence tests and ensure reordered delivery cannot change a value under one vintage. Keep original release surprise and later revision information named separately. |
| CL17 | predictor/prediction_provider + M5PHET interpreter | Deliver the second question-first vertical slice: user question and dataset description -> validated multi-horizon task -> existing native forecast engine -> typed output. Reuse configured interpreter access or deterministic supported parsing; never silently map unsupported prose to a preset. Compare a bounded existing DEV checkpoint batch with native outputs on the same tensors. No full-model retraining or rereading reserved test. |
| CL18 | news collector + quality + broker code | Implement the collector adapter, durable queue and replay/revision tests now, independently of real-feed entitlement. Prepare the predeclared labelled relevance evaluation and factual-economy controls. Build MT5 demo/Alpaca paper adapters and refusal/recovery tests; actual connections require documented access and existing risk mandate, not guessed credentials. |
| CL19 | integrator | Publish runnable commands, scoped acceptance, source revisions, exact tests and costs, with one status row for ALL lanes including hierarchy/causal/policy/DOIN integration. Keep a machine-readable ready/held queue and name a concrete missing object for each actual external hold. |

CL14/15 do not block CL16/17/18 implementation. Calendar use in a scientific
model does depend on CL16. A broken real-feed entitlement does not prevent
collector implementation, recorded-input replay, interpreter or native adapters.
Other family owners remain feature-eng for hierarchy, causal-inference for causal
engines (preserve untracked exploratory files), agent-multi/gym-fx for policy.
DOIN is transversal and optional, not the policy engine. No new repository.

For F3, test untruncated lengths at each SDK limit, individual options included,
and differential sequence assembly against the pinned upstream implementation.
Use the actual tokenizer in an existing isolated environment without GPU/model
loading where available. Avoid a hand-maintained approximation that silently
changes meaning; preserve the declared upstream normalization of special tokens.

For CL17, training preprocessing remains in its native owner: the interpreter
selects an explicit task/schema, never invents data columns, scalers, checkpoint
compatibility or uncertainty calibration. A supported request either reaches the
native engine with recorded inputs or asks for the exact missing field/refuses.
MCP/skills may expose that bounded interface; RAG remains optional and no hold.

For CL18 quality, retain the original relevance objective. Factual attribution is
a separate task with its own labels, not a replacement chosen because two examples
look good. Include unrelated stories, no-economy/ambiguous cases and controlled
asset-metadata changes with unchanged story text. Freeze annotation, deduplication,
split and selection before evaluation. Report macro-F1, per-class counts, majority
baseline and coverage; small uncalibrated confidence is not proof of being wrong.
No inference from relevance or rate-cut labels to trading profitability.

Prefer the external 5090 for eligible bounded model work after fresh admission,
use existing remaining budgets and report their ledger. Do not preempt, duplicate
completed work, reset allocations or occupy a GPU just to claim utilization.
CPU implementation and annotation must continue while GPU work runs. This review
authorizes no real-capital execution and introduces no new training campaign.

## Verification

Proposed-runtime environment: news-signal 90 passed; M5PHET 141 passed/1 skipped
(DuckDB absent); calendar 25 passed; ECL CPU gates 54 passed/6 deselected (data/model
dependent). The source-pin regression test with e752ed1 fails as reproduced;
the separate fresh dependency-resolved installation produces the same refusal.
These are deliberately different environments, not one combined green total.
Retained two-question parity was recomputed independently. Current live state,
real tokenizer execution and live-governance closure were not checked here.
