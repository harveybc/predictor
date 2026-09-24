# CL14-CL19 review and CL20-CL24 execution orders

Written for Satoshi (implementation) and the owner (status). 2026-09-24.

## Findings, reproduced independently

Candidates: news-signal `317dd35`, feature-eng `d081d0f`, M5PHET
`3f4a833`, predictor `c4b52c06`. References below are candidate source lines.
Evidence: [CPU probe and results](../evidence/CL14_CL19_MUSASHI_20260924/).
These are temporary fixture failures, not evidence that retained production
results were corrupted. No GPU, feed, broker, live database or training used.

Reproduce using separate candidate checkouts at the revisions above and a clean
venv; normal pip resolution must provide the pinned M5PHET dependency. Set
`NEWS_CHECKOUT` and `FEATURE_CHECKOUT` to those checkouts, and `VENV` to the venv:

```bash
"$VENV/bin/pip" install "$NEWS_CHECKOUT[test]"
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 "$VENV/bin/python" \
  docs/audits/evidence/CL14_CL19_MUSASHI_20260924/probe.py \
  --news "$NEWS_CHECKOUT" --feature "$FEATURE_CHECKOUT" --out /tmp/cl14-results.json
```

The probe records observed behaviour, not a passing repaired implementation;
its exit status alone is not acceptance. Compare each observation and preserve
the positive paths as well as the deliberately failing cases.

1. **High: queue transitions validate corrupted content by re-signing it.**
   `news-signal/src/news_signal/collector.py:138`: `_transition` ignores failed
   integrity. A corrupted headline is initially excluded by `pending`, then
   `retry` makes it dispatchable with integrity OK. At line 132, reoffering
   that invalid entry also reports ACCEPTED despite losing exclusive creation:
   the returned digest is not on disk and no valid item is pending. At line 80,
   reading checks a self-hash but not path/content identity: intact B under A's
   path is served as DUPLICATE of A with no reported integrity failure.
2. **High: shadow recovery can claim a write that did not happen.**
   `news-signal/src/news_signal/shadow.py:223`: when identical bad bytes already
   exist in quarantine, `_quarantine` leaves the invalid active file in place.
   `put` ignores a second unsuccessful exclusive write. Through installed `ask`,
   first repair succeeds; repeating the same corruption reports REPLACED_INVALID
   while disk integrity remains FAILED and returned/stored digests differ.
3. **High, ML: publication boundary still selects by receipt order.**
   `feature-eng/app/economic_calendar.py:255` and `:276`: a newer consensus 2.7
   published at 13:20 arrives before older 2.5 published at 13:00. For actual 2.9,
   release surprise is 0.4, but latest-published consensus gives **0.2**. Further,
   `_published` at line 228 silently replaces missing publication with receipt;
   a post-release consensus can again turn release surprise into zero. A revision
   received before the delayed original is labelled release_actual, and the
   original becomes revised_actual. Reception is eligibility, not source chronology.
4. **Medium: unrelated source identifiers become revisions of one event.**
   `news-signal/src/news_signal/collector.py:119` and `:174`: two sources using
   the same event_id yield one event and a cross-source revision. Dedup keys
   include source through content, but revision linking and reporting do not.

## Accepted progress and limits

- A genuinely clean pip installation resolved M5PHET `3f4a833` from the declared
  pin. Installed `ask` returned SHADOW_ONLY with explicit NON_MODEL_FIXTURE;
  a new process recovered one intact record. The installation finding is closed.
- Recorded collection independently yields 11 events plus one revision;
  second pass adds nothing (13 duplicates). This is collection, not prospective
  capture, classification quality, or successful downstream acknowledgement.
- News suite: **117 passed, 1 skipped** (upstream differential dependency).
  Calendar focal suites: **37 passed**. New adversarial probe still exposes the
  findings above despite those green suites.
- Retained two-question real-Laya native parity is not withdrawn or remeasured.
  No new model-quality or doctoral measurement; no new financial inference.
- CL17 is **implementation not started**, not an external permission dependency.
  Existing feed/broker access limits real canaries, not adapter code or fixtures.

## CL20-CL24: concurrent implementation, not another global hold

Satoshi owns implementation. Freeze this probe unchanged as PRE, add behavioural
tests before repairs, retain POST and positive controls. Inspect current jobs,
leases, worktrees and remaining allocations before dispatch; never duplicate or
stop independent valid work. Use disjoint Hermes/subagent worktrees if available,
one integrator, and no credentials in prompts. These orders add no training budget.

### CL20: queue and shadow lifecycle

Validate schema, source/content/input/key/path identity and integrity on EVERY
public consume/transition path, not just `pending`. Corruption is retained and
refused or explicitly recovered from independently supplied valid input; never
re-hashed into truth by retry/fail/ack. Return success only for an actual durable
winner that reads back under the requested identity. Preserve receipt history;
if original receipt cannot be trusted, do not fabricate point-in-time availability.

Exercise offer/get/pending/retry/fail/ack, invalid and wrong-key records, repeated
identical quarantine bytes, competing repair/writers, restart and interrupted
publication. Do not let recovery delete another writer's valid replacement.
Namespace revisions and counts by source plus source event identity; cross-source
story linkage requires an explicit separate relationship, not an identifier match.

### CL21: calendar source chronology

Select release consensus by publication order among admissible records, retaining
receipt-based eligibility and the separately named available quantity. Missing
publication is UNKNOWN, not observed_at or scheduled event_time. Select original
ACTUAL and revisions by declared source lineage/chronology, not arrival order;
if original or ordering is unknowable, name the missing fact and retain the other
valid quantities. Define tied publication instants and conflicting source vintages.
Test all permutations of insertion order AND independently varied publication/
receipt order, late original, early revision, missing clock, ties and as-of views.
Do not use price movement to infer economic surprise. Keep consumer ML gates on
affected calendar outputs, not on unrelated forecasting/classification work.

### CL22: deliver the outstanding CL17 question-to-forecast slice now

Start this lane alongside CL20/21, not after their audit. Use existing owners
M5PHET and predictor/prediction_provider, no new generic forecasting framework.
Public input: data/reference, description and a user-authored natural-language
question. Bind interpreted target/horizon/frequency/output type to an explicitly
supported native task, validate it before engine execution, and expose the resolved
task in the receipt. Missing/ambiguous semantics require a typed clarification;
unsupported horizons, units or uncertainty must not silently become defaults.

Reuse a compatible retained trained DEV checkpoint and its real preprocessing,
input/output shapes, feature order, target transformation and native inference.
No ideal oracle, fixture, synthetic answer or new baseline may stand in for that
engine acceptance. An interpreter may select a task, never invent predictions.
Compare identical inputs against an independent native-engine invocation, including
output rows, horizons, scale and values under a declared numeric rule. Demonstrate
two supported paraphrases and negative/ambiguous requests. Test interpretation
separately from numerical engine parity. Do not claim probabilistic calibration
unless actual model outputs and calibration evidence support it.

No full training, heldout scoring or nine-checkpoint replay needed. First commit
the adapter and acceptance tests. If a real checkpoint/access prerequisite is
missing, name the exact artifact and finish everything independent of it; do not
describe an unwritten runner as an owner decision. Use configured interpreter
access or a clearly declared bounded supported grammar without claiming general
language understanding. No raw private-data upload or generated-code execution.

### CL23: recorded source through a real consumer, plus broker preparation

Connect recorded collection -> validated queue -> installed M5PHET provider ->
durable evaluation receipt -> acknowledgement tied to that event and evaluation.
The fixture integration must survive restart after result persistence but before
ack, provider refusal and corrupted retry without false success or lost records.
Keep fixture-labelled evidence distinct from real Laya. Reuse retained model
identity/parity and run only any necessary bounded acceptance within an existing
remaining allocation; external RTX5090 first after admission. No gratuitous GPU
load, new training or claim of live capture from recorded files.

In parallel, prepare the independently labelled quality-set protocol and existing
MT5 demo/Alpaca paper interfaces with deterministic rejection/recovery tests. Actual
feed capture and broker canaries still require documented entitlement, confirmed
demo/paper endpoint and existing risk mandate. Never use real-capital credentials.

### CL24: one integrated return and persistent queue

Report each lane as delivered, running, executable next, or missing a concrete
external object. Include CL22's actual user command and native parity, not only
validator counts. Preserve earlier accepted results and distinguish fixed cases
from untested capabilities. Do not stop after CL20/21 when independent CL22/23
implementation remains. Update both project states and the existing queue, with
current evidence rather than historical GPU snapshots. No new scientific result
means NO_NEW_MEASUREMENT; no new quality measurement means exactly that.
