# CL01-CL07 return: the first real business slice runs

Orders: `docs/handoffs/SATOSHI_M5PHET_CLASSIFICATION_FIRST_2026_09_24.md` (4b55b693)
and M5PHET `docs/CLASSIFICATION_FIRST_DELIVERY.md` (f5ea67a).
Evidence: `docs/audits/evidence/CL01_CL07_20260924/`.

**English news relevance to EURUSD -- `related` / `unrelated` / `unclear` -- now runs
end to end through the installed M5PHET provider registry against real Laya weights,
and its answers are bit-for-bit those of the SDK called directly.**

Read the two results below as two different questions. The first says the wrapper
changed nothing. The second says the classifier is not good at this task on the only
labelled rows that exist today. Both are true at once.

## What ran, and on what

| Fact | Value | Attribution |
|---|---|---|
| Device | external RTX 5090, `GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8` | MEASURED in process |
| SDK | `laya` 0.3.11 at `1e28ac20c0896b1c37a744cd11f740eb98f8b178` | read from `direct_url.json` |
| Checkpoint | `convaiinnovations/laya`, 117 files, sealed `bd12df8877899246...` | hashed before and after load |
| Task | `news_relevance_eurusd.v1`, questions `d747653767...` | digest identical on both sides |
| Entry point | `m5phet.providers` -> `laya_news` | installed discovery, `refused: {}` |
| Runtime | M5PHET `satoshi/runtime-p02-20260924` merged with `f5ea67a` | 139 tests |
| Adapter | news-signal `satoshi/classification-first-20260924` | 49 tests |
| Admission | `crispdm-run -m 6G -t 900 -n laya-eurusd-pilot2` | fresh, external 5090, nothing preempted |

## CL01: parity, with no tolerance to adjust

`tools/direct_sdk_reference.py` imports `laya`, `torch` and the standard library, and
**nothing from `news_signal`**. It duplicates the state serialization, the question set
and the call settings by hand, because a witness that shared our code would only
reproduce our own mistakes. Its question digest matches the shipped task's exactly, so
the duplication is verified rather than assumed.

| Comparison | Result | Scope |
|---|---|---|
| Direct SDK vs registry path | **12 of 12 distinct inputs equal** | every exposed decision field: `type`, `choice`, the whole probability map including key order, `confidence` |
| Same wrapper, new process | **12 of 12 equal** | process reload, reported separately |
| SDK `predict_batch` vs `system_one` | 13 of 13 labels equal, **0 of 13 probability maps equal**, max delta 0.0218 | a property of the SDK, not of the wrapper |
| Batch vs permuted batch | 13 of 13 identical, delta 0.0 | order invariance holds |

The comparator carries no tolerance parameter, so there was nothing to widen after
seeing the results. It is exercised by a test that feeds it a 1e-4 nudge and a reordered
probability map and requires it to fail on both.

13 corpus files are 12 distinct inputs: one row is an exact duplicate of another, which
is the point of including it.

The upstream README describes batched results as differing "in the 4th decimal" on GPU.
Observed here: 2.18e-2, two orders of magnitude larger. The wrapper calls `system_one`
one state at a time, so parity is untouched; this is recorded as an upstream finding,
not repaired by us.

## CL02-CL07

| ID | What was exercised | Result |
|---|---|---|
| CL01 | real SDK direct vs real provider, same inputs and settings | 12/12 exactly equal, no fixture presented as a model |
| CL02 | relevant, unrelated, ambiguous, negated, duplicate, revised, instruction-like | 13 of 13 answered; the instruction-like item was classified, not obeyed; `execution_authorized` false everywhere |
| CL03 | wrong language, oversize body, over token budget, future receipt, out-of-scope asset, absent weights | 5 of 5 refused with their own typed reason before any forward pass; absent weights refuse `MODEL_NOT_FITTED` before loading |
| CL04 | restart, order, permutation, single vs batch | identities preserved; a result whose declared population does not match its input is refused; batch reported separately |
| CL05 | absent or broken provider, unsupported combination | the request names the provider it failed on; a registration failure is per provider; an undeclared combination refuses before load |
| CL06 | persisted result, restart, replay | 17 records over 16 identities: **11 actionable, 5 retained refusals**, 1 revision linked to what it supersedes, 0 integrity failures, 0 broker calls, `inference_performed: false` |
| CL07 | labels sealed before scoring | sealed corpus `SEAL.json`; **macro-F1 0.3333**, 6 of 13 correct, coverage 1.0 |

CL07 in full: `related` precision 5/8 recall 5/7; `unrelated` precision 1/1 recall 1/5;
`unclear` 0 of 1. The model answers `related` or `unclear` for most items a reader would
call irrelevant. **This is a smoke test.** The labels are author-written, the bank is 13
rows, and it is sealed by digest so a later edit is visible -- but no sample this size
measures business quality, and no power claim is made from it. The direct SDK and the
wrapper produce identical labels on the same rows, which is why the same number appears
for both: it is the model's number, not the adapter's.

## Cost, measured

| Quantity | Value |
|---|---|
| Cold load | 5.18 s |
| First inference | 0.680 s |
| Warm inference | median **10.8 ms**, min 10.6 ms |
| Whole corpus, 18 records | 6.02 s wall, 6.29 s CPU |
| Peak VRAM | 2.47 GB allocated / 2.67 GB reserved, of 33.7 GB |
| Peak RSS | 3.39 GB |
| Refusals | 5, none of which reached the model |

Well inside the 600 CPU s / 900 wall s allocation; the second admission was used because
the first run exposed defects, and the first run's artefacts are kept under
`attempt_1_superseded/` rather than deleted.

## Defects found and repaired

1. **The GPU identity check could never pass.** `CUDA_VISIBLE_DEVICES` takes `GPU-<uuid>`;
   `torch.cuda.get_device_properties().uuid` prints the bare uuid. Compared literally, the
   adapter would have refused every real GPU with `GPU_UUID_NOT_OBSERVED` -- a refusal
   standing in for a check nobody had run. `same_gpu` normalises the prefix; its test pins
   the form torch actually prints, and the receipt now records the observed uuid as MEASURED.
2. **A refusal reported the runtime's observation and hid the provider's reason.** The
   envelope said coverage was not established; `LANGUAGE_NOT_VALIDATED` lived one level
   down. The receipt now carries both.
3. **A retained refusal counted as an actionable event.** CL02 requires failures to be
   kept; CL06 counts decisions. Storing both in one place let the first inflate the
   second -- 16 actionable events where there were 11. Refusals are now counted apart, and
   `replay_v2.json` is that count, re-derived on a different host from the stored bytes.
4. **The parity comparator keyed rows by `event_id`.** A revision shares its original's
   identifier, so one of the two rows was silently dropped and 12 comparisons were reported
   as 11. It keys by input identity now, which is what "compare by input identity" meant.

## What this does not establish

No live news feed. No calibration: the checkpoint itself warns that it ships invalid
temperatures, which is recorded in the run and is why `confidence` travels as
uncalibrated. No trading claim, no broker call, no order of any kind. No statement that
the classifier is fit for the EURUSD relevance task -- the opposite, on the evidence
available. No new training, and the doctoral numbers are untouched: R0 0.371174,
R1 0.374584, R2 0.368596 stand exactly as returned at `e36c1c6d`.

## Release status, kept apart

* `origin/master@6165089` ships the design documents and **does not contain** the
  `d9ffc4d` runtime.
* `satoshi/runtime-p02-20260924` now carries the runtime, the evidence layer and the
  classification contract, merged without losing either branch or the owner's README
  edits. 139 tests.
* The real pilot is the run recorded here. A passing contract test is neither a release
  nor a measured model result.

## Other lanes

Untouched by this block and still open: prospective news capture, the economic calendar
CAL01-CAL12, MT5 demo and Alpaca paper interfaces, the forecasting, hierarchical-regime,
causal and policy adapters. The financial lane remains blocked on the operator-declared
availability contract (HTTP 422). None of these are marked done by contract tests.

## Commits

| Repository | Branch | Head |
|---|---|---|
| news-signal | `satoshi/classification-first-20260924` | `bb281cc` |
| M5PHET | `satoshi/runtime-p02-20260924` | `3e36396` |
| predictor | `satoshi/rp132-rp134-20260923` | this return |
