# Laya architecture and reuse decision

Owner asks to reuse the real open-source implementation, including considering a
fork. Scope: upstream source, pinned Hub JSON configurations, local adapter and
bounded CPU tests. No model weights downloaded, no model quality/latency measured,
no active experiment changed. Satoshi remains implementation owner.

## Versioned evidence

- Current news-signal dependency: Laya 0.3.11,
  `1e28ac20c0896b1c37a744cd11f740eb98f8b178`.
- Upstream main inspected: 0.3.18,
  `84ca7240348e6fb3b8145d2d9f6266d5125747d8`. This is a candidate, not an adopted
  upgrade. The browser's cached main README still showed 0.3.11; Git source and
  pyproject at the full commit resolve that discrepancy.
- Hub metadata/configs inspected by immutable revision (NOT weights):
  english `convaiinnovations/laya@cf7c54c0586eede67d827dfaab8cd2d2007273e9`;
  multilingual `convaiinnovations/laya-multilingual@af8d1acfcd010319dd7784409e5f2243363ece9e`;
  typed `convaiinnovations/laya-typed-decisions@1a793eb568e6718f15941d08f85432581df534e3`.
- Source LICENSE and model-card metadata declare Apache-2.0. Preserve upstream
  license/attribution for redistributed code/weights; pin the actual artifacts.

## Actual model, not a replacement chat LLM

The SDK builds an encoder plus decision heads, not an autoregressive decoder.
One sequence combines task type/instructions, marked candidate options and state;
multiple question rows can share a batched forward pass. It is not one universal
state embedding reused without question conditioning.

| Config | Encoder blocks | Width / attention heads | Encoder FFN | Configured total token limit |
|---|---:|---|---:|---:|
| English | 28 | 1024 / 16 | 2624 | 512 |
| Multilingual | 22 | 768 / 12 | 1152 | 1024 |
| Typed decisions | 28 | 1024 / 16 | 2624 | 1024 |

These are from encoder/config.json and rl_agent_config.json at the revisions
above. Backbones are ModernBERT-large / mmBERT-base / ModernBERT-large. Encoder
activation is GELU. Context limits include instructions/options, not only news.
An architectural positional maximum is not evidence of accuracy at that length.

The decision module adds a three-type embedding and two pre-norm Transformer
encoder layers (width d, d/64 heads, FFN 4d, default ReLU, dropout 0.1). Candidate
marker representations pass through LayerNorm -> Linear(d,d) -> GELU -> Linear(d,1).
A separate pooled-state/statistics head is Linear(d+4,256) -> GELU -> Linear(256,2)
for the inspected action configuration. Temperature-adjusted probabilities become
choice, ordinal score or noul. Reuse these modules and weights through the SDK;
do not reproduce this architecture inside M5PHET.

Sources: [model implementation](https://github.com/NandhaKishorM/laya/blob/84ca7240348e6fb3b8145d2d9f6266d5125747d8/laya/common.py),
[English configuration](https://huggingface.co/convaiinnovations/laya/blob/cf7c54c0586eede67d827dfaab8cd2d2007273e9/encoder/config.json),
[multilingual configuration](https://huggingface.co/convaiinnovations/laya-multilingual/blob/af8d1acfcd010319dd7784409e5f2243363ece9e/encoder/config.json).

## Reuse map

| Responsibility | Owner / reuse decision |
|---|---|
| Encoder, tokenizer, decision heads, batching, question encoding | Laya Agent, unchanged |
| Language/checkpoint routing, resident model cache | Laya Router when needed; pinned local snapshots/attached Agents, not a duplicate router |
| Supported schema -> typed question conversion | Laya structured helpers after compatibility validation, not our own compiler |
| Local audit callbacks | Laya hooks connected to our evidence API; hooks do not themselves grant governed authority |
| Optional HTTP/MCP, ONNX, compilation/fast path | Existing upstream integrations; only deploy what the measured application needs |
| As-of clocks, input/source entitlement, task/model/calibration identity | M5PHET/application boundary |
| Optional data-gov/DOIN, local evidence and OLAP | Our existing adapters, not a second Laya engine |
| Forecasting, representation, RL and causal identification | Existing specialized engines via M5PHET; not forced into Laya classification |

news-signal already calls the pinned `laya.Agent.system_one`; it does NOT implement
a new encoder. Its current output surface is choice only. The 0.3.18 structured
module can save new work: enums, booleans and bounded integer levels; not continuous
forecast distributions, arbitrary nested JSON or causal estimands. Keep raw detailed
answers. Ordinal expected score and projected most-probable integer are distinct.

Sources: [structured API](https://github.com/NandhaKishorM/laya/blob/84ca7240348e6fb3b8145d2d9f6266d5125747d8/docs/structured.md),
[router](https://github.com/NandhaKishorM/laya/blob/84ca7240348e6fb3b8145d2d9f6266d5125747d8/laya/router.py),
[agent](https://github.com/NandhaKishorM/laya/blob/84ca7240348e6fb3b8145d2d9f6266d5125747d8/laya/agent.py).

## Reuse is not blind acceptance

1. CPU probes of upstream structured helpers reproduce `const: false` projecting
   to True, absent required output projecting to {}, and enum [1,"1"] collapsing
   to one label. Narrow schema prechecks and full response validation are still
   needed. Reject unsupported cases rather than inventing a schema compiler.
2. Legacy confidence is entropy concentration for choice/score, but max probability
   for noul. New answer_confidence is max probability; structured DecisionResult
   still copies legacy confidence. Retain definitions and raw answers, never share
   thresholds across these meanings. No shipped confidence establishes financial
   calibration. news-signal currently keeps probabilities explicitly uncalibrated.
3. The typed checkpoint config retains option-count temperature overrides while
   also supplying per-type temperatures; overrides take precedence in Agent.
   Record effective calibration, warnings and clamp behavior, not just three base
   numbers. Do not infer calibration quality from metadata or reuse test labels.
4. Agent may move to CPU on OOM. Explicit device/UUID checks, bounded workers and
   admission remain necessary; silent fallback cannot count as the requested path.
5. Upstream schema docs mention a structured extra; the inspected pyproject does
   not declare it. Test installed dependencies, not only README installation text.

Source probes: ../evidence/LAYA_ARCHITECTURE_2026_09_24/probe.py and results.json.
Upstream standalone CPU scripts: structured **45/0**, structured API **35/0**,
confidence **28/0** reported checks, run with GPU hidden and Hub offline in the
existing trading-stack interpreter. They are script counters, not a full pytest
suite or weight-backed inference. An initial pytest collection in base Python
failed on missing torch (3 errors); no dependency was installed into that runtime.

## Fork decision

Prior design was an independent framework consuming Laya; no fork was created.
Keep **M5PHET independent + upstream SDK dependency** as the default. This preserves
optional backend dependencies and five task families without maintaining Laya's
engine, servers, GPU kernels and training implementation ourselves.

A **thin compatibility fork of Laya** is justified only for a reproduced necessary
internal fix that cannot be handled at the supported boundary and has not landed
upstream. Keep upstream history/remote, isolated patch commits and original names/
licenses. Offer fixes upstream when approved; record the exit/rebase path. It is
an implementation dependency, not M5PHET's product identity. Do not rename a copied
Laya tree as if it implemented the five families. No fork or public issue/PR was
created during this review.

## Satoshi addendum: reuse before extending

This supplements the active integration continuation, without stopping experiments.
Inspect these pins before coding overlapping features. Reuse Agent first; adopt
Router/structured/hooks only where needed. Freeze compatibility tests before any
upgrade; do not roll a running experiment onto main or relax its identity.

Compare direct SDK and wrapper on identical checkpoint, state, question order,
token budgets, dtype, device and calibration. Cover choice/noul/score, missing and
invalid outputs, multilingual routing, truncation, batching order, raw versus
projected score and device fallback. The reviewed schema counterexamples must
refuse at our boundary; support no broader subset than actually verified.

Run the already-authorized bounded real-weight pilot under admission. Software
tests do not establish fit on a particular GPU. External5090 remains first choice
for GPU work; do not preempt valid jobs. Measure actual memory/latency instead of
assuming model-size ratios or requiring all checkpoints resident on the coordinator.

If a fork is required, first record the failing upstream test, minimal patch and
maintenance/reintegration plan; no parallel engine implementation. Update both
adapter pins and consumer tests together only after acceptance. Keep governance
and DOIN optional, and no real-capital execution. Report what was reused, the
remaining glue, versions, real-weight coverage and any unimplemented capability.
