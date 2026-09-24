# NEWS-LIVE: parallel news-model track for MT5 and Alpaca

Owner approved 23-sep local / 24-sep UTC. This is an additive business track,
not a replacement for doctoral forecasting, extractor/core pretraining, weekly
RL, or FIN-LOSS-OPT. It does not wait for protocol-B retention repairs.

## Outcome and boundaries

Deliver a working local Laya news pipeline with live inputs, then bounded
MT5 **demo** and Alpaca **paper** execution through the existing LTS contracts.
No real-capital orders, replacement of existing strategies, or claim that news
sentiment predicts returns. No further routine engineering approval is needed.
Missing account/data entitlements are facts to name, not permission to fabricate.

Repository: [news-signal](https://github.com/harveybc/news-signal). It owns news
interpretation only, with unique `news_signal` imports. The old trading-signal
repo is retired. LTS owns risk and execution; trading-contracts owns shared
intent/report schemas; data-gov owns authenticated delivery/accounting.

## Finite stages and acceptance

| Stage | Owner | Observable completion |
|---|---|---|
| NEWS-ADAPTER | Musashi -> Satoshi | Prototype strict typed CLI/tests now; real frozen Laya weights + input/response/model identity and host measurements next |
| NEWS-COLLECT | Satoshi | Licensed source, actual receipt time, revisions, dedup, missingness, symbol mapping, durable restart and governed evidence; collect prospective data immediately once admitted |
| NEWS-SHADOW | Satoshi | Resident model, bounded queue, separate news/price/combined observations, reference labels, calibration/latency/cost metrics; no action authority |
| NEWS-PAPER | Satoshi | Existing LTS risk route to BOTH MT5 demo and Alpaca paper, real acknowledgements/fills and restart reconciliation, isolated profiles |
| NEWS-EVALUATE | Satoshi + Musashi | Predeclared matched prospective comparison; domain ML and paper economics separately; uncertainty and failed units visible |
| NEWS-REAL | Owner + reviewer | NOT AUTHORIZED by this plan; separate capital/risk/release decision after evidence |

The prototype hashes are NOT accepted data-gov receipts. The fixture backend
is NOT Laya. The real-weight adapter is implemented but not smoke-tested yet;
no financial accuracy, calibration, latency or profitability measured by Musashi.

## Model and scientific questions

First engine: pinned Laya English checkpoint, one asset supplied from verified
mapping, short licensed headline/excerpt; typed relevance/event/tone questions.
Pin SDK source and checkpoint bytes independently, numerical dependencies and
device observed in the inference child. Unsupported language, token overflow,
nonfinite/contradictory response, stale input or device fallback refuse.

Do NOT map favourable tone directly to a market order. Evaluate (a) classification
quality and calibration, (b) added causal information for the trading policy,
(c) cost/latency and execution behavior as separate questions. The deterministic
policy is outside the classifier. `action.act_probability` is not trading authority.
The earliest available quote follows first receipt PLUS measured inference and
execution latency, not the article's claimed publication time.

Freeze prospective collection/start, checkpoint, tasks and changes. Public news
used in pretraining may contaminate historical evaluation; prefer prospective
outcomes after the freeze and keep historical diagnostics explicitly qualified.
Use chronological event/issuer/time-block splits, no revision backfill, no duplicate
event across evaluation partitions. Fit any probability calibration on its own
past slice. Compare majority/rules and matched FinBERT for classification where
its actual task matches; compare no-trade and existing price-only policy for
execution. No arbitrary published metric may be called directly comparable.

Required report: per-class counts, macro-F1, confusion, precision/recall, Brier/NLL
and ECE definition; OOD/abstention/coverage, received-to-feature p50/p95/p99 latency,
input and realized execution age; paper return after spread/fees/latency, drawdown,
turnover, exposure and Sharpe with sampling interval and temporal support. Return
forecast errors only if a numeric-return task is actually defined. Tiny sample
metrics are descriptive; no fixed demonstration count proves business readiness.

Broader typed-ML engines are an RFC in news-signal, not a new doctoral dependency.
Embedding/extractor, fusion branch and final policy-head placements each need a
separate ablation. DOIN can later tune bounded valid configurations; this is not
automatic evidence that Laya replaces predictor or an RL agent.

## Hardware and concurrency

External RTX5090 remains first choice for substantial GPU work. The owner asked
whether the coordinator's RTX4070 8GiB can host this smaller model: yes, it is a
candidate. Snapshot 24-sep about01:51Z: 8188MiB total, 6485MiB free, 46C,
about19GiB host RAM available, no compute process reported, desktop utilization
38%. This is a resource observation, NOT a measured Laya memory requirement.

Start a bounded real-weight batch=1 pilot on the coordinator after fresh admission,
2 CPU threads, eager inference, no training/compile/Router preload of multiple
checkpoints. Measure cold-load RSS/VRAM and warm latency. Keep it resident there
only if it meets the declared news workload without affecting services/desktop.
Use WORKER_A independently if necessary, or queue on external5090 after its lease
releases. Never silently select the internal GPU or start a second RP135 job.
SDK load and inference can fall back to CPU: adapter must refuse and record this.

Hermes/subagents: distinct coding/collector/broker-validation worktrees, bounded
budgets and leases. One dispatcher owns each GPU/collector/execution loop. Agents
do not receive broker secrets in prompts. Do not reactivate expired RP135 Hermes
task. News operation does not require stopping a running scientific experiment.

## MT5 / Alpaca integration specifics

Use `lts.app.demo_execution_service.DemoExecutionService.process_intent` before
any broker mutation. Alpaca then uses `AlpacaL1Executor.consume_pending`; MT5 uses
the validated LTS order -> `Mt5ExecutionStore` -> signed MQL5 bridge path. Do not
call transport `submit` or bridge `enqueue` as a substitute for L0 acceptance.

Read-only source review found adoption tests still needed for Alpaca account
shorting eligibility, MT5 netting/hedging and restricted symbol trade modes.
Start long/flat on explicitly eligible instruments and check close semantics;
no new shorts. `AlpacaModelRunner.tick(allow_execution=False)` still performs
monitor/drain work: it is NOT a no-network shadow harness. Use a zero-network
sink for shadow tests and a separate approved profile for paper canaries.

No symbols, account identities, API keys or private machine names enter public
configuration. Resolve the already approved demo/paper mandate locally. If none
exists, implement/tests/collector continue; report the exact missing mandate or
credential reference. Never inspect unrelated secret stores or print secrets.

## Repository and submission

Source is runnable prototype code with explicit limitations, README, test design,
agent instructions and a broader typed-ML RFC. Submission text uses the current
prototype status, not a claim of completed MT5/Alpaca integration. The independent
community form's Type is Integration; source URL is news-signal. Hidden Website
field stays empty. Owner submits; no external form was sent in this work.
