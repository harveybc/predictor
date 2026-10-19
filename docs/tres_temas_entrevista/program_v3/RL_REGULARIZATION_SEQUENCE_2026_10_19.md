# Direct SAC Regularization Sequence

Owner authorization2026-10-19: no manual approval between the current early
stopping run and the following regularization ablations. Preserve the active
scientific contract/runtime until its authenticated closure.

## Order

| Order | Treatment | Location |
| --- | --- | --- |
| 0 | Current paired early stop, no regularizer | unchanged active5090 run |
| 1 | dropout0.05 only | after both hidden critic activations |
| 2 | L2 1e-6 only | native Adam critic parameter groups, including biases |
| 3 | dropout0.05 + L2 1e-6 | same critic locations and optimizer |
| 4 | BatchNorm only | critic hidden Linear -> BN -> ReLU |

No BatchNorm/dropout combination. Actor and entropy optimizer remain unchanged;
this first bounded study isolates critic regularization. It is not a full
DroQ/CrossQ reproduction, nor a claim that these are optimal strengths.
Both target critics remain in eval mode; native target running-stat handling
and exact save/load/best-checkpoint restore are required.

## Identical Comparison

Same15 selected inputs, their repaired data/normalization,13 observations,
stride2 observed open-market bars,32/32 units, seed7, leverage50, sizing,
SL/TP, weekend/session/daily-entry rules, costs and full2020-23 TRAIN/2024
VALIDATION. Fresh independent training for each treatment; no donor or TEST.
Same paired mean daily log-growth early stopping, every5000 transitions,
patience10, min_delta1e-8, cap100000. No temporal SAC dispatch.

The mean can improve while VALIDATION still loses. Report both full-period
roles, net profit, drawdown, Sharpe definition, trades, win rate, profit factor,
costs and gap. Compare with the no-regularizer reference, not gap alone.
Poor generalization may involve both overfitting and market distribution
shift; the experiments, not the gap by itself, determine whether a treatment
helps. No guaranteed profitability or blanket diagnosis is claimed.

## Autonomous Execution

agent-multi `tools.selected_rl_regularization_sequence` prepares four immutable
request_v5 configs/contracts, pins each predecessor and renders finite systemd
workers/timers. A successor checks actual predecessor tasks, checkpoint file
digests, OLAP report counts/readback and closed backup, plus fresh free RAM,
disk and GPU ownership. Then a bounded synthetic GPU admission smoke verifies
finite gradient updates and exact archive/prediction roundtrip before the fit.
One heavy worker on5090 only; never use5070Ti as a second shared-RAM slot.

Every checkpoint/final role is published to OLAP with explicit regularization
parameters/scope, config/code/data/weight identities and trading metrics.
Publication and second-machine collection use separate finite timers; no
agent watches logs. An unsuccessful trading result does not block the next
arm. Integrity or exhausted-execution failures are typed, not relabelled as
successful scientific completion. Status is callable for the sequence and
every campaign; ETA for an unmeasured treatment is explicitly provisional.

## Engineering Evidence

Tests first: unsupported profiles, missing sequence implementation and paired
predecessor closure exposed red cases. Verify all new cases and the existing
queue/monitor/publication/collector tests before deployment. Synthetic native
engine tests exercise all four profiles through two inference subprocesses,
unchanged parent replay/RNG and exact best policy/optimizer/buffer restoration.
These are engineering evidence, not measured financial results.

Source design: agent-multi `docs/work_plan/47_SAC_REGULARIZATION_SEQUENCE.md`.
Deployment pins and real admission state are recorded in the execution return;
do not call queued arms running or their GPU admission already verified.

## Deployment Return

Deployed2026-10-19, scientific runtime agent-multi `d9e635f7`, renderer
`381da73c`. Four pinned contracts prepared, worker timers plus publication
enabled and each checked once: all WAIT_PREDECESSOR. Second-machine collector
enabled, first no-close read PENDING as expected. Real systemd parse and
EnvironmentFile resolution checked after correcting generated path settings.
182 focused CPU tests pass; the final renderer regression also checks the
actual systemd parser and rejects ignored non-absolute path warnings.
GPU admissions and financial fits remain future timer-owned actions, not
claimed measurements. The current unregularized experiment is unchanged.
