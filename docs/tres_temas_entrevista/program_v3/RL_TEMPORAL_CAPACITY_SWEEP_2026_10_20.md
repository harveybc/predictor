# Autonomous Gradual Temporal Capacity Study

Owner approved2026-10-20: automate and execute the entire diagnostic ladder on
the preferred5090,one heavy worker at a time,no owner/agent intervention between
trials. The closed fixed-budget baseline is retained,not retrained or modified.
Implementation contract: `agent-multi/docs/work_plan/52_TEMPORAL_CAPACITY_SWEEP.md`.

## Frozen Context And Learning

Same15 selected features,13 observed open-market hourly bars,stride2,includes t;
TRAIN2020-23 and full VALIDATION2024;100000 transitions per model,four full
observations at25k/50k/75k/100k,no early stop,final100k selected. Same seed7
for the exploratory ladder,declared costs,leverage50,SL/TP,volume budget,
entry quota and weekly session restrictions. No TEST,DOIN,pretraining,
regularizer,reward change,refit,window change or downsampling change.

## Exact Order

| Stage | Trials | Frozen settings |
| --- | --- | --- |
| 1 | CoreConv1D8 and16 filters | branches2,hidden[8],one core layer |
| 2 | Hidden[16] and[32] for actor and twin Q | provisional stage1 architecture |
| 3 | Per-feature branches4 and8 filters | provisional stage2 architecture |
| 4 | Two causal coreConv1D layers | provisional stage3 widths |

Both widths run even if the first disappoints. Each stage also retains the
previous incumbent. Compare both candidates to that same frozen stage-start
incumbent:validation mean daily log growth improves by>1e-8,maxDD does not
increase beyond1e-8 numerical tolerance,positive TRAIN-minus-VALIDATION daily
growth gap does not exceed max(previous gap,0)+1e-8,and at least one validation
trade closes. Eligible candidates rank by validation daily growth,then lower
maxDD. These are conservative exploratory promotion rules,not business risk
thresholds or statistical certificates. All rejected outcomes remain visible.

If the ladder promotes a different architecture,run baseline and finalist
with seeds17 and29,using retained seed7 for the third paired comparison.
Report all three shared seeds and their paired daily-growth differences,not
the best seed. Inconsistent or failed confirmations remain INCONCLUSIVE.
At most7 exploratory fits plus4 conditional confirmation fits,excluding bounded
retries. The stagewise result is provisional;interactions need later joint
contrasts before claiming any global optimum. Windows/resolution follow later.

## Automation And Evidence

Native SAC queue owns all learning and inference. A locked finite worker
materializes pinned JSON candidates in order;an independent timer publishes
the complete checkpoint reports and authenticated curves. A successor waits
for verified closure/readback,not for a human. A separate machine discovers
new closed trials and verifies their physical copies. All outcomes retain
effective config,architecture,seed,data/selection/code/checkpoint identities.
Sweep JSON/CSV/Markdown and hierarchical status/ETA are automatic.

Resource waits are retryable;warehouse outages do not trigger refits. Failed
or interrupted TRAIN is retained as partial and gets at most one fresh-root
retry,never a fake exact continuation from policy-only weights. Failed final
validation can adopt the completed TRAIN. Exhausted individual candidates
are recorded and skipped. Proven data/config/version/integrity contradictions
fail closed rather than manufacture successful trials. Permanent hardware
failure cannot be promised away by automation.

Baseline measured approximately70min including monitoring and final reporting.
Initial range for7-11 new fits is roughly8-15h,provisional until wider models
are measured. Status must distinguish compute completion,warehouse closure
and independent-copy completion. Scientific acceptance and deployment state
will be appended after actual worker verification;this plan alone is not a
claim that a GPU fit started.

## Deployment Verified

Native runtimeee35cb3b;specb7cca8ea;first contract446e378c.219 local focused
regressions passed,including real-engine depth2/seed17 monitor and legacy
FIXED monitor;35 actual-worker cases passed in30.13s,1.5GiB peak,no swap.
Systemd rendered units parsed;worker/publisher timers and second-machine
collector are enabled. First GPU admission passed16 native updates and exact
reload on attempt1. The core8 candidate started TRAIN09:50:37UTC and reached
5000 transitions in104.9758s,with branches2/hidden[8]/depth1/seed7. Core16
follows automatically;no baseline weights or interrupted fit are imported.
No new financial result is claimed from startup. Root status and the automatic
trial ledger determine progress;this dated paragraph is not a monitoring loop.
Follow-up startup check:10000 transitions in221.296s,no restart/no BLOCKED
marker. Independent collector verified the closed baseline,pending0;its
operational memory cap was corrected for native library imports without
changing the scientific runtime or training limits.
