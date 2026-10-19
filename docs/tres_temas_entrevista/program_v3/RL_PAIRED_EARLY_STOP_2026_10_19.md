# Direct SAC Generalization Experiment

## Reference Result

The selected-input direct SAC at leverage50 completed frozen-policy TRAIN
2020-2023 and VALIDATION2024. From USD10000: TRAIN net+35111.73, return+351.12%,
max drawdown11.91%, daily nonannualized Sharpe0.1437,828 closed trades,
52.42% wins, profit factor1.657; VALIDATION net-19.24, return-0.19%, drawdown
10.45%, Sharpe-0.0053,204 closed trades,46.08% wins, profit factor0.996.
Both respect at most one entry per NY day and end flat. This is a large
generalization gap, not proof all of it is overfitting rather than regime
change. The partial training percentages previously shown were not validation.

## Approved Next Experiment

Owner follow-up: new temporal SAC experiments are paused. Its existing
reference fits completed, so this pauses future dispatch rather than killing
a running fit. The direct-only early-stop successor continues unchanged.
Engineering tests are not financial evidence that a regularizer works.

Keep the same15 selected variables, their bytes/normalization,13 observations
at stride2,32/32 SAC heads, seed7, leverage50, relative margin budget0.1,
mandatory ATR SL/TP, business calendar and cost scenario. Only add checkpoint
monitoring and early stopping; do not enlarge the network or add refitting.

Every5000 transitions evaluate the saved frozen checkpoint on full TRAIN and
full VALIDATION through the existing curated inference pipeline in separate
processes. Neither evaluation learns or writes replay in the parent. Metric:

`M = -0.5 * (mean_daily_log_growth_TRAIN + mean_daily_log_growth_VALIDATION)`.

Group observed equity by NY business-day close, including initial capital
as prior equity. Do not average raw profits over unequal periods. Patience10
evaluations, absolute min_delta1e-8, cap100000 transitions. Restore the best
checkpoint's complete policy and optimizer state; retain actual work counters
and selected-policy step separately. Incomplete/nonfinite/nonpositive-equity
rollouts cannot count as full-period eligible checkpoints.

BatchNorm, dropout and weight decay are OFF. The owner's instruction prohibits
combining BatchNorm and dropout in this experiment and requires separate
future ablations. No regularization ablation is launched automatically here.
Validation serves checkpoint tuning: it is not an independent holdout.

## Deployment And Continuation

Contract `ac7921533f6c4c95fb7a7de2e5cf6a7307415ca0c8cf7739c908027baa091ad6`;
agent-multi `e119e8ad`, gym-fx `df1b602`. Requestv4 contains exactly one arm,
SAC_FLAT: two final train/validation tasks, not a new architecture sweep.
Selection remains `553fc047` within admitted EURUSD core; no new all-catalogue
or causal-identification completion claim.

The resource-aware systemd successor is enabled and waiting for the pinned
existing leverage50 campaign `37e06192` to close. It verifies its task artifacts,
warehouse readback and closed backup, then checks free RAM, disk and GPU before
launching on5090. A minute timer retries waiting conditions; an active service
cannot be launched twice. Existing runtime/scientific configuration is untouched.

Every completed checkpoint pair is published through the existing OLAP outbox
with content hashes, checkpoint step, score, patience, growth and full trading
metrics for each split. Up to20 pairs per fit plus two final reports. Final
closure requires actual delivery/readback counts; closed SQLite backup,
checkpoints, configs, traces, summaries and monitor records are then collected
and verified on the second physical disk. No active SQLite/WAL/locks are copied.

Engineering evidence:157 focused CPU tests passed. A real SAC synthetic test
executed gradient updates, two isolated Backtrader inference processes,
confirmed unchanged parent Python/NumPy/Torch RNG and replay, and restored the
exact best policy and actor/critic optimizer state without changing actual-work
counters. Unit cases cover patience boundaries, trace support/NaNs, same units,
direct-only request, monitor publication, collector hashes and predecessor
refusal. First real GPU checkpoint/readback remains unverified until execution.

Projected duration after start: about2-3h for20 full checkpoint evaluations,
not a measured completion ETA. Status exposes dispatch wait, training progress,
current monitor split/progress, score and patience. No agent watches logs.
