# Paired Early-Stop SAC Result

Closed contract `ac792153`, selection `553fc047`, runtime `e119e8ad`.
Completed100000 transitions and20 full TRAIN/VALIDATION checkpoint pairs.
Stop reason TRANSITION_CAP, not exhausted patience; best checkpoint100000,
patience0/10, approved mean daily log-growth objective unchanged. This run
does not establish that early stopping itself caused an improvement.

| Metric | TRAIN2020-2023 | VALIDATION2024 |
| --- | --- | --- |
| Net profit USD | 36107.593618467516 | 75.18786705539242 |
| Return percent | 361.07593618467515 | 0.7518786705539293 |
| Max drawdown percent | 11.844102885997385 | 10.636129215553442 |
| Daily nonannualized Sharpe | 0.14566234906170974 | 0.000042528364652847275 |
| Closed trades | 828 | 204 |
| Winning trades percent | 52.41545893719807 | 45.588235294117645 |
| Profit factor | 1.6548897336417623 | 1.0150589809594726 |

Initial equity USD10000 per role. VALIDATION is barely positive under the
fixed analytical cost scenario; the generalization gap remains large and
near-zero Sharpe is not convincing trading skill. No independent holdout,
broker-realism certificate, or profitable-deployment claim follows.

All42 reports delivered/read back, zero pending/mismatch; authenticated close
and SQLite backup complete. Second physical collection reports COMPLETE with
artifact hashes verified. The ordinary queue journal's historical
COMPLETED_OUTBOX_PENDING is superseded by the authenticated CLOSE evidence,
not interpreted as an outstanding delivery.

## Successor And Operational Repair

Dropout0.05 started on the preferred5090 after closure. First check15:05Z:
TRAIN RUNNING,1000 steps, attempt1, no restarts. Its bounded GPU smoke passed
24 synthetic transitions/20 updates plus checkpoint/prediction roundtrip;
this smoke is separate from financial results, which are not complete yet.

An operational defect delayed handoff: systemd's StartLimitIntervalSec900 /
Burst3 also counted normal minute-by-minute ExecCondition waits. Corrected
the generated worker limits (renderer `a7b4da3e`) to disable that unrelated
global start throttle; native queue max2 actual task attempts, finite timeouts,
memory/GPU gates and integrity refusal remain unchanged. Ten targeted tests
pass after the correction, including real systemd parsing. Cleared the old
rate counter and initiated the first worker once; native timers own all
subsequent checks/handoffs. Scientific runtime stays `d9e635f7`, all four
sealed contracts and their scientific settings unchanged. L2, dropout+L2
and standalone BatchNorm remain sequential successors, not measured runs.
