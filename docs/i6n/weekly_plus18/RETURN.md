# Weekly Plus18 Launch

Campaign i6dmo-weekly-plus18-20261017, runtime bd21fc2f. On the original
rtx5090 class the fit-free preflight confirmed all 82 original weight tensors
and six short forwards exactly. All 53 original calendar periods are retained;
the first empty period was skipped exactly as in the parent. The first real
fit was observed at epoch 4 with a fresh heartbeat and 260 optimizer updates.
No new completed predictive measurement is claimed by this launch return.

The original mean TRAIN/weekly VALIDATION monitor, patience 10, min_delta
1e-8 and max_epochs 200 are unchanged. The only model-config delta is the
appended 18-hour horizon. Short rows, target bytes, masks, input windows,
scales and naive values were verified against authenticated parent results.
Adding a seventh loss term can change gradients and the selected epoch.

All 52 nonempty periods run sequentially without an agent. A CPU timer syncs
immutable terminals, reports every horizon's own original-scale MAE/naive,
publishes and reads back, pools the complete year, verifies two physical
mirrors and disables only this campaign's timers. A stopped partial fit is
retained as FAILED_INTERRUPTED, not silently retrained. GPU allocation is
4352 MiB, at least 1.25 times the measured maximum of the parent campaign;
no parallel slot on the second shared-RAM GPU. Large artifacts stay on the
configured storage volume, SQLite and locks on the local Unix filesystem.

Registry automation records each period and adds the seven-horizon annual
procedure only when every expected period is terminal and all required
member profiles authenticate. It never promotes a partial year's best week.
TRAIN/VALIDATION checkpoint selection makes this retrospective DEVELOPMENT
2024, exactly as for the parent. TEST remains closed; no trading claim.

Verification: 92 focused experiment/continuation tests and 22 registry tests
passed. The two existing protobuf deprecation warnings remain. Public code
contains no source bars, credentials, private addresses or host identities.
