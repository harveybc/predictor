# Secondary Weekly Window And Core Launch

Runtime f2cab53742dba821bfffbcc1a9c0a87eb7e0c3c3 is deployed on the secondary
rtx4090. Its finite worker and coordinator publisher timers are active. First
real training check: window13, 2024-01-07 period, epoch1, fresh progress,
14426 MiB observed VRAM. This is an observed start, NOT a completed scientific
result. The independent primary rtx5090 twelve-head campaign remains running.

## Evidence

- 40 focused window/weekly worker tests and 24 registry tests pass. No full-suite
  clean claim. Initial test collection failed for the missing new module before
  implementation. The later historical worker AST assertion rejected undeclared
  dispatch changes until its exact operational-only allowlist was recorded.
- Preparation manifest:
  `42c90ee4c9c338c888340beba962b02410998f745ea8403aaafc0ddcfb6936a8`.
  All 53 calendar periods retained. There are 51 nonempty common-support
  periods, two explicit common empty periods, 408 planned actual fits, 16
  empty-period skips and 212 conditional skips; 636 logical cells total.
- First preparation stopped before sealing at the final partial calendar week:
  no common observed inputs for the largest window. No fit ran. The explicit
  common-empty disposition was added and preparation completed, without
  imputing missing holiday observations or discarding valid individual heads.
- Fit-free actual-device preflight PASS for all twelve arms. Window-only
  controls: 1901 parameters and identical seeded initial weights. Core[4]:
  2595 parameters. Core[4,2]: 2397 parameters. Core changes are explicitly
  intentional, not mislabelled bitwise reproductions of the control.
- Seed6102026, original batch256/AdamW .001/.0001, patience10/min_delta1e-8,
  max200, original mean TRAIN/current-week VALIDATION checkpoint monitor.
  Frozen trainer and scientific layer builders retained. No TEST or DOIN.
- First empty-period report delivered and read back, pending0/mismatch0.
  Per-fit publication, annual per-arm registry and two verified physical
  mirrors are automated. Final mirror closure remains pending by design.
- No agent or Satoshi monitoring loop was installed. Query the generated
  status once when requested; timers drive the dependency transitions.

## Demo Clarification

The owner's destination is MT5 DEMO or Alpaca PAPER only. A governed positive
net-profit simulation under configured costs authorizes automatic demo
promotion, with account-mode/endpoint verification and no TEST or second human
approval gate. This launch does not claim that the prediction_provider/LTS
deployment adapter has already been installed or that a profitable compatible
pair already exists. Those operational receipts must precede such a claim.

All results remain retrospective DEVELOPMENT2024 checkpoint tuning, not
independent OOS evidence. The seven-head parent and primary active experiment
are immutable. The registry keeps each complete annual arm separate and
preserves model, data, mask, seed, checkpoint and stopping provenance.
