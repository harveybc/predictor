# Satoshi: recovered baseline is CLOSED; repair the remaining automation

Read `docs/i6n/direct_multi_output_static/RETURN_REPORT_RECOVERY_20261015.md`.
Musashi deployed runtime `a169fcba` from
`codex/i6dmo-report-recovery-20261015`; active campaign
`i6dmo-9e1ebad89ff2`. Both fits and reports COMPLETE, pair hook executed
PAIR_NOT_ELIGIBLE, closure CLOSED. All campaign timers are disabled.

Do not start another MG2/MG12 fit or reopen the closed campaign.
Integrate the repair/evidence branches into your coordination branch after
checking its current state; integration is not an instruction to redeploy.
The source campaign `i6dmo-72809b6c88a6` is retained FAILED with short.fit
COMPLETE and short.report UNKNOWN_DESIGN_FIELD; its idle timers are disabled.

The repaired actual report path is covered on MG2 and MG12, without a fake
arrays_loader. Completed-fit reuse retains the source fit's actual code pin,
identity and digests; new units record the successor pin. Do not bypass the
reuse verifier or relabel a completed fit with a new training code pin.

The reports contain two eligible short heads (6/12 h) and zero long heads.
110 reports / 3952 metrics are readback verified; 58 files verify per mirror.
No live scientific work needs agent monitoring.

Next bounded engineering task, CPU only, without fits or TEST:

1. Make pair.eligibility depend on authenticated report metrics and the sealed
   minimum-head policy only. It must compute a negative decision even when
   strategy OHLC pins are absent. The real worker currently stops too early
   at STRATEGY_INPUT_PIN_MISSING:strategy_bars. Preserve real source pin checks
   for the actual trading stage. Add a real-worker regression, not just a fake
   trainer or an injected arrays_loader.
2. Align strategy.validation with the real adapter. The graph creates kind
   strategy, but the real run_unit currently accepts only pair. It also needs
   authenticated short/long forecast upstreams even though its direct graph
   dependency is pair.eligibility. Exercise both qualified and disqualified
   paths through actual hooks, without a scientific fit.
3. Make terminal scientific disposition + fully delivered outbox + verified
   mirrors advance to explicit CLOSE automatically. Preserve the closure's
   input proof files before mutable publication/mirror receipts are rewritten.
   The present recovery used the existing close CLI once after verification.
4. Keep source-code identity, imported-fit reuse and negative decisions visible
   in status. A live service failure must not appear indefinitely as waiting.

Use separate implementation and review ownership. Return code pins, executable
CPU regressions and one end-to-end finite automation receipt. No deployment
over the closed scientific identity, no agent polling, no new owner approval.

The published source return's phrase "339 of 1587 observations covered" is
inverted relative to its underlying ALFRED audit: 339 were outside the declared
date coverage. Correct that narrative in an addendum without changing evidence
or claiming look-ahead proved solely by ALFRED database ingestion dates.
