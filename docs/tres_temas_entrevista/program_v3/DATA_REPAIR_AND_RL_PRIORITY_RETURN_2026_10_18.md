# Data Repair And RL Priority: Session Return

Scope: EURUSD/GBPUSD source recovery and current work-plan correction. This is
not whole-inventory/lake admission, final feature selection or an RL launch.

## Measured Data Work

The native hourly source produced612 of636 requested pair-week files across
2020-2025. Alternative acquisition recovered84 of110 requested dates,100,610
native minute observations per pair within that request. The remaining26 dates
are23 Saturdays, one Sunday and two Friday holidays by date label, all returning
NO_REQUIRED_PAIRS; none is thereby certified as a legitimate closure. A parser
error rejecting valid next-midnight completed-minute endpoints was fixed and
the bounded acquisition rerun; prices were not interpolated or time-shifted.

Independent clock diagnostics ran on actual available source files:108
pair/clock/month or aggregate comparisons, strict complete60-minute support,
explicit clock hypotheses and unverified alternative quote side. No source
was spliced, automatically assigned a clock or admitted. Extra diagnostic
overlap files are counted separately from the110-date recovery request. The
governed warehouse contains4 reports with exact readback and zero pending or
mismatched deliveries; the physical cube contains1122 metric rows. Its2,895,872
bytes hash to3770cfb70fb717e039175f4972ecd501c80144a0270aac860f1324c40704fbc7,
matching the local closed copy, bulk-volume copy and second-host mirror.

Focused verification:80 passing tests across acquisition, native quality,
clock comparison, recovery/status, publication/physical schema and affected-cube
cleanup controls. This is not a claim about every repository's full suite.
Real publication and physical readback ran in addition to synthetic tests.

## Plan Changes

Master, active execution text, queue, checklist and method states now agree:
repair/admit data -> verify preprocessing -> recompute affected selection ->
RL environment/baselines -> existing SAC compact/modular -> justified RL
contrasts and paper. Forecasting and NEAT are deferred. The old active-state
numerical results/dispatch orders were removed from these control documents;
this does not claim every historical Git object or unrelated copy is purged.

The adaptation insight is recorded as a qualitative hypothesis only. A
successful fitted-week score is not prospective prediction. Increasing update
frequency is an experiment, not a guaranteed monotonic improvement. Retired
models/metrics are not restored. RL is evaluated by net trading performance and
risk, not required to beat a price-persistence MAE. Existing preprocessing and
representation code is reused where accepted, not rebuilt or presumed accepted.

## Remaining Work And Handoff

Actual source clocks/quote side, holiday/session coverage, source replacement,
whole required-variable inventory, admitted-only lake deployment and prepared
support remain unfinished. Recomputed selection and RL scientific runs have
not started. No defensible completion ETA for those unresolved tasks is yet
measured. Bounded recovery and publication services are installed; they do not
implement automatic semantic admission or an RL successor campaign.

SATOSHI_DATA_AND_RL_ORDER_2026_10_18.md is a prepared handoff, not a message
delivered to another session. It allows two independent bounded lanes: primary
source/calendar/inventory evidence and RL literature/runtime gap mapping. No
fits, GPU, live deployment, deletion or periodic monitoring. The owner may ask
Satoshi to read and execute that file while Musashi continues data closure.
