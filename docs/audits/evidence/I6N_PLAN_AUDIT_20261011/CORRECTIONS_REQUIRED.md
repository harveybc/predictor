# I6-N corrections required before implementation

These are design changes for the plan author. None has been made. Each step
lists the findings it closes and the tests that must exist before any fit.

1. **Calendar artifact (N0 prerequisite).** Versioned instrument calendar with
   timezone, DST, holidays and early closes; bar label convention; segment list
   for 2023 inner weeks and 2024 VALIDATION derived from it; fit cutoff = C of
   the previous week and release before O; split membership by target time.
   Closes AUD-08, AUD-10, AUD-11. Tests: WMC01, WMC02, WMC06 from
   `WMC_COUNTEREXAMPLES.md`.
2. **Single eligibility function and support table.** E/M by origin, h and role
   with strict h < R(t); per-h/week/role support table sealed with digest;
   power/row budget per representative. Closes AUD-07, AUD-29. Tests: WMC02,
   WMC03, WMC05.
3. **Feature manifest provenance.** TRAIN-only manifest per fold cutoff, or an
   explicit RETROSPECTIVE_DEVELOPMENT declaration carried in identity, with TEST
   as the only prospective evidence. Closes AUD-01.
4. **Input slot and alignment rules.** Placeholder/presence/mask per route;
   origin-anchored temporal reduction; origin-anchored lag-hour position
   encoding; Transformer order and norm placement; readout gene or drop the
   claim. Closes AUD-06, AUD-09, AUD-16, AUD-17.
5. **Search design.** Input transform and output mode as N2 genes (or
   contrasted at >=2 windows); absolute L values; per-family representatives;
   recipe rules with the 12 (L, delta_x, n) triples; DEAP fitness, operators,
   repair, tolerance and stopping sealed numerically; seed-noise calibration;
   inner-only monitor; FULL_RETRAIN and init; fit hyperparameters stated with
   an LR check; multi-output w_h and s_h; ARIMA order/convergence policy.
   Closes AUD-03, AUD-04, AUD-05, AUD-12, AUD-14, AUD-15, AUD-19.
6. **Fitness and populations.** Fitness = S_h on sealed common support, forecast
   failure typed and disqualifying; U_h only after the strategy node, with a
   recipe-per-h map. Closes AUD-13.
7. **Strategy repair node** in the DAG before any U_h or strategy result
   (pinned plugin, horizon map, masks, chronological SL/TP, abstention,
   calendar). Closes AUD-02.
8. **Identity and OLAP.** Identity fields listed (contract version, calendar id,
   offsets, cut, bar convention, mask-function digest, manifest digest, row
   digests); population fact table for S_h/U_h; custody fields mapped to
   gov_terminal tables with full readback; copy rule aligned. Closes AUD-26,
   AUD-27, AUD-31. Test: WMC10.
9. **Genotype and executor specification.** Genotype schema and canonicalizer;
   named missing components (I6-N model factory without the n<=241 cap,
   conditional DEAP driver, DOIN plugin); typed failure classes, watchdog,
   stop-drain, durable PAUSED/SUPERSEDED state checked by claim(). Closes
   AUD-18, AUD-20, AUD-25.
10. **Placement and budget.** One placement authority and corrected role map;
    host-level lease across campaigns; admission against live available memory;
    pilots for n>241 and new families; itemized budget with measured or
    NOT_MEASURED cost per class. Closes AUD-21 to AUD-24, AUD-28.

Re-audit scope after revision: the revised proposal and contracts by digest,
the support table, the sealed DEAP/fitness values and the identity schema.
