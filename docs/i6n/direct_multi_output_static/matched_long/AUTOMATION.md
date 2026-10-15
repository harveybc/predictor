# Finite RAW Long Campaign

Authority: [RAW priority](../RAW_PRIORITY_20261015.md). The independent release
addendum accepts the executable bytes in `RELEASE_ACCEPTANCE.json`, not a
scientific result. Do not restart R2, DOIN or any closed short experiment.

## Execution

Use a private launch JSON with schema `i6dmo.raw_long_launch.v1` and explicit
absolute paths for `work_root`, `source_deployment`, `input_dir`,
`short_campaign`, and `strategy_inputs`. Never put credentials into it.
The source deployment supplies existing role interpreters, governed launcher,
publisher EnvironmentFile path, data roots and physical mirror destinations.
Preserve the engineering population; select a new production work root.

From the committed reviewed checkout, with `$SPEC` naming that private JSON:

```bash
python -m tools.i6dmo.raw_long_campaign install --spec "$SPEC"
```

The bounded launcher prepares the population under the frozen source commit,
authenticates the retained short report and strategy inputs, deploys exact
reviewed bytes, checks actual role imports/runtime and fresh resources, and
enables campaign-specific workers. It retries resource admission automatically
without reducing the fit cap. It does not recreate or retry scientific terminal
failures. The launcher stops after scientific close or technical finalization.

## Status

```bash
python -m tools.i6dmo.raw_long_campaign status --root "$WORK_ROOT"
python -m tools.i6dmo matched-long status --root "$WORK_ROOT/campaign" --level campaign
```

Both commands are finite reads. Execution contains per-unit progress, measured
ETA when available, typed failures, OLAP delivery/readback and mirror state.
No agent should remain polling logs. GPU work is on the preferred 5090 and,
if independently admissible, the 4090. The 5070 Ti is not a parallel slot.
Coordinator handles bounded CPU scoring/publication only.

## Fixed Sequence

Two RAW fits and their reports -> matched comparison -> retained-short pair
eligibility -> strategy only if eligible -> governed OLAP readback -> two
physical mirrors -> immutable CLOSE -> campaign-only timer shutdown.
An ineligible pair is a complete scientific negative, not a missing backtest.
No TEST unit exists. No smoothing, target cleaning, refitting short, pretraining,
or weekly retraining is added to this static contrast.

Own naive per horizon, original price scale and exactly paired business-masked
rows are mandatory. The long horizons are 18, 24, 36, 48, 60 and 72 elapsed
hours. Both input arms span 144 market hours, include t and retain elapsed-time
and closure context. MG12m has 13 observations, MG1l has 145. Flatten readouts
make their parameter counts differ; this is not an isolated sampling ablation.
The calendar and trading costs remain assumed, not broker-certified. Existing
feature-selection development reuse and unresolved source-age clocks must be
retained in every interpretation. TEST stays closed.
