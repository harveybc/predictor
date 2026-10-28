# Combined index: data closure (D) and RL preparation (R), 2026-10-18

Order: `SATOSHI_DATA_AND_RL_ORDER_2026_10_18.md` @ `09ba0c7a`. Two independent
lanes, each returned once. No training, GPU, deployment, service/config/DOIN
change, lake/warehouse alteration, old-result restoration or periodic
monitoring. Musashi integrates reviewed work; nothing here is merged or live.

| Lane | Repository / branch | Commits | Return |
| --- | --- | --- | --- |
| D | financial-data `satoshi/data-closure-20261018` (base `dd14510cf`) | `9f7b518b3`, `9c371fe88` | `DATA_CLOSURE_RETURN.md` (branch root) |
| R | predictor `satoshi/rl-preparation-20261018` (base `09ba0c7a`) | `7ef557cb` | `RL_PREPARATION_RETURN.md` (this directory) |

## Lane D, essentials

- Tests first: 12 new D6 tests; 56 passed, 1 pre-existing skip, under 2 GiB/120 s.
- D1: reports read once; Europe/Berlin has the lowest overlap p95 (EURUSD 0.00023
  vs fixed UTC+1 0.00063 vs UTC+2 0.00195) on 34 pair-dates only; stated as an
  overlap score, NOT a verified timezone or quote side.
- D2: Forexite primary page declares «Período, GMT+1»; our retrieved family is
  the MetaStock `.zip`, price type, DST and minute convention UNSTATED by the
  provider. Observed: winter dates agree with fixed UTC+1 = Berlin; every summer
  overlap date is one hour off from fixed UTC+1 (counterexample archive
  `300320.zip`). Both statements retained; no shift, offset or merge. Side
  UNVERIFIED (bid/ask/mid within 0.00004).
- D3: FXCM schedule from docs plus 306 observed weeks; missing hours typed as
  SOURCE_FILE_ABSENT (EURUSD 975 h), CALENDAR_RESOLUTION_CANDIDATE (126 h),
  MISSING_QUOTE_OPEN_SESSION (2 h); 72 Saturday rows in 2024–2025 are a source
  fact for the calendar ruling, not an auto-closed label.
- D4: no new acquisition needed; HistData witness shows the lake file's UTC label
  is inconsistent with the provider's EST declaration (candidate witness, not a
  lake change).
- D5: inventory candidate of 394 records (370 catalog, 5 trading-required,
  19 model candidates); against the local lake 33 files missing and 104 present
  files with zero rows in 2020–2025; no hard-coded population size.
- Next object: a Forexite overlap day across an October DST end inside already
  acquired weeks; command in the return. Calendar ruling and the HistData label
  question go to Musashi. Nothing admitted, replaced or selected.

## Lane R, essentials

- Gap matrix over the 17 business-contract requirements: 10 IMPLEMENTED_AND_TESTED,
  5 IMPLEMENTED_UNTESTED, 7 MISSING, 2 UNKNOWN; six concrete runtime defects
  with file:line (financing never debited on the backtrader path even when
  enabled; NaN→0.0 in metrics; continuous enlargement invisible to the weekly
  overlay; reward one bar behind the action; `evaluate_test_split` default for
  legacy configs; hard-coded ETH constants in the adaptation tool).
- Stack: SB3 2.9.0 / gymnasium 1.3.0 / torch 2.13.0; `sb3_contrib` NOT installed,
  so TQC/CrossQ/QR-DQN/MaskablePPO need an environment change not made here.
- 12 primary papers plus SB3/Contrib docs and FINRA Notice 26-10 read on
  2026-10-18; business mismatches stated per paper; no benchmark result is a
  trading guarantee.
- Candidate table (existing SAC first; modular representation via an SB3
  features-extractor adapter, never replaced by an MLP silently; DQN on a
  declared Discrete(4)); synthetic CPU test spec RL00–RL10; six brainstorming
  hypotheses; PDT note (framework effective 2026-06-04, transition to
  2027-10-20, securities margin accounts, FX not mentioned; account state in
  private configuration); finite RL0→RL4 manifest, none authorised, no ETA.

## Limits

Clock evidence rests on 34 pair-dates and cannot determine the quote side;
inventory cadence is inferred. RL line numbers are valid only at gym-fx
`e287ea2` / agent-multi `e4d01fa0`; RL10 stays UNKNOWN; decision-at-t causality
cannot be certified until lane D's bar-timestamp semantics are ruled.
