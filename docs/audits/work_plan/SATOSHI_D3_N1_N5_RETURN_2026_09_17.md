# Return: D3 N1–N5 — utility results recovered, calibration scoped, governance before work, descriptive pilot

Order: `docs/handoffs/MUSASHI_D3_M1_M6_REVIEW_AND_N1_N5_2026_09_17.md` (`bb5e3ed`), over the
review of `b6facb8`. Executed without pausing between blocks. `NON_GOVERNING` throughout; no
reserve, no public confirmation, no financial data, no GPU. D3 mechanics were not repeated.

**PRE** (reviewer's reproducer at `b6facb8`): `EMPTY_CALIBRATION 0 0.0 DOES_NOT_ADVANCE`,
`EMPTY_CALIBRATION 0 nan DOES_NOT_ADVANCE`; cube: `utilreh-v4-utility-rehearsal` three COMPLETED
terminals with **0 metric rows**. **POST**: the reproducer's records cannot be sealed
(`ProtocolRefusal`: zero simulations / non-finite rate / unknown generator); the cube holds the
three recovered deltas at generation 2; the new rehearsal reports every metric from the
verified file.

## N1 — the measured results, recovered and traceable

* **Regression frozen first** (`test_N1_the_parents_score_is_the_verified_output_file…`,
  `test_N1_a_missing_altered_or_discordant_output…`): after a COMPLETED child the parent reads
  the file the child named and the runner re-hashed, checks bytes, schema
  (`df_utility_contrast.v1`), contrast and protocol identity, finite values and outcome
  agreement; the score returned **is** that result. Absent, altered, discordant or non-finite →
  `SCORE_UNVERIFIED`, no score, no fabricated zero. A completed attempt is re-verified from its
  `outcome.json` and never re-run. The envelope writes `UNAVAILABLE` (stored as null), never `0`,
  where a measurement is absent.
* **Recovery of `utilreh-v4`** (`tools/df_utility_recover.py`, receipt
  `utility_rehearsal_v4/RECOVERY.json`): each `contrast.json` re-hashed against the digest its
  child declared (that version had not persisted the runner's own digest — said so in the
  receipt; the files predate the schema field — labelled `LEGACY_FILE_WITHOUT_SCHEMA_FIELD`),
  protocol identity checked against the freeze; **three generation-2 terminals** sent through the
  outbox (same status, real metrics, tags naming the original terminal digest and the recovery),
  reconciled `missing_units: []`; **cube content equal to the files value by value** —
  `cusum −0.18881876086493116`, `delta_run_length 0.019637893298198927`,
  `mad_extremes −0.18871307727173697` (12 metric rows at generation 2); corrective DEVELOPMENT
  envelope `57e5f959…` (`utility-rehearsal-utilreh-v4-recovery-2`) naming the one it corrects
  (`da341cc5…`), loaded. Generation 1 and the original envelope stay as history; nothing was
  re-measured or promoted.
* **Two errors of mine on the way, recorded**: (a) the first recovery attempt refused every file
  (schema field) and still emitted an envelope with nothing recovered, which the loader took
  before it could be closed — `utility-rehearsal-utilreh-v4-recovery` (`2e68209e…`, four
  `UNAVAILABLE` units); disposition manifest prepared
  (`docs/audits/evidence/d3_k5_20260917/N1_DISPOSITION_PENDING.json`, `ACCIDENTAL_OPERATIONAL_INGESTION`,
  not admissible), to be published in the next stop/publish/start window — no restart just for
  it; the tool now refuses to act on an empty recovery; (b) a re-run of the recovery rebuilt the
  generation-2 terminals with fresh report timestamps (v4 stored no real instants), data-gov
  refused them as generation conflicts (409), and the three envelopes were disposed
  `INVALID_ENVELOPE`; the tool now takes the original terminal's instants so a re-run builds the
  same bytes. Terminal outbox: 0 pending, 0 awaiting adjudication.

## N2 — calibration verifiable and explicitly scoped

Red first, then closed (`tests/test_df_utility_harness.py`, N2 rules): zero simulations,
NaN/inf/negative rates, unknown generator, incomplete denominator (`scored + failed ≠ n_sims`),
zero scored, per-simulation digest, protocol numeric domains — all refuse to seal; transfers
to another operator, protocol, length or family do not decide (`INCONCLUSIVE_UNCALIBRATED` with
the reason); the decision is gated on the **Clopper–Pearson upper bound** at the predeclared
confidence, not the point estimate (`sims_required_for_zero(0.0125, 0.95) = 239`).

The record (`df_utility_calibration.v1`) carries: generator and whether it is a **null of no
effect** (`white_null`: independent increments, dependent levels; `ar1_features_independent_target`:
AR(1) features with an independent target — dependence without effect; `ar1_null` is a
structured positive-control-like diagnostic and is **refused** as a calibration null), the
sealed plan (`calibration_plan` in the protocol: generator, `n_sims`, `n`, bound confidence —
fixed before running, never raised to obtain a pass), attempted/scored/failed/advances, rate,
upper bound, `alpha_adjusted`, seed, length, operator kind/spec/params, protocol base identity,
family, margin, blocks, window, target, model, **every simulation** (index, seed, outcome,
delta) under a digest, the harness digest, cost. `with_calibration` keeps the whole record; the
consumer recounts and verifies scope and identity.

## N3 — governance before the work

`run_rehearsal(cfg, gov, trace, isolated)`: freeze-pre write-once → **register the calibration
campaign** → per operator `before_run` → isolated calibration child (same ceilings; the record is
its verified output; its terminal carries the child's real instants and cost) → mechanics in an
isolated child (rehearsal) or the verified cells (pilot) → seal per operator → **register the
contrasts campaign** → per contrast `before_run` → isolated child → terminal with the child's
instants and cost → reconcile → envelope. Registrations are persisted (`CAMPAIGNS.json`) and
reused on resume; a root frozen under another code identity is not resumed; completed attempts
are not re-run and rebuild identical terminals.

`tests/test_df_utility_run_order.py` — **4 rules** with a stub governance and stub children,
order observed through callbacks: a registration refusal starts no child; `register` precedes
every child and `before_run` precedes each; instants and costs are the children's; a resume
rebuilds identical terminals; another code identity refuses. Proved in production too:
`utilreh-v5` (attempt 1 fell in the parent after the calibration child had completed —
regression added; attempt 2 under the fixed commit re-posted the campaign and data-gov refused
409 **before any child**); kept as evidence.

## N4 — governed rehearsal and descriptive pilot

**Rehearsal `utilreh-v7`** (root `utility_rehearsal_v7/`, code `be1f170`; v5 and v6 kept as evidence — v5:
parent crash after the calibration child + 409 on a cross-commit resume; v6: whole chain in order,
every terminal refused `invalid started_at` because the runner records `+00:00` and data-gov
takes `Z`, fixed and disposed):

| step | outcome |
|---|---|
| calibration campaign `utilreh-v7-utility-calibration` | registered before any child; 3 isolated calibration children, 239 simulations each on the exchangeable null at n = 2400: `mad_extremes` 0/239, upper bound **0.01246** ≤ 0.0125 (54.4 s CPU); `delta_run_length` 0/239, 0.01246 (50.5 s); `cusum` 1/239, **0.01969** > 0.0125 (175.4 s) — decides nothing; terminals with the children's instants and costs, reconciled `[]` |
| mechanics (eligibility) | isolated child; all three `MECHANICALLY_ACCEPTED` on the fabricated series |
| contrasts campaign `utilreh-v7-utility-contrasts` | registered before any contrast; `before_run` before each child |
| contrasts | `mad_extremes` **DOES_NOT_ADVANCE** (Δ = −0.1887), `delta_run_length` **DOES_NOT_ADVANCE** (Δ = +0.0196), `cusum` **INCONCLUSIVE_UNCALIBRATED** (Δ = −0.1888, descriptive — its own record's bound exceeds α/4), slow control **RESOURCE_EXCEEDED** (`WALL_TIME_LIMIT`, 6.0 s, no partial score) |
| reconciliation | both campaigns `missing_units: []`, 0 pending |
| content (`N4_UTILREH_V7_CONTENT_CHECK.json`) | **all equal**: every calibration and contrast metric the cube holds equals the verified file, value by value |
| envelope | DEVELOPMENT `2c1b7d31…`, loaded |

Descriptive of that fabricated series; no claim about representations.

PILOT_PLACEHOLDER

## N5 — closure

SUITES_PLACEHOLDER

Open, with owners: the empty corrective envelope's disposition (manifest ready; next
publication window); Metabase, index, terms: separate fronts.

Ending: ENDING_PLACEHOLDER
