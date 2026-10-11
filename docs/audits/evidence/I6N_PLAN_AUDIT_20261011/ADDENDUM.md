# I6-N plan audit: addendum

Date: 2026-10-11. Auditor: Satoshi. Base: audit commit `4d3af09d`. The original
`DICTAMEN.md`, `FINDINGS.json`, CSVs, `WMC_COUNTEREXAMPLES.md` and `probes/`
are unchanged and remain the record of what was first claimed. This addendum
answers the owner's six review points (order
`SATOSHI_I6N_BOUNDED_ENGINEERING_2026_10_11.md`, sha256 `243fc9e1...`, section 3).
No model was trained, no service, timer, GPU, broker or campaign was started
or stopped, and no TEST row (dated 2025-01-01 or later) was read.

**Verdict unchanged: REVISE_BEFORE_IMPLEMENTATION.** Several original claims
were overstated; the corrected record is the table below and
`ADDENDUM_FINDINGS.json`. Only AUD-02 stays CRITICAL, and only for U_h and
strategy results. No finding is withdrawn entirely; specific sentences are.

## Vocabulary

DEMONSTRATED_DEFECT: shown by retained, reproducible evidence on the concrete
object. RISK: plausible failure not yet tied to the concrete route.
OWNER_REQUIREMENT: required by the owner's contract; omission is
non-compliance. METHODOLOGICAL_ALTERNATIVE: a recommendation the owner may
decline, with its cost, not executed. Calendar-dependent numbers are labelled
with their calendar: `NY-assumed` (Forex Sunday 17:00 to Friday 17:00
America/New_York, offsets 12+12, Thursday 00:00 cut) or `observed coverage`
(first/last bar per week in the retained INPUT, which is data coverage and NOT
a confirmed market calendar). Neither authorizes production use.

## The six points

1. **Feature provenance retained.** `addendum/provenance/provenance_check.py`
   proves, from manifests identified by digest, that the 20 RAW features equal
   the union of the 14 EURUSD FS4 weekly winners. The verifier found one
   output line depended on PYTHONHASHSEED; the integrator fixed the sort key
   and retained `provenance_check_v2.out` (two seeds, identical except
   run_utc). v1 is kept as history.
2. **Development use of 2024 is legitimate if declared.** A TRAIN-only
   frontier fixed two candidate sets per target before any VALIDATION read;
   VALIDATION 2024 then made 14 binary choices. Consequences: VALIDATION 2024
   results that use these features are development evidence, not independent
   evaluation; 2023 screens are not point-in-time; TEST 2025 is untouched. The
   minimal correction is a declaration plus identity fields. A TRAIN-only
   manifest is an alternative, and no redo of the selection is ordered.
3. **'24 evaluations are not a GA' is retracted.** Per the DEAP documentation
   a genetic algorithm is defined by its operators and generational loop. N2
   is a small, budgeted genetic search seeded by N1, likely inconclusive. The
   requirement stands only as the plan's own: seal fitness, population,
   operators, selection, repair, tolerance and stop rule before fitting.
4. **The owner's early-stopping criterion stands.** The 0.5*(TRAIN + inner)
   monitor is kept. An inner-only monitor is a costed alternative (0 fits as a
   diagnostic, 16 fits as a paired refit). Seed calibration (36 fits per
   representative) is an alternative, not a requirement.
5. **Architecture tied to concrete routes with known-weight controls.**
   With all-ones and one-hot kernels, impulse inputs, and positive and
   negative controls (84/84 configurations agree with the analytic receptive
   field), t is never lost on N1 routes or stride-1 routes. Loss of t depends
   on the padding convention chosen for stride-2 reduction, which the plan
   does not fix: a RISK plus an OWNER_REQUIREMENT gap. The only demonstrated
   defect in inherited code is latent: `GroupedCausalConv1D` (a Satoshi-lane
   component) drops t on odd lengths, but all its callers use even lengths.
   Keras 'same' padding is not an equivalent fix (non-uniform, non-anchored
   grid). The earlier claim that `_residual_temporal_stage` raises on odd n is
   retracted.
6. **Calendar and strategy attribution.** Support counts are now computed
   from explicit calendar parameters for NY-assumed, observed coverage and a
   9-setting sensitivity set. The expressions (B-A)-h and
   min(cut-A, (B-A)-h) hold only for an hourly aligned grid, half-open
   intervals, no other cuts or absences and integer durations; otherwise the
   eligible timestamps are enumerated by h, role and split. For the strategy,
   the inspected blobs are distinguished from what an executor would load:
   no executor loads a strategy for I6-N today, and resolution depends on the
   interpreter (legacy site-packages copy, inspected copy, ImportError or no
   entry point). The contaminated original output is kept as history: it is a
   revision-1 run followed by an appended tail of a revision-2 run. The clean
   replay is in a new file with command, exit code and hashes of script and
   tested module. The revision timeline itself is reconstructed from an
   unretained transcript.

## Per-finding table

Full fields (risk, owner requirement, alternative, evidence) are in
`ADDENDUM_FINDINGS.json`.

| ID | After review | Severity | Demonstrated defect | Retracted / corrected | Minimal correction |
|---|---|---|---|---|---|
| I6N-AUD-01 | QUALIFIED | MAJOR | The plan (P:21) does not declare that the 20 features were chosen with VALIDATION-2024 information; identity carries only the freeze digest, no selection split/year field. | 'Contradiction' and 'selected on VALIDATION 2024' as broad selection: a TRAIN-only frontier admitted 2 sets per target (28 of 568) before any VALIDATION read; VALIDATION made 14 binary choices; 12 of 20 members enter only via RANDOM_K control sets; all 14 winners score below naive. Retracted: 'N4 letting VALIDATION choose is a defect'. | Add a provenance line and identity fields: freeze digest, selection_split=VALIDATION, selection_year=2024, ranking folds inner_2019..inner_2023, evaluation_independence=DEVELOPMENT_ONLY; label 2023 screens not point-in-time. |
| I6N-AUD-01b | NEW | MINOR | I6-E-M's 53rd segment [2024-12-30, 2025-01-01) overlaps FS4's sealed TEST week 52 [2024-12-30, 2025-01-06). | - | Declare one VALIDATION/TEST boundary used by both protocols. |
| I6N-AUD-02 | CONFIRMED (core) / QUALIFIED (scope) | CRITICAL for U_h and strategy results only | On the inspected blobs (policy 0940589, plugin 3a19b3f; identical at heuristic-strategy 79977b2, a213b65, 5c87a25): early close is unordered min/max and never consults TP, so TP@2h/SL@12h and SL@2h/TP@12h decide identically for all seven variants, buy and sell (owner expects hold vs close); no index->horizon_hours->target_timestamp map (positional Prediction_h_{i+1}); no explicit availability-mask contract (NaN drop only). | 'checkout = origin/master 5c87a25' (HEAD is 79977b2, same blobs); 'no two-long abstention' as a defect (owner-proposed rule to audit); 'no calendar' as plugin defect; entry order-blindness as defect (conforms to 3.1). | Add a strategy node before any U_h/strategy result: pin the plugin actually imported (commit, blob, import path, interpreter), implement 3.1 with tie/equality/no-crossing semantics ratified by the owner, and rerun counterexample_v2 against that pin. |
| I6N-AUD-03 | QUALIFIED | MAJOR | Grid fact: an absolute 24 h context is reachable only for h in {4,6,12} among the 12 horizons and 168 h for none (L={1,2,4,6}*h). | 'Contradicts the 2*h law' (the grid is {1,2,4,6}*h). | Declare L proportional to h by design and state what N3 can and cannot conclude. |
| I6N-AUD-04 | QUALIFIED | MAJOR | The N4 transfer rule is undefined: admissible delta_x differs by h (h=3: {3,1}; h=5: {5,1}; k=9 only for 18,36,72), while P:163 requires explicit windows. | 'Not per-horizon optimization' as a defect (P:163-164 declares the transfer scope). | Publish the transfer rule and the resulting 12 (L, delta_x, n) triples before N4. |
| I6N-AUD-05 | QUALIFIED | MAJOR | - | '24 evaluations are not a GA / local screening' (DEAP eaSimple is defined by its operators and generational loop, no minimum evaluations); the unretained '~1.6e6 configs'; 'single seed' as a defect (owner rule P:46). | Seal the numeric GA specification and the tolerance basis before any N2 fit. |
| I6N-AUD-06 | QUALIFIED | MAJOR | Latent: inherited GroupedCausalConv1D (tools/fs4_temporal_predictor.py:253-297, Satoshi lane) drops t on odd lengths, contradicting its docstring; every current caller enforces even lengths (window = 4*latent_steps), so it is not active. | '_residual_temporal_stage raises on odd n' (it builds and drops t; its caller refuses n not divisible by the factor product, not odd n in general). | Specify the padding/anchoring rule and add known-weight acceptance tests for t coverage, grid uniformity and no look-ahead on every planned route. |
| I6N-AUD-06b | NEW | MINOR | - | - | Declare receptive field per N2 configuration in its identity. |
| I6N-AUD-07 | QUALIFIED | MAJOR | - | Numbers presented without calendar attribution: '96 h', '67 entry origins', 'Mon 05:00-Tue 04:00' are NY-assumed; '1,272 eligible vs 3,712' compared E with scored rows (like-for-like is M=1,234 NY / 1,222 observed-coverage vs 3,712, of which 2,478 are not WMC-eligible); 'holiday week zero h=72' was not executed (now: 0 under observed coverage, 24 E / 11 M under NY). | Enumerate eligible timestamps by h, role and split from the versioned calendar; (B-A)-h and min(cut-A,(B-A)-h) hold only for an hourly aligned grid, half-open intervals, no other cuts or absences and integer durations. |
| I6N-AUD-08 | QUALIFIED | MAJOR | Data inconsistency: in 19 of 21 US/EU DST-gap weeks of 2019-2024 the observed session boundaries are 1 h earlier in NY time than in 2012-2018 gap weeks and in aligned weeks. | 'Bar label convention not fixed': the concrete route labels hourly bars by END (i6e_price_metrics.py:18-28; fs4_weekly_wrapper.py:271); the open-label leak scenario is withdrawn for this route. | Resolve the DST-week offset from the raw pre-2025 source before N0 sealing; state the END-label convention in the plan. |
| I6N-AUD-09 | QUALIFIED | MAJOR | - | 'Conv1D reads zeros' as a leak: on the inherited route placeholders never reach the output (0.0 with valid controls; negative control leaks). | State the placeholder, presence channel and per-route masking, including any LSTM adapter. |
| I6N-AUD-10 | QUALIFIED | MAJOR | - | 'MWC does not say whether origin or target decides'; the '58/94 h=2 rows' framing (56 of them have 2025 origins). | Declare the straddling-week rule. |
| I6N-AUD-11 | QUALIFIED | MAJOR | For I6-E (pre-contract object): Monday 00:00 UTC fit cutoffs fall after the open in 53/53 weeks (NY); post-open bars enter inner-validation targets only, not gradient rows. | 'CONTRADICTION' for I6-N; 'fits can see 86 post-open bars' (inner targets only). | Define segments from the versioned calendar; do not reuse the I6-E segmenter. |
| I6N-AUD-12 | QUALIFIED | MAJOR | - | 'Monitor inner-only' as a required correction. | Name the weeks, the update mode and the stop records; resolve the two ambiguities. |
| I6N-AUD-13 | QUALIFIED | MAJOR | - | 'S_h unpaired / no intersection rule'. | Specify aggregation, failure handling and the recipe map (C2 implements the pairing contract). |
| I6N-AUD-14 | QUALIFIED | MAJOR | Four values are code-only: huber_delta, min_delta, max_seconds, max_updates. | 'Hyperparameters absent' (specified by reference at I6E:117-120); 'contradicts FLP'. | State the four values and mark the recipe provisional. |
| I6N-AUD-15 | CONFIRMED | MAJOR (N4-M only) | w_h and s_h values are absent. | - | Declare w_h and s_h from TRAIN eligible labels before N4-M. |
| I6N-AUD-16 | QUALIFIED | MAJOR | Inherited transformer_conv refuses blocks=0 (blocks=0 repair unspecified). | 'Aliases for delta_x>=7' (understated); 't invariant to L' as a requirement. | Specify encoding, order, norm placement and the blocks=0 rule. |
| I6N-AUD-17 | QUALIFIED | MINOR | - | Model-level defect: the flag is inert under the last-state head, not wrong. | Drop 'return_sequences=True' as a behavioural claim or add a readout gene. |
| I6N-AUD-18 | QUALIFIED | MAJOR | - | 'Nothing can consume' becomes 'no existing component consumes'. | Specify genotype schema and canonicalizer; name the missing components. |
| I6N-AUD-19 | QUALIFIED | MAJOR | P:165 gives no TRAIN criterion for ARIMA resolution. | ARIMA(1,1,0) as a defect (owner order I6E:111); weekend-irregularity claim. | Seal the resolution criterion and convergence policy. |
| I6N-AUD-20 | CONFIRMED | MAJOR (risk) | - | - | Durable PAUSED/SUPERSEDED state checked by claim(). |
| I6N-AUD-21 | QUALIFIED | MAJOR (risk) | Two placement authorities conflict in static code. | Snapshot evidence not retained. | Name one placement authority. |
| I6N-AUD-22 | QUALIFIED | MAJOR (risk) | GPU.lock is per campaign root. | Timer observation not retained. | Host-level lease across campaigns. |
| I6N-AUD-23 | QUALIFIED | MAJOR (risk) | - | 'VERIFIED' -> OBSERVED_SNAPSHOT_NOT_RETAINED. | Admit against live available memory. |
| I6N-AUD-24 | CONFIRMED | MAJOR | The i6m factory caps n<=241 and refuses grid n=2, 289, 433. | - | Pilot before admitting n>241. |
| I6N-AUD-25 | CONFIRMED | MAJOR | Existing components lack stop-drain, stale-progress watchdog, DEAP-capable queue, typed failure classes. | - | Specify them for the I6-N executor. |
| I6N-AUD-26 | CONFIRMED | MAJOR | The plan lists no identity fields; I6-E identity lacks protocol fields. | - | List identity fields (contract, calendar, offsets, cut, bar convention, mask digest, manifest provenance, row digests). |
| I6N-AUD-27 | CONFIRMED | MAJOR | OLAP has no S_h/U_h population table or custody fields. | - | Population fact table and custody mapping (C2 emits compatible records). |
| I6N-AUD-28 | QUALIFIED | MAJOR | Pilots, annual control under new labels, ARIMA TRAIN choice and recounts are outside the ceilings. | Seed calibration as an unbudgeted required cost. | Itemize every cost class. |
| I6N-AUD-29 | QUALIFIED | MINOR | - | 'Never binds' without calendar qualification. | State the rule's effect per calendar setting. |
| I6N-AUD-30 | CONFIRMED | MINOR | Inherited reducer/adapter details unstated. | - | State inherited values. |
| I6N-AUD-31 | CONFIRMED | MINOR | Copy rule conflict P:200-201 vs CURRENT_EXECUTION:106-108. | - | Align the copy rule. |
| I6N-AUD-32 | QUALIFIED | INFO | - | 'VERIFIED' -> VERIFIED_AT_AUDIT_TIME_NOT_RETAINED (inline commands). | None. |

## Changes to CORRECTIONS_REQUIRED.md (read with this addendum)

- Step 3: the minimum is the provenance declaration and identity fields; the
  TRAIN-only manifest is an alternative.
- Step 4: specify the padding/anchoring rule and known-weight tests; the
  origin-anchored convention is a recommendation, not the only admissible fix.
- Step 5: the inner-only monitor, seed calibration, LR grid, absolute L
  values and N2 genes for transform/output are alternatives with cost. The
  sealed GA specification, transfer triples, w_h/s_h, the four code-only fit
  values and the provisional-recipe declaration remain requirements.
- Step 1: add resolution of the DST-week session offset from the raw
  pre-2025 source and the straddling-week rule.

## Conflict of interest

`GroupedCausalConv1D`, `fs4_temporal_predictor.py` and the FS4 weekly winners
manifest come from Satoshi's lanes. Findings on them are reported against
that lane and rest on executed, retained checks verified by separate agents.
The C1/C2 components ordered in parallel are written by Satoshi's agents and
are reviewed by separate agents; this addendum does not certify them.

## Limits and open items

- Broker/instrument calendar: unverified. All counts are NY-assumed or
  observed coverage.
- The DST-week offset cause (clock or session) needs the raw 5-minute source.
- Strategy resolution on worker hosts: unverified (no ssh sweep).
- Live warehouse readback: no token-free read path.
- `WEEKLY_PLAN.json` in the FS4 preliminary state is rewritten in place every
  few minutes (built_utc only; content digest unchanged). The writer was not
  identified or stopped.
- DEAP quotes come from the fetch tool; re-check before external citation.
