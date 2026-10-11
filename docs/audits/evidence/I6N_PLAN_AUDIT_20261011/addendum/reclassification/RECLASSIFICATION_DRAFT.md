# I6-N plan audit addendum: reclassification lane (draft for the integrator)

Date: 2026-10-11. Lane: reclassification (owner review points 2, 3 and 4, plus provisional classes for the rest).
Original audit: `docs/audits/evidence/I6N_PLAN_AUDIT_20261011` at commit `4d3af09d`. This lane did not modify it.
Object: the audited worktree on `codex/i6d-weekly-walk-forward-20261009`, an uncommitted draft over `269c2d13`. Its digests are re-printed in `quote_check.out`; the proposal is `d4341e72...`.

**Verdict: REVISE_BEFORE_IMPLEMENTATION, unchanged.** The audit's proposed solutions are not accepted automatically. No model was trained. No service, timer, GPU or broker was touched. No TEST row, market data, credential or token was read.

## Vocabulary

- **DEMONSTRATED_DEFECT**: shown by retained, reproducible evidence on the concrete object.
- **RISK**: a plausible failure not yet tied to the concrete route, or one whose evidence was not retained.
- **OWNER_REQUIREMENT**: the owner's written contract requires it. Omitting it is non-compliance.
- **METHODOLOGICAL_ALTERNATIVE**: a recommendation. The owner may decline it, and declining is never a compliance failure.
- **AMBIGUITY_FOR_OWNER**: genuine contract ambiguity. The audit records it and proposes no replacement.

Document aliases used below:

- P = proposal
- WMC = market-week contract `604f9480`
- BWC = weekly contract `d64206b8`
- FLP = FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md `5032acc6`
- PMS = PROJECT_METHOD_STATE.json `ee8a23f2`
- I6E = I6E multi-horizon plan `129cea8c`

## Retained check of this lane

| Item | Value |
| --- | --- |
| Script | `quote_check.sh`, sha256 `a4935ed0b5dd78ee3c394f886d00708cf4c404235d8d51f1e3da69fc8bea0590` |
| Command | `CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 512M -t 5m -n i6n-add-reclassification -- bash -c 'bash ./quote_check.sh "$0" "$1" > quote_check.out' <OBJ> <AUD>`, run from this directory. OBJ is the audited worktree root; AUD is the original audit directory. |
| Inputs | Every quoted document and probe file, with its sha256 printed at the top of the output. These are documents only: no data rows. |
| Output | `quote_check.out`, sha256 `5b0a6796cb4565d1289ed16b5d9e239417fb68393f091a86b18d1ec5e37570e6`. It holds the exact quoted line ranges and the evidence-retention facts below. |
| Run date | 2026-10-11 |

One earlier run redirected crispdm-run's own stdout and produced a truncated file. That file was discarded. The retained output is byte-identical to a direct run.

## Mandatory corrections from the owner review

1. **AUD-05: genetic algorithm.** The claim that 24 evaluations "amount to local screening, not a GA" is retracted.
   - The DEAP documentation defines the algorithm by its explicit operators and loop. Quoted from the documentation pages, fetched 2026-10-11:
     - "DEAP is a novel evolutionary computation framework for rapid prototyping and testing of ideas. It seeks to make algorithms explicit and data structures transparent."
     - `eaSimple(population, toolbox, cxpb, mutpb, ngen)` is described as "First, it evaluates the individuals with an invalid fitness. Second, it enters the generational loop where the selection procedure is applied to entirely replace the parental population."
   - Population size and `ngen` are free parameters, and there is no minimum number of evaluations. N2 is therefore a **small, budgeted genetic search seeded by N1**, likely inconclusive.
   - What remains required is that fitness, population, generations, selection, operators, repair, DEAP seed, tolerance and stopping rule be explicit.
     - Tolerance and stopping are already the plan's own requirement (P:140-142: "Se sella la tolerancia y regla de parada antes del fit"), so they are an OWNER_REQUIREMENT.
     - The owner's review endorses requiring the rest.
   - The "~1.6e6 configs" figure appears in no retained output and is withdrawn.

2. **AUD-05: seed calibration.** Calibrating with 3 seeds × 3 configs × 4 folds is a METHODOLOGICAL_ALTERNATIVE, not a requirement.
   - Cost: 36 fits per representative, or 72 for h=4 and h=72. That is 12.2% or 24.3% of the 296-fit N1-N3 ceiling.
   - It is justified only if the tolerance is meant to exceed training-seed noise.
   - Zero-fit bases exist: the paired between-fold dispersion of the N1 configs, or a declared practical resolution (FLP:107-110).
   - The owner's rule P:46 (single seed, no identical repetitions) stands.

3. **AUD-12 and AUD-14: early stopping.** The monitor is the owner's choice and the audit does not replace it.
   - Owner choice: `0.5*(TRAIN + inner)`, patience 5, best-weight restore (P:142, and already in I6E:120).
   - "Monitor inner-only" is recast as an optional contrast:
     - At zero fits, the evaluator history already logs `train_loss` and `validation_loss` per epoch, so the epoch an inner-only monitor would pick can be reported without refits.
     - As a paired refit with `monitor="validation_loss"` (the evaluator supports it at line 395): 2 configs × 2 representatives × 4 folds = 16 fits, or 5.4% of 296.
   - Genuine ambiguities, recorded for the owner only:
     - **Weighting:** the implementation is an unweighted mean of the epoch-running TRAIN loss and the end-of-epoch inner loss. Is that what "promedio" means?
     - **Which "inner":** the per-fit four-week chronological tail (I6E:117-118; `i6m_data.py:51`) and the "cuatro semanas internas de 2023" screening weeks (P:103) share one name.
     - **BWC:69:** "La selección de checkpoint usa una cola cronológica interna" sits beside a monitor that also includes the TRAIN term. The owner's explicit choice stands.

4. **AUD-14: learning rate and FLP.** There is no direct contradiction.
   - FLP is the registered `financial_loss_policy` (PMS). FIN-LOSS-OPT is a separate task with status DESIGNED.
   - FLP:8 makes the MAE/Huber × Adam/AdamW comparison mandatory in financial forecasting.
   - FLP:23-25 and FLP:50-52 place that comparison **before the recipe is frozen** for the R0/R1/R2 and representation contrasts.
   - FLP:83-84 tunes LR/decay (and delta for Huber) on train plus inner validation only.
   - P:143 uses the inherited recipe "inicialmente", and I6-N sits before R1/R2 (P:6).
   - The resulting OWNER_REQUIREMENT: declare the I6-N recipe provisional, and do not freeze it or present it as the financial recipe before FIN-LOSS-OPT.
   - The `{3e-4, 1e-3, 3e-3}` LR check is a METHODOLOGICAL_ALTERNATIVE costing 32 extra fits (10.8%).
   - Open for the owner: whether FLP:8 requires the factorial inside I6-N itself.

5. **AUD-01: feature provenance.** The finding is reframed. Using 2024 for development is allowed (BWC:45-47: the procedure is selected on validation and frozen before test).
   - What follows from it:
     - The I6-N VALIDATION 2024 results cannot be presented as an independent evaluation (BWC BW12, line 149).
     - The 2023 screens are not a point-in-time simulation.
     - TEST 2025 stays untouched and is the only confirmation.
   - No automatic redo of the selection is ordered. A TRAIN-only manifest is an optional separate arm.
   - The exact set-equality evidence (X-A5) was an inline check. The provenance lane must retain it.

6. **AUD-06, 07, 08, 09, 10, 11, 16, 17, 29 and AUD-02.** These are marked pending lane evidence. The classes given for them here are provisional.

## Table per finding

| ID | Before | After | Demonstrated defect / owner requirement (short) | Recast as risk, alternative or ambiguity | Minimal correction | Pending lane |
| --- | --- | --- | --- | --- | --- | --- |
| AUD-01 | CRIT CONTRADICTION | QUALIFIED, MAJOR | P:21 does not declare provenance while I6A:51 admits "selected using later validation". BW12 forbids calling VALIDATION 2024 independent. | The exact equality is unretained (RISK). A TRAIN-only manifest is an ALT. | Provenance line and identity field. Label VALIDATION 2024 as development and the 2023 screens as not point-in-time. TEST is the confirmation. No redo. | provenance |
| AUD-02 | CRIT NO_IMPL | QUALIFIED, CRIT for U_h only | Variants A and C: TP-first equals SL-first on the inspected file. P:34-35 requires verifying the effective plugin. | Output not reproducible from the retained script. B and D-G are uninformative. Loaded plugin unresolved. | Strategy node that pins the plugin actually imported, before any U_h result. | strategy |
| AUD-03 | CRIT NO_ESP | QUALIFIED, MAJOR | Grid fact: 24 h is unreachable for h=2,3,5 and 168 h for every h. | Sequential N2 then N3 is a RISK. Genes and absolute L are ALTs. MASTER:315 is a default (ambiguity). | Declare L as h-proportional by design and state what N3 can conclude. | none |
| AUD-04 | CRIT NO_ESP | QUALIFIED, MAJOR | The transfer rule is undefined for non-representative h. P:163 requires explicit windows. | "Not per-h optimization" retracted: it is a declared scope. | Rule plus the 12 (L, delta_x, n) triples before N4. | calendar (support) |
| AUD-05 | CRIT NO_ESP | QUALIFIED, MAJOR | Tolerance and stop have no values (OWNER_REQ, P:140-142). Fitness, population, operators, repair and DEAP seed are absent. | "Not a GA" retracted. Seed calibration is an ALT (36 or 72 fits). Single seed is owner rule P:46. | Seal the numeric GA specification and the tolerance basis. | none |
| AUD-06 | CRIT NO_ESP | QUALIFIED, MAJOR | P:128-130 requires coverage of t but gives no alignment rule. | Synthetic, random-weight probe: a RISK. Known-weight controls are needed. | Per-mode padding and alignment rule, tested with known weights on the actual route. | architecture |
| AUD-07 | MAJ AMBIGUO | QUALIFIED | Support table already required (WMC:122, P:65-68). | Counts use an assumed calendar (RISK). "Poor representative" is a judgement. | Build the support table from the versioned calendar. | calendar |
| AUD-08 | MAJ AMBIGUO | QUALIFIED | WMC:53-55 requires a versioned calendar. | Label convention is ambiguous. The DST anomaly uses an assumed open time. | Declare and test the bar label. | calendar |
| AUD-09 | MAJ NO_ESP | QUALIFIED | WMC:61-67 requires masks, distances and no invented prices; P:37 is silent on placeholders. | Fractions use the assumed calendar. Bit-identical invariance is an ALT. | State placeholder, presence and per-route mask. | calendar / alignment |
| AUD-10 | MAJ NO_ESP | CONFIRMED | TEST is sealed (P:43; BWC:123-125). | WMC:77-78 is ambiguous on origin vs target. | Declare the split-membership rule at every boundary. | calendar |
| AUD-11 | MAJ CONTRADICTION | QUALIFIED | P:43 copies the I6-E "53 segmentos" with an undeclared basis. | P:40-41 already adopts the closure cutoff, so "contradiction" is overstated. Reusing the I6-E segmenter is a RISK. | Derive the segments and screening weeks from the calendar. | calendar |
| AUD-12 | MAJ AMBIGUO | QUALIFIED | update_mode is mandatory (BWC:59). Weeks must be sealed (P:104). The owner's monitor is kept. | Inner-only monitor is an ALT (0 or 16 fits). Three ambiguities in "promedio" and "inner". | Name the weeks, declare update_mode, confirm the monitor as written. | none |
| AUD-13 | MAJ NO_ESP | QUALIFIED | WMC:182-205 defines S_h and U_h. P:68 already seals common support. | "Unpaired" partly retracted. Fold aggregation and failure rule are ambiguous. | State aggregation, failure rule and the recipe-per-h map. | none |
| AUD-14 | MAJ NO_ESP | QUALIFIED | Four code-only fit values. FLP ordering and labelling (OWNER_REQ). BWC:159-163 stop records. | "Absent" and "contradicts" corrected. "Favours capacity" is a RISK. LR grid is an ALT (32 fits). | List the values. Recipe PROVISIONAL_PENDING_FIN_LOSS_OPT. Record stop reason and updates. | none |
| AUD-15 | MAJ NO_ESP | QUALIFIED | WMC:100,104 require declared w_h and s_h; values are missing. | The 3x gradient figure is calendar-derived (RISK). | Seal before N4-M only. | none |
| AUD-16 | MAJ AMBIGUO | QUALIFIED | none | Order and naming are ambiguous. The ClockView aliasing is unretained (probe computed arithmetic only). | One order rule and the positional encoding formula. | architecture |
| AUD-17 | MAJ AMBIGUO | QUALIFIED, MINOR | Retained, with weights copied: return_sequences=True plus last step equals False under P:92. | Readout gene is an ALT. | Declare the readout or note the equivalence. | architecture |
| AUD-18 | MAJ NO_ESP | QUALIFIED | P:133-135 requires inactive-gene exclusion but gives no canonicalizer. | Existing components cannot consume the genotype (RISK, static). | Genotype schema and canonicalizer. | none |
| AUD-19 | MAJ NO_ESP | QUALIFIED | P:165 has no TRAIN selection criterion. | ARIMA(1,1,0) is the owner's order (I6E:111). "Cannot produce" becomes a RISK. 46/128 unretained. | Selection criterion and failure policy. | none |
| AUD-20 | MAJ NO_IMPL | CONFIRMED (RISK) | P:12-17 records the pause and keeps the terminals. | Durability is operational, from static code. | Fate of QUEUED jobs; durable pause or a written ban on redeploying. | none |
| AUD-21 | MAJ AMBIGUO | QUALIFIED | none | Snapshot not retained (X-F1). | Name one placement authority. | none |
| AUD-22 | MAJ NO_ESP | QUALIFIED | P:197 requires one child per host. | Timer observation not retained. | State how this is enforced across campaigns. | none |
| AUD-23 | MAJ VERIFIED | QUALIFIED | P:72 requires 1.25 × peak. | "VERIFIED" becomes OBSERVED_SNAPSHOT_NOT_RETAINED. | Admit against live memory. | none |
| AUD-24 | MAJ NOT_MEASURED | CONFIRMED | P:72 and P:207 require pilots. | The estimate is unmeasured. | Pilot before n>241. | none |
| AUD-25 | MAJ NO_IMPL | CONFIRMED | P:192-194 automation; P:166 typed failures. | Missing in existing components. | Define the failure classes. | none |
| AUD-26 | MAJ NO_IMPL | CONFIRMED | WMC:210-211 and WMC10 put the contract in identity. | Old terminals could be adopted (RISK). | List identity fields, including manifest provenance. | none |
| AUD-27 | MAJ NO_ESP | CONFIRMED | WMC:201 requires S_h/U_h counts and digests in the OLAP. | Schema design is an ALT. | Name where they are stored and read back. | none |
| AUD-28 | MAJ AMBIGUO | QUALIFIED | Class pilots and other items are not budgeted. | "Seed calibration (none planned)" retracted as an unbudgeted cost. | Itemize the remaining costs. | none |
| AUD-29 | MIN VERIFIED | QUALIFIED | WMC:150-152 proposes the rule for audit. | Holds only under the assumed calendar. | Declare that entry geometry varies by origin. | calendar |
| AUD-30 | MIN NO_ESP | CONFIRMED | Inherited values are unstated. | none | State them or cite where they are written. | none |
| AUD-31 | MIN AMBIGUO | CONFIRMED | none | Copy rule: P:200 vs CURRENT_EXECUTION:106-108. | Align the copy rule. | none |
| AUD-32 | INFO VERIFIED | QUALIFIED | none | Inline X-F1, not retained. | Retain the query if cited again. | none |

Totals: 24 QUALIFIED and 8 CONFIRMED. No whole finding is retracted; 34 specific claims are retracted or recast (listed in the JSON `overclaims`). Of the six original CRITICAL findings, only AUD-02 stays CRITICAL after review, and only for U_h and strategy results, provisionally until the strategy lane reports.

## Other overclaims found

- **Critical findings rest on unretained evidence.**
  - EVIDENCE_MANIFEST X-A5 (feature equality) and X-F1 (resource snapshot, QUEUE.db counts) are inline checks with no script or output.
  - AUD-01 (CRITICAL), AUD-19, AUD-21, AUD-22, AUD-23 and AUD-32 cite them.
- **The AUD-02 output does not match its script** (`probes/a/strategy_counterexample.out`, 24 lines):
  - Lines 17-19 repeat as lines 21-23.
  - Line 20 (`calculate_entry_geometry signature has no short argument: False`) is printed by no line of the retained script.
  - The file concatenates at least two runs of two script versions.
  - The duplicated variant lines between the LONG and SHORT blocks are expected, because their text omits the direction.
  - Only variants A and C ever trigger a close, so only they can show time-order blindness.
- **AUD-06** used `Conv1D(4, kernel=2, strides=2)` with random seeded weights (`check.py:20,22`). A "no change" result there needs a known-weight control.
- **AUD-16:** the "ClockView" origin position is `(n-1)*spacing` arithmetic (`check.py:52`), not an execution of `i6e_masked_model.py`.
- **CORRECTIONS_REQUIRED overstates what is required.**
  - Step 3 puts the TRAIN-only manifest first.
  - Step 5 lists seed calibration, the inner-only monitor and an LR check as required.
  - Only the provenance declaration and the sealed tolerance with its basis are required. The rest are alternatives.
- **DICTAMEN:** "contains one methodological contradiction (I6N-AUD-01)" becomes "undeclared validation-informed feature provenance".

## Open for the integrator

- Reconcile the provisional classes with these evidence lanes:
  - provenance: AUD-01
  - strategy / effective plugin: AUD-02
  - calendar: AUD-07, 08, 09, 10, 11, 29
  - architecture with known weights: AUD-06, 16, 17
- Owner ambiguities collected:
  - **Monitor:** weighting and which "inner" (AUD-12).
  - **FLP:** whether FLP:8 applies inside I6-N (AUD-14).
  - **Window default:** whether MASTER:315 binds the I6-N window grid (AUD-03).
  - **Split membership:** origin or target decides (AUD-10).
  - **Fitness:** fold aggregation and the consumed recipe (AUD-13).
  - **Operations:** fate of the 3,762 QUEUED jobs (AUD-20), the placement authority (AUD-21) and the copy rule (AUD-31).
