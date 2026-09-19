# Errata to the RP9–RP16 return (2026-09-19) — issued under RP17/RP18

Applies to [SATOSHI_PROGRAM_RP9_RP16_RETURN_2026_09_19.md](SATOSHI_PROGRAM_RP9_RP16_RETURN_2026_09_19.md) and to
`docs/audits/evidence/d3_k5_20260917/RP14_ARCH_STAGE_EFFECTS.json` / `RP14_ARCH_STAGE_TABLES.md` (kept as
published; frozen copies under `RP17_PRE/`). Dictum: [MUSASHI_RP9_RP16_REVIEW_2026_09_19.md](MUSASHI_RP9_RP16_REVIEW_2026_09_19.md),
reproduced on the reviewed checkout `6f6c1a0` with the preserved run root: `RP17_PRE/RP16_REVIEW_REPRODUCED_PRE.json`
is byte-equal to the dictum's `results.json`.

## Withdrawn

1. **"γ is POSITIVE everywhere: the advantage is SMALLER with the planted lagged dependence."** Withdrawn. The
   published γ subtracted two different contrasts (the sequence–summary pair at r = 0; the average of four
   fusion arms at r = 1). On the SAME pair (sequence − summary, same donor, both replicates) the values are
   A −0.0749, B −0.0678, C −0.0675, 0 −0.0549 (`RP18_ARCH_STAGE_EFFECTS_v2.json`, equal to the dictum's
   recomputation to 1e-10). No claim about the sign of the balanced factorial γ is made: it is NOT ESTIMABLE
   with the measured cells (the 2 × 2 exists at r = 1 only).
2. **"Most of the sequence–summary gap is the readout, not preserving history."** Withdrawn as an attribution.
   The readout effect at r = 1 (last − pooled, balanced) is −0.08 … −0.11 and is reported as descriptive;
   the common pair still compares two complete procedures (Conv1D core vs Dense core, a donor trained for the
   sequence receiver), so it does not isolate history preservation causally.
3. **`interpretable = true` by beating the persistence naive.** Withdrawn. Adequacy is now reported apart:
   persistence naive, the MASE denominator (train seasonal-naive MAE, a denominator), the linear reference with
   its point gap AND the historic +0.03 criterion, the oracle. Under that criterion ARCH-B at H2 h3 is
   within +0.03 of the linear reference (gaps +0.0192 / +0.0178) and only fails the point value; in DX it
   exceeds the tolerance (+0.039).
4. **"ARCH-0 and ARCH-A are as good as or better than the TCN at 1/4 and 2/3 of its cost."** Withdrawn as a
   general efficiency claim. Costs are now compared on the exact intersection of 24 task × seed pairs per
   architecture, by host stratum, with the donor's cost amortised (`RP21_ARCH_STAGE_ADEQUACY.json`); 1 100
   updates with patience 8 do not show comparable convergence (best checkpoints touch the allowance in
   several H2 cells), so no ordering of architectures is asserted.
5. **"H2 — no architecture makes the grouping matter."** Rephrased: with two replicates, |e(h)| ≤ 0.015 is a
   descriptive difference; neither equivalence nor absence of effect is claimed.
6. `donor.delta` definition: it is the SAME common pair with the other donor (sequence_dsum − summary_dsum)
   − (sequence − summary), per replicate; the earlier text compared it verbally with the averaged d1.

## Kept

- The 112 measured cells, their arrays, weights, replays and terminals; the dictum recomputed 120 attempts'
  MASE from the arrays with a maximum difference of 7.77e-16 — a bounded numeric confirmation of the
  score layer, not a certification of every layer.
- The DX successor (4/4) as an adequacy diagnostic of a distinct condition; it decides no regime.
- The RP9–RP13 closure and backfill results, within the scopes the dictum states (replay cache: reuse
  outside its scope is invalidated, see RP19).

## Frozen PRE for F4/F5 (inventory and criteria before correction)

- `RP17_PRE/E1_FAMILIES_v1_PRE.json`: the v1 manifest with W = 96 / h = 1 / ≥ 200 blocks and `eligible: true`
  for the four public families; declared inadequate by the dictum (W in physical units differs per family; a
  sampling rate is not a period; blocks are not independent units; usable windows after gaps, availability
  and purge were not counted; prior exposure of the reserve candidates not declared).
- `RP17_PRE/RP14_ARCH_STAGE_EFFECTS_v1_PRE.json`, `…TABLES_v1_PRE.md`, `df_mod_e0_arch_verify_v1_PRE.py.txt`: the
  estimator and reading now withdrawn.

## 13C dependency

The OWNER_DECISION items of 13C are resolved for DEVELOPMENT by 13D (cash-spot BTC/ETH scenario, equity 1,
long/flat, no leverage, weekly UTC retraining, funding zero by definition of the cash-spot scenario only).
The real operating conditions (universe, venue, capital, lot minimums, SLA, funding of derivatives) remain
UNKNOWN and are not invented.
