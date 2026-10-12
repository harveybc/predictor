# I6N causal trace audit: independent review

Reviewer: independent of the author. Object: `tools/i6n_causal_trace_audit.py` and the
evidence at `5a9858c1`, with the addendum at `45a741e9`. Scope follows the owner's narrowed
order: four adversarial fail-closed mutations, plus the checks already finished before the
order was narrowed. All runs were CPU only (`CUDA_VISIBLE_DEVICES=""`), under `crispdm-run`.
No fit, no TEST row, no campaign restart, and nothing was written under a campaign root. Each
mutation ran once against its own copy of the data and campaign roots, outside the real roots.

**Verdict: FIX_REQUIRED (minor).** The fail-closed behaviour holds. One machine-readable state
claims too much (STLFSI4, section 4). One redaction gap remains (section 5). Neither affects
category (1) as the addendum states it.

## 1. Tests and reproduction

- `pytest tests/test_i6n_causal_trace_audit.py`: **16 passed**, exit 0.
- The rerun of the exact RETURN.md command into a scratch output directory exited 0 with
  `COMPLETE` and the same counts as the committed evidence: 67 PROVEN_SAFE, 9 PROVEN_LEAK and
  15 RISK_UNRESOLVED.
- `ALIGNMENT_CASES.jsonl`, `FEATURE_PROVENANCE.csv` and `TRACE_GRAPH.json` are byte-identical.
  The committed C1, C2.*, E7 and E8 rows, the champion replay and all 27 committed local
  candidates are identical.
- Some diffs are more than timestamps. All of them come from the live campaign advancing
  between runs, and none comes from non-determinism:
  - The archive census grew from 47 to 49 candidates (E1/E2/E6 49/49 exact).
  - The local fits grew from 27 to 28 (F1 SAFE).
  - E3b went from 137/188 to 144/196.
  - `WEEKLY_PLAN.json` was rewritten by a live process at 21:50Z. Its digest moved from
    `7688a643…` to `88321930…`, and the winners and validation weeks are unchanged.

## 2. Fail-closed mutations

The audited origin was t = 2023-03-22T12Z, the "ordinary" case of fold 2023Q1_DCA. The
column-0 sentinel was +1e6. Mutated inputs were re-signed so that the run gets past the digest
gates. The tensor check called the campaign's own `cell.prepare` and `cell.windows` directly
on the baseline copy and each mutated copy.

| Mutation | Exit | Terminal / changed states | Direct check |
|---|---|---|---|
| M1a: row t+2h (the next bar after t) mutated | 0 | COMPLETE. B2, B3, A5×3, D1×3 and E3 go to RISK_UNRESOLVED, because the bytes and support no longer match the pinned ones. | Q1 window at t is **byte-identical**. Q1 scaler and target scale are identical. The Q2 scaler changes, because that row is Q2 TRAIN. |
| M1b: row at t mutated | 0 | COMPLETE. Same digest flags as M1a. | **Only the rightmost slot changes.** Earlier slots are identical, and the Q1 scaler is identical. |
| M2: one naive origin id shifted +1 h (Q1 candidate 180be682, index 5; count kept at 208) | 0 | COMPLETE. **F1 goes from PROVEN_SAFE to PROVEN_LEAK** (parity break). | — |
| M3: `CLOCK.json` `input_sha256` changed by one hex digit | **1** | **COMPLETE_WITH_PHASE_ERRORS**. A6×3 go to RISK_UNRESOLVED. The data, features, folds, archive and alignment phases are refused with ERR.* = RISK_UNRESOLVED ("source_clock digest mismatch"). No SAFE verdict is emitted downstream. | — |
| M4: Q1 fold cutoff moved one week later (7 INNER days become TRAIN) | 0 | COMPLETE. D2 flagged (recomputed 03-31T21Z, frozen 04-07T21Z). D3 flagged (train 18652/18695, inner 213/208). A5 and D1 flagged. **E3: the fitted scaler differs from the archive on 57 fold fits.** G1 goes to RISK_UNRESOLVED. | — |

**Does any result refute category (1)?** No. No mutation produced a direct leak from rows
after t into inputs, targets, masks or scalers. Every boundary violation I injected was
reported and none was passed as SAFE.

## 3. Headline findings re-derived without the runner

- **Feature selection (C2).** The union of the 14 EURUSD RAW winners in `TEST_FREEZE.json`
  (sha `f9ac4879…`) contains exactly the 20 `CLOCK.json` features, in the same order.
  `WEEKLY_PLAN.json` has 52 VALIDATION weeks, from 2024-01-01 to 2024-12-30. This confirms
  FUTURE_RELATIVE_TO_2023_FOLDS.
- **Fitness (E7/D7).** The plan defines `per_fold = 0.5*(err_TRAIN/naive_TRAIN +
  err_INNER/naive_INNER)`. I replayed the receipt for Q1 candidate 180be682 exactly:
  1.0000191443641377. Half of each fold's fitness is therefore the in-sample TRAIN ratio. That
  supports the reading "in-sample ranking regularizer, not a leak".

## 4. Spot-checks

- **px.log_co, px.close_loc and fx.gbpusd.logret_1h.** The audit's fixed -5 h clock (61 %)
  undersells the match. A DST-aware vendor clock (-5 h in summer, -6 h in winter) reproduces:
  - price at 93.4 %;
  - log_co at 93.3 %;
  - close_loc at 93.3 %;
  - gbpusd logret at 93.3 %.

  This supports "bar ending at t". It stays below 0.95, so RISK_UNRESOLVED is correct and
  there is no overclaim. px.rv5 matches no tested formula (<0.4 %), which is consistent with
  the audit.
- **STLFSI4: overclaim.** I reproduced the as-of lag of +23 h at 100 %.
  - All source dates are Fridays, and the file holds a single vintage (realtime_start
    2026-05-01).
  - FRED releases this weekly index the following Thursday (external knowledge; no release
    metadata is retained). The campaign therefore uses each value about 5 days before it was
    published.
  - At +120 h only 58.6 % of hours match.
  - The addendum lists this under category (4), but `FINDINGS.csv`
    `C3.fred.stress.stlfsi4.level` still says **PROVEN_SAFE**. Under the order's rules it must
    be at least RISK_UNRESOLVED.
  - This is pre-publication availability, not use of an observation after t. The tensor
    machinery is clean, so it does not refute category (1) as the addendum frames it. It is
    the one item for the owner to rule on.

## 5. Privacy scan of the committed files

- No host names, usernames, home paths, UUIDs, IPs or tokens appear in the evidence.
- **Test fixtures:** `tests/test_i6n_causal_trace_audit.py` lines 211-213 contain a real
  island host label and sample home/IP strings as redaction fixtures. Replace them with
  neutral placeholders.
- **Redaction gap:** `TRACE_GRAPH.json` (4 edges) and `FINDINGS.csv` E9 carry the absolute
  path `<data-root>/ISLAND_2/...`. The island is redacted, but the path is not
  mapped to `<data-root>`.

## Fixes required

1. Change `C3.fred.stress.stlfsi4.level` to RISK_UNRESOLVED or a declared availability leak,
   with a release-lag bound of at least 6 days. Treat all FRED inputs as revised
   single-vintage data.
2. Redact the paths of other islands' data roots.
