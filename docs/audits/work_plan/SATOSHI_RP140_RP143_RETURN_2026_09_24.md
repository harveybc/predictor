# RP140–RP143: protocol A consolidated with a four-horizon mean, B at ten of twelve, and the matched ECL contrast made executable

Orders [RP140–RP143](../../handoffs/MUSASHI_SOTA_RP140_RP143_2026_09_23.md) at `5c9a14d5`, after the RP136–RP139 review.
Branch `satoshi/rp132-rp134-20260923`, preserved separate. Experimental results first; software is section 5.

## 1. Protocol A is consolidated: twelve cells pooled, four horizons, a mean for the first time

The composition binds the four original-device replays, the retained replay histories and Musashi's live custody audit to the
authoritative root by identity. Every cell now passes the stated pooling rule, so the four-horizon mean exists:

| horizon | reproduced MSE / MAE (mean over 3 seeds) | seed SD (MSE) | published | difference (MSE) | status |
|---|---|---|---|---|---|
| 96 | 0.135496 / 0.232884 | 0.003973 | 0.133 / 0.230 | +0.00250 | OPERATIONAL_AGREEMENT |
| 192 | 0.157645 / 0.252163 | 0.003799 | 0.154 / 0.248 | +0.00365 | OPERATIONAL_AGREEMENT |
| 336 | 0.164480 / 0.262985 | 0.003200 | 0.162 / 0.261 | +0.00248 | OPERATIONAL_AGREEMENT |
| 720 | 0.190226 / 0.290617 | 0.006937 | 0.184 / 0.284 | +0.00623 | OPERATIONAL_AGREEMENT |
| **four-horizon mean** | **0.161962 / 0.259662** | — | 0.158250 / 0.255750 | +0.003712 | computed over a denominator of four |

Matched persistence on the same rows and windows, from each cell's retained catalog: naive z-MAE 0.9455, 0.9507, 0.9613 and
0.9754 at T = 96, 192, 336 and 720. The agreement rule is the predeclared operational band, **not** statistical equivalence
and not exact equality to a rounded published table.

**The qualification that travels with every row.** No cell of this campaign records a MEASURED training-device UUID:
attribution is UNKNOWN for four cells and INFERRED_GPU_MEMORY for eight. Exact reproduction on an observed device is
repeatability of the stored predictions under a reloaded checkpoint, and it does not establish which physical device produced
them. The review made that point about the coordinator cell; the composition shows it holds for all twelve, and my earlier
"on their original device" wording is corrected in a dated errata beside the original.

## 2. Protocol B: ten of twelve cells trained and accepted

The campaign runs on its existing service; I supervised and launched nothing. Measured from its own accepted terminals:

| horizon | seeds accepted | author float32 MSE | author float32 MAE |
|---|---|---|---|
| 96 | 3 | 0.125551, 0.125849, 0.126375 | 0.220453, 0.220934, 0.221487 |
| 192 | 3 | 0.144051, 0.142953, 0.143947 | 0.238342, 0.237668, 0.237899 |
| 336 | 3 | 0.152274, 0.152304, 0.153511 | 0.250876, 0.251083, 0.252075 |
| 720 | 1 of 3 | 0.179536 | 0.275546 |

Two cells remain: seed 2022 is training and seed 2023 is queued. These are recorded scores, not accepted science: no
same-device replay, no closure, no verified horizon mean. Table 9 publishes 0.126 / 0.220 at H96 and its per-horizon searched
lookback is unresolved, so it is not an exact matched comparator and the closeness of these numbers is not agreement. A
versus B will be labelled a multi-parameter RECIPE comparison, never a context effect.

**Allocation, measured before each remaining child.** Continuation scope 11,614 CPU s of the declared 24,000 at 01:34Z, 48 %.
Measured per cell: 1,258 s at H96, 1,341 at H192, 1,362 at H336. Two cells remain, so the allocation fits with roughly 45 %
headroom, and the 8 h wall window that opened at 22:20Z has about 5 h left. Memory: the scope's peak is 5.40 GiB against the
7 GiB Musashi raised it to, with `oom` and `oom_kill` both zero.

## 3. The matched ECL contrast is now executable, and its pilot ran

The review is right that the existing E1 pilot is not an ECL adapter: it predicts one channel at one offset while the
reference evaluates horizon by 321 channels. `tools/df_ecl_modular.py` is the matched path.

- **Task identity is the reference's by construction.** The arguments come from a sealed protocol-A cell of the same horizon,
  so the split boundaries, the scaler fit population, the window geometry and the channel order are the author's. The targets
  are compared with the reference's own through a second code path and hash identically over the test windows.
- **The model is the approved architecture with one change**: the readout emits `Dense(pred_len x 321)` reshaped and added to
  the last observation broadcast over the horizon, which is the approved persistence-skip head generalised from one step to H.
  The detector keeps the approved names, so R0/R1/R2 apply unchanged.
- **Causality is proved over row identities, on TARGET support.** The author's val and test splits deliberately begin
  `seq_len` rows before their own boundary, so their input context reaches into the preceding split. That is his published
  protocol; it is declared rather than hidden, and the disjointness proved is of target support, which is what a fit could
  leak through. The auto-encoder's internal validation is a chronological tail of the outer TRAIN origins purged by
  `seq_len + pred_len`, never the outer validation. A future perturbation leaves each window's tensors unchanged. The split
  boundaries are re-derived with the author's own formulas and checked against his own split arrays.
- **The obsolete rule is corrected twice**: in the preparation with a dated note, and in a dated successor to 13E that erases
  nothing.

The authorised **TRAIN-ONLY pilot** ran on WORKER_A while B held the 5090:

| quantity | measured |
|---|---|
| cost probe, after warm-up | 0.0279 s per AE step, 0.0288 s per R0 step |
| auto-encoder | 23,797 steps over 53 passes, 361 s; masked loss 0.35999 to 0.14284, purged inner validation 0.25378 to 0.16439 |
| R0 | 23,329 steps over 41 passes, 448 s; loss 0.44293 to 0.29182 |
| regimes | R1 imports and freezes the detector, which receives no gradient; R2 imports the same bytes and does receive one |
| cost | 1,206.1 CPU s and 820.1 s wall, inside the declared 1,800 / 1,800 |

It computes no test score, selects nothing on any validation and makes **no claim about H1**. Its own first attempt is
preserved beside it: that run proved the path but silently executed one pass instead of its prescription, because a finite
sequence yields at most `len(seq)` batches per epoch. The prescription is now executed as whole passes.

## 4. The financial lane, investigated without reading a byte

The blocker is stated in the service's own words: HTTP 422, "resource availability contract required". The client records six
fields per delivery and three of them have no evidence yet, including the two a causal claim depends on: when each bar became
observable, and the worst completion lag. The inspected parser's EST-as-UTC defect is a demonstrated behaviour of that parser
and **not** evidence about the producer of the retained bytes, so a timezone correction now would be a guess dressed as a
repair. There is still no matched financial reference, and the electricity work does not supply one. The reserve was not
touched and no fit was run.

## 5. The software, reported separately

`compose_evidence` binds evidence objects per cell by design, record, checkpoint, prediction, report and accepted-terminal
identity, admits a retained replay history in all three of its recorded shapes at two stated binding strengths, takes a score
from a bound report row when the record carries none, and derives the consolidated table. It reads no array, recomputes
nothing, edits no report and sets no verified flag. The refusals the orders name are tests: foreign design, changed
checkpoint, changed predictions, missing terminal, missing population, a history whose recorded checkpoint is another one,
and one that neither records an identity nor reproduces the record's metric.

Suite **146 passed** at `87db6944` on WORKER_A, clean worktree, no skips.

## 6. Cost and refusals

Measured this round on WORKER_A: about 400 CPU s of tooling tests, 146 s for the final suite, 1,206 CPU s for the pilot. On
WORKER_B: 0.95 CPU s of read-only composition. No GPU seconds on the campaign device. A 6 GiB scope was refused twice on
WORKER_A for want of free memory and I asked for less rather than forcing it. I did not run the suite on WORKER_A while the
pilot held it.

## 7. Review request

(a) The pooling rule and whether the four-horizon mean may stand on it, given that no cell has MEASURED training-device
attribution. (b) The two binding strengths for retained replay histories, in particular METRIC_AND_SHAPE_BOUND. (c) The
matched ECL adapter: the one architectural change, the target identity check, and whether the causality proof on target
support is the right statement given the author's overlapping input context. (d) The pilot's prescription rule and its first
attempt, preserved. (e) The financial contract investigation and what it says cannot proceed.
