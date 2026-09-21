# Household literature benchmark — preparation (point 12), nothing executed

Registry: `BENCHMARK_CONTRACT_REGISTRY.json` (every source below read **at its origin**; decisions
from fields). Reproduction config: `GASPARIN_2019_REPRODUCTION_CONFIG.json`. Runner refusal without a
contract: `tools/df_benchmark_contract.require`, called by `df_e1_phase1.run` and `df_e1_huber.validate`.

## 1. Sources found for UCI 235, and what each one actually specifies

| field | **ours** (RP55/RP63/Huber) | Gasparin, Lukovic, Alippi 2019/2022 | Saad Saoud, AlMarzouqi, Hussein 2022 | Vaygan, Rajabi, Estebsari 2021 | Kim & Cho 2019 |
|---|---|---|---|---|---|
| read at | run artifacts | arXiv:1907.09207 PDF §7.1–7.4, Tables 3–5 | arXiv:2207.02589v2 PDF §2–3, Tables 1–3, Fig. 6 | arXiv:2109.12498 PDF §III–IV, Table I | Energy 182:72–81 — **abstract only** (paywalled) |
| code revision | this branch | **not public** | not public | not public | not public |
| target | Global_active_power, kW, minute‑averaged | same, **resampled to 15 min** | same, at minutely / hourly / daily / weekly (rule unstated) | same, 1‑minute | not verifiable |
| resolution | 60 s | 900 s | 3 600 s (hourly row) | 60 s | — |
| input window | 60 steps (1 h) | 4 days = 384 steps | not stated | pools of 720 min | — |
| horizon | **60 steps = 60 min, one value** | **96 steps = 1 day, MIMO** | one step of the row's resolution, recursive multistep | **not declared** | — |
| split | DEV 28 d / 7 d inside the family train span; test never read | test = last year (35 040); train 103 301; validation = one month in five | first 3 years train / last year test | 67/23 **within pooled weeks** | — |
| missing | withdrawn | imputed by same‑slot mean across years | not stated | averaged over available data | — |
| transform / scaler | z‑score, train windows | none stated for IHEPC | SWT (db1, 3 levels) over the **whole series**, bands scaled to [0,1] | not stated | — |
| metric | MAE kW; MAE_z | RMSE, MAE averaged over N pairs × 96 steps; NRMSE% (train max/min); R² | RMSE, MAE, MAPE | RMSE, MAE (eq. 16–17) | RMSE (abstract) |
| naive | persistence at h, identical rows | **none** | none | none | — |
| models / tuning | ARCH‑A modular, tcn_w; declared settings | FNN, DFNN, TCN, ERNN, LSTM, GRU (Rec/MIMO), seq2seq TF/SG; grid search (Table 4); 10 repeats | CNN‑LSTM‑SWT, Transformer‑SWT; RMSProp 1e‑3, batch 32, 100 epochs | DRNN, TPRNN; N=720, M=14 | CNN‑LSTM |
| published value (kW) | — | Table 5: GRU‑MIMO RMSE 0.75±0.00 / MAE **0.52±0.00**; TCN 0.76 / 0.54 | Table 1 hourly: Transformer‑SWT 0.4183 / 0.2637; CNN‑LSTM 0.5957 / 0.3317 | Table I: TPRNN 0.37 / 0.19 | none public |
| **comparability with ours** | — | **NOT_COMPARABLE** (target construction 15‑min, horizon 96‑step MIMO, split, missing policy) | **NOT_COMPARABLE** and **CAUSALITY_UNVERIFIED** (whole‑series SWT before the split — the battery's `full_series_dwt_as_time_row` class; hourly target ≠ minute power 60 min ahead) | **NOT_COMPARABLE** (horizon undeclared; pooled split) | **NOT_COMPARABLE** (fields unverifiable) |

No verified matched literature value exists for our 60‑minute‑ahead minute‑power task. **None of
the numbers above enters a comparison column.** The closure table says NOT_COMPARABLE with these
reasons and the planned comparison below.

## 2. The reproduction lane (Gasparin 2019) — designed, costed, not run

Gasparin's IHEPC protocol is the only one fully specified: target construction, resolution,
window, horizon, split, missing policy, metric formulas, grid and repeats. It can be reproduced
**as their task**, and our model can be run **under their contract**, with one naive added that
they did not report. `GASPARIN_2019_REPRODUCTION_CONFIG.json` fixes every field; the runner change
it needs is a 15‑minute resampler and a 96‑step MIMO head on the existing loader — gated by its own
acceptance tests before any fit:

1. the resample reproduces their Table 3 counts (test 35 040 = 365 × 96; train 103 301) — a count
   check on the same bytes, before a model exists;
2. the same‑slot‑mean imputation is computed from **train years only** (their text says "across the
   different years"; if the intended rule uses test years it is a leak, and the reproduction is then
   labelled FORENSIC, not a corrected comparison — the two are separate objects);
3. MAE and RMSE follow their formulas (mean over pairs and over the 96 steps), NRMSE with train
   max/min, R² per window;
4. a persistence naive at 96 steps on identical rows is added as a column they do not have;
5. prefix invariance of the resampled series (the 15‑min mean of a slot reads that slot only).

Reference model: **GRU‑MIMO with Table 4's IHEPC configuration** (L=1, n_H=50, λ=0.0005, dropout
0.0), and TCN (L=8, n_H=50, k=2, M=32, λ=0.005, dropout 0.1). Ours: the modular core with a 96‑step
MIMO head, same window and rows. Ten repeated trainings as they do. **Cost**: at their sample sizes
(103 301 train windows of 384 × 1 channel) a GRU‑MIMO epoch is far cheaper than our minute task;
a 200‑update cost pilot per model precedes the campaign; projection must fit the round's ceiling with
closure reserved. Expected deliverable: the paper's metric first (MAE/RMSE kW over 96 steps), then
MAE_z and skill against the added naive — never mixed in one row.

## 3. The matched‑domain lane (our task) — the comparison that would actually answer the owner

Re‑execute Gasparin's GRU‑MIMO and TCN **under our contract** (minute power, W60, h60, DEV rows,
persistence naive) with their Table 4 settings scaled by a declared rule. That yields the only
literature comparator our closure table can ever place in the comparison column. Cost: two
architectures × 3 seeds at W=60 ≈ the phase‑1 scale (2 024 s for 9 cells). Not launched: it is a
scientific training and waits for review of this packet.

## 4. What is NOT done and is said

- No benchmark is closed. Every legacy and literature comparison remains NOT_COMPARABLE.
- The Saad Saoud recipe replicates a whole‑series wavelet preprocessing; a reproduction of it is
  forensic diagnostics only and must be identified as such, never as a corrected comparison.
- Kim & Cho's protocol cannot be transcribed from the abstract; its numbers exist here only as
  re‑executed by a third party and are recorded, not used.

## Correction dated 2026-09-21 (RP69) — preserved text above, corrected here

- The TCN row of Table 4 has **no n_H**: the line "TCN (L=8, n_H=50, k=2, M=32, …)" above transcribed a value the article does not give. The TCN's width is M=32 filters with kernel 2 and L=8 (RP66/GASPARIN_2019_REPRODUCTION_CONFIG.json v2).
- "Table 3 sample sizes" are the article's SAMPLE counts, not an enumeration of admissible windows; a count match is not proof of reproduction.
- The article is NOT fully specified (optimizer, LR, batch, epochs, initialization, regularized tensors, split dates, imputation population, seeds, code are unknown); any statement above that calls it the "only fully specified" source is withdrawn.
- The matched-domain lane (§3) is now MEASURED for the GRU family: tools/df_gru_reference.py adapted to our contract, executed as block DEV_MATCHED (SATOSHI_RP66_RP73_RETURN_2026_09_21.md): GRU adapted MAE_z 0.531273 vs modular 0.568045, persistence 0.676560, 3 paired seeds. The TCN re-execution under our contract is not run.
