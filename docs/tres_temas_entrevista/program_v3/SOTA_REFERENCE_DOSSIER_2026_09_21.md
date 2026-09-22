# RP91 — reference dossier for the full ECL321 benchmark (primary sources read on 2026-09-21)

Status: DISCOVERY CLOSED for this round; the selection below is justified by protocol-matched published values and code
availability, not by convenience. Nothing here is a measurement of ours. The household task, the custom 32-client panel
and the adapted GRU are NOT ECL reproductions and are HISTORICAL_DEV_ONLY (RP90).

## The benchmark

"Electricity"/ECL in the long-term-forecasting literature is the **processed file distributed with Autoformer/TimesNet**
(`electricity.csv`: 26 304 hourly rows × 321 clients + `date`, 2016-07-01 02:00 → 2019-07-02 01:00, no missing values),
derived from UCI ElectricityLoadDiagrams20112014 (Trindade 2015). It is served by the Time-Series-Library dataset
repository on Hugging Face (`thuml/Time-Series-Library`, revision `2b66e59ee19dac8f6f19fb5d4997f289fdfea357`,
LFS object sha256 `7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad`, 95 581 762 bytes, licence
CC-BY-4.0 per the dataset card). The distributor's processing steps (from the raw 370-client 15-minute UCI file) are not
re-derived here; the raw UCI321 panel we hold (140 256 × 370, 15-min) is a different object and is not interchangeable.

Common protocol of every paper below (TimesNet/Time-Series-Library `Dataset_Custom`): chronological split
train = int(0.7 n) = 18 412 rows, test = int(0.2 n) = 5 260 rows, vali = the remaining 2 632 rows; the vali/test
segments are preceded by L rows of context; `StandardScaler` fit on the train rows only, applied to all channels; all 321
channels are inputs and targets (`features M`); **metrics MSE and MAE in the normalized space**, mean over every test window ×
horizon step × channel; horizons T ∈ {96, 192, 336, 720}. Window counts for L = 96, T = 96: (18 317, 2 633, 5 261),
exactly the sizes the papers print — verified on the delivered file.

## Candidates and their published ECL values (MSE / MAE, normalized)

| method (venue) | protocol | 96 | 192 | 336 | 720 | avg | code |
|---|---|---|---|---|---|---|---|
| PatchTST/64 (ICLR 2023, arXiv:2211.14730, Table 3) | L = 512 | 0.129/0.222 | 0.147/0.240 | 0.163/0.259 | 0.197/0.290 | 0.159/0.253 | yuqinie98/PatchTST @204c21e (script: L = 336 for ECL) |
| PatchTST/42 (Table 3) | L = 336 | 0.130/0.222 | 0.148/0.240 | 0.167/0.261 | 0.202/0.291 | 0.162/0.254 | same |
| PatchTST self-supervised (Table 4) | L = 512, pretrain | 0.126/0.221 | 0.145/0.238 | 0.164/0.256 | 0.193/0.291 | 0.157/0.252 | same |
| iTransformer (ICLR 2024, arXiv:2310.06625, Table 5, five seeds) | L = 96 | 0.148±0.000/0.240±0.000 | 0.162±0.002/0.253±0.002 | 0.178±0.000/0.269±0.001 | 0.225±0.006/0.317±0.007 | 0.178/0.270 | thuml/iTransformer @c2426e6 |
| **TimeFilter (ICML 2025, arXiv:2501.13041, Table 8, three runs)** | **L = 96** | **0.133/0.230** | **0.154/0.248** | **0.162/0.261** | **0.184/0.284** | **0.158±0.005/0.256±0.006** | TROUBADOUR000/TimeFilter @dffde87 |
| TimeFilter (Table 9) | L searched {192,336,512,720} | 0.126/0.220 | 0.143/0.237 | 0.153/0.252 | 0.177/0.275 | 0.150/0.246 | script offers L = 512 only |
| NPMixer (preprint 2026, arXiv:2605.07476, Table 1, three seeds) | L searched {96,336,512} | not in the main text | | | | 0.152/0.249 | **NOT FOUND**: the only GitHub link in the text is SRSNet's; per-horizon values are in a supplementary table not in the PDF |
| others in NPMixer's Table 1 (same searched-L protocol): SRSNet 0.161/0.255, CycleNet 0.157/0.251, Amplifier 0.163/0.256, TimeKAN 0.164/0.258, PatchTST 0.162/0.254, DLinear 0.166/0.264 | | | | | | | not read at source in this round |

Per-horizon PatchTST and iTransformer values were read from the papers' own tables; TimeFilter's from its appendix Tables 8/9
and its Table 7 (standard deviation over three runs of the four-horizon average: MSE 0.005, MAE 0.006). Paper texts are in the
scratch extraction; the pinned clones are under `~/.local/state/crispdm-data-foundation/sota_sources/`.

## Ranking under a resolved protocol

* **Fixed context L = 96 (the "unified" setting):** TimeFilter 0.158/0.256 < iTransformer 0.178/0.270. PatchTST publishes
  no L = 96 ECL row. Leader on both MSE and MAE: **TimeFilter**.
* **Searched / long context:** TimeFilter 0.150/0.246 (Table 9) < NPMixer 0.152/0.249 (searched {96,336,512}) < PatchTST
  self-supervised 0.157/0.252 < PatchTST/64 0.159/0.253. Leader on both metrics: TimeFilter, by margins (0.002 MSE, 0.003
  MAE) smaller than TimeFilter's own three-run dispersion (0.005 / 0.006); NPMixer's per-horizon values and its code could
  not be obtained, so its claim stays **unverified, not refuted**.
* MAE and MSE do not name different leaders on ECL among the candidates read; both leaders are retained explicitly anyway.

**Selected reference: TimeFilter, protocol A = Table 8, L = 96** — fully specified by the author script (`scripts/ECL.sh`)
plus `run.py` defaults, with an author-reported dispersion that lets a numerical-agreement criterion be frozen. Protocol B
(Table 9) is retained as a secondary target: the script's L = 512 invocation is the only published recipe for it and the
paper does not state which L was chosen per horizon.

## Missing artifacts and paper/code disagreements (recorded, not resolved by guessing)

1. TimeFilter: dropout (0.5 for T = 96, 0.4 otherwise), patience 3, cosine LR schedule, n_heads 4, α = 0.1, top_p = 0.5,
   RevIN on, are script/defaults only — the paper's Table 6 states patch 32, 2 layers, lr 1e-3, d_model = d_ff = 512,
   15 epochs, batch 16. The paper reports three runs; `run.py` fixes seed 2021 with no CLI path to vary it. `requirements.txt`
   omits `sktime` and `patool`, which the code imports at module level (unused on the forecasting path).
2. TimeFilter's training loop evaluates the **test** loss every epoch (logging only; early stopping uses the validation
   loss). Recorded as part of the author path; not used by us for any decision.
3. iTransformer: the paper says batch size 32 uniformly; the ECL script uses 16. The script sets lr 5e-4, 3 layers,
   d_model = d_ff = 512. The test loader uses batch 1 with drop_last True (PatchTST's test loader also drops the last
   incomplete batch) — a reduction detail that changes the scored population slightly; TimeFilter's loader keeps every window.
4. PatchTST: the ECL script trains with L = 336 (PatchTST/42); the /64 numbers need L = 512, set by hand. OneCycleLR with
   pct_start 0.2, 100 epochs, patience 10, batch 32 — not all in the paper's text.
5. NPMixer: no code location found in the paper; per-horizon ECL values not in the main text.

## What this dossier does not do

It selects nothing for the financial or RL stages, it does not treat a preprint claim as verified, and it does not use our
auxiliary MAE_z anywhere: the reproduction is scored in the paper's own normalized MSE/MAE.
