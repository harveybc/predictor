# Protocol B (TimeFilter Table 9, long context): the author's recipe, mapped exactly, NOT executed

Status: **SPECIFIED AND SEALED FOR EXECUTION; NO CELL RUN; NO RESULT CLAIMED.**
Scope: this is the separate deliverable RP113 asks for. Protocol A (Table 8, L = 96) does not complete it and its numbers
are never reported as protocol B's.

## What the paper publishes and what the code offers

Table 9 of TimeFilter (ICML 2025, arXiv:2501.13041) reports Electricity/ECL for an input length **searched over
L ∈ {192, 336, 512, 720}** and gives one number per horizon plus the average:

| T | 96 | 192 | 336 | 720 | avg |
|---|---|---|---|---|---|
| MSE / MAE | 0.126 / 0.220 | 0.143 / 0.237 | 0.153 / 0.252 | 0.177 / 0.275 | 0.150 / 0.246 |

The authors' released script (`scripts/ECL.sh` at the pinned revision dffde87e) contains exactly **one** long-context block,
with `seq_len=512` for all four horizons. Its arguments, verbatim:

```
--task_name long_term_forecast --is_training 1 --root_path ./data --data_path electricity.csv
--model_id ECL_512_$pred_len --model TimeFilter --data custom --features M
--seq_len 512 --label_len 48 --pred_len {96,192,336,720}
--e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321
--patch_len 128 --des Exp --learning_rate 0.001 --batch_size 16 --train_epochs 15
--d_model 512 --d_ff 512 --dropout 0.5 --top_p 0.0 --itr 1
```

Differences from protocol A (L = 96, Table 8) that matter and are not defaults: **seq_len 512**, **patch_len 128**
(so the number of patches per channel is 4, as in protocol A), **top_p 0.0**, **dropout 0.5 at every horizon**.

## The choice mapping, stated instead of assumed

1. **Which L produced each published cell is NOT published.** Table 9 names a searched set; the paper does not say which
   member of {192, 336, 512, 720} each column used. The script offers only L = 512.
2. Therefore an L = 512 run reproduces **the only published recipe for protocol B**, and its agreement may be reported
   **only against the Table 9 column for the same horizon**, with the explicit statement that the paper's value may come
   from another L. Running L = 512 and calling it "Table 9 reproduced" without that statement would be a substitution.
3. The remaining three input lengths have **no published recipe** (no patch length, dropout or `top_p` is given for them).
   Executing them would be our own search, not a reproduction, and is out of scope here.
4. The frozen agreement rule, the metric space and reduction, the data identity, the seeds and the replay rule are the
   same as protocol A's; only the recipe above changes. Seeds {2021, 2022, 2023}, as in protocol A.

## What actually changes, verified against the pinned loader and model

Read from the code at dffde87e with the governed ECL file, not from comments
([PROTOCOL_B_PROPERTIES.json](../../audits/evidence/d3_k5_20260917/RP114/PROTOCOL_B_PROPERTIES.json)):

| property | protocol A (L = 96, patch 32) | protocol B (L = 512, patch 128) |
|---|---|---|
| patches per channel `(L − P)/P + 1` | 3 | 4 |
| token sequence per sample `C × patches` | 963 | 1284 |
| graph mask elements `3 × tokens²` | 2,782,107 | 4,945,968 |
| training windows (T = 96 … 720) | 18221 / 18125 / 17981 / 17597 | 17805 / 17709 / 17565 / 17181 |
| validation windows | 2537 / 2441 / 2297 / 1913 | identical |
| **test windows** | 5165 / 5069 / 4925 / 4541 | **identical** |

Two corrections to the earlier version of this document, both from Musashi's RP113 review and both confirmed above:

1. The input is not tokenised 512/96 ≈ 5.3 times more finely. The patch length grows with the input, so the token count rises
   from 963 to 1284 (×1.33) and the mask from 2.78 M to 4.95 M elements (×1.78). What grows per token is the patch projection's
   input width (32 → 128 values per patch).
2. The test population does NOT shrink because L grows. The pinned custom loader prepends `seq_len` rows at the test border, so
   the count is `num_test − pred_len + 1` at a fixed split and horizon: 5260 − T + 1 in both protocols, exactly the populations
   already measured for protocol A. Training windows are the only population that shrinks (by 416 windows, ~2 %).

## Cost: UNMEASURED

No protocol B cell, and no bounded protocol B pilot, has been executed. The earlier "2–4× protocol A, 6–16 h for twelve cells,
VRAM fits" sentence was an assumption derived from a wrong token-count argument; it is withdrawn. **The runtime, VRAM, host
memory, disk and scoring cost of protocol B are UNMEASURED.** Protocol A's measured costs on the admitted external RTX 5090
(15.3–15.6 min per T = 192/336 cell, 20.6–20.8 min per T = 720 cell, ≤ 5.2 GiB VRAM, 2.7–7.1 GiB peak host RSS) describe
protocol A only and are not admission evidence for protocol B.

Before any protocol B campaign is scheduled, the resource question is settled by a **bounded, governed, exact-recipe pilot** on
the admitted external device: the author's L = 512 arguments unchanged, a fixed small number of optimizer steps and one
validation pass, measuring wall time per iteration, peak VRAM, peak host RSS, disk and the cost of one bounded evaluation —
with no test scoring, no checkpoint selection and no number that could influence a recipe choice. The finite population (12
cells) is then projected from those measurements and stated as a projection.
## What executing it would require

The same governed path as protocol A: seal a design with `protocol="Lsearched"`, acquire the same governed ECL file, execute
one cell at a time on the **admitted external RTX 5090** (UUID checked before each dispatch, host thermals monitored), close
with replays on that device, catalog every cell, then delete predictions under the corrected per-content gate. No part of it
may reuse protocol A's cells, and no protocol A number may appear in a protocol B row.
