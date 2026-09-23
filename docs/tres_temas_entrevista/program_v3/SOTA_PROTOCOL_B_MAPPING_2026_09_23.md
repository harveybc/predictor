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

## Cost, from measured protocol A costs on the admitted device

Protocol A on the external RTX 5090 measured 15.5 min (T = 192), 15.3 min (T = 336) and 20.8 min (T = 720) per cell, with a
peak host RSS of 2.7–7.1 GiB and ≤ 5.2 GiB of VRAM. Protocol B multiplies the token count per channel by 512/96 ≈ 5.3 at the
patch embedding while keeping four patches per channel, and its test population shrinks (fewer windows fit). A conservative
projection is **2–4× protocol A's wall time per cell**, i.e. roughly 30–80 min per cell and **6–16 h for twelve cells** on one
device, with VRAM well inside the 32 GiB of the admitted 5090 and host RSS bounded by the same disk-backed evaluation.
This is a projection from measured cells, not a measurement.

## What executing it would require

The same governed path as protocol A: seal a design with `protocol="Lsearched"`, acquire the same governed ECL file, execute
one cell at a time on the **admitted external RTX 5090** (UUID checked before each dispatch, host thermals monitored), close
with replays on that device, catalog every cell, then delete predictions under the corrected per-content gate. No part of it
may reuse protocol A's cells, and no protocol A number may appear in a protocol B row.
