# RP61 — literature configurations traced to code

Two references were read **at their source** for this matrix, not from memory or from a summary. What
each one actually specifies is below, next to what our code does, next to what this round ports.
Nothing here transfers a published number to our task: the tasks are different, and that difference
is the first row of every column.

## 1. The matrix

| | Bai/locuslab TCN | TensorFlow forecasting tutorial | Our E1 successor | This round's port |
|---|---|---|---|---|
| source read | `github.com/locuslab/TCN/blob/master/TCN/tcn.py` (raw), paper arXiv:1803.01271 | `tensorflow.org/tutorials/structured_data/time_series` | `tools/df_e1_pilot.py`, design `143abb57…` | `tools/df_tcn_reference.py` |
| commit / revision | the repository's `master` as fetched 2026-09-21 | the published tutorial as fetched 2026-09-21 | run `satoshi-e1-successor-20260920` | this branch |
| task | generic sequence benchmarks (adding problem, copy memory, P-MNIST, word/char LM) | **weather**, temperature at Max Planck BGC | **household electric power**, one household | same as ours |
| sampling | task-dependent | 10 min resampled to **1 hour** | **1 minute** | 1 minute |
| horizon | next token / class | 1 step (1 h), or 24 steps out | **60 steps = 60 minutes** | 60 minutes |
| split | per benchmark | 70 / 20 / 10 by time, no shuffle | 28 d train / 7 d validation inside the family's train span, test never read | identical to ours |
| normalisation | none in the block; task-level | z-score, **fitted on train only** | z-score, train only, **window grain** | identical to ours |
| block | `weight_norm(Conv1d)`→chomp→**ReLU**→**Dropout 0.2**, **twice**, +1×1 residual, then ReLU | Conv1D(32, k=3, relu) + Dense(32, relu) + Dense(1) | **one** Conv1D(16, k=3, **ELU**) + 1×1 projection skip, **no** weight norm, **no** dropout, no activation after the sum | the Bai block, ported |
| dilations | 2^i per level, kernel 2 by default | none (plain conv) | [1, 2, 4, 8, 16], kernel 3 | 2^i, kernel 2 |
| receptive field | 1 + 2(k−1)·Σ2^i | 3 steps | 60 (measured, `tools/df_e1_receiver.py`) | 63 |
| init | conv weights ~ N(0, 0.01) | Keras defaults; residual head **zero-initialised** | Keras defaults | N(0, 0.01), as the source |
| optimiser | Adam, with gradient clipping and per-task schedules | Adam, **default lr 1e-3** | Adam, **lr 3e-3** | **ours**, so the block is the only difference |
| loss / monitor | task loss | **MSE**, MAE reported as a metric | **MSE** in z-space; early stopping on validation MSE | ours |
| batch / stopping | per task | 32, max 20 epochs, patience 2 | 64, ceiling 4 000 updates, patience 3 epochs, restore best | ours |
| calendar inputs | — | **yes**: day and year as sin/cos, wind as x/y components | **none** | none in phase 1 |
| residual/delta head | — | **yes** for the residual LSTM: predicts a delta on the input, zero-initialised | no: predicts the level | no in phase 1 |

## 2. What this says about our core, precisely

Our "TCN" shares a name and dilated causal convolutions with the reference, and differs in five
specific ways: one convolution per block instead of two, ELU instead of ReLU, no weight
normalisation, no dropout, and no activation after the residual sum. **That is not the reference's
block.** Any expectation of the paper's behaviour on the strength of the name is unfounded, and the
port in `tools/df_tcn_reference.py` exists so the difference can be measured rather than argued.

Its width was chosen by a rule stated before any fit: the channel count whose trainable-parameter
total is closest to our own core's. Widths 12/16/20/24/32 give 3 109 / 5 297 / **8 061** / 11 401 /
19 809 parameters against our 8 127, so 20 was taken. The contrast is therefore about the block, not
about capacity.

## 3. What is NOT transferable, and is not being transferred

- The tutorial's numbers are **weather at one-hour sampling, one step ahead**. Ours is household
  power at one-minute sampling, sixty steps ahead. Its skill figures say nothing about our task.
- The paper's benchmark results are sequence-modelling tasks with their own recipes (clipping,
  schedules, much larger widths). Porting the block does not port those results.
- TS2Vec (arXiv:2106.10466) is a **contrastive** objective with its own protocol. It is evidence that
  pretraining objectives differ from each other; it is not evidence that our masked auto-encoder
  should win, and it is not used as one.
- The UCI source holds nearly four years at one-minute sampling. That does not authorise opening any
  reserve, and it does not turn this pilot's 35-day slice into four years.

## 4. What the tutorial contributes that we have not tried

Two concrete, cheap ideas whose provenance is this reference, kept for the ordered sequence and NOT
mixed into the first contrast:

1. **Calendar inputs as sin/cos** of the daily and yearly position. Our consumed columns carry no
   calendar at all. RP59 measured the train autocorrelation at the daily lag (0.317) as *lower* than
   at the hourly lag (0.403), so a daily context is a hypothesis to test, not a certainty.
2. **A delta head, zero-initialised**, so that an untrained model starts exactly at persistence
   instead of at a random level. RP60 measured our untrained model at MAE 1.57–2.45 kW against
   persistence's 0.617, so where a fit starts from is not a detail here.

Both belong to a later phase, after the temporal and availability checks the order requires. Moving
them now would mix the architecture contrast with an input change, which is exactly what RP62
forbids.

## 5. The one independent neural comparator chosen for the first phase

`tools/df_tcn_reference.py` — the Bai block, ported, parameter-matched, trained with our own
optimiser, budget, loss and stopping on the same rows, same labels and same seeds. LSTM and CNN
families from the tutorial are **enumerated here for a later contrast and are not launched**.
