# Classification: literature replication before product comparison

Musashi, 2026-09-28. Status: SOURCE_REVIEWED_SHORTLIST, not a sealed experiment,
not downloaded/registered data, not a measured classifier ranking.
This adds a scientific classification lane; it does not replace Weather/Traffic,
financial forecasting, RL, or the M5PHET product plan.

## Questions and boundaries

1. Can we reproduce a named strong published model under its exact evaluation?
2. Does the M5PHET adapter preserve that model's outputs on identical inputs?
3. Does the resulting classifier work on independently labelled business data?

Keep these separate from natural-language routing into a typed task. A router
can choose the correct task while the classifier answers it incorrectly.
Laya is a candidate provider, not the assumed state of the art of every task.
An old dataset used by recent literature is eligible; dataset age is not model
recency. Record both the original dataset citation and the recent benchmark.

## Initial portfolio

| Priority / dataset | What it tests | Reference to reproduce or qualify |
| --- | --- | --- |
| Financial: FOMC / Trillion Dollar Words | Hawkish, dovish and neutral monetary-policy language | Original RoBERTa-large recipe [1] as an anchored reference; FinMTEB [2] and recent DCS / LabelFusion-TS [6,7] as newer candidates, with distinct protocols |
| Financial: Financial PhraseBank, exact agreement subset | Financial sentiment; NOT EURUSD relevance or next-price direction | Original dataset as used by the selected recent financial benchmark [2]; Fin-E5 is a candidate pending accessible weights, recipe and per-task result, not an assumed executable artifact |
| General: BANKING77 | Fine-grained intent with many similar labels | MTEB/MMTEB task version [3]; Qwen3-Embedding-8B plus the exact benchmark classifier/evaluator [4], subject to dated same-task ranking review |
| General: AG News | News topic categorization and native Laya parity | Reproduce Laya's exact published sampled run [5] first; separately evaluate the full official test against the strongest reproducible same-protocol reference |
| Second stage: MASSIVE, English and Spanish | Multilingual intent, matching our interface languages | MMTEB protocol [3]; Laya's sampled option-set experiment [5] is a separate task, not the full-label benchmark |

FinMTEB is an EMNLP 2025 financial benchmark, with original code [2]. Its paper
reports MAP for classification; do not silently substitute accuracy or macro-F1.
Pin the actual evaluator and resolve paper/code discrepancies before sealing.
Fin-E5's overall rank does not establish the best individual classification row.

Qwen3-Embedding [4] supplies open model sizes including 8B. Its overall embedding
rank is not a BANKING77 SOTA claim. The reference includes the trained downstream
classifier, labelled training subset, repeats and aggregation, not just the encoder.
Do not call this zero-shot if labels train the head. Direct Laya zero-shot is a
different supervision regime; report it separately, even on the same test rows.

FOMC [1] declares CC BY-NC 4.0 for released resources. Keep it research-only
pending rights review, with no commercial trading deployment or inferred rights
from a permissively licensed wrapper. Newer FOMC papers [6,7] are preprints,
not interchangeable protocols: continuous stance scoring, time-series-augmented
classification, and sentence-only classification are different interventions.

Laya's own tables [5] use sampled populations and distinguish checkpoints.
They also describe label-option budget limitations and temperature changes.
Do not compare a sampled result with a full test result, truncate 77 options,
or replace option ordering, prompt, checkpoint, batching or temperature silently.

## How the strongest reference is selected

Before OUR test scores, retain a dated candidate table from primary papers,
official model cards and evaluator result artifacts. For each task rank only
the SAME dataset revision, label space, split, supervision regime and metric.
Record published value, precise table/result file, model/weights/code revision,
licence, resource need and why every excluded higher row is not reproducible or
not comparable. No universal 'best model' and no self-reported leaderboard title
without the underlying artifact. The names above are starting candidates, not
a declaration that a 2025 model still leads every September 2026 task.

A missing checkpoint is a missing artifact, not permission to invent its model
or claim its score for a substitute. Keep a reproducible historical anchor AND
the strongest accessible current challenger; label the distinction. Quantizing,
changing pooling/max length, or using fewer labels is a new variant, not exact
replication. Checkpoint evaluation reproduction and full training reproduction
must also have different labels and costs; do not promise pretraining from scratch.

## Required experiment contract

Freeze dataset source/revision/licence, per-split row IDs and counts, duplicates,
class map and label order, exact subset/sampling seed, train/dev/calibration/test
roles, model/tokenizer/weights, prompt serialization, truncation, precision,
batching, max length, pooling, downstream head and fitting recipe. For training
reproduction include updates, optimizer, loss, schedule, early stopping and seeds.
Unknown pretraining contamination is UNKNOWN, not 'clean'.

Reproduce the primary published metric and scale exactly. Separately record
accuracy, macro/weighted/per-class F1 and confusion counts where meaningful.
For probabilities, use NLL/Brier and ECE with declared binning/calibration split;
an entropy-derived confidence is not P(correct). Preserve abstention coverage
and selective risk: never remove refusals from the accuracy denominator silently.
Fit majority/stratified baselines using train labels; score them on identical
test rows. Any business keyword baseline is fixed on development data.

Use native author inference first and an independently invoked M5PHET adapter
second. Compare labels and ordered probabilities with predeclared numerical
criteria, complete coverage and actual output-type validation. No shared wrapper
helper serving as both implementation and independent oracle.

Register permitted resources via the existing lake/data-gov procedure. Warehouse
receipts must include task, supervision regime, dataset/split/population, label
mapping, scorer/averaging, calibration identity, provider/checkpoint and protocol.
Extend classification metric receipts explicitly: the TSL MSE/MAE contract is
NOT a classification contract. Prove idempotency and rejected malformed values
against a temporary warehouse before accepting scientific rows. No fake scores.

Final table: dataset/task, model and regime, published metric/value, reproduced
value, paired naive, gap, coverage, comparability, cost and limitations. Native
replication and wrapper parity precede our extensions. A separate untouched
business corpus tests generalization; sentiment/topic accuracy does not establish
news relevance, causal effect or profitability.

## References (primary sources checked 2026-09-28)

[1] A. Shah, S. Paturi, and S. Chava, "Trillion Dollar Words: A New Financial
Dataset, Task & Market Analysis," ACL, pp. 6664-6679, 2023,
doi: 10.18653/v1/2023.acl-long.368.
https://aclanthology.org/2023.acl-long.368/

[2] Y. Tang and Y. Yang, "FinMTEB: Finance Massive Text Embedding Benchmark,"
EMNLP, pp. 3620-3638, 2025, doi: 10.18653/v1/2025.emnlp-main.179.
https://aclanthology.org/2025.emnlp-main.179/
Code: https://github.com/yixuantt/FinMTEB

[3] K. Enevoldsen et al., "MMTEB: Massive Multilingual Text Embedding Benchmark,"
arXiv:2502.13595, 2025. https://arxiv.org/abs/2502.13595
Evaluator: https://github.com/embeddings-benchmark/mteb

[4] Y. Zhang et al., "Qwen3 Embedding: Advancing Text Embedding and Reranking
Through Foundation Models," arXiv:2506.05176v3, 2025.
https://arxiv.org/abs/2506.05176v3
Code: https://github.com/QwenLM/Qwen3-Embedding

[5] NandhaKishorM, "Laya benchmarks," project benchmark documentation, 2026,
accessed Sep. 28, 2026. Not a peer-reviewed SOTA ranking.
https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md

[6] Y. Tang and Y. Yang, "Mind the Shift: Decoding Monetary Policy Stance from
FOMC Statements with Large Language Models," arXiv:2603.14313v1, 2026.
https://arxiv.org/abs/2603.14313v1

[7] M. Schlee, F. Lukassen, and C. Weisser, "LabelFusion-TS: Fusing Large
Language Models, Transformer Encoders, and Financial Time Series for
Monetary-Policy Stance Classification," arXiv:2608.11753v1, 2026.
https://arxiv.org/abs/2608.11753v1
