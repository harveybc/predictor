# RP92 — source-and-protocol lock: TimeFilter on the official processed ECL (frozen before any test score)

Machine-readable lock: the `lock` block of the sealed design (`DESIGN.json` of the reproduction root, digest recomputed by
`tools/df_sota_repro.py validate`), produced by `df_sota_repro.py seal` from the author's own files — never transcribed.

| item | frozen value | how it is bound |
|---|---|---|
| paper / table | TimeFilter, ICML 2025 (arXiv:2501.13041); Table 8 (L = 96) primary, Table 9 (L = 512 script) secondary; Table 7 dispersion | `PAPER` in the tool; dossier |
| code | github.com/TROUBADOUR000/TimeFilter @ `dffde87e4fff0fdeeebbacde03dc1e432e15b3a1`, clean tree; sha256 of run.py, exp/*, models/TimeFilter.py, layers/*, data_provider/*, utils/tools.py, utils/metrics.py, utils/timefeatures.py, scripts/ECL.sh, requirements.txt | `lock.source`; every cell records the digests it ran with; drift fails the closure |
| invocation | the `python -u run.py …` lines of `scripts/ECL.sh` parsed with their loops (L = 96: T ∈ {96,192,336,720}); `run.py`'s argparse extracted by AST so EVERY effective default is the author's (80 arguments per cell) | `cells[*].argv`, `cells[*].effective_args` |
| dataset | `thuml/Time-Series-Library` @ `2b66e59e…`, `electricity/electricity.csv`, sha256 `7e45845d…`, 95 581 762 bytes, 26 304 × (date + 321), hourly, no missing; delivered ONLY through governance (lake `sota_benchmarks`, whole-resource AS_IS, availability UNDECLARED, date ranges refused) | `lock.dataset`; delivery digest checked per unit |
| partitions | Dataset_Custom borders: train [0, 18412), vali [18412−L, 21044), test [21044−L, 26304); windows per split for L = 96: T96 (18317, 2633, 5261), T192 (18221, 2537, 5165), T336 (18077, 2393, 5021), T720 (17693, 2009, 4637) | `lock.partitions`; BENCH_DATA records the counts; the closure re-derives the test targets with the author loader |
| preprocessing | StandardScaler fit on the train rows only, all channels; no inversion (metrics in normalized space); timeF marks unused by the model | BENCH_DATA scaler digests anchored by the accepted prepare terminal |
| model | TimeFilter: patch 32, 2 graph blocks, d_model 512, d_ff 512, n_heads 4, α 0.1, top_p 0.5, pos 1, RevIN (use_norm 1), dropout 0.5 (T 96) / 0.4 (T 192–720) | effective args |
| training | Adam(lr 1e-3), loss MSE + 0.05·MoE routing loss, batch 16, shuffle, 15 epochs, EarlyStopping(patience 3, delta 0) on validation MSE, cosine LR per epoch, checkpoint = best validation epoch, restored before test | `lock.training` (code reference) |
| evaluation | `utils.metrics.metric` on the concatenated float32 test predictions: MSE, MAE over every window × step × channel; test loader shuffle False, drop_last False; per horizon; the paper averages the four horizons | `lock.evaluation`; recomputed bitwise and in float64 at closure |
| seeds | 2021 (the author's fixed seed), 2022, 2023 — set by the same three calls run.py makes | `lock.seeds` |
| environment | ours: torch 2.13.0+cu130, numpy 2.5.1, pandas 3.0.3, scikit-learn 1.9.0, Python 3.12; author: torch 2.3.1, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, A100 40 GB — **declared divergence**; GPUs: RTX 4070 Laptop 8 GB (omega), RTX 4090 Laptop 16 GB (dragon), RTX 5070 Ti 12 GB (gamma) | `lock.environment_*`; each cell records its host/GPU |
| operational patches (no mathematical effect, tested) | import shims for `sktime.datasets`/`patoolib` (refuse any call); `np.Inf` alias restored for the author's EarlyStopping; run.py `__main__` replicated to vary the seed; `metric` wrapped to capture the arrays the author scores | `lock.operational_patches`; tests |
| numerical agreement (frozen) | per horizon and for the average: NUMERICAL_AGREEMENT if \|mean₃ − published\| ≤ 2·σ_paper + 0.0005 (σ_paper = 0.005 MSE / 0.006 MAE, Table 7); PARTIAL if ≤ 3·σ_paper + 0.0005; DISAGREEMENT otherwise; seed dispersion reported beside it | `lock.agreement`; `df_sota_repro.agreement` |
| replay rule (frozen) | fresh process, the author's `test(test=1)` reloading the checkpoint on CPU: allclose(atol 1e-4, rtol 1e-4) on predictions, metric of the replayed predictions within 1e-5 of the stored one; both maxima reported | `lock.replay_rule` |
| custody | every cell's predictions, checkpoint and record are artifacts of its own governed unit; the closure hashes the bytes on disk, binds them to the accepted terminal, checks config digest and tags (horizon, seed, L, arm), re-derives the targets, recomputes the metrics, replays the checkpoint | `verify_sota_run`; tests fail on substituted predictions, altered checkpoint, changed scaler, wrong horizon/channels/window/target, another reduction, missing artifact |

Tests fail on a different scaler, channels, window, horizon, target, reduction, checkpoint or substituted predictions
(`tests/test_df_sota_repro.py`). Protocol fidelity and numerical agreement are separate verdicts in `SOTA_TABLE.json`.
