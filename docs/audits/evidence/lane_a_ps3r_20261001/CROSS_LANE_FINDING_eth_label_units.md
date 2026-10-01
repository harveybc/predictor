# Cross-lane finding: lane B's ETH business-target labels sit 1000x the declared horizon ahead

Found by lane A (Satoshi, successor technical lead), 2026-10-01. Reproduced independently by the coordinator; accepted as decisive.

**Where.** feature-eng `9c8e1a0`, `tools/progressive_selection.py:629`:

```python
ts = pd.to_datetime(pd.Series(stamps), format=...).astype("int64").to_numpy() // 10**9
```

**Mechanism.** In the pinned env (worker_b `envs/tensorflow`, pandas 3.0.1), `to_datetime` returns `datetime64[us]`. The `// 10**9` assumes nanoseconds, so it yields kiloseconds.

**Measurement 1 — the computed step.** Run under crispdm-run 1G on worker_b:

```
3.0.1 datetime64[us] [1506571, 1506585] [14]
```

Two consecutive 4 h bars give a step of 14 instead of 14400. `build_targets` adds `h*3600` in those units, so each label lands at 1000 × h. Y_s@4h is therefore read about 167 days ahead, and Y_l@24h about 1000 days ahead.

**Measurement 2 — the missing-label counts confirm it.** The counts in lane B's own `runs/eth_4h/target_status.csv` are exactly what the 1000× offset predicts:

| Label | Missing in target_status.csv | Missing predicted by a 1000× offset |
|---|---|---|
| Y_s@4h | 1009 | ≈ 1000 bars |
| Y_l@24h | 6000 | 6000 |
| Y_l@48h | 11994 | ≈ 12000 |
| Y_l@72..144h | 13699 (all of TRAIN) | all |

**Measurement 3 — what correct seconds give.** Using the raw CSV timestamps in seconds (predictor `examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv`, TRAIN = 13 699 rows), the missing labels are 21 / 28 / 34 / 52 at h = 24 / 48 / 72 / 144. The step histogram is 14400 s × 13690, 28800 × 5, 43200 × 2, 115200 × 1.

**Consequence.** The ETH PS2 relevance scores, tiers and work list `f75260a6` were computed against the wrong targets. The coordinator has withdrawn acceptance of that list. Lane B is ordered to fix the conversion with an explicit unit, add an FS02 real-stamp test plus a mutation, and republish as a CORRECTION.

**Effect on lane A.** `PILOT_SELECTION.json` keeps its rule and seed (20261001) but is marked `PROVISIONAL_SUPERSEDED_LABELS`. It will be redrawn on the republished sha.
