# SOTA reproduction table — Hu, Y. et al. TimeFilter: Patch-Specific Spatial-Temporal Gr…

Protocol Lsearched (Table 9 (L searched in {192, 336, 512, 720}; the script offers L = 512 only)); design bcc7e3d3c09c; protocol fidelity: **FAITHFUL_WITH_DECLARED_ENVIRONMENT_DIVERGENCE**; complete: True

| dataset / protocol | model @ revision | T | published MSE / MAE | replicated mean MSE / MAE (sd, n) | difference | matched naive MSE / MAE | training | cost | replay max|Δ| | agreement |
|---|---|---:|---|---|---|---|---|---|---|---|
| ECL official processed (TSL), L=512, T=96, split 7/1/2, normalized space | TimeFilter @ dffde87e4fff | 96 | 0.126 / 0.220 | 0.1259 / 0.2210 (sd 0.0004 / 0.0005, n=3) | -0.0001 / +0.0010 | 1.5878 / 0.9455 | 2021:15ep best14; 2022:15ep best15; 2023:15ep best15 | 1.05 h wall, 6.6 GiB peak | 0.00e+00 | OPERATIONAL_AGREEMENT / OPERATIONAL_AGREEMENT |
| ECL official processed (TSL), L=512, T=192, split 7/1/2, normalized space | TimeFilter @ dffde87e4fff | 192 | 0.143 / 0.237 | 0.1437 / 0.2380 (sd 0.0006 / 0.0003, n=3) | +0.0007 / +0.0010 | 1.5962 / 0.9507 | 2021:15ep best13; 2022:15ep best15; 2023:15ep best11 | 1.12 h wall, 6.6 GiB peak | 0.00e+00 | OPERATIONAL_AGREEMENT / OPERATIONAL_AGREEMENT |
| ECL official processed (TSL), L=512, T=336, split 7/1/2, normalized space | TimeFilter @ dffde87e4fff | 336 | 0.153 / 0.252 | 0.1527 / 0.2513 (sd 0.0007 / 0.0006, n=3) | -0.0003 / -0.0007 | 1.6178 / 0.9613 | 2021:15ep best15; 2022:15ep best14; 2023:15ep best6 | 1.13 h wall, 6.6 GiB peak | 0.00e+00 | OPERATIONAL_AGREEMENT / OPERATIONAL_AGREEMENT |
| ECL official processed (TSL), L=512, T=720, split 7/1/2, normalized space | TimeFilter @ dffde87e4fff | 720 | 0.177 / 0.275 | 0.1790 / 0.2753 (sd 0.0012 / 0.0009, n=3) | +0.0020 / +0.0003 | 1.6468 / 0.9754 | 2021:15ep best3; 2022:15ep best3; 2023:15ep best8 | 1.29 h wall, 6.7 GiB peak | 0.00e+00 | OPERATIONAL_AGREEMENT / OPERATIONAL_AGREEMENT |

Average over the four horizons: MSE 0.1503 vs published 0.15 (OPERATIONAL_AGREEMENT); MAE 0.2464 vs 0.246 (OPERATIONAL_AGREEMENT).

Unexecuted or unverified cells: none.
Agreement rule (frozen before any test score): per horizon and for the four-horizon average (formed within each seed first), the three-seed mean of the replicated metric is in OPERATIONAL_AGREEMENT with the published value when |mean - published| <= 2 x std_paper + 0.0005 (rounding half-unit); OPERATIONAL_PARTIAL when <= 3 x std_paper + 0.0005; OUTSIDE_OPERATIONAL_MARGIN otherwise. std_paper is the paper's std over its three runs of the FOUR-HORIZON AVERAGE (Table 7), borrowed per horizon as a predeclared operational margin — not a published per-horizon error bar, not statistical equivalence; the replicated seed dispersion is reported beside it and never replaces the criterion

* = early stopped. Metrics are the author's `metric()` on the concatenated float32 test predictions in the normalized space (no inversion); the paper prints three decimals. Training completion per seed: epochs run and the checkpointed epoch. Naive = persistence on identical windows.
