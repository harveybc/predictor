# SOTA reproduction table — Hu, Y. et al. TimeFilter: Patch-Specific Spatial-Temporal Gr…

Protocol L96 (Table 8 (L = 96)); design 9b49010da99d; protocol fidelity: **NOT_ESTABLISHED: L96_h96_s2021: fresh-process reload through the author's test() exceeds the frozen replay rule (0.0019083023071289062); L96_h96_s2022: fresh-process reload through the author's test() exceeds the frozen replay rule (0.00040149688720703125); L96_h96_s2023: fresh-process reload through the author's test() exceeds the frozen replay rule (0.010064072906970978)**; complete: False

| dataset / protocol | model @ revision | T | published MSE / MAE | replicated mean MSE / MAE (sd, n) | difference | matched naive MSE / MAE | training | cost | replay max|Δ| | agreement |
|---|---|---:|---|---|---|---|---|---|---|---|
| ECL official processed (TSL), L=96, T=96, split 7/1/2, normalized space | TimeFilter @ dffde87e4fff | 96 | 0.133 / 0.230 | — / — (sd — / —, n=0) | — / — | nan / nan | — | 0.00 h wall, 0.0 GiB peak | nan | NO_MEASUREMENT / NO_MEASUREMENT |
| ECL official processed (TSL), L=96, T=192, split 7/1/2, normalized space | TimeFilter @ dffde87e4fff | 192 | 0.154 / 0.248 | — / — (sd — / —, n=0) | — / — | nan / nan | — | 0.00 h wall, 0.0 GiB peak | nan | NO_MEASUREMENT / NO_MEASUREMENT |
| ECL official processed (TSL), L=96, T=336, split 7/1/2, normalized space | TimeFilter @ dffde87e4fff | 336 | 0.162 / 0.261 | — / — (sd — / —, n=0) | — / — | nan / nan | — | 0.00 h wall, 0.0 GiB peak | nan | NO_MEASUREMENT / NO_MEASUREMENT |
| ECL official processed (TSL), L=96, T=720, split 7/1/2, normalized space | TimeFilter @ dffde87e4fff | 720 | 0.184 / 0.284 | — / — (sd — / —, n=0) | — / — | nan / nan | — | 0.00 h wall, 0.0 GiB peak | nan | NO_MEASUREMENT / NO_MEASUREMENT |

## Executed cells NOT verified by the closure (values shown, never pooled into a mean)

| T | unit | MSE / MAE (author float32) | custody | replayed MSE / MAE | replay max|Δ| | check that failed |
|---:|---|---|---|---|---|---|
| 96 | L96_h96_s2021 | 0.13332 / 0.23055 | ACCEPTED_ARTIFACT_CHAIN | 0.13331516 / 0.23055443 | 1.91e-03 | L96_h96_s2021: fresh-process reload through the author's test() exceeds the frozen replay rule (0.0019083023071289062) |
| 96 | L96_h96_s2022 | 0.14008 / 0.23778 | ACCEPTED_ARTIFACT_CHAIN | 0.14008181 / 0.23777533 | 4.01e-04 | L96_h96_s2022: fresh-process reload through the author's test() exceeds the frozen replay rule (0.00040149688720703125) |
| 96 | L96_h96_s2023 | 0.13309 / 0.23032 | ACCEPTED_ARTIFACT_CHAIN | 0.13309242 / 0.23032171 | 1.01e-02 | L96_h96_s2023: fresh-process reload through the author's test() exceeds the frozen replay rule (0.010064072906970978) |
| 192 | L96_h192_s2021 | 0.16086 / 0.25599 | ACCEPTED_ARTIFACT_CHAIN | nan / nan | nan | REPLAY_PENDING: not replayed on this host (memory placement); the row stays UNVERIFIED until replayed |

Average over the four horizons: MSE None vs published  (NOT_COMPUTED); MAE None vs  (NOT_COMPUTED).

Unexecuted or unverified cells: ['L96_h96_s2021', 'L96_h96_s2022', 'L96_h96_s2023', 'L96_h192_s2021', 'L96_h192_s2022', 'L96_h192_s2023', 'L96_h336_s2021', 'L96_h336_s2022', 'L96_h336_s2023', 'L96_h720_s2021', 'L96_h720_s2022', 'L96_h720_s2023'].
Agreement rule (frozen before any test score): per horizon and for the four-horizon average, the three-seed mean of the replicated metric agrees with the published value when |mean - published| <= 2 x std_paper + 0.0005 (rounding half-unit); PARTIAL when <= 3 x std_paper + 0.0005; DISAGREEMENT otherwise. std_paper is the paper's std over its three runs of the four-horizon average (Table 7), applied per horizon as the only dispersion the paper reports; the replicated seed dispersion is reported beside it and never replaces the criterion

* = early stopped. Metrics are the author's `metric()` on the concatenated float32 test predictions in the normalized space (no inversion); the paper prints three decimals. Training completion per seed: epochs run and the checkpointed epoch. Naive = persistence on identical windows.
