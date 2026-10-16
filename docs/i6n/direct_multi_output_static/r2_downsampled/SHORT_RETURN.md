# Short MG2m R0 Versus R2

**CLOSED: R0_RETAINED.** Campaign `i6dmo-fed57e1d7696`, executing source
`5d430951`. All 25 units completed, 116/116 governed reports read back,
177 files verified in each physical mirror. CLOSE SHA256:
`31a46a1dee5846791d6b6cd83dc77c92c36e1d88bdcc3cd0181ca9bdc58c8b49`.
The separately sealed long MG12m sequence follows automatically. No TEST,
trading, cadence change or core pretraining ran.

## Measured Result

Original-price MAE, identical supported DEVELOPMENT-2024 rows for each
horizon. Each horizon has its own naive; its value is identical across the
two arms only because that horizon's actual origins/targets are matched.
Inputs: 19 selected RAW features, 13 market-time observations over 24 open
hours including t; hourly predictions, static training, six direct heads.

| Horizon | Naive for this horizon | R0 MAE | R2 MAE | R0 relative error | R2 relative error |
|---|---:|---:|---:|---:|---:|
| 2 h | 0.000746664 | 0.000746697 | 0.000746993 | +0.004425% | +0.044042% |
| 3 h | 0.000928891 | 0.000928973 | 0.000929012 | +0.008874% | +0.012987% |
| 4 h | 0.001079279 | 0.001079101 | 0.001079846 | -0.016506% | +0.052514% |
| 5 h | 0.001224979 | 0.001225024 | 0.001224809 | +0.003669% | -0.013919% |
| 6 h | 0.001362063 | 0.001361934 | 0.001362768 | -0.009498% | +0.051720% |
| 12 h | 0.001994833 | 0.001994087 | 0.001994321 | -0.037392% | -0.025655% |

Relative error is `(MAE_h - naive_h) / naive_h`; negative is better. The
frozen equal-head mean is R0 **-0.007738%**, R2 **+0.020282%**. R0 is retained
at tolerance 1e-8; R0 has eligible heads 4/6/12, R2 has 5/12. R2 improves
h5 but not the complete six-head recipe. This one-seed development contrast
does not prove that pretraining in general is harmful or that any head is
independently confirmatory. Selection used 2024 development; TEST stays closed.

Both fits completed 12 epochs, restoring epoch 2, with identical core/head
initialization and trainable branches. Fit times were 50.72 s R0 and 50.98 s
R2; host cgroup peaks were 2.43/2.54 GiB. The 19 original donors were adopted,
not retrained; their original cost and INNER-only monitor remain disclosed.
The fresh R0 reproduced the retained RAW recipe's per-head errors exactly;
its strict code-identity reuse predicate required this one new reference.

Exact unrounded values, report/import digests, measured costs and comparison
are in `SHORT_RESULT.json`. The strict production comparison verified both
reports, own naives, row/support metadata and authenticated 19-branch import.
Reconstruction/probe metrics were diagnostic and did not choose the winner.
