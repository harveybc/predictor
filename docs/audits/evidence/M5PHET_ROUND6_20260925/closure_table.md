### WP09 closure -- corpus 7e4789e5835d6e19, seal 31257d47b87a, 450 rows

Error definition: `error = 1 - macro_f1`. Numbers are rendered with 6 decimals, fixed.

| arm | metric | scale | n | model error | naive reference | naive error | skill | literature value | literature source | comparability |
|---|---|---|---|---|---|---|---|---|---|---|
| laya_zero_shot | macro_f1 | three classes, unweighted mean of per-class F1 | 450 | 0.622240 | majority_class (euro_area), same rows | 0.833333 | 0.253311 | NOT_CARRIED | NOT_CARRIED | COMPARABLE: same sealed rows, same metric, same naive reference |
| keyword_baseline | macro_f1 | three classes, unweighted mean of per-class F1 | 450 | 0.824004 | majority_class (euro_area), same rows | 0.833333 | 0.011195 | NOT_CARRIED | NOT_CARRIED | COMPARABLE: same sealed rows, same metric, same naive reference |
| laya_zero_shot vs keyword_baseline | macro_f1 | three classes, unweighted mean of per-class F1 | 450 | 0.622240 | keyword_baseline (country name verbatim), same rows | 0.824004 | 0.244857 | NOT_CARRIED | NOT_CARRIED | COMPARABLE: same sealed rows, same metric |
