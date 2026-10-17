# Search For The Requested Six-Win Short Reference

**Superseded reference question:** the owner subsequently supplied the matched
MG2m table and explicitly identified it as the reference. See
[recovered MG2m and best-result registry](../best_registry/RETURN.md).
This earlier census does not identify a different reference on the owner's
behalf. Reproducing unmatched MG2 did not reproduce matched MG2m.

## Census

A standard-library-only script scanned retained I6-DMO result/receipt/report
metadata on all three machines, under a 512 MiB admission per invocation.
It read 789 metadata files, including mirrors, kept split identity and required
exactly six distinct horizons 2/3/4/5/6/12. Stored relative errors had to agree
with (MAE-naive)/naive. Two requested roots were absent and recorded as such;
there were no malformed-score errors. This is a census of these retained
locations and supported metadata formats, not proof about unavailable history.

At census time a static annual six-win reference was not identified. Three six-win
validation vectors were identified, all from the current continuation campaign,
not the historical MG2 or MG2m control. Their stored prediction-file hashes
were verified and MAE/naive independently recomputed from the saved arrays.

## Located Six-Win Cases

Relative error percentages; negative is better than each horizon's own naive.

| Horizon | Monthly January block | Week 22-26 January | Week 18-22 March |
|---|---:|---:|---:|
| 2h | -0.407264% | -1.763408% | -0.571904% |
| 3h | -0.252193% | -4.204443% | -0.537111% |
| 4h | -0.428141% | -2.703153% | -0.446262% |
| 5h | -0.469459% | -4.484354% | -0.169953% |
| 6h | -0.744153% | -4.665349% | -0.533355% |
| 12h | -0.089284% | -9.775447% | -1.871264% |

The January monthly block is 2024-01-01 through 2024-02-02 under full-week
grouping, not the full year. Weekly periods are also partial. These are
13-slot/2-open-hour inputs with a small core and LSTM heads, 1740 parameters,
not the 4882-parameter historical Dense-head MG2 model. They cannot be
substituted as the intended historical reference merely because they win 6/6.
This audit did not rerun their training or assert annual profitability.

Exact MAE, horizon-specific naive, counts, periods, weights, result hashes and
scan coverage are in [SIX_WIN_LOCATOR.json](SIX_WIN_LOCATOR.json).
All three source cases belong to i6dmo-continuation-41c3a7698150:
cad_short_monthly_MONTHLY-2024-01,
cad_short_weekly_WEEKLY-2024-01-21 and
cad_short_weekly_WEEKLY-2024-03-17.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES='' python tools/i6dmo/short_reference_inventory.py \
  --root /path/to/retained/runs \
  --root /path/to/retained/artifacts \
  --out /path/to/audit/INVENTORY.json
```

No TensorFlow, model fitting, TEST prices or credentials are used. The script
rejects duplicate/missing short horizons, mismatched signs and invalid counts;
six focused CPU tests pass. No worker/timer/model/experiment configuration was
changed. The main campaign remains independent and running automatically.
