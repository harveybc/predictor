# Governed run: predictor through data-gov

`tools/governed_run.py` is the predictor lab path of data-gov's
`docs/04_FLOW_V2.md` (section 8). It downloads every distinct input of a
config through data-gov under an experiment key, runs `app/main.py` on CPU
with every output redirected under an output directory, and reports the
results CSV as metrics together with the sha256 of each dataset, the hash of
the effective config and the code commit.

## Toy run

Prerequisites: data-gov at `http://127.0.0.1:5055` with the
`predictor_examples` lake (in-process files lake over
`examples/data_downsampled`) and the `olap_cube` lake pointing at the OLAP
lake service on `:5057`; a service key for `predictor` in a file.

```bash
cd <predictor checkout>
/path/to/python tools/governed_run.py \
  --load_config examples/config/phase_1_daily/phase_1_ann_1575_1d_config.json \
  --experiment-key toy-ann-1575-1d \
  --gov-url http://127.0.0.1:5055 --api-key-file var/data_gov_api_key \
  --lake predictor_examples --lake-root examples/data_downsampled \
  --metrics-lake olap_cube --out-dir var/governed/toy-ann-1575-1d \
  -- --epochs 2 --max_steps_train 300 --max_steps_test 300 --mc_samples 2 \
     --execution_purpose ARCHIVAL_REPLAY_NON_AUTHORITATIVE
```

Everything after `--` goes to `app/main.py` as long-form flags (short flags
are dropped by predictor's config merger). The last flag is needed because the
toy config has no eligibility manifest: predictor's eligibility gate refuses
a manifest-less run unless it is declared an archival replay.

Interpreter: the same one that runs predictor (TensorFlow, pandas). The tool
itself is stdlib only; it does not import data-gov's `app` package because
that would shadow predictor's own.

What happens, in order:

1. The six input keys (`x_train_file`, `y_train_file`, `x_validation_file`,
   `y_validation_file`, `x_test_file`, `y_test_file`) are resolved against
   the checkout root, deduplicated by resolved path, and mapped to lake
   resources relative to `--lake-root` (`phase_1/normalized_d4.csv`). Each
   distinct file is downloaded with `GET /api/v1/download` under
   `X-Experiment-Key`, streamed in 1 MiB chunks while hashing, checked
   against `X-Content-SHA256`, and kept as
   `<cache>/<lake>/<sha256><ext>` (`--cache-dir`, default `~/.cache/data-gov`,
   expanded at run time). A mismatch deletes the partial file and fails.
   Every file is downloaded on every run: the download row in data-gov's
   accounting is what makes the later report VERIFIED.
2. `<out-dir>/governed_config.json` is the config with the inputs pointing at
   the cached files and `results_file`, `output_file`, `uncertainties_file`,
   every `*_plot_file`, `save_model`, `save_config` and `save_log` under
   `<out-dir>` (basenames kept; predictor's default basenames when the config
   leaves a key unset). Committed samples under `examples/results/` and the
   working files in the repository root are never written.
3. `app/main.py --load_config <out-dir>/governed_config.json <extra>` runs
   with `CUDA_VISIBLE_DEVICES=""` and `PYTHONPATH=<checkout>`.
4. `config_sha256` is computed after the run from the effective config
   predictor wrote to `save_config`: the six input keys replaced by
   `gov:<lake>/<resource>@<sha256>`, output paths by their basenames,
   `save_config`/`save_log` dropped, compact sorted JSON. The canonical text
   is kept in the receipt. `code_commit` is `git rev-parse HEAD` with
   `-dirty` when `git status --porcelain` is non-empty.
5. The results CSV becomes metric rows (`Train MAE H24` -> metric `MAE`,
   split `train`, horizon 24, value = Average, std_dev/min_value/max_value)
   and is posted to `POST /api/v1/experiments/<key>/metrics` with the
   datasets (each with `role` = its config key), `config_sha256`,
   `code_commit`, `project`, `phase` (default: the config's parent directory)
   and `tags.plugin`.
6. `<out-dir>/GOVERNED_RUN.json` holds all of the above and the receipt
   (`report_sha256`, `lineage`, per-dataset lineage with `event_id`).

Any failure exits 1 with one line on stderr (`governed_run: <reason>`), and
the receipt, when the output directory exists, records `status: failed` and
the reason.

## Where to look afterwards

- **Receipt:** `<out-dir>/GOVERNED_RUN.json`; the run's own outputs
  (`*_results.csv`, `*_prediction.csv`, plots, `predictor_model.keras`,
  `config_out.json`) sit next to it.
- **data-gov usage:** `GET /api/v1/experiments/<key>/usage` lists one
  `download` allow row per distinct input (with `sha256` and
  `source_sha256`) and one `write_metrics` allow row whose `detail` is the
  canonical report with its lineage.
- **Cube:** `gov_report` has one row per report (`lineage = VERIFIED` when
  every dataset was served under the key), `gov_dataset` its datasets with
  `sha256`, `role`, `lineage`, `reason`, `event_id`, `source_sha256`,
  `gov_metric` the metric rows, and `gov_metric_current` the rows of the
  latest report per `(experiment_key, lake_id)`. For governed keys this is
  the record; the legacy ETL (`fact_performance`) is not run for them.

```sql
SELECT r.experiment_key, r.lineage, d.role, d.resource_id, d.sha256, d.event_id
FROM gov_report r JOIN gov_dataset d USING (report_sha256)
WHERE r.experiment_key = 'toy-ann-1575-1d' LIMIT 20;
```

## Tests

```bash
cd <predictor checkout> && python -m pytest -q tests/test_governed_run.py   # pure functions
cd olap/lake && python -m pytest -q tests                                    # lake write path
```
