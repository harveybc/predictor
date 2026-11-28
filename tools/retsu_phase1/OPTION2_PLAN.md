# Opción 2 — búsqueda acotada del encoder

Este archivo manda. No se abre la opción 3 ni la 4. No se diseña la estrategia. No se rehace el STL ni la selección de columnas. No se predicen 2024 durante la búsqueda.

Hallazgos que entran en el diseño. El encoder actor completo y la cabeza lineal vencieron al último bloque del core y a la cabeza de trayectoria. El ancho del core se queda en 16. La cabeza se queda en 3e-4. En horizontes cortos el encoder ganó con 3e-6 y en largos con 3e-5. El dropout del core actor está en 0: el 0.1 del archivo de configuración cae en el crítico, y el crítico no se entrena. El módulo vivo solo acepta etapas `tcn` y `pointwise`. LSTM, TFT y transformer no están en ese extractor y no se inventan aquí.

La búsqueda usa DEAP sobre el horizonte 6, que es el cierre temprano con más margen entre los horizontes cortos. El ajuste de cada individuo usa filas anteriores a 2023-01-01. La aptitud es `MAE naive − MAE del modelo` en `[2023-01-01, 2023-07-01)`. Más alto es mejor. H2 2023 y 2024 no entran en esa aptitud. El tope de la búsqueda es 400 épocas. El presupuesto es población 8 y 2 generaciones, como máximo 24 entrenamientos.

Después se elige el mejor genoma de todos los evaluados, no solo el de la última población. Empate dentro de 1e-8: menos bloques de core, luego menor learning rate del encoder, luego dropout 0, luego `tcn`, luego ramas compartidas. Ese ganador se reentrena en el horizonte 6 con el protocolo completo, ajuste anterior a 2023-07-01 y early stopping en H2 2023. El mismo genoma se transfiere una vez al horizonte 24, con su propia máscara y con el learning rate largo ya medido, 3e-5. 2024 se predice una vez por horizonte, después de esos dos refits.

```option2
{
  "schema": "retsu_option2.v1",
  "search_horizon": 6,
  "transfer_horizon": 24,
  "population": 8,
  "generations": 2,
  "crossover_probability": 0.5,
  "mutation_probability": 0.5,
  "tournament": 3,
  "search_seed": 20261125,
  "max_trainings": 24,
  "search_max_epochs": 400,
  "refit_max_epochs": 2000,
  "transfer_encoder_lr": 3e-5,
  "head_lr": 3e-4,
  "weight_decay": 0.004,
  "recipe_identity": "recipe_cb69e8f60044af7f",
  "phase2_dir": "/var/tmp/retsu-phase1-causal-p2",
  "stl_series": "/var/tmp/retsu-phase1/stl_series.npz",
  "out_dir": "/var/tmp/retsu-phase1-option2",
  "finished_dirs": ["/var/tmp/retsu-phase1-causal", "/var/tmp/retsu-phase1-causal-p2", "/var/tmp/retsu-phase1", "/var/tmp/retsu-phase1-h12-h36"],
  "score_start": "2023-01-01T00:00:00Z",
  "monitor_start": "2023-07-01T00:00:00Z",
  "holdout_start": "2024-01-01T00:00:00Z",
  "min_pool": 1575,
  "min_monitor": 200,
  "alone_monitor_24": 1074,
  "alone_holdout_24": 2397,
  "alone_pool_24": 18074,
  "row_keep_fraction": 0.95,
  "reduce_lr_grace": 20,
  "reduce_lr_patience": 20,
  "reduce_lr_factor": 0.2,
  "lr_floor": 4.8e-7,
  "es_patience": 80,
  "es_start_epoch": 100,
  "min_delta": 1e-8,
  "batch": 256,
  "seed": 7,
  "genes": [
    ["core_blocks", [1, 2, 3]],
    ["core_kernel", [3, 5]],
    ["core_dilation_base", [1, 2, 3]],
    ["core_activation", ["silu", "gelu"]],
    ["core_dropout", [0.0, 0.1]],
    ["encoder_lr", [3e-6, 1e-5, 3e-5]],
    ["core_kind", ["tcn", "pointwise"]],
    ["branch_mode", ["shared", "pointwise"]]
  ],
  "baseline": [1, 0, 1, 0, 0, 0, 0, 0]
}
```
