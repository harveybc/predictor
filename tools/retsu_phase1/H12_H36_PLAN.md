# 12 h y 36 h — incremento estacional

Este archivo manda para esta corrida. No se abre la opción 2, 3 ni 4. No se diseña la estrategia. No se rehacen el STL del target, la máscara de columnas ni los campeones de 1–6 h, 24 h y 48 h.

Cada horizonte tiene su propia máscara de la misma sesión y su propio modelo. 12 h no se recorta a las filas de 48 h. 2024 se predice una vez por horizonte, después de entrenar los dos y de restaurar cada campeón. Ninguna época mira 2024.

Entradas: las 164 columnas de la fase 2, identidad `recipe_cb69e8f60044af7f`. El encoder es el DoubleTCN semilla 7, completo y descongelado, con cabeza lineal de una salida. El learning rate del encoder es el del campeón vecino: 3e-6 para 12 h y 3e-5 para 36 h. La cabeza queda en 3e-4. Esos dos valores se declaran aquí; no hay grilla.

El target es `estacional(t+h) - estacional(t)` de la serie ya guardada. El naive es el incremento cero. La validación es H2 2023.

```h12-h36
{
  "schema": "retsu_h12_h36.v1",
  "horizons": [12, 36],
  "encoder_lr": {"12": 3e-6, "36": 3e-5},
  "head_lr": 3e-4,
  "weight_decay": 0.004,
  "scope": "full_encoder",
  "head": "linear",
  "recipe_identity": "recipe_cb69e8f60044af7f",
  "phase2_dir": "/var/tmp/retsu-phase1-causal-p2",
  "stl_series": "/var/tmp/retsu-phase1/stl_series.npz",
  "base_dataset": "/var/tmp/retsu-phase1/dataset.npz",
  "out_dir": "/var/tmp/retsu-phase1-h12-h36",
  "finished_dirs": ["/var/tmp/retsu-phase1-causal", "/var/tmp/retsu-phase1-causal-p2", "/var/tmp/retsu-phase1"],
  "min_pool": 1575,
  "min_monitor": 200,
  "alone_pool": {"12": 22647, "36": 13514},
  "alone_monitor": {"12": 1362, "36": 794},
  "alone_holdout": {"12": 3009, "36": 1791},
  "row_keep_fraction": 0.95,
  "reduce_lr_grace": 20,
  "reduce_lr_patience": 20,
  "reduce_lr_factor": 0.2,
  "lr_floor": 4.8e-7,
  "es_patience": 80,
  "es_start_epoch": 100,
  "min_delta": 1e-8,
  "max_epochs": 2000,
  "batch": 256,
  "seed": 7
}
```
