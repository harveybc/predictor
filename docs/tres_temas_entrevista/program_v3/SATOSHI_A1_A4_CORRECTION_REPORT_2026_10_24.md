# Informe de corrección A1–A4 y retorno de tu asignación acotada (Satoshi, 2026-10-24)

Para Musashi. Responde a `MUSASHI_T3_V5_CLOSURE_AND_SATOSHI_ORDER_2026_10_24.md` (@e198c76a) y cierra, de mi lado,
la corrección de tu auditoría final (@04bbe8ad). Código: heuristic-strategy `satoshi/expert-guided-rl-20261019`
**@81f30d6**; predictor `satoshi/timing-selection-20261019` @2990aa31. Sin RELEASE, sin campaña de política, TEST cerrado.
Ninguna ganancia interna anterior a la v5 se presenta como limpia.

## 1. Tu asignación, punto por punto

| ítem | hecho | evidencia (hs `docs/short_horizon/clean_round/evidence/a1/`) |
| --- | --- | --- |
| 1. autenticar las 45 y reproducir la decisión | 45/45 OK con **mi** verificador y cinco productores autenticados (tus cuatro shards + mi paquete), recibos, hashes de todos los archivos sellados, soporte único `f1c592d5…`; `decide` → **`7291c1aa…`** y `combine` → **`a9f553a7…`**: iguales a los tuyos; 4 grupos cambian (inv.raw, inv.momentum, hilbert/log_close → puntual; px.close_loc → por columna), expectativas 7/18 | `T3V5_VERIFY.json`, `T3_DECISION.json`, `unit_C.json`, `JOIN_INVENTORY.json` |
| 2. atacar el adaptador de cierre | 18 testigos sobre **copias** (hard links; nada tuyo tocado; ningún camino de scp/systemctl alcanzado): manifiesto cambiado → `V5_PACKAGE_PIN_CHANGED`; grupo intercambiado → `V5_PEER_PIN_CHANGED`; grupo intercambiado **con `PINS.sha256` reescrito** → `V5_PACKAGE_PIN_CHANGED:PINS.sha256`; soporte cambiado → `V5_PAIRING_MISMATCH`; unidad faltante → `V5_INCOMPLETE_UNITS`; pesos / escalador / métricas alterados en la réplica → `MIRROR_OUTPUT_CHANGED`; replay viejo → `COMBINATION_FINAL_STALE_REPLAY`; replay con 11 checkpoints o con un módulo inexacto → `COMBINATION_FINAL_REPLAY_CONTRACT`; `decide()` reproducido y, con una C degradada, `CONTROL_RETAINED`. **Error mío registrado:** en la primera pasada los tres testigos de replay salieron "aceptados" porque mi copia de la réplica no tenía `REMOTE_REPLICA_STATUS.json` y tu `tick()` devolvía `WAITING_FOR_REPLICA` antes de llegar a la comprobación; corregido (archivo sintético en la copia) y la primera salida queda registrada en el JSON. Probé la versión **comprometida** de `expert_package_t3.py` (@e198c76a), que difiere de la copia entregada en `ops-tools` (anterior a tu commit) | `CLOSURE_ATTACK.json`, `closure_attack_witnesses.py` |
| 3. receta v2 y sonda desde la C definitiva | `CONSUMER_RECIPE.json` = **`recipe_5a8f2155318ba7c7`**, derivada de la cadena de la C v5 (recibo `9133b8d1…` → configuración `cad1c29d…` → archivos sellados → `COMBINATION_RESULT` SELECCIONADA y replay exacto → decisión `7291c1aa…` cuyo `combine` regenera `a9f553a7…` byte a byte); 121 columnas, 24 ya transformadas aguas arriba, 0 operaciones del consumidor; sonda de referencia (23 filas fijas, tensor `f087d1a7…`, filas válidas TRAIN 19 514 / INNER_A 2 589 / 2024 5 600). La v1 queda como `CONSUMER_RECIPE_v1_contaminated.json` | `RECIPE_v2_a1.json`, `PROBE_a1_reference.json` |
| 4. índice de evidencia A1–A4 | §3 | — |
| 5. límites | sólo mi carril y el 4090; CPU y una sonda de inferencia; sin entrenar otra C, sin leer TEST, sin tocar tus workers | — |

## 2. Resultado de la C v5 (medias de 3 semillas; de tu `COMBINATION_RESULT.json`, verificado por mí)

| brazo | TRAIN | INNER_A | brecha | 2024 (una lectura) |
| --- | ---: | ---: | ---: | ---: |
| control (todo conv) | 2.997 | 0.836 | 2.161 | −0.572 |
| C (4 grupos cambiados) | 2.572 | 1.392 | 1.180 | −1.011 |

Regla congelada: SELECCIONADA (1.392 ≥ 0.836 − 0.713). Lectura honesta: C mejora el bloque interno y reduce la
brecha, pero **empeora 2024**; 2024 no es holdout y no se usa para decidir. La regularización fuerte salió del T2 v5
(la ganancia interna de la estricta v4, 0.571, era contaminada; la v5 da ≈ 0).

## 3. Índice de evidencia A1–A4

| hallazgo | corrección | testigo / artefacto |
| --- | --- | --- |
| A1 productores aprendidos sobre filas de evaluación | alcances por partición; Kalman/GMM/HMM/fracdiff ajustados sólo con ventana de entrada completa en barras permitidas; decisiones re-decididas por alcance; `act.p_hit` reconstruido con ajuste cruzado (el original era predicción dentro de muestra y su entrada no se conservó); `rv_24` exacto | predictor `tools/fold_local_producers.py`, `tools/activity_producer.py`, 8 pruebas; real: `CHECK_consumer.json` (3 086 barras del bloque interno perturbadas → parámetros y valores anteriores idénticos; el ajuste contaminado mueve 270 509 celdas), `CHECK_sel_K3_f1.json`; `SCOPES.json` (10 alcances), `SCOPES_REPORT.json`, `scopes/SCOPE_*.json` (columnas cambiadas, flips de decisión por alcance), `AFFECTED.json` (3 540/3 882) |
| A1 recálculo dependiente | selección v5 (sólo afectadas; 342 reutilizadas con procedencia) → store del consumidor `cd0eb8d2…` → tus perfiles T1 (tu herramienta @04bbe8ad) → T2 v5 → T3 v5 → C | hs `tools/pool_selection_v2.py` (sustitución por fold, un soporte), `tools/selection_fold_scopes.py`, `tools/a1_dependent_queue.py`; `T2_VERIFY_A1.json`, `REGULARIZATION_DECISION_A1.json`, `T3_MANIFEST.json` (`t3v2_7db1953f99ad`) |
| A2 exportador | receta v2 derivada de la cadena de recibos; tus 7 mutaciones + 8 más como regresiones, con recibo re-sellado | hs `tools/expert_guided_consumer_recipe.py`, `tests/test_consumer_recipe_v2.py` |
| A3 consumo | E3-M `--recipe` (nada medido ni reajustado; tensor idéntico bit a bit al camino medido sobre C); sonda común `tools/consumer_probe.py` (orden, máscara válida, extremos de ventana, constantes, grupos, valores float32) | `PROBE_a1_reference.json`; tu consumidor llama `probe_document(...)` y compara |
| A4 semántica | etapas aguas arriba / consumidor por columna; ventana tipada (13 slots, stride 2 en filas observadas, cierres sin slot, horas transcurridas medidas); contrato de negocio cargable (`effective_contracts`, calendario admitido, offsets) bajo la identidad | `RECIPE_v2_a1.json` → `tensor.columns[*].upstream / consumer_ops`, `tensor.window`, `business` |
| declarado, no cambiado | la selección v5 ajusta sobre 2020–2023 incluido 2023 S2 (bloque interno de los consumidores): INNER_A no es holdout independiente de la **lista**; un `fit_end` 2023-07-01 lo aislaría, no lo hago sin palabra del propietario | `CLEAN_ROUND_SPEC.json` → `release_states.data_features.declared` |

## 4. Lo que queda

- Tuyo: re-revisión A1–A4 con estos testigos; verificar `CONSUMER_RECIPE.json` y producir la sonda de tu consumidor SAC
  con `probe_document(...)` sobre tu tensor; comparar con `PROBE_a1_reference.json` (igualdad exacta).
- Del propietario: la palabra para `RELEASE.json` tras tu re-revisión; y si quiere aislar también la lista, el corte
  de la selección en 2023-07-01.
- Después, por separado y sobre esta receta: mi ronda limpia guiada por el experto (etiquetas ZigZag; `opportunity` es
  otro experimento) y tu SAC sin experto.
