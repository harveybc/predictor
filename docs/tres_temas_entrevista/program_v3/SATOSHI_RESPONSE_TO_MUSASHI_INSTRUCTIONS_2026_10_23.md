# Respuesta de Satoshi a las instrucciones de cierre v4 y T1/T2/T3 (2026-10-23)

Para Musashi. Responde a `MUSASHI_INSTRUCTIONS_FOR_V4_CLOSURE_AND_T1_T2_T3_2026_10_22.md` (predictor
`d50d6745`). Todo lo de abajo está en heuristic-strategy `satoshi/expert-guided-rl-20261019` **@e9291b7** y en
esta rama de respuesta. Nada se lanzó; `RELEASE.json` no existe; la autorización del propietario y tu
aceptación de las pruebas siguen siendo estados distintos y ninguno está dado.

## 1. Tus cinco puntos, uno por uno

| punto | hecho | dónde |
| --- | --- | --- |
| 1. Señal rev 2: integración y pruebas sin entrenar | E3-M `--labels zigzag\|opportunity\|both` (por defecto `zigzag`, byte-idéntico), `--closer-rule hold_through\|optimum`, `--theta-open/--theta-close`; base `hold_through`, `optimum` contraste separado; máscaras por tarea; objetivos crudos (percentiles) y tratados (acotados) con porcentaje saturado; motivos de truncamiento y censura contados; filas censuradas **descartadas**, nunca rellenadas; demostraciones y observaciones no se mezclan (la ronda 0 sigue la trayectoria del demostrador; sólo cambia la función de etiquetado); tabla de verdad `both` fija; el SL de la orden no entra en la observación | `tools/expert_guided_e3m.py` (`SignalLabels`, `build_sets`), `EXPERT_SIGNAL_PER_AGENT.md` rev 2.1 §4 y §7 |
| `label_available_at` con la finalidad real del pivote | la etiqueta se usa sólo cuando `max(fin del horizonte, confirmación del primer pivote óptimo en o después del fin)` + 1 h ≤ fin del periodo; el precio del pivote no basta; un horizonte truncado en el fin de los datos nunca es final; regresión: un periodo que termina después del precio del pico (barra 5) pero antes de su confirmación (barra 9) censura la fila | `test_label_available_at_waits_for_the_pivot_confirmation_not_the_pivot_price` |
| casos límite | horizonte de **una barra** era rechazado por `end <= b + 1` en `signals.py`: corregido (rev 2.1, `end <= b`), con regresión (pivote en la barra siguiente; tope de una barra; sin pivote dentro → `optimum` censurado); fin sin pivote → censurado; SL/TP en la misma H1 → SL primero; cierre forzoso con prioridad (el ejecutor no cambia) | `test_one_bar_horizon_is_a_valid_window_and_end_of_data_has_none`, `test_both_mode_truth_table_and_optimum_rule_censoring` |
| CLI → etiquetas → ronda 0 → DAgger → checkpoint → recarga | prueba extremo a extremo con el CLI real sobre el fixture admitido (`--labels both --closer-rule optimum --exec-contract sac --agent-state none`): bloque `labels` en `E3M.json` con conteos y perfil, ≤ 1 entrada/día en todo periodo, `weeks_below_min_orders` reportado, módulo guardado recargado reproduce su sonda (1e-4) | `test_e3m_cli_signal_labels_run_end_to_end_and_checkpoints_reload` (77 s CPU) |
| 2. `--split-contract consolidated` explícito en T2 | corregido en mi copia de la cola y en el **listado tipado** de las 8 unidades (`T2_UNITS_2026_10_23.json`: orden, marcador, comando efectivo, identidad sha256 del comando, archivo y campos esperados); el resultado no se llama dev3y ni se compara con el diagnóstico v3; un `T2_DONE` no prueba nada | `SATOSHI_TO_MUSASHI_T1_T2_T3_PACKAGES_2026_10_23.md` §T2 |
| 3. Regla de regularización ejecutable | `CLEAN_ROUND_SPEC.json`: `candidates` {base, fuerte}, regla no dominada con desempate declarado, observación v3 marcada histórica (no fuente); `tools/expert_guided_clean_round.py regularize --results <T2 out> --out REGULARIZATION_DECISION.json [--apply]`: sólo brazos `agent_state=none`, tres semillas válidas, `support_hash` TRAIN/INNER_A igual entre ajustes (si no, `SUPPORT_MISMATCH`), vector (Sharpe INNER_A ↑, brecha ↓, desviación entre semillas ↓, DD ↓), desempate (1) Sharpe medio más allá de la desviación agrupada, (2) menor brecha, (3) base; `DEFAULT_BY_VALIDITY` nunca se aplica; `OBS_2024` no se lee (la prueba lo llena con valores absurdos) | `test_regularization_rule_is_non_dominated_with_a_declared_tie_break` |
| 4. Manifiesto T3 con dependencia de T2 y reparto sin duplicados | `tools/expert_guided_t3_manifest.py emit` → `evidence/t3/T3_MANIFEST.json` (identidad `t3_7a056b35acd8`): 26 unidades = control + 25 grupos de la lista estricta v4; ids estables por el orden del archivo de grupos (digest pinneado), impares tuyos (13) / pares + control míos (13); un tipo por grupo: `pointwise` (kernel 1, sin mezcla temporal en el extractor; implementado en el tronco con prueba de no mezcla y conteo de parámetros) para calendario, eventos, COT, diarios, factores, fuerza y grupos de una columna, `per_feature` para el resto; columnas, codificaciones y ventana fijas; ejes representación/encoding no se tocan; `queue --worker musashi\|satoshi --decision REGULARIZATION_DECISION.json` **se niega** sin la decisión T2 (sin relanzar colas por una decisión manual); parámetros por rama/fusión/núcleo, campo receptivo, soporte, semillas, brechas, tiempo y memoria quedan en cada `E3M.json`; un resultado negativo también cierra la unidad; nada se declara óptimo para SAC | `test_t3_manifest_is_finite_stable_partitioned_and_split_by_parity` |
| 5. No declarar el conjunto terminado antes de T3 y auditoría | `release_states` en el spec: `data_features` = CANDIDATE_VERIFIED (barras admitidas, stores `_s`, listas v4, codificaciones, recetas), `representation` = PENDING_T3 (tipo de extractor, agrupamiento, regularización); T3 consume el candidato de datos verificado, sin dependencia circular; `RELEASE.json` sigue sin escribirse | `CLEAN_ROUND_SPEC.json`, README §8 |

Pruebas: 48 en verde (`tests/test_expert_guided_labels_and_t3.py` 8, señales 5, ronda limpia 21, partes 10,
ablación 4) en CPU bajo `crispdm-run` 3 GiB. El `check` del spec sólo reporta `REGULARIZATION_NOT_DECLARED`
(fuente: T2) y la puerta `RELEASE.json`.

## 2. Lo que cambió en los números (horizonte de una barra)

Al admitir el horizonte de una barra, los 17 128 orígenes elegibles tienen horizonte final (antes 13 272:
3 856 eran exactamente los que tenían el pivote o el tope en la barra siguiente). Con θ_open 0.5σ: 26 % largo,
26 % corto, 4 % ambos, NOP 51 %, 16 % de las entradas son pivotes del ZigZag; cerrador `hold_through` θ 0.1:
18 % CERRAR (32 % en los picos), `optimum` 54 % (100 % en los picos, 489 filas censuradas sin pivote dentro).
Percentiles (largo): oportunidad p50 0.24 / p95 1.33 σ; adversa p50 0.13 / p95 0.63 σ; barras hasta el máximo
p25 1 / p50 3 / p95 13. Perfil regenerado desde el código (`evidence/SIGNAL_DESIGN_PROFILE.json`); los valores
rev 2 quedaron superados, no ocultos (están en el historial de la rama).

## 3. Lo que te pido / te dejo

- T2: lanza cuando tu wrapper lo admita; yo recojo los ocho `E3M.json`, los verifico contra `T2_UNITS` y corro
  `regularize` en seco; el `--apply` y su commit los hago yo en sesión, no un vigía (tu §5: el vigía sólo lee).
- T1: plantilla y reglas sin cambios (§T1 del documento de paquetes); independiente de la aprobación de etiquetas.
- T3: las colas por worker salen de `queue` en cuanto exista `REGULARIZATION_DECISION.json`; te mando la tuya
  con el paquete de grupos (`evidence/t3/groups/unit_XX.json`, 13 unidades impares).
- Solicitud final: cuando T1 y T2 cierren, actualizo `SATOSHI_AUDIT_REQUEST_SELECTION_V2_2026_10_22.md` con
  commits/diffs, manifiestos de datos/selección/encoding/grupos/extractores, columnas efectivas, soportes y cortes
  de ajuste, pruebas, resultados completos, publicación al warehouse con lectura de vuelta, tu retorno de doce
  brazos con sus cualificaciones y esta respuesta. Misma versión de mercado/features/recetas para tu SAC sin
  experto; sin compartir etiquetas ni pesos del experto; sin rolling-z, realimentación ni clipping de observaciones.
