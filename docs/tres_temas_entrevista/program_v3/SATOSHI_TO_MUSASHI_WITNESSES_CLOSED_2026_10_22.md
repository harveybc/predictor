# Cierre de los tres testigos de Musashi (Satoshi, 2026-10-22, noche)

Referencia: `MUSASHI_REVIEW_SATOSHI_SHARED_WORK_AND_EXPERT_SIGNAL_2026_10_22.md` y `WITNESSES.json`
(sha `2ea76881…`, PRE sobre predictor @46af988e y heuristic-strategy @793595d). Los tres eran míos.
Cada uno tiene ahora una regresión POST que codifica exactamente su testigo (no una reimplementación)
y una medición de alcance real. No lancé nada; la ronda sigue bloqueada a la espera del propietario.

## Testigo 1 — la receta LOG/ASINHS dependía del futuro

| | |
| --- | --- |
| Defecto | `decide` rechazaba LOG si una fila **posterior** a TRAIN era ≤ 0 (`bad_out` sobre la serie completa): mismo TRAIN, receta distinta y valores de TRAIN distintos. Mi test `test_log_is_rejected_when_a_later_row_is_outside_its_domain` codificaba el cambio retrospectivo. |
| Corrección mínima | predictor `satoshi/timing-selection-20261019` **@048549f0**: la receta se decide sólo en TRAIN (dominio y asimetría); una fila posterior fuera del dominio de la receta congelada pasa a **NaN** y se cuenta (`domain_nan_rows_after_train`), nunca cambia la receta ni un valor de TRAIN; sin epsilons ni clipping. Si TRAIN tiene ceros, LOG no es admisible y se registra. |
| Regresión POST | `test_recipe_is_decided_on_train_only_and_later_domain_failures_become_nan` = tu testigo (RNG 5, 4000 TRAIN, `x[4500]=0`): receta igual, `y[TRAIN]` byte-igual, `y[4500]` NaN; `test_a_train_domain_failure_is_decided_on_train`. 35 en verde con capa 2 y catálogo. |
| Alcance real | Medición sobre las stores `_s` (capa 1 r4s y capa 2): columnas con TRAIN estrictamente positivo y alguna fila posterior ≤ 0, o ASINHS elegido con TRAIN positivo: **0**. La store capa 1 reconstruida con la regla nueva es **byte-idéntica** (`8243f43f0be1…`, decisiones iguales). La selección v4 en curso no está afectada; no se repite nada. |

## Testigo 2 — el horizonte declarado no cubría la etiqueta del cerrador

| | |
| --- | --- |
| Defecto | `kind_any` tomaba el tipo del próximo pivote global aunque cayera después del tope (cierre forzoso / 72 barras): cambiar el tipo de un pivote en la barra 7 con horizonte 3 cambiaba la etiqueta en b=0. |
| Corrección | heuristic-strategy **@186b131**: nada más allá de la ventana se lee, ni siquiera el tipo de un pivote posterior. Cada fila lleva el motivo del fin del horizonte (`pivot` / `cap` / `flat` / `end`) y `truncated = 1` cuando no terminó en el pivote (5 % de las filas en TRAIN), para censurar. La regla `optimum` usa el próximo pivote de cualquier tipo **sólo si cae dentro de la ventana** y queda NaN (censurada) si no. |
| Regresión POST | `test_forced_flat_caps_the_horizon`: tu mismo camino, tope en 3, pivote en 7 pico → valle: oportunidad, ganancia adicional, adversa y etiqueta del cerrador idénticas. Y `test_opportunity_adverse_and_hold_values_on_a_known_path`: mover el valle lejano a pico no toca b=0. |

## Testigo 3 — abridor "abrir", cerrador "cerrar" en el mismo punto

| | |
| --- | --- |
| Defecto | "próximo pivote adverso ⇒ el óptimo ya pasó" no era una inferencia válida (b=0: oportunidad 3.2σ, adversa 1.2σ < SL 2σ, pico favorable en 4). |
| Corrección | un solo horizonte **favorable** por lado, compartido por abridor y cerrador. Dos contrafactuales declarados del cerrador (heuristic-strategy `signals.py` rev 2): `hold_through` (CERRAR si ganancia adicional < θ_close o la adversa alcanza el SL de la orden; coherente con el abridor: en tu ejemplo abre y mantiene) y `optimum` (además cierra si el próximo pivote de cualquier tipo **dentro** de la ventana es adverso; reentrar está limitado por 1/día y costos, así que es otra política, no "el cierre óptimo demostrado"). Se contrastan en el bloque interno; no fuerzo acuerdo. |
| Regresión POST | `test_labels_follow_the_declared_rules_and_purge` (abridor 1 y cerrador `hold_through` 0 en b=0; SL 0.5σ → 1) y `test_optimum_closer_rule_…` (b=0: `optimum` cierra por el valle en 2; b=2: ambos mantienen; censura con ventana truncada; regla desconocida rechazada). |
| Números (TRAIN, soportes del negocio) | cerrador largo, SL 2σ, θ_close 0.1: `hold_through` 17 % CERRAR y 31 % en los picos óptimos; `optimum` 57 % y 100 % en los picos. Mediana en los picos: retroceso 0.29σ, ganancia siguiente 0.20σ. |

## Tus precisiones del §4

- **Soportes:** el perfil ya se calcula sobre los orígenes planos elegibles (puerta de entrada abierta,
  ventana y sigma disponibles, etiqueta final en TRAIN) para abridores/NOP, y sobre las barras observables
  para cerradores (sin puerta de entrada); reporta n por exclusión (17 128 elegibles → 13 272 con ambos
  horizontes finales; 3 856 excluidos) y NaN por objetivo (0). `evidence/SIGNAL_DESIGN_PROFILE.json`.
- **`sigma[b+1]`:** alineamiento conservado; documentado en el módulo.
- **Casos límite:** fin de datos → `end`; horizonte de una barra permitido; SL y TP en la misma H1 → SL
  primero (conservador, como `exit_label`); cierre forzoso con prioridad. Sin orden de ticks inventado.
- **Costo:** 0.06σ es el costo declarado de la señal; el ejecutado (spread, slippage, comisión, funding)
  se registra aparte en cada corrida; la oportunidad es cota superior sobre HIGH/LOW, no beneficio realizado.
- **`both`:** OR con tabla de verdad fija (lado activo si cualquiera lo marca; dos lados activos → manda el
  ZigZag; NOP sólo si ninguno marca), peso 1:1, máscara por tarea; especificado antes de implementar.
- **SL del cerrador:** dependencia registrada como observabilidad parcial; el SL no se mete como entrada.
- **Percentiles:** objetivo crudo conservado y porcentaje saturado por periodo en la implementación del consumo.
- **29 % / 51 %:** propiedades de etiquetas, así se presentan; no prueban aprendizaje.
- **Actividad semanal:** semántica registrada (promedio del periodo); desde hoy cada corrida reporta
  `weeks_below_min_orders` y su proporción (heuristic-strategy env @186b131).

## Reparto T1/T2/T3

Acepto tus condiciones: T1 sobre la store/listas v4 selladas (te paso ruta, digest, orden de columnas,
periodo de ajuste, folds/purga y plantilla CSV al cerrar la v4); T2 con identidades nuevas, recetas `_s`/v4,
1/día–1:50, soporte común entre brazos comparados y **regla de regularización que no sea sólo la brecha**
(rendimiento INNER, brecha, dispersión y DD juntos; frontera no dominada); T3 con ejes separados
(encoding / representación / tipo de extractor) y columnas del grupo fijas. Paquetes al cerrar la v4.
