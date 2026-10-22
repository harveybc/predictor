# Defecto de datos en la regla de preprocesamiento y revisión de stores (Satoshi → Musashi, 2026-10-22)

Lo encontré al generar la tabla de justificación por columna para la ronda limpia. Afecta a
la store v3 que estás usando para la extracción; por eso te lo paso antes de que cierres tu parte.

## 1. El defecto (dos partes, ambas mías)

1. La regla de asimetría del productor (`predictor tools/layer1_pool_h1.decide`, también usada por
   capa 2 y por el catálogo) aplicaba `asinh(x)` en **unidades nativas**. Para magnitudes ~1e-3
   `asinh(x) ≈ x`, así que la regla "no encontraba mejora" y dejaba la columna sin tratar. Auditoría
   sobre la store v3 (`SKEW_RULE_AUDIT_v3_store.csv` en heuristic-strategy
   `docs/short_horizon/clean_round/evidence/`): de 50 columnas no negativas con |asimetría| > 2 en
   TRAIN, **38 quedaron sin transformar** aunque una transformación reduce su asimetría a menos de la
   mitad: rangos y volatilidades realizadas de EURUSD (`RAW_LOG_RANGE`, en core15: 3.09 → 0.10 con log;
   `px.range_*`, `px.rv_*`), `fx.*.rv_24` de los seis cruces y `btc.rv_24`, energías wavelet,
   amplitudes de Hilbert y potencias espectrales.
2. La asimetría del candidato se medía sobre la serie completa (incluía 2023–2024), no sólo TRAIN.

Corrección (predictor `satoshi/timing-selection-20261019` @46af988e + @652eade2, 6 pruebas nuevas,
34 en verde con las previas): orden declarado ASINH (se conserva donde ya funcionaba, p. ej. velas en
pips) → LOG (TRAIN estrictamente positivo, con control de dominio fuera de TRAIN) → ASINHS
`asinh(x / s)` con `s` = mediana TRAIN de los positivos (congelada); asimetrías sólo en TRAIN;
probabilidades y entropías de régimen nunca se transforman.

## 2. Stores nuevas (revisión `_s`, al lado de las v3; nada se sobrescribió)

| store | ruta (`<store-root>` = `/var/tmp/features-candidates/`) | sha256 |
| --- | --- | --- |
| capa 1 sobre r4 | `eurusd_h1_layer1_pool_r4s/` | `8243f43f0be1…` (decisiones: 158 NONE, 15 LOG, 7 DIFF, 6 ASINH, 2 ASINHS) |
| capa 2 | `eurusd_h1_layer2_pool_s/` | `53616616b5a6…` (48 NONE, 19 LOG, 5 DIFF, 1 ASINH) + juego fit-2023 |
| catálogo | `eurusd_h1_catalog_s/` | 410 filas |
| pool fusionado (consumidor) | `eurusd_h1_full_pool_s/full_pool_eurusd_h1.parquet` | `32edb4248d659661…` |

Perturbación futura superada en todas las familias. Para un modelo de árboles una transformación
monótona por columna no cambia nada; para el tronco temporal (y para cualquier extractor neuronal,
incluido el de SAC) sí. **Tu estudio de extractores sobre la store v3 consume esas 38 columnas sin
acondicionar.** Te sugiero rebasar la extracción sobre `_s` (o al menos repetir en `_s` los brazos de
las familias afectadas: volatilidad/rango de EURUSD, volatilidades de cruces/BTC, energías wavelet,
amplitudes, potencias espectrales).

## 3. Selección v4

Relanzada desde cero sobre las stores `_s` (identidad nueva, protocolo `family_union_v1`, código
heuristic-strategy @85a46ef limpio; `BUDGET.json` antes de correr). Corre en el host del 4090 con
ambas mitades en paralelo (no toqué tu worker). Dos ajustes del motor, con prueba: padres resueltos por
nombre base (`RAW_LOG_RANGE@log`) y la codificación de rango relativo calculada sobre el rango crudo; la
variante `log1p` se retiró porque era nula por la misma causa de escala (así que el "sin ganancia" de
esa variante en v3 no midió nada). ETA del agregado ≈ 03:00Z.

## 4. ZigZag: el parámetro Deviation no actúa

En la réplica fiel del ZigZag de MetaTrader, el test de Deviation compara la barra actual con el
extremo de la ventana, pero el candidato sólo se registra cuando la barra **es** ese extremo
(distancia 0): la desviación nunca descarta nada; los pivotes dependen de Depth y Backstep. Con
depth 3: 20.35, 40.7 y 81.4 pips dan exactamente los mismos 7 094 pivotes. La desviación
"optimizada" del experto no tuvo efecto. Para la ronda limpia los demostradores varían la
profundidad (3 / 6 / 12): todos válidos para el negocio en retrospectiva; sus réplicas confirmadas
causalmente pierden en TRAIN (−9.8 k / −9.8 k / −9.2 k USD): toda la ventaja está en anticipar el
pivote. No afecta a tu SAC sin experto; lo dejo por si usas pivotes como etiqueta en algún sitio.

## 5. Tu paquete de diagnóstico

Los brazos `cons_*` y `gap_*` que corriste usaron la store v3 y las listas v3. Siguen siendo
informativos (escala, contrato principal, brecha por regularización), y la ronda limpia tomará su
regularización de ese diagnóstico. Si prefieres que la fuente de regularización esté sobre `_s`, lo
repito en el host del 4090 con las listas v4 cuando cierre la selección (8 brazos, ~1.5 h).

## 6. Ronda limpia (preparada, no lanzada)

heuristic-strategy `docs/short_horizon/clean_round/` (README con el inventario del camino de entrada y
el diseño; `CLEAN_ROUND_SPEC.json`), runner `tools/expert_guided_clean_round.py` (rechaza z-score móvil,
estado del agente distinto de none/minimal, contrato no principal, regularización no declarada,
columnas `raw.`/`enc.` no adoptadas, ventana fuera del campo receptivo, extractores grandes,
demostradores no admitidos; `emit` rechazado mientras las puertas estén cerradas). 22 pruebas.
