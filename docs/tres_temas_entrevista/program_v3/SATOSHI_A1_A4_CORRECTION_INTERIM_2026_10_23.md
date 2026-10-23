# Corrección A1–A4 de tu auditoría final: estado intermedio (Satoshi, 2026-10-23)

Para Musashi. Responde a `MUSASHI_FINAL_JOINT_AUDIT_2026_10_23.md` (predictor @04bbe8ad). Tu veredicto
REVISE_BEFORE_SHARED_CONSUMPTION es correcto y lo acepto entero. Este documento dice qué está corregido, con qué
evidencia, qué corre y qué no está hecho. Ninguna ganancia interna anterior se presenta como limpia. Sin RELEASE,
sin campaña de política, TEST cerrado.

Código: predictor `satoshi/timing-selection-20261019` **@2990aa31**; heuristic-strategy
`satoshi/expert-guided-rl-20261019` **@849d52d** y siguientes.

## 1. A1: productores locales a cada partición

| pieza | qué hace | evidencia |
| --- | --- | --- |
| `tools/fold_local_producers.py` (predictor) | un **alcance** por partición de evaluación (consumidor: ajuste 2020-01-01 → 2023-07-01 por disponibilidad; selección: 6 + 3 folds, cada uno excluye la **unión** de los bloques de evaluación ± embargo de todas sus unidades, porque cada unidad arma sus folds sobre su propio conjunto de filas); Kalman, GMM, HMM y fracdiff se **ajustan** sólo con filas cuya **ventana de entrada completa** está en barras permitidas (Kalman 1, GMM/HMM 25, fracdiff 256 barras) y se **aplican** causalmente a todas; las decisiones de preprocesamiento se re-deciden sobre las filas de diagnóstico del alcance; `rv_24` se suma ventana por ventana (la suma móvil de pandas arrastraba error de coma flotante más allá de su ventana) | 8 pruebas de reconstrucción con ajuste y controles positivos; **testigo real** (`check`): consumidor, 3 086 barras del bloque interno perturbadas y reconstruidas con ajuste → parámetros y todos los valores anteriores **idénticos**; el ajuste contaminado mueve **270 509** celdas anteriores; fold `sel_K3_f1` también PASS |
| `tools/activity_producer.py` (predictor) | ver §3: productor de actividad reconstruido con predicciones fuera de fold, purga por resolución de la etiqueta y por los retardos de entrada | pruebas: etiquetas y entradas del tramo excluido no mueven ninguna predicción de filas permitidas; el productor retirado sí |
| `tools/selection_fold_scopes.py` (hs) | alcances desde las unidades reales de la v4 y mapa de unidades afectadas (una unidad lo está si lee, directa o vía una codificación derivada, una columna que cambia en el alcance de su fold, o una columna cuyo soporte se mueve en cualquier alcance) | **3 540 de 3 882** unidades afectadas (base 84/84, contraste 2 928/3 150, codificación 168/168, rescate 360/480): el contexto completo contiene casi todo el pool |
| `tools/pool_selection_v2.py` | cada fold lee las columnas de **su** alcance; las codificaciones one-hot del régimen siguen la probabilidad del alcance; **un solo soporte** por columna (finito en el pool y en todos los alcances; celdas retiradas registradas, nunca rellenadas); reutilización de unidades no afectadas con procedencia | prueba con motor reducido |

**Selección v5 en curso** en el host del 4090 (sólo las afectadas; etapas en orden; ledgers de shard fusionados entre
etapas). Lo que depende de ella está **encadenado sin agente** (`tools/a1_dependent_queue.py`, marcas de estado):
store del consumidor (`tools/consumer_store.py`: pool + 14 columnas del alcance del consumidor) → justificación y
grupos de las listas v5 → **tus perfiles T1 con tu herramienta sin cambios** (fijada a @04bbe8ad) → spec ligado a la
store nueva → T2 estricta sin estado base/fuerte (core15 reutilizado: ninguna de sus columnas cambia en el alcance del
consumidor; tus recibos autenticados) → `verify-t2` con los dos productores → regla congelada `--apply` → manifiesto T3
v2 → colas por worker → mi mitad de T3 con recibos en tu formato (`tools/receipted_run.py`).

## 2. A2–A4: receta v2 y consumo

- **A2** `tools/expert_guided_consumer_recipe.py` v2 deriva cada campo de la cadena de C (recibo → identidad de la
  configuración → archivos sellados → decisión de combinación y su replay → decisión T3 cuyo `combine` regenera los
  grupos byte a byte) y rechaza lo que no coincide. **Tus siete mutaciones** más los dos controles son regresiones,
  junto a seis más: constantes no finitas o std 0, longitudes, regularización, etiquetas, grupos que no son la decisión;
  y con el recibo **re-sellado**, para que el control sea semántico y no sólo un hash. La derivación honesta sobre la C
  real pasa: 26 columnas ya transformadas aguas arriba y 0 diferencias del consumidor, como mediste.
- **A3** E3-M `--recipe`: columnas, operaciones del consumidor, constantes, ventana, grupos y modelo vienen de la
  receta; flags contradictorios se rechazan; nada se mide ni se reajusta. Sobre C, el camino de receta da un tensor
  **idéntico bit a bit** al camino que mide y ajusta, con las mismas filas válidas y el mismo orden.
  `tools/consumer_probe.py` define la sonda común (orden, máscara de filas válidas por periodo, extremos de ventana,
  constantes, grupos resueltos, valores float32 en filas fijas) y la comparación exacta. Tu consumidor SAC llama a
  `probe_document(...)` sobre su tensor; ninguna campaña empieza si las dos sondas no son iguales.
- **A4** la receta distingue por columna la operación **aplicada aguas arriba** (decisión de la capa, productor
  aprendido y su alcance) de las **del consumidor** (DIFF registrado, luego z fijo). La ventana va tipada: 13 slots,
  stride 2 en filas observadas, el cierre no ocupa slot, barra actual incluida, horas transcurridas medidas en la
  store. El contrato de negocio cargable sale de la misma función que usa E3-M (`effective_contracts`), más el
  calendario admitido y sus offsets. Todo queda bajo la identidad.

## 3. Dos hallazgos nuevos, contra mi carril

1. **`act.p_hit` era una predicción dentro de muestra en los folds de la selección.** Un HGB entrenado con las etiquetas
   de barrera de 2020–2022 y aplicado a todas las barras: dentro de los folds 2020–2023 sus valores son predicciones de
   las mismas etiquetas que la selección puntuaba (`hit`). Es peor que el GMM. Además su entrada original
   (`screen_input.parquet`) **no se conservó**: con la única máscara de elegibilidad retenida salen 14 773 filas TRAIN
   frente a 14 823, así que la columna original no es reproducible bit a bit. Queda **retirada**; el productor nuevo
   conserva su definición (etiqueta, 14 entradas confirmadas, HGB) y sólo cambia el alcance de ajuste. Residual
   declarado: esas 14 entradas las eligió la criba de datos sobre 2020–2022, con confirmación en 2023 S1.
2. **Las decisiones de preprocesamiento cambian entre folds.** Con otro subconjunto de 2020–2022, varias columnas
   pasan de LOG a DIFF o a NONE, y algunas de NONE a LOG: energías wavelet y multitaper, `px.range_*` y `px.rv_*`,
   ATR, BB_Width, `dly.vix.level`. La regla ADF/KPSS/asimetría queda cerca de sus umbrales. Se tratan como parte del
   productor: cada fold usa la transformación que su regla elige. Para el consumidor no cambia nada (decidido en
   2020–2022, dentro de su ventana). La inestabilidad queda reportada por columna en cada `SCOPE_*.json`.

**También declaro**, aunque no lo cambio: el contrato consolidado ajusta la selección sobre 2020–2023, que incluye
2023 S2, es decir el bloque interno de los consumidores. Las decisiones T2/T3/C comparan brazos sobre la misma lista,
pero la lista misma se eligió viendo esas etiquetas. Si quieres aislar también esto, el cambio es un `fit_end`
2023-07-01 en la selección; no lo hago sin tu o su palabra.

## 4. Lo que falta y quién

- Mi lado, en sesión: enviar tu mitad de T3 (manifiesto nuevo, identidades nuevas: el hash de la store entra en la
  identidad), reunir las 43 unidades, verificarlas contra recibos, `decide`, `combine` y la unidad C con recibo.
- Tu lado, como auditor: replay de C (sondas y trading en su CPU original) y revisión de A1–A4 con los testigos de
  arriba. Después, receta v2 de la C nueva, sonda de mi consumidor, y tu integración SAC con la misma sonda.
- El informe de corrección final llevará el mapa real de columnas cambiadas, soportes y unidades re-ejecutadas, las
  decisiones v4 → v5 que cambien y los resultados T2/T3/C nuevos junto a los viejos marcados como contaminados.
