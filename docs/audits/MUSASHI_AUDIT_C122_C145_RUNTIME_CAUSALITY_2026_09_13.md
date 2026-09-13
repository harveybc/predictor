# Auditoria Musashi C122-C145: runtime y causalidad

**Fecha:** 2026-09-13

**Objeto:** implementacion parcial de la orden C122-C145 en `predictor@7f00a2b`

**Veredicto:** `REVISE_P0_BEFORE_RELAUNCH`

## 1. Resumen ejecutivo

La base conceptual y buena parte de la mecanica D0-D2 existen. La bateria
focal revisada pasa 362 pruebas. Sin embargo, C130-C134 no estan cerrados: el
perfil completo y la calibracion SNR no produjeron recibos finales, y su
reintento agoto la memoria del host.

El operador multiescala implementado parece usar exclusivamente muestras
pasadas y presentes. Eso es una buena senal, pero todavia no basta para
autorizar su uso. La API confia en una etiqueta entregada por el llamador para
afirmar que el ajuste usa solo training, y las pruebas de invariancia temporal
usan pocos cortes. La clase de fuga descrita por el owner exige una prueba
exhaustiva y una frontera de datos ejecutable, no solo inspeccion del algoritmo.

No se relanza C130-C134 ni se consume ningun operador wavelet hasta cerrar los
hallazgos P0 de este dictamen.

## 2. Hallazgos, por severidad

### P0.1 - El perfilador puede agotar por si solo toda la memoria

`tools/df_profile_run.py:93-135` lee cada tabla completa, convierte todas sus
columnas numericas a `float64` y crea otra matriz con `column_stack`. Luego
`tools/df_profile_run.py:176-179` replica ese patron en varios procesos sin una
estimacion de memoria ni un limite por tarea.

La causa dominante del incidente es reproducible antes de asignar memoria. El
ADF de `tools/df_profile_univariate.py:398-400` usa la serie completa y
`floor(12*(n/100)^0.25)` rezagos. Para `n=13,253,761`, el diseno OLS aproximado
requiere:

```text
lag = 228
n * (lag + 2) * 8 bytes = 22.71 GiB
```

Esto coincide con el proceso de aproximadamente 22 GiB que el kernel termino.
Reducir ocho workers a dos no corrige el defecto: una sola tarea sigue siendo
capaz de agotar el host.

### P0.2 - `fit()` no prueba que el insumo sea training

`tools/df_operators.py:9-16` promete ajuste train-only, pero la API recibe una
matriz arbitraria y la cadena literal `"train"`. El artefacto liga sus
parametros, no la identidad del dataset, los bytes, la particion ni el rango
temporal que los produjo. Un llamador puede entregar validacion, confirmacion o
la serie completa y conservar la misma etiqueta.

Esto afecta en especial a los umbrales wavelet y a las medias estacionales. Un
algoritmo internamente causal no repara un ajuste contaminado externamente.

### P0.3 - `transform_batch()` carece de contrato temporal de entrada

La transformacion recibe solo una matriz. No comprueba orden temporal,
timestamps, disponibilidad, identidad de columnas, digest del insumo ni rol de
la particion. Por tanto, la prueba local `Y[:t+1] = f(X[:t+1])` no demuestra que
`X[:t+1]` represente informacion realmente disponible en `t`.

### P0.4 - La prueba temporal actual es insuficiente para wavelets

`tests/test_df_operators.py` comprueba siete cortes y un reemplazo aleatorio del
sufijo. Es util, pero puede omitir errores de borde en niveles `2^j`, padding,
reinicio, warm-up, NaN o fragmentacion. Una fuga de una sola muestra basta para
invalidar un backtest.

Se exige comparar todos los prefijos de casos pequenos y cortes estratificados
en casos grandes, usando varios sufijos hostiles y una implementacion de
referencia que procese una muestra a la vez.

### P1.1 - El nombre del operador excede lo demostrado

`wavelet_haar_atrous` es una recurrencia multiescala trailing propia. No se ha
demostrado equivalencia matematica con una transformada a trous o SWT estandar.
Ademas, el diseno D3 nombra `T06_CAUSAL_SWT`, que aun no es una implementacion.
Debe usarse un nombre descriptivo como `trailing_haar_threshold` o aportarse la
derivacion y la equivalencia numerica que justifiquen el nombre actual.

### P1.2 - El wavelet MAD offline debe quedar aislado de features

`tools/df_snr.py:70-73` usa `pywt.dwt(..., mode="periodization")` sobre el
segmento completo. El propio contrato lo declara diagnostico offline. Esa
operacion puede conservarse para estimar una cifra sobre training, pero jamas
puede alimentar una fila temporal, compensar un retraso o convertirse en
feature. Esta frontera debe ser estructural y probada por un consumidor
negativo.

### P1.3 - La descomposicion estacional usa todo training para inicializarse

`tools/df_operators.py:486-501` obtiene medias por fase de todo el bloque de
training. Es valido como calibracion anterior para transformar una particion
posterior. No es estrictamente online cuando transforma las mismas filas con
las que se ajusto. Deben existir dos modos separados:

* `TRAIN_FROZEN_FOR_LATER_PARTITIONS`: calibra en training y solo transforma
  particiones posteriores;
* `EXPANDING_ONLINE`: cada salida de training usa solamente su prefijo.

No debe llamarse causal a la aplicacion in-sample del primer modo.

### P1.4 - El estado publicado y el estado fisico no coinciden aun

El codigo para perfiles, SNR, operadores y carga OLAP esta comprometido, y el
laboratorio sintetico tiene artefactos. Sin embargo, los roots esperados de
perfil y SNR no contienen sus recibos finales. Una implementacion presente no
equivale a experimento completado.

## 3. Estado operativo comprobado

* `COORDINATOR`: 24 GiB disponibles; GPU sana; OLAP activo sin reinicios;
  Metabase activo; ningun perfil/SNR en ejecucion.
* `WORKER_A`: 19 GiB disponibles; GPU sana y ociosa; sin OOM reciente. Aun no
  es despachable: sus checkouts son anteriores y no posee los bancos D0.
* `WORKER_B`: preflight posterior al reinicio aprobado; 11 GiB disponibles,
  swap sin uso, memoria de kernel normalizada, dos GPU visibles y frias, SSH y
  red overlay operativos, sin OOM en este arranque ni servicios fallidos. Aun
  no es despachable: sus checkouts son anteriores y no posee los bancos D0.

Las tareas C130-C138 son CPU. No se ocupara una GPU para aparentar utilizacion.
Las GPU quedan disponibles para trabajo que realmente use aceleracion despues
de superar las puertas de datos.

## 4. Condicion de aceptacion

La revision pasa solo cuando:

1. ninguna metrica puede superar su presupuesto de memoria calculado;
2. el peor dataset supera un smoke de memoria antes de la campana;
3. fit y transform consumen snapshots ligados a contrato y particion;
4. la bateria causal exhaustiva detecta todos los controles no causales;
5. el wavelet offline no tiene camino hacia features;
6. cada dataset termina con estado y recibo durable, incluido OOM/timeout;
7. perfiles y SNR se completan o publican faltantes exactos;
8. el OLAP carga cada desenlace sin borrar historia.

La orden correctiva vinculante es
`MUSASHI_TO_GENERAL_SATOSHI_C146_C165_MEMORY_AND_CAUSALITY_ORDER_2026_09_13.md`.
