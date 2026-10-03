# Contrato rector de evaluación semanal del negocio

Estado metodológico: `S3_ACCEPTANCE_TEST_DESIGN`. Este documento define el
comportamiento exigido; no afirma que todos los runners ya lo implementen.

## 1. Regla de negocio

El artefacto final no es un modelo estático sino un **procedimiento de
actualización y decisión**. En producción los modelos de pronóstico, los
cabezales de la estrategia heurística y las políticas RL se actualizan durante
la pausa semanal. Por tanto, el veredicto financiero principal se obtiene con
walk-forward semanal, no ajustando una vez y puntuando un año completo con los
mismos pesos.

Para cada semana de evaluación `W_k`:

1. se fija su calendario, primer instante de decisión, cutoff de información,
   inicio/fin de ajuste y release antes de leer outcomes de `W_k`;
2. el corpus de desarrollo es el intervalo de cuatro años calendario que
   termina en el cutoff de `W_k`; ninguna fila entra por estar almacenada si su
   `available_time` supera el cutoff;
3. scaler, imputación aprendida, selección adaptativa permitida, early stopping,
   pretraining y ajuste se calculan solo con información realizada antes de
   `W_k`, con purga derivada del máximo soporte de input, target y tenencia;
4. el modelo liberado se usa únicamente en `W_k`; se guardan su semilla,
   conjunto de filas, digesto de datos, estado inicial/final y presupuesto;
5. al cerrarse `W_k`, sus labels pueden incorporarse a `W_{k+1}` solo cuando su
   disponibilidad real lo permita. Nunca corrigen retrospectivamente `W_k`.

Validation y test son años externos completos divididos en todas sus semanas
elegibles consecutivas. Si el contrato del dataset enumera 48 semanas, deben
producirse 48 disposiciones; una semana excluida conserva razón y población. No
se reemplaza una semana fallida por otra ni se promedia solo lo que terminó.

## 2. Modos incompatibles y configurables

`evaluation_mode` es obligatorio y forma parte de la identidad:

- `BUSINESS_WEEKLY_WALK_FORWARD`: modo primario EURUSD. Actualiza antes de cada
  semana y puntúa esa semana. El procedimiento se selecciona sobre el año de
  validation y se congela antes de recorrer una única vez el año de test.
- `BUSINESS_MONTHLY_WALK_FORWARD`: aproximación de coste y sensibilidad. Se
  actualiza en la primera semana evaluada de cada mes calendario y se reporta
  separada; nunca sustituye el
  veredicto semanal.
- `LITERATURE_STATIC`: reproduce exactamente un paper que ajusta una vez y
  puntúa el bloque completo. Sus cifras son comparables con ese paper, no una
  medición del negocio semanal.

No hay fallback silencioso entre modos. Una tabla que los compare conserva dos
columnas de protocolo y no promedia sus filas.

`update_mode` también es obligatorio:

- `FULL_RETRAIN_ROLLING_4Y`: inicia cada semana desde la inicialización sellada
  y ajusta sobre la ventana móvil. Es el único brazo cuya memoria aprendida se
  limita estrictamente al corpus declarado.
- `WARM_UPDATE_ROLLING_4Y`: parte del checkpoint anterior y refresca con la
  ventana móvil, opcionalmente con un presupuesto menor. Los pesos pueden
  conservar memoria anterior a cuatro años; esa herencia se declara y este
  brazo no recibe la etiqueta de memoria estricta.

La selección de checkpoint usa una cola cronológica interna al corpus de cuatro
años. Tras fijar época/actualizaciones, `FULL_RETRAIN_ROLLING_4Y` puede reajustar
desde la misma inicialización sobre toda la ventana con ese presupuesto. El
procedimiento exacto queda sellado; no se elige retrospectivamente por el score
de la semana externa.

## 3. Qué se congela y qué se actualiza

Antes del test se congelan: tarea y horizontes; política de selección de
características; arquitectura y grupos; hiperparámetros; update mode; cadencia;
costes, riesgo y ejecución; regla de fallback; semillas; naive; y catálogo de
métricas. Durante test solo cambian datos ya disponibles, pesos producidos por
la política congelada y estado legítimo del negocio.

El manifiesto de características del experimento confirmatorio es fijo por
defecto. Un selector adaptativo semanal es otro brazo: su algoritmo y presupuesto
se congelan antes del test, y cada conjunto semanal se deriva únicamente del
rolling TRAIN correspondiente. No se comparan un manifiesto fijo y uno
adaptativo como si fueran el mismo tratamiento.

En RL, equity, posiciones, órdenes pendientes, costes y financiación continúan
entre semanas; cambia la política disponible, no se reinicia el negocio. En
forecasting y estrategia heurística, cada predicción registra el modelo semanal
que la produjo. Solo pronósticos que superen estrictamente su naive pareado en
el protocolo de validation elegible pueden entrar a la estrategia.

Para RL, `FULL_RETRAIN_ROLLING_4Y` reinicia actor, crítico, optimizadores y
replay bajo la semilla sellada. `WARM_UPDATE_ROLLING_4Y` declara por separado
checkpoint padre, estado de optimizadores y tratamiento del replay; un replay
que contiene transiciones anteriores a la ventana impide reclamar memoria
estricta. La política puede cambiar cada semana, pero el entorno financiero no
se reinicia. Para forecasting, los cabezales corto/largo y toda representación
trainable siguen el update mode declarado. Los parámetros de la estrategia
heurística permanecen congelados durante test salvo que validation haya elegido
explícitamente una política de actualización semanal para ellos.

## 4. Fronteras ejecutables requeridas

La implementación se divide en componentes verificables, no en un script que
pueda saltarse pasos:

1. `BusinessProtocol` valida modo, cadencia, update mode, años externos y la
   identidad del procedimiento seleccionada exclusivamente en validation.
2. `WeekCalendar` enumera todas las semanas esperadas y conserva una disposición
   por semana completada, fallida, no elegible o atendida por fallback.
3. `PointInTimeWindow` resuelve los cuatro años calendario, disponibilidades,
   soporte de targets, purga, filas y digestos. No acepta un CSV ya partido como
   prueba de procedencia.
4. `WeeklyTrainer` tiene adaptadores separados para forecasting, cabezales de la
   heurística, SAC y DQN, pero todos reciben la misma identidad de semana/cutoff.
5. `WeeklyExecutor` usa el controlador/runtime semanal existente para releases,
   fallback y continuidad del estado financiero; esos componentes no certifican
   por sí solos el entrenamiento semanal.
6. `ObjectiveFirewall` oculta arrays, labels y métricas del año de test a
   predictor, selector, DOIN y callbacks hasta cerrar el recorrido congelado.
7. `LiteratureStaticRunner` conserva la receta publicada y rechaza emitir el
   esquema o las etiquetas del negocio semanal.

La ejecución mensual es un brazo completo con identidad propia, no un muestreo
de la salida semanal. El coste de los 48 (o el denominador
sellado real) reajustes forma parte del resultado. Si el cómputo obliga a un
piloto, se reduce la población declaradamente para estimar coste; no se publica
ese piloto como año de validation o test.

## 5. Pruebas de aceptación predeclaradas

| ID | Propiedad observable |
| --- | --- |
| BW01 | El enumerador produce exactamente la lista sellada de semanas validation/test y una disposición por cada una. |
| BW02 | Perturbar cualquier byte posterior al cutoff de `W_k` no cambia datos, scaler, selección, pesos ni predicciones de `W_k`. |
| BW03 | Una label de `W_k` puede afectar `W_{k+1}` tras estar disponible, pero nunca `W_k`. |
| BW04 | La ventana FULL empieza cuatro años calendario antes del cutoff y excluye todo dato no disponible, con fronteras DST probadas. |
| BW05 | Warm-start registra checkpoint padre y memoria heredada; no puede declararse memoria estricta de cuatro años. |
| BW06 | Cambiar semanal por mensual cambia identidad, calendario y denominador; no reutiliza métricas semanales. |
| BW07 | Validation puede elegir el procedimiento; ninguna métrica de test cambia selección, hiperparámetros o reglas. |
| BW08 | El recorrido de test ejecuta una sola vez el procedimiento congelado, aunque produzca un checkpoint por semana. |
| BW09 | Un release tardío usa el fallback declarado y conserva la semana; no desplaza su frontera ni mira el modelo futuro. |
| BW10 | Forecast, heurística, SAC y DQN consumen la misma identidad de semana, cutoff, costes y disponibilidad cuando se comparan. |
| BW11 | Estado financiero continúa entre semanas y no se reinicia al cambiar checkpoint. |
| BW12 | Scalers, imputadores, selección, AE de rama y H-CORE nunca ajustan con la semana externa puntuada. |
| BW13 | Cada MAE/MSE incluye naive sobre idénticos orígenes, horizonte y escala; fallo contra naive impide invocar estrategia. |
| BW14 | `LITERATURE_STATIC` reproduce el protocolo publicado y rechaza etiquetas de negocio semanal. |
| BW15 | Una semana ausente, fallida o no elegible aparece en el cierre; no se reduce silenciosamente el denominador. |
| BW16 | Reinicio y reanudación no repiten una semana completada y preservan checkpoint, semilla, datos y recibos. |
| BW17 | Full retrain y warm update reciben mismos orígenes y presupuesto declarado; toda diferencia restante queda atribuible al estado inicial/modo. |
| BW18 | La frecuencia de actualización y el coste completo se reportan; un modo mensual no se promociona como semanal por economía. |

## 6. Evidencia mínima por semana y cierre

Cada unidad conserva `week_id`, fronteras UTC, cutoff y release; modo de
evaluación/actualización; cuatro años resueltos; datasets y disponibilidades por
digesto; filas de fit/inner-validation/score; purga; manifiesto de features;
scalers; checkpoint padre e inicial; semilla; actualizaciones, early stopping y
checkpoint elegido; predicciones temporales hasta verificar el catálogo;
métricas y naive pareados; costes de cómputo; y estado financiero de entrada y
salida. El cierre agrega primero por semana y luego por año, sin tratar barras
autocorrelacionadas como réplicas independientes.

## 7. Referencias de diseño

Este contrato es una decisión de negocio, no una copia de un paper. Es
consistente con la evaluación rolling out-of-sample usada en finanzas: Gu,
Kelly y Xiu reestiman periódicamente y desplazan validation; Cartea et al.
entrenan ventanas consecutivas y despliegan en la semana siguiente. Las
referencias justifican el principio walk-forward, no nuestros cuatro años ni
nuestra cadencia semanal, que son requisitos operativos propios.

- Gu, Kelly, and Xiu, “Empirical Asset Pricing via Machine Learning,” *Review
  of Financial Studies*, 2020, https://doi.org/10.1093/rfs/hhaa009.
- Cartea et al., “Statistical Predictions of Trading Strategies in Electronic Markets,”
  *Journal of Financial Econometrics*, 2024,
  https://doi.org/10.1093/jjfinec/nbae025.
