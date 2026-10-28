# Fase 4: extractibilidad y selección predictiva

Estado: diseño de aceptación y controlador implementado; runner científico pendiente.
Autoridad: plan maestro v3, contrato BUSINESS_WEEKLY_WALK_FORWARD y cierres de
fases 1-3. No existe aún un ganador de características.

## 1. Población congelada

Consumir `CANDIDATES_FOR_VALIDATION.json` y su `PHASE_3_FILTER_COMPLETE.json`
para EURUSD y ETH. EURUSD: 365 características admisibles, 14 targets y 686
subconjuntos; ETH: 78, seis y 294. Deducir las características únicas de esos
subconjuntos: una característica compartida por cien subconjuntos recibe un
solo ajuste por pliegue y brazo. Mantener los controles ALL_ADMISSIBLE,
UNIVARIATE_MI, CAUSAL_SUPPORTED y RANDOM_K, y la semilla 0. No reemplazar una
celda fallida por otra semilla. No se lee TEST durante diseño ni selección.

La fase 1 conservó 34 pares EURUSD y tres ETH con respaldo causal; no se usan
como filtro binario. NOT_IDENTIFIED es incertidumbre, no ausencia de efecto.
Las equivalencias exactas de fase 2 se sustituyen por el representante
declarado; los clusters de correlación son propuestas de reducción, no bajas
automáticas. Cada candidato se liga a su target, horizonte, K, conjunto
ordenado de miembros e identidad de fase 3.

## 2. Extractibilidad: tres brazos emparejados

Para cada característica única y pliegue TRAIN `inner_2019` a `inner_2023`,
alinear ventanas históricas de 24 horas transcurridas usando solo valores con
`available_time <= origin`. El tamaño 24 es el primer valor, versionado y
optimizable después. Los tres brazos usan idénticos orígenes, soporte,
normalizador ajustado dentro del TRAIN de ese pliegue, patrón de corrupción,
semilla y presupuesto de puntuación:

- `RAW`: la señal de entrada perturbada, puntuada directamente contra la señal
  limpia solo en los puntos ocultados.
- `RANDOM_ENCODER`: encoder/decoder inicializado con semilla 0 y nunca ajustado;
  misma arquitectura y latente que el brazo entrenado.
- `TRAINED_ENCODER`: Conv1D causal en el encoder, reducción temporal 24 -> 12 -> 6,
  latente configurable; decoder simétrico, entrenamiento con máscaras o ruido
  generado exclusivamente del TRAIN. Early stopping sobre una cola interna
  purgada de TRAIN y restauración del mejor checkpoint.

Las variables estacionales conocidas al origen (hora/día/mes codificados
cíclicamente) pueden entrar como contexto idéntico en los dos encoders. No
entra el target de pronóstico, el resultado de negocio ni el dato futuro.
El decoder existe para medir reconstrucción; el encoder conserva secuencia
temporal `(batch,6,channels)` para el siguiente hito. Guardar peso inicial,
final y elegido, digesto de entradas, cortes de disponibilidad, código, semilla,
época escogida, pico de RAM/VRAM, segundos CPU/pared y resultados por pliegue.

Puntuación: MAE/MSE solo sobre puntos originalmente ocultos y soporte idéntico;
la entrada RAW corrupta es la referencia pareada. Reportar mejora de
TRAINED respecto a RANDOM y RAW, incertidumbre entre pliegues, y error por
escala/estación sin promediar los faltantes como cero. Una mala reconstrucción
no descarta la característica; una buena no demuestra utilidad predictiva.
Las transformaciones admitidas se calculan causalmente dentro de cada ventana;
ningún smoother global puede convertir el futuro en contexto.

## 3. Control de coste

Primero ejecutar un piloto de **coste** con una característica de alta y otra
de baja cobertura, en un pliegue. Medir memoria real del proceso y duración,
fijar cap = 1,25 x pico si cabe; si no cabe, colocar en otro host y registrar
el motivo. El piloto no decide selección. Una semilla por celda; repetir sólo
una celda cuando un fallo técnico haya impedido obtener un terminal válido,
con límite absoluto de tres intentos. El controlador no dispara trabajos
idénticos ya cerrados.

Después ejecutar por tandas: primero los controles RAW y RANDOM; luego
TRAINED en los cinco pliegues sobre toda característica con soporte TRAIN.
Esta primera campaña prioriza evidencia comparable antes que una poda
adaptativa todavía no implementada. Una característica sin observaciones TRAIN queda
`NOT_AVAILABLE_FOR_TRAIN`, sin entrenar modelos. Conservar el denominador
de todas las características y todas las razones de no ejecución.

## 4. Selección mediante utilidad predictiva

Los 980 subconjuntos son *candidatos*, no 980 entrenamientos semanales.
Unir subconjuntos con miembros idénticos por target antes de entrenar.
En TRAIN, usar ranking y extractibilidad para priorizar una frontera corta
por familia y K; conservar ALL_ADMISSIBLE y los tres controles en esa
frontera. El criterio y presupuesto se sellan antes de abrir VALIDATION.

Comparar allí, sobre los mismos orígenes, un predictor temporal fijo con
entrada RAW y con encoder congelado; incluir RANDOM congelado como control.
El predictor no colapsa la dimensión temporal antes de fusionar. No cambiar
su arquitectura ni presupuesto entre subconjuntos. Medir MAE/MSE, naive en
las mismas filas y escala, dispersión entre semanas, cobertura, coste,
número de características y mejora marginal por remoción. El naive debe
superarse estrictamente antes de invocar la estrategia de trading.

El modo primario ejecuta `FULL_RETRAIN_ROLLING_4Y` semanal antes de cada
semana elegible del año completo de VALIDATION, con `available_time`, purga,
early stopping interno y disposición para toda semana. El modo mensual y el
modo estático de literatura usan identidades y tablas separadas. Elegir el
procedimiento por la media de todas las semanas, desempatar por menor número
de características y coste. Congelar selector, modelo, preprocesamiento y
regla de empate antes de recorrer TEST una sola vez. La comparación
raw/encoder no sustituye los posteriores R0/R1/R2 sobre la arquitectura
modular final.

## 5. Automatización y estados

`tools/fs4_campaign.py` planifica características únicas × pliegues ×
{RAW,RANDOM_ENCODER,TRAINED_ENCODER} en SQLite WAL. `init` es idempotente,
`claim` exclusivo, `heartbeat` renueva el arriendo, `complete` verifica
identidad/semilla/digestos/métricas finitas y `status` muestra conteos por
población y ETA **solo** cuando hay tiempos observados y workers activos.
`tools/fs4_worker.py` invoca el runner bajo `crispdm-run`, persiste primero
el terminal local y después entrega el resultado al coordinador por SSH.
El runner científico se registra como un ejecutable real separado; hasta
que esté integrado, ninguna celda de entrenamiento puede figurar completa.

El nivel de subconjuntos y el semanal necesitan controladores equivalentes:
plan sellado, terminal por conjunto/semana, reanudación, lectura de vuelta
al warehouse y cierre sobre el denominador completo. `STATUS.json` de cada
nivel debe incluir estado, esperado, completo, fallido, activo, último
latido, velocidad y ETA o `null` con razón. El agente lee esos estados al
cierre o en petición humana; ningún agente monitorea logs en bucle.

## 6. Aceptación

FS4-01: los dos candidatos que comparten un miembro generan un solo ajuste
por miembro/pliegue/brazo. FS4-02: permutar columnas no cambia identidades.
FS4-03: perturbar futuro o TEST no cambia tareas ni resultados de TRAIN.
FS4-04: variar semilla, población o fila cambia digest y no reutiliza pesos.
FS4-05: RANDOM no ejecuta optimizador; TRAINED registra contador y checkpoint
restaurado. FS4-06: faltante numérico, población ausente o NaN rechazan.
FS4-07: reinicio adopta terminal sin reentrenar; arriendo perdido conserva
diagnóstico y nunca inventa resultado. FS4-08: status/ETA salen del mismo
almacén de tareas, sin inferencia desde logs. FS4-09: los tres brazos usan
idénticas filas y máscaras. FS4-10: ninguna reconstrucción por sí sola elige
característica. FS4-11: la validación semanal recorre todas las semanas y
ajusta solo datos disponibles antes del cutoff. FS4-12: TEST permanece
cerrado hasta que el ganador y la regla estén sellados. FS4-13: cada métrica
se escribe en el OLAP y se coteja por lectura de vuelta; el repositorio
guarda código, esquema, manifiesto y digestos, no el snapshot binario.

Hito posterior: I5-P compara RAW con reconstrucción causal de las entradas
finalmente elegidas, sin alterar el target; RAW sigue si no hay ganancia.
Después formar ramas, comparar R0/R1/R2 y preentrenar el núcleo. NEAT, RL y calendario
económico siguen en el orden del plan maestro.
