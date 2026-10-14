# I6-E-M: horizontes de negocio, primero downsampling y despues horario

## Superseding dispatch amendment, 2026-10-14

The per-horizon campaign below is retained history/design, not the next
execution. Do not resume its downsampled/hourly queues. The active experiment
is [I6-DMO](../../i6n/direct_multi_output_static/PLAN.md), short six-head DS2
static first, long six-head second, then frozen-pair trading evaluation.
Retain all completed metrics and original identities; no silent conversion
of weekly or single-horizon results into static multi-output results.

Actualizado: 2026-10-11. Estado: EJECUTOR IMPLEMENTADO Y PROBADO; despliegue
y medicion de coste registrados separadamente en el estado operativo.
Este es el subplan vigente para la extension de I6-E. Sustituye la grilla
historica de dieciocho brazos a 72 h como siguiente despacho, sin borrar sus
registros ni modificar los ejecutores/checkpoints sellados.

## 1. Pregunta, alcance y nombres

EURUSD, entradas horarias y prediccion emitida en cada origen horario elegible.
Horizontes fisicos: **2, 3, 4, 5, 6, 24, 48, 72, 96 y 120 horas**.
No incluir 1 h en esta campana. Primero ejecutar todos los brazos downsampled;
despues todos los controles horarios, de menor a mayor ventana.

El nombre visible obligatorio es **naive para el horizonte de X horas**.
No llamar al resultado un naive global, ni sustituirlo por el naive de 1 h.
Cada fila de metrica incluye `horizon_hours`, `naive_horizon_hours`,
`spacing_hours`, `forecast_steps`, unidad y digest de origenes puntuados.
El naive se calcula desde los precios de esas filas, no se copia de otra tabla.

Para k ticks downsampled de separacion delta, el horizonte fisico es h=k*delta:

```
MAE modelo(h) = mean(abs(prediccion_precio(t+h) - precio(t+h)))
naive para el horizonte de h horas = mean(abs(precio(t) - precio(t+h)))
```

Ambos MAE estan en USD/EUR original. MSE esta en (USD/EUR)^2. Si el modelo
entrena con log-return, deshacer su normalizacion y reconstruir el precio
como P(t)*exp(retorno_predicho) ANTES del calculo. Los errores en log-return
se guardan aparte y nunca se comparan con un naive en precio.

Convencion vigente del Maestro: `skill = (MAE_h - naive_h) / naive_h`.
Un valor NEGATIVO mejora; positivo empeora; cero iguala. El porcentaje es
100*skill. Conservar la formula en cada resultado y en el OLAP. La convencion
historica `1-MAE/naive` permanece solo como diagnostico explicitamente distinto;
no cambiar retroactivamente sus registros. La elegibilidad usa MAE_h < naive_h.

Tambien guardar el naive para cada horizonte j*delta, j=1..k, de la trayectoria
downsampled, con su propio soporte exacto y conteo. Por ejemplo, delta=24 h y
k=3 produce naives de 24, 48 y 72 h. El error del pronostico final de ese brazo
se compara SOLO con el naive de 72 h. No comparar la prediccion final con el
naive de su primer tick. Si cambia h, recalcular. Si dos brazos llegan al mismo
t+h sobre las mismas filas, sus naives recalculados deben coincidir; no inventar
una diferencia por cambiar la unidad de ticks del modelo.

## 2. Grilla completa y orden

La historia cubierta es 2*h horas y todos los extremos incluyen el origen t.
Cada ventana contiene t-(n-1)*delta,...,t-delta,t, sin suavizar o interpolar.
Las muestras se desplazan con CADA origen horario, no con una grilla fija diaria.

| h (horas) | Horario: n / historia | k=1: delta / n | k=2: delta / n | k=3: delta / n | k=4: delta / n |
| --- | --- | --- | --- | --- | --- |
| 2 | 5 / 4 h | 2 / 3 | 1 / 5 (=horario) | no exacto | no exacto |
| 3 | 7 / 6 h | 3 / 3 | no exacto | 1 / 7 (=horario) | no exacto |
| 4 | 9 / 8 h | 4 / 3 | 2 / 5 | no exacto | 1 / 9 (=horario) |
| 5 | 11 / 10 h | 5 / 3 | no exacto | no exacto | no exacto |
| 6 | 13 / 12 h | 6 / 3 | 3 / 5 | 2 / 7 | no exacto |
| 24 | 49 / 48 h | 24 / 3 | 12 / 5 | 8 / 7 | 6 / 9 |
| 48 | 97 / 96 h | 48 / 3 | 24 / 5 | 16 / 7 | 12 / 9 |
| 72 | 145 / 144 h | 72 / 3 | 36 / 5 | 24 / 7 | 18 / 9 |
| 96 | 193 / 192 h | 96 / 3 | 48 / 5 | 32 / 7 | 24 / 9 |
| 120 | 241 / 240 h | 120 / 3 | 60 / 5 | 40 / 7 | 30 / 9 |

No exacto significa que la fuente de 1 h no permite delta=h/k entero. Retener
la exclusion; no fabricar datos de 30/45 minutos o redondear el horizonte.
Hay 41 solicitudes logicas y **38 configuraciones unicas**. Las tres solicitudes
equivalentes al horario apuntan al mismo resultado, nunca a otro entrenamiento.

**Etapa D: 28 brazos con delta>1.** Orden inicial por n ascendente (3,5,7,9),
luego h ascendente; producir resultados para todos, no solo los ganadores.
**Etapa H: 10 brazos horarios.** Orden h=2,3,4,5,6,24,48,72,96,120; n ascendente
de 5 a 241. Inicia automaticamente cuando D tiene todas sus disposiciones.
Las solicitudes equivalentes al horario se resuelven en H, sin duplicacion.

Dos veces el horizonte es una HIPOTESIS de contexto controlada, no una ley
matematica ni una garantia de vencer al naive. No concluir imposibilidad por
el fracaso de esta grilla finita.

## 3. Modelo, controles y entrenamiento

Primario NEURAL R0: 20 rasgos RAW del manifiesto seleccionado retenido, mismas
mascaras de fuente y mercado; no afirmar seleccion optima para cada nuevo h.
Un modelo de salida escalar por h/resolucion/semana. Una semilla: 6102026.

- Por feature: Conv1D causal, canales 16, kernel 3; sin colapso temporal.
- Fusion: concatenacion por canales; conservar todas las posiciones temporales.
- Posicion fisica inmediatamente despues de fusion; proyeccion Conv1D 1x1 a 32.
- Dos bloques Transformer completos: dos cabezas, key_dim 16, FFN 64->32,
  Add y LayerNormalization en atencion y FFN; mascara causal y de mercado.
  Dense de FFN actua por posicion, NO aplana el eje temporal.
- Nucleo: tres etapas Conv1D residual causal, kernel 3, canales 24->16->8,
  strides 1, proyeccion del residual y normalizacion; conservar tiempo.
- Cabezal: ultimo estado temporal y salida lineal escalar.

34921 parametros verificados SOLO en el adaptador compacto existente; exigir
el mismo conteo y las mismas formas de pesos en la extension. El adaptador
actual acepta n={3,5,9}; n=7 y las ventanas horarias requieren una extension
probada en un MODULO NUEVO, no una edicion del codigo de la corrida sellada.
No incrementar capas/unidades de un brazo por haber visto su VALIDATION.
Si se necesita otro tamano, declararlo como contraste de capacidad separado.

Control ARIMA(1,1,0), sin drift, ajuste semanal sobre cuatro anos completos de
historia de precio, no solamente n observaciones. Forecast final a k=h/delta
pasos, con el mismo t+h y origenes puntuados del brazo neuronal. Preservar
slots de mercado ausentes, no comprimir dias cerrados en filas consecutivas.
ARIMAX anterior queda retenido; no es requisito para esta grilla primaria.

TRAIN movil de cuatro anos calendario antes del cutoff semanal; inner cuatro
semanas cronologicas, con purga real de etiquetas por h. AdamW, learning_rate
0.001, weight_decay 0.0001, Huber segun contrato heredado, batch 256, max_epochs
50, patience 5, monitor promedio TRAIN/inner-validation, restaurar mejor peso.
No early stopping con la semana externa. Pesos iniciales iguales entre brazos
compatibles; no pesos finales de otra semana salvo contraste nuevo declarado.

R1/R2 requieren donantes compatibles con frecuencia, ventana, escalado,
cronologia y TRAIN semanal; nunca cargar los donantes horarios de 24 muestras
como si fueran donantes de 3/5/7/9 muestras. Es una etapa posterior separada,
no 38 campanas R2 improvisadas antes de resolver la pregunta de muestreo.

## 4. Calendario, filas y TEST

VALIDATION es el ano calendario 2024 completo. El calendario nuevo incluye
**53 segmentos semanales**, con el fragmento final de dos dias; no llamar ano
completo a los 52 segmentos historicos que terminan el 30 de diciembre.
TEST 2025 permanece reservado, con sus 53 segmentos y reajustes semanales;
no se despacha para escoger configuraciones. El primer segmento parcial de
2025 usa cutoff del lunes anterior, no datos entre ese cutoff y el origen.

El numero exacto de filas por h se mide al autenticar los inputs, ANTES de
produccion: t observado, precio positivo en t y en t+h, endpoint horario exacto,
target sin staleness y etiqueta dentro del split declarado. La mascara de
entrada permite huecos explicitos, nunca cotizaciones inventadas. Dentro de
un h todos los brazos usan el mismo vector sellado de origenes; entre h las
poblaciones pueden variar y sus conteos se reportan separados.

No leer extremos posteriores al fin de VALIDATION para completar sus ultimos
horizontes. Un segmento sin origenes elegibles es NOT_APPLICABLE con razon y
conteo, no MAE cero y no un entrenamiento inutil. Los candidatos sin filas se
conservan en el denominador de disposiciones. El TEST no se abre en esta orden.

Primario: 38*53=2014 disposiciones neuronales previstas, no 2014 entrenamientos
garantizados. ARIMA tiene su propio denominador de 2014 disposiciones. No contar
los controles o los pilotos como repeticiones de semilla del modelo primario.

## 5. Automatizacion sin ninera

Esta seccion es el CONTRATO del ejecutor `tools/i6m_*`. La prueba del software
no sustituye el coste y las mediciones de produccion. Estado live: `STATUS.json`
del directorio de campana; instrucciones completas: `I6M_OPERATOR_GUIDE.md`.

1. Un comando maestro autentica grilla, codigo, fuentes, manifest de features,
   ventanas, calendarios y filas por h; inicializa cola durable y outbox.
2. Preflight y pilotos TRAIN miden numero de parametros, VRAM y pico cgroup de
   hijo+envoltorio. Reserva 1.25*pico si cabe con supervisor y servicios vivos.
   No usar el tamano del dataset como estimacion de RAM del entrenamiento.
3. Workers reclaman unidades exclusivas (modelo,h,delta,semana,seed,diseno).
   D usa ambos hosts admisibles; H tambien, con ventanas menores primero y
   las grandes asignadas preferentemente a la GPU preferida tras medir coste.
4. GPU preferida: 5090; un solo hijo en ese host, nunca otro en la 5070 Ti.
   GPU secundaria: 4090 cuando su envolvente cabe. Coordinator: solo controles
   CPU pequenos, cola, ETL y estado; no TensorFlow pesado en el escritorio.
5. La etapa H arranca cuando D tiene todos sus terminales tipados, SIN nueva
   aprobacion, visita del usuario, reporte de agente ni condicion de skill.
   Los controles CPU pueden coexistir con las tareas GPU admitidas.
6. Verificar y adoptar terminales anteriores SOLO con identidad completa:
   mismos inputs, origenes, calendario, init, modelo, entrenamiento y codigo.
   Los resultados historicos de 52 semanas no certifican la poblacion nueva.
   No repetir una unidad validada. Fallos cientificos quedan tipados; continuar
   las unidades independientes. No reentrenar automaticamente para esconderlos.
7. Outbox idempotente: metricas, costes, naives por h, digests y disposiciones
   al OLAP, lectura exacta de vuelta y copias fisicas. Un fallo temporal del
   warehouse NO detiene calculo; la entrega reintenta sin volver a entrenar.
8. Al cerrar D y H, generar informe y PNG desde terminales, sin agente que
   vigile logs. No declarar COMPLETE si faltan entregas, filas o identidades.

El ejecutor final tendra init/run/status a nivel campana, etapa, h, brazo y
semana. STATUS conserva completed/running/queued/failed/not_applicable, lease,
latido, host/dispositivo observado, ultimo avance, causa y proxima unidad.
ETA por clase de n y host desde pilotos/terminales reales, sin dividir por
GPUs inactivas o reservar tiempos imaginarios. No prometemos ocupacion 100%:
ningun recurso admisible se deja sin una unidad preparada compatible; un fallo
de hardware/RAM detiene esa unidad, no la ciencia ni los carriles independientes.

## 6. Aceptaciones y siguiente accion exacta

MH01: h=k*delta y contexto=(n-1)*delta=2h, sin periodos fraccionales inventados.
MH02: 38 identidades unicas, tres alias horarios, todos los diez h presentes.
MH03: D precede H; H orden ascendente; no nueva autorizacion entre etapas.
MH04: MAE y naive por h en precio original y mismas filas; rechazo de otro h.
MH05: perturbacion futura, mascaras, cronologia y save/reload del modelo real.
MH06: semanas parciales y targets fuera del split explicitos; TEST sin lectura.
MH07: restart/adopcion sin duplicacion; ninguna convergencia fallida es un naive.
MH08: WAL/outbox y readback OLAP, error de entrega no vuelve a entrenar.
MH09: costo/memoria medidos por clase; un hijo por host preferido.
MH10: estado/ETA consultable y cierre autonomo de las DOS etapas.

Siguiente accion: desplegar el pin del ejecutor con `tools.i6m_deploy`; el primer
fit de semana cero de cada clase/host mide TRAIN y se adopta sin reentrenar.
No convertir las pruebas de software en afirmaciones de exactitud financiera.

```bash
python -m tools.i6e_horizon_plan plan --output <nuevo-plan.json>
python -m tools.i6e_horizon_plan check --output <nuevo-plan.json>
python -m pytest -q tests/test_i6e_horizon_plan.py
```

Estos comandos solo generan/verifican el PLAN. No leen datasets ni usan GPU.
La grilla versionada esta en `I6E_MULTI_HORIZON_PLAN.json`.
