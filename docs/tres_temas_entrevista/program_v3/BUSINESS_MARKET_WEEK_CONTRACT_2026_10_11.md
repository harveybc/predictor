# Calendario operativo, supervision y reajuste semanal

Estado: requisito de negocio actualizado por el propietario el 2026-10-11;
diseno de implementacion pendiente de auditoria. No afirma cumplimiento de
los runners existentes. Prevalece sobre grillas experimentales anteriores.

## 1. Invariantes y configuracion

Todos los experimentos que reclamen representar el negocio comparten este
calendario: forecasting, seleccion supervisada, preentrenamiento condicionado
por targets, heuristica y RL. Las comparaciones de literatura usan su modo
separado y no cambian silenciosamente el protocolo de negocio.

- Reajustar cada semana durante el cierre del mercado, antes de la reapertura.
  TRAIN: cuatro anos calendario anteriores al cutoff de informacion, con
  disponibilidades verificadas y targets ya realizados. El cutoff ordinario
  es el cierre de mercado C de la semana anterior; el fit ocurre entre C y O
  siguientes. No usar un cutoff del lunes posterior a la reapertura como si
  ese modelo hubiera sido entrenado durante el cierre.
- No mantener posiciones durante el fin de semana. Posiciones y ordenes
  pendientes deben quedar liquidadas/canceladas antes del plazo forzoso B.
- B = cierre de mercado C menos `force_flat_before_close_hours`, default 12.
- A = apertura de mercado O mas `entry_delay_after_open_hours`, default 12.
- No abrir antes de A ni desde el corte de nuevas entradas hasta B. Default
  propuesto para ese corte: jueves 00:00 en la zona del calendario de mercado,
  configurable; el propietario indico jueves sin fijar una hora concreta.
- Entre ese corte y B solo gestionar/cerrar posiciones existentes; nunca
  reabrir por una senal de prediccion o RL. Desde B hasta A siguientes, plano.

Los offsets son configurables en JSON; la prohibicion de posiciones de fin
de semana es invariante del modo BUSINESS. El modo LITERATURE_STATIC permite
desactivar el reajuste para reproducir un paper. El mensual sigue como
sensibilidad identificada, nunca sustituye el primario semanal.

Contrato de configuracion propuesto, no archivo listo para despliegue:

```json
{
  "evaluation_mode": "BUSINESS_WEEKLY_WALK_FORWARD",
  "retraining": {"cadence": "weekly", "during_market_closure": true, "rolling_years": 4},
  "market_calendar_id": "REQUIRED_VERSIONED_INSTRUMENT_CALENDAR",
  "force_flat_before_close_hours": 12,
  "entry_delay_after_open_hours": 12,
  "new_entry_cutoff": {"weekday": "THURSDAY", "local_time": "00:00", "timezone": "MARKET_CALENDAR"},
  "weekend_positions_allowed": false,
  "forecast_target_policy": "SAME_OPERATIONAL_WEEK",
  "input_history_policy": "ALL_PAST_AVAILABLE_OBSERVATIONS",
  "short_horizons_hours": [2, 3, 4, 5, 6, 12],
  "long_horizons_hours": [18, 24, 36, 48, 60, 72]
}
```

Resolver O y C desde calendario versionado del instrumento/broker, zona,
DST, festivos y cierres anticipados. Un hueco en un CSV no identifica por si
solo un cierre del mercado. No imponer una hora UTC constante todo el ano.
Para activos 24/7 como ETH se requiere una pausa semanal de negocio explicita;
no atribuirles un cierre de mercado Forex que no tienen.

## 2. Entradas, origenes y objetivos son tres cosas diferentes

**Entradas:** conservar datos observados y disponibles hasta t, incluido t,
tambien durante las 12 h posteriores a apertura y previas a cierre en las
que no abrimos/tenemos posiciones. El precio y las features siguen siendo
contexto observable en produccion. No operar no convierte esos datos en ruido.
Una ventana puede alcanzar sesiones previas si su extension lo requiere.
Conservar timestamps, distancia temporal, mascaras de mercado y disponibilidad.
No pegar viernes con lunes como horas consecutivas ni inventar precios.

**Origenes:** producir supervision de forecasting para instantes t en los que
la estrategia puede usarla. `entry_origin` exige A<=t<min(corte_entradas,B).
`management_origin` exige A<=t<B, aunque el jueves ya no abra operaciones.
La evaluacion principal no depende de que una simulacion particular haya
abierto efectivamente una posicion: eso seleccionaria filas por la estrategia.
Metricas de uso real son diagnosticos adicionales con denominador propio.

**Objetivos:** para cada h, q=t+h debe existir exactamente, estar disponible
cuando corresponda y quedar en la MISMA semana operativa: A<=q<B. Tambien
debe pertenecer al split y cumplir purga/cutoff. Un q de la semana siguiente
no se acepta por caer otra vez dentro de horario operativo.
Esta es la interpretacion de excluir horas FUERA de apertura+12/cierre-12.
No recortar un h invalido a B ni cambiar el target por la siguiente barra.

Separar la elegibilidad conocida en t de la existencia futura de una label:

E(t,h,rol)=origen_utilizable(t,rol)
           AND misma_semana_operativa(t,t+h) AND A<=t+h<B.
M(t,h,rol)=E(t,h,rol) AND target_observado_exactamente(t+h)
           AND disponibilidad_y_split_correctos.

E usa solo el calendario y la informacion disponible en t. La estrategia en
vivo nunca consulta si una barra futura acabara existiendo. M incorpora la
label historica para entrenamiento y scoring. La disponibilidad real de una
prediccion (inputs, checkpoint y ejecucion correctos) se registra aparte;
una ausencia no se convierte en prediccion naive ni en error cero.

Aplicar M en TRAIN, inner-validation, VALIDATION y TEST. En direct multi-output
la mascara es por cabezal: un target largo invalido no borra targets cortos
validos. Un ejemplo sin ninguna label elegible no contribuye al fit. Un h sin
soporte no se entrena/puntua como si tuviera error cero. Escalado de target y
pesos de loss por h se ajustan exclusivamente sobre labels TRAIN elegibles.
Recalcular la mascara con la semana historica de cada ejemplo, no con la
semana del ajuste. Seleccionar labels validas ANTES de calcular residuos:
multiplicar NaN por cero no es enmascaramiento correcto. Normalizar cada loss
por su numero de labels elegibles y declarar los pesos entre horizontes.
Un head invalido no aporta gradiente de su label; los parametros compartidos
pueden seguir aprendiendo de los otros heads validos de ese ejemplo.

En extractores no supervisados, las ventanas de entrada elegibles pueden
contener toda esa historia observada; reconstruirla no autoriza a supervisar
precios futuros fuera de la actividad. Los donantes heredan calendario,
soporte, disponibilidad y cutoff, ademas de sus formas y pesos.

## 3. Horizontes compatibles y consumo de predicciones

Con una semana abierta de unas 120 h, A..B abarca unas 96 h; el intervalo
semiabierto exige h<B-A. Los 120 y 144 h no caben, y 96 h tampoco tiene soporte
en ese ejemplo. Esto es incompatibilidad de calendario, no fracaso del modelo.
El propietario autorizo cambiar largos para ajustarlos al negocio.

Nueva propuesta: corto 2,3,4,5,6,12 h; largo 18,24,36,48,60,72 h. Son seis
por familia. Son candidatos operativamente posibles, no horizontes optimos
demostrados. Generar tabla de origenes por h/semana/rol antes de entrenar.
Los historicos 24..144 h se conservan con su protocolo original.

Al acercarse B, disminuyen los horizontes validos. Emitir por prediccion
`horizon_hours`, `target_timestamp`, `eligible_until`, `available` y razon.
No rellenar heads no utilizables con el precio actual, cero o un oracle.
El adaptador de estrategia debe consumir estas mascaras; un modelo que emite
doce numeros sin indicar su elegibilidad no cumple el contrato.

En cada origen calcular R(t)=B-t, en horas transcurridas. Un horizonte fijo h
solo puede usarse si h<R(t), ademas del resto de E. La igualdad queda fuera
porque el target cae en B, cuando ya debe haberse liquidado. Ejemplo de largos
permitidos por el plazo (no autoriza entradas despues del corte del jueves):

| Horas restantes hasta B | Horizontes largos permitidos (h) |
| --- | --- |
| 80 | 18,24,36,48,60,72 |
| 50 | 18,24,36,48 |
| 30 | 18,24 |
| 20 | 18 |
| 10 | ninguno |

Los cortos siguen exactamente la misma regla. Nunca convertir h=72 en un
horizonte variable de las horas restantes. Al invalidarse un largo se conserva
el resto del vector valido; no se elimina el origen entero. Una unica funcion
versionada de elegibilidad debe servir al dataset, adaptador y evaluador.

Apertura: usa largos validos para extremos, candidatos, umbrales y TP/SL.
Regla minima propuesta para auditar: al menos dos largos distintos disponibles
antes de considerar una entrada; menos implica abstencion. No afirmar que
esto ya coincide con el plugin, ni introducir una regla de parcialidad oculta.
Cierre temprano: ordenar todos los cortos/largos validos y buscar SL antes
que TP. La falta de predicciones no desactiva SL/TP reales o cierre forzoso.
El cutoff del jueves solo prohibe abrir, no elimina sus labels de gestion.

## 4. Ejecucion, RL y reajuste

Liquidacion por calendario tiene prioridad sobre cualquier accion de modelo.
La regla debe vivir en el componente comun del entorno/ejecutor, no depender
de que un agente RL la aprenda. Cancelar entradas pendientes que pudieran
ejecutarse despues del corte. Verificar fills y posicion cero; llamar close()
sin fill no prueba cumplimiento. Con fill en barra siguiente, anticipar la
decision para que la ejecucion no ocurra despues de B. Faltas de datos/fill
son fallos de ejecucion retenidos, no liquidaciones ideales inventadas.

Equity/caja/costes continuan entre semanas, pero posiciones y pendientes no:
deben estar a cero. RL conserva calendario, mascaras de acciones, costes y
regla forzosa durante entrenamiento y evaluacion. Cerrar una semana no borra
perdidas ni reinicia capital. Un release tardio no se usa retrospectivamente;
su fallback debe preservar todas estas reglas y quedar identificado.

## 5. Metricas, antiguedad y consecuencias para el plan

Por h y rol, calcular MAE/MSE y naive usando exactamente M, unidades de precio
original y mismos origenes. Skill=(MAE_h-naive_h)/naive_h, negativo mejora.
Contabilizar excluidos por origen inactivo, q fuera de banda, cruce de semana,
endpoint ausente, cutoff y split. Las mascaras preceden a resultados y nunca
dependen de si la prediccion acerto. Agregar todo el ano por errores/conteos.
Publicar diagnosticos sin filtro solo bajo un protocolo distinguible.

Para cada h, S_h es el conjunto de origenes historicos admitidos por M y con
prediccion verificable. Sobre EXACTAMENTE ese conjunto:

MAE_h = sum_{t en S_h} abs(prediccion(t,h)-precio(t+h)) / len(S_h).
Naive_h = sum_{t en S_h} abs(precio(t)-precio(t+h)) / len(S_h).

Ambos usan la misma escala original, target_timestamp y precio observado en t,
tambien con inputs downsampled. No compartir un naive entre horizontes ni
reutilizar uno de filas sin filtro. Si Naive_h=0, skill queda indefinido con
razon; si S_h esta vacio, no hay score. Publicar ademas cobertura sobre todas
las filas elegibles y fallos de prediccion: omitirlos del denominador evaluable
no puede ocultarlos ni permitir aprobar una corrida incompleta.

Para el resultado de USO REAL requerido por el propietario, registrar antes
del desenlace las predicciones que la estrategia efectivamente consulta en
cada decision, incluyendo las consultadas que no disparan orden ni cierre.
Identidad: activo, origen, h, target_timestamp, checkpoint semanal, rol,
calendario y protocolo. U_h es ese conjunto con label observable al puntuar.
Calcular otra pareja MAE_h/Naive_h con la misma formula y exactamente U_h.
Persistir conteos, exclusiones y digest de filas de S_h y U_h en el OLAP.
No sustituir U_h por trades ganadores o solo por el horizonte que disparo la
accion. No duplicar una prediccion al agregar roles. El score de oportunidades
S_h y el score de uso real U_h deben nombrarse por separado: las posiciones
pueden depender del modelo, por lo que U_h puede diferir entre estrategias.
Una simulacion de trading basada en naive con posiciones propias es otro
experimento; no define el naive de error sobre las predicciones utilizadas.

Todas las decisiones, calendarios, mascaras/digests, conteos por h, modelo
semanal, costes y metricas al OLAP con readback y copia fisica. Incluir este
contrato en la identidad de config/donantes/cache/DEAP/DOIN y resultados.
El calendario y sus restricciones son condiciones de negocio, no genes que
el optimizador pueda desactivar para mejorar fitness.

Resultados anteriores no certifican automaticamente este protocolo. Si se
retuvieron predicciones, pueden recontarse sin nuevos fits como diagnostico;
si el entrenamiento incluia labels ahora excluidas o cutoff incompatible,
recontar no equivale al nuevo entrenamiento. Conservar ambos con identidades
distintas. Reajuste semanal es fidelidad al negocio, no garantia de vencer
al naive; el calendario tampoco justifica eliminar retrospectivamente errores.

## 6. Aceptacion requerida

- WMC01: O/C, DST/festivos y huecos de fuente no se confunden.
- WMC02: fronteras A/B exactas, jueves solo entrada, targets misma semana;
  h=72 valido con R=73, invalido con R=72 y R=48, sin renombrarlo ni borrar
  cortos validos. La ausencia posterior de una barra no cambia E historica.
- WMC03: 120/144 h sin soporte se reportan NO_ELIGIBLE_ROWS, no skill cero.
- WMC04: input de hora sin operar sigue presente si se conoce en t; futuro no.
- WMC05: mascaras TRAIN/inner/outer/test y naive por h coinciden; multisalida
  conserva cabezales validos aunque otros no tengan label; labels NaN excluidas
  antes de residuos, denominador cero explicito. Probar las parejas S_h/U_h
  con identidad exacta de filas, escala y conteos, incluidos fallos de forecast.
- WMC06: modelo nuevo no opera antes de su release, ni con cutoff del lunes
  se declara entrenado antes de reapertura; cuatro anos resueltos por semana.
- WMC07: no nuevas entradas desde corte, pendiente cancelada, plano en B;
  fill tardio/faltante falla, no se simula cierre que nunca ocurrio.
- WMC08: heuristica y SAC/DQN comparten restricciones, equity continua.
- WMC09: LITERATURE_STATIC separado; BUSINESS rechaza desactivar la regla.
- WMC10: cache y OLAP distinguen protocolo anterior de este sucesor.

Referencia de calendario, solo ejemplo externo y no calendario del broker
del usuario: OANDA publica horas Forex por instrumento y zona de Nueva York,
incluyendo diferencias de apertura/cierre. Usar el calendario contratado y
su version historica para el experimento:
https://www.oanda.com/bvi-en/cfds/hours-of-operation/
