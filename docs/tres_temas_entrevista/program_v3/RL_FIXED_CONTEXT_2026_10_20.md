# SAC: contexto temporal con presupuesto fijo

## Decision del propietario

Se mantiene desactivado el early stopping costoso del SAC nativo. No se ha
demostrado que el sobreentrenamiento sea la causa principal del fallo de
generalizacion. El contraste BC/DAgger del monitor es un lote separado de diez
celdas; se termina sin ampliarlo automaticamente ni trasladar su monitor al SAC.

## Siguiente secuencia autorizada

Referencia nativa autenticada: 13 observaciones cada 2 barras abiertas, semilla
7, 100000 transiciones, actor Conv1D minimo y criticos nativos. Cambiar solamente
el stride de entrada a 1 y despues a 4, conservando 13 observaciones, las mismas
15 variables y arquitectura, costes, normalizacion y restricciones de negocio.
Los contextos abarcan 12, 24 y 48 intervalos de barras observadas; no equivalen
necesariamente a horas transcurridas si atraviesan un cierre. No se inventan
barras ausentes y se incluye la barra actual completada. La ejecucion sigue
siendo horaria. No cambia el numero de parametros.

TRAIN completo 2020-2023; VALIDATION completo 2024; TEST cerrado. Se mantienen
los controles congelados a 25k/50k/75k/100k, sin paciencia, seleccion de mejor
checkpoint ni restauracion. Cuenta inicial 10000, apalancamiento 50, SL/TP y
volumen obligatorio, una posicion y como maximo una apertura diaria, cierre
forzoso/reapertura con offsets de 12 horas. Costes son supuestos declarados,
no una certificacion de costes de broker.

Dos fits iniciales. Solo si mejora el crecimiento diario de VALIDATION sin
empeorar brecha TRAIN-VALIDATION ni drawdown y con actividad, se compara el
finalista con stride 2 en semillas compartidas 17 y 29. Maximo seis fits nuevos;
el control de semilla 7 se reutiliza desde su cierre autenticado. Este diseno
prueba contexto y resolucion conjuntamente, no sus efectos por separado.

## Ejecucion automatica

Reutilizar cola nativa, publicador y colector existentes. Un worker pesado
secuencial en la GPU preferente, despues del cierre verificado del lote del
monitor, con admision fresca de RAM/disco/GPU. Timers persistentes, intentos
acotados con raices nuevas, sin repetir fits por retraso de publicacion. OLAP
con lectura de vuelta, pesos/configuraciones y copia fisica independiente.
No se toca DOIN ni el carril paralelo de Satoshi.

Desplegado: runtime `9aa461dd717a3a305f196b6ee6977ac076ef2f7c`, spec
`faab4d09ef98d4e1cf1fed8613fc79cbd16e06a61a81b15a2c8a8926736b8861`.
Timers de worker/publicacion y colector independiente habilitados. Primer tick:
`WAIT_PREDECESSOR`, sin fit ni resultado financiero nuevo. El lote previo iba
8/10, sin fallos, y se mantiene intacto. No se afirma que SAC haya arrancado
antes de su relevo real. Regresiones: 125 con integracion nativa, 120 CPU con
sondas de muestreo (conjuntos solapados), 16 contextuales finales y 35 en worker.

Relevo automatico comprobado: el lote previo cerro10/10 sin fallos. Primer SAC
stride1 ya entrenando a01:20:37UTC del21:1000/100000 transiciones en12.4763s,
15 variables,13 instantes,actor1062/criticos1638 parametros. Contrato
`eaeb5fd40805100db8343bbca8ecf16203fce5d6f7fdfbec39e73382519c8112`.
Admision GPU y recarga exacta verificadas; ES=false, final100000. Recibo
`LAUNCH_VERIFIED.json` retenido. No hay resultado financiero nuevo en este corte.
