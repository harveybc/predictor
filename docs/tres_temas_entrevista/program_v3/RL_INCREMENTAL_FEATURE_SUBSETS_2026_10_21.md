# Seleccion incremental de entradas para SAC

Orden del propietario2026-10-21. Estado: PLANIFICADO, no desplegado.
No modifica el barrido de ventana activo, su subconjunto15 ni su identidad.
La seleccion previa admite candidatas; no demuestra que todas juntas sean el
mejor subconjunto para RL. Esta prueba es seleccion supervisada por desempeno
del consumidor SAC, no repeticion del saneamiento ni del inventario original.

## Lugar en el programa

Despues del grid fino de ventana y antes del contraste de extractores/capacidad:
buscar una base compacta, despues afinar los componentes y revisar interacciones.
Confirmar la mejor combinacion subconjunto/ventana/encoder antes de pasar al
reajuste trimestral/mensual/semanal, que permanece AL FINAL. La representacion
y los subconjuntos optimos pueden depender entre si: no fijarlos para siempre.

## Prioridad inicial de prueba

Lista exhaustiva de las15 entradas actuales, agrupando el par horario.
IMPORTANTE: prioridad percibida para ordenar experimentos, NO ranking de
importancia empirica, resultado de ablacion ni efecto causal identificado.
Antes de lanzar, adjuntar las metricas disponibles de la seleccion anterior
con su definicion, periodo y soporte; si no existen para una variable, indicar
NO_MEDIDO. No inventar puntuaciones para justificar este orden.

| Prioridad | Entradas actuales | Motivo de prueba / posible redundancia |
| --- | --- | --- |
| 1 | RAW_CLOSE_LOG_RETURN | Movimiento reciente, base compacta para patrones temporales |
| 2 | RAW_LOG_RANGE | Amplitud intrabar, complemento al retorno |
| 3 | RAW_ELAPSED_HOURS | Distingue tiempo observado de tiempo transcurrido y cierres |
| 4 | hod_sin, hod_cos | Par horario completo; sesiones, se incorpora como grupo |
| 5 | dow_cos | Estacionalidad semanal; falta su pareja seno en el subconjunto actual |
| 6 | ATR | Escala local de variacion; puede solaparse con rango/std |
| 7 | rolling_std_24 | Volatilidad contextual; comparar aporte incremental a ATR/rango |
| 8 | ADX | Fuerza de tendencia, distinta de direccion |
| 9 | MACD_Histogram | Momento/tendencia a otra escala |
| 10 | DI_minus | Componente direccional; rescatar tambien con ADX si falla solo |
| 11 | RSI | Oscilacion/momento, posible solape con MACD y otros osciladores |
| 12 | Stochastic_D | Posicion relativa suavizada, posible redundancia con WilliamsR |
| 13 | WilliamsR | Posicion dentro de rango; candidato a redundancia con Stochastic_D |
| 14 | HIGH | Nivel absoluto: contrastar su aporte a la politica frente a retornos/rangos |

No confundir seleccion de observaciones de la politica con los datos del
simulador: OHLC, calendario, ATR requerido por ejecucion, precio de referencia,
balance, posicion, presupuesto, SL/TP y estado del negocio siguen disponibles
para sus funciones obligatorias. Inventariar tambien los campos de `state`:
retirar una columna de `features` no prueba eliminar informacion equivalente
si sigue entrando por otra ruta. No borrar series del lago para hacer este test.

## Seleccion hacia adelante con rescate acotado

1. Congelar el orden, grupos, criterios y presupuesto antes del primer fit.
   Reutilizar como referencia el modelo15 de la ventana elegida SOLO si el
   protocolo, transformaciones y artefactos son compatibles y autenticados.
2. Base pequena: retorno log, rango log y horas transcurridas. Entrenar desde
   cero, no importar pesos de otra dimension como si fuese el mismo modelo.
3. Incorporar las demas candidatas en el orden indicado, una por una o como
   grupos inseparables (seno/coseno). Comparar cada S+grupo con el S aceptado
   hasta ese punto. Preservar el orden canonico original de columnas dentro de
   cada subconjunto; prioridad de evaluacion no reordena silenciosamente inputs.
4. Retener el grupo si mejora el criterio congelado de VALIDATION/riesgo/brecha
   TRAIN-VALIDATION. Si no mejora, no incluirlo en el incumbente de esta pasada;
   registrar descartado provisional o inconcluso, no inutil en cualquier modelo.
5. Rescate acotado de interacciones: hasta4 grupos/pruebas predeclarados, por
   ejemplo ADX+DI_minus y volatilidad ATR+std, si fallaron individualmente.
   La seleccion greedy depende del orden; no se presenta como optimo global.
6. Confirmar el subconjunto final frente a las15 con semillas compartidas7/17/29,
   conservando todas las semillas. La primera pasada puede requerir hasta12
   fits; rescate hasta4 y confirmacion adicional hasta4: techo provisional20
   nuevos fits, fijado definitivamente al sellar el driver y los grupos.
7. Revisar localmente la ventana y el encoder para los finalistas; despues
   continuar el afinamiento de componentes. Si nada mejora la referencia,
   conservarla y registrar lo aprendido; no frenar el resto del programa.

Comparar TRAIN2020-2023 y VALIDATION2024 completos, normalizados a la misma
unidad diaria y soporte temporal, sin seleccionar un ano favorable. TEST
permanece cerrado. Mismo negocio, costes, presupuesto100k y checkpoints fijos;
no ES costoso ni reajuste en este contraste. No se exige profit positivo:
elegimos la mejor base relativa para la fase posterior de adaptacion.
Reportar retorno neto, Sharpe con convencion explicita, DD, ordenes, ganadoras,
PF, crecimiento diario, brecha, numero de inputs y parametros actor/criticos.
Reducir variables tambien reduce capacidad en ramas/cabezal: declararlo como
efecto del sistema; usar un control de capacidad acotado si se atribuye la
mejora exclusivamente a informacion y no a simplificacion del modelo.

## Calendario y transformaciones

- Hora: sin(2*pi*hora/24), cos(2*pi*hora/24), mismo reloj y significado de barra
  que la raiz admitida, con timezone/DST versionados. No confundir etiqueta de
  barra, cierre y disponibilidad. Dia semanal: sin(2*pi*dia/7), cos(2*pi*dia/7),
  lunes0; el reloj debe ser el mismo y no comprimir el fin de semana a cinco
  dias para fabricar una periodicidad. Probar pares como grupos.
- La adicion de `dow_sin` es UNA PRUEBA SEPARADA de codificacion, sobre el mismo
  subconjunto y receta. No introducirla a escondidas en la seleccion15. Comparar
  dow_cos actual contra el par completo; completar un par no garantiza mejora.
- La configuracion actual declara rolling_zscore256 y ninguna columna binaria.
  Inventariar la ruta efectiva por campo antes de cambiarla. Para calendario,
  contrastar por separado conservar el par en[-1,1] frente al escalado causal
  actual; no mezclar esa variacion con incluir/excluir variables.
- No ajustar escaladores, imputadores ni descomposiciones al concatenado global
  TRAIN+VALIDATION ni con observaciones posteriores al origen. La eleccion de
  subconjuntos SI usa VALIDATION como desarrollo, declaradamente; no se presenta
  ese resultado como confirmacion independiente ni se consulta TEST.
  No interpolar cierres, no padding sintetico, no suavizado centrado ni wavelets
  con soporte futuro. Toda transformacion aprendida se ajusta en TRAIN; una
  actualizacion rolling debe depender solo de lo disponible hasta t.
- Conservar el tick actual completado y las ventanas de barras observadas.
  Verificar invariancia de prefijo/perturbacion futura por cada transformacion,
  inclusion de t y disponibilidad. La informacion de posicion/riesgo no se poda.

## Hipotesis doctoral y persistencia

Preservar como linea posterior la hipotesis de que cada variable/familia puede
necesitar normalizacion, procesamiento y extractor distintos. No imponer ahora
una transformacion global nueva ni un extractor uniforme por conveniencia.
Separar empiricamente seleccion, transformacion y extraccion; probar primero
un componente por vez y despues interacciones de finalistas. No es un hallazgo
demostrado ni una orden de modificar la campana activa.

Para cada intento: manifiesto ordenado del subconjunto/grupos, candidatas
incluidas/excluidas y motivo, metrica y delta respecto a S y a15, semilla,
ventana/stride, parametros y codigo, transformacion por campo, datos/fechas,
pesos y hashes. Publicar por el OLAP/outbox actual con readback y segunda copia.
Indice navegable y resumen de un parrafo por subfase, incluidos los descartes.

Antes de ejecutar: pruebas de subconjunto exacto y sin fugas por otra ruta,
pares ciclicos inseparables, igualdad de soporte, episodios/estado obligatorio,
config no seleccionada rechazada, ninguna fila de TEST, recarga de cada shape,
reanudar sin duplicar fits y ledger de todos los resultados. El driver/configs
de este futuro lote AUN NO estan implementados ni encolados; la orden presente
es dejar este diseno documentado sin cambiar el trabajo en ejecucion.
