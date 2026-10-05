# Fases 2 y 3 de seleccion: dependencia, redundancia y filtros

Estado: aprobado para implementacion y ejecucion automatizada el 2026-10-05.
Autoridad: complemento del plan maestro v3. Sustituye cualquier interpretacion
que llame seleccion final a los 34 pares EURUSD o a los tres pares ETH con
respaldo causal de la fase 1.

## 1. Lugar exacto en la secuencia

1. **Fase 1, completa:** perfiles individuales, calidad temporal, asociaciones
   feature-target y escalera causal. Sus salidas son insumos; `SELECTED` en el
   sobre historico significa respaldo de aquella regla causal, no manifiesto
   final de caracteristicas.
2. **Fase 2, activa:** dependencia, redundancia y complementariedad entre
   caracteristicas, calculadas solo sobre TRAIN y con estabilidad temporal.
3. **Fase 3:** filtros reproducibles que producen rankings y trayectorias de
   subconjuntos por activo, target y horizonte.
4. **Fase 4:** extractibilidad raw/random/trained, AE/DAE y alternativas
   temporales sobre poblaciones congeladas de fase 3 y controles emparejados.
5. Validacion wrapper bajo el contrato semanal del negocio; despues siguen
   arquitectura modular, preentrenamiento de ramas, R0/R1/R2 y H-CORE.

No se entrena ningun predictor, extractor, CVAE, politica RL o NEAT durante las
fases 2 y 3. Las GPU quedan fuera de este alcance.

## 2. Poblaciones e identidades

- EURUSD: 366 features, 14 targets/horizontes, identidad y TRAIN del cierre de
  fase 1 `phase1-eurusd-final:94d20c038d55e152`.
- ETH: 83 features, 6 targets/horizontes, identidad y TRAIN del cierre
  `phase1-final:ETH:29d2f745f5d9e87c`.
- Cada poblacion produce `n*(n-1)/2` pares unicos: 66,795 para EURUSD y 3,403
  para ETH. Ningun alias se cuenta como evidencia independiente.
- Toda fila conserva feature izquierda/derecha ordenada, poblacion, fold,
  soporte compartido, lag, metodo, parametros, codigo, entrada y digest.
- El test y la validacion externa permanecen cerrados. Ajuste, discretizacion,
  escalado y estimadores se resuelven dentro de cada fold TRAIN.

## 3. Fase 2: matriz entre caracteristicas

### 3.1 Puerta de identidad y equivalencia

Antes de estimar dependencia:

- igualdad byte a byte sobre filas compartidas;
- igualdad numerica dentro de tolerancia declarada;
- relacion afin exacta o monotona casi perfecta como diagnostico;
- fuente, subyacente, transformacion, unidad, disponibilidad, cobertura y
  antiguedad;
- soporte comun, faltantes conjuntos y desacuerdo de mascaras.

Los alias forman grupos. No se borra una columna: se conserva una disposicion y
un representante propuesto por calidad point-in-time, cobertura y coste.

### 3.2 Metricas por par

Calcular sobre filas alineadas y finitas:

- Pearson, Spearman y Kendall tau;
- informacion mutua con estimador y semilla declarados;
- distance correlation como control no lineal;
- correlacion cruzada en horas transcurridas `{0,1,2,6,24,48,168}` cuando
  exista soporte suficiente;
- estabilidad por folds: media, dispersion, signo, minimo/mximo y numero de
  folds validos;
- soporte efectivo y estados `MEASURED`, `INSUFFICIENT_SUPPORT`,
  `NOT_APPLICABLE` o `FAILED`.

La dependencia es simetrica salvo las relaciones con lag. Un valor alto propone
redundancia; no autoriza exclusion por si solo.

### 3.3 Salida y cierre

La fase 2 termina solo con:

- denominador completo de pares y disposicion para cada celda esperada;
- tablas OLAP `feature_pair_metrics`, `feature_alias_groups` y
  `feature_redundancy_clusters`, o sus equivalentes versionados;
- lectura de vuelta que reconcilia conteos y digests;
- snapshot DuckDB comprimido, SHA-256 y release recuperable;
- `PHASE_2_COMPLETE.json` generado desde evidencia, nunca escrito a mano.

## 4. Fase 3: metodos de seleccion por filtros

Ejecutar por activo, target y horizonte con exactamente la misma poblacion:

1. clustering jerarquico por distancia de Spearman absoluto, eligiendo un
   representante por relevancia, disponibilidad, cobertura y coste;
2. mRMR con informacion mutua para maxima relevancia y minima redundancia;
3. JMI para complementariedad conjunta respecto al target.

Controles obligatorios: `ALL_ADMISSIBLE`, ranking univariado por MI,
`CAUSAL_SUPPORTED`, y `RANDOM_K` con semilla fija. La evidencia causal modifica
una variante declarada del ranking; `NOT_IDENTIFIED` vale cero evidencia causal,
no rechazo. No sumar escalas heterogeneas sin normalizacion predeclarada.

Cada metodo entrega una trayectoria ordenada y subconjuntos para
`K={4,8,12,16,24,32}` limitados por el numero de features disponibles. No se
elige un K ganador en esta fase. Se conservan ranking completo, score y sus
terminos de relevancia, redundancia, complementariedad, causalidad y coste.

La fase 3 termina con `PHASE_3_FILTER_COMPLETE.json`, lectura de vuelta del
warehouse y un manifiesto de **candidatos para validacion**, no una declaracion
de mejora predictiva. La eleccion final ocurre despues mediante wrapper temporal
con el contrato `BUSINESS_WEEKLY_WALK_FORWARD`.

## 5. Automatizacion y recursos

Un solo driver durable descubre trabajo faltante desde terminales por contenido,
no desde memoria del agente. Debe:

- materializar el plan completo antes de calcular;
- dividir pares deterministamente por hash en shards;
- asignar a omega los shards menores y a gamma/dragon los mayores;
- procesar una columna o bloque acotado cada vez, con hilos numericos limitados;
- escribir primero un terminal local atomico y luego enviar al warehouse;
- reanudar sin repetir terminales aceptados;
- emitir `STATUS.json` cada minuto con esperado, completo, fallido, activo,
  velocidad y ETA calculable;
- continuar de fase 2 a fase 3 sin intervencion cuando pase la compuerta;
- abstenerse y conservar diagnostico ante soporte insuficiente;
- detener solo el shard defectuoso; los demas hosts siguen trabajando.

Los agentes implementan, prueban, despliegan una vez y leen retornos terminales.
No supervisan celdas ni ejecutan comandos repetitivos a mano.

## 6. Pruebas de aceptacion

- Mutar una fila futura no cambia ninguna salida anterior.
- Cambiar el orden de columnas no cambia identidad ni resultados por par.
- Un reinicio produce los mismos bytes y no duplica filas.
- Un alias sintetico es agrupado; una relacion no lineal sintetica es detectada
  por MI/distance correlation aunque Pearson sea bajo.
- Un par con soporte insuficiente se abstiene.
- Ningun proceso abre validation/test.
- Un recibo del warehouse con identidad ajena se rechaza.
- El cierre falla si falta una sola disposicion esperada.
- mRMR y JMI rechazan matrices incompletas o poblaciones mezcladas.
- Las doce EURUSD y tres ETH de fase 1 se etiquetan como respaldo causal, no
  como seleccion final.

## 7. Lo que queda despues

Fase 4 compara extractibilidad sobre los subconjuntos de fase 3 y controles.
Luego un wrapper temporal compara los conjuntos con un modelo fijo bajo
walk-forward semanal de cuatro anos, con sensibilidad mensual, naives pareados
y una sola semilla inicial. Las descomposiciones amplias y sus barridos quedan
como extension posterior guiada por evidencia, sin bloquear estas fases.
