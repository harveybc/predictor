# Orden vigente: cierre automatico de seleccion, fase 1

Esta orden sustituye cualquier despacho anterior que mantenga activos PS3-R,
FS-REP, FS-PRED, autoencoders, CVAE, NEAT, arquitectura modular, RL o nuevos
experimentos de literatura. No borra su evidencia. Los pausa hasta que la fase
1 descrita aqui quede cerrada.

## Resultado obligatorio

Entregar una ejecucion automatica y reanudable sobre dos poblaciones separadas:
366 candidatas EURUSD y 83 candidatas ETH. No mezclar sus denominadores ni sus
targets. Para cada unidad dataset-feature debe existir exactamente un terminal
que contenga:

1. Identidad inmutable de recurso, dataset, columna, bytes, intervalo TRAIN,
   calendario/frecuencia, codigo y version de metricas.
2. Inventario temporal: filas, inicio/fin, timestamps esperados/observados,
   fechas ausentes, gaps, duplicados, desorden, nulos, no finitos, cobertura,
   tramos constantes y stale runs.
3. Perfil PS1 completo: escala y cuantiles, MAD/IQR, asimetria, curtosis,
   outliers y severidad, volatilidad, tendencia, ACF/PACF, estacionariedad,
   estacionalidad, espectro, entropia/informacion y coste medido.
4. Celdas crudas de los tres escalones causales contra los target packs EURUSD
   y ETH cuando sus datos point-in-time existen. Si no existen, estado explicito
   `NOT_AVAILABLE` o `NOT_APPLICABLE`; nunca ausencia silenciosa.
5. Un `feature_selection_envelope.v1` validado y entregado al propietario del
   warehouse. El worker no abre DuckDB directamente.

La escalera EURUSD ya posee evidencia retenida para 366 x 14 = 5.124 celdas por
escalon. Debe validarse y adoptarse por digesto, no repetirse. ETH es una
poblacion independiente de 83 features, 18.085 filas y horizontes 1--6 barras
de 4 horas; requiere su propio manifiesto point-in-time y su propia ejecucion.

Al terminar ambas poblaciones, finalizadores independientes deben aplicar
BH/FDR global por dataset/target/horizonte/escalon, publicar decisiones finales
y marcar `PHASE_1_COMPLETE` solo cuando el warehouse lea de vuelta 366/366
identidades EURUSD y 83/83 identidades ETH sin contradicciones ni pendientes.

## Dos comandos, ningun codigo por columna

El producto debe exponer estas dos operaciones conceptuales mediante CLI:

```text
run-column --unit <inventory-record> --target-pack <EURUSD|ETH> --out <dir>
finalize-inventory --inventory <manifest> --terminals <dir> --warehouse <url>
```

Toda diferencia entre columnas, fuentes, frecuencias y targets viene de datos y
configuracion versionada. Se prohibe crear scripts especiales por feature.

## Ejecucion distribuida

- CPU solamente durante esta fase. No reservar ni usar GPU.
- Un proceso de serie por host para que una columna defectuosa no derribe otra.
- Particion determinista por coste estimado de filas/bytes: las unidades mas
  pequenas van a omega; las mayores se balancean entre gamma y dragon. Para
  EURUSD, la particion auditada inicial es 73/146/147; cualquier cambio debe
  conservar el mismo algoritmo y publicar los tres digestos de membresia.
- Omega: limite por hijo <= 2 GiB. Gamma/dragon: limite inicial <= 4 GiB salvo
  medicion retenida que justifique otro valor.
- Claims exclusivos por identidad. Un terminal valido se adopta; nunca se
  repite. Un hijo interrumpido se reanuda o se reemplaza sin duplicar resultados.
- `rc=75`, rechazo de admision y OOM son estados reintentables, no evidencia
  cientifica y no autorizan una decision de seleccion.
- El estado se actualiza tras cada terminal e incluye total, completos,
  no disponibles, fallidos, pendientes, unidad activa por host, tiempos
  observados y ETA recalculada.

## Warehouse y salida visible

El API propietario recibe atomicamente perfiles, relaciones, evidencia causal,
decisiones y recibo. Debe ofrecer vistas para perfil, escalera causal,
seleccion, cobertura y fallos. La consola HTTP del warehouse es la interfaz
operativa; no afirmar que Metabase consulta DuckDB mientras no exista ese
driver en el despliegue.

El cierre produce un snapshot verificado mediante la herramienta de snapshot,
incluyendo tablas de seleccion, conteos y digestos. No se acepta una copia con
`cp` ni se versiona un binario mutable como si fuera evidencia. Se publica el
manifiesto/digesto y una ubicacion descargable gobernada.

## Orden exacto de implementacion y prueba

1. Pruebas rojas del contrato por columna, finalizacion global, envelope OLAP,
   replay, contradiccion, recuperacion y denominador incompleto.
2. Implementar los tres componentes ya despachados en paralelo: worker causal,
   endpoint del warehouse y orquestador del inventario.
3. Integrarlos en entorno desechable con tres series pequenas: una completa,
   una con fechas ausentes y una no disponible.
4. Adoptar por digesto la evidencia causal EURUSD existente y ejecutar solo las
   unidades o metricas realmente ausentes; ejecutar ETH completo en los tres
   hosts bajo su manifiesto independiente.
5. Finalizar BH/FDR, reconciliar contra el warehouse, crear snapshot y verificar
   restauracion en servicio desechable.
6. Publicar retorno con comandos exactos, commits, pruebas, cobertura 366/366,
   URL real de consulta y digesto del snapshot.

## Condiciones de cierre

No declarar fase 1 terminada si falta cualquiera de estas condiciones:

- EURUSD 366/366 y ETH 83/83 con terminal explicito.
- Perfiles requeridos completos o abstencion nombrada por metrica.
- EURUSD y ETH con cobertura causal declarada por target/horizonte.
- BH/FDR calculado sobre el denominador completo, no por columna aislada.
- 0 contradicciones de identidad y 0 envelopes pendientes.
- Warehouse consultado de vuelta y snapshot verificado.
- Dashboard de cobertura muestra 100 % para cada poblacion por separado.

## Memoria de omega

El follower `fs-rep-follower.service` sufrio dos OOM confinados por su antiguo
`MemoryMax=256M`. El valor operativo comprobado es `MemoryHigh=384M` y
`MemoryMax=512M`; conservarlo si el servicio sigue vivo durante la transicion.
No reiniciar omega, gamma ni dragon por este incidente: las tres estan sanas al
corte de esta orden.
