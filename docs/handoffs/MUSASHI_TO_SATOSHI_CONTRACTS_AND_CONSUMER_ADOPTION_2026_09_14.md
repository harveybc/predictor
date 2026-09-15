# Ordenes: contratos de datos y adopcion real de consumidores

Fecha: 2026-09-14. Responsable: Satoshi. Revision: Musashi.
Prioridad: datos reproducibles y uso de los hosts ya desplegados.

## Leer primero y no repetir trabajo terminado

1. [Acta productiva de Musashi](MUSASHI_STORE_HOSTS_PRODUCTION_ACCEPTANCE_2026_09_14.md).
2. [Plan obligatorio actualizado](../integracion_workplan_2026_09_10/09_ADOPCION_DATA_LAKE_DATA_WAREHOUSE_2026_09_14.md).
3. Tu retorno `SATOSHI_HOSTS_PUBLISHED_AND_D2_CORRECTIONS_2026_09_14.md`,
   revision predictor `329af81`, y las correcciones D2 alli citadas.
4. [data-gov](https://github.com/harveybc/data-gov),
   [data-lake](https://github.com/harveybc/data-lake),
   [data-warehouse](https://github.com/harveybc/data-warehouse),
   [financial-data/store](https://github.com/harveybc/financial-data/tree/master/store),
   [predictor/olap/store](https://github.com/harveybc/predictor/tree/master/olap/store).

PUBLICACION Y TRANSICION YA RESUELTAS: ambos repos son publicos, PR 44 esta
integrado en master y ambos hosts estan sirviendo. No pedir otra autorizacion
para esos hechos ni reiniciar de nuevo por inercia. Verificar la identidad
que sirve realmente, sin cambiar ramas en el directorio de datos vivo.

## A1. Conciliar el estado operativo

- Leer el acta y comprobar salud, identidad instalada, inventario, terminal
  del micro-run y conciliacion. Verificar que el cargador sigue avanzando;
  no basta el estado active de un proceso.
- Incorporar los servicios persistentes a la matriz de recuperacion y al
  runbook. No hacer un reinicio de maquina como ensayo en esta orden.
- Actualizar todos los bloqueadores viejos: publicacion privada, PR pendiente
  y transicion sin ejecutar ya no son pendientes del owner.
- No borrar diagnosticos previos ni renombrarlos como evidencia cientifica.

Cierre: matriz de diferencias entre el retorno anterior y los hechos actuales,
con referencias a recibos, no otra campana ni otro terminal equivalente.

## A2. Resolver el primer contrato financiero desde su productor

Seleccionar un recurso realmente requerido por el siguiente experimento;
reutilizar censo, procedencia y caracterizacion existentes. No empezar otro
censo masivo por falta de enlace entre artefactos.

Seguir hasta el productor y documentar: origen, version ejecutada, variables,
unidades, frecuencia, significado de la marca temporal, momento de cierre,
latencia/disponibilidad defendible, revisiones y derechos de uso documentados.
Separar uso historico offline de disponibilidad en vivo. Un nombre de columna,
un hash o un permiso de descarga no demuestra esas propiedades.

Si la fuente no permite demostrar una propiedad, buscar la documentacion
primaria o construir un sucesor con el productor trazado. No rellenar UNKNOWN
con una estimacion conveniente. Registrar que queda desconocido y por que.
No ampliar a los miles de recursos hasta cerrar este primer recorrido.

Pruebas antes de implementar: extremos temporales, zona horaria, barra
incompleta, revisiones, particiones disjuntas y cambio de cola futura que no
altera salidas ya disponibles. Wavelets y transformaciones centradas no
reciben excepciones; medir y declarar cualquier retraso causal.

Cierre: contrato candidato, evidencia del productor, prueba sobre bytes y
configuracion candidata. La instalacion de un contrato que cambie la
elegibilidad cientifica requiere revision; preparar todo sin tocar holdouts
ni esperar al owner para investigar lo que el equipo puede resolver.

## A3. Adopcion por consumidor, empezando por los que ya funcionan

Orden: preprocessor -> feature-eng -> feature-extractor -> predictor.
Usar los paquetes publicados, los endpoints nuevos y el fixture sintetico
separado para pruebas mecanicas con resultados registrados como NON_GOVERNING.
Reutilizar las pruebas gobernadas ya realizadas donde sus identidades sigan
siendo validas; ejecutar solo el recorrido que falta probar.

Para cada consumidor: configuracion reproducible; entrada verificada; hashes
del resultado; costo medido; terminal y conciliacion en el cubo configurado.
Probar exito, fallo, salida antigua y reintento en entorno desechable primero.
Una prueba mecanica acotada por consumidor en produccion queda dentro de esta
orden, sin entrenamiento pesado ni datos financieros sin contrato.

No basta que exista un wrapper: verificar que el punto de entrada real lo
consume. No agregar llamadas remotas dentro del entrenamiento o transformacion.
El resultado local y el outbox deben sobrevivir si el warehouse no responde.

agent-multi, DOIN y live: inventariar los puntos reales y preparar pruebas
offline; implementar la integracion que falte en su repositorio propietario.
No declarar un camino inexistente como activo. Esta orden no activa trading,
publicacion de genes, entrenamiento GPU ni campanas cientificas nuevas.

## A4. Pruebas de reproducibilidad y causalidad de la integracion

Aplicar la metodologia de diseno guiado por pruebas: requisitos -> pruebas
de aceptacion/sistema -> componentes/integracion -> unitarias; implementar
de abajo arriba. Mantener una matriz persistente requisito-prueba-resultado.

Cubrir identidad de entrada y configuracion, alineacion x/y/fechas, ajuste
solo en entrenamiento, continuidad por fragmentos y reinicio, prefijos y
perturbacion del futuro. Los controles deliberadamente no causales deben
fallar. Declarar el alcance de la evidencia; no prometer ausencia universal
de fuga ni datos perfectos. No reabrir auditorias de otras superficies.

## A5. Cerrar documentacion sin alterar D2

Conservar los conteos corregidos 47+6, 39 calibraciones SNR, cinco perdidas
candidatas y dos de controles, 138 filas cambiadas. La comprobacion reportada
de 288 decisiones es un diagnostico de alcance finito, no una nueva tolerancia.
No emitir una aprobacion cientifica nueva por esta orden.

Preparar el diagnostico estrecho del Kalman si falta: factores de hardware y
entorno separados, presupuesto CPU previo, criterio definido antes de medir.
No atribuirlo a la CPU como causa demostrada ni ampliar tolerancias con los
resultados vistos. D3 permanece sujeto a su revision cientifica separada.

READMEs: actualizar ejemplos instalables, proveedores, uso con agentes,
configuracion activa/pendiente y estado por consumidor. Los hechos de
produccion deben enlazar al acta; lo pendiente no se presenta como terminado.

## Recursos y entrega

Trabajo CPU acotado; consultar memoria disponible antes de repartir unidades
entre los tres hosts. Limites por proceso y concurrencia por memoria medida.
No ocupar GPUs solo para mostrarlas ocupadas. No tocar el dataset vivo como
checkout, el historial del cubo, las campanas cerradas ni los servicios sanos.

Entregar un unico MD con: resultados, tabla de adopcion por consumidor,
contrato financiero candidato, pruebas exactas en cada tip final, recibos
gobernados, limites y bloqueadores con responsable y siguiente accion.
Actualizar las etapas del plan al completar cada una. Empujar las ramas;
publicar por PR cualquier cambio no revisado en el codigo del servicio vivo.

No hay una nueva decision del owner necesaria para empezar A1-A5. Si una
fuente requiere una accion exclusiva suya, documentar el recurso y la accion
exactos; continuar con los bloques independientes mientras se resuelve.
