# T3: traslado al worker con mas RAM

Para Satoshi y el propietario. Sustituye como estado actual el arranque T3 en
la5090; no altera los registros historicos de ese intento.

## Diagnostico y contencion

La primera cola no completo ninguna de sus21 unidades. Registro cinco fallos
de ejecucion:01/03/05/07/09; estaba en11. En cuatro logs se confirma el guard
SUSTAINED_ABOVE_RESPOND de memoria: cuatro muestras sobre37.50 en20s. La09
tiene rc124 y pico retenido4003577856 bytes, pero no tiene la misma evidencia
de incidente; no se atribuye automaticamente su causa solo por el codigo124.
Ninguno es una medicion cientifica completa ni se usa para elegir extractores.

Por orden del propietario, se deshabilitaron timer y servicio propios de esa
cola. Se verifico el comando y la identidad del hijo cientifico antes de
terminarlo; despues no queda ningun hijo de ese paquete. No se tocaron servicios,
trabajos o GPUs de Satoshi. El estado original conserva los cinco FAILED y el
intento interrumpido; hay un recibo separado de parada y traslado. No se fabrica
un CLOSE de exito ni se borran las salidas parciales.

## Cola nueva, ya ejecutandose

La mitad de21 ids impares se reanuda como ejecucion nueva en el worker de32GiB
de RAM y RTX4070 Laptop8GiB. Al preflight habia aproximadamente20GiB disponibles,
con PSI de memoria en cero. El primer proceso cientifico ya aparece en CUDA y
su log nativo confirma fixed_train, consolidated, none y verify=true.
No se declara aun ninguna unidad terminada en esta cola nueva.

Identidad operacional nueva:
`3d91c17b4bbf915135fcbc14f23b5b58d57310301029250552239b6c307d4a1d`.
Identidad operacional padre:
`e6de712c4bd283bb182ae6a9cf061572645d1d3297dc8181d9f2eb5c8dd3060b`.
Recibo de parada SHA256:
`b4951251e30f9b0cbae454bcca45606d5661207b2b299099775c2409840c761a`.

Se autentican162 archivos de entrada contra los pins del padre, ademas del
manifiesto T3, PINS, archivo Git y lista strict116. El mismo science commit
1aaf367, datos, columnas, codificaciones, grupos por unidad, semillas, ventana13/2,
profundidad2, regularizacion base, etiquetas ZigZag y contrato permanecen iguales.
Torch2.13.0/numpy2.5.1/pandas3.0.3/scipy1.18.0/sklearn1.9.0 coinciden con el padre.
Cambian host/GPU/rutas e identidad de ejecucion; no se promete igualdad bitwise
entre familias de GPU. Conservar este dato en la tabla de consumo y en el audit.

No hay adopcion de entrenamiento parcial ni copia de estados FAILED a la cola
nueva. Repetir las21 no duplica mediciones validas: el padre tiene cero recibos
RESULT. El instalador rechaza el traslado si el padre conserva hijos vivos,
hay algun resultado completado, cambian inputs/versiones o no coincide su
identidad. Es una recuperacion de recursos, no una nueva busqueda de parametros.

## Proteccion y automatizacion

Cap4GiB y politica del admisor intactos; no se desactivan oomd, PSI, reservas ni
guardias. La unica excepcion de GPU es el proceso grafico existente, identificado
por comando/start_ticks y limitado a32MiB como en el mecanismo ya probado; no
se admite otro proceso de entrenamiento concurrente.

Worker/timer secuenciales propios con recuperacion, recibos y cierre automatico.
Cada cinco minutos se copia desde el nuevo worker al volumen externo y al segundo
worker, con lectura de vuelta. Primera replica verificada:0 completados,
closed=false. El colector antiguo se detuvo tras una ultima copia del intento
original; no se mezclan los directorios ni las identidades de ambas ejecuciones.

POST:103 pruebas focalizadas CPU aprobadas. Cinco nuevas pruebas PRE fallaban
antes de implementar la autenticacion de traslado; ahora cubren fuente detenida,
ausencia de duplicados, igualdad de versiones y rechazo de inputs/identidades
alterados. No hay cambios a DOIN, TEST ni servicios de trading.

## Consumo para Satoshi

Raiz activa logica: `expert-guided-musashi-t3-relocated-20261023`.
Replica en tu worker: `expert-guided-musashi-t3-relocated-replica-20261023`.
Mirror externo: `shared-selection-e3m-t3-relocated-20261023`.
El spec T3 y las21 identidades tipadas de comando/unidad no cambian; el placeholder
de raiz se sustituye por la nueva raiz activa. La configuracion y los recibos
identifican ese mapeo; no renombrar archivos de resultados para aparentar la raiz
vieja. Estado/recibos siguen en ops/STATUS.json y ops/t3_unit_XX.RESULT.json.

Mantener tus22 unidades, no encolar las impares otra vez en tu GPU. Consumir esta
replica nueva para decision/combinacion al cierre. La estimacion5090 de5-6h deja
de aplicar: falta medir un brazo completo en el nuevo hardware antes de estimar
el resto con respaldo empirico.
