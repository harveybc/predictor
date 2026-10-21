# Vecinos inmediatos y estabilidad entre semillas

Autoridad: continuacion del propietario2026-10-21. Objetivo: completar el
afinamiento omitido alrededor de13 y medir su estabilidad, no exigir ganancia
estatica ni declarar agotadas las alternativas. Los experimentos previos quedan
intactos. Repeticion fresca13 verificada exactamente: mismos pesos, trazas y
metricas; publicacion10/10, copia independiente completa, TEST no leido.

## Diseno congelado antes del arranque

| Orden | Ventana | Semilla | Origen |
| --- | --- | --- | --- |
| Control | 13 | 7 | Cerrado y reproducido exactamente |
| 1 | 12 | 7 | Nuevo desde cero |
| 2 | 14 | 7 | Nuevo desde cero |
| 3 | 13 | 17 | Nuevo desde cero |
| 4 | 12 | 17 | Nuevo desde cero |
| 5 | 14 | 17 | Nuevo desde cero |
| 6 | 13 | 29 | Nuevo desde cero |
| 7 | 12 | 29 | Nuevo desde cero |
| 8 | 14 | 29 | Nuevo desde cero |

Todas las confirmaciones son obligatorias aunque gane13 en semilla7. Modelo
Conv1D minimo:15 ramas de2 filtros,nucleo4/kernel3/profundidad1,head8;
stride2, ventana incluye t. Doce/trece/catorce observaciones abarcan22/24/26
intervalos observados. Arquitectura nominal fija; los pesos del cabezal y
la inicializacion pueden depender de la forma, como ya documenta la auditoria.
No afirmar capacidad total identica ni atribucion causal exclusiva al contexto.

TRAIN2020-2023, VALIDATION2024, TEST2025 cerrado. Mismos15 campos, transformacion
causal, calendario, SL/TP/volumen, apalancamiento50, una entrada diaria y12h
de margen de apertura/cierre. Mismos costos asumidos.100000 transiciones,
99000 updates, cuatro diagnosticos25k/50k/75k/100k, checkpoint final100k,
sin early stopping costoso, seleccion de checkpoint ni reajuste periodico.

Preservar metricas nativas TRAIN/VALIDATION:profit,retorno,DD,ordenes,ganadas,
profit factor y Sharpe diario no anualizado. Comparar ademas sobre el sufijo
comun35 con hashes de timestamps identicos. Las posiciones y cuentas previas
no se resetean: no es un contrafactual con cuenta inicial identica. Por vecino,
informar diferencia diaria pareada frente a13 para7/17/29,media,DD y brecha.
No elegir semillas convenientes. Promocion semilla7 conserva la regla previa;
confirmacion de estabilidad se reporta aparte, sin inventar un optimo global.

## Ejecucion y pruebas

Nuevo schema temporal_local_window_sweep.v1 sobre el ejecutor nativo existente;
sin tocar SAC, gym, DOIN ni cierres anteriores. Contratos/spec/runtime fijados,
un proceso pesado en5090 y ninguna tarea paralela en5070Ti. Verificacion GPU
renovada, recursos antes de cada despacho, prueba sintetica de forma/gradientes/
recarga en GPU antes de cada fit, dos intentos acotados, espera automatica por
recursos, ledger durable, outbox/readback OLAP y recolector fisico independiente.
No cierres humanos entre ensayos; el cierre exige todos los informes esperados.
Errores de integridad no se eluden. Los recursos y codigos se registran por rol.

PRE:3 fallos esperados antes de implementar. POST:63 pruebas focales; incluyen
replay completo8/8 sin mejora, semillas obligatorias, espera de publicacion,
cierre tras readback, rechazo de presupuesto alterado, autenticacion del
predecesor y gradiente/recarga reales CPU para12/14 (actor1030/1094,
criticos1574/1702). No son resultados financieros ni sustituyen la admision GPU.

Siguiente etapa: seleccion incremental de entradas ya planificada, despues
de leer este lote. Reajuste periodico sigue al final. Este documento no declara
esa siguiente etapa implementada o lanzada.

## Arranque real verificado

YA RUNNING en la GPU preferente: ventana12/seed7,4000/100000 transiciones
en80.6848s, inicio23:52:34UTC del21 de octubre. Runtime agent-multi
d0f0554812f0f16f4a4e18322a192553415c80bc; spec
2d3cd84c2e15955798b626f78cb6dcf6f0b25f9712d27c36bf02ce780c3bd847;
primer contrato75a887f6427b4fad572cba6e17d1ae5fbb23bb6b019307bbc366150eecfb6bdb.
Codigo de agente y pipeline byte-identico al runtime6916e49a de la reproduccion;
solo se amplio el orquestador, pruebas y documentacion. Prueba GPU sintetica:
16 updates, recarga exacta, actor1030/criticos1574.26 pruebas en worker verdes;
63 focales locales y2 integraciones del motor real verdes. Una invocacion local
previa fallo por ruta inexistente del checkout gym; corregida la ruta, pasan.
No se modifico codigo ni resultado por ese error de invocacion.

Timers worker/publicacion/recoleccion habilitados, una sola tarea pesada,
5070Ti sin proceso de entrenamiento. Recibo GPU renovado antes del sello.
Memoria/disco admitidos;~72GiB libres en worker,~673GiB en archivo independiente.
El control ya tiene2/2 nuevos informes comunes con readback,cero pendientes y
copia fisica verificada. Los sucesores no requieren ordenes intermedias.
Duracion provisional del lote~9-11h, basada en4207s del control completo;
primera comparacion12/14~2h20min desde el inicio, no medicion final nueva aun.
Todos los resultados se registran, independientemente del signo.
