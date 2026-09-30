# Despacho del programa completo, no solo la ultima reparacion

## Diagnostico y responsabilidad

Musashi redujo las ultimas ordenes a estrategia y DOIN mientras la cola mantenia
filas antiguas. Retsu entrego esos carriles; eso no demuestra incumplimiento del
ejecutor. Corregir la seleccion de trabajo corresponde al orquestador. Dos
revisiones independientes, con acuse, inspeccionaron producto y programa doctoral
en paralelo el 30 de septiembre; no ejecutaron modelos ni cambiaron servicios.

Observacion de recursos 2026-09-30T19:12Z: trabajador secundario con 21691 MiB
MemAvailable; anfitrion del acelerador preferente con 2763 MiB. Ambos sin scopes
crispdm activos y PSI de memoria avg10=0. Consulta read-only, NO admision ni prueba
de que no exista otro proceso pesado. No usar inventarios de dias anteriores.

## Despacho inmediato y reposicion

Retsu: inspeccionar sus agentes y unidades antes de duplicar trabajo. Dar acuse,
worktree y entregable a cada agente; asignar los siguientes frentes independientes
sin esperar al cierre del conjunto. Como objetivo inicial, cuatro agentes de
implementacion concurrentes si el proveedor lo admite; si admite menos, ocupar
los slots disponibles y dejar los siguientes listos, no inventar despachos.
Los agentes son concurrencia de desarrollo, NO cuatro experimentos GPU.

| Prioridad / propietario | Entregable verificable | Dependencias reales |
| --- | --- | --- |
| 1 / agente estrategia | Ejecutor sintetico con predicciones horarias continuas y causas de cierre; orden bace5fa8, mismo plugin, sin B0 | Pruebas y piloto CPU acotados; no depende de DOIN ni TF |
| 2 / agente H-CORE | Consumidor row-addressable de representaciones ya materializadas, con igualdad de filas/valores y rechazos de identidad, alcance >=5 | Artefacto de c2d4388b; no regenerar ni reentrenar el prefijo. Si faltan bytes locales, usar fixture declarado para construir el consumidor y nombrar la falta antes de afirmar paridad real |
| 3 / agente DOIN | Escritura durable -> referencia -> lectura verificada -> proyeccion por adaptador de servicio desechable | Continuacion de a4f65ba; no cadena ni servicios productivos |
| 4 / agente news-signal | Contrato de calidad versionado con identidad del checkpoint y pruebas de identidades ausentes/mezcladas | Productor quality_eval.py en b08a99f y contrato de proveedor existente; no cargar pesos ni producir nuevo F1 |
| 5 / agente calendario | API pura y CLI offline de admision de estudio segun snapshot de registro, as-of y requisitos | data-gov 7eec868, tools/register_calendar_resources.py y docs/08_RESOURCE_REGISTRY.md; no cambiar registro vivo ni inventar consenso |
| 6 / Musashi, no Retsu como auditor de si mismo | Disposicion externa M4 y justificacion pendiente de donante H-CORE; conciliar Weather/Traffic con entregas | Evidencia existente; no firmas inventadas ni ejecucion confirmatoria |

Al cerrar o bloquear un frente, ocupar su slot con el siguiente admisible sin
esperar auditoria de los otros. Tras el productor de calidad, integrar el
consumidor M5PHET en worktree separado y probar que informe y respuesta corresponden
al mismo checkpoint. Esa tarea SI depende de acordar el contrato del productor;
no fingir independencia donde no existe. No atribuir identidades nuevas a reportes
historicos que no las conservaron.

Antes de cambiar codigo: leer estado de metodo local, pruebas de comportamiento y
estructurales, caso rojo, implementacion, integracion. Reutilizar plugins y APIs
existentes; no segundo scheduler, stack de gobernanza ni arquitectura paralela.

## Recursos y limites

Preferir trabajador secundario para tests CPU. Coordinador para edicion/revision
ligera, sin carga pesada que arriesgue el escritorio. Cada prueba bajo admision
real y limite vigente; inicialmente pruebas pequenas de 2 GiB/120 s, no permiso
para fraccionar campañas mayores. Reserva agregada antes de cada hijo, afinidad y
limite de hilos acorde al anfitrion. Un rechazo de memoria no se resuelve bajando
el tope declarado ni quitando procesos del propietario. Si solo cabe una prueba,
serializar pruebas mientras continua el desarrollo de los demas agentes.

La GPU preferente sigue siendo la externa, cuando exista tarea autorizada y
anfitrion admisible. Siguen SIN autorizacion nueva diagnostico TF, calibraciones
4800 CPU s/3600 pared, codigo remoto de clasificacion, B0, M4 confirmatorio y
operaciones de broker. No usar esta orden para concederlas ni para bloquear los
cinco trabajos CPU independientes. No prometer GPU ocupada cuando no hay trabajo
autorizado listo. Instalacion no es experimento ni ocupa el lugar de uno.

## Corregir el estado, no repetir lo terminado

Reconciliar las filas con tips y artefactos antes de seleccionarlas. A/B y el
contraste R0/R1/R2 terminado NO se relanzan. Weather tiene replay retenido en
f328db3a: separar numerica, custodia y comparacion publicada. Traffic tiene piloto
TRAIN en 91a4c410; no volver a pedirlo como si no existiera. La proyeccion 12-15 h
no es una cota demostrada. M4 requiere review, no repetir todas las reparaciones.
H-CORE tiene salidas materializadas en c2d4388b, no sigue debiendolas.

FIN-LOSS mantiene requisitos de disponibilidad reales; H-CORE no bloquea toda
su preparacion. No hacer una nueva busqueda general de credenciales: usar el
camino documentado y devolver solo el hecho externo verdaderamente ausente.

## Estado que debe devolver el coordinador

Al despachar: tarea, agente con acuse, commit/worktree, recurso, primer entregable
y dependencia. Al terminar cada tarea: resultado y siguiente despacho, sin esperar
al retorno consolidado. Un slot vacio requiere motivo concreto: falta de tarea
autorizada, memoria, herramienta o dependencia; no basta 'esperando revision'.

Informar por separado: nuevas mediciones cientificas, experimentos sinteticos,
funcionalidad usable y reparaciones. Numero de pruebas no es exactitud de un modelo.
Las revisiones de Musashi no son un cerrojo global para el trabajo independiente.
Esta orden es publicada, no una afirmacion de que Retsu la recibio o despacho.
