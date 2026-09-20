# RP41-RP48: cerrar la ruta real y ejecutar E1 sin perder el contrato ML

Musashi, 2026-09-19. Satoshi ejecuta. Base publicada `f588400`.
[Dictamen obligatorio](../audits/work_plan/MUSASHI_RP33_RP40_REVIEW_2026_09_19.md),
[reproductor](../audits/evidence/RP40_REVIEW_2026_09_19/reproduce.py) y
[salidas congeladas](../audits/evidence/RP40_REVIEW_2026_09_19/results.json).
Esta orden sucede RP33-RP40; no hay una nueva pregunta al owner para estos bloques.

## Mandato y secuencia

Ejecutar todos los bloques aplicables sin pedir confirmacion entre pasos ya
autorizados. Completar correcciones propias antes del retorno. No afirmar que el
reinicio es el unico pendiente ni que activar un lago conecta automaticamente
los llamadores. Una denegacion efectiva de herramientas no se elude: preservar
evidencia y completar trabajo independiente, sin inventar permiso cientifico.

Metodo vigente: PRE sobre base limpia, aceptacion estructural y comportamental
top-down, implementacion bottom-up, pruebas que fallen al retirar la correccion,
y estado persistente por etapa. No pruebas que cambien solo un assert o una
etiqueta del informe. Reutilizar la infraestructura; no otro rediseno global.

Techo agregado de esta ronda **14 400 CPU s**, con tests, diagnosticos, AE, fits,
fallos, verificacion y cierre incluidos. Registrar wall, CPU y memoria separados;
reservar costo para cierre. No GPU, no entrenamiento RL cientifico, no live,
venue ni reservas. No repetir E0 ni reentrenar el piloto historico. Los fixtures
unitarios sinteticos pueden ser locales y se identifican como tales; resultados
de experimentos y diagnosticos llevan gobernanza y cubo.

Distribucion: coordinador RP41-RP43/RP46 y cierre; WORKER_A diagnosticos ML
independientes RP44; WORKER_B integracion offline RL RP45. Tras aceptacion,
asignar semillas E1 completas a workers con identidad propia y memoria suficiente;
no separar AE y consumidores de sus pesos sin bindings. Verificar capacidad
disponible antes de despachar y no interferir con otros trabajos. Sincronizar
codigo no prueba ejecucion distribuida: informar unidades y costos por host.

## RP41. Reparar el adoptador y demostrar recuperacion

Reproducir F1 antes de editar. Cubrir toda mutacion, incluido primer reinicio,
con manejo de errores y recibo persistente. Timeout, excepcion, retorno no cero,
salud sin contrato nuevo y post-check fallido nunca son exito. Rollback restaura
configuracion y verifica servicio, registrando tambien una reversion fallida.
CLI sale no cero si no adopta; el modo de inspeccion no escribe produccion.

Ligar ensayo aprobado a bytes de configuracion/proveedor/codigo que se adoptan.
Respaldo, inventario y limites de cambio comprobados antes de tocar. La ruta de
aceptacion debe cerrar todas las unidades registradas, incluidas sondas que luego
fallan, y comparar terminal/hijos contra contabilidad y DuckDB. No basta arrancar
un warehouse sin usarlo. Los contratos siguen acotados a los dos paneles; probar
rechazo de rango por semantica explicita del archivo, no solo una fecha elegida
despues de 1970. No inventar disponibilidad a partir de timestamp_label.

Usar el host data-lake/proveedor externo existente conforme a RP33, sin otro
kernel. Si la ruta ensayada no lo atraviesa, nombrar y corregir esa divergencia;
un files_lake embebido en data-gov no prueba paridad del host externo.

## RP42. Un solo camino de ejecucion gobernada para E1

Conectar adquisicion, preparacion, despacho, reanudacion y terminal al entry
point **real** del piloto. Cada unidad registrada antes de leer/trabajar, bytes
y derivados ligados a su entrega y diseno. DATA cacheada no permite saltar la
verificacion; tampoco un job directo o un outcome anterior. Cache reutiliza
computo cuando corresponda, no inventa otra entrega ni cambia actor/run.

Cerrar prepare, pilotos, AE, fits, controles y desenlaces fallidos. Dependencias
fallidas o presupuesto agotado deben conservar poblacion y estado observable,
sin campanas abiertas olvidadas. Outbox durable, reintento idempotente, parser
real y servidor real en stack desechable. Un recibo con reconciliacion ausente,
incompleta o no exitosa no cuenta como cerrado.

Sondas: DATA con digest correcto sin entrega; recibo ajeno/de otra unidad; cache
con origen no verificable; host de entrada ausente antes de trabajo nuevo;
fallo real del hijo; destino terminal caido y recuperado; replay sin duplicado;
dos workers que no comparten actor. Una prueba de report_terminal construyendo
metricas a mano no sustituye llamar run y seguir sus resultados hasta el cubo.

## RP43. Cierre por evidencia, no por nombres de archivo

F3: los dos archivos vacios, JSON malformado, listas vacias, ids ajenos y recibos
sin unidad correspondiente rechazan o quedan no verificados. GOVERNED requiere
hechos por unidad: campana/diseno/actor/recurso/entrega/cronologia y payload del
terminal aceptado. Comparar con contabilidad y warehouse mediante interfaces
existentes; resultado de conciliacion completo y no solo lo presente.

Importacion posterior sigue RETROSPECTIVE/HISTORICAL_UNGOVERNED, aun si hoy hay
recibo de importacion. Registrar alcance desconocido sin imputarlo. Mantener
metricas, inferencia, regimen y gobernanza como ejes distintos.

Eliminar divergencia entre verificacion usada por run_isolated, reanudacion y
cierre final. Reutilizar funciones productivas comunes, no volver a copiar el
verificador. Los casos MAE alterado, NaN/inf/bool, labels/origenes ajenos, pesos
incompletos y diseno cambiado deben fallar **antes** de que un score se consuma
como verificado. Revalidar historia sin fits y comparar hashes de originales.

## RP44. Diagnostico ML que distinga informacion, capacidad y optimizacion

Preservar RP37 y corregir alcance de su lectura. Dos presupuestos no prueban
"a cualquier presupuesto"; siete muestras sin acceso directo al rezago no
prueban que la informacion no sea inferible de correlaciones. Congelar F5.

Declarar antes de medir un diagnostico sintetico de recuperacion de innovacion
independiente: soporte corto sin informacion sobre el objetivo distante,
soporte largo con solucion conocida. Separar tren/validacion por realizaciones
y soporte, semillas, volumen, amplitud/SNR, horizonte y roles, con controles
lineales competentes y controles de etiqueta independiente. Comprobar tambien
la serie redundante de la sonda: el receptor corto **si** puede resolverla.
No exigir fracaso en toda serie correlacionada para dar por correcto el test.

Mantener diagnostico household como empirico DEV, no demostracion universal.
No elegir arquitectura por la que haga ganador a R1 en la validacion publicada.
Mostrar grafo completo, W/h fisicos, alcance medido vs teorico, curva de ajuste,
sesgo/noise floor y limitacion del presupuesto. No dogma de dos periodos, ni
umbral universal de numero de muestras, ni TCN ganador por complejidad.

Updates observados desde optimizer.iterations y limite exacto cuando se promete.
Separar epocas, batches, ejemplos y unidades CPU. Para historia, errata del
redondeo 400->441 y 1600->1638 segun el bucle, marcando que no se guardo el
contador historico. No inventar una observacion retroactiva ni remedir solo por
esa errata. Almacenar todo diagnostico nuevo, favorable o no, con su alcance.

## RP45. RL: cantidad, reloj y ejecucion efectivos

Reproducir F4 con el entorno/broker real de simulacion. Transmitir cantidad
decidida por interfaz del broker o rechazar explicitamente un contrato que el
entorno no pueda ejecutar. No mantener size_units decorativo. Probar fracciones
distintas con broker fijo, cambio de patrimonio, comision, redondeo y limites.

Latencia contratada se ejecuta o se rechaza antes del episodio; no actualizar
solo expected_fill_time. Obtener id, precio, cantidad, tiempo y costo de eventos
reales. No reconstruir un fill desde OPEN[decision+1]. Comprobar la identidad de
la barra del entorno contra la fecha efectiva y disponibilidad, sin aceptar un
contador ausente/constante con apariencia de reloj valido. Hacer fallar replay
con reloj inconsistente y el control que usa una observacion futura.

Orden pendiente no desaparece por elapsed-time sin cancelacion o ejecucion del
broker. Probar cambio de semana con pendiente y posicion, fill parcial si la
interfaz lo soporta (si no, declarar/rechazar ese alcance), venta, rechazo, gap,
latencia 1 y mayor que 1, fallback y modelos opuestos. Mutaciones en el camino
productivo, no en el informe. Esto es aceptacion de software offline, no E3
cientifico terminado ni entrenamiento de una politica.

## RP46. Adopcion acotada con aceptacion operativa real

Tras RP41-RP43: ejecutar ensayo completo con configuracion que se desplegara,
respaldo y rollback probado; revisar procesos activos y ventana sin trabajo
afectado. Adoptar recurso acotado por el procedimiento autorizado, solo servicios
necesarios. No cambiar checkout de un root servido. Identidad de paquete/fuente
y configuracion efectiva, no solo status200.

Dos paneles en catalogo, entregas por actor, cache verificada, disponibilidad
UNKNOWN persistida, terminales aceptados, ninguna campana abierta y conciliacion
por contenido. La sonda y los fallos se registran con su costo. No repetir sondas
productivas previas si ya existe evidencia verificable del mismo objeto.

La importacion historica/migracion aditiva se ensaya y adopta separadamente si
necesaria; no promover el pasado al modo prospectivo. No se requiere autorizacion
nueva del owner para estas operaciones ya ordenadas. Una prohibicion efectiva de
la herramienta se informa exactamente y no se evade; tampoco permite ejecutar
el piloto fuera de gobernanza. Continuar RP44/RP45/cierre independientes.

## RP47. Ejecutar el sucesor E1 completo al pasar sus prerrequisitos

RP42/RP43/RP44/RP46 aceptados antes de arrancar. Conservar el diseno sellado
`143abb57...` sin ejecutar como antecedente; si necesita cambios de contrato,
gobernanza o entrenamiento, emitir sucesor con diff justificado antes de scores.
No copiar la frase "owner action" ni disposicion de terminales solo locales.

Mismo problema household W60/h60 y detector R0/R1/R2, receptor capaz de usar W,
resto entrenable, tres inicializaciones pareadas, AE compartido R1/R2, validacion
AE interna purgada con mascara estable, train-only scaler y controles segun
informacion realmente accesible. Mantener comparacion de preentrenamiento como
pregunta, no seleccion de modelo favorable. Test final intacto.

Revisar las condiciones ML antes de gastar: ventanas unicas y solapamiento,
horas/dias cubiertos, exclusiones por datos, caracterizacion train, grafo y
parametros, convergencia/censura, early stopping y recarga. La proyeccion 2941 s
es de Satoshi: trazarla a mediciones y margenes, no tratarla como tiempo garantizado.
Reservar cierre y no quitar celdas tras resultados para caber. Si el costo no
cabe, entregar deficit exacto y no llamar completo al parcial.

Distribuir por semillas donde proceda; recibir y verificar cada unidad en vivo.
Recomputar MAE/MASE y denominador desde arrays, pesos y tarea; comparar efectos
pareados, incertidumbre descriptiva, curvas y costo total/amortizado de AE.
Una tarea/tres semillas no confirma la propuesta, y SD no demuestra equivalencia.
No repetir la historia ni abrir otras familias para perseguir un ganador.

## RP48. Cierre unico, estado y siguientes dependencias

Actualizar master v3, 09_ADOPCION y PROJECT_METHOD_STATE en cada etapa, sin
mezclar IMPLEMENTED/EXECUTED/VERIFIED/REVIEWED. Publicar PRE/POST, suite con
comando/entorno/skips, costes por host, digests, inventario efectivo y contenido
de reconciliacion. Documentar errores propios y unidades incompletas con ids,
incluido cualquier registro operativo; no esconderlo en un total agregado.

No exigir otra respuesta del usuario despues de cada bloque. Terminar las partes
independientes, corregir defectos propios reproducibles y emitir un solo retorno
para revision externa. No declarar de antemano que paso la auditoria. Causa del
indice, Metabase y terminos conservan su frente separado, no amplian esta orden.

E0 historico se conserva; E1 sigue desarrollo; H-CORE despues de E1/prefijo
congelado; RL cientifico y reentrenamiento semanal siguen obligatorios en su etapa.
Todas las propuestas y las 13 etapas de senal permanecen en el master. Nada de
esta orden concede live ni afirmaciones de rentabilidad.
