# Plan maestro v3: programa doctoral y negocio data-centric

**Nombre y alcance general aprobados: M5PHET (24-sep UTC).**
[M5PHET](https://github.com/harveybc/M5PHET) es un framework de tareas ML tipadas,
no un programa generico de investigacion. Sus cinco familias de tareas son:
clasificacion, regresion/forecasting, representacion/no supervisado, RL e inferencia
causal. DOIN es transversal para evaluacion/busqueda distribuida. No imponer Laya
en todos: seleccionar herramientas open-source por referencia reproducida, tarea,
licencia, exactitud/calibracion y costo. Implementado ahora: contrato minimo de
clasificacion consumido realmente por news-signal; los otros motores son roadmap,
no capacidades demostradas. El formulario enlaza ambos repositorios y omite nombre
personal. No se modifica la historia Git ni se promete anonimato de cuenta.
La expansion no bloquea el trabajo experimental ni el carril MT5 demo/Alpaca paper.
Diseno de producto concretado: estado + tarea/esquema -> salida tipada mediante
proveedor compatible. Reutilizar Laya directamente para sus decisiones; M5PHET
anade composicion entre motores, no otro clasificador. Primer input estructurado:
dataset economico existente, con calendario/consenso/actual/revisiones por vintage
y disponibilidad por campo. Sigue representacion jerarquica de mercado,
forecasting multihorizonte con incertidumbre declarada y RL con los mismos inputs.
La inferencia causal exige estudio identificado aparte. Casos UC01-06 y pruebas
P01-P09/CAL01-CAL12 en M5PHET; especificados, no implementados. Ver RP150 ampliada.

**Actualizacion vigente, 24-sep UTC: experimentos + NEWS-LIVE en paralelo.**
[Ordenes RP144-RP151](../handoffs/MUSASHI_RP144_RP151_NEWS_AND_EXPERIMENTS_2026_09_24.md)
y [carril Laya/MT5/Alpaca](program_v3/NEWS_LIVE_PARALLEL_2026_09_24.md).
Retorno RP140-RP143 `07140d03` incorporado como reportado, no como una nueva
auditoria independiente: A agrupa12 celdas, media MSE/MAE0.161962/0.259662;
B10/12 terminales de entrenamiento al cierre reportado, aun requiere replay
y cierre; adaptador doctoral ECL implementado y piloto TRAIN-only ejecutado.
La siguiente comparacion R0/R1/R2 se somete a sus propias pruebas ML y referencia
A, sin esperar retencion ni Laya. No rehacer los entrenamientos aceptados.

Owner aprueba NEWS-LIVE: repo [news-signal](https://github.com/harveybc/news-signal),
Laya local -> features causales -> politica evaluada -> riesgo/ejecucion LTS
existente -> **MT5 demo y Alpaca paper**, primero shadow con datos en vivo.
Prototipo CLI probado; pesos reales, calibracion, collector gobernado y enlaces
broker aun pendientes. Ningun permiso para operar capital real ni reemplazar
estrategias existentes. Publicar el estado honesto en README/formulario.
El coordinador4070/8GiB es candidato para inferencia residente tras piloto medido;
5090 externa mantiene prioridad para cargas GPU sustanciales, sin competir con B.
Hermes/subagentes trabajan por carriles con leases y worktrees separados.
La propuesta typed-ML ampliada es un RFC separado, no capacidad ya demostrada.
Los siguientes bloques fechados son historia y no revierten esta actualizacion.

**Revision RP139, 23-sep al cierre del dia:**
[dictamen](../audits/work_plan/MUSASHI_RP136_RP139_REVIEW_2026_09_23.md) y
[ordenes RP140-RP143](../handoffs/MUSASHI_SOTA_RP140_RP143_2026_09_23.md).
Seis celdas L512 medidas; H336s2021 entrenando en5090. Custodia H192s2021
comprobada en warehouse por Musashi; replica numerica exacta, UUID productor
historico aun UNKNOWN. Limite del scope7GiB, slice compartido8GiB, sin reinicio.
Preparar e implementar en paralelo el contraste modular ECL de horizonte completo
x321 canales. AE con validacion interna purgada dentro de TRAIN, no DEV externo.
Piloto gobernado TRAIN-only acotado autorizado tras pruebas ML; no depende de
retencion ni de terminar B si su referencia es A. No permiso adicional del owner.

**Estado vigente, 23-sep tras regreso del owner:** refrigeracion confirmada en
todos los equipos; GPUs elegibles tras admision, 5090 externa primera opcion.
[Ordenes RP136-RP139](../handoffs/MUSASHI_SOTA_RP136_RP139_2026_09_23.md).
Dos celdas L512 H96 entrenadas: MSE/MAE normalizados del autor
0.125551/0.220453 y 0.125849/0.220934; aun no son una media de tres semillas ni
reproduccion exacta de Tabla 9. Continuacion bajo servicio persistente
`crispdm-rp135-continuation-20260923.service` con flock real: H96 semilla2023
entrenando en la5090 a22:24Z (PID856355); nueve celdas siguientes en cola.
Hermes `t_28e5c655` queda bloqueado/superado operacionalmente: su limite desde
el inicio matutino expiraba cada reintento. No lanzar un ejecutor duplicado.
Los servicios centrales no estaban caidos y no se reiniciaron. Segundo terminal
recuperado sin entrenar otra vez, con errata de timestamps/conflicto a conciliar.
La primera nueva entrega detecto ademas el lago SOTA detenido y deshabilitado:
se habilito/inicio su unidad existente, sin cambiar receta, datos o configuracion.
Intento historico paralelo terminado: metricas retenidas H96 identicas, pero
VAULT_CHANGED por identidad del implementador impidio replay; coordinador
rechazado por carga GPU competidora. Seguimiento de ambos enRP138, sin otra
aprobacion del owner. Satoshi publica el retorno de reparaciones (`e473efbe`).
Asignacion de continuacion: 24 000 s CPU / 8h wall para las diez celdas y su
verificacion, previa a ejecutarlas, conservando aparte los costos anteriores.
Las restricciones de viaje y estados "aun no entrenado" que siguen abajo son
historia. No confundir este relevo con autorizacion para borrar predicciones.

**Revision vigente RP131, 23-sep: avanzar experimentos y separar retencion.**
[Dictamen independiente](../audits/work_plan/MUSASHI_RP128_RP131_REVIEW_2026_09_23.md)
y [ordenes RP132-RP135](../handoffs/MUSASHI_SOTA_RP132_RP135_2026_09_23.md).
Los casos anteriores estan reparados; dos nuevas brechas afectan certificados
de borrado y el informe de revalidacion, no demuestran errores en los scores.
Musashi corrige la orquestacion: **bloquear borrados no bloquea experimentos**.
No falta permiso del owner; no se espera otra revision para iniciar RP135.

- Frente experimental activo: **SOTA-REPRO, receta publicada L512 de TimeFilter**,
  321 clientes ECL, cuatro horizontes x tres semillas, solo 5090 externa admitida.
  Diseno y piloto de costo existentes; unas cuatro horas GPU de entrenamiento
  proyectadas, mas evaluacion/cierre. No es una promesa de duracion total.
- Pregunta: rendimiento de esa receta fuerte con contexto largo, antes de las
  intervenciones doctorales. No es un efecto causal aislado del contexto ni
  reproduccion exacta de Tabla 9: su lookback por horizonte no esta publicado.
- Frente de retencion: RP132-RP134, maximo 1 800 s CPU dentro del techo total de
  14 400 s; sin nuevos borrados. No reconstruir todo ni repetir once reducciones.
- Refrigeracion pendiente solo bloquea los replays en sus dispositivos originales.
  Se conservan T96 y el alcance incompleto de T192 s2021; no se relajan tolerancias.
- Capacidad: inventario real antes de admitir RP135, incluidos arrays retenidos y
  temporales; reserva de disco de 50 GiB, sin asumir borrados ni equipos nuevos.

Estado honesto: las ultimas rondas avanzaron herramientas, no hipotesis doctorales.
RP135 tiene **ejecutor Hermes despachado (`t_28e5c655`), entrenamiento aun no
confirmado**; no se confunde un agente corriendo con una celda medida. Satoshi
se acopla a esa tarea y su lease, no lanza otra campana duplicada. Los defectos de
datos, entrenamiento, metricas, gobernanza o capacidad de la nueva ruta siguen
siendo causas validas para detener esa ruta, no para inventar un resultado.

**Orden adicional del owner, 23-sep: ejecucion concurrente permanente.**
[Politica](program_v3/CONCURRENT_EXECUTION_2026_09_23.md) y
[cola persistente](program_v3/EXPERIMENT_EXECUTION_QUEUE.json).
Satoshi despacha la experimentacion en checkout inmutable mientras otro
worktree/agente repara y otro analiza/prepara el siguiente diseno. Usar instancias
Hermes/subagentes disponibles sin duplicar campanas ni interferir con otras
tareas. Ninguna espera entre pasos para pedir "continua". La admision termica y
de capacidad permanece; no llenar GPUs con trabajo ajeno al plan. Cada auditoria
abre con resultados nuevos o trabajo real en curso y la causa de cualquier
inactividad; no presentar tests de software como avances experimentales.

**Revision historica RP127, 23-sep (superada por RP131):** [dictamen independiente](../audits/work_plan/MUSASHI_RP122_RP127_REVIEW_2026_09_23.md).
Las comprobaciones numericas nuevas detectan los valores falsos anteriores.
Persisten dos brechas reproducidas en la integracion: catalogo sin referencia
numerica habilita borrado, y aceptacion de otro diseno pasa por la ruta destructiva.
Ademas un booleano pasa como conteo numerico cero. No demuestra corrupcion real;
no repetir entrenamiento ni las once reducciones cuya evidencia siga siendo valida.
Orden de esa revision: [RP128-RP131](../handoffs/MUSASHI_SOTA_RP128_RP131_2026_09_23.md).
T96 mantiene replay pendiente y arrays; T192 semilla 2021 mantiene su score
historico y catalogo pendiente de regeneracion identica. Las GPUs originales
siguen retenidas hasta confirmacion fisica; las reparaciones no esperan ese hecho.
Sin GPU, reinicio ni borrado real por Musashi en esta revision.

**Revision historica RP121, 23-sep (superada por RP127):** [dictamen independiente](../audits/work_plan/MUSASHI_RP114_RP121_REVIEW_2026_09_23.md).
Tres fallos reproducidos en fixtures CPU: catalogo acepta ACF/cuantiles falsos y
correlacion nula aunque esta definida; registro local reetiqueta evidencia
diagnostica como cierre; API permite borrar sin cadena aceptada mediante una
opcion explicita. No prueba corrupcion de resultados reales ni exige reentrenar.
Suite focal independiente: 85 passed, 1 skipped. No GPU ni borrado real por Musashi.
Orden de esa revision: [RP122-RP127](../handoffs/MUSASHI_SOTA_RP122_RP127_2026_09_23.md).
Corregir y terminar cobertura numerica por familia/celda, preservando scores y
scorer oficial. T96 sigue MEDIDO / REPLAY_UNVERIFIED; no promedio verificado de
cuatro horizontes ni borrado de sus arrays. Hay trabajo independiente de la
refrigeracion de WORKER_A. Solo 5090 externa elegible tras admision; sin otro
permiso de ingenieria. Errata propia: A usa patch32 (tres parches), B patch128
(cuatro); la afirmacion anterior de cuatro en ambos era un error de Musashi.

**Revision historica RP113, 23-sep (superada por RP121):** [dictamen independiente](../audits/work_plan/MUSASHI_RP106_RP113_REVIEW_2026_09_23.md).
Cinco contraejemplos ejecutados impiden aceptar el cierre: informe rehasheado y
regeneracion ficticia pueden cambiar el score; borrado sin aprobacion/respaldo;
divisor float32 distinto de NumPy para N=16.777.217; catalogo con estimadores
imposibles aceptado con todos sus oraculos internos en cero. No demuestra
corrupcion de las mediciones reales ni requiere repetir entrenamientos. T96
esta MEDIDO con replay pendiente, no carece de datos. No mas borrados hasta
cerrar aceptacion independiente, scorer y validacion numerica por familia.
Orden de esa revision: [RP114-RP121](../handoffs/MUSASHI_SOTA_RP114_RP121_2026_09_23.md).
Solo la 5090 externa es elegible tras admision. Las reparaciones independientes
no necesitan otro permiso del owner ni esperan refrigeracion de WORKER_A.
Musashi no uso GPU, no borro predicciones reales ni reinicio servicios en esta revision.

**Revision historica RP105, 22-sep (alcance superado por RP113):** [dictamen independiente](../audits/work_plan/MUSASHI_RP98_RP105_REVIEW_2026_09_22.md).
Doce celdas TimeFilter entrenadas; doce terminales aceptados tras la adopcion
BIGINT y el reenvio de los tres sobres T=720 por Musashi. Historia del almacen
intacta durante la adopcion. No es aun reproduccion exacta cerrada: faltan
verificaciones numericas y el scorer original float32 de T=336/720.
Dos fallos reproducidos bloquean nuevos borrados: cierre historico que acepta
metricas reescritas y borrado de copias por nombre sin exigir identidad de bytes.
Los seis catalogos reales ya retenidos coinciden con el informe pre-borrado y
sus records/checkpoints con el almacen vivo; no se declaran corruptos.
Orden de esa revision: [RP106-RP113](../handoffs/MUSASHI_SOTA_RP106_RP113_2026_09_22.md).
Reparar, terminar scoring/replays y aplicar la retencion autorizada, sin repetir
entrenamientos ni rebajar la receta. GPU de WORKER_A pendiente de confirmacion
fisica de refrigeracion; coordinador sin compute pesado, sin caducidad automatica.
Ninguna prediccion de produccion borrada por Musashi en esta revision.

**Capacidad de disco, 22-sep:** [presupuesto medido sin hardware nuevo](program_v3/STORAGE_BUDGET_2026_09_22.md).
La campana SOTA primaria de 12 celdas cabe en disco con retencion de metricas y
checkpoints y borrado posterior de predicciones. El total de todas las propuestas
NO tiene aun una poblacion finita cerrada; no se declara que todo cabe. Cada
asignacion usa el espacio real por filesystem, reserva operativa y crecimiento
acumulado; no cuenta RAM/tmpfs, un NAS futuro ni borrados todavia no realizados.

## Restriccion de viaje historica (22-sep; levantada explicitamente el 23-sep)

Preferencia permanente del owner: **la RTX 5090 externa de WORKER_B es la primera
opcion GPU**, especialmente para tareas individuales, incluso cuando se levanten
las restricciones de los otros equipos. Distribuir por tiempo a resultado medido,
no por reparto uniforme. Respetar excepciones de replay en dispositivo original.
Tras una interrupcion electrica, verificar que regreso la eGPU, no solo el host;
sin reinicio automatico ni fallback silencioso. No interrumpir la ronda activa
para aplicar esta actualizacion documental.

El coordinador esta de viaje sin refrigeracion externa: **sin nuevo trabajo
pesado de CPU/GPU, replay o compresion local**. Aclaracion posterior del owner:
**solo la RTX 5090 externa de WORKER_B es elegible para nuevo trabajo GPU**.
La RTX 5070 Ti interna de WORKER_B y la GPU de WORKER_A siguen suspendidas;
el regreso previsto manana por la tarde no levanta esas restricciones.
Sin fallback al coordinador ni a GPUs internas. Servicios y escritorio siguen
activos. Verificar UUID fisico y dispositivo usado por el hijo, conexion,
utilizacion, memoria, compatibilidad y temperatura antes de admitir la eGPU.
Su disipador no refrigera la CPU del portatil: limitar y vigilar tambien el
trabajo CPU del host. No basta que no aparezca un proceso de compute.
Politica y orden complementaria:
[termica y retencion](program_v3/THERMAL_AND_ARTIFACT_RETENTION_2026_09_22.md).
Comprobar predicciones en el worker; no replicar todos los arrays al portatil.
**Enmienda posterior del owner, 22-sep:** no comprimir ni conservar un archivo
permanente de predicciones. Tras el analisis independiente y la persistencia
verificada del catalogo de metricas pertinente, eliminar las copias inventariadas
de las predicciones, incluida la ultima, con recibo de eliminacion y alcance
historico explicito. No borrar antes de ese analisis ni prometer que las metricas
responden toda pregunta futura. STEP-12 generara sus propias predicciones cuando
tenga diseno ejecutable; no obliga a retener todos los pilotos actuales.

## Prioridad vigente: reproduccion del estado del arte (21-sep-2026)

Decision explicita del owner: **SOTA-REPRO primero**. Quedan suspendidas las
nuevas mediciones de los pilotos simplificados E0/E1, contexto electrico y
FIN-LOSS-OPT hasta establecer la referencia publicada correspondiente. No
repetir ni ampliar esos pilotos como sustituto de una reproduccion fiel.
Orden anterior ejecutada parcialmente: [RP90-RP97](../handoffs/MUSASHI_SOTA_FIRST_RP90_RP97_2026_09_21.md).
Orden activa: [RP136-RP139](../handoffs/MUSASHI_SOTA_RP136_RP139_2026_09_23.md).
Politica: [SOTA_FIRST](program_v3/SOTA_FIRST_2026_09_21.md).

La prioridad inmediata es el benchmark Electricity/ECL de 321 clientes horarios
que el owner acaba de citar, distinto de UCI235 (hogar individual por minuto).
No se trasplantan scores, ventanas, escaladores ni conclusiones entre ambos.
La referencia seleccionada es TimeFilter (ICML 2025), revision del autor
dffde87e, protocolo L96. La seleccion no prueba supremacia sobre toda publicacion
de 2026. El protocolo de lookback buscado de la Tabla 9 necesita su propio enlace
por celda a la receta del autor; no se identifica automaticamente con L512.

Los resultados exploratorios previos se conservan como **HISTORICAL_DEV_ONLY**:
fuera de la seleccion activa, recomendaciones de arquitectura/loss y decisiones
de preentrenamiento o negocio. Esta disposicion no declara falsos sus valores
ni modifica recibos historicos. La exclusion ejecutable de vistas/selectores
fue incorporada por RP90; se conserva su evidencia y alcance. No se borra historia para
ocultar errores. Controles ingenuos siguen siendo instrumentos de comparacion,
no candidatos sustitutivos del modelo de referencia.

La reproduccion conserva datos, protocolo, arquitectura completa, entrenamiento
y metricas oficiales. El presupuesto se adapta a la receta, no se reduce la
receta para llamarla reproduccion. Despues de reproducir y auditar, se retoman
las hipotesis doctorales y la revalidacion financiera/RL, con la referencia fuerte.
La exactitud del protocolo es obligatoria; la igualdad numerica se comprueba,
no se promete ni se consigue ajustando reiteradamente contra test.

## Historial anterior (no autoriza nuevas campanas)

Actualizado: 2026-09-21. Responsable del programa y revision: Musashi.
Ejecucion delegada: Satoshi. Estado: E0-DEV, etapa ARCH y piloto E1 EJECUTADOS;
piloto historico no gobernado y SUCESOR E1 EJECUTADO; sin confirmacion.

**Vigente:** [RP66-RP73: correcciones y referencia emparejada](../handoffs/MUSASHI_POST_HUBER_RP66_RP73_2026_09_21.md).
**Ejecutado por Satoshi (21-sep, RP82-RP89, pendiente de revision):** [retorno](../audits/work_plan/SATOSHI_RP82_RP89_RETURN_2026_09_21.md): una sola autoridad de verificacion (custodia de la preparacion aceptada, denominador del contrato, identidad de configuracion y etiquetas) consumida por tabla y cierres (70 filas: 39 verificadas, 31 preservadas con alcance calificado); comparador atado a identidad; selector financiero v4 por estrato loss x optimizador; intervalos por bloques con soporte declarado y certificados de cobertura (ninguno a 26/2); bloque de contexto rezago diario (paciencia 10, 3 anfitriones): rezago 0.5339 vs modular 0.5416 MAE_z, -0.0077 en 3/3 (persistencia 0.6766; 5/6 censuradas); piloto de costo financiero sellado y RECHAZADO por el servicio (422: recurso sin contrato de disponibilidad) — accion del operador, sin campana.
**Ejecutado por Satoshi (21-sep, RP74-RP81, pendiente de revision):** [retorno](../audits/work_plan/SATOSHI_RP74_RP81_RETURN_2026_09_21.md): custodia por cadena de artefactos aceptada (64 filas: 45 verificadas, 19 historicas preservadas como METRIC_ANCHORED), comparador verificado por la ruta estricta, poblacion financiera = tensores consumidos, seleccion por configuracion con semillas emparejadas, intervalos por bloques con soporte declarado; bloque arquitectura x calendario (paciencia 10, 15 celdas, tres anfitriones contabilizados): GRU+calendario 0.4798, modular+calendario 0.4927, GRU 0.5290, modular 0.5391, control 0.5446 MAE_z (persistencia 0.6766; 7 de 15 censuradas en 4 000); Q2 sigue limitado (111 000 s); finanzas disenadas v3, sin campana.
**Ejecutado por Satoshi (21-sep, pendiente de revision):** [retorno RP66-RP73](../audits/work_plan/SATOSHI_RP66_RP73_RETURN_2026_09_21.md): contratos tipados enlazados al runtime, tabla de cierre por contenido (49/49), bloque DEV emparejado ejecutado y cerrado (GRU adaptada 0.5313 MAE_z vs modular 0.5680, persistencia 0.6766; 3/3 semillas), Q1 calendario (0.4868 vs control 0.5529) y Q3 volumen (112 d 0.5575, 56 d 0.5608, 28 d 0.5680) ejecutados y cerrados; Q2 NOT_EXECUTED por limite medido (4.9 s/actualizacion en W=1440); finanzas: tarea por tiempo transcurrido y runner gobernado con FL01-FL08 ejecutables sobre datos sinteticos, sin campana cientifica.
**Revision de 4ef9f71:** [dictamen ejecutado](../audits/work_plan/MUSASHI_POST_HUBER_REVIEW_2026_09_21.md).
Historicos preservados; contrato y tabla no aceptados como verificadores suficientes.
Antes de medir: corregir el alcance 67 vs 60, frecuencia de validacion al variar
volumen, escala comun, horizonte financiero por timestamps y deltas Huber cero.
RP72 autoriza condicionalmente un bloque DEV acotado de referencia GRU adaptada
frente al modular, despues de sus pruebas y piloto de costo, sin nueva consulta
al owner. Fase 2 por bloques completos despues; finanzas cientificas, reserva y
reproduccion completa del articulo siguen pendientes de sus revisiones especificas.
**Regla transversal desde 21-sep:** [contrato de comparabilidad con literatura](PROGRAM_METRICS_CONTRACT_v1.md#adicion-obligatoria-comparabilidad-con-literatura-2026-09-21)
obligatorio para TODOS los dominios, incluido finanzas. Antes de entrenamiento
cientifico: target/transformacion/metrica y protocolo de referencia fijados;
reproduccion publicada o referencia reejecutada en la misma tarea del negocio,
sin comparar cifras de protocolos diferentes. Implementacion transversal en
los runners: pendiente de Satoshi, tarea BENCHMARK-CONTRACTS; no declarada hecha.
**Cierre obligatorio, incluido el actual:** tabla en el mensaje al owner con
error del modelo, metrica/escala, naive en las mismas filas, mejora relativa y
valor comparable de literatura con fuente; por tarea/horizonte/split. Punto 14
de las ordenes anteriores, mantenido en RP73. Si falta una referencia comparable, declararla pendiente
con razon; nunca sustituirla por cifras incompatibles ni inventadas.
RP57-RP64 y el factorial electrico Huber/AdamW estan ejecutados y preservados.
**Adicion aprobada:** [comparacion financiera obligatoria de losses y optimizers](program_v3/FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md).
No hay loss ganadora para trading; no descartar mejoras normalizadas de 1e-5/1e-6
por pequenas. No buscar el OLAP antiguo ni repetir heuristic-strategy como requisito.
**Comparacion electrica obligatoria:** el punto 12 de la orden anterior preparo
una referencia reproducible UCI 235 y nuestro modelo bajo identicos target,
horizonte, particiones, transformacion y metrica publicada, con ingenuo sobre
las mismas filas. Entregables: matriz de protocolo, configuracion reproducible,
tests y presupuesto. MAE_z complementa, no sustituye esa metrica. La preparacion
no equivale a haber ejecutado la reproduccion. La autorizacion limitada nueva
es RP72, no la etiqueta de un contrato ni el documento de preparacion.

Antecedente de [RP57-RP64](../handoffs/MUSASHI_PROGRAM_RP57_RP64_2026_09_20.md):
[Revision ML y comparadores](../audits/work_plan/MUSASHI_E1_ML_BASELINES_REVIEW_2026_09_20.md):
el sucesor SI se ejecuto en RP55. Musashi recomputo nueve fits sobre 10 020
origenes conservados, con naive/ridge sobre filas iguales. R0 MAE 0.546929 kW
vs persistencia 0.617372 (+11.41%); ridge 0.545495 (+11.64%).
Evaluados en log1p(kW), R0 0.266344 vs naive 0.279127 (+4.58%).
No son porcentajes de aciertos ni replicas de un entrenamiento logaritmico.
El cociente historico usa persistencia h60, no MASE convencional m1.
Corregir presentacion/etiqueta sin cambiar la propuesta; diagnosticar datos,
objetivo, stopping, contexto, volumen y arquitectura antes de repetir R0/R1/R2.
TCN propia adaptada no equivale a replica del articulo; referencia independiente
obligatoria. No asumir que mas epocas mejoran validacion.
RP56 declara cierre gobernado; esta revision verifica arrays locales, no reemplaza
la auditoria completa de adopcion/warehouse. Historicos conservados, sin ganador
universal ni confirmacion.
No hay nueva decision del owner para el desarrollo autorizado. No repetir E0.
Antecedentes financieros: el owner identifica una mejora valida en fase 3 y
un resultado anterior afectado por fuga wavelet. RP58 debe separar ambos por
linaje. El TCN NEAT antes citado es CAUSALITY_UNVERIFIED, no benchmark aceptado.
[13D](program_v3/13D_MUSASHI_BUSINESS_DISPOSITION_2026_09_19.md)
resuelve el escenario de simulacion: capital real y exposicion no bloquean DEV.

**Invariante de E1:** R0 = detector aleatorio entrenable; R1 = detector
preentrenado congelado; R2 = los mismos pesos iniciales preentrenados ajustables.
El resto del modelo se entrena en los tres. Crudo/agrupado/fusion y arquitectura
son factores separados; nunca redefinen estos codigos. El checker documental
verifica el contrato; gradientes/estados/pesos verifican su ejecucion.

**Invariante de ejecucion:** nuevos experimentos registran campana y entrega
ANTES de preparar/fitear; terminal local no reemplaza contabilidad y cubo.
Importacion historica es evidencia retrospectiva separada, nunca una campana
prospectiva reconstruida despues. Registrar recursos es trabajo tecnico del
ejecutor por el procedimiento existente, no una nueva decision del owner.

**Antecedente aprobado, 18-sep:** comparar ARCH-A (Conv1D local, referencia), ARCH-B
(convolucion dilatada/TCN), ARCH-C (Conv1D + GRU/LSTM) y control sin extractor
aprendido dentro de E0 antes de elegir el procedimiento. Hipotesis
[H-CORE](CORE_PRETRAINING_HYPOTHESIS_2026_09_18.md)
incorporada despues del desarrollo del extractor y su prefijo congelado verificable.
Retorno RP1-RP8 recibido en `2347c7b`; esta adicion se adopta en el sucesor,
sin cambiar la corrida sellada. [Revision ejecutada y limites](../audits/work_plan/MUSASHI_RP1_RP8_REVIEW_2026_09_18.md):
69 records sin discrepancia en la comprobacion local de datos/errores, pero cierre
incompleto, descriptor de tendencia incorrecto y atribucion H3 limitada.
No borrar el piloto ni repetirlo entero: corregir, demostrar reuso y ejecutar
[RP9-RP16](../handoffs/MUSASHI_MOD_E0_RP9_RP16_2026_09_18.md).
La correccion de tendencia mantiene las 15 particiones del piloto en el reanalisis.
Ese era el estado al emitir RP9-RP16. La etapa informativa ARCH-A/B/C/0 ya se
ejecuto; su lectura queda corregida por el dictamen de 19-sep. H-CORE sigue
despues de E1 y del prefijo verificable.

## 1. Autoridad, alcance y correccion de rumbo

Este es el indice operativo vigente. Sustituye la secuencia inmediata del master
v2, del plan CRISP-DM de 12-sep y de las ordenes sinusoidales A/C de 18-sep.
No cambia las hipotesis de las propuestas, no reabre reservas, no borra resultados
y no concede aprobacion cientifica a una implementacion por tener tests verdes.
Si una propuesta cambia, actualizar primero su trazabilidad aqui.

La fuente principal es la [propuesta modular editable](../propuesta_doctoral_representaciones_temporales_modulares.tex),
especialmente las secciones de procedimiento, experimentos y contrastes; su
[PDF](../propuesta_doctoral_representaciones_temporales_modulares.pdf) conserva su contenido.
Se comprobaron identicos los bytes de la fuente en la rama de trabajo y master.
El [inventario previo](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/integracion_workplan_2026_09_09/02_INVENTARIO_PROPUESTAS.md)
contiene **seis objetos cientificos**, no cinco. Se organizan en cinco frentes
operativos porque senales contiene P-PRE y P-TRN, pero sus contrastes NO se fusionan.
P-3F es arquitectura compartida, no una septima tesis.

La revision abarca el master completo, los tres patches, el plan CRISP-DM, la
integracion jerarquica, los protocolos de las propuestas y el estado 12C/12D/12E.
Los documentos extensos de cada STEP siguen como especificaciones subordinadas;
su literatura y cada implementacion no quedan certificados por esta revision.

Hallazgos corregidos:

1. La campana C1-C6 de seno univariado no contrasta agrupacion ni fusion. Se retira
   como siguiente campana; sus comprobaciones compatibles se reutilizan como tests.
2. E0 de P-MOD es un bloque multivariado de desarrollo y posterior reserva H2/H3,
   no una prueba general de que una red aprende senos.
3. El plan CRISP-DM mezclaba entregables P-L2 con P-MOD y condicionaba RL a una
   cadena completa I0-I10. Se reemplaza por dependencias cientificas por experimento.
4. La cobertura D3 de operadores no cubre feature engineering, redes modulares,
   seleccion de variables ni utilidad de trading. Los estados se separan.
5. El plan I-INFO contenia cotas falsas de informacion sin ruido y un rango efectivo
   mal nombrado. Rige el [contrato corregido de metricas](PROGRAM_METRICS_CONTRACT_v1.md).

## 2. Trazabilidad de todas las propuestas

| ID / frente | Fuente y pregunta | Experimentos obligatorios y comparadores | Metricas y entregable |
|---|---|---|---|
| P-MOD / modular | [Fuente](../propuesta_doctoral_representaciones_temporales_modulares.tex): perfiles, grupos, contextos y fusion para pronostico y RL | E0-DEV mecanismos; E1 familias publicas de desarrollo; E0-CONF H2/H3 y E2 H1 reservados; E3 aplicacion financiera/RL independiente | MASE y efectos H1/H2/H3, incertidumbre y costo; procedimiento fijado, tablas/figuras por tarea, comparacion RL obligatoria |
| P-L2 / seleccion de memoria RL | [Fuente](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/propuesta_doctoral_seleccion_multifidelidad_rl.tex): cuando comprar curvas parciales y abstenerse | Piloto, desarrollo, calibracion y confirmacion por entorno base; POPGym y cambio de contexto CARL; PPO y cinco codificadores; referencias exhaustiva, fija, aleatoria, multifidelidad y decision forzada segun protocolo | H1 costo/regret, H2 riesgo-cobertura, H3 transferencia; pasos de entorno reales, costo inicial/amortizado y abstenciones; no usar una sonda de forecasting como resultado RL |
| P-CAP / capacidad y dimensionamiento | [Fuente](02_memorizacion_generalizacion_dimensionamiento.md): memorizar, generalizar y estimar tamano suficiente | H1 etiquetas aleatorias/mesetas por familia-tamano-precision; H2 reglas conocidas y log-loss; H3 prediccion de N_min frente a curvas, parametros y proxies; MLP principal y GRU temporal secundaria | C_mem operacional, perdidas log2, N_min en rejilla con censura, error de dimensionamiento y costo; M3/M4 anteriores solo si su contrato coincide |
| P-PRE / senales | [Fuente](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/propuesta_doctoral_preprocesamiento_informacional.md): ruido plantado, denoising y preservacion | H1 degradacion por perturbacion, H2 tratamiento causal segun regimen, H3 conservar residuo; raw / D / D+residuo, receptores adecuados y controles de capacidad; sintetico y publico separados | Error de tarea, sesgo/calibracion SNR, extremos, retardo y disponibilidad, costo; nunca equiparar residuo con ruido real |
| P-TRN / senales | [Matriz](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/tesis_transformaciones_temporales/01_MATRIZ_HIPOTESIS_EXPERIMENTOS.md): utilidad/transferencia de un grafo pequeno con abstencion | T0-T5 / E0-E8: contrato, calibracion, preservacion, global vs por variable, utilidad publica, seleccion bajo presupuesto, abstencion, familias reservadas, paridad DOIN; identidad, fijo/aleatorio y busqueda sin transferencia | H1 utilidad condicional, H2 transferencia/costo, H3 abstencion; curvas riesgo-cobertura y preservacion. Mantener signo de efecto propio del protocolo |
| P-INC / incentivos | [Fuente](04_incentivos_red_descentralizada_multidominio.html): asignacion con demanda financiada y nodos adaptativos | H1 regla uniforme/estatica/reactiva/propuesta; H2 nodos fijos/aleatorios/codiciosos/aprendidos; H3 desviaciones rentables o limites; simulador HPO-B y segundo dominio obligatorio de entrenamiento real compacto | Demanda financiada no atendida, desalineacion L1, utilidad de desviaciones, arrepentimiento, HHI y costo; unidad campana, Holm; sin pagos reales ni dependencia de desplegar DOIN |

Fuentes complementarias conservadas: [master v2](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v2.md),
[patch 001](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/tres_temas_entrevista/WORKPLAN_PATCH_001_HISTORICAL_CHAIN_AND_COMPRESSION_EXTENSIONS.md),
[patch 002](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/tres_temas_entrevista/WORKPLAN_PATCH_002_AGENT_AUDIT_CONSTRAINTS.md),
[patch 003](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/tres_temas_entrevista/WORKPLAN_PATCH_003_FINAL_REVIEW_CAUSAL_EVENTS_METAOPT.md),
[integracion jerarquica](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/integracion_workplan_2026_09_09/04_PLAN_INTEGRACION_JERARQUICO.md).
No reabrir la fusion de tesis rechazada ni sustituir P-MOD por P-L2.

## 3. Donde estamos realmente

La siguiente tabla resume evidencia registrada en las ramas, NO una auditoria
de procesos vivos ni una nueva conciliacion del cubo realizada en esta revision.
Antes de reutilizar un resultado, comprobar sus artefactos, poblacion y contrato.

| Trabajo | Evidencia disponible | Lo que NO establece |
|---|---|---|
| D0-D2 | Perfilado reportado de 715 datasets; calibraciones condicionadas a regimen, 47+6 resultados candidatos y 39 calibraciones SNR tras correcciones | No caracterizacion semantica completa de toda variable, ni denoising optimo por entrada, ni informacion sin ruido conocida en mercado |
| D3 mecanica | v1-v3, 511 unidades / 6390 celdas; estados de causalidad, disponibilidad y sensibilidad de sondas | No utilidad de las representaciones ni aceptacion automatica de variantes/configuraciones nuevas |
| Utilidad anterior | utildev-v1: 18 contrastes sin avance, 18 descriptivos, cero propuestas; ridge W4, incremento observado, tres operadores, escenario limitado | No comparacion de redes modulares, no refutacion general del preproceso, no resultado de trading |
| Adecuacion neuronal | Implementaciones y pilotos de costo; 12D/T sin factorial completo; control C1-C6 retirado como campana siguiente | No validacion del extractor modular ni suficiencia del volumen/contexto |
| Gobernanza/OLAP | Rutas data-gov -> lake/proveedores -> consumidores -> warehouse DuckDB, outbox y herramientas de conciliacion | No causalidad por tener hash; no correccion ML por tener terminal COMPLETED |
| E0/E1/E2/E3 modular | Propuesta y requisitos escritos; primera comparacion H2/H3 no demostrada por lo anterior | Ninguna hipotesis doctoral modular queda ya confirmada |

[Mapa de cobertura 12C](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/integracion_workplan_2026_09_10/12C_REPRESENTATION_COVERAGE_MAP_2026_09_18.md)
y [protocolo semanal/RL 12E](https://github.com/harveybc/predictor/blob/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/integracion_workplan_2026_09_10/12E_WEEKLY_BUSINESS_PROTOCOL_AND_RL_COUNTERPART_2026_09_18.md)
delimitan las brechas. No convertir un estado antiguo "no ejecutado" en un estado
actual, ni una mecanica ejecutada en hipotesis probada.

## 4. Primer experimento modular historico: MOD-E0-DEV

Esta seccion conserva el diseno y su razon cientifica. **No es la orden de
ejecucion actual:** la prioridad SOTA-first posterior y la cola de la seccion 5
gobiernan los nuevos ajustes. No repetir estos pilotos para ocupar una GPU.

**Pregunta:** al variar heterogeneidad temporal y relaciones retardadas entre
variables, que aportan agrupar por perfiles y conservar secuencias hasta la fusion,
bajo informacion, entrenamiento y recursos comparables?

Es el piloto de mecanismos de E0, ya previsto en la propuesta, NO su confirmacion
reservada. Su resultado sera una tabla de efectos y limites del procedimiento,
mas parametros de costo/precision para E1 y las reservas. Puede encontrar ausencia
de beneficio: eso sigue siendo un resultado del proyecto si los controles son
informativos y el experimento no confunde falta de aprendizaje con falta de efecto.

| Aspecto | Orden sinusoidal retirada | MOD-E0-DEV |
|---|---|---|
| Datos | Una variable, un periodo, amplitudes/fases diversas | Varias variables y grupos temporales; periodicidad, persistencia, tendencia/eventos; pares con/sin dependencia retardada |
| Intervencion | Volumen de ventanas con predictor fijo | H2: asignacion por perfiles vs aleatoria; H3: fusion temporal vs resumen temprano |
| Modelo | Ridge y Conv1D sobre una ventana | Detector -> integrador -> adaptacion por grupo -> fusion -> nucleo -> cabezal; controles que cambian solo la intervencion |
| Conocimiento adquirido | Adecuacion de esa implementacion en una rejilla de senos | Respuesta inicial sobre los mecanismos que justifican la arquitectura doctoral |
| Infraestructura | Motivo central del ensayo | Controles internos reutilizados; se corrigen solo fallos observados |
| Limite | Ni agrupacion ni RL ni mercado | Desarrollo sintetico multivariado, todavia no H1 publico ni beneficio economico |

### 4.1 Banco y variables del experimento

- H2: niveles predefinidos de diferencias de periodicidad o persistencia; mantener
  muestreo, numero de variables, volumen y distribucion del ruido. Cambiar amplitud
  no sustituye a heterogeneidad temporal. Grupos latentes solo para diagnosticar,
  nunca suministrados al agrupador. Perfiles calculados solo en entrenamiento.
- H3: pares con/sin relaciones entre variables con retraso. Conservar en distribucion
  los procesos marginales, comprobarlo en replicas y no usar un simple shuffle
  temporal como unico control. Demostrar que la dependencia cambia la informacion
  disponible para el objetivo; correlacion cruzada sola no basta.
- Periodos, retardos, horizonte y observacion se derivan de la pregunta del generador
  y de su escala muestreada. No heredar P=8/16/32/41, W=4/17 ni n=2048 porque ya
  existen. La propuesta no prescribe esos numeros: Satoshi debe entregar el calculo
  y las alternativas antes de medir, usando entrenamiento/desarrollo exclusivamente.
- Registrar numero de trayectorias/configuraciones independientes, valores unicos,
  ventanas, ciclos utiles y filas tras calentamiento/purga, sin intercambiarlos.
  Una rejilla determinista no produce replicas independientes por cambiar nombres.
- Separar datos de desarrollo, validacion del ajuste y reserva cientifica. Los
  controles limpios deterministas con MASE indefinido quedan como diagnosticos MAE;
  no agregar ruido ni epsilon despues de ver resultados para arreglar el denominador.

### 4.2 Aprendices, agrupacion y contrastes

**Comparacion aprobada para desarrollo E0:** ARCH-A como referencia inicial
(Conv1D causal de una/dos capas e integrador identidad), frente a ARCH-B
(bloque causal dilatado tipo TCN) y ARCH-C (Conv1D + GRU o LSTM con secuencias).
Anadir ARCH-0 sin extractor aprendido con adaptacion dimensional si hace falta.
Todas usan el modular con fusion y cabezal reales; no elegir ganador por facilidad
de implementacion. Controlar nucleo/cabezal, acceso a informacion y oportunidades
de ajuste; comparar efectos por arquitectura/regimen y costo total. Pocas capas
no garantizan contexto suficiente: calcular alcance rama+nucleo. Ver el documento
H-CORE, seccion 2, para controles, criterios y adopcion despues del retorno de Satoshi.

- Agrupacion: ACF, bandas Welch y fuerza de tendencia/estacionalidad; escalado de
  descriptores y exclusion de constantes en desarrollo; enlace promedio/distancia
  euclidiana segun propuesta. Estabilidad por segmentos. Feature selection y
  agrupacion son decisiones diferentes: no retirar variables entre brazos H2.
- H2: mismos numero/tamano de ramas, arquitecturas, contextos, informacion,
  preproceso comun, inicializaciones emparejadas y oportunidades de ajuste. Varias
  asignaciones aleatorias predefinidas, no permutaciones que solo renombren modulos.
- H3: las dos alternativas reciben **las mismas activaciones de un extractor
  entrenado y congelado**. Solo combinacion y cabezales se ajustan. Conservar el
  eje temporal en el brazo secuencial; definir exactamente el resumen temprano.
  Igualar capacidad de los controles dentro de tolerancia registrada y medir costo.
- R0 es la base de desarrollo desde cero; R1/R2 y preentrenamiento se comparan en
  desarrollo E1, no se presuponen mejores. En H3, congelar el extractor despues de
  su entrenamiento comun no implica haber elegido R1 para H1. Verificar gradientes,
  pesos y estados de normalizacion de los componentes que deben permanecer fijos.
- El grafo real debe corresponder al contrato; un Flatten anterior a la fusion no
  implementa el brazo de secuencias de H3. Existe tal helper en
  `predictor_plugins/common/utils.py:build_branch`; eso obliga a trazar la ruta
  usada, no demuestra que todos los plugins pasen por ella. Reusar plugins cuando
  cumplen, implementar solo el componente faltante cuando no.
- Persistencia, referencia estacional, predictor estadistico pertinente y oraculo
  causal del generador diagnostican dificultad/suelo. El baseline lineal no es el
  unico receptor del tratamiento. Verificar aprendizaje de un control positivo
  pertinente al mecanismo con el modelo neural real y el mismo soporte disponible.

### 4.3 Medidas e interpretacion

H2: `e(h)=MASE(perfiles,h)-MASE(aleatorio,h)`; pendiente de e respecto a h.
H3: `d_r=MASE(secuencias,r)-MASE(resumen,r)`; `gamma=d_1-d_0`.
Negativo favorece el metodo. Guardar errores crudos por objetivo/origen y el
denominador train compartido. MAE/MSE/RMSE complementan, no sustituyen el estimando.

El piloto informa curvas, efectos descriptivos, variabilidad, precision alcanzada
y costo; **no** emite apoyo confirmatorio a H2/H3. Antes de la reserva, E1 fija
margen relevante y plan de precision; el protocolo principal usa F+4 intervalos
Bonferroni y bootstrap jerarquico con unidades emparejadas. Semillas, ventanas y
origenes no inflan el numero de tareas independientes.

Guardar tambien grupos encontrados, estabilidad, activaciones/forma temporal,
retardos de disponibilidad, soporte usado, parametros, updates observados, curvas
train/val, checkpoint seleccionado, predicciones recargadas, pico de memoria y
costo completo. Early stopping restaura el mejor checkpoint de validacion: no
demuestra por si solo ausencia de sobreajuste ni selecciona por el test.

## 5. Cola y dependencias, sin cadena artificial

Cola vigente desde 23-sep; sustituye las filas antiguas "Ahora A/B" de RP57-RP63.
El estado detallado de celdas vive en
[EXPERIMENT_EXECUTION_QUEUE.json](program_v3/EXPERIMENT_EXECUTION_QUEUE.json).

| Carril vigente | Trabajo | Dependencia real / siguiente resultado |
|---|---|---|
| GPU, primero | RP135: receta publicada TimeFilter L512, 12 celdas, solo 5090 externa admitida | Preparacion gobernada, diseno sellado y capacidad; MSE/MAE del autor, naive y receta documentada; no espera borrados ni replay historico |
| CPU, independiente | RP132-RP134: dos consumidores de evidencia | Arbol separado, tope 1 800 s CPU; conservar arrays hasta reparar, no detener entrenamiento independiente |
| Analisis/diseno concurrente | Contraste modular frente a referencia aceptada, luego R0/R1/R2 y H-CORE segun dependencias | Preparar intervencion/controles sin tocar el modelo que entrena; no afirmar que la arquitectura de electricidad gana en finanzas |
| Preparacion financiera concurrente | FIN-LOSS-OPT y E3 semanal forecasting/RL | Resolver contratos y referencia financiera antes de ajustar; no abrir reserva ni convertir un fallo documental electrico en bloqueo de diseno |
| Espera fisica localizada | Replay A T96 y catalogo T192 s2021 en dispositivos originales | Confirmacion de refrigeracion + admision; no retiene la cola B |

Las filas siguientes son mapa historico y dependencias cientificas conservadas,
no una segunda cola de despacho ni permiso para revivir pilotos suspendidos.

| Orden | Trabajo / responsable ejecutor Satoshi | Dependencia y salida |
|---|---|---|
| Ejecutado, revision parcial | Piloto multivariado RP1-RP8 | 66 celdas y 3 pilotos; conservar efectos descriptivos, no aceptacion confirmatoria |
| Historico A | RP57-RP59: comparadores, antecedentes antiguos y auditoria de datos | Naive/lineal/modelo por filas y escalas identicas; no inferir log1p ni unidades por magnitud |
| Historico B | RP60-RP63: diagnostico ML y referencia de literatura | Loss/monitor, curvas, capacidad, contexto y volumen por separado; conservar el alcance de esos diagnosticos |
| E0, antes de elegir | MOD-ARCH-COMPARE: ARCH-A/B/C y ARCH-0 en el modular propio | Efectos descriptivos recalculados; aceptacion compuesta pendiente de RP26, sin ganador universal |
| Paralelo C | Aceptacion RL offline de RP53 conservada para revision | RP56 reporta reparacion de terminales sin fill; no aceptacion cientifica ni entrenamiento RL por ese hecho |
| Despues D | E1 publico: desarrollar perfiles, seleccion, contextos, R0/R1/R2 y comparadores | Piloto de mecanismos y datos admisibles; fijar procedimiento, margenes, precision y presupuestos antes de reservas |
| Despues D, extension | MOD-FROZEN-PREFIX y MOD-CORE-PRETRAIN: hipotesis H-CORE del owner | Comparacion R0/R1/R2 y receptor adecuado completados; fijar/materializar prefijo; nucleo desde cero vs preentrenado ajustable vs congelado, con cabezales y costo total |
| Despues E | E0-CONF H2/H3 y E2 H1 en reservas independientes | Reglas E1 fijadas y revision ML; no reutilizar test de diagnosticos anteriores |
| Obligatorio F | E3 forecasting aplicado y RL con reentrenamiento semanal | Datos temporalmente aptos, entorno/modelo y contrato de ejecucion validados; evaluacion propia aunque H1-H3 no sean positivos |
| Carriles propios | P-PRE/P-TRN, P-CAP, P-L2, P-INC | Sus protocolos, modelos y unidades; compartir datos/metricas/codigo solo si equivalencia demostrada, no multiplicar claims sobre el mismo resultado |

H-CORE es un experimento de desarrollo propio, no requisito para E0 ni aprobacion
presupuesta del preentrenamiento. No modifica H1-H3 o sus pruebas reservadas sin
una revision previa a la confirmacion. Su comparacion RL pertenece a E3; un resultado
de reconstruccion o forecasting no la reemplaza. No se pierde si R0 supera R1/R2.

No hacer depender E0-DEV de que todas las transformaciones ganen previamente,
de terminar DOIN, de Metabase o de resolver datos financieros que no consume.
Un operador experimental causal puede compararse en DEVELOPMENT con esa etiqueta;
no obtiene licencia publica/financiera por participar. Mantener controles de datos
y clasificacion reales: este plan no convierte un rechazo de runtime en permiso.

La aplicacion semanal debe fijar demanda desde las configuraciones ejecutables,
horas de corte/publicacion/decision/ejecucion, horizonte, historiales, actualizacion
semanal, objetivos y disponibilidad por variable. No elegir ETH porque haya un CSV.
Desconocidos de adquisicion no se sustituyen por hora de cierre de barra.

## 6. Cobertura completa de senales y representaciones

Cada fila es una pregunta experimental, NO una transformacion obligatoria en
cascada. El [master v2 y sus STEPs](https://github.com/harveybc/predictor/tree/a5630c8a2ef0eb1dbb427873ddf21af735cd4b21/docs/tres_temas_entrevista)
conservan el detalle. Las celdas sin utilidad medida siguen pendientes.

| ID | Experimento/control requerido | Registro/encaje |
|---|---|---|
| STEP-01 | Resolucion, operador de observacion y aliasing; comparar igual intervalo fisico antes de ampliar historia | Eventos vs barras agregadas, disponibilidad; E0/E1/E3 |
| STEP-02 | Identificabilidad/calibracion SNR bajo perturbacion conocida vs estimadores e identidad | Error/sesgo por regimen y abstencion; P-PRE/T0-T1 |
| STEP-03 | Denoising global vs por variable, raw/D/raw+D+residuo y capacidad igualada | Utilidad, extremos, retardo, missingness; P-PRE/P-TRN |
| STEP-04 | Resolucion uniforme vs por distribucion/variable, companding; raw vs cuantizado vs aumentado | Distorsion, ocupacion, bits nominales/empiricos y error; no bits = capacidad neuronal |
| STEP-05 | Fuente/contextos/diccionarios, MDL y longitudes con coder fijado | Costo de descriptor y codificacion, no K ni fuente libre de ruido |
| STEP-06 | Amplitud/fase/frecuencia/tiempo-frecuencia vs raw bajo informacion igual | Disponibilidad, fase, bordes, reconstruccion/prefijos; wavelet causal real |
| STEP-07 | Detectores Conv1D/TCN/recurrentes/atencion y patrones locales/largos frente a referencias | Grafo, campo receptivo efectivo, curvas y adecuacion; P-MOD |
| STEP-08 | Canal o dominio definido, correccion/equalizacion vs identidad | Distorsion/transferencia y utilidad; normalizar no demuestra equalizacion |
| STEP-09 | Factor comun/unico y relaciones retardadas vs raw, conservando X/C/U | No suponer compartido = ruido; utilidad condicional; grupos/fusion |
| STEP-10 | Alineacion fija vs recuperacion temporal dinamica con tiempos de emision | Error de retardo, costo y ausencia de futuro; eventos asincronos |
| STEP-11 | Robustez con mascaras/corrupcion y codificador reutilizado vs sin preentrenar | R0/R1/R2, reconstruccion auxiliar y tarea; no nuevo AE obligatorio |
| STEP-12 | Enrutamiento adaptativo por calidad vs fijo/aleatorio, missingness y abstencion | Utilidad neta, estabilidad, cambios y costo; no oracle gating |
| STEP-13 | Ancho/compute por rama igual/fijo/marginal, interacciones y ablacion de ramas | Frontera error/costo, informacion igualada; no independencia MIMO supuesta |

Compresion permanece transversal: **C1** sparse (06/07), **C2** sparse convolucional
(07), **C3** refinamiento progresivo (06/12), **C4** informacion lateral (05/09/13),
**C5** residuo jerarquico (03/05/06), **C6** tasa-distorsion latente (extractor/nucleo,
despues de controlar la representacion), **C7** duracion/eventos (05/07).
Medir tarea, preservacion, tasa/descriptor y costo; ninguna es un compresor de pesos
que por si solo estime memorizacion. No se ejecutan las siete de golpe.

**FE:** feature-eng tiene un carril propio dentro de 06/07: calendario cuando el
dominio lo justifica, rezagos, estadisticas trailing, indicadores y features de
eventos, comparados con raw y controles de informacion/capacidad. Cada plugin
versiona config, dependencias, fit, disponibilidad y salidas; seleccion anidada
train/val con estabilidad, nunca filtros ajustados al test. El inventario de
plugins no demuestra que se hayan evaluado exhaustivamente.

**EVENT:** procesos irregulares, observacion agregada y tiempo de disponibilidad
se tratan como contratos propios, no interpolacion que inventa muestras.
**META:** selector de pipeline/metaoptimizacion aprende de filas de experimentos
con campanas retenidas, costo incluido; se difiere hasta existir resultados
admisibles suficientes. No usar los mismos runs para ajustar y afirmar transferencia.

## 7. Negocio, RL y modelos de referencia

**FIN-LOSS-OPT es obligatorio antes de fijar la receta de forecasting financiero.**
Su [contrato](program_v3/FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md) exige el
factorial MAE/Huber x Adam/AdamW, defaults identificados y tuning acotado,
receptores/contextos adecuados, validacion semanal y sensibilidad numerica a
mejoras marginales. Diseno en paralelo a E1; ejecucion al cumplir los contratos
financieros, no condicionada a un ganador electrico. Estado: NO EJECUTADO.
MAE_z/RMSE_z y skill frente al ingenuo se reportan por horizonte y fold, con
escala train compartida, valores completos e incertidumbre pareada. La mejora
economica reportada por el owner motiva esta precision, no garantiza beneficios.
No requiere repetir la estrategia heuristica; tampoco reemplaza la evaluacion
RL E3 ni cambia el estimando confirmatorio de las propuestas.

Forecasting publico E1/E2 conserva las referencias de la propuesta: ingenuos,
estadistico multivariado pertinente, DLinear, PatchTST, iTransformer, DUET y uno
entre MTST/Pathformer/TimeMixer elegido antes de resultados de desarrollo. No
reducir sus arquitecturas para igualar parametros; esa restriccion aplica a los
controles internos con tolerancia declarada. Presupuesto/oportunidad de tuning
comparable y coste completo para todos. Agrupacion adicional segun protocolo.

RL E3 es obligatorio, no un apendice opcional ni una microcorrida de transporte.
Usar el agente/entorno de las configs ejecutables, historial causal, reset/estado,
reward, acciones, restricciones, capital, fees, slippage, funding si aplica,
tiempo de ejecucion y politica de ordenes. Comparar raw vs modular con la misma
informacion y entrenamiento semanal: train -> validacion interna -> semana
futura intacta. Separar mejora por mas datos/resoluciones de mejora arquitectonica.

Entregables E3: retorno neto, drawdown, Sharpe con supuestos, turnover, exposicion,
incertidumbre por semanas/regimenes, pasos observados y tiempo total de reentrenar.
Comparar politicas triviales y un control no modular competente. Cabezas de
volatilidad/pronostico/intervalos requieren calibracion y predicciones fuera de
muestra antes de alimentar al agente; no labels verdaderos ni futuro como input.
Paper/live requiere validacion operativa separada; esta ronda no envia ordenes reales.

P-L2 tiene ademas su propio banco publico RL y contraste de seleccion; no esperar
a que E3 financiero sea viable para desarrollarlo. P-INC puede simularse sin DOIN
operativo. P-CAP no se satisface contando parametros del extractor de P-MOD.

## 8. Disciplina ML y trazabilidad utilizadas, no decorativas

BENCHMARK-CONTRACTS es un requisito transversal de cada nueva campana, no una
cola que espere a terminar electricidad. El contrato de metricas define la
matriz de equivalencia bibliografica y los dos carriles. Historicos se clasifican
por alcance sin borrarlos ni reentrenarlos automaticamente. El trabajo actual
de Satoshi debe preparar e implementar los controles faltantes por entry point
antes de proponer su siguiente entrenamiento cientifico comparable.

Antes de cada campana, requisito -> test de aceptacion -> diseno de sistema ->
componentes/integracion -> tests unitarios; despues implementacion y verificacion
de abajo arriba. Reutilizar tests existentes por ruta productiva, no reescribir
una simulacion del callback para probar el propio test.

La ficha numerica obligatoria contiene: pregunta/estimando; observacion/target;
generador o fuentes; muestreo/periodos/retardos; soporte disponible a cada decision;
N bruto/util e independiente; contexto/campo receptivo/horizonte; capas, unidades,
activaciones/parametros; escalado/preproceso; optimizer/lr/batch/presupuesto;
early stopping y recarga; curva de volumen/convergencia; controles positivos,
nulos y adversos; incertidumbre/multiplicidad; costo. Cada numero lleva origen,
supuestos y sensibilidad. No se heredan defaults sin esa ficha.

Wavelet/STL/rolling se prueban con prefijos, futuros alterados, padding/bordes,
missingness, tiempo de emision y restart, en el camino realmente consumido. Un
gemelo no causal debe ser detectado en comparaciones sensibles; cero emisiones
no es pase. Una ventana causal no vuelve causal un fit o reconstruccion global.

Datos, roles, tiempos, labels y grafo ejecutado deben llegar a un lector verificable.
Todos los intentos (incluidos fallos/inconclusos) pasan por data-gov y terminan en
DuckDB mediante el warehouse. Comparar poblacion y contenido contra contabilidad
independiente; conservar predicciones/labels para rederivar metricas. Un hash no
prueba semantica ni causalidad. El test automatico de este plan solo prueba su
estructura documental, **no** aprueba experimentos.

## 9. Estado durable y orden vigente

[PROJECT_METHOD_STATE.json](program_v3/PROJECT_METHOD_STATE.json) conserva la etapa,
cola, responsables y alcance autorizado. [Chequeo documental](program_v3/check_plan.py)
detecta omisiones de frentes, pasos, carriles y dependencias. No llena resultados.

[Orden vigente RP74-RP81: auditoria ML y continuacion acotada](../handoffs/MUSASHI_RP74_RP81_2026_09_21.md).
Satoshi debe incorporar esta revision a su work plan y publicar retorno completo,
sin pedir "continua" por cada paso. Musashi revisara tanto la adecuacion ML como
la implementacion y los resultados; no se atribuye revision a Satoshi en nombre
de Musashi. La orden publicada no prueba que Satoshi ya la este ejecutando.

### Historico: paquete post-Huber (21-sep): preparacion, no ejecucion

- **Fase 2 sellada** (`6b475094…`, 27 celdas, `SEALED_NOT_EXECUTED`): calendario, contexto diario y
  volumen como preguntas separadas; el confundido ventana/profundidad **medido** (W=60: 5 bloques,
  8 127 parametros; W=1440: 10 bloques, 12 047) y separado factorialmente (canal de rezago diario,
  ventana larga con profundidad propia, ventana larga con profundidad fija, ventana corta con nucleo
  profundo); control de calendario permutado con la misma capacidad; conteo de filas/ventanas/
  etiquetas/exposicion por separado; adecuacion sin afirmar convergencia. Diez reglas de aceptacion.
- **FIN-LOSS-OPT disenado** (`0df0ca76…`, `DESIGNED_NOT_STARTED`): tarea congelada desde el recurso
  gobernado EURUSD 1h, horizontes 6 h / 72 h declarados, pliegues semanales, MAE_z, delta de Huber
  desde una escala residual causal de train, busqueda acotada e igual por familia; FL01–FL08 con 11
  reglas verdes y 5 `xfail` estrictos que nombran el runner ausente. No hay ganador financiero.
- **BENCHMARK-CONTRACTS parcial**: contrato versionado, comparabilidad decidida por campos, rechazo
  sin contrato en los dos runners de las proximas campanas (2 de 37 puntos de entrada reales);
  inventario de cobertura publicado; NO se declara cumplimiento en todos los dominios.
- **Benchmark bibliografico UCI 235**: cuatro fuentes leidas en su origen; ninguna es comparable con
  la tarea de 60 minutos (Gasparin 2019: 15 min, 96 pasos, sin naive; Saad Saoud 2022: SWT sobre la
  serie completa, CAUSALITY_UNVERIFIED; Vaygan 2021: horizonte no declarado; Kim & Cho 2019: de pago).
  Configuracion de reproduccion de Gasparin declarada y costeada; no ejecutada.
- **Tabla de cierre** generada desde artefactos verificados y el warehouse (31 filas, 31/31 digests),
  con NO_NEW_MEASUREMENT y NOT_COMPARABLE con razon y comparacion pendiente en cada fila.
- Correcciones publicadas junto al retorno RP57–RP64 (RP63 cambio perdida y monitor; etiquetas
  barajadas no son cota; el salto de persistencia ya existe en `tcn_w`).

### Revision de RP66-RP73 y continuacion RP74-RP81 (21-sep)

[Dictamen de Musashi](../audits/work_plan/MUSASHI_RP66_RP73_REVIEW_2026_09_21.md):
CHANGES_REQUIRED, mediciones conservadas. Revision `1880caa` publicada y disponible en
checkouts aislados de ambos workers, sin cambiar sus arboles existentes ni reiniciar servicios.
Musashi recalculo las 18 predicciones nuevas y comparo arrays, pesos y registros contra los
artefactos de terminal publicados: 18/18, sin discrepancia. Esto no equivale a una nueva
consulta al warehouse vivo ni a un replay independiente de pesos.

- **Resultado util**: calendario MAE_z 0.486819 frente a control aleatorizado 0.552871;
  referencia GRU adaptada 0.531273 frente a modular original 0.568045. Naive comun 0.676560.
  Una semana DEV, tres semillas; no ranking universal ni evidencia de ganancia financiera.
- **Evidencia**: corregir custodia de historia sin ancla independiente y comparador que acepta
  una unidad prepare sin predicciones. No borrar ni reentrenar historia automaticamente.
- **ML financiero**: corregir ventanas con features no finitas, seleccion que elige semillas,
  bootstrap degenerado y contrato temporal antes de una campana cientifica financiera.
- **Parada/costo**: presupuesto alcanzado sigue censurado aunque coincida con paciencia;
  validacion se cuenta una vez. Q2 corregido ~111000 s sigue fuera del techo. No se ha demostrado
  que el cuello de botella sea construir ventanas; perfilar antes de optimizar.
- **Siguiente medicion condicional**: arquitectura {modular, GRU adaptada} por entradas
  {originales, calendario}, 12 celdas, mas 3 controles de calendario aleatorizado; paciencia
  10 eventos a cadencia fija de 200 updates, techo 4000, mismo train/DEV/escalador. Diseno
  informado por DEV previo, no confirmatorio. Ejecutar completo solo si pasa aceptacion y costo.
- FIN-LOSS-OPT sigue obligatorio, no selecciona receta desde electricidad; implementacion y
  aceptacion sintetica se corrigen ahora y la ficha cientifica concreta vuelve a revision.
  RL semanal, extraccion por grupos y preentrenamiento del nucleo mantienen sus dependencias.

### Revision RP74-RP81 y continuacion RP82-RP89 (21-sep)

[Dictamen](../audits/work_plan/MUSASHI_RP74_RP81_REVIEW_2026_09_21.md) y
[ordenes activas](../handoffs/MUSASHI_RP82_RP89_2026_09_21.md):
CHANGES_REQUIRED, sin reinicio general de experimentos. Musashi reprodujo las
15 celdas nuevas desde pesos en procesos separados y recalculo errores: todas
pasan la tolerancia existente. No fue una consulta nueva al warehouse vivo.

- **Informacion util**: GRU+calendario MAE_z 0.479838 vs GRU 0.528993;
  modular+calendario 0.492654 vs modular 0.539087; naive comun 0.676560.
  Calendario mejora en ambos receptores y los tres bloques semilla/host.
  Comparador = nuestra GRU adaptada, no el valor publicado de otra tarea.
- **Alcance ML**: 7/15 censuradas, una semana DEV informada por resultados previos.
  No ganador universal ni beneficio financiero. Comparar tiers no aisla paciencia:
  las trayectorias de semillas 2/3 cambiaron antes de la parada al cambiar de host.
- **Prioridad RP82-RP85**: cierre real, identidad del escalador y comparador;
  selector financiero que ejecute A/B/C/D y no mezcle 10 vs 16 candidatos;
  incertidumbre sobre la misma poblacion que el estimando. Contraejemplos
  ejecutables conservados, no afirmaciones inferidas solo de inspeccion.
- **Siguiente pregunta experimental**: rezago diario causal en el receptor W60
  con receta tier 2 y poblacion comun, no ventana W1440 de costo no autorizado.
  Medicion condicional a aceptacion y presupuesto completo; reutilizar evidencia
  solo por equivalencia comprobada. No repetir el bloque de 15 por rutina.
- **Finanzas**: v4 con factorial, temporalidad y presupuesto adecuados; piloto
  gobernado de costo solo en train bajo RP88. Huber/MAE y Adam/AdamW obligatorios,
  sin eleccion desde electricidad; sin busqueda financiera completa ni reserva.
  RL semanal y preentrenamiento de extractor/nucleo conservan sus dependencias.

### Revision RP82-RP89 y continuacion RP90-RP97 SOTA-first (21/22-sep)

**Ejecutado por Satoshi (22-sep, RP90-RP97, pendiente de revision):** [retorno](../audits/work_plan/SATOSHI_RP90_RP97_RETURN_2026_09_22.md).
Disposicion HISTORICAL_DEV_ONLY implementada en los caminos reales de seleccion (resolver, ranking activo de la tabla,
cierres); hallazgos RP89 de Musashi reparados y re-probados (POST en `docs/audits/evidence/d3_k5_20260917/RP90/`).
Referencia seleccionada: **TimeFilter (ICML 2025) L=96, Tabla 8** ([dossier](program_v3/SOTA_REFERENCE_DOSSIER_2026_09_21.md),
[lock](program_v3/SOTA_PROTOCOL_LOCK_2026_09_21.md)); ECL oficial adoptado como recurso gobernado (lago `sota_benchmarks`).
Reproduccion fiel PARCIAL por poblacion: T=96 tres semillas verificadas (media MSE 0.1355 / MAE 0.2329 frente a 0.133 / 0.230
publicados: acuerdo numerico bajo la regla congelada), T=192 una semilla (0.1609 / 0.2560), T=336/720 sin host
([asignacion RP94](program_v3/SOTA_ALLOCATION_RP94_2026_09_22.md): deficit de memoria nombrado, receta no encogida).
Ordenes del owner del 22-sep: ningun trabajo en omega hasta el 24-sep (retencion termica de viaje); sin compresion de
predicciones y boveda exhaustiva de metricas por celda para que Musashi borre los arrays tras su analisis independiente.
Siguiente: T=192 semillas 2022/2023 y replay en omega desde el 24-sep; decision del owner sobre el techo de memoria para
T=336/720; solo tras aceptacion de la reproduccion se planifican R0/R1/R2 modulares y H-CORE contra esta referencia.

**Ejecutado por Satoshi (22-sep, RP98-RP105, pendiente de revision):** [retorno](../audits/work_plan/SATOSHI_RP98_RP105_RETURN_2026_09_22.md),
evidencia en `docs/audits/evidence/d3_k5_20260917/RP103/`. Los cinco hallazgos del dictamen RP97 corregidos con contraejemplos
PRE/POST (boveda v3 con estados y poblacion exacta, replays re-ejecutados en cada cierre sin cache, promedio dentro de cada
semilla); traza de rutas: la discrepancia CPU/GPU es NUMERICAL_ONLY; evaluacion y cierre acotados en memoria sin encoger el
modelo (adaptador en disco, lector por trozos, objetivos en archivo). Las 12 celdas ejecutadas (T=192 s2022/s2023, T=336 x3 y
T=720 x3 en la RTX 5090 de WORKER_B con dos parches operativos declarados: DataLoader sin procesos hijos y umbrales de malloc
de glibc). Cierre consolidado en WORKER_B: **T=192 media 0.1576 / 0.2522 y T=336 media 0.1645 / 0.2630 en ACUERDO OPERACIONAL**
con la regla congelada (replays bit-identicos en el mismo dispositivo); sus 6 arrays borrados con recibos en WORKER_B y WORKER_A
(11.16 GB + 2.29 GB), catalogos y checkpoints conservados. T=96 sin verificar en este cierre (replay entre GPUs 4090->5090 falla
la regla puntual 1e-4 con metricas bit-identicas; faltan 3 x 22 s de replay en la GPU de WORKER_A: decision del owner por el
disipador). T=720 ejecutado pero sus terminales rechazados por el almacen OLAP (columna INT32 de tamano de artefacto): paquetes
corregidos e instalados, **falta que el owner reinicie `crispdm-data-warehouse-olap.service` en omega** (reinicio negado a
Satoshi por el entorno); copias de omega intactas durante la retencion. Siguiente: reinicio del almacen -> `report` x3 -> cierre
con T=720; replays en WORKER_A para T=96; borrados en omega tras la retencion; solo con las cuatro horizontes verificadas se
forma el promedio dentro de cada semilla.

**Ejecutado por Satoshi (23-sep, RP106-RP113, pendiente de revision):** [retorno](../audits/work_plan/SATOSHI_RP106_RP113_RETURN_2026_09_23.md),
evidencia en `docs/audits/evidence/d3_k5_20260917/RP106/`, `.../RP111/` y `.../RP112/`. Los tres hallazgos del dictamen RP105
corregidos con su PRE congelado y su POST reproducido: la verificacion historica de una celda borrada ahora se RESUELVE POR
DIGESTO contra el informe de cierre original y se ATA al registro, al checkpoint, al catalogo y a la cadena de terminales
aceptada (rechazos tipificados; el marcador de borrado ya no es autoridad de nada); el borrado es por CONTENIDO bajo frontera
exclusiva, con aprobacion atada al informe aceptado y a un respaldo de metadatos, estado por ruta y reanudacion. La metrica
oficial del autor (float32) se completa dentro de la memoria real: `author_metric_exact` reproduce BIT A BIT
`utils.metrics.metric` replicando el arbol de suma por pares de numpy (probado contra la funcion del autor y contra
`np.add.reduce`). Las tres celdas T=336 borradas se REGENERARON por INFERENCIA desde sus checkpoints en la RTX 5090 externa:
bit a bit identicas a las originales (digestos de predicciones y objetivos iguales a los del registro), lo que recupero su
metrica oficial; su catalogo se rederivo de esos bytes y coincide con el retenido. Atribucion de dispositivo con clase de
evidencia (MEDIDA / MASCARA / INFERIDA / DESCONOCIDA): solo la medida certifica repetibilidad en el mismo dispositivo. Cierre en
WORKER_B (informe bc9d3388…): **T=192 0.1576/0.2522, T=336 0.1645/0.2630 y T=720 0.1902/0.2906 en ACUERDO OPERACIONAL** con la
regla congelada; T=96 sigue SIN MEDICION porque su replay entre GPUs falla la regla puntual (la regla no se ensancha) y su
replay en el dispositivo original espera la confirmacion de refrigeracion del owner; el promedio de cuatro horizontes sigue SIN
CALCULAR. Aceptacion independiente de los doce catalogos (nueve pasan con todos los oraculos exactamente 0.0) y borrado
autorizado de los tres arreglos T=720 (12.59 GB) con recibos por ruta. Protocolo B (Tabla 9, L=512) queda especificado y
costeado como entrega separada, sin ejecutar. Colocacion: unica GPU admitida la RTX 5090 externa de WORKER_B, verificada por
UUID fisico antes de cada despacho y afirmada dentro del hijo; sin respaldo a GPU interna ni al coordinador.

**Ejecutado por Satoshi (23-sep, RP114-RP121, pendiente de revision):** [retorno](../audits/work_plan/SATOSHI_RP114_RP121_RETURN_2026_09_23.md),
evidencia en `docs/audits/evidence/d3_k5_20260917/{RP114,RP119}/`. Los cinco contraejemplos del dictamen RP113 quedaron
congelados como PRE y reparados: (1) resolver un digesto no es aceptarlo, asi que informes de cierre, aceptaciones de catalogo y
regeneraciones se publican ahora como terminales gobernados y la verificacion historica exige que el informe sea evidencia
ACEPTADA; (2) una regeneracion fabricada ya no reemplaza nada: debe estar aceptada y ser coherente con el catalogo retenido, y
una rechazada se reporta e ignora sin destruir el vinculo original; (3) el borrado exige informe aceptado, aceptacion de catalogo
y respaldo verificado en bytes, y su frontera de desvinculacion prueba por descriptor e inodo que lo borrado eran los bytes
aceptados; (4) la media float32 del autor divide por el conteo como numpy (entero de 64 bits promovido a float64), no por
float32(N): se corrigio, se verso la ruta y se midio el impacto real, **un solo numero cambio en toda la poblacion, el MAE de
T=336 semilla 2021, en 3·10⁻⁸**; (5) la aceptacion de catalogos valida ahora el dominio de cada familia y, donde hay arreglos,
los recomputa con una implementacion independiente. Las seis celdas afectadas por (4) se regeneraron por inferencia desde sus
checkpoints en la 5090 externa admitida: **bit a bit identicas a las originales**. Cierre sucesor (informe 9c491f38…):
**T=192 0.1576/0.2522, T=336 0.1645/0.2630 y T=720 0.1902/0.2906 en ACUERDO OPERACIONAL** con la metrica propia del autor;
**T=96 pasa a MEDIDA_SIN_REPLAY_ACEPTADO** (no "sin medicion"), con sus valores fuera de toda media y su fallo entre GPUs
preservado. Los doce catalogos aceptados y publicados (peor oraculo interno 1.1·10⁻¹⁶). Protocolo B: corregido (3 y 4 parches por
canal, poblacion de prueba identica) y **medido** con un piloto acotado en la 5090 (0.055-0.060 s por paso, 6.6 GiB de VRAM),
proyectado en ~4 h para doce celdas; sin campana lanzada. Sin borrados nuevos en esta ronda.

**Ejecutado por Satoshi (23-sep, RP122-RP127, pendiente de revision):** [retorno](../audits/work_plan/SATOSHI_RP122_RP127_RETURN_2026_09_23.md),
evidencia en `docs/audits/evidence/d3_k5_20260917/{RP122,RP127}/`. Los tres hallazgos del dictamen RP121 reproducidos y
reparados: (1) el catalogo aceptaba autocorrelaciones y cuantiles plausibles pero falsos y trataba un valor ausente como
diferencia cero -> se declara el inventario finito de estimadores (14 familias con poblacion, eje, orientacion, reduccion,
parametros, condicion de indefinicion y tolerancia) y una implementacion independiente los recomputa sobre LA MISMA poblacion
(autocorrelacion sobre origenes de ventana por paso fijo, cuantiles por la CDF del histograma declarado, linea base desde el
cargador del autor), comparando campo por campo con cobertura explicita; (2) la aceptacion leia el TIPO de evidencia del
registro local -> ahora tipo, sujeto, rol y diseno se leen del terminal aceptado y el registro es solo una pista; (3) la API
publica permitia `require_acceptance=False` -> el parametro se elimino y las pruebas inyectan una cadena aceptada fiel.
Aplicado a las doce celdas: los tres catalogos de T=96 y ocho de las nueve celdas borradas (regeneradas bit a bit identicas en
la 5090 admitida) pasan **las catorce familias sin discrepancias**; la novena, T=192 semilla 2021, fue entrenada en el
coordinador y NO se regenera identica en la 5090, de modo que su catalogo no puede certificarse sin su dispositivo original y
la aceptacion lo rechaza en vez de aparentarlo. Las puntuaciones no cambian: T=192 0.1576/0.2522, T=336 0.1645/0.2630 y
T=720 0.1902/0.2906 en acuerdo operacional; T=96 medido sin replay aceptado; promedio de cuatro horizontes SIN CALCULAR sobre
denominador cuatro. El retorno anterior conserva su texto con una **fe de erratas fechada** de sus tres afirmaciones
demasiado amplias. Suites en WORKER_A: 98 + 117 + 106.

**Ejecutado por Satoshi (23-sep, RP128-RP131, pendiente de revision):** [retorno](../audits/work_plan/SATOSHI_RP128_RP131_RETURN_2026_09_23.md),
evidencia en `docs/audits/evidence/d3_k5_20260917/{RP128,RP131}/`. Los tres hallazgos del dictamen RP127 reproducidos (PRE en
WORKER_B) y reparados (POST): (1) una aceptacion que nunca ejecuto la referencia numerica podia autorizar un borrado -> cada
aceptacion lleva ahora una CLASE explicita y un certificado derivado de su propio contenido (FULL_INDEPENDENT_NUMERIC solo si se
comparo cada familia declarada, sin campos sin revisar ni discrepancias; si no, DOMAIN_AND_INTERNAL_ONLY con alcance restringido
y conservado como diagnostico), y el consumidor destructivo valida ese certificado -clase, catalogo cubierto, poblacion contra la
forma del registro, version y digesto del inventario- en vez de un booleano; ademas se separo lo que certifica el catalogo de lo
que verifica la celda; (2) los consumidores destructivos no exigian el diseno cientifico -> ahora es obligatorio y se lee del
campo propio del terminal, con la ausencia y la contradiccion rechazadas; (3) un booleano se comparaba igual a un numero -> los
booleanos solo se comparan con booleanos, a cualquier profundidad. Revalidacion de solo lectura de las doce celdas bajo las
compuertas reparadas: **once con aceptacion numerica completa** (T=96 por sus propios certificados, ocho borradas por sus
certificados de regeneracion) y T=192 semilla 2021 solo con dominio e internos; los tres arreglos de T=96 siguen retenidos y
**ninguna celda es elegible para borrado hoy**. Puntuaciones sin cambios. Fe de erratas fechada en el retorno anterior. Sin GPU
en esta ronda; suites en WORKER_A: 104 + 117 + 106.

**Ejecutado por Satoshi (23-sep, RP132-RP134, entregado bajo RP136, pendiente de revision):**
[retorno](../audits/work_plan/SATOSHI_RP132_RP134_RETURN_2026_09_23.md), evidencia en
`docs/audits/evidence/d3_k5_20260917/{RP132,RP134}/`, rama separada `satoshi/rp132-rp134-20260923`. Los dos contraejemplos del
dictamen RP131 reproducidos sin reparar (una aceptacion aceptada con CERO campos y otra con DISCREPANCIA en `global.mae`, ambas
con resumenes verdes, ambas certificaban completo y **borraban el arreglo**) y luego reparados: el certificado ya no lee
`families_complete`, `unchecked`, `disagreements`, `fully_independent` ni `pass`, sino los 43 campos declarados del inventario
ligado -presencia, estado aplicable, diferencia finita y tipada, tolerancia propia de la familia, caso indefinido declarado- y
deriva cobertura y resumenes, de modo que un registro contradictorio se **rechaza** en vez de repararse. El inventario bajo el
que se lee un registro es explicito; los nombres de familia no establecen version. La revalidacion (esquema v2) separa la
evidencia numerica registrada localmente de la **aceptada hoy** bajo diseno, tipo, sujeto y rol, exige a los resumenes la misma
relacion validada que a sus filas, etiqueta como INSPECCION HISTORICA la corrida sin almacen, lee la verificacion de la celda del
cierre **actual** y sustituye `deletion_eligible_today` por una VISTA PREVIA CONDICIONAL que enumera lo que no comprobo; la
compuerta completa vive solo en el borrado. Reverificacion de los doce registros retenidos en WORKER_B (solo lectura, 0.24 s de
CPU): **once fuentes en pie** -tres catalogos de T=96 bajo el inventario ligado y ocho regeneraciones bajo una reutilizacion
fechada a nivel de campo cuyos limites quedan escritos: los parametros de familia (bins e intervalo del histograma, lista de
cuantiles, binning de la informacion mutua, rezagos de autocorrelacion, tamano de bloque temporal) **no** quedan establecidos-
y T=192 semilla 2021 sin evidencia numerica, con sus cuatro razones. Ninguna celda con vista previa favorable, **ningun borrado
de produccion**, puntuaciones intactas. Costo medido: 393 s de CPU de los 1.800 reservados; sin GPU. Suite 124 pasan, 1 omitida.

**Ejecutado por Satoshi (23-sep noche, RP136-RP139, pendiente de revision):**
[retorno](../audits/work_plan/SATOSHI_RP136_RP139_RETURN_2026_09_23.md), evidencia en
`docs/audits/evidence/d3_k5_20260917/{RP137,RP138}/`. **Resultado experimental primero:** las tres celdas retenidas de T=96
se reproducen **bit a bit en su dispositivo original** (RTX 4090 de WORKER_A): diferencia maxima absoluta 0.0 y 159.164.640
de 159.164.640 elementos iguales por celda, metricas replicadas identicas al registro, custodia de cadena aceptada y filas
verificadas sin problemas; la discrepancia previa entre dispositivos (1,9e-4 a 2,7e-4 en la 5090) queda como hecho de
portabilidad, no de la medicion. Ademas **L96_h192_s2021 -la celda que nunca se regeneraba identica- tambien es exacta en su
propio dispositivo** (RTX 4070 del coordinador): 312.412.608 de 312.412.608 elementos iguales; esa corrida se hizo sin
credenciales de gobernanza, asi que la custodia no se comprobo y la fila NO queda verificada: es una comprobacion ausente, no
fallida. El bloqueo que lo impedia era `VAULT_CHANGED` por un solo campo: el digesto de la implementacion del catalogo. Se
localizaron ambos digestos recorriendo las 64 revisiones del archivo y hasheando el fuente de `metrics_vault`; **toda la
diferencia es una linea**, el guardia de finitud que paso a su version acotada en memoria (RP101), que no alimenta ningun
acumulador ni reduccion. `METRIC_IMPLEMENTATION_LINEAGE` declara ese predecesor con sus revisiones, el diff exacto y por que
no puede mover un numero; el catalogo retenido conserva sus bytes y su identidad aceptada, un digesto no declarado sigue
rechazando y un predecesor declarado nunca explica una diferencia numerica. Campana L512 supervisada y viva: cinco celdas
entrenadas o en curso de doce, ruta gobernada verificada con respuestas autenticadas reales, correccion de procedencia
aditiva para la celda 2 (solo se reconstruyeron `started_at` y `finished_at`; la ventana medida era 14:59:01Z a 15:20:09Z),
libro de costos medido (2.517 s de CPU del carril previo; proyeccion de unos 13.100 s frente a la asignacion de 24.000) y
latido cada dos minutos. Siguiente intervencion doctoral **preparada y NO autorizada a ajustar**. Suite 128 pasan, 1 omitida.

**Ejecutado por Satoshi (24-sep, RP140-RP143, pendiente de revision):**
[retorno](../audits/work_plan/SATOSHI_RP140_RP143_RETURN_2026_09_24.md), evidencia en
`docs/audits/evidence/d3_k5_20260917/{RP140,RP142}/`. **Protocolo A consolidado:** las doce celdas entran en el mismo
denominador bajo una regla de agrupacion declarada (terminal aceptado, cierre que verifico la celda, y un replay ligado a sus
propias identidades que pasa la regla congelada). T=96 0.135496/0.232884, T=192 0.157645/0.252163, T=336 0.164480/0.262985,
T=720 0.190226/0.290617, los cuatro horizontes en ACUERDO OPERACIONAL, y por primera vez **media de cuatro horizontes
0.161962/0.259662** frente a 0.158250/0.255750 publicados; persistencia pareada en las mismas filas 0.9455/0.9507/0.9613/0.9754.
La calificacion que acompana cada fila: **ninguna celda registra el UUID MEDIDO del dispositivo que la entreno** (UNKNOWN en
cuatro, INFERRED_GPU_MEMORY en ocho), asi que una reproduccion exacta es repetibilidad de las predicciones almacenadas y no
una afirmacion de mismo dispositivo; mi redaccion anterior queda corregida en fe de erratas fechada. **Protocolo B:** diez de
doce celdas aceptadas (H96 0.125551/0.125849/0.126375; H192 0.144051/0.142953/0.143947; H336 0.152274/0.152304/0.153511;
H720 0.179536 de tres), sin replay ni cierre: puntuaciones registradas, no ciencia aceptada; la Tabla 9 conserva su lookback
sin resolver. Asignacion medida antes de cada hijo: 11.614 s de CPU de 24.000. **Contraste doctoral ejecutable:** adaptador ECL
emparejado (argumentos resueltos de la propia referencia, 321 canales en su orden, objetivo y salida [lote, horizonte, 321],
arquitectura aprobada con un unico cambio en la lectura) con causalidad **probada sobre el soporte de objetivos** y el
solapamiento de contexto del autor declarado; piloto SOLO-TRAIN ejecutado en WORKER_A dentro de 1.800/1.800 s, sin puntuar el
test y sin afirmacion sobre H1. Lane financiero investigado sin leer un byte. Suite 146 pasan.
