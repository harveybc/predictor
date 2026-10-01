# Satoshi: seleccion progresiva y continuacion modular en paralelo

Para Satoshi (orquestacion/ejecucion), Musashi (revision) y el propietario.
Fecha local 30-sep-2026; inspeccion 01-oct aproximadamente 01:53 UTC.

## 1. Mandato y precedencia

El propietario aprobo incorporar seleccion progresiva, causalidad y evaluacion
de representaciones al plan maestro, sin detener los desarrollos independientes.
Lee primero el [subplan aprobado](../tres_temas_entrevista/program_v3/FEATURE_SELECTION_REPRESENTATION_WORK_PLAN_2026_09_30.md)
y su estado metodologico. Esta orden complementa la campana M01-M06, C07 y S07;
no abre ocho reconstrucciones nuevas de lo que ya funciona. Integra aditivamente
las entregas `09f129c5` y los tips que su RETURN enumera. No repitas Traffic H96
ni los 16 ajustes para producir otro informe. Conserva historia y artefactos.

Sigue vigente la [reconciliacion arquitectonica](SATOSHI_MODULAR_RECONCILIATION_2026_09_30.md).
Donde una cola vieja pide entrenamiento con ramas de 12 pasos como si fueran el
default aprobado, esta orden la sustituye. No modifica presupuestos, licencias,
mandatos financieros o asignaciones de riesgo. Publicacion de esta orden no es
acuse de recibo ni despacho de un agente.

## 2. Estado comprobado y correccion que precede al siguiente ajuste modular

Retorno leido: M06 `09f129c5`, ocho carriles retornados. M01 probado `64a91a74`,
retorno `1bb63f51`, remanifest `41955f50`; M02 `d1e2e77c`; M03 `4ddcce48`;
M04 `6ced97e7`, pin de campana `8f724b12`; M05 predictor `a3218b78`, LTS
`4e88c01`; C07 `788082f6`; S07 predictor `53c6f47a`, estrategia `d08ae00`.
Verifica los tips de nuevo antes de integrar, sin pisar trabajo posterior.

M01 aun contiene el motor monolitico con branch_steps=12. El config vivo de M02
tambien declara window=24, branch_steps=12, factores del nucleo [2,1,1]. No es
solo ausencia de un ancestro Git: codigo y configuracion siguen difiriendo.
El donante CPU estaba en rama 162/321 (162 terminadas), no 132; ese latido no
certifica ni el nucleo ni la arquitectura corregida.

**No avanzar directamente a v4 R1/R2 mediante remanifest de estos donantes.**
El remanifest puede documentar compatibilidad de metadatos con la misma funcion;
no cambia una rejilla 12 en 24 ni prueba paridad con otro grafo. Los pesos
anteriores son evidencia del diseno anterior, no falsos ni inutiles por decreto.

Satoshi y M01: integrar `da4ce7b4` como especificacion/codigo candidato de la
arquitectura corregida, conservando fachada, entry points, semver, gramatica,
defaults efectivos, serializacion, reanudacion y early stopping de M01/M02.
No hacer cherry-pick ciego sobre los cambios del otro agente. Una sola
implementacion publica y un solo adaptador de parametros para M04.

Default horario a comprobar con Keras real, no solo JSON:

- Rama por entrada: (B,24,1) -> Conv1D causal -> (B,24,16).
- Fusion por canales: (B,24,16*F); F real, no cuatro entradas fijas.
- Codificacion posicional inmediatamente tras fusion; proyeccion por instante,
  dos Transformer completos con atencion/FFN, sus Add y normalizaciones.
- Nucleo: Conv1D residuales, tiempo 24->12->6->6, canales 32->16->8.
- Sin flatten temporal antes del cabezal; Dense por ultimo eje no colapsa tiempo.
- Early stopping y restauracion del mejor checkpoint en ramas, nucleo y predictor;
  RL conserva su evaluacion por episodios. R0/R1/R2 se comprueban por pesos/updates.

Para el trabajo viejo activo: no editar ni matar a ciegas. Comprobar alcance,
guardar estado/recibos y usar su frontera segura de checkpoint si continuar solo
produciria donantes incompatibles. No encadenar el nucleo viejo automaticamente.
Se pueden recuperar capas compatibles como variante explicitamente definida,
pero exige identidad y prueba nuevas, no etiquetar el viejo encoder como nuevo.

## 3. Despacho concurrente, sin duplicar propietarios

Satoshi coordina e integra; no implementa todo secuencialmente. Reutiliza agentes
existentes solo si realmente siguen disponibles; si no, despacha reemplazos con
acuse e ID. Hasta seis agentes al inicio, incrementables segun memoria medida;
numero de agentes no implica seis fits simultaneos. Worktrees separados,
propietario unico por modulo, mensajes directos productor-consumidor.

| Carril | Responsable y trabajo inmediato | Primer entregable / dependencia local |
| --- | --- | --- |
| A Motor y donantes | M01 + M02: reconciliar una sola arquitectura, puerto de tests/early-stop/resume y manifiestos | Commit integrado instalado, grafo Keras PNG y pruebas de formas/rejillas/regimenes/paridad. Luego donantes nuevos y nucleo sobre fusion correcta. No espera estudios causales. |
| B Datos y seleccion | M03: PS0/PS1, reutilizar perfiles; PS2 reversible y muestra exploratoria | Cobertura por metrica/feature/fold, huecos y lista de lotes admisibles; pruebas FS01/02/15/16/19. No espera A. |
| C Causalidad y bibliografia | Agente causal-inference + feature-extractor: especificar PS3-C y shortlist PS3-R | Episodios/targets/ajuste/soporte, tres peldanos de seccion 5; referencias oficiales y comparadores reproducibles. No exige VAE ni una intervencion fisica. No espera A/B completos. |
| D Optimizacion | M04: preparar cola corregida y coste por etapas; consumir commit A al estar listo | Piloto completo medido y lote DOIN finito; R0 mientras se preentrena, R1/R2 solo con donantes compatibles. No espera PS3-C/PS7. |
| E Producto/negocio | M05: mantener carril semanal y adaptar consumidor instalado, shadow/paper con mandato existente | Contrato/escala/poblacion, replay y prueba fin a fin del runner que realmente consume; no solo existencia del adapter. No promociona fixture ni score electrico a dinero real. |
| F Recursos/evidencia | M06: reconciliar leases/procesos/colas, corregir estado y enlazar warehouse | Informe vivo, cuotas medidas, proyeccion de metricas por identidad, PNG/ETA desde mismo estado; revisa ADM-DEADCACHE sin tratar una resta de cache como memoria garantizada. |

Al liberar A, su agente de preentrenamiento toma PS3-R con los extractores de
referencia y controles. C07, S07 y clasificacion mantienen sus expedientes y
dependencias locales: no se vuelven prerrequisitos del motor o los perfiles.
Calendario y M5PHET reciben contratos reutilizables, no implementaciones paralelas
de la misma causalidad. La inferencia causal reside en causal-inference;
feature-eng produce datos/perfiles y feature-extractor representaciones.

## 4. Orden de pruebas y experimentos

El subplan completo contiene PS0-PS7, FS01-FS20 y los tres peldanos causales.
Prioridades de esta tanda:

1. Perfilar todas las entradas admisibles por lotes; no computar primero todas
   las descomposiciones de todas las columnas. Reutilizar 3.455/15.256 filas
   reportadas y explicitar 15.228 columnas distintas y huecos por metrica.
2. Pilotar unas 20 entradas/grupos estratificados, incluida exploracion de baja
   prioridad. Seleccion por aporte a targets cortos/largos y politica, no
   self-forecast ni reconstruccion perfecta. Mantener control de todas las
   entradas admisibles y grupos con sinergia.
3. Comparar orden causal-primero, representacion-primero e intercalado progresivo
   a presupuesto total emparejado. Cotizar antes de escalar. No gastar todo el
   presupuesto en ganadores aparentes del primer cribado.
4. Probar referencias contrastivas, enmascaradas y latentes junto al control AE
   cuando sean reproducibles/admisibles. VAE/generacion es variante secundaria.
   No inventar puntuacion reconstructiva para encoder sin decoder ni usar pesos
   preentrenados en test como si fueran TRAIN-only.
5. Comparar adiciones, retiradas con reajuste, agrupaciones y regimenes en la
   arquitectura temporal. Nuevos grupos/cambios de ramas invalidan upstream del
   nucleo donante: rematerializar y entrenar el que corresponda.

Disenar pruebas de comportamiento de arriba abajo por incremento, luego
implementar de abajo arriba. No anunciar FSxx verdes por tests de otra familia.
Los controles sinteticos calibran mecanismos; medir utilidad en filas reales
reservadas por el protocolo y naive pareado, con precision suficiente para
diferencias 1e-5/1e-6. No declarar ganancias financieras por MAE solamente.

## 5. Recursos y pendientes humanos: dependencias locales

Preferencia: 5090 externa para ajustes individuales largos. Host RAM, VRAM,
temperatura y leases se verifican juntos; las dos tarjetas de worker_a comparten
RAM y no son dos anfitriones independientes. Worker_b aloja CPU/GPU admisibles;
coordinador conserva escritorio, nada pesado para llenar un indicador de uso.
Trabajo util y cola preparada, no entrenamiento ficticio ni GPUs ocupadas por
obligacion cuando no hay una tarea valida. Entre fits, validacion y transferencia
tienen estado y tiempo propios. Un rechazo reubica tareas independientes.

COST-01 anterior pidio 8 GiB solo despues de fallar construccion con 6.44 GB y
cero updates; es del grafo anterior. Reperfilar el grafo corregido completo,
incluido optimizador/materializacion. No usar ese numero como medida del sucesor.
Mantener caps/salud del escritorio. No reusar presupuestos gastados como si
siguieran disponibles; informar remanente y coste propuesto en el acuse.

Sobre las cinco decisiones del retorno:

- ADM-DEADCACHE: revisar el fix `0928dc06` y DEPLOY con pruebas de cargas vivas,
  cache compartida, reclaim parcial, reserva y memory.max. Cache estimada limpia
  no es permiso para sobreasignar RAM. Preparar despliegue reversible; donde la
  autorizacion vigente no cubra reemplazar tooling compartido, formular una
  sola accion concreta al propietario. Mientras tanto usar slots admisibles.
- Slab/persistencia: diagnostico de recursos, no tratamiento demostrado. No
  activar modo persistencia/root timer ni reiniciar hosts por inferencia.
- Historia Git: correccion hacia delante ya existe. No force-push sin concesion
  explicita; no es dependencia de los experimentos ni de esta integracion.
- C07: conservar solicitud exacta de ejecucion de codigo remoto/licencia no
  comercial. No deducirla de la aprobacion de este subplan. Otros proveedores
  ya autorizados pueden continuar por su propio contrato.
- S07: reserva leida no se vuelve holdout limpio al renombrarla. Proponer una
  nueva ventana prospectiva o declarar desarrollo y documentar decision. No
  bloquear mecanica sintetica, perfiles ni comparaciones retrospectivas validas.

## 6. Informe exigido y cadencia

Acuse en los primeros 15 minutos **desde recepcion efectiva**: revision leida,
IDs/agentes/tareas aceptadas, estado real del donante antiguo, commit de
integracion previsto, slot candidato, remanente de asignacion y ETA del siguiente
entregable. No afirmar despacho sin acuse de cada agente.

Heartbeat por ajuste <=60 s; STATUS atomico <=5 min; resumen cada 30 min mientras
haya trabajo activo, y al terminar/fallar cada celda. Detencion del agente y
final del job son eventos distintos. Corregir entradas antiguas como "piloto
queued" cuando batch1 ya termino. Nada de enviar el mismo bloqueo como progreso.

Retorno Markdown + STATUS/RESULTS JSON/CSV, con enlaces por commit y ruta:

1. Resultado nuevo primero: modelo/config/arquitectura, dataset/split/poblacion,
   metrica/escala/reduccion, valor, naive mismas filas, referencia publicada
   comparable o no disponible, semillas/dispersion, coste y limite del resultado.
2. Hardware por rol/GPU/UUID, job/PID/lease, etapa/updates, temperaturas y memoria;
   libre/espera/build/fit/validacion distinguidos, siguiente tarea y causa concreta.
3. Progreso con denominadores: FS pruebas aceptadas/20, perfiles por metrica y
   columnas, celdas finalizadas/planificadas por lote. FS aprobados no equivalen
   a porcentaje del programa doctoral. Registrar evidencia para cada avance.
4. ETA por etapa: restante x tasa medida, intervalo observado, dependencias y
   hora de estimacion. El ETA de una rama no es el ETA de 321 ramas+nucleo.
   Para tarea no perfilada, entregar primero hora prevista del piloto y despues
   estimacion medida; no una fecha global sin fundamento.
5. PNG de progreso desde ese mismo estado, pruebas/commits/evidencia, cambios
   frente al plan, pendientes con objeto/dueno/accion minima. Diferenciar
   programado, ejecutandose, implementado, verificado y demostrado cientificamente.

M06 conserva la tabla por horizonte: el titular M04 0.397174 MAE_z frente a
0.851406 persistencia tiene skill negativo en h1/h23/h24 en las dos semillas.
Es validacion L24/H1..24, arquitectura anterior, NOT_COMPARABLE con la tabla
publicada ECL. Traffic H96 conserva 0.375199/0.251143 MSE/MAE frente a
0.375/0.251 publicados. Este parte no reentrena ni reaudita numericamente esas
celdas; no cambiar su clase de evidencia por citarlas aqui.

Integrar cada entrega aceptada al estar lista. No esperar al cierre de todos
para alimentar la cola siguiente; no convertir este subplan en otra barrera
entre datos reales, optimizacion y el consumidor de negocio.
