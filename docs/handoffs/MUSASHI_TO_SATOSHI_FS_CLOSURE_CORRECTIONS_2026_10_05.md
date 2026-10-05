# Correcciones de cierre para Satoshi

Fecha: 2026-10-05 08:31Z. Autoridad: Maestro/Musashi. Esta orden es aditiva a
`SATOSHI_FEATURE_SELECTION_CLOSURE_2026_10_05.md`. No detiene PS3-R, PS4 ni
FS-CAUSAL. Corrige dos defectos antes de que un DRAFT pueda convertirse en FINAL.

## 1. Veredicto de auditoria

La campana sigue el spearhead correcto: seleccion antes de ARCH/H-CORE/NEAT/RL;
366 candidatas conservadas; TEST cerrado; `NOT_IDENTIFIED` neutral; FS-GEN
fail-closed; ambas GPU procesando PS3-R. No hay desviacion general.

Hallazgos que requieren reparacion:

1. **FS-CLOSE no es business-faithful.** `run_closure` ajusta una sola vez con
   TRAIN 2012-05..2023-12 y puntua todo 2024. Puede conservarse como
   `LITERATURE_STATIC_VALIDATION_DIAGNOSTIC`, pero no puede elegir el manifiesto
   EURUSD ni satisfacer C9/C11. Nuestro contrato principal exige walk-forward
   semanal: cuatro anos calendario as-of antes de cada semana y un ajuste nuevo
   antes de puntuar esa semana.
2. **FS-PRED esta bloqueado por memoria, no avanzando.** A las 08:30Z el proceso
   `3593336` llevaba 2 h 32 min, estado `D`, `mem_cgroup_handle_over_high`, sin
   escribir desde 06:12Z. El scope usa `MemoryHigh=3.6 GiB`, `MemoryMax=4 GiB`,
   pico 4,120,571,904 bytes. El slice padre esta sobre su high de 12 GiB. El ETA
   08:44Z es invalido mientras esto no se repare.
3. **Los nombres de estado exageran PS3-C.** `366/366 PS3-C` significa cobertura
   del join, no causalidad terminada. FS-CAUSAL es la medicion real y estaba en
   45/47 chunks al auditar. Reportar ambos por separado.
4. **PLUS_REP no prueba una representacion codificada.** Los refits actuales solo
   disponen de features raw; G4 para familias entrenadas es `NOT_APPLICABLE`.
   Estas decisiones son de extractibilidad/probe y deben llamarse provisionales.
   La comparacion real raw versus latente pertenece a M4 despues del manifiesto.

## 2. Reparacion inmediata de FS-PRED

No repitas las 111 celdas terminales ni borres salidas. El runner es resumible.

1. Deten solamente `fs-pred-runner.service` y su scope actual; no toques
   `feature-selection-4090-claims`, `fs-causal-driver` ni `fs-close-refit`.
2. Conserva todos los JSON completos. La celda ChronoEpilogi que no produjo
   terminal puede reiniciarse; registra el intento interrumpido.
3. Amplia la politica persistente de `crispdm-batch.slice` en Dragon a
   `MemoryHigh=18G`, `MemoryMax=20G`. Hay 30 GiB fisicos y la VM MT5 permanece
   apagada; no uses esa ampliacion para lanzar un segundo entrenamiento GPU.
4. Relanza FS-PRED con nombre nuevo y `-m 5376M`. Esta cifra es el techo de
   produccion: ceil practico de 1.25 x el pico observado de 4,120,571,904 bytes.
   No bajes el modelo ni los datos.
5. Prueba PRE/POST: antes, progreso mtime estancado + estado D + high events;
   despues, la siguiente celda terminal aparece y `MemoryPeak <= MemoryMax`.
6. Anade watchdog: progreso sin cambio durante
   `max(10 min, 3 * p90_de_celda)` y proceso en memory pressure es `STALLED`, no
   RUNNING. Un PID vivo no valida progreso. Recalcula ETA solo despues de tres
   celdas POST.

## 3. Sustituir el cierre estatico por BUSINESS weekly walk-forward

Escribe primero pruebas que fallen contra `run_closure` actual:

- un unico fit para todo 2024 no puede producir `FINAL` de negocio;
- toda semana puntuada tiene su `WeekSpec`, cutoff, cuatro anos calendario
  exactos, digest de filas de fit y pesos/update identity propios;
- ninguna fila disponible despues del cutoff entra al fit;
- se puntuan todas las semanas completas consecutivas derivadas del calendario
  2024, sin hard-codear 48/52;
- cada semana lleva naive en exactamente las mismas filas;
- TEST 2025 sigue inaccesible;
- reinicio/reanudacion no repite una semana terminal.

Implementacion:

1. Reutiliza `business_weekly_protocol.py`, `business_asof_window.py`,
   `business_weekly_training.py`, `manifest_weekly_rows.py` y
   `business_weekly_score.py`. No construyas un segundo framework semanal.
2. Congela primero los conjuntos K=24 de `ALL_ADMISSIBLE`, `PRED_BEST`,
   `PLUS_CAUSAL` y `PLUS_EXTRACTIBILITY_EVIDENCE`; `KNOCKOFF` solo si queda
   calibrado. No leas VALIDATION para crear o modificar conjuntos.
3. Para cada semana completa de 2024 y cada conjunto, ajusta el mismo head ridge
   desde la receta declarada usando exactamente los cuatro anos calendario
   anteriores disponibles al cutoff. Puntua solo la semana siguiente.
4. El modo primario es `BUSINESS_WEEKLY_WALK_FORWARD` con `FULL_RETRAIN` para
   este head barato. `FINE_TUNE` y mensual quedan como sensibilidades futuras,
   no sustituyen el primario.
5. Agrega MAE/MSE o log-loss/Brier por target/horizonte, naive pareado, skill,
   semanas elegibles, dispersion semanal, coste y digests. El ganador usa el
   agregado predeclarado sobre todas las semanas, no la mejor semana.
6. El cierre estatico existente se conserva con nombre y clase diagnostica,
   separado. No autoriza C9, C11, estrategia ni promocion.
7. `FINAL_SELECTION_MANIFEST.json` solo pasa a FINAL cuando el cierre semanal,
   las 366 disposiciones, FS-PRED completo, causal final y PS3-R/PS4 137/137
   pasan. La compuerta debe rechazar cualquier `evaluation_mode` distinto de
   `BUSINESS_WEEKLY_WALK_FORWARD` para el manifiesto de negocio.

## 4. Causal, generacion y representacion

- Deja terminar FS-CAUSAL. Al cerrar, sincroniza `causal_evidence.jsonl` y
  recalcula estados por feature. Las 117 celdas provisionales son
  `IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS`, no 117 features ni verdad
  causal incondicional.
- FS-GEN ya dio el resultado correcto: 2 features con generador calibrado, 0
  features con seleccion knockoff mayoritaria; brazo 5
  `NOT_CALIBRATED/EMPTY`. No repitas ni fuerces un ganador.
- Renombra la salida FS-REP actual como
  `PROVISIONAL_EXTRACTIBILITY_DISPOSITION`. Puede priorizar RAW o una familia
  para M4, pero no afirma mejora downstream de latentes mientras G4 entrenado
  sea `NOT_APPLICABLE`.
- `PLUS_REP` pasa a `PLUS_EXTRACTIBILITY_EVIDENCE` en el cierre de features. El
  experimento M4 posterior medira raw versus latente con los extractores reales.

### Disposicion obligatoria para ausencia total de TRAIN

`fred.credit.bamlh0a0hym2.logret_1d` no es una victoria de RAW. Sus tres
terminales pesados (baseline, MTAE y past-to-current) fallaron porque la serie
no contiene ninguna observacion util en TRAIN. FS-GEN confirma la misma causa
en sus cinco pliegues con `GENERATOR_FIT_REFUSED:no observed TRAIN values to
fit normalization`.

1. No reintentes esas celdas y no reduzcas el denominador de 366 candidatas.
2. Registra la fuente/feature como `NOT_AVAILABLE_FOR_TRAIN` con la causa
   `NO_OBSERVED_TRAIN_VALUES`; representacion y extractibilidad quedan
   `NOT_APPLICABLE`, nunca `RAW` ni `NO_TRAINED_ADVANTAGE`.
3. Excluyela de todos los conjuntos que entren modelos, conservandola en el
   inventario y en el manifiesto con su disposicion explicita.
4. Agrega una regresion: si RAW y todas las familias entrenadas carecen de
   observaciones TRAIN, el agregador no puede caer por defecto a RAW. Conserva
   aparte el caso valido donde RAW tiene soporte y solo fallan las familias
   entrenadas.

## 5. Reporte y orquestacion

Continua en paralelo:

- 5090 y 4090: PS3-R sin cambios ni duplicados;
- Dragon CPU: FS-CAUSAL hasta 47/47 y FS-PRED reparado;
- CPU libre: pruebas e implementacion del cierre semanal;
- follower: PS4/FS-REP y DRAFT, nunca FINAL anticipado.

Devuelve un ACK inmediato con agentes/worktrees y, despues, un retorno con:

1. PRE/POST de la presion de FS-PRED y tres celdas nuevas;
2. ETA nueva basada solo en POST;
3. tests del rechazo del cierre estatico;
4. contrato semanal sellado antes de leer 2024;
5. causal 47/47 y conteos feature-level finales;
6. progreso GPU/REP y lista exacta de objetos aun faltantes.

No esperes autorizacion adicional y no detengas carriles independientes.
