# Satoshi: cierre prioritario de seleccion de caracteristicas

Fecha: 2026-10-05. Autoridad: Maestro/Musashi. Esta orden sucede a
`SATOSHI_CANONICAL_SELECTION_FIRST_EXECUTION_2026_10_03.md` sin invalidar sus
contratos ni su evidencia. El objetivo unico es emitir cuanto antes un
manifiesto de seleccion EURUSD defendible para desbloquear ARCH. No abras NEAT,
H-CORE, calendario como entrada, estrategia ni RL antes de este cierre.

## 1. Estado de partida comprobado

Corte vivo: `2026-10-05T03:06Z`.

| Carril | Estado | Ritmo / ETA local |
|---|---:|---:|
| alternativas PS3-R, RTX 5090 | 244/274, 0 fallos, una celda viva | mediana 195 s; 30 restantes: 1.6-2.1 h |
| baseline PS3-R, RTX 4090 | 31/86, 0 fallos, una celda viva | mediana 1286 s; 55 restantes: 19.7-25.0 h |
| baseline sucesora, RTX 5090 | 0/51, espera el final de alternativas | medir ETA al cerrar las primeras tres |
| perfil PS4 incremental | 31 descubiertas, 31 aceptadas y perfiladas | follower activo, 0 pendientes |
| denominador | 366 candidatas; 137 en matriz pesada | ninguna seleccion final emitida |

Procesos observados: `yh.lqd.logret_5d/past_to_current_siamese` en la 5090 y
`yh.slv.logret_1d/identity,random,ae,dae` en la 4090. No los detengas, no repitas
sus celdas y no cambies su receta. Los servicios de sincronizacion, perfilado y
estado estan activos en omega.

## 2. Regla de orquestacion

Despacha al menos seis agentes con worktrees y conjuntos de escritura
disjuntos. El agente principal integra continuamente: no espera que terminen
todos para revisar el primero. Hermes/OpenCode puede hacer inventarios, pruebas,
tablas, adaptadores acotados y documentacion. Cada agente acusa rama, tip,
archivos que escribira y primer entregable antes de trabajar.

| Lane | Recurso | Entregable |
|---|---|---|
| FS-GPU | 5090 + 4090 | terminar PS3-R, transferir automaticamente al sucesor y luego robar trabajo restante sin duplicar |
| FS-PRED | CPU | todos los selectores predictivos/common-K sobre folds TRAIN identicos |
| FS-CAUSAL | CPU | selector causal auditable y escalera de tres peldaños, con abstencion honesta |
| FS-REP | CPU mientras corren GPU | decision raw/random/AE/DAE/MTAE/P2C por candidata y coste |
| FS-GEN | CPU o GPU que quepa, no bloqueante | knockoff/generador condicionado como control calibrado |
| FS-CLOSE | CPU | refits emparejados, estabilidad, manifiesto 366/366 y grafico de progreso |

La 5070 Ti comparte 14 GiB de RAM de host con la 5090: no fuerces concurrencia
si la admision no conserva la receta exacta. Omega no recibe el modelo pesado de
14-20 GiB. CPU o GPU ociosa debe tomar una tarea que quepa; nunca se inventa una
replica ni se encoge un modelo para aparentar ocupacion.

Semilla por defecto: una, retenida en cada recibo. Maximo tres semillas solo en
el desempate final de un metodo estocastico o para una comparacion publicada que
lo exija. No repitas una celda terminal valida.

## 3. FS-GPU: terminar extractibilidad sin espera manual

1. Conserva los runners actuales y sus claims exclusivos.
2. Al cerrar las 30 alternativas, la 5090 inicia automaticamente las 51
   baselines de lotes 001/003 ya declaradas.
3. Despues de esas 51, la 5090 toma, en frontera de celda, baselines todavia
   pendientes del lote 002. Debe acuñar un claim atomico compartido con Dragon;
   Dragon omite una celda reclamada o terminal. No mates una celda de Dragon.
4. El follower PS4 consume cada terminal autenticada. La matriz de cierre exige
   por candidata pesada: raw/identity, random, AE, DAE, masked temporal AE y
   past-to-current, o `NOT_APPLICABLE`/`FAILED` con razon y recibo.
5. Publica ETA de cada cola desde duraciones observadas: mediana y p90, numero de
   trabajadores y hora absoluta. Un waiter no cuenta como trabajo GPU.

## 4. FS-PRED: comparadores modernos bajo un solo contrato

Antes de implementar, congela pruebas de poblacion, folds, targets, horizonte,
K, semilla, ausencia de VALIDATION/TEST y rechazo por identidad. Reutiliza
`feature_selection_campaign.py`; no crees otro orquestador general.

Ejecuta sobre las 366 admisibles y los mismos folds TRAIN:

1. `ALL_ADMISSIBLE`, `RANDOM_K`, Spearman, MI/JMI/CMIM/mRMR y redundancia.
2. Elastic Net/group penalty con estabilidad entre folds.
3. ExtraTrees con permutacion temporal por bloques y **refit** al retirar.
4. seleccion marginal secuencial bajo presupuesto y gate de grupos temporal;
5. ChronoEpilogi como comparador prioritario de subconjuntos predictivos
   alternativos. Usa implementacion oficial compatible si licencia, pin y API
   son verificables; de lo contrario implementa un adaptador fiel y marca el
   alcance que no se pudo reproducir. No lo llames causal.

Primario `K=24`; sensibilidad sellada `K={8,16,24,32,48}`. Toda familia entrega
ranking completo y conjunto K, incluso si falla una sensibilidad. Mismo modelo,
inicializacion, updates, filas, target y naive para cada refit.

## 5. FS-CAUSAL: evidencia, no decoracion

No reutilices las 1,076 estimaciones historicas como identificacion. El estado
actual es 0/279 identificadas bajo las compuertas reparadas y esto es neutral,
no rechazo. Integra tres clases de evidencia:

1. **Peldaño 1, asociacion:** dependencia condicional lagged, placebos,
   negativos, ganancia OOF y correccion BH por familia de hipotesis.
2. **Peldaño 2, intervencion historica soportada:** episodios `A=a` y controles
   `A=a'` recuperados de TRAIN, misma prehistoria/regimen/calendario, overlap y
   balance; matching/propensity y DR o g-computation. Reporta ATT/ATE/CATE solo
   dentro del soporte observado.
3. **Peldaño 3, contrafactual de episodio:** SCM temporal ajustado en TRAIN,
   abduccion del ruido factual, cambio exclusivo de A a una alternativa
   historicamente soportada, propagacion y reconstruccion factual; placebos y
   sensibilidad. Nunca afirmes que el outcome individual fue observado.

Como comparadores de descubrimiento/cribado, evalua SyPI cuando sus restricciones
de grafo sean defendibles y un metodo consciente de estacionariedad/retardos
(PCMCI+ o minimal separating sets). ARROW puede evaluarse como acelerador de un
metodo base, no como evidencia adicional. Cada metodo declara supuestos,
condicionamiento, max lag, CI test, multiplicidad, soporte y razon de abstencion.

La salida por feature es `SUPPORTED`, `CONTRADICTED` o `NOT_IDENTIFIED`, separada
por target/horizonte/peldano. Solo `CONTRADICTED` robusto puede pesar contra una
feature; `NOT_IDENTIFIED` no elimina.

## 6. FS-REP: extractibilidad como selector de representacion

La extractibilidad decide **como representar** una candidata, no si merece vivir
por si sola. Para cada feature pesada, compara en las mismas filas:

- raw/identity;
- random encoder;
- AE y DAE;
- masked temporal AE;
- past-to-current/siames.

Mide reconstruccion normalizada, ACF/espectro/extremos/DTW cuando aplique,
estabilidad, dimension efectiva, probe hacia targets corto/largo/barrera,
ganancia incremental con refit y coste. Resultado: `RAW`, familia entrenada o
`NO_TRAINED_ADVANTAGE`. Reconstruccion pobre es diagnostico; utilidad downstream
manda. Reconstruccion buena sin utilidad no selecciona.

## 7. FS-GEN: generacion/knockoffs, acotado y fail-closed

No selecciones una feature porque un VAE/GAN la imita. Implementa una rama
secundaria que no retrase el manifiesto principal:

1. generador condicionado solo con informacion disponible: calendario conocido,
   mascara/delta y pasado; target futuro prohibido;
2. validacion de fidelidad temporal, colas, espectro, dependencia cruzada,
   cobertura por regimen y prefix invariance;
3. si y solo si pasa diagnosticos de intercambio/condicionalidad, ejecuta
   Model-X multi-knockoffs o group knockoffs con FDR declarado;
4. si no pasa, emite `NOT_CALIBRATED` y usa el generador solo para stress test,
   negativos y futura ampliacion de entrenamiento.

El resultado generativo es una columna de evidencia y estabilidad. No sustituye
utilidad externa, causalidad ni refit.

## 8. FS-CLOSE: decision final y artefactos obligatorios

Compara como minimo, a K comun:

1. ALL_ADMISSIBLE;
2. mejor selector predictivo/redundancia;
3. anterior + evidencia causal;
4. anterior + representacion elegida por extractibilidad;
5. knockoff/generativo, solo si quedo calibrado.

La decision usa utilidad externa emparejada y parsimonia: MAE y MSE por
horizonte, naive de las mismas filas, skill, estabilidad/Jaccard por fold,
coste, latencia y memoria. Ningun resultado va a estrategia si no supera
estrictamente su naive pareado. VALIDATION se usa para elegir al final; TEST
permanece sin leer hasta congelar metodo y manifiesto.

Entrega atomica:

- `feature_dispositions.csv`: exactamente 366 filas y una disposicion por fila;
- `representation_dispositions.csv`: las 137 pesadas y todos sus controles;
- `selector_sets.json`: metodos, K, orden, folds, targets, semillas y digests;
- `paired_refit_metrics.parquet`: poblacion y naive explicitos;
- `causal_evidence.jsonl`: tres peldaños o abstencion por feature;
- `FINAL_SELECTION_MANIFEST.json`: conjunto primario y sensibilidades;
- `STATUS.json` y `MASTER_MILESTONE_PROGRESS.png` generados desde evidencia.

El cierre falla si falta una de las 366 disposiciones, si mezcla poblaciones, si
un conjunto K no tiene ranking completo, si usa TEST, si confunde
`NOT_IDENTIFIED` con rechazo o si una metrica carece de naive pareado.

## 9. Reporte de Satoshi

Publica un primer acuse tras despachar agentes y un retorno al completar cada
hito, sin esperar el cierre total. Formato obligatorio:

1. commit/ramas/worktrees y agentes realmente activos;
2. tabla por host: proceso, celda, GPU/RAM, inicio y ETA;
3. conteos PS3-R/PS4, selector, causal y manifiesto sobre sus denominadores;
4. mediciones nuevas, separadas de software/test;
5. semillas, filas, target, horizonte, modelo, naive, coste y digests;
6. fallos/abstenciones con objeto faltante y siguiente accion ya despachada;
7. progreso visual actualizado.

No pidas autorizacion rutinaria ni dejes una GPU esperando una revision CPU. Un
bloqueo de un metodo no detiene los otros. No marques seleccion completa hasta
que el manifiesto 366/366 y los refits comunes sean verificables.

