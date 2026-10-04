# Orden para Retsu: reparar M1/M2 y continuar el camino critico

Base autoritativa: `satoshi/canonical-exec-20261003` en el tip que contiene este archivo. Lee primero `CURRENT_EXECUTION.md`, `EXPERIMENT_EXECUTION_QUEUE.json` y `RETSU_M1_M2_AUDIT_2026_10_04.md`. No uses handoffs fechados como autoridad. No detengas ni dupliques los hijos GPU vivos.

Ejecuta cinco carriles independientes en paralelo. Devuelve cada carril cuando termine; no esperes a los demas. Seed 0 para pilotos nuevos; no repitas una celda terminal autentica.

## A. Reparacion del scorer semanal M1 (CPU)

Continua desde `retsu/m1-weekly-score-gate-20261003@39a45e88`. Escribe primero pruebas rojas para estos cinco casos y conserva PRE/POST:

1. Una semana con modelo corto digest A y modelo largo digest B se acepta si tarea, cutoff, split, poblacion y origenes coinciden. Los dos digests quedan ligados a sus forecasts.
2. Cambiar el digest dentro del mismo forecast o reutilizar familia+horizonte contradictorios se rechaza.
3. `to_json -> from_json` conserva todas las diferencias pareadas por origen; el digest las cubre y una mutacion se rechaza.
4. El cierre anual llama a las semanas `longitudinal_evaluation_units`; no infiere replicas independientes. Publica `independence_status=NOT_ESTABLISHED` y no copies el denominador a `independent_replicates`.
5. Configura para la tarea financiera el error primario como MAE en la escala declarada de cada target. MSE es secundario. La compuerta exige MAE estrictamente menor que el naive de las mismas filas en todos los horizontes requeridos.

No entrenes modelos. No marques BW13 completo hasta que las nuevas pruebas y las 13 existentes pasen en proceso limpio. Retorno: commit, comandos, conteo, PRE/POST y lista exacta de campos serializados.

## B. Reparacion causal del ledger M2 (CPU)

Continua desde `retsu/m2-readiness-ledger-20261003@5da79f34`, sin reescribir evidencia historica.

1. Sustituye la fuente causal por los tres `laneC/reanalysis_48ae17c/batch_*/ps3c_join.json` y fija `producer_revision=48ae17c` mas hash del join en cada fila.
2. Congela un test que muestre que la fuente vieja produce 65 identificadas + 13 mixtas y que el ledger reparado produce 0 identificadas sobre 279, 279 `NOT_IDENTIFIED` y 87 fuera del join/pending. `NOT_IDENTIFIED` nunca es `REJECTED`.
3. Si una candidata no tiene evidencia PS3-R/PS4 requerida, su PS5 es `NOT_READY_EVIDENCE_INCOMPLETE`; solo el diseno global puede decir `PREPARED_NOT_TRAINED`.
4. Recalcula los conteos y el REPORT. No ejecutes PS5 ni emitas seleccion.

Retorno: commit, pruebas, hashes de los tres joins, tabla de conteos antes/despues.

## C. Ingestor continuo PS3-R/PS4 (CPU, independiente de A/B)

Crea un modulo probado que sustituya la lista manual de features terminales. Debe escanear solamente raices permitidas declaradas en config: baseline Dragon/lane E y alternativas Gamma/lane F.

Para adoptar una celda exige: `status=COMPLETED`, feature esperada, seed 0, familias exactas, folds exactos, revision permitida, input digest, results digest recomputado y ausencia de otra terminal contradictoria. Un directorio parcial o vivo es `NOT_TERMINAL`, nunca cero ni completado. Lane F queda como evidencia de familia alternativa, no reemplaza baseline.

Pruebas obligatorias: hash errado, feature errada, familia faltante, seed distinta, manifiesto parcial, dos terminales contradictorias y adopcion positiva de los tres artefactos `tv.hilbert_amp`, `tv.kalman_dev`, `tv.stl_dev`. El ledger debe actualizarse sin editar una constante por feature.

## D. Perfil PS4 incremental (CPU Omega)

Mientras las GPU continuan, perfila cada feature PS3-R terminal autentica que aun no tenga PS4. Usa solo TRAIN y los cinco folds existentes; prohibido target, validation externa y test. Ejecuta bajo `crispdm-run -m 2G -t 120s`, una unidad a la vez, y persiste una unidad atomica por feature-fold para reanudar.

Primero generaliza el profiler con un fixture `series.npz`; luego procesa las terminales disponibles. No esperes a que terminen las 366. Publica conteos `MEASURED/PENDING/FAILED` y hashes. PS4 no selecciona features por si solo.

## E. Continuacion GPU (sin detener trabajos vivos)

- Gamma 5090: deja terminar el driver lane F, una sola celda, siguiendo el plan sellado despues de `tv.stl_seasonal`. No concurras con la 5070 Ti mientras la RAM compartida no admita ambos hijos.
- Dragon 4090: deja terminar `fx.eurjpy.logret_1h` y continua la siguiente baseline no reclamada, una celda.
- No bajes topes, no repitas terminales, no lances NEAT, arquitectura modular, H-CORE, RL ni estrategia antes del manifiesto final M2.
- Tras cada terminal, el carril C debe adoptarla por manifiesto y el carril D debe poder perfilarla sin una edicion manual.

## Formato de retorno

Empieza con resultados nuevos, despues incidencias. Una fila por carril: estado, commit, pruebas, denominador, completadas/pendientes, recurso, pico, pared y siguiente accion. Reporta siempre 366 como denominador de seleccion; separa medicion de extractor, perfil PS4, evidencia causal y decision final. Incluye estado vivo de 5090/4090 y ETA basada en mediana observada, no una fecha inventada.
