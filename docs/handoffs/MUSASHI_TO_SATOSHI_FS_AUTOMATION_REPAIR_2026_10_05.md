# Orden inmediata: reparar la automatizacion de FS y terminar la seleccion

Fecha: 2026-10-05
De: Musashi
Para: Satoshi
Prioridad: P0; ejecutar sin nueva autorizacion

## Veredicto de auditoria

La seleccion NO esta terminada. A las 17:45Z el estado real era:

- FS-CAUSAL 5124/5124 y FS-REP 137/137, pero FS-REP contiene decisiones
  provisionales alimentadas por 25 falsos terminales;
- PS3-R 111/137 y PS4 102/137;
- FS-PRED 278/980, con una celda viva sin terminal desde 16:13Z;
- FS-CLOSE refits 146/146 del conjunto actualmente disponible, pero C7 sigue
  incompleto y el cierre semanal no puede arrancar;
- las GPU 5090 y 4090 estaban ociosas.

El runner `steal_batch_002_5090.sh` convirtio 25 rechazos de admision en
`FAILED.json`. Los `stdout.log` dicen literalmente
`REFUSED ABOVE_SLICE_CEILING -- 8.79G exceeds ... 8.00G` y `nothing was
started`. No son fallos cientificos. El cap de 8998M fue contaminado por el
pico de otro host; las 51 baselines reales de gamma miden un maximo de
6272532480 bytes, cuyo margen 1.25x es 7478 MiB. 7936M satisface el margen y
cabe en el slice existente.

## A. Reparacion test-first de estados de ejecucion

Despacha un agente Hermes de codigo y otro de prueba, con write sets separados.

1. Congela PRE con un fixture de `crispdm-run` que devuelve 75 y escribe
   `REFUSED ... nothing was started`.
2. En los drivers GPU, `rc=75` o un recibo `REFUSED` sin `started_at` significa
   `RETRYABLE_ADMISSION_REFUSAL`, nunca `FAILED` ni terminal cientifico.
3. Tras el primer rechazo de admision, el driver debe detener esa cola y
   conservar las celdas restantes como `PENDING`; no debe quemar toda la cola.
4. `FAILED` solo es valido si el recibo demuestra que el hijo fue iniciado.
5. `ps3r_claims`, el status y ambos workers deben ignorar como terminal cualquier
   marcador de fallo sin prueba de inicio.
6. Pruebas POST obligatorias: rechazo 75 no terminal; error posterior a inicio
   si terminal; peer no salta un rechazo; reinicio retoma exactamente una vez.

## B. Recuperar las 25 celdas sin repetir resultados validos

1. Enumera exactamente los 25 `FAILED.json` con `rc=75` bajo
   `selection_successor/baseline/batch_002`. Publica lista y digest.
2. Mueve esos marcadores a evidencia historica de intentos rechazados. No los
   borres ni los presentes como resultados.
3. Supercede sus claims `FAILED` con `PENDING_AFTER_ADMISSION_REFUSAL`.
4. Revierte solamente esas 25 disposiciones FS-REP a `PENDING_BASELINE`; no
   cambies las otras 112 decisiones ni las 272 alternativas validas.
5. Conserva aparte `fred.credit.bamlh0a0hym2.logret_1d` como
   `NOT_AVAILABLE_FOR_TRAIN`; esa es la unica falla por datos y no se reintenta.

## C. Relanzamiento paralelo gobernado

1. 5090 de gamma: nueva generacion de runner, cap 7936M, una celda a la vez,
   recorriendo los 25 pendientes en reversa.
2. 4090 de dragon: puede tomar simultaneamente desde el frente con cap 8000M,
   siempre que FS-PRED mantenga margen y los claims impidan duplicados.
3. No uses la 5070 Ti simultaneamente con la 5090 en gamma por RAM compartida.
4. El follower PS4 y FS-REP deben consumir cada terminal valido en cuanto
   aparece. No esperes a que terminen las 25.
5. Publica ETA despues de dos terminales validos por host. Nunca derives ETA de
   rechazos de admision.

## D. FS-PRED: automatizacion por celda, no por target monolitico

El proceso en dragon esta consumiendo CPU, pero no produjo terminal desde
16:13Z. La ETA 18:20Z no es valida porque ignora esta observacion censurada.

1. Captura stack, metodo/fold actual, CPU, RSS y tiempo sin progreso antes de
   intervenir.
2. Activa el watchdog ya ordenado: stale = sin terminal por
   `max(10 min, 3*p90_del_metodo)`. CPU al 100% no invalida stale.
3. Conserva las 278 celdas. Interrumpe solo la celda actual si excedio su limite.
4. Ejecuta cada method-cell en subprocess propio, con recibo y timeout propio;
   un metodo no puede inmovilizar todo un target durante 20 horas.
5. Reanuda desde la primera celda ausente. Un timeout es `TIMED_OUT_RETRYABLE`,
   no ranking vacio ni metodo completo.
6. Recalcula ETA con tres terminales POST e incluye duraciones censuradas.

## E. Cierre semanal y final automatico

`fs_close_weekly.py` y el contrato `BUSINESS_WEEKLY_WALK_FORWARD` ya existen.
No construyas otro framework.

1. Cuando FS-PRED complete rankings, genera PRED_BEST y sus refits C7.
2. En cuanto C7, PS3-R y PS4 esten completos, el follower debe lanzar
   automaticamente el cierre semanal 2024.
3. Cada semana: FULL_RETRAIN con cuatro anos calendario anteriores al cutoff,
   score de la semana siguiente, naive en las mismas filas, TEST cerrado.
4. Ningun estado intermedio o cierre estatico puede promover el manifiesto.

## F. ACK y retorno

ACK inmediato con agentes, worktrees y servicios. El siguiente parte debe
mostrar, sin narracion sustitutiva:

- PRE/POST de rc75 y conteo 25 -> 0 falsos terminales;
- GPU, celda activa y ETA de los 25 pendientes;
- FS-PRED POST, metodo/fold activo, terminales y ETA corregida;
- PS3-R/PS4/FS-PRED/C7 numeradores reales;
- estado del cierre semanal y lista exacta de objetos faltantes.

No pidas autorizacion adicional. No repitas celdas terminales validas. No
despaches trabajos posteriores a seleccion mientras quede este camino critico.
