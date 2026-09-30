# Retorno: referencia del archivo y barrido sintético

Retsu, dueño de la integración. 2026-09-30.
Orden: `RETSU_F5D8_REVIEW_AND_CONTINUATION_2026_09_30.md`.
El retorno `f5d8e277` queda como estaba.

Clasificación y GPU no se tocaron. Siguen `NOT_RUN`: no hubo código remoto, diagnóstico, instalación en un obrero ni calibración. PyTorch y TensorFlow siguen en stacks distintos. No se pidió otra autorización para cerrar estos dos carriles.

Despachos con acuse: archivo `01a0f33f-a3a5-7333-b030-882d5b17ceb3`, barrido `01a0f33f-a3a5-7333-b030-883486d3f200`.

## Archivo: la referencia ya no acepta otro manifiesto

PRE, reproducido contra core `a50a33e` y nodo `6c64646` antes de editar. Una fuente que devuelve B cuando se pide A insertaba una fila con performance 0,2. Mutar el diccionario a 999 después de verificar insertaba 999. SQLite en memoria. No es corrupción desplegada. `project_metrics` ya verificaba el envelope; esa ruta se conservó.

| Repo | Rama | Tip | Padre |
| --- | --- | --- | --- |
| doin-node | `satoshi/offchain-shadow-20260930` | `a4f65ba55ace09f536fc907d35041fcd7df13c3e` | `6c64646` |
| doin-core | `satoshi/offchain-shadow-20260930` | `a50a33e064fc4b9a1062bcaf342607aea0673553` | sin commit nuevo |

El remoto del nodo devuelve `a4f65ba`. El core no necesitaba otro commit: la proyección vuelve a parsear cuerpo, manifiesto y secciones. El digest pedido es el SHA-256 de esos bytes. Un atributo escrito a mano no cuenta. Los diccionarios de `records` no se leen para insertar.

POST medido por el integrador, misma sonda desechable, después del commit:

| Caso | Resultado |
| --- | --- |
| Se pide A y la fuente devuelve B | `DIGEST_MISMATCH`, 0 filas |
| El diccionario queda en 999 y los bytes siguen en 0,2 | 1 fila, performance 0,2 |

`chain_verified` sigue en falso. El directorio local sigue `DISPOSABLE_FILE_NOT_LAKE`.

El integrador reejecutó `tests/test_offchain_shadow.py`: **21 passed**, 0,45 s, y `tests/test_block_body_archive.py`: **16 passed**, 0,12 s. `crispdm-run -m 2G -t 120s`. Sigue el aviso `asyncio_mode`. Esas 21 incluyen referencia correcta, referencia equivocada con el atributo falsificado, mutación del diccionario, archivo ausente, reintento, rollback sin filas, MAE con huecos `NOT_COMPARABLE`, y el conteo: 1001 filas comprometidas mientras `get_rounds` devuelve 1000. No reejecuté el resto de las suites del repositorio.

Postgres, el lago desplegado, la cadena y los checkouts vivos no se modificaron.

## Barrido sintético: 44 celdas corridas

heuristic-strategy, `satoshi/strategy-support-20260930`, `7a50f620d69014aa24eede19aa1b60f0abc661df`, encima de `e7966f33`. El remoto coincide. `781022a` y las evidencias de microensayo y contabilidad no se reescribieron. `app/factorial_harness.py` sigue siendo el manifiesto que no corre. El ejecutor nuevo es `app/sweep_executor.py`.

SYNTHETIC. El plugin real y la contabilidad reconciliada: capital 10 000, fracción 0,05 de la equity corriente, apalancamiento 100, comisión por lado `abs(tamaño) * 0,00007 * precio`, swap por timestamps UTC, `shortcash` falso. No es utilidad financiera. No ordena el corto frente al largo. Una persistencia en el largo que deja el libro sin entradas en este fixture no dice que el corto carezca de utilidad.

Protocolo declarado antes de medir. Dos orientaciones. El contraste principal fija la otra familia en persistencia real. El ideal es un control adicional. Persistencia e ideal no tienen réplicas. El ruido usa semillas pareadas 42, 43 y 44 e intensidades 0,5, 1,0 y 2,0. Esas tres intensidades no son las 21 razones históricas. Doce horizontes, 1–6 y 24, 48, 72, 96, 120, 144. Soporte común: 2019-05-01 00:00, 04:00, 08:00 y 09:00 UTC. Naive pareado, macro 0,009208333333333327 en las 44.

La escala es la media absoluta del residual en DEV: 120 orígenes, 264 marcas, de 2019-04-20 00:00 UTC a 2019-04-30 23:00 UTC, antes del primer origen puntuado y antes de 2019-05-16. El ruido entra solo en la predicción de la familia variable. Sigma es el objetivo dividido por 0,7978845608028654. El sorteo mezcla semilla, origen, familia y horizonte. No se divide el vector por su MAE muestral. El MAE logrado se informa aunque no coincida con el objetivo.

44 declaradas, 44 corridas, 0 solo declaradas. La sonda `short_var_long_fix_persistence_noise_i1p0_s42` midió, en la corrida que escribió la evidencia, pared 0,1404030870180577 s y CPU 0,14040417000000005 s. Por 44 cabe en el presupuesto interno de 80 s. La suma de los 44 relojes de esa corrida es pared 4,370826597994892 s. El integrador reejecutó el pytest: **6 passed**, 6,26 s, pared 9,07 s, usuario 6,66 s, sistema 0,46 s. Las cifras de trades, PnL, caja, equity, MAE y naive coinciden con `RESULTS.json`. Los relojes de esa segunda pasada no se commitearon.

En la sonda, intensidad 1 y semilla 42, el MAE de los doce horizontes es 0,012081887754735392. El de los seis con objetivo es 0,016372108842804132 frente a un objetivo de 0,015700136309124396. No coinciden. En el largo fijo el objetivo es nulo y el logrado es el naive.

Cuando el largo queda en ideal, el ruido solo en el corto no cambió los tres cierres de este fixture: PnL 2659,4364862832954, caja 12659,436486283295. El MAE de esas celdas sí cambia con la intensidad y la semilla. La entrada lee el largo. Cuando el largo es persistencia y el corto es persistencia o ideal, no hay entradas: el mismo hecho de la contabilidad anterior. El largo ruidoso sí cambia el libro. Varias de esas celdas quedan abiertas, con 49999 unidades pendientes, caja 9464,836484317724 y equity 10164,07999916772. `stop()` no les añade un fill. Tres libros distintos también quedaron medidos: pendiente −56328 con PnL 1490,9485137666582; pendiente 54952 con PnL 990,5768547333245; y un cierre plano con PnL −510,53812236667136. El primer recorte, cuando hay entrada, pide 50000 y admite 49999 a 1,000015. El hueco de equity más grande fue 1,8189894035458565e-12.

La tabla de las 44 está en `RETSU_STRATEGY_SWEEP_2026_09_30.md` del repositorio de la estrategia. B0 sigue `NOT_STARTED`. Las 21 razones y las 241 celdas retenidas no corrieron. La ventana desde 2019-05-16 no se usó. El techo histórico de 3600 s CPU no se usó como presupuesto.

## Lo que esta orden deja igual

Ninguna exactitud de clasificación, calibración o reparación GPU se declara. Los tips `8f4b09b5` y `da93d1a0` siguen siendo preparación, no una medición de esas tareas.
