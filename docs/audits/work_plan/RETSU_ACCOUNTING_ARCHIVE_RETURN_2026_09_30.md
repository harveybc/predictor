# Retorno: contabilidad, archivo verificado y preparación

Retsu, dueño de la integración. 2026-09-30.
Orden: `RETSU_ACCOUNTING_ARCHIVE_CONTINUATION_2026_09_30.md`.
Dictamen: `MUSASHI_RETSU_D158_REVIEW_2026_09_30.md`.
El retorno anterior, `RETSU_PARALLEL_RETURN_2026_09_30.md`, queda como estaba.

La pregunta sobre el diagnóstico de 300 s y las calibraciones de 4 800 s CPU / 3 600 s de pared no concede esas piezas. Siguen sin ejecutarse. El código remoto de clasificación sigue fuera. El sobre de 12 GiB no se redujo y no se pidieron 18 GiB. Estas correcciones no son una precisión científica nueva.

Despachos: estrategia `01a0f1cb-5aa5-7f31-8756-d9f3ecf544d1`, DOIN `01a0f1cb-5aa5-7f31-8756-da04189d66bd`, GPU y clasificación `01a0f1cb-5aa5-7f31-8756-da113bb16165`.

## Microensayo sintético reconciliado

heuristic-strategy, `satoshi/strategy-support-20260930`, `e7966f33fca9946d6b2b9e45e28bc5f8ba20f6d8`, encima de `781022a`. El remoto devuelve ese mismo tip. La evidencia `STRATEGY_MICRO_20260930` no se reescribió: caja fixture 2 000 000, tamaños de 1e6 y los cuatro PnL de ese commit siguen siendo el diagnóstico legacy.

El sucesor usa el mismo plugin, variante E, TP 0,9, SL 2,0, decisión al cierre y fill en la apertura siguiente. Capital 10 000, fracción de margen 0,05 sobre la equity corriente, apalancamiento 100. La moneda de cuenta es la cotizada. El nocional es unidades por precio y el margen es ese nocional dividido por 100. Quinientos de margen permiten 50 000 de nocional antes de costos: 50 000 unidades a precio 1 y 25 000 a precio 2. El mínimo de volumen no eleva el margen. Si el tamaño permitido queda bajo el mínimo, la orden es Margin y no queda posición.

`CommInfo.margin` no es esa fracción. Con `commtype` vacío, un valor verdadero convierte la comisión en un monto fijo por unidad. El sucesor deja `margin=None`, fija `commtype=COMM_PERC`, `percabs=True`, `stocklike=True`, `leverage=100` y `shortcash=False`. El largo y el corto depositan colateral. El efectivo que un corto recibiría por el nocional no se acredita y no es beneficio. Al cierre se libera el colateral.

Un solo libro registra spread, slippage, comisión y swap. Spread 2 pips y slippage 1 pip, pip 0,00001, van dentro del precio del fill y no se restan otra vez. La comisión es por lado: `abs(tamaño) * 0,00007 * precio`. El 0,00007 sigue siendo `commission_per_lot / 100000`. No es una tarifa fija de 7 ni un round-trip de 7. El swap es 10 por lote de 100 000 unidades por 24 horas de timestamp UTC, en las dos direcciones, debitado antes de la decisión que lee el saldo. Un hueco de barras cobra las horas del reloj. `duration_bars` no es ese reloj.

PRE, reproducido antes de tocar el código: en el legacy plano, caja menos PnL reportado era 33,33333333369228 frente a swap 33,33333333333337 en ideal/ideal, y 37,5000000003929 frente a 37,5 en persistence/ideal. El swap se restaba del PnL reportado y no de la caja.

El integrador reejecutó los tres pytest bajo `crispdm-run -m 2G -t 120s`: **41 passed**, 2,53 s de pytest, pared 5,36 s, usuario 3,02 s, sistema 0,36 s. CPU. Sin mercado. Mismo soporte, orígenes 2019-05-01 00:00, 04:00, 08:00 y 09:00 UTC. MAE macro: media sin ponderar de los doce horizontes. Naive pareado, macro 0,009208333333333327, igual en los cuatro. SYNTHETIC. No es utilidad financiera.

| Brazo (corto / largo) | Cierres | PnL realizado | Caja = equity | Pendiente | MAE macro | MAE corto | MAE largo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ideal/ideal | 3 | 2659,4364862832954 | 12659,436486283295 | 0 | 0 | 0 | 0 |
| persistencia/ideal | 3 | 1965,8373773999754 | 11965,837377399974 | 0 | 0,005312500000000002 | 0,010625000000000004 | 0 |
| ideal/persistencia | 0 | 0 | 10000 | 0 | 0,0038958333333333254 | 0 | 0,007791666666666651 |
| persistencia/persistencia | 0 | 0 | 10000 | 0 | 0,009208333333333327 | 0,010625000000000004 | 0,007791666666666651 |

Hueco de equity 0 en los cuatro. Hueco de caja en plano: 0, salvo persistencia/ideal con −1,5916157281026244e-12. El PnL realizado resta comisión y swap una sola vez.

Fills frente al legacy: mismos relojes, mismo lado y mismo precio. Cambia el tamaño. ideal/ideal: 49999 a 1,000015, −49999 a 1,029985, −56328 a 1,019985, 56328 a 1,000015, 63035 a 1,000015, −63035 a 1,000985. El primer fill pide 50 000 y el slippage a 1,000015 lo recorta a 49 999, con margen 499,99749985 frente a un presupuesto de 500. Los tamaños siguientes suben porque el presupuesto es el 5 % de la equity corriente: 56 328 y 63 035. A las 09:00 los dos brazos con trades están largos 63 035. La mezcla `0,6 * 0,99 + 0,4 * 1,003` sigue bajo el stop 0,9998 y la familia corta pide el cierre en ideal/ideal. En persistencia/ideal esa hora no pide nada y el stop cierra a las 10:00; el fill de salida es a las 11:00 a 0,989985, tamaño −63035. Los dos brazos planos siguen en 0 fills. MAE igual no es la misma decisión. Cuatro puntos no ordenan el corto frente al largo.

El primer trade cerrado de ideal/ideal acumula comisión 7,10485790000007 en los dos lados y swap 0,41665833333333335, que es 2/24 de un lote parcial de 49 999 entre las 01:00 y las 03:00. `duration_bars` de ese tramo sigue en 3.

Sonda con corte a las 01:00 y capital 10 000: fill 1,000015, tamaño 49 999, cierres 0, realizado 0, pendiente 49 999, caja 9496,50251765105, equity 10045,74903250104, hueco de equity 0. `stop()` llama a `close()` y no fabrica un fill. La observación legacy a caja 10 000, un corto y PnL 396,32, no se repitió subiendo el capital.

Estados vistos en los brazos con trades: Accepted, Submitted, Completed e in_flight. Margin, Rejected y Cancelled se ejercieron en fixtures del mismo pytest: un rechazo deja la posición en cero y una cancelación no completa.

`app/factorial_harness.py` deja 18 celdas listas: corto variable con largo ideal fijo y al revés; persistencia, ruido de MAE equivalente e ideal. Semillas 42, 43 y 44. Escalas solo en DEV. `execute_manifest` levanta `B0_NOT_STARTED`. El manifiesto no se ejecutó. Holdout y presupuesto B0 siguen fuera. El techo histórico de 3 600 s CPU de las 241 celdas no es presupuesto de B0.

## Archivo: el ETL consume bytes verificados

La sonda PRE, corrida antes de editar, insertaba performance 999 junto al digest original del manifiesto, perdía el MAE, devolvía el `round_id` viejo a otro experimento y rechazaba un segundo manifiesto del mismo cuerpo con CONFLICT. Esas cifras están en `docs/audits/offchain_shadow_pre_20260930.json` del nodo. Los huecos de caja de esa sonda son del carril de estrategia.

| Repo | Rama | Tip publicado | Padre |
| --- | --- | --- | --- |
| doin-core | `satoshi/offchain-shadow-20260930` | `a50a33e064fc4b9a1062bcaf342607aea0673553` | `dae92be` |
| doin-node | `satoshi/offchain-shadow-20260930` | `6c646465158b0d7596409105d259d9ec91efa1ef` | `212c263` |
| data-lake | `satoshi/offchain-byte-store-20260930` | `e82b33e3ad3103b1b2a420327e4fac02e611e40a` | `7ac4ace` |

El remoto devuelve esos tres tips. El checkout vivo del lago no se editó. Consenso, poda, Postgres y el servicio no se tocaron.

POST, medido por el integrador bajo `crispdm-run -m 2G -t 120s`: doin-core **16 passed**, 0,12 s; doin-node **14 passed**, 0,30 s; descubrimiento más byte store **14 passed**, 0,02 s. Sigue el aviso preexistente `asyncio_mode`. Una sección con performance 999 y los digests originales lanza `DIGEST_MISMATCH` antes de insertar. `metrics={'MAE': 0.1}` permanece, con sus parámetros. Escala, población, reducción y lo no declarado quedan `NOT_COMPARABLE`. El mismo contenido devuelve el registro anterior. Otro experimento, otro intento, otra performance, otros parámetros o otra métrica con el mismo id lanzan `CONFLICT`. Un fallo a mitad de la proyección hace rollback. Dos manifiestos del mismo bloque tienen el mismo `body_digest` y distinto `manifest_digest`: el segundo se guarda al lado y el primero sigue legible. Una contradicción del índice no se convierte en reintento. Dos escritores concurrentes dejan un solo registro verificado. Un archivo truncado recupera con `verified: false` y no se borra. `chain_verified` sigue en falso. El directorio local sigue etiquetado `DISPOSABLE_FILE_NOT_LAKE`.

`GovernedByteStore` escribe y lee bytes por SHA-256 en SQLite de memoria o en archivo temporal, exige un grant y rechaza `HASH_MISMATCH` y `POSTGRES_REFUSED`. `write_metrics` sigue guardando un informe, no bytes. El registro de data-gov no autoriza esta entrega. `deployed_lake` queda en falso. No hay ruta HTTP nueva. Esto no es el lago desplegado.

Cobertura parcial. Hecho en el archivo desechable: reconstrucción desde bytes, sección forjada, inventario, tres digests, MAE y calificadores, idempotencia, rollback, sucesor de manifiesto, dos escritores, truncado, y el byte store con grant. No cubre AT01, AT04, AT05, AT06, AT09 ni AT10. O02 y O06 no tienen cifra. No es un prerequisito nuevo para experimentos que ya pueden correr.

## GPU: procedimiento escrito, diagnóstico sin correr

predictor `satoshi/gpu-env-repair-20260930`, `da93d1a0163d25bb6210c165c74d479b2b4075e8`, encima de `c5aa8dd4`. El remoto coincide. El integrador reejecutó el test del pin: **1 passed**, 0,01 s. No importa TensorFlow.

El pin de preparación es `tensorflow==2.21.0`, `nvidia-cuda-runtime-cu12==12.5.82`, `nvidia-cudnn-cu12==9.3.0.75`, más los hermanos `nvidia-*-cu12` del mismo piso. 44 ruedas, 2 576 641 397 bytes, rehasheadas del prefijo del coordinador, sin nueva descarga. Ese prefijo tiene `libcudart.so.12` y no tiene `libcudart.so.13`. El texto del wheel declara CUDA 12.5.1, cuDNN 9 y `sm_60`, `sm_70`, `sm_80`, `sm_89`, `compute_90`. No se copió `.repair-env`. No se creó venv. El procedimiento queda `WRITTEN_NOT_RUN`. El intérprete relativo, si después hay concesión, es `.worker-diag-env/bin/python`. El diagnóstico no va envuelto en `CUDA_VISIBLE_DEVICES` vacío.

Ocupación releída por el integrador, solo UUID, sin colocar un proceso:

| UUID | Papel | MiB usados / total | Utilización | Compute |
| --- | --- | ---: | ---: | --- |
| `GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8` | preferente | 10 / 32607 | 0 % | 12,0 |
| `GPU-b77fc3ad-db77-b648-dc15-ec79b65e2519` | no es destino | 14 / 12227 | 18 % | 12,0 |
| `GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9` | rechazo retenido | 14 / 16376 | 6 % | 8,9 |
| `GPU-612d1e0c-33de-d5cc-56eb-06c0ae424326` | excluida | 638 / 8188 | 34 % | 8,9 |

La lectura del carril tenía 7 % en la tarjeta del rechazo y 647 MiB / 29 % en la excluida, cuyo compute no había consultado. La preferente mide 12,0, fuera de la lista del wheel, y el pin no se instaló ahí. La 8,9 del rechazo está dentro de la lista y tampoco recibió venv. El choque del pin CUDA 12.5 con sonames CUDA 13 sigue `NOT_DEMONSTRATED`. Soname ausente, driver y CUDA/cuDNN del build que rechazó siguen `NOT_RETAINED`.

El comando de diagnóstico, no lanzado, usa `crispdm-run -m 6G -t 300s`, el intérprete relativo, el UUID del rechazo y `--verify`. Una admisión fresca. Las calibraciones siguen en 4 800 s CPU y 3 600 s de pared sumados, solo si ese diagnóstico pasa. Esta nota no las concede.

## Clasificación: bytes pinados, ensayo sin correr

predictor `satoshi/banking77-closure-review-20260930`, `8f4b09b5a692df3f5985d6f8a76d98410af2516a`, encima de `e1d04ad9`. El remoto coincide. El integrador reejecutó **21 passed**, 1,44 s, bajo `crispdm-run -m 2G -t 120s`. No redescargó. Volvió a hashear el árbol gitignored.

Modelo `jinaai/jina-embeddings-v5-text-small`, revisión `46ed7da5b47e4bca710b756313fafaf4110c6bd1`: 1 369 721 378 bytes del hub, MEASURED. `model.safetensors` 1 192 133 208 bytes, sha256 `045fa75ff963a528cda2589fb1ca0a9ad848b53511780ed4f08f6fe10f6167c3`. Los tres Python se guardaron como bytes y no se importaron. Eso no certifica su seguridad. Licencia CC BY-NC 4.0, solo investigación.

Corpus `mteb/banking77`, revisión `0fd18e25b25c072e09e0d92ab615fda904d66300`, JSON lines, sin ejecutar `prepare_data.py`. Train 10 003 filas, 77 etiquetas, soporte 35–187, 1 245 265 bytes, sha256 `d411780d8c0e18e166f5664c6cfe90dc9de399d722aa7cde282e31a771323ea7`. Test 3 080 filas, 77 etiquetas, soporte 40, 365 101 bytes, sha256 `fb1b0043ded745b8767687084786e6dd0a5f0ce03243b6131992a1c7ae2c2595`. No hay split dev. Corpus del hub: 1 612 800 bytes. Juntos, 1 371 334 178 bytes en 80,856321 s, según el registro de adquisición. Dos archivos locales `REVISION` suman 82 bytes y no están en esa cuenta del hub.

La herramienta de bajada solo trae bytes. La de ejecución sigue sin descargar y sin cargar el modelo, y sale si faltan las variables del dueño. Esas variables no se definieron. `trust_remote_code` no se concedió. No hay score.

Piloto, escrito y sin ejecutar. No hay dev en el corpus. En orden de `train.jsonl`, las dos primeras apariciones de cada etiqueta son la sonda y las dos siguientes el dev: 154 + 154 filas. El test de 3 080 no entra. Después se conserva el protocolo de referencia ya pinado. Un timeout a mitad del corpus no es un score. mteb 2.9.0 no tiene tope de filas, así que el piloto no es una invocación más corta de la tarea oficial.

CPU o GPU queda `PROPOSAL / UNDECIDED`. No hay tiempo de inferencia. Los 80,86 s son transferencia. El camino GPU espera la concesión del diagnóstico. La tarjeta preferente mide compute 12,0, fuera de la lista del wheel de TensorFlow 2.21, y ese pin no es el stack de esta solicitud: `torch==2.14.0` sigue dependiendo de librerías NVIDIA de CUDA 13, que no se instalaron. El camino CPU no necesita la concesión GPU y tampoco se cronometró, así que no queda elegido.

## Lo que esta orden deja igual

B0 sigue `NOT_STARTED`. El barrido de 241 celdas no corrió. Ningún broker real, servicio, VM, intérprete compartido ni base poblada se modificó. Los caches `.repair-env` y `.closure-env` siguen en disco. El código remoto, el diagnóstico GPU y las calibraciones siguen sin concesión.
