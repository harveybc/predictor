# Consumidor por filas del prefijo ya materializado

Fecha: 2026-09-30.
Base: `c2d4388b3dd7bdad1cbf03911c7a4be4009494c5`.
Rama: `retsu/hcore-row-consumer-20260930`.

## Alcance

Busqué una oración que defina «alcance» en los documentos de DR, de H-CORE y del programa, en los worktrees de solo lectura `predictor-dr056-20260926` y `predictor-classification-plan-20260928`.

No hay tal oración. El término aparece como el ámbito de un encargo. En H-CORE la frase más cercana es una instrucción, no una definición: «Calcular el alcance rama+nucleo y verificarlo» (`docs/tres_temas_entrevista/CORE_PRETRAINING_HYPOTHESIS_2026_09_18.md`). El plan maestro remite a esa sección y tampoco lo define. El artefacto DR05 usa la palabra inglesa reach para el campo medido; no es una definición de alcance. No convertí ese reach en un protocolo.

alcance: NOT_DEFINED_IN_REPO

Casos de fila implementados, cinco: coincidencia exacta, desajuste de valores, desajuste del número de filas, identidad incorrecta y artefacto ausente. Con los bytes reales se rechazan además una fila mutada, un valor mutado, un digest incorrecto y una versión o identidad incorrectas.

## Bytes

Los archivos nombrados por `docs/audits/evidence/DR05_DR06_20260926/PREFIX_OUTPUT_MATERIALIZATION.json` están presentes en el almacén que ese registro ya nombra. El sha256 medido coincide con el registrado. Esto es anterior a la paridad, y no depende del fixture.

| archivo | filas | sha256 |
|---|---:|---|
| `prefix_output_train.npy` | 40080 | `8a24071667bfedbeaf882b2869f3482b289e42f06ff3d3adb8d531745c382522` |
| `prefix_output_validation.npy` | 10020 | `0e6dd5bb871553621fa1f5b12343dfce7fff6425c5bf3d5d203defc894e17432` |
| `origins_train.npy` | 40080 | `6f76b30bbc4cef3a40df066bf3e7453bb203ffd084ce85c91bbfcc3f843c504a` |
| `origins_validation.npy` | 10020 | `ee2e01ce8793b059727bf468159c3b19bc79d9eeeabff59b447c43cecbcfbd1f` |
| `MANIFEST.json` | — | `589a9e935f2aaa62db8da95b1d0b97bd6c1aa742673cdfd618f6ae6f379c9276` |

La versión ligada es `dr05.prefix.fd0fde9b2a913257`. Los digest de identidad del registro, también coincidentes entre el manifiesto y el artefacto, son data `70485ac9d1a41ac86d0910d23928a15c1aa88737c6261343e4c7dc248e30c194`, design `143abb57d97daa07e3f5228eadf4e3f1a0deb5fe30eb95ca065663f76ff888a7`, panel `b3192c0bcb117b2ee120a906dbcfb9550cd907abff74fea9bc2b1aa320ebc8db` y donor_weights `c374076c49c9631803bd16f85b4a577a67466a38e60371040af7d4ebe4ae3845`.

Paridad contra esos bytes: el conteo de filas y el tramo de orígenes igualan al registro; el sha256 del archivo cubre cada byte; la primera y la última ventana igualan el digest que ya registró la recarga en proceso fresco (helper `sha_array` de DR05); la primera ventana, leída dos veces, es igual byte a byte. Elementos de extremo: 2880 en train y 2880 en validation. No se copió el tensor entero a un segundo arreglo.

El fixture `tests/fixtures/hcore_row_consumer/` es sintético, versión `synthetic.hcore.not-the-pinned-prefix`. Que pase no es paridad con el prefijo anclado. El registro anclado frente a ese fixture se rechaza con `WRONG_VERSION`.

## Rechazos

`EXACT_MATCH`, `VALUE_MISMATCH`, `ROW_COUNT_MISMATCH`, `WRONG_IDENTITY`, `MISSING_ARTIFACT`, `MUTATED_ROW`, `MUTATED_VALUE`, `WRONG_DIGEST`, `WRONG_VERSION`.

Sobre una copia en memoria de una fila real: valor mutado, fila mutada, desajuste de valores y desajuste de conteo. Sobre el registro, sin escribir el almacén: digest, versión e identidad de los pesos del donante. Un directorio que no está: `MISSING_ARTIFACT`. El almacén real no se modificó.

## Qué no hace

No regenera el prefijo, no reentrena, no usa GPU ni TensorFlow, no abre la reserva, no ejecuta una comparación H-CORE y no firma una justificación del donante.

## Prueba

`CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -n hcore-row-consumer -- python -m pytest tests/test_hcore_row_consumer.py -q -s`

12 passed in 0.24s.
