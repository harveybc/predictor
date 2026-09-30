# Acuse: despacho del programa, cuatro frentes en ejecución

Retsu, coordinador. 2026-09-30.
Orden publicada: `RETSU_PROGRAM_PARALLEL_DISPATCH_2026_09_30.md`.
Este archivo es el acuse de recepción y de despacho. No es un retorno de resultados.
Los retornos `f5d8e277` y `d845563e` quedan como estaban.

Cuatro agentes de implementación recibieron acuse de la herramienta de despacho.
Son concurrencia de desarrollo en CPU. Cada prueba queda bajo `crispdm-run -m 2G -t 120s`.
No hay experimento de GPU en este despacho. Al despachar, el coordinador tenía memoria libre suficiente para admisiones de 2 GiB. El acelerador visible en el coordinador queda fuera de destino.

El calendario, el consumidor M5PHET y el frente 6 no están en ejecución. El motivo de cada uno va en la tabla.

| Prioridad | Tarea | Agente con acuse | Commit de partida / worktree | Recurso | Primer entregable | Dependencia |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Ejecutor sintético con predicciones horarias continuas y causas de cierre. Mismo plugin. B0 queda sin arrancar. | `01a0f3c4-4478-7523-a618-713a28d99a05` | heuristic-strategy `7a50f620` en `satoshi/strategy-support-20260930`, worktree `strategy-support-20260930` | CPU, 2 GiB / 120 s por invocación | Piloto sintético horario, o el ejecutor probado y su coste si la grilla ampliada no cabe | Orden `bace5fa8`, sección A de `RETSU_D845_REVIEW_AND_NEXT_2026_09_30.md`. Independiente de DOIN y de TensorFlow |
| 2 | Consumidor por filas de representaciones ya materializadas. Igualdad de filas y de valores. Rechazo de identidad. Alcance al menos 5 | `01a0f3c4-4478-7523-a618-714211b84786` | predictor `c2d4388b`. Worktree nuevo `predictor-hcore-consumer-20260930`, rama `retsu/hcore-row-consumer-20260930` | CPU, 2 GiB / 120 s | Consumidor y pruebas de igualdad y de rechazo | Artefacto de `c2d4388b`. El prefijo no se regenera ni se reentrena. Si faltan los bytes, fixture declarado y la falta nombrada |
| 3 | Escritura durable, referencia, lectura verificada y proyección por adaptador de servicio desechable | `01a0f3c4-4478-7523-a618-715fbb3b115c` | doin-node `a4f65ba`, doin-core `a50a33e`, data-lake `e82b33e`. Worktrees `doin-node-offchain-shadow-20260930`, `doin-core-offchain-shadow-20260930`, `data-lake-offchain-byte-store-20260930` | CPU, 2 GiB / 120 s. SQLite desechable | El siguiente AT pendiente que empiece por escritura durable y lectura por hash | Continuación de `a4f65ba`. Cadena y servicios productivos quedan fuera |
| 4 | Contrato de calidad versionado con identidad del checkpoint. Pruebas de identidad ausente y de identidades mezcladas | `01a0f3c4-4478-7523-a618-716973eabdfb` | news-signal `b08a99f`. Worktree nuevo `news-signal-quality-identity-20260930`, rama `retsu/quality-checkpoint-identity-20260930` | CPU, 2 GiB / 120 s | Contrato versionado y esas pruebas | `quality_eval.py` y el contrato de proveedor en `b08a99f`. Pesos sin cargar. F1 nuevo sin producir |
| 5 | API pura y CLI offline de admisión de estudio | slot libre, todavía sin agente | data-gov `7eec868`. El checkout vivo tiene `docs/00_CONTRATO.md` modificado; el agente usará un worktree nuevo | CPU, cuando ocupe un hueco | API y CLI sobre un snapshot | `tools/register_calendar_resources.py` y `docs/08_RESOURCE_REGISTRY.md`. El registro vivo no se cambia |
| 6 | Disposición externa de M4 y justificación del donante H-CORE. Conciliación de Weather y Traffic | Musashi | Evidencia ya publicada: Weather `f328db3a`, Traffic `91a4c410` | Fuera de este despacho | Fuera de este despacho | Retsu no audita su propio frente |

## Huecos y su motivo

El calendario espera el primer hueco que deje un cierre o un bloqueo de los cuatro. El tope inicial de este despacho es cuatro agentes de implementación. La tarea ya está especificada y el commit `7eec868` existe.

El consumidor M5PHET espera el contrato del productor de calidad. Va en otro worktree y tiene que demostrar que el informe y la respuesta citan el mismo checkpoint. Los reportes históricos que no guardaron identidad siguen sin una identidad nueva.

El frente 6 permanece con Musashi.

## Lo que este acuse no ejecuta

Siguen `NOT_RUN`, y esta orden no los concede: diagnóstico de TensorFlow, calibraciones de 4 800 s CPU / 3 600 s de pared, código remoto de clasificación, B0, M4 confirmatorio y operaciones de broker.

Tampoco se relanzan el contraste ya cerrado ni las entregas Weather `f328db3a` y Traffic `91a4c410`.

No hay medición científica nueva en este archivo. No hay experimento sintético nuevo. No hay reparación nueva. Los cuatro frentes están despachados; sus commits de resultado llegarán uno a uno, sin esperar un retorno consolidado.
