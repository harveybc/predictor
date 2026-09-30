# Retorno de los cuatro carriles

Retsu, dueño de la integración. 2026-09-30.
Orden: RETSU_PARALLEL_CONTINUATION_2026_09_30.md, después de
MUSASHI_RETSU_A2DFB150_REVIEW_2026_09_30.md.
El diagnóstico GPU, las dos calibraciones y la ejecución de código remoto
siguen sin concesión. No se ejecutaron. No bloquean lo que sigue.

El despacho, con los cuatro trabajadores, está en
`RETSU_PARALLEL_DISPATCH_2026_09_30.md`.

## Microensayo sintético — lo único con comportamiento medido

Repositorio heuristic-strategy, `781022a4702244d866cf86bb69343673340e9eff`,
encima de `1b25d31`. El integrador reejecutó
`tests/unit_tests/test_strategy_micro_20260930.py` y
`tests/unit_tests/test_strategy_support_20260930.py` bajo
`crispdm-run -m 2G -t 120s`: **28 passed**, 1,40 s. CPU. Sin mercado.

PRE, antes de reparar F2 y F3: 16 fallos y 1 paso. El paso era el hueco de
mercado, que no es un valor no finito. POST de esos 17 contratos: todos pasan.
La admisión de calibración mira orígenes, objetivos, escalas y residuos.
Mutar una barra reservada no cambia un parámetro ajustado en DEV. Horizontes
bool, fraccionarios, NaN o duplicados se rechazan. La zona del fixture es UTC
declarada.

El ensayo usa el plugin real `plugin_long_short_predictions`, variante E,
decisión al cierre y orden de mercado en la apertura siguiente. Doce horizontes
de horas transcurridas, 1–6 y 24–144. Cuatro orígenes sintéticos el
2019-05-01 00:00, 04:00, 08:00 y 09:00 UTC. Costos del plugin: spread 2 pips,
slippage 1 pip, pip 0,00001, comisión 0,00007, swap 10 por lote y día,
TP 0,9, SL 2,0.

El efectivo declarado es **2.000.000**. Es del fixture. El bróker simulado
exige el nocional completo porque el plugin no fija margen, y con 10.000 el
largo pedido no llena. Esa observación también quedó medida: a 10.000 hubo
1 trade cerrado, PnL realizado 396,32, y el largo de las 00:00 no cerró.
No es un quinto brazo. TP y SL no se cambiaron.

Todo lo de abajo es **SYNTHETIC**. No es utilidad financiera. La agregación
es la media sin ponderar de las doce MAE por horizonte. El naive, en las
mismas filas, es el cierre del origen contra el cierre futuro. Su macro es
0,009208333333333327 en los cuatro brazos.

| Brazo (corto / largo) | Trades cerrados | PnL realizado | Equity marcada | Caja | Unidades pendientes | MAE macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ideal/ideal | 3 | 50453,09666666601 | 2050486,43 | 2050486,43 | 0 | 0 |
| persistencia/ideal | 3 | 39449,69999999956 | 2039487,20 | 2039487,20 | 0 | 0,0053125 |
| ideal/persistencia | 0 | 0 | 2000000 | 2000000 | 0 | 0,0038958333333333254 |
| persistencia/persistencia | 0 | 0 | 2000000 | 2000000 | 0 | 0,009208333333333327 |

La MAE ideal es 0 en cada horizonte. La de persistencia/persistencia coincide
con el naive. Exposición final de los cuatro brazos: 0. El libro está plano,
así que la equity marcada coincide con la caja.

Traza de la variante E, no una separación supuesta. En ideal/ideal, a las
09:00, el largo está abierto con stop 0,9998 y el cierre 1,002 está entre el
stop y el TP 1,009. El mínimo largo es 1,003 y no cruza el stop. La mezcla
`0,6 * 0,99 + 0,4 * 1,003 = 0,9952` sí lo cruza, y el plugin pide cerrar.
Esa salida la dispara la familia corta. En persistencia/ideal la misma barra
sigue abierta: la familia corta es el cierre 1,002, la mezcla es 1,0024, y
el cierre llega a las 10:00 por el stop. Las dos familias de persistencia en
el largo no abren. La familia larga fija la entrada y el TP/SL. La familia
corta cambia la salida.

`stop()` llama a `close()` con una posición de 1.000.000 abierta en un feed
cortado a las 01:00. Después de `stop()` la posición sigue en 1.000.000 y los
trades cerrados siguen en 0. `close()` dentro de `stop()` no obtuvo un fill.
PnL realizado 0. Unidades pendientes 1.000.000. Equity marcada 2000914,99895.
Caja 999914,99895. No se borró ningún trade que cruzara el corte. No hay
liquidación forzada. La convención terminal del sucesor B0, no corrido, es:
las posiciones pendientes siguen abiertas.

B0 sigue **NOT_STARTED**. El saldo histórico de 3.600 s de CPU no se usó
como presupuesto. Igual MAE no implica iguales decisiones. No hubo barrido.

## GPU — el lanzador, no el diagnóstico

`c5aa8dd4d800984b3f6864d7a108a33585115ab2`, encima de `3f910fbf`.
El integrador reejecutó la fase POST de
`docs/audits/gpu_launcher_probe_20260930.py` bajo `crispdm-run -m 2G -t 120s`
con `CUDA_VISIBLE_DEVICES` vacío. `ok` es true. No se importó TensorFlow y
no se mapeó CUDA.

PRE, biblioteca C inocua: asignar `LD_LIBRARY_PATH` en el mismo proceso
devuelve rc 1; un proceso nuevo arrancado con ese directorio devuelve 42.
POST: la entrada pública arranca un hijo que devuelve 42, y el supervisor no
carga la biblioteca. La ruta integrada (`governed_run` y `supervise`) ya
entregaba el entorno al hijo y sigue haciéndolo. Si falta la dependencia
transitiva, el hijo devuelve 1 y el supervisor devuelve 3, `REFUSED`.

Narración corregida: el entorno aislado **sí se instaló** (2.576.641.397 bytes
de ruedas y 5.135.683.136 en `site-packages`). No se reparó en el obrero, no
se desplegó y no se probó en GPU. El pin CUDA 12.5 no se presenta como válido
para todas las arquitecturas.

El diagnóstico queda escrito y **NOT_RUN**: una admisión, 300 s de CPU,
300 s de pared, 6 GiB de RAM de anfitrión, sin entrenamiento. Es un límite
de gasto, no una huella medida. El intérprete sería el `python3.12` del
prefijo `.repair-env` de este worktree, en el coordinador. No está probado
en el obrero y no se transporta. Las calibraciones de 4.800 / 3.600 y la
envolvente de 12 GiB no se tocaron. No se pidieron 18 GiB.

## Clasificación — una solicitud, no una corrida

`e1d04ad93e4c74440f0b6c191a720ef0d2d1a395`, encima de `b5e540ea`.
El integrador reejecutó `tests/test_b77_executable_request_20260930.py`:
**11 passed**, 0,04 s.

Ningún metadato de las 87 versiones exige Python 3.13. La lista para el
CPython 3.12 existente cierra en las mismas 87 versiones. Los binarios cp313
no se instalaron encima de 3.12. 65 ruedas universales rehasheadas suman
2.284.173.652 bytes (MEASURED). 22 reemplazos cp312 no descargados suman
946.243.017 bytes (INDEX_METADATA). Los pesos, 1.369.721.378 bytes, están
**NOT_PRESENT**.

La prueba negativa borra un fichero de un snapshot sintético local: rechazo,
0 bytes descargados, sin red. `trust_remote_code` sigue sin concesión.
Ausencia de `subprocess` o `eval` en los tres ficheros remotos no certifica
seguridad. Licencia CC BY-NC 4.0, sin aprobación comercial. La solicitud
está en `RETSU_BANKING77_EXECUTABLE_REQUEST_2026_09_30.md` y su estado es
**NOT_RUN**. Dispositivo pedido: CPU.

## DOIN — archivo desechable, no un lago

doin-core `dae92beab3b8b8839639564865c6f4b538ee9c53`.
doin-node `212c26380f3ba6d52c291846c942bc159c7b4de6`.
El integrador reejecutó los dos módulos con el código en `PYTHONPATH`:
**11 passed** y **5 passed**.

`ResourceRegistry.append` escribe una fila de catálogo. No copia los bytes
del recurso. `write_metrics` guarda un informe del consumidor. El adaptador
de esta entrega es un directorio local. Etiqueta: **DISPOSABLE_FILE_NOT_LAKE**.

En el bloque fixture: la lectura devuelve los mismos bytes y los dos digestos;
un byte volteado es `DIGEST_MISMATCH`; el segundo put no duplica el registro;
un fichero ausente es `MISSING` y uno ilegible es `UNREADABLE`. La proyección
desechable (SQLite en memoria, misma forma de llamada que el registro de
rondas) escribió 3 filas, candidato no ganador incluido, `chain_verified`
falso, y la segunda proyección insertó 0. Una ronda sin bloque queda
`UNANCHORED`. No se cambió el consenso, no se podó, no se abrió Postgres y
no se migró ninguna cadena. AT01–AT10 siguen **NOT_RUN**. Este corte toca
identidad de serialización, retención sombra y reintento. No toca poda,
migración de proveedor ni ahorro medido de la flota.

## Publicación

Las cuatro entregas anteriores se publicaron y el tip remoto se leyó de
vuelta, igual al local: GPU `3f910fbf`, clasificación `b5e540ea`, retorno
`a2dfb150`, estrategia `1b25d31`. Los diff de esas entregas y los de las
correcciones de arriba se revisaron: no añaden claves ni rutas de casa.
Las correcciones se publican encima, sin force, después de este retorno.
Un tip que no aparezca en la lectura remota de esa publicación no está
publicado.

Postgres, Metabase, los puertos de chat, la VM de MT5 y la GPU en uso no se
tocaron. Los caches `.repair-env` (7,3 GiB) y `.closure-env` (3,1 GiB) siguen
en su sitio. No se borraron.

— Retsu
