# Retorno de la orden posterior a 788f1cfa

Retsu, dueño de la integración. 2026-09-30.
Orden fuente: la revisión de Musashi del mismo día sobre `788f1cfa`, `4d548a82`,
`c89a8ff4` y heuristic-strategy `4bb763d`.
El dueño autorizó ejecutar esa orden. La misma autorización repitió los candados:
no hubo instalación concedida, no hubo asignación concedida, la petición sigue
en 4.800 s de CPU y 3.600 s de pared sumados entre hijos, condicionada a un
entorno funcional y a una admisión vigente, y los 18 GiB siguen retirados.

**Ninguna exactitud nueva quedó medida.** Tampoco hubo resultado financiero,
score de clasificación ni piloto de GPU. Lo que sigue son preparación, una
revisión estática y pruebas sintéticas. Esas tres cosas van separadas.

El despacho, con los identificadores de los tres trabajadores, está en
`RETSU_POST_CONSOLIDATION_DISPATCH_2026_09_30.md`.

## Lo que esta ejecución no gastó

La petición de las dos calibraciones secuenciales queda igual que en la orden:
2 × 2.400 s de CPU y 2 × 1.800 s de pared, envolvente de dispositivo de 12 GiB
observada, tope de anfitrión de 12 GiB. No se lanzó. El humo de diagnóstico
tampoco: pide una asignación acotada que esta nota no concede.

El techo histórico de la estrategia, 3.600 s de CPU, es otra cifra. El diseño
`4bb763d` ya había gastado 1.461,693 s en 241 celdas. La resta es 2.138,307 s.
Esa resta no es una asignación nueva. B0 no empezó. Esas 241 celdas no se
repitieron.

Una reejecución de las nueve pruebas sintéticas de la estrategia pasó bajo
`crispdm-run -m 2G -t 180s`: 9 passed, 0,58 s. Eso no es B0 y no toca la petición
de 4.800 / 3.600.

## Carril GPU — `3f910fbf4d0f29a5381fe23f40318f879b3f6a81`

Rama `satoshi/gpu-env-repair-20260930`, desde `4d548a82`.

La biblioteca que TensorFlow no pudo abrir queda **NOT_RETAINED**. La frase
retenida es: «Cannot dlopen some GPU libraries … Skipping registering GPU
devices». En el obrero admitido, Python 3.12.13 / TensorFlow 2.21.0 registró
cero dispositivos mientras `libcuda.so.1` cargaba y el UUID ya publicado
verificaba. `FRAMEWORK_REGISTERED_NO_DEVICE` nombra el fallo. No nombra la
causa. La versión del controlador no está en la evidencia retenida.

La tabla oficial de TensorFlow 2.21 para Linux
(https://www.tensorflow.org/install/source) fija CUDA 12.5 y cuDNN 9.3, Python
3.10–3.13, y no trae columna de controlador. El entorno aislado se pinó a ese
suelo: `tensorflow==2.21.0` con `nvidia-cuda-runtime-cu12==12.5.82` y
`nvidia-cudnn-cu12==9.3.0.75`. Ese árbol no contiene `libcudart.so.13`. El
contrato que falló en el obrero sí tenía sonames de CUDA 13 en el camino del
hijo. Alinear el pin a la tabla oficial es la preparación pedida. No es una
reparación demostrada.

Medido en disco, y recontado por el integrador:

| cantidad | bytes |
|---|---:|
| 44 ruedas descargadas | 2.576.641.397 |
| ficheros regulares en `site-packages` | 5.135.683.136 |

El humo escrito y sin ejecutar es `crispdm-run` lanzando
`tools/df_placement_contract.py --verify`, el mismo veredicto de tres hechos
que usa `tools/df_e1_block.py`. Rollback: abandonar el entorno `.repair-env`
de ese worktree (7,3 GiB, ignorado por git). Ningún intérprete existente, ningún
controlador y ningún servicio fue modificado. La envolvente de 12 GiB no se
redujo.

## Carril de clasificación — `b5e540ea6a40c60d7d63a8086d96fe5f03225d49`

Rama `satoshi/banking77-closure-review-20260930`, desde `c89a8ff4`.

Se leyó, sin ejecutarlo, el código Python del repositorio
`jinaai/jina-embeddings-v5-text-small` en la revisión
`46ed7da5b47e4bca710b756313fafaf4110c6bd1`. El árbol de esa revisión tiene tres
ficheros Python. Licencia CC BY-NC 4.0. Esta revisión no aprueba un despliegue
comercial.

Operaciones concretas, todas en tiempo de llamada, ninguna corrida aquí:

- `modeling_jina_embeddings_v5.py:37-40` llama a `snapshot_download` con
  `allow_patterns=["adapters/*"]` cuando la ruta no es un directorio local.
- `modeling_jina_embeddings_v5.py:57-60` y `custom_st.py:44-47` llaman a
  `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`.

No hay `subprocess`, `eval`, `exec`, `pickle` ni `torch.load` en esos tres
ficheros. El cuerpo de los módulos, en la importación, solo importa y define
clases.

El cierre de dependencias del conjunto de nivel 1, resuelto para CPython 3.13
en manylinux, es **87 ruedas y 3.230.034.941 bytes**, recontados en disco.
Las cinco ruedas directas de ese cierre suman 16.129.124 bytes. La cifra
anterior de 16.228.123 bytes era solo el grupo directo y no es este cierre.
La mayor parte del cierre es `torch==2.14.0` con la pila CUDA 13 que el índice
Linux resuelve por defecto. No se sustituyó un índice de CPU.

El tamaño instalado queda **NOT_MEASURED_NO_PYTHON_3_13**. No había un
intérprete 3.13. No se instaló uno, y no se hizo pasar un 3.12 por 3.13.
`trust_remote_code` sigue en falso. Los pesos del modelo no se descargaron.
No hubo score ni fila de almacén.

La petición de aprobación, todavía sin conceder, es una sola: ejecutar la
revisión `46ed7da5…` con ese cierre pinado, bajo CC BY-NC 4.0 y solo como
investigación, lo que exige decidir `trust_remote_code` a sabiendas de las dos
llamadas de arriba. Hasta esa decisión el ensayo de investigación queda
**NOT_RUN**.

## Carril de estrategia — `1b25d31a39c715e8631cec6aea0be2a0382072f8`

Repositorio heuristic-strategy, rama `satoshi/strategy-support-20260930`,
desde `4bb763d`.

La semántica de replicación queda la del plugin: decisión al cierre, llenado
con orden de mercado en la apertura siguiente. Las órdenes protectoras del
bróker siguen siendo un experimento con nombre propio. No se añadieron.

La variante E queda escrita en la configuración de base como resolución
explícita. `historical_run_recovered` es falso. El manifiesto de las 241 celdas
no se abrió: su variante queda **NOT_CHECKED**. El default de la firma anidada,
que decía `D` mientras `plugin_params` decía `E`, quedó alineado a `E`.

Las predicciones nuevas descritas como 1–6 h y 24–144 h usan horas
transcurridas y excluyen un origen al que le falta la barra exacta de ese
plazo. Los generadores viejos siguen, con unidad **filas**, no horas.

En el fixture sintético de cinco orígenes de desarrollo, dos quedan inelegibles
porque su objetivo de 144 h cae en o después de 2019-05-16 00:00. La duración
de la operación es **UNBOUNDED_TRADE_DURATION**: el plugin no tiene un plazo
máximo de tenencia, y `max_trades_per_5days` cuenta entradas, no cierra una
posición. Por eso los cinco orígenes quedan
**NOT_SEPARATED_BY_ORIGIN_CUT**. No se inventó una purga de seis días. La
ventana reservada no se reescribió. Estas cuentas son del fixture sintético.
No son un resultado de mercado.

Auditoría de acceso previo: las filas reservadas ya fueron leídas. Hay
lanzadores que cargan el CSV horario completo antes de filtrar por año, hay
artefactos comprometidos con marcas en o después de 2019-05-16, y el propio
diseño del 2026-09-29 verificó el fichero hasta 2020-04-29 22:00. Una etiqueta
nueva no crea un no-uso histórico.

La calibración de ruido, correlación y normalización, en el constructor nuevo,
rechaza una marca en o después del corte. No se ajustó ningún modelo de ruido
sobre datos reales.

## Qué queda delante del dueño

1. Asignación diagnóstica, si la quiere, para correr el humo de tres hechos
   con el pin de `3f910fbf` en un entorno nuevo. Las dos calibraciones siguen
   detrás de otra concesión, y detrás de un entorno que de verdad registre el
   dispositivo. La envolvente de 12 GiB no se ofrece reducida.
2. Una decisión explícita de `trust_remote_code` para la revisión
   `46ed7da5b47e4bca710b756313fafaf4110c6bd1`, más un intérprete Python 3.13,
   antes de cualquier corrida de BANKING77 nativa. El cierre a pedir cabe en
   3.230.034.941 bytes de descarga. El tamaño instalado sigue sin medir.
3. Nada bloquea la estrategia en una elección del dueño. B0 no está pedido
   por esta orden. Si más adelante se pide, el soporte de salida de las
   operaciones sigue sin cota: un corte de orígenes no separa el periodo
   reservado.

Postgres, Metabase, los puertos de chat y la tarjeta que ya estaba en uso no
se tocaron. Los entornos de medición quedan en los worktrees, fuera de git:
`.repair-env` (7,3 GiB) y `.closure-env` (3,1 GiB). Abandonarlos es el rollback.

— Retsu
