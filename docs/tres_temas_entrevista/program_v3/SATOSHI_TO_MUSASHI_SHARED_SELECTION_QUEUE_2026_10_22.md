# Cola compartida de la selección v2 (para Musashi, 2026-10-22)

Orden del propietario: la mitad de la cola pendiente de la selección v2 la ejecuta
Musashi en el host del 5070 Ti/5090 (CPU solamente), la otra mitad Satoshi en el host
del 4090. La cola es determinista y fragmentada por índice: `--shard 0/2` = tu mitad,
`--shard 1/2` = la mía. Cada unidad escribe una línea en `LEDGER.jsonl`; una unidad
fallida lleva traceback y se reintenta al reanudar; nada se duplica entre mitades.

## Qué hay ya en tu host

Directorio `<psel-root>` = `/var/tmp/psel-v2-20261022/` (mismo en ambos hosts):

| Ruta | Contenido |
| --- | --- |
| `code/` | heuristic-strategy `satoshi/expert-guided-rl-20261019` @38ca383 (archivo del árbol, sin .git) |
| `data/admitted/`, `data/sel15/` | EURUSD admitido (sha c80847bb…) + CSV sel15 (sólo para etiquetas y calendario) |
| `data/eurusd_h1_layer1_pool_r4/` | pool capa 1 sobre cross-asset r4 (201 columnas) |
| `data/eurusd_h1_layer2_pool/` | pool capa 2 (79 columnas) |
| `data/eurusd_h1_catalog/FEATURE_CATALOG.csv` | catálogo (370 filas) |
| `psel_remote.sh` | lanzador: `bash psel_remote.sh <etapa> <shard>` |
| `out_full/LEDGER.jsonl` | ledger local (se fusiona con el del otro host entre etapas) |
| `LOG_full` | marcas `<etapa> shard <k/2> start|DONE|FAILED` (la etapa base, lanzada antes, marca en `LOG`) |

La etapa `base` de tu host la lancé yo antes de esta orden y sigue corriendo bajo
mi proceso; déjala terminar (marca `base shard 0/2 DONE` en `LOG`). Tu parte empieza
en la etapa `contrast`.

## Protocolo entre etapas (lo hace mi driver, tú sólo lanzas)

1. Cuando ambos hosts tengan `base shard … DONE`, mi driver fusiona los dos ledgers
   (último registro por clave) y copia el ledger fusionado a `out_full/LEDGER.jsonl`
   en ambos hosts. La regla de "tarea informativa" (pérdida CV del HGB por debajo del
   prior con el pool completo) se lee de ese ledger fusionado: por eso ninguna mitad
   puede empezar `contrast` antes de la fusión.
2. Mi driver escribe en `DRIVER.log` (en mi host) `contrast launched on dragon`.
   Tú lanzas en tu host, cuando veas el ledger fusionado actualizado (mtime posterior
   a tu `base … DONE`):

   ```bash
   cd /var/tmp/psel-v2-20261022 && setsid nohup bash psel_remote.sh contrast 0/2 > nohup_contrast.log 2>&1 < /dev/null & disown
   ```
3. Igual para `rescue` tras la fusión de `contrast`, y para la etapa nueva `encoding` (misma información en distintas codificaciones; se lanza después del primer agregado, cuando el ledger fusionado vuelva a actualizarse; el código `code/` ya está actualizado en tu host con esa etapa):

   ```bash
   cd /var/tmp/psel-v2-20261022 && setsid nohup bash psel_remote.sh rescue 0/2 > nohup_rescue.log 2>&1 < /dev/null & disown
   cd /var/tmp/psel-v2-20261022 && setsid nohup bash psel_remote.sh encoding 0/2 > nohup_encoding.log 2>&1 < /dev/null & disown
   ```
   Si tu mitad de `encoding` no ha empezado 2 h después de que arranque la mía, mi driver la toma también en dragon (sin duplicar: las unidades hechas se saltan).
4. El agregado (`aggregate`) lo corro yo sobre el ledger fusionado y publico
   RANKING / CONTRASTS_PARENT_COMPONENTS / FAMILY_DECISIONS / SHORTLIST.

Si una etapa termina `FAILED` en tu host, relanza el mismo comando: las unidades
hechas se saltan, las fallidas se reintentan. Si vuelve a fallar, mándame la línea
del ledger con `error`/`traceback` y no toques el código: lo corrijo, actualizo el
archivo `code/` en ambos hosts y te aviso.

## Recursos

`psel_remote.sh` corre bajo `crispdm-run -m 3G`, `CUDA_VISIBLE_DEVICES=""` (CPU
solamente; no toca la 5090 ni la 5070 Ti). Si la admisión rechaza por headroom,
espera o pide menos memoria sólo si es un comando distinto (la herramienta no
acepta bajar la petición de un comando idéntico tras un rechazo). Sin interpolar,
sin rellenar, sin leer 2025: el código lo impone.

## Lo que mide cada etapa (resumen; detalle en el docstring de `tools/pool_selection_v2.py`)

- `base`: HGB (iteraciones elegidas en el bloque interno de cada fold) y tronco
  temporal causal (2 unidades, ventana 24) sobre el pool completo, 6 folds purgados
  dentro de TRAIN 2020-2023 (embargo 96), permutación agrupada por bloques de 72.
- `contrast`: por familia, en dos contextos (`full`, `core15`): sin/con reentrenado;
  familias con padre declarado: none / P / Cj / subgrupos / Cgroup / P+Cgroup, mismas
  filas (hash de timestamps), 3 folds.
- `rescue`: 20 pares de familias no retenidas, contexto core15.
- `encoding`: por familia de codificación (hora del día, régimen GMM/HMM, reloj de eventos, escala de rango, nivel de precio): variantes con la misma información, mismas filas; estructurales con ambos modelos, monótonas sólo con el temporal (un árbol es invariante a ellas).

Nada de esto certifica utilidad en trading: el consumo real se prueba después con
los subconjuntos finalistas en E3-M FULL (escala fija TRAIN, ablación de estado).
