# Cierre de T2, C01/C02 resueltos y T3 v2 desde el perfil de cada grupo (Satoshi, 2026-10-23)

Para Musashi. Responde a `MUSASHI_REVIEW_T1_ANNEX_AND_T2_VERIFIER_2026_10_22.md` (@29c23bd3) y a tu cierre de T2.
Código: heuristic-strategy `satoshi/expert-guided-rl-20261019` **@1aaf367**. No es la auditoría final ni autoriza SAC
ni la ronda guiada; `RELEASE.json` no existe.

## 1. C01 y C02, cerrados con tus sondas como regresiones

| hallazgo | corrección | regresión |
| --- | --- | --- |
| C01: ausencia, `null` o productor desconocido pasaban como ZigZag heredado; una lista lanzaba AttributeError | `label_mode_of(e, producer)`: la ausencia es ZigZag heredado **sólo** con productor autenticado: identidad de tu configuración (sha256 del JSON con claves ordenadas, como la calcula tu worker) = `config_sha256` del recibo, y el `tools/expert_guided_e3m.py` que esa configuración fija = el blob de 186b131 en el repositorio (`61e0fceb…`), que no emite `labels`. `null` y tipos inválidos → `MALFORMED`; productor no autenticado → `ABSENT_WITHOUT_AUTHENTICATED_PRODUCER`; un campo presente que el productor fijado no puede emitir → rechazado. No se escribe nada en ningún `E3M.json` | `test_label_mode_requires_an_authenticated_legacy_producer_musashi_c01`, `test_producer_authentication_compares_the_pinned_source_with_the_science_commit` |
| C02: el mismo resultado bajo una unidad alterada pasaba | `verify_t2_receipt`: nombre, comando efectivo = comando de la unidad, identidad tipada **recalculada** del comando, dropout/wd, estado, lista, split, columnas solicitadas y consumidas = lista, soporte TRAIN/INNER_A del recibo = el del resultado, y el hash de **cada** archivo sellado (E3M, scaler, sonda y los 12 checkpoints) contra tu recibo; además soporte único por lista y estado. `regularize` exige ese documento `all_ok` y que cada brazo sea el archivo verificado (un cambio posterior se rechaza) | `test_receipt_binds_identity_regularization_list_and_every_sealed_file_musashi_c02`, `test_regularize_refuses_a_result_that_is_not_the_verified_file` |

Sobre tus ocho resultados reales: **8/8 OK**, productor autenticado, soporte único por lista (core15 `78b5834e…`,
estricta `f1c592d5…`). `evidence/t2/T2_VERIFY.json`. 52 pruebas en verde en CPU.

## 2. Regularización decidida por la regla congelada

**base** (dropout 0.1, wd 1e-4). Los dos ajustes quedan no dominados: el fuerte sólo gana en la dispersión entre
semillas de la lista estricta (0.47 frente a 0.79). El desempate 1 no alcanza: la ventaja de base en Sharpe medio
(0.67) no supera la dispersión agrupada de semillas (0.75). El desempate 2, menor brecha media, elige base. 2024 no
se leyó. Aplicada al spec con la fuente ligada al hash de la decisión; `check` del spec válido, sólo falta la puerta
del propietario. `evidence/t2/REGULARIZATION_DECISION.json`. Tus brazos con estado son tu contraste, no fuente.

Anexo T1 corregido según tu revisión: sin símbolo de igualdad para correlación de rangos; los 10 pares son el grafo
de vecinos máximos, no una enumeración; T3 condicionado a las demás entradas presentes.

## 3. T3 v2 (reemplaza al manifiesto de 26 unidades, que no llegó a lanzarse)

El propietario pidió que el extractor de cada grupo se elija **desde el perfil del grupo**. El manifiesto anterior
asignaba un único tipo por nombre de familia; eso era una regla de nombre, no de perfil. T3 v2
(`tools/expert_guided_t3_manifest.py`, identidad `t3v2_8875abf84a63`):

- **Candidatos**: cada grupo multi-columna prueba `pointwise` **y** `per_feature` contra el control `conv`; cada grupo
  de una columna prueba `pointwise`. Nada se descarta por el perfil.
- **Expectativa preregistrada desde tu T1** (escrita antes de cualquier resultado; sólo se reporta su tasa de
  acierto): memoria persistente (ACF₁ ≥ 0.9 y ACF₂₄ ≥ 0.5 o sin rezago no significativo ≤ 48) → `pointwise ≥ conv`;
  casi blanca (|ACF₁| < 0.1) → `conv ≥ pointwise`; redundancia interna alta (≥ 0.95) → `conv ≥ per_feature`; baja
  (< 0.5) → `per_feature ≥ conv`.
- **Decisión congelada por grupo**: una alternativa reemplaza al control sólo si su Sharpe INNER_A medio supera al del
  control por más que la dispersión agrupada de semillas **y** su brecha no es mayor por más que la dispersión de
  brechas, con el mismo soporte y tres semillas válidas; si no, el control se queda. Un resultado negativo cierra la
  unidad. Luego una **unidad de combinación** con todas las elecciones confirma que las ganancias se suman.
- **Fijos**: lista estricta v4, archivo de grupos (digest), columnas y codificaciones, regularización T2, ventana
  13/2 (la de T2), modelo de la ronda (2/2/16), contrato 1/día–1:50, ≥ 2 órdenes/semana, consolidated, sin estado,
  etiquetas ZigZag. La regularización se midió a profundidad 1 y se lleva a profundidad 2: declarado.
- **Reparto**: 43 unidades; impares tuyas (21), pares y control míos (22). Listados tipados con comando efectivo e
  identidad: `evidence/t3/T3_UNITS_{musashi,satoshi}.json`.

## 4. Estado y lo que te pido

- **Mi mitad corre** en el host del 4090 desde un paquete con pins verificados (control primero).
- **Tu mitad está preparada** en tu host, en un directorio mío nuevo `satoshi-t3-for-musashi-20261023` (no toqué los
  tuyos): código @1aaf367, datos, lista, grupos, `T3_UNITS_musashi.json`, `t3_queue_musashi.sh`; `PINS.sha256`
  verificado (`d6fd6415…`). La cola exige `ROOT` y `GPU` explícitos: en ese host el índice 0 es la 5070 Ti y el 1 la
  5090; usa tu wrapper con UUID como en T2. Si prefieres tu supervisor, los comandos tipados del listado son la
  fuente; la identidad es el sha256 del comando con `<root>`.
- Al cerrar ambas mitades: `decide` (regla congelada), `combine` y la unidad C en mi host; luego la solicitud final de
  auditoría con datos, encoding, grupos, métricas, extractores, ambos consumidores y tus retornos.
