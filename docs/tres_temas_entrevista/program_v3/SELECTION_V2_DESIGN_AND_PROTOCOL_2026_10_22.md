# Selección v2 del pool completo: diseño, codificación, protocolo y orquestación (2026-10-22)

Para Musashi y para el propietario. Este documento es el mapa de todo lo que corre
desde el 2026-10-22 en la selección de entradas del carril experto (ZigZag → RL
guiado). Sustituye como referencia a las selecciones v1–v6b y al screen D. Nada
de lo descrito certifica utilidad en trading: eso lo decide el consumo real
(§7). Rutas privadas fuera de Git se escriben como `<store-root>` =
`/var/tmp/features-candidates/` en los hosts del proyecto.

## 1. Documentos y commits que hay que leer

| Qué | Dónde (repo · rama · commit) |
| --- | --- |
| Brief de correcciones (Musashi) | predictor · `codex/data-coverage-repair-20261018` · 7633cd37 · `docs/tres_temas_entrevista/program_v3/SATOSHI_SELECTION_CORRECTIONS_AND_AUDIT_BRIEF_2026_10_21.md` |
| Respuesta de Satoshi al brief | predictor · `satoshi/reply-to-musashi-20261022` · 2ebe0ac7 · `…/SATOSHI_REPLY_TO_CORRECTIONS_BRIEF_2026_10_22.md` |
| Cola compartida (mitad de gamma) | predictor · misma rama · e34f8f13 · `…/SATOSHI_TO_MUSASHI_SHARED_SELECTION_QUEUE_2026_10_22.md` |
| Reparación de datos: cruces 2023 | financial-data · `satoshi/fx-crosses-recovery-20261022` · 34f1aa72c · `work_plan/FX_CROSSES_RECOVERY_20261022.md`, `_scripts/recover_independent_hf_crosses.py` |
| Store cross-asset r4 (+XAUUSD) | feature-eng · `satoshi/cross-asset-h1-20261020` · bce7c06 · `examples/config/cross_asset_eurusd_h1_r4.json`, `docs/cross_asset_eurusd_h1_r4/MANIFEST.json` |
| Pool capa 1 (sobre r4) | predictor · `satoshi/timing-selection-20261019` · 50b0e257 · `tools/layer1_pool_h1.py`, `docs/timing_selection/layer1_pool_20261022/` |
| Pool capa 2 + catálogo | predictor · misma rama · e04b496a · `tools/layer2_pool_h1.py`, `tools/feature_catalog_h1.py`, `tools/lib/{native_wavelet,gaussian_hmm,kalman_llt}.py`, `docs/timing_selection/layer2_pool_20261022/` |
| Escalado fijo, SCALER.json, ablación de estado, contrato de periodos | heuristic-strategy · `satoshi/expert-guided-rl-20261019` · 5e10f7c, 5fda47a, 379de4b, 696a774 · `app/short_horizon/expert_guided/data.py`, `env.py`, `tools/expert_guided_e3m.py`, `tests/test_expert_guided_{scaling,ablation}.py` |
| Motor de selección v2 | heuristic-strategy · misma rama · `tools/pool_selection_v2.py` (docstring = protocolo) |
| Plan vivo | heuristic-strategy · `satoshi/work-plan-20261019` · `docs/short_horizon/SATOSHI_WORK_PLAN.md` |
| Literatura de la fase de datos | predictor · `satoshi/timing-selection-20261019` · `docs/timing_selection/LITERATURE_DATA_PHASE_2026_10_21.md` |

## 2. Datos: qué cambió y por qué

1. **Hueco 2023 en los cruces.** El lago HistData (`market_data/forex/g10/*/1h.parquet`)
   pierde 25–40 % de las barras de feb–jul 2023 y ~8 % en H2 en todos los pares; el
   store r3 heredaba 40 % de NaN en `fx.*.ret_1` y 75 % en `rv_24` en 2023H1.
   Reparación: los seis cruces, USDCAD, NZDUSD y XAUUSD recuperados desde la misma
   fuente independiente H1+M30 (etiquetas UTC de apertura) que produjo el EURUSD
   admitido; identidad M30→H1 con 0 discrepancias; cobertura 2023H1 = 1.000;
   reloj corroborado contra el lago (offset 0: 0.0 pips mediana y p95 en ~28.8 k
   horas comunes; controles ±1..3 h: 2.8–12.5 pips). Sin interpolación ni relleno:
   la única hora ausente del productor (2023-11-29 13:00Z) queda listada.
   XAUUSD no tiene testigo horario independiente (declarado, no corroborado) y su
   `rv_24` no se puede calcular en la reja FX por la pausa diaria de metales
   (cobertura 0.03 %; excluida por regla, no juzgada).
2. **Pool capa 1 reconstruido sobre r4** (201 columnas): inventario técnico, pool de
   precio multi-escala, velas en pips, fractales, cross-asset r4, reloj de eventos,
   actividad, SSA causal; diagnóstico por columna en TRAIN (ADF/KPSS, asimetría,
   curtosis) y decisión NONE/DIFF/ASINH documentada (175/7/6).
3. **Pool capa 2** (79 columnas, todas causales y probadas con perturbación futura
   bit-idéntica ≤ t): wavelet db4 nativa (ventana 256, nivel 4) sobre log-cierre y
   retornos; Hilbert (ventana 128, detrend OLS dentro de la ventana; borde derecho
   declarado); multitaper DPSS (NW 3.5, K 5, ventana 256; 4 bandas, entropía
   espectral, periodo dominante); familia espectral de fuerza USD/EUR sobre r4;
   Kalman de tendencia local con Q/R por máxima verosimilitud en TRAIN (q_slope
   en la frontera cero: la pendiente aporta poco, documentado); GMM k=3 y HMM
   gaussiano k=3 (EM en numpy, **filtro forward**, nunca forward-backward) sobre
   `logret_1, rv_24, log_range`; diferenciación fraccional d=0.3 (ventana fija
   256; tabla ADF por d). Segundo juego de parámetros con fit hasta 2023-01-01
   para el brazo de desarrollo de tres años.
4. **Catálogo** generado desde los productores (370 filas): fuente, padre, familia,
   componente, algoritmo real, parámetros, reloj, disponibilidad, cobertura
   mensual, calentamiento, faltantes, outliers (6·MAD), estadísticas ajustadas,
   productor + hash, estado, decisión de preprocesamiento. 1 fila PENDING
   (consenso/sorpresas de calendario: decisión del propietario sobre la fuente).

## 3. Codificación de la información (lo que ya está y lo que se mide)

Principio del propietario: tan importante como la entrada correcta es la
codificación que maximiza la información útil que llega al modelo.

| Clase de información | Codificación base (documentada por columna) | Alternativas que mide la etapa `encoding` (§5.4) |
| --- | --- | --- |
| Cíclica (hora, día) | `sin/cos`; sesiones binarias | one-hot 24; sólo sesiones |
| Régimen (GMM/HMM) | probabilidades de pertenencia ordenadas por volatilidad + entropía (+ sorpresa HMM) | one-hot duro del `argmax` |
| Fase (Hilbert) | `cos/sin` de la fase | — (el ángulo crudo rompe la continuidad; no se contrasta) |
| Niveles no estacionarios | diferencia (decidido por ADF/KPSS); desde heuristic-strategy @`stationarity rule` el cargador con escala fija mide cada entrada en TRAIN y diferencia (o rechaza, `--nonstationary refuse`) los niveles antes de ajustar constantes, con la tabla ADF/KPSS en SCALER.json | diferencia + nivel relativo a 480 barras |
| Magnitudes de cola pesada | `asinh` sólo si redujo la asimetría medida | `log1p`; ratio a la media móvil de 24 barras |
| Reloj de eventos | horas desde / horas hasta, lineales | decaimiento `exp(−h/4)`; ventana ±2 h |
| Escala final de todo | z-score con constantes de TRAIN (amplitud preservada) | — |

Nota metodológica: para un modelo de árboles, una transformación monótona por
columna (`log`, ratio a constante) no cambia nada; por eso esas variantes se
juzgan con el tronco temporal, y las estructurales (cambian el conjunto de
columnas) con ambos modelos.

Uso posterior (petición del propietario): la codificación ganadora por familia y
las métricas de señal (estacionariedad, cola, periodicidad, entropía espectral,
persistencia de régimen) son la base para decidir **qué extractor conviene a qué
grupo de entradas** (ramas convolucionales por grupo, extractor compartido entre
familias afines, embedding para categóricas) en el rediseño del RL guiado.

## 4. Filas, etiquetas y periodos

- Filas: orígenes planos observables con puerta de entrada abierta, dentro de
  TRAIN consolidado 2020-01-01..2024-01-01 (19 459 filas por tarea); toda etiqueta
  final antes del corte; TEST (≥ 2025) nunca leído; 2024 se lee una vez al final
  en el modelo consumidor.
- Siete tareas: `opener3` (pivote anticipatorio del ZigZag: NONE/LONG/SHORT),
  `hit` (barrera ±2σ antes de 72 barras), `up_given_hit`, `sl_first_long/short`
  (orden hipotética en la apertura siguiente, TP/SL 2σ: SL o cierre forzoso
  negativo antes del TP), `net_long/short` (resultado neto en σ, regresión).
- Folds purgados y contiguos dentro de TRAIN (6 en la base, 3 en contrastes),
  embargo 96 barras; bloque interno (último 15 % de las filas de entrenamiento
  de cada fold, con embargo) para elegir iteraciones del HGB y parada temprana
  del temporal: el fold de validación nunca decide nada.

## 5. Etapas del motor (`tools/pool_selection_v2.py`; BUDGET.json se escribe antes de correr)

1. **base**: HGB (profundidad 4, hoja mínima 200, L2 1.0, iteraciones por bloque
   interno) y tronco temporal causal (2 unidades por rama, ventana 24, stride 1)
   sobre el pool completo; permutación agrupada por **bloques de 72 filas**
   (columnas y slots de ventana del grupo permutados juntos), 3 semillas.
   Regla declarada: una tarea es informativa si la pérdida CV del HGB queda por
   debajo del prior con el pool completo; las demás se reportan y no ordenan.
2. **contrast**: por familia (57), dos contextos (`full`, `core15`): sin/con
   reentrenado; familias con padre: none / P / Cj / subgrupos / Cgroup / P+Cgroup,
   mismas filas (hash de timestamps + cobertura nativa), 3 folds, tamaño de modelo
   registrado.
3. **encoding** (nuevo, §3): misma información en distintas codificaciones, mismas
   filas (la intersección de todas las variantes), contexto = resto del pool;
   decisión = mejor variante con lo90 > 0 frente a la base; si no, se conserva la
   base.
4. **rescue**: 20 pares de familias no retenidas, contexto core15.
5. **aggregate**: RANKING.csv, CONTRASTS_PARENT_COMPONENTS.csv, ENCODING.csv,
   FAMILY_DECISIONS.csv, SHORTLIST.json, SUPPORT.json, SELECTION_V2.md.

Regla de decisión (declarada antes de leer): RETAINED si en alguna tarea y
contexto la ganancia reentrenada tiene cota inferior unilateral 90 % > 0, o la
permutación por bloques la tiene en algún modelo; columnas retenidas de una
familia descompuesta = las del mejor brazo con lo90 > 0 frente a `none`; lo demás
NOT_RETAINED_WITH_THIS_EVIDENCE (nunca "inútil"); RESCUED_INTERACTION por pares.

## 6. Resultado de la etapa base (pool completo, 84 unidades, 0 errores)

| tarea | HGB CV | prior | HGB train | temporal CV | informativa |
| --- | --- | --- | --- | --- | --- |
| opener3 | 0.4465 | 0.6963 | 0.3646 | 0.6470 | sí |
| hit | 0.1657 | 0.1723 | 0.1407 | 0.1942 | sí (marginal) |
| up_given_hit | 0.7080 | 0.6945 | 0.6358 | 0.6956 | no |
| sl_first_long | 0.6987 | 0.6925 | 0.6607 | 0.7313 | no |
| sl_first_short | 0.7025 | 0.6945 | 0.6607 | 0.6954 | no |
| net_long (MSE) | 1.7150 | 1.6845 | 1.5909 | 1.7685 | no |
| net_short (MSE) | 1.7150 | 1.6845 | 1.5909 | 1.8147 | no |

Lectura honesta: con 260 columnas y dos sesgos inductivos distintos, ninguna
tarea de dirección o resultado neto de orden bate el prior fuera de fold; el HGB
se detiene en 23–48 nodos (no encuentra estructura). La anticipación de pivotes
y la actividad sí tienen señal. Familias dominantes para `opener3`: `inv.raw`
(retornos y rango crudos), `px.close_loc`, velas; para `hit`: `act.p_hit`,
`cot.eur`. Los contrastes y la etapa de codificación corren sobre estas dos
tareas; las otras cinco siguen en el ledger como NOT_INFORMATIVE_WITH_THIS_POOL.

## 6b. Resultado del E3-M sin rolling z-score sobre el CSV heredado (cerrado 03:39Z)

FULL head 16, escala fija TRAIN, 15 inputs heredados (incluye el NIVEL crudo `HIGH`):
INNER_A Sharpe por semilla −0.02 / +2.41 (inválida: 1.7 órdenes/semana) / −4.68, media −0.76
frente a +0.86 (+1.00/+0.42/+1.17) con rolling-z; TRAIN neto +27 732 USD en una semilla
(vs −4 317 con rolling). Diagnóstico: con constantes fijas un nivel de precio es un proxy
de fecha; el rolling-z lo escondía. Corrección estructural: el cargador con escala fija
mide la estacionariedad de cada entrada en TRAIN (ADF/KPSS, la regla del catálogo) y
diferencia los niveles antes de ajustar constantes, registrando la decisión en
SCALER.json (`stationarity`); E3-M `--nonstationary {diff,refuse,keep}`, por defecto
`diff`. La prueba de integración CLI → tensor → SCALER.json → recarga verifica que
`HIGH` sale como `HIGH@diff`. Los brazos finalistas ya consumían el pool (niveles
decididos por el catálogo), así que son la comparación limpia.

## 7. Consumo real y cierre

- Cola maestra secuencial en el host del 4090 (`dragon_queue.sh`): S1 espera el
  barrido de reajuste; S2 E3-M FULL+JOINT core15 con escala fija TRAIN; S3 espera
  los finalistas; S4 tres brazos sobre el **mismo store** (pool completo
  materializado, 31 116 × 280, sha `2f7b3b8c…`): core15 / shortlist /
  core15+shortlist, E3-M FULL head 16, escala fija, SCALER.json junto a cada
  checkpoint, ablación de estado disponible; 2024 una sola lectura.
- Cola compartida: mitad de gamma (shard 0/2 de contrast, encoding y rescue) la
  ejecuta Musashi con el mismo script; el driver del coordinador fusiona ledgers
  entre etapas y empuja los finalistas a la cola de dragon.
- Solicitud de auditoría (§11 del brief) al cerrar: commits exactos, diffs,
  pruebas, ledger, catálogo → subconjunto → columnas del tensor → modelo →
  métricas enlazados por hashes, comandos reproducibles, y lo que no se cerró
  con su razón.

## 8. Dirección registrada del propietario para la siguiente ronda de RL

Entrenamiento **limpio**: sin realimentación de entradas del agente (ablación
`none`/`minimal` primero), sin z-score móvil, escala fija TRAIN, entradas del
pool seleccionado con su codificación ganadora, y **múltiples señales ZigZag**
(varios umbrales/horizontes como demostraciones) para el RL guiado por el
experto, evaluando si así los distintos módulos entrenan mejor. La selección y
las mecánicas de negocio son la base; se avanza con cuidado y sin atajos.
