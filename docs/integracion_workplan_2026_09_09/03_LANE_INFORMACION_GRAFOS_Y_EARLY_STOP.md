# 03 — Lane I-INFO: información, grafos y early stopping

**Origen:** orden de Harvey, 2026-09-09.  
**Estado:** PROPUESTO. No hay plugin. No hay columna en el cubo.  
**Disciplina ya escrita:** STEP 05 (entropía, MDL, surprisal) y recorte Retsu 5-sep: **MDL ≠ Kolmogorov ≠ \(C\) de MacKay ≠ \(U^*\)**.

Esta lane no es una tesis. Es **instrumentación obligatoria** de todo run que a partir de I1 se acepte en el cubo de laboratorio.

## 1. Por qué existe

Harvey: seguir con modelos en valores por defecto, preproceso incompleto y sin métricas de información/grafo impide buscar patrones después. El cubo debe poder contrastar, entre epochs y entre runs:

1. estimación de información de **cada serie de entrada** y del **target**;
2. estimación de complejidad del **modelo**;
3. métricas de **grafo/red con pesos**;
4. el punto de **overfitting** (val vs train, y el epoch de early stop).

Objetivo declarado: insights *a posteriori*. Hipótesis extra: un criterio de parada alternativo. Esa hipótesis **no** se implementa como default.

## 2. Lo que **no** vamos a afirmar

| Afirmación prohibida | Por qué |
|---|---|
| “Medimos la complejidad de Kolmogorov” | \(K(\cdot)\) es incomputable. Lo que cabe es un **proxy**: longitud comprimida bajo compresores fijos |
| “Los bits de Shannon son el ancho Conv1D” | Recorte STEP 05 / P-PRE |
| “Huffman como input del extractor” | El bitstream no es la geometría que el modelo aprende |
| “La capacidad de red de STEP 09” | STEP 09 dijo expresamente que **no** estima network Kolmogorov capacity |
| “H-ES ya funciona” | No hay un solo run instrumentado |

Nombre honesto del proxy: **longitud de descripción comprimida** \(L_C(\cdot)\), no \(K(\cdot)\).

## 3. Tres objetos que no se mezclan

Sea un run \(r\) en epoch \(e\).

### 3.1 Información de los datos (entradas)

Para cada serie/variable \(k\) en el split de **entrenamiento** (nunca test):

| Código | Cantidad | Cómo (mínimo viable) | Precaución |
|---|---|---|---|
| `D.bytes` | bytes del array serializado (float32 o alfabeto \(Q\)) | dump canónico, endian fijo | el dtype cambia \(L_C\) |
| `D.L_zstd` / `D.L_lzma` / `D.L_bz2` | longitud comprimida | mismos niveles fijos en todos los runs | varios compresores = sensibilidad, no “el \(K\)” |
| `D.H0` | entropía de orden cero del alfabeto \(Q\) | STEP 05, train only | alfabeto provisional hasta 4B |
| `D.H_ctx` | rate con contexto (si 5B abre) | \(-\log_2 p(x_t\mid context)\) | si no baja vs H0, no hay rama de fuente |
| `D.SNR_hat` | SNR estimada causal | STEP 02–03 | en series naturales **no** se afirma \(n_t\) conocido |
| `D.I_free_synth` | bits de la señal limpia | **solo** si el generador es conocido (E0, T1, P-CAP) | prohibido en finanzas como “información libre de ruido verdadera” |

**Información libre de ruido:** en sintético se **calcula** (generador). En series naturales se **acota por abajo** con \(L_C\) de la serie cuantizada y se **acota por arriba** con \(L_C\) de la serie cruda. Nunca se etiqueta el residuo de un filtro como \(N\) verdadero.

### 3.2 Información del target

Mismas columnas con prefijo `Y.` sobre el objetivo del run (rendimiento, clase, acción, etc.).  
Además:

- `Y.H0`, `Y.L_zstd` del vector de etiquetas/valores de train;
- si hay cabezal probabilístico: `Y.nll_bits` (pérdida log2) en train y val — ya es la unidad de P-CAP.

### 3.3 Complejidad del modelo

| Código | Cantidad | Cómo |
|---|---|---|
| `M.n_params` | cuenta de pesos entrenables | Keras `count_params` |
| `M.bytes_raw` | serialización canónica de pesos (sin optimizer state) | mismo protocolo en todos los plugins |
| `M.L_zstd` / `M.L_lzma` | longitud comprimida de esos bytes | proxy de descripción del modelo **en ese epoch** |
| `M.L_ratio` | `M.L_zstd / M.bytes_raw` | 1 = incompresible (ruido en pesos); baja = estructura |
| `M.graph.*` | métricas de grafo, §4 | snapshot del grafo de pesos |

Opcional posterior (no I1): `M+D.L_zstd` de concatenar pesos+datos — solo como diagnóstico NCD, no como “información mutua”.

## 4. Métricas de grafo / red con pesos

Tratar cada capa densa o Conv1D como grafo dirigido ponderado: nodos = unidades, arista = peso.

Mínimo (todas en valor absoluto de pesos, umbral relativo fijado en desarrollo, **un** umbral para todos los runs de un experimento):

| Código | Métrica | Para qué Harvey la quiere |
|---|---|---|
| `G.density` | densidad tras umbral | ¿el modelo se vuelve denso al overfittear? |
| `G.weight_entropy` | entropía de \(\lvert w\rvert\) normalizados | concentración de magnitud |
| `G.spec_radius` | radio espectral de \(\lvert W\rvert\) (capas cuadradas o vía unfold Conv) | estabilidad / explosión |
| `G.eff_rank` | rango efectivo (suma de singular values / máx) | dimensión ocupada |
| `G.modularity` | modularidad (si la capa es interpretable como bloques) | ¿aparecen comunidades al memorizar? |
| `G.path_mean` | camino medio en el grafo umbralizado | conectividad |

Si una métrica no es definible para esa capa (Conv no cuadrada, etc.), se registra `NA` y no se inventa un isomorfo.  
No se llama a esto “capacidad de Kolmogorov de la red”.

Costo: un snapshot por epoch es caro en TFT/Transformer. Contrato:

- **cada epoch** en modelos chicos (ANN/CNN/TCN diario, E0, P-CAP);
- **cada `log_every` epochs** (fijo, p.ej. 10) en modelos grandes;
- siempre: al mejor val, al early-stop, y al último epoch.

## 5. Contrastes que el cubo debe poder hacer

Consultas objetivo (no se ejecutan hasta poblar I1):

1. `M.L_zstd(e)` vs `val_loss(e)` — ¿la descripción del modelo sigue creciendo cuando val ya empeora?
2. `M.L_zstd / Y.L_zstd` y `M.L_zstd / D.L_zstd` vs epoch de mínimo val.
3. `D.H_ctx` y `D.L_zstd` vs MASE fuera de muestra (¿datos más incompresibles ⇒ techo más alto?).
4. `G.eff_rank` y `G.weight_entropy` vs gap train–val.
5. En sintético: `D.I_free_synth` vs `M.L_zstd` en el epoch óptimo.

## 6. Hipótesis H-ES (early stopping informacional)

**Enunciado (falsable):** existe un estadístico \(S_e\) computable en \(e\) **sin mirar el conjunto de prueba**, función de

\[
S_e = f\!\left(\frac{M.L_C(e)}{D.L_C},\ \frac{M.L_C(e)}{Y.L_C},\ G(e),\ L_{\mathrm{train}}(e), L_{\mathrm{val}}(e)\right)
\]

tal que el primer \(e\) que cumple una regla prerregistrada \(R(S_e)\) está más cerca del epoch de mínimo val-loss de un holdout de *desarrollo* que `early_patience` sobre val-loss, en tareas **reservadas**.

**Falla si:** en las tareas de confirmación, \(R\) no mejora (o empeora) el val-loss final ni el cómputo, frente a paciencia estándar, con el mismo presupuesto.

**Prohibido hasta que H-ES gane:**

- sustituir `early_patience` en campañas F2;
- parar por \(S_e\) en E2/E3 de P-MOD;
- declarar “el overfitting es un fenómeno de información” en el PDF doctoral.

Piloto de \(f\) y \(R\): **después** de tener ≥ N runs instrumentados (N se fija en I0; propuesta: 30 runs CPU de ANN/DLinear en E0/T1). No se diseña \(f\) mirando el test de P-MOD.

## 7. Encaje con STEP 05 y P-CAP

- STEP 05 aporta `D.H0`, `D.H_ctx`, MDL como **diagnóstico de representación**, no como H-ES.
- P-CAP aporta pérdida log2 y \(\widehat C_{mem}\). `M.L_C` es **otro** eje: descripción de pesos, no bits memorizados de etiquetas aleatorias. Se registran los dos. No se identifican.
- Compresores C1–C7 (sparse, RD latente) son **modelos**, no el medidor \(L_C\). El medidor usa zstd/lzma/bz2 de **bytes ya serializados**.

## 8. Dónde se calcula (repos)

| Medidor | Repo | Momento |
|---|---|---|
| `D.*` / `Y.*` sobre CSV/ventanas | `preprocessor` (fit en train) + `predictor` al armar ventanas | una vez por split, no por epoch |
| `M.*` / `G.*` | `predictor` y `feature-extractor` al final de epoch | callback, CPU, opcional GPU-off |
| escritura al cubo | `predictor/olap` (schema `public` del ETL v2) | throwaway primero; ver doc 05 |

No se calcula \(L_C\) de datos en el GPU training loop. Se cachea.
