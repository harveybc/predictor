# Dictamen de Satoshi — verificación de la propuesta de transformaciones e inserción en el plan de optimización DOIN

**Fecha:** 2026-09-05
**Dictamina:** General Satoshi III, a solicitud del propietario
**Objeto:** propuesta doctoral de Musashi (commit `5449bed`) + paquete
tres-fuentes de Retsu (compuertas G1-G12, PATCH 003)
**Veredicto:** `ACCEPT_WITH_INTEGRATION_NOTES` — todo verificado; la
inserción en el plan DOIN es viable y queda especificada abajo.

---

## 1. Verificación (todo cuadra)

| Afirmación | Verificación |
|---|---|
| SHA-256 del PDF `bb051911a9d7…` | **EXACTO** en `docs/` y en `~/Downloads/` (byte-idénticos entre sí) |
| 10 páginas, 8+2 de referencias, 11 pt | **EXACTO** (24 referencias IEEE, [1]-[24], DOIs presentes) |
| Commit `5449bed` en `docs/agent-onboarding-20260816` | **EXACTO** — 8 archivos, 1178 inserciones; `.tex` (383 líneas) y `.bib` (243) committeados → PDF reproducible |
| Sin problemas visuales/referencias rotas | Inspección de las 10 páginas: consistente (ecuaciones 1-3, Tabla 1, citas resueltas) |
| Novedad acotada vs auto-sklearn/Auto-FP/FFORMA/Monash | Sustentada en §3.2/§3.4 con [1][4][12][19] como comparadores, no decorado |
| Digests del paquete Retsu (`a59228e9…`, `c292d6b9…`) | **EXACTOS** contra los archivos vivos |
| Coherencia interna de los 4 companions | Verificada; una divergencia deliberada anotada en §4.3 |

## 2. Juicio técnico de la propuesta

**Fortalezas que la hacen de nuestra casa:** hipótesis falsables con
condiciones de refutación explícitas (H1-H3); nulos publicables como
resultado válido; costo completo en el reloj (diagnóstico + fallos +
transformación + entrenamiento); familias retenidas como unidad de
generalización; y sobre todo la **separación de cuatro autoridades**
(sintético calibra → público decide utilidad → registro licencia →
dominio re-valida, «ninguna capa acuña el veredicto de la siguiente»)
— es la misma gramática de autoridad no-sustituible que gobierna
N4/Screen B. Los 8 gates de incorporación (doc 02 §6) y el contrato
de operador (`fit/transform`, digest de parámetros, mismo prefijo →
mismos bytes) son directamente implementables con la maquinaria de
sellado/refusal que ya existe en `agent-multi`.

**La respuesta directa a la preocupación del propietario** (ruido y
variación entre ejecuciones contaminando la evaluación de
experimentos posteriores): la dan tres piezas de esta propuesta —
(a) el contrato determinista del operador (T1/E0: igualdad
incremental-vs-lotes, digest de artefacto), (b) el banco de verdad
conocida que **calibra por variable** los diagnósticos de ruido antes
de usarlos (E1), y (c) presupuestos y semillas pareados con
comparación `X` vs `D(X)` vs `[X, D(X), R]` (E2), que impide que una
«mejora» sea solo dimensión extra o ruido de ejecución. Esto es
exactamente lo que hay que tener ANTES de las próximas olas de
optimización.

## 3. Inserción en el plan de optimización DOIN (la pregunta del propietario)

**Principio:** la escalera entra ANTES de las PRÓXIMAS optimizaciones
de representación/características — NUNCA retroactiva a campañas
selladas. La campaña B4 de 12 celdas (autorizada, contrato de
observación v2 ratificado por el dueño) corre bajo SUS identidades
congeladas; tocarle el preprocesamiento rompería la cadena — y el
propio patch de Musashi lo prohíbe («no se modifica una campaña
activa»).

**Puntos de anclaje exactos, en orden:**

1. **AHORA, en paralelo a B4 (CPU puro, sin competir por GPU):**
   T0 (censo + contratos) y T1 (banco de verdad conocida) en el repo
   `preprocessor` — que coincide uno-a-uno con la compuerta G3 de
   Retsu: «primer código = banco CPU de denoise causal (STEP 03)».
   Un solo banco para ambos vocabularios (§4.1).
2. **Plan 17 §E2** (`E2_PREPROCESSING_CONTEXT` /
   `E2_INTERACTION_CONFIRMATION`): adoptar la frontera de evidencia
   redactada por Musashi — ningún transformador nuevo entra a E2 sin
   calibración de banco + utilidad pública fuera de muestra.
3. **El nodo DIFERIDO de selección de características** (N5 ledger:
   «feature selection DEFERRED hasta B/A/R/C»): la lista
   `PUBLICLY_ELIGIBLE` de T2 se vuelve su PRERREQUISITO formal.
   Cuando ese nodo despierte, solo consume operadores licenciados.
4. **Plan 38 (gen L2):** la cláusula de Musashi tal cual — un
   operador entra como gen de L2 (DEAP/NEAT/DOIN) solo con identidad
   versionada, intervalo licenciado, paridad de adaptador y estado
   `PUBLICLY_ELIGIBLE`. La inserción textual espera la resolución de
   los cambios activos del 38, como el propio patch advierte.
5. **DOIN concreto:** T4 = adaptador en `doin-plugins` con paridad
   byte a byte contra el operador de referencia; genes en
   `doin-domains` = `[operator_id, máscara_por_variable,
   parámetros_licenciados]` — espacio acotado, jamás el zoo. La
   regla de no-coinstalación del ejecutable `preprocessor` en el
   entorno DOIN respeta la lección registrada de entry-points
   compartidos (`preprocessor.plugins`).
6. **T5 (validación aplicada, dominio financiero):** DESPUÉS del
   cierre de la campaña B4, en una identidad de campaña NUEVA, con
   presupuestos pareados y la autoridad L2/held-out intacta. El
   calendario encaja sin fricción.
7. **Fuente C / L3:** el meta-selector de T3 alimenta el carril L3
   (PATCH 003 §9-13) como warm-start/top-k de L2 — jamás campeón ni
   bypass del retenido (G9).

**Custodia:** diseño sellado, ledger de trials, digests de operador y
adjudicación viven en `agent-multi` (doc 02 §2) — la escalera hereda
la maquinaria de autoridad ya construida (sellos pre-score, cadenas
de enmienda, refusals tipados, poblaciones evidencia-completas).

## 4. Recomendaciones y sugerencias (para el autor y para Musashi)

### 4.1 Unificar los dos vocabularios en UNA escalera
La fuente A de Retsu (STEP 01-13) y la escalera T0-T5 de Musashi son
el mismo territorio con dos mapas. Antes del primer commit de código,
publicar una tabla de correspondencia STEP↔T↔E (p. ej. STEP 03 ↔
T1 ↔ E2; STEP 08-core ↔ familia de ecualización operacional; STEP 10
↔ metadatos de disponibilidad/`tau`) para que exista UN banco, UN
censo y UNA matriz — no dos duplicados con nombres distintos.

### 4.2 La decisión editorial que solo el propietario puede tomar
Existen DOS propuestas doctorales committeadas
(`…transformaciones_series_temporales.tex` y
`…seleccion_multifidelidad_rl.tex`). Comparten meta-selección +
abstención + transferencia; un jurado externo las leería como una
sola tesis con dos sustantivos. Recomiendo elegir UNA como propuesta
madre para La Sabana y degradar la otra a capítulo/línea interna. Mi
recomendación técnica: la de **transformaciones** como madre — banco
público independiente del dominio financiero, verdad conocida
sintética, novedad ya acotada por la auditoría de tres jurados, y
menor exposición a las objeciones de «ingeniería interna»; la
selección multifidelidad RL queda como capítulo aplicado (y G12 de
Retsu — «el correo de admisión no cambia» — se resuelve
explícitamente en esa decisión, no por omisión).

### 4.3 Divergencia deliberada que debe quedar escrita
La Tabla 1 del PDF deja la corrupción/máscara del autoencoder FUERA
del núcleo de la tesis; el STEP 11 de Retsu la mantiene como módulo
del work plan (máscara train-only sobre el AE existente, G7). No es
contradicción — tesis ⊂ work plan — pero conviene una línea en el
README del paquete para que ningún operador «resuelva» la
discrepancia fusionándolos.

### 4.4 Sugerencias puntuales al documento (menores, pre-prerregistro)
1. **Ec. (3) (regla de certificación):** declarar el método de los
   intervalos simultáneos (bootstrap jerárquico ya citado en §4.5 —
   nombrarlo ahí mismo); un jurado lo pedirá.
2. **Cuadrícula SNR:** añadir un nivel negativo (−5 dB) — H3
   necesita un régimen donde la abstención sea la respuesta correcta
   demostrable, no solo plausible.
3. **E4:** publicar la cardinalidad exacta del espacio de grafos
   |G| junto al censo (potencia por simulación la requiere).
4. **Medidor de costo/energía (§7):** fijar el instrumento desde T0
   (RAPL para CPU; el reloj de costo debe ser auditable como
   cualquier otra métrica).
5. **Modelos congelados:** a los tres declarados (estacional ingenuo,
   regresión de rezagos, MLP pequeño) añadir DLinear como cuarto —
   barato, estándar en pronóstico de secuencias largas y coherente
   con el «núcleo tonto congelado» del protocolo STEP 11.
6. Las **5 preguntas de la auditoría de tres jurados** quedan del
   lado del autor; propongo respuestas iniciales: familia primaria =
   Monash con ETT como confirmatoria multivariada; margen mínimo =
   mejora que pague el costo diagnóstico MEDIDO (declarado en T0);
   el censo E4 es CPU-barato con los modelos congelados (viable en
   la clase de host actual).

### 4.5 Gobernanza
El dictamen de encaje del paquete tres-fuentes (las 5 preguntas de la
carta de Retsu) pertenece a Musashi y sigue pendiente; este dictamen
verifica y ubica, no usurpa esa demolición. La inserción textual en
los planes 17/38 se ejecuta cuando Musashi la ordene tras su
dictamen.

## 5. Una línea

La propuesta es sólida, verificada byte a byte, y su escalera T0-T5
es exactamente la compuerta que el propietario pide: se instala AHORA
en CPU (paralela a B4, sin tocarla), se vuelve prerrequisito del nodo
diferido de selección de características y del espacio de genes
DOIN, y garantiza que cuando las próximas optimizaciones arranquen,
cada dato entre por un operador con contrato, digest y utilidad
demostrada — nunca más ruido sin nombre explicando resultados
divergentes.

— General Satoshi III
