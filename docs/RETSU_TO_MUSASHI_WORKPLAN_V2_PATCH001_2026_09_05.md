# Retsu → Musashi — PATCH 001 y work plan v2: mapa sí, “cerrado” no

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**No es una orden de campaña.** No reescribo STEP 01–07. No implemento equalización, sparse coding ni STEP 08. No toco GPU. No mezclo esto en el `.tex` de La Sabana. Este plan **no sustituye** el ledger de trading/GPU.

**Insumos** (Downloads → `docs/tres_temas_entrevista/`, hashes verificados):

| Archivo | SHA-256 |
|---|---|
| `MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v2.md` | `cc044782…992d` |
| `WORKPLAN_PATCH_001_HISTORICAL_CHAIN_AND_COMPRESSION_EXTENSIONS.md` | `db326402…9a7d` |
| `COMPRESSION_ANALOGIES_FORMALIZATION_FOR_AGENT_AUDIT.md` | `60c713e7…94f0` |

El agente de work plan hizo lo que pedía Harvey: un mapa único, la deuda histórica de la cadena, y el carril de compresión aparte de la transmisión. Abajo: qué acepto, qué recorto, y las siete preguntas del §13 de analogías.

---

## 0. Dictamen en una línea

**ACK PATCH 001** como índice de *protocolos*. STEP 08 = equalización, no pulse shaping. El carril C1–C7 se registra como transversal, **no** como tesis paralela ni como paso entre 07 y 08.

**RECORTE de estado:** “Protocol closed” en la tabla §9 del master es falso operativamente. 01–07 están *especificados y recortados*. Cero plugins. Cero curvas. El primer *hecho* sigue siendo el banco CPU de STEP_03, no escribir el 13.

---

## 1. Qué acepto del patch

1. **No reescribir** los STEP 01–07 por un cambio de hoja de ruta. Los recortes RETSU siguen vigentes.  
2. **Orden restante:** 08 equalización → 09 interferencia/crosstalk → 10 sincronía → 11 redundancia controlada → 12 adaptativo → 13 asignación multi-rama. Sustituye el “pulse shaping” que STEP_07 dejaba como siguiente. Pulse shaping, si aparece, se discute *dentro* de 08/10 contra el modelo de datos, no como paso suelto.  
3. **Carril de compresión transversal**, no insertado entre 07 y 08. Varios conceptos cruzan etapas.  
4. **Contrato de dominio:** no un juego fijo de transforms; un contrato para descubrir, falsar y combinar plugins. Eso es compatible con el stack (preprocessor / extractor / predictor).  
5. **Disciplina §2 del master:** causalidad, qué destruye, \(P\) fuera de muestra, capacidad igualada, falsable. Ya era la regla de 03–07.

Harvey: no perder tiempo en barridos de modelos mientras los datos no estén en forma. **De acuerdo en el espíritu.** El medidor de “en forma” es un núcleo *congelado y tonto* (DLinear / ANN diario), no la ausencia de modelo. Sin ese palo, “perfeccionar datos” no termina.

Este documento es el **índice de investigación informacional**. No es el work plan de Paper/Demo/A100. Si hay conflicto de hueco, tú dispones el calendario; yo no reasigno GPU.

---

## 2. Recortes al master y al patch

**Estado.** Sustituir en la cabeza de quien lea:

| Lo que dice v2 | Lo que es |
|---|---|
| STEP 01–07 protocol closed | Protocolo escrito + recorte RETSU. Experimentos **no** empezados |
| Compression lane formalized for audit | Auditado aquí. Implementación **no** |
| STEP 08 next | Siguiente *protocolo*. No siguiente plugin |

**“Information to knowledge”.** Operacional: lo que el extractor convierte en parámetros. No se llama inteligencia. No se fusiona con \(C\)/\(U^*\) del tema 2. C6 (rate del latente) **no** es MacKay.

**STEP 08, antes de que lo escriban.** Equalización \(\neq\) el z-score que `plugin_normalizer` ya hace. El análogo útil es una *función de transferencia de fuente*: broker/instrumento/régimen de vol como “canal”, no un teléfono. Whitening para detección se queda en 7B. Flujos invertibles aprendidos no son el primer experimento. Si 08 es StandardScaler con otro nombre, no se abre.

**STEP 09 vs C4.** Compartido/único es el mismo objeto que side information. No dos tesis. PCA/ICA/residualización cruzada; no se etiqueta causalidad.

**STEP 11.** Añadir redundancia \(\neq\) quitarla (05). El AE de `feature-extractor` **ya es** el análogo de reconstrucción enmascarada / DAE. No se construye un segundo autoencoder “de canal”. 11 se recorta a: *¿el AE que ya existe, con corrupción controlada, endurece el latente?* Eso es experimento sobre código vivo, no un repo nuevo.

**STEP 12–13.** Controlador por SNR/OOD y presupuesto de ramas. Necesitan los medidores de 03–05. No se implementan para “evitar” el banco de ruido.

**C5 vs \(E\neq N\).** Residual jerárquico no es denoising. \(\hat X_0+R_1\) puede ser tendencia + innovación (05) o banda DWT (06). No se relabela el residuo de STEP_03.

---

## 3. Carril de compresión — respuestas al §13

### 1. ¿Qué cubre el código actual?

Casi nada de C1–C7 como plugins. Lo que *sí* existe y no se reinventa:

| Pieza viva | Relación |
|---|---|
| `preprocessor` normalizer A/B | No es 08. Escala, no canal |
| `feature-extractor` AE (CNN/LSTM/VAE) | C6/STEP_11 *sin* \(+\lambda R_Z\). Reconstrucción, no detector |
| predictor TCN/CNN/TFT | Detector aprendido, no MF (STEP_07 §10) |
| `agent-multi` event-token / rush | C7 más adelante, no ahora |
| STL / wavelet plots en resultados del extractor | 06, no sparse coding |

### 2. Prior art a reusar, no reescribir

- Sparse coding: DictionaryLearning / K-SVD / SPAMS. Olshausen–Field no se rediscubre.  
- CSC: biblioteca (p. ej. sporco), no un Conv1D “esparcido” a mano.  
- MiniRocket ya es el control de 7D; C2 tiene que ganarle.  
- Side information: \(H(X\mid Y)\) de STEP_05.  
- Residuos jerárquicos: DWT de 06.  
- Latent RD: Ballé et al. (hyperprior); no un entropy model casero en el primer commit.  
- Duration coding: run-length es trivial; el valor está en el *estado* (régimen/rush), que aún no está protocolizado como detector.

### 3. ¿Hipótesis independientes o parafraseo?

| Id | Veredicto |
|---|---|
| C1 sparse | Solapa STEP_07 plantillas/shapelets. Independiente *solo* si el diccionario es sobrecompleto y \(\alpha\) es la rama, no si es otro Conv1D |
| C2 CSC | Solapa 07; comparación útil **después** de 7A: CSC vs Conv1D vs MiniRocket |
| C3 successive refinement | Es STEP_12. No se abre ahora |
| C4 side info | STEP_05 H5.8 + STEP_09. Un objeto |
| C5 hierarchical residual | 03/05/06 con otro nombre si no se distingue \(N\) vs \(E\) vs detalle wavelet |
| C6 latent RD | Independiente y **tarde**. Extractor/core. No retrasa el banco de ruido |
| C7 duration | Independiente y tarde. Event-token |

### 4–5. ¿Standalone o doblar en 08–13?

No hay fase “C” de tres meses. Se *enganchan*:

- C4 diagnóstico → tabla de STEP_05 (sin código nuevo).  
- C2 → 07 después de 7A.  
- C4-H2 common/private → 09.  
- C3 → 12.  
- C6 → extractor, post-4B/5B y con AE ya entrenable.  
- C7 → event-token, post-7G.

### 6. Experimento mínimo para falsar

| Id | Mínimo |
|---|---|
| C1/C2 | Tras 7A: diccionario train-only vs MiniRocket vs crudo, DLinear, paralelo \([X,\alpha]\) |
| C4-H1 | \(H(X_j\mid X_{-j})\) vs \(H(X_j)\) en alfabeto provisional de 05. Si no hay ganancia, no hay agrupación |
| C4-H2 | Espera 09 |
| C5 | Si DWT de 06 ya separa escalas, no se abre otro residual |
| C6 | \(\lambda R_Z\) en el AE existente; frontera Pareto. No un extractor nuevo |
| C7 | Hasta que exista un estado simbólico causal |

### 7. Librerías

No reinventar Huffman (ya dicho), ni K-SVD, ni sporco, ni MiniRocket, ni el AE del extractor.

**Prioridad del §12 de analogías:** no la sigo al pie. Side-info primero *como tabla* (gratis). Sparse/CSC **no** antes de 7A. Latent RD no es “high-priority later” si “later” se lee como “antes de perfeccionar datos”: es *después* del medidor de SNR.

---

## 4. Sugerencias concretas (las que pediste)

1. En el master, una línea bajo la tabla: *closed = protocolo + recorte; open = experimento*.  
2. El primer commit de *código* sigue siendo STEP_03 causal (EWMA/trailing MA + contaminación post-split + DLinear). El mapa 08–13 no lo aplaza.  
3. STEP_08, cuando se escriba: una página de “qué no es” (z-score, GMF, denoise) y un canal operacional (instrumento / vol / fuente).  
4. STEP_11 se escribe contra el AE que ya existe, o no se escribe.  
5. C6 se anota en `feature-extractor` como extensión de pérdida, no como work package de predictor.  
6. No barrer TFT/PatchTST/SAC hasta 7A y una curva \(P(\mathrm{SNR})\) o \(P(b)\). El núcleo tonto sí se usa: es el palo, no el barrido.  
7. El carril C no entra al pitch de entrevista. El correo sigue siendo el tema 1.

---

## 5. ACK para que pasen a redactar STEP 08

PATCH 001 **aceptado** como hoja de ruta de protocolos, con los recortes de estado y de equalización \(\neq\) normalizer.

Carril C1–C7 **registrado**, no implementado.

STEP 08 **puede protocolizarse** (no implementarse) con el recorte del §2.

STEP 01–07 **no se reescriben**.

— Retsu
