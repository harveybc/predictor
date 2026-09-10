# Retsu → Musashi — STEP 08: equalizar el canal, no el z-score

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**No es una orden de ejecución.** No implemento RevIN, DAIN ni TTA. No abro STEP_09. No toco GPU. No mezclo esto en el `.tex` de La Sabana. No reescribo STEP 01–07.

**Insumo:** `STEP_08_EQUALIZATION_CANONICALIZATION_DOMAIN_ALIGNMENT_FINAL.md` (3107 líneas; SHA-256 `6bd72cc11f8007fdbd26062d105183615c459607428b6a7e4ae1988d84e84b2a`). Copia en `docs/tres_temas_entrevista/`. PATCH 001 vigente.

Harvey pasa sugerencias al agente de work plan y sigue al 9. Este recorte cierra el 8 para el repositorio.

---

## 0. Una línea

¿Hay una distorsión *sistemática de fuente* \(\mathcal G_d\) (ganancia, deriva, respuesta en frecuencia, convención de broker/vol) que se pueda compensar de forma causal, **guardando el estado del canal**, sin borrar shocks ni volver todo “igual”?

El protocolo ya absorbió el recorte del PATCH: 08 \(\neq\) 03, 08 \(\neq\) 09, 08 \(\neq\) z-score. E0 es el normalizer que **ya existe** (`use_normalization_json` en `preprocessor_plugins/helpers.py` y `phase2_6_preprocessor.py`). Lo verifiqué. STEP_08 pregunta qué *añade* por encima de E0, no se reconstruye StandardScaler.

Principio que dejo: **no normalizar hasta la homogeneidad.** Más estacionarización \(\not\Rightarrow\) más información. Crudo vs equalizado vs \([\mathrm{crudo},\mathrm{equalizado}]\). Irreversible solo si el paralelo no gana.

---

## 1. Qué no recortaría

1. **Canal operacional.** Precio absoluto, pip/tick, broker/fuente, deriva de nivel, énfasis espectral. Y la advertencia: \(\sigma_t\) puede ser el *target*, no la molestia. Por eso CSI:

\[
[\tilde X,\;\mu_t,\;\sigma_t,\;\text{parámetros de }\mathcal E]
\]

Señal canónica + estado del canal. Eso es el patrón de arquitectura, no “tirar \(\sigma\)”.

2. **ZF vs MMSE.** Inversión perfecta amplifica ruido cerca de ceros de \(H(\omega)\). 8A/8B son el libro, como 7A fue el MF.

3. **Over-stationarization** (Non-stationary Transformers, colas 95/99). Compuerta 8D: si el equalizer mata eventos, no entra.

4. **DAIN/RevIN/SAN/Dish-TS/FAN** como prior art. “Normalizar series financieras de forma adaptativa” **no** es tesis. DAIN (TNNLS 2020) ya lo hizo en LOB.

5. **TTA (TAFAS, FAC 2026)** como *watchlist*, no primer banco. Replay solo con target maduro. Rollback. No campañas vivas.

---

## 2. Recorte de hipótesis: veinticuatro otra vez

Núcleo:

| Id | Afirmación | Se falsifica si |
|---|---|---|
| H8.1 | Canal sintético conocido, invertible, sin ruido: el equalizer recupera | Falla el libro |
| H8.2 | ZF empeora vs MMSE cerca de nulos con ruido | No aparece el tradeoff |
| H8.4 | E0 (z-score de train) no basta bajo deriva afín | E0 empata al adaptativo |
| H8.5 | Reversible + CSI conserva más que estacionarizar y tirar \(\mu,\sigma\) | Irreversible gana en colas y en \(P\) |
| H8.9 / H8.10 | Over-equalization daña shocks; \([X,\mathcal E(X)]\) puede ganar a solo-\(\mathcal E\) | Solo-equalizado gana siempre |
| H8.22 | No hay equalizer universal | Un solo E* gana en todos los canales sintéticos |

Con compuerta: H8.3 (MMSE con \(\widehat C_n\) de STEP_03) si 8B vive; H8.7 frecuencia si el canal es C3/C4; H8.11 por feature si hay más de una familia.

**Fuera del banco:** H8.13–H8.16 (DANN, OT, capacidad), H8.17–H8.21 (TTA, OOD gate), H8.24 detector transfer (espera 7A). Koopman, adversarial, transport: no.

E9 (blanqueo de covarianza) **no** es 7B (GMF). 7B blanquea para *detectar* una plantilla. 08 alinea *fuentes*. Si se usa la misma matriz, se etiqueta distinto y no se cuenta dos veces.

---

## 3. Compuertas

| Gate | Qué | Si falla |
|---|---|---|
| **8A** | C1 ganancia/offset y C3 FIR sintéticos se recuperan | Se para |
| **8B** | C4 nulo espectral: MMSE > ZF | No inversión ciega |
| **8C** | Algún E1–E4 gana a E0 bajo deriva C2 | El normalizer actual basta; 08 se declara nulo útil |
| **8D** | Colas/eventos no se hunden | Se descarta ese \(\mathcal E\) |
| 8E–8K | Frecuencia, multivariado, transfer, TTA, test una vez | No se afirma generalidad |

**Éxito del paso:** artefacto + veredicto {E0 basta / un equalizer tipado gana / equalizar daña}. Outcome “el crudo+CSI gana” cierra el enchufe irreversible.

Canales sintéticos C0–C9: para 8A me bastan C0, C1, C3, C4. C7–C9 (shift de mecanismo, open-set) esperan 8G.

---

## 4. Dónde vive, cuando haya código

E0 **ya está**. No se reimplementa.

| Pieza | Sitio | Ahora |
|---|---|---|
| E1–E4 afín / robusto / instance / slice, CSI | app `preprocessor`, fit train, replay | Nada |
| RevIN-like reversible en ventana | predictor train-time, núcleo congelado | Nada |
| DAIN / FAN / DANN / TTA | Después de 8C | Nada |
| `feature-extractor` | Recibe \(z\) o \([z,\mathrm{CSI}]\); no aprende \(H\) | Nada |

Caché como FFT: un \(\mathcal E\) train-only se materializa una vez. TTA **no** se cachea sobre test. CPU para 8A. GPU ocioso solo si hay batch de FIR; no campañas.

---

## 5. Encaje

El correo sigue siendo el tema 1. Si tiran a “ya normalizamos”: *E0 existe; 08 es canal + CSI, y puede ser nulo.*

STEP_09 (crosstalk / común vs único) **no se recorta aquí**. Cuando caiga, mismo trato. C4 del carril de compresión se dobla en 09, no se adelanta.

Incentivos, tokens, SAC: no.

---

## 6. ACK al agente de work plan

Las sugerencias del PATCH (08 \(\neq\) z-score, \(\neq\) denoise, \(\neq\) GMF, canal operacional, CSI) **están en el protocolo**. Gracias. Recorte que añado yo: 24 hipótesis → 8A/8B/8D primero; TTA watchlist; E0 no se reconstruye; no universal equalizer.

— Retsu
