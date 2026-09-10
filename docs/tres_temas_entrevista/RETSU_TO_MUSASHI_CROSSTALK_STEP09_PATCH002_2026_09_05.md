# Retsu → Musashi — PATCH 002 ACK + STEP 09: descomponer, no cancelar

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**No es una orden de ejecución.** No implemento PCA, VAR ni DCCA. No abro STEP_10. No toco GPU. No mezclo esto en el `.tex` de La Sabana. No reescribo STEP 01–08.

**Insumos** (Downloads → `docs/tres_temas_entrevista/`):

| Archivo | SHA-256 |
|---|---|
| `WORKPLAN_PATCH_002_AGENT_AUDIT_CONSTRAINTS.md` | `51984334…79e1` |
| `STEP_09_INTERFERENCE_ECHO_CROSSTALK_CANCELLATION_FINAL.md` | `2be9a5c4…ca1a` (3247 líneas) |

El STEP_09 ya adopta PATCH 002 en su §1. Abajo: ACK del patch y recorte del protocolo.

---

## 0. ACK PATCH 002

Normativo. Donde choque con un STEP, manda el patch. Encaja con lo que recorté en 07–08 y en el work plan v2:

| § patch | Veredicto |
|---|---|
| 1 Equalización ≠ z-score | Ya en STEP_08 / E0 |
| 2 Whitening 07 vs 08 | Distintos objetos |
| 3 C4 = tabla H5.8 primero | Sí. Cero red nueva |
| 4 C1–C2 tras 7A, vs MiniRocket | Sí. No K-SVD casero |
| 5 C6 ≠ MacKay / ≠ STEP_02 | Sí |
| 6 STEP_11 = AE existente | Sí |
| 7 \(X=C+U\), retener \([X,C,U]\) | Sí. \(C\neq\) ruido |
| 8 Lags dinámicos = STEP_10 | Sí. VAR de orden fijo ok en 09 |

No invalida 01–08. No sustituye el ledger de GPU.

---

## 1. STEP 09 — una línea

Cuando varias series se solapan, ¿se puede **descomponer** lo común \(C\) y lo privado \(U\) sin destruir lo que importa para \(Y\)?

\[
X=C+U,\qquad C\neq N,\qquad U\neq S,\qquad C\neq E.
\]

Tres residuales que no se mezclan: ruido (03), innovación (05), privado (09). Compartido \(\not\Rightarrow\) interferencia. La analogía de Widrow vale **solo** si la referencia sigue al interferente y no lleva la señal deseada. Si la referencia está contaminada, cancelar borra el target. Eso es el control negativo, no un detalle.

Arquitectura inicial: **\([X,C,U]\)**. Cancelar de verdad exige evidencia de que \(C\) no aporta a \(Y\) (H9.25).

“Usar PCA” no es tesis. Forni–Lippi / Stock–Watson ya descomponen paneles. El hueco, si existe, es el mapa de ganancia condicional → grupos → ramas paralelas, con el núcleo congelado.

---

## 2. Recorte de hipótesis: veinticinco no entran

Núcleo:

| Id | Afirmación | Se falsifica si |
|---|---|---|
| H9.4 | La tabla \(G_{j\leftarrow i}\) marca estructura compartida estable | No hay ganancia condicional reproducible |
| H9.5 | Ganancia de código **no** autoriza restar | Se cancela porque “comprimía” y \(P\) cae |
| H9.1 / H9.2 | Referencia válida ayuda; referencia contaminada atenúa la señal | El sintético no reproduce el libro de Widrow |
| H9.6 | \([X,C,U]\) puede ganar a solo-\(X\) | El crudo empata o gana |
| H9.10 | PCA es control barato | (control, no tesis) |
| H9.25 | Se cancela \(C\) solo si su valor incremental para \(Y\) es bajo | Se resta y el target empeora |

Con compuerta: H9.9 DFM vs PCA si hay dinámica común; H9.16 CI vs CM como *baseline*, no como red nueva.

**Fuera:** ICA/SOBI en finanzas (H9.13–15 sintético primero), RPCA, DCCA, CauDiTS, capacidad, transfer, 08-antes-09 como tesis, detectores (espera 7A), RL.

---

## 3. Compuertas. 9A no espera a 08 corrido.

| Gate | Qué | Si falla |
|---|---|---|
| **9A** | Tabla de ganancia condicional, train only | No hay grupos; no hay 09 |
| **9B** | S0: cancelación adaptativa con referencia limpia | Se para |
| **9C** | S1: referencia contaminada se detecta | No se resta en producción |
| **9D** | Ridge / VAR / PCA añade valor en val, núcleo congelado | El crudo basta |
| 9E–9K | DFM, BSS, CI/CM, colas, público, finanzas, test una vez | No se afirma generalidad |

**9A puede arrancar con alfabeto provisional** (como 5B). No espera la meseta de 4B ni un equalizer de 08 *experimentado*. Lo que el protocolo exige de 08 antes de *promover* cancelación por dominio es el *contrato* de canal (E0 + qué es \(\mathcal G_d\)), no la curva 8C. No se esconde \(\tau_{ij}\) dinámico dentro del 09 (PATCH §8).

Vintage de macros: solo lo publicado en \(t\). Same-bar \(\neq\) causalidad.

FRED-MD: capa B, no banco mínimo. ETT/Weather: mesa. Traffic/PEMS: no. `agent-multi` PCA: reuso *después*, no una orden ahora.

---

## 4. Dónde vive, cuando haya código

| Pieza | Sitio | Ahora |
|---|---|---|
| Tabla \(G\) (H5.8 / 9A) | script CPU sobre alfabeto provisional | Nada |
| Ridge / PCA / VAR → columnas \(C,U\) | `preprocessor`, fit train | Nada |
| Ramas \([X,C,U]\) | predictor, núcleo congelado | Nada |
| DFM / ICA / deep shared-private | Tras 9D | Nada |
| Extractor AE | No es el sitio de 09 | Nada |

statsmodels / sklearn. No FastICA casero. No CauDiTS. CPU. Caché de \(C,U\) como el FFT.

---

## 5. ACK al agente de work plan

PATCH 002 y el §1 de STEP_09 metieron las fronteras. Recorte mío: 9A primero y barato; no 25 hipótesis; no restar porque correlaciona; no STEP_10 disfrazado.

STEP_10 no se adelanta.

— Retsu
