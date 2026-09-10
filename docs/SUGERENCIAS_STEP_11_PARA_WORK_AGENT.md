# Sugerencias para STEP 11 — pasar al work agent al abrir el protocolo

**De:** Retsu  
**Para:** Harvey → work agent  
**Cuándo:** al empezar STEP 11. STEP 10 no se toca aquí.  
**PATCH 002 §6 es normativo.** Donde choque, manda el patch.

---

## 1. Una línea

STEP 11 pregunta si **redundancia deliberada** (máscara, corrupción, vistas múltiples) hace más robusto el latente **del extractor que ya existe**. No pregunta si hace falta otro autoencoder.

---

## 2. Qué no es

- No un segundo repo de AE.
- No DAE “desde cero” si `feature-extractor` ya reconstruye ventanas (CNN/LSTM/Transformer/VAE).
- No códigos de Hamming/Reed–Solomon metidos al CSV.
- No Huffman/ANS como input (eso murió en STEP 05).
- No C6 (\(L_{\mathrm{task}}+\lambda R_Z\)) disfrazado de canal. C6 es rate del latente, **después** del medidor de SNR; no es MacKay.
- No STEP 03: plantar ruido para *medir* degradación ≠ entrenar con corrupción para *endurecer*.
- No TTA ni campañas GPU vivas.

---

## 3. Contrato del experimento

Cadena:

```text
CSV (preproceso congelado)
  → extractor EXISTENTE (misma arquitectura, misma interfaz de latente)
  → núcleo tonto congelado (DLinear / ANN diario)
  → P fuera de muestra
```

Brazos, como siempre:

1. AE actual, sin corrupción extra (baseline).  
2. El **mismo** AE, corrupción/máscara **solo en train**, parámetros de corrupción fijados en desarrollo.  
3. Paralelo: latente limpio + latente robusto, si el core lo admite sin ensancharse.

Éxito del paso: curva + veredicto {ayuda / empata / empeora}. Un nulo es tesis. “Siempre regularizar con máscara” no lo es.

---

## 4. Corrupción causal y sin leakage

- Máscara / ruido / dropout de features: solo sobre \(x_{\le t}\) de **train**.  
- No usar val/test para elegir la tasa de máscara más de una vez; prerregistrar la grilla.  
- No enmascarar el target.  
- Si la máscara es temporal, no “agujeros” que se rellenan con futuro (mismo borde vivo que el denoise).  
- Replay en val/test: **sin** corrupción, o con corrupción etiquetada como estrés OOS, nunca mezclar las dos en la misma curva.

---

## 5. Prior art a citar, no a reimplementar

- Denoising AE / masked reconstruction (Vincent et al.; MAE como analogía de *máscara*, no de ViT).  
- El código vivo: `feature-extractor` `encoder_plugin_*` + `decoder_plugin_*`.  
- Consistency / multi-view: solo si el baseline de máscara en el AE actual no cierra 11A.

No HIVE-COTE, no foundation models, no un Transformer nuevo “de canal”.

---

## 6. Compuertas que pido

| Gate | Qué | Si falla |
|---|---|---|
| **11A** | El AE actual se reentrena en el banco chico (ETT o phase_1_daily) y reproduce conducta conocida | No se interpreta ninguna máscara |
| **11B** | Al menos un esquema de corrupción train-only mueve \(P\) de forma estable vs baseline | Nulo: 11 cierra, se queda el AE actual |
| **11C** | El latente no se ensancha; comparación a igual dimensión de interfaz | Era capacidad, no redundancia |
| **11D** | Colas / eventos no se hunden | Esa corrupción se descarta |

No PatchTST. No Traffic/PEMS. No `agent-multi`.

---

## 7. Hipótesis máximas (no quince)

- **H11.1** — Corrupción train-only en el AE existente puede mejorar \(P\) OOS vs el mismo AE sin ella.  
- **H11.2** — No es universal: hay un régimen de tasa de máscara/SNR donde empeora.  
- **H11.3** — El latente robusto \(\neq\) el residuo de STEP_03 ni el \(U\) de STEP_09.

Falla como tesis si se afirma “el preentrenamiento con máscara siempre ayuda”.

---

## 8. Dónde vive el código, si algún día se implementa

Solo `feature-extractor`: flag de corrupción en el train del plugin actual. Fit train. Misma `encoder.keras` de interfaz. Predictor no se engorda. Preprocessor no se toca para esto.

CPU. `CUDA_VISIBLE_DEVICES=""` salvo GPU ociosa autorizada para *ese* retrain. No campañas.

---

## 9. Relación con 03 y 08

| | STEP 03 | STEP 11 |
|---|---|---|
| Ruido | Dial de *evaluación* | Regularizador de *entrenamiento* |
| Dónde | preprocessor → entradas | extractor → latente |
| Nulo | Denoise no ayuda a \(P\) | Máscara no ayuda al latente |

No se cuentan dos veces.

---

Paso 10 (sync / \(\tau_{ij}\)): no se escribe aquí. Cuando llegue, recorte aparte.
