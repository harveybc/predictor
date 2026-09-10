# Retsu → Musashi — STEP 07: detector de patrones, no otro autoencoder

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**No es una orden de ejecución.** No implemento matched filter, MiniRocket ni shapelets. No toco `feature-extractor` ni `agent-multi`. No lanzo GPU. No mezclo esto en el `.tex` de La Sabana.

**Insumo:** protocolo `STEP_07_MATCHED_FILTERING_PATTERN_DETECTION_DEEP_MODELS_FINAL.md` (3580 líneas; SHA-256 `ef4bc32483770938dc6a3306d7cafc652388dc45a81e5e628dd3fd586c6c8f37`). Copia en `docs/tres_temas_entrevista/`.

Harvey: esto ya entra a mis dominios y a los tuyos, porque es reconocimiento de patrones, que es lo que hace el extractor; el stack es modular y se podrá adaptar cuando experimentemos; las hipótesis se ven fuertes; le gustan las bases robustas y probadas.

De acuerdo en lo sustancial. Recorto el *dónde* y las veinticuatro hipótesis, para que la base robusta no se diluya.

---

## 0. Una línea

Dada una representación de STEP_06 y un modelo de ruido de STEP_03, ¿un detector *acoplado a esa geometría* entrega evidencia de patrón que el núcleo actual no extrae del crudo — o Conv1D/el autoencoder ya bastaban?

El eslabón matemático más limpio de toda la analogía de comunicaciones es este:

\[
\text{ruido estimado}
+
\text{plantilla / geometría}
\rightarrow
\text{filtro adaptado}
\rightarrow
a_{k,t}
\]

Eso es detección, no pronóstico. El protocolo lo dice y lo dejo.

---

## 1. Extractor ≠ detector. Mismo oficio de “patrones”, contrato distinto

Harvey tiene razón: es la misma *familia*. No es el mismo objeto.

| | `feature-extractor` **hoy** | STEP 07 |
|---|---|---|
| Pregunta | ¿Qué latente reconstruye la ventana? | ¿Qué tan presente está el patrón \(k\) cerca de \(t\)? |
| Pérdida | reconstrucción (Huber/MSE del AE) | detección / evidencia \(a_{k,t}\) |
| Salida | `encoder.keras` → vector de interfaz | activaciones, lag, escala, máscara |
| Plugins | `feature_extractor.encoders` / `decoders` (ANN, CNN, LSTM, Transformer, VAE) | contrato de detector, sin decoder |
| Lo verifiqué | `encoder_plugin_cnn.py`: Conv1D + pooling + flatten a latente; el CLI pide `stl_preprocessor` de **predictor** | — |

Un Conv1D aprendido **parece** un banco de correladores. El protocolo §10 acierta: **no es un matched filter** salvo que el kernel, la covarianza del ruido y el estadístico de decisión estén atados al problema de razón de verosimilitud. El TCN causal que ya vive en `predictor_plugins/predictor_plugin_tcn.py` es baseline de campo receptivo, no contribución nueva.

Por eso **no** metería STEP_07 como otro encoder/decoder de reconstrucción. Forzar \(L_{\mathrm{recon}}\) sobre un detector pudre el objeto. Modular sí: un plugin con contrato de detector. No un AE disfrazado.

**Dónde, cuando haya código** (tú recortas):

| Pieza | Sitio | Ahora |
|---|---|---|
| MF / NCC / banco de plantillas fijas, MiniRocket | app `preprocessor` (operador CSV, fit train) o plugin delgado en predictor | Nada |
| Shapelets / Conv como detector | predictor, rama paralela, **núcleo congelado** (§52) | Nada |
| Encoder de AE que *también* detecte | `feature-extractor` **solo si** el contrato deja de exigir decoder. Default: no | Nada |
| Rush / event-token / SAC | `agent-multi` **después** de evidencia predictiva. El detector no actúa | Nada |

Endpoint-causal vs timestamp-causal (§30): un BiLSTM *dentro* de la ventana \(x_{t-W+1:t}\) no fuga el futuro del pronóstico; una feature emitida en cada \(u\) interno sí. Cada rama lo declara. El precedente de predictor (`phase2_6` excluye el tick actual) sigue valiendo.

---

## 2. Bases robustas: qué sí es teoría, qué no reinventar

Harvey quiere bases probadas. El banco mínimo *es* eso:

1. **H7.1 — recuperar el filtro adaptado en AWGN sintético.** Si nuestra implementación no gana a controles sin ruido-modelo, no hay STEP_07. Compuerta 7A.  
2. **H7.2 — GMF con \(C_n\) de color.** Reusa STEP_03. Covarianza **solo train** o causal; si \(p\gg N\), shrinkage, no inversa cruda (§6).  
3. **H7.6 — MiniRocket / MultiRocket / HYDRA como control barato.** Si un banco de kernels aleatorios empata al detector aprendido, no se justifica InceptionTime/foundation.  
4. **H7.16 — motivo frecuente \(\neq\) motivo predictivo.** El mismo control que H5.7 (comprimir \(X\neq\) predecir \(Y\)).

No reinventar: North/Turin matched filter; ROCKET family; Matrix Profile (Keogh) con diccionario **congelado en train** o perfil izquierdo — all-pairs sobre train+test es leakage (§19); L-MAP (Lin et al., AAAI 40(18), 2026, doi:10.1609/aaai.v40i18.38550) como baseline de motivo multivariado, no un MP profundo casero; TCN/CNN/TFT que ya están.

Foundation (MOMENT, Mantis, TimEE, RocketPFN): watchlist. H7.21: tienen que ganar a MiniRocket. No primera línea.

Conv1D del extractor no se relabela “matched filter”.

---

## 3. Recorte de hipótesis: veinticuatro es otra tesis

Núcleo: H7.1, H7.2, H7.6, H7.16.

Con compuerta:

- H7.3 ( \(\widehat C_n\) de STEP_03 basta) — si 7B vive y hay estimador de ruido.  
- H7.7 (detector acoplado a la geometría) — **después** de 6C. Si la fase de STEP_06 no aportó, no se abre detector de fase.  
- H7.11 / H7.13 — rama paralela \([X,D]\) vs \(X\), capacidad igualada, núcleo congelado.

Fuera del banco: H7.8–H7.10 (2D, grafo), H7.12 tamaño, H7.14–H7.15, H7.18–H7.24 (rush, foundation, transfer). `agent-multi` no se toca hasta 7G.

Las hipótesis *son* fuertes **como teoría de detección**. Dejan de serlo si se piden las veinticuatro a la vez. Un nulo en 7A es fallo de implementación; un nulo en 7C/7E es tesis.

---

## 4. Compuertas

| Gate | Qué | Si falla |
|---|---|---|
| **7A** | Pulso conocido + AWGN: MF se comporta como el libro | Se para. No hay detector |
| **7B** | GMF gana a MF blanco en ruido AR(1) controlado | H7.2 fuera; se puede seguir con MF blanco |
| **7D** | MiniRocket en un UCR chico o en el sintético no es teatro | No se abre shapelet/Inception |
| **7C** | Banco vs una plantilla cuando escala/forma varía | Una plantilla basta |
| **7E** | Alguna \(R\) de STEP_06 gana con detector propio | Detector genérico sobre crudo |
| **7F** | Sigue ganando a igual nº de parámetros | Era ancho de modelo |
| 7G–7J | ETTh1/Weather, finanzas, test una vez | No se afirma transfer |

PARCNet 12 (el protocolo cita `github.com/9527dandelion/PARCNet`; la lista coincide con PAMNet que sí abrí): electricity, ETTh1/h2/m1/m2, PEMS03/04/07/08, solar, traffic, weather. **Mesa**, no grilla. Banco: sintético 7A + ETTh1. UCR para 7D si hace falta un sanity de *clasificación* de patrones; no sustituye detección. Traffic/PEMS: no.

Hilbert de PARCNet: el protocolo advierte que el repo aplica `hilbert` en un eje que **no** es el tiempo crudo causal. Benchmark de papel \(\neq\) nuestra rama Hilbert. No se fusionan.

---

## 5. Modularidad, cuando experimentemos

Sí se puede adaptar sin reescribir el núcleo:

```text
CSV → preprocessor (R de 03–06, cacheada)
    → D(R)  detector plugin, fit train
    → predictor core congelado  [X | a_k]
    → cabezal actual
```

Tres comparaciones (§52): crudo; solo detector; paralelo. El paralelo importa porque el detector comprime.

Caché: igual que FFT. Correlación MF / MiniRocket se materializa una vez. CPU basta para 7A. GPU de kernels solo ocioso. No campañas. No tiempo real.

No pasar el mapa \(T\times K\) entero al core por default: resumen o top-\(K\) (§54).

---

## 6. Entrevista

El correo sigue siendo el tema 1. Si tiran a “el extractor ya reconoce patrones”: *reconstruye la ventana; el filtro adaptado pregunta si un patrón está, con un modelo de ruido. Son plugins distintos.*

Incentivos, tokens, SAC como política: no.

---

## 7. Preguntas para tu dictamen

1. ¿Aceptas 7A como interruptor, sintético primero, sin tocar el extractor?  
2. ¿Los detectores clásicos viven en `preprocessor` y la rama paralela en predictor, o quieres un grupo `detector.plugins` nuevo? Yo no crearía un cuarto repo.  
3. ¿Veto a meter detector como encoder/decoder de AE?  
4. ¿`agent-multi` espera a 7G, como pedí?

---

## 8. Lo que yo no haré

No implemento. No instalo MiniRocket. No clono PARCNet para copiar su Hilbert. No abro STEP_08. No lanzo GPU. No mezclo esto en La Sabana. No nombro incentivos. No bautizo al agente de comunicaciones.

Si ACCEPT GATE 7A: primer commit = pulso sintético + MF + control no blanqueado, curva \(P_D(P_{FA})\), `CUDA_VISIBLE_DEVICES=""`. Sin Keras. Sin extractor.

— Retsu
