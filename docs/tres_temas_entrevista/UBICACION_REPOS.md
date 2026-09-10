# Dónde vive cada cosa

**Fecha:** 2026-09-05  
Hay **dos** “preprocessors”. No son el mismo código. DOIN usa uno; el repo viejo es el otro.

---

## Cadena real, de izquierda a derecha

```text
CSV crudo
    → preprocessor          (app de CSV: recorta, parte D1–D6, normaliza, JSON)
    → feature-eng           (features de dominio, si se usan)
    → feature-extractor     (entrena el autoencoder; encoder.keras)
    → predictor             (entrena el modelo predictivo; ventanas + STL en *sus* plugins)
    → prediction_provider / LTS / gym-fx / doin-plugins
```

Ningún eslabón hace el trabajo del de al lado. Los AGENTS.md de los tres repos lo dicen y coincide con el código.

---

## Los dos preprocessors

| | Repo `preprocessor` | `predictor/preprocessor_plugins/` |
|---|---|---|
| Ruta | `/home/harveybc/Documents/GitHub/preprocessor` | este repo, carpeta `preprocessor_plugins/` |
| Qué hace | Un CSV → D1–D6 + `normalization_config_*.json` | Ventanas, STL, log-returns, anti-naive, *en el momento de entrenar/inferir* |
| Cómo carga plugins | Por **archivo** en `app/plugins/`. No usa entry points en runtime | setuptools grupo `preprocessor.plugins` |
| Plugins | `plugin_default`, normalizer, unbiaser, trimmer, cleaner, feature_selector_* | `default_preprocessor`, `stl_preprocessor` |
| Último código | 2026-03-27; AGENTS.md 2026-08-16 | 2026-04-07 |
| ¿Lo usa DOIN? | **No.** `doin-plugins` hace `load_plugin("preprocessor.plugins", …)` y eso lo **registra predictor** | **Sí.** Inferencia y optimización DOIN cargan esto |
| ¿Lo usa feature-extractor? | No en el train. Lee CSVs ya normalizados | **Sí**, pide `stl_preprocessor` del grupo de predictor. Sin predictor instalado, el pipeline del extractor muere: *Plugin stl_preprocessor not found* |

El nombre del grupo `preprocessor.plugins` es compartido. gym-fx también lo usa. Co-instalar predictor + gym-fx + el app `preprocessor` mezcla el grupo. Un entorno por aplicación.

---

## El extractor que no recordabas

**Repo:** `/home/harveybc/Documents/GitHub/feature-extractor`  
Entrena autoencoders Keras (CNN, LSTM, transformer, VAE). Plugins `feature_extractor.encoders` / `decoders`. No es el preproceso. No es predictor. No tiene runtime DOIN: produce `.keras` que otros cargan.

Si el denoising va *antes* del extractor, el extractor no se toca: recibe CSVs (o ventanas) ya tratados.

---

## Dónde poner cada línea doctoral

| Línea | Documentos | Código cuando se implemente |
|---|---|---|
| 1 Selección de representaciones RL | `predictor/docs/` y `tres_temas_entrevista/01_*.pdf` | `gym-fx` / `agent-multi` cuando toque; **no** ahora |
| 2 Memorización / dimensionamiento | `tres_temas_entrevista/02_*.pdf` | laboratorio chico, no producción |
| 3 Ruido / SNR / denoise causal | protocolo STEP_03 + cartas en `tres_temas_entrevista/` | **operadores y banco CSV: repo `preprocessor`** |
| 3b Resolución de amplitud / cuantización | protocolo STEP_04 + recorte Retsu en `tres_temas_entrevista/` | **mismo sitio, después de 4A–4B; no hay plugin aún** |
| 3c Modelo de fuente / surprisal | protocolo STEP_05 + recorte Retsu en `tres_temas_entrevista/` | **diagnóstico 5A–5B con alfabeto provisional; ramas después; no hay plugin** |
| 3d Amplitud / fase / tiempo-frecuencia | protocolo STEP_06 + recorte Retsu | **caché causal de rFFT; 6A–6C; Hilbert de banda después; no hay plugin** |
| 3e Detector / filtro adaptado | protocolo STEP_07 + recorte Retsu | **7A sintético; MF/MiniRocket no son AE; rama paralela en predictor después; no hay plugin** |
| 3f Equalización / canal | protocolo STEP_08 + recorte Retsu | **8A sintético; E0 = normalizer vivo; CSI; no TTA; no hay plugin nuevo** |
| 3g Común / privado (crosstalk) | protocolo STEP_09 + PATCH 002 + recorte Retsu | **9A tabla primero; \([X,C,U]\); no restar; no hay plugin** |
| 3h Sync / lead–lag | protocolo STEP_10 + recorte Retsu | **10A contrato \(A_i(t)\); metadato \(\tau\) antes de shift; no DTW default; no hay plugin** |
| 3i Redundancia / endurecer AE | protocolo STEP_11 + recorte Retsu | **solo `feature-extractor`; flag train-only; 11A necesita `stl_preprocessor` de predictor; no hay flag aún** |
| 3j Router / AMC | protocolo STEP_12 + recorte Retsu | **elige modos validados; 12A necesita ≥2 modos; no MoE; no hay router** |
| 3k MIMO / presupuesto | protocolo STEP_13 + recorte Retsu | **composite 64:32:32:32; fusión congelada; no NAS; no hay barrido** |
| Carril C | PATCH 001–003 | **C1–C2 tras 7A. C6 tarde. C ≠ MacKay** |
| Carril eventos / L3 | PATCH 003 + review final | **panel+LP; L3 no bypasea L2; no DML para ruteo si el contrafactual se ve** |
| Propuesta a Satoshi | `SATOSHI_PROPUESTA_TRES_FUENTES_…` | **tres fuentes → sistema modular; no es el PDF de La Sabana** |
| Entrevista / PDFs de admisión | `predictor/docs/` está bien como archivo de las tres | — |

Las propuestas doctorales se quedan en `predictor/docs/` (y la carpeta `tres_temas_entrevista/`). No las muevas al preprocessor: no todo es preproceso, y mañana el correo sale de aquí.

---

## Este punto concreto (denoise causal, y cuantización después)

**Sí: mejor en `preprocessor` que en predictor**, para los *operadores* (CSV in → columnas o CSVs tratados + parámetros fit en train). Eso es exactamente el oficio de ese repo, aunque esté viejo. Es la ocasión de usarlo otra vez, con plugins nuevos, sin resucitar el unbiaser/boruta de 2025.

**Pero** si solo vive ahí, **DOIN y el train actual no se enteran**. El camino vivo es `predictor/preprocessor_plugins` vía `doin-plugins`. El enchufe a producción, *si* H2 gana, es un segundo paso: o los D4–D6 denoised que predictor ya lee, o una llamada delgada desde `stl_preprocessor` al mismo operador. No al revés: no engordar STL ahora.

Banco CPU mínimo: plugin nuevo en `preprocessor/app/plugins/` (EWMA / trailing MA, modos `causal` | `delayed_centered` | `centered_forecast_fill`) + script de contaminación post-split y curva. Sin TensorFlow si se puede. Sin GPU. Sin tocar campañas.

**STEP 04 (cuantización / companding).** Mismo repo, *después* de las compuertas 4A–4B del recorte. El barrido uniforme de \(b\) en modo reconstrucción puede compartir CSV y modelo tonto con STEP_03; el cruce SNR\(\times b\) usa la contaminación post-split; denoise\(\times Q\) espera a que H2 no sea nulo. WaveToken, TOTEM, TimeVQVAE e INT8 de pesos no se implementan. En el app `preprocessor` hay un puntero en `docs/README.md`; el protocolo largo vive aquí, no allá.

**STEP 05 (modelo de fuente).** Diagnósticos \(H_0\)/\(H_k\) pueden usar un alfabeto provisional; no esperan la meseta de 4B. Congelar “el mejor \(Q\)” sí. Ramas de innovación/surprisal esperan 5B. Huffman/ANS no son input. \(E\neq N\). BCT-X e IB/CIB son baselines ajenos, no el primer plugin.

**STEP 06 (fase / FFT).** Caché causal de rFFT/STFT/Hilbert-*por-ventana*. No Hilbert de la serie completa. No en el tick. GPU de kernels solo con dispositivo ocioso; no tocar campañas. Banco: ETTh1 + Weather + sintético. Los doce LTSF son mesa; Traffic/PEMS no se barren.

**STEP 07 (detector).** Familia de patrones, **contrato distinto** al AE de `feature-extractor` (reconstrucción vs \(a_{k,t}\)). MF/NCC/MiniRocket: operador, no decoder. TCN/CNN de predictor: baseline, no se reimplementan. Rama paralela con núcleo congelado. `agent-multi` después de evidencia. Matrix Profile solo con diccionario de train. No cuarto repo.

**PATCH 001–003 / master v2.** Cadena 01–13 cerrada como *preguntas*. PATCH 003: no serial obligatorio; carril causal de eventos; L3 warm-start; `causal-inference` no es infra. Primer código: banco CPU STEP_03. Carta a Satoshi: `docs/SATOSHI_PROPUESTA_TRES_FUENTES_CONOCIMIENTO_MODULAR_2026_09_05.md`. Musashi después de su dictamen.

**STEP 08 (equalización).** No es STEP_03 ni 09 ni el z-score. E0 = `use_normalization_json`. \(\mathcal G_d\) + CSI. TTA watchlist.

**STEP 09 (crosstalk).** \(X=C+U\). Primero tabla H5.8. \([X,C,U]\). No restar. No \(\tau_{ij}\) dinámico.

**STEP 10 (sync).** Reloj \(\neq\) publicación \(\neq\) información usable. \(A_i(t)\) primero. Metadato \(\tau\) + LOCK; no arrastrar el futuro.

**STEP 11 (canal / máscara).** Solo el AE de `feature-extractor`. Corrupción train-only, misma \(\dim Z\). Empate cierra.

**STEP 12 (AMC / router).** Elige entre paquetes congelados. IQS causal. Oráculo primero. No MoE.

**STEP 13 (presupuesto).** 12 = cuál modo; 13 = cuánto a cada rama. Ancla: composite `64:32:32:32`. Fusión/cabezales fijos. FLOPs ≠ latencia. No Shannon-bits. No NAS. HF del composite usa `padding=same` (nota causal, no el barrido).

---

## Qué no hacer

- No crear un cuarto repo.
- No poner el denoise, el cuantizador, el modelo de fuente, el FFT ni el detector-como-AE en `feature-extractor`.
- No mezclar este código en el `.tex` de La Sabana.
- No co-instalar el app `preprocessor` en el env de predictor/DOIN.
- No clonar la rama `wavetoken` para “completar” código que el autor no publicó.
- No alimentar el extractor con bitstreams Huffman/ANS/zstd.
- No calcular FFT/Hilbert sobre la serie completa y recortar.
- No recalcular el espectro en cada época: caché.
- No relabelar Conv1D como matched filter.
- No dejar que el detector actúe en SAC.
- No tratar STEP 01–07 como experimentos cerrados.
- No abrir sparse coding / C6 antes de 7A y del medidor de SNR.
- No sustituir el normalizer por “equalización”; E0 es baseline, no se reescribe.
- No TTA sobre campañas vivas ni sobre test sin target maduro.
- No cancelar \(C\) porque correlaciona; no usar macros sin vintage.
- No desplazar el futuro hacia \(t\); no DTW retrospectivo como feature.
- No un segundo AE para STEP 11; no ensanchar el latente.
- No un MoE/SAC como primer router; no rutea modos que no pasaron su gate.
- No llamar bits de Shannon a los canales Conv1D; no NAS en STEP 13.
- No construir un segundo autoencoder para STEP 11.
- No parar ni reasignar GPU de campañas vivas.
