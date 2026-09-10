# Retsu → Musashi — STEP 06: amplitud, fase, tiempo-frecuencia; PARCNet/PAMNet; GPU vs caché

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**No es una orden de ejecución.** No implemento STFT, Hilbert ni FEDformer. No abro STEP_07. No toco campañas GPU vivas, Postgres ni Metabase. No mezclo esto en el `.tex` de La Sabana.

**Insumo:** protocolo `STEP_06_AMPLITUDE_FREQUENCY_PHASE_TIME_FREQUENCY_REPRESENTATIONS_FINAL.md` (2953 líneas; SHA-256 `7f60279474484a06e0423dd81fda41d6e69ace569258f2b4c01ef1772270edc3`). Copia en `docs/tres_temas_entrevista/`.

Harvey está de acuerdo con el recorte de 03–05 y pregunta dos cosas concretas: (1) si los doce bancos de PARCNet sirven para comparar; (2) qué de FFT/Hilbert se puede **precalcular o mandar a GPU** para no repetir el preproceso en cada corrida, y **no** hacerlo en tiempo real.

---

## 0. Una línea

¿Un modelo finito generaliza mejor si se le entregan coordenadas explícitas (magnitud, fase, tiempo-frecuencia) en vez de pedirle que las infiera del crudo — sin leakage, y con la fase de Hilbert solo después de separar bandas?

El protocolo acierta en el principio: representación invertible \(\neq\) representación aprendible. DFT completa no crea información; puede bajar la carga de un modelo chico. Eso no es tesis de “usar Fourier”. FEDformer, FreTS, FITS, TimesNet, TimeMixer, WPMixer, WaveToken, FreDF ya cubren el territorio obvio.

---

## 1. Qué no recortaría

1. **Fase no es residuo.** Oppenheim–Lim 1981. Representación circular o pares \((\Re,\Im)\). Cerca de \(|X[k]|\approx 0\) la fase es inestable: se enmascara.
2. **STFT/wavelet de borde vivo = ventana *trailing*.** Ventana centrada \(t\pm W/2\) es el mismo leakage que el denoising. Padding no puede meter futuro.
3. **Hilbert de la serie cruda no es frecuencia instantánea.** Boashash: primero banda/IMF, después \(A(t),\phi(t)\). H6.12. HHT/EMD es rama experimental, no default (mezcla de modos, extremos, causalidad).
4. **Control de surrogates.** Aleatorizar fase conservando \(|X_f|\) (Theiler / Schreiber). Si el pronóstico no cae, la “fase útil” era teatro.
5. **Capacidad igualada.** Un transform que gana porque el modelo tiene más parámetros no es un resultado de representación (compuerta 6G).
6. **Coherencia \(\neq\) causalidad.**

---

## 2. PARCNet, PAMNet, y los doce bancos

### 2.1 Qué verifiqué, y qué no

**PARCNet** (el del protocolo, [24]): Li, Chai, Song, Liu, Liu. *PARCNet: Phase-aware Residual Correction Network for Efficient Multivariate Time Series Forecasting*, *Knowledge-Based Systems* 348:116370, 2026, doi:10.1016/j.knosys.2026.116370. El protocolo dice que usa amplitud y fase de la **señal analítica / Hilbert**. ScienceDirect bloqueó el PDF desde esta máquina (Cloudflare). **No pude enumerar sus doce datasets desde el artículo.** No invento la lista.

**PAMNet** (abierto, arXiv:2605.02938, 1 May 2026): Zhou et al., *Cycle-aware Phase-Amplitude Modulation Network*. **Sí usa doce bancos**, y los nombra. **No es Hilbert.** Es un índice cíclico \(t=\tau_{\mathrm{end}}\bmod c\) (hora del día, etc.) con embeddings aprendidos de “fase” y “amplitud” de ciclo. Analogía de AM, no transformada analítica. Distinto objeto. Lo dejo como prior art de *fase cíclica aprendida*, no como réplica de Hilbert.

Si Harvey o el otro agente tienen el PDF de PARCNet, se pega la tabla y se cierra. Mientras, comparo contra la suite LTSF que **sí** está publicada en PAMNet y que es la misma familia que STEP_06 §60.

### 2.2 Los doce de PAMNet (Tabla 1, verificada)

| # | Dataset | Vars | Freq. típica | ¿Nos sirve para *comparar*? | ¿Entra al banco mínimo? |
|---|---|---|---|---|---|
| 1 | ETTh1 | 7 | 1 h | **Sí.** Ya era el candidato de STEP_03. Mismo reloj que 1 h propio | **Sí. Primero** |
| 2 | ETTh2 | 7 | 1 h | Sí, réplica de transformador | Después de ETTh1 |
| 3–4 | ETTm1 / ETTm2 | 7 | 15 min | Sí, pero no es nuestro \(\Delta t\) | No al inicio |
| 5 | ECL / Electricity | 321 | 1 h | Sí como mesa pública; caro en canales | No. 321 \(\times\) STFT explota |
| 6 | Traffic | 862 | 1 h | Mesa pública; **no** para barrer representaciones | No |
| 7 | Weather | 21 | 10 min | Sí, no estacionario, tamaño razonable | **Sí. Segundo** |
| 8 | Solar | 137 | 10 min | Opcional | No |
| 9–12 | PEMS03/04/07/08 | cientos | 5 min | Tráfico de sensores. No es finanzas ni 1 h/4 h | **No** |

**Exchange Rate** (STEP_06 §60, Informer/Lai) **no** está en esos doce y es el más cercano a FX. Si hay que elegir un tercero, Exchange antes que PEMS.

**Veredicto.** Los doce sirven como *mesa de comparación* si algún día se afirma transfer a LTSF. No sirven como grilla de trabajo. Banco: **ETTh1 + Weather + sintético (sinusoide/chirp/AM)**. Electricity/Traffic/PEMS son presupuesto de campaña, no de preproceso. Finanzas propias después, como siempre.

PARCNet/PAMNet **no se reimplementan**. “Usar fase” ya no es novedad. El hueco sigue siendo la cadena SNR → \(Q\) → fuente → coordenadas explícitas, causal, con surrogate.

---

## 3. Recorte de hipótesis: dieciocho es otra tesis

Núcleo:

| Id | Afirmación | Se falsifica si |
|---|---|---|
| H6.1 | Algún \(T\) invertible gana al crudo con **mismos** parámetros | El crudo empata o gana |
| H6.2 / H6.3 | Complejo o \((A,\phi)\) gana a solo-magnitud; la fase aporta | Magnitud basta |
| H6.4 | Surrogate de fase empeora más que conservar \(|X_f|\) | Aleatorizar fase no mueve \(P\) |
| H6.12 | Hilbert útil **después** de separar bandas, no sobre el precio crudo | Hilbert crudo \(\ge\) Hilbert de banda |

H6.5–H6.11, H6.13–H6.18 (STFT vs wavelet, bandas, coherencia, scattering, loss en frecuencia, tamaño de modelo): **después** de 6C. Si 6C falla (la fase no aporta en FFT global), no se abre Hilbert/CWT/HHT.

Dieciséis familias R0–R15 no se lanzan. El §75 del protocolo ya pide el experimento barato: crudo vs \(|\mathrm{FFT}|\) vs \(\phi\) vs \((\Re,\Im)\) vs crudo+complejo, más surrogate. Eso es el banco. STFT/DWT/Hilbert entran si 6C vive.

---

## 4. Compuertas

| Gate | Qué exige | Si falla |
|---|---|---|
| **6A** | Sintético: el transform recupera \(A\), \(f\), \(\phi\) conocidos | No se interpreta ningún espectro financiero |
| **6B** | Soporte causal auditado (trailing; hash de padding) | **Se para.** Un resultado con leakage no se publica |
| **6C** | Complejo o fase vs solo-magnitud, capacidad igualada | No Hilbert, no CWT, no HHT |
| 6D | Tiempo-frecuencia en no estacionario (Weather) | FFT global basta para este banco |
| 6G | Sigue ganando a igual nº de parámetros | Era inflado de modelo |
| 6H | ETTh1 o Weather, splits publicados | No se afirma LTSF |
| 6J | Test una vez | — |

**Éxito del paso:** artefacto + veredicto {fase aporta / no aporta / solo en SNR alto}. Outcome A (el crudo gana) cierra y no se enchufa.

---

## 5. GPU, pre cálculo, y por qué el costo no es el FFT

Harvey: *hay bastante trabajo; ver qué se puede hacer en GPU; FFT y otras cosas precalcular o GPU; no en tiempo real; repetir preproceso es caro.*

De acuerdo, con un matiz: **lo caro de repetir no es el FFT, es reentrenar.** Un rFFT de una ventana 512, miles de ticks, decenas de series, en CPU (NumPy/FFTW) es barato. El desperdicio es volver a transformar **dentro de cada época** o **en cada semilla** porque nadie cacheó.

### 5.1 Contrato: materializar, no stream

Nada de esto es inferencia en el tick. Es laboratorio offline.

1. Fijar \((dataset, split, T, W, causal, padding, versión)\).  
2. Calcular **una vez** las representaciones.  
3. Escribir `repr_<hash>.parquet` / `.npy` + manifiesto JSON.  
4. El train solo **lee columnas**. Cambiar \(b\), denoiser o semilla no recalcula el espectro si el input de \(T\) no cambió.

Causality: se precalcula la **ventana trailing** en cada \(t\) del split, no un FFT/Hilbert de la serie entera para luego recortar. `scipy.signal.hilbert(serie_completa)` usa el futuro. Eso no se cachea.

Forma GPU-amigable del rolling FFT:

\[
X_{\mathrm{win}} \in \mathbb{R}^{B \times C \times W}
\quad\xrightarrow{\texttt{torch.fft.rfft}}
\quad
\mathbb{C}^{B \times C \times (W/2+1)}
\]

con \(B\) = ticks apilados (strided view, sin copiar). Hilbert de ventana = el mismo rFFT, filtro \(-j\) en positivos, irFFT.

### 5.2 Qué va a GPU, qué a CPU, qué no se toca

**GPU de *kernels de transform*, no de campaña de trading.** No paro ni pido las A100 que ya corren. Si hay un dispositivo ocioso y Harvey lo autoriza, un job corto de materialización. Si no, CPU + caché basta para ETTh1/Weather.

| Operador | ¿Precalculable? | ¿GPU útil? | Notas |
|---|---|---|---|
| FFT / rFFT rolling | **Sí** | Sí, cuFFT / `torch.fft` en batch | Prioridad 1 de STEP_06 |
| STFT trailing | **Sí** | Sí (`torch.stft`) | Misma caché |
| Hilbert por ventana (FFT) | **Sí** | Sí | No Hilbert de serie completa |
| DWT (PyWavelets) | Sí | Poco; CPU ok en 1-D | GPU si Kymatio/pytorch-wavelets y hay cola |
| CWT / scattering | Sí | Kymatio CUDA | **Después** de 6C |
| Multitaper | Sí (DPSS CPU + FFT) | Solo la FFT | |
| Surrogates de fase | Sí | Sí (FFT, randomizar \(\phi\), iFFT) | Se cachean *como dataset* |
| Cross-espectro pares | Cuidado | GPU | \(O(m^2 F)\). No Traffic/PEMS |
| EWMA / trailing MA (STEP_03) | Sí | No hace falta | CPU |
| Cuantización uniforme / \(\mu\)-law (04) | Sí | No | CPU |
| Lloyd–Max | Sí, una vez en train | No | CPU |
| \(H_k\), AR, surprisal (05) | Parcial | No | CPU secuencial |
| EMD / HHT | Frágil | **No** | Iterativo, extremos; no default |
| CTW / BCT-X | Diagnóstico | No | CPU |
| Entrenar DLinear/ANN del banco | Cada semilla | GPU **si** el transform ya está en disco | Aquí sí duele CPU al repetir |
| Entrenar PatchTST / proyecto | No en el banco mínimo | GPU de campaña — **no** | |

**No se hace en tiempo real:** ningún STFT/Hilbert “al tick” para este laboratorio. Si más adelante hay enchufe a producción, el operador causal ya estará en el CSV D4–D6 o en un plugin replay; no se rediseña como stream.

### 5.3 Orden de costo, si el banco se abre

1. CPU: sintético 6A + ETTh1 crudo vs rFFT trailing cacheado vs surrogate. DLinear.  
2. Si 6C vive: Weather, misma caché.  
3. GPU **solo** si el batch de ventanas no cabe cómodo en CPU o si se entrena más de un modelo no lineal. Materializar primero; entrenar después.  
4. Hilbert de banda, DWT, coherencia: otra pasada de caché, no mezclada con la primera.

Repetir experimentos de *preproceso* entonces cuesta un `hash lookup`, no un FFT.

---

## 6. Dónde vive, cuando haya código

Igual que 03–05. **Ahora: nada.** STEP_07 (filtro adaptado) no se abre.

Puntero en `preprocessor/docs/`. Protocolos en `tres_temas_entrevista/`.

---

## 7. Entrevista

El correo sigue siendo el tema 1. Veinte segundos del 3: ruido/SNR. Si tiran a fase: *coordenadas explícitas, no modular un portadora; Hilbert no sobre el precio crudo; “usar fase” ya está en PARCNet/PAMNet.*

---

## 8. Preguntas para tu dictamen

1. ¿Banco = ETTh1 + Weather + sintético, y los doce solo como mesa de citas?  
2. ¿Caché causal de rFFT/Hilbert-por-ventana como artefacto obligatorio antes de cualquier train?  
3. ¿GPU de kernels solo si hay dispositivo ocioso, sin tocar campañas?  
4. ¿PARCNet queda como cita de papel hasta que alguien baje el PDF y liste los doce?

---

## 9. Lo que yo no haré

No implemento. No descargo Traffic/PEMS. No lanzo GPU. No abro STEP_07. No mezclo esto en La Sabana. No nombro incentivos.

Si ACCEPT GATE 6A–6C: primer commit = sintético + rFFT trailing cacheado en ETTh1, DLinear, surrogate, `CUDA_VISIBLE_DEVICES=""` salvo orden explícita de un GPU ocioso para *solo* materializar.

— Retsu
