# Preprocesamiento informacional de series: ruido, SNR y transferencia al extractor

**Propuesta de investigación doctoral**  
Harvey Demian Bastidas Caicedo  
Borrador para entrevista — 2026-09-05  
Programa de interés: doctorado por tesis.  
No es la propuesta que se envía por correo. Esa es la de selección de representaciones para RL.  
No hay membrete de universidad admitida.

**Una línea.** ¿Un preproceso que estima ruido y lo trata de forma causal entrega al extractor más estructura aprovechable que la serie cruda, o solo suaviza y empeora la predicción?

---

## Resumen

En comunicaciones, una cadena de operaciones —muestreo, estimación de ruido, filtrado, cuantización, cambio de representación— permitió acercarse al límite de un canal con ruido conocido. El módem V.90, en el extremo digital de la red telefónica, alcanzó del orden de 56 kbit/s de bajada; el dato sirve aquí como motivación de *ingeniería por etapas*, no como constante que se trasplante a un extractor de características.

En aprendizaje automático ocurre otra cosa. Al extractor se le pide que convierta la observación en parámetros entrenados, es decir, en la primera forma de conocimiento del sistema modular: extractor, núcleo, cabezales. Si la entrada mezcla estructura y ruido de forma opaca, no se le puede pedir al extractor lo que la representación no hace explícito. Quitar ruido, sin embargo, no garantiza una mejor predicción: puede borrar innovaciones que el modelo necesitaba. Eso es empírico, no un slogan.

Esta tesis no reconstruye la pila V.90. Estudia **un tramo**: ruido, relación señal–ruido y transferencia al extractor, en series públicas, con ruido plantado cuya potencia se conoce. Los tramos 4–7 —amplitud, fuente, coordenadas, detector— ya tienen protocolo y **no entran al núcleo** hasta sus compuertas. El análogo de fuente no es Huffman. El de fase no es una portadora. El de detección no es otro autoencoder: el extractor *reconstruye* la ventana; el filtro adaptado pregunta si un patrón está, con un modelo de ruido. Conv1D no es matched filter salvo que el kernel y \(C_n\) estén atados a la razón de verosimilitud. Códigos de canal siguen en el catálogo.

Los mercados y las series propias quedan fuera de la calibración: su ruido no se conoce y el objetivo se contamina al medirlo. Entran, si acaso, como comprobación posterior, igual que en la línea de representaciones para RL.

---

## 1. Problema

Un modelo modular no empieza en el núcleo. Empieza en lo que el extractor *ve*. El extractor se puede entrenar aparte y luego conectarse al núcleo y a los cabezales. Esa decisión de diseño es anterior a elegir si el núcleo es una convolución, un transformador o un agente.

Tres hechos se confunden con facilidad:

1. El muestreo impone un techo. Para periodo \(\Delta t\), Nyquist es \(1/(2\Delta t)\). Lo que está por encima no es un patrón que el extractor pueda recuperar de esa serie.
2. La observación es \(x_t = s_t + n_t\). En un canal de enseñanza, \(n_t\) se conoce. En una serie natural, no. La definición de ruido es parte del problema.
3. Un denoiser \(D\) produce \(\hat s_t = D(x_t)\). Puede subir la SNR aparente y bajar el error fuera de muestra, o puede subir la SNR aparente y *empeorar* la predicción porque \(D\) se comió la innovación.

La analogía con comunicaciones es una **hipótesis de trabajo**, no una equivalencia. Shannon da la capacidad de un canal gaussiano. No da la capacidad de un extractor Keras. El experimento que sí se puede hacer es plantar ruido de potencia conocida, estimar SNR, aplicar un denoiser causal, y medir si el extractor gana fuera de muestra frente a la serie cruda y frente a pasar señal y residuo por ramas distintas.

---

## 2. Objetivos

**General.** Determinar, con series públicas y ruido plantado, si un preproceso causal de estimación de ruido y filtrado mejora la información que el extractor convierte en parámetros, frente a no preprocesar.

**Específicos.**

1. Fijar una definición operacional de señal y ruido cuando \(n_t\) se planta, y un estimador de SNR que no use el futuro.
2. Contrastar tres entregas al extractor: crudo; denoised; señal y residuo en ramas separadas.
3. Comprobar si el ranking se sostiene cuando el ruido deja de ser gaussiano blanco, o declarar el límite.

---

## 3. Pregunta e hipótesis

**Pregunta.** ¿Bajo qué condiciones un preproceso causal que estima SNR y trata el ruido aumenta, en un extractor congelado en protocolo, el desempeño fuera de muestra respecto de la serie cruda, sin usar información futura?

**H1 — Ruido plantado empeora.** Añadir ruido gaussiano de potencia creciente a las entradas, con el extractor y el núcleo fijos en protocolo, degrada una métrica prerregistrada. Falla si la degradación no es monótona ni detectable frente a la variación de semillas.

**H2 — Denoising no es gratis.** Un denoiser causal, con parámetros fijados sin ver la prueba, no mejora de forma estable el desempeño fuera de muestra respecto del crudo en todos los SNR. Se reporta en qué régimen ayuda, en cuál empata y en cuál empeora. Falla como tesis si se afirma mejora universal.

**H3 — Ramas frente a supresión.** Entregar señal estimada y residuo en ramas distintas no es peor, en el régimen donde H2 empeora, que tirar el residuo. Falla si tirar el residuo gana de forma estable: entonces el ruido plantado no llevaba innovación útil para ese extractor.

**Siguientes tramos, con compuerta, no en el correo.** STEP 04–06 como antes. STEP 07: recuperar el filtro adaptado en AWGN sintético (H7.1); GMF con \(C_n\) de color (H7.2); MiniRocket como control barato (H7.6); motivo frecuente \(\neq\) predictivo (H7.16). Un Conv1D/TCN ya existente es baseline, no contribución. El detector no se mete como encoder/decoder de reconstrucción. Ninguno es esta tesis.

No hay hipótesis sobre capacidad de canal del extractor, ni sobre V.90, ni sobre incentivos.

---

## 4. Marco

**Sistema.** Cadena modular: preproceso \(P\) → extractor \(E\) → núcleo → cabezal. \(E\) se entrena con \(P\) congelado o se reentrena; el contrato lo fija en el piloto. El núcleo y el cabezal se eligen simples a propósito: el objeto es \(P\), no ganar un banco de pronóstico.

**Ruido plantado.** \(x_t = s_t + n_t\), \(n_t\) gaussiano de potencia \(\sigma^2\) conocida. \(s_t\) es la serie pública original. Así SNR es un dial, no una metáfora.

**Estimador de SNR.** Se prerregistra uno, más un control ingenuo. Candidatos de laboratorio, no todos a la vez: residuo STL, MAD sobre coeficientes wavelet, potencia de innovación. El que entre a H2–H3 tiene que ser causal: en el instante \(t\) solo usa \(\{x_u: u\le t\}\) y parámetros ajustados en desarrollo.

**Denoiser.** Un filtro causal de la misma familia que el estimador. No se elige el que mejor se vea en la prueba.

**Lo que no entra al núcleo de esta tesis.** Segundo AE, VQ, WaveToken como código, Huffman como input, Hilbert crudo, HHT default, detector como AE, SAC como router, sparse coding paralelo, TTA, DANN, restar factores, alinear con futuro, NAS, bits de Shannon como anchos Conv1D. Equalización (08) es canal + CSI. Crosstalk (09) es \([X,C,U]\). Sync (10) es \(\tau\) causal. Canal (11) es máscara en el AE existente. Router (12) elige modos ya validados. Presupuesto (13) pregunta si 64:32:32:32 ya es Pareto. Companding dentro de \(Q\). INT8 de pesos no es el objeto.

**Cuatro objetos que no se mezclan** si el tramo 4 se abre: valor reconstruido; símbolo; token/embedding; codebook latente. El banco, si existe, aísla el primero. Tokenizar cambia el modelo.

**Separación de las otras dos líneas.** La tesis de representaciones para RL elige un *codificador de estado* bajo presupuesto, con abstención. Esta elige un *preproceso* delante de un extractor. La de memorización y dimensionamiento mide techo y uso de una red sobre datos de complejidad conocida. Tres objetos. Un correo, una.

---

## 5. Método

Estudio experimental, prerregistrado. Unidad: serie × SNR × preproceso. Semillas anidadas del entrenamiento; no son la muestra.

**Datos.** Bancos públicos de series, no el sistema de producción. Candidatos de piloto: un conjunto de clasificación o pronóstico de la UCR o ETT, con todas las series presentes en el checkout o descargables por script. Si una serie falta, no se usa. Finanzas propias: comprobación posterior, no H1–H3.

**Fase 0.** Contrato: métrica, SNR de la grilla, denoiser, regla causal, partición temporal. Hash antes de resultados.

**Fase 1 — H1.** Grilla de \(\sigma\). Curva de la métrica contra SNR. Control: etiquetas o horizonte permutados.

**Fase 2 — H2.** Mismo extractor. Crudo vs denoised. Ablación de hiperparámetros del denoiser en desarrollo, no en prueba.

**Fase 3 — H3.** Crudo vs denoised vs dos ramas. Si el piloto no sostiene dos ramas, H3 se declara fuera de alcance y se publica el recorte.

**Cómputo.** CPU por defecto. Modelos pequeños. Cada fase exige el artefacto de la anterior. FFT/STFT/Hilbert-por-ventana se **materializan una vez** (caché causal) y el train lee columnas; no se recalculan por época ni en el tick. GPU de *kernels* de transform solo si hay dispositivo ocioso y Harvey lo autoriza; no se tocan campañas vivas. Lo caro de repetir es reentrenar, no el rFFT.

**Tramo 4, si H1 sostiene el dial.** Compuerta 4A: un modelo tonto reproduce conducta conocida en un banco público. Compuerta 4B: existe la curva \(P(b)\) en modo reconstrucción. Si todo \(b\) grueso daña, se para. Si hay meseta o empate, eso *es* resultado. WaveToken es comparador de papel.

**Tramo 5, diagnóstico vs representación.** Compuerta 5A: el estimador de \(h_X\) se valida en sintético. Compuerta 5B: el contexto baja el rate frente a orden cero; si no, no hay rama de fuente. Un alfabeto *provisional* basta para 5A–5B; congelar el mejor \(Q\) de STEP 04 espera 4B. Surprisal e innovación esperan 5B. IB/CIB y MDL no son el primer banco. BCT-X es baseline ajeno, no un plugin nuestro.

**Tramo 6, coordenadas.** Compuerta 6A: sintético recupera \(A,f,\phi\). Compuerta 6B: sin leakage (trailing). Compuerta 6C: complejo o fase vs solo-magnitud, mismos parámetros; si falla, no se abre Hilbert/CWT. Banco público: ETTh1 + Weather. Los doce LTSF son mesa de citas, no grilla.

**Tramo 7, detección.** Compuerta 7A: pulso conocido en AWGN; el MF se comporta como el libro. Compuerta 7B: GMF vs MF en ruido coloreado, \(C_n\) de train. Compuerta 7D: MiniRocket no es teatro. El TCN/CNN del predictor es comparador, no se reimplementa. El extractor de AE no se relabela detector. `agent-multi` espera evidencia predictiva.

**Tramo 8, canal.** Compuerta 8A: ganancia/FIR sintéticos se recuperan. Compuerta 8B: MMSE gana a ZF en nulos con ruido. Compuerta 8D: las colas no se hunden. Si E0 empata, 08 es nulo útil. Crudo vs equalizado vs ambos, con CSI. TTA fuera.

**Tramo 9, común/privado.** Compuerta 9A: tabla de ganancia condicional (H5.8), sin red nueva. Compuerta 9B/9C: referencia limpia vs contaminada (Widrow). Ramas \([X,C,U]\). No se resta \(C\) salvo H9.25. \(C\neq N\neq E\). Lags dinámicos son STEP 10.

**Tramo 10, sincronía.** Compuerta 10A: contrato point-in-time \(A_i(t)\). Compuerta 10B: delay sintético se recupera. Compuerta 10E: metadato de lag, no reemplazo duro. No DTW por default. No macros por timestamp limpio.

**Tramo 11, redundancia.** Compuerta 11A: el CNN daily del extractor se reentrena. Compuerta 11B: veredicto estable {ayuda, empata, empeora}; empate cierra. Compuerta 11C: misma \(\dim Z\). Compuerta 11D: colas. No es STEP_03. No es C6.

**Tramo 12, enlace adaptativo.** Compuerta 12A: ≥2 modos heterogéneos. Compuerta 12B: el IQS predice al oráculo. Compuerta 12C: una regla gana al estático o se queda el campeón. Aprendido solo si gana a la regla. No MoE. No RL. No es la abstención de L2.

**Tramo 13, presupuesto.** Compuerta 13A: alguna rama tiene curva no plana. 13B: \(P\) cambia con \(B\). Ancla: composite 64:32:32:32, fusión congelada. FLOPs ≠ latencia. Dinámico solo después de 12. Un nulo (“el ratio actual basta”) es tesis.

**Tres años** solo si esta línea se convirtiera en matrícula. Hoy es el tercer tema de entrevista y, si el piloto rinde, un artículo durante el doctorado de representaciones.

---

## 6. Contribuciones

1. Un protocolo causal de ruido plantado, SNR y tres entregas al extractor.
2. Un mapa de regímenes: dónde denoise ayuda, empata o empeora.
3. Un recorte explícito: la analogía queda como catálogo. STEP 04–13 están escritos y gated. El router no inventa modos. El presupuesto no son bits de Shannon. Si el oráculo no gana, un modo basta. Si 64:32:32:32 ya es Pareto, también.

---

## 7. Riesgos

| Riesgo | Tratamiento |
|---|---|
| Trasplantar Shannon o V.90 como teorema del extractor | No se hace. Motivación de etapas, no constante. |
| Denoising que usa el futuro | Contrato causal. Filtros centrados y descomposiciones sobre la serie completa quedan fuera de H1–H3. |
| Alta frecuencia = ruido | No. El ruido es el que se planta, o el residuo que el contrato define. |
| Segunda tesis escondida en la de RL | Objeto disjunto. No se mezcla en el PDF que se envía. |
| Dataset de producción | Fuera. Público primero. |
| El otro agente abre cuantización y fase a la vez | 4A–4B, 5A–5B, 6A–6C. Hilbert crudo y HHT fuera. |
| Huffman / ANS como input del extractor | No. El análogo es el modelo de fuente. H5.15 es control negativo. |
| Comprimir \(X\) = predecir \(Y\) | No. H5.7 / H5.12. IB/CIB es el control con target, no un IB casero. |
| Innovación = ruido de STEP_03 | No. \(E\neq N\). |
| MDL de la fuente = \(C\) de la red | No. Tema 2 es techo de parámetros. STEP 05 es descripción de la serie. |
| \(b_{AWGN}\) como bits del cuantizador | No. Es IM gaussiana por muestra. Se prueba asociación con \(b^*\), no identidad. |
| Reinventar WaveToken / VQ / INT8 / CTW | WaveToken sin código. BCT-X es baseline ajeno. INT8 fuera. |
| Distorsión de reconstrucción como éxito | El pronóstico manda. Colas 95/99 y \(r_{\mathrm{clip}}\) se reportan. |
| FFT/Hilbert de la serie completa | Leakage. Ventana trailing; se cachea, no se calcula en el tick. |
| Recalcular el espectro en cada época | Caché por hash. GPU de kernels solo con dispositivo ocioso; no campañas. |
| Los doce LTSF como grilla | Mesa de citas. Banco: ETTh1 + Weather + sintético. |
| Conv1D / AE = filtro adaptado | No. El detector tiene contrato propio: \(a_{k,t}\), no latente de reconstrucción. |
| Equalizar = z-score | No. E0 ya existe. 08 es \(\mathcal G_d\) + CSI. Over-stationarization es fallo. |
| Compartido = ruido / cancelar | No. \(X=C+U\), \([X,C,U]\). Tabla primero. |
| Alinear = hacer el tensor síncrono | No. Metadato \(\tau\); no arrastrar \(t+k\) a \(t\). |
| Canal = otro autoencoder | No. Máscara train-only en el AE actual. Empate cierra. |
| Adaptar = MoE / SAC | No. Router sobre modos ya medidos. \(G_{\mathrm{oracle}}\) primero. |
| MIMO = bits de canal en la red | No. Anchos/latencia del composite. \(C\) de MacKay es el tema 2. |
| Matrix Profile all-pairs en train+test | Leakage. Diccionario congelado en train o perfil izquierdo. |

**Resultado mínimo.** Una curva de desempeño contra SNR plantado, y un veredicto de si el denoiser causal gana, empata o pierde. Un nulo es tesis. Un “el preproceso siempre ayuda” no lo es. Si H1 sostiene el dial, el artefacto siguiente es \(P(b)\), no un tokenizer. Si hay alfabeto, el de fuente es \(H_k\) vs \(k\). Si hay espectro, es crudo vs magnitud vs fase vs surrogate, no PAMNet reimplementado.

---

## 8. Referencias de anclaje

Shannon, C. E., *Bell Syst. Tech. J.*, 1948.  
Shannon, C. E., IRE National Convention Record, 1959 (rate–distortion).  
Nyquist, H., *Trans. AIEE*, 1928.  
ITU-T, Recommendation V.90, 1998.  
ITU-T, Recommendation G.711, 1988.  
Gray, R. M. and Neuhoff, D. L., *IEEE Trans. Inf. Theory*, 1998.  
Lloyd, S. P., *IEEE Trans. Inf. Theory*, 1982.  
Max, J., *IRE Trans. Inf. Theory*, 1960.  
Rabanser, S. et al., arXiv:2005.10111, 2020.  
Ansari, A. F. et al., Chronos, *TMLR*, 2024.  
Masserano, L. et al., WaveToken, *ICML*, PMLR 267:43248–43275, 2025.  
Huffman, D. A., *Proc. IRE*, 1952.  
Willems, F. M. J. et al., Context-Tree Weighting, *IEEE Trans. Inf. Theory*, 1995.  
Rissanen, J., *Automatica*, 1978.  
Tishby, N. et al., Information Bottleneck, Allerton, 1999.  
Delétang, G. et al., Language Modeling Is Compression, *ICLR*, 2024.  
Papageorgiou, I. and Kontoyiannis, I., BCT-X, *Int. J. Forecasting*, 42(2):474–491, 2026.  
Li, X. et al., CIB-MTSF, *IJCAI*, 2025.  
Maasoumi, A. and Racine, J., *J. Econometrics*, 2002.  
Oppenheim, A. V. and Lim, J. S., *Proc. IEEE*, 1981.  
Boashash, B., *Proc. IEEE*, 1992.  
Li, Z. et al., PARCNet, *Knowledge-Based Systems* 348:116370, 2026.  
Zhou, Y. et al., PAMNet, arXiv:2605.02938, 2026.  
Turin, G. L., *IRE Trans. Inf. Theory*, 1960.  
Dempster, A. et al., ROCKET, *Data Min. Knowl. Disc.*, 2020.  
Lin, M. et al., L-MAP, *AAAI*, 40(18), 2026.  
Donoho, D. L., *IEEE Trans. Inf. Theory*, 1995.  
Cleveland, R. B. et al., *J. Off. Stat.*, 1990.  
Zhang, C. et al., *ICLR*, 2017.  
Zhou, H. et al., Informer, *AAAI*, 2021.  
Dau, H. A. et al., UCR Time Series Archive, 2018.  
Zeng, A. et al., DLinear, *AAAI*, 2023.
