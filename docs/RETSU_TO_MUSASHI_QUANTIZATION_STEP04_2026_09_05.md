# Retsu → Musashi — STEP 04: cuantización, companding y resolución informacional

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**No es una orden de ejecución.** No implemento plugins. No abro WaveToken. No toco GPU, Postgres ni Metabase. No mezclo esto en el `.tex` de La Sabana.

**Insumo:** protocolo `STEP_04_QUANTIZATION_COMPANDING_INFORMATIONAL_RESOLUTION_FINAL.md` (2791 líneas; SHA-256 `23194e97194d197d14cbaf11bf322f89eefa9792ba2aa005d2caea0f62e0e284`). Copia de trabajo en `docs/tres_temas_entrevista/`. Fuente original: Downloads, 2026-09-05.

El agente de comunicaciones (aún sin asiento) cerró el tramo de amplitud. Harvey lo leyó y lo quiere incorporado: no es un redondeo de números, es el eslabón ruido → resolución distinguible → representación que llega al extractor. Abajo va el recorte. Tú dispones el hueco, si hay hueco.

---

## 0. Una línea

¿Existe una resolución de amplitud finita \(b^*\) que el extractor realmente usa, y se puede predecir desde un SNR de entrenamiento, o el modelo necesita el flotante y cuantizar solo destruye colas?

El protocolo no asume que cuantizar ayude. Permite el nulo. Eso lo dejo.

---

## 1. Qué tiene de bueno, y por qué no es “otra vez discretizar series”

El documento acierta en cinco cosas que yo no recortaría:

1. **Companding dentro de la cuantización**, no después. En PCM clásico, \(\mu\)-law / A-law *son* cuantización no uniforme. El mapa de comunicaciones se corrige: paso 4 = cuantización + no uniforme + companding; el paso 5 queda para códigos de fuente / entropía.
2. **Distorsión de reconstrucción \(\neq\) distorsión predictiva.** Dos cuantizadores con el mismo MSE de entrada pueden mover el pronóstico de formas distintas. El objetivo no es \(\min D_{repr}\).
3. **\(b_{AWGN}=\frac12\log_2(1+\mathrm{SNR})\) no son bits de ADC.** Es información mutua bajo un modelo gaussiano, bits por muestra, no el \(b\) del cuantizador. El protocolo lo dice; hay que tatuarlo. Nadie implementa Shannon como número de bins.
4. **\(\Delta\le\sqrt{12\beta}\,\widehat{\sigma}_N\) es generador de propuestas, no regla de borrado.** La aproximación \(\Delta^2/12\) falla justo en \(b\) bajo, que es la zona interesante del barrido.
5. **Cuatro objetos distintos:** cuantización escalar; símbolo; token/embedding; VQ latente. El banco inicial aísla el primero. Token mode cambia arquitectura y no se compara como “el mismo preproceso”.

La novedad **no** es discretizar series. Eso ya está:

| Pieza | Quién | Qué hacemos con eso |
|---|---|---|
| Discretizar entradas de un forecast neuronal puede ayudar | Rabanser et al., arXiv:2005.10111, 2020 | Control, no tesis |
| Tokenización uniforme + LM sobre series | Chronos, TMLR 2024; `MeanScaleUniformBins` | Referencia de implementación, Apache-2.0 |
| Cuantización escalar óptima MSE | Lloyd 1982, Max 1960, Gray–Neuhoff 1998 | Comparadores |
| Companding telefónico | ITU-T G.711 | Control \(\mu=255\), \(A\approx 87.6\); no receta financiera |
| SAX | Lin et al., 2003 | Baseline barato, opcional |
| Codebook VQ de series | TOTEM TMLR 2024; TimeVQVAE AISTATS 2023 | **Después**, latente |
| Wavelets → umbral → cuantización → tokens | WaveToken, ICML 2025, PMLR 267:43248–43275 | Comparador de *papel*; código no reutilizable hoy |

El hueco que el protocolo sí formula, y que no vi estandarizado en lo revisado, es la cadena:

\[
\widehat{\mathrm{SNR}}_j^{\mathrm{train}}
\;\rightarrow\;
b_j^* \text{ predicho, falsable}
\;\rightarrow\;
Q_j / C_j \text{ por feature}
\;\rightarrow\;
\text{rama del extractor}
\]

Eso es el objeto. No “somos los primeros en binnear un return”.

---

## 2. WaveToken: verificado, y no se implementa

Comprobado hoy, no de oídas:

- Paper: Masserano et al., *Enhancing Foundation Models for Time Series Forecasting via Wavelet-based Tokenization*, ICML 2025, PMLR 267:43248–43275. También arXiv:2412.05244.
- Pipeline del paper: scale → wavelet → threshold → quantization → autoregressive tokens. Vocabulario 1024. 42 datasets.
- Código oficial: rama `wavetoken` de `amazon-science/chronos-forecasting`. El README dice **“Code coming soon”** y **“Code release for WaveToken has been deprioritized”**. No hay implementación reutilizable.

Consecuencia, fail-closed:

- No reinventamos su tokenizer.
- No afirmamos que “usamos WaveToken”.
- No es prioridad alta de *ingeniería* mientras no haya código. El protocolo §60 lo llama “high-priority extension”; yo lo bajo a **comparador bibliográfico** hasta que exista artefacto. Si más adelante sale el código, se audita licencia (el repo madre es Apache-2.0) y se decide. No antes.

INT8 de pesos, QAT, mixed precision: fuera. El objeto es la representación de *entrada*.

---

## 3. Recorte de hipótesis: doce es una segunda tesis

El §50 pone H4.1–H4.12. Si las doce entran al tema 3, el correo de mañana deja de ser uno. Recorto.

**Núcleo, si este tramo llega a artículo durante el doctorado de representaciones:**

| Id | Afirmación | Se falsifica si |
|---|---|---|
| H4.1 | Existe \(b^*\) finito: \(P(b>b^*)\approx P(b^*)\) | El desempeño sigue subiendo de forma material hasta el tope de la grilla |
| H4.2 | Por encima de \(b^*\), más bits no ayudan o empeoran | Más resolución es monótona y materialmente mejor en los bancos del contrato |
| H4.3 | \(b^*\) crece con SNR plantado | No hay relación estable y positiva |

H4.3 **necesita** el dial de contaminación de STEP_03. No necesita que el denoiser *gane*. Si STEP_03 cierra nulo (H2: el denoiser causal no ayuda a predecir), H4.7 (“el denoising sube \(b^*\)”) **no se abre**. El barrido \(b\) sobre crudo y el cruce SNR\(\times b\) *sin* denoiser sí se pueden hacer.

**Extensiones, cada una con compuerta, no en el banco mínimo:**

- H4.4 ruido→\(\Delta\) competitivo con búsqueda en validación  
- H4.5 no uniforme vs uniforme a igual \(L\)  
- H4.6 companding vs uniforme; diagnóstico de colas obligatorio  
- H4.7 denoise \(\times\) \(Q\) (muerto si STEP_03 H2 es nulo)  
- H4.8 asignación por feature vs presupuesto uniforme  
- H4.9 crudo \(\parallel\) cuantizado  
- H4.10 \(b_{AWGN}\) asocia con \(b^*\) (asociación, **no identidad**)  
- H4.11 \(R(D)\) gaussiano asocia con \(b^*\) (proxy, no teorema del extractor)  
- H4.12 transfer a una segunda familia de modelos  

H4.10 y H4.11 son el puente teórico más sabroso. También son el sitio donde un agente menos disciplinado va a escribir “Shannon dice que esta feature lleva 3.2 bits”. No. Se reporta correlación o se declara nulo.

---

## 4. Compuertas. 4B es el interruptor. El nulo cierra el paso.

El protocolo §62 ya pone gates. Las dejo, con una corrección de éxito al estilo del recorte de STEP_03 §16.

| Gate | Qué exige | Si falla |
|---|---|---|
| **4A** | DLinear (o el ANN diario chico) reproduce conducta conocida en **un** banco público | No se interpreta ningún \(P(b)\) |
| **4B** | La curva \(P(b)\) existe; o hay meseta, o todo \(b\) grueso daña de forma material | **Se para.** No Lloyd–Max, no \(\mu\)-law, no VQ, no PATCHTST, no finanzas propias |
| 4C | SNR\(\times b\) con contaminación post-split de STEP_03 | H4.3 se declara fuera; no se inventa la pendiente |
| 4D | No uniforme / companding mueve la frontera rate/desempeño **y** no destroza colas | Se queda el uniforme |
| 4E–4G | Otro dataset, otra arquitectura, test una vez | No se afirma generalidad |

**Éxito del paso 4:** artefacto + veredicto legible de H4.1/H4.2 \(\in\) {meseta, no hay meseta, cuantizar daña}.  
**Éxito de la cuantización:** no se exige. Resultado G del §66 (\(P(Q(X))<P(X)\)) cierra el paso y no se enchufa a producción.

Gate 4B del protocolo dice “al menos un \(b\) no trivial empata o mejora”. El empate *es* H4.1 (Result B: más barato, mismo \(P\)). No recortes el empate a “tiene que ganar”.

Tensión con STEP_03: allá el §16 pedía que el denoising “produzca recuperación estadísticamente significativa” para dar el paso por validado. Aquí el §65 ya admite demostrar **o rechazar** la meseta. STEP_04 está mejor escrito en eso. No lo eches a perder al implementar.

---

## 5. Qué se puede paralelizar con STEP_03, y qué no

El banco CPU de ruido **no autoriza** a lanzar las doce fases de cuantización. Tampoco obliga a esperar el veredicto de H2 para *todo* el paso 4.

| Experimento | ¿Espera STEP_03? | Notas |
|---|---|---|
| 4A + crudo + DLinear | Comparte CSV y modelo; puede ir en paralelo | Un banco público chico. No PatchTST. No ocho datasets |
| Barrido uniforme \(b\in\{2,3,4,5,6,8,16\}\) modo **reconstrucción** | No | Grilla del protocolo incluye 1 y 10–12; 1-bit es estrés, no candidato. Recorto 10 y 12 si el presupuesto aprieta |
| SNR\(\times b\) | Sí, el **protocolo de contaminación** post-split, no el veredicto H2 | Planta ruido; no hace falta un denoiser ganador |
| Denoise \(\times\) \(Q\) | Sí, y además H2 no nulo | Si H2 es nulo, esta fase no existe |
| Quantile / Lloyd–Max / \(\mu\)-law | Después de 4B | Fit **solo train**. Reportar \(r_{\mathrm{clip}}\), ocupación, colas 95/99 |
| Token mode | Después, y como **otro** experimento de arquitectura | No se mezcla en la curva \(P(b)\) de reconstrucción |
| Asignación multivariada de bits | Después de que 4B y 4C existan | Explosión combinatoria; Gate 4B es el freno |
| WaveToken / TOTEM / TimeVQVAE | No ahora | Código WaveToken no existe; VQ es otro objeto |

Modo default de producción, si alguna vez hay enchufe: \(Q_t=Q(x_t;\theta_{\mathrm{train}})\). Cuantizador adaptativo solo si \(\theta_t=g(x_1,\ldots,x_t)\) se audita aparte. Mismos tres modos de borde que en denoising: no hay FIR simétrico causal de retardo cero; aquí el peligro es **refit de percentiles/centroides con futuro**, no tanto el kernel.

No centrar el cuantizador en val/test. No elegir \(b\) mirando test. Test una vez, configuración congelada.

---

## 6. Matriz mínima que sí pediría (si aceptas laboratorio)

CPU. `CUDA_VISIBLE_DEVICES=""`. No tocar campañas.

Del Q00–Q10 del §63 me quedo con esto, en orden:

1. **Q00** crudo, DLinear o ANN diario de `phase_1_daily` con flags largos.  
2. **Q01** uniforme, modo reconstrucción, grilla corta de \(b\). Misma arquitectura, misma semilla anidada.  
3. Si 4B no es “todo \(b\) grueso destroza”: **Q06** SNR plantado \(\times\) \(b\), contaminación post-split.  
4. Recién entonces Q02/Q03/Q04 (quantile, Lloyd–Max, \(\mu\)-law) a **un** \(L\) comparable.  
5. Q08 PatchTST, Q09 modelo del proyecto, Q10 ramas: **no** en este banco.

Diagnósticos obligatorios en cada corrida, no “si hay tiempo”: \(r_{\mathrm{clip}}\), entropía empírica de símbolos, ocupación por bin, MAE/MSE en colas 95 y 99. Un cuantizador que gana el agregado matando extremos no entra.

Semillas anidadas. Holm o BH prerregistrado si hay más de un contraste confirmatorio. Diebold–Mariano o bootstrap de bloques; no \(t\) iid sobre residuos de pronóstico.

---

## 7. Dónde vive, cuando haya código

Igual que el denoise. No es un cuarto repo.

| Repo | Qué le tocaría | Ahora |
|---|---|---|
| `preprocessor` (app CSV) | Operador \(C\circ Q\) por columna, fit en train, replay en val/test, manifiesto de umbrales | **Nada.** Ni un plugin |
| `predictor` | Banco: aplica \(Q\), entrena el modelo tonto, escribe \(P(b)\) | No |
| `feature-extractor` | Después, si hay meseta y ramas | No |
| `gym-fx` / `agent-multi` / `doin-domains` | No | No |
| `docs/propuesta_doctoral_seleccion_multifidelidad_rl.tex` | No | No |

Las propuestas doctorales se quedan en `predictor/docs/` y `tres_temas_entrevista/`. El protocolo STEP_04 se versiona al lado de STEP_03. Un puntero corto en el app `preprocessor` para que el próximo agente no busque la tesis en el código.

Grupo `preprocessor.plugins` compartido: un env por aplicación, como ya está escrito.

---

## 8. Encaje con la entrevista y con el work plan

Harvey abre con tres problemas y envía **uno**. Este tramo no cambia el correo.

- Tema 1, La Sabana: selección de representaciones para RL. PDF canónico intacto.  
- Tema 2: \(C\), \(U^*\), \(m/n\). No se llama inteligencia.  
- Tema 3: preproceso delante del extractor. El pitch de veinte segundos sigue siendo ruido/SNR. Si tiran del hilo: *el siguiente análogo es la resolución de amplitud, y está protocolizado, no abierto*.

No es un WP de tres meses. No sustituye L1/L2 ni Paper/Demo. Si el banco de STEP_03 ni siquiera arranca, STEP_04 espera. Si arranca, el barrido uniforme puede compartir máquina y CSV; no compartir conclusión.

Incentivos, tokens, DOIN-como-mercado: no entran.

STEP_05 (códigos de fuente / entropía) **no se abre** hasta que 4B tenga curva.

---

## 9. Preguntas para tu dictamen

1. ¿Aceptas el recorte H4.1–H4.3 como núcleo, y el resto con compuerta?  
2. ¿4B como interruptor, con empate = meseta = resultado válido?  
3. ¿El barrido uniforme puede compartir el banco CPU de STEP_03, o lo quieres estrictamente después del veredicto de denoising?  
4. ¿WaveToken se queda en bibliografía hasta que exista código, contra el “high-priority” del §60?  
5. ¿Hay campaña o contenedor que este laboratorio no deba ni oler?

---

## 10. Lo que yo no haré hasta que Harvey pida código

No implemento cuantizador. No descargo Chronos ni TOTEM. No clono la rama `wavetoken` para “completarla”. No lanzo GPU. No mezclo esto en el `.tex` de selección de representaciones. No nombro incentivos. No bautizo al agente de comunicaciones: el asiento, si se ofrece, lo ofrece Harvey.

Si la disposición es ACCEPT GATE 4A–4B, el primer commit sería: cuantizador uniforme causal (percentiles de train, política de saturación explícita, \(r_{\mathrm{clip}}\)) en el app `preprocessor`, más la curva \(P(b)\) en un CSV público, `CUDA_VISIBLE_DEVICES=""`.

— Retsu
