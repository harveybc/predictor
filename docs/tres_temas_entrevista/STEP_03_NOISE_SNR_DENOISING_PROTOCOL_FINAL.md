# Paso 3 — Protocolo final de estimación de ruido, SNR y denoising para series de tiempo

**Estado:** Versión final para incorporación al repositorio de documentación  
**Ámbito:** Series de tiempo financieras y multivariadas con splits temporales fijos  
**Split asumido:** 4 años de entrenamiento + 1 año de validación + 1 año de test  
**Periodicidades objetivo:** 1 h y 4 h  
**Propósito:** Definir un protocolo falsable, reproducible y causal para estudiar el impacto del ruido, estimar SNR y evaluar si el denoising mejora realmente la información aprovechable por modelos de Machine Learning.

---

## 1. Objetivo científico

El objetivo no es demostrar que el denoising “funciona”, sino someter a prueba la hipótesis:

\[
H_1:\quad P(D(X)) > P(X)
\]

donde:

- \(X\) es la representación original;
- \(D(\cdot)\) es un operador de estimación/atenuación de ruido;
- \(P(\cdot)\) es una métrica de desempeño predictivo fuera de muestra.

La hipótesis nula correspondiente será:

\[
H_0:\quad P(D(X)) \leq P(X)
\]

El protocolo debe distinguir entre ruido añadido artificialmente, ruido estimado en datos reales y estructura erróneamente clasificada como ruido.

---

## 2. Causalidad y uso de los splits

Regla fundamental:

> Toda decisión de diseño o parametrización del estimador de ruido/denoising debe aprenderse exclusivamente con el dataset de entrenamiento.

Sea:

\[
\mathcal{D}_{train},\quad \mathcal{D}_{val},\quad \mathcal{D}_{test}
\]

con:

- \(\mathcal{D}_{train}\): 4 años;
- \(\mathcal{D}_{val}\): 1 año;
- \(\mathcal{D}_{test}\): 1 año.

### 2.1. Training

Se permite usar training para:

- estimar distribuciones;
- estimar SNR;
- ajustar thresholds;
- estimar escalas de ruido;
- ajustar parámetros de wavelets, STL, VMD, filtros o estimadores equivalentes;
- calibrar hiperparámetros del denoiser;
- construir curvas controladas de degradación por ruido;
- definir transformaciones causales;
- construir estadísticas auxiliares.

### 2.2. Validation

Validation se utiliza exclusivamente para:

- comparar configuraciones previamente definidas;
- seleccionar hiperparámetros de modelo;
- early stopping;
- evaluar generalización fuera de training.

No debe utilizarse para reestimar parámetros estadísticos que deberían haber quedado fijados con training.

### 2.3. Test

Test se mantiene completamente fuera del proceso de decisión y se utiliza únicamente para la evaluación final.

No se permite usar test para:

- seleccionar thresholds;
- ajustar SNR;
- escoger métodos;
- escoger arquitecturas;
- escoger ventanas;
- decidir si conservar o descartar una transformación.

---

## 3. Causalidad operacional

No basta con ajustar el método usando únicamente training.

Toda transformación destinada a producción debe satisfacer:

\[
\hat{x}_t = f(x_1,\ldots,x_t)
\]

y nunca:

\[
\hat{x}_t = f(x_{t-k},\ldots,x_t,\ldots,x_{t+k})
\]

si \(x_{t+1},\ldots,x_{t+k}\) no estaban disponibles en el instante \(t\).

Esto implica, entre otras cosas:

- filtros centrados requieren una versión causal o deben quedar restringidos a análisis offline;
- medias móviles centradas no son válidas para inferencia online;
- STL sobre la serie completa puede inducir leakage;
- normalización global usando datos futuros es inválida;
- transformaciones espectrales deben construirse en ventanas causales;
- parámetros adaptativos deben actualizarse únicamente con pasado disponible.

---

## 4. Modelo conceptual de señal y ruido

Partimos de:

\[
X_t = S_t + N_t
\]

donde:

- \(X_t\): observación;
- \(S_t\): componente estructurada potencialmente útil;
- \(N_t\): componente considerada ruido bajo una definición explícita del dominio.

La relación señal-ruido clásica será:

\[
\mathrm{SNR}=\frac{P_S}{P_N}
\]

o en decibelios:

\[
\mathrm{SNR}_{dB}=10\log_{10}\left(\frac{P_S}{P_N}\right)
\]

En series financieras \(S_t\) y \(N_t\) no son observables directamente. Por ello, la estimación de SNR depende del modelo de descomposición y no debe tratarse como una propiedad absoluta de la serie.

---

## 5. Hipótesis experimentales

### 5.1. Hipótesis A — Degradación monotónica

Al incrementar progresivamente la amplitud del ruido añadido:

\[
X_t^{(\alpha)} = X_t + \alpha N_t
\]

se espera degradación del desempeño.

Esto genera una:

**Noise-to-Performance Degradation Curve**

### 5.2. Hipótesis B — Recuperación por denoising

Aplicando:

\[
\hat S^{(\alpha)} = D(X^{(\alpha)})
\]

se espera:

\[
P(\hat S^{(\alpha)}) > P(X^{(\alpha)})
\]

para un rango significativo de niveles de ruido.

Esto genera una:

**Denoising Recovery Curve**

### 5.3. Hipótesis C — Consistencia en datos reales

Sobre datos reales:

\[
P(D(X_{real})) > P(X_{real})
\]

La mejora debe ser compatible con la magnitud de SNR estimada a partir de training.

### 5.4. Hipótesis D — El residual no debería conservar capacidad predictiva fuerte

Sea:

\[
\hat N = X - D(X)
\]

Si:

\[
P(\hat N) \gg P_{baseline}
\]

entonces el supuesto “ruido” contiene información predictiva relevante y el denoiser está eliminando señal útil.

---

## 6. Experimento de ruido controlado

Se generan múltiples niveles controlados de SNR, por ejemplo:

\[
\mathrm{SNR}_{dB}\in\{\infty,40,30,20,15,10,5,0\}
\]

La lista exacta se define experimentalmente.

La contaminación debe hacerse **después de haber separado train/validation/test**. Nunca se debe contaminar la serie completa antes del split.

---

## 7. Tipos de perturbación

### 7.1. Ruido en una sola feature

Para feature \(j\):

\[
X'_j = X_j + N_j
\]

y:

\[
X'_i = X_i,\quad i\neq j
\]

Objetivo: medir sensibilidad individual de cada entrada.

### 7.2. Ruido independiente en múltiples features

\[
X'_j = X_j + N_j
\]

con perturbaciones independientes entre features.

Objetivo: estudiar degradación acumulativa e interacción.

### 7.3. Ruido correlacionado

\[
N_j(t)=a_jN_c(t)+\epsilon_j(t)
\]

Objetivo: modelar shocks compartidos y perturbaciones sistémicas.

### 7.4. Ruido heteroscedástico

\[
N_t\sim\mathcal{N}(0,\sigma_t^2)
\]

con:

\[
\sigma_t^2=f(V_t)
\]

Objetivo: aproximar condiciones más realistas de mercado.

---

## 8. Estimadores de ruido candidatos

El protocolo debe permitir comparar varias familias:

- STL;
- wavelets;
- multitaper;
- VMD;
- EMD/EEMD/CEEMDAN;
- state-space/Kalman;
- métodos robustos basados en residuos o innovación.

Para wavelets puede incluirse el estimador robusto:

\[
\hat\sigma=\frac{\mathrm{median}(|d_i|)}{0.6745}
\]

sobre coeficientes de detalle apropiados.

---

## 9. Validación automática del residual

Sea:

\[
R_t=X_t-\hat S_t
\]

El residual candidato debe evaluarse con múltiples criterios:

- autocorrelación residual;
- Ljung–Box;
- picos y concentración espectral;
- entropía espectral;
- dependencia no lineal;
- estabilidad temporal;
- capacidad predictiva residual.

Un residual con periodicidades, autocorrelación persistente o capacidad predictiva significativa no debe considerarse ruido puro.

---

## 10. Selección automatizada de parámetros

Sea un conjunto de parámetros \(\theta\).

Se busca:

\[
\theta^*=\arg\max_\theta J(\theta)
\]

donde \(J\) balancee:

- reducción de ruido;
- whiteness;
- ausencia de estructura espectral;
- baja predictibilidad residual;
- preservación de información útil;
- estabilidad fuera de muestra.

La función definitiva debe validarse empíricamente.

---

## 11. Benchmarks externos

Antes de probar el protocolo únicamente sobre datos propios, debe validarse con datasets y modelos ampliamente utilizados en forecasting.

### Datasets candidatos

- ETTh1;
- ETTh2;
- ETTm1;
- ETTm2;
- Electricity;
- Traffic;
- Weather;
- Exchange Rate.

### Modelos candidatos

- DLinear como baseline simple;
- PatchTST u otro modelo moderno ampliamente utilizado.

Objetivos:

1. verificar que la implementación base reproduce aproximadamente resultados conocidos;
2. medir degradación por ruido artificial;
3. medir recuperación mediante denoising;
4. comprobar si el efecto es dependiente de arquitectura.

---

## 12. Experimentos con datos propios

Una vez validado el protocolo externo, aplicar a:

- EURUSD;
- ETHUSDT;
- demás series disponibles;
- periodicidad 1 h;
- periodicidad 4 h;
- datos técnicos;
- datos fundamentales;
- múltiples ramas de features.

Siempre respetando:

\[
4\text{ años train}+1\text{ año validation}+1\text{ año test}
\]

---

## 13. Integración con arquitectura modular multi-branch

La arquitectura modular permite no destruir información irreversiblemente.

Para una entrada \(X\), pueden construirse ramas:

\[
B_1=X
\]

\[
B_2=\hat S
\]

\[
B_3=\hat N
\]

\[
B_4=\mathrm{Wavelet}(X)
\]

\[
B_5=\mathrm{Spectral}(X)
\]

Cada rama produce:

\[
Z_i=E_i(B_i)
\]

El core consolida:

\[
Z_{core}=C(Z_1,\ldots,Z_k)
\]

y las cabezas predictivas producen:

\[
\hat Y_h=H_h(Z_{core})
\]

Esto permite comparar:

- raw only;
- denoised only;
- residual only;
- raw + denoised;
- raw + denoised + residual;
- múltiples representaciones paralelas.

---

## 14. Matriz mínima de ablación

| Experimento | Raw | Denoised | Residual | Ruido añadido |
|---|---:|---:|---:|---:|
| A | Sí | No | No | No |
| B | No | Sí | No | No |
| C | No | No | Sí | No |
| D | Sí | Sí | No | No |
| E | Sí | Sí | Sí | No |
| F | Sí | No | No | Sí |
| G | No | Sí | No | Sí |
| H | Sí | Sí | Sí | Sí |

---

## 15. Métricas

Deben incluir, según el problema:

- MAE;
- MSE/RMSE;
- \(R^2\);
- métricas por horizonte;
- estabilidad por seed;
- intervalos de confianza;
- degradación relativa;
- recuperación relativa.

Definiciones útiles:

\[
\Delta P_{noise}=P_{clean}-P_{noisy}
\]

\[
\Delta P_{recovery}=P_{denoised}-P_{noisy}
\]

y, cuando sea matemáticamente apropiado:

\[
R_{recovery}=\frac{\Delta P_{recovery}}{\Delta P_{noise}}
\]

---

## 16. Criterios de éxito

El paso se considerará validado si se cumplen simultáneamente:

1. la degradación por ruido es reproducible;
2. la curva noise-to-performance es estable;
3. el denoising produce recuperación estadísticamente significativa;
4. el efecto se reproduce en más de un dataset;
5. el efecto se reproduce en más de una arquitectura;
6. el residual no contiene fuerte capacidad predictiva;
7. no existe leakage;
8. el beneficio persiste en validation;
9. el beneficio se confirma finalmente en test.

---

## 17. Criterios de falsación

La hipótesis de utilidad del denoising será rechazada si ocurre consistentemente cualquiera de los siguientes casos:

- no existe degradación sistemática al aumentar ruido;
- el denoising no mejora performance;
- el denoising empeora performance;
- el residual conserva capacidad predictiva significativa;
- la mejora solo aparece en training;
- desaparece en validation/test;
- depende exclusivamente de un dataset;
- depende exclusivamente de una arquitectura;
- requiere información futura;
- es inestable ante pequeñas variaciones de parámetros.

---

## 18. Resultado esperado

Este paso no pretende producir una transformación fija.

Pretende responder empíricamente:

> **¿Cuánta información predictiva se pierde por ruido y cuánto puede recuperarse causalmente?**

Los posibles resultados incluyen:

- \(D(X)\) superior a \(X\);
- \((X,D(X))\) superior a cualquiera individual;
- \((X,D(X),R(X))\) superior;
- el denoising no ayuda.

Todos son resultados científicamente válidos.

---

## 19. Principio rector

> **No eliminar ruido por intuición, sino medir qué perturbaciones deterioran la capacidad predictiva, estimar qué parte de los datos se comporta como ruido, atenuarla causalmente y verificar mediante experimentos falsables si la información aprovechable por el modelo aumenta.**

---

## 20. Estado del proyecto

- **Paso 1 — Muestreo / Nyquist:** cerrado.
- **Paso 2 — Estimación de ruido / SNR:** incorporado a este protocolo.
- **Paso 3 — Denoising y validación experimental:** cerrado como protocolo de investigación.
- **Paso 4 — Cuantización / resolución efectiva:** siguiente tema a estudiar.
