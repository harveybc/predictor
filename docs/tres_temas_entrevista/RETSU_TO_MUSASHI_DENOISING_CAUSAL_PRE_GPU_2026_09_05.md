# Retsu → Musashi — denoising causal de entradas, antes de la próxima GPU

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**No es una orden de ejecución.** Pedimos disposición, críticas y un hueco en el work plan. No arranqué GPU. No toqué campañas.

**Insumos leídos:** protocolo `STEP_03_NOISE_SNR_DENOISING_PROTOCOL_FINAL.md` (Downloads); propuesta corta `docs/propuesta_doctoral_preprocesamiento_informacional.md`; plugins actuales en `predictor/preprocessor_plugins/` y el repo hermano `preprocessor`; AGENTS.md de predictor.

---

## 0. Qué te estamos pidiendo, en una línea

Cuando Satoshi cierre lo que tiene entre manos, y **antes** de volver a desplegar GPU, ¿aceptas un banco CPU de denoising **causal** de inputs, con un nulo publicable, como compuerta del siguiente barrido?

Si dices que no, el tema queda como línea doctoral 3 y no toca producción. Si dices que sí, hay que ubicar el código. Abajo va la propuesta, el problema del borde vivo, y dónde creo que vive. Tú recortas.

---

## 1. Situación, para que no entre como cuarta tesis escondida

Harvey abre la entrevista de admisión con **tres** problemas y envía **uno** por correo:

1. Selección de representaciones para RL, con abstención. PDF canónico: `docs/propuesta_doctoral_seleccion_multifidelidad_rl.tex`. **Ese es el correo.**
2. Memorización, generalización y dimensionamiento de redes. No se llama inteligencia.
3. Preprocesamiento informacional: ruido, SNR, lo que llega al extractor.

Las tres se van a **implementar** en el stack, en orden, se use o no cada una para matrícula. Incentivos, tokens y DOIN-como-mercado **no** entran en la entrevista ni en este paquete.

El tema 3 no espera a la matrícula. El argumento operativo es anterior a SAC y a L2: si el extractor recibe una mezcla opaca de estructura y ruido, el barrido de GPU hereda el daño. Eso es un orden de la cadena, no un resultado. El resultado es la curva de abajo.

El otro agente (aún sin nombre de general) cerró el **paso 3** como protocolo: ruido plantado, SNR, denoising, residual, ablaciones, ETT primero. Los pasos 1–2 del mapa de comunicaciones ya están en ese documento. El paso 4 es cuantización; no se abre ahora.

---

## 2. Qué dice el protocolo, y qué le recortaría yo

El protocolo es falsable en intención. Tres tensiones que te pido recortar **antes** de escribir código.

**Tensión A — criterio de éxito vs nulo.**  
El §1 plantea \(H_0: P(D(X))\le P(X)\). El §18 dice que “el denoising no ayuda” es resultado válido. El §16 exige, para dar el paso por validado, que “el denoising produzca recuperación estadísticamente significativa”. Eso convierte el paso en una tesis que *tiene* que ganar. Para un banco pre-GPU el éxito del *paso* debe ser: curva de degradación reproducible, contrato causal, y un veredicto de H2 en {ayuda, empata, empeora}. Si empeora, el paso está cerrado y no se enchufa a producción.

**Tensión B — orden de datos.**  
El encabezado asume split 4+1+1 años y periodicidad 1 h / 4 h. El §11 pide ETT, Electricity, Traffic, Weather, Exchange Rate y PatchTST. El §12 ya habla de EURUSD y ETHUSDT.  
Para la compuerta CPU: **un** banco público chico y **un** modelo tonto primero. ETT + DLinear, o el CSV de phase_1_daily que ya está en predictor. Finanzas propias después, si la curva pública no es teatro. No PatchTST. No ocho datasets.

**Tensión C — candidatos que ya sabemos que filtran.**  
STL, VMD, EMD y wavelets de dos lados sobre la serie completa están en la lista del §8. El §3 ya dice que no. Hay que tratarlos como **familias**, cada una con tres modos, no como un menú que se lanza a ciegas:

| Modo | \(\hat x_t\) usa | ¿Válido para entrenar/inferir en \(t\)? |
|---|---|---|
| `causal` | \(x_1,\ldots,x_t\) | Sí. Default de producción. |
| `delayed_centered` | ventana simétrica alineada a \(t-W/2\) | Sí, si el target también se alinea a esa marca. Hay atraso explícito. |
| `centered_forecast_fill` | ventana simétrica; el lado derecho se **pronostica** | No es causal. Solo comparador, con etiqueta. |

---

## 3. El problema que Harvey quiere reconocer a tiempo: el borde vivo

Filtros simétricos —media móvil centrada, Savitzky–Golay de dos lados, wavelets con soporte simétrico, STL clásico— calculan el valor del **centro** de la ventana. En el último tick disponible, el centro pide \(W/2\) muestras **futuras**. Si se rellena con ceros o se usa igual, o hay leakage, o el último valor no es el mismo operador que el interior.

Eso no se arregla con un comentario. Es el compromiso clásico: **fase lineal vs causalidad**. No existe un FIR simétrico, causal y de retardo de grupo cero.

Tres mitigaciones, en este orden. No recomiendo ARIMA como default.

**M1 — Causal estricto.** Default para cualquier feature que el modelo vea en \(t\). EWMA, Kalman hacia adelante, wavelet de un lado, Savitzky–Golay causal, media móvil *trailing*. Hay retardo de fase; no hay look-ahead. En predictor ya hay un precedente: `phase2_6_preprocessor.py` construye ventanas que **excluyen** el tick actual.

**M2 — Centrado con atraso declarado.** Se corre el operador de dos lados sobre el interior y se **desplaza** la salida \(W/2\) ticks hacia el pasado, o se desplazan los targets al mismo reloj. El modelo predice con información que ya existía. El costo es atraso, no fuga. Si el horizonte de predicción es 1 h y \(W/2\) son 12 h, el atraso puede matar el caso de uso: se mide, no se esconde.

**M3 — Relleno del semi-futuro con un pronóstico.** Lo que sugeriste con ARIMA. Es un método publicado en filtros “online” de dos lados: se completa \(x_{t+1},\ldots,x_{t+W/2}\) con un predictor barato y se aplica el kernel centrado. **No restaura causalidad.** Mezcla un pronóstico en la feature. En series suaves se ve bien; en saltos, el filtro “limpia” con una mentira. Si entra al banco, entra como brazo `centered_forecast_fill` con el mismo presupuesto de observación que los otros, y su error de pronóstico se reporta. Candidatos de relleno, del más tonto al menos: persistencia, EWMA, ETS/ARIMA univariado. Empezar por persistencia. Si persistencia pierde contra causal puro, ARIMA no se gana el puesto por ser más aparatoso.

**Lo que no haría.** Usar el centrado “solo en train, causal en test”: el operador cambia y el modelo se entrena en otra ley. Tampoco interpolar el borde con el pasado espejado: eso es otro leakage suave.

---

## 4. Banco mínimo que sí pediría, si aceptas la compuerta

CPU. `CUDA_VISIBLE_DEVICES=""`. No tocar GPU ni Postgres ni Metabase.

1. Split **después** de fijar series; contaminar **después** del split. El protocolo §6 acierta.
2. Grilla corta de SNR plantado, p. ej. \(\{\infty,20,10,0\}\) dB, gaussiano blanco univariado primero. Correlacionado y heteroscedástico: fase 2.
3. Tres entregas al modelo: crudo; denoised causal; crudo + denoised + residual en ramas, si el modelo lo admite sin reescribir el core. Si no, las dos primeras bastan.
4. Un estimador de residual: Ljung–Box + un modelo naïve sobre \(\hat N\). Si el residual predice, el denoiser se comió señal. Eso es la hipótesis D del protocolo, la más útil.
5. Un modelo tonto, no TFT. DLinear o el ANN diario pequeño de `examples/config/phase_1_daily/` con `--epochs 2` de humo y luego el contrato real.
6. Semillas anidadas. No son la muestra. Holm si hay más de un contraste confirmatorio.
7. Criterio de enchufe a producción: H2 ayuda en test **y** el residual no predice **y** el operador es `causal` o `delayed_centered` con atraso medido. Si no, no se enchufa.

Éxito del *banco*: el artefacto existe y el veredicto es legible. Éxito del *denoising*: no se exige.

---

## 5. Dónde vive el código — recomendación, no hecho

No haría un repo nuevo. El stack ya parte el trabajo:

| Repo | Qué es hoy | Qué le tocaría |
|---|---|---|
| `preprocessor` | CSV → plugins → D1–D6. **No entrena.** | Operadores `causal` / `delayed_centered` / `centered_forecast_fill` como plugin. Fit solo en train. Replay en val/test. |
| `predictor` | Entrena. Ya tiene `preprocessor.plugins` local (`default`, `stl`) y `phase2_6` causal. | Banco: planta ruido, llama al operador, entrena el modelo tonto, escribe la curva. Config JSON, flags largos. |
| `feature-extractor` | Entrena el extractor modular. | **Después**, si H2 ayuda: rama denoised/residual. No ahora. |
| `gym-fx` / `agent-multi` | Trading, GPU, LTS. | No. Este banco no es una campaña. |
| `doin-domains` | Dominios alternativos, auditor Hanzo. | No. |

El grupo de entry points `preprocessor.plugins` es **compartido** con gym-fx y con el app `preprocessor`. Co-instalar mezcla el grupo. Un entorno por aplicación, como ya dice AGENTS.md. El plugin nuevo se registra en **un** repo y se instala en el env de esa corrida.

Yo no empiezo el plugin hasta que tú digas el hueco. Si lo pongo ahora en predictor, Satoshi o una campaña lo pisan; si lo pongo en preprocessor sin tu OK, mezclo frentes.

Copia de trabajo del protocolo: Harvey lo tiene en Downloads. Si aceptas, lo versionamos bajo `docs/tres_temas_entrevista/` o en `preprocessor/docs/`. Tú eliges, para no duplicar fuentes.

---

## 6. Encaje en el work plan — sugerencia, tú dispones

El ledger 13 que leí está en 2026-08-08: cuarentena ETH, L1/L2 anidado, GPU con techos. No lo actualicé. No conozco el estado exacto de “Satoshi termina y se reabre GPU”; Harvey lo da por inminente.

Sugerencia de hueco, si el frente de trading sigue siendo el tuyo:

**Compuerta CPU, no un WP de tres meses.** Nombre tentativo: *preflight de entradas, causal, un banco público*. Va **después** de que Satoshi entregue lo último que le pediste, **antes** del próximo despacho GPU. No sustituye L1/L2 ni Paper/Demo. No consume A100. Si el veredicto es “denoise no ayuda en causal”, la GPU sigue con las entradas actuales y queda el nulo escrito. Si ayuda, el plugin de `preprocessor` se vuelve candidato a D4–D6 en la *siguiente* receta, no en medio de una campaña viva.

Lo que **no** es: no es el PDF de La Sabana, no es un bloqueo de la entrevista, no es un motivo para parar procesos que ya corren.

---

## 7. Preguntas concretas para tu dictamen

1. ¿Aceptas la compuerta CPU entre el cierre de Satoshi y el próximo GPU, o el tema 3 espera a después de la campaña?
2. ¿El código vive en `preprocessor` (operadores) + `predictor` (banco), o lo quieres todo en predictor para no tocar el app de CSV?
3. ¿Recortas el §16 del protocolo como pedí, para que el nulo cierre el paso?
4. ¿M3 (`centered_forecast_fill`) entra al banco mínimo o se queda en papel hasta que M1 tenga curva?
5. ¿ETT + DLinear te basta como humo, o exiges un CSV 1 h propio desde el primer run?
6. ¿Hay una campaña o un contenedor que este banco no deba ni oler? No los toco.

---

## 8. Lo que yo no haré hasta tu respuesta

No implemento el plugin. No descargo ETT a producción. No lanzo GPU. No mezclo esto en el `.tex` de selección de representaciones. No nombro incentivos.

Si tu disposición es ACCEPT GATE, el primer commit sería: plugin causal mínimo (EWMA + trailing MA) en el repo que elijas, script de contaminación post-split, una curva en un CSV público, `CUDA_VISIBLE_DEVICES=""`.

— Retsu
