# Retsu → Harvey — G12: el correo no se cambia por omisión

**Fecha:** 2026-09-06  
**De:** Retsu  
**Para:** Harvey  
**Copia:** Satoshi, Musashi  
**No es una orden de ejecución.** No cambio TITULOS. No sustituyo el PDF. No implemento T0+T1.

Satoshi, en `3116c362`, verificó el PDF de transformaciones (`bb051911a9d7…`, 10 pp, `5449bed`) y recomienda esa propuesta como **madre de La Sabana**, con la multifidelidad-RL como capítulo aplicado. Dice, literal, que G12 debe resolverse por decisión suya explícita, no por omisión.

Usted me lo puso delante. Registro el dictamen. **No elijo por usted.**

---

## 1. Hecho

Hay dos PDFs doctorales committeados, ambos con meta-selección + abstención + transferencia:

| | Transformaciones | L2 / multifidelidad-RL |
|---|---|---|
| Fuente | `docs/propuesta_doctoral_transformaciones_series_temporales.tex` | `docs/propuesta_doctoral_seleccion_multifidelidad_rl.tex` |
| PDF | 10 pp, SHA `bb051911a9d7…cad` | 9 pp, SHA `23aece900cc8…ddf` |
| Commit de fuente | `5449bed` | canónico vigente de La Sabana |
| Banco | público + sintético de verdad conocida; finanzas **después** | POPGym / CARL / ARLBench; RL con abstención |
| Correo **hoy** (G12) | no | **sí** — tema 1 de `TITULOS_ENTREVISTA.md` |

Un jurado que reciba uno y encuentre el otro en el repo verá una tesis con dos sustantivos. Satoshi no exagera.

G12 se cerró para que el paquete tres-fuentes **no** se colara en el correo. No se cerró contra un segundo PDF que entonces no existía.

---

## 2. Las dos opciones (diga A o B)

### A — Ratificar G12. El correo sigue siendo L2 RL.

- Mañana se envía `propuesta_doctoral_seleccion_multifidelidad_rl.pdf`.  
- Transformaciones queda como suelo de representaciones / work plan / posible segunda propuesta, no como PDF de admisión.  
- Coherente con Musashi `570f517` §7.2 («no tocar el PDF que se enviará de inmediato») **antes** de que existiera `5449bed`.  
- Riesgo: el jurado pregunta por qué hay otra propuesta committeada con la misma gramática (selector, abstención, transferencia).

### B — Levantar G12. El correo pasa a transformaciones.

- Mañana se envía `propuesta_doctoral_transformaciones_series_temporales.pdf`.  
- L2/RL queda como capítulo aplicado (representación para el agente, no la pregunta madre).  
- Es la recomendación técnica de Satoshi: banco público independiente de lo financiero, verdad conocida, novedad ya recortada por tres jurados internos, menor exposición a «ingeniería interna».  
- Yo entonces sí actualizo `TITULOS_ENTREVISTA.md`, el README de entrevista, y reescribo G12: el correo es el tema de transformaciones; L2 no desaparece, deja de ser el adjunto.  
- Riesgo: el pitch de veinte segundos y el PDF que el comité ya espera (si alguien lo anticipó como RL) cambian de objeto. Usted es quien sabe si el correo ya salió o no.

No hay opción C silenciosa. No hay “mientras tanto alinear README al correo nuevo”.

---

## 3. Lo que yo hago mientras espera

- Dictamen de verificación: `DICTAMEN_RETSU_SATOSHI_TRANSFORMACIONES_VERIFICADO_2026_09_06.md`.  
- Carta a Musashi: T0+T1 = G3, un banco; no insertar planes 17/38; tabla STEP↔T↔E de papel.  
- Nota en el README del expediente de transformaciones: tesis ⊂ work plan (Tabla 1 vs STEP 11).  
- **No** edito el `.tex` de transformaciones con las menores de Satoshi §4.4 (−5 dB, DLinear, \|G\|, RAPL, método de la ec. 3) hasta que usted lo pida.  
- **No** lanzo T0+T1. **No** GPU. **No** B4.

Cuando escriba **A** o **B**, ejecuto solo ese lado.

— Retsu
