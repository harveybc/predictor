# 01 — Lenguaje atacable

**Vuelve a:** [dictamen principal](../DICTAMEN_RETSU_FUSION_A_B_2026_09_02.md)

Un jurado no necesita un paper de 2019 para tumbar una propuesta. Le basta una palabra que ustedes no pueden definir en una frase. B está llena de ellas. A también, en menor medida. Satoshi no ejecutó esta objeción: la trató como estilo. No es estilo. Es el flanco más barato.

Regla: **si no hay protocolo de medición, la palabra no entra al título ni al objetivo general.**

---

## Palabras que B pone en la portada y son regalos

### 1. *Confiable* (título, footer ×7, objetivo general)

**Dónde:** `<title>`, `<h1>` “Metaoptimización **confiable**…”, footer de las 7 páginas, §1.2 “abstenerse cuando esa transferencia no sea **confiable**”.

**Ataque en sala:** “¿Confiable según qué norma? ¿Calibración de intervalos? ¿No-inferioridad? ¿ISO/IEC 24028? ¿Confianza del usuario? Usted usa la misma palabra para las tres.”

B **nunca** define *confiable*. Lo más cerca es abstención cuando la evidencia no basta. Eso se llama **selectiva**, **calibrada**, o **con abstención**. No se llama confiable.

*Confiable* en español de propuestas doctorales es primo de *trustworthy AI*: seguridad, robustez, equidad, trazabilidad, accountability. Ustedes no van a entregar ese stack. Ponerlo en el título es un cheque que el jurado cobra en el minuto 3.

**Sustituto:** *Metaoptimización selectiva* / *con abstención* / *bajo procedencia estratégica*.  
Objetivo general: “abstenerse cuando la evidencia no alcanza un umbral de calibración **predeclarado**”.

### 2. *Trazas verificables* / *trazas verificadas* (resumen y pregunta)

**Ataque:** “¿Verificadas por re-ejecución, por firma, por lookup tabular, por quorum, o por el candidato?”

En `doin-domains`, D1 verifica por **lookup** (oráculo exacto). WP3 verifica holdout. El PoO de `doin-core` registra un incremento que **alguien** marcó `verified_performance`. Ninguno de esos verbos es el de B.

**Sustituto:** “trazas con procedencia registrada y, cuando el dominio lo permite, **re-ejecutables**”. En HPO-B: “consultas a una tabla”. No *verificables*.

### 3. *Salida segura* (§1.3 objetivo 2)

**Ataque:** “¿Segura contra qué pérdida? ¿Safety? ¿Fail-closed? ¿O solo ‘volver a BO’?”

**Sustituto:** “salida **sin transferencia**” / “abstención”. La palabra *segura* es de sistemas críticos. No la tienen.

### 4. *Información útil* (distinción central §2.1, tabla 3.1)

B distingue capacidad / información adquirida / información útil. Bien como intuición. Mal como término: *útil* es utilidad, no bits. Un teórico de la información preguntará “¿útil en el sentido de Shannon, de decisión, o de generalización?”. B cita Xu–Raginsky y PAC-Bayes-compresión, y luego Entropy Search. Tres “informaciones”.

**Sustituto en el cuerpo:** tres nombres feos y distintos — *capacidad de la familia*, *ajuste al entrenamiento*, *ganancia esperada de información sobre el óptimo por unidad de costo*. Si no sobreviven la ablación, se borran. No se llaman “teoría de la información” en el resumen.

### 5. *Inteligencia* (incluso para negarla)

§2.1: “esta tesis no multiplicará conexiones por dos para estimar la **inteligencia** de una red”. La negación **introduce** el concepto. Un jurado malicioso lee: el candidato pensó en eso.

**Sustituto:** no mencionar inteligencia de redes. MacKay, si sobrevive, es “número de dicotomías de un umbral lineal”, punto.

### 6. *Red / entorno distribuido y trazable* (resumen)

El resumen de B mete DOIN en el segundo párrafo. El lector de comité oye blockchain. B luego dice que el resultado no depende de ella. Entonces **no va en el resumen**.

**Sustituto:** una línea en viabilidad, no en el lead.

### 7. *Calidad variable* (pregunta abierta §2.3)

Suena a ruido de medición. El objeto de la fusión es **calidad estratégica**. Si no dicen *estratégica* / *manipulable*, el adversario de H2 parece un script del candidato. Satoshi vendió la endogeneidad; B aún habla como si el mundo ensuciara las trazas solo.

---

## Palabras de A que no deben reentrar

A ya no es el objeto. Si reaparecen, el PDF se bifurca.

| Frase de A | Por qué no |
|---|---|
| *evaluación confiable* de LLM/agentes | Mismo pecado, otro dominio. Oráculo parcial **eliminado**. |
| *Correlated Agreement* / *elicitación entre pares* | Sin verdad parcial no hay peer prediction. Citarla es reabrir Gao. |
| *mercado descentralizado* en la pregunta | Akash es un mercado descentralizado de CPU. Ustedes no. |
| Hayek “el precio transmite conocimiento” | En el flujo real **no hay precio**. Hayek no aplica. Citado en A, es un error de categoría: un economista lo cierra en un minuto. Ver [08](08_FLUJO_MINERO_NO_MERCADO.md). |
| *acuerdo informado* como verdad | Sigue siendo cierto en abstracto. En HPO-B hay tabla. No lo necesitas. Dejarlo para la imposibilidad, no para el slogan. |

---

## Palabras que parecen técnicas y no lo son (ambas)

| Palabra | Uso suelto | Uso permitido |
|---|---|---|
| *óptimo* | “ubicación del óptimo” sin decir simple regret vs. best-seen | *configuración de mejor desempeño tabulado en el horizonte H* |
| *garantía* | “garantía útil” §3.3 de B | *cota bajo supuestos (α, s, δ)* o *región de imposibilidad* |
| *robusto / robustez* | B §2.1 topología y precisión | *no-inferioridad bajo cambio de espacio*, con margen |
| *novedad* | B §2.3 “la novedad no puede ser…” | no auto-declarar novedad; dejar que el SOTA la delimite |
| *universal* | B dice que no lo es; OptFormer sí lo pone en el título ajeno | citar el título de Chen et al. entre comillas, no adoptarlo |
| *descentralizado* | maestría, DOIN, A | solo como runtime, nunca como hipótesis |
| *verificación* | PoO, domains, B, Gensyn | siempre con complemento: *de qué* |

---

## Título candidato (ninguno usa *confiable*)

En orden de honestidad:

1. **Metaoptimización selectiva a partir de trazas de procedencia estratégica**  
2. **Transferencia de hiperparámetros con abstención y corrupción acotada del corpus**  
3. **Aprendizaje de políticas de búsqueda bajo evidencia manipulable**

Si el capítulo de mecanismo cabe: subtítulo, no título — *con un mecanismo de re-ejecución que acota la fracción corrupta*.

Si no pueden jurar el subtítulo, usen (2) y KEEP B. No pongan *red*, *descentralizada*, *confiable*, *inteligente*, ni *DOIN* en el `<h1>`.

---

## Prueba de fuego (una frase)

Antes de mandar el PDF: cada adjetivo del título debe responderse con un **número o un protocolo** en la misma página. *Selectiva* → umbral de abstención predeclarado. *Estratégica* → mejor respuesta contra el protocolo publicado. *Acotada* → \(\delta\) de H2. *Confiable* → no hay número. Por eso se va.
