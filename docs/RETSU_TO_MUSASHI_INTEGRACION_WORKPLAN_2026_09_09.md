# Retsu → Musashi — integración de propuestas, métricas de información y cola experimental

**Fecha:** 2026-09-09  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  

**No es orden de GPU, live, B4, reset del cubo poblado, ni de cambiar G12.**

Harvey pidió recoger las propuestas que nacieron al escribir el doctorado, mirar dónde vas en el work plan, e integrar **todo** —incluida su propuesta modular vigente y una lane de métricas de información / grafos / cubo— aunque la cola experimental se reinicie. Prefiere reordenar a seguir entrenando con preproceso por defecto y sin esas métricas.

Paquete:

`docs/integracion_workplan_2026_09_09/`

Empieza por `00_LEER_PRIMERO.md`. El plan ejecutable es `04_PLAN_INTEGRACION_JERARQUICO.md` (I0–I14).

## Dónde te sitúo (sin re-auditar F1/F2 hoy)

Dos work plans:

1. **Negocio / cinco frentes** — tu carta del 7-sep. F2 B4 no corría; F1 live no autorizado; F3 académico abierto; F5 separado.
2. **Informacional / STEPs 01–13** — master v2 + PATCH 001–003. 01–07 están *escritos*. Yo recorté el 5-sep: **protocol closed ≠ hecho**. El siguiente *protocolo* era 08. El siguiente *hecho* no lo era. T0/T1 del preprocessor existen; T2 no está sellado.

La cola que Harvey quiere ahora: **instrumentar información y grafos (I1–I4) y ejecutar el banco de ruido (I5) antes de STEP 08 y antes de otro barrido de modelos.**

## Qué integras (sin fusionar tesis)

| ID | Qué |
|---|---|
| P-MOD | representaciones modulares — tex que Harvey edita (15 pp, 9-sep) |
| P-L2 | selector multifidelidad RL — G12 sigue aquí hasta que él diga |
| P-CAP | memorización / dimensionamiento |
| P-TRN / T0–T5 | transformaciones |
| P-PRE / STEP 03 | ruido-SNR-denoise |
| P-3F | suelo extractor-núcleo-cabezal (work plan) |
| P-INC | incentivos; F5, no el PDF |
| I-INFO | lane nueva: `D.*`/`Y.*`/`M.*`/`G.*` + hipótesis H-ES |

Tesis ⊂ work plan. El cubo **registra**. No prueba H1 de P-MOD.

## Hipótesis de Harvey (I4)

Patrón entre complejidad del modelo (compresión de pesos + grafo), información de datos/target, y overfitting. Early stopping informacional como **alternativa a probar**, no como default.

Yo no dejo escribir “Kolmogorov” en una columna del cubo. \(K\) es incomputable. El medidor es longitud comprimida \(L_C\) con zstd/lzma/bz2 y manifiesto fijo. En series naturales no hay “bits libres de ruido” verdaderos; en sintético sí, porque hay generador. MDL de STEP 05 no es \(K\) ni \(C\) de MacKay. P-CAP mide otra cosa (bits de etiquetas aleatorias). Se registran aparte.

## Qué te pido

1. Leer I0–I14. Si el ledger de F2/F1 se movió el 8–9, anotas la divergencia; no reescribes el paquete para acomodar un relanzamiento de B4.
2. Abrir I1 en Postgres **throwaway**. El bug `metric_value` del ETL se trata ahí, no en el cubo real.
3. Poner I2/I3 (datos/modelo) **antes** de sellar T2 y **antes** de E2 de P-MOD, para no reentrenar “porque faltaban métricas”.
4. I5 = primer hecho de la cadena. I8 E0 de P-MOD puede ir en paralelo CPU.
5. H-ES no toca `early_patience` de campañas hasta que falle o gane en reservadas.
6. No me despaches T3–T5, P-L2 ni STEP 08 como “ya”. Sus precondiciones están en I7/I11/I12.

Si hay pelea de hueco, tú calendarias. Yo no reasigno GPU.

Retsu
