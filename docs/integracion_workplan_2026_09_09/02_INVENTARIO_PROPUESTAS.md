# 02 — Inventario de propuestas

Todas viven bajo `docs/` de `predictor`. El directorio `docs/tres_temas_entrevista/` es el índice histórico (el README de ahí **aún dice** que la #1 L2/RL es la elegida para La Sabana; eso quedó **desactualizado** respecto del objeto que Harvey edita el 9-sep). G12 (el adjunto del correo) **no se cambia aquí**.

## 1. Objetos doctorales (preguntas distintas)

| ID | Nombre | Fuente canónica | Pregunta (una línea) | Unidad | Qué **no** es |
|---|---|---|---|---|---|
| P-MOD | Representaciones temporales modulares | `docs/propuesta_doctoral_representaciones_temporales_modulares.tex` (PDF 15 pp, commit de hoy) | ¿Un procedimiento de perfiles→grupos→campos receptivos→fusión mejora el pronóstico fuera de muestra frente a controles de arquitectura? | Tarea de pronóstico (conjunto × objetivo × horizonte) | Ni el selector L2, ni DOIN, ni el cubo como contribución |
| P-L2 | Selección multifidelidad RL | `docs/propuesta_doctoral_seleccion_multifidelidad_rl.tex` + `docs/tesis_sac/` | ¿Cuándo comprar curvas parciales y cuándo abstenerse al elegir un codificador de memoria bajo cambio de tarea? | Tarea RL pública + presupuesto de fidelidades | No diseña el espacio de codificadores |
| P-CAP | Memorización / dimensionamiento | `docs/tres_temas_entrevista/02_memorizacion_generalizacion_dimensionamiento.md` | ¿Calibrar capacidad de memorizar + pérdida temprana estima el tamaño suficiente? | Familia × tamaño × tarea con generador conocido | No es \(C\) de MacKay ni Kolmogorov |
| P-TRN | Transformaciones temporales | `docs/propuesta_doctoral_transformaciones_series_temporales.tex` + `docs/tesis_transformaciones_temporales/` | ¿Un grafo pequeño de operadores, con abstención, transfiere entre tareas temporales bajo presupuesto? | Tarea × grafo de operadores | `predictor` no es el sustrato; T0–T5 |
| P-PRE | Preproceso informacional (ruido/SNR) | `docs/propuesta_doctoral_preprocesamiento_informacional.md` + `docs/tres_temas_entrevista/03_preprocesamiento_informacional.md` | ¿Denoise causal entrega al extractor más estructura que la serie cruda? | Serie pública + ruido **plantado** | No es V.90; no es capacidad de canal del extractor |
| P-INC | Recompensas / autoselección multidominio | `docs/tres_temas_entrevista/04_incentivos_red_descentralizada_multidominio.*` | Controlador de \(\pi_d(t)\) frente a nodos que aprenden | Dominio × política publicada | No es el ledger-como-verdad; antecedente 04_antecedente_* es **otra** pregunta |
| P-3F | Tres fuentes → sistema modular | `docs/SATOSHI_PROPUESTA_TRES_FUENTES_CONOCIMIENTO_MODULAR_2026_09_05.md` | Arquitectura de conocimiento (extractor / núcleo / cabezales) como **suelo** | Work plan, no tesis | Satoshi ACCEPT WITH GATES; no se vende como doctorado |

Expedientes de trabajo (no son una quinta tesis):

- `docs/tesis_representaciones_modulares/` — contrato, U01–U12, Takeshi R2.
- `docs/tesis_sac/` — L2/RL.
- `docs/tesis_transformaciones_temporales/` — T0–T5.
- `docs/fusion_a_b/` — fusión A/B **rechazada**; no reabrir.
- `docs/propuesta_doctoral_doin_borrador.*` y `...aprendizaje_multiagente...` — borradores; no son el objeto vigente.

## 2. Relación permitida (no fusión)

```text
P-PRE + STEPs 01-05 + T0/T1     →  suelo de datos / información
P-TRN (T0-T5)                   →  operadores versionados que sobreviven gates
P-MOD (E0-E3)                   →  cómo agrupar y ramificar *después* de tener perfiles
P-CAP                           →  cómo no sobredimensionar el núcleo
P-L2                            →  cómo *elegir* entre artefactos ya construidos, con abstención
P-3F                            →  mapa extractor-núcleo-cabezal (work plan)
P-INC + F5                      →  incentivos entre dominios; fuera del núcleo doctoral
I-INFO (este paquete, doc 03)   →  métricas que *todos* los runs deben dejar en el cubo
```

Una mejora en P-PRE no prueba P-MOD. Un selector L2 no diseña ramas. Un proxy de compresión no es la tesis P-CAP.

## 3. Hipótesis doctorales que **sí** hay que experimentar (resumen)

Tomadas de los textos, no inventadas.

### 3.1 P-MOD (E0–E3)

- **H1** diseño basado en datos vs control principal (sin ramas o modular sin perfiles).
- **H2** escalas / campos receptivos (pendiente en sintético).
- **H3** agrupación y fusión (dos diferencias).
- **E0** sintético; **E1** desarrollo público; **E2** familias reservadas; **E3** finanzas + RL, **no** sostiene H1–H3.

### 3.2 P-PRE (H1–H3 de ruido)

- Ruido plantado degrada.
- Denoise causal **no** es gratis en todos los SNR.
- Señal+residuo en ramas vs tirar el residuo.

### 3.3 P-TRN (T0–T5)

- Calibración con perturbación conocida → utilidad pública → transferencia → adaptador DOIN → validación aplicada.
- Identidad, búsqueda sin transferencia o abstención pueden ser el resultado correcto.

### 3.4 P-CAP

- \(\widehat C_{mem}\) por familia/tamaño/precisión.
- Ajuste específico vs generalización en bits (pérdida log2).
- Estimar \(N_{\min}\) con presupuesto inicial vs curvas y proxies sin entrenamiento.

### 3.5 P-L2

- Best-arm / abstención bajo ruido y deriva; fidelidades del **mismo** candidato.
- Banco público de RL; finanzas como confirmación posterior.

### 3.6 I-INFO (nueva, Harvey 9-sep) — ver doc 03

- Completitud de métricas de información y grafo en el cubo.
- Contraste complejidad-modelo vs información de datos/target.
- **H-ES:** ¿un criterio informacional anticipa el epoch de overfitting mejor que la paciencia sobre val loss?

## 4. Qué Musashi **no** fusiona

1. No pegar P-INC en el PDF modular.
2. No promover STEP 11 (máscara AE) a capítulo de P-MOD (ya recortado: tesis ⊂ work plan).
3. No tratar P-3F como tercera tesis.
4. No reabrir `fusion_a_b`.
5. No usar el cubo como prueba de P-MOD: el cubo **registra**; H1–H3 se prueban con el protocolo de P-MOD.
