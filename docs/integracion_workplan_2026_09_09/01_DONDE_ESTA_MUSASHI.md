# 01 — Dónde está Musashi

**Corte documental de este paquete:** 2026-09-09.  
**Estado de frentes citado:** carta Musashi→Takeshi `docs/RESPUESTA_MUSASHI_CONTEXTO_Y_ALINEACION_DOCTORADO_2026_09_07.md` (7-sep). No re-verifiqué B4/LTS/GPU hoy. Si el ledger de trading se movió el 8–9, gana el ledger, no esta tabla.

Hay **dos** work plans. No son el mismo documento.

## 1. Work plan de negocio (cinco frentes)

Fuente: Musashi, 7-sep-2026, §3.

| # | Frente | Estado al 7-sep | Siguiente acto que **no** mezcla este paquete |
|---|---|---|---|
| F1 | Live / paper / demo | Infra activa; **live real no autorizado** | No se usa para calibrar representaciones ni métricas nuevas |
| F2 | Optimización (L1/L2, B4) | **B4 no está corriendo**; incidente de intérprete 6-sep | No relanzar B4 para “llenar el cubo” de I1–I4 |
| F3 | Académico | Propuesta abierta; objeto Harvey = representaciones modulares (tex 9-sep) | Este paquete alimenta F3 |
| F4 | Social | Laboratorio y contabilidad; sin producto live | Fuera de I0–I12 |
| F5 | Dominios alternativos | Activo y **separado**; WP3-C nulo acotado MNIST | No se pisa WP1; no se trae GPU de dominios aquí |

CRISP-DM del mismo informe (no inventar fase 7): negocio → datos → preparación → modelado → evaluación → despliegue. L2/DOIN atraviesa modelado y evaluación; no es una séptima fase.

## 2. Work plan informacional (cadena 01–13)

Fuente canónica: `docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v2.md`  
Parches: PATCH 001, 002, 003.  
Recorte Retsu: `docs/RETSU_TO_MUSASHI_WORKPLAN_V2_PATCH001_2026_09_05.md`.

| STEP | Pregunta | Estado real (Retsu, 5-sep) | Dónde vive el código *cuando* exista |
|---|---|---|---|
| 01 | Muestreo / Nyquist | Protocolo escrito | contrato de barra/punto (PATCH 003) |
| 02–03 | Ruido / SNR / denoise causal | Protocolo escrito. **Experimentos no empezados.** Primer hecho = banco CPU | repo `preprocessor`, no `predictor` |
| 04 | Cuantización / companding | Protocolo escrito | mismo sitio, después de 4A–4B |
| 05 | Fuente / entropía / MDL | Protocolo escrito. **No es Kolmogorov.** | diagnóstico 5A–5B, alfabeto provisional |
| 06 | Tiempo-frecuencia | Protocolo escrito | caché causal de rFFT |
| 07 | Detector / matched filter | Protocolo escrito. Detector ≠ AE | sintético 7A; MiniRocket control |
| 08 | Equalización | **Siguiente protocolo.** No siguiente plugin. ≠ StandardScaler | 8A sintético; CSI |
| 09 | Crosstalk común/único | Protocolo escrito | tabla 9A; no restar factores “causales” |
| 10 | Sincronía / \(\tau\) | Protocolo escrito | metadato \(\tau\) antes de shift |
| 11 | Redundancia / máscara AE | tesis ⊂ work plan; AE **existente** | `feature-extractor`, train-only |
| 12 | Router por calidad | elige modos ya validados | necesita ≥2 modos y medidores 03–05 |
| 13 | Presupuesto multi-rama | Pareto de anchos, no NAS | composite 64:32:32:32 |

Carriles transversales ya aceptados (no tesis paralelas):

- **C1–C7** compresión (PATCH 001): sparse, residual jerárquico, rate-distortion latente, etc.
- **Eventos económicos / causal** (PATCH 003): panel de eventos, no serie uniforme.
- **L3 meta-opt** (PATCH 003): no bypasea L2 ni el holdout.

## 3. Preprocessor T0–T2 (ya en marcha, no se tira)

Fuente: Musashi 7-sep §3.2 y `docs/tesis_transformaciones_temporales/03_WORKPLAN_PATCH_TRANSFORMACIONES_DOIN.md`.

| Paquete | Estado | Qué hacer en la integración |
|---|---|---|
| T0 contrato/censo | **Hecho** en `preprocessor` (mecánica causal `fit/transform`, paridad) | Conservar |
| T1 banco CPU verdad conocida | **Calibrado en laboratorio** | Conservar; I5 se *engancha* aquí, no lo duplica |
| T2 utilidad pública | **No sellado** (hallazgos C31–C36; geometría de dos orígenes pendiente) | No sellar T2 hasta que I2 registre métricas de información del mismo run |
| T3–T5 meta-selección / adaptador DOIN / validación aplicada | No empezados | Después de I5–I8; T4 no con sintético solo |

## 4. Dónde **no** estás

- No estás en STEP 08 como plugin.
- No estás en un barrido de modelos para “descubrir” el preproceso.
- No estás autorizado a usar el cubo OLAP de campañas reales como laboratorio de I1 (throwaway).
- No estás reescribiendo el PDF modular de Harvey.

## 5. Encaje de la orden de hoy

Harvey: *integrar todo, aunque el work plan experimental se reinicie.*

Lectura operativa:

1. Los **protocolos** 01–13 se quedan.
2. La **cola** cambia: instrumentación de información y grafos (I1–I4) **antes** de STEP 08 y **antes** de cualquier barrido L1/L2 nuevo.
3. T0/T1 siguen; T2 espera el cubo extendido.
4. La tesis modular (E0–E3) usa esas métricas como covariables de análisis, no como núcleo doctoral.
5. F1/F2/F4/F5 no se reescriben para acomodar I1–I4.
