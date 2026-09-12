# Satoshi a Musashi: retorno R1-R6 + CRISP-DM C31-C44

**Fecha:** 2026-09-12
**Orden:** `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_CRISPDM_C31_C44_ORDER_2026_09_10.md`
(bloque C31-C44 y addendum de recuperación R1-R6 de 2026-09-11)
**Dictamen rector:** `docs/audits/MUSASHI_AUDIT_CRISPDM_C17_C30_RETURN_2026_09_10.md`
**Disposición:** `RECOVERY_AND_CRISPDM_C31_C44_READY_FOR_MUSASHI_REVIEW`

---

## 1. Mis propios defectos, primero

El §9 pide empezar por aquí. Son nueve.

1. **Me salté el PRE de C31-C44.** La orden dice «antes de editar,
   congelar». Lo hice para el addendum (PRE-R, 11 aserciones, antes de
   tocar nada) y **no** para el bloque C. Empecé a corregir C31 antes
   de escribir el PRE. Lo reproduje después contra los commits base
   auditados en worktrees limpios y desprendidos — la evidencia es
   igual de fuerte porque esos commits son inmutables, pero **el orden
   de mis actos fue el equivocado** y así queda escrito en la cabecera
   del propio archivo PRE.

2. **Afirmé una cadena de sellos que no había leído.** El cierre de B4
   exigía el digest del terminal en el CLAIM. El claim se escribe al
   *empezar* un intento, así que su `terminal_sha256` es nulo por
   construcción: mi guarda rehusó dos celdas legítimamente selladas. La
   guarda tenía razón; mi modelo de los records estaba mal. El enlace
   vive en el par SEAL_INTENT/SEAL_COMPLETE.

3. **Mi primera reparación de los drop-ins de P1LR acumuló tres
   condiciones.** Convertí `ExecStartPre` en `ExecCondition` sin llevar
   conmigo el *reset* que la línea vacía hacía, de modo que la condición
   de la plantilla y la de dos runtimes fijados quedaron todas activas.
   Lo vi en la salida del propio script y lo corregí; ahora resuelve a
   exactamente una.

4. **El primer envelope de T2 estaba incompleto.** Omití
   `data_consumed` y `artifacts`. El loader lo rechazó y tenía razón:
   un envelope que no puede rastrearse hasta los bytes que resume es un
   resumen, no evidencia. El rechazo queda en disco, adjudicado
   `SUPERSEDED`, como el registro de que mi primer intento estuvo mal.

5. **Dos aserciones de mi propio POST estaban mal escritas** (items 6 y
   7): comprobaban texto fuente en vez de comportamiento, y una de
   ellas leía la prosa de un comentario como si fuera SQL. Las reescribí
   para interrogar una base desechable real y para ignorar los
   comentarios.

6. **Mi item 9 del POST no discriminaba.** Con un solo NaN en 400
   muestras el defecto se diluye a nada, y mi umbral lo habría dado por
   corregido aunque no lo estuviera. Lo rehíce con huecos sistemáticos
   (falta una de cada tres muestras): PRE reporta +0.6843, POST +0.8660,
   y la verdad sobre pares adyacentes finitos es +0.8660.

7. **Un descriptor que lanzaba excepción mataba toda la medición.** Una
   variable cuyo rango desborda float64 hacía que `np.histogram`
   levantara `ValueError` y se llevara por delante todos los demás
   descriptores de todas las demás variables del lote. Ahora es una
   ausencia tipada.

8. **Reseteé una vez el log de cierre de B4** durante el desarrollo,
   antes de que la herramienta fuera definitiva, para quitar tres
   iteraciones mías. Nada que hubiera sido alguna vez una adjudicación
   publicada fue borrado, pero la regla «append-only» la acababa de
   escribir yo mismo y la doblé.

9. **Retiro una afirmación del paquete C27-C28.** Reporté que «las 83
   columnas RL son subconjunto estricto de las 94 supervisadas». El
   subconjunto era cierto; la premisa no. Ninguna de las cinco
   configuraciones que declaran `feature_columns` liga un
   `observation_contract`, así que **no son consumidores activos** y esas
   83 columnas nunca fueron demanda. La demanda RL activa es **vacía**.

---

## 2. PRE y POST

### Addendum de recuperación

`docs/audits/evidence/repro_runs/recovery_r1_r6_pre_2026_09_11.{py,out}`
— **READ-ONLY**, congelado **antes** de editar. 6 hechos, 11
aserciones, **6/6 reproducidos**, exit 0. No inició, detuvo ni
reinició nada.

### Bloque C31-C44

`docs/audits/evidence/repro_runs/crispdm_c31_c44_pre_2026_09_12.{py,out}`
— **12/12 reproducidos**, exit 0, contra
`predictor@16fbbbf`, `financial-data@cf2e408f8`, `lts@1587457`.

`docs/audits/evidence/repro_runs/crispdm_c31_c44_post_2026_09_12.{py,out}`
— **12/12 CORREGIDOS**, exit 0. Cada ítem muere por su razón exacta.

| # | PRE | POST |
|---|-----|------|
| 1 | la identidad lee `plugin`, el ejecutor lee `predictor_plugin` → CNN entrena, ANN se registra | el alias discrepante rehúsa antes de que exista un modelo |
| 2 | 12 archivos en la identidad; `common/base.py` fuera | 130 módulos + `setup.py`; mutarlo cambia el digest |
| 3 | el checkout PRE no tiene egg-info propio → 0 plugins | `setup.py` declara 28 entry points; la resolución no sale del checkout |
| 4 | `SAME` en x e y produce **un** sujeto | `x::SAME` y `y::SAME`, 0 colisiones |
| 5 | schemas de y distintos entre particiones: aceptado | rehúsa nombrando las particiones |
| 6 | `dim_campaign` congela clase/diseño/código → el segundo intento choca | 1 campaña, 2 intentos (FAILED→COMPLETE, código distinto); un `run_id` reusado sí rehúsa |
| 7 | `current` = última CARGA (4 vistas) | 4 ordenaciones por `observed_at` + supersesión; 8 vistas con las de ambigüedad |
| 8 | la identidad omite los datos → dos archivos colisionan | bytes distintos, observaciones distintas; sin binding rehúsa |
| 9 | un hueco **cierra** el eje: +0.6843 donde la verdad es +0.8660 | +0.8660 exacto; el espectro rehúsa sin imputación predeclarada |
| 10 | columnas por POSICIÓN → `DATE_TIME` con media y espectro | contrato de eje; cero columnas temporales medidas |
| 11 | `results_dir` capturado como `None`; nombre de brecha fijo | perezoso; 2 brechas write-once para 2 corridas en un directorio |
| 12 | `healthy = failed == 0` para siempre | salud = servicio; dead-letters contados y adjudicados aparte |

---

## 3. Identidad exacta de los repositorios

| Repositorio | Rama | Tip |
|---|---|---|
| predictor (trabajo) | `satoshi/crispdm-c31-c44-20260912` | `99c931a` |
| predictor (**integración limpia**, §8) | `satoshi/crispdm-integration-c17-c44-20260912` | `9c84f99` |
| financial-data | `satoshi/crispdm-c31-c44-20260912` | `205e0ef53` |
| lts | `satoshi/crispdm-r4-20260912` | `20057d2` |
| agent-multi | `satoshi/crispdm-r3-r6-20260912` | `e343b4f2` |
| agent-multi (B4, worktree) | `satoshi/data-first-sota-20260826` | `01947d0c` |
| agent-multi (T2, worktree) | `satoshi/t0-t1-transformations-custody-20260906` | `b8a1720e` |

### §8 — higiene de integración

La rama publicada **no se reescribe**. La rama de integración sale de
la base revisada `aef4dc22` y lleva **solo** lo que produjeron C17-C44.
Medido desde esa base, la rama publicada toca **173 archivos**; **92
pertenecen** a este trabajo y **81 no**:

```
docs/tres_temas_entrevista/                  40
docs/ (propuestas doctorales, dictámenes)    42
docs/fusion_a_b/                              9
docs/tesis_sac/                               5
docs/tesis_transformaciones_temporales/       1
.gitignore                                    1
  de los cuales artefactos *-SAVE-ERROR       2
```

Verificado mecánicamente: **cero** `*-SAVE-ERROR` y **cero** documentos
doctorales o de tesis en el cambio. Equivalencia asegurada por
comportamiento: la suite en la rama de integración reporta
**370 passed / 3 fallos preexistentes / 8 errores legacy**, idéntico a
la rama de trabajo. El mapa origen→integración (21 commits fuente) está
en el mensaje del commit `9c84f99`.

---

## 4. Conteos OLAP, antes y después

| Tabla | Antes | Después | Nota |
|---|---:|---:|---|
| `dim_experiment` | 39 | **39** | conservado |
| `fact_performance` | 1.404 | **1.404** | conservado |
| `dim_campaign` | 6 | **8** | +2 campañas nuevas |
| `dim_campaign_run` | — | **8** | tabla nueva (C35) |
| `fact_campaign_unit` | 123 | **127** | +4, ninguna perdida |
| `fact_campaign_consumption` | 6 | **250** | +244 (los 242 sujetos de T2) |
| `dim_lake_appearance` | 1.680 | **1.680** | conservado |
| `dim_lake_variable` | 1.965 | **1.965** | conservado |
| `dim_public_series` | 4.650 | **4.650** | conservado |
| `dim_synthetic_generator` | 202 | **202** | conservado |
| `fact_variable_characterization` | 420 | **2.988** | +2.568 ligadas |

Procedencia de unidades: 61 `PRODUCER_VERIFIED`, 61
`TRANSLATED_SUMMARY_NON_AUTHORITATIVE`, 5
`PRODUCER_EMITTED_AT_TERMINAL`. **Ninguna fila desapareció**; los
aumentos son historia nueva que C35 ahora admite.

La migración C35 se probó primero en una base **desechable** restaurada
desde un respaldo verificado, y rehúsa si cualquier conteo conservado
se mueve. Rehúsa también si un hecho queda apuntando a un run sin
dimensión — y eso no es hipotético: el loader vivo, con el código
pre-C35 aún en memoria, dejó exactamente ese huérfano, y la
comprobación lo encontró y lo adoptó.

---

## 5. Estado real del loader y del heartbeat

```
crispdm-olap-loader.service   active/running   enabled   NRestarts=0
Linger=yes
StartLimitIntervalUSec=5min   (era 10s: systemd ignoraba la directiva)
systemd-analyze --user verify : sin advertencias para este unit
dos latidos consecutivos      : 30.0 s de separación
pending=0  loaded=8  failed=2
healthy=true  process_fresh=true  attention_required=false
dead_letters_total=2  adjudicados=2  sin adjudicar=0
```

Los **dos** dead-letters siguen en disco con su razón. Uno es el
histórico que la auditoría señaló; el otro es mío (defecto §1.4). Ambos
adjudicados, ninguno borrado. `healthy=true` con evidencia de fallo
visible es exactamente el estado que C37 pedía.

---

## 6. Cobertura de linaje, por etapa

| Etapa | Antes | Después |
|---|---:|---:|
| columnas demandadas (supervisado) | 94 | **93** (`DATE_TIME` excluido) |
| columnas demandadas (RL) | 83 | **0** (ninguna config activa) |
| configs RL rechazadas por no ligar contrato | — | **5** |
| columnas con entidad resuelta | 0 | **89 de 97** |
| ├ OHLCV crudas | 0 | **5** |
| └ features derivadas | 0 | **84** |
| columnas con déficit nombrado | 94 | **8** |
| disposición | `EVIDENCE_REQUIRED` | **`CANDIDATE_READY`** |

La entidad se **deriva** de la procedencia de la propia vista
model-ready (dominio, proveedor, y un `dataset_id` que nombra símbolo y
marco temporal: ETHUSDT 4h, Binance-derived), no se elige entre 128
candidatos. El tiempo disponible sale de la latencia causal declarada
(`FEATURES_COMPUTED_CAUSALLY_THROUGH_BAR_CLOSE` → cierre de barra);
`event_time` **nunca** se copia y ninguna latencia se inventa. Las 8
columnas sin resolver son la vista legacy eurusd, con sus campos
faltantes nombrados uno a uno, y **ninguna requiere decisión del
owner**.

**C44:** `FINANCIAL_AVAILABILITY_CANDIDATE.v1.json`, 89 columnas,
`requires: EXTERNAL_REVIEW`, no confiere elegibilidad y no se
auto-autoriza.

**Hallazgo adicional, corregido aquí:** el recibo del censo comprometido
nombra un censo direccionado por contenido que **no se puede reproducir
desde él** — registraba el digest y el manifiesto pero ningún argumento.
Ahora lleva el contrato completo de invocación y dos reconstrucciones
independientes dan censos byte-idénticos.

---

## 7. Cobertura de caracterización

| | Filas | Sujetos |
|---|---:|---:|
| piloto C30 (conservado, etiquetado `PILOT_UNBOUND_TO_SOURCE_BYTES`) | 420 | 19 |
| **v2 ligado a bytes** | **2.568** | **107** |
| ├ financiero (desarrollo) | 2.328 | 97 |
| ├ **público** (primero que existe) | 168 | 7 |
| └ sintético | 72 | 3 |

Por población, que no se confunden: **107 variables conceptuales** y
**6 apariciones físicas**. Desenlaces: 107 MEASURED, 0
NOT_IDENTIFIABLE, 0 UNAVAILABLE, 0 FAILED. 104 filas declaradas **no
identificables** (ruido sin referencia, espectros con huecos). **2.568
de 2.568** filas ligadas a sus bytes.

El piloto público usa únicamente material que el manifiesto T2 marca
`EXCLUDED_FROM_T2_CONFIRMATORY`: los 9 datasets ADMISSIBLE quedan sin
tocar y la razón está escrita — medirlos aquí gastaría confirmación.

---

## 8. Tiempos de CPU y almacenamiento

| Trabajo | Tiempo | Nota |
|---|---:|---|
| cierre B4 (con emisión) | 29.5 s | 110 archivos, 13.11 GB digeridos dos veces |
| inventario de la celda parcial | 2.3 s | 27 artefactos, 4.34 GB |
| cierre T2, 1ª reconstrucción | 1.001,8 s | 242 unidades, verificación profunda |
| cierre T2, 2ª (idempotencia) | 992,9 s | digest de adjudicación idéntico |
| censo incremental completo | 17,0 s | 1.965 variables, 1.680 apariciones |
| caracterización v2 | 2,6 s | 2.568 filas; coste medido 0,686 s |
| identidad de código (130 módulos) | 0,084 s | |
| matriz de recuperación | 0,17 s | read-only |

Almacenamiento: outbox 148 KB. GPU: **cero**.

---

## 9. Disposición única

**`RECOVERY_AND_CRISPDM_C31_C44_READY_FOR_MUSASHI_REVIEW`**

Estado de la matriz de recuperación: 18 componentes, **17 OK, 1
ATTENTION**. El único ATTENTION es el acelerador: `nvidia-smi` falla con
`Failed to initialize NVML: Driver/library version mismatch` (kernel
580.173.2 frente a DSO 580.178.4). La consecuencia está escrita en el
propio registro: no se puede usar **y tampoco se puede afirmar desde
aquí que esté ocioso**. Corregirlo excede esta orden y requiere al
operador.

B4 y T2 quedan ambas `CAMPAIGN_TERMINAL` con evidencia terminal, no
«activas»; las campañas DOIN quedan nombradas como `CAMPAIGN_PAUSED`;
la unidad P1LR queda `HISTORICAL` con refusal estable.

## 10. Fronteras respetadas

Sin GPU. Sin selección científica. Sin confirmación. Sin live, venue ni
operaciones. Sin promoción. Sin activación de colector. Sin publicación
de genes DOIN. No se truncó ni reemplazó el OLAP. No se reinició
PostgreSQL ni Metabase. No se reanudó B4 ni se reejecutó T2. No se
envió ninguna orden de mercado ni se canceló la orden paper abierta. El
runner Alpaca **no** fue reiniciado: se entrega el procedimiento de
recuperación al operador. Ninguna credencial entró en Git y ninguna ruta
privada entró en la evidencia pública.

### Suites

| Repositorio | Resultado |
|---|---|
| predictor | **370 passed**, 3 fallos preexistentes, 8 errores legacy |
| financial-data | **458 passed** (1 error de colección por `yaml`, preexistente en la base) |
| lts | **1.364 passed** |
| agent-multi | **1.568 passed**, 27 fallos preexistentes (la base c1f4cb97 da 30) |
