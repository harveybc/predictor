# Satoshi a Musashi: retorno B4 R11-R14, T2 R7-R10 y CRISP-DM C45-C52

**Fecha:** 2026-09-12
**Orden:** `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_CRISPDM_C45_C52_AND_T2_R7_R10_ORDER_2026_09_12.md`
**Dictamen rector:** `docs/audits/MUSASHI_AUDIT_RECOVERY_AND_CRISPDM_C31_C44_2026_09_12.md`
**Disposición:** `B4_AND_T2_READJUDICATION_SUBMITTED_AND_C45_C52_COMPLETE_READY_FOR_MUSASHI_REVIEW`

---

## 1. Mis propios defectos, primero

Siete.

1. **El contador de filas leía la clave equivocada.** Reporté `0
   cargadas` cuando 38.364 filas habían entrado en el cubo.
   `load_rows` devuelve `{"fact_variable_characterization": n}` y yo
   leía `"loaded"`.

2. **Emití los desenlaces de lote como *events*, que el loader no puede
   cargar.** Catorce lotes fueron directos al dead-letter. Un desenlace
   que el cubo no puede recibir no es un desenlace reportado. Los
   reemití como envelopes de campaña y adjudiqué los rechazos sin
   borrarlos.

3. **Las filas de descriptor sólo existían para las variables
   medidas**, así que el cubo mostraba 1.612 variables donde 1.965
   tenían terminal. Ahora cada variable lleva su fila de disposición.

4. **El propio registro del cierre B4 entraba en el snapshot que
   publica**, de modo que cada re-cierre era una adjudicación nueva de
   la misma evidencia: el record cambiaba aquello que describía.
   Excluido, con la exclusión nombrada.

5. **Mi parseo de `git status --porcelain` se comía el primer carácter
   de la ruta** (`ools/…`). Y la submission ensuciaba el árbol que ella
   misma reporta; ahora se excluye explícitamente.

6. **Dos aserciones de mi POST leían prosa como si fuera código** — el
   ítem 13 fallaba porque el docstring dice, correctamente, que
   `final_adjudication()` *no* se llama. Lo reescribí sobre el AST:
   llamadas, no texto.

7. **Un fichero fijado ausente reventaba con traceback** en vez de
   rehusar tipado.

Un hallazgo adicional, que mi propia prueba encontró y no la auditoría:
`shift(-1)` se parsea como `UnaryOp`, no como `Constant`, así que **todo
desplazamiento hacia adelante parecía ningún desplazamiento** y una
feature con fuga habría quedado clasificada `CAUSAL`. Corregido; sobre
los datos reales no cambió ninguna clasificación, y lo digo así.

---

## 2. PRE y POST

**Esta vez el PRE fue primero.** El retorno anterior tuvo que confesar
que el PRE del bloque C se escribió después; este archivo se escribió y
se corrió antes de tocar nada, y los árboles de trabajo estaban
intactos cuando produjo su salida.

* `crispdm_c45_c52_pre_2026_09_12.{py,out}` — **16/16 reproducidos**,
  exit 0. Los ítems de B4 y T2 trabajaron sobre **copias** de sus
  raíces; el digest estructural de ambas raíces reales se comprobó
  antes y después: **sin cambios**.
* `crispdm_c45_c52_post_2026_09_12.{py,out}` — **16/16 CORREGIDOS**,
  exit 0, raíces reales **intactas**.

| # | PRE | POST |
|---|-----|------|
| 1 | el terminal se parsea de una apertura y se hashea de otra: se adjudica 123.4 mientras el disco dice 999999 | se adjudica lo que se hasheó; el digest es de los bytes consumidos |
| 2 | el ledger por barra se hashea de una apertura y se cuenta de otra | un reemplazo que conserva el número de filas ya no llega a la adjudicación |
| 3 | el intent se parsea y se reabre para hashearlo | todo sale de `custody.read`; no hay segunda apertura |
| 4 | `status.json` se relee tras el inventario: se adjudica la época 1999 | se adjudica la 111, desde la misma lectura del inventario |
| 5 | una celda que aparece después conserva `NOT_STARTED` | la ausencia se decide desde un snapshot y publica su digest |
| 6 | demanda leída de un registro de investigación | 137 configs ejecutables / 87 inalcanzables; el registro es otro universo |
| 7 | el conjunto RL vacío produce `subset=true` | `NOT_APPLICABLE` |
| 8 | 93 nombres junto a 97 sujetos sin distinguir grano | seis granos materializados por separado |
| 9 | toda columna entra como input | 2 targets con `contract_role=target` |
| 10 | 84 features con una frase y una fórmula | 11 lookbacks y 5 ficheros productores distintos |
| 11 | el linaje no mira los repos productores | 250 ficheros parseados en 3 repos, cada uno ligado por commit |
| 12 | ledger verde con 107 de 1.965 | **1.965/1.965** |
| 13 | el record verificado se reabre para puntuarlo | el screen consume `snap.record(...)`, la misma instancia |
| 14 | identidad mixta llamada «revisada» | 10 ficheros, cada uno con su origen; no se llama revisada |
| 15 | sin batería para el cierre T2 | 17 pruebas dedicadas |
| 16 | `p = 1.3125` | tabla simétrica y acotada 0..6; el valor publicado queda superado |

---

## 3. Identidades

| Repositorio | Rama | Tip |
|---|---|---|
| predictor | `satoshi/crispdm-c45-c52-20260912` | `8a3545e` |
| financial-data | `satoshi/crispdm-c45-c52-20260912` | `19fe375a1` |
| agent-multi (B4) | `satoshi/data-first-sota-20260826` | `c907495d` |
| agent-multi (T2) | `satoshi/t0-t1-transformations-custody-20260906` | `5fb2849e` |
| agent-multi | `satoshi/crispdm-r3-r6-20260912` | `e343b4f2` (sin cambios) |
| lts | `satoshi/crispdm-r4-20260912` | `20057d2` (sin cambios) |

La submission de B4 registra el commit `ac788210`, que era el HEAD
cuando se generó; al enmendar el commit el hash cambió a `c907495d`. La
autoridad es el `surface_sha256` de los ficheros, no el hash del
commit, y lo digo porque un artefacto que se describe a sí mismo no
puede fijar el commit que lo contiene.

---

## 4. B4 y T2: frontera exacta

### B4 R11-R14

* **La raíz preservada NO se abrió para escritura.** Todo el trabajo se
  hizo sobre una copia (`~/.local/state/crispdm-pre-copies/b4_v7`,
  13 GB) tomada antes de empezar. El digest estructural de la raíz real
  es idéntico antes y después del PRE y del POST.
* **2 COMPLETED_VERIFIED / 1 QUARANTINED_PARTIAL / 9 NOT_STARTED**,
  `SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT`, reproducidos con **40
  lecturas descriptor-first**.
* Costes: 17.773,3 s y 17.909,1 s desde los terminales; la celda
  detenida 124.993,6 s, declarado **LOWER_BOUND** — no bajo un cargo
  que no puedo rederivar del todo.
* Debilidad de custodia declarada: los artefactos existentes son
  `0o664` (group-writable). Rehusar eso haría ilegible la evidencia
  real; aceptarlo en silencio sería lo deshonesto. Se registra en cada
  artefacto y el modo estricto sí rehúsa.
* `B4_READJUDICATION_SUBMISSION_2026_09_12.json`, `requires:
  EXTERNAL_REVIEW`. **Cero relanzamiento, cero entrenamiento, cero GPU,
  cero promoción.**

### T2 R7-R10

* **La raíz preservada NO se abrió.** El record de cierre anterior se
  cita desde la copia tomada antes de esta orden.
* `UnitSnapshotRoot` lee cada unidad **una vez** e implementa la
  superficie que el verificador fijado usa, subclasando `ResultsRoot`
  sin llamar a su `__init__`. Ni un byte del código fijado cambia.
* **Declarado:** el cierre ya **no llama** a `final_adjudication()`,
  porque re-resuelve la raíz y reabriría la evidencia. Sus tres
  garantías se reproducen sobre el snapshot.
* Identidad: 7 ficheros del checkout revisado, **3 del tip** (el
  reconstructor, el cierre y la custodia) porque el record es anterior
  a ellos. **No se llama identidad revisada.**
* El valor p `1.3125` queda **superado** por 1.0 y el defecto se
  elimina del código; el envelope histórico no se reescribe. La
  corrección amplía en un fichero la divergencia del tip respecto al
  record fijado, y se reporta.
* `T2_READJUDICATION_SUBMISSION_2026_09_12.json`, `requires:
  EXTERNAL_REVIEW`. **Cero reentrenamiento, cero descarga.**

---

## 5. Demanda, candidatos, targets y granos

| Universo | Contenido |
|---|---|
| `ACTIVE_EXECUTION_DEMAND` | 137 configs ejecutables, 87 inalcanzables; **162** sujetos x, **162** sujetos y, **2** targets; RL activo **0** (5 borradores rechazados por no ligar contrato) |
| `RESEARCH_BANK_CANDIDATES` | **97** columnas en 2 datasets registrados **sin consumidor ejecutable** |
| `CONFIRMATORY_RESERVED` | **9** datasets públicos admisibles |

Granos, separados: **30** nombres únicos de columna, **159** sujetos
`(dataset, columna)`, 162 x, 162 y, 2 targets, 97 candidatos de banco.
El reparto 137/87 es el que documenta `AGENTS.md`, derivado aquí de
forma independiente.

`rl_is_subset_of_supervised = NOT_APPLICABLE`. Ningún conjunto vacío
produce una afirmación positiva.

---

## 6. Cobertura del DAG, por clase

| Clase | Columnas |
|---|---:|
| `CAUSAL` | **52** |
| `UNRESOLVED_PRODUCER` | **45** |
| `NON_CAUSAL` | 0 |
| `EXTERNAL_LATENCY_REQUIRED` | 0 |

250 ficheros productores parseados en `feature-eng`,
`feature-extractor` y `financial-data`, cada repositorio ligado por
commit. Las 52 resueltas llevan repositorio, commit, fichero, línea,
símbolo, digest de código, entradas directas, lookback, shift,
centrado y política de ventana — **11 lookbacks distintos** y **5
ficheros productores distintos**.

**Productores no localizados (45):** `BC-BO`, `BH-BL`, `BH-BO`,
`BO-BL`, las cinco `close_sma_ratio_*`, las seis `ema_*`, las seis
`sma_*`, y el resto de derivadas del export Project3. No se afirma
nada sobre ellas.

**Cinco columnas resuelven sólo en rutas retiradas o invalidadas**
(`ema_200`, `ema_50`, `return_20`, `return_5`, `rsi_14`): el grafo
describe código que puede no ser el que construyó la vista, y así se
declara.

---

## 7. Cobertura 1.965/1.965 y costes

```
MEASURED           1.505
NOT_IDENTIFIABLE     460   (identificadores temporales: son el eje)
UNAVAILABLE            0
FAILED                 0
                   ─────
                   1.965 / 1.965     completo
```

Poblaciones separadas: **1.965** variables conceptuales frente a
**1.680** apariciones físicas. Nunca se multiplican.

| Trabajo | Tiempo |
|---|---:|
| caracterización completa del lago | **75,8 s** |
| cierre B4 sobre la copia (con emisión) | 17,0 s |
| censo incremental completo | 17,0 s |
| DAG causal (250 ficheros) | < 2 s |
| tres universos (224 configs) | < 2 s |

Almacenamiento: copias PRE 14 GB (efímeras), estado de
caracterización 7,9 MB. **GPU: cero.**

---

## 8. Conteos OLAP, antes y después

| Tabla | Antes | Después |
|---|---:|---:|
| `dim_experiment` | 39 | **39** |
| `fact_performance` | 1.404 | **1.404** |
| `dim_campaign` | 8 | **9** |
| `dim_campaign_run` | 8 | **9** |
| `fact_campaign_unit` | 127 | **183** |
| `fact_campaign_consumption` | 250 | **264** |
| `dim_lake_variable` | 1.965 | **1.965** |
| `dim_lake_appearance` | 1.680 | **1.680** |
| `fact_variable_characterization` | 2.988 | **40.749** |

Las seis consultas de aceptación (`olap/acceptance_queries_c51.sql`,
salida en `acceptance_queries_c51_2026_09_12.out`) pasan: 1.965
disposiciones terminales, cobertura por banco y estado, poblaciones
separadas, **cero** descriptores de selección, **cero** observaciones
duplicadas y cada conteo auditado por encima de su suelo.

Loader: `healthy=true`, backlog 0, **16 dead-letters, los 16
adjudicados**, ninguno borrado.

---

## 9. Estado post-reinicio

**El reinicio del owner todavía no ha ocurrido.** `uptime` sigue en
2026-09-11 22:53:03 y `nvidia-smi` continúa fallando con
`Driver/library version mismatch`. La matriz de recuperación mantiene
la GPU como única atención tipada. Nada de esta orden lo requería: todo
fue CPU.

---

## 10. Disposición y fronteras

**`B4_AND_T2_READJUDICATION_SUBMITTED_AND_C45_C52_COMPLETE_READY_FOR_MUSASHI_REVIEW`**

Sin GPU. Sin selección. Sin confirmación. Sin live, venue ni
operaciones. Sin campaña nueva. B4, M4 y T2 **no** se reentrenaron. Las
raíces preservadas de B4 y T2 **no** se abrieron para escritura y sus
digests estructurales son idénticos antes y después. Ningún servicio
live se reinició, ninguna orden se modificó y ninguna operación se
envió. C52 existe como diseño y **no ejecuta nada**.

### Suites

| Repositorio | Resultado |
|---|---|
| predictor | **372 passed**, 3 fallos preexistentes, 8 errores legacy |
| financial-data | **473 passed** (1 error de colección por `yaml`, preexistente) |
| agent-multi B4 | **45 passed** |
| agent-multi T2 | **17 passed** |
| lts | 1.364 passed (sin cambios en esta orden) |

### Pendiente P2

`E2` (ruido repetitivo del runner Alpaca) **no** se abordó en esta
ronda. Es P2 opcional y exigía reproducir offline un estado del runner
live; preferí no tocarlo antes de que la revisión de B4 y T2 cierre.
Lo declaro como no hecho en vez de dejarlo implícito.
