# Entregables académicos acumulativos — paquetes P0–P6

**Fecha:** 2026-09-10
**Base revisada:** `predictor@aef4dc22e7a6039eb349b75caf5e9b5823607d5e`
**Alcance:** inventario, contratos y reja de elegibilidad. **No contiene una
conclusión científica sobre ningún selector ni ninguna transformación.**

Este documento cumple el §7 de la orden: cada paquete deja pregunta y decisión
CRISP-DM, unidad estadística y particiones, tabla de datasets/variables/
operadores y costos, diagrama del flujo causal, comparadores y regla de
abstención, ledger de intentos incluidos negativos, tabla reproducible con
script y digests, y un texto de alcance reutilizable sin jerga interna.

---

## 0. Definición operacional obligatoria: *fidelidad*

> **Fidelidad** es la cantidad parcial de entrenamiento o de evaluación que se
> usa para estimar, a menor costo, el resultado que se obtendría con el
> presupuesto completo.

Una fidelidad se declara con tres números: la fracción del presupuesto de
entrenamiento consumida, la fracción del conjunto de evaluación consultada, y
el costo medido en segundos de reloj y memoria máxima. Un resultado obtenido a
fidelidad parcial **nunca** se reporta como si fuera el resultado a presupuesto
completo; se reporta como una estimación con su fidelidad al lado.

En este ciclo no se ejecutó ninguna curva de fidelidad: el preflight de P5 es
mecánico y no puntúa. La definición queda fijada aquí para que las curvas
parciales de I5–I9 la usen sin renegociarla.

---

## 1. Pregunta y decisión CRISP-DM por paquete

| Paquete | Pregunta que motiva la tarea | Decisión que pretende mejorar |
|---|---|---|
| P0 | ¿La base revisada se reproduce desde los bytes, o solo desde su informe? | Si se puede construir sobre ella sin re-auditarla |
| P1 | ¿Cuántas *variables* existen realmente, y qué se sabe de cada una? | Qué universo puede siquiera competir |
| P2 | ¿Puede un índice común servir a tres bancos sin volverlos intercambiables? | Qué evidencia puede sostener qué afirmación |
| P3 | ¿Qué autoriza el uso de una variable o un operador? | Qué entra a un modelo, un grupo o un gen |
| P4 | ¿Dónde se ajusta hoy algo fuera de entrenamiento? | Qué resultados históricos pueden citarse |
| P5 | ¿Cómo se compararía un selector sin decidirlo después de ver los números? | Qué se ejecutará cuando haya autorización |
| P6 | ¿Dónde vive la memoria del programa? | Qué se puede consultar sin abrir cada raíz de estado |

---

## 2. Unidad estadística y particiones

- **Unidad exterior congelada (P5):** `tarea × origen × serie`.
- **Prohibido como unidad exterior:** semilla, réplica, época, checkpoint. Una
  semilla es una repetición anidada, nunca una observación independiente. La
  prohibición es ejecutable: `build_design` rechaza un diseño que declare
  cualquiera de esas cuatro como unidad exterior.
- **Split exterior:** declarado antes de seleccionar y jamás re-cortado.
- **Ajustado solo dentro de entrenamiento:** imputación, escalas,
  transformaciones y el propio selector.
- **Validación:** consultable **una vez** por comparador y por origen. La
  consulta repetida es optimización manual y está prohibida.

En P1 la unidad es distinta y se declara aparte para no confundirlas: la
**aparición física** (un corte materializado) y la **variable conceptual** (una
identidad lógica con sus apariciones). Contar apariciones como variables fue el
error concreto que este censo elimina.

---

## 3. Tabla de datasets, variables, operadores y costos

| Banco | Autoridad | Datasets | Variables / generadores | Costo medido |
|---|---|---|---:|---|
| Público (T2) | `PUBLIC_FORECASTING_EVIDENCE` | 10 | 4 650 series admisibles | 32 760,29 s de reloj cobrados por el ledger encadenado |
| Sintético (T1+M4) | `SYNTHETIC_KNOWN_MECHANISM_CALIBRATION_ONLY` | — | 202 generadores (192 T1 + 10 familias M4) | actualizaciones de optimización (M4), no segundos |
| Financiero | `FINANCIAL_DOMAIN_DEVELOPMENT_ONLY` | 1 680 apariciones | 1 965 variables conceptuales / 420 entidades | censo completo en 3,5 s sin releer el lago |

**Costo del censo (P1), medido:** 3,5 s de reloj, 0 bytes de los 14 436 534 039
declarados releídos, 0 apariciones digeridas físicamente en el primer censo, 12
perfiles de valores muestreados (0,71 % de los cortes).

**Operadores:** el único operador con evidencia pública en este ciclo es el
denoiser `D` evaluado por T2. Su veredicto es negativo (§6).

---

## 4. Diagrama del flujo causal de datos

```text
  fuentes declaradas               (provenance.json, data_dictionary.md,
  en financial-data                 features/MANIFEST.json, contrato de
        │                           disponibilidad)
        ▼
  ┌─────────────────┐   aparición física: corte, periodo, esquema, bytes
  │  CENSO  P1      │──────────────┐
  └─────────────────┘              │  variable conceptual: id estable,
        │                          │  semántica, unidad, disponibilidad,
        │                          │  linaje, lista de apariciones
        ▼                          ▼
  ┌───────────────────────────────────────────┐
  │  ÍNDICE COMÚN DE BANCOS  P2               │
  │  público │ sintético │ financiero         │  ← la autoridad viaja
  │  (la clase de autoridad NO se mezcla)     │    con cada fila
  └───────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────┐
  │  REJA  PUBLICLY_ELIGIBLE  P3              │  ← manifest revisado
  │  un consumidor, sin camino de permiso     │    externamente
  │  por omisión                              │
  └───────────────────────────────────────────┘
        │                │                │              │
        ▼                ▼                ▼              ▼
  preprocessor      predictor        agent-multi     doin-domains
  (antes de         (antes de        (al FORMAR el   (antes de un
   materializar)     ventanas)        universo)       gen L2)
        │                │                │              │
        └────────────────┴────────┬───────┴──────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │  SOBRE DE CAMPAÑA  P6     │
                    │  clases separadas:        │
                    │  NON_GOVERNING · MECHANICAL
                    │  DEVELOPMENT · CALIBRATION
                    │  CONFIRMATION             │
                    └───────────────────────────┘
                                  │
                                  ▼
                        cubo OLAP (aditivo)

  doin-plugins verifica ids y digests en cualquier punto,
  y no selecciona ni promueve.
```

La dirección importa: la reja está **antes** de todo consumo, y el cubo
**después** de toda adjudicación. Nada vuelve hacia atrás: un resultado no
puede conceder elegibilidad a la variable que lo produjo.

---

## 5. Comparadores y regla de abstención

Comparadores congelados en el diseño sellado `aa846d6d`:

1. todas las variables mecánicamente admisibles (el techo a superar);
2. filtro de estabilidad y redundancia;
3. información mutua estimada solo en entrenamiento;
4. modelo lineal regularizado;
5. control aleatorio del mismo tamaño;
6. *(excluido)* el selector actual de `agent-multi`.

**Regla de abstención:** cuando el intervalo del contraste primario contiene a
la vez el margen de no-inferioridad (0,02) y el cero, el resultado es
`INCONCLUSIVE`. No se redondea hacia el resultado preferido. Un comparador que
falla no-inferioridad, o falla preservación de extremos, o cae bajo el piso de
estabilidad (Jaccard 0,5), queda **retirado** y no vuelve a entrar en una
corrida posterior.

**Multiplicidad:** Holm sobre la familia completa por tarea, α = 0,05, con la
familia congelada antes de puntuar.

---

## 6. Ledger de intentos — incluidos los negativos y los inconclusos

| Intento | Resultado | Estado |
|---|---|---|
| T2, operador de denoising `D` sobre seis paneles públicos | **DOES_NOT_ADVANCE** — `electricity_weekly` −0,0365 fuera del margen; estimando primario −0,001048; 3/6 signos positivos | **Negativo, publicado** |
| M4, escalera de predicción M2 vs M1 | ganancia pareada −0,41983 → M2 **no avanza** | **Negativo, publicado** |
| B4 v7, campaña RL pareada | `QUARANTINED_RUNTIME_STALL`, cero unidades adjudicables | **Sin resultado, publicado como tal** |
| Selector actual de `agent-multi` como comparador de P5 | **excluido**: los selectores históricos disponibles no pasan la auditoría de frontera P4 | **Exclusión razonada** |
| Instancias del contrato de disponibilidad en `financial-data` | 0 encontradas sobre un esquema que declara 8 campos obligatorios | **Hueco, no rellenado** |
| Diccionarios de datos del lago | 622 semánticos, 330 *stubs* autogenerados que declaran no ser garantía semántica, 2 no parseables | **Cobertura tipada, no inflada** |
| Punto de llamada del gen L2 en `doin-domains` | ausente en la rama actual | **Consumidor instalado sin sitio de llamada, declarado** |

Ningún intento fue retirado del ledger por ser desfavorable.

---

## 7. Tabla reproducible: script y digests

| Artefacto | Script que lo produce | Digest |
|---|---|---|
| Censo del lago (resumen) | `financial-data/_scripts/build_incremental_census.py` | `65d0efe6ff60351f` |
| Recibo del censo | idem | `2dfc9cedf69194b8` |
| Índice común de bancos | `predictor/tools/build_bank_index.py` | `0ad564dead5ef35e` |
| Diseño de selección sellado | `predictor/olap/selection_design.py` | `aa846d6db9f30572` |
| Preflight mecánico (sin puntaje) | idem | vinculado a `aa846d6db9f30572` |
| Sobres de campaña | `predictor/tools/build_campaign_envelopes.py` | T2 `397354dc`, M4 `628ac2f9`, B4 `b879fa00` |
| Recibo de backfill | `predictor/tools/backfill_campaign_envelopes.py` | `86700ccf8ce74091` |

Cada script acepta un instante fijo (`--censused-at`, `--indexed-at`,
`--as-of`) para que dos ejecuciones del mismo estado produzcan el mismo digest.

---

## 8. Texto de alcance reutilizable (sin jerga interna)

> Antes de comparar métodos de selección de variables construimos un censo del
> conjunto de datos que separa dos cosas que suelen confundirse: cuántas veces
> aparece físicamente una columna y cuántos conceptos distintos existen. En
> nuestro caso 1 680 archivos declaran 7 860 columnas, que corresponden a 1 965
> variables conceptuales sobre 420 entidades. El censo registra explícitamente
> lo que la fuente no declara —unidad, licencia, momento de disponibilidad— en
> lugar de inferirlo, y publica esos huecos como resultado: en este conjunto,
> ninguna de las 1 965 variables tiene un contrato de disponibilidad temporal
> verificable, aunque el esquema del contrato exista.
>
> Sobre ese censo definimos una condición de admisión revisada externamente que
> toda variable u operador debe satisfacer antes de entrar a un modelo, y la
> aplicamos en un único punto de decisión compartido por los cinco componentes
> del sistema. La condición no tiene camino de permiso por omisión: una entrada
> no listada, una revisión vencida o una evidencia sustituida bajo una etiqueta
> positiva son rechazos, no advertencias.
>
> También auditamos las fronteras de ajuste existentes con pruebas ejecutables
> en lugar de lectura de código. Encontramos y corregimos un caso en que la
> normalización se ajustaba por separado sobre entrenamiento, validación y
> prueba, y etiquetamos como no autoritativos dos selectores que calculan su
> estadística sobre el conjunto completo porque no reciben ninguna frontera de
> partición. La prueba que fija la corrección tiene la forma adecuada para este
> tipo de defecto: se modifica el futuro y se demuestra que el pasado queda
> idéntico bit a bit.
>
> El diseño de comparación se selló antes de calcular cualquier puntaje e
> incluye un control aleatorio del mismo tamaño, un criterio de no-inferioridad,
> un requisito de preservación de valores extremos y una regla explícita de
> abstención. En esta etapa solo se ejecutó una verificación mecánica que
> comprueba que el diseño puede construirse; esa verificación es incapaz de
> producir un puntaje por construcción.

---

## 9. Lo que este documento **no** afirma

- No afirma que ningún selector funcione: no se puntuó ninguno.
- No afirma que el lago esté documentado: 1 675 de 1 965 variables no tienen
  semántica declarada y ninguna tiene unidad ni disponibilidad.
- No afirma que el operador de denoising sirva: el único veredicto público
  disponible es negativo y está limitado a seis paneles.
- No afirma que el cubo contenga todos los experimentos: contiene los tres
  productores del backfill inicial y lo dice en su recibo.
