# Retorno C17-C30 — reja de dos fases, OLAP terminal continuo y primera caracterización

**Fecha:** 2026-09-10
**Orden:** `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_CRISPDM_C17_C30_ORDER_2026_09_10.md`
**Dictamen:** `docs/audits/MUSASHI_AUDIT_CRISPDM_C1_C16_RETURN_2026_09_10.md`

**Declaración explícita:** no se ejecutó GPU, confirmación, live, venue,
selección científica ni publicación DOIN. No se creó ningún record externo de
Musashi. Ninguna fila OLAP fue borrada. Los componentes aceptados en el dictamen
no se reescribieron: el censo de 14,4 GB no se repitió, la población 1.680/1.965
no se alteró, MTM no se reabrió y DOIN L2 sigue declarado no implementado.

**Disposición:** `CRISPDM_C17_C30_READY_FOR_MUSASHI_REVIEW`

---

## 1. Tips PRE y POST

| Repositorio | PRE | POST | Commits |
|---|---|---|---:|
| `predictor` | `0e27f3c` | `456c6e4` | 6 |
| `financial-data` | `7e895b4c0` | `cf2e408f8` | 1 |
| `preprocessor` | `05fa1f4` | `c246505` | 1 |
| `agent-multi` | `35aeb226` | `c1f4cb97` | 1 |
| `doin-domains` | `79b8d2a` | `26ec568` | 1 |
| `doin-plugins` | `43064fb` | `10f6026` | 1 |

Rama en `predictor`: `satoshi/crispdm-c17-c30-20260910`; los otros cinco siguen
en `satoshi/crispdm-census-gate-20260910`.

---

## 2. PRE: los diez puntos, y once reproducidos

`docs/audits/evidence/repro_runs/crispdm_c17_c30_pre_2026_09_10.{py,out}`,
congelado en `predictor@05000ab` antes de tocar una línea. **Los diez puntos del
§1 reprodujeron, más uno que encontré reproduciéndolos.** Extractos exactos:

- `y_validation` de `1` a `999` → sujetos, `data_digest` y `partitions_digest`
  **idénticos**;
- reja en el offset 11235, `config.update(optimal_params)` en 11992, pipeline en
  12402, **sin allowlist ni rederivación** entre medias;
- `gate_run()` sella `submitted_at` con el reloj de esa invocación y exige
  inmediatamente un record de ese digest — **sin fase `SUBMIT_ONLY`**;
- manifest cargado con `undeclared_top_level_field` y
  `undeclared_nested_field`;
- envelope con campo anidado extra **y `wall_seconds=NaN`** aceptado y
  digerido;
- el único `emit()` después del pipeline **sin `try/except`**;
- **cero** unidades systemd del loader y **ningún** directorio outbox
  productivo;
- **cinco** inserts con `ON CONFLICT DO NOTHING` sin columna de versión;
- `CONSUMING_CODE` = **cuatro** archivos de la reja;
- T2/M4/B4 con **cero** referencias a outbox.

---

## 3. Tabla C17-C30

| # | Corrección | Evidencia |
|---|---|---|
| **C17** | Dos fases reales | `SUBMIT_ONLY` deriva, persiste una submission content-addressed y **sale**; `EXECUTE_REVIEWED` rederiva, exige igualdad con la persistida y consume el record. **Probado en procesos separados**; un byte cambiado entre fases rehúsa |
| **C18** | Conjunto consumido completo | todo `x` y `y` con encabezado, columnas y digest; target del lado `y` es sujeto con su lado y rol; digests canónicos sobre la lista ordenada completa; `y` mutado, omitido, renombrado o intercambiado cambia identidad |
| **C19** | Contrato inmutable | allowlist explícita de hiperparámetros (datos, target, plugins, autoridad rehúsan) + rederivación completa con igualdad exigida justo antes del pipeline |
| **C20** | Identidad del código real | 12 módulos estáticos + **cada plugin resuelto por el mismo registro de entry points que usa el run**, hasheado por bytes, con inventario; cambiar el pipeline plugin cambia el digest; plugin irresoluble o inhasheable rehúsa |
| **C21** | Schemas exactos recursivos | campos no declarados en manifest, entrada, disponibilidad y digests rehúsan; bool-como-número rehúsa; artefactos de productor con parser estricto y digest canónico |
| **C22** | Un terminal por desenlace | `terminal_run` envuelve el run COMPLETO: `COMPLETE`, `FAILED`, `INCONCLUSIVE`, `REFUSED`, `QUARANTINED`, con fase y tipo del fallo sin convertirlo en resultado; un fallo del outbox deja una **brecha operacional tipada** junto a los resultados |
| **C23** | Productores en el punto terminal | el run terminal **ES una unidad**: celda = dataset consumido, candidato = plugin, costo medido, épocas, estado; `units=[]` desapareció |
| **C24** | Loader activo | servicio `systemd --user` **activo**, 0 reinicios, un consumidor, heartbeat, credenciales fuera de Git, sin reiniciar PostgreSQL ni Metabase |
| **C25** | Inventario versionado | clave `(id lógico, observación)`; observación repetida idempotente, distinta ⇒ versión nueva; cuatro vistas `current` deterministas; nada se borra |
| **C26** | Envelope ligado al productor | parser estricto + self-digest rederivado + digest del archivo + identidad del verificador |
| **C27-C28** | Puente derivado | cadena columna→variable→aparición→familia→contrato sobre 94 columnas, con la etapa de ruptura nombrada; **ninguna configuración fue editada** |
| **C29** | Disposición | `FINANCIAL_AVAILABILITY_EVIDENCE_REQUIRED` |
| **C30** | Caracterización | 420 filas versionadas, tres bancos separados, sin selección |

---

## 4. Submission y templates no autorizantes

- **Submission**: `eligibility/review.py` la construye, la persiste
  content-addressed en `<state_root>/eligibility_submissions/` y la vuelve a
  consumir. Lleva `grants_nothing` y el gate rehúsa tratarla como decisión.
- **Template de record**: `review_record_template()` — cada valor es un
  `<placeholder>`, lleva una clave extra `_template_note`, y el lector estricto
  rechaza ambas cosas. **No autoricé ni instalé ningún record real y no existe
  ninguno.**
- **Diseño de selección v2**: sigue siendo una submission de candidato; su
  self-digest prueba cronología y nada más.

---

## 5. Inventario de archivos `x/y` y módulos ligados

**Datos** (por rol, ambos lados): `x_train/x_validation/x_test` y
`y_train/y_validation/y_test`, cada uno con su clave de config, ruta,
encabezado, columnas, bytes y digest. El `data_digest` es canónico sobre la
lista ordenada `(rol, lado, clave, sha)`; el `partitions_digest` añade la ruta y
el contrato temporal, porque **qué archivo juega qué rol sí es parte del
contrato de particiones** aunque una ruta nunca sea la identidad de un sujeto.

**Código ligado (12 módulos + plugins resueltos):** `app/main.py`,
`app/config.py`, `app/config_merger.py`, `app/config_handler.py`,
`app/plugin_loader.py`, `app/data_handler.py`, `app/data_processor.py`,
`eligibility/{gate,consumed,integration,review,strict}.py`, más los módulos de
los plugins `predictor`, `optimizer`, `pipeline`, `preprocessor` y `target`
resueltos por entry point y hasheados por bytes.

---

## 6. Estado y latidos del loader

```text
systemctl --user is-active crispdm-olap-loader.service  ->  active
MainPID 3352297   NRestarts 0
```

| | `published_at_epoch` | pending | loaded | failed | healthy |
|---|---:|---:|---:|---:|---|
| Latido 1 | 1789065824.925 | 0 | 0 | 0 | true |
| Latido 2 | 1789065855.074 | 0 | 1 | 0 | true |

**30,1 s de separación**, la cadencia declarada. Entre ambos, un run REAL de
`predictor` emitió su terminal y el servicio lo cargó de extremo a extremo sin
intervención humana.

**Recuperación** (`crispdm_c24_recovery_2026_09_10.out`, base **desechable**,
nunca el cubo real): base caída → 1 pendiente, 0 fallidos, nada perdido; base de
vuelta → cargado; segundo drenaje → 0 intentos, filas sin cambio; re-emisión →
0 intentos, filas sin cambio.

---

## 7. Conteos OLAP e historia versionada

| Tabla | Antes | Después |
|---|---:|---:|
| `dim_experiment` | 39 | **39** |
| `fact_performance` | 1.404 | **1.404** |
| `fact_results_summary` | 1.404 | **1.404** |
| `fact_campaign_unit` | 120 | **123** (60 traducidas + 60 verificadas + 3 nacidas en terminal) |
| `dim_lake_appearance` | 1.680 | **1.680** |
| `dim_lake_variable` | 1.965 | **1.965** |
| `dim_public_series` | 4.650 | **4.650** |
| `dim_synthetic_generator` | 202 | **202** |
| `fact_variable_characterization` | ausente | **420** |

**Historia versionada probada** en base desechable: misma observación ⇒ 1
versión; observación con digest y disponibilidad distintos ⇒ 2 versiones, la
vista `current` muestra la nueva y **ambas** permanecen.

**Tercer estado de procedencia:** un terminal **nacido** en el productor no es
un resumen traducido — llamarlo así habría puesto al productor vivo por debajo
de una relectura posterior. Las 60 traducidas siguen siendo 60.

---

## 8. Cobertura de disponibilidad por familia y variable

| | |
|---|---:|
| Columnas consumidas examinadas | **94** |
| Resueltas a familia con contrato completo | **0** |
| Sin resolver en `column_to_variable` | **89** |
| Sin resolver en `variable_to_entity` | **5** |
| Configs que declaran ambas cosas | **0** |
| Columnas RL que no están en la vista supervisada | **0** |

Las 89 son features **derivadas** (indicadores técnicos y estadísticos)
computadas aguas abajo del lago: ninguna variable conceptual del censo las
lleva, y su disponibilidad es propiedad de una **derivación que nadie ha
declarado**. Las 5 son `OPEN`, `HIGH`, `LOW`, `CLOSE` y `VOLUME`, que existen en
**128 entidades cada una** sin que la vista declare de cuál instrumento salió.

**Hallazgo que cierra el hueco de otra forma:** las 83 columnas RL son un
**subconjunto estricto** de las 94 supervisadas — ambos consumidores leen la
misma vista física. Por eso no hizo falta editar ninguna configuración.

---

## 9. Ledger y resumen de C30

| | |
|---|---:|
| Intentos | **19** (19 medidos, 0 inconclusos, 0 fallidos) |
| Filas de descriptor | **420** |
| Banco financiero (desarrollo) | 348 |
| Banco sintético (calibración) | 72 |
| Filas **no identificables**, declaradas | **16** |
| Costo total medido | **0,158 s** |

Cada descriptor lleva su contrato: una longitud comprimida se llama longitud
comprimida bajo un compresor declarado, **nunca** contenido de información ni
complejidad de Kolmogorov; una entropía es la de su cuantización declarada, no
la de la variable; la estacionariedad es un proxy **descriptivo**, no una
prueba de hipótesis. **La cifra de ruido se reporta únicamente para los tres
generadores sintéticos**, cuyo componente limpio se conoce por construcción; en
las 16 variables financieras se registra explícitamente `NOT IDENTIFIABLE` con
su razón. Ningún selector se puntuó, ninguna confirmación se tocó, ninguna GPU
se usó, y `assert_no_selection` rehúsa cualquier descriptor cuyo nombre suene a
elección.

---

## 10. Conteos, tomados del terminal

| Batería | Repositorio | Conteo |
|---|---|---:|
| C17-C21 reja | `predictor` | **28** |
| aceptación §6 C17-C30 | `predictor` | **21** |
| OLAP C12-C15 | `predictor` | **21** |
| C1-C5 (migrada a dos fases) | `predictor` | **28** |
| consumidores cruzados | `predictor` | **25** |
| censo | `financial-data` | **41** |

**Suite completa `predictor` en `456c6e4`: 272 passed / 3 failed / 8 errores de
colección.** Los 3 fallos y los 8 errores son **preexistentes**, reproducidos
en la base revisada `aef4dc22`. **`financial-data`: 41 passed.**

**POST:** `crispdm_c17_c30_post_2026_09_10.{py,out}`, exit 0, **13/13
CORRECTED**. El PRE queda congelado: sobre el código corregido sus propios
localizadores ya no describen el programa.

---

## 11. Defectos propios y limitaciones

1. **Dos defectos de evaluación perezosa míos.** El envoltorio terminal
   capturaba el `config` y la clave de campaña **antes** de que el run los
   construyera, así que el primer terminal reportó `UNAVAILABLE` en todo salvo
   el costo. Ambos se leen ahora en el instante de cierre.
2. **`campaign_key` demasiado grueso.** Todos los runs compartían
   `predictor_run`, y el guarda de identidad C13 los rechazó — correctamente.
   La campaña es ahora el **experimento** y `run_id` viajó al hecho.
3. **Una columna ausente en un INSERT** y **un `KeyError`** por mover `run_id`
   fuera del diccionario de identidad; ambos aparecieron en carga real y se
   corrigieron antes de cualquier escritura definitiva.
4. **Un import mal escrito** (`importlib.util` vía `__import__`) rompió la
   resolución de plugins en la primera corrida real.
5. **Confundí raíz de datos con raíz de código**: la identidad de código
   dependía de dónde vivían los datos. Separadas.
6. **Mi taxonomía de procedencia estaba incompleta**: clasificaba como
   "traducido" un sobre nacido en el productor, que es la procedencia más
   fuerte que existe.
7. **Cinco tests míos codificaban el contrato viejo** (una sola fase, `run_id`
   en la dimensión, emisor solo-éxito). Están migrados, con la migración dicha
   en el propio test.

**Limitaciones declaradas:** el loader es un servicio de usuario, no de
sistema; si la sesión del usuario termina, systemd lo detiene — no lo presento
como alta disponibilidad. La caracterización cubrió 16 variables financieras y
3 generadores sintéticos, un alcance deliberadamente pequeño; **no hay banco
público en ella**. El puente examinó las columnas de los consumidores
**activos**, no las 1.965 del lago.

---

## 12. Efectos no ejecutados

Ninguna selección científica, confirmación, GPU, live, venue ni publicación
DOIN. `TRADING_L2_PUBLICATION_PATH_NOT_IMPLEMENTED` se conserva y no fabriqué
un punto de llamada. No emití manifest de elegibilidad, índice sucesor ni
diseño sucesor, porque los tres habrían ligado un conjunto vacío.

**Me detengo en `CRISPDM_C17_C30_READY_FOR_MUSASHI_REVIEW`.**
