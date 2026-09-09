# 04 — Plan de integración jerárquico

Numeración estable. Un paquete no empieza si su precondición está en rojo.  
Harvey ordenó **reiniciar la cola experimental**, no reescribir protocolos.

Leyenda de estado al 9-sep-2026:

- **HECHO** — artefacto versionado
- **ESCRITO** — protocolo, cero curvas
- **NO** — no existe
- **BLOQUEADO** — existe pero no se toca aquí

---

## I0. Contrato de integración

**Salida:** este directorio + carta `docs/RETSU_TO_MUSASHI_INTEGRACION_WORKPLAN_2026_09_09.md`.

0.1 Conservar STEP 01–13 y recortes RETSU.  
0.2 Conservar T0/T1. No sellar T2.  
0.3 Conservar G12 = L2 hasta orden de Harvey.  
0.4 P-MOD es el objeto doctoral que Harvey edita; no se fusiona con P-L2/P-CAP/P-INC.  
0.5 I-INFO es instrumentación, no tesis.  
0.6 N para H-ES piloto = 30 runs CPU (modificable en I0 si Harvey baja el piso; no se sube en silencio).  
0.7 CPU por defecto. GPU solo con sello de campaña **distinto** de este paquete.

---

## I1. Cubo de laboratorio (información + grafo + epoch)

**Precondición:** I0. Postgres throwaway. No `reset_olap.py` del cubo real.

1.1 Extender schema según doc 05 §4 (`fact_run_epoch`, `fact_split_information`).  
1.2 Savepoint en el backfill `metric_value` (bug conocido del ETL).  
1.3 Manifiesto JSON de compresor/umbral por `experiment_key`.  
1.4 Gate: un run sin `fact_split_information` se rechaza.

**Salida:** DDL + ETL en rama; cubo throwaway vacío listo.  
**No:** migrar el cubo poblado.

---

## I2. Medidores de datos y target (`D.*`, `Y.*`)

**Precondición:** I1. Repo `preprocessor` + escritura desde `predictor` al armar splits.

2.1 Dump canónico de cada serie de train/val (no test para ajustar \(Q\)).  
2.2 `L_zstd`, `L_lzma`, `L_bz2`, `H0`.  
2.3 `I_free_synth` solo si el generador está en el manifiesto.  
2.4 `SNR_hat` solo con estimador causal ya elegido en T1/STEP 03; si no hay, `NA`.  
2.5 Cachear; no meter zstd en el training loop.

**Salida:** plugin/script CPU; fixture con hash.  
**Enganche:** T1 existente; no duplicar el banco de ruido.

---

## I3. Medidores de modelo y grafo (`M.*`, `G.*`)

**Precondición:** I1. Callback en `predictor` y `feature-extractor`.

3.1 Serializar pesos float32 C-order por nombre.  
3.2 `M.L_*` cada epoch en modelos chicos; cada `log_every` en grandes; siempre al best/stop/last.  
3.3 Grafo: density, weight_entropy, spec_radius, eff_rank; el resto `NA` si no aplica.  
3.4 `CUDA_VISIBLE_DEVICES=""` en la verificación.

**Salida:** callback + test de hash sobre un ANN de juguete.

---

## I4. Registro de parada y piloto H-ES

**Precondición:** I2 y I3 verdes; ≥30 runs CPU ANN/DLinear (E0 o daily chico con `--epochs` bajo y patience real).

4.1 Loguear `stop_reason` y epoch de min val.  
4.2 Congelar \(f\) y \(R\) **antes** de mirar tareas de confirmación.  
4.3 Contrastar \(R\) vs `early_patience` en tareas reservadas.  
4.4 Resultado permitido: nulo. Se publica.

**Prohibido:** cambiar el default de campañas F2 o de E2/E3.

---

## I5. Primer *hecho* de la cadena informacional = banco ruido (STEP 03 / P-PRE)

**Precondición:** T0 HECHO, T1 lab HECHO, I2 en el mismo run.

5.1 No reescribir STEP_03. Ejecutar el banco CPU que el protocolo ya pide.  
5.2 Tres entregas: crudo / denoised / señal+residuo.  
5.3 Escribir `D.*` y desempeño del extractor/núcleo **congelado** (DLinear/ANN).  
5.4 H1–H3 de P-PRE.

**Salida:** curvas SNR vs métrica + filas de cubo. Esto **reinicia** la cola: es el primer experimento, no STEP 08.

---

## I6. Diagnóstico de fuente (STEP 04–05)

**Precondición:** I5 no tiene que “ganar”; tiene que **existir**. 5A sintético antes de finanzas.

6.1 Alfabeto provisional; `H0`; 5B contexto vs orden cero.  
6.2 Si 5B no baja rate: se documenta “casi sin memoria” y no se abre rama CTW.  
6.3 MDL como diagnóstico, no como \(K\).  
6.4 No Huffman-as-input.

---

## I7. Cerrar T2 (transformaciones / preprocessor) **con** I2

**Precondición:** C31–C36 de T2 resueltos; dos orígenes; I2.

7.1 T2 mide utilidad pública **y** deja `D.*`/`Y.*`.  
7.2 Operadores: `LAB_CALIBRATED` / `PUBLICLY_ELIGIBLE` / `REJECTED`.  
7.3 No T3 sin T2 sellado.

---

## I8. P-MOD E0 (sintético de representaciones modulares)

**Precondición:** perfiles implementables; TCN causal; I2–I3 en el run.

8.1 Generadores de escalas/desfases conocidos (contrato `tesis_representaciones_modulares/01_`).  
8.2 H2/H3 en E0; **no** H1 de E2.  
8.3 Registrar `M.*`/`G.*` para no tener que reentrenar después “porque faltaban métricas”.  
8.4 No E3 live. E3 es histórico y no sostiene H1–H3.

**Paralelo permitido:** I8 puede avanzar en CPU **a la vez** que I5 si no pelean GPU. I5 no espera a E0.

---

## I9. P-CAP laboratorio de capacidad

**Precondición:** I3 (hace falta `M.L_C` y pérdida log2).

9.1 \(\widehat C_{mem}\) en MLP chicos.  
9.2 No identificar `M.L_C` con bits memorizados.  
9.3 No entra al PDF P-MOD como resultado.

---

## I10. T3–T5 y adaptador DOIN (P-TRN)

**Precondición:** T2 sellado; operadores `PUBLICLY_ELIGIBLE`; paridad de adaptador.

10.1 Meta-selector propone población; **no** campeón; **no** holdout retenido.  
10.2 T4 CPU. T5 GPU solo con sello.  
10.3 No tocar B4.

---

## I11. P-L2 selector (después del espacio)

**Precondición:** hay al menos un espacio de artefactos (I8/I10) que no sea “cinco nombres en un JSON”.

11.1 Fidelidad = mismo candidato, distinto presupuesto.  
11.2 Banco RL público. Finanzas no son la evidencia principal.  
11.3 Abstención es resultado válido.

---

## I12. STEP 08–13 (protocolos ya escritos; experimentos después de medidores)

**Precondición:** I5–I6 existen (aunque H2 de P-PRE sea nulo). Router 12 necesita ≥2 modos validados.

12.08 Equalización con CSI; ≠ z-score del normalizer.  
12.09 Común/único; no causalidad.  
12.10 \(\tau\) causal.  
12.11 Máscara train-only en AE **existente**.  
12.12 Router elige modos; no MoE.  
12.13 Presupuesto de ramas; no NAS.

Carriles C1–C7, eventos, L3: según PATCH 001–003. C1–C2 después de 7A. L3 no bypasea L2.

---

## I13. P-INC y frente F5

**Precondición:** ninguna de I1–I12. Carril **separado**.

13.1 WP1 de `doin-domains` inmutable.  
13.2 No GPU de trading. Hanzo audita.  
13.3 No se mete en el PDF P-MOD.

---

## I14. Encaje con los cinco frentes (negocio)

| Frente | Este paquete | Qué no hace |
|---|---|---|
| F1 live/paper/demo | nada | no calibra I-INFO; no capital |
| F2 optimización / B4 | I1–I4 **no** se montan sobre B4 | no relanzar B4 para llenar el cubo |
| F3 académico | I0, I8, inventario doc 02 | no fusionar tesis; no cambiar G12 |
| F4 social | nada | — |
| F5 dominios | I13 | no mezclar con preprocessor financiero |

Si hay conflicto de hueco CPU/GPU: **Musashi dispone el calendario**. Retsu no reasigna A100.

---

## Orden de ejecución (cola reiniciada)

```text
I0
 └─ I1 cubo throwaway
     ├─ I2 D/Y
     │    └─ I5 banco ruido STEP03/P-PRE     } primer hecho de la cadena
     │         └─ I6 STEP 04-05
     │              └─ I7 T2 con métricas
     │                   └─ I10 T3-T5
     │                        └─ I11 P-L2
     │                        └─ I12 STEP 08-13
     └─ I3 M/G
          ├─ I4 piloto H-ES (tras ≥30 runs)
          ├─ I8 P-MOD E0          } paralelo CPU con I5
          └─ I9 P-CAP lab

I13 F5  ── paralelo, otro repo, otro auditor
I14 F1/F2/F4 ── no se abren para esto
```

**Qué se “reinicia”:** todo lo que era “siguiente = escribir STEP 08” o “siguiente = barrer modelos”.  
**Qué no:** T0/T1, protocolos, recortes, G12, WP1, campañas selladas.

---

## Compuertas globales (fail-closed)

G1. Sin I1 throwaway, no hay ETL nuevo.  
G2. Sin hash de I2/I3, no hay H-ES.  
G3. Sin generador, no hay `I_free_synth`.  
G4. Sin 5A, no se interpreta entropía financiera.  
G5. Nulo se publica.  
G6. PDF P-MOD no cita H-ES como resultado.  
G7. Flags largos. CPU. No parar GPU/Postgres/Metabase ajenos.

---

## Lista de experimentos (mínima, para no perder el hilo)

1. Fixture I2: una senoidal + ruido conocido → `I_free_synth` y `L_C` reproducibles.  
2. Fixture I3: ANN 8-8-1, pesos fijos en el test → `M.L_zstd` hash.  
3. Banco P-PRE (I5): SNR grid, tres entregas, DLinear congelado.  
4. 30× ANN/DLinear daily-chico (I4): val-loss vs `M.L_C(e)`.  
5. E0 P-MOD (I8): dos escalas, H2/H3.  
6. 5A (I6): fuente sintética de \(h_X\) conocido.  
7. T2 (I7) cuando C31–C36 cierren.

Todo lo demás espera a que 1–3 existan.
