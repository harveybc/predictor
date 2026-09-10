# Retsu → Musashi — STEP 13: repartir presupuesto, no bits de Shannon

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**No es una orden de ejecución.** No barro anchos del composite. No NAS. No GPU. No mezclo esto en el `.tex` de La Sabana. **No redacto el documento final** hasta que Harvey avise que el cierre/correcciones están en Downloads.

**Insumo:** `STEP_13_MULTIPLEXING_MIMO_BUDGET_ALLOCATION_MULTI_BRANCH_FINAL.md` (3056 líneas; SHA-256 `b2e7c59439a29b3d140f4c8ba7017cd561033db38a2b4c1a24edd82fd8bd90a6`).

---

## 0. Una línea

Con ramas útiles y presupuesto finito \(B\), ¿hace falta asignar capacidad según valor marginal y redundancia, o el 64:32:32:32 del composite ya está en la frontera?

Homólogo: water-filling / bit-loading / MIMO. **No** se llama bits de Shannon a los canales Conv1D. \(c_i\) es ancho/FLOPs/latencia medible. C6 (\(R_Z\)) y el \(C\) de MacKay del tema 2 son otros números.

12 = *cuál* modo. 13 = *cuánto* a cada rama del modo activo. El primer experimento es **estático**, con \(a\) congelado. Dinámico solo después de 12.

---

## 1. Ancla concreta — verificado

`predictor_plugin_composite.py`, `close_window_only`: CLOSE Conv1D causal → 64; HF 15m → 32; HF 30m → 32; point → 32. Concat + fusión. Ratio **64:32:32:32**, \(B=160\) canales pre-fusión.

H13.2 convierte un default de ingeniería en pregunta falsable. Un nulo (“ese ratio ya sirve”) es tesis.

Nota, no es experimento de 13: las ramas HF usan `padding="same"`, no causal. Si 13 se corre, o se declara endpoint-causal sobre la ventana, o se corrige *antes* y no se mezcla con el barrido de anchos.

Fusión y cabezales (BiLSTM, Flipout) **congelados**. Solo filtros pre-fusión. Igualdad de parámetros o se declara el desvío.

---

## 2. Recorte de hipótesis

Núcleo:

| Id | Afirmación | Se falsifica si |
|---|---|---|
| H13.1 | Igual no es óptimo a algún \(B\) | El igual empata todo |
| H13.2 | 64:32:32:32 es testeable; puede ser ya Pareto | El actual queda en la frontera |
| H13.3 | Rendimientos decrecientes | Utilidad no cóncava / umbral |
| H13.6 | \(c_i=0\) puede ganar a presupuesto apretado | Todas las ramas piden \(c_i>0\) |
| H13.10 | Ahorrar FLOPs \(\neq\) bajar latencia | (control: hay que medirlo) |

H13.4 greedy si 13.3 cóncavo. H13.5 redundancia usa 09, no se inventa. H13.11 dinámico **después** de 12. H13.8 SNR no basta para asignar. H13.12 contrato de dominio, no tesis.

No NAS. No slimmable/OFA primero. No Alloc-MoE. Una unidad de presupuesto por experimento: **ancho pre-fusión** primero.

---

## 3. Compuertas

| Gate | Qué | Si falla |
|---|---|---|
| **13A** | Al menos una rama tiene curva de utilidad no plana | No hay 13 |
| **13B** | \(P\) cambia con \(B\) | El ancho actual basta |
| **13C** | \(\Delta P/\Delta C\) estable en bloques de train | No greedy |
| **13D** | Greedy/knapsack ≈ mejor buscado | Interacciones; no water-filling ciego |
| **13F** | Latencia de pared, no solo FLOPs | El “ahorro” es papel |
| **13G** | Colas | Esa asignación se descarta |
| **13H** | Test una vez | — |

A diferencia de 12, **13A puede usar el composite actual** sin esperar denoise/fase. Sigue siendo un experimento de *capacidad*, no de datos. No se lanza como campaña GPU antes del banco CPU de STEP_03. Cuando toque: flags de anchos, fusion fija, `CUDA_VISIBLE_DEVICES=""` salvo orden.

---

## 4. Dónde

Solo `predictor_plugin_composite.py` (o config de anchos). No extractor. No preprocessor. No agent-multi. No cuarto repo.

---

## 5. Documento final

Cuando Harvey avise que el cierre y las correcciones están en Downloads, armo **un** índice/mapa de la cadena 01–13 + patches + recortes RETSU, sin reescribir los STEP. Hasta entonces no lo genero.

— Retsu
