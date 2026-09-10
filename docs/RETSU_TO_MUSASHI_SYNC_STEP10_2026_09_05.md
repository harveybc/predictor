# Retsu → Musashi — STEP 10: alinear el reloj, no el tensor

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**No es una orden de ejecución.** No implemento GCC, DTW ni grafos de lead–lag. No abro el código de STEP_11. No toco GPU. No mezclo esto en el `.tex` de La Sabana.

**Insumo:** `STEP_10_SYNCHRONIZATION_TEMPORAL_ALIGNMENT_TIMING_RECOVERY_FINAL.md` (3277 líneas; SHA-256 `17f30f0ac4664defaa9a4d1c7b17524cebda73559a3a319ddd74471d1094735c`). PATCH 002 §8 vigente. Sugerencias del 11 ya pasadas al work agent.

---

## 0. Una línea

Reloj \(\neq\) observación \(\neq\) publicación \(\neq\) instante en que la información es usable. Si hay un \(\tau_{ij}\), se estima con incertidumbre, se emite como metadato, y **no se arrastra el futuro hacia \(t\)**.

Principio: **nunca alinear timestamps solo para que el tensor parezca síncrono.** A veces lo correcto es *no* sincronizar y modelar tiempo irregular (GRU-D / máscara de edad). Ese nulo está en el protocolo y lo dejo.

Lead–lag en finanzas **no** es tesis (Hayashi–Yoshida, grafos 2022–2026). El hueco es el contrato point-in-time \(A_i(t)\) + lag con lock + vista causal opcional, encima de 05/06/09.

---

## 1. Qué no recortaría

1. **Cuatro fenómenos, cuatro métodos.** Error de reloj; muestreo asíncrono; retraso de propagación; warping elástico. DTW no es el estimador de un \(\tau\) fijo.
2. **No se mueve el futuro atrás.** Si \(i\) adelanta a \(j\) en 2 barras, en \(t\) se usa \(X_i(t)\) para horizonte \(\approx 2\). No se fabrica un feature de \(j\) en \(t\) con \(X_j(t+2)\).
3. **Frontera 09.** VAR de orden fijo = 09. Estimar/actualizar \(\tau_{ij}(t)\) = 10. No se esconde sync en el cancelador.
4. **Frontera 06.** Fase/coherencia se *calculan* en 06; 10 las convierte en delay + confianza. No se recalcula el FFT.
5. **Pares.** No \(O(m^2 L)\). Candidatos: semántica, tabla H5.8, coherencia, grupos de 09.
6. **soft-DTW del TCN** (`predictor_plugins/common/losses.py`) es *pérdida* predicción–target. No es alineación de entradas. No se relabela.

---

## 2. Recorte de hipótesis: veinticuatro no entran

Núcleo:

| Id | Afirmación | Se falsifica si |
|---|---|---|
| H10.1 | Correlación cruzada recupera delay fijo a SNR decente | Falla el sintético T0 |
| H10.9 | Metadato \(\hat\tau\) puede ayudar *sin* desplazar la serie | Solo el shift duro gana, o nada gana |
| H10.11 | Reemplazo duro es más arriesgado que paralelo \([X,\tau]\) | Tirar el crudo gana siempre |
| H10.18 | Alineación retrospectiva sin causalidad infla \(P\) | (control: tiene que cumplirse) |
| H10.19 | Macros/eventos se alinean por *publicación*, no por timestamp limpio | Vintage ignorado y “mejora” |
| H10.24 | El método depende de la clase de desalineación | Un sincronizador único gana en T0–T5 |

Con compuerta: H10.4 \(\tau(t)\) si el fijo falla; H10.7 DTW solo si hay warp elástico *y* causal; H10.16 no interpolar por default.

**Fuera del banco:** grafos CGLR/TLGNN/DGLASA, DCTW, OT, Neural CDE, RL. 2026 papers = comparadores de papel.

Coherencia baja ⇒ no se fía el delay de fase (06/10). Periodicidad común genera \(\tau\) falso (T7). Nyquist: “sub-sample delay” más fino que \(\Delta t\) es teatro salvo interpolación declarada.

---

## 3. Compuertas

| Gate | Qué | Si falla |
|---|---|---|
| **10A** | Contrato \(A_i(t)\): zona horaria, cierre de barra, vintage | No se interpreta ningún \(\tau\) |
| **10B** | T0: \(\hat\tau\) recupera el delay conocido | Se para |
| **10D** | Al menos un par con lag estable vs surrogate | No hay 10 de datos reales |
| **10E** | Metadato de lag mueve \(P\) o calibra, núcleo congelado | Se declara nulo; no se shiftea |
| 10F | Vista alineada causal *además* del metadato | Solo \(B_\tau\) |
| 10G–10K | Dinámico, DTW, irregular, transfer, test una vez | No se afirma generalidad |

Arquitectura segura: crudo + \(\tau,q,\mathrm{LOCK}\) + vista alineada **opcional**. Modo C (reemplazo duro) no es default.

---

## 4. Dónde vive, cuando haya código

| Pieza | Sitio | Ahora |
|---|---|---|
| Contrato de timestamps / \(A_i(t)\) | manifiesto JSON + preprocessor | Nada |
| \(\hat\tau\) fijo, GCC, rolling causal | preprocessor, fit train, replay | Nada |
| Rama \(B_\tau\) | predictor, núcleo congelado | Nada |
| DTW/soft-DTW de *entradas* | Tras 10H | Nada; no reusar la loss del TCN como feature |
| Grafos / Neural CDE | No | No |

CPU. Caché de \(\hat\tau(t)\) como el FFT. No campañas.

---

## 5. STEP 11

El protocolo ya apunta a 11 con PATCH 002. Las sugerencias están en `docs/SUGERENCIAS_STEP_11_PARA_WORK_AGENT.md` y Harvey dijo que las pasó. Yo no redacto el 11.

— Retsu
