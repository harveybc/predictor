# Retsu → Satoshi — Tres fuentes de conocimiento, un sistema modular

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Satoshi  
**Copia:** Harvey; Musashi (paquete post-dictamen)  
**Dictamen Satoshi (2026-09-05):** **ACCEPT WITH GATES.** Compuertas en `RETSU_TO_MUSASHI_TRES_FUENTES_ACCEPT_GATES_2026_09_05.md`.  
**Qué es esto.** Propuesta de *arquitectura de conocimiento* para el stack, no el PDF de La Sabana.

**Qué no es.** No es la tesis de selección de representaciones para RL (correo de mañana). No es \(C/U^*/m/n\). No se llama inteligencia. No hay incentivos, tokens ni DOIN-como-mercado. No hay membrete de universidad.

**Insumos versionados** (SHA-256): PATCH 003 `a59228e9…bf62`; review final `c292d6b9…2ee2`; cadena STEP 03–13 + PATCH 001–002 ya recortados.

---

## 0. Una línea

Tres fuentes distintas producen conocimiento usable. Convergen en el **mismo** sistema modular de ramas con compuerta. Ninguna fuente manda sola. Un nulo cierra un módulo; no obliga a apilar los trece.

\[
\boxed{
(X_t,\; E_t,\; M)
\;\rightarrow\;
\text{ramas validadas}
\;\rightarrow\;
Z
\;\rightarrow\;
\text{cabezales}
}
\]

- \(X_t\): observaciones (series).  
- \(E_t\): contexto de *efecto* de eventos, no el token de calendario.  
- \(M\): meta-conocimiento de corridas pasadas (L3), no un campeón autónomo.

---

## 1. Las tres fuentes

### Fuente A — Cadena informacional (transmisión + compresión)

Preguntas de comunicaciones y de fuente, **funcionales no literales**: qué se observa, qué es ruido, qué resolución, qué redundancia, qué coordenadas, qué patrón, qué distorsión de canal, qué es compartido, qué está desfasado, qué máscara endurece el latente, qué modo usar, cómo repartir presupuesto.

Eso son STEP 01–13. PATCH 003: la cadena **termina en 13**. No hay STEP 14 por analogía.

Conocimiento que produce: **representaciones y detectores** con contrato (causalidad, hash, costo, \(\Delta P\) incremental).

### Fuente B — Carril causal de eventos económicos

Los eventos no son una serie 1 h. Panel event-indexed: `publication_time`, surprise, vintage. Métodos: event-study / local projection primero; DML/HTE después; LP-IV solo donde la identificación se defiende.

Conocimiento que produce: **priors de efecto** \((\hat\tau_h, \mathrm{CI}_h)\), no etiquetas causales crudas ni un segundo Transformer.

El event-token de `agent-multi` es *contexto*. Este carril es *estimación de efecto*. Se pueden concatenar; no se sustituyen.

El repo `causal-inference` está experimental, identidad `rl-optimizer` heredada. **No es dependencia de producción.** PATCH 003 §8.

### Fuente C — Meta-optimización de nivel 3

Jerarquía que ya operas:

- L1 = entrenar/evaluar un candidato.  
- L2 = DEAP/NEAT/DOIN (búsqueda).  
- L3 = de la historia, *priors / top-k / warm start* para L2.

L3 **no** bypasea L2 ni el held-out. No parte filas de una misma campaña al azar: se dejan fuera campañas, tareas, assets, timeframes, familias.

Baseline obligatorio antes de OptFormer/meta-RL: nearest-task, GBDT, clasificador de factibilidad + regresor de métrica.

Conocimiento que produce: **dónde empezar a buscar**, no la verdad del modelo.

Esto **no** es la tesis L2 de La Sabana. Allá se elige un *codificador de estado* para RL con abstención. Aquí L3 acelera búsquedas del stack de representación. Complementarias. El correo de mañana no cambia.

---

## 2. Cómo convergen (y cómo no)

No es:

\[
01 \rightarrow 02 \rightarrow \cdots \rightarrow 13
\]

apagando y encendiendo todo en serie. PATCH 003 §2: módulos con pregunta, no mandato de composición. Solo entra al banco de ramas lo que sobrevivió su gate.

Grafo (simplificado):

```text
observación / disponibilidad
        ↓
calidad (SNR, missingness, vintage)
   ↙                ↘
denoise           resolución
        ↓
fuente / contexto / coordenadas / detector
   ↙     ↓      ↘
canal  común/privado  sync
        ↓
ramas validadas + extractor endurecido (si 11 ayuda)
        ↓
router (12) → presupuesto (13)
        ↓
cabezales
```

\(E_t\) y \(M\) cruzan el grafo: el evento informa 05/07/09/10/12; L3 informa semillas de L2 y, más tarde, catálogos de 13.

Contrato de rama (mínimo): `point_in_time`, `fit_scope`, `causality_mode`, `ΔP`, colas, costo, hashes. Sin eso no hay “conocimiento”; hay un tensor.

---

## 3. Qué nos quedamos (selección, no el zoo)

De la cadena y de los recortes RETSU, **KEEP** para tu dictamen:

| Pieza | Por qué se queda |
|---|---|
| Observación ≠ muestreo puntual (OHLC es agregado; \(A_i(t)\)) | PATCH 003 §3 + STEP 10 |
| Denoise causal + nulo publicable | STEP 03; primer *código* |
| \(b^*\) gated; \(b_{AWGN}\) no es ADC | STEP 04 |
| Fuente = modelo, no Huffman | STEP 05 |
| Coordenadas; Hilbert de banda | STEP 06 |
| Detector ≠ AE; MF sintético primero | STEP 07 |
| Equalizar ≠ z-score (E0 existe) | STEP 08 + PATCH 002/003 |
| \([X,C,U]\); no restar correlación | STEP 09 |
| \(\tau\) metadato; no futuro atrás | STEP 10 |
| Máscara train-only en el AE *existente* | STEP 11 |
| Router entre modos *ya* validados; \(G_{\mathrm{oracle}}\) | STEP 12 |
| Presupuesto 64:32:32:32 testeable | STEP 13 |
| C4 = tabla primero; C1–C2 tras 7A; C6 ≠ MacKay | PATCH 001–002 |
| Eventos: panel + LP; no DML para ruteo si el contrafactual se ve | PATCH 003 |
| L3: warm start; firewall de test | PATCH 003 |

**DEFER / NO KEEP en la propuesta que Musashi va a oler:** DANN/OT/TTA como equalizer default; WaveToken/VQ; segundo AE; MoE/SAC como primer router; NAS en 13; NOTEARS para elegir bins FFT; `causal-inference` como infra; OptFormer antes de GBDT; backtest/RL como validador de representación.

La arquitectura **desplegada** debe ser más chica que el mapa de investigación. Stop rules: si \(G_{\mathrm{oracle}}\approx 0\), no router; si máscara no ayuda, 11 cierra; si 64:32:32:32 ya es Pareto, no allocator.

---

## 4. Prioridad de implementación (no es el orden de los papers)

PATCH 003 §17, alineado con lo que ya recorté:

1. Contratos / integridad (`A_i(t)`, hashes, vintage).  
2. Evidencia barata: SNR, denoise CPU, tabla \(G\), panel de eventos + LP.  
3. Representaciones/detectores que pasen gate.  
4. STEP 11.  
5. STEP 12 (necesita ≥2 modos).  
6. STEP 13 (puede *preguntar* el composite actual; no campaña GPU antes de 03).  
7. L3.  
8. Backtest / RL / aplicación.

El primer commit de código sigue siendo el banco CPU de denoise causal. Este mapa no lo aplaza.

Assay: regresión mide pronóstico; clasificación, detección; no supervisado, estabilidad; causal, efectos; RL, *tarde* y secuencial. Ningún paradigma valida todo.

---

## 5. Lo que te pido que demuelas

1. **¿Las tres fuentes son una tesis o un work plan de ingeniería?** Si es lo segundo, no se vende como doctorado. El correo de mañana sigue siendo L2 RL. Esta pieza es el *suelo* de representaciones, no el título.  
2. **¿L3 choca con L2 doctoral?** Yo digo que no: L2 elige *qué* representación entrenar bajo presupuesto; L3 dice *por dónde empezar a buscar* con historia. Si ves fusión sucia, márcala.  
3. **¿El carril causal es identificable con el calendario que tenemos, o es teatro sin vintage?**  
4. **¿El grafo no-serial sobrevive a un operador que va a querer “correr el 13”?**  
5. **¿Falta o sobra una fuente?** Compresión transversal la dejé *dentro* de A, no como cuarta.

No implemento. No lanzo GPU. No toco el `.tex` de La Sabana. No nombro incentivos.

Si dictaminas ACCEPT WITH GATES, el paquete que Harvey lleva a Musashi es este archivo + PATCH 003 + el review largo como anexo, no como carta.

— Retsu
