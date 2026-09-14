# D3 — diseño y pruebas de aceptación: cuantización/compresión, representaciones tiempo-frecuencia y detectores (D2-R7)

Fecha: 2026-09-14. Estado: **DISEÑO, sin ejecutar**. No hay scoring, ni feature
selection, ni modelos. La rama cruda se conserva siempre; un diagnóstico completo
con abstención es una salida válida. Ninguna cifra SNR de D2 sintético se toma
como verdad para elegir el operador de una variable real.

## 1. Entrada y salida del contrato de plugin

Cada operador D3 es un plugin de `feature-eng` (grupo `feature_eng.plugins`) o de
`preprocessor` con el mismo contrato mínimo que los operadores D2
(`tools/df_operators.py`): `fit(X_train_prefix) -> state`, `transform(X, state)`
causal, `describe() -> spec`. La entrada cruda se conserva en la salida (columnas
`*_raw`) y la transformación se etiqueta con su `spec_sha256` y `state_sha256`.

| campo de la especificación | obligatorio | significado |
|---|---|---|
| `bytes_state`, `params` | sí | tamaño y contenido del estado ajustado; hash de bytes |
| `fit_scope` | sí | `NONE` \| `TRAIN_PREFIX_ONLY`; nunca calibración ni confirmación |
| `lookback_samples` | sí | máximo de muestras pasadas que consume una salida en `t` |
| `output_availability` | sí | `t + delay_samples`: cuándo la salida en `t` está completa |
| `warm_up_samples` | sí | salidas marcadas `NOT_AVAILABLE` al inicio de cada segmento |
| `delay_samples` | sí | retardo de grupo nominal medido con impulso; `0` solo con prueba |
| `cost_cpu_seconds_per_1000` | sí | medido en el piloto CPU con un hilo |
| `applicability` | sí | familias/regímenes donde se declara aplicable (del banco) |
| `chunk_restart` | sí | `IDEMPOTENT` \| `STATEFUL_WITH_CHECKPOINT`; prueba de reinicio |

## 2. Operadores propuestos (banco inicial, sin ampliar la propuesta doctoral)

| grupo | operador | estado de ajuste | lookback | disponibilidad | control no causal deliberado |
|---|---|---|---|---|---|
| cuantización/compresión | cuantizador uniforme por deciles de train (`k` niveles) | bordes de deciles (train) | 0 | `t` | ninguno (sin ventana) |
| cuantización/compresión | SAX/PAA causal (segmento trasero) | tabla de breakpoints (train) | `segment` | `t` | PAA centrado (control) |
| cuantización/compresión | delta-encoding + run-length (compresión) | ninguno | 1 | `t` | ninguno |
| tiempo-frecuencia | STFT trasera (ventana `w`, salto `h`) | ninguno | `w` | `t` | STFT centrada (control) |
| tiempo-frecuencia | wavelet trasera (Haar/db4, `L` niveles) | ninguno | `2^L` | `t` | wavelet centrada + desplazamiento hacia atrás (control: nunca causal) |
| tiempo-frecuencia | filtro banda trasero (Butterworth causal) | coeficientes (fijos) | orden | `t` | filtfilt (control) |
| detectores | CUSUM causal de cambio de media | umbral `h`, deriva `k` (train) | acumulado | `t` | CUSUM bidireccional con lookahead (control) |
| detectores | detector de extremos MAD trasero | mediana/MAD (train) | `w` | `t` | MAD centrado (control) |
| detectores | detector de régimen de varianza (ventana trasera) | ninguno | `w` | `t` | ventana centrada (control) |

Cada operador expone prefijo y sufijo futuro alterado, bordes y reinicio por
chunk a la auditoría de causalidad existente (`df_wavelet_audit`/prefijos):
el control no causal DEBE fallarla; el candidato DEBE pasarla.

## 3. Pruebas de aceptación (antes de implementar)

1. **Prefijo:** `transform(X[:n])[:n-lookback-delay] == transform(X)[:n-lookback-delay]` bit a bit.
2. **Sufijo alterado:** cambiar `X[t+1:]` no cambia la salida en `[0, t]`.
3. **Borde/warm-up:** las primeras `warm_up_samples` salidas son `NOT_AVAILABLE`, no 0.
4. **Chunk/reinicio:** procesar en dos trozos con checkpoint reproduce el procesamiento entero.
5. **Retardo medido:** impulso unitario → `delay_samples` observado == declarado.
6. **Control no causal:** el control declarado falla 1–2; se registra, nunca se promueve.
7. **Disponibilidad:** la marca temporal de cada salida respeta el contrato del recurso
   (`availability` del lago, GOV-N2): ninguna salida antes de `label + completion_lag_max`.
8. **Coste:** piloto CPU de un hilo ≤ presupuesto declarado; memoria ≤ 2 GiB por proceso.
9. **Aplicabilidad:** un operador declarado inaplicable a una familia produce `NOT_APPLICABLE`, no un número.
10. **Rama cruda:** la salida conserva `*_raw`; ninguna decisión elimina la rama cruda.

## 4. Vínculo con `feature-eng` y el banco

Los operadores se registran como plugins de `feature-eng` (entry points), con
entradas gobernadas (Flow v3, `tools/governed_run.py`) y salida bajo el terminal.
El banco D3 = banco sintético D2 (regímenes con eventos: pasos, impulsos, motivos,
régimen de media/varianza, tendencia) + los tres recursos toy contratados; el
banco financiero queda cerrado hasta que sus contratos tengan evidencia (07 de
data-gov). Dependencias concretas: adjudicador reparado (R2), vista de cobertura
vigente (R6), micro-run productivo reconciliado (N3) para toda medición que
fundamente decisiones.

## 5. Lo que este diseño no hace

No ejecuta D3, no selecciona features, no actualiza los ~cientos de indicadores
de `feature-eng`, no prueba modelos, no amplía la propuesta doctoral, y no toma
ninguna cifra sintética como prueba de utilidad pública o financiera.
