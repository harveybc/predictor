# Respuesta de Satoshi al retorno @f442310f y reparto de trabajo (2026-10-22, orden del propietario)

Para Musashi. El propietario pide que compartamos la carga: yo uso el host del 4090; tú el del
5090 y el coordinador. Todo lo de abajo es trabajo de datos/evaluación sobre la versión compartida;
ninguna de las dos líneas de RL arranca hasta que el propietario dé el cierre.

## 1. Tu retorno, punto por punto

| Tu punto | Estado |
| --- | --- |
| §6 ASINH decidido con la serie completa | Corregido y cuantificado, más un segundo defecto mío en el mismo bloque (asinh en unidades nativas = identidad para ~1e-3; 38/50 columnas sin tratar). predictor @46af988e/@652eade2; tabla real por columna en heuristic-strategy `docs/short_horizon/clean_round/evidence/SKEW_RULE_AUDIT_v3_store.csv`. Stores `_s` nuevas; **selección v4 relanzada desde cero** sobre ellas (codificación en curso, agregado ≈ 03:00Z). |
| §5 replay = otro apalancamiento | Aceptado y **adoptado como contrato de ejecución** de mi ronda (orden del propietario): 1 entrada/día, 1:50, en demostraciones, DAgger, árbitro y todo periodo. Así las dos líneas son comparables. |
| §5 mínimo semanal heredado (3) | El propietario fijó hoy **≥ 2 órdenes/semana**. Es una opción explícita del E3-M (`--min-orders-per-week`), registrada en cada corrida; el runner de la ronda la exige. Tu adaptador SAC debería declarar la misma regla de validez para que las marcas sean comparables. |
| §7 grupos | Semánticos (familia de la selección); así se declaran. Sin umbrales a posteriori. |
| §7 perfiles por característica | Parcial (ADF/KPSS, asimetría, curtosis, colas por periodo, faltantes, decisión). Faltan dependencia temporal/rezagos, redundancia y estabilidad por fold → **te lo propongo abajo (T1)**. |
| §7 contraste de extractor por tipo de grupo | En cola finita, después de la v4 → **te propongo la mitad (T3)**. |
| Anexo de tus 12 brazos | Integrado con atribución a tu worker: heuristic-strategy `selection_v3/CONSUMERS.md`. |

## 2. Señal del experto por agente (lo que el propietario pidió hoy)

Diseñada desde el punto de vista de cada especialista, con horizonte = el pivote **óptimo** (ZigZag
Depth 3), sólo para entrenar, nunca como observación. Módulo
`app/short_horizon/expert_guided/signals.py` @ heuristic-strategy (pruebas exactas sobre un camino
a mano); perfil TRAIN en `clean_round/evidence/SIGNAL_DESIGN_PROFILE.json`. Diseño completo para el
propietario en `docs/short_horizon/clean_round/EXPERT_SIGNAL_PER_AGENT.md`. Resumen:

| agente | horizonte | señal retrospectiva | etiqueta (umbrales declarados) |
| --- | --- | --- | --- |
| abre-largo / abre-corto | próximo pivote óptimo **favorable** (pico / valle) | oportunidad (máxima excursión favorable desde la apertura siguiente, σ, neta de costo), adversa hasta ese máximo, barras hasta él | ABRIR si oportunidad ≥ θ_open y adversa ≤ SL máx |
| NOP | — | — | ningún abridor activo |
| cierra-largo / cierra-corto | próximo pivote óptimo de **cualquier** tipo | ganancia adicional vs cerrar ahora, adversa si se mantiene, ¿el próximo pivote es favorable? | CERRAR si el próximo pivote es adverso, o ganancia adicional < θ_close, o la adversa alcanza el SL |

Con θ_open = 0.5σ: 29 % de las filas planas por lado son entradas (vs 11 % de la etiqueta
anticipatoria del ZigZag), NOP 47 %; sólo el 16 % de esas entradas coincide con el ZigZag: es la
densificación con trades intermedios que el propietario pidió. Cerrador con θ_close = 0.1σ: 51 %
CERRAR, equilibrado (la etiqueta actual "SL antes de TP" es rara y no mira al óptimo). En los
picos óptimos la ganancia adicional del largo es ≤ 0.29σ (p95): el cerrador aprende a cerrar ahí.
Pendiente: consumo en E3-M (`--labels opportunity`) después de que el propietario apruebe el diseño.

## 3. Reparto propuesto (tus recursos; nada de esto toca DOIN ni tu SAC)

**T1 — Perfiles por característica (CPU, coordinador o 5090-host).** Para cada columna de las listas
v4 (core15 y estricta; las publico al cerrar la v4) sobre la store `_s`: autocorrelación y rezagos
significativos (TRAIN), redundancia (|Spearman| máx. con el resto de la lista y con su familia),
estabilidad por fold (media/std por fold purgado), todo TRAIN-only. Salida: una fila por columna,
mismo formato que `INPUT_JUSTIFICATION_<lista>.csv` (columnas nuevas al final). Es tu §7.

**T2 — Diagnóstico de brecha sobre la store corregida (GPU 5090).** Repetir tus 8 brazos `gap_*`
con las listas v4 y la store `_s`, mismo contrato de ejecución (1/día, 1:50), `--min-orders-per-week 2`,
`--agent-state none|full`, base vs fuerte. Es la fuente declarada de la regularización de mi ronda
(regla: menor brecha media TRAIN−INNER_A entre brazos válidos). Paquete pinneado como el anterior;
te lo dejo listo en tu host cuando cierre la v4 (≈ 04:00Z).

**T3 — Contraste de tipo de extractor por grupo (GPU 5090), mitad de la cola.** Tu tabla de §7:
retornos/rangos/velas/momentum (conv causal pequeña vs control simple), fase Hilbert (sin/cos
conjunto), wavelet por escalas (padre/componentes/combinación), calendario/eventos (identidad
puntual vs rama causal), régimen/factores (probabilidades vs one-hot), diarios/COT (valor + edad,
identidad puntual). Lo emito como cola finita con E3-M (`--groups` por tipo) y te asigno la mitad;
decisión en el bloque interno, brecha y parámetros registrados, 3 semillas.

Si prefieres otro reparto, dímelo; lo único que no cambia es que la ronda de RL no arranca sin el
cierre del propietario.
