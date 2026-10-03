# Satoshi: ejecución canónica selection-first

Fecha: 2026-10-03. Esta orden sustituye las colas fechadas y cualquier orden de
usar NEAT como optimizador. No sustituye evidencia histórica.

## 1. Autoridad

Lee, en este orden: el plan maestro v3; FEATURE_SELECTION_REPRESENTATION_WORK_PLAN;
MODULAR_STACK_WORK_PLAN; PROJECT_METHOD_STATE; EXPERIMENT_EXECUTION_QUEUE y
CURRENT_EXECUTION. No reconstruyas el plan desde retornos RP, STATUS M06,
mensajes anteriores ni ramas fechadas.

## 2. Corrección inmediata

- Confirma que no exista proceso o servicio NEAT de la campaña incorrecta.
- No trates 321 columnas ECL ni 83 ETH como selección final.
- Conserva sus artefactos sólo como diagnóstico del protocolo que ejecutaron.
- DEAP propone configuraciones. DOIN evalúa candidatos. NEAT queda fuera hasta
  una representación final congelada.
- No uses R3: no está definido.
- No ejecutes H-CORE antes de seleccionar arquitectura y régimen R0/R1/R2.

## 3. Orquestación paralela

Despacha agentes con worktrees y escritura disjunta. Usa Hermes/OpenCode para
inventarios, fixtures, tablas, documentación y tests acotados; reserva el agente
principal para integración y decisiones de contrato.

| Lane | Recurso | Trabajo | Inicio |
|---|---|---|---|
| A | CPU | EURUSD PS0/PS1: fuentes, disponibilidad, targets y folds | ya |
| B | CPU | PS2: redundancia, relevancia y muestra exploratoria | primer lote A |
| C | CPU | PS3-C: escalera causal de tres peldaños | primer lote A |
| D | CPU ingeniería | plugin univariado, contratos y pruebas | ya |
| E | RTX 5090 | PS3-R: extractor/control sobre supervivientes | primer lote B |
| F | 5070 Ti o 4090 | extractor alternativo en las mismas filas | primer lote B |
| G | GPU libre | referencia publicada fiel independiente | ya, si está sellada |
| H | CPU/servicios | LTS, MT5 demo, Alpaca paper y M5PHET | continuo |

Gamma comparte RAM entre 5090 y 5070 Ti. Dos procesos sólo si ambos sobres caben
con reserva de host medida. La 5090 recibe el trabajo largo de mayor prioridad.
Dragon toma la alternativa o referencia y puede usar toda su RAM admisible: la
VM MT5 no se ejecuta ni se reserva hasta tener extractores y H-CORE listos para
inferencia. Omega queda para replays pequeños.
No esperes todos los lanes para alimentar el siguiente lote.

## 4. Lane A: contrato e inventario

El manifiesto principal es EURUSD. ETH queda separado como desarrollo/control.
Incluye sin omisiones silenciosas:

- FXMacroData: evento, país/moneda, consenso, actual, previo, revisión,
  importancia, publicación y recepción;
- precio, volumen y spread disponibles; Yahoo Finance y Alpaca donde la licencia,
  frecuencia y cobertura point-in-time correspondan;
- tasas, índices, VIX/SP500, fundamentales y fuentes inventariadas;
- retornos, volatilidad, técnicos y régimen de mercado de feature-eng;
- wavelet, multitaper, Hilbert, STL y Kalman como variantes con identidad y
  pruebas causales/prefix, no como columnas asumidas útiles.

Targets: Y_s h=1..6 h, Y_l h=24,48,...,144 h, Y_b de barrera y J_policy.
Define soporte, purga y as-of join por tiempo transcurrido. Selección sólo en
folds internos de TRAIN; validación externa no selecciona; test no se lee.

PS1 mide sobre todas las admisibles: missingness, constantes, escala/colas,
volatilidad, ACF, tendencia, ADF/KPSS, estacionalidad, períodos/espectro y coste.
Cada celda queda MEASURED, FAILED, NOT_APPLICABLE o PENDING.

## 5. Lane B: prioridad reversible

Por target, horizonte y fold: asociaciones robustas y utilidad OOF; redundancia
condicional y clusters de perfiles; grupos de dominio y sinergias; y una muestra
exploratoria fuera del ranking. Produce una lista provisional, nunca un descarte
final. Mantén el control de todas las admisibles.

## 6. Lane C: escalera causal

Primer estudio completo: eventos económicos hacia EURUSD.

Define primero la unidad exacta: episodio histórico anclado en t; tratamiento A
como tipo/nivel/cambio observado (sorpresa económica, transición de régimen o
cruce), outcomes Y_s/Y_l/Y_b posteriores; e historia H disponible antes de t.
Umbrales, DAG y ventanas se fijan en TRAIN.

1. Asociación/relevancia: estima dependencia condicional y ganancia OOF de A/X
   sobre Y dados H, con placebos temporales y controles negativos.
2. Intervención observada: **busca en los datos** episodios con A=a y controles
   con A=a' que compartan tipo de evento, calendario, régimen y prehistoria H.
   Comprueba overlap y balance; usa matching/propensity y estimación doubly robust
   o g-computation. Estima ATT/ATE/CATE sólo dentro del soporte observado.
3. Contrafactual del mismo episodio: ajusta el SCM temporal en TRAIN; abduce los
   shocks del episodio factual; cambia sólo A a un nivel alternativo que exista
   en el soporte histórico; conserva historia, no descendientes y demás shocks;
   propaga descendientes hasta Y. Exige reconstrucción factual, episodios
   históricos análogos, placebos y sensibilidad.

No se interviene físicamente el mercado y no se necesita simulador externo: los
tratamientos y los análogos se recuperan de la historia. Tampoco se finge que el
outcome contrafactual individual fue observado.

NOT_IDENTIFIED es una salida válida y no descarta automáticamente una feature.
No llames causal a una permutación de latente ni a intervenir la red.

## 7. Lanes D/E/F: plugin de extractibilidad

Implementa en feature-extractor una interfaz univariada temporal:

    signal        (B,T,1)
    observed_mask (B,T,1)
    delta_time    (B,T,1)
    calendar      (B,T,C_known)
    latent        (B,T,D)

Calendar incluye sin/cos de hora, día de semana y día del año; sesión y feriado
sólo si estaban publicados. El target futuro no entra al encoder operacional.
Puede supervisar probes o condicionar un generador offline TRAIN-only.

Primer control implementable: stem Conv1D causal; bloques TCN residuales
dilatados sin pooling temporal; contexto estacional proyectado por instante y
fusionado por canales; latent temporal; decoder causal; early stopping y mejor
checkpoint restaurado.

No lo declares ganador. Compáralo con identidad/raw, encoder aleatorio, AE/DAE,
un masked temporal autoencoder y un candidato past-to-current/siamés. Investiga
código oficial y licencia de TimeSiam y Ti-MAE antes de adoptar. CVAE
condicionado es variante generativa secundaria.

Piloto: ventana horaria 168, latent D=8 y una semilla. Ventanas 48/168/720, D,
profundidad y familia pasan a DOIN después del coste. La salida conserva T.

Métricas por feature/fold: MAE/MSE original y normalizado; ACF, espectro,
extremos/eventos y DTW cuando aplique; estabilidad y dimensión efectiva; probes
iguales raw/random/trained hacia Y_s/Y_l/Y_b; aporte incremental; retirada con
reajuste; coste/memoria/updates/latencia. Fidelidad generativa va aparte.

Reconstrucción pobre dispara diagnóstico, no rechazo automático. Reconstrucción
buena sin utilidad downstream tampoco selecciona.

## 8. Decisión y downstream

Con igual soporte y presupuesto compara: todas las admisibles;
predictivo/redundancia; anterior más causal; anterior más extractibilidad; y
preferencia generativa opcional. Incluye adición, retirada con reajuste,
sinergias y reincorporación. Publica el manifiesto final.

Después, y sólo después:

ARCH -> ramas preentrenadas -> E1 R0/R1/R2 -> elegir/fijar prefijo -> H-CORE ->
transferencia del núcleo -> congelar representación -> Dense vs NEAT -> SAC/DQN
raw vs modular.

Las reproducciones ECL, Weather y Traffic mantienen la receta del autor. Nuestra
selección es una intervención posterior emparejada.

## 9. Pruebas y reporte

Pruebas mínimas antes del código: futuro perturbado; target prohibido en encoder
operacional; rejilla temporal; calendario conocido; raw/random/trained; folds
TRAIN-only; NOT_IDENTIFIED no rechazado; manifiesto fail-closed; donor/digest;
save/load; early stop; mismo-row naive; y cero invocaciones de estrategia cuando
skill sea menor o igual a cero.

Actualiza un solo STATUS atómico; no gastes contexto repitiendo comandos.
Retorna al cerrar cada hito:

1. resultado o artefacto nuevo;
2. denominador de features/celdas/folds;
3. modelo, target, población, métrica, naive y skill;
4. semilla, coste, hardware y digest;
5. checklist maestro actualizado;
6. siguiente trabajo ya despachado;
7. bloqueador sólo si requiere credencial, licencia, hardware físico o broker.

No pidas permiso rutinario. Después del primer lote de cada familia, calcula ETA
por tasa observada y actualiza el mismo estado.
