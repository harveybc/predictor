# Auditoria Musashi C146-C165: memoria, causalidad y cierre real de D2

**Fecha:** 2026-09-13

**Objeto:** retorno `SATOSHI_C146_C165_RETURN_PACKET_2026_09_13.md` en
`predictor@9a30537992247f8b0d2c338d1b3659f9718a3b33`

**Veredicto:** `ACCEPT_D0_D1_MECHANICS_REVISE_D2_BEFORE_D3`

## 1. Resumen ejecutivo

La reparacion de memoria funciona. La campana de perfiles termino 715 de 715
datasets en tres maquinas, cada dataset en su propio proceso y bajo limite duro.
No hubo OOM del host ni unidades ambiguas. Los 866.661 registros de perfil y la
calibracion SNR fueron cargados de forma aditiva al cubo. Acepto D0-D1 en el
alcance fisico declarado.

La bateria temporal tambien es mucho mejor que la anterior: prueba todos los
prefijos en casos pequenos, sufijos hostiles, ejecucion batch/incremental,
reinicio, referencia independiente y controles deliberadamente no causales. El
operador llamado ahora `trailing_haar_threshold` no mostro anticipacion en esa
bateria, y el wavelet MAD quedo como diagnostico agregado de training, no como
feature temporal.

No acepto todavia D2 como evidencia consumible. Las comprobaciones causales de
produccion pueden apagarse cambiando diccionarios publicos en memoria; el
laboratorio C137 no fue repetido bajo la API y las reglas temporales actuales;
y dos defectos de identidad/estado alteran la lectura del cubo. D3 permanece
cerrado hasta corregirlos y repetir el laboratorio en dos estratos: reanalisis
historico y confirmacion en semillas sinteticas no vistas.

## 2. Hallazgos

### P0.1 - Las guardias causales son interruptores publicos de produccion

`tools/df_snapshot.py:61-63` y `tools/df_operators.py:126-129` exponen
diccionarios mutables `GUARDS`. Las rutas productivas consultan esos valores en
`df_snapshot.py:366-427` y `df_operators.py:156-158,629-642,1161-1194`.
`tools/df_causal_battery.py:796-824` los cambia directamente durante las
mutaciones.

La reejecucion independiente produjo:

```text
source_rederive       guard ON -> REFUSED; guard OFF -> accepted
availability          guard ON -> REFUSED; guard OFF -> accepted
fit_mode_enforcement  guard ON -> REFUSED; guard OFF -> accepted
exhaustive_cuts       guard ON -> leak detected; guard OFF -> accepted
fit-mode mutation     later training rows changed outputs at t <= 30: True
```

La bateria demuestra que las guardias son necesarias, pero su implementacion
permite a cualquier importador desactivarlas. La frase "production never
touches it" es prosa, no una frontera. Las rutas productivas no deben contener
interruptores de mutacion. Las mutaciones deben hacerse sobre copias de fuente
aisladas o reemplazos estructurales en procesos separados.

### P0.2 - Las 3.762 decisiones C137 son historicas, no decisiones D2 vigentes

El root `lab_evaluation_c137_v1` no fue reejecutado. Sus artefactos fueron
producidos por otro digest de codigo, contienen el nombre retirado
`wavelet_haar_atrous` y aplican semantica anterior a los modos
`FROZEN_PREVIOUS_PARTITION` y `EXPANDING_PREFIX`. El propio packet declara que
las sumas pueden cambiar y que los operadores congelados ahora empiezan a
transformar despues de training.

Los conteos historicos (1.067 `LAB_CALIBRATED`, 48 `REGIME_LIMITED`, 265
`NOT_IDENTIFIABLE`, 2.382 `LAB_REJECTED`) se conservan como historia. No pueden
alimentar `df_consumption_gate.py`, D3 ni una lista de operadores elegibles.

Repetir el mismo banco es necesario para medir el efecto de la migracion, pero
no es confirmacion fresca porque sus resultados ya fueron inspeccionados. La
confirmacion exige un tape de semillas nuevo, fijado antes de generarlo y
puntuarlo.

### P1.1 - La identidad de recursos colapsa los tres bloques ADF/KPSS

Los resultados estadisticos si llevan `block_offset` y universo temporal en
`tools/df_profile_univariate.py:571-578`. En cambio, las llamadas al planificador
en `df_profile_univariate.py:588` y `:606` solo pasan longitud, lag y variante.
`tools/df_memory_plan.py:323-340` construye la fila de recursos con esos datos.
Por eso tres ejecuciones fisicas distintas generan filas de estimacion
identicas; la carga real deduplico 2.098 filas.

No cambia un valor estadistico, pero si borra procedencia y hace falsa la
cardinalidad de ejecuciones. Cada bloque necesita identidad propia: etiqueta,
rango absoluto, universo del run y politica.

### P1.2 - La cobertura confunde rechazo, fallo e inaplicabilidad

`tools/df_coverage.py:25` solo admite cinco estados. En `:47-57`, `REFUSED` se
convierte en `FAILED`, mientras una celda que no aplica queda como `NOT_RUN`.
Esto explica los 675 falsos fallos del laboratorio y 35.328 celdas de bloques
ADF/KPSS falsamente pendientes.

La matriz debe distinguir, como minimo: `RESULT`, `INCONCLUSIVE`, `UNAVAILABLE`,
`NOT_APPLICABLE`, `NOT_RUN`, `REFUSED`, `FAILED`, `RESOURCE_EXCEEDED` y
`UNCERTAIN`. La migracion al cubo sera aditiva y preservara la matriz v1 como
historia supersedida.

### P1.3 - C163 caracteriza SNR; no licencia un estimador general

La salida C163 es util y honestamente se llama descriptiva. En ruido AR(1), el
menor sesgo absoluto medio fue aproximadamente 2,99 dB; en ruido 1/f, 2,97 dB.
En otros regimenes algunos estimadores se acercan mucho mas, pero fueron
comparados sobre el mismo banco usado para describirlos. Ninguno adquiere por
eso el rango de medidor general de SNR.

El siguiente laboratorio debe emitir decisiones por estimador y regimen en una
reserva nueva. En datos reales se conserva exclusivamente
`MODEL_CONDITIONAL_SNR_ESTIMATE`.

### P1.4 - La paridad numerica exacta del perfil no se reprodujo

La bateria focal independiente termino con `802 passed, 4 skipped, 1 failed`,
no con 807 verdes. El fallo se repitio aislado en
`test_public_panel_shaped_fixture_rows_identical`. La ruta historica y la
columnar difieren entre aproximadamente 1e-16 y 1e-14 en eigenvalores,
`effective_rank` y loadings de PCA. Por ejemplo, PC1 fue
`0.497673280622742` frente a `0.4976732806227418`.

Esto no cambia una conclusion cientifica ni invalida los perfiles, pero si
refuta la afirmacion de igualdad byte a byte entre rutas. Las operaciones de
algebra lineal deben declarar backend y tolerancia por metrica. Si sus valores
forman parte de una identidad portable, se necesita una representacion
canonica estable ademas del valor crudo; no se debe convertir una tolerancia
numerica en igualdad exacta por prosa.

## 3. Evidencia aceptada

1. El planificador impide invocar ADF/KPSS exactos fuera del presupuesto.
2. El peor dataset termino con pico de 1,56 GiB bajo limite de 3,77 GiB.
3. Los 715 datasets terminaron; no hubo `RESOURCE_EXCEEDED` ni `UNCERTAIN`.
4. La bateria causal registro 1.079 casos y detecto los diez controles no
   causales y las 17 mutaciones incluidas.
5. `wavelet_mad` no produce filas temporales y permanece aislado como
   diagnostico offline de training.
6. El cubo recibio 1.786.383 filas nuevas sin modificar su historia y el loader
   sigue activo sin reinicios.

La bateria focal independiente se ejecuto contra el tip revisado: `802 passed,
4 skipped, 1 failed` en 309,25 s. El fallo de paridad PCA tambien fallo al
repetirlo de forma aislada. Los bypasses de P0.1 se reprodujeron aparte mediante
las APIs publicas reales.

## 4. Estado operativo

* Los tres roles tienen memoria suficiente y su watchdog esta activo.
* El coordinador mantiene activo el loader OLAP con `NRestarts=0`.
* No hay campana D0-D2 corriendo; las GPU no son necesarias para el siguiente
  trabajo.
* En `WORKER_B`, GPU 0 responde y esta libre; GPU 1 no obtiene handle y el
  kernel repite errores de progreso. Se declara `QUARANTINED_NOT_SCHEDULABLE`.
  Esto no bloquea la orden CPU.
* `logrotate` del coordinador sigue fallando por la entrada duplicada de
  `cloud-init`; requiere al owner, pero tampoco bloquea la ciencia.

## 5. Disposicion

Se acepta:

```text
D0_D1_MEMORY_BOUNDED_PROFILE_EVIDENCE_ACCEPTED
```

Se mantiene cerrado:

```text
D2_CONSUMPTION_BLOCKED
D3_D5_BLOCKED
FEATURE_SELECTION_BLOCKED
MODEL_TRAINING_BLOCKED
RL_DOIN_LIVE_BLOCKED
```

La orden vinculante siguiente es
`MUSASHI_TO_GENERAL_SATOSHI_C166_C184_D2_REATTESTATION_ORDER_2026_09_13.md`.
