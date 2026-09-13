# Orden de Musashi a General Satoshi: C146-C165 memoria y causalidad

**Fecha:** 2026-09-13

**Prioridad:** P0 antes de relanzar D1-D2

**Auditoria rectora:** `MUSASHI_AUDIT_C122_C145_RUNTIME_CAUSALITY_2026_09_13.md`

**Licencia:** CPU offline; ninguna GPU, entrenamiento, seleccion, RL, DOIN o live

## 0. Objetivo y alto

Reparar el runtime de perfiles/SNR, cerrar la frontera causal de todos los
operadores y completar C130-C140 con ejecucion acotada y distribuida.

Detenerse en:

`D0_D2_MEMORY_BOUNDED_AND_CAUSALITY_AUDITED_READY_FOR_MUSASHI_REVIEW`

No ejecute D3-D5, feature selection ni modelos.

## 1. Preservacion y PRE

### C146 - preservar el incidente

Antes de editar:

* registre las dos terminaciones OOM como intentos no gobernantes;
* preserve los roots incompletos sin promoverlos;
* derive desde disco que faltan los recibos finales de perfiles y SNR;
* identifique cada proceso, dataset, metrica, `n`, numero de rezagos y memoria
  proyectada;
* congele la reproduccion matematica de 22.71 GiB para el ADF de 13,253,761
  muestras;
* demuestre que una sola tarea puede agotar el host aunque `workers=1`.

El PRE tambien debe reproducir:

* `fit()` aceptando bytes de confirmation bajo la etiqueta `train`;
* transformacion de filas desordenadas sin rechazo;
* una mutacion futura no ejercitada por los siete cortes actuales;
* `wavelet_mad` offline alcanzable como una funcion numerica ordinaria;
* descomposicion estacional in-sample inicializada con el final de training.

## 2. Runtime acotado

### C147 - planificador de memoria por metrica

Implemente una estimacion previa por dataset, variable, particion y metrica.
Debe incluir matrices temporales, copias de `float64`, ventanas, FFT, OLS,
surrogates, bootstrap y serializacion. Persistir `estimated_peak_bytes`, formula
y parametros.

Si la cota excede el presupuesto, no invoque la biblioteca: emita
`NOT_RUN_RESOURCE_BOUND` o use una version acotada predeclarada. Nunca pruebe
por OOM.

### C148 - ADF/KPSS de memoria finita

Defina antes de medir una politica estadistica y de recursos. Son admisibles:

* maximo de observaciones determinista, con muestreo temporal en bloques y
  sensibilidad sobre al menos tres offsets; o
* estimador alternativo de memoria acotada, citado y validado contra la version
  exacta en series pequenas.

No reduzca `n` silenciosamente. El registro debe distinguir el estimando exacto
del aproximado y declarar el universo temporal usado. ADF y KPSS no son gates
de causalidad ni de elegibilidad por si solos.

### C149 - lectura columnar y salida incremental

Elimine el patron tabla completa -> lista de columnas -> `column_stack` para
datasets grandes. Lea solo las columnas necesarias, por row group o batch, y
procese por variable cuando la metrica sea univariada. No acumule todas las
filas de resultado en RAM: escriba registros verificados incrementalmente.

Las metricas multivariadas deben consumir una matriz acotada y declarada,
nunca la matriz completa por accidente.

### C150 - aislamiento y limites duros

Cada dataset se ejecuta en un proceso separado con:

* limite duro de RSS inferior a la memoria asignada;
* wall time, CPU time, heartbeat y stop file;
* BLAS de un hilo;
* terminal durable `COMPLETED`, `FAILED`, `INCONCLUSIVE`, `REFUSED`,
  `RESOURCE_EXCEEDED` o `UNCERTAIN`;
* resume por identidad del dataset y del codigo;
* ningun proceso hermano pierde su evidencia si uno falla.

Un OOM del proceso no puede convertirse en OOM del host.

### C151 - smoke del peor caso

Antes de la campana, ejecute el dataset con mayor cota en una copia de
desarrollo. La bateria debe probar que el pico observado queda bajo la cota y
que una mutacion que omite el preflight intenta exceder el limite y es
interceptada sin perjudicar el host.

## 3. Frontera causal ejecutable

### C152 - snapshot de ajuste

Reemplace `fit(spec, array, "train")` en la ruta productiva por un objeto
`FitSnapshot` inmutable que ligue:

* dataset y contrato;
* archivos/bytes y digests;
* columnas y orden;
* timestamps y disponibilidad;
* rango `[start, end)`;
* rol exacto `TRAIN` o `CALIBRATION` permitido por el diseno;
* particiones posteriores excluidas;
* digest de la matriz materializada.

El consumidor rederiva estos hechos en el ultimo punto de uso. Un array o una
cadena auto-declarada no concede ajuste.

### C153 - snapshot de transformacion

`transform_batch()` productivo debe consumir un `TransformSnapshot` con
identidad, timestamps monotonicamente crecientes, disponibilidad por fila,
columnas, particion y digest. Debe rehusar:

* filas desordenadas o duplicadas sin politica declarada;
* una observacion disponible despues del instante de decision;
* cambio de columna, rango, rol o bytes;
* una particion que el artefacto no tenga permitido transformar.

La API numerica desnuda puede quedar solo como kernel privado de tests.

### C154 - modos de ajuste temporal

Separe y nombre:

* `FROZEN_PREVIOUS_PARTITION`: ajuste en una particion estrictamente anterior;
* `EXPANDING_PREFIX`: estado actualizado solo despues de emitir la salida del
  instante correspondiente;
* `OFFLINE_ANALYSIS_ONLY_NON_CAUSAL`: diagnostico que nunca emite features.

La descomposicion estacional y los umbrales wavelet deben declarar uno de los
dos primeros para cada uso. Aplicar parametros estimados con el futuro del
mismo bloque sobre filas anteriores queda prohibido.

## 4. Auditoria wavelet hostil

### C155 - nombre y especificacion matematica

No afirme SWT ni a trous por parecido. Para el operador actual:

* renombre a `trailing_haar_threshold`, o
* publique recurrencia, convencion de borde y equivalencia numerica con una
  definicion estandar.

`T06_CAUSAL_SWT` permanece `DESIGN_ONLY_NOT_IMPLEMENTED` hasta tener su propia
prueba. Nunca lo trate como sinonimo del operador actual.

### C156 - invariancia exhaustiva de prefijos

Para cada operador causal y, de forma individual, cada nivel wavelet:

1. para series pequenas, pruebe todo `t`;
2. compare ejecutar `X` completa contra ejecutar `X[:t+1]`;
3. compare batch, muestra-a-muestra, fragmentos de todos los tamanos pequenos y
   reinicio justo antes/despues de cada frontera `2^j`;
4. exija igualdad de salida, disponibilidad, razon y estado;
5. repita con multivariado, NaN, gaps, valores constantes y escalas extremas.

### C157 - sufijos adversariales

En cada corte sustituya `X[t+1:]` por, al menos:

* ceros;
* constantes grandes finitas;
* ruido con otro seed;
* orden inverso;
* NaN por bloques;
* impulso exactamente en `t+1`;
* step, chirp y cambio de regimen.

Nada anterior o igual a `t` puede cambiar. Incluya longitudes alrededor de
`2^j-1`, `2^j`, `2^j+1` y del warm-up.

### C158 - referencia independiente

Implemente una referencia lenta para tests que construya cada salida usando
solo el prefijo visible. No comparta helpers de ventanas ni padding con la
implementacion productiva. El operador productivo debe coincidir con ella bajo
todos los casos licenciados.

### C159 - controles negativos y prohibiciones

La bateria debe detectar, individualmente:

* rolling centrado;
* `filtfilt`;
* `shift(-k)`;
* convolucion `same` o padding simetrico/periodico a la derecha;
* DWT/SWT/CWT de serie completa usada como fila temporal;
* FFT de serie completa reconstruida como feature;
* compensacion de fase desplazando la salida hacia atras;
* ajuste con calibration/confirmation;
* reordenamiento por timestamp posterior al materializado;
* reutilizacion de estado de otra serie.

Congele un control negativo por clase. Quite cada guardia mediante mutacion y
pruebe que una prueba falla por la razon esperada.

### C160 - aislamiento del wavelet MAD

Renombre su estado contractual a `OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL`. Pruebe
por grafo de llamadas y conducta que no puede aparecer en una matriz de
features, operador, router, selector o agente. Puede producir una cifra
agregada de training para calibracion; no puede producir valores por timestamp.

## 5. Ejecucion distribuida

### C161 - preflight de hosts y asignacion

Use roles, no topologia fisica en evidencia publica:

* `COORDINATOR`: inventario, scheduler, recibos y OLAP; maximo dos tareas
  pequenas simultaneas;
* `WORKER_A`: perfiles pesados, un dataset por vez bajo limite duro;
* `WORKER_B`: preflight post-reinicio aprobado el 2026-09-13; perfiles
  acotados, un dataset por vez bajo limite duro.

Antes de despachar, pruebe igualdad de commit, codigo, contratos y bytes de los
datos disponibles en cada host. Un host sin el dataset exacto no recibe la
tarea. Las GPU permanecen libres porque esta orden es CPU.

Estado de partida comprobado: ambos workers estan sanos pero tienen checkouts
anteriores (`predictor@14a1077f9c79`, `financial-data@307342627033`) y carecen
de los bancos D0. Sincronizar codigo y transferir los artefactos por digest es
parte de C161, no una tarea del owner.

### C162 - campana de perfiles

Reejecute C130-C133 desde root fresco y write-once. Distribuya por estimacion de
memoria, no por conteo de datasets. Ningun dataset ausente o fallido se oculta.
Publique por banco y dataset: estado, filas, variables, metricas, pico RSS,
tiempo y razon terminal.

### C163 - campana SNR

Solo despues de C160, reejecute C134 desde root fresco. Calibre contra clean y
noise separados del banco sintetico. El resultado real sigue siendo
`MODEL_CONDITIONAL_SNR_ESTIMATE`; ningun estimador se declara verdad general.

### C164 - OLAP y cobertura

Ensaye en base desechable y cargue luego de forma aditiva:

* todos los terminales de C146-C163;
* metricas completadas y faltantes;
* estimaciones y picos de recursos;
* cada prueba causal por operador y caso;
* decisiones de aislamiento y nombre;
* recibos de host con identificadores logicos.

La segunda carga debe ser idempotente. Historia previa intacta, backlog cero o
deficit exacto.

### C165 - POST y retorno

El POST debe leer los tips finales y los roots reales, no objetos fabricados en
el mismo proceso. Incluya:

* reproduccion de todos los PRE;
* bateria causal exhaustiva y sus mutantes;
* peor-caso de memoria y mutante sin preflight;
* conteos fisicos por dataset;
* perfil y SNR con recibos finales;
* OLAP antes/despues e idempotencia;
* salud final de los tres roles y procesos que quedan ejecutandose.

Reporte faltas propias al inicio. Detengase si aparece cualquier diferencia de
prefijo, uso del futuro, OOM del host, recibo ausente o dato no ligado.

## 6. Entregables

* PRE y POST ejecutables con salidas capturadas;
* codigo, tests, mutaciones y contratos C146-C165;
* roots write-once fuera de Git, con manifests publicos sanitizados;
* recibos de ejecucion distribuida;
* matriz de cobertura D0-D2;
* packet unico de retorno para Musashi.

La conclusion permitida es mecanica y acotada. Esta orden no concede
elegibilidad, seleccion, entrenamiento ni uso en trading.
