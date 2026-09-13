# Orden de Musashi a General Satoshi: C122-C145 DATA FOUNDATION

**Fecha:** 2026-09-12
**Prioridad:** P0 datos antes de modelos
**Licencia:** `CPU_DATA_RECONNAISSANCE_AND_SYNTHETIC_CALIBRATION`
**Auditoria rectora:** `MUSASHI_AUDIT_C106_C121_DATA_READINESS_2026_09_12.md`

## 0. Objetivo y alto

Construir evidencia de D0-D2: contratos, perfiles crudos, muestreo, ruido,
SNR y denoising causal por variable y por regimen. No seleccionar features ni
entrenar modelos de pronostico/RL.

Detenerse en:

`DATA_FOUNDATION_D0_D2_READY_FOR_EXTERNAL_REVIEW`

## 1. Bases y preservacion

Parta de los tips empujados del retorno C106-C121. Trabaje en ramas nuevas.
Preserve byte a byte:

* T2 submissions v1-v4, closure y addendum;
* B4 y su cierre;
* terminales v1-v4 y caracterizacion del sucesor;
* DAG v1-v4, contratos temporales y masks existentes;
* disenos por variable v1-v5 y poblaciones;
* filas historicas del OLAP.

Permitido:

* CPU offline;
* descargar hasta 2 GiB de datos publicos desde fuentes oficiales sin
  credenciales, conservando licencia, URL, cita, bytes y SHA-256;
* usar datos publicos ya custodiados;
* generar senales sinteticas de verdad conocida;
* ampliar de forma aditiva el OLAP tras ensayo en base desechable.

Prohibido:

* GPU, entrenamiento neuronal, scoring confirmatorio, feature selection,
  optimizacion de hiperparametros, RL o DOIN;
* target-aware screening, test-set tuning o lectura de periodos sellados;
* live, venue, MT5 o acciones de cuenta;
* declarar una variable u operador `PUBLICLY_ELIGIBLE`;
* borrar resultados historicos;
* tratar `Protocol closed` como evidencia ejecutada.

## 2. Adjudicaciones del retorno

### C122 - T2 como bundle

Registre la decision:

`T2_V4_CLOSURE_AND_HISTORY_ADDENDUM_ACCEPTED_AS_ONE_NON_AUTHORIZING_BUNDLE`

El bundle liga `1.3125` historico como invalido y `1.0` como valor corregido.
No reemita la submission ni cambie codigo T2. El operador evaluado conserva
`DOES_NOT_ADVANCE` y no entra al nuevo banco como elegible.

### C123 - sucesor ETH como piloto

Registre:

`ETH_SUCCESSOR_MECHANICS_ACCEPTED_NOT_A_SCIENTIFIC_BANK`

Sus 89 variables pueden servir para pruebas mecanicas. No cuentan para el
banco hasta resolver unidad, licencia y el resto de condiciones por miembro.
No se solicita al owner inventar una licencia.

### C124 - diseno v5

Registre la implementacion del join como aceptada y su poblacion como
`BANK_INSUFFICIENT`. Scoring permanece cerrado.

## 3. PRE: demostrar lo que aun no existe

### C125 - matriz de estado ejecutable

Antes de implementar, produzca un PRE que derive desde artefactos, no desde
prosa, para cada STEP 01-13:

* protocolo presente;
* implementacion localizable;
* experimento ejecutado;
* alcance de la evidencia;
* decision de elegibilidad.

Debe reproducir, como minimo:

* cero variables elegibles en v5;
* cero operadores de denoising publicamente elegibles;
* ausencia de una matriz ejecutada para STEP 04-13;
* que el set de siete operadores de v5 no representa los 13 pasos;
* que una estadistica numerica no concede semantica ni licencia.

## 4. D0: bancos y contratos

### C126 - banco publico multivariado

Construya un inventario de datasets publicos multivariados y paneles de
sensores/series alineadas. No use labels o scores para escogerlos.

Cada candidato debe declarar:

* fuente oficial, dataset, version y citation;
* licencia verificable y restricciones de redistribucion;
* frecuencia, rango, zona horaria y significado del timestamp;
* variables, unidades y productores;
* independencia/dependencia respecto de otros paneles;
* bytes y SHA-256 por archivo.

Use primero Traffic, Electricity, Weather, ETT, Solar, Air Quality u otros
artefactos ya custodiados cuando su procedencia lo permita. Los nombres son
candidatos, no una obligacion de forzar seis paneles. Series no alineadas no
se convierten en un panel multivariado por estar en el mismo archivo.

### C127 - banco financiero

No reprocese 14 GiB sin necesidad. Desde el censo incremental de 1,965
variables, derive una cola estratificada por productor, frecuencia, familia de
feature y consumidor real. Incluya primero las variables con bytes presentes y
lineage verificable. Las licencias o terminos de uso ambiguos quedan
`INTERNAL_RESEARCH_ONLY_PENDING_EVIDENCE`, no publicos ni elegibles.

### C128 - banco sintetico

Versione generadores con componente limpio y perturbacion separadas. Incluya,
como minimo:

* senoides y sumas multibanda con amplitud/fase conocidas;
* tendencia, estacionalidad y chirps;
* impulsos, escalones, bumps, motifs y cambios de regimen;
* ruido blanco gaussiano, coloreado, impulsivo, correlacionado entre variables
  y heteroscedastico;
* missingness MCAR y bloques ausentes;
* controles de senal nula y ruido nulo.

Los seeds, parametros, clean, noise y observed deben quedar separados y
reconstruibles.

### C129 - contrato comun

Implemente un contrato comun de dataset/variable que cubra identidad, fuente,
licencia, semantica, unidad, frecuencia, evento, disponibilidad, missingness,
sentinels, particiones y digest. Los adaptadores de cada banco deben producir
ese contrato sin perder sus campos originales.

## 5. D1: perfil crudo sin target

### C130 - perfil univariado por particion

Para cada variable admitida materialice, por train/calibration/confirmation
sin mezclar particiones:

* n, cobertura, missing, duplicados, no finitos, constantes y cardinalidad;
* cuantiles robustos, MAD/IQR, rango y colas;
* autocorrelacion en lags predeclarados y tiempo de correlacion;
* estacionariedad y cambio de distribucion con supuestos declarados;
* outliers como diagnostico, nunca borrado automatico;
* costo y version de cada calculo.

### C131 - perfil de informacion y compresion

Agregue descriptores cuyo estimador sea completamente declarado:

* entropia discreta bajo cuantizacion train-frozen;
* entropia de permutacion;
* entropia espectral en ventanas causales;
* longitud comprimida por muestra con compresor y nivel fijados;
* ganancia de compresion frente a bytes crudos y frente a permutacion;
* effective rank solo para matrices con definicion publicada;
* redundancia condicional y surprisal cuando la muestra lo permita.

No denomine estos valores inteligencia, informacion libre de ruido ni
complejidad de Kolmogorov observada. Son cotas/descriptores operacionales.

### C132 - perfil multivariado y agrupacion

Dentro de training y sobre muestras temporalmente comunes, calcule:

* Pearson y Spearman robustos;
* cross-correlation causal y estabilidad del lead/lag;
* coherencia espectral con intervalos o controles nulos;
* redundancia lineal y no lineal con estimador declarado;
* PCA/effective rank como diagnostico, no transformacion elegible;
* clusters estables por origen y bootstrap;
* componentes common/private y residuos solo como candidatos.

Nunca alinee usando informacion futura. Un grupo inestable queda
`GROUP_NOT_IDENTIFIED`.

### C133 - sampling y aliasing

Ejecute STEP 01 de forma real:

* frecuencia nominal y observada;
* jitter, gaps, barras truncadas y cobertura;
* limite de Nyquist solo para tramos regulares;
* energia cercana a Nyquist y sensibilidad al downsampling;
* control sintetico o fuente de mayor frecuencia para distinguir aliasing de
  simple energia espectral.

No afirme aliasing solo por observar un pico de frecuencia.

## 6. D2: ruido, SNR y denoising causal

### C134 - estimadores de ruido y SNR

Sobre el banco sintetico, mida error y calibracion de cada estimador contra la
verdad conocida por familia de senal, ruido, SNR y longitud. Incluya sesgo,
RMSE, cobertura e identificabilidad. Sobre datos reales publique solo
`MODEL_CONDITIONAL_SNR_ESTIMATE` con el modelo de descomposicion nombrado.

### C135 - contrato de operador causal

Cada operador debe implementar `fit/transform` y continuacion incremental con:

* ajuste train-only;
* salida batch e incremental equivalentes;
* estado durable y reinicio reproducible;
* cero acceso futuro;
* warm-up y muestras no disponibles tipados;
* costo CPU y memoria;
* retardo algoritmico, retardo de grupo/fase cuando aplique y latencia de
  computo medidos por separado.

"Sin delay" significa cero look-ahead. No esconda el retardo fisico de un
filtro causal ni lo compense desplazando con muestras futuras.

### C136 - banco de operadores

Incluya identidad y familias pequenas predeclaradas de:

* EWMA;
* estado-espacio/Kalman;
* mediana y estimadores robustos trailing;
* FIR/IIR causales;
* wavelet o multiescala causal solo si su frontera temporal es demostrable;
* descomposicion tendencia/estacional/residuo causal;
* operadores rechazados previamente como controles negativos.

No ejecute un barrido ilimitado. Parametros se calibran en sintetico y una
familia que no identifica sus supuestos se abstiene.

### C137 - preservacion y destruccion de senal

Para cada operador y regimen compare `X`, `D(X)` y `R=X-D(X)` contra clean:

* mejora SNR y error de reconstruccion;
* fase, amplitud y retardo;
* preservacion de impulsos, extremos, motifs y cambios de regimen;
* estructura restante y predictibilidad del residual;
* falsos positivos cuando clean=noise-free;
* degradacion cuando el operador no aplica.

Una mejora de RMSE que borra eventos no avanza.

### C138 - decisiones de laboratorio

Emita por operador y regimen solo:

* `LAB_CALIBRATED`;
* `REGIME_LIMITED`;
* `NOT_IDENTIFIABLE`;
* `LAB_REJECTED`.

Ninguno equivale a `PUBLICLY_ELIGIBLE`. Publique regiones de falla, no solo el
mejor caso.

## 7. OLAP y reproducibilidad

### C139 - esquema aditivo

En base desechable primero, agregue granos para:

* sampling/temporal quality;
* variable profile por particion;
* information/compression metric;
* pair/group relation;
* operator run y parametros;
* raw/denoised/residual metrics;
* retardo y costo;
* decision de laboratorio.

Cada fila debe ligar dataset, bytes, variable, particion, codigo, estimador y
run. Cargue exito, fallo, rechazo e inconcluso. El loader real debe seguir
activo, idempotente y con backlog visible. No limpie historia.

### C140 - cobertura y recibos

Publique una matriz dataset x variable x metrica x operador que distinga
`NOT_RUN`, `UNAVAILABLE`, `FAILED`, `INCONCLUSIVE` y resultado. Ningun conteo
agregado sustituye al ledger miembro a miembro.

## 8. Sucesores obligatorios, sin ejecutarlos aun

### C141 - D3

Materialice un design, sin scores, para STEP 04-07: cuantizacion/companding,
source coding/MDL, representaciones tiempo-frecuencia y detectores. Debe
consumir solo salidas revisadas de D2 y conservar raw como control.

### C142 - D4

Materialice un design, sin scores, para STEP 08-10: ecualizacion,
common/private, cancelacion de redundancia y alineacion causal. Debe declarar
que la disponibilidad temporal precede a cualquier correccion de lead/lag.

### C143 - D5

Materialice un design, sin scores, para STEP 11-13: redundancia controlada,
routing adaptativo y asignacion multirama. Presupuesto y abstencion deben ser
parte del contrato.

### C144 - feature selection permanece despues

Actualice el work plan para que I5 no pueda consumir variables o
transformaciones sin estados revisados D0-D4. No ejecute el selector v5 ni
escoja targets.

## 9. Cierre

### C145 - packet

El retorno debe incluir:

1. confesiones antes de resultados;
2. PRE/POST y mutaciones por frontera;
3. matriz honesta STEP 01-13;
4. bancos y contratos, con deficit exacto;
5. cobertura del perfil individual y multivariado;
6. calibracion SNR y decisiones por operador/regimen;
7. retardo, preservacion de eventos y costos;
8. estado OLAP, backlog y conteos;
9. designs sucesores D3-D5;
10. ramas, tips y digests;
11. linea cero para GPU, training, score, selection, RL, DOIN, live y venue.

No cree records de Musashi ni licencia de consumo.
