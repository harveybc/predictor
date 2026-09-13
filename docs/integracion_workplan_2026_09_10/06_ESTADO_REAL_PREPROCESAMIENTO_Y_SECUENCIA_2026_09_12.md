# Estado real del preprocesamiento y secuencia vinculante

**Fecha:** 2026-09-13
**Estado:** `AUTHORITATIVE_STATUS_SUPPLEMENT`

Este suplemento evita confundir cuatro estados distintos:

1. `PROTOCOL_WRITTEN`: existe una especificacion falsable.
2. `IMPLEMENTED_MECHANICS`: existe codigo y pasa pruebas mecanicas.
3. `EVIDENCE_OBTAINED`: el experimento se ejecuto en el alcance declarado.
4. `ELIGIBLE`: una revision permite consumir el resultado en la siguiente fase.

Ninguno implica automaticamente el siguiente.

## 1. Estado CRISP-DM I0-I10

| Fase | Estado | Evidencia actual | Puerta siguiente |
|---|---|---|---|
| I0 decisiones y roles | Parcial | contratos de campanas cerradas | contrato por nuevo banco y uso |
| I1 inventarios | En curso | lago financiero censado; banco publico sin panel elegible | datasets con licencia, semantica y tiempo |
| I2 perfil de calidad e informacion | Evidencia obtenida en D1 | 715 datasets, 866.661 filas de perfil, ejecucion acotada por memoria | corregir identidad de bloques y cobertura v2 |
| I3 calibracion sintetica | Mecanica actual; evidencia historica no consumible | SNR C163 descriptivo; C137 usa la semantica anterior | reanalisis de migracion y confirmacion en semillas frescas |
| I4 utilidad publica | Un resultado negativo | EWMA `DOES_NOT_ADVANCE` en seis paneles T2 | probar solo operadores calibrados nuevos |
| F0-F5 ingenieria de caracteristicas | Descubrimiento completo; requisitos en revision | auditoria de `feature-eng` y matriz FBR-001..012 | cerrar casos y pruebas antes de implementar |
| I5 seleccion | Bloqueada | diseno v5, poblacion cero | minimo seis paneles y 30 variables elegibles |
| I6 pronostico | Bloqueada | sin contrato nuevo | salida revisada de I5 |
| I7 representaciones | Bloqueada | protocolos solamente | evidencia I2-I6 |
| I8 L2/DOIN | Bloqueada | sin universo L1 elegible | manifest congelado |
| I9 RL/trading offline | Bloqueada para entradas nuevas | campanas anteriores cerradas por su identidad | evidencia atribuible de I8 |
| I10 financiero/live | Bloqueada | piloto temporal, no licencia operativa | revalidacion separada |
| Gobernanza data-gov | Implementacion beta bajo auditoria | descarga con hash y reporte nominal de exitos | cerrar registro de campana, entrega confirmada, todos los terminales, outbox y reconciliacion |

## 2. Estado STEP 01-13

| Paso | Protocolo | Implementacion | Evidencia | Elegibilidad |
|---|---|---|---|---|
| 01 muestreo/Nyquist | Escrito | Comun bajo contratos | Perfil acotado de 715 datasets | Pendiente de revision por variable/regimen |
| 02 ruido/SNR | Escrito | Comun y acotado | 513 unidades; resultado descriptivo y sesgo dependiente del regimen | Ningun estimador general |
| 03 denoising | Escrito | API causal y bateria exhaustiva; requiere quitar switches de mutacion | C137 historico no compatible con la API actual | Ningun operador vigente |
| 04 cuantizacion | Escrito | No comun | No | No |
| 05 entropia/compresion | Escrito | No comun | No | No |
| 06 tiempo/frecuencia | Escrito | Codigo historico disperso | No bajo contrato nuevo | No |
| 07 deteccion | Escrito | No como banco comun | No | No |
| 08 ecualizacion | Escrito como plan | No | No | No |
| 09 redundancia/crosstalk | Escrito como plan | No | No | No |
| 10 sincronizacion | Escrito como plan | No | No | No |
| 11 redundancia controlada | Escrito como plan | No | No | No |
| 12 routing adaptativo | Escrito como plan | No | No | No |
| 13 asignacion multirama | Escrito como plan | No | No | No |

## 3. Orden vinculante

```text
G0 submission gobernada de campana
 -> D0 contratos e inventario
 -> D1 perfil crudo por variable y por grupo
 -> D2 muestreo + ruido/SNR + denoising causal
 -> [D3-D5 procesamiento de senal || F0-F5 ingenieria de caracteristicas]
 -> union de manifests elegibles
 -> I5 seleccion de variables
 -> I6 pronostico supervisado
 -> I7-I9 representaciones, L2/DOIN y RL
 -> I10 validacion financiera/live
 -> G5 reconciliacion de terminales, recibos y OLAP
```

`G0-G5` envuelve la secuencia completa. No reemplaza las rejas cientificas y
no hace llamadas dentro del ciclo de aprendizaje. Toda ejecucion que pueda
cambiar una decision es `GOVERNING`; pruebas y sondas mecanicas pueden ser
`NON_GOVERNING`, pero no se convierten en evidencia sin repeticion gobernada.

Una fase puede terminar con todos los operadores rechazados. Eso no permite
saltar a la siguiente usando entradas no caracterizadas.

## 4. Principios de tratamiento de senal

* Se conserva siempre una rama cruda. Una transformacion no reemplaza el dato
  original antes de demostrar utilidad.
* "Causal" significa que la salida en `t` usa solo informacion disponible en
  `t`; no significa retraso cero. Se mide retardo de grupo, respuesta al
  impulso, warm-up y latencia de computo.
* La compensacion que necesite muestras futuras queda fuera de produccion.
* SNR real es dependiente del modelo de senal y ruido. Se publican estimador,
  supuestos, error contra verdad sintetica y sensibilidad; nunca un numero
  absoluto sin modelo.
* Todo ajuste ocurre en training. Validation decide y test confirma.
* Las relaciones multivariadas se estiman despues de declarar disponibilidad
  y dentro de la particion permitida.
* Los targets no participan en inventario, perfil crudo ni seleccion del banco.
* Feature selection comienza despues de D0-D4 y solo consume variables y
  operadores elegibles. Tambien exige F5 para cualquier feature derivada.
* (C144) Esa regla es ejecutable. `tools/df_consumption_gate.py` rechaza
  cualquier variable u operador sin registros de revision externa para cada
  etapa D0, D1, D2, D3 y D4. En D2 solo se aceptan `LAB_CALIBRATED`, o
  `REGIME_LIMITED` dentro de sus regimenes.
* Un registro del productor no cuenta como revision, y la reja nunca concede
  `PUBLICLY_ELIGIBLE`.
* Sin esos registros el selector v5 no se ejecuta y no se escogen targets.

## 5. OLAP

El cubo es memoria append-only. Debe recibir:

* identidad y contrato del dataset;
* perfil por variable, particion y grupo;
* diagnosticos de muestreo, ruido e informacion;
* identidad, parametros, retardo, costo y salida de cada operador;
* estados `COMPLETED`, `FAILED`, `INCONCLUSIVE`, `REFUSED` y `REJECTED`;
* comparaciones raw/transformada/residual;
* recibo de cada carga.

Las campanas nuevas escriben mediante data-gov y un outbox durable. El cliente
no recibe credenciales del cubo. Una transferencia solo cuenta para linaje
despues de que el cliente verifica los bytes; una caida temporal conserva el
terminal como pendiente. El contrato y los bloqueadores de beta estan en
`08_GOBERNANZA_TRANSVERSAL_DATA_GOV_2026_09_13.md`.

No se borran resultados viejos para hacer espacio. Si se requiere particion o
archivo historico, se conserva la identidad y la consulta unificada.

## 6. Incidente de ejecucion C130-C134

El primer intento completo de perfiles y SNR no produjo recibos finales. El
perfilador lanzo ocho procesos y dos fueron terminados por OOM. La causa
dominante esta identificada: el ADF exacto sobre 13,253,761 muestras y 228
rezagos requiere aproximadamente 22.71 GiB solo para su matriz OLS. Una sola
tarea puede agotar el host; bajar el numero de workers no es correccion
suficiente.

Ademas, la causalidad local de los operadores no esta aun ligada a una
particion fisica: la API acepta una matriz arbitraria acompanada por la etiqueta
`train`. Los operadores wavelet y de descomposicion no pueden consumirse hasta
tener snapshots de fit/transform ligados a bytes, tiempo y particion, y superar
una prueba exhaustiva de invariancia de prefijo.

La reparacion C146-C165 completo 715 de 715 datasets sin OOM del host. El peor
caso observo 1,56 GiB bajo un limite de 3,77 GiB. La campana SNR tambien termino
y el cubo recibio sus resultados de forma aditiva.

Persisten cuatro fronteras antes de aceptar D2: los guards causales son mutables
desde produccion, C137 no se reejecuto con la API actual, la cobertura mezcla
rechazo, fallo e inaplicabilidad, y la paridad PCA es numerica pero no exacta.

Estado: `D0_D1_ACCEPTED_D2_REATTESTATION_REQUIRED`.

## 7. Proxima puerta

La orden C166-C184 elimina los interruptores de mutacion de produccion, corrige
identidad y cobertura, repite C137 como reanalisis y ejecuta una confirmacion
sintetica fresca. D3-D5, seleccion y entrenamiento permanecen cerrados hasta la
revision externa de ese resultado.

En paralelo se cierra la beta de data-gov. No se reinician los servicios vivos
ni se declara obligatorio el camino hasta que el ensayo en puertos alternos y
base desechable pase la auditoria. La autorizacion acotada del propietario para
reiniciar 5055-5057 ya esta registrada y solo aplica despues de esa reja.

En paralelo solo se permite trabajo de definicion F0-F3 bajo DGPD: requisitos,
casos de uso, pruebas y arquitectura. No se puntuan features ni se modifica una
campana cerrada. Estado ejecutable:
`FEATURE_BANK_METHOD_STATE.json`.
