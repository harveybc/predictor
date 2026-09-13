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
| I2 perfil de calidad e informacion | Piloto | descriptores fisicos historicos y sucesor ETH | perfil por variable, particion y grupo |
| I3 calibracion sintetica | Parcial | EWMA/Kalman/mediana en banco limitado | familias de ruido y operadores ampliadas |
| I4 utilidad publica | Un resultado negativo | EWMA `DOES_NOT_ADVANCE` en seis paneles T2 | probar solo operadores calibrados nuevos |
| I5 seleccion | Bloqueada | diseno v5, poblacion cero | minimo seis paneles y 30 variables elegibles |
| I6 pronostico | Bloqueada | sin contrato nuevo | salida revisada de I5 |
| I7 representaciones | Bloqueada | protocolos solamente | evidencia I2-I6 |
| I8 L2/DOIN | Bloqueada | sin universo L1 elegible | manifest congelado |
| I9 RL/trading offline | Bloqueada para entradas nuevas | campanas anteriores cerradas por su identidad | evidencia atribuible de I8 |
| I10 financiero/live | Bloqueada | piloto temporal, no licencia operativa | revalidacion separada |

## 2. Estado STEP 01-13

| Paso | Protocolo | Implementacion | Evidencia | Elegibilidad |
|---|---|---|---|---|
| 01 muestreo/Nyquist | Escrito | Parcial en contratos temporales | Un dataset piloto | No |
| 02 ruido/SNR | Escrito | Parcial | Sintetica limitada | No |
| 03 denoising | Escrito | Parcial | EWMA negativo en T2; laboratorio limitado | Ningun operador |
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
D0 contratos e inventario
 -> D1 perfil crudo por variable y por grupo
 -> D2 muestreo + ruido/SNR + denoising causal
 -> D3 cuantizacion + compresion + tiempo/frecuencia + detectores
 -> D4 ecualizacion + redundancia + sincronizacion
 -> D5 robustez + routing + asignacion multirama
 -> I5 seleccion de variables
 -> I6 pronostico supervisado
 -> I7-I9 representaciones, L2/DOIN y RL
 -> I10 validacion financiera/live
```

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
  operadores elegibles.
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

Estado: `D1_D2_RELAUNCH_BLOCKED_BY_MEMORY_AND_CAUSALITY_AUDIT`.

## 7. Proxima puerta

La orden C146-C165 corrige memoria y causalidad, completa D1-D2 con recibos
finales y deja D3-D5 como sucesores obligatorios. No abre seleccion ni
entrenamiento.
