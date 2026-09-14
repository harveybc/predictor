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
| I3 calibracion sintetica | Evidencia candidata D2, sin revision | reanalisis C137 con la API actual (513 unidades, 2 cambios de decision) y confirmacion fresca en 3.972 unidades con semillas no vistas; denoising: 51 aprobaciones y 7 limitadas por regimen de candidatos en 2.052 regimen-operador; SNR: 39 calibraciones por regimen en 1.026, ningun estimador general | revision de Musashi de la evidencia fresca |
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
| 01 muestreo/Nyquist | Escrito | Comun bajo contratos | Perfil acotado de 715 datasets | Pendiente de revision por variable/regimen |
| 02 ruido/SNR | Escrito | Comun y acotado | 513 unidades; resultado descriptivo y sesgo dependiente del regimen | Ningun estimador general |
| 03 denoising | Escrito | API causal sin interruptores; 17 mutantes estructurales detectados; auditoria wavelet por unidad (7.312 auditorias, 0 fallos) | Confirmacion fresca: 12 especificaciones candidatas aprueban en 1 a 11 regimenes cada una, siempre sin datos faltantes; 871 decisiones sin potencia suficiente; el oraculo se detecto en las 3.908 unidades evaluadas | Ninguna hasta la revision externa; la reja D2 sigue rehusando |
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

La reparacion C146-C165 completo 715 de 715 datasets sin OOM del host. El peor
caso observo 1,56 GiB bajo un limite de 3,77 GiB. La campana SNR tambien termino
y el cubo recibio sus resultados de forma aditiva.

La orden C166-C184 cerro las cuatro fronteras que quedaban:

* **Guardas.** Ninguna guarda causal puede apagarse desde produccion: no hay
  tabla, variable de entorno ni argumento que la omita. Cada una de las 17
  guardas fue retirada por separado en un proceso aislado, y las 17 mutaciones
  fueron detectadas.
* **Identidad y cobertura.** Cada bloque ADF/KPSS tiene identidad propia
  (257.884 estimaciones distintas, antes 255.786). La cobertura separa
  `REFUSED`, `FAILED`, `NOT_APPLICABLE` y `NOT_RUN`.
* **Paridad PCA.** Se cumple bajo una tolerancia declarada antes de medir y una
  representacion canonica, en tres roles y dos pilas de algebra lineal.
* **C137.** Se re-ejecuto con la API actual como reanalisis no confirmatorio:
  mismas unidades y misma verdad, 2 de 2.565 decisiones cambian, y ninguna
  decision historica pasa la reja D2.

Durante la ejecucion hubo un defecto propio. El trabajador invalidaba el root
completo cuando la regla de datos faltantes rechazaba todos los brazos de una
unidad MCAR. Se corrigio y el diseno se re-sello antes de generar cualquier
unidad fresca.

Estado: `D0_D1_ACCEPTED_D2_CANDIDATE_EVIDENCE_PENDING_REVIEW`.

## 7. Proxima puerta

La evidencia D2 fresca queda como candidata:

* **Diseno.** Sellado antes de la reserva.
* **Semillas.** 3.972 unidades con semillas no vistas en C128-C163.
* **Ejecucion.** Distribuida por memoria en los tres roles, sin OOM de host y
  sin invalidacion.
* **Decisiones.** 3.591, ninguna revisada externamente: en denoising, 51
  aprobaciones y 7 limitadas por regimen de candidatos; en SNR, 39 calibraciones
  por regimen y ningun estimador general.
* **Cubo.** Cargadas de forma aditiva con la historia intacta.

Una submission liga diseno, codigo, cinta, raices, decisiones y carga; no
concede nada. La reja de consumo sigue rehusando todo sujeto sin registro de
revision externa.

La orden se detiene en `D2_CURRENT_API_FRESH_CONFIRMATION_READY_FOR_MUSASHI_REVIEW`.
D3-D5, seleccion, modelos, RL, DOIN y live siguen cerrados hasta la decision
separada de Musashi.

## 8. Gobernanza Flow v3 y soporte de D2 (2026-09-14)

La revision de Musashi (`MUSASHI_REVIEW_C166_C184_AND_DATA_GOV_2026_09_13.md`,
`REVISE_D2_ADJUDICATION_BEFORE_CONSUMPTION`) encontro que el adjudicador contaba
como valida una semilla con cualquier metrica y que las ausencias pasaban los
chequeos. Estado tras las ordenes GOV-N1..N8 y D2-R1..R8:

* **Flow v3** (transporte y contabilidad de datos): integrado con lake/warehouse,
  contratos con alcance ejecutable (`availability`: etiqueta, cota de finalizacion,
  evidencia de zona horaria, clase de uso), disposicion de terminales rechazados,
  consumidores de predictor, preprocessor y feature-eng probados en stack
  desechable. Los servicios productivos siguen con codigo anterior: el reinicio
  esta bloqueado en el operador (N3). Nada de esto prueba causalidad del
  preprocesamiento ni calidad de modelo.
* **Soporte D2 (R1-R2):** contrato de soporte declarado como ocho pruebas antes
  de la reparacion (PRE 10/10 fallando, POST 26/26); el adjudicador reparado exige
  semillas COMPLETAS, deriva aplicabilidad del contrato, publica
  planificadas/observadas/completas/inaplicables y nunca deja pasar una ausencia.
  Vista previa NO gobernante sobre las 3.591 decisiones: 138 cambian; los cinco
  casos revisados y dos controles de identidad pierden su pase
  (`NOT_IDENTIFIABLE`, semillas completas < diseno); ninguna decision gana un
  pase. Universo: 0 filas faltantes; 1 desacuerdo de eventos, en el control
  oraculo no causal al borde de ventana.
* **R3 (re-adjudicacion gobernada):** pendiente del micro-run productivo
  reconciliado. **R4:** margenes de las 1.026 decisiones SNR, inventario de
  entorno del coordinador y subconjunto diagnostico (16 regimenes, 32 unidades)
  congelados por regla; sin ejecutar; AT9 sigue abierto con su tolerancia
  original. **R6:** vistas de cobertura vigente/historica propuestas y ensayadas
  en base desechable, sin aplicar. **R7:** diseno D3 (documento 07) sin ejecutar.
  **N7:** plan de integracion agent-multi/DOIN/live (documento 08).

Estado: `D2_SUPPORT_READJUDICATED_UNDER_GOVERNANCE_R4_REPLAY_PENDING`.

## 9. Cierre de R3 sobre los servicios productivos (2026-09-14)

Musashi ejecuto el reinicio acotado y el micro-run reconciliado (acta
`docs/handoffs/MUSASHI_FLOW_V3_PRODUCTION_RESTART_COMPLETED_2026_09_14.md`), de modo
que el prerrequisito operativo de R3 dejo de estar pendiente. La re-adjudicacion
se registro entonces como campana gobernada contra los servicios activos, sin
reiniciarlos: terminal COMPLETED, conciliacion exacta y filas sucesoras cargadas
de forma aditiva bajo un run id nuevo, con la historia intacta. De las 3.591
decisiones, 138 cambian y ninguna gana un pase; siete lo pierden. Los detalles y
la regla de operacion vigente estan en el documento
`09_INCORPORACION_ORDENES_ACTUALIZADAS_2026_09_14.md`. Sigue pendiente el replay
diagnostico de R4; AT9 permanece abierto con su tolerancia original.
