# Auditoria Musashi: ronda 7 C87-C105

**Fecha:** 2026-09-12
**Retorno:** `SATOSHI_C87_C105_RETURN_PACKET_2026_09_12.md`
**Tip revisado de predictor:** `6a0dce448036a77d9e2b3c1c2037f5457b63f8d7`
**Disposicion global:** `ACCEPT_PARTIAL / REVISE_DESIGN`
**GPU, entrenamiento, confirmacion y live:** no autorizados

## 1. Disposiciones

| Frente | Disposicion |
|---|---|
| T2 | `CORRECTED_EXPECTATION_ACCEPTED_FOR_READ_ONLY_READJUDICATION` |
| B4 | `V4_ACCEPTED_AS_NON_AUTHORIZING_FINAL_CLOSURE` |
| terminales v4 | `ACCEPTED_WITH_PHYSICAL_AND_STATISTICAL_SCOPE` |
| `FEATURE_DAG.v4` | `ACCEPTED_PROSPECTIVE_SUCCESSOR_ONLY` |
| contrato temporal v2 | `ACCEPTED_FOR_SUCCESSOR_BINDING_WORK` |
| diseno por variable v4 | `REVISE_NO_LICENSE` |

No queda una decision del propietario. Las correcciones y el siguiente
trabajo son offline y pueden ejecutarse por Satoshi.

## 2. T2: el valor correcto es 1.0 y no es ajuste post-resultado

El alto de Satoshi fue correcto. Una diferencia debia detener la
publicacion. La diferencia, sin embargo, no abre una pregunta estadistica
nueva:

* `1.3125` no pertenece al intervalo de una probabilidad;
* para `k=3`, `n=6` y nula Bernoulli 0.5, la prueba bilateral exacta
  predeclarada es `min(1, 2 * min(P(X<=k), P(X>=k))) = 1.0`;
* la tabla exacta para `k=0..6` es
  `1/32, 7/32, 11/16, 1, 11/16, 7/32, 1/32`.

La cronologia es decisiva:

1. El codigo que declara esa formula y la tabla fue comprometido en T2
   como `5fb2849eb7dc2723d00db1e4f365fbba19a35672` a las 03:20:07 -05.
2. El diseno publico que declara `1.3125` superado por `1.0` fue
   comprometido en predictor como
   `86ca8ca88b8d1f43c37c8a262e39182ee1a4d772` a las 09:53:09 -05.
3. El replay endurecido de esta ronda se comprometio despues, en
   `4d0b315952838b93ec207e2d1faa1655a47183b5` a las 18:25:50 -05.

Por tanto, la regla que produce `1.0` existia antes del replay. El error
de ronda 7 esta en la construccion de la expectativa: el modo template de
`t2_hardened_readjudicate.py` calculo el digest con el
`screen_adjudication` historico, que todavia contiene `1.3125`, aunque al
mismo tiempo agrego el bloque que lo declara superado. El replay
endurecido, en cambio, recalculo el campo y obtuvo `1.0`.

Tome la evidencia historica inmutable, conserve todos sus campos y
reemplace solo el campo matematicamente invalido mediante la formula ya
publicada. El digest resultante es
`1be80a0ab6d091794f7ce3ec97c2dbf88920911037c0415c383043bd95ca3a6d`,
exactamente el digest producido por el replay endurecido.

Instale el record externo
`MUSASHI_T2_HARDENED_READJUDICATION_REVIEW_RECORD.json` para el tip
`4bf38b6f4ba2a2e3e97f331f52619757d33be538`, con SHA-256
`c75d8d6ef4f1a1af67768d8c53507b231c316d9373b9d8dc52c22a0bdfdff1ff`.
El record concede exclusivamente la readjudicacion de solo lectura:
reentrenamiento, descargas, ejecucion de modelos, promocion y cualquier
permiso de ejecucion permanecen en `false`.

## 3. B4: cierre aceptado

Recompare v3 con v4 y ejecute las baterias focales. Coinciden:

* generacion de campana;
* digest del ledger;
* conteos `2 COMPLETED_VERIFIED / 1 QUARANTINED_PARTIAL / 9 NOT_STARTED`;
* desenlace `SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT`;
* costos de las dos celdas completas;
* digests de terminal y serie por barra de ambas celdas completas.

El digest de inventario de la celda parcial cambia porque v4 incorpora
los nuevos bindings de componente. No cambia su clase ni se consume como
resultado. La submission v4 queda aceptada con SHA-256 de archivo
`63ea37901c8756cd9669d18548176d337efeb08f1bb457a3e395eb90c7be7317`.

B4 termina aqui. No hay base para relanzar, completar por etiqueta,
comparar modelos ni gastar mas GPU en esta poblacion.

## 4. Terminales, DAG y tiempo

### Terminales v4

Acepto la poblacion y la recomputacion con este limite semantico:

* 1,501 variables tienen descriptores numericos recomputados desde los
  bytes ligados;
* eso no significa que 1,501 variables tengan semantica, unidad, rol o
  licencia verificados;
* cuatro columnas de fecha almacenadas como enteros quedan
  `SEMANTICALLY_UNRESOLVED` y ningun valor numerico suyo es elegible;
* las 460 variables no identificables conservan autoridad del productor,
  no autoridad de recomputacion independiente.

El cubo puede conservar las seis capas de evidencia. Ninguna capa, por si
sola, licencia una variable para un experimento.

### `FEATURE_DAG.v4`

Acepto los 89 `CAUSAL_ACTIVE` solo para el dataset sucesor producido por
el rerun prospectivo. El dataset historico conserva cero. La igualdad de
valores con el historico no traslada procedencia ni identidad.

### Contrato temporal v2

Acepto la separacion entre disponibilidad, completitud, regularidad y
elegibilidad. La cuenta correcta es:

* 20 barras truncadas dentro de las 18,085 filas del dataset sucesor;
* 21 en el origen completo, porque una queda antes de la primera fila;
* 8 gaps dentro del dataset;
* 10,282 muestras elegibles y 7,803 no elegibles bajo el contrato actual.

El contrato v2 aun nombra el dataset historico. Debe producirse un
contrato nuevo ligado a los bytes y al identificador del sucesor antes de
que sus 89 columnas puedan entrar al banco.

## 5. Hallazgo P0 nuevo: el derivador no implementa la interseccion

La prosa de v4 exige que cada miembro cumpla simultaneamente terminal,
semantica, lineage, temporalidad, licencia y missing policy. La funcion
`derive_population()` no ejecuta esa interseccion:

1. cuenta los terminales recomputados, pero no los une con las columnas;
2. cuenta las variables semanticamente declaradas, pero tampoco las une;
3. forma `panels` usando solo columnas `CAUSAL_ACTIVE` y la presencia de
   un contrato temporal para el dataset;
4. asigna `members` como la suma de esas columnas activas.

Contraejemplo ejecutado: seis datasets con cinco columnas activas y
contrato temporal, pero cero terminales y cero variables semanticamente
declaradas, producen:

```text
verdict=BANK_SUFFICIENT_FOR_REVIEW panels=6 members=30
terminals_independently_recomputed=0 semantic_variables=0
```

Esto podria abrir el screen cuando se agregue el sexto panel aunque
ninguna variable satisfaga el contrato. El diseno v4 no recibe licencia.

La correccion debe hacer un join exacto por identidad de dataset y
variable/columna. Una cardinalidad agregada nunca reemplaza la
correspondencia miembro a miembro.

## 6. Verificacion independiente

Ejecute en CPU:

* predictor: 54 focales verdes;
* B4: 82 focales verdes;
* T2: 77 focales verdes;
* financial-data: 144 focales verdes.

La GPU esta disponible, sin proceso de computo, pero permanece fuera de
alcance. El cargador OLAP esta `active/running`, con cero reinicios. No se
reinicio ningun servicio.

La reproduccion adicional queda en
`docs/audits/evidence/repro_runs/musashi_round7_additional_pre_2026_09_12.py`;
su salida capturada queda en el archivo homonimo con extension `.out`.

## 7. Siguiente paso

Satoshi debe consumir el record T2 ya instalado, ejecutar una sola
readjudicacion de solo lectura y publicar la submission v4 si y solo si
el digest coincide. En paralelo puede corregir el join de poblacion,
ligar el contrato temporal al sucesor, construir su censo semantico y
preparar el banco de seis paneles. La orden completa es C106-C121.
