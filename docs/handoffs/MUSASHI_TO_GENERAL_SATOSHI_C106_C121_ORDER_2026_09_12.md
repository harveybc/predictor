# Orden de Musashi a General Satoshi: C106-C121

**Fecha:** 2026-09-12
**Auditoria rectora:** `MUSASHI_AUDIT_ROUND7_C87_C105_2026_09_12.md`
**Prioridad:** P0 T2 y poblacion; P1 sucesor y banco
**Licencia:** `READ_ONLY_RECOMPUTATION_AND_CPU_MECHANICS`

## 0. Bases y fronteras

Trabaje en ramas nuevas desde estos tips empujados:

| Superficie | Tip |
|---|---|
| predictor | `6a0dce448036a77d9e2b3c1c2037f5457b63f8d7` |
| financial-data | `7d9862a331b98f3d07d1fd606c6963e1e5575dbe` |
| B4 | `ff52ca7d9b771978dfe60bfd15e351ffbe3d1d22` |
| T2 | `4bf38b6f4ba2a2e3e97f331f52619757d33be538` |

Preserve byte a byte:

* B4 v7 y submissions v1-v4;
* T2 original, su copia privada y submissions v1-v3;
* evidencia historica que contiene `1.3125`;
* divergence report de ronda 7;
* terminales v1-v4;
* `FEATURE_DAG.v1-v4`, manifests v1-v2;
* contratos temporales v1-v2 y su mask;
* disenos por variable v1-v4.

Permitido:

* CPU offline;
* una readjudicacion T2 de solo lectura bajo el record externo instalado;
* censo, contratos, manifests, validadores, fixtures y dry-runs sin labels;
* adquisicion publica limitada bajo C120.

Prohibido:

* GPU, entrenamiento, scores nuevos, confirmacion o promocion;
* live, venue, MT5, servicios de trading o acciones de cuenta;
* reabrir B4 o completar sus celdas;
* editar evidencia historica o el record externo de Musashi;
* emitir records en nombre de Musashi;
* relajar una regla para obtener una poblacion no vacia.

## 1. P0: cerrar T2 con la correccion preexistente

### C106 - congelar el PRE correcto

Antes de editar:

1. Reproduzca que el candidato de ronda 7 tiene `1.3125` y digest
   `a989d02d3673c04c8eaa7976050ddc0b7b3e7dec3e0e6a175f04a7ef02b004ca`.
2. Reproduzca que el replay endurecido produce `1.0` y digest
   `1be80a0ab6d091794f7ce3ec97c2dbf88920911037c0415c383043bd95ca3a6d`.
3. Pruebe por historia Git que la formula bilateral estaba comprometida
   en `5fb2849eb7dc2723d00db1e4f365fbba19a35672` antes de
   `4d0b315952838b93ec207e2d1faa1655a47183b5`.
4. Congele la falla del template: usa el `screen_adjudication` historico
   sin aplicar al propio campo la correccion que ya calcula al lado.

El PRE debe correr por API/CLI real. No basta buscar texto.

### C107 - consumir el record externo, no reconstruirlo

El record real ya esta instalado en su pathname fijo:

`MUSASHI_T2_HARDENED_READJUDICATION_REVIEW_RECORD.json`

SHA-256 esperado:

`c75d8d6ef4f1a1af67768d8c53507b231c316d9373b9d8dc52c22a0bdfdff1ff`

Debe verificar antes de imports:

* schema, revisor, decision, fecha y alcance exactos;
* tip endurecido `4bf38b6f4ba2a2e3e97f331f52619757d33be538`;
* tree y superficie completa;
* record historico, root preservado e inventario;
* expectativa corregida
  `1be80a0ab6d091794f7ce3ec97c2dbf88920911037c0415c383043bd95ca3a6d`;
* todos los grants en `false`.

No copie, regenere, chmod, sustituya ni reescriba el record. Si cualquier
hecho no coincide, `REFUSED` antes de abrir evidencia.

### C108 - replay unico de solo lectura

Ejecute desde un checkout desprendido, limpio y exacto de `4bf38b6f`.
Use la evidencia historica versionada
`T2_COMPLETION_RECONSTRUCTION_AND_SCREEN_ADJUDICATION_2026_09_10.json`
y el root preservado que liga el record.

Condiciones:

* cero entrenamiento y cero descarga;
* ningun entry point del executor puede escribir claims o unidades;
* 242 registros se verifican con el verificador endurecido;
* la semilla MLP omitida rehusa tipada, nunca `KeyError`;
* el checkout, el record y el inventario se revalidan antes y despues;
* intento unico. Una diferencia nueva vuelve a detener la publicacion.

### C109 - submission T2 v4

Solo si C108 coincide exactamente, publique en un commit B nuevo una
submission v4 que declare:

* `242 COMPLETED_VERIFIED / 0`;
* `DOES_NOT_ADVANCE`;
* estimando `-0.001048443391358884`;
* los seis efectos por panel, iguales a la evidencia historica;
* `signs_positive=3` y `sign_test_exact_p_two_sided=1.0`;
* historia preservada: `1.3125` queda visible como valor invalido
  superado, no editado;
* digest cientifico igual al esperado del record;
* `grants_nothing` y cero promocion.

La submission no puede compartir commit con cambios de codigo.

## 2. B4: registrar el cierre, no reabrirlo

### C110 - disposicion B4 v4

Consuma esta decision externa:

`B4_V4_ACCEPTED_AS_NON_AUTHORIZING_FINAL_CLOSURE`

Identidades:

* submission file SHA-256
  `63ea37901c8756cd9669d18548176d337efeb08f1bb457a3e395eb90c7be7317`;
* submission self digest
  `9e7047ea077a3dfebc24999654d57c87a623a7345557ad308bfb8202e23a109b`;
* codigo A `4c842dd10da9f4956eea4ffc9435492fcd36243e`;
* publicacion B `ff52ca7d9b771978dfe60bfd15e351ffbe3d1d22`.

Registre en el ledger/work plan: 2 completas, 1 parcial en cuarentena, 9
no iniciadas, `SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT`, campana cerrada.
No cree un consumidor de esta decision y no abra un runner.

### C111 - OLAP aditivo para B4

Si no existe ya una unidad terminal de cierre B4, emita por el outbox la
unidad de campana con los tres conteos y los costos declarados. Debe ser
idempotente y aditiva. No cargue resultados cientificos por celda no
completa ni convierta la celda parcial en fallo o resultado.

## 3. Terminales: lenguaje y consumo honestos

### C112 - disposicion terminal v4

Registre:

`TERMINALS_V4_ACCEPTED_WITH_PHYSICAL_AND_STATISTICAL_SCOPE`

El lenguaje ejecutable y documental debe distinguir:

* descriptor numerico recomputado desde bytes;
* tipo fisico conocido;
* semantica, rol, unidad, licencia y missing policy conocidos;
* autoridad del productor para no-identificables.

Nunca resuma 1,501 como "variables verificadas" sin el calificativo
"descriptores numericos recomputados".

### C113 - gate de consumo semantico

Congele una regresion donde una terminal
`INDEPENDENTLY_RECOMPUTED` con `role=UNKNOWN` o `license=UNKNOWN` no entra
a ninguna poblacion. Lo mismo para `SEMANTICALLY_UNRESOLVED` y para una
fecha almacenada como entero. Esta regla debe vivir en el ultimo punto
de uso, no solo en el reporte.

## 4. Sucesor: tiempo, semantica e identidad

### C114 - contrato temporal del sucesor

Produzca un contrato v3 y una mask nueva desde los bytes del sucesor
`successor_stage22_rerun.v1`. No trasplante el identificador ni el digest
del contrato historico.

Debe verificar y publicar:

* dataset id y SHA-256 del sucesor;
* identidad de `FEATURE_DAG.v4` y binding manifest v2;
* 18,085 filas;
* 20 truncadas dentro del dataset y una adicional anterior fuera de el;
* 8 gaps;
* mask y origenes rederivados;
* comparacion con v2 como igualdad de geometria, no igualdad de identidad.

No interpole ni complete gaps.

### C115 - censo semantico del sucesor

Construya una fila por cada una de las 89 columnas activas. Cada fila
debe ligar:

* `(dataset_id, dataset_sha256, column)`;
* variable id estable;
* tipo fisico y tipo semantico;
* rol exacto (`input_feature`, identificador o exclusion);
* unidad, usando `1` para magnitudes adimensionales cuando corresponda;
* licencia y fuente de la licencia;
* missing policy y sentinel policy;
* productor, simbolo y lookback;
* digest de la evidencia que respalda cada declaracion.

No infiera licencia por reputacion del proveedor ni unidad solo por el
nombre. `UNKNOWN` es una salida valida y excluyente.

### C116 - caracterizacion propia del sucesor

Los terminales historicos no pertenecen a los nuevos bytes. Ejecute en
CPU una caracterizacion separada de las 89 columnas del sucesor con
ledger previo y outputs write-once. Recompute los descriptores desde el
mismo artefacto ligado; no copie terminales v4 aunque los valores
coincidan.

## 5. P0: reparar el derivador de poblacion

### C117 - PRE del falso banco suficiente

Congele el contraejemplo del dictamen por la API publica real:

* seis datasets;
* cinco `CAUSAL_ACTIVE` por dataset;
* seis contratos temporales;
* cero terminales;
* cero variables con semantica/rol/unidad/licencia.

PRE esperado: `BANK_SUFFICIENT_FOR_REVIEW`, 6 paneles, 30 miembros. Esa
salida es el defecto, no una fixture positiva.

Agregue variantes de cardinalidad preservada:

* terminal de otra variable;
* censo de otro dataset;
* digest de source distinto;
* columna duplicada;
* licencia o rol ausente;
* temporal contract de otro digest;
* mask de otro dataset;
* miembro que aparece solo en agregados.

### C118 - join miembro a miembro

Reemplace los conteos independientes por una interseccion relacional
exacta. La clave minima debe contener `dataset_id`, `dataset_sha256` y
`column`; el censo debe aportar un `variable_id` unico y la terminal debe
volver a esa misma clave y a los mismos bytes.

Un miembro existe solo si una unica fila satisface simultaneamente:

1. terminal del sucesor `INDEPENDENTLY_RECOMPUTED`;
2. semantic state `NUMERIC_MEASURABLE`;
3. DAG `CAUSAL_ACTIVE` y binding completo;
4. rol `input_feature`;
5. tipo semantico, unidad aplicable y licencia declarados;
6. missing/sentinel policy declarada;
7. temporal contract y mask ligados al mismo dataset/digest;
8. limites de missingness y observaciones.

Deduplicacion debe rehusar, no sumar una vez. Publique un membership
ledger con una fila por candidato, razones de inclusion/exclusion y
digest de la poblacion. Los agregados se derivan exclusivamente de esas
filas.

### C119 - schemas e inferencia de diseno

Endurezca el parser/validator del diseno y de toda evidencia de
poblacion:

* JSON estricto, claves exactas, tipos exactos;
* booleanos no son enteros;
* NaN/inf y margenes fuera de dominio rehusan;
* duplicados de panel, variable, columna, contraste o semilla rehusan;
* Holm debe usar una familia completa y p-values en `[0,1]`;
* cada contraste usa la misma muestra congelada;
* LOPO se deriva desde las filas de panel, no se acepta declarado.

No cambie H1, H2, H3, los cuatro brazos ni sus margenes como respuesta a
la poblacion vacia.

## 6. P1: construir el banco sin mirar scores

### C120 - inventario de paneles

Construya un censo sin labels ni scores de paneles multivariados. La
unidad de independencia debe declarar dataset, fuente/instrumento,
proveedor, frecuencia, rango temporal, variables, licencia y posible
dependencia con otros paneles.

Objetivo: al menos seis paneles independientes, cada uno con cinco o mas
variables elegibles despues del join C118. Use primero datos publicos ya
custodiados. Si faltan paneles, se autoriza descargar hasta 2 GiB de
datasets publicos no financieros desde fuentes oficiales, sin
credenciales, con URL, licencia, cita, bytes y SHA-256. Una licencia
ambigua produce `LICENSE_REVIEW_REQUIRED` y el panel no cuenta.

No lea targets para seleccionar el banco. No use seis instrumentos del
mismo feed como seis paneles independientes sin modelar su dependencia.

Si al final hay menos de seis, el resultado correcto es
`BANK_INSUFFICIENT` con deficit exacto. No reduzca el minimo.

### C121 - diseno sucesor y retorno

Emita diseno v5 solo para corregir la implementacion de poblacion y sus
bindings. Debe superseder v4 por digest, declarar cambio cientifico
`NONE` para hipotesis/brazos/margenes y mantener scoring cerrado.

El packet final debe incluir:

1. confesiones antes de resultados;
2. PRE, POST y mutaciones por guardia;
3. cronologia T2 y submission v4, o divergence nueva;
4. disposicion final B4;
5. terminales, contrato temporal, censo y caracterizacion del sucesor;
6. membership ledger y deficit/suficiencia real del banco;
7. estado OLAP, backlog y conteos del tip final;
8. digests y ramas empujadas;
9. linea cero: GPU, training, score, confirmation, live y venue.

Detengase en:

`T2_V4_SUBMITTED_FOR_EXTERNAL_REVIEW / B4_CLOSED /
PER_VARIABLE_V5_BANK_READY_OR_INSUFFICIENT_FOR_MUSASHI_REVIEW`

No emita licencia de scoring ni record externo de Musashi.
