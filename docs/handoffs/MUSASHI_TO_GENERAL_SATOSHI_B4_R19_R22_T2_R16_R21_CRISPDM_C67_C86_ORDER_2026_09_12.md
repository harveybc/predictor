# Orden de Musashi a General Satoshi: B4 R19-R22, T2 R16-R21 y CRISP-DM C67-C86

**Fecha:** 2026-09-12  
**Base del dictamen:** `MUSASHI_AUDIT_CRISPDM_C53_C66_B4_T2_2026_09_12.md`  
**Prioridad:** P0 integridad de evidencia; P1 datos y diseno  
**Autorizacion vigente:** `MECHANICS_ONLY_CPU_NO_SCORES`

## 0. Fronteras y bases

Trabaje en ramas nuevas desde estos tips empujados:

| Repositorio / frente | Base |
|---|---|
| B4 | `0eab2705` |
| T2 | `26b4214b` |
| snapshot T2 recuperable | `f070eeb1` |
| predictor | `9bb90fa` |
| financial-data | `d53f24b6a` |

Antes de editar, congele PRE ejecutables para cada contraejemplo. No
modifique B4 v7, `t2_successor`, las terminales v1/v2, el cubo poblado
ni los registros externos existentes. No cree un record en nombre de
Musashi. No ejecute GPU, entrenamiento, screen, confirmacion, live,
venue ni publicacion DOIN. El loader OLAP permanece activo; no se
reinicia para esta orden.

## 1. P0-A: identidad de cada hoja inventariada

### C67 / B4-R19 / T2-R16 — PRE obligatorio

Reproduzca en las tres copias de `descriptor_custody.py`:

1. fotografiar un directorio que contiene `terminal.json=1`;
2. renombrar esa hoja dentro del mismo directorio;
3. crear una hoja sustituta, mismo nombre, modo, longitud y mtime, con
   valor distinto;
4. demostrar que el lector actual consume la sustituta;
5. repetir con mutacion in-place de igual longitud despues del
   inventario;
6. en T2, repetir con un `ARRAYS_<uid>.npz` antes de su lectura tardia.

El PRE debe registrar los hechos iniciales y finales de archivo. No
basta reemplazar el directorio: ese caso ya esta cubierto.

### C68 / B4-R20 / T2-R17 — correccion

`DirSnapshot` debe conservar por nombre los hechos obtenidos durante el
inventario. Antes de aceptar una lectura:

* el `fstat` inicial del descriptor abierto debe coincidir con el
  inventario en tipo, device, inode, uid, mode, size, mtime_ns y
  ctime_ns;
* un segundo `fstat` despues de leer debe coincidir con el primero;
* una divergencia produce refusal tipado y ningun valor derivado de
  esos bytes puede entrar al resultado;
* el artefacto publicado debe ligar los hechos inventariados y los dos
  `fstat`, no solamente el directorio padre;
* todos los descriptores se cierran en exito y excepcion.

Elimine la divergencia entre las tres copias: mismo contrato, mismos
fixtures y mismo digest de implementacion, o documente ejecutablemente
por que una copia debe diferir.

### B4-R21 — nueva readjudicacion sobre copia

Reejecute el cierre B4 solo sobre una copia privada de la raiz
preservada. Exija el nuevo binding por hoja y publique submission v3 con
protocolo A/B: A contiene codigo y tests empujados; B agrega solamente
la submission generada desde A limpio. El resultado esperado sigue
siendo 2 completas, 1 parcial, 9 no iniciadas; cualquier cambio debe
detenerse y explicarse, no normalizarse.

### B4-R22 — aceptacion focal

Ademas de la bateria existente:

* hoja reemplazada despues del inventario;
* hoja reemplazada y restaurada por nombre;
* mutacion in-place de igual longitud;
* mutacion durante lectura por bloques;
* mismo modo y mtime, inode distinto;
* mutante que elimina cada una de las comparaciones pre/post.

## 2. P0-B: identidad correcta para la readjudicacion T2

### T2-R18 — no repinar historia

El `MUSASHI_T2_SUCCESSOR_EXECUTION_RECORD` historico es inmutable. No
cambie su commit ni sus siete digests. Implemente un contrato separado:

`agent_multi.t2_readjudication_review_record.v1`

Ese record, que Musashi instalara despues del retorno, debe ligar como
minimo:

* digest del execution record historico y su commit;
* los siete digests de codigo historico;
* commit limpio del snapshot de readjudicacion;
* digest y lista exacta de toda la superficie de readjudicacion;
* identidad de la raiz preservada y su inventario;
* digest de la adjudicacion candidata;
* decision, fecha canonica, alcance `READ_ONLY_READJUDICATION`;
* `retraining=false`, `downloads=false`, `grants_execution=false`.

La herramienta candidata entrega solamente un template y un verificador
que lee una ruta externa fija. Sin record real debe detenerse en
`READJUDICATION_REVIEW_RECORD_REQUIRED` antes de abrir la evidencia.

### T2-R19 — un solo checkout del reproductor

Construya desde `f070eeb1` un sucesor que contenga toda la superficie y
las correcciones R16-R18. Todos los imports del reproductor salen de
ese checkout. El gate nuevo no exige que `HEAD` sea el commit historico:
exige simultaneamente los siete digests historicos y el commit/digests
del reproductor que el record nuevo revise.

### T2-R20 — reejecucion sobre copia, sin puntuar de nuevo

Tras fixture de review aislado, reproduzca desde una copia privada los
242 registros y la tabla binomial corregida. No entrene ni regenere
predicciones. La submission v3 debe declarar claramente:

* resultado historico ejecutado bajo la identidad vieja;
* readjudicacion ejecutada bajo la identidad nueva;
* mismos 242 efectos y estimando, o stop por divergencia;
* el resultado sigue sin conceder promocion.

### T2-R21 — bateria

Incluya hoja NPZ sustituida despues del inventario, execution record
repinado retroactivamente, snapshot con un archivo omitido, mezcla de
imports entre checkouts y record de readjudicacion autoemitido por el
candidato. Todos deben rehusar por su causa exacta.

## 3. P0-C: verificacion profunda de las 1.965 terminales

### C69 — PRE obligatorio

Congele por API publica:

1. fuente ausente produce hoy `TERMINALS_VERIFIED_EXACT`;
2. descriptor fabricado produce hoy `TERMINALS_VERIFIED_EXACT`;
3. `appearance` de otra variable o de otro archivo es aceptado;
4. ruta absoluta, traversal y symlink de fuente alcanzan la lectura;
5. terminal con clave extra, tipo incorrecto o outcome desconocido;
6. ledger y censo autoconsistentes pero con poblacion sustituida.

### C70 — nombre honesto inmediato

Mientras no se recalculen los valores, la verificacion solo puede emitir
`TERMINAL_POPULATION_AND_SOURCE_BINDING_VERIFIED`. Retire
`TERMINALS_VERIFIED_EXACT` de todo resultado que no haya recalculado los
descriptores. El estado `MEASURED` del productor se conserva como
`PRODUCER_DECLARED_MEASURED`, no como conclusion del revisor.

Una fuente ausente, no regular, fuera de la raiz, intercambiada o no
ligada al appearance vuelve el conjunto divergente.

### C71 — binding variable -> appearance -> archivo

Valide esquemas y tipos exactos de ledger, censo y terminal. Cada
terminal debe ligar su `appearance` exacto; ese appearance debe ligar
la variable y un unico archivo fisico. No se permite escoger el primer
appearance de una lista. Lea cada fuente por componentes contenidos,
sin seguir enlaces, y derive digest y hechos del mismo descriptor.

### C72 — verificador cientifico independiente

Construya una implementacion de revision que no importe las formulas del
productor y que, desde bytes fuente, vuelva a calcular los descriptores
publicados para cada variable evaluable. Debe:

* enumerar por nombre los descriptores esperados y sus unidades;
* obtener los valores publicados desde la evidencia durable/OLAP, no
  desde un conteo terminal;
* comparar valores, support, missingness y outcome con tolerancias
  predeclaradas;
* exigir correspondencia exacta de las filas de descriptor con los
  1.965 sujetos;
* emitir v3 de forma aditiva; v1 y v2 quedan byte-intactos;
* no conceder elegibilidad: solo producir evidencia que otro gate puede
  consumir.

Si una formula no puede implementarse independientemente, clasifiquela
`NOT_INDEPENDENTLY_VERIFIABLE`; no la cuente como medida verificada.

### C73 — OLAP

El loader debe ingerir la supersesion v3 de forma aditiva y preservar
las filas historicas. Una vista nueva debe distinguir
`PRODUCER_DECLARED`, `SOURCE_BOUND` e `INDEPENDENTLY_RECOMPUTED`.
Backlog o dead-letter no se ocultan. Pruebe todo primero en base
desechable; el cubo real solo recibe artefactos despues de verificacion.

## 4. P0-D: `FEATURE_DAG.v3`

### C74 — PRE de siete familias

Congele los seis contraejemplos del dictamen y uno de binding:

1. reasignacion local con fuga en la ultima definicion;
2. ventana encadenada 10 despues de 20;
3. callable adelantado dentro de `apply`/`map`/`transform`;
4. `iloc[-1]` y otros indices posicionales constantes;
5. cuerpo de funcion anidada que contamina el scope exterior;
6. dos productores de la misma columna cuyo orden cambia el veredicto;
7. un productor causal de nombre coincidente que nunca genero el
   dataset fisico.

### C75-C77 — analisis conservador

Reemplace la heuristica por un analisis que respete scope, orden de
asignacion y flujo de control. Cuando varias definiciones alcanzan una
salida y no puede probarse cual gobierna, emita `UNRESOLVED`.

Requisitos minimos:

* nunca recorrer cuerpos anidados como si fueran el cuerpo exterior;
* versionar reasignaciones y propagar dependencias de la definicion que
  alcanza el uso;
* analizar argumentos, keywords y callables pasados a metodos;
* todo indice posicional no probado como historico es `UNRESOLVED`;
* ventanas consecutivas componen su alcance (`10` y `20` -> `29` para
  ventanas inclusivas), y shifts suman dependencia;
* la lista de metodos seguros no puede ocultar funciones de usuario;
* ciclos, dispatch dinamico, ramas ambiguas y llamadas desconocidas
  quedan `UNRESOLVED`.

### C78 — binding de procedencia

No elija productores por nombre ni por orden. Cree un manifest que
ligue:

`dataset_id + physical_sha256 -> repository + commit + file + symbol + output`

El binding debe derivarse de la procedencia disponible o quedar
`UNRESOLVED_PRODUCER_BINDING`. Un productor encontrado pero no ligado
puede llamarse `STATIC_CAUSAL_CANDIDATE_UNBOUND`, nunca
`CAUSAL_ACTIVE`.

### C79 — prueba dinamica complementaria

Para cada productor ligado y ejecutable, agregue una prueba de
invariancia de prefijo: perturbar solo el futuro no puede cambiar ningun
valor anterior al corte. Esto complementa, no reemplaza, el analisis
estatico. Un productor no ejecutable queda con evidencia estatica
solamente y no entra a la licencia.

### C80 — publicacion aditiva

Publique `FEATURE_DAG.v3`; v1 y v2 no se reescriben. Reporte los conteos
por clase despues de todas las guardias. No hay minimo positivo que
deba fabricarse.

## 5. P1-A: semantica temporal separada

### C81 — contrato ETH H4

Materialice evidencia reproducible para el dataset Project 3:

* CSV sha `1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f`;
* 18.085 filas emparejadas contra el parquet fuente por `DATE_TIME`;
* `DATE_TIME == open_time`;
* OHLCV con error maximo `0.0`;
* `information_complete_not_before = close_time`;
* `timestamp_semantics = BAR_OPEN`, `bar_seconds = 14400`;
* `provider_delivery_latency = UNOBSERVED`.

El artefacto debe ligar ambos archivos por digest y consumir los bytes
que verifica. No generalice esta decision al EURUSD legacy.

### C82 — dos nociones de disponibilidad

Separe en schema y codigo:

1. `causal_information_bound`: cuando el dato podria estar completo por
   definicion del periodo;
2. `operational_delivery_bound`: cuando se observo o garantizo que el
   proveedor lo entrego.

Para experimentos offline, E5a exige la primera. Para live, E5b exige
ambas. `UNOBSERVED` nunca se convierte en cero. La propuesta no obtiene
viabilidad de ejecucion live por esta orden.

### C83 — fixtures

Cubra BAR_OPEN, BAR_CLOSE, PUBLICATION, huecos, barras irregulares,
close_time contradictorio, latencia ausente, dataset trasplantado y
EURUSD sin procedencia. Una disponibilidad causal offline no debe abrir
la compuerta live.

## 6. P1-B: diseno v3; sin screen

### C84 — especificacion completa

Escriba un sucesor v3, sin puntuar, que fije:

* pregunta e hipotesis primaria;
* poblacion exacta derivada de terminales v3 + DAG v3 + E5a;
* paneles y unidad estadistica;
* operadores, parametros y regla de abstencion;
* modelos, ventanas/origenes, semillas y presupuesto por brazo;
* metricas primaria/secundarias y polaridad;
* margenes numericos, multiplicidad y minimo de paneles;
* construccion exacta del control A3 sin fuga ni informacion nueva;
* missingness, costo total, reglas de retiro y resultado nulo.

Todos los brazos usan el mismo espacio de datos y presupuesto. El
diseno debe poder validarse campo a campo y rehusar cualquier delta
despues de review.

### C85 — licencia

Esta orden no concede scoring. Puede ejecutar parsers, materializacion
sin targets, dry-runs, fixtures y pruebas CPU. El entry point de score
debe detenerse en `EXTERNAL_DESIGN_REVIEW_AND_LICENSE_REQUIRED` antes
de leer labels o construir modelos.

### C86 — retorno

Entregue un unico packet con:

1. defectos propios primero;
2. PRE y POST por hallazgo;
3. commits A/B de B4 y T2;
4. conteos medidos en los tips finales;
5. digests y deltas de v1/v2/v3;
6. prueba de que B4 v7, T2 original y terminales v1/v2 no cambiaron;
7. estado OLAP y backlog;
8. lista exacta de records que Musashi debe autorar despues;
9. linea expresa: cero GPU, cero scores, cero confirmacion, cero live.

## 7. Criterio de cierre

El retorno es revisable cuando los contraejemplos P0 dejan de producir
un resultado aceptado, las identidades historicas permanecen intactas y
los nuevos artefactos dicen exactamente que prueban. Hasta entonces:

`B4_AND_T2_READJUDICATION_REVISE / FEATURE_DAG_REVISE / DEVELOPMENT_SCREEN_CLOSED`.
