# Orden Musashi a General Satoshi: B4 R11-R14, T2 R7-R10 y CRISP-DM C45-C52

**Fecha:** 2026-09-12
**Prioridad:** P0/P1, CPU
**Base revisada predictor:** `9c84f99`
**Retorno auditado:** `e4f5cde`
**Dictamen rector:**
`docs/audits/MUSASHI_AUDIT_RECOVERY_AND_CRISPDM_C31_C44_2026_09_12.md`

## 1. Objetivo y orden de ejecucion

Cerrar cuatro deudas antes de cualquier seleccion:

1. re-adjudicar B4 desde artefactos consumidos por descriptor;
2. re-adjudicar T2 desde una unica evidencia verificada y una identidad de
   codigo completa;
3. separar demanda ejecutante de datasets candidatos y construir el DAG
   causal real de cada feature;
4. dar disposicion a las 1.965 variables conceptuales del lago.

Ejecutar primero los PRE. Luego trabajar en paralelo asi:

- **P0-A:** B4 R11-R14, hasta submission para revision;
- **P0-B:** T2 R7-R10, hasta submission para revision;
- **P0-C/P1:** C45-C52 completos, CPU y OLAP;
- **P2 opcional:** correcciones de entry points y ruido Alpaca, sin reiniciar
  ni tocar servicios live.

Si B4 o T2 llega a su frontera de review, continuar C45-C52; no quedar ocioso.

## 2. PRE obligatorio

Congelar antes de editar, usando copias temporales y APIs publicas:

1. intercambiar el terminal B4 entre lectura, hash y parseo permite verificar
   unos bytes y clasificar otros;
2. intercambiar el ledger por barra entre hash y conteo cambia lo consumido;
3. reemplazar claim, intent o seal despues de parsearlo y antes de hashearlo
   produce un binding de otro objeto;
4. reemplazar objetos de una celda parcial despues del inventario cambia su
   adjudicacion;
5. crear un artefacto despues del chequeo de ausencia conserva indebidamente
   `NOT_STARTED`;
6. ningun config ejecutable referencia los dos CSV del registro CRISP-DM, pero
   `demand_columns()` publica demanda supervisada 93;
7. el conjunto RL vacio produce `rl_is_subset_of_supervised=true`;
8. 93 nombres unicos se reportan junto a 97 sujetos `(dataset, columna)` sin
   distinguir los granos;
9. una columna sin target ni config consumidor entra como input activo;
10. dos features con productores/ventanas diferentes reciben el mismo
   `upstream` y la misma disponibilidad generica;
11. retirar o cambiar el productor real de una feature no cambia su lineage;
12. el ledger de caracterizacion termina verde con 107 sujetos aunque 1.858 de
   las 1.965 variables no tienen outcome;
13. sustituir un `RECORD` T2 entre `final_adjudication()` y
   `adjudicate_screen()` permite verificar unos bytes y puntuar otros;
14. cambiar `t2_completion_reconstruction.py` o
   `t2_campaign_closure.py` no cambia la identidad denominada revisada;
15. no existe bateria dedicada al cierre T2 que mate los dos casos anteriores.
16. con tres signos positivos entre seis, la adjudicacion T2 publica el valor p
    imposible `1.3125`; la implementacion no produce una tabla bilateral
    simetrica y acotada para 0..6.

Los PRE de B4 y T2 usan copias de sus raices; las evidencias originales son
read-only.

## 3. P0-A: B4 R11-R14

### R11. Consumo descriptor-first real

- Abrir cada artefacto desde la raiz retenida con `openat` y `O_NOFOLLOW` por
  componente.
- Verificar con `fstat` archivo regular, propietario y modo desde ese mismo
  descriptor.
- Leer bytes completos una sola vez; hash, parseo, conteo y reglas semanticas
  consumen esos bytes, nunca una reapertura por ruta.
- Aplicar a terminal, ledger por barra, claim, lease, intentos, sellos,
  checkpoints y objetos de la celda parcial.
- Un cambio posterior puede provocar refusal o ser irrelevante; jamas puede
  cambiar los bytes que llegan a la adjudicacion.

### R12. Inventario y estados desde snapshots

- Construir un inventario exacto desde entradas de directorio validadas y
  snapshots inmutables.
- Derivar `COMPLETED_VERIFIED`, `QUARANTINED_PARTIAL` y `NOT_STARTED` solo de
  ese inventario. No mezclar `exists/is_file/glob/stat` con lecturas posteriores.
- Ligar archivos extra, faltantes y duplicados. La aparicion de un archivo
  despues del snapshot no puede alterar esa ejecucion.
- Preservar la raiz real B4 byte-intacta; toda prueba destructiva usa copia.

### R13. Identidad completa y bateria adversarial

La submission debe ligar `b4_campaign_closure.py`, todos sus helpers
alcanzables, commit y arbol limpio. Agregar pruebas individuales para los cinco
PRE, symlink de hoja y directorio, archivo permisivo, propietario incorrecto,
extra/faltante/duplicado y mutacion de cada capa descriptor-first. Un mutante
que restaure reapertura por ruta debe morder.

### R14. Frontera B4

Reproducir `2/1/9` y `SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT` sobre una copia,
emitir submission de re-adjudicacion y detenerse. Musashi revisara la identidad
antes de abrir la raiz preservada. Cero relanzamiento, entrenamiento, GPU o
promocion; la celda en cuarentena no se reanuda.

## 4. P0-B: T2 R7-R10

### R7. Snapshot verificado unico

- El verificador profundo debe devolver objetos inmutables que contengan los
  bytes verificados, los arrays reconstruidos, digests y hechos derivados.
- La adjudicacion del screen consume esos objetos. No puede volver a resolver
  ni abrir `RECORD`, `ARRAYS`, manifest, censo o diseno por ruta.
- Si la API actual no puede devolverlos, crear una lectura descriptor-first
  unica por unidad y hacer que verificacion y adjudicacion compartan esa
  misma instancia.
- Un swap, rename, symlink o cambio posterior puede causar refusal, pero jamas
  cambiar el score consumido.

### R8. Identidad completa del cierre

La submission de re-adjudicacion debe ligar al menos:

- los siete archivos ya fijados por el record de ejecucion;
- `t2_completion_reconstruction.py`;
- `t2_campaign_closure.py`;
- todo helper nuevo alcanzable por ambos;
- commit, arbol limpio y version de dependencias numericas usadas.

No llamar "reviewed identity" a una mezcla. El candidato puede emitir
digests y pedir revision; no puede autorizarse.

### R9. Bateria adversarial

Agregar pruebas para:

1. record intercambiado entre verify y score;
2. arrays intercambiados;
3. reconstructor mutado;
4. closure mutado;
5. checkout revisado sucio o en otro commit;
6. poblacion con extra/faltante/duplicado;
7. segundo cierre idempotente;
8. resultado negativo original reproducido sin escritura ni entrenamiento.
9. tabla completa del test binomial bilateral exacto para 0..6 positivos:
   `0.03125, 0.21875, 0.6875, 1.0, 0.6875, 0.21875, 0.03125`.

Cada mutacion debe morder una prueba individual. No publicar conteos antes de
leer la salida del tip final.

La reparacion del valor p supersede el campo defectuoso y declara que no cambia
el estimando, los efectos por panel ni el veredicto gobernado por dano. No se
reescribe el envelope historico que contiene `1.3125`.

### R10. Frontera de ejecucion

Preparar la submission de codigo y detenerse antes de reabrir la raiz real.
Musashi revisara esa identidad. Tras review separado se re-derivara el mismo
T2 desde los 242 records, se supersedera el envelope anterior de forma aditiva
y se comprobara si `DOES_NOT_ADVANCE` permanece. Cero reentrenamiento y cero
descarga.

## 5. P0-C: verdad de demanda y linaje

### C45. Tres universos separados

Publicar estructuras distintas:

1. `ACTIVE_EXECUTION_DEMAND`: datos, lados `x/y`, roles, targets y configs
   realmente alcanzables por el entry point ejecutante;
2. `RESEARCH_BANK_CANDIDATES`: datasets registrados para desarrollo que aun
   no tienen consumidor ejecutable;
3. `CONFIRMATORY_RESERVED`: datos cuyo examen gastaria confirmacion.

Un registro o header por si solo nunca es consumo. Derivar cada consumidor
desde su config efectivo, ruta de datos, contrato y codigo exactos. Si hoy no
hay consumidor activo supervisado ni RL, el resultado correcto es cero/cero.

### C46. Granos y roles exactos

Materializar y validar por separado:

- `unique_column_names`;
- `dataset_column_subjects`;
- `active_x_subjects`;
- `active_y_subjects`;
- `targets`;
- `research_bank_candidates`.

Cada sujeto minimo liga `(dataset, side, role, column)`. Un conjunto vacio
produce `NOT_APPLICABLE`, nunca `true` en pruebas de subconjunto o cobertura.
El schema fisico debe coincidir con la version del filename.

### C47. DAG causal por feature

Para cada columna candidata de la vista Project3, localizar su productor real
en `financial-data`, `feature-eng`, `feature-extractor` o declarar que no fue
localizado. Cada nodo debe contener:

- dataset y columna de salida;
- repositorio, commit, archivo, funcion/clase y digest de codigo productor;
- entradas directas y sus roles;
- lookback, lead/shift, alineacion y politica de ventana;
- `event_time` de cada entrada;
- latencia causal/publicacion;
- formula ejecutable de `earliest_available_time`;
- clase `CAUSAL`, `NON_CAUSAL`, `EXTERNAL_LATENCY_REQUIRED` o
  `UNRESOLVED_PRODUCER`.

El grafo debe ser no serial y admitir varias entradas. Una frase comun para
las 84 features no satisface C47. Probar al menos retorno con lag, estadistico
rodante, indicador tecnico, OHLCV crudo y una feature no localizable.

### C48. Supersesion honesta de C42-C44

- Conservar byte-intactos bridge/lineage/candidate anteriores como historia.
- Emitir una version nueva que diga que los 89 registros previos eran
  candidatos de banco, no demanda activa.
- Solo una feature con DAG y disponibilidad derivada puede entrar a una
  submission futura.
- La revision externa sigue siendo obligatoria; ninguna herramienta candidata
  concede `PUBLICLY_ELIGIBLE`.

## 6. P1: caracterizacion completa

### C49. Denominador 1.965/1.965

Procesar las 1.965 variables conceptuales del censo en lotes deterministas y
reanudables. Cada `variable_id` debe terminar exactamente una vez en:

- `MEASURED`;
- `NOT_IDENTIFIABLE`;
- `UNAVAILABLE`;
- `FAILED`.

No exigir numero donde no existe. Registrar por descriptor su valor o razon de
ausencia. Separar variable conceptual de aparicion fisica y no multiplicar el
denominador por archivos.

### C50. Custodia y reanudacion

- Ledger pre-resultado con las 1.965 identidades.
- Claims y terminales write-once por lote/variable.
- Fuente, bytes, ventana, codigo, costo y protocolo ligados.
- Reanudar despues de interrupcion sin duplicar ni perder outcomes.
- Un productor que crashea una variable no cancela el lote.
- El outbox recibe exitos, ausencias, fallos e inconclusos.

La corrida anterior de 107 sujetos se conserva como una observacion previa; no
se edita para fingir cobertura.

### C51. OLAP y consultas de aceptacion

El cubo debe conservar, como minimo, los conteos auditados: 39 experimentos,
1.404 performance, 8 runs, 127 unidades y las 2.988 caracterizaciones actuales.
Agregar historia de C49 aditivamente. Publicar consultas que prueben:

1. 1.965 variables con una disposicion terminal;
2. cobertura por banco y por estado;
3. apariciones fisicas separadas;
4. ninguna seleccion emitida;
5. reingestion idempotente.

### C52. Sucesor de preprocesamiento por variable, solo diseno

Con los resultados de C47-C51, preparar un diseno **sin scores** para evaluar
preprocesamiento por variable. Debe comparar, con presupuesto pareado:

- `X` sin transformacion nueva;
- `D_j(X_j)` por variable y operador candidato;
- `[X_j, D_j(X_j), X_j-D_j(X_j)]` cuando tenga sentido;
- control de igual dimension/capacidad.

La transformacion EWMA global de T2 no avanza automaticamente: su resultado
negativo se respeta. Una regla por variable es una hipotesis nueva y debe
ganarse su propia licencia en desarrollo antes de cualquier confirmacion. El
diseno debe fijar causalidad, costos, abstencion, unidad estadistica y regla de
retiro; no ejecutar seleccion todavia.

## 7. Residuales P2

### E1. Entry points duplicados con valor identico

Extender la politica para dos distribuciones que publican el mismo
`(group, name, value)`: registrar y ligar todas las distribuciones o rehusar.
Agregar prueba que invierte el orden de instalacion y conserva exactamente el
mismo testigo o produce refusal.

### E2. Repeticion Alpaca

Reproducir offline el estado `pending_entry_target_changed` con decision
due-bar terminal. Si la imposibilidad de revision es el estado esperado,
convertir iteraciones siguientes en un no-op idempotente tipado y conservar el
heartbeat. No cancelar, reemplazar ni enviar orden; no reiniciar el servicio.
La correccion queda como candidato para revision independiente.

## 8. Estado operativo y frontera del owner

La actualizacion NVIDIA `580.178.04` se instalo despues del ultimo boot, que
continua ejecutando `580.173.02`. El **owner debe reiniciar Omega una vez**.
Satoshi no descarga drivers, no descarga kernels y no intenta descargar/reload
del modulo.

Despues del reinicio, ejecutar solo:

1. `nvidia-smi`;
2. la matriz read-only `tools/recovery_matrix.py --json`;
3. comprobacion de loader, PostgreSQL, Metabase, supervisor, sesiones y timers.

El resultado esperado es 18/18 `OK`. Si la GPU sigue no enumerable, detenerse
con diagnostico. Aunque quede verde, **no lanzar una campana GPU**: B4 ya es
terminal sin veredicto y el frente activo es data-centric/CPU.

## 9. Aceptacion

1. Los dieciseis PRE mueren por su razon exacta.
2. B4 no reabre artefactos y reproduce 2/1/9 desde snapshots descriptor-first.
3. T2 no reabre evidencia verificada y su identidad incluye todo el codigo de
   cierre; ningun valor p sale de `[0,1]`.
4. Demanda activa y banco de investigacion no comparten etiqueta ni contador.
5. Targets nunca entran como inputs por ausencia de metadata.
6. Cada feature resuelta tiene DAG causal ejecutable; lo desconocido queda
   desconocido.
7. Cobertura conceptual exacta 1.965/1.965 con outcomes terminales.
8. OLAP conserva toda la historia y carga todos los desenlaces.
9. El diseno por variable existe, pero cero seleccion y cero confirmacion se
   ejecutan.
10. B4, M4 y T2 no se reentrenan.
11. Ningun servicio live se reinicia, ninguna orden se modifica y ninguna
    operacion se envia.

## 10. Reporte de retorno

El §7 debe comenzar por defectos propios y reportar:

- PRE/POST y mutaciones con conteos del tip final;
- identidades de todos los repos modificados;
- B4 y T2: frontera exacta y si las raices reales fueron leidas o no;
- demanda activa, candidatos, targets y granos separados;
- cobertura DAG por clase y lista de productores no localizados;
- cobertura 1.965/1.965 y costos CPU/almacenamiento;
- conteos OLAP antes/despues;
- estado post-reinicio, si ya ocurrio;
- disposicion unica y fronteras respetadas.

**No hay decision adicional del owner para preparar B4 R11-R14, T2 R7-R10 o
C45-C52.**
La unica accion del owner es reiniciar Omega para alinear el driver NVIDIA.
