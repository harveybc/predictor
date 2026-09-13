# Orden de Musashi a General Satoshi: C166-C184 cierre verificable de D2

**Fecha:** 2026-09-13

**Prioridad:** P0 antes de D3

**Auditoria rectora:**
`MUSASHI_AUDIT_C146_C165_MEMORY_CAUSALITY_2026_09_13.md`

**Licencia:** CPU offline y OLAP aditivo. Sin GPU, modelos, feature selection,
D3-D5, RL, DOIN, live ni venue.

## 0. Objetivo y alto

Eliminar los bypasses de la frontera causal, reparar procedencia y cobertura,
y obtener decisiones D2 bajo la API temporal actual mediante un reanalisis
historico separado de una confirmacion sintetica fresca.

Detenerse en:

```text
D2_CURRENT_API_FRESH_CONFIRMATION_READY_FOR_MUSASHI_REVIEW
```

Nada de esta orden concede elegibilidad ni abre D3 por si mismo.

## 1. PRE y preservacion

### C166 - congelar los contraejemplos antes de editar

Reproducir mediante APIs publicas reales y preservar salida:

1. apagar `df_snapshot.GUARDS["source_rederive"]` permite consumir una matriz
   mutada y redigerida que no sale de los bytes del contrato;
2. apagar `availability` permite una observacion disponible despues de la
   decision;
3. apagar `fit_mode_enforcement` permite ajustar
   `trailing_haar_threshold` en modo expanding y hace que filas futuras de
   training cambien salidas anteriores;
4. apagar `BATTERY_GUARDS["exhaustive_cuts"]` deja pasar la fuga de una muestra;
5. las tres ventanas ADF/KPSS producen estimaciones de recursos de identidad
   igual;
6. `df_coverage` transforma `REFUSED` en `FAILED` y una metrica inaplicable en
   `NOT_RUN`;
7. una decision C137 historica puede parecer consumible aunque su codigo, nombre
   y modo temporal no sean los actuales;
8. la paridad publica historica/columnar difiere en PCA, effective rank y
   loadings pese a exigir igualdad exacta.

Preservar byte a byte C137, C156, C162, C163, C164 y las raices historicas.
Ninguna se reescribe.

## 2. Frontera causal sin interruptores

### C167 - quitar `GUARDS` de produccion

* `df_snapshot.py` y `df_operators.py` no contendran diccionarios, flags,
  variables de entorno ni parametros que omitan una comprobacion causal.
* Todas las comprobaciones se ejecutaran incondicionalmente en cada ruta
  productiva.
* Asignar atributos llamados `GUARDS`, reemplazar una tabla importada o pasar
  configuracion adicional no puede cambiar la conducta.
* `df_causal_battery.py` tampoco tendra un interruptor que reduzca los cortes en
  una ejecucion que pueda publicarse como bateria completa.

### C168 - mutaciones estructurales y aisladas

Las pruebas de mutacion deben copiar la superficie minima a un directorio
temporal, quitar una guardia en la fuente o AST y ejecutarla en un proceso
nuevo. Cada mutante debe:

* demostrar primero que la fuente productiva intacta rehusa;
* demostrar despues que quitar solo esa guardia hace fallar el test esperado;
* no importar el modulo mutado en el proceso que produce evidencia;
* guardar digest de fuente original, mutante y prueba que mordio;
* incluir el caso wavelet que cambia `t <= 30` al modificar filas 31-59.

La bateria oficial solo puede emitir `PASS` si usa la lista completa de todos
los prefijos declarados; un subconjunto se llama `MECHANICS_SAMPLE`, nunca
evidencia causal.

## 3. Identidad y cobertura

### C169 - identidad de cada bloque ADF/KPSS

Cada fila de estimacion debe ligar, antes de invocar la libreria:

* `block_offset` (`exact`, `start`, `middle`, `end`);
* rango absoluto `[start, end)` y universo del finite run;
* longitud y lag realmente usados;
* digest de la politica y variante exacta/aproximada.

La clave OLAP debe distinguir esos hechos. Recalcule las filas afectadas desde
los terminales C162, sin repetir ADF/KPSS. La cardinalidad ofrecida y cargada
debe cerrar; los 2.098 colapsos anteriores quedan registrados como v1, no
borrados.

Congele tambien el contraejemplo de paridad numerica. Para descriptores basados
en eigendescomposicion:

* registre version de NumPy y backend de algebra lineal;
* declare una tolerancia absoluta/relativa por metrica antes del POST;
* pruebe el mismo fixture en los tres roles;
* conserve el valor crudo, y use una representacion canonica separada si el
  valor debe participar en digests portables;
* retire toda afirmacion de igualdad byte a byte que solo sea equivalencia
  numerica.

### C170 - matriz de cobertura v2

Implemente estados separados:

```text
RESULT, INCONCLUSIVE, UNAVAILABLE, NOT_APPLICABLE, NOT_RUN,
REFUSED, FAILED, RESOURCE_EXCEEDED, UNCERTAIN
```

El grid esperado debe declarar aplicabilidad por dataset, tipo de variable,
particion, metrica, operador y politica. No infiera `NOT_APPLICABLE` solo porque
falta una fila. `REFUSED` no es fallo numerico; `REJECTED` es una decision
cientifica, no indisponibilidad. Cuando convivan estados, la precedencia debe
estar especificada y probada.

Migre de forma aditiva. Preserve la matriz v1, publique el mapa v1->v2 por celda
y demuestre que los totales salen del ledger miembro por miembro.

## 4. Reanalisis D2 bajo la API actual

### C171 - diseno superseding antes de resultados

Antes de ejecutar una evaluacion nueva, selle un diseno D2 v2 que fije:

* banco, contratos y roles temporales;
* operador, parametros y modo de ajuste actuales;
* rama cruda `X`, transformada `D(X)` y residual `X-D(X)`;
* metricas de mejora, distorsion, retardo, eventos, extremos y costo;
* reglas de abstencion, soporte y datos faltantes;
* unidad de analisis = generador/semilla independiente, no fila temporal;
* seleccion de parametros solo en desarrollo historico;
* confirmacion primaria solo en la reserva fresca, sin retuning;
* multiplicidad por familia de operadores y regla de no inferioridad;
* presupuesto, timeout, memoria y asignacion por roles.

Use C137 solamente para estimar dispersion y escoger candidatos. Determine y
fije antes de generar la reserva el numero de semillas por regimen, con minimo
10 y maximo 30, buscando al menos 80% de potencia o una precision equivalente.
Si el maximo no alcanza, el regimen queda `UNDERPOWERED`, no se maquilla.

### C172 - reanalisis historico de migracion

Ejecute el banco C137 intacto desde una raiz fresca y write-once con el codigo
actual, sin copiar decisiones. Publique:

* igualdad de unidades y truth arrays;
* diferencias de salida y decision frente a C137 por operador/regimen;
* causa de cada flip: nombre, modo temporal, rango transformado, aritmetica,
  abstencion o correccion real;
* conteos `REFUSED`, `FAILED`, `INCONCLUSIVE` y resultados sin fusionarlos.

Este estrato se etiqueta `HISTORICAL_MIGRATION_REANALYSIS_NON_CONFIRMATORY`.
No concede consumo.

### C173 - reserva sintetica fresca

Tras C171 y C172, materialice un tape nuevo de semillas que no haya aparecido en
C128-C163. Selle el tape antes de generar datos y derive cada unidad con el
generador ya revisado. No inspeccione plots, metricas ni muestras para cambiar
el diseno.

Cada unidad debe conservar clean, noise, observed, missing mask, eventos,
contrato y particiones TRAIN/CALIBRATION/CONFIRMATION. Train ajusta;
calibration es diagnostico; solo confirmation gobierna la prueba fresca. Una
misma semilla y unidad deben alimentar todos los brazos pareados.

### C174 - ejecucion distribuida D2

Distribuya por costo y memoria entre los tres roles CPU. Un dataset por proceso,
limites duros, terminal durable y resume por identidad. Ninguna GPU se usa.
`WORKER_B/GPU1` queda excluida del inventario despachable, pero el CPU del host
permanece habilitado.

Ejecute para cada candidato y sus controles la API productiva
`FitSnapshot -> fit -> TransformSnapshot -> transform`. Los kernels privados no
pueden producir evidencia. Incluya controles identity/raw, operador rechazado y
oraculo no causal; el ultimo debe ser detectado, nunca competir como brazo.

## 5. Adjudicaciones

### C175 - SNR por estimador y regimen

Use C163 como desarrollo, no confirmacion. En la reserva fresca, por estimador y
regimen, derive error contra SNR verdadero, sesgo, error absoluto, intervalo y
tasa `NOT_IDENTIFIABLE`.

Una decision `SNR_CALIBRATED_FOR_REGIME` requiere, en confirmation:

* limite superior del IC 95% del error absoluto medio <= 1 dB;
* al menos 90% de cobertura del intervalo nominal;
* tasa `NOT_IDENTIFIABLE` <= 10%;
* ninguna combinacion de sesgos opuestos puede cancelar el error absoluto.

Si no pasa, use `SNR_REGIME_LIMITED`, `SNR_NOT_IDENTIFIABLE` o `SNR_REJECTED`.
Nunca emita un estimador general a partir del promedio entre perturbaciones.
En datos reales la unica etiqueta sigue siendo
`MODEL_CONDITIONAL_SNR_ESTIMATE`.

### C176 - denoising por operador y regimen

Derive las decisiones con el contrato C171 y solo la reserva fresca. Para
`LAB_CALIBRATED` deben cumplirse simultaneamente mejora, no inferioridad,
preservacion de eventos/extremos, limite de retardo, fuga de senal al residual y
costo. Un promedio favorable no puede ocultar una familia o semilla destruida.

Publique `LAB_REJECTED`, `REGIME_LIMITED`, `NOT_IDENTIFIABLE` y
`UNDERPOWERED` con igual detalle que los pases. Mantenga la rama cruda incluso
si un operador pasa.

### C177 - auditoria wavelet repetida en la ruta de laboratorio

Para `trailing_haar_threshold`, compruebe sobre las unidades realmente
evaluadas:

* todo prefijo y cada nivel;
* cambios adversariales del sufijo;
* fronteras de warm-up y potencias de dos;
* batch, step, chunk y restart;
* que ninguna compensacion de retraso mueve la salida al pasado;
* que `wavelet_mad` solo produce una cifra agregada de TRAIN.

Una sola diferencia anterior a la muestra perturbada invalida todo el root, no
solo esa fila.

## 6. OLAP, revision y cierre

### C178 - carga aditiva y trazabilidad

Ensaye en una base desechable y cargue despues:

* estimaciones de recursos v2 por bloque;
* cobertura v2;
* reanalisis historico separado de confirmacion fresca;
* resultados SNR y denoising por unidad;
* decisiones, no decisiones y costos;
* terminales y recibos de los tres roles.

Segunda carga idempotente, historia intacta, cero fila silenciosamente
deduplicada. El loader debe permanecer activo y sin reinicios.

### C179 - submission de revision, no auto-licencia

Prepare una submission que ligue diseno, codigo, contratos, seed tape, roots,
ledgers, decisiones y carga OLAP. El candidato no crea record de Musashi ni
convierte su submission en autoridad. `df_consumption_gate.py` debe seguir
rehusando sin revision externa.

### C180 - salud operacional

Al terminar, publique salud por roles. Declare `WORKER_B/GPU1` cuarentenada
mientras `nvidia-smi` no pueda obtener su handle y existan errores de progreso
en el kernel. No intente repararla ni la use en esta orden. `logrotate` queda
como tarea del owner y no puede bloquear D2.

### C181 - POST independiente

El POST debe ejecutarse desde tips finales y raices fisicas en procesos nuevos.
Debe incluir todos los PRE C166, mutantes estructurales, recuentos fisicos,
comparacion C137->reanalisis, prueba de reserva no vista, adjudicaciones frescas,
carga OLAP e idempotencia.

Pruebas de aceptacion minimas:

1. no existe interruptor publico capaz de omitir una guardia causal;
2. los 17 mutantes estructurales muerden por separado;
3. un operador con fuga de una muestra no puede publicar evidencia;
4. cada bloque ADF/KPSS tiene identidad distinta;
5. `REFUSED != FAILED != NOT_APPLICABLE != NOT_RUN` en codigo y cubo;
6. la paridad PCA se cumple bajo una tolerancia y representacion canonica
   predeclaradas, no bajo una igualdad falsa;
7. ninguna decision C137 historica atraviesa el gate D2;
8. ninguna decision fresca se deriva de calibration ni del promedio de filas;
9. cada resultado se rederiva desde arrays y contratos ligados;
10. ninguna GPU, D3, selector, modelo, RL, DOIN o live fue tocado.

### C182 - estado documental

Actualice el estado ejecutivo con hechos, no con nombres de archivos. D0-D1
permanecen aceptados; D2 queda como evidencia candidata hasta la revision de
Musashi; D3 sigue cerrado.

### C183 - faltas propias y retorno

Liste al inicio del packet todo defecto, corrida fallida, mutacion que no mordio,
conteo corregido o diferencia de diseno. Cite siempre el conteo del tip final.

### C184 - alto obligatorio

Detengase en:

```text
D2_CURRENT_API_FRESH_CONFIRMATION_READY_FOR_MUSASHI_REVIEW
```

No ejecute D3 aunque todos los resultados sean favorables. Musashi debe revisar
la evidencia fresca y emitir una decision separada.
