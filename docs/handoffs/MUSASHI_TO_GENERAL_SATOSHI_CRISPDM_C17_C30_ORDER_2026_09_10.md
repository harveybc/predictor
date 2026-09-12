# Orden Musashi a General Satoshi - CRISP-DM C17-C30

**Fecha:** 2026-09-10

**Base obligatoria:** `predictor@0e27f3c` y los cinco tips hermanos declarados
en su retorno C1-C16

**Dictamen:** `docs/audits/MUSASHI_AUDIT_CRISPDM_C1_C16_RETURN_2026_09_10.md`

## 0. Objetivo

Cerrar la reja como protocolo ejecutable de dos fases, ligar todos los datos y
todo el codigo realmente consumidos, convertir OLAP en un flujo terminal
continuo y versionado, y comenzar el siguiente trabajo CPU de disponibilidad y
caracterizacion sin abrir seleccion cientifica, confirmacion, GPU, live ni
publicacion DOIN.

Los componentes aceptados en el dictamen no se reescriben. En particular: no
rehacer el primer censo completo salvo por una prueba dirigida; no alterar la
poblacion 1.680/1.965; no borrar historia OLAP; no reabrir MTM; no presentar
DOIN L2 como implementado.

## 1. PRE obligatorio

Antes de editar, congelar con APIs publicas y salidas exactas estos
contraejemplos:

1. cambiar solo bytes de `y_validation_file` deja iguales sujetos,
   `data_digest` y `partitions_digest`;
2. un optimizador devuelve una clave de datos o plugin y esa clave llega al
   pipeline despues de la reja;
3. una submission revisada en la primera invocacion no puede gobernar la
   segunda porque `submitted_at` se regenera;
4. manifest con campo top-level y campo de entrada no declarados es aceptado;
5. envelope con `NaN` y campos anidados extra es aceptado;
6. una excepcion del pipeline produce cero entrada terminal en el outbox;
7. no hay loader OLAP persistente activo ni heartbeat productivo;
8. una nueva version de una aparicion/variable con el mismo id logico queda
   ignorada por `ON CONFLICT DO NOTHING`;
9. cambiar el modulo de preprocesador o predictor no cambia el digest de codigo
   revisado;
10. T2/M4/B4 no tienen emision terminal en su siguiente ejecucion.

El PRE se compromete antes de cualquier correccion. No insertar paths privados,
identificadores de maquina ni credenciales en evidencia publica.

## 2. P0 - reja y autoridad

### C17 - protocolo real de dos fases

Implementar una unica secuencia productiva:

1. `SUBMIT_ONLY`: deriva los datos y codigo, escribe una submission
   content-addressed y sale antes de optimizador/pipeline;
2. el revisor externo produce su record sobre esos bytes;
3. `EXECUTE_REVIEWED`: vuelve a derivar todos los hechos, exige igualdad exacta
   con la submission persistida y consume el record;
4. solo entonces se permite continuar.

La fase ejecutante no genera una submission con un timestamp nuevo. Submission,
record y manifest deben tener identidades separadas. Una submission y un
template conceden cero autoridad. Mantener el replay historico expreso.

### C18 - conjunto consumido completo

Derivar y ligar por rol:

- todos los `x_*` y `y_*` realmente declarados;
- encabezado, columnas y digest de cada archivo;
- target separado cuando vive en `y`;
- digest canonico de datos sobre la lista ordenada completa;
- digest canonico de particiones que incluya `x`, `y`, rol y contrato temporal.

Un `y` mutado, omitido, renombrado o intercambiado debe cambiar identidad o
rehusar. No asignar el mismo digest parcial a todos los sujetos.

### C19 - contrato inmutable despues de la reja

El resultado del optimizador solo puede modificar una lista explicita de
hiperparametros. Claves de datos, particiones, target, plugins, manifest,
autoridad, codigo, output de evidencia o alcance quedan prohibidas.

Antes del pipeline, rederivar la identidad completa y exigir igualdad con la
submission revisada. Probar con optimizador real o doble que `x_test_file`,
`target_column` y `preprocessor_plugin` son rechazados antes de consumir datos.

### C20 - identidad del codigo realmente ejecutado

La submission debe ligar la resolucion efectiva de entry points y los bytes de
los modulos cargados para predictor, optimizer, pipeline, target, preprocessor,
data handler, config merge y reja. Un cambio en cualquiera debe cambiar la
identidad. No usar una lista de cuatro archivos como sustituto del grafo real.

La identidad debe ser reproducible en otro checkout limpio del mismo commit.
Si un modulo no puede localizarse o hashearse, rehusar tipado.

### C21 - schemas exactos y tipos

Aplicar un parser JSON comun que rechace claves duplicadas y constantes no
finitas, y schemas exactos recursivos para:

- manifest y cada entrada;
- submission y record;
- temporal availability, digests, costos, parametros e IO schema;
- envelope, identidad, particiones, presupuesto, terminal, consumo, artefactos
  y unidades;
- artefactos originales de T2/M4/B4.

Todo digest debe ser 64-hex minusculo canonico. Numeros rechazan bool, NaN e
infinito. Fechas son UTC canonicas y no futuras. Campos desconocidos rehusan.

## 3. P0 - OLAP terminal continuo

### C22 - emision para todo desenlace

Encapsular el run para que exactamente un envelope terminal durable sea emitido
en `COMPLETE`, `FAILED`, `INCONCLUSIVE`, `REFUSED` o `QUARANTINED`, incluso si
optimizer, pipeline, scoring o persistencia de resultados falla.

El envelope debe contener el tipo y fase del fallo sin convertirlo en resultado
cientifico. Una falla local del propio outbox debe dejar una brecha operacional
tipada junto a los resultados, no solo un `print` que desaparece.

### C23 - productores en el punto terminal

Integrar el outbox en los puntos terminales futuros de predictor, T2, M4 y B4.
El backfill actual se conserva como historia; no cuenta como integracion viva.
Cada unidad/candidato terminal debe aparecer con su identidad y costo, no solo
un resumen de campana con `units=[]`.

Cuando no exista un nuevo call site DOIN, conservar
`TRADING_L2_PUBLICATION_PATH_NOT_IMPLEMENTED` y no fabricar uno en esta orden.

### C24 - loader persistente activo

Entregar y activar un servicio CPU supervisado para drenar el outbox. Debe:

- iniciar automaticamente y reiniciarse solo si termina inesperadamente;
- tener un unico consumidor efectivo;
- publicar heartbeat, atraso, pendientes, cargados y fallidos;
- sobrevivir PostgreSQL no disponible sin perder ni duplicar entradas;
- no reiniciar PostgreSQL ni Metabase;
- usar configuracion externa ya existente, sin credenciales en Git.

La aceptacion exige evidencia de proceso real activo, dos heartbeats separados,
una entrada cargada de extremo a extremo y recuperacion tras una base desechable
temporalmente no disponible. No usar el cubo real para la prueba de caida.

### C25 - inventario OLAP versionado

Sustituir `ON CONFLICT(logical_id) DO NOTHING` como modelo de historia. Una nueva
observacion del mismo id con distinto census/index digest debe insertarse como
version nueva o supersesion aditiva. Exponer una vista actual determinista sin
borrar versiones anteriores.

Repetir la misma version debe ser idempotente; cambiar disponibilidad, digest,
semantica o pertenencia debe producir una version nueva. Preservar 39/1.404 y
las 60+60 filas actuales.

### C26 - envelope ligado al productor

Los builders consumen bytes una vez con el parser estricto y ejecutan el
verificador propio del productor, no solo una lista de campos y un self-digest.
El envelope liga el archivo, el veredicto rederivado y la identidad del codigo
que lo rederivo. B4 sin artefacto permanece no gobernante.

## 4. P1 - disponibilidad y contrato comun

### C27 - alcance financiero derivado

Resolver el hueco `observation_contract` versus `feature_columns` mediante un
artefacto de puente derivado de:

- contrato de observacion v2 ratificado;
- headers fisicos de las vistas activas;
- manifests y codigo productor de cada familia;
- tareas supervisadas y RL actualmente registradas.

No editar configuraciones para fingir coincidencia. El puente debe demostrar
la correspondencia columna -> variable conceptual -> aparicion -> familia ->
contrato temporal, o marcarla `UNRESOLVED` con razon.

### C28 - contratos de disponibilidad

Para cada familia activa, derivar de evidencia local los ocho campos del
contrato. Nunca usar event time como available time por defecto. Proveedor,
licencia, revision, timezone y latencia deben venir de artefactos/codigo
identificables.

Primera salida obligatoria:

- familias completas y candidatas a revision;
- familias bloqueadas agrupadas por campo y fuente que falta;
- cobertura exacta de las 94 columnas supervisadas y 83 RL;
- lista minima de documentos que realmente requiere el owner, solo si la
  evidencia no existe en ningun repositorio.

La meta no es forzar un conteo positivo. Cero sigue siendo valido si la
evidencia lo exige.

### C29 - manifest y diseno sucesor

Solo despues de C27-C28, generar:

1. submission de manifest para el conjunto resuelto, no record de revision;
2. indice sucesor ligado al nuevo censo/contratos;
3. diseno de seleccion sucesor ligado a universo no vacio y a la validacion
   anidada existente.

Si el conjunto sigue vacio, la disposicion es
`FINANCIAL_AVAILABILITY_EVIDENCE_REQUIRED`; no ejecutar seleccion.

## 5. P2 - siguiente trabajo cientifico CPU

### C30 - caracterizacion antes de seleccionar

Preparar y, solo para sujetos con procedencia suficiente, ejecutar la primera
caracterizacion por variable del plan data-centric. Debe producir filas OLAP
versionadas, no elegir features ni entrenar modelos pesados.

Por variable y particion de desarrollo, medir como minimo:

- cobertura, missingness, duplicados, irregularidad temporal y revisiones;
- distribucion, escala robusta, extremos y estabilidad por ventana;
- autocorrelacion, estacionariedad descriptiva y contenido espectral;
- entropia/compresibilidad como descriptores empiricos, sin llamarlos capacidad
  cognitiva ni complejidad de Kolmogorov exacta;
- diagnosticos de ruido/SNR solo donde exista referencia, replica o generador
  que los haga identificables;
- costo completo de calcular cada descriptor.

Separar estrictamente:

- sintetico: calibra diagnosticos;
- bancos publicos: evalua transferencia fuera del dominio;
- financiero: desarrollo y revalidacion de dominio.

No usar confirmacion reservada, no puntuar selector, no ejecutar GPU y no
promover operadores. Entregar el ledger de intentos incluidos, fallidos e
inconclusos; los nulos tambien entran al OLAP.

## 6. Bateria minima de aceptacion

1. Los diez PRE rehusan por su causa exacta.
2. Submission estable -> record externo -> ejecucion funciona en dos procesos
   separados; cambiar un byte entre fases rehusa.
3. Mutar cualquier `x` o `y` cambia la identidad.
4. El optimizador no puede cambiar datos, target ni plugins.
5. Cambiar cualquier plugin ejecutado cambia `code_digest`.
6. Manifest/envelope/productor con extra, duplicado, bool numerico, NaN, inf,
   digest no canonico o fecha futura rehusa.
7. Exito, fallo, inconcluso, refusal y cuarentena producen exactamente un
   terminal cada uno.
8. El loader queda activo y carga una entrada nueva sin duplicarla.
9. Un segundo censo distinto crea version OLAP nueva; repetirlo no.
10. Los 39 experimentos, 1.404 performances y 120 unidades actuales quedan
    intactos.
11. La cobertura financiera se deriva; no se auto-declara elegibilidad.
12. C30 escribe mediciones y costos por variable sin emitir seleccion ni usar
    confirmacion/GPU.

## 7. Entrega

Un solo packet de retorno con:

- tips PRE/POST de cada repo tocado;
- salidas exactas de los diez PRE y POST;
- submission y templates no autorizantes;
- inventario de archivos `x/y` y modulos ligados;
- estado y dos heartbeats del loader;
- conteos OLAP antes/despues y prueba de historia versionada;
- cobertura de disponibilidad por familia y variable;
- ledger y resumen de C30;
- defectos propios y limitaciones;
- declaracion explicita de que no se ejecuto GPU, confirmacion, live, venue,
  seleccion cientifica ni publicacion DOIN.

**Stop:** no crear ningun record externo de Musashi. Al terminar, detenerse en
`CRISPDM_C17_C30_READY_FOR_MUSASHI_REVIEW`.
