# I6-N: trayectorias, swarm, semanas completas y archivo externo

Estado: IMPLEMENTACION_Y_EJECUCION_AUTORIZADAS, 2026-10-11.
Este subplan sustituye las autorizaciones pendientes y el orden secuencial
N1/N2/N3 del borrador I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md. No reescribe
mediciones anteriores ni certifica componentes todavia no desplegados.
La solicitud actual del propietario autoriza a Satoshi a implementar,
integrar y ejecutar, sin otra visita para cada etapa lista.

## 1. Punto de partida y prioridades

Base de codigo: predictor `codex/i6n-integration-20261011@259e94df`.
Contrato superior: BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md.
Primero fidelidad al negocio; despues datos seleccionados y soporte; despues
busqueda conjunta y evaluacion. No reabrir seleccion, denoising, NEAT o RL.
48 candidatos endpoint terminaron; son ingenieria historica, no un ano
validado. Sus checkpoints, identidades y recibos son reutilizables solo cuando
coincidan todos los componentes efectivos de un nuevo trabajo. No repetirlos
por cambiar de nodo, nombre de directorio o representacion del genotipo.

## 2. Contrato cientifico fijo

- EURUSD y las 20 features RAW retenidas. Seleccion informada por 2024:
  desarrollo retrospectivo declarado, nunca confirmacion independiente.
- Cortos [2,3,4,5,6,12] h; largos [18,24,36,48,60,72] h. Emision horaria.
- Reajuste semanal, cuatro anos calendario precedentes por ajuste, purga de
  labels no conocidas al cutoff, inner temporal y escalado TRAIN-only.
- Cierre forzoso antes de cierre de mercado menos 12 h; entradas desde
  apertura mas 12 h, sin nuevas entradas desde jueves 00:00 del calendario
  configurado. Calendario nominal no pasa a certificado por decreto.
- C1 gobierna origen, target, misma semana operativa y mascara individual.
  No eliminar labels cortas validas porque falten largas. Ventana incluye t;
  puede conservar observaciones pasadas de periodos donde no operamos.
- Naive para cada horizonte, precio original, mismos origenes y mascaras.
  Skill=(MAE_h-naive_h)/naive_h. S_h separado de U_h; U_h requiere consumo
  observado de la estrategia, no se inventa a partir de oportunidades S_h.
- Seed 6102026; una repeticion. Early stopping: media TRAIN/inner, patience 5,
  min_delta 1e-6, 50 epocas maximas, restauracion del mejor checkpoint.
  No usar la semana exterior para stopping o fitness del candidato.
- TEST 2025 cerrado. No datos TEST para ajustar dimensiones, vocabularios,
  calendarios aprendidos, escalado, seleccion o identidad del experimento.

## 3. Espacio conjunto, desde pequeno

No identificar la mejor ventana con nucleo fijo y despues congelarla.
Historia, resolucion, transformacion, familia, profundidad, filters por capa,
reduccion temporal, adaptador y supervision se optimizan conjuntamente.
El evaluador y la identidad contienen TODOS los genes activos; los inactivos
se canonicalizan para evitar alias. Un escalar t+h renombrado no es otro fit.

Ramas iniciales fijas: Conv1D causal, filters=16, kernel=3, stride=1;
fusion por features y posicion inmediatamente despues de fusion.
Nucleo minimo inicial: entrada Conv1D filters=8, una reductora Conv1D filters=4,
stride=1, sin Transformer ni LSTM, cabezal temporal final. Es una semilla
barata, no un ganador ni un limite permanente de capacidad.

| Familia de parametros | Dominio autorizado |
| --- | --- |
| Historia L | max(24,m*h), m en [1,2,4,6], deduplicando equivalentes |
| Separacion entrada dx | divisores horarios enteros de h y L; incluye 1 |
| Muestras | 1+L/dx, incluyendo t; sin suavizado ni interpolacion |
| Entrada | identity o difference, con mascara y unidades por feature |
| Supervision | endpoint o trayectoria real [dy,2dy,...,h] |
| dy independiente de dx | h/k para k en [2,3,4] si es entero; endpoint equivale a k=1 |
| Entrada nucleo | Conv, Transformer, Conv+Transformer |
| Filters/d_model inicial | [8,16,32,64,128], condicionado por coste |
| Bloques Transformer | 0 para Conv; 1 o 2 para rutas Transformer |
| Reductoras | 1 a 3; filters_i en [4,8,16,32,64,128], no crecientes |
| Strides temporales | 1 o 2 por capa, right-anchored; ninguna/progresiva/final |
| Adaptador | ninguno, Conv causal o LSTM return_sequences=True |
| Ancho adaptador | [8,16,32,64], condicionado por coste |

Usar las reglas del modelo para rechazar reducciones que pierdan t o vacien
la secuencia. No truncar a 241 muestras: probar contratos hasta 433 antes
de admitir esa clase. No mezclar filtros (features) con longitud temporal.
Los rangos son explorables, no una afirmacion de que todos caben en memoria.
Empezar por clases <=25 muestras/ancho<=32; despues <=55/ancho<=64; finalmente
horarias mayores, hasta 433/ancho128, solo con piloto de coste admitido.
Cada ampliacion mantiene libres los genes anteriores: no congela la ventana.

N1 de capacidad es la inicializacion de la misma busqueda, no una seleccion
irreversible previa. No cruce obligatorio: mantener la implementacion
existente 80% mutacion condicional / 20% inmigracion, minimizacion DEAP.
No sustituir DEAP por NEAT ni fingir que el adaptador local es un swarm.

## 4. Secuencia automatica y presupuestos

1. Pruebas de trayectorias, recuperacion DOIN y custodia; fixtures pequenos
   etiquetados. Medir coste con datos TRAIN reales antes de ampliar clases.
2. Screening de los 12 horizontes: maximo 24 candidatos efectivos NUEVOS por
   horizonte, poblacion hasta 8, 3 generaciones, una semilla. Inicializar con
   el nucleo minimo y conservar pequena alternativa. Primero downsampling,
   despues ampliar a horario sin producto cartesiano exhaustivo.
3. Cada candidato se puntua en cuatro folds internos predefinidos por
   trimestre de 2023, como el diseno anterior: ultimo cutoff de mercado
   completo del trimestre; 28 dias internos anteriores al cutoff, con purga.
   Reajuste y cuatro anos anteriores por fold. Fitness = media no ponderada
   entre folds de 0.5*(MAE_TRAIN/naive_TRAIN + MAE_inner/naive_inner).
   Usar sumas/conteos dentro de cada split, no promediar batches.
   Maximo 12*24*4=1152 ajustes nuevos; cache verificada reduce el total.
   Si faltan historia o procedencia, materializarlas/contrastarlas antes de
   ese fold: no reducir anos, reemplazar por una semana de 2024 o saltarlo.
4. Traza por salida: escalas ajustadas solo TRAIN, pesos iguales entre heads
   validos; el score de busqueda compara exclusivamente el endpoint h sobre
   soporte congelado por fold/h. Intermedios tienen sus propios naives y
   quedan como diagnostico. No permiten cambiar filas del endpoint.
5. Finalistas: mejor fitness completo y control minimo (o segundo candidato
   pequeno no equivalente). Conservar todas las semanas 2024 elegibles, con
   refit semanal y pesos iniciales aleatorios por defecto. Maximo 2*12*53=1272
   disposiciones; generar el numero real desde calendario, no asumir 48.
   Registrar tambien media semanal, pero MAE anual principal es suma(error)
   / suma(n), naive identico; no declarar anual un brazo con huecos sin resolver.
6. Contraste direct multi-output: corto y largo separados, presupuesto del
   plan anterior de hasta 212 disposiciones, tras el coste e integracion del
   punto 5; no llamar trayectoria auxiliar a la familia de horizontes de negocio.

La configuracion fija tolerancia de mejora absoluta de fitness 1e-4 y paciencia
de busqueda 3 generaciones; el presupuesto siempre limita. No prometer un
optimo global con 24 candidatos. Si la generacion admisible se agota, cerrar
con la razon, sin rellenar con duplicados. Declarar cualquier falta de soporte.
El comparador ARIMA posterior conserva sus hasta 636 disposiciones del plan;
no ocupa una GPU ni bloquea la implementacion del pronosticador.

Todos los pasos tienen plan/check/start/status/collect y estado durable.
La terminacion de una etapa activa la siguiente si pasan sus pruebas y coste,
no espera a un agente. Etapas independientes continuan ante un fallo localizado.
Planear el coste completo y publicar ETA por throughput de clases medidas,
con intervalo y marca unknown cuando no haya medicion, nunca ETA inventada.

## 5. Swarm y cortes de energia

Un nodo desaparecido no autoriza dos fits concurrentes del mismo candidato.
Claims con lease y fencing token, checkpoint y resultado durable; al recuperar:
adoptar terminal verificado; reanudar checkpoint completo si incluye optimizador,
RNG, epoch/stopping; si no existe, marcar INTERRUPTED y aplicar la politica
explicita de un unico intento de recuperacion, no bucle de reentrenamiento.
Ese intento es necesario por fallo externo, queda ligado al anterior, cuenta
en el limite de intentos y nunca se reporta como una replica independiente.
Maximo tres intentos fisicos por identidad incluso con fallos externos; una
sola medicion cientifica. OOM no autoriza retry automatico con menor cap.
El reinicio no reinicia presupuesto, identidad o generacion. Servicios
instalados y habilitados, no solo unidades transitorias perdidas al reiniciar.
No fall back silencioso de GPU a CPU. No abrir broker ni TEST al arrancar.

## 6. Cadena liviana y OLAP

Modo objetivo `external_payload`: cuerpo canonico completo (incluidas metricas)
durable en data-lake/objeto por contenido, despues compromiso en cadena con
hash del cuerpo, hash del manifiesto, locator y campos de consenso/enlace
previo necesarios. Hash de cuerpo, manifiesto y bloque/header son distintos.
No incluir en el hash del cuerpo su propio hash/header final de forma circular.
No quitar firmas ni validaciones del protocolo de DOIN para ahorrar espacio.

Antes de aceptar bloque/referencia: PUT durable -> GET exacto -> comparar
digest -> commit del header/referencia. Un registro de catalogo no es bytes.
ETL recibe referencia finalizada, relee bytes, verifica cadena y manifiesto,
proyecta metricas en warehouse y relee. Inbox/outbox idempotentes, claves de
experimento/candidato/fold/h/head/rol/poblacion/split/metrica/unidad/version,
id de bloque y estado de finalidad. Reorg revoca la proyeccion canonica sin
borrar el historial. Hash prueba integridad, no disponibilidad ni veracidad.

Datos para meta-analisis: genotipo canonico y config efectiva, horizonte/dx/dy/L,
capas y filters, identidad de features y fuente, calendario, TRAIN/inner,
seed, pesos/stopping, error y naive con sumas/conteos, S_h/U_h, costes/recursos,
intentos/fallos y parentesco, hashes y procedencia. Candidatos fallidos tienen
disposicion, no score inventado. El metaoptimizador queda futuro: estos datos
no autorizan entrenarlo con TEST ni contaminar evaluaciones retenidas.

Destino: servicio OLAP desplegado existente, contrato `gov_metric` y metadatos
de DOIN versionados; no crear otro cubo incompatible o SQLite de juguete como
sustituto. Migraciones aditivas probadas primero en instancia desechable.
Las metricas locales/outbox pueden producirse mientras se integra el archivo;
no se afirma publicacion on-chain hasta verificar la cadena real.
Snapshot analitico fisico local y copia verificada en segundo host. En Git
solo codigo, configs sin secretos, manifiestos y resumenes; NO binarios de
datos/OLAP ni uploads de releases grandes. No borrar resultados retenidos.

## 7. Recursos y autonomia

Priorizar 5090. 4090 despues de comprobar que el reinicio pendiente resolvio
NV_ERR_NO_MEMORY/slab y que cumple admision. 4070 del coordinador puede hacer
piloto acotado pequeno solo si deja 8 GiB de RAM para escritorio/servicios y
2 GiB de VRAM libres; no convivir con CPU pesada del mismo host.
5070 Ti NO se activa por defecto: comparte RAM con 5090. Solo admitir ambas
si suma de reservas completas (1.25*picos medidos de clases, mas envoltorios)
cabe en host y slice con reserva del host intacta. No aprender esa cota
provocando OOM. Si aparece presion, dejar de asignar a 5070 y drenarla;
el watchdog interrumpe su unidad identificada si se llega al limite duro.
No matar procesos por patrones amplios ni tocar escritorio/broker/VM.

La caida de un worker no paraliza pruebas CPU, ETL o al otro worker. Agentes
escriben y verifican codigo en worktrees; scripts gestionan ejecucion y ETA.
Ningun agente hace sleep/poll de logs. Reportar al terminar un entregable o
cuando una dependencia humana irreducible lo requiera, sin pedir aprobaciones
ordinarias ya concedidas. No eludir controles de acceso.

## 8. Despues de este frente

Seguir master: R0/R1/R2, nucleo preentrenado, estrategia cronologica, SAC/DQN
con el mismo calendario/costes y comparadores de negocio. Registrar como
futuro, sin comenzar ahora: expertos de apertura/cierre por dia de la semana,
especialistas por regimen, nuevas heuristicas y metaoptimizacion. Exigen
comparador compartido, cobertura por dia y prueba de valor fuera de muestra.
