# Estado de ejecución vigente

Observado: 2026-10-03 04:31-04:35 UTC. Autoridad: plan maestro v3 consolidado.
Este archivo reemplaza el snapshot del 30 de septiembre.

## Corrección de rumbo

La selección de características no está terminada. Las 321 columnas de ECL y las
83 de la tabla ETH son inventarios/admisibilidad, no conjuntos seleccionados.
Las campañas que usaron NEAT como optimizador de hiperparámetros quedan
diagnósticas y fuera de la cola. NEAT sólo volverá después de una representación
final congelada, como cabezal comparado contra Dense.

El estado M06 observado aún anuncia colas NEAT históricas y una orden
`plan_revision=b327b771`. Sus procesos no existen en gamma y la cola no prueba
trabajo vivo. Este archivo y EXPERIMENT_EXECUTION_QUEUE.json las sustituyen.

## Máquinas observadas

| Host | GPU | Observación | Trabajo científico GPU |
|---|---|---|---|
| omega | RTX 4070 8 GiB, 50 C, 57 %, 1.0 GiB | actividad de escritorio/servicios; 15 GiB RAM disponible | ninguno identificado |
| gamma | RTX 5070 Ti 12 GiB, 29 C, 0 % | 11 GiB RAM disponible | ninguno |
| gamma | RTX 5090 32 GiB, 38 C, 0 % | comparte RAM con 5070 Ti | ninguno |
| dragon | RTX 4090 16 GiB, 32 C, 0 % | 13 GiB RAM disponible; VM MT5 no está en uso | ninguno |

Servicios de operación observados: Alpaca model runner en omega; bridge/model
runner MT5 en dragon, pero **no la VM MT5**; campaign supervisors en los tres
hosts. Hasta que ramas y H-CORE preentrenados estén listos para inferencia, no se
reserva RAM para esa VM: dragon queda disponible para CPU/GPU experimental. No
se infiere rentabilidad ni actividad de órdenes de la existencia de servicios.

## Estado científico

| Frente | Estado honesto | Siguiente objeto |
|---|---|---|
| Selección financiera EURUSD | NOT_STARTED como manifiesto completo | contrato de fuentes/targets/folds y PS0/PS1 |
| Selección ETH | 83 features TRAIN-only; selección incompleta | tabla versionada selected/rejected/pending provisional |
| Escalera causal | componente de factibilidad reportado; no ganador | estudio EURUSD de tres peldaños |
| Extractibilidad | AE piloto previo, no matriz por superviviente | plugin univariado y controles raw/random/trained |
| Modular | motor/adapter en integración; selección upstream ausente | controles ARCH después del manifiesto |
| E1 R0/R1/R2 | evidencia histórica sobre diseño anterior | campaña nueva sólo con ramas seleccionadas |
| H-CORE | no elegible | esperar prefijo ganador de E1 |
| DOIN | integración disponible; sin ganador del diseño vigente | optimizar candidato elegible después de E1 |
| NEAT | campañas incorrectas detenidas | esperar representación final congelada |
| RL | integración previa; ningún resultado real nuevo aceptado | SAC/DQN raw vs modular tras selección |
| Literatura | ECL/Weather/Traffic retenidos | verificación CPU o celdas fieles independientes |
| M5PHET | clasificación/forecasting parcial | siguiente proveedor real E2E independiente |
| Trading | servicios paper/demo vivos | consumir sólo predictor financiero naive-eligible |

## Despacho inmediato paralelo

1. **CPU-A, prioridad crítica:** EURUSD PS0/PS1: fuentes pagadas y públicas,
   disponibilidad, targets, folds y matriz básica.
2. **CPU-B:** PS2: redundancia y relevancia por Y_s/Y_l/Y_b, manteniendo muestra
   de exploración e interacciones.
3. **CPU-C:** PS3-C: escalera causal EURUSD, con episodios de calendario,
   ajuste/soporte, placebos y SCM temporal.
4. **GPU-5090:** PS3-R: coste y primer extractor univariado sobre supervivientes
   provisionales, raw/random/trained; una semilla.
5. **GPU-5070 Ti:** familia alternativa de extractor sobre las mismas filas si
   la RAM compartida admite ambos; si no, preparar datos y tomar el relevo.
6. **GPU-4090:** referencia pública fiel o extractor alternativo independiente;
   después, controles ARCH cuando exista manifiesto.
7. **GPU-4070:** replays pequeños y verificaciones; proteger escritorio.
8. **Ingeniería:** compuerta fail-closed del manifiesto, integración
   feature-extractor y contratos de warehouse.
9. **Operación:** mantener shadow/paper; no promover modelos sin naive positivo.

## Gate siguiente

El siguiente hito es un manifiesto financiero versionado que compare cinco
conjuntos: todas admisibles; predictivo/redundancia; +causal; +extractibilidad;
y preferencia generativa opcional. Sólo entonces se fijan ramas definitivas y
comienza ARCH/E1. `NOT_IDENTIFIED` no se convierte en rechazo automático.

ETA se emitirá después del primer lote medido de PS1, PS3-C y PS3-R. Antes de
esos pilotos, una fecha total sería inventada.
