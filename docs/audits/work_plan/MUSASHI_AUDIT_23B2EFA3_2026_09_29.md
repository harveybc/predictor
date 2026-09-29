# Dictamen parcial de la solicitud 23b2efa3

Musashi, 2026-09-29. Revision independiente del numero AG News, integracion
QRM01 y diseno QRM02. No es una certificacion global de los diecisiete tips.
No cambie servicios, limites, reservas, modelos ni pesos; no lance GPU.

## Hallazgos

### F1 - Alto: el supervisor admite evidencia ajena como coste de este intento

`tools/df_cell_scope.py` en c1033dc6, lineas 579-634: lee cualquier JSON
preexistente de record_path. MEASURED y salida cero bastan para
usable_for_costing; no exige identidad de celda, intento, etapa, reloj, limite
ni reserva confirmada. La comparacion de scope es condicional: si no vio el
scope o falta el inode del hijo, no impide aceptar.

Probe independiente, launcher sustituido por un doble sin carga:
- registro anterior de otra celda, reloj 1, sin scope observado ni lease:
  usable_for_costing=true;
- pico -1 con status MEASURED: usable_for_costing=true.

No demuestra que los picos historicos sean falsos. Refuta la capacidad de esta
compuerta para certificar la atribucion que declara. Exigir contrato de intento
fresco, tipos y dominios, identidad completa y evidencia de reserva por una ruta
que tambien funcione para hijos breves; ausencia no es coincidencia.

### F2 - Alto: el productor real y el consumidor no hablan el mismo esquema

`df_e1_block.py:820` escribe host_ram dentro de cell_scope y `:973` entrega el
cell.json entero a supervise. Este busca host_ram en la raiz (`:604`). El probe
con la forma exacta anidada produce UNKNOWN y usable_for_costing=false aunque
contenga el pico. La prueba de scopes aislados no cubre esta integracion.

Arreglar productor/consumidor explicitamente por version de esquema y probar
run_units -> child -> cell.json -> supervisor -> CELL_SCOPE con un workload
pequeno. No aceptar alternativamente cualquier campo que tenga el mismo nombre.

### F3 - Alto: el piloto propone telemetria de otro framework y otra colocacion

El modelo de Q2 es TensorFlow/Keras (`df_e1_block.py:602-625`, entrenamiento
en :693). `df_cell_scope.py:285-319` consulta exclusivamente el asignador de
PyTorch. QRM02 en 1fb6b387 se basa en esas APIs para vigilar 12 GiB. El asignador
de PyTorch no contabiliza las asignaciones TensorFlow. Ademas `run_units:976`
fuerza CUDA_VISIBLE_DEVICES vacio; sellar ese runner no establece un piloto GPU.

Declarar y verificar el dispositivo realmente usado, mantener la receta de
TensorFlow y medir su asignador con una API apropiada; memoria del proceso y
telemetria total del dispositivo, si se usan, llevan alcance separado. Una API
no disponible produce UNKNOWN. No importar PyTorch solo para medir TensorFlow.
No portar el modelo de framework como supuesto arreglo operativo.

Una comprobacion posterior de estadisticas no puede prometer abortar ANTES de
una asignacion que excede el umbral: corregir tambien esa frase del diseno.
CPU/wall y cada etapa necesitan una parada ejecutable, no solo una tabla.

## Lo que si sobrevive

Reconte desde per_row del artefacto nativo de 19a37baf, sin usar sus agregados:
400 identidades distintas, diagonal 381, accuracy 0.9525 y macro-F1
0.946754652762913. Consulte directamente la fuente del autor en
https://raw.githubusercontent.com/NandhaKishorM/laya/010bacef/research/results/app_benchmark_results.json
que publica 0.9525 y 0.9468. Igualdad exacta de accuracy y coincidencia de F1 a
precision publicada, confirmadas. La fuente dice in_training=true.

Esto acredita el recuento y la coincidencia con la referencia, no una nueva
inferencia independiente, igualdad de predicciones del autor (no publicadas),
generalizacion retenida ni utilidad financiera. No avalo el superlativo
"primera de toda la campana" sin censar toda su historia.

Weather: el retorno f328db3a distingue correctamente replay y autorizacion;
leer cero filas resuelve la pregunta sobre custodia, no la falta de custodia.
No repetido ni certificado independientemente el replay en esta revision.
Traffic: paridad del evaluador sobre pesos entrenados no es reproduccion de
calidad del modelo. La horquilla 12-15 h sigue siendo proyeccion, no cota
garantizada por dos horizontes. Tampoco reejecutada aqui.

Dos picos distintos por si solos NO prueban scopes distintos: el mismo scope
puede tener distintos maximos en dos instantes. Identidad, membresia, vida del
scope y reserva son la prueba relevante. No retiro la observacion de scopes
distintos reportada; retiro esa inferencia como demostracion suficiente.

## Recursos y disposicion

Lectura SSH no mutante del trabajador secundario en esta revision:
MemoryCurrent=2690031616 B, MemoryMax=15032385536 B (14 GiB),
MemAvailable=21626597376 B. Una lectura posterior de memory.stat da
file=2506440704 B, shmem=806244352 B, anon=0, kernel=183590912 B.
Son instantaneas distintas; shmem es parte de file, no se suma otra vez.
El residual ya no es 4.9-5.1 GiB y no todo file es cache descartable.
No cambie el techo ni hice reclaim ni borre /dev/shm.

NO recomiendo aprobar QRM02 contra c1033dc6 hasta F1-F3 reparados; despues
reevaluar el pedido finito de 4800 CPU s/4800 s pared, sin pedir al propietario
decisiones tecnicas de esquema. No se concede asignacion aqui.
La solicitud de 18 GiB sigue SIN APROBAR: volver a admitir contra el estado
actual, identificar residual y su propietario; nunca borrar memoria compartida
ni reducir una reserva para pasar. Una excepcion requeriria limites agregados,
reserva de escritorio y restauracion del techo probadas y autorizacion expresa.

Para picos por etapa, la documentacion del kernel indica que el reset de
memory.peak afecta a lecturas por el MISMO descriptor abierto. Un write_text
seguido de read_text abre otro descriptor y no establece ese experimento.
Fuente: https://docs.kernel.org/admin-guide/cgroup-v2.html (memory.peak).
Agregar prueba del descriptor y conservar el watermark de vida por separado.

## Reproduccion y alcance

`python docs/audits/qrm_scope_probe_20260929.py`: tres contraejemplos y recuento;
resultados en el JSON adyacente. Fuente del instrumento cargada desde c1033dc6.
No launcher real, no API de broker, no GPU, no mutacion de servicios.
Suite original: 29 passed, 2 deselected (`-k 'not real_launcher'`), 4 segundos.
Los tres contraejemplos sobreviven esa suite. El probe usa dobles de proceso y
observador para aislar la compuerta, no simula que hizo una medicion de kernel.

## Ordenes paralelas de reparacion y avance

1. Agente A: F1/F2, congelar probe, tests rojos primero, integracion real pequena.
2. Agente B: F3 y QRM02 ejecutable, contra el runner corregido de A. Mientras
   tanto preparar costes Traffic por los horizontes restantes, sin suponer
   monotonia ni pedir otra campana completa; respetar asignaciones existentes.
3. Agente C: asumir propiedad del defecto de identidad por fila de metricas en
   CB04/warehouse. Una metrica secundaria no hereda la identidad de la primaria;
   separar clase de evidencia y evitar doble conteo entre terminales. Usar
   almacen desechable y sucesores, no reescribir historia. BANKING77 sigue como
   siguiente ensayo de clasificacion bajo su propia autoridad y admision.
4. Agente D: asumir el caso fixture en contrato productor/almacen: conservar
   pruebas declaradas como tales, impedir promocion como modelo real o badge.
   No prohibir una palabra aislada como sustituto de validar procedencia.

Satoshi integra cada entrega sin parar carriles independientes. No nueva
campana doctoral ni financiera autorizada por este dictamen. El diseno del
barrido corto/largo de heuristic-strategy permanece un carril separado, sin
reemplazar la estrategia original ni confundir error igual al naive con sus
predicciones. No anunciar agentes activos hasta despacharlos y obtener su id.
