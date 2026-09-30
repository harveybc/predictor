# Retsu: contabilidad ejecutable, archivo verificable y adquisicion completa

Musashi, 2026-09-30. Suplemento a RETSU_PARALLEL_CONTINUATION_2026_09_30.md.
Leer MUSASHI_RETSU_D158_REVIEW_2026_09_30.md y ejecutar la sonda PRE conservada.
No bloquear carriles independientes. Tres agentes: estrategia, DOIN y preparacion
GPU/clasificacion; Retsu integra y publica incrementalmente. Usar los mismos
worktrees aislados o sucesores, nunca editar bajo servicios vivos.

## A. Estrategia: corregir ejecucion sin cambiar las decisiones

Conservar 781022a y sus cifras como diagnostico legacy sintetico. Crear sucesor
de contabilidad: mismo plugin/variante E, TP/SL y close-only/next-open; reparar
la convencion financiera, no redisenar la heuristica ni elegir parametros por PnL.

Tests primero, con escenarios deterministas long y short:

1. Capital 10,000, fraccion de margen configurable y no superior al 5%, leverage
   100, conversion declarada entre moneda de cuenta, nocional y unidades. Oraculo
   simple con moneda de cuenta igual a cotizada: 500 de margen permiten hasta
   50,000 de nocional antes de costos, NO 50,000 unidades con cualquier precio.
   Minimos/redondeo/costos/gap al fill nunca elevan el margen sobre el limite.
   No subir el capital del fixture para esconder un Margin ni simular fills.
2. Broker y calculo de tamanos comparten esa convencion. Identificar que hace el
   motor existente antes de escoger su API. Probar colateral y liberacion al cierre,
   compras y ventas simetricas; el efectivo acreditado de un corto no es beneficio.
3. Ledger unico de spread, slippage, comision y funding/swap. Definir unidades y
   si la comision es por lado o round-trip. Debitar costos en el reloj declarado
   antes de las decisiones que consumen ese saldo. Reconciliar caja, equity,
   PnL realizado/no realizado y costos, sin doble conteo. En fixture con huecos,
   no llamar horas a conteos de barras; usar timestamps o rollover declarado.
4. Registrar Accepted/Completed/Rejected/Margin/Cancelled y orden en vuelo. Un
   rechazo no deja falsa posicion ni falso cierre. Test de stop() pendiente sigue.
5. Ejecutar otra vez los cuatro brazos sinteticos sobre mismo soporte con el
   sucesor, capital 10,000 y fraccion/leverage declarados; informar diferencias
   de fills frente a legacy. MAE por horizonte, familia y agregado, naive pareado,
   trades, capital, margen, costos, PnL y equity. No exigir ganancia ni monotonicidad.

No correr B0 de mercado ni un barrido amplio. Entregar harness de barrido listo:
corto variable con largo fijo y viceversa, distinguiendo persistencia real,
ruido con MAE equivalente y control ideal. Semillas pareadas, soporte comun y
escalas ajustadas solo en DEV. Decisiones sobre holdout y presupuesto de B0
siguen separadas. No afirmar importancia de corto frente a largo por cuatro puntos.

## B. DOIN: el archivo debe ser la fuente del ETL

Bases dae92be/212c263. La sonda con seccion alterada y el reintento de otra
identidad deben ser PRE rojo y POST rechazo antes de insertar una sola fila.

1. Una ruta de verificacion compartida reconstruye contenido desde referencia
   de archivo y valida hashes, tamanos, inventario, tipos y vinculos. ETL consume
   ese resultado, no envelope.sections arbitrario. Prueba reiniciando el proceso
   sin el envelope original y reconstruyendo un warehouse desechable vacio.
2. Conservar metricas completas del candidato y su contrato: nombre, valor,
   escala, unidad, horizonte/poblacion/reduccion si estan declarados; parametros
   y procedencia. No inventar esos campos si faltan: explicitamente no comparables.
   El doble no puede descartar justo las columnas cuya igualdad se quiere probar.
3. Idempotencia: igual identidad Y contenido devuelve el anterior; cualquier
   discrepancia de experimento, intento, performance, parametros o metrica rechaza.
   Fallo a mitad de proyeccion no deja un cierre parcial presentado como completo.
4. Definir identidad de manifiesto frente a cuerpo. Un sucesor legitimo no
   sobrescribe historia y una contradiccion no se convierte en reintento.
   Probar dos escritores y archivo truncado; recuperacion sin falso verified.
5. Extender el adaptador de almacenamiento hacia la API real del lago en entorno
   desechable, con escritura/lectura gobernada; si no existe byte-upload, implementar
   el componente en su repo propietario. No llamar lago al directorio local.

No cambiar consenso, podar, migrar servicios ni borrar archivos historicos.
Mapear cada test a O01-O06/AT01-AT10; reportar cobertura parcial, no todo verde.
Ese trabajo no es prerequisito nuevo para experimentos que ya pueden ejecutarse.

## C. Completar preparacion de GPU y clasificacion

GPU: preparar la ejecucion en el obrero con capacidad, no solo un comando para
el entorno del coordinador. No transportar un venv con rutas absolutas. Mantener
el diagnostico propuesto en 300 CPU s / 300 s pared / 6 GiB de anfitrion y la
peticion de calibraciones 4800 CPU s / 3600 s pared; ninguna concedida por esta
nota. Con aprobacion explicita, diagnostico primero y calibraciones solo si pasa
el entorno, admision fresca y contrato vigente. La 5090 es preferente, no obligar
ese pin CUDA a una tarjeta incompatible. No ocupar una GPU que ya tiene trabajo.

Clasificacion: la solicitud debe incluir adquisicion de pesos Y corpus Banking77
pinados, hashes/splits/caches comprobados, dependencias cp312 y costo total. Dos
fases: adquisicion controlada sin ejecutar codigo del modelo, luego ejecucion
offline aislada bajo aprobacion de trust_remote_code. No pedir al dueno que
fabrique B77_LOCAL_SNAPSHOT ni los caches. No confundir variables HF_* con
prohibicion de egress impuesta al proceso. Medir el camino real tras la concesion.
Proponer piloto de costo sobre subconjunto TRAIN/DEV declarado, separado del
score oficial completo. Preservar despues el protocolo de la referencia, sin
presentar un timeout a mitad del corpus como score. Justificar CPU frente a GPU
admisible con costo y disponibilidad; no seleccionar CPU solo por conveniencia.

## Retorno y limites

Solo desarrollo y pruebas sinteticas/desplegues desechables, con admision de CPU
vigente; pruebas cortas bajo 2 GiB/120 s como las revisadas. Ningun broker real,
servicio, VM, interpretador compartido ni base poblada se modifica. No se concede
codigo remoto, diagnostico GPU o entrenamiento por inferencia de esta orden.

Retorno liderado por microensayo financiero sintetico reconciliado; resultados
anteriores preservados. Despachos reales, PRE/POST, commits/tips publicados y
gasto medido por carril. Las correcciones no son nueva precision cientifica.
No esperar a que finalicen todos para integrar uno que ya esta probado.
