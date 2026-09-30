# Retsu: continuacion paralela despues de a2dfb150

Musashi, 2026-09-30. Retsu sustituye temporalmente a Satoshi como coordinador.
Leer primero `../audits/work_plan/MUSASHI_RETSU_A2DFB150_REVIEW_2026_09_30.md`.
No reiniciar lo terminado ni esperar a Satoshi. Esta orden suplementa las
anteriores; no cancela los experimentos independientes, Traffic o M5PHET.

## Resultado requerido, no otra ronda solo documental

Cuatro carriles con agentes y acuses reales, worktrees separados y un integrador.
Despachar A, B y C al comienzo; D puede disenar en paralelo y tomar el siguiente
slot de implementacion. Paralelismo de agentes no significa sobreasignar RAM.
Prioridad de GPU: 5090 externa si anfitrion y framework son admisibles; las
otras solo por una razon tecnica registrada. No competir con cargas existentes.

Al menos el carril B debe terminar con un resultado de comportamiento medido
sobre datos sinteticos usando el plugin real, no otro helper sin consumidor.
Separar siempre resultado sintetico, diagnostico de recurso y resultado de mercado.

## A. GPU: corregir el entry point y dejar un diagnostico ejecutable

Base 3f910fbf; conservar el entorno aislado ya instalado, no descargarlo otra vez.
Corregir la narracion: se instalo un entorno nuevo; no se reparo ni desplego aun.
Reproducir F1 de la sonda de Musashi como PRE, luego reparar CLI y probar POST.

El supervisor prepara el env y arranca un proceso nuevo. Nada de cambiar
LD_LIBRARY_PATH dentro del mismo proceso y llamarlo equivalente. No importar
TensorFlow ni cargar librerias CUDA desde el supervisor. El hijo conserva el
entorno efectivo y verifica identidad, registro del framework y operacion real.
Prueba sin GPU con biblioteca C inocua, mas prueba del entry point publico y
de la ruta integrada. Incluir una negativa donde falta una dependencia transitiva.

El diagnostico GPU propuesto queda preparado con limite separado de 300 CPU s,
300 s de pared y 6 GiB de RAM de anfitrion, una ejecucion, admision fresca y
sin entrenamiento. Esa cifra es un limite de gasto, no una huella medida.
Su ejecucion requiere la aprobacion del dueno; esta orden no la inventa.
Antes de ejecutar, identificar en el host elegido el interprete y snapshot
instalado: el entorno preparado esta en el coordinador, no probado en el obrero.
Instalar/transportar al obrero solo bajo el alcance autorizado, sin copiar a
ciegas un venv con rutas absolutas. No cambiar drivers ni bibliotecas globales.

Con aprobacion, retener stdout/stderr completos, tf build_info, driver actual,
UUID fisico, librerias buscadas y error concreto del loader. Probar una operacion
con resultado consumido/sincronizado, no solo visibilidad. Si falla, informar
causa localizada o evidencia insuficiente; no lanzar intentos interminables.
No asumir que el pin CUDA 12.5 sirve para todas las arquitecturas de GPU.

Las dos calibraciones siguen siendo otra etapa: 4800 CPU s y 3600 s de pared
sumada entre hijos, dos hijos secuenciales. Solo despues de concesion explicita,
GPU probada y admision del sobre vigente de 12 GiB. No pedir 18 GiB ni encoger
modelo, lote, precision o poblacion para esconder el rechazo. Mostrar los limites
host/dispositivo por separado. El usuario puede aprobar ambas etapas juntas,
pero las calibraciones nunca se ejecutan si el diagnostico falla.

## B. Estrategia: de helpers a un ensayo sintetico util

Base heuristic-strategy 1b25d31. Tests PRE/POST de F2 y F3 antes de reparar.
Mantener la estrategia existente, variante E explicita, decisiones al cierre y
fills de mercado en la apertura siguiente. No introducir protectoras ni cambiar
TP/SL en nombre de una correccion. Legacy por filas sigue accesible y etiquetado.

1. Contrato de horizontes enteros positivos, unicos y sin coercion silenciosa.
   Rechazar bool/fracciones/NaN/inf. Precios finitos, timestamps validos y unicos,
   orden temporal y zona declarados. No adivinar la zona de un CSV historico.
2. Conectar el generador elapsed-hour al harness nuevo, con ruta explicita de
   configuracion. Una prueba debe atravesar generador, columnas consumidas por
   el plugin, decisiones, fills y resultado. No basta process_data legacy verde.
3. La admision de calibracion recibe soporte de todos los valores consumidos,
   no solo origenes. Derivar targets, escalas y residuos dentro de DEV. Mutar
   futuros reservados no debe cambiar ningun parametro ajustado en DEV.
4. Materializar un feed DEV truncado antes de la frontera; ningun evento de
   broker simulado ni costo se toma de despues. Reportar posiciones pendientes,
   equity marcada y PnL realizado por separado al final. Probar si close() en
   stop() realmente obtiene un fill; no asumirlo. No eliminar a posteriori trades
   que atraviesan el corte: sesgaria la poblacion. Cualquier liquidacion forzada
   debe ser una convencion explicita de un sucesor, no la estrategia original.
5. Ejecutar un microensayo con OHLC sintetico determinista y las doce predicciones
   1..6h / 24..144h, mismo soporte y costos. Primero verificar por una traza que
   corto afecta las salidas y largo las entradas/TP/SL en esta variante; E puede
   usar ambas familias en la salida, por tanto no presumir separacion funcional.
6. Contrastar ideal/ideal, persistencia/ideal, ideal/persistencia y
   persistencia/persistencia. Calcular MAE y naive por horizonte y una agregacion
   declarada. Si no abre/cierra operaciones que permitan observar ambos caminos,
   el fixture no cumple: construir escenarios de subida, caida y giro con oraculos
   de decisiones fijados antes de ejecutar. Reportar trades, costos, capital,
   exposicion y resultado; no imponer que la senal ideal maximice el profit.

Esto es prueba sintetica de comportamiento autorizada como desarrollo, CPU-only,
sin mercado, con admision 2 GiB y 120 s de pared por prueba como la revision;
no optimizador ni barrido de 241 celdas. Si no cabe, reducir el fixture de prueba,
no un experimento sellado. No reutilizar el saldo historico como presupuesto B0.
El sucesor B0 debe quedar listo con costo y convencion terminal antes de solicitar
su corrida sobre datos reales. Para el futuro barrido distinguir persistencia
real de ideal+ruido con igual MAE: igual MAE no implica iguales decisiones.

## C. Clasificacion: quitar el bloqueo artificial y cerrar la solicitud real

Base b5e540ea. Resolver primero metadatos para CPython 3.12 existente y plataforma
real; no intentar instalar cp313 en cp312. Reutilizar ruedas universales verificadas.
No instalar otro interprete ni descargar otra pila completa hasta justificar
necesidad y disco. Mteb 2.9.0 admite 3.12 segun METADATA; falta la resolucion
completa. Conservar diferencias frente a las versiones de la referencia.

Cerrar transitivamente la ruta de carga: todos los ficheros, adaptadores,
tokenizador y dependencias con revision/hash. El snapshot_download interno sin
revision no debe poder traer una revision nueva: usar snapshot local completo,
offline/local-only comprobado, sin red ni credenciales de broker/datos privados
en el proceso. Prueba negativa con un fichero faltante: rechazo, no descarga.
No interpretar ausencia de subprocess/eval como certificacion de seguridad.

Entregar UNA solicitud util para el ensayo de investigacion: codigo concreto,
licencia, snapshot/pesos, dependencias compatibles, instalacion aislada, disco
total incluyendo caches, dispositivo, piloto de coste y publicacion de metricas.
El dueno decide trust_remote_code explicitamente. No otorgado aqui. La etiqueta
de investigacion no concede uso comercial. No gastar GPU en fixtures esperando
esa decision; continuar B/D y la preparacion del evaluador con datos sinteticos.

## D. DOIN: empezar la implementacion aislada de archivo externo

Adoptar b4dc63cb y su `DOIN_OFFCHAIN_BODY_PLAN_2026_09_30.md`, O01-O06/AT01-AT10.
Primer corte: contrato y tests de almacenamiento shadow, sin cambiar consenso:
serializar el cuerpo existente, escribir a un adaptador temporal de archivo,
leer/verificar hash, conservar manifiesto y proyectar metricas en warehouse
desechable de la misma interfaz. Incluir candidatos no ganadores y reintentos.
Tests antes del codigo; cambios en repos propietarios, no un framework nuevo
en predictor. Descubrir API real de escritura del lago y extender su adaptador
si hace falta; no suponer que registrar un recurso sube sus bytes.

Entregar ida y vuelta ejecutada sobre un bloque fixture con cambios de bytes,
reintento y fallo de archivo. Reportar alcance local/disposable honestamente.
Nada de migrar cadena, podar, borrar el unico cuerpo o reiniciar servicios.
Los compromisos on-chain/versionado y piloto multinodo vienen despues, sin
retener los carriles cientificos ni el producto M5PHET mientras tanto.

## Integracion, publicacion y retorno

- Publicar las cuatro entregas previas locales por sus ramas, previa revision de
  secretos; verificar tips remotos. No forzar push ni mezclar ramas vivas. Publicar
  las correcciones encima preservando historia y PRE. No decir publicado sin verificar.
- Reconciliar la cola existente por carril y fecha; no crear otro scheduler.
  Retsu integra continuamente; un agente bloqueado entrega lo minimo concreto y
  toma otra tarea independiente. No prometer agentes que no se despacharon.
- Mantener costo, limite y autorizacion separados. Ninguna tarea interrumpe MT5,
  chat, Postgres, Metabase o trabajo de GPU ajeno. No borrar los caches retenidos
  automaticamente: inventariar contenido/reuso/tamano antes de proponer limpieza.
- Retorno liderado por el microensayo sintetico y comportamiento medido, luego
  GPU real solo si autorizada, pruebas y bloqueos. Para resultados cientificos:
  modelo, datos, metrica/escala, naive de mismas filas y referencia comparable;
  si no hubo medicion, decirlo. No convertir pruebas sinteticas en utilidad financiera.
- M4 y las seis celdas doctorales mantienen sus condiciones de aprobacion. Este
  paquete no firma una confirmacion ni altera el work plan cientifico.
