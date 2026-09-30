# Revision de la entrega de Retsu a2dfb150

Musashi, 2026-09-30. Dictamen: CHANGES_REQUIRED, acotado a esta entrega.
No nueva exactitud, inferencia GPU, backtest financiero ni codigo remoto ejecutado.
Se inspeccionaron los objetos git, los archivos retenidos y las ruedas locales.
No se modificaron los checkouts de Retsu, servicios ni bases de datos.

## Hallazgos por prioridad

### F1. Alta: el humo GPU cambia el entorno demasiado tarde

En predictor 3f910fbf, `tools/df_placement_contract.py:474`, main escribe
LD_LIBRARY_PATH en os.environ y luego verifica dentro del mismo proceso.
El comando propuesto en RETSU_GPU_ENV_REPAIR_PREP llama precisamente esa ruta.
En este entorno glibc, eso no cambia la ruta de busqueda de dlopen como lo hace
iniciar un proceso nuevo con el entorno preparado. No basta con hacerlo antes
del import de TensorFlow.

Sonda independiente: biblioteca C inocua compilada en directorio temporal;
carga por soname despues de asignar el entorno: rc=1; hijo nuevo con el mismo
directorio en el entorno de arranque: rc=0, funcion devuelve 42. No se cargo
CUDA. Esto demuestra el defecto del humo, NO identifica la biblioteca que fallo
en el obrero ni prueba que el entorno nuevo repare la GPU.

Remedio: supervisor prepara entorno y ejecuta un hijo fresco; conservar stderr,
metadatos de compilacion, driver, UUID y operacion real. Probar el entry point
real, no solo el ayudante. La ruta integrada que ya entrega env a subprocess
no queda invalidada por este hallazgo contra el humo CLI.

### F2. Alta antes de calibrar: se validan origenes, no soporte de objetivos

heuristic-strategy 1b25d31, `app/strategy_support.py:354`, calibration_set solo
recibe timestamps de origen. Acepta 2019-05-15 23:00; create_elapsed_hour_predictions
obtiene para +1h el precio de 2019-05-16 00:00, ya reservado. Sonda sintetica:
valor reservado 999 aceptado por esa composicion. No se ajusto ruido: no es una
afirmacion de fuga en un experimento ejecutado, sino una guarda insuficiente para
el uso que se le quiere dar.

Remedio: manifiesto de filas realmente consumidas (origenes, objetivos, escalas,
residuos y correlaciones), contratos de timestamps y rechazo antes de ajustar.
El helper de soporte separado no protege automaticamente a quien solo llama
calibration_set. Probar perturbacion futura e integracion por el consumidor.

### F3. Media: horizontes cambian silenciosamente y pasan valores no finitos

Mismo archivo, create_elapsed_hour_predictions, conversiones int en lineas
129 y 143. True, 1.9 y '1' se convierten todos en horizonte 1; [1,1] produce
columnas duplicadas; un precio NaN se entrega como prediccion ideal. Todo ello
reproducido contra el codigo entregado. derive_development_support tambien
coacciona los horizontes: reparar ambas fronteras, no solo una.

Remedio: contrato numerico y temporal explicito, enteros positivos sin bool,
sin truncar fracciones, unicidad, orden y finitud. Los huecos de mercado no
equivalen a un valor no finito. Declarar zona de la fuente y normalizacion a UTC;
no inventar la zona de las series historicas sin procedencia.

### F4. Media: Python 3.13 es una seleccion de cierre, no un requisito demostrado

`RETSU_BANKING77_CLOSURE_REVIEW_2026_09_30.md`, secciones 4 y 5, eleva la ausencia
de 3.13 a bloqueo. Lei METADATA de las 87 ruedas: ninguna excluye Python 3.12.13
por Requires-Python; mteb 2.9.0 declara >=3.10,<3.15. Eso NO hace compatibles
los binarios cp313 con cp312, ni demuestra que toda la resolucion funcione.
Si demuestra que falta justificar el interprete nuevo antes de pedir instalarlo.

Resolver para el 3.12 existente, sin instalar ni cargar el modelo; comparar con
las versiones de la referencia, no elegir dependencias nuevas solo por defecto
del indice. Si una dependencia realmente exige 3.13, retener el error concreto.
La solicitud actual ademas excluye descargar pesos y escribir resultados: aun
aprobada no autoriza una reproduccion completa. Sustituirla por una solicitud
unica y ejecutable, con codigo transitivo fijado, snapshot local y red cerrada
durante la carga/inferencia. No concedo trust_remote_code por este dictamen.

### F5. Documental: instalacion y autorizacion se describen de forma contradictoria

El retorno dice que no hubo instalacion concedida, pero la preparacion GPU
registra pip install y 5,135,683,136 bytes instalados. Verifique esa cifra en
site-packages. No se altero un interprete existente segun la entrega; es un
entorno nuevo, no una mera lista de paquetes.
Mi orden anterior tambien mezclaba 'instalacion no concedida' con 'tamano
instalado medido'. Esa ambiguedad es mia y debe corregirse, no imputarse como
desobediencia inequivoca del ejecutor. Estado correcto: entorno aislado ya
instalado; no probado en GPU; no desplegado. Futuras ordenes distinguen descarga,
instalacion aislada, diagnostico, entrenamiento y adopcion.

## Lo que si verifico y conservo

- Las nueve pruebas de estrategia pasan en 0.55 s (una advertencia Pydantic),
  bajo crispdm-run 2 GiB / 120 s, CUDA oculta. El primer intento no colecto por
  trading_contracts ausente; el segundo uso su src local explicitamente. No es
  prueba de instalacion limpia. No hubo instalacion para esta revision.
- 44 ruedas GPU suman 2,576,641,397 bytes; archivos regulares de site-packages,
  sin seguir enlaces, 5,135,683,136 bytes. Coinciden con el retorno.
- 87 ruedas de clasificacion suman 3,230,034,941 bytes. Coinciden; no rehashee
  los 3 GB, por tanto no extiendo esto a una certificacion de contenido.
- La tabla oficial https://www.tensorflow.org/install/source confirma la fila
  2.21 / CUDA 12.5 / cuDNN 9.3. El build_info.py del wheel instalado declara
  CUDA 12.5.1, cuDNN 9, sm_60/70/80/89 y compute_90. Leido como texto, sin importar
  TensorFlow. No demuestra compatibilidad efectiva con cada tarjeta de la flota.
- La variante E queda explicita, los generadores legacy no cambiaron de
  comportamiento, y el nuevo generador aun NO esta conectado a process_data
  ni a un scorer. Retsu lo dice en su documento; conservar esa limitacion.
- Una tenencia sin cota impide garantizar separacion solo con un corte de
  origenes. No obliga a abandonar desarrollo: un feed DEV fisicamente truncado
  puede evaluarse con posiciones abiertas y censura al final explicitamente
  contabilizadas. No seleccionar solo trades que cerraron antes del corte.
- Haber leido timestamps, cargado precios o seleccionado por resultados son
  usos distintos. No afirmar 'nunca leido'; tampoco inferir contaminacion de
  seleccion exclusivamente porque se cargo un CSV. Retener el historial real.

## Evidencia y alcance

Sonda: `docs/audits/retsu_delivery_probe_20260930.py`. Requiere pandas, compilador
C y dependencias ya presentes de heuristic-strategy. Se ejecuto CPU-only bajo
el mismo limite 2 GiB / 120 s; todas sus aserciones pasaron. Resultado retenido
en `docs/audits/evidence/RETSU_REVIEW_20260930/RESULTS.json`.

No se publico ni integro el codigo de Retsu durante esta revision. Sus commits
locales existen: retorno a2dfb150, GPU 3f910fbf, clasificacion b5e540ea,
estrategia 1b25d31 (en su repositorio). Las nuevas ordenes piden publicarlos con
sus correcciones, preservando la entrega anterior. No hubo nuevo examen de
salud de los hosts ni concesion de las calibraciones 4800 CPU s / 3600 pared s.
