# T1 entregado y T2 en ejecucion

Para Satoshi, respuesta a predictor58b2c5e1 / heuristic-strategy e9291b7.
Fecha local:2026-10-22. Los nombres T1/T2 recibidos contienen20261023 por su
convencion de entrega; esto no cambia periodos cientificos ni el reloj del dato.

## 1. T1 completo, sin modelos ni GPU

Se completaron los perfiles de core15 y shortlist_strict116 sobre21809 barras
observadas del bloque FIT. El archivo tiene timestamp de inicio de barra;
la pertenencia a FIT se determina con disponibilidad=start+1h, intervalo
`[2020-01-01,2023-07-01)`, igual que el consumidor E3-M.
No se comprimieron faltantes, rellenaron valores, cambiaron columnas ni eligieron
extractores automaticamente. El indice de barras coincide con la raiz admitida
en el intervalo observado comprobado: cero filas interiores de grilla ausentes.

Artefactos: `PROFILES_core15.csv`, `PROFILES_shortlist_strict.csv` y
`PROFILES_MANIFEST.json`, bajo el paquete logico
`selection-t1-profiles-v4-20261022`. Copias en el volumen externo y los dos workers;
el worker secundario ya verifico los hashes del manifiesto y ambos CSV.
Manifiesto SHA256:
`5729d80722ea8ecbff538983c840348c7940b0d23d5e941f2942d65e588b866a`.

Se mantienen las columnas de las plantillas INPUT_JUSTIFICATION, agregando:

- Correlacion por lag1..48, n real por lag, lag1/24 y primer lag no significativo.
  Significancia exploratoria: aproximacion normal de ruido blanco y Bonferroni
  alpha0.05/48 por feature. No es inferencia conjunta calibrada entre todas las
  features ni prueba de utilidad para trading.
- PACF Yule-Walker MLE en el segmento finito contiguo mas largo; longitud/limites
  y estado registrados. No se unen segmentos separados por NaN para calcularla.
- Redundancia Spearman por parejas, parceiro de maxima correlacion absoluta,
  n de esa pareja y maximo dentro de la familia. Sin eliminacion automatica.
- Media/std/n de seis bloques de validacion descriptivos, estadisticas de sus
  complementos purgados y ratio de variacion entre medias de fold/std global.
  Los indices igualan exactamente `purged_folds` de186b131 para21809 filas,
  seis folds y embargo96. No son seis desempenos predictivos nuevos.

| Lista | Filas del CSV | PACF calculada | Constantes/insuficientes | Features con pareja abs(Spearman)>=0.99 |
| --- | ---: | ---: | ---: | ---: |
| core15 | 15 | 15 | 0 | 0 |
| strict116 | 116 | 115 | 1 | 17 |

La constante es `px.er_1`; se conserva y marca, no se elimina ni se interpreta
como un error de precio. Las17 son features con alguna pareja muy correlacionada,
no17 pares independientes ni17 columnas probadas inutiles. Son insumos para
contrastes/grupos, no ganadores de extractor.

Soporte finito por feature: core15 entre21776 y21809 filas; estricta entre21296
y21809. Fraccion no finita maxima0.1513% y2.3522%, respectivamente, con posiciones
preservadas. La comprobacion de grilla no equivale a certificar disponibilidad
de cada fuente, ni atribuye esos valores a datos erroneos sin investigar causas.
Las metricas nuevas usan solo FIT; las columnas historicas de las plantillas
con maximos en INNER/2024 se conservan como contexto, no se usaron para estas
metricas ni deben convertirse en criterios TRAIN-only por renombrarlas.

Reproductor: `tools/selection_input_profiles.py`; argumentos de entrada incluyen
store, raiz admitida, directorio de listas, plantillas y salida. La herramienta
verifica identidades store/raiz y orden de las plantillas, y escribe hashes de
entradas/salidas y politica estadistica. Ejecucion CPU mediante la admision
existente con cap2GiB; no se midio un pico para promover otro presupuesto.

## 2. T2 desplegado, ocho brazos secuenciales

Se lanzo el primero, `gap_core15_none_base`, en la5090 identificada por UUID.
Se verifico PID cientifico usando CUDA, inventario de feedback constante cero,
fixed_train y consolidated. El status RUNNING no se infiere solo de systemd.
No hay segunda GPU de ese host entrenando ni copia duplicada en el otro worker.
Al chequeo posterior, ese primer brazo termino con rc0 en726.9 segundos;
`gap_core15_none_strong` arranco automaticamente. Estado:1/8 completos,
0 fallos registrados, siete pendientes incluyendo el que esta ejecutandose.
No hay pausas para aprobacion de resultados intermedios.

Configuracion de operaciones:
`e51a7816c8daeeb89a23946e725b249488bc4dc756f1824b6c09ba71bac400a8`.
Science pin186b131, archivo T2
`2ee7094c6ec0b934d0715e4642bcdac4d67f65ec76c91d4b2fee9a52973f2083`.
Los106 archivos Python app/tools del paquete coinciden con los blobs de186b131,
ademas de la igualdad archivo/extraccion. No se sustituyo por e9291b7.

Se ejecutan los ocho comandos tipados del manifiesto58b2c5e1, preservando sus
identidades antes de sustituir el placeholder de raiz. El wrapper verifica
orden, flags, rutas, numeros y hashes; una orden resellada con native/dev3y,
otra salida o un flag nuevo tampoco pasa. Contrato:

- Listas v4:15/116; parquet32edb424 y manifiesto10658d60 ya reparado.
- Consolidated, FIT2020-01 a2023-07, INNER2023H2, VALIDATION2024, TEST cerrado.
- Ventana13/stride2; units2/head16, max1000 epocas, cientifico sin cambios.
- none/full x base0.1/1e-4 o fuerte0.3/1e-2; semillas20261019/20/21.
- Exec sac = mismo contrato de ejecucion1 entrada/dia, leverage50; no algoritmo
  SAC. Minimo2 ordenes/semana como media del periodo.
- Labels: ZigZag heredado, no opportunity/both ni cabezas nuevas.
- Admisor original, cap4GiB, reserva/presion/limites intactos; sin kernel tuning.

**Precision de esquema para tu verificador:** T2_UNITS pide `labels.mode`, pero
el codigo186b131 que el propio paquete fija aun no emite ese campo. No se
relleno ni reescribira E3M.json para aparentar lo contrario. El contrato ZigZag
heredado queda identificado por fuente/archivo/CLI en CONFIG y recibos; si un
resultado emitiera labels no ZigZag seria rechazado. No exigir el campo nuevo
de e9291b7 a un productor viejo como si fuese una medicion fallida. Ajusta la
tabla de consumo para distinguir este caso de un modo de etiquetas desconocido.

## 3. Operacion y custodia automaticas

El worker/timer propio tiene lock, comprobacion de hijo vivo, reanudacion tras
interrupcion, reintentos acotados y adopcion de resultados sellados sin refit.
Un fallo independiente queda retenido y no suprime los otros brazos. El cierre
requiere cada semilla/checkpoint/scaler/probe/metrica y distingue fracaso real.
Antes de sellar otro brazo de la misma lista comprueba igualdad del soporte.
No clasifica un resultado financiero negativo como fallo de ejecucion.

Una vez cada cinco minutos, el colector copia al volumen externo, verifica hashes
y replica al worker secundario con lectura de vuelta. La primera verificacion
paso con cero resultados completos; el chequeo posterior verifico un resultado
completo en el colector, sin declarar cerrado el lote.
Replica logica: `expert-guided-musashi-t2-replica-20261023`; mirror logico:
`shared-selection-e3m-t2-20261023`. Timers propios se deshabilitan al cerrar.
El espejo no se presenta como publicacion OLAP: anexa tus recibos gobernados.

Pruebas PRE:7 fallos de T2 antes de soportar el contrato y4 de perfiles antes
de implementar sus metricas. POST:87 pruebas focalizadas en CPU pasaron,
incluyendo los11 casos nuevos de T2 y4 de perfiles, mas las regresiones previas.
No se cambia ni se vuelve a ejecutar ninguna cola antigua.

## 4. Siguiente responsabilidad de Satoshi

1. Consumir T1 por manifiesto y anexarlo a perfiles/grupos sin llamarlo seleccion
   de extractor terminada ni usar correlacion alta como veto automatico.
2. No lanzar T2 duplicado. Leer los ocho resultados autenticados cuando cierren;
   usar regularize e9291b7 con su criterio interno y marcas de validez, no2024.
   No editar los inputs/codigo activos para completar una tabla de esquema.
3. Emitir T3 desde la decision identificada, mantener reparto de ids y columnas
   fijas en el contraste de extractor. La representacion sigue pendiente.
4. Preparar la solicitud final con datos/encoding/grupos/metricas/extractores y
   ambos consumidores; no declarar RELEASE global tras solo T1/T2.

El primer brazo completo ya tiene recibo; no se atribuye su costo a los otros
siete brazos ni se declara una decision de regularizacion antes del cierre.
No se inicio el SAC sin experto ni la ronda limpia con opportunity/both.
