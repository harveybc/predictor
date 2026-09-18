# Contrato de metricas del programa: version 1

Fecha: 2026-09-18. Corrige la interpretacion de I-INFO de 9-sep; no cambia valores
historicos ni afirma haber implementado todos estos medidores. Satoshi inventaria
productores existentes y completa los que exige el piloto antes de usar sus salidas.

## Identidad y alcance

Cada hecho lleva propuesta/hipotesis, campana, dataset/variable/target, particion,
brazo/replica/checkpoint, contrato de metrica/version/unidad, codigo/entorno,
poblacion o soporte, valor y estado. Estados: MEDIDO, NO_APLICA,
NO_IDENTIFICABLE, NO_MEDIDO o FALLIDO, con motivo; no convertir faltantes en cero.
Parametros, observaciones y semillas independientes se cuentan por separado.

Registros numericos tipados y finitos; politica explicita para denominadores cero,
constantes y datos vacios. Conservar fuentes que permitan recalcular, no solo un
digest del propio resumen. El warehouse DuckDB recibe via gobernanza; no abrir
el archivo productivo desde varios procesos para instrumentar cada epoch.

## D / Y: entradas y objetivos

Medir por variable/target y particion permitida: identidad fisica/semantica,
observacion/disponibilidad, unidades/muestreo, longitud cruda y util, huecos,
constantes, cuantiles/extremos y escalas de dependencia. Perfiles de diseno y fit
solo train; ninguna metrica de test vuelve al selector.

- `bytes_raw`, dtype, endian, orden, forma, mascara y precision de serializacion.
- Longitud lossless con coder/nivel/version fijos, incluidos headers; cociente
  comprimido/crudo puede superar 1. No es porcentaje de ruido ni bits aprendidos.
- H0 discreta: `-sum p_i log2(p_i)` en alfabeto/quantizer declarado, ajustado train.
  H0 no es entropia diferencial ni tasa con memoria. Frecuencias constantes dan 0.
- Tasa contextual/prequential: perdida log2 con modelo/contexto declarado, costo
  del descriptor y evaluacion fuera del ajuste. Comparar H0/no-contexto; no llamar
  a un compresor informacion mutua por concatenar dos archivos.
- SNR real solo de componentes plantados y convencion de potencia especificada;
  SNR estimada con intervalo, calibracion, regimen y abstencion separados.
- Amplitud/fase/espectro, persistencia, dependencia cruzada y estabilidad con sus
  estimadores/parametros y disponibilidad; no inferir causalidad de correlacion.

**Correccion obligatoria:** L_C(cuantizado) y L_C(crudo) NO son cotas inferior y
superior de informacion libre de ruido. La descomposicion S+N no se identifica
solo comprimiendo X. Tampoco conocer un generador basta para llamar a su semilla
o a sus bytes "cantidad de informacion limpia": hace falta definir variables
aleatorias, distribucion, precision y entropia/condicional objetivo. Sin ello se
registran componentes y longitudes, no `D.I_free_synth` con unidades inventadas.

El fundamento de la entropia requiere una distribucion de mensajes, no su utilidad
semantica: [Shannon, 1948, seccion 6](https://swh.princeton.edu/~wbialek/rome/refs/shannon_48.pdf).
La separacion entre longitud, ruido y utilidad anterior es una restriccion del
contrato, no un supuesto de que Shannon proporcione un estimador para nuestros datos.

## M / G: modelo y grafo

- Parametros totales, entrenables y congelados separados; `count_params` total
  no es siempre numero entrenable. Pesos/activaciones/estado de optimizer se
  serializan y contabilizan por separado.
- Longitud de pesos con protocolo fijo; version del grafo y cuantizacion si existe.
  Ni pesos repetidos ni una longitud corta demuestran capacidad disponible.
  Simetrias/permutaciones de redes impiden interpretar bytes como conocimiento unico.
- Al inicio, en checkpoints predefinidos, mejor validacion y final: normas de pesos,
  gradientes cuando se miden, errores train/val, updates, parametros y coste.
  Frecuencia fija antes del run; no introducir todos los SVD en cada minibatch.
- Para singular values no negativos `s`, si `sum(s)>0`, definir
  `p=s/sum(s)` y `effective_rank_entropy=exp(-sum(p*ln(p)))`;
  `stable_rank=sum(s*s)/max(s)^2`; `nuclear_ratio=sum(s)/max(s)` son otras medidas.
  La ultima NO se etiqueta como la primera. Cero matriz: resultado no definido
  por esas formulas; convencion separada con estado explicito.
  [Roy y Vetterli, 2007, definicion 1](https://infoscience.epfl.ch/bitstreams/2907ab8a-23f5-481d-bb07-1d56a3f3511f/download).
- Radio espectral solo para operador cuadrado especificado. La matriz rectangular
  de una capa tiene norma espectral, no radio espectral. La adyacencia de un DAG
  tiene radio 0 por estructura: no diagnostica su capacidad ni estabilidad.
- Grafo: declarar nodos/aristas, direccion, peso/distancia, umbral, tratamiento de
  Conv/recurrentes y componentes desconectados. Densidad, excentricidad, distancias
  y modularidad solo cuando esa definicion les da sentido. Una desconexion no es
  distancia cero; guardar cobertura y NA. Misma regla entre brazos comparables.

Antes de habilitar un medidor: fixtures constante, vacio, matriz cero, rango uno,
identidad, grafo desconectado, dtype y renombrado. No imponer coste cuadratico a
una red grande sin presupuesto. La falta de definicion debe aparecer, no ocultarse.

## Rendimiento, capacidad y parada

P-MOD: MAE/MSE/RMSE, MASE train-scaled compartido, efectos metodo-control,
calibracion/anchura de intervalos cuando hay cabezal probabilistico, agregacion
por tarea/familia segun propuesta. P-TRN conserva su signo propio si usa
loss(identity)-loss(transform); la metrica incluye su definicion, no un `delta` ambiguo.

P-CAP: capacidad operacional memorizada y log-loss en bits requieren tareas y
distribuciones propias, no longitud del fichero de pesos. Censura de N_min y
fallos numericos quedan en denominadores conforme a su protocolo.

H-ES informacional: recolectar trayectorias primero, ajustar la regla solo en
tareas DEV, evaluar en tareas nuevas con regla fija frente a paciencia estandar,
perdida seleccionada, distancia al checkpoint de referencia y costo. Seleccionar
regla y juzgarla en las mismas curvas seria reutilizar la evaluacion. No cambiar
la parada productiva por una correlacion exploratoria entre compresion y val-loss.

## Negocio y coste

RL: retorno neto con capital y costos explicitos, Sharpe y periodicidad, drawdown,
turnover, exposicion, episodios/semanas/regimenes y pasos observados. Pronostico
acertado no equivale a orden ejecutable en un pico conocido retrospectivamente.

Costo: wall separado de suma CPU/GPU, concurrencia, memoria, updates, perfilado,
preentrenamiento, busqueda, fallos, cache/verificacion e ingesta. Si no se mide
potencia, no llamar a horas GPU consumo electrico observado.

## Uso critico del cubo

Cada cierre compara contenido y poblacion contra contabilidad independiente,
recalcula perdidas desde arrays y muestra filas excluidas con motivo. Paneles de
analisis: efecto por tarea/grupo/regimen/contexto/modelo; error vs volumen/costo;
trayectorias de generalizacion vs M/G; disponibilidad/extremos vs transformacion.
No son nuevas hipotesis confirmatorias si se descubren explorando el cubo.
Historicos con distinto contrato conservan su version y no adquieren garantias
nuevas por ser migrados o mostrar un digest correcto.
