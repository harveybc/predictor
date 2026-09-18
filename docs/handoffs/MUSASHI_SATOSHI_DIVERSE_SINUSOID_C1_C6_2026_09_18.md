# Orden unica para Satoshi: C1-C6, banco diverso y curva de aprendizaje

> SUPERSEDIDA como proxima campana por [RP1-RP8](MUSASHI_PROGRAM_RESTART_RP1_RP8_2026_09_18.md).
> Conservar calculos y tests compatibles como controles internos. No ejecutar
> este barrido ni heredar sus numeros como diseno del programa doctoral.

Fecha: 2026-09-18. Responsable de ejecucion: Satoshi. Revision ML: Musashi.
Sustituye A1-A5 como proxima campana. B1-B5 fue borrador retenido, no emitido.
Ejecutar C1-C6 hasta el cierre sin microaprobaciones. No es entrenamiento RL.

## C1. Pregunta, alcance y sustitucion

Pregunta: puede el pipeline aprender a predecir el siguiente valor de sinusoides
de periodo conocido, transfiriendo a fases y amplitudes retenidas, y como cambia
su error al aumentar ejemplos de entrenamiento distintos bajo igual techo de
actualizaciones? Esto no pregunta si una red puede aproximar senos en general:
existe literatura y una recurrencia analitica. Es adecuacion de nuestro instrumento.

No lanzar el factorial anterior ni entrenar la unica onda repetida como campana
principal. Conservar resultados anteriores; detener solo despacho incompatible,
cerrar intentos con costo, sin parar otros trabajos/servicios. No reiniciar el
proyecto. Si hay evidencia compatible ya producida, verificar antes de reutilizar.

P=8, W=17, h=1 se conservan. Cambia explicitamente la poblacion: en vez de ocho
ventanas repetidas, trayectorias con fases y amplitudes distintas. No ruido,
wavelet, STL, indicadores ni busqueda de arquitectura en este control. No claims
de utilidad financiera, no linealidad compleja, frecuencia desconocida o RL.

## C2. Banco determinado ANTES de puntuar

    x[t] = A*sin(2*pi*t/8 + phi), t=0..24 POR trayectoria
    W=17, decisiones t=16..23, etiqueta x[t+1]
    25 valores -> 8 ventanas distintas por trayectoria
    phi = (pi/4)*(q/256), A = a/28

La fase se parametriza modulo un paso (pi/4), porque recorrer las ocho decisiones
ya cubre el ciclo completo. Variar fase por un paso entero solo permuta ventanas:
esa duplicacion NO contara como nueva diversidad. A>0 evita equivalencia por signo.

| Particion | Amplitud: numerador a | Fase: numerador q | Trayectorias | Ventanas |
|---|---|---|---:|---:|
| Train maximo | 14+4*j, j=0..7 | 4*i, i=0..63 | 512 | 4096 |
| Validacion | 15+4*j, j=0..6 | 8*i+2, i=0..31 | 224 | 1792 |
| Test | 17+4*j, j=0..6 | 4*i+3, i=0..63 | 448 | 3584 |

Todos son controles dentro del intervalo de amplitudes [0.5,1.5]; val/test
intercalados entre amplitudes train, no extrapolacion fuera del intervalo. Fases
y amplitudes de val/test no se usan en train. No afirmar cobertura continua ni
independencia estadistica: es una rejilla determinista, no una muestra i.i.d.

Total: 1184 trayectorias, 29600 valores brutos, 9472 ventanas. Train/val/test
son conjuntos de trayectorias distintos; nunca concatenar para crear ventanas.
Cada fila lleva identidad de trayectoria y t local, fuera de las features.

Curva train anidada: 8,16,32,64 fases x 8 amplitudes x 8 ventanas =
512,1024,2048,4096 ventanas. Orden de fases: bit-reversal de enteros 0..63 en
seis bits; primeros 8/16/32/64. Asi cada nivel cubre el intervalo uniformemente,
no aumenta solo desde una esquina. Misma val/test para todos los niveles.
Esta escalera es una decision de diseno para medir suficiencia, NO un minimo
neural demostrado. No llamar suficientes a 4096 antes de observar la curva.

Comprobar identidad canonica (a, secuencia de fases modulo 2048), donde
fase(t)=(256*t+q) modulo 2048, y comprobar tambien ventanas numericas del generador
real. Detectar duplicados aun con nombres distintos. Contar ventanas, trayectorias,
fases, amplitudes, minibatches y updates separadamente. Solapamiento intra-trayectoria
se declara: ventanas distintas no equivalen a observaciones independientes.

Entrada: unicamente 17 valores historicos. No t, A, phi, q, a, particion, ni
etiqueta futura al modelo. Estandarizacion escalar x sobre filas unicas 0..23
por trayectoria TRAIN del nivel correspondiente; y sobre etiquetas 17..24.
Nunca fit por ventana, por trayectoria ni sobre val/test. Registrar scaler; ambos
aprendices del nivel usan el mismo. Invertir escala para calcular errores.

## C3. Aprendices y presupuesto de ajuste comparable

Controles: persistencia, recurrencia x[t+1]=sqrt(2)*x[t]-x[t-1], OLS sin
penalizacion con dos ultimos rezagos. Ridge con intercepto no penalizado, W=17:
minimiza SUM(error^2)+1*SUM(coeficientes^2), no una media con lambda distinto.

Conv1D fija de A: seis capas causales, 16 filtros/capa, kernel=3, dilataciones
1/2/4/8/16/32, ReLU; ultima posicion -> Dense(1) lineal. Sin pooling, BatchNorm,
dropout ni residuales. 4001 parametros; RF teorico 127, W real 17: el resto es
padding, no contexto observado. Seis capas no son un minimo ni optimo demostrado.

Adam lr=.001, beta_1=.9, beta_2=.999, epsilon=1e-7, amsgrad=false; batch=64,
MAE estandarizado. Semillas NN 101/202/303, emparejadas entre niveles; registrar
Python/NumPy/framework y minibatches. Un ridge y tres NN por nivel = 4+12 ajustes.
Ninguna seleccion de la mejor semilla. Codigo y entorno ligados antes de medir.

CAMBIO explicito respecto a A: igualar techo a 12800 updates por NN, NO 200
epocas para cada volumen (eso daria presupuestos de optimizacion distintos).
Validar cada 64 updates observados; 200 comprobaciones maximas. Early stopping
por MAE val, mode=min, min_delta=0, paciencia=20 COMPROBACIONES, restaurar primer
minimo estricto. Documentar epocas equivalentes (ejemplos procesados/M), updates
y checkpoint, no llamar epocas a esas comprobaciones. Entrenar ciclos barajados
del conjunto train, sin fabricar ejemplos nuevos ni resetear el optimizador.

Usar callback/lazo real y probado con el framework existente, no una simulacion
en tests. Un limite de recursos o techo de updates no demuestra convergencia.
Guardar predicciones val por comprobacion, errores y checkpoint elegido para
rederivar el minimo. Curvas train/val en escala original. Test inaccesible hasta
restaurar y verificar seleccion. No usar test para ajustar lr, paciencia, arquitectura,
volumen o margen. Probar que alterar test no cambia pesos ni seleccion.

## C4. Pruebas anteriores al entrenamiento

| ID | Prueba estructural y conductual requerida en el camino real |
|---|---|
| D1 | Conteos/nesting/diversidad; copias renombradas y fase +2*pi no son ejemplos nuevos |
| D2 | Particiones disjuntas por trayectoria/parametros; no ventanas entre trayectorias |
| D3 | Labels h=1; futuro/prefijo; control de fuga deliberada detectado |
| D4 | Fit train-only, train invariante a alteracion val/test; metadata fuera del tensor |
| D5 | Grafo/parametros observados; updates realmente ejecutados y techo respetado |
| D6 | Early stopping real restaura minimo; curva adversaria que empeora lo prueba |
| D7 | Historial/checkpoint corrupto detectado por rederivacion; vacio no pasa |
| D8 | Recarga y errores desde arrays; OLS/recurrencia max_abs_error <=1e-10 float64 |
| D9 | Loader realmente lee entrega gobernada; outbox/fallo/reintento con costo |

Requisitos -> pruebas de aceptacion -> diseno -> tests unitarios/integracion ->
implementacion -> aceptacion. Mantener estado persistente por etapas y evidencia.
El script de Musashi adjunto verifica matematica/geometria, NO reemplaza D1-D9
contra el generador, framework, loaders y runner productivos.

## C5. Ejecutar y decidir sin ampliar oportunistamente

Pruebas D1-D9 primero. Piloto de costo train/val SIN test, modelo y nivel maximo,
sin elegir hiperparametros por resultados. Techo acumulado 14400 s CPU, contando
piloto, proyeccion con 25% holgura, limites de memoria por hijo. Si la proyeccion
no cabe, registrar el deficit medido; no exceder ni recortar por scores. No pedir
otra aprobacion cuando cabe y pasan las pruebas. No inventar mas gates burocraticos.

Distribuir ajustes independientes entre los tres roles segun recursos observados,
CPU, identidades propias y entregas data-gov. No exigir maquinas ocupadas ni
reiniciar servicios para llenar los tres roles. Registrar asignacion y uso reales.

Registrar campana DEVELOPMENT antes de trabajo; ligar generador/banco/diseno/
codigo/entorno. Todas las condiciones planificadas se reportan, fallen o pasen;
no detener por error alto de un volumen pequeno. Mecanica rota o limite excedido:
conservar intento y detener despacho afectado, no continuar con scores invalidos.
Todos los desenlaces, recursos y artefactos al warehouse DuckDB, conciliacion por
contenido y poblacion archivos -> padre -> contabilidad -> warehouse.

Punto primario de aceptacion predeclarado: M=4096, LAS TRES semillas. Error medio
por trayectoria de |pred-y|/A, luego media no ponderada entre trayectorias test,
<=0.05 y menor que persistencia con la misma agregacion. A se usa SOLO al evaluar,
nunca como feature. Este error relativo sustituye el umbral bruto A=1 anterior.
Es tolerancia funcional explicita, NO significacion ni margen de literatura.
Reportar tambien MAE bruto, por amplitud/fase/semilla y peor trayectoria.

No elegir M por test: curvas de todos los M son descriptivas predeclaradas.
Si el maximo falla, diagnosticar train/val, actualizaciones, recarga y controles;
no concluir inutilidad de las representaciones. Si pasa, concluir solo adecuacion
en esta rejilla retenida de amplitud/fase, no suficiencia financiera ni dominio
continuo. Una meseta mala no es suficiencia; una buena no garantiza fuera de rejilla.

## C6. Entrega completa y siguiente trabajo

Tabla de los 16 ajustes, controles y fallos: volumen unico, trayectorias,
train/val/test, error relativo, baseline, epoca equivalente y update del minimo,
update de parada, costo/RSS, recarga, identidad y conciliacion. Curvas y graficas
por amplitud/fase, predicciones/etiquetas persistidas, comandos reproducibles.
Guardar solo mejor checkpoint no basta para verificar un minimo: conservar la
evidencia val predeclarada. Confirmar loader y outbox avanzando, no solo HTTP 200.

Publicar retorno, work plan/estado, commits y sincronizacion. No parar tras un
paso para pedir "continua". Si aparece defecto resoluble en alcance, corregir con
regresion y preservar intentos; cambios cientificos se declaran como sucesor antes
de ejecutarse, sin usar test expuesto como reserva fresca.

Preparar despues, sin lanzar en este bloque: sensibilidad P16/P32 (h/P constante
separado de h=1), ruido y controles no lineales. Contraparte RL obligatoria: mismas
ventanas gobernadas, politica/estado/accion, tiempo de decision/ejecucion, costos,
baseline causal y oraculo con las mismas restricciones; posterior validacion del
negocio con reentrenamiento semanal. No sustituir RL por MAE de este control.

## Fuentes y limites

- [Finn et al., 2017, seccion 5.1, pagina 5](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf):
  regresion de senos con amplitudes/fases variadas. Es metaaprendizaje con coordenada
  como input, no nuestro forecasting ni una justificacion de estos numeros.
- [Hoiem et al., 2021](https://proceedings.mlr.press/v139/hoiem21a.html): curvas de
  aprendizaje para investigar volumen/modelo, no minimo universal.
- [Keras EarlyStopping](https://keras.io/api/callbacks/early_stopping/): semantica
  de minimo/restauracion; no garantia de ausencia de memorizacion.

Los niveles, rejillas y tolerancia son decisiones de ingenieria declaradas para
medir, no supuestos optimos derivados de publicaciones. Contar diversidad no
demuestra por si mismo que un modelo este bien ajustado.

## Comprobacion independiente del diseno, no del aprendizaje

Musashi ejecuto `python docs/handoffs/check_diverse_sinusoid_2026_09_18.py`:
4 pruebas pasan (conteos/anidacion, particiones, equivalencia de fase/duplicados,
diversidad numerica y recurrencia). 9472 ventanas numericas distintas comprobadas
con redondeo a 11 decimales; los calculos analiticos usan identidad racional.
No hay entrenamiento, resultado cientifico ni acceso a produccion en esta prueba.
Satoshi debe repetirla y probar D1-D9 sobre su implementacion real; no usar estas
cuatro pruebas como sustituto del ensayo gobernado o de la revision ML.
