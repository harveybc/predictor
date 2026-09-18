# Orden aprobada: control limpio P=8, W=17

Fecha: 2026-09-18. Owner: "aprobado", tras proponer P=8/W=17 como control
inicial y P=16/32 como sensibilidad posterior. Ejecutor: Satoshi. Diseno: Musashi.
Estado: EMITIDA, aun sin resultados. Alcance: DEVELOPMENT, prediccion sintetica.

## A1. Sustitucion y continuidad

No lanzar mas hijos del factorial 12D/T3 con periodos heredados por sorteo.
Si sigue activo, detener su despacho, cerrar ordenadamente los intentos con costo
observado y conservar la evidencia. No detener servicios u otras campanas.
La comprobacion anterior por df_adequacy/adequacy-v no encontro procesos en las
tres maquinas; verificar de nuevo incluyendo hijos antes de actuar.

Esta orden levanta la pausa general SOLO para este control. Implementar, probar,
ejecutar y cerrar A1-A5 sin pedir permiso entre pasos. No borrar ni reetiquetar
resultados antiguos. No lanzar aun barridos de ruido/sensibilidad, seleccion,
confirmacion, trading ni RL. A5 prepara su siguiente diseno, no los ejecuta.

## A2. Datos y calculos

Pregunta: aprende el pipeline real una relacion periodica limpia conocida, con
contexto suficiente, y se reproduce su error desde las predicciones guardadas?
No es una prueba de patrones no lineales complejos.

    x[t] = sin(2*pi*t/8), amplitud 1, fase 0, t entero
    P = 8 muestras por periodo
    W = 2*P+1 = 17 valores; duracion = W-1 = 16 = 2 periodos
    h = 1 muestra; avance de fase = 360/8 = 45 grados
    x[t+1] = sqrt(2)*x[t] - x[t-1]

P=8 es el control compacto aprobado, no un minimo universal ni un periodo del
mercado. Ocho fases por ciclo y una recurrencia exacta hacen el control verificable.
W cubre dos ciclos por requisito; dos rezagos ya bastan para la recurrencia limpia.
Correccion de Musashi: retiro P=32 derivado de una tolerancia de interpolacion.
La cota era correcta, pero interpolar no es requisito de muestras analiticas
exactas. Ese borrador no se publico ni ejecuto. No atribuir P=8 a un paper.

Entrada: solo valores x[t-16:t+1]. Target: x[t+1]. No fase, periodo, posicion
del ciclo, tick absoluto ni metadata como features. Tiempo en muestras sinteticas,
no ticks financieros. Rango [-1,1], sin ruido, sin wavelet/STL/denoising ni features
adicionales: se aisla la capacidad del camino crudo antes de comparar tratamientos.

4096 ventanas train, 512 validacion, 1024 test, stride=1. Son ocho fases repetidas,
NO miles de ejemplos independientes ni prueba de suficiencia de volumen/diversidad.

    N = 4096+512+1024 + 3*(W+h-1) = 5683 muestras brutas

| Particion | Decisiones inclusivas | Soporte bruto, inputs y etiquetas |
|---|---|---|
| Train | 16..4111 | 0..4112 |
| Validacion | 4129..4640 | 4113..4641 |
| Test | 4658..5681 | 4642..5682 |

Los soportes son disjuntos y las filas iguales para todos los aprendices.
StandardScaler escalar x ajustado sobre filas unicas 0..4111, compartido entre
rezagos; scaler y sobre etiquetas 17..4112. Validacion/test solo transforman.
Errores y predicciones en unidades originales. Metadata fuera de los tensores.

## A3. Aprendices, ajuste y aceptacion predeclarada

Persistencia x[t] sin ajuste; recurrencia analitica como control, no candidato.
OLS de dos rezagos sin penalizacion como prueba algebraica adicional.
Ridge lambda=1 con intercepto y W=17 como baseline aprendido; documentar la
convencion de regularizacion y contrastarla con el solver real.

Conv1D: seis capas causales de 16 filtros, kernel 3, dilataciones 1/2/4/8/16/32,
ReLU, ultima posicion a Dense(1) lineal. Sin pooling, BatchNorm, dropout ni
residuales. Llamarla Conv1D dilatada, no replica de una TCN residual.

    RF = 1 + 2*(1+2+4+8+16+32) = 127
    parametros = (3*1+1)*16 + 5*(3*16+1)*16 + 17 = 4001

RF es teorico: el contexto efectivo sigue siendo W=17, el resto es padding, no
datos. Se mantiene el grafo para preparar sensibilidad hasta W=65 sin cambiarlo.
No se afirma que seis capas sean necesarias u optimas para una sinusoide.

MAE, Adam lr=0.001, batch=64, max_epochs=200; early stopping por MAE de validacion,
patience=20, restaurar mejor checkpoint. Tres inicializaciones fijadas antes de
medir, registradas con entorno exacto; no son tres replicas de datos. No imponer
un segundo limite pequeno de updates que silenciosamente trunque el ajuste.

Aceptacion operativa: recurrencia y OLS float64 max_abs_error <=1e-10;
cada Conv1D test MAE <=0.05 (amplitud 1), mejor que persistencia, en las tres
inicializaciones. Persistencia aqui tiene MAE=0.5: el criterio exige al menos
90% de reduccion. Es criterio funcional declarado, NO significacion estadistica
ni margen demostrado por literatura. Reportar ridge sin exigirle el umbral NN.

Si falla: FAILED_ADEQUACY, curvas y diagnostico train/val/updates, sin avanzar a
representaciones ni concluir fracaso del dominio. No afinar sobre test. Cualquier
sucesor declara esa exposicion; no convierte fases repetidas en reserva nueva.

## A4. Pruebas y ejecucion

Antes de entrenar, fijar y pasar pruebas estructurales y conductuales:

| Requisito | Evidencia requerida |
|---|---|
| P/W/h y soportes | Calculos, identidades de filas y etiquetas |
| Causalidad | Prefijos/futuro perturbado invariantes; control con fuga detectado |
| Escalado | Futuro cambiado no modifica scaler ni train |
| Aprendizaje real | Grafo, parametros, updates observados, curvas y checkpoint |
| Recarga | Predicciones equivalentes tras cargar el modelo |
| Error real | Recomputado desde arrays retenidos, incluida persistencia |
| Datos usados | Entrega gobernada y digest en el lector productivo |
| Cierre | Archivo -> padre -> contabilidad -> DuckDB por contenido |
| Fallos | Timeout/error/reintento con costo y terminal sin duplicados |

Una prueba sin filas no pasa. El script de calculos adjunto verifica aritmetica,
no reemplaza pruebas contra loaders/modelos desplegados.

Un ridge y tres Conv1D, mas controles analiticos; no un nuevo factorial.
Piloto de costo train/val SIN test. Proyeccion con 25% holgura, techo de 14400 s
CPU acumulados y limite de memoria por hijo. Si no cabe, publicar necesidad medida
y estado exacto, no mutilar celdas por resultado ni exceder el presupuesto.
Distribuir ajustes independientes entre los tres roles con capacidad disponible;
si uno no admite trabajo, usar otro y registrar la razon medida. CPU en este
control, no sweep GPU. Recibir datos por data-gov con identidad propia.

Registrar campana antes de computar; ligar generador/datos/diseno/codigo/entorno/
semillas. Todos los desenlaces, costos y artefactos a DuckDB y conciliados. No
reiniciar servicios para probar. Publicar cierre aunque no pase aceptacion.

## A5. Entrega y siguiente etapa

Tabla por aprendiz/semilla: MAE train/val/test, persistencia, error de recurrencia,
ejemplos/updates observados, tiempo/RSS, parada, recarga y revision ejecutada.
Figuras de prediccion/residuos y curvas; arrays localizables, conciliacion completa.
Actualizar work plan y estado persistente, commit, push y sincronizacion habitual.

Preparar SIN ejecutar sensibilidad P=16/32, W=33/65: separar h/P constante
(h=1/2/4) de h=1 fijo, que son preguntas diferentes. Contexto, soporte, fases y
volumen comparables, sin cambiar fechas al variar W inadvertidamente.
Curva posterior con ruido: separar ventanas/ciclos/realizaciones/semillas de
optimizacion; una meseta mala no demuestra suficiencia. Controles no lineales,
tratamientos por variable, negocio semanal y RL siguen obligatorios y pendientes,
no validados por este control. Una suite verde tampoco demuestra adecuacion ML.

Fuentes ya revisadas y alcance limitado:
- [Bai et al. 2018](https://arxiv.org/abs/1803.01271): contexto y convolucion causal
  dilatada, no este grafo ni P=8.
- [Hoiem et al. 2021](https://proceedings.mlr.press/v139/hoiem21a.html): medir curvas
  de aprendizaje, no un minimo de muestras para esta tarea.
- [DeepLOB](https://www.oxford-man.ox.ac.uk/wp-content/uploads/2020/03/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books.pdf):
  referencia financiera de contexto/volumen, no adecuacion de nuestro control.
