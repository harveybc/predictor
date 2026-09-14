# Orden a Satoshi: soporte de D2, portabilidad y siguiente diseno

Fecha: 2026-09-13. Prioridad: P0 calidad de decisiones, P1 preparacion D3.
Esta orden usa IDs D2-R1..R8; no renumera C185-C204 ni la adopcion en curso.

## 0. Inicio y alcance

Continue la orden `MUSASHI_TO_GENERAL_SATOSHI_FLOW_V3_ADOPTION_ORDER_2026_09_13.md`.
No la reinicie, no duplique implementaciones ni modifique su rama mientras
integra. La propuesta v2 es antecedente; Flow v3 gobierna la adopcion actual.
El ensayo y despliegue ya tienen su secuencia: no pedir otra autorizacion del
owner por haber recibido un packet anterior a esas correcciones.

Puede empezar R1-R2 y la preparacion R4/R7 ahora en un worktree separado.
R3 y toda medicion nueva que fundamente decisiones requieren el micro-run
Flow v3 reconciliado de la adopcion. Registrar esta dependencia y continuar
con los bloques independientes mientras se satisface; no detener todo.

Insumos:

- `predictor@d26fdc9`, retorno C166-C184 y codigo D2.
- Auditoria `MUSASHI_REVIEW_C166_C184_AND_DATA_GOV_2026_09_13.md`.
- Reproductor `musashi_d2_missing_metrics_review.py` junto a esa auditoria.
- `DECISIONS.jsonl` con SHA-256
  `f4958c88f8caa6b78d67fb7ff00c2a6a697276aa9b19b6b73411b78d97622510`.
- Diseno C171, tape C173, reserva y roots C174 conservados; obtener sus
  digests completos de bytes, no copiar abreviaturas como si fueran hashes.
- Plan vigente `06_ESTADO_REAL_PREPROCESAMIENTO_Y_SECUENCIA_2026_09_12.md`.

## R1 - PRE y contrato de soporte antes de editar

Congele las cinco salidas del reproductor y la tabla real de cinco decisiones
afectadas. No requiere entrenar, descargar ni regenerar la reserva.

Defina una tabla por unidad/variable/metrica con: aplicabilidad derivada,
soporte requerido/observado, estado, razon y componente responsable. Distinga
`NOT_APPLICABLE` de `INCONCLUSIVE`, ausencia de registro, fallo numerico y
falta de soporte. La verdad sintetica puede justificar aplicabilidad en el
evaluador, nunca convertirse en entrada del transformador.

Declare antes de corregir los tests estructurales y de comportamiento:

1. Retirar una metrica que prueba dano no transforma rechazo en pase.
2. Retardo o residual INCONCLUSIVE no satisface sus limites.
3. Evento inexistente por contrato es inaplicable, no medido como cero.
4. Evento presente con metrica faltante no se convierte en inaplicable.
5. Variable sin soporte permanece en el universo y denominador declarados.
6. Semillas sin contraste primario no se cuentan como contraste completo.
7. Menos de dos pares aplicables de no inferioridad no da `passed=True`.
8. SNR sin alguna variable requerida no mejora por excluirla del promedio.

Use la metodologia de diseno guiado por pruebas: requisitos -> aceptacion ->
integracion/unidades -> codigo -> verificacion ascendente. Estado y matriz
persistentes despues de cada bloque, no solo memoria de chat.

## R2 - Reparacion acotada del adjudicador

Corregir la derivacion de soporte de `decide_denoising` y revisar el mismo
patron en `decide_snr`. El loader tambien debe comprobar el universo esperado
de unidades/variables/metricas contra el contrato, no solo contar las filas
que llegaron.

El numero publicado para cada contraste es el de sus pares completos. Una
semilla con una metrica cualquiera no se llama semilla completa. Conservar
conteos de planificadas, observadas, completas e inaplicables por motivo.
Ninguna ausencia puede satisfacer una condicion requerida. Una metrica
legitimamente indefinida exige la disposicion que el contrato permita, no
imputacion cero ni un nuevo umbral escogido mirando esta confirmacion.

Los margenes, parametros, semillas, particiones y metricas quedan intactos.
Si una semantica no estaba definida, publiquela como aclaracion posterior a
resultados y resuelva conservadoramente el caso actual. No llamarla
predeclaracion. Un cambio de hipotesis exige diseno futuro, no parche oculto.

Incluir pruebas de omision por variable y semilla, presencia parcial y
estados inconclusos, ademas del control positivo completo. Mostrar que cada
regresion falla al retirar la correccion correspondiente. No ampliar a una
reescritura general de infraestructura.

## R3 - Re-adjudicacion desde evidencia existente, gobernada

Registrar una campana de revision Flow v3 que consume los arrays/resultados
historicos exactos como recursos de evidencia. Su recibo es actual; la
produccion original conserva su fecha y procedencia anterior a Flow v3.
No inventar descargas pasadas ni repetir toda la campana por cambiar el
transporte de resultados.

Re-derivar soporte desde las unidades conservadas, una unidad por vez.
Emitir decisiones sucesoras y tabla anterior->nueva para las 3.591 entradas,
incluyendo rechazos e inconclusos; ninguna fila historica se sobrescribe.
Recontar las 58 decisiones candidatas favorables y las 39 calibraciones SNR,
no solo las cinco detectadas. Explicar cada cambio con sus observaciones.

Primero re-adjudicar, no volver a ejecutar operadores. Si falta un hecho que
no puede reconstruirse, declarar el subconjunto exacto que lo necesita; no
rellenar ni lanzar una nueva confirmacion global. Estas salidas se someten a
revision, no conceden D3 o elegibilidad por si solas.

## R4 - AT9: portabilidad numerica sin mover la tolerancia historica

Mantener AT9 abierto con su criterio original. Producir primero desde los
registros existentes los margenes de las 1.026 decisiones SNR frente a TODAS
sus condiciones, no solo el error medio de la unidad encontrada.

Separar igualdad de bytes, tolerancia numerica y estabilidad de la decision.
Inventariar CPU, NumPy/SciPy/statsmodels, BLAS, hilos, precision, optimizador,
convergencia, limites y semilla bootstrap. El inventario publico usa roles,
no topologia privada. Instrumentar en copia diagnostica sin modificar el
codigo que produjo las observaciones historicas.

Congele un subconjunto de diagnostico antes de reejecutarlo: incluir el caso
AT9, controles de estimadores no iterativos, los cinco regimenes Kalman
calibrados y los casos mas cercanos a cada limite segun el inventario previo.
Elegir semillas por regla fija sobre IDs, no por el resultado del nuevo
replay. Un mismo input corre en los tres roles bajo limites medidos.
Es sensibilidad post-resultado, no confirmacion fresca ni prueba de toda la
poblacion. Reportar fallos de convergencia y cambios de intervalo/estado.

No ajustar tolerancias a 0.006 por conveniencia ni afirmar portabilidad
universal por esa muestra. Entregar una politica prospectiva por estimador:
entorno de referencia y alcance restringido, o nuevo algoritmo con prueba
independiente. Los componentes no afectados pueden someterse por separado;
Kalman no debe mantener bloqueado el diseno de todo el proyecto.

## R5 - Recursos y resultados, sin OOM deliberado

Para R3-R4: CPU, un proceso por unidad, un hilo de algebra lineal, inicialmente
un proceso por host, limite duro 2 GiB/proceso, maximo dos por host tras medir
el piloto; maximo 6 horas CPU agregadas y 4 horas de reloj para el diagnostico.
El inventario vivo debe respetar las reservas y trabajos ya presentes.
Si el coste estimado no cabe, reducir el subconjunto por la regla registrada
antes de ejecutar; no intentar las 3.972 unidades por inercia.

Nada de GPUs ni reinicios de maquinas. Usar los tres roles solo cuando haya
trabajo independiente y memoria libre; el caso AT9 si requiere comparacion
entre roles. No provocar un OOM para probar el limite. Reportar cada terminal,
incluidos resource-exceeded/inconclusos mediante la traduccion de estados
documentada por Flow v3, y conciliar outbox con cubo.

## R6 - Cobertura OLAP y actualizacion de propuesta

Preparar una vista de cobertura vigente con seleccion EXPLICITA de version y
codigo; conservar una vista historica con ambas matrices. No borrar ni
deduplicar sin procedencia. Probar que agregar las dos versiones no duplica
el denominador de la vista vigente. Ensayar en base desechable y publicar el
SQL propuesto antes de aplicarlo mediante la ruta de adopcion.

Actualizar la propuesta `05_PROPUESTA_MUSASHI_BETA_2026_09_13.md` con un
encabezado de supersesion y enlaces a Flow v3. Corregir la secuencia de beta
y la asignacion de reinicios, conservando el relato historico. Separar
IMPLEMENTED, DEPLOYED y GOVERNED_RUN_PROVEN con pruebas reales, no promises.

Cerrar el caso sintetico con el adaptador comun: generador+config+semillas,
hash de arrays materializados, roles observed/clean/noise/mask y particiones.
Si la API requiere un recurso descargable, registrar ese artefacto una vez y
reutilizar cache verificada. No disfrazar indices sinteticos como tiempos de
publicacion financiera. Clean/noise solo llegan al evaluador, nunca al fit.
Las sondas puramente mecanicas siguen exentas y marcadas NON_GOVERNING.

## R7 - Preparar D3 sin puntuar ni seleccionar todavia

Crear diseno y pruebas de aceptacion para cuantizacion/compresion,
representaciones tiempo-frecuencia y detectores, con entrada cruda conservada.
Vincularlo con feature engineering y `feature-eng` por contrato de plugins,
sin actualizar cientos de indicadores ni probar modelos por anticipado.

Por cada operador propuesto especificar bytes/parametros/estado de ajuste,
lookback, disponibilidad de salida, warm-up, retardo, costo y limites de
aplicacion. Incluir prefijos, sufijos futuros, bordes, chunk/restart y
control deliberadamente no causal. Una wavelet centrada y desplazar el
resultado hacia atras nunca son tratamiento causal de produccion.

No exigir que denoising gane en todas partes: rama cruda con diagnostico
completo y abstencion es una salida valida. D2 sintetico no demuestra utilidad
en datos publicos o financieros. Ninguna cifra SNR desconocida se toma como
verdad para elegir automaticamente el operador de una variable real.

No ejecutar D3 ni feature selection. Presentar su diseno con dependencias
concretas y un banco suficiente; sin ampliar la propuesta doctoral.

## R8 - Retorno unico

Entregar commits limpios por repo, PRE/POST medidos, tabla de impacto de las
3.591 decisiones, soporte completo de los cinco casos, informe AT9, coste por
rol, reconciliacion Flow v3, vista OLAP propuesta, diseno D3 y estado de
metodologia actualizado. Citar SHA completos leidos de artefactos.

Aceptar como salida honesta que un candidato quede no identificable o pierda
su pase. No elegir otro margen para recuperarlo. Detenerse en:
`D2_SUPPORT_READJUDICATED_PORTABILITY_SCOPED_D3_DESIGN_READY_FOR_REVIEW`.

No emitir records del revisor ni activar seleccion, entrenamiento, DOIN, RL o
live. No hay una accion nueva del owner en esta orden: las dependencias son
de implementacion, evidencia y revision tecnica.
