# Orden vigente RP1-RP8: ejecutar trabajo del programa, no otro control aislado

**Historica tras el retorno `2347c7b`:** siguiente orden vigente
[RP9-RP16](MUSASHI_MOD_E0_RP9_RP16_2026_09_18.md).
Conservar esta orden como contrato de la ejecucion anterior, no volver a lanzarla.

Fecha: 2026-09-18. Emisor: Musashi por la instruccion del owner de revisar,
corregir y ejecutar el programa completo con Satoshi. Responsable: Satoshi.

## Alcance y fuentes obligatorias

Leer el [plan maestro v3](../tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md),
la [propuesta modular](../propuesta_doctoral_representaciones_temporales_modulares.tex),
el [contrato de metricas](../tres_temas_entrevista/PROGRAM_METRICS_CONTRACT_v1.md)
y el [estado persistente](../tres_temas_entrevista/program_v3/PROJECT_METHOD_STATE.json).
El master contiene enlaces a TODOS los otros protocolos. No sustituir P-MOD por
el selector L2 ni reducir el programa a forecasting. Respetar las hipotesis de
cada documento; no declarar que un resultado sirve para otro sin contrastarlo.

**Retirada la orden C1-C6 como proxima campana**, incluido su periodo/ventana/red
como defaults del programa. No ejecutar su barrido ni P16/P32 como siguiente paso.
Preservar evidencia ya producida y pruebas reutilizables. Cerrar con costo cualquier
intento incompatible propio, sin interrumpir cargas ajenas ni borrar historia.

Esta orden autoriza diseno, implementacion, pruebas y piloto **DEVELOPMENT** de
MOD-E0-DEV en los tres roles disponibles. No autoriza reservas confirmatorias,
operacion financiera real ni convertir datos retrospectivos en point-in-time.
No hay aprobaciones adicionales por cada paso normal. Resolver defectos dentro
del alcance con regresiones y continuar hasta el cierre RP8; un resultado nulo
no es motivo para pedir permiso ni para reajustar mirando la prueba.

## RP1. Reconciliar y registrar el trabajo correcto

Integrar el master v3 y la correccion de los documentos historicos de la rama
`musashi/program-plan-restart-20260918`; correccion historica publicada en
[`4a53b49`](https://github.com/harveybc/predictor/commit/4a53b493fc9e4cc901c752f72d25b07ca974d615).
Publicar identidad de origen y diff. Importar los documentos/estado del plan sin
reemplazar codigo o evidencia de la rama de trabajo por los de master.
Releer las ordenes vigentes de master por nombre, no usar una copia C anterior.
Actualizar el estado persistente y 09_ADOPCION antes de despachar.

Reverificar solo la evidencia que se reutilizara: datos/generador, perfiles,
operadores, pruebas de causalidad, loader, early stopping, checkpoint y metricas.
La compatibilidad incluye contrato cientifico, soporte, entorno y ruta ejecutada;
no basta que coincida el nombre de una funcion. No repetir los 715 perfiles, D2,
T2 o todas las antiguas calibraciones como peaje del piloto modular.

Registrar campana y unidades antes de trabajo gobernado. Separar comprobaciones
de aceptacion, coste de desarrollo y contraste cientifico. Un JSON de plan o tests
del plan no dan elegibilidad ni equivalen a medir el modelo.

## RP2. Disenar MOD-E0-DEV antes de entrenar

Entregar una ficha numerica ejecutable y legible, con derivaciones reproducibles:

1. H2: tareas multivariadas con niveles de heterogeneidad temporal; perfiles vs
   varias asignaciones aleatorias, manteniendo arquitectura/contexto/informacion.
2. H3: tareas emparejadas con y sin dependencia retardada; mismos marginales en
   distribucion; mismo extractor fijo y mismas activaciones entre brazos.
3. Combinar procesos periodicos y autorregresivos; incluir tendencia/evento como
   caso de fallo diagnostico predefinido si no forma parte del contraste principal.
   No convertir todos los factores en un factorial gigante antes del piloto.
4. Muestreo, periodos/persistencias, retardos, ventanas, horizontes y N: derivarlos
   del mecanismo, del soporte necesario y de cuantas unidades quedan tras separar
   particiones. Expresar ventanas tambien en tiempo/ciclos. No existe minimo
   universal de dos periodos ni 2000 muestras por red; comprobar sensibilidad y
   curva de aprendizaje en desarrollo, no adivinar suficiencia.
5. Demostrar que el objetivo es predecible con informacion causal y que el control
   positivo distingue el mecanismo de agrupacion/fusion. Declarar suelo/oraculo,
   referencias, fuentes de dificultad y que informacion no ve el modelo.
6. Volumen: trayectorias/configuraciones distintas, replicas independientes,
   ventanas, datos unicos, mini-batches y updates por separado; limites de purga
   derivados de soportes reales, no una resta fija que elimina o anade filas entre
   brazos. Calcular el denominador MASE y su politica de cero antes de puntuar.
7. Arquitecturas completas por brazo: capas/tipos/unidades/activaciones, contextos,
   RF, grafo, trainable/frozen, optimizer/lr/batch, presupuesto/early stopping.
   Cada numero lleva motivacion y alternativa de sensibilidad. La figura del PDF
   es una ilustracion, no obliga a copiar sus anchos como optimo.
8. Unidad y estimandos H2/H3 del master; precision del piloto y regla para derivar
   el tamano posterior mediante simulacion. No llamar confirmatorio a este piloto.

Separar desarrollo y reserva cientifica por parametros/generadores/semillas, no
solo filas de la misma trayectoria. No generar ni abrir la reserva en esta orden.
Fijar el primer diseno y presupuesto antes de producir sus resultados. Cambios
tras resultados DEV se documentan como sucesores de desarrollo y no conservan
pretension confirmatoria. Publicar la justificacion, no esperar otra microorden.

## RP3. Pruebas de aceptacion primero; implementar solo brechas

Usar requisitos -> aceptacion -> sistema -> componentes -> integracion -> unitarias,
luego implementacion -> unitarias -> integracion -> sistema -> aceptacion alpha.
Mantener estado en disco y enlaces requisito/test/evidencia. Cuando una pieza ya
esta implementada y su prueba real sirve, reutilizar ambas.

| ID | Obligacion antes de puntuar |
|---|---|
| ML01 | Generador reproduce ecuaciones, poblacion y distribucion; etiquetas/metadata/grupo latente no entran como features |
| ML02 | Cambiar futuro/validacion/test no cambia fit, grupos ni valores emitidos previamente; roles y alineacion por identidad/tiempo |
| ML03 | H2 realmente cambia asignacion, no nombres; preserva tamanos, modelos, contextos y presupuesto por brazo |
| ML04 | H3 conserva marginales en distribucion y cambia dependencia relevante; el control no destruye autocorrelacion para simular independencia |
| ML05 | H3 usa activaciones identicas del mismo extractor; congelacion de pesos/estado; resumen vs secuencias localizable en grafo real |
| ML06 | Grupo -> detector/integrador/adaptador -> fusion -> nucleo/cabezal preserva forma temporal donde corresponde; gradientes reales |
| ML07 | Aprendizaje positivo con receptor neural real; referencia/oraculo adecuados; curva train/val y volumen distinguen falta de entrenamiento de falta de utilidad |
| ML08 | Early stopping selecciona por validacion, restaura pesos reales, recarga reproduce y test no influye; no inferir overfit solo por gap |
| ML09 | Metricas desde arrays independientes del resumen; denominadores/poblacion/pesos compartidos; ejemplos vacios o no finitos no pasan |
| ML10 | Permutaciones, semillas y ventanas no inflan la unidad; calculo de e(h), d0/d1/gamma y multiplicidad sin invertir signos |
| ML11 | Loader lee bytes entregados, CPU/GPU updates observados, costes completos, limites y estados de parada; sin fichero de muestra alternativo |
| ML12 | Fallo real deja terminal/outbox; conciliacion por poblacion y contenido contra contabilidad independiente; metricas informacionales con version y NA razonado |

Para cada bloque critico, una alteracion deliberada del camino productivo debe
hacer fallar la prueba correspondiente. No es necesario probar todas las
dependencias del universo: cubrir generador, fit, fusion, tiempo, metrica y
persistencia que consume este experimento. Guardar PRE/POST cuando se corrige
un defecto existente, antes de cambiarlo, sin reescribir los resultados previos.

ML07 exige adecuacion del receptor y del control, **no** que ganen H2/H3. No
seleccionar generadores por producir una ventaja modular ni eliminar casos donde
un control competente iguale o supere al metodo. Precisamente eso se quiere medir.

Trazar plugins y entry points reales de predictor/feature-extractor/feature-eng/
preprocessor. No declarar modelo modular porque un wrapper se llame asi. El helper
`build_branch` inspeccionado empieza con Flatten: no reutilizarlo como si
conservara el eje temporal. Comprobar quienes lo consumen antes de modificarlo.
Implementar variantes acotadas compatibles con el sistema de plugins, sin romper
configuraciones historicas ni fabricar una segunda plataforma.

## RP4. Instrumentar lo necesario y ejecutar el piloto completo

Reutilizar data-gov, data-lake/proveedores, data-warehouse DuckDB y outbox existentes.
Primero comprobar identidad de servicio y salud de loader por avance/pendientes,
no solo 200. No migrar motores ni reiniciar servicios por conveniencia del ensayo.

Aplicar contrato de metricas del master: D/Y por entrada y target una vez; M/G en
checkpoints con frecuencia y costo declarados. Primera pasada inventaria lo que
ya se calcula; implementar faltantes aplicables antes de producir las unidades
que prometen medirlos. Registrar NO_APLICA/NO_IDENTIFICABLE/NO_MEDIDO con razones
reales; no usar esas etiquetas para ocultar una metrica requerida sin implementar.
No afirmar que este piloto ya contrasta P-CAP o H-ES: aqui recolecta descriptores.

Piloto de costo train/val antes de la rejilla; limite acumulado **14400 segundos
CPU** heredado como techo operativo, no como justificacion del tamano estadistico.
Proyeccion incluyendo 25% de holgura, intentos y costo de instrumentacion. Si no
cabe, disenar una etapa DEV mas pequena antes de sus scores conservando ambos
contrastes y todas las condiciones del bloque; no eliminar celdas que salieron mal.
Si tampoco cabe una etapa informativa, entregar ese deficit medido, sin afirmar
que se completo el experimento. No consumir una reserva para salvar el presupuesto.

Distribuir unidades independientes en coordinador/WORKER_A/WORKER_B segun memoria,
cargas y costo medidos; cada rol con identidad y entregas propias. CPU es suficiente
para esta primera etapa si el piloto lo confirma. GPU solo si se incluye en el
diseno de costo autorizado y se respeta trabajo en curso; no arrancar GPU por
llenar las tres maquinas. Informar asignacion, cola, utilizacion y memoria reales.

Ejecutar H2 y H3 de desarrollo, no detener por un efecto negativo. Si el control
positivo falla, diagnosticar y corregir dentro de DEV antes de aceptar scores:
conservar intentos, versionar cambios y volver solo sobre evidencia afectada.
Si sigue sin separar, el resultado es limitacion del instrumento, no refutacion
de la propuesta. No emitir ADVANCES ni elegir representacion financiera.

## RP5. Reconciliar resultados y responder la pregunta cientifica

Por condicion y replica: perfiles/grupos, W/h/RF/N efectivo, regimen R0/R1/R2,
grafo/parametros, entrenamiento observado, MAE/MASE y denominador, e(h), d0, d1,
gamma, dispersion, costo y disponibilidad; train/val/evaluacion DEV separados.
Graficas de prediccion, curvas, grupos y efectos por mecanismo. No mostrar solo
un agregado que oculta que el modelo no aprendio o que una familia se perjudica.

Recalcular desde predicciones/labels persistidos, comprobar congelacion real del
extractor H3 y la igualdad de informacion H2. Archivos -> padre -> contabilidad ->
warehouse, todos los hijos/estados incluidos. La consulta debe poder encontrar
cada unidad por propuesta, hipotesis, diseno, brazo, condicion, replica y fase.
Preservar intentos fallidos y tests historicos como historia, no duplicar ciencia.

El retorno explicara: que cambio, cuanto cambio en unidades originales y MASE,
que mecanismo parece contribuir o fallar y que falta para la confirmacion.
No decir "mejora" sin su diferencia, comparador, incertidumbre y alcance.

## RP6. Preparar el siguiente paso real y la contraparte RL

Sin esperar un resultado positivo para trabajar en ello:

- Reutilizar censo y perfilar solo el subconjunto faltante del banco E1 publico
  multivariado; presentar familias distintas DEV/reserva, licencias, disponibilidad,
  variables/objetivos, geometria y costo. No descargar o recaracterizar todo otra vez.
- Inventariar demanda de forecasting/RL desde configs realmente ejecutables;
  fijar ciclo semanal y decisiones de negocio, datos/horizontes, recompensa,
  acciones/ejecucion/costos y brechas. No adivinar activos ni activar live.
- Dejar diseno E1 con referencias de la propuesta, R0/R1/R2 y seleccion de contexto/
  grupos sobre validacion; protocolo E3 forecast+RL obligatorio, no condicionado
  a que H1 sea positivo ni a desarrollar el selector L2 de otra propuesta.
- Dejar las tareas siguientes de P-PRE/P-TRN, P-CAP, P-L2 y P-INC con identificador,
  fuente, dependencia y entregable. No implementar todas en esta ronda ni
  convertirlas en nuevas propuestas sin plan. Ninguna queda fuera del programa.

## RP7. Reusar infraestructura sin relegar ciencia

No desarrollar un nuevo dashboard, biblioteca de grafos o motor de orquestacion
como sustituto del piloto. Resolver solo brechas consumidas por RP1-RP6; mantener
el resto en el plan con siguiente tarea y evidencia necesaria. Metabase, el
incidente historico del indice y terminos financieros no son prerrequisitos de
un banco sintetico propio que no los usa.

Controlar el carril feature-eng con config reproducible y salidas gobernadas;
la futura seleccion de features es anidada y posterior a caracterizacion, no
meter todas las columnas en el modelo ni normalizarlas todas por costumbre.
La reserva de P-MOD, los bancos de P-L2/P-CAP y los datos usados por P-INC no se
mezclan para contar mas replicas. No borrar negativos o diagnosticos anteriores.

## RP8. Cierre integral

Actualizar estado persistente al completar cada etapa, no rellenarlo al final
desde memoria. Ejecutar comprobacion del plan y pruebas de todas las rutas tocadas
con sus entornos y exclusiones declarados; no presentar tests focales como suite total.
Publicar diseno, fichas, contratos, tablas, graficas, costos, conciliacion y retorno
con commits/links GitHub. Sincronizar codigo y entradas por las rutas gobernadas.

Cerrar con disposicion factual por bloque: diseno, implementado, ejecutado,
verificado e inferencia son campos distintos. Si falta una pieza, el paquete
no puede decir "todo terminado". Identificar el objeto exacto, impacto y siguiente
accion; resolver lo de ingenieria que este al alcance antes de escalarlo al owner.

La revision externa del piloto sera de Musashi. Esta orden NO afirma que esa
revision o la campana ya ocurrieron. Entregar una sola solicitud de revision al
completar el alcance, sin detenerse entre tareas normales para pedir "continua".
