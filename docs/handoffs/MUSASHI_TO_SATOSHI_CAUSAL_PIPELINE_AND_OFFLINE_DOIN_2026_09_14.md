# Ordenes: causalidad del pipeline y replay offline DOIN

Fecha: 2026-09-14. Ejecutor: Satoshi. Revisor: Musashi.
Autorizacion del owner ratificada; no se requiere otra para empezar.

## Punto de partida, ya resuelto

Leer [el acta de activacion y cuatro consumidores](MUSASHI_SYNTHETIC_CATALOG_AND_FOUR_CONSUMERS_ACCEPTANCE_2026_09_14.md).
El catalogo sintetico ampliado YA ESTA ACTIVO. Los cuatro consumidores YA
completaron su micro-run por el host nuevo y el cubo contiene los doce
desenlaces adicionales. No reiniciar otra vez ni pedir al owner que ejecute
ese procedimiento. Reconciliar el estado y continuar, conservando los datos.

Esta orden ejecuta el trabajo causal pendiente de N6 y el replay pendiente
de N8; no es otra ronda de publicacion o de pruebas de infraestructura.
La investigacion financiera puede avanzar en paralelo.

## P1. Roles de columnas y contrato de entrada explicito

El incidente de la fecha ISO entrando en un tensor revela que no basta con
quitar esa columna del fixture. Implementar en los repositorios propietarios
la seleccion explicita de features, targets y metadata temporal, siguiendo
las convenciones existentes y evitando un framework nuevo innecesario.

Pruebas antes del cambio: metadata temporal como texto y como numero;
columna adicional no declarada; permutacion de columnas; target presente
en el archivo x; cambio de configuracion. Los modelos deben consumir SOLO
las features declaradas, en orden estable, conservando metadata en el recibo.
Una entrada antigua sin contrato necesita una migracion explicita, no una
seleccion heuristica que permita incorporar targets inadvertidamente.

Conservar el fixture desplegado y sus identidades. Para las pruebas nuevas
crear un sucesor con roles y contratos nuevos, sin sobrescribirlo.

## P2. Banco pequeno que pruebe causalidad, no solo esquemas

Generar una trayectoria sintetica conocida y dividirla cronologicamente,
con ventanas x/y y horizonte definidos. Validacion y prueba empiezan despues
del entrenamiento y de su embargo cuando corresponda. Las tres semillas
del fixture anterior no equivalen a tres particiones temporales.

Incluir: tendencia, componentes periodicas, saltos/extremos, ruido conocido
y faltantes. Usar un subconjunto pequeno, no otro barrido masivo. Registrar
que pruebas aplican a cada regimen. La disponibilidad es un reloj SIMULADO
con regla declarada, no una observacion del proveedor de datos.

Para indicadores tecnicos, producir los valores mediante feature-eng desde
OHLC sintetico consistente. No tomar columnas arbitrarias con nombres RSI,
ATR o MACD como evidencia de su calculo correcto. Cotejar al menos un caso
simple por operador con una referencia independiente.

## P3. Ejecutar las pruebas de procesamiento pendientes

Sobre los operadores y plugins realmente usados por las cuatro configuraciones:

1. Ajustar en train y congelar el estado antes de transformar validacion/test.
2. Cambiar y ampliar la cola futura: el estado de train y el prefijo ya
   disponible no deben cambiar bajo el mismo contrato.
3. Alinear features, targets, fechas y disponibilidad despues de lookbacks,
   recortes, horizontes y missingness; no cerrar huecos del calendario.
4. Probar el tratamiento del borde derecho y cualquier padding de filtros.
5. Ejecutar un control centrado no causal, un scaler ajustado con test y un
   desplazamiento hacia el futuro. Deben fallar por la propiedad buscada.
6. Medir el retraso de cada operador. No prometer denoising sin retraso como
   propiedad general. Wavelets no reciben excepciones.
7. Batch/incremental y restart solo donde exista una API con estado; declarar
   no aplicable con razon donde no exista, sin afirmar esa equivalencia.

Orden de implementacion: unitarias -> integracion de transformaciones ->
pipeline real -> aceptacion. Mantener una matriz persistente de requisito,
prueba, codigo, resultado y siguiente accion. No relegar estas pruebas a
"la suite del consumidor" sin ejecutarlas e identificar sus casos concretos.

Si aparece fuga, conservar la reproduccion, corregir el operador o excluirlo
del conjunto causal con razon, y seguir con los operadores que pasan. No
borrar resultados previos ni detener toda la campana por un plugin descartable.

## P4. Cerrar dos huecos pequenos del arnes, sin nueva campana pesada

- Clasificacion: invocar el parser real y un entry point acotado. Probar
  GOVERNING por defecto, NON_GOVERNING en campana/recibo y la incompatibilidad
  de declarar evidencia gobernante desde un replay no autoritativo. La
  reconstruccion de argparse y los asserts de texto no son esos ensayos.
- Ruta: conservar que discover puede no estar concedido. Registrar ruta no
  observable cuando corresponda y aceptar evidencia del recibo/configuracion
  del operador; no inferir "local" ni "http probado" desde campos null.
- Outbox: extender la prueba real de pendiente-recuperacion a los wrappers
  que no comparten exactamente la misma implementacion. En los que si la
  comparten, demostrar el punto de llamada y probar su uso. Todo fallo de
  destino se simula en stack desechable, nunca derribando produccion.

## P5. Instrumentar un replay offline de agent-multi/DOIN

Reutilizar el contrato comun de resultados y los adaptadores existentes.
Crear una extension solo si hay un campo semantico necesario que no cabe.
El replay debe consumir datos sinteticos identificados, ejecutar una unidad
pequena de trabajo real y entregar metricas, costos y desenlace al warehouse
configurado. No introducir llamadas remotas dentro de cada paso del agente.

Primero stack desechable, despues un micro-run NON_GOVERNING en produccion
si pasa. Esta ejecucion acotada esta autorizada. Sin broker, venue, operaciones
financieras, publicacion de genes, blockchain productivo ni GPU. No detenerse
por no existir aun el nombre de esquema que se propuso en una carta.

## P6. Terminar la investigacion temporal y la propuesta retrospectiva

Corregir PR 2 para distinguir reloj de recepcion y reloj de publicacion.
El test actual pasa received_time como publication y lo marca MEASURED;
no es evidencia de publicacion. La semantica debe venir de un contrato del
productor y la medicion debe comprobar su consistencia, no adivinar su rol.

Las duraciones de las 21 barras permiten categorias geometricas, no prueban
por si solas por que el productor las genero asi. Mantener finalizacion
desconocida donde falta evidencia. Consultar las fuentes primarias pendientes
sobre derechos y revisiones; asignar al owner solo acciones exclusivas que
realmente hagan falta, con el recurso y la accion concretos.

ARCHIVE_RETROSPECTIVE es una direccion aceptada para no inventar disponibilidad.
Completar pruebas de compatibilidad extremo a extremo: proveedor, host,
data-gov, cliente, recibo, cubo y consumidor. UNKNOWN no puede convertirse en
cero en ningun punto. Probar entrega de archivo retrospectivo y rechazo de
solicitudes que prometan point-in-time; conservar los holdouts vigentes.
No instalar una politica financiera nueva sobre archivos reales en esta orden.
Esta tarea no bloquea P1-P5 ni requiere otra aprobacion para investigarla.

## P7. Resultados, publicacion y recursos

Los resultados experimentales, incluidos fallos e inconclusos, se registran
por data-gov con generador/semilla/configuracion y hashes. Las unitarias pueden
ser locales; sus resultados quedan ligados al commit y a la matriz de pruebas.
Sin borrar el cubo ni reutilizar IDs para otra evidencia.

CPU acotada: unidades pequenas, techos de memoria y tiempo, concurrencia segun
memoria disponible en cada maquina. No ocupar GPUs por apariencia de actividad.
Los experimentos cientificos D3 conservan su diseno y revision separados; esta
orden no modifica resultados D2, sus tolerancias ni sus licencias por regimen.

Entregar un packet con lo EJECUTADO, no solo propuestas: tabla de operadores
causales/descartados, pipeline compuesto probado, adopcion offline DOIN,
compatibilidad retrospectiva, investigacion documental y limites restantes.
Actualizar el plan conforme termina cada bloque y empujar las ramas. Cambios
en codigo de servicios se presentan por PR con pruebas; no actualizar el
servicio sano a una revision que no corresponda a la evidencia aceptada.

El siguiente alto es para revisar resultados nuevos, no para volver a pedir
la autorizacion de empezar ni la activacion que Musashi ya realizo.
