# Ordenes: disponibilidad veraz y adopcion del host real

Fecha: 2026-09-14. Ejecutor: Satoshi. Revisor: Musashi.
Continua A1-A5; no repetir publicacion ni despliegue ya aceptados.
Base: [dictamen con reproducciones](../audits/work_plan/MUSASHI_REVIEW_A1_A5_CONTRACT_AND_ADOPTION_2026_09_14.md).

## Decisiones ejecutivas

- PR 1 de financial-data NO se instala en su forma actual. Conservar el
  candidato y las mediciones; corregir por una revision sucesora trazable.
- Se conservan los doce terminales y los exitos mecanicos anteriores.
  No son prueba del host de entrada nuevo: corregir solo ese alcance.
- Las tareas de fixtures e integracion pueden empezar AHORA, en paralelo
  a la investigacion del productor financiero. No dependen de sus derechos
  de uso ni de una nueva decision del owner.
- No abrir D3, GPU, live, confirmacion cientifica o nuevos barridos.

## N1. Separar los relojes y la evidencia

Seguir el productor real del parquet, no inferirlo por el nombre de columnas.
Separar inicio/fin de ventana, finalizacion, publicacion, recepcion/ingestion
y version revisada. Anotar que existe realmente y que sigue desconocido.

La zona UTC en el esquema es evidencia de representacion temporal, no de
latencia ni de finalizacion. No copiar close_time como disponibilidad probada
ni convertir UNOBSERVED en cero. Un corte diario tambien tiene frontera.

Si solo se puede demostrar tiempo de evento, producir un contrato descriptivo
de archivo retrospectivo que no prometa disponibilidad point-in-time ni live.
Si el esquema actual no lo representa, proponer una extension minima con
pruebas del significado, sin relajar los contratos de consumidores existentes.
Una caracterizacion de archivo historico puede seguir siendo util sin afirmar
que el dato se conocia en aquella fecha.

Clasificar las 21 barras truncadas con evidencia de construccion; distinguir
parciales, cierres finales y agregados incompletos. Excluir lo no demostrado
del conjunto que requiere barras finales, conservando origen y razon.

## N2. Corregir las pruebas y la derivacion

Antes de editar, congelar los cuatro contraejemplos del dictamen y los casos
reales necesarios. Corregir el test live para probar la restriccion de ESTE
recurso, no la falta de zona horaria de otro contrato. Verificar UTC exacto,
rechazos tipados y hechos de completitud; no aceptar cualquier excepcion.

El generador debe separar hechos descriptivos y elegibilidad. Una medicion
fallida, tiempos incompatibles o procedencia insuficiente no producen un
contrato listo para instalar. Probar cada motivo y el caso positivo valido.
Una exclusion declarada puede permitir un sucesor valido; no rechazar sin
diagnostico todo el dataset por una fila si puede aislarse justificadamente.

Agregar recepcion al dia siguiente, revision tardia y barra no final. Para
transformaciones, ejecutar un control centrado deliberadamente no causal
contra el mismo verificador de prefijos/futuro del operador correcto. No
presentar pruebas de cortes de fecha como prueba de una transformacion.

## N3. Publicar fixtures adecuados sin esperar al contrato financiero

Preparar fixtures deterministas con los esquemas que consumen realmente
feature-eng y feature-extractor. Reutilizar generadores ya trazados cuando
sirvan; no renombrar datos financieros como sinteticos. Declarar generador,
semilla, roles x/y, horizonte, frecuencia, tiempos y hashes de salida.

Servirlos mediante data-lake + proveedor externo en una raiz SEPARADA del
lago financiero. Ampliar la instancia sintetica existente preservando el
fixture de dos filas y su identidad. Probar la configuracion candidata en
puerto alterno antes de activarla. Preparar entradas suficientes para los
cuatro consumidores, con roles y particiones explicitamente definidos.

La activacion acotada de este catalogo sintetico y los ensayos mecanicos
NON_GOVERNING estan incluidos en la orden. Usar ventana sin entregas,
respaldo de configuracion y reversibilidad. Reiniciar solo el servicio que
necesite cargar una configuracion nueva; no tocar PostgreSQL ni el loader.
La configuracion pendiente no es la activa; conservar rutas distintas.

## N4. Completar el recorrido de los cuatro consumidores

Preprocessor, predictor, feature-eng y feature-extractor deben consumir el
recurso sintetico por http_lake hacia el host instalado, NO por files_lake.
El arnes debe recibir lago y recurso por configuracion y registrar el
recorrido real, la identidad del proveedor y los recibos.

Completar primero feature-eng y feature-extractor; repetir para los otros
dos solo la prueba necesaria para demostrar la ruta que faltaba. Un pipeline
debe producir su salida real, con chequeos de esquema, alineacion, finitud
y conteos. No basta un comando vacio ni un fracaso transportado correctamente.

Probar la ruta con servicios desechables antes del micro-run productivo.
Verificar que quitar el host de entrada hace fallar la prueba por la causa
esperada; asi se distingue de un adaptador local que sigue funcionando.
Registrar costo, version instalada, entrada, salida, terminal y conciliacion.

## N5. Probar el reintento real y la clasificacion

En stack desechable, crear un terminal pendiente por caida del destino;
recuperarlo por el wrapper, verificar una sola fila y reenviar ese terminal.
Otra descarga desde cache debe dejar un recibo de cache verificable. No usar
pending=0 como sustituto de estos ensayos. Reutilizar pruebas previas solo
cuando coincidan codigo, ruta y contrato, y citar esa identidad.

Agregar tests del nuevo --classification de predictor: por defecto GOVERNING,
NON_GOVERNING persistido en campana y recibo, rechazo de valores invalidos.
Un replay ARCHIVAL_REPLAY_NON_AUTHORITATIVE no puede presentarse luego como
evidencia cientifica gobernante. Cubrir esa combinacion sin ampliar permisos
ni reconstruir los mecanismos de gobernanza ya aceptados.

## N6. Completar A4 donde falta, con presupuesto pequeno

Matriz requisito -> prueba -> resultado por consumidor: ajuste solo en train,
alineacion x/y/fechas, futuro que no altera el prefijo, batch vs incremental
cuando aplique, reinicio cuando haya estado, retraso causal declarado.
Aplicabilidad explicita: un consumidor puramente batch no necesita fingir
una API incremental, pero debe documentar la frontera temporal que usa.

Los controles negativos deben fallar por la causa buscada. No afirmar que
un hash elimina fuga ni que un pipeline ejecutado demuestra utilidad del
preprocesamiento. Las hipotesis cientificas conservan su plan independiente.

## N7. Investigar derechos y revisiones, sin inventar una decision del owner

Consultar fuentes primarias del proveedor y la documentacion del productor;
registrar enlaces, fecha y alcance concreto. Distinguir acceso a datos,
analisis, redistribucion y publicacion de artefactos. No asignar una licencia
por analogia con el codigo del SDK o con otro dataset.

Una segunda captura solo compara dos versiones; no demuestra una politica
general ni ausencia futura de revisiones. Si requiere una accion exclusiva
del owner, nombrar exactamente cual despues de agotar lo investigable.
Esta investigacion no bloquea N3-N6.

## N8. Continuacion DOIN y entrega

En los repositorios propietarios, definir y probar en aislamiento el resultado
gobernado de un replay offline agent-multi/DOIN, reutilizando el terminal comun
donde alcance. Crear un esquema nuevo solo si faltan campos semanticos reales.
No bloquear el comienzo por la ausencia de un nombre de esquema propuesto.
Sin publicacion de genes, transacciones, venue ni servicio live.

Entregar un unico packet con tabla por consumidor: ruta de entrada, resultado
de pipeline, costo, reintento REAL, alcance causal, terminal y conciliacion.
Separar transporte local legado, host nuevo, mecanica y evidencia cientifica.
Actualizar el work plan y README al cerrar cada etapa; no eliminar historia.
Empujar ramas y PRs con tests al tip final. No instalar el contrato financiero
sucesor sin revision. D2 y sus tolerancias permanecen intactos.

CPU y memoria acotadas: paralelizar solo bloques independientes dentro de los
limites medidos. No detenerse esperando al owner cuando otra tarea pueda avanzar.
