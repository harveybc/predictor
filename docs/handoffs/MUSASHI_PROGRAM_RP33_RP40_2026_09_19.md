# RP33-RP40: E1 gobernado, validacion ML comparable y runtime RL real

Fecha 2026-09-19. Satoshi ejecuta; Musashi revisa. Base `60daac9`.
[Dictamen con contraejemplos](../audits/work_plan/MUSASHI_RP25_RP32_REVIEW_2026_09_19.md).
Esta orden sucede RP25-RP32. No hay una nueva decision del owner para registrar
los paneles publicos y ejecutar desarrollo. Se conserva 13D para simulacion;
no se pide capital real, venue o permiso de live para estos bloques.

## Limites y metodo

Ejecutar los ocho bloques completos sin pedir aprobacion entre pasos previstos.
PRE antes de corregir, pruebas top-down, implementacion bottom-up, evidencia y
estado persistentes. No repetir toda E0 ni borrar el piloto household. No abrir
reservas ni GPU, no entrenamiento RL cientifico ni operaciones financieras.
No alterar el programa doctoral para justificar un defecto del runner.

Techo **14 400 CPU s agregado**, incluyendo tests, pilotos, fallos, AE, fits,
verificacion y cierre. Preflight de memoria/tiempo por fase, holgura para cierre.
Tres hosts segun tareas independientes y carga; asignar semillas completas a
workers es posible sin separar AE y fits de esa semilla. Medir transferencia y
reuso por entrega gobernada, no afirmar que sincronizar cuesta mas sin medirlo.
No lanzar nada cientifico hasta resolver el prerrequisito de gobernanza.

## RP33. Registrar los datos y adoptar la ruta antes del siguiente trabajo

Resolver el objeto faltante, no devolverlo como permiso del owner. Inventariar
configuracion/servicios actuales y usar data-lake + proveedor externo y data-gov
existentes. Publicar un recurso acotado para household y, si cabe sin ampliar
riesgos ni contratos, el panel electricity ya caracterizado. No recenso global,
no copia suelta por scp ni nuevo backend. Fijar bytes, productor/parser, licencia
de la fuente ya documentada, roles y uso ARCHIVE/DEV; UNKNOWN temporal permanece
UNKNOWN, no elegibilidad point-in-time o live por tener un hash.

Probar contrato y recorrido completo primero con stack desechable; preparar
adopcion por el procedimiento operativo existente con respaldo, paridad y
reversion. Cualquier activacion permitida se acota al servicio/config necesarios,
sin tocar otros trabajos. No evadir una denegacion real de la herramienta:
registrar el objeto/operacion exactos y completar bloques independientes; no
inventar que la autorizacion cientifica del owner falta. Config pendiente no es
despliegue. No comenzar entrenamientos mientras solo exista config pendiente.

Aceptacion: cada host ejecutante usa identidad propia, obtiene el recurso por
HTTP/gobernanza, verifica hash, consume realmente esos bytes, registra campana
ANTES de preparar/fitear y cierra terminal -> outbox -> contabilidad -> DuckDB
por contenido. Host de entrada ausente debe impedir trabajo nuevo (cache solo
si verificada y autorizada por la ruta existente). Una sonda de formato no basta.

## RP34. Conservar el piloto no gobernado y reparar su cierre

Congelar originales y cronologia. Importacion retrospectiva con identidad,
fecha real de importacion, fecha original de ejecucion y ausencia de entrega
gobernada al ejecutar. No registrar hoy una entrega como si hubiera ocurrido
antes ni reutilizar sus terminales para afirmar adopcion prospectiva. Usar la
disposicion/importacion del warehouse existente; si no expresa este alcance,
extenderla aditivamente con tests primero. Reconciliar las 15 unidades al alcance
historico; conservar fallos/incidentes y no disfrazar esa carga como nuevo score.

Reparar close/verified_unit contra el job/diseno y poblacion registrados. Refusar
NaN/inf/bool/arrays vacios, MAE alterado, origenes/labels/denominador ajenos,
unidad trasplantada, horizonte cambiado y diseno con digest viejo O reparado.
Recomputar todas las metricas almacenadas desde arrays validados contra DATA y
la fuente/transformacion aceptada. Verificar pesos por replay en proceso fresco
con codigo y entorno declarados; AE/decoder y R1/R2 ligados a los bytes importados.
No aceptar un summary de gradientes como sustituto de un experimento ejecutado.
Poblacion ausente/parcial se declara por ids, no por all(presentes).

Probar los contraejemplos en el entry point de cierre completo. Comparar una
sola instancia verificada, no releer luego un rec distinto. Precisar aceptacion
de metricas vs inferencia vs regimen vs gobernanza; no una etiqueta que confunda
todo. Reverificar el piloto sin reentrenar. Corregir la equivalencia del donante
E0 (updates 1100 -> 0 no puede pasar), incluyendo tarea, entrenamiento y las
identidades que la union promete. Recerrar desde evidencia, no repetir fits.

## RP35. Contrato ejecutable y soporte de entrada

Validar dominios antes del enumerador: W/h enteros positivos para forecasting,
sin bool, fracciones de splits validas, rejilla/roles/targets/finitud coherentes.
Reconstruccion con h=0 es otra tarea, nunca acierto trivial del pronostico.
Pruebas desde loader y runner completos para horizon 0/-1, roles contradictorios,
metadata, huecos, ultima ventana y labels por identidad, no igualdad de valores.
Si mask_ffill se ofrece, emitir y consumir realmente la mascara de entrada;
si no, rechazar ese modo hasta implementarlo. Withdraw sigue permitido.

Granos de escalado declarados (filas unicas vs ventanas repetidas), mismo ajuste
train para comparadores; con datos futuros alterados, estados y entradas previas
son iguales. En validacion de modelos controlar soporte de informacion: lineal
de siete rezagos y de W completo se nombran aparte; estacional con historia fuera
de W es referencia de mayor informacion. No inferir igualdad por mismo target.

## RP36. Preentrenamiento y stopping medibles

Mantener ruido de entrenamiento variable si el diseno lo declara. Validacion AE
con mascara/banco determinista por origen/canal, invariante a epoch, orden de
lotes, reanudacion y evaluacion repetida; no resembrar por indice de batch si el
reordenamiento cambia el estimulo. Separar validacion interna de preentrenamiento
dentro de train, con soporte purgado, de la validacion de desarrollo supervisada.
Nunca test. Guardar identidades del banco y checkpoints necesarios.

Pruebas con el camino run_unit/WindowBatches/fit real: pesos fijos dan mismo
criterio; callback elige minimo comprobable; evaluate de pesos restaurados
reproduce ese minimo dentro de tolerancia declarada; reanudacion con estados
necesarios conserva la trayectoria al alcance elegido. AE diagnostico no prueba
mejor forecasting; R1 congela solo detector, R2 parte del MISMO detector.

Presupuesto alcanzado sigue siendo censura de optimizacion aun si el mejor
checkpoint fue anterior. Reportar pendiente, duracion y criterio de adecuacion,
no "no truncado" por argmin != ultima epoca. No ampliar stopping tras ver un
resultado favorable; cualquier cambio de diseno es sucesor predeclarado.

## RP37. Adecuacion del contexto y diseno sucesor de E1

No se exige ganar al lineal ni dos periodos por dogma. El objeto es comparar
preentrenamiento en un receptor capaz de utilizar el soporte declarado. Preservar
el detector Conv1D local de dos bloques; no preentrenar automaticamente el nucleo
(H-CORE sigue despues). Seleccionar el nucleo/readout con criterios de tarea,
curvas y soporte en TRAIN, no por cual da R1 ganador en la validacion publicada.

Opcion de desarrollo explicita: nucleo temporal causal dilatado de bajo costo
que cubra W sin cambiar detector/adaptador. Derivar capas/dilataciones/parametros
desde W y probar alcance con perturbaciones/gradientes; si otra solucion local
mas simple satisface el mismo requisito a menor costo, justificarla antes de
scores. No confundir soporte teorico con capacidad aprendida. Usar controles
de rezago distante y horizonte para que un receptor de siete muestras falle y
el receptor completo tenga una solucion conocida; incluir tarea sin estructura
predecible y control lineal competente. Solo diagnosticos DEV acotados.

Ficha numerica: W/h en muestras y tiempo, caracterizacion train, rango/unidades,
datos/ventanas utiles por split, variables que realmente entran, grafo entero,
alcance por rama/nucleo/readout, optimizador, learning rate, updates y stopping.
R0/R1/R2 con mismo grafo/resto e iniciales pareadas. No busqueda libre de modelos,
no aumento de datos/capas por prejuicio; presupuestar AE y suficiente ajuste con
curvas. Queda prohibido afirmar adecuacion solo porque el presupuesto cabe.

## RP38. Piloto E1 sucesor realmente gobernado

Una vez RP33-RP37 pasan aceptacion, sellar y ejecutar el piloto household DEV
con tres semillas pareadas y controles de soporte declarado. Mantener el panel,
target y horizon salvo cambio justificado por adecuacion antes de outcomes;
usar subparticiones train/validacion explicitadas. La validacion anterior ya es
DEV observada, no reutilizarla como confirmacion. Reserva final intacta.

Costo/memoria antes de lanzar; distribuir por semillas completas donde reduzca
wall medido, con cierre y cache gobernados por worker. No cortar brazos o semillas
tras resultados para caber. Si no cabe el diseno completo, priorizar reparaciones
y una ficha de presupuesto verificable, sin resultados parciales como experimento
completo. No iniciar otra auditoria general de infraestructura en su lugar.

Reportar MAE/MASE (definicion h de denominador explicita), diferencias pareadas,
cobertura, curvas, mejor checkpoint y censura, costo AE/fit/inferencia/metricas,
costo total vs costo amortizado. SD de semillas no es umbral de equivalencia;
una tarea con tres inicializaciones no da precision sobre otras familias. Todo
resultado, incluso negativo, llega al cubo por el flujo real antes del cierre.

## RP39. RL: conectar controlador, modelo y broker reales

Reutilizar entorno y broker desplegados de la ruta offline. `_run` del test no
es ese entorno y no sirve como aceptacion de integracion. El componente decide
modelo disponible y ejecuta SU inferencia, o valida identidad de la propuesta
contra esa seleccion; no etiquetar una accion externa como si la hubiera emitido
el modelo anterior. Fixture con modelos opuestos prueba fallback de verdad.

Validar reloj UTC, dominios/finitud, latencia positiva, fallback soportado,
precio/cash/equity/costos. Equity = efectivo + valor de posiciones; ordenes
pendientes reservan efectivo y forman parte del estado continuo. Cantidad/precio
al fill por API real; limites sobre patrimonio y comisiones no se prueban con
notional <= una variable que el test llama equity. Cierre flat sin short.

Tests integrados: cambio de semana con posicion Y orden pendiente, gap real,
delay de inferencia/publicacion/ejecucion, release tarde y modelo ausente, falta
de efectivo, NaN/inf, propuesta incompatible y modelo ajeno. Mutantes cambian
la ruta real y deben hacer fallar las mismas pruebas de aceptacion. Sin live ni
entrenamiento RL esta ronda; no reportar E3 cientifico completado.

## RP40. Una entrega completa, con limites ciertos

PRE/POST, suite y skips con entorno, ids/rutas de cada resultado, CPU/memoria
por fase/host, cronologia de importacion historica y nuevas entregas separadas.
Reconciliar datos -> arrays -> inferencia -> padre -> contabilidad -> warehouse
por contenido, sin promover documentos locales a recibos. Preservar originales.
Reusar implementaciones que ya probaron la propiedad; tests de schema solos no
prueban ciencia, contador de asserts no prueba cobertura.

Actualizar plan y estado de cada bloque; 5 frentes/6 propuestas, E3 semanal y
H-CORE permanecen. Publicar cambios y una sola solicitud final de revision. Las
operaciones tecnicas ya encargadas no se aparcan como decisiones del owner.
Salida `RP33_RP40_E1_GOVERNED_VALIDATION_AND_WEEKLY_RUNTIME_READY_FOR_REVIEW`.
