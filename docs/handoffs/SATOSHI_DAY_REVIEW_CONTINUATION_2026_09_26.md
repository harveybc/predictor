# Continuacion tras auditoria de jornada: DR01-DR08

Musashi, 2026-09-26. Lee primero el
[dictamen](../audits/work_plan/MUSASHI_DAY_REVIEW_2026_09_26.md).
Esta orden reconcilia los carriles existentes: no reinicia el programa, no
autoriza capital real, no aumenta presupuestos y no sustituye al plan cientifico
por el plan de producto M5PHET. Incorpora suplementos posteriores antes de
despachar; un snapshot de este documento no prueba que un trabajo siga ausente.

## DR01. Proteger el coordinador sin parar trabajo independiente

1. Registrar los cuatro incidentes OOM identificados por journal, unidades,
   intentos, limites, costos perdidos y desenlace. No atribuir todos los casos a
   RAM fisica agotada: distinguir OOM del cgroup y presion del usuario/oomd.
2. **No despachar nuevos ajustes pesados ni barridos masivos al coordinador**
   mientras no exista admision compartida efectiva. No matar procesos vivos,
   cerrar aplicaciones del propietario ni modificar recursos bajo un hijo.
   Cerrar cada intento en su frontera normal y registrar cualquier fallo.
3. Implementar en el lanzador comun una reserva atomica por host que todas las
   rutas de computo usen. Bajo lock: leer capacidad fresca, reservas de cargas
   vivas, techo del slice, reserva del escritorio y presion; admitir o encolar;
   conservar la reserva hasta terminar el arbol de procesos. Recuperar leases
   tras fallo sin liberar la RAM de un hijo que siga vivo. No basta serializar
   el instante de lectura; el compromiso debe durar toda la carga.
4. Usar el pico del ARBOL/cgroup, no solo RSS del proceso principal; ligar el
   piloto a su evidencia; derivar los bytes de `--cap` sin otro entero discordante.
   Exigir presupuesto agregado observado, no solo RLIMIT del hijo. Pruebas con
   reloj/recursos simulados: dos peticiones que caben solas pero no juntas,
   hijo huerfano, lease vencido con proceso vivo, limite padre, PSI alto, OOM,
   memoria liberada, cap contradictorio y comando con `--` interior preservado.
5. Validar sin presionar RAM real; desplegar solo para lanzamientos futuros.
   No deshabilitar oomd, aumentar swap/topes, vaciar caches ni insistir despues
   de un rechazo. Un OOM no se convierte en simple reintento con limite mayor.

5090 externa es primera opcion para NUEVO trabajo GPU compatible. Las demas
maquinas siguen elegibles con admision y refrigeracion observadas. No cambiar
CPU/GPU o entorno de un experimento sellado sin declararlo y comprobar equivalencia;
reubicar CPU en otro host cuando sea adecuado. GPU libre no obliga a fabricar
un experimento ni a ejecutar uno sin datos, diseno o presupuesto.

## DR02. Un solo indice de despacho para todos los carriles

Actualizar `EXPERIMENT_EXECUTION_QUEUE.json`, el estado del programa y el indice
de asignaciones con una fila actual por A/B, R0/R1/R2-ECL, H-CORE, M4,
FIN-LOSS-OPT/E3, calendario, M5PHET cinco familias/chat, noticias, MT5 demo y
Alpaca paper. Para cada fila: commit, artefacto/recibo, estado observado/fecha,
dependencia REAL, trabajo elegible siguiente, recurso, coste restante y lease.
Clasificar snapshots como historia; no reactivar A/B ni el contraste ya terminado.

Mantener dos mapas ligados: producto M5PHET y experimentos doctorales/negocio.
El framework facilita las tareas, no certifica utilidad ni reemplaza sus ensayos.
La correccion de una familia no retiene las demas. Trabajar en worktrees aislados;
Hermes/subagentes pueden cubrir conjuntos disjuntos con un integrador y la misma
admision de recursos, no como otra via de lanzamiento.

Conciliar tambien el consumidor `df_e1_seal`: no puede seguir devolviendo
OUTSIDE incondicionalmente para una revision que el estado llama descargada.
Representar el dictamen actual y su alcance con identidad verificable, pruebas
de ausencia/contradiccion y rechazo por motivo real. El PASS de `check_plan`
es cobertura documental, no autorizacion cientifica.

## DR03. Q2 y gobernanza sin repetir por rutina

Conservar los intentos NON_GOVERNING y sus costos; no cambiarles retrospectivamente
la custodia. Resolver el cliente/actor con las referencias documentadas existentes,
sin buscar secretos indiscriminadamente, copiarlos al repo o inventar autorizaciones.
Probar la ruta entrega -> lector -> terminal -> warehouse con una unidad mecanica
pequena antes del siguiente fit cientifico. Si falta una referencia concreta,
nombrarla; no crear otro runner que evite esa dependencia.

La recuperacion y reportes CPU sobre evidencia conservada continuan. Antes de
repetir una celda, escribir que decision vigente la necesita, por que el registro
actual no basta y que contraste cubrira. Reparar el presupuesto acumulado en la
ruta ejecutable. Nunca interpretar null de cierre como error numerico cero.

## DR04. M4: reparar verificabilidad antes de confirmacion

Congelar los probes M4 del dictamen como PRE y convertirlos en pruebas de la
ruta publica real, sin construir datos confirmatorios nuevos. Reparar:

- autoridad ligada a la implementacion revisada; no solo HEAD registrado;
- censo exacto, semillas distintas, identidad completa familia/ruido/generador;
- evidencia cruda/autenticada y rederivacion numerica antes de VERIFIED;
- attrition y poblacion completa para TODOS los contrastes, incluido el 15;
- reanudacion con esquema completo, brazos, linaje, estado y disyuncion;
- accounting acumulado e interrupcion/concurrencia sin dobles ejecuciones.

Positivos y negativos bajo fixtures DEVELOPMENT, incluido el entrypoint completo.
No crear ni simular una firma de auditor externo. No correr CONFIRMATION hasta
que el protocolo, la implementacion y el record revisado correspondan entre si.
Corregir la frase cero arrays, distinguiendo prueba de disyuncion, materializacion,
ajuste y puntuacion. El auditor revisara el candidato reparado; esto no es su aprobacion.

## DR05. Corregir el razonamiento ML y avanzar el diseno correcto

Retirar el uso de la diferencia de etiquetas barajadas como piso de resolucion
o ruido del contraste. Retirar la mezcla kW/error escalado y los bloqueos globales
derivados. Ofrecer el mismo techo no implica mismos updates consumidos; decidir
si el estimando es receta bajo early stopping, coste igual o updates iguales y
declararlo antes de comparar. Conservar los controles utiles con su alcance real.

MOD-FROZEN-PREFIX puede desarrollarse, pero no se da por entregado porque haya
paridad de CSV/panel. Materializar la SALIDA del prefijo fijado
preprocesamiento/grupos/detector/adaptador/fusion, con estados aprendidos, reloj,
fila/split, forma y version. Probar directo vs cache, recarga en proceso nuevo,
mutacion de pesos/estado y causalidad. No abrir la reserva para construirlo.

Satoshi queda encargado de REDACTAR el candidato confirmatorio con hipotesis,
estimandos, tareas, particiones/reservas, multiplicidad, precision, presupuesto y
parada; Musashi revisara. Autoria del candidato no es firma del auditor ni permiso
de ejecucion. Usar el programa vigente y la evidencia ECL; el Huber recuperado es
antecedente household, no una receta ganadora universal ni financiera.

Eliminar H-CORE como bloqueo de TODO E3. Sus comparaciones RL mantienen sus
dependencias, pero preparar la referencia financiera y el protocolo semanal
forecasting/RL continua con sus propios datos, contratos y riesgos. No usar
electricidad como sustituto de validacion financiera.

## DR06. Lags, retencion y estado del checkout

Conservar Huber recuperado tal cual: la identidad independiente pasa. Matizar
cronologia fisica y `write_once`; no repetir barridos de todos los blobs para
demostrar un hecho ya ligado por un digest publicado.

Recalcular lags con offsets de reloj y mascara de parejas finitas, no borrando
posiciones. Nombrar ACF y Pearson por separado; la inversion dia/semana persiste,
sin convertirla en selector automatico de ventanas. Es reanalisis, no entrenamiento.

No limpiar ni comprometer archivos personales para hacer pasar identidad de
codigo en el checkout primario. El ignore de caches no lo vuelve limpio: hay
tambien documentos y otras evidencias sin seguimiento. Usar checkout de ejecucion
limpio y fijado; dejar los archivos del propietario intactos.

## DR07. LTS y aplicaciones en paralelo

Retirar impacto cero y comparar dependencias transitivas de timers, plugins y
entorno, no solo entrypoints. Ensayar runtime fijado en worktree/venv dedicado;
adopcion coordinada y rollback antes de cambiar rutas vivas. No restaurar ramas
debajo de procesos/timers; ninguna llamada mutante al broker por esta orden.

M5PHET/chat, los cinco proveedores, captura y aplicaciones siguen su trabajo
independiente. Reportar por familia una llamada real entrada -> motor -> salida
-> persistencia, native parity donde corresponda, limites y calidad; fixture no
equivale a modelo ni a utilidad financiera. El calendario conserva UNKNOWN y
rechaza estudios sin reloj/consenso admisibles. No detener mejoras mecanicas
porque falte ese dataset; no fingir la identificacion causal que falta.

## DR08. Entrega consolidada, no "continua" por cada subpaso

Completar los grupos elegibles sin pedir autorizacion repetitiva. Actualizar el
estado al terminar cada tarea y continuar las independientes. No cerrar el paquete
como completo si quedan corridas vivas: informar propietario/lease/heartbeat y
quien hace el cierre. Si un presupuesto o requisito real se agota, registrar la
espera SOLO en ese carril, sin inventar resultados para mantener una GPU ocupada.

Entregar PRE/POST, commits publicados y dependencias, ledger por host e incidentes,
estado real de servicios/campanas, y tabla experimental con tarea/horizonte/split,
metrica/escala, modelo, naive mismas filas, skill, referencia y comparabilidad.
Resultados previos se etiquetan como previos; ninguna suite sustituye a una medicion.
Sin nueva medicion, decirlo. No exigir al propietario redactar protocolos tecnicos
o restaurar ramas para suplir tareas del ejecutor/orquestador.
