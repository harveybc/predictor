# Revision de d845563e y siguiente paso sintetico

## Dictamen acotado

Acepto la reparacion de los dos contraejemplos del archivo en a4f65ba: la ruta
compara el hash solicitado con los bytes del manifiesto y reconstruye los
registros desde bytes verificados, sin consumir los diccionarios mutables.
El conteo usa COUNT(*) dentro de la transaccion, sin el tope de lectura de 1000.
Reejecute tests/test_offchain_shadow.py: 21 passed, 0.43 s. No certifica un lago
desplegado ni consenso: chain_verified sigue falso. No encontre un nuevo bloqueo
en el alcance de esas reparaciones; no es una auditoria completa de DOIN.

Reejecute tests/unit_tests/test_strategy_sweep_20260930.py contra 7a50f62:
6 passed, 6.48 s; el barrido completo ejecuto 44 celdas. Compare recursivamente
RESULTS.json regenerado contra el objeto Git: TODOS los campos coinciden
exactamente salvo wall_seconds/cpu_seconds por celda. La ejecucion del barrido
midio 4.469907441 s CPU y 4.471066055 s pared, dentro de 2 GiB/120 s del wrapper.
No hubo GPU, mercado, servicios, calibracion financiera ni codigo remoto.

Los tests escriben los dos JSON en la carpeta de evidencia versionada. Esta
revision restituyo exclusivamente los dos archivos regenerados por su propia
ejecucion, conservando el contenido publicado. Cambiar los tests a tmp_path y
hacer explicita la exportacion de una nueva evidencia; correr pytest no debe
reescribir silenciosamente una medicion retenida.

## Resultado y limite cientifico

Es un barrido sintetico reproducible, no solo un manifiesto. El ruido cambia MAE;
con largo ideal, los nueve casos de corto ruidoso mantienen tres cierres y PnL
2659.4364862832954. Persistencia larga con corto ideal/persistente no entra.
El largo ruidoso modifica el libro y puede dejar posiciones abiertas: comparar
solo PnL realizado seria insuficiente; conservar equity y exposicion final.

No hay ranking corto/largo. Solo existen cuatro origenes con pronosticos. El
plugin revisa el cierre predictivo solo cuando hay una fila para ese instante;
fuera de ellos mantiene los controles de TP/SL pero no una nueva prediccion.
Por eso no extrapolar esta insensibilidad del fixture a un cierre temprano
alimentado cada hora. Los 44 casos comparten una trayectoria; las semillas son
ruido de prediccion, no mercados independientes. Los baselines duplicados entre
orientaciones son controles compartidos, no observaciones independientes.

## Trabajo paralelo siguiente

A. Estrategia: pruebas de comportamiento primero. Crear soporte SINTETICO horario
continuo, con pronosticos en cada decision elegible y cola futura suficiente para
144 h. Sin cambiar variante E, TP/SL, dimensionado, comision ni reloj de fills.
Separar periodos de calibracion sintetica y evaluacion. Declarar antes de ejecutar
trayectorias (tendencia, oscilacion y reverso), longitudes, semillas y presupuesto.
Cubrir controles que efectivamente activen y no activen cierre temprano. Registrar
por decision disponibilidad, posicion, umbral, familia consultada y motivo de
salida. Reconciliar esos motivos con orden y fill, sin forzar que el corto gane.

Conservar orientaciones con fijo persistencia e ideal, e intensidades pareadas.
Reportar por horizonte y familia MAE y naive de las mismas filas. Intensidad 1
significa escala DEV, NO igualdad con naive en evaluacion: cualquier cruce debe
salir del MAE logrado y puede quedar sin encerrar en la grilla. No inventar linea
naive ni interpolar fuera del rango. Mantener PnL realizado, equity neta, costes,
drawdown y exposicion; no presentar Sharpe de cuatro decisiones como evidencia.

Construir y probar dentro del limite CPU de pruebas vigente (2 GiB/120 s por
invocacion), sin trocear una campana mayor para eludirlo. Un piloto pequeno debe
medir coste; si la grilla ampliada no cabe, entregar el diseno y el ejecutor
probado, no ejecutar la campana. Nada de B0 ni datos de mercado reservados.
No implementar aun estrategias corto-only/largo-only redisenadas: retirar una
entrada de la estrategia actual y redisenar la estrategia son preguntas distintas.

B. DOIN: continuar el siguiente test de aceptacion pendiente del plan offchain
con servicios desechables, empezando por escritura durable y lectura por hash
antes de proyectar metricas; fallos de escritura no deben producir compromiso.
Reutilizar los adaptadores existentes, sin migrar cadena ni reiniciar servicios
productivos. Enumerar AT realmente ejecutados y lo que todavia depende del lago
desplegado. La revision aceptada de este parche no autoriza despliegue.

C. Mantener GPU, calibraciones financieras y codigo remoto de clasificacion
sin ejecutar. No son dependencias de A/B. Retsu coordina ambos frentes en paralelo
con admision CPU real, acuse de cada agente y worktrees separados. No se requiere
otra decision del propietario para el desarrollo y las pruebas aqui delimitadas.

Retorno: commits, resultados sinteticos nuevos cuando existan, comparacion con
los controles, cobertura real de decisiones, recursos y pruebas. Separar progreso
de integracion de calidad predictiva o utilidad financiera.
