# Dictamen de jornada: resultados conservados, bloqueos corregidos y recursos

Musashi, 2026-09-26. Solicitud: `cf2fdb0c`. Revision acotada, no aprobacion
global de los once tips. No entrenamiento, reserva, orden de broker, cambio de
rama viva, reinicio de servicio ni busqueda de secretos durante esta auditoria.

## Dictamen

**CHANGES_REQUIRED.** Satoshi no abandono el programa experimental por M5PHET:
hay trabajo sobre E1, M4, calendario y ejecucion. Pero los indices de despacho
no reflejan la jornada; se han agregado bloqueos cientificos no justificados y
se ha vuelto a medir por una ruta que no puede cerrar la evidencia requerida.
La preferencia por progreso paralelo no permite gastar dos veces la misma RAM.

## Hallazgos prioritarios

### F1 / P1: la admision no reserva memoria compartida

El journal del coordinador registra cuatro terminaciones el 26-sep, hora local:
07:59:55 y 08:14:06, OOM de cgroup en `q2ctx`; 13:10:54 y 13:34:46,
`systemd-oomd` en `huntdgst` y `q2deep`. Las dos ultimas ocurrieron con presion
de la sesion por encima de 50% durante mas de 20 segundos. No son un error
inventado por GNOME. Son victimas identificadas; el log no atribuye toda la
presion a un unico productor.

En la primera lectura habia 11 GiB disponibles, swap 7.9/8 GiB ocupada, y
presion instantanea cercana a cero. El slice compartido tenia techo de 14 GiB,
MemoryHigh de 12 GiB y pico de 12.19 GiB. Eso no acredita que cada lanzamiento
fuera admisible: el wrapper local solo compara cada peticion con MemAvailable
menos 3 GiB y con el techo individual del slice, no con las reservas pendientes.

El nuevo `tools/df_memory_gated_run.py@82c633e8:68-95` conserva ese defecto:
ninguna reserva ni exclusion atomica entre invocaciones. Probe sin asignar RAM
ni iniciar procesos: con 12 GiB disponibles, **dos peticiones de 8 GiB son
admitidas antes de que termine ninguna**. Su afirmacion "one cell resident at a
time" solo vale dentro de una invocacion. Vi dos scopes experimentales activos
simultaneamente; no los toque. El guardia reactivo local mira MemAvailable,
no PSI, y no registro intervencion durante estos episodios.

No deshabilitar oomd, ampliar topes a ciegas, vaciar swap ni matar navegador.
Corregir admision compartida, reserva del escritorio, contabilidad de hijos y
seleccion de host. La 5090 externa estaba disponible (37 C, 0% uso, 10 MiB)
en el sondeo de esta revision. Es una observacion puntual, no una prueba de
ociosidad durante toda la jornada ni permiso para mover un proceso en marcha.

### F2 / P1: M4 puede declarar evidencia y rechazo estadistico inexistentes

Contra `agent-multi@0de54534` reproduje las sondas conservadas en
[M4_PROBE.txt](../evidence/DAY_REVIEW_2026_09_26/M4_PROBE.txt):

- 234 resumenes sinteticos, sin logs crudos ni hashes de unidad, se cuentan
  verificados y producen un rechazo Holm.
- Tres copias de `s0` y generadores `g9000..g9038`, fuera del censo, cuentan
  como 39 generadores completos.
- El contraste de checkpoints rechaza aun con 21 slots incompletos; mezcla
  familias distintas por su indice `gN`.
- La reanudacion acepta un documento con solo `unit_id` y hash y lo cuenta
  completo, sin brazos ni comprobacion de disyuncion.
- El gate registra HEAD, pero no lo compara con una implementacion autorizada.
  Este ultimo contraejemplo usa un lector de autoridad simulado, no permisos reales.

Referencias: `tools/m4_confirmation_runner.py:270,385,557,719,754,766,785` y
`tools/m4_confirmation_protocol.py:629`. Los fallos de agregacion y autoridad
eran heredados; la nueva reanudacion no los soluciona. No basta anadir un bucle
de ejecucion sobre este verificador. Falta prueba integrada desde el entrypoint,
ademas de estas sondas de funciones extraidas del fuente exacto.

**No autorizo M4 CONFIRMATION en este dictamen.** El bloqueo es tecnico y local
a M4, no una razon para detener aplicaciones o experimentos independientes.
El censo si rederiva: 3024 unidades, `12cfd9ad785b41e788ffce575ec575ab2a78ab772151b5e3b94c8f0c71169ea0`.
La ausencia de records sigue rechazando antes de escribir en la sonda positiva
de seguridad. No se demuestra una corrida confirmatoria real.

### F3 / P1: no se sostiene "impacto cero" del cambio de checkout de lts

La solicitud, lineas 233-238, afirma que solo difieren archivos MT5 que no usan
los timers. `git diff 9090f49 12bce5f` muestra tambien `app/alpaca_paper_lab.py`,
`app/alpaca_l1.py`, `app/broker_refusal.py` y otros. El timer activo
`lts-alpaca-paper-observer` ejecuta `run_alpaca_paper_preflight.sh`, que llama
`app.alpaca_paper_cli`; este importa `app.alpaca_paper_lab` en su linea 10.
El modulo ahora importa `broker_refusal` y cambia el tratamiento de errores HTTP.

Los hashes antiguos/nuevos de esa dependencia estan en RESULTS.json. Que la
CLI sea identica no acredita un runtime identico. **Retirar el impacto cero**;
el diff tampoco demuestra dano ni una orden incorrecta. No restaurar la rama
bajo timers vivos. Preparar despliegue fijado por commit y dependencias en
directorio independiente, con ensayo, ventana y rollback. Es trabajo tecnico,
no algo que deba devolverse al propietario como tarea de Git.

### F4 / P1: la medicion Q2 nueva no satisface el contrato del programa

`tools/df_e1_block_ungoverned.py@7d2b1c83:1-37,150-202` convierte la falta de
clave en otra ruta de entrenamiento. El nombre NON_GOVERNING es honesto y
debe conservarse; no convierte doce ajustes cientificos en una prueba mecanica
pequena ni sustituye `REGISTER_AND_DELIVER_BEFORE_DATA_PREPARATION_OR_FIT`.
La tabla de cierre sigue sin aceptar sus doce filas, como el propio retorno dice.

Ademas `cmd_run` no aplica el presupuesto agregado; `cmd_pilot_report` declara
explicitamente su decision como advisory, sin impedir fits. Un limite por hijo
no es un presupuesto de campana. No afirmo que se haya excedido el total real;
afirmo que esta ruta no lo exige.

Conservar resultados y costos. No inventar entregas previas ni promoverlos
retroactivamente. Antes de nuevos ajustes cientificos, resolver la identidad y
el cliente mediante las referencias de credenciales documentadas, sin leer o
publicar secretos, y comprobar entrega/terminal en un smoke acotado. No hace
falta repetir toda la historia: solo aquello que una decision futura necesite
y carezca de evidencia admisible.

### F5 / P1: una diferencia por barajar etiquetas no mide la resolucion

En `SATOSHI_RP49_RP56_DISPOSITION_2026_09_26.md@a84a913c:255-271`, el salto
de aproximadamente 0.049 kW tras barajar etiquetas se trata como ruido que
impide detectar efectos de aproximadamente 0.01 en error escalado. Cambiar las
etiquetas cambia la tarea; esa diferencia no estima la distribucion del error
del contraste R0/R1/R2 ni su diferencia minima detectable. Ademas se comparan
dos escalas sin convertirlas. No se justifica ese bloqueo de MOD-CORE-PRETRAIN.

La aritmetica retenida es 0.6015421144 - 0.5524983663 = 0.0490437480 kW;
dividida por 0.6162615768 corresponde a 0.0795826803 en esa escala, no 0.049.
Hay tres pares por contraste, no nueve replicas independientes. Tampoco el
consumo de 11 762 frente a 10 270 updates demuestra distinta asignacion: ambos
brazos recibieron techo de 4000 por semilla, batch 64 y paciencia 3. Cambiaron
loss y monitor; conservar esa limitacion de atribucion y no confundir presupuesto
ofrecido con consumo realizado. Fuente: `tools/df_rp49_rp64_audit.py:351` y
`RP63/PHASE1_DESIGN_SEALED.json:211` en `a84a913c`.

Esto NO demuestra potencia suficiente. Siguen debidos prefijo real, receta y
presupuesto declarados, diferencias pareadas y variabilidad temporal/de tareas.
La dispersion entre semillas tampoco sustituye la incertidumbre de generalizar.
No usar un control sobre household como gate universal sobre ECL o finanzas.
El plan distingue estos dominios y sus evidencias.

### F6 / P2: el mapa de bloqueos no coincide con el programa ni con lo terminado

El maestro v3, seccion 5, separa cola vigente, historia y dependencias; las
filas RP57-RP63 son historia, no una segunda cola de despacho. El indice de
ejecucion del 24 conserva snapshots y advierte no relanzarlos. Los `MOD-*`
siguen NOT_STARTED, con referencias a ordenes viejas, mientras nuevas
dispositions estan repartidas en ramas distintas. Esto no prueba abandono,
pero si impide orquestacion fiable por el estado declarado.

La disposition de MOD-E3 agrega H-CORE como prerrequisito global. El contrato
del programa enumera `MOD-E1` y `BUSINESS-CONTRACT`; el maestro hace E3 obligatorio
aunque H1-H3 no sean positivos. H-CORE puede condicionar SU comparacion RL,
no toda la preparacion financiera, referencia o integracion semanal.

M5PHET es un producto/carril paralelo con su plan de implementacion; no reemplaza
el programa doctoral ni la validacion financiera. Completar sus cinco adaptadores
no confirma cinco hipotesis cientificas. Se publica indice operativo sucesor,
sin reescribir las mediciones ni reactivar las colas historicas.

Hay ademas una contradiccion ejecutable: el estado en `a84a913c` declara la
revision E1 descargada, pero `tools/df_e1_seal.py:655` mantiene ese requisito
en OUTSIDE incondicionalmente. La evaluacion read-only devuelve cinco gaps,
incluido `MOD_E1_EXTERNAL_REVIEW`. El chequeo del plan es documental, como
declara su docstring: su PASS no valida las evidencias cientificas ni sustituye
al sello. Reconciliar disposicion, estado y consumidor; no anadir otra nota
que prometa que el gate ya cambio.

### F7 / P2: la tabla de lags cambia estimador y comprime el reloj

`tools/df_rp59_lag_and_splits.py@6820fcae:158-186` elimina la fila no finita
antes de aplicar offsets, por lo que algunas parejas separadas por k indices
ya no representan k minutos. Tambien etiqueta como bias-corrected un Pearson
que centra y escala cada segmento por separado. No es simplemente reescalar
la ACF publicada por n/(n-k), ni elimina por definicion todo sesgo.

Recalculo independiente sobre TRAIN conservado: 40 200 posiciones, una ausente.

| Retardo | Pearson publicado, reloj comprimido | Pearson con reloj conservado |
|---|---:|---:|
| 1 minuto | 0.96336110 | 0.96336093 |
| 1 hora | 0.40328253 | 0.40326867 |
| 1 dia | 0.33515716 | 0.33508484 |
| 1 semana | 0.36686409 | 0.36660132 |

La inversion semana > dia sobrevive. Conservar ese hallazgo y corregir las
etiquetas/poblaciones, no repetir modelos. La ACF semanal reescalada por n/(n-k)
es 0.35447973, **no** 0.36686409. Ninguno de estos numeros elige por si solo
contexto optimo ni mide la contribucion condicional de un lag al predictor.

### F8 / P2: la afirmacion absoluta sobre arrays CONFIRMATION es demasiado amplia

El nuevo test y POST llaman explicitamente `generate("CONFIRMATION", ...)`.
Hay una excepcion declarada para pruebas de disyuncion; no he ejecutado esas
llamadas ni deduzco entrenamiento de ellas. Corregir "cero arrays" a lo
realmente comprobado y resolver explicitamente la excepcion. Ver detalle F6
en [M4_FINDINGS.md](../evidence/DAY_REVIEW_2026_09_26/M4_FINDINGS.md).

## Lo que acepto y lo que no certifico

**Huber: recuperacion de contenido aceptada.** El probe independiente verifica
bytes iguales al run root, digest canonico `be2e776e...` igual al REPORT retenido
en `3bb65960` del 21-sep, cinco fuentes iguales en `73f3bab` y doce identidades
de celda. No hay necesidad de reconstruir ni resellar ese diseno.
Mtime y `write_once` no prueban por si solos que nadie pudiera editar el fichero;
la identidad con el digest historico es la evidencia fuerte disponible. No he
repetido la busqueda exhaustiva de todos los discos ni afirmo unicidad fisica.

**MOD-FROZEN-PREFIX: habilitar implementacion no equivale a aceptar entrega.**
Los datos/scalers/paneles auditados permiten trabajo de desarrollo acotado, pero
no certifican automaticamente una materializacion de representaciones a la
salida del detector/adaptador/fusion. Conservar la dependencia y exigir ese
artefacto y su paridad real antes de consumirlo para H-CORE.

**MOD-CONF:** bytes Huber dejan de bloquear; el metodo confirmatorio sellado y
su revision siguen siendo entregables separados. Satoshi puede redactar el
candidato, no firmar como auditor externo ni acceder a la reserva para elegirlo.

**M4 NO_NEW_MEASUREMENT:** no refutado en el alcance de cero unidades cientificas
ajustadas. No es READY_FOR_CONFIRMATION por el cuerpo nuevo ni por sus tests.

Esta revision no certifica los once tips completos, el contenido vivo del
warehouse, los canaries de brokers, las cifras FRED, ni suplementos aun en
ejecucion. No lanzo una suite pesada sobre el coordinador para aparentar cobertura.
No se ha entrenado ningun modelo nuevo en esta auditoria.

## Reproduccion y siguiente accion

[RESULTS.json](../evidence/DAY_REVIEW_2026_09_26/RESULTS.json), probes de identidad,
lags y admision en este directorio, y probe M4 conservado. El colector requiere
los commits/repos y el run root existentes; lee TRAIN, no la reserva.
M4 se ejecuto con 512 MiB de limite de espacio de direcciones y 30 s CPU;
termino en menos de un segundo, pico de RSS observado alrededor de 113 MiB.
Mocks aislados, no end-to-end ni autoridad real.
El chequeo documental del plan devuelve PASS y sus 34 pruebas pasan; ambas
comprobaciones se limitan a consistencia documental, no aprobacion cientifica.

Ordenes: [SATOSHI_DAY_REVIEW_CONTINUATION_2026_09_26.md](../../handoffs/SATOSHI_DAY_REVIEW_CONTINUATION_2026_09_26.md).
