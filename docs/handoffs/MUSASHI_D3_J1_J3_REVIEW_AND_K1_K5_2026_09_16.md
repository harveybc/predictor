# D3: revision J1-J3 y orden K1-K5

## Alcance y decision

Retorno revisado: predictor `8364714b2fbe68b1ca55f0e64e8e8bd682ee70f7`.
Esta revision examina codigo y ejecuta pruebas CPU; no ha repetido la campana,
recontado el cubo vivo ni auditado independientemente sus 511 terminales.
Las cifras de campana del retorno siguen siendo evidencia presentada por Satoshi.

Musashi ejecuto, bajo trading-stack:

```bash
python -m pytest -q tests/test_d3_temporal_contract.py tests/test_d3_operators.py tests/test_d3_mechanics_pipeline.py tests/test_d3_matrix.py
```

Resultado: **93 passed en 10.38 s**. Esto no basta para aceptar la matriz:
dos contraejemplos ejecutados sobre fixtures desechables revelan un hueco real.
No se afirma que la evidencia real haya sido alterada.

1. `df_d3_matrix.aggregate()` acepta una unica fila de veredicto favorable,
   sin las doce pruebas, aunque COLLECT declara trece filas. Devuelve
   `MECHANICALLY_ACCEPTED: 1`, `tests: {}`, `rows: 1`, `mismatched: 0`.
2. Al eliminar ese archivo, devuelve cero unidades verificadas de una recogida,
   cero filas y `mismatched: 0`, sin rechazo. El texto conserva un conteo antiguo
   de discrepancias y no verifica nuevamente los bytes que acaba de resumir.

Reproductor adjunto: `reproduce_d3_matrix_review_2026_09_16.py`. No toca el run.
El reductor puede servir como resumen, pero no como verificador independiente.
La aceptacion externa de los seis operadores queda pendiente de K1, no anulada
por una supuesta fuga que no hemos observado.

Dos precisiones al retorno:

- El gemelo sin salidas NO obtiene aceptacion en el codigo revisado: el agregador
  de la bateria lo marca fallido con una explicacion incorrecta. Hay que propagar
  insuficiencia, no afirmar que se demostro que el gemelo era causal.
- Una representacion con ventana trailing no es necesariamente sin memoria.
  Que la perturbacion no cruce un umbral en el primer punto no identifica por si
  solo un retraso; tampoco justifica convertir cualquier retraso observado en
  UNIDENTIFIED. La regla necesita controles que distingan esos casos.

## Orden ejecutable completa

Continuar K1-K5 sin pedir permiso entre bloques. La autorizacion operativa previa
cubre implementacion, ensayos aislados, mecanica CPU gobernada y publicacion de
resultados. No concede utilidad cientifica, seleccion financiera, GPU ni live.
No reiniciar servicios ajenos, no borrar evidencia ni repetir J1-J3 enteros por
comodidad. Conservar v1 y todos los intentos. Un bloqueo local no detiene los
otros bloques independientes.

### K1. Verificar antes de resumir

Escribir primero pruebas que reproduzcan ambos casos anteriores. Separar resumen
exploratorio de matriz verificada. Para esta ultima, derivar la poblacion esperada
del manifiesto congelado: unidades, variables, operadores, pruebas y version.
Revalidar los digests de los archivos consumidos contra sus recibos y ligar estos
al diseno. No usar `output_verified` como prueba vigente suficiente.

Exigir identidad, unicidad y cobertura exactas por unidad/variable/operador/test;
recalcular el veredicto desde las pruebas obligatorias y la politica de estados.
Ausencia, duplicado, fila inesperada, conteo discrepante, resultado contradictorio
o archivo cambiado deben impedir el sello de matriz verificada. Distinguir una
unidad fallida registrada de una fila perdida; las primeras no desaparecen del
denominador. Resolver intentos desde el ledger, sin duplicarlos ni escoger por score.

Probar tambien: archivo modificado tras COLLECT, prueba fallida con veredicto
favorable, registro de otra variable y recibo duplicado. Reagregar v1 desde sus
bytes conservados, sin volver a medir y sin sobrescribir su matriz original.
Reportar cualquier diferencia y su causa. Si falta evidencia recuperable, marcar
el alcance incompleto; no reconstruir resultados inventados.

### K2. Enmienda de sonda, no ajuste de resultados

Congelar una enmienda sucesora antes de medir con ella. La construccion de la
sonda usara solo el ajuste de entrenamiento y la resolucion declarada; nunca
validacion, test ni amplitudes buscadas hasta obtener un pase. Definir casos de
entrenamiento constante, rangos extremos y saturacion.

Separar: excitacion identificable, primer cambio observado, y coincidencia con
la respuesta declarada. Una sonda sin excitacion identificable es UNIDENTIFIED;
un retraso real incompatible con el contrato debe seguir fallando. Pruebas con
cuantizador de dominio conocido, escalas distintas, un operador deliberadamente
retardado y STFT con inicio declarado uno. La regla no puede convertir el control
retardado en aprobado o borrar su contradiccion mediante abstencion universal.
No modificar los umbrales o parametros de los operadores para salvarlos.

### K3. Gemelos y disponibilidad escasa

Propagar INSUFFICIENT_TEST cuando el gemelo carezca de comparaciones observables;
registrar numero de emisiones y comparaciones para cada control. Una infraccion
demostrada del gemelo cuenta como deteccion; ausencia de evidencia no cuenta.
Conservar un control centrado con datos completos que la bateria deba detectar.

Probar ventanas con NaN aislado, bloques, MCAR, frontera de soporte y restart.
Separar causalidad, cobertura de emision e inaplicabilidad por missingness. No
introducir interpolacion futura ni cambiar el soporte 50 para rescatar wavelet.
No convertir falta de cobertura en utilidad ni en una prueba de ausencia de fuga.

### K4. Errores de ingesta observables

Reproducir sobre stack desechable el documento invalido que hoy termina en 503
y la perdida del cuerpo de respuesta reintentable. Validar el documento en la
frontera: entrada invalida conocida devuelve rechazo permanente tipado (400);
una indisponibilidad transitoria sigue siendo reintentable; un defecto interno
no se disfraza de error del cliente capturando todo AttributeError.

Persistir en el outbox estado HTTP, clase, razon y diagnostico acotado sin secretos.
Probar error permanente, 503 transitorio, recuperacion y segundo drenaje sin
duplicado, con cliente y loader reales. Mantener los 74 sobres adjudicados y su
historia. Si se requiere adopcion, usar el procedimiento existente con ensayo,
respaldo y comprobacion posterior; no hacer mantenimiento ajeno a este bloque.

### K5. Medir solo lo afectado y cerrar

Una vez verdes K1-K3 y sellada la enmienda, ejecutar piloto gobernado y despues
la mecanica sucesora de los operadores/pruebas afectados en TODO su alcance
predeclarado, no solo los casos que antes fallaron. Recibos nuevos; los resultados
no afectados pueden referenciar evidencia anterior solo si codigo y contrato
relevantes no cambiaron, con justificacion de alcance verificable.

Distribuir CPU por memoria disponible entre trabajadores con identidad propia.
El coordinador participa solo si sus cargas actuales lo permiten. No ocupar tres
maquinas por apariencia ni competir con GPU en curso. Mantener limites por unidad,
costo de intentos fallidos, heartbeat, reanudacion y gobernanza por resultado.

Entregar matriz sucesora validada, delta v1/sucesora con causas, terminales y
conciliacion por contenido contra la contabilidad independiente, suites con
entornos y exclusiones, y commits exactos. Actualizar work plan y trazabilidad.
No anunciar cero backlog basandose solo en ausencia de errores HTTP.

Preparar el siguiente experimento de utilidad de representaciones por variable
con rama cruda, presupuestos comparables y ajuste train-only, sin ejecutarlo:
pasar mecanica NO significa mejorar prediccion, informacion o trading.
La causa del incidente de indice, Metabase y los terminos siguen como frentes
separados; no inventar cierres ni convertirlos en bloqueadores de esta mecanica.

## Salida esperada

`D3_SUCCESSOR_MECHANICS_AND_MATRIX_READY_FOR_EXTERNAL_REVIEW`, o estado parcial
con objeto faltante y responsable exactos. No detenerse tras cada commit para
preguntar si continuar. Ninguna nueva decision del owner es necesaria para K1-K5.
