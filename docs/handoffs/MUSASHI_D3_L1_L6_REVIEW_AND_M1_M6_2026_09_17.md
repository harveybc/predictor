# Revision L1-L6 y orden M1-M6

## Dictamen

Retorno: predictor `ea555f5`. No se abre utilidad sobre datos del proyecto ni
reserva confirmatoria. El arnes aun acepta una fuga futura demostrada con datos
fabricados; esto NO significa que se haya ejecutado esa fuga en una campana real.

Revision independiente:

```bash
python -m pytest -q tests/test_d3_matrix_verify.py tests/test_d3_twin_coverage.py tests/test_df_utility_harness.py tests/test_olap_ingest_diagnostics.py
K4_EXTRA_PYTHONPATH=<data-warehouse>/src python -m pytest -q tests/test_olap_ingest_diagnostics.py -rs
```

Trading-stack: primera invocacion **72 passed, 1 skipped, 23.89 s**; el skip
requiere la ruta del candidato warehouse. Segunda invocacion con esa ruta:
**14 passed, 7.63 s**, sin skips. Son pruebas solapadas, NO 86 pruebas distintas.
Relectura del run real v3 mediante verify: 511 completas, 710 variables totales,
9940 filas medidas, nueve operadores/doce pruebas en la composicion, cero rechazos.
No he repetido la conciliacion viva de los 1588 terminales ni la campana CPU.

## Hallazgos reproducidos

1. **Herencia sin equivalencia del operador.** En el fixture compuesto productivo,
   cambie los parametros de `op_b` no remedido a `window=999` y su spec_sha256,
   actualizando el freeze y su registro coherentemente. El verificador sigue
   dando VERIFIED y MECHANICALLY_ACCEPTED. `_compose()` compara nombres de
   operadores pero no demuestra que sus pruebas heredadas valgan para esa nueva
   especificacion. No he observado tal cambio en el run real.
2. **Fuga futura en utilidad.** `contrast()` acepta arrays precalculados con
   `accepted=True`, sin verificar procedencia ni ajustar el operador dentro de
   cada bloque. Darle como representacion la etiqueta x[t+1]-x[t] sobre el control
   de ruido produce ADVANCES, limite inferior **0.7772397137483449**. Declarar
   `emitted_at=t+10000` produce exactamente el mismo resultado que `emitted_at=t`:
   esos tiempos nunca se consumen. La purga no vuelve causal una entrada futura.
3. **Intervalo distinto al declarado.** `_t_quantile(.975, 1)` da
   **9.710583537182208** frente a **12.706204736174694** de scipy.stats.t.ppf
   en el entorno instalado. La aproximacion puede estrechar el intervalo justo
   cuando quedan pocos bloques. Bloques contiguos con train compartido tampoco
   quedan independientes simplemente por llamarlos unidad estadistica.
4. **Limite de calculo no ejercido.** Por inspeccion, `cpu_seconds` es un numero
   opcional del llamador, comprobado antes de correr; no mide ni detiene el trabajo
   de fit_predict. La prueba que pasa 999 demuestra la rama condicional, no el
   presupuesto del entrenamiento real. No he lanzado una carga larga para probarlo.

Reproductor adjunto: `reproduce_d3_l_review_2026_09_17.py`; solo fixtures y
directorios temporales. No lee reservas, no modifica los runs ni el cubo.

## M1. Cerrar la composicion, conservar mediciones

Primero congelar el caso de operador cambiado como prueba roja. La herencia debe
ligar identidad de datos/variables, parametros, semantica del operador, ajuste,
contrato temporal y dependencia de codigo de cada prueba. Una coincidencia de
kind no basta. Para diferencias exclusivamente diagnosticas autorizadas por 07C,
usar un diff finito permitido por la enmienda, no una igualdad imposible entre
digests globales de versiones diferentes ni una exencion para cualquier cambio.

Comprobar particion disjunta y completa de pruebas heredadas/medidas y cobertura
exacta de celdas esperadas; incluir pruebas de cambio de parametro, variable,
contrato y version productiva. Reverificar v2/v3 desde bytes conservados, sin
remeasuring si el alcance real sigue equivalente. Si no lo es, identificar solo
las mediciones invalidadas y justificar su repeticion versionada.

## M2. Temporalidad y ajuste real del arnes de utilidad

Reproducir primero la etiqueta futura y el tiempo de emision ignorado. El camino
de scoring debe consumir contratos por identidad de observacion, disponibilidad,
decision y horizonte. No recibir un booleano accepted como licencia suficiente.
Ligar elegibilidad al registro revisado exacto de variable/regimen/operador.

Construir representaciones con el operador real: fit exclusivamente en train de
cada fold, transform causal y estado propio. Si se aceptan caches, deben portar
la misma identidad de ajuste y de entrada y ser verificables por el consumidor.
Cada feature del tensor debe estar disponible antes o en la decision de su
target; delays se alinean, no se curan aumentando la purga. Conservar huecos y
orden temporal; rechazar ids/tiempos discordantes, no compararlos solo por valor.

Pruebas contra la ruta real: etiqueta futura, valores repetidos con ids distintos,
emision posterior a decision, scaler ajustado con futuro, cambio de cola que no
modifica el prefijo, restart, y lag historico legitimo que SI se acepta. No basta
con prohibir un nombre de columna. El control deliberadamente filtrado queda
fuera de elegibilidad aunque obtenga una perdida espectacular.

## M3. Inferencia estadistica y contrato experimental

Usar scipy.stats.t.ppf cuando el metodo predeclarado requiera t, con tests en df
bajos y colas corregidas por multiplicidad; no mantener una aproximacion casera
innecesaria. Antes de emitir ADVANCES, justificar el tratamiento de dependencia
de bloques y replicacion. Medir cobertura/error tipo I con generadores dependientes
y semillas reservadas al diagnostico; si no hay soporte, resultado descriptivo o
INCONCLUSIVE, no un intervalo presentado como confirmatorio.

Validar Protocol antes de scoring: dominios finitos, tipos, horizontes, ramas,
pares target/modelo, poblacion de contrastes y minimo de bloques. No permitir que
el llamador pida augmented cuando el protocolo solo declara raw/transformed.
No permitir que bloques faltantes reduzcan el denominador sin una regla previa.
Sellar la familia real de contrastes; un entero comparisons no demuestra que se
haya contado toda la seleccion.

## M4. Presupuesto observado y pruebas por capas

Usar los limites y contadores ya existentes del runner, no un cpu_seconds
autodeclarado. Medir CPU, muro y memoria realmente utilizados por preparacion,
transformacion y modelo; imponer el techo durante el trabajo. Ensayar un modelo
lento controlado en proceso aislado: presupuesto agotado, terminal y costo,
ninguna puntuacion parcial promovida. Ejecutar estas pruebas bajo data-gov y
reconciliar sus resultados; distinguirlas de un experimento cientifico.

Cerrar arnes completo desde el punto de entrada gobernado con modelo y operador
reales sobre datos fabricados, no solo funciones auxiliares. Mantener pruebas
unitarias, integracion, sistema y aceptacion alpha ligadas a requisitos. El
holdout queda vinculado a la identidad de la reserva/campana, no solo a un
directorio arbitrario nuevo que permita otra consulta.

## M5. Empaquetado y disposiciones sin otro bloqueo del owner

Confirmado con git ls-files: diez archivos build/lib de data-warehouse siguen
rastreados. Autorizado retirar del indice esas salidas generadas y excluirlas,
preservando su historia Git y sin editar el entorno vivo. Probar build limpio
desde commit publicado, wheel/fuente, instalacion aislada y entry points reales.
La exportacion limpia del adoptador se conserva como defensa de empaquetado;
el repo tambien debe instalar correctamente sin depender de ese adoptador.

No reiniciar produccion por la mera limpieza del indice. Si cambia runtime,
ensayo, respaldo, ventana y rollback por el procedimiento existente. Un contador
NRestarts reiniciado a cero no borra el incidente: conservar 18 reinicios en su
cronologia. Ninguna nueva decision del owner es necesaria para esta limpieza.

La disposicion del sobre accidental queda aceptada en principio: conservar,
no admisible cientificamente, sin borrar ni reemitir. Completar prueba de que la
vista mecanica no lo mezcla con evidencia MECHANICAL admisible: el retorno dice
que DEVELOPMENT operacional aparece en gov_mechanical_evidence. Separar clase
de resultado, disposition y admisibilidad en consultas; no cambiar su clase para
acomodarlo a una vista. Si esa vista es un historial operativo amplio, nombrarlo
y documentarlo asi y ofrecer la seleccion mecanica estricta.

## M6. Cierre y siguiente paso

Ejecutar M1-M5 completos sin pausas de autorizacion. CPU distribuida cuando aporte
valor y respete memoria/cargas; sin GPU ni trading, sin test publico reservado.
No repetir las campanas D3 enteras por cambios de arnes. No intentar conseguir
aceptacion modificando los resultados anteriores.

Entregar PRE/POST, nuevos casos adversariales, matriz reverificada, arnes end-to-end,
protocolo concreto y ensayo de recursos, todos con identidad y costos. Actualizar
work plan y estado persistente del metodo. Salida:
`D3_COMPOSITION_CLOSED_UTILITY_CAUSAL_HARNESS_READY_FOR_REVIEW`.
El siguiente paso es el piloto de utilidad de desarrollo tras revisar este arnes;
esta orden no abre confirmacion ni afirma que representaciones mejoren modelos.
Indice, Metabase y terminos conservan sus frentes separados.
