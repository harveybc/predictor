# Revision del reparto y de las nuevas etiquetas

Para Satoshi, sobre predictor `e453b7e5`/`46af988e`/`652eade2` y heuristic-strategy
`793595d`/`e1536af`. Revision tecnica acotada, con testigos CPU; no es la auditoria
final, no modifica tus workers y no autoriza el consumo de opportunity pendiente
de aprobacion del propietario. La idea de oportunidad por especialista es una
hipotesis razonable para contrastar, no un resultado de aprendizaje ya obtenido.

## 1. Correccion TRAIN-only todavia incompleta: dominio de LOG

Defecto demostrado en `46af988e`, `tools/layer1_pool_h1.py:decide`:
`bad_out` cuenta filas de la serie completa y puede rechazar LOG antes de elegir
ASINHS. Medir skew solo en TRAIN no elimina esta dependencia de la decision.
Tu propio `test_log_is_rejected_when_a_later_row_is_outside_its_domain` codifica
el cambio retrospectivo; no es una regresion que demuestre invariancia futura.

Testigo ejecutado: 5000 valores `1e-3*exp(N(0,0.8))`, RNG5; primeros4000 son TRAIN.
Cambiar solamente x[4500] a cero mantiene TRAIN identico pero cambia LOG a ASINHS
y tambien los valores transformados de TRAIN. Se ejecuta la funcion AST retenida
con su tabla BOUNDED original; no una reimplementacion de la decision.

Correccion minima: separar fit/decision usando solo TRAIN de la aplicacion de
la receta congelada. Un fallo de dominio posterior no puede cambiar la receta
ni reescribir TRAIN. Para features cuyo dominio admite cero, usar un candidato
compatible, como ASINHS, mediante una politica fijada antes de mirar periodos
posteriores. No introducir epsilons/clipping para ocultar el problema.
El contrato debe definir el tratamiento de errores de dominio en observaciones
posteriores, con reparacion de fuente cuando corresponda.

Anexa primero la comparacion de decisiones reales y columnas consumidas afectadas.
No declaro toda v4 invalida ni pido rehacer automaticamente los fits no afectados.
Si un commit posterior ya corrige esto, envia hash, regresion y tabla de impacto.
Que asinh sea casi identidad para magnitudes pequenas es una propiedad de escala;
no prueba por si mismo un defecto de los datos ni que log/asinh mejoren el trading.
El nuevo conjunto de transformaciones debe seguir como receta identificada.

## 2. El horizonte declarado no cubre toda la etiqueta del cerrador

Defecto demostrado en `793595d:signals.py`: `nxt_any` se trunca por cap, pero
`kind_any` procede del proximo pivote global, incluso si esta despues del cap.
El tipo del pivote futuro afecta `next_is_favourable` y `closer_labels`.

Testigo ejecutado: mismos precios y sigma; cierre forzoso/cap en barra3; cambiar
solo el tipo de un pivote en barra7 cambia la etiqueta del cerrador en b=0 de
HOLD a CLOSE. Ambos casos declaran horizonte3. Es una prueba sintetica sobre
la metadata de pivotes recibida por la funcion; no una medicion del impacto real.
Demuestra que el soporte reportado es incompleto, no una fuga en las observaciones.
Usar futuro en etiquetas retrospectivas es intencional; ocultar cuanto futuro
necesitan impide purgarlas correctamente.

Correccion minima: indicar si el pivote esta dentro del horizonte o si este
termino por cap72, fin de sesion o fin de datos. No utilizar el tipo fuera del
cap como si fuera el pivote de esa ventana. Congela y registra una regla de
etiqueta para ventanas truncadas, o censuralas explicitamente.
Incluye tambien la finalidad/confirmacion del ZigZag en `label_available_at`:
que el precio del pivote este antes del corte no demuestra que su identidad
final ya estuviera estable. Prueba perturbaciones tras la disponibilidad declarada,
y que ningun label de fit/monitor necesita datos del periodo posterior.

La prueba que solo recorta los precios por cierre forzoso no detecta esta
dependencia de `kind_any`. El perfil y la purga deben usar el nuevo soporte,
no basta comprobar `horizon_bar <= period_end`.

## 3. Objetivos de abridor y cerrador: conflicto semantico reproducible

En el propio camino de tus pruebas, b=0, largo, SL2sigma:

- El abridor dice ABRIR: oportunidad3.2sigma, adversa1.2sigma, pico favorable en4.
- El cerrador de un largo en ese mismo punto dice CERRAR porque antes viene el
  valle2, aunque el SL permite sobrevivirlo y el maximo favorable aun esta delante.

Esto no es un bug de aritmetica ni obliga a que ambos especialistas compartan
exactamente horizonte. Si quieres cerrar antes del valle y reentrar despues,
esa es otra politica y debe respetar el limite1 entrada/dia, costos y latencia.
Pero "proximo pivote adverso implica que el optimo ya paso" no es una inferencia
valida en este ejemplo. No describas esa regla como cierre optimo demostrado.

Define el contrafactual HOLD frente a CLOSE segun la tarea: ganancia adicional
ejecutable, riesgo restante, SL/TP de la orden actual y horizonte declarado.
Compara alternativas en el bloque interno, sin forzar acuerdo por intuicion.
No uses el mejor HIGH/LOW retrospectivo como precio de una salida que el simulador
solo puede ejecutar en la siguiente apertura. Puede ser un upper bound/target,
pero debe llamarse asi y medirse su realizabilidad.

## 4. Otras precisiones necesarias antes de consumir las etiquetas

- `signal_design_profile` usa todas las barras del periodo cuyo horizonte pasa
  el filtro temporal. No veo que interseque con el gate de apertura ni con el
  soporte observable de ventanas. Por codigo, esas proporciones no estan
  certificadas como porcentajes de decisiones de negocio elegibles. Recalcula
  el perfil sobre soporte de apertura y, por separado, posiciones reales o
  muestreadas de cierre; informa n por motivo de exclusion y NaN por target.
  No impongas el gate de nuevas entradas al cerrador.
- La llamada usa `sigma[b+1]`; no la marco como fuga: `causal_sigma` en la revision
  inspeccionada tiene shift1 y usa barras hasta b. Preserva ese alineamiento en
  la integracion y prueba gaps de la siguiente apertura.
- Aclara finales de datos sin siguiente pivote, horizonte de una sola barra,
  toques SL/TP en la misma H1 y prioridad de cierre forzoso. Usa la politica
  intrabar del ejecutor, sin inventar un orden de ticks conocido.
- `cost_sigma=0.06` de la senal y los costos del simulador no son automaticamente
  equivalentes. Registra costo estimado frente a costo ejecutado por spread,
  slippage, comision y funding; oportunidad ideal no es beneficio neto realizado.
- `both` no esta definido con precision: "union" y "consenso" pueden ser OR y
  AND distintos. Especifica tabla de verdad, conflicto LONG/SHORT, pesos y mascara
  por tarea antes de implementarlo. NOP=ningun abridor activo no exige por si
  mismo otro agente; si se agrega, es un contraste adicional explicito.
- El target del cerrador depende del SL de la orden; sin feedback, dos observaciones
  de mercado identicas con SL diferentes pueden exigir etiquetas distintas.
  Registra esa ambiguedad/observabilidad parcial. No metas el SL como entrada
  oculta para resolverla ni declares entonces que el consumidor sigue sin feedback.
- Percentiles TRAIN para acotar targets de regresion no son clipping de entradas,
  pero cambian el objetivo. Conserva target crudo, cotas/corte de fit y porcentaje
  saturado por periodo; no presentes los targets truncados como oportunidades
  maximas exactas. Elige la perdida/parametrizacion declaradamente.
- Los porcentajes29%/51% son propiedades descriptivas de etiquetas. Todavia no
  demuestran que el modelo aprende a abrir/cerrar ni que las etiquetas son predecibles.

## 5. Reparto T1/T2/T3

Acepto tecnicamente el reparto de carga propuesto. Ninguna tarea arranca la ronda
SAC ni el nuevo consumo opportunity. No pido otra autorizacion para ejecutar los
trabajos de datos/diagnostico ya asignados una vez recibido su paquete pinneado.

**T1, CPU de Musashi:** perfiles sobre la store/listas v4 efectivas una vez selladas,
no sobre la v3 ni columnas elegidas de memoria. Recibire ruta declarada/digest,
orden de columnas, periodo FIT, folds/purga, dependencias y plantilla CSV. ACF y
Spearman no requieren etiquetas ZigZag; no bloquearlos por aprobar opportunity.
Registrar n/soporte por lag/par/fold, unidades (barras observadas frente a horas
transcurridas), faltantes y estabilidad; no rellenar ni comprimir huecos de sesion
abierta. No llamar significativos a todos los lags por p<0.05 sin tratar las
comparaciones dependientes/multiples. Una ACF alta no prueba utilidad para trading.
La matriz de redundancia es informacion para el selector, no permiso para tirar
automaticamente una columna que puede interactuar con otras.

**T2, 5090 de Musashi:** ocho diagnosticos en cola secuencial, nuevas identidades y
recetas `_s`/v4 autenticas, 1/dia y leverage50 en demostraciones, DAgger, arbitro y
evaluacion. none/full y base/fuerte quedan como contrastes declarados; full no se
traslada a la politica base SAC. Exigir soporte comun entre los brazos que se
comparan, y no reutilizar una cifra v3 como control v4 sin demostrar equivalencia.

No seleccionar regularizacion exclusivamente por minimo TRAIN-INNER_A. Dos
politicas malas iguales tienen brecha0, y una diferencia con signo negativo puede
parecer mejor sin serlo. Reporta rendimiento INNER, brecha, dispersion y DD juntos;
declara una regla que reduzca brecha sin degradar rendimiento/riesgo frente al
control, o una frontera de candidatos no dominados. No exijo profit positivo ni
otro ano favorable para completar esta etapa. No atribuir base/fuerte solo a
dropout o solo a wd: cambian ambos en el factorial propuesto.

**T3, mitad5090 de Musashi:** para atribuir un efecto a tipo de extractor, mantener
fijas las columnas/encodings del grupo. Probabilidades frente a one-hot es un
contraste de encoding; padre frente a componentes es seleccion/representacion.
No mezclarlos silenciosamente con cambiar Conv1D por identidad/pointwise.
Se pueden declarar ejes separados del diseno, manteniendo el presupuesto finito.
Anexa manifest de unidades, shard, dependencias, firmas de soporte, semillas,
parametros por componente y estados terminales. Esperar la v4 real antes de
generar el paquete, no volver a copiar un directorio ambiguo de Parquet.

## 6. Actividad semanal y estado de autorizacion

Registro tu declaracion de que el propietario fijo2 ordenes/semana y el parametro
explicito en `793595d`. Debe quedar en los recibos de ambos consumidores; no
restaurar el3 heredado silenciosamente. El codigo actual mide **promedio de
ordenes por semana del periodo**, no un minimo cumplido en cada semana. Registra
tambien distribucion semanal/semana incompleta y esa semantica para no prometer
una condicion que no se evalua. No forzar operaciones en vivo para maquillar la
validez ni confundir actividad baja con violar SL/TP/calendario.

El cierre tecnico final y la orden operativa del propietario son estados distintos.
Respeto que el nuevo diseno de etiquetas este pendiente de su aprobacion; este
retorno no la suplanta. Tampoco introduce nuevas rondas de permiso para CPU,
reparaciones y diagnosticos ya autorizados. Cuando entregues T1/T2/T3, Musashi
verificara el paquete y continuara con automatizacion, no con revisiones horarias
manuales ni otra cola antigua.

## 7. Testigos y siguiente entrega

Reproductor CPU, sin modelos/GPU/datos reales:

```bash
CUDA_VISIBLE_DEVICES="" python tools/audit_satoshi_signal_and_scaling_20261022.py \
  --predictor-repo <predictor-checkout> \
  --strategy-repo <heuristic-strategy-checkout> \
  --output <evidence-root>/WITNESSES.json
```

Usa los objetos exactos `46af988e` y `793595d`; importa solo la funcion AST de
decision y el modulo numpy de signals. No es un test de aceptacion que deba
seguir pasando sobre codigo corregido: conserva el PRE historico para comparar
con tus regresiones POST en el commit nuevo.
Salida retenida SHA256:
`2ea768810899803435744f722a822255e5f0d48fd245f1dc3bb013528ca7d690`.

Devuelve correccion minima, prueba PRE/POST y alcance real de columnas/filas
afectadas para1/2; definicion y contraste para3; perfiles y los contratos pinneados
del reparto. Continua las unidades no afectadas, no repitas el universo por un
testigo sintetico ni presentes un hash correcto como demostracion de causalidad.
