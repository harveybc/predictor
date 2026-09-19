# Revision ML de RP17-RP24

Fecha: 2026-09-19. Revisor: Musashi. Base:
[`de8ae34c4d878441e89f60053b13e5c7a71a235d`](https://github.com/harveybc/predictor/tree/de8ae34c4d878441e89f60053b13e5c7a71a235d).
Disposicion: **mediciones conservables; cierre compuesto y ficha E1 requieren
correccion; RL semanal aun no demostrado como politica ejecutable**.

## F1. Alto: al unir campanas se pierde la validacion de sus cierres

[`merge_successor`, linea 375](https://github.com/harveybc/predictor/blob/de8ae34c4d878441e89f60053b13e5c7a71a235d/tools/df_mod_e0_arch_verify.py#L375)
promete ligar cada cierre con su diseno, pero no llama a esa validacion. Construye
una poblacion nueva desde los disenos y descarta las poblaciones originales.
Tampoco exige estado VERIFIED de los donantes heredados que omite de la union.

Cinco reproducciones sobre copias, pasando por merge_successor -> effects:
identidad ajena del cierre padre; poblacion padre vacia; poblacion sucesora vacia;
donantes heredados con PROBLEMS; ventana sucesora 999 con identidad sin actualizar.
**Las cinco producen gamma factorial ESTIMATED en las cuatro arquitecturas.**
No se afirma que esos defectos esten en las mediciones originales. El defecto
es que el cierre compuesto no puede descartarlos. Reparar el contrato de union
y verificar la equivalencia cientifica, no repetir el entrenamiento.

## F2. Alto: R0/R1/R2 de E1 no son los de la propuesta

[13E, tabla de diseno](https://github.com/harveybc/predictor/blob/de8ae34c4d878441e89f60053b13e5c7a71a235d/docs/tres_temas_entrevista/program_v3/13E_E1_TASK_SHEET_2026_09_19.md)
los redefine como crudo / agrupacion / agrupacion+fusion. La fuente doctoral,
[lineas 378-398](https://github.com/harveybc/predictor/blob/de8ae34c4d878441e89f60053b13e5c7a71a235d/docs/propuesta_doctoral_representaciones_temporales_modulares.tex#L378),
los define como detector desde cero, preentrenado congelado, preentrenado
ajustable. R1 congela el detector, no automaticamente todo el extractor.

Son factores distintos. Ejecutar 13E literalmente no contestaria la comparacion
de preentrenamiento y confundiria regimen con representacion. Corregir la ficha,
el manifiesto y tests por nombres de parametros/gradientes antes de E1; no cambiar
la propuesta para hacerla coincidir con una implementacion accidental.

## F3. Alto: el soporte E1 no corresponde a sus roles ni a su geometria temporal

[`family_contract`, linea 127](https://github.com/harveybc/predictor/blob/de8ae34c4d878441e89f60053b13e5c7a71a235d/tools/df_e1_tasks.py#L127)
usa todas las columnas numericas, aunque household declara seis entradas y un
target. Las tablas por-columna se calculan sobre las siete, no sobre el target
declarado. Experimento sobre el callable real, n=5000, W=60, h=1:

| Fixture | Ventanas train reportadas |
|---|---:|
| Completo | 3380 |
| Un target ausente, entradas declaradas completas | 3319 |
| Una metadata numerica ajena con un ausente | 3319 |
| Un salto temporal de un minuto, valores completos | 3380 |

El target ausente deberia retirar una etiqueta, no 61 origenes bajo los roles
declarados; la metadata no deberia cambiar nada. El hueco se informa pero el
contador sigue tratando filas consecutivas como ticks consecutivos. Las pruebas
existentes de prefijo llaman `df_mod_e0.make_windows`, no un loader E1 completo;
la de escalado no ajusta ni altera un scaler. Los cinco tests verdes no prueban
la propiedad end-to-end que sus nombres sugieren.

Tambien hay que declarar acceso a historia del target: persistence usa su pasado,
mientras 13E no lo incluye en las entradas household. La comparacion primaria
debe igualar informacion; una ablation sin historia propia puede existir aparte.
Las mascaras de targets no convierten ceros de clientes ausentes en entradas
medidas. La regla primer no-cero debe ser un proxy declarado, no prueba de fecha
de alta; una columna siempre cero requiere estado explicito.

## F4. Alto: las pruebas RL no ejercitan un contrato semanal long/flat

[`test_e3_weekly_env.py`](https://github.com/harveybc/predictor/blob/de8ae34c4d878441e89f60053b13e5c7a71a235d/tests/test_e3_weekly_env.py)
prueba piezas reales del broker, pero el release/fallback es una funcion escrita
dentro del test y el diccionario `late` nunca se ejecuta en el entorno. No hay
controlador de ciclo semanal sometido a esa prueba. El cambio de semana se cuenta
por llamadas a step, no por una asercion de timestamps del lunes tras el warmup.

Reproduccion con SU entorno/config/fixture: long, luego accion 2 (llamada flatten
en el test) produce posiciones **[-1, 0, 1]**, minimo **-0.005 unidades**. Es una
reversion a short, no flat. El test cash-spot evita 2; no demuestra que el camino
de ejecucion la rechace o transforme conforme a 13D. No es fallo general de un
entorno que tambien admite short: falta adaptar y probar el escenario elegido.

El fixture de precio pone open[t+1]=close[t], por lo que no distingue ambos
precios por si solo. Probar timestamps de informacion/decision/orden/fill, salto
de apertura y latencia positiva. Publicar limites de los tests actuales sin
afirmar que ya validan disponibilidad de datos, release tardio y operativa semanal.

## F5. Medio: un cero espectral no es ausencia de estructura lenta

Welch usa nperseg <=32768. En household de un minuto, el periodo no nulo maximo
resoluble es **22.7556 dias**. La banda >35 dias no tiene bins; su cero es por
construccion, no evidencia de ausencia de drift. Declarar NO_RESUELTO fuera del
soporte espectral. Fijar resolucion, segmentos, mascara y ponderacion antes de
usar espectros para elegir contextos. Las 32 primeras columnas no representan
sin mas todas las 370; potencia agregada y periodicidad por-variable son granos
distintos. No pedir dos periodos por axioma: justificar contexto y alcance del
modelo por tarea y comparadores, como manda el plan.

## F6. Medio: DST permanece discrepante y requiere tratamiento de tarea

UCI dice 96 medidas por dia, hora de ceros en marzo y agregacion de dos horas
en octubre; son horas de dias de 23/25 horas, **no 23/25 registros** como dice
la cadena `producer_notes` del codigo. [Fuente primaria](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014).
Relei rangos 00:00-06:00 de los cuatro cambios de marzo sobre el panel existente
con digest verificado: ninguna fila es cero para TODOS los clientes. Confirma
el desacuerdo reportado; no demuestra que el productor o el parser sea culpable.

Trazar esos rangos al archivo crudo y al parse receipt ya existente. Mientras
sea desconocida la semantica de esas filas, declararlas ambiguas o excluir su
soporte en una tarea retrospectiva acotada; no basta anadir un DST flag. Un
archivo estatico tampoco demuestra que nunca haya habido revisiones.
Household conserva timestamps con mediciones faltantes segun su productor:
el eje regular no implica observaciones completas.
[Fuente primaria](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

## Lo que si se sostiene

Recalculo independiente, aritmetica explicita sobre las cuatro celdas de cada
replica/regimen, sin reutilizar el estimador para generar el valor esperado:

| Arquitectura | Fusion r0 | Fusion r1 | Gamma factorial |
|---|---:|---:|---:|
| A | -0.007623 | -0.030045 | -0.022422 |
| B | +0.004078 | -0.018190 | -0.022268 |
| C | -0.005888 | -0.030309 | -0.024421 |
| 0 | -0.011330 | -0.034722 | -0.023393 |

Coincide con Satoshi. MASE recalculado desde predicciones/targets/denominadores
fisicos de las **16 celdas nuevas**: finito, diferencia maxima **4.44e-16**.
No se infiere igualdad de pesos ni conciliacion viva de esta comprobacion.
El efecto describe procedimientos completos con distintos cores, dos replicas;
ni ausencia de efecto en r0 ni superioridad universal quedan demostradas.

## Evidencia y alcance de esta auditoria

[Reproductor](../evidence/RP24_REVIEW_2026_09_19/reproduce.py) y
[resultados](../evidence/RP24_REVIEW_2026_09_19/results.json). Ejecutar desde la
publicacion del script, apuntando al checkout revisado, con `trading-stack`:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B \
  docs/audits/evidence/RP24_REVIEW_2026_09_19/reproduce.py \
  --repo <checkout-de8ae34> --rl --panels-root <public-panels-c126-v2> \
  --run-root <mod-e0-arch-stage-v1-rc> --out /tmp/rp24-review.json
```

Sin las opciones de roots no hace las lecturas fisicas; sin --rl no carga gym-fx.
Tests existentes ejecutados: efectos/E1/granos **17 passed**, RL **6 passed**.
Sus verdes coexisten con los contraejemplos nuevos; no reejecute la suite completa.
Mi primer script fallo al serializar `action_space.n` numpy.int64; conversion a
int corregida y ejecucion completa repetida, sin tocar ninguna evidencia fuente.
Sin entrenamiento, GPU, reservas, reinicios ni escritura al warehouse. La
conciliacion viva del retorno sigue siendo evidencia de Satoshi, no una lectura
actual hecha por mi. Archivos fuente de los cierres preservados por digest.

## Disposicion

Conservar etapa original y controles nuevos. Aceptar el recalculo descriptivo
al alcance indicado, corregir cierre antes de promover resultados. No sellar
13E v1 como experimento de preentrenamiento. Ninguna decision pendiente del owner
impide reparar estos puntos. [Orden RP25-RP32](../../handoffs/MUSASHI_PROGRAM_RP25_RP32_2026_09_19.md).
