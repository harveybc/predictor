# Revision ML: comparar contra referencias y diagnosticar antes de repetir

Musashi, 2026-09-20. Base de codigo leida: `de88764`.
Corrijo mi estado conversacional anterior: el sucesor E1 SI se ejecuto.
Esta revision recomputa sus arrays locales; no sustituye una auditoria completa
de RP49-RP56 ni una conciliacion independiente del warehouse vivo.

## Comparacion que faltaba

10 020 origenes de validacion exactamente iguales, nueve fits, targets
reconstruidos desde DATA, persistencia/dia reconstruidos directamente, errores
almacenados cotejados. Fuentes hasheadas antes/despues, sin cambios. No fits nuevos.
El lineal es el ya ajustado por Satoshi; no lo reentrene.

| Metodo | MAE kW | Mejora MAE vs persistencia | MAE z-score | MAE log1p(kW) | Mejora log vs persistencia |
|---|---:|---:|---:|---:|---:|
| Persistencia h60 | 0.617372 | 0.00% | 0.676560 | 0.279127 | 0.00% |
| Naive diario | 0.731659 | -18.51% | 0.801804 | 0.330894 | -18.55% |
| Ridge | 0.545495 | +11.64% | 0.597791 | 0.269098 | +3.59% |
| R0 (media 3 semillas) | 0.546929 | +11.41% | 0.599364 | 0.266344 | +4.58% |
| R1 (media 3 semillas) | 0.556734 | +9.82% | 0.610109 | 0.271802 | +2.62% |
| R2 (media 3 semillas) | 0.552660 | +10.48% | 0.605644 | 0.268779 | +3.71% |

Mejora = 100*(1-MAE_modelo/MAE_referencia), AMBOS en validacion y sobre
las mismas filas. No es porcentaje de aciertos. log1p se aplica punto a punto
a predicciones y observaciones en kW; no es log1p(MAE), ni un modelo reentrenado
con objetivo logaritmico. El log cambia el peso relativo de los errores:
R0 supera ligeramente a ridge en esta escala pero no en kW. No elegir ahora
la escala que de un ganador; el criterio principal del protocolo permanece.

El denominador publicado 0.6162615768 es persistencia h60 en train, no un paso.
En la franja train consumida, las diferencias consecutivas dan 0.0851237797
(40 257 pares finitos; dos excluidos, sin cerrar el hueco). Por eso MASE m=1
no es el 0.8875 publicado. El periodo estacional de una futura comparacion
convencional debe declararse, no escoger 60 solo por ser el horizonte.
No reescribir los resultados historicos: publicar errata y campos distintos.
La propuesta doctoral mantiene MASE convencional y sus metricas complementarias.

## Antecedente antiguo localizado

**Aclaracion posterior del owner (20-sep):** la unica mejora valida que recuerda
fue en fase 3, despues de corregir una fuga causal en la descomposicion wavelet
de una entrada; antes hubo un error anormalmente bajo (~0.001). Sospecha que el
CSV citado aqui pertenece a una ejecucion afectada. NO he establecido esa
identidad. Los numeros siguientes son transcripcion/comparacion aritmetica de
un resumen, no evidencia aceptada de capacidad predictiva. Disposicion:
CAUSALITY_UNVERIFIED, no apto como benchmark hasta reconstruir ambos linajes.

`examples/results/phase_1_daily/phase_1_tcn_neat_1d_results.csv` contiene:

| Horizonte rotulado | MAE test | Naive MAE test | Mejora desde resumen redondeado |
|---|---:|---:|---:|
| H9 | 0.003196 | 0.003204 | +0.25% |
| H12 | 0.003724 | 0.003804 | +2.10% |
| H15 | 0.004387 | 0.004336 | -1.18% |
| H18 | 0.004768 | 0.004843 | +1.55% |
| H21 | 0.005118 | 0.005279 | +3.05% |
| H24 | 0.005445 | 0.005672 | +4.00% |

Esto confirma que hay comparaciones antiguas modelo/naive que merecen recuperarse.
NO identifica todavia la corrida exacta 0.02/0.018 recordada por el usuario ni
prueba que sus errores sean log1p del target. El config champion adyacente declara
log1p de FEATURES, normalizacion JSON y otro loss. Se necesita configuracion
efectiva y codigo productor, no inferir la escala desde el tamano del numero.
El commit historico que introdujo ese CSV es `63d042995cf3020d83aeef29cbcdb04810ba6b30`.
No comparar sus porcentajes con household como ranking entre negocios.

## Problemas e hipotesis concretas

1. **Metrica y stopping no alineados.** Se optimiza/selecciona MSE en z-score,
   pero la pregunta principal se juzga con error absoluto. No es automaticamente
   un bug; son objetivos estadisticos diferentes. Hay que comparar de forma
   controlada loss/monitor y mantener un criterio de seleccion predeclarado.
2. **Mas updates no implica mejor generalizacion.** En R0_s1 train MSE baja
   0.8788 -> 0.6273; validacion toca 0.7179 en epoca 3 y sube a 0.7693.
   R0_s3 tiene el mejor checkpoint en epoca 1. Las cuatro corridas al techo
   tampoco terminan con una mejora monotona de validacion. Es compatible con
   sobreajuste/inestabilidad/cambio de distribucion, no una causa ya aislada.
   Retirar que el error observado sea una cota inferior del error alcanzable:
   no existe esa garantia; el presupuesto solo limita lo que fue explorado.
3. **TCN adaptada, no replica fiel de Bai.** Nuestro bloque tiene una Conv1D
   con ELU y suma residual, sin weight normalization ni dropout. El codigo
   oficial usa dos convoluciones con weight normalization, ReLU y dropout
   por bloque. Esa diferencia debe declararse y contrastarse, no prometer
   rendimiento de un articulo por compartir el nombre TCN.
4. **Contexto y cobertura limitados.** Solo 60 minutos de siete canales, sin
   variables de calendario en la lista consumida; 28 dias train/7 validacion.
   40 mil ventanas solapadas no son 40 mil situaciones independientes. Esto
   no demuestra que falte una ventana diaria: exige medir informacion adicional
   y separar volumen, contexto y calendario sin cambiar todo a la vez.
5. **Preentrenamiento no es universal.** Reconstruir ventanas enmascaradas no
   garantiza aprender la informacion necesaria a h60. No confundirlo con los
   objetivos contrastivos y protocolos de otras publicaciones.

## Literatura consultada y uso preciso

- [Hyndman/Koehler, MASE](https://robjhyndman.com/publications/another-look-at-measures-of-forecast-accuracy/):
  justificacion del escalado entre series, no porcentaje de aciertos.
- [Transformaciones, FPP3](https://otexts.com/fpp3/ftransformations.html):
  invertir una transformacion no preserva automaticamente la media condicional;
  no aplicar correccion de sesgo indiscriminada cuando el objetivo es MAE.
- [TensorFlow, ejemplo oficial](https://www.tensorflow.org/tutorials/structured_data/time_series):
  comparadores lineal/CNN/LSTM, referencias, features temporales y prediccion
  residual. Su tarea es clima, NO el benchmark household de esta corrida.
- [Bai et al.](https://arxiv.org/abs/1803.01271) y
  [implementacion de TemporalBlock](https://github.com/locuslab/TCN/blob/master/TCN/tcn.py):
  especificacion para una referencia independiente, no resultados transferibles
  automaticamente a otros datos/horizontes.
- [TS2Vec](https://arxiv.org/abs/2106.10466): objetivo contrastivo jerarquico;
  no evidencia de que nuestro AE congelado deba ganar.
- [Fuente UCI](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption):
  casi cuatro anos de muestreo por minuto disponibles en el origen. Eso no
  autoriza abrir reservas ni convierte la submuestra del piloto en cuatro anos.

## Entregable ejecutado y limites

[Resultados por celda, curvas y hashes](../evidence/ML_BASELINE_REVIEW_2026_09_20/results.json).
`tools/forecast_comparison.py` crea reporte fuera del run y rehusa poblaciones,
labels y denominadores distintos. 14 tests pasan; fase inicial de tests rechazo
por modulo aun no implementado, despues positivos y perturbaciones de archivos.
No es validador de gobernanza: inspeccion de evidencia existente, no nueva campana.

Comando (CPU, numpy; sin cargar TensorFlow):

```bash
python tools/forecast_comparison.py --run <run-conservado> --legacy-csv examples/results/phase_1_daily/phase_1_tcn_neat_1d_results.csv --out <reporte-nuevo-fuera-del-run.json>
python -m pytest -q tests/test_forecast_comparison.py
```

No se demostro una inversion de labels ni que todos los modelos sean inutiles.
Si se demostro que la presentacion ocultaba comparadores necesarios y que hay
causas ML comprobables antes de gastar en otra replica. Ejecutar
[RP57-RP64](../../handoffs/MUSASHI_PROGRAM_RP57_RP64_2026_09_20.md).
