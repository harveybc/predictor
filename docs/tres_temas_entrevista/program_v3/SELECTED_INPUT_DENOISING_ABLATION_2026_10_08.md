# I5-P: reconstrucción causal de entradas seleccionadas

Estado: `PILOT_RUNNING`. El manifiesto final de I5 está congelado y el primer
piloto de arquitectura terminó; falta la comparación predictiva semanal. Precede al
contraste de arquitectura I6-A. No cambia la campaña semanal RAW/encoder que
ya corre, no reabre selección y no lee TEST.

## Pregunta y controles

¿La reconstrucción causal de cada serie **ya seleccionada** mejora la
predicción frente a entregarla observada? Comparar en idénticas semanas,
orígenes, rasgos, presupuesto, semilla y cabezal temporal:

1. `RAW`: valor observado, control y respaldo por defecto.
2. `RECONSTRUCTED`: último valor reconstruido por un decoder autenticado desde
   la ventana causal de 24 horas que termina en t; el resto de la secuencia se
   obtiene con la misma regla para cada origen anterior.
3. `LATENT`: salida temporal del encoder congelado, como contraste de
   representación y no como prueba de eliminación de ruido.
4. `RAW_LAG_MATCHED`: sólo si el soporte de `RECONSTRUCTED` o `LATENT` termina
   antes de t, control RAW con exactamente el mismo corte informativo.

Los encoders existentes se reutilizan por digesto si son compatibles. Un
autoencoder entrenado para recuperar observaciones tras corrupción artificial
no demuestra que quite el ruido económico de esas observaciones; no se usará
su MAE de reconstrucción como criterio de victoria.

## Causalidad y objetivo

Para emitir la entrada en t, ninguna lectura, normalización, máscara o
checkpoint puede depender de datos disponibles después de t. No se promedian
estimaciones de t provenientes de ventanas que terminan después de t. Probar
el soporte efectivo del **decoder**, no sólo la forma de su salida: el encoder
actual 24→12→6 tiene un último estado que lee hasta t−3 h. Si el decoder no
puede reconstruir t con soporte demostrado hasta t, se etiqueta el rezago y
se compara con `RAW_LAG_MATCHED`; para afirmar retraso cero hará falta otro
brazo versionado y reentrenado con soporte t, no desplazar pesos de fase.

El target predictivo, sus filas, escala, horizonte, naive y definición de
negocio permanecen **byte-identicos** entre los brazos. No se suaviza el
target. Un objetivo suavizado sería otro experimento, no esta ablación.

## Ejecución y decisión

Usar los rasgos del manifiesto final, sin expandir el inventario. Piloto de
coste sobre un conjunto pequeño y uno grande; medir pico cgroup/VRAM y tiempo
por celda antes de fijar caps. Mantener ventana 24 y el Conv1D compacto
existente como primera opción; LSTM/Transformer por rasgo sólo se justificarán
con beneficio pareado frente al coste medido, sin búsqueda abierta ni más de
tres semillas. Asegurar que el número de parámetros/activaciones del ganador
permita R2 completo de ramas y núcleo; no aceptar R1 obligatorio por memoria
sin registrar esa renuncia.

Puntuar todas las semanas elegibles de VALIDATION con el procedimiento semanal
de cuatro años. Reportar MAE/MSE y naive de las mismas filas, skill, cobertura,
coste, error por régimen y diferencia pareada frente a RAW y `RAW_LAG_MATCHED`.
Si no hay ganancia verificable o si falla la prueba de soporte, RAW avanza a
I6-A. Publicar métricas y disposiciones en el warehouse con lectura de vuelta,
identidad de pesos/configuración/datos, snapshot físico y prueba de reinicio.
TEST permanece cerrado hasta el finalista sellado del programa.

La aceptación exige: perturbación de futuro sin efecto en entradas anteriores;
perturbación de x_t para comprobar si realmente alcanza la salida en t;
invariancia de bytes del target/naive; misma población semanal entre brazos;
rechazo de decoder/checkpoint ajeno; y un resultado RAW válido cuando el
denoiser no sea admisible o no aporte valor.

## Avance verificado, 2026-10-08

El encoder v1 tiene tres horas de rezago efectivo en su ultimo estado. El
control `RAW_LAG3_DIAGNOSTIC` ya esta implementado fuera de la cola semanal:
exige una referencia RAW retenida, conserva sus filas, target y naive, y usa
un corte de tres horas transcurridas. Sus 20 pruebas con TensorFlow real pasan;
no hay todavia una medicion predictiva real de este control.

El brazo versionado `fs4_causal_conv_24_12_6_oc_v2` aplica padding causal
izquierdo antes de cada reduccion temporal. Conserva 24 horas de entrada y
latente 6x8. Una perturbacion de la hora de origen alcanza tanto el ultimo
latente como la reconstruccion de esa hora; la v1 no la alcanza. La primera
celda TRAIN real, `px.close_loc`/`inner_2023`/semilla 0, termino en la 4070
de omega con MAE de reconstruccion 0.905266, contra 0.904511 para v1,
0.903787 para RAW y 1.182497 para persistencia, sobre 25 806 puntos ocultos.
Entrada, filas y mascara tienen los mismos digestos que v1. Tiempo de pared
103.8 s, pico cgroup 1.88 GB, VRAM 0.42 GB. Es un piloto de soporte y coste,
no evidencia de beneficio predictivo ni de limpieza de ruido economico.

El siguiente cierre requiere checkpoints v2 para todos los rasgos del conjunto
seleccionado que se contraste, entradas reconstruidas emitidas causalmente
por origen, y la comparacion semanal pareada contra RAW (y RAW_LAG3 para v1)
con target y naive intactos. Si el beneficio no aparece, avanza RAW.
