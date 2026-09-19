# Revision externa RP1-RP8: conservar el piloto, corregir el cierre y el alcance

Fecha: 2026-09-18. Revisor: Musashi. Codigo examinado: predictor `2347c7b`.
Disposicion: **REVISIONS_REQUIRED_PRESERVE_DESCRIPTIVE_PILOT**.
Orden sucesora: [RP9-RP16](../../handoffs/MUSASHI_MOD_E0_RP9_RP16_2026_09_18.md).

## Hallazgos por prioridad

### F1, alta: el cierre no verifica la poblacion y su resultado puede contradecir al padre

[`verify()` en la revision examinada](https://github.com/harveybc/predictor/blob/2347c7b/tools/df_mod_e0_verify.py#L195)
itera los directorios que encuentra; no deriva lo esperado de DESIGN, los trabajos
y los registros de campana. Una copia con REPORT honesto pero attempts vacio
produce `all_verified=true, parent_equal=true`. Otra con solo seis de las 69
unidades produce esos mismos booleanos y un efecto H3. Si se cambia el MASE del
padre a 999999, sale `parent_equal=false` pero `all_verified=true`; el CLI usa el
segundo para su codigo de salida. El calculo de efectos excluye silenciosamente
la celda discrepante y todavia informa tres unidades aunque el par util sea menor.

La comparacion con warehouse recorre la misma poblacion encontrada y solo algunos
agregados. No deriva la poblacion completa desde la contabilidad independiente.
`--no-warehouse` es un alcance local legitimo, no permiso para verificar un vacio.
No aceptar un cierre completo ni un intervalo con denominador variable por omision.

### F2, alta: integridad local de los archivos no demuestra identidad del problema ML

[`verify_cell()`](https://github.com/harveybc/predictor/blob/2347c7b/tools/df_mod_e0_verify.py#L119)
regenera X pero solo compara su digest con el declarado; no compara las etiquetas,
filas, baselines ni denominadores con esa regeneracion. Sobre copias coherentes
en sus recibos locales, sin modificar la contabilidad:

| alteracion | respuesta del verificador local |
|---|---|
| etiquetas sustituidas por predicciones | aceptada; MASE 0 |
| denominador multiplicado por 100 | aceptada; 0.5821267022 pasa a 0.0058212670 |
| indices de fila desplazados 100000 | aceptada |
| MAE y MASE del baseline lineal puestos en 999999 | aceptada |
| archivo de pesos reemplazado por texto | aceptada |

La ultima comprobacion tampoco abre los pesos ni prueba la congelacion: confia
en `extractor_weight_change` y `prediction_parity_after_reload` del record.
Esto demuestra brechas de verificacion, **no** que los originales esten alterados.
Hay que ligar datos, grafo, predicciones y todos los campos publicados a fuentes
independientes, no remedir ciegamente una campana porque su verificador era debil.

### F3, alta: descriptor de tendencia equivocado, con impacto de datos ya acotado

[`trend_seasonal_strength()`](https://github.com/harveybc/predictor/blob/2347c7b/tools/df_mod_e0.py#L181)
usa Var(X) en el denominador de la fuerza de tendencia, en vez de Var(T+R).
Sobre una sinusoide pura de periodo 24 y longitud 240 devuelve **0.9931997484**;
la formula `max(0, 1-Var(R)/Var(T+R))`, con la MISMA descomposicion, devuelve **0**.
Referencia primaria: [Hyndman y Athanasopoulos, FPP3, 4.3](https://otexts.com/fpp3/stlfeatures.html).
Su promedio movil tampoco debe publicitarse como ejecucion de STL.

Reanalice los perfiles de las 15 combinaciones (nivel, r, semilla) conservadas,
sin entrenar: **15/15 mantienen la misma particion** al corregir esta formula.
Eso permite una correccion de descriptores con prueba de equivalencia aguas abajo,
no una sustitucion silenciosa del codigo que produjo los resultados historicos.
La convolucion centrada usada solo para caracterizar el bloque de entrenamiento
no es por si misma fuga del test; su uso como feature emitida online seria otro
contrato y requeriria otra comprobacion causal.

### F4, alta: una prueba de finitud no llama al medidor que afirma comprobar

[`mase()`](https://github.com/harveybc/predictor/blob/2347c7b/tools/df_mod_e0.py#L312)
acepta NaN y devuelve NaN con estado MEDIDO. ML09 pasa porque la propia prueba
lanza `_raise()` antes de llamar a `mase`, no porque el medidor rechace la entrada.
Asimismo la prueba del oraculo que dice cambiar el futuro repite el mismo
generador sin perturbacion. La prueba de invariancia ML02 si cambia test, pero
no demuestra todo el alcance anunciado sobre validacion y fit.
Corregir las pruebas para que ejerciten las funciones y rutas reales, incluyendo
una mutacion deliberada que haga fallar cada propiedad, no una copia del algoritmo.

### F5, media: H3 es un contraste de dos procedimientos, no aislamiento completo de fusion

[`build_modular()` y `run_cell()`](https://github.com/harveybc/predictor/blob/2347c7b/tools/df_mod_e0.py#L359)
comparan secuencia -> Conv1D(16,3) -> ultima posicion con promedio global ->
Dense(32). Ademas, el extractor congelado fue optimizado previamente con el
primer tipo de receptor. Compartir activaciones y tener 920 vs 808 parametros
entrenables es util, pero no elimina diferencias de readout, optimizacion y
compatibilidad del preentrenamiento. No demuestra que H3 sea falso ni que su
diferencia numerica sea ficticia: limita la atribucion exclusiva a la fusion.

H2 no tiene equivalencia demostrada: diferencias pequenas con tres trayectorias
no prueban ausencia de efecto. La pendiente observada es positiva, no favorable
a H2. El bootstrap de tres trayectorias es una descripcion muy limitada; sus
2000 remuestreos no equivalen a 2000 replicas. El r=0 es ausencia del enlace
plantado observable, no debe venderse como ausencia de toda predictibilidad
cruzada, dadas las componentes periodicas comunes.

### F6, media: RP4 y RP6 no estan completamente materializados

`_metrics` en `tools/df_mod_e0_run.py` publica agregados de error, updates,
parametros y ARI, no la matriz D/Y/M/G requerida por el contrato de metricas.
No hay trayectoria de pesos/activaciones/gradientes en checkpoints que permita
reconstruir retroactivamente lo que no se guardo. Arrays, curva y checkpoint
final si permiten recuperar una parte: hay que distinguirla de NO_MEDIDO.

13B lista criterios para E1, no el manifiesto seleccionado desde el censo.
Su minimo de 20 ciclos no esta justificado y excluye por construccion procesos
sin periodo identificable. Su tabla mezcla SAC/PPO/DQN, 1h/4h, distintos costos
y presets, pero su resumen los reduce a DQN/PPO, 1h, twelve y 0.001. Enumerar
JSON no prueba que sus fuentes/plugins/contratos sean ejecutables actualmente.
Esto debe ser un inventario factual, no convertir defaults historicos en negocio.

Errata verificable de 13A: a h=3, P_B=24*(1+0.5*3)=60, no 72;
W=48 cubre W/P=0.8 ciclos o (W-1)/P=0.78333 entre observaciones, no 0.67.
Se requieren sensibilidad y curva de volumen/contexto, no prometer suficiencia
con un numero de ciclos fijo ni elegirlo mirando el resultado de reserva.

## Lo que si comprobe y se conserva

[Reproductor](../evidence/RP_REVIEW_2026_09_18/reproduce.py) y
[salida medida](../evidence/RP_REVIEW_2026_09_18/results.json).
Trabaje en un worktree separado, solo CPU; ningun entrenamiento ni consulta o
escritura productiva al warehouse, ningun servicio tocado. Las nueve sondas
locales anteriores y el control de formula estan documentados; originales:
**497 archivos, digests antes/despues identicos**.

Comprobacion independiente de los 69 records originales: filas, etiquetas,
baseline ingenuo, oraculo, denominadores desde generador/geometria; MAE y MASE
agregados de los cuatro predictores desde arrays; digest de pesos. **69 sin
discrepancias** en ese alcance. Esto no es todavia replay de inferencia ni
conciliacion viva de todos los hijos del cubo. No afirmo haber hecho esas dos cosas.

El verificador original sobre originales reproduce la tabla de efectos:
H2 e = [-0.008900, +0.005794, -0.000081, +0.005645], pendiente +0.003776;
H3 d0=-0.022696, d1=-0.090721, gamma=-0.068025. Conservar como descriptivo del
procedimiento ejecutado, no como confirmacion, equivalencia H2 o utilidad trading.

Pruebas focales existentes ejecutadas: **7 passed, 4 deselected**:
`pytest tests/test_df_mod_e0.py -q -k 'ML01 or ML02 or ML04 or ML09 or ML10 or RP2'`
en `trading-stack`, CPU, sin bytecode. No es la suite total de 1306 reportada.
Que esas siete pasen junto a los contraejemplos es precisamente la brecha de cobertura.

## Fallo y secuencia

Acepto existencia/ejecucion del piloto y la aritmetica local al alcance anterior;
**no acepto cierre integral ni atribucion mecanistica exclusiva**. No reiniciar
todo, no borrar negativos, no abrir reserva. Corregir el verificador y reusar
lo que pase; completar instrumentacion antes de nuevos entrenamientos.

Incorporar ahora la aprobacion publicada en master `1b82220`: ARCH-A/B/C/0 dentro
de E0, con A como referencia y ningun ganador; H-CORE despues de E1 y del prefijo
congelado verificable. La TCN de este piloto no satisface por si sola esa comparacion.
Los siguientes trabajos estan autorizados en RP9-RP16, sin nuevas microaprobaciones.
