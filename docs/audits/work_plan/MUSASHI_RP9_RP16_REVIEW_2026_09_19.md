# Revision ML de RP9-RP16

Fecha: 2026-09-19. Revisor: Musashi. Revision examinada:
[`6f6c1a00bf2fe8e285f821dfb72d12505a9049ed`](https://github.com/harveybc/predictor/tree/6f6c1a00bf2fe8e285f821dfb72d12505a9049ed).
Disposicion: **CORRECCIONES REQUERIDAS; conservar mediciones; no ganador ni confirmacion**.

## Hallazgos, por importancia

### F1. Alto: gamma cambia de contraste entre regimenes e invierte la lectura de H3

[`effects`, lineas 94-114](https://github.com/harveybc/predictor/blob/6f6c1a00bf2fe8e285f821dfb72d12505a9049ed/tools/df_mod_e0_arch_verify.py#L94)
promedia los brazos de fusion que esten presentes. En r=0 hay `sequence` y
`summary`; en r=1 hay tambien `sequence_gap` y `summary_last`. Restar esos dos
promedios no estima una interaccion del mismo contraste. El bootstrap repite la
formula, no la valida. El test de RP16, linea 217 de `test_df_mod_e0_arch.py`,
codifica precisamente esta mezcla y pasa en la revision examinada.

Recalculo de validacion con **el mismo par sequence - summary**, mismo donante
y las dos replicas, desde los registros guardados:

| Arquitectura | d0 | d1, par comun | gamma, par comun | gamma publicado, mezcla |
|---|---:|---:|---:|---:|
| A | -0.065428 | -0.140286 | -0.074858 | +0.035383 |
| B | -0.027711 | -0.095495 | -0.067784 | +0.009521 |
| C | -0.064308 | -0.131836 | -0.067528 | +0.033999 |
| 0 | -0.074920 | -0.129794 | -0.054874 | +0.040198 |

Control independiente sobre el callable real: errores 0.5 para ambos readouts
last y 1.0 para ambos pooled, identicos en r=0/r=1. No hay interaccion ni efecto
de fusion; el calculador informa gamma=+0.5 en las cuatro arquitecturas.

Retirar la afirmacion de que la dependencia reduce la ventaja del par original.
El gamma factorial equilibrado por readout NO es estimable con lo medido: faltan
los controles r=0. El par comun sigue comparando procedimientos completos, no
aisla causalmente la conservacion de historia: `build_modular`, linea 532, usa
Conv1D para secuencia y Dense para resumen, ademas de un donante dependiente del
receptor. El efecto de readout r=1 es descriptivo; su magnitud no demuestra que
explique casi toda la brecha. `donor.delta` usa el par comun pero su definicion
lo compara verbalmente con el d1 promediado: corregir tambien esa etiqueta.

### F2. Alto: el ultimo calculador admite pares incompletos y un diseno ajeno

[`effects`, linea 53](https://github.com/harveybc/predictor/blob/6f6c1a00bf2fe8e285f821dfb72d12505a9049ed/tools/df_mod_e0_arch_verify.py#L53)
no liga la identidad del diseno con el cierre ni exige los miembros de cada
contraste. Cambiar `design_sha256` por **64 ceros** deja sus
resultados intactos. Marcar `H3__r1__s2__A__summary` como PROBLEMS sigue dando
gamma=+0.058733 y `n_replicates=2`, con otra ponderacion. El mismo patron permite
que H2 promedie solo los controles aleatorios supervivientes.

Un cierre global parcial por DX no invalida automaticamente pares H3 completos,
pero cada efecto necesita poblacion exacta, pesos fijos, replicas completas y
su propio estado. No basta filtrar etiquetas VERIFIED.

### F3. Alto: una restauracion no finita verifica y el replay cacheado no liga codigo

[`verify_attempt`, linea 381](https://github.com/harveybc/predictor/blob/6f6c1a00bf2fe8e285f821dfb72d12505a9049ed/tools/df_mod_e0_close.py#L381):
en una copia en memoria del replay real de H2 h3/s1/A, sustituir
`restore_abs_diff` y `recorded_best_validation_loss` por NaN conserva VERIFIED
sin problemas. La comparacion `>` no demuestra finitud. La comprobacion de
activaciones tiene el mismo patron por inspeccion; esa variante no se ejecuto
en este dictamen y debe tener su propio test.

[`run_replays`, linea 430](https://github.com/harveybc/predictor/blob/6f6c1a00bf2fe8e285f821dfb72d12505a9049ed/tools/df_mod_e0_close.py#L430)
liga archivos de datos/pesos/job/donante, pero no codigo ni entorno del replay.
Un fixture desechable con esos hashes iguales y codigo declarado distinto se
reutiliza sin iniciar proceso. No prueba que los replays historicos sean falsos;
prueba que el cache no acredita reproduccion bajo la implementacion actual.
Conservar versiones historicas; invalidar solo reuso fuera del alcance probado.

### F4. Alto para E1: inventario no equivale a geometria ni a independencia

`program_v3/E1_FAMILIES.json` declara las cuatro familias elegibles, usa W=96,
h=1 y estima soporte desde rangos de fechas. W=96 son 96 minutos en household,
24 horas en electricity, 16 horas en appliances y 4 dias en Beijing. Frecuencia
de muestreo no demuestra periodo diario; 200 bloques sin solapamiento no
demuestran 200 unidades independientes ni potencia. Falta contar las ventanas
efectivamente admisibles por split despues de huecos, disponibilidad y purga.

Esto tiene consecuencias fisicas documentadas: electricity codifica clientes
que aun no existian con ceros y trata especialmente las horas de cambio de
horario portugues (ceros en marzo y agregacion en octubre).
[Fuente primaria UCI](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014).
Household contiene cerca de 1.25% de mediciones faltantes aunque conserve las
marcas de tiempo, ademas de variables con unidades diferentes.
[Fuente primaria UCI](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

Separar catalogo elegible por licencia de tarea ejecutable. UCI es un repositorio,
no una unica fuente fisica: compartir catalogo no demuestra dependencia de los
paneles. A la inversa, 370 clientes no son automaticamente 370 replicas
independientes. Los candidatos de reserva ya tienen RESULT de caracterizacion
en el censo; "no abiertos esta ronda" no significa datos nunca inspeccionados.
Declarar exposicion previa y fijar que reserva de resultados aun es defendible.

### F5. Medio: criterio ML y eficiencia comparativa mal delimitados

En H2 h3, B supera en error a la referencia lineal por **0.019160 y 0.017812
MASE**, ambas dentro de la barra historica `linear + 0.03`. No alcanza su valor
puntual, pero no incumple esa tolerancia. En DX si hay un incumplimiento distinto
(aproximadamente 0.039). El campo `receiver_adequate` solo compara con ingenuo;
no sustituye ninguno de los dos criterios. Ademas el ingenuo de `prepare()` usa
ultima observacion; llamarlo seasonal-naive confunde predictor con denominador.

1100 updates con paciencia 8 no demuestra convergencia comparable de A/B/C/0.
Costo medio de todas las celdas mezcla pilotos, roles y sensibilidad de donante
que solo A/B ejecutan, y hardware. Publicar costo por tarea emparejada, donante
y reutilizacion por separado; no inferir inferioridad general de TCN.
H2 con dos replicas no prueba equivalencia ni "ningun efecto". El bootstrap de
dos semillas no crea precision poblacional ni prueba poder para un margen.

### F6. Medio: las metricas D/Y no describen toda la composicion y soporte consumidos

[`data_metrics`, linea 83](https://github.com/harveybc/predictor/blob/6f6c1a00bf2fe8e285f821dfb72d12505a9049ed/tools/df_mod_e0_metrics.py#L83)
define senal como s+periodic+cross. DX suma ademas `deterministic`; el residuo
maximo de x-(senal+ruido) es **4.749** en el caso probado. La formula publicada
es explicita, pero no representa toda la senal de ese diagnostico: conservarla
como componente o versionar SNR total, sin sustituir silenciosamente historia.
Se describen filas base del split, no por separado el soporte de ventanas X y
filas Y desplazadas por horizonte. Eso no acredita todas las metricas de datos
efectivamente consumidos que pide el contrato D/Y. Corregir grano y denominador.

## Evidencia propia y limites

[Reproductor](../evidence/RP16_REVIEW_2026_09_19/reproduce.py) y
[salida medida](../evidence/RP16_REVIEW_2026_09_19/results.json).
Ejecutado con Python de `trading-stack`, CPU, numpy, y codigo revisado:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -B docs/audits/evidence/RP16_REVIEW_2026_09_19/reproduce.py \
  --repo <checkout-de-la-revision-6f6c1a0> \
  --run-root <root-preservado-mod_e0_arch_stage_v1> \
  --output /tmp/rp16-review-results.json
```

Sin `--run-root` reproduce los defectos del estimador/cache con la evidencia
commiteada; la prueba de restauracion y la lectura fisica requieren ese root.
El script se publica en master; el codigo y la evidencia examinados se leen
del checkout fijado mediante `--repo`, no del runtime distinto de master.
Recalculo independiente de MASE del modelo desde arrays: **120 intentos, cero
discrepancias**, diferencia maxima 7.77e-16. No es un nuevo replay de pesos ni
una nueva conciliacion viva del warehouse. Diseno/cierre originales intactos;
mutaciones solo en copias en memoria o fixtures desechables. Sin entrenar,
sin GPU, sin modificar servicios, base real, evidencia original o reservas.

Tres tests existentes ejecutados: enumeracion arquitectonica, composicion DX y
efectos RP16: **3 passed**. El ultimo pasa pese a F1: evidencia concreta de que
la suite no cubria la pregunta correcta. No reejecute la suite completa de
Satoshi; sus 2056/37/3/8 siguen siendo conteos del ejecutor, no certificacion mia.
Publicacion documental: `check_plan.py` PASS y sus **17 tests passed**. Esto
comprueba cobertura/dependencias del plan, no valida una hipotesis cientifica.

## Disposicion y siguiente trabajo

Aceptar la preservacion y reutilizacion acotada de las mediciones; rechazar la
lectura mezclada de gamma, la elegibilidad E1 completa y las garantias de replay
fuera de su alcance. No borrar ni repetir todas las 112 celdas. Corregir primero
calculo y pruebas; completar solo el control faltante bajo sucesor DEV.

[Ordenes RP17-RP24](../../handoffs/MUSASHI_PROGRAM_RP17_RP24_2026_09_19.md).
[Respuesta a las preguntas de negocio](../../tres_temas_entrevista/program_v3/13D_MUSASHI_BUSINESS_DISPOSITION_2026_09_19.md).
No falta una decision del owner para esos trabajos. Real capital/riesgo/live
siguen sin inventarse y no son condicion para desarrollo publico o sintetico.
