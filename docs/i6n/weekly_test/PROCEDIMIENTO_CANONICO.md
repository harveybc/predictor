# TEST semanal: primero predecir, despues reajustar

Orden explicita del propietario, 2026-10-17. Este documento sustituye el
procedimiento equivocado que entrenaba mirando la misma semana que iba a
evaluar. Leerlo antes de modificar, ejecutar o interpretar esta campana.

## Secuencia Obligatoria

1. Cargar los pesos guardados del ultimo checkpoint util de VALIDATION de cada
   familia: corto y largo. Cargar tambien SUS escaladores y configuracion.
   No inicializar otro modelo al azar ni entrenar antes de la primera prediccion.
2. Hacer inferencia para la primera semana de TEST con ese checkpoint. Cada
   entrada termina en el tick t conocido; nunca puede contener t+1 ni posteriores.
   Guardar y sellar las predicciones antes de consultar los resultados de la
   semana para puntuar o seleccionar un nuevo checkpoint.
3. Al transcurrir esa semana, calcular y registrar MAE y naive propios de cada
   horizonte, sobre las mismas filas, mascaras y escala de precio original.
4. Entonces reajustar desde los pesos anteriores, sin reinicializarlos. TRAIN
   abarca los cuatro anos de calendario ANTERIORES AL INICIO DE ESA SEMANA ya
   evaluada. Por ejemplo, al comienzo de TEST incorpora VALIDATION y los tres
   ultimos anos de TRAIN, conforme a las fechas exactas del corte semanal.
5. Conservar la metrica original de paciencia:
   M = (loss_TRAIN + loss_de_la_semana_YA_TRANSCURRIDA) / 2.
   Paciencia 10, mejora minima 1e-8 y maximo 200 epocas. Esa semana ahora conocida
   puede guiar el reajuste para LA SIGUIENTE, nunca cambiar su prediccion pasada.
6. Guardar el nuevo checkpoint y SUS nuevos escaladores. Usarlos para inferencia
   de la segunda semana de TEST. Anotar el resultado; despues reajustar usando
   los cuatro anos anteriores al inicio de esa segunda semana. Repetir en orden
   cronologico hasta terminar TEST.

| Momento | Prediccion con | Reajuste posterior | Destino del reajuste |
| --- | --- | --- | --- |
| Semana TEST 1 | Ultimo checkpoint VALIDATION | TRAIN: cuatro anos antes de semana 1; monitor: semana 1 ya conocida | Semana TEST 2 |
| Semana TEST 2 | Checkpoint posterior a semana 1 | TRAIN: cuatro anos antes de semana 2; monitor: semana 2 ya conocida | Semana TEST 3 |
| Semana TEST n | Checkpoint posterior a semana n-1 | Nunca modifica la prediccion sellada de semana n | Semana TEST n+1 |

No sustituir este orden por entrenar con la semana n y luego puntuar la misma
semana n. No congelar un modelo para todo el ano. No introducir otro holdout,
otro criterio de paciencia, una busqueda nueva o un modelo nuevo.

## Lo Que No Cambia

- Arquitectura, pesos de partida autenticos, features y orden de columnas.
- Ventana de 13 observaciones cada dos horas de mercado abierto, incluido t.
- Entradas sin suavizado; horas de cierre no cuentan como observaciones.
- Semilla 6102026, AdamW, learning_rate .001, weight_decay .0001, batch 256.
- Corto conserva sus siete salidas reales: 2,3,4,5,6,12,18 h. Largo conserva
  18,24,36,48,60,72 h. No eliminar una salida del checkpoint que se esta cargando.
- Mascaras por horizonte, apertura +12 h y cierre forzoso -12 h. La historia
  de entrada puede cruzar una semana anterior sin inventar barras de fin de semana.
- Error relativo = (MAE_h - naive_h) / naive_h; negativo es mejor. Cada horizonte
  tiene SU naive. Agregacion anual por sumas de errores y conteos de pares.
- Guardado de .keras, pesos, escaladores, configuracion, predicciones y hashes;
  metricas en OLAP con lectura de vuelta y copias fisicas verificadas.

Los escaladores se ajustan exclusivamente con TRAIN del reajuste correspondiente.
En inferencia se usan los del checkpoint, no los de la semana que aun no ha
sido reajustada. El optimizador conserva sus parametros originales y se crea
para cada ajuste, como hacia el runner; los pesos de la red SI se heredan.

## Casos De Borde

- Una semana sin pares validos no tiene MAE ficticio ni permite early stopping:
  se registra SIN_PARES y se conserva el checkpoint para la siguiente semana.
- El checkpoint inicial debe haber terminado su semana de VALIDATION antes de
  cualquier origen de TEST que puntue. Nunca tomar un checkpoint entrenado en TEST.
- Una caida despues de sellar la prediccion no autoriza reemplazarla por la
  salida del modelo reajustado. Se retoma el reajuste desde su donante anterior.
- La adaptacion con semanas de TEST ya observadas es una evaluacion prequential
  del negocio. Las etiquetas de semanas posteriores no pueden modificar sus
  predicciones anteriores. Los supuestos FRED y de calendario se conservan como
  limitaciones declaradas, no se convierten en certificados por esta ejecucion.

## Pruebas Obligatorias

PS01: primer donante es el ultimo VALIDATION autentico, pesos exactos.
PS02: orden predict -> sellar -> score -> update -> siguiente semana.
PS03: variar targets de la semana actual no cambia su prediccion sellada.
PS04: variar semanas futuras no cambia el reajuste de una semana anterior.
PS05: TRAIN mide cuatro anos antes del inicio de la semana observada.
PS06: init del reajuste coincide con pesos del checkpoint usado para predecir.
PS07: se usan escaladores del donante en inferencia y de TRAIN en el reajuste.
PS08: formula y parametros originales de parada permanecen intactos.
PS09: reanudacion no sobrescribe una prediccion; resultados viejos no se adoptan.
PS10: publicacion, mirrors, estado y sucesion son automaticos y acotados.
