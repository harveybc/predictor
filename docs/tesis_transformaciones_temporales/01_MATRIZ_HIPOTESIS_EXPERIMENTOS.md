# Matriz de hipótesis y experimentos

**Propósito:** impedir que una mejora de reconstrucción, suavidad o ajuste interno se presente como una mejora de pronóstico.

## 1. Variables y unidades

Una tarea se define por:

```text
fuente de datos
variable o conjunto objetivo
frecuencia
horizonte de pronóstico
regla de disponibilidad
orígenes temporales de evaluación
```

La unidad de generalización es una tarea o una familia completa retenida. Una semilla, una ventana o un horizonte correlacionado no cuenta como otra tarea independiente.

Para una tarea `tau`, una variable `j` y una transformación `d`:

```text
delta(tau,j,d) = pérdida_fuera_de_muestra(identidad)
                 - pérdida_fuera_de_muestra(d)
```

Un valor positivo favorece la transformación. El meta-selector debe estimar el signo, el orden y la incertidumbre de `delta` sin consultar el periodo retenido.

## 2. Banco de verdad conocida

### Señales base

- Donoho-Johnstone: Blocks, Bumps, HeaviSine y Doppler.
- Procesos de espacio de estados multivariados con componentes comunes y privadas.
- Oscilaciones con frecuencia variable y cambios abruptos.
- Sistemas no lineales de baja dimensión con dinámica conocida.

### Perturbaciones

- gaussiana blanca;
- temporalmente correlacionada;
- correlacionada entre variables;
- heteroscedástica;
- impulsiva;
- faltantes en bloques y dispersos;
- retrasos de disponibilidad por variable.

La cuadrícula inicial de SNR verdadera es `{infinito, 20, 10, 5, 0}` dB. Los niveles se aplican tanto de forma homogénea como heterogénea entre variables.

### Qué se puede afirmar

Cuando `X = S + N` se conoce por construcción:

```text
SNR_j = 10 log10(sum(S_j^2) / sum(N_j^2))
```

Se puede medir error de estimación de SNR, ordenamiento entre canales, reconstrucción de `S` y preservación de eventos. Fuera de este banco, `S` y `N` no están identificados y esas cantidades dejan de ser verdad observable.

## 3. Bancos públicos

### Cobertura amplia

- Monash Time Series Forecasting Archive.
- Una muestra predeclarada de M4.

### Confirmación multivariada

- ETT.
- Electricity.
- Traffic.
- Weather.

El censo inicial documenta licencia, frecuencia, tamaño, faltantes, número de tareas independientes, disponibilidad temporal y horizonte. Ningún conjunto se incorpora solo porque sea popular.

## 4. Modelos de pronóstico congelados

La primera fase usa modelos deliberadamente sencillos:

1. pronóstico estacional ingenuo;
2. regresión lineal sobre rezagos;
3. un perceptrón pequeño con presupuesto fijo.

Después se añade un solo modelo secuencial confirmatorio. Una transformación que solo funciona al cambiar simultáneamente arquitectura, presupuesto y objetivo no identifica el efecto del preprocesamiento.

## 5. Experimentos

### E0. Contrato temporal

**Pregunta:** ¿cada operador produce en línea exactamente lo que habría producido al recorrer la serie sin mirar el futuro?

**Pruebas:** disponibilidad por variable; ajuste solo en desarrollo; igualdad entre ejecución incremental y por lotes permitida; rechazo de ventanas centradas en modo desplegable; costo y retraso publicados.

**Falla:** cualquier dependencia de prueba, timestamp futuro o ajuste global no declarado.

### E1. Calibración del diagnóstico de perturbación

**Pregunta:** ¿los diagnósticos baratos recuperan el orden de ruido entre variables y distinguen regímenes?

**Métricas:** error de SNR donde existe verdad; correlación de rango; cobertura de intervalos; calibración por tipo de perturbación.

**Falla:** un diagnóstico entrenado en ruido blanco se presenta como general o no supera un estimador que solo conoce la familia de datos.

### E2. Eliminación de ruido y conservación

**Brazos:** `X`; `D(X)`; `[X,D(X),R]`, donde `R = X-D(X)`.

**Métricas sintéticas:** ganancia de SNR, error de reconstrucción, desplazamiento temporal y conservación de extremos.

**Métricas de aprendizaje:** MASE o pérdida escalada, CRPS/pinball cuando aplique, desempeño en colas y utilidad incremental del residuo.

**Falla:** mejora la reconstrucción pero empeora el pronóstico; el residuo conserva señal útil; el efecto desaparece bajo presupuesto pareado.

### E3. Decisión global frente a decisión por variable

**Pregunta:** ¿tratar cada variable por separado aporta cuando las perturbaciones son heterogéneas?

**Controles:** mismo presupuesto de parámetros; mismo conjunto de transformaciones; dimensión de entrada controlada; perturbación homogénea como control negativo.

**Falla:** la mejora proviene únicamente de añadir canales o no aparece en tareas retenidas.

### E4. Censo tarea-grafo

Se evalúa exhaustivamente un espacio pequeño en un subconjunto público para construir una matriz factual de pérdida, costo, fallos y aplicabilidad. El espacio inicial contiene identidad, una familia de filtros hacia atrás, una representación tiempo-frecuencia, una descomposición común-privada y combinaciones de profundidad máxima dos.

**Falla:** la matriz es demasiado dispersa para separar tareas de configuraciones o los fallos no se conservan como evidencia.

### E5. Selección bajo presupuesto

**Comparadores:** identidad; mejor grafo fijo en desarrollo; búsqueda aleatoria; Hyperband; BOHB; vecino por tarea; modelo de árboles.

**Métricas:** arrepentimiento simple, mejor pérdida por evaluación, costo hasta quedar dentro de `epsilon` del mejor grafo conocido, costo de diagnósticos y tasa de configuración inválida.

**Falla:** el meta-selector no supera los comparadores sencillos después de contabilizar todo el costo.

### E6. Abstención

**Acciones:** recomendar; pedir una evaluación adicional; usar identidad; abstenerse.

**Métricas:** cobertura, riesgo entre recomendaciones, frecuencia y magnitud de transferencia negativa y costo adicional.

**Falla:** la aparente seguridad se obtiene absteniéndose casi siempre o calibrando el umbral sobre prueba.

### E7. Generalización pública

Se retienen familias completas. El entrenamiento, la calibración y la prueba no comparten series ni transformaciones ajustadas. Se prueba sensibilidad a frecuencia, horizonte, longitud, faltantes y cambio de distribución.

**Falla:** el resultado existe solo dentro de una colección o depende de la identidad del conjunto.

### E8. Transferencia a DOIN

Solo los operadores aprobados entran como genes tipados o configuraciones del dominio. El meta-selector propone el arranque; DOIN ejecuta la búsqueda y el periodo retenido decide.

**Comparadores:** contrato vigente; mejor grafo fijo público; lista corta del meta-selector; búsqueda DOIN sin warm start.

**Falla:** el ahorro público no se conserva, el adaptador cambia la semántica del operador o la recomendación intenta reemplazar la validación propia del dominio.

## 6. Correspondencia con hipótesis

| Hipótesis | Evidencia principal | Evidencia confirmatoria | Resultado que la refuta |
|---|---|---|---|
| H1: efecto condicional predecible | E1-E3 | E4 y familias públicas retenidas | no hay información incremental o hay fuga |
| H2: transferencia reduce costo | E4-E5 | E7 | no supera búsqueda sin historia con costo completo |
| H3: abstención reduce transferencia negativa | E6 | E7 y cambio aplicado | no mejora riesgo a cobertura mínima |

E8 no crea una cuarta hipótesis. Comprueba si los hallazgos sobreviven al traslado de dominio.

## 7. Disciplina estadística

- El piloto estima varianza y factibilidad; no elige la dirección del efecto.
- El margen de relevancia práctica, las familias de contrastes y el presupuesto se fijan antes de abrir la prueba.
- La potencia se calcula por simulación con la estructura de tareas observada.
- Se usan comparaciones pareadas en los mismos orígenes y presupuestos.
- La multiplicidad se corrige por familias predeclaradas.
- Si faltan tareas independientes, el resultado es `INCONCLUSIVE`; no se reemplazan por más semillas.
- Todo fallo, tiempo agotado y grafo no aplicable permanece en la matriz.
