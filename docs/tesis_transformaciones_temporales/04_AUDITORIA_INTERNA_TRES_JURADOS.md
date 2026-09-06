# Auditoría interna de la propuesta desde tres jurados

**Documento auditado:** `propuesta_doctoral_transformaciones_series_temporales.tex`

**Veredicto:** `ACCEPT_FOR_AUTHOR_REVIEW / REVISE_AFTER_COMMENTS_BEFORE_SUBMISSION`

## Jurado A: aprendizaje automático y estadística

### Objeciones

1. Un banco sintético diseñado por el autor puede confirmar exactamente los regímenes que incorporó.
2. Una semilla o ventana no es una tarea independiente.
3. El efecto de denoising puede confundirse con mayor dimensión o presupuesto.
4. Una regla de abstención puede parecer segura porque casi nunca recomienda.
5. La potencia no puede improvisarse después de observar los resultados.

### Respuesta incorporada

- El sintético solo calibra diagnósticos; la utilidad principal se decide en familias públicas retenidas.
- La tarea/familia es la unidad de generalización.
- Se controlan parámetros y se compara `X`, `D(X)` y `[X,D(X),R]`.
- Riesgo y cobertura se reportan juntos.
- El piloto estima varianza; margen, contraste y presupuesto se congelan antes de prueba.

### Riesgo residual

La definición final de tarea y el número efectivo de familias dependen del censo. Si no hay potencia, la propuesta debe aceptar un resultado inconcluso.

## Jurado B: AutoML y metaaprendizaje

### Objeciones

1. AutoML ya busca preprocesadores y modelos completos.
2. Auto-FP ya compara secuencias de preprocesamiento.
3. FFORMA, FFORMPP y otros métodos ya transfieren información entre series.
4. Hyperband y BOHB ya asignan presupuestos de forma eficiente.
5. “Aprender qué transformación usar” no basta como novedad.

### Respuesta incorporada

La propuesta reconoce esos vecinos y limita la novedad a su intersección: selección por variable, contratos temporales verificables, transferencia entre tareas y abstención con costo completo. Los vecinos existentes son comparadores, no decorado bibliográfico.

### Riesgo residual

La revisión sistemática puede encontrar un método que cubra esa intersección. En ese caso se reduce la afirmación y se estudia una diferencia medible; no se cambia de nombre al método existente.

## Jurado C: series temporales y sistemas

### Objeciones

1. En datos reales no hay una separación observable entre señal y ruido.
2. “Causal” puede confundirse con inferencia causal.
3. Una transformación por lotes puede esconder información futura.
4. DOIN y los repositorios locales pueden convertir la propuesta en ingeniería interna.
5. El costo de construir diagnósticos puede borrar cualquier ahorro.

### Respuesta incorporada

- SNR verdadera se limita al laboratorio.
- El documento usa “temporalmente admisible” y define disponibilidad.
- Se exige equivalencia incremental y ajuste solo en desarrollo.
- La evidencia primaria es pública e independiente; DOIN es aplicación posterior.
- Diagnóstico, fallos, transformación y entrenamiento entran en el mismo reloj de costo.

### Riesgo residual

La equivalencia entre operador de referencia y adaptador aplicado tendrá que probarse antes de cualquier campaña de dominio.

## Prueba de lenguaje

Se retiraron del documento principal:

- nombres de fases internas y etiquetas de campañas;
- nombres de repositorios como fundamento científico;
- afirmaciones de que AutoML no hace preprocesamiento;
- uso de SNR verdadera en datos naturales;
- “causal” como adjetivo ambiguo del filtro;
- blockchain, tokens, mercados e incentivos;
- promesas de rentabilidad.

Se conservaron únicamente términos técnicos necesarios y definidos: grafo de transformaciones, transferencia negativa, abstención, arrepentimiento simple, SNR en laboratorio y validez temporal.

## Preguntas que debe resolver la revisión del autor

1. ¿El programa receptor exige una extensión o formato institucional específico?
2. ¿Qué familia pública debe ser primaria después del censo de tareas?
3. ¿Cuál será el margen mínimo de mejora que justifica el costo adicional?
4. ¿Se dispone de los recursos para evaluar el espacio exhaustivo de referencia?
5. ¿Qué modelos simples representan mejor el tipo de pronóstico que se quiere generalizar?

Ninguna de estas preguntas exige cambiar la pregunta madre. Sí deben quedar resueltas antes del prerregistro confirmatorio.
