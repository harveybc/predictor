# Auditoría previa a la propuesta doctoral L2/RL

**Veredicto:** `DRAFT_AUTHORIZED_WITH_EXPLICIT_LIMITS`  
**Fecha:** 2026-09-03

## 1. Qué se corrigió antes de redactar

| Riesgo | Corrección aplicada |
|---|---|
| “AutoRL no existe” o “la práctica es folclor” | Eliminado. AutoRL y sus comparadores principales se reconocen y citan. |
| Trading como dominio principal | Rechazado. El banco público de RL es primario; trading es confirmación externa posterior. |
| Ocho celdas como corpus de metaaprendizaje | Rechazado. Se declara que no existe todavía un corpus L1 suficiente. |
| Sondas diseñadas y validadas en el mismo stack | Rechazado como evidencia doctoral. Los umbrales se calibran y prueban en tareas públicas no vistas. |
| Promesa de predecir el retorno final | Reemplazada por calibración del riesgo de una decisión con opción de escalar o abstenerse. |
| Dos teoremas no relacionados | Reducidos a un solo resultado de selección con rechazo y su región complementaria de imposibilidad. |
| Thresholdout como garantía heredada | Eliminado. Solo inspira separación de datos; no se trasplanta su teorema. |
| MacKay como capacidad de una red profunda | Corregido. El resultado de 2K bits se limita al clasificador lineal de umbral y a su experimento de memorización. |
| Comparación solo contra random/manual | Rechazada. Hyperband/ASHA y BOHB/SMAC-HB son obligatorios; un vecino multifidelidad se añade si sus supuestos se conservan. |
| Semilla como unidad de análisis | Corregido. La unidad es tarea o contexto no visto; semillas anidadas. |
| Abstención vacua | Cerrado con piso de cobertura y costo explícito de escalamiento. |
| “Semanas convertidas en horas” | Eliminado por ser contrafactual. El documento usa un presupuesto medido en FRE y una envolvente que P0 debe reemplazar. |

## 2. Afirmaciones locales que sí sobreviven

1. Existe una superficie real de selección de representaciones: familias, ventanas, fusión, preentrenamiento y arquitectura.
2. La evaluación SAC completa es suficientemente costosa y ruidosa para que la asignación de fidelidad importe.
3. Los diagnósticos internos no ordenan las familias de forma simple: hay señal relativa en algunas, ausencia de habilidad absoluta y un posible cuello de fusión.
4. El camino mecánico del extractor fuerte fue ejecutado con presupuesto acotado, pero no produjo evidencia de desempeño ni checkpoint promovible.

## 3. Qué no se afirmará

- que alguna familia esté “muerta”;
- que la propuesta ya haya ahorrado cómputo;
- que el extractor propio sea estado del arte;
- que una sonda contenga información útil por su nombre o motivación teórica;
- que dos bits por peso midan inteligencia, conocimiento o conciencia;
- que un resultado financiero demuestre generalidad;
- que la infraestructura existente sea una contribución científica por sí misma.

## 4. Ataques de jurado y respuesta honesta

### “Esto ya es Hyperband o BOHB”

Esos métodos asignan presupuesto entre configuraciones. La pregunta adicional es cuándo una evidencia barata, cuyo error cambia entre tarea y configuración, permite una recomendación y cuándo obliga a escalar o abstenerse. Si el método no supera esos comparadores, H1 falla.

### “ProxyBO e iMFBO ya ponderan proxies o fidelidades”

Correcto. Por eso son vecinos directos. La propuesta no reclama ponderación dinámica como novedad; estudia calibración de daño y abstención bajo cambio de tarea, con una región de imposibilidad cuando la fidelidad no es identificable.

### “Un surrogate de RL no es confiable”

Ese es un supuesto que se pone a prueba, no se niega. El método puede decidir no usarlo. Si no aporta cobertura segura frente a controles simples, se elimina y se reporta el resultado negativo.

### “La abstención siempre gana si nunca decide”

No bajo el contrato: debe mantener una cobertura mínima fijada en el piloto y todo escalamiento consume el mismo presupuesto comparado.

### “¿Dónde está la inteligencia artificial?”

El objeto es un sistema que aprende a elegir representaciones y presupuesto para otro proceso de aprendizaje, estima incertidumbre sobre sus propias decisiones y generaliza o se abstiene ante tareas no vistas. La tesis se apoya en AutoRL, metaaprendizaje, selección de modelos y aprendizaje selectivo; la infraestructura financiera no es su centro.

## 5. Riesgos residuales

1. **Novedad:** debe confirmarse con revisión sistemática y no solo con esta pasada dirigida.
2. **Integración:** ARLBench todavía ofrece una superficie arquitectónica limitada; la extensión debe ser pública y pequeña.
3. **Compatibilidad de contexto:** CARL es adecuado para cambios controlados, pero su integración exacta con el banco primario debe probarse en P0.
4. **Costo:** 930 GPU-horas es una envolvente conservadora, no una medición local. P0 puede obligar a reducir el espacio.
5. **Teoría:** la garantía depende de una envolvente de error calibrable. Si no existe, el aporte formal será la imposibilidad y el aporte empírico, la caracterización del fallo.

## 6. Decisión

El pivote puede convertirse en propuesta doctoral porque ahora tiene una pregunta, tres hipótesis, un dominio público, comparadores fuertes, una unidad de análisis y una salida negativa legítima. El memo de Satoshi y la página del fenómeno quedan como acta interna de origen; no son la base textual del PDF.
