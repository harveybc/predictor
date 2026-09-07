# Respuesta a la auditoría de Astra sobre la propuesta de codificadores de memoria

**Fecha:** 2026-09-06
**Auditoría recibida:** `08_AUDITORIA_ASTRA_CODIFICADORES_MEMORIA_2026_09_06.md`
**Versión auditada por Astra:** PDF SHA-256 `0860a4f2975042fbf30400d5920443a7ca46415c58ba2e0cedc2ca2886eb470f`
**Disposición:** `P1_RESUELTOS / LISTA_PARA_LECTURA_FINAL_DEL_AUTOR`

La auditoría fue correcta en su diagnóstico general: no había una razón para cambiar de tema, pero las garantías estadísticas, el estimando de riesgo y la contabilidad de costos aún admitían interpretaciones incompatibles. La revisión conserva la pregunta doctoral y cierra los seis grupos P1 con una sola ruta metodológica.

## 1. Disposición de los seis P1

### P1-01. Cobertura secuencial

Se eligió una ruta única. El modelo jerárquico se usa para predicción y adquisición; la validez se calibra sobre la trayectoria completa de una política de consulta congelada en una malla finita. Cada tarea de calibración aporta un solo puntaje máximo sobre candidatos y etapas. El umbral puede detener la trayectoria, pero no reordenarla ni cambiar la evidencia previa. La referencia final conserva la incertidumbre de las semillas ocultas.

La garantía se declara simultánea a lo largo de la trayectoria y marginal para una tarea nueva bajo intercambiabilidad. Se retiró cualquier lectura de cobertura condicional universal. Si la construcción no puede justificarse frente a la adquisición y parada adaptativas, el resultado se limitará a calibración empírica y el componente de certificación de H2 será inconcluso.

### P1-02. Riesgo observable

El arrepentimiento se define como

`R_t^N = [V_t(c*) - V_t(c_hat)] / s_t`,

donde `s_t` es una escala positiva obtenida mediante una política de referencia fija, semillas separadas y una regla congelada antes de calibración. El mismo umbral `epsilon` se usa en la recomendación y en H1-H3.

La clasificación se hace con un intervalo para el arrepentimiento, no con un empate entre candidatos. Los casos no resueltos permanecen en el denominador y cuentan como posibles perjuicios en la cota conservadora. La unidad primaria es el entorno base, con igual peso; dificultades, contextos y semillas quedan anidados.

### P1-03. Soporte, calibración y dependencia

Se separaron cuatro roles: piloto, desarrollo, calibración y prueba confirmatoria. Los pliegues agrupados sirven para desarrollar el procedimiento, pero no se presentan como certificación de un selector fijo. Después de desarrollo se congelan el selector y la regla de consulta; otras tareas calibran los umbrales y un último grupo se abre una sola vez.

La cuenta 15/59 queda explícitamente identificada como un caso binomial ideal que no incluye desarrollo, calibración, abstenciones, referencias inciertas ni multiplicidad. La ampliación a otro banco depende de un censo de compatibilidad y del presupuesto total. Si el soporte no alcanza, se publica la curva riesgo-cobertura y H2 queda inconclusa en su parte de certificación.

### P1-04. Contrastes y cronología

H1 es la conjunción de superioridad en costo y no inferioridad en arrepentimiento. H2 compara el riesgo conservador de las recomendaciones emitidas con el de la variante obligada sobre todas las tareas, junto con cobertura y costo. H3 reutiliza el contrato de medición, pero requiere soporte propio bajo cambio de contexto.

Se distinguen `delta` (límite de riesgo), `delta_int` (error de cobertura de intervalos) y `alpha` (nivel inferencial). Cada hipótesis admite evidencia favorable, evidencia contraria o resultado inconcluso. Se establecieron dos hitos públicos: registro inicial del diseño y registro confirmatorio después de desarrollo y calibración, antes de abrir la prueba.

### P1-05. Controles y vecinos directos

Se añadió como control decisivo un único codificador fijo elegido en desarrollo y aplicado sin consultar curvas de la tarea nueva. También se incorporó ifBO como competidor completo de adquisición por curvas, y se discutieron DyHPO y la optimización multifidelidad sensible al costo con transferencia de curvas.

La novedad ya no se atribuye a la adquisición incremental, la transferencia, la parada o la suma de esos componentes. La brecha se restringe a adquirir evidencia parcial de codificadores de memoria y decidir selectivamente bajo generalización entre tareas de RL.

### P1-06. Costo y amortización

El costo primario es tiempo de pared bajo un perfil fijo de recursos y concurrencia uno. Termina al disponer de la política seleccionada completamente entrenada o al emitir una abstención terminal. Cobra todos los tramos, descriptores, recuperación y fallos consumidos; CPU-horas, GPU-horas, memoria y ejecución asíncrona se reportan aparte.

Se separan costo común de evaluación, costo previo del corpus y del ajuste, y costo marginal por tarea. La amortización se presenta en dos escenarios, con corpus construido desde cero o disponible para todos los métodos compatibles, y con incertidumbre. No se reportará un punto de equilibrio finito si el ahorro marginal no se distingue de cero.

## 2. Correcciones P2 incorporadas

- El título usa ahora `en tareas no vistas`, evitando confundir generalización entre tareas con cambio dentro de un episodio.
- `Fidelidad` y `semilla aleatoria` se definen en lenguaje llano desde el resumen.
- El objetivo 2 usa una cota superior de arrepentimiento en vez de un margen ambiguo entre candidatos.
- La imposibilidad cubre toda la interacción accesible a la política de consulta, no la coincidencia de una historia aislada.
- Se fijarán horizonte disponible, atención, truncamiento recurrente, límites y normalización.
- El punto de partida del selector es un proceso gaussiano jerárquico; la adquisición inicial maximiza la reducción esperada de la cota de arrepentimiento por segundo.
- Se corrigieron autoría y enlace oficial de ASHA, metadatos de ARLBench y versión de la referencia de control selectivo.
- El resumen se redujo y se retiró repetición defensiva sin eliminar restricciones metodológicas.
- Para H3 se distingue el meta-selector congelado del reentrenamiento de cada candidato y se define qué significa ocultar contexto.

## 3. Decisiones de alcance

No se añadieron nuevas familias de arquitecturas, dominios ni hipótesis. Tampoco se incorporó cada preprint reciente sugerido: se eligieron los vecinos necesarios para delimitar la contribución sin convertir la propuesta en una revisión bibliográfica. El estudio financiero permanece como réplica opcional y no puede alterar H1 ni H2.

## 4. Verificación editorial

- Documento final: 11 páginas en tamaño carta y fuente base de 11 puntos.
- Referencias: 32 entradas, todas citadas y resueltas por Biber.
- Compilación: `latexmk -g -pdf -interaction=nonstopmode -halt-on-error`, sin referencias indefinidas ni cajas desbordadas.
- Revisión visual: 11/11 páginas, sin solapamientos, cambios arbitrarios de tamaño ni tablas o ecuaciones cortadas.
- PDF final SHA-256: `ba8502c21f0f918123607e43487ea83aabccbb0a6612e5ed5d51dcafd423cf7b`.

## 5. Resultado

Los seis P1 quedan resueltos en el texto. La limitación científica restante es deliberada y visible: el piloto debe determinar si existe un banco compatible y presupuesto suficiente para sostener la certificación de H2. Si no existe, la tesis conserva H1 y el análisis empírico de riesgo-cobertura, pero no afirmará una garantía que el soporte no permite.
