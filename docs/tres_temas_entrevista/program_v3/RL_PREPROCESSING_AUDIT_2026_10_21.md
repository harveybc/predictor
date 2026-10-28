# Auditoria del escalado efectivo de RL

Solicitud del propietario: rastrear rolling z-score256 y transformaciones
adicionales que puedan afectar los resultados. Revision solo lectura; no se
altero la cola activa12/13/14 ni sus contratos,datos o modelos. TEST no leido.

## Resultado verificado

- Viene del default de gym-fx ad8bbc41,2026-05-03. Musashi lo conservo y el
  adaptador de entradas seleccionadas9c286e0e lo exige desde2026-10-19. No hay
  evidencia aqui para adjudicarselo a Satoshi ni para decir que256 fue escogido
  por un contraste de este SAC. Git usa un autor compartido, no identifica agente.
- Activo: media/desviacion por variable sobre hasta256 barras H1 disponibles,
  incluida la ultima barra completa, sin posteriores. No son256 ticks2h. La
  ventana13x2 de la red no es el unico contexto de su procesamiento.
- Se aplica a las15 variables, incluidos RSI y senos/cosenos. La lista de
  excepciones esta vacia. Los ciclos no conservan sus coordenadas nativas.
- Hay clipping a[-10,10], heredado del plugin y no explicito en el config de
  tarea. Es reproducible por el pin de codigo, pero debe quedar explicito en
  nuevos experimentos. Existe fallback std<1e-8 ->1.
- NaN->0 e Inf->limites existe como defensa generica; este contrato comprueba finitud de
  ventanas e historial antes. Las dos matrices auditadas son finitas; no se
  demostro imputacion de faltantes en esta ruta.
- Recompensa=(equity_nueva-equity_previa)/capital_inicial*1000. Escala del
  aprendizaje, no del profit en USD reportado. No se encontro segunda
  normalizacion de features en los wrappers/build/extractor inspeccionados,
  ni wavelets/filtros centrados/suavizado en el constructor de observaciones.
  Esto no vuelve a certificar el productor de cada feature anterior.

## Sonda real sin entrenamientos

Control13 cerrado y copiado:24862 filas TRAIN,6221 VALIDATION, hashes de
CSV/manifiestos verificados. Doce testigos del plugin exactos y perturbaciones
posteriores sin efecto. Se contaron todas las ventanas completas13x2, no la
mascara de entradas del negocio:24838/6197 origenes. Celdas solapadas recortadas:

| Campo | TRAIN | VALIDATION |
| --- | --- | --- |
| RAW_LOG_RANGE | 16 | 1 |
| RAW_CLOSE_LOG_RETURN | 19 | 1 |
| RAW_ELAPSED_HOURS | 1036 | 245 |
| Total | 1071 | 247 |

No son conteos de trades ni barras unicas. Sin clipping en los otros12 campos;
los campos ciclicos cambian de escala, aunque no llegan al limite10.

## Interpretacion y siguiente contraste propuesto

Causal no significa apropiado ni optimo. Tampoco queda probado que rolling256
sea la causa de las perdidas. La eleccion de ventana/input/modelo actual es
condicional a esa transformacion. No modificarla a mitad de una corrida.

Antes de consolidar una receta final, proponemos comparar z-score fijo ajustado
solo con TRAIN frente a rolling256, manteniendo todo lo demas. Luego aislar
ciclos sin reescalar y clipping desactivado. Guardar estadisticas del escalador
con los pesos y hacer explicitos todos los defaults. Las combinaciones y
transformaciones por familia se contrastan despues. Son propuestas: no estan
implementadas ni encoladas y no tienen resultados. No reemplazar un256 sin
justificar por otro numero arbitrario.

Detalle tecnico: agent-multi docs/work_plan/58_PREPROCESSING_DEFAULT_AUDIT.md.
La cola actual sigue intacta y automatizada; el efecto economico del escalado
queda pendiente de experimentar, no resuelto por esta auditoria.
