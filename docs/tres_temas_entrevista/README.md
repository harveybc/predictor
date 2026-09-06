# Expediente de propuestas doctorales

Este directorio conserva juntas las cuatro propuestas desarrolladas hasta ahora. El nombre
`tres_temas_entrevista` es histórico; ya no describe el contenido completo del expediente.

## Propuestas

| # | Propuesta | Documento | Fuente editable | Estado actual |
|---|---|---|---|---|
| 1 | Selección secuencial de codificadores de memoria para aprendizaje por refuerzo bajo cambio de tarea, con abstención calibrada | [PDF](01_seleccion_representaciones_rl.pdf) | [LaTeX](01_seleccion_representaciones_rl.tex) · [bibliografía](01_seleccion_representaciones_rl.bib) | **Elegida para la postulación a La Sabana** |
| 2 | Memorización, generalización y dimensionamiento de redes neuronales | [PDF](02_memorizacion_generalizacion_dimensionamiento.pdf) | [Markdown](02_memorizacion_generalizacion_dimensionamiento.md) · [HTML](02_memorizacion_generalizacion_dimensionamiento.html) | Alternativa y línea complementaria |
| 3 | Selección automática de transformaciones de entrada para el pronóstico de series temporales | [PDF](03_transformaciones_series_temporales.pdf) | [LaTeX](03_transformaciones_series_temporales.tex) · [bibliografía](03_transformaciones_series_temporales.bib) | Alternativa doctoral e incorporada al plan experimental |
| 4 | Aprendizaje multiagente para distribuir cómputo entre tareas de optimización de inteligencia artificial | [PDF](04_incentivos_red_descentralizada_multidominio.pdf) | [HTML](04_incentivos_red_descentralizada_multidominio.html) | Alternativa sobre recompensas, autoselección y verificación multidominio |

La propuesta 1 es el documento que se prepara para enviar. Las demás no deben fusionarse con ella
por semejanza superficial: cada una tiene una pregunta, una unidad experimental y criterios de
falsación diferentes.

## Historia de la propuesta 4

La versión actual estudia un controlador de recompensas financiadas y la autoselección de dominios
por una población de nodos. No requiere que los nodos revelen sus costos ni supone que el ledger
determine la calidad del trabajo.

También se conserva la versión anterior, recuperada del commit `7ae8643`:

- [Validación e incentivos entre pares para mercados descentralizados de inferencia y optimización (PDF)](04_antecedente_validacion_incentivos_entre_pares.pdf)
- [Fuente HTML](04_antecedente_validacion_incentivos_entre_pares.html)

Ese antecedente estudiaba además evaluación con verdad parcial, jueces automáticos y Correlated
Agreement. No es la versión vigente ni debe citarse como si describiera el controlador multiagente
actual. Se conserva porque documenta la evolución de la pregunta y evita que vuelva a perderse.

La lectura de Astra refuerza una distinción que una próxima revisión de esta línea deberá mantener:
el precio de inferencia, la recompensa por optimización, la remuneración de verificadores y una
eventual emisión monetaria son decisiones diferentes. El ledger aporta orden y trazabilidad, no
verdad. Los retos sintéticos posteriores al compromiso pueden reducir la memorización de un test
fijo, pero no sustituyen la validación de que el generador representa el dominio.

## Integridad de las copias

| Archivo PDF | Paginas | SHA-256 |
|---|---:|---|
| `01_seleccion_representaciones_rl.pdf` | 11 | `0860a4f2975042fbf30400d5920443a7ca46415c58ba2e0cedc2ca2886eb470f` |
| `02_memorizacion_generalizacion_dimensionamiento.pdf` | 10 | `d07c99255672bf7100d8381d335041c024aca670a431cd7dab5f58d0318a4091` |
| `03_transformaciones_series_temporales.pdf` | 10 | `bb051911a9d77fd8603e8735315bcee1a52be1d2f29e8c92de36aa2362329cad` |
| `04_incentivos_red_descentralizada_multidominio.pdf` | 8 | `a9b6c7d45b5dfe7facef8eb7bc7b49321ec8181b01522d70708f36bf7f542f17` |
| `04_antecedente_validacion_incentivos_entre_pares.pdf` | 4 | `8b56799447fdb19e871d8bab0ca25c4dc9a68f31f9ea93d7bb77efd72d4d6b03` |

Las copias de este directorio son un punto de acceso común. Las fuentes de trabajo originales siguen
en sus ubicaciones de desarrollo; cualquier revisión futura debe actualizar también esta copia y su
digest.
