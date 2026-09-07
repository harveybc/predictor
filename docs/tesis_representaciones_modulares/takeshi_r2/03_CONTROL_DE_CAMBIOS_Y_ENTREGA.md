# Control de cambios y retorno solicitado

**Revisión 2 · 7 de septiembre de 2026**

## 1. Qué cambia frente al primer paquete

| Recomendación o supuesto anterior | Tratamiento vigente | Justificación |
|---|---|---|
| El detector separado requería un propósito propio aún no concretado | Explicitarlo como submódulo con interfaz y posibilidad de preentrenamiento | Harvey precisó el objetivo auxiliar y su infraestructura previa |
| Preentrenamiento y ajuste presentados como posibilidades futuras | Reconocer exportación inspeccionada, configuraciones y uso histórico declarado | Evitar atribuir al doctorado la construcción de capacidad existente |
| Conservar aprendizaje conjunto como ruta principal sin distinguir inicialización | Mantenerlo como ruta disponible; documentar R0/R1/R2 y fijar régimen antes de confirmación | La arquitectura modular no determina cómo se inicializan o congelan sus pesos |
| Evitar añadir autoencoders por reflejo | Incorporar su papel justificado como antecedente y alternativa acotada | La aclaración del autor aporta una razón técnica concreta; no habilita búsqueda exhaustiva |
| Figura con extractor tratado como caja única | Describir detector, integración y adaptación dentro de cada rama | Hace visible la decisión de frontera sin exigir tres redes separadas |
| Longitud temporal común dibujada como caso principal | Mantenerla solo si el método elegido la cumple | El encoder CNN inspeccionado reduce resolución |
| Figuras gráficas y código TikZ en el ZIP | Sustituirlos por descripciones estructuradas de nodos y enlaces | Formato vigente solicitado para esquemas; evita incorporar gráficos desactualizados |
| Decoder tratado solo como componente auxiliar | Añadir preservación de pareja original y linaje del encoder ajustado | El ajuste puede cambiar la semántica latente |
| Generación sintética como posible extensión | Mantenerla fuera del núcleo, sin experimento ni garantía adicionales | Decisión explícita del autor |

## 2. Vigencia de los ocho P1 de la auditoría

| Grupo | Estado en este paquete | Actualización asociada |
|---|---|---|
| P1-01. Decisiones de hipótesis | Conservado; no resuelto por esta actualización | Mantener distinción entre beneficio, equivalencia, perjuicio e incertidumbre |
| P1-02. Historia accesible y cabezal | Conservado y ampliado | Incluir alcance y resolución del detector y conexiones posteriores |
| P1-03. Regla y heterogeneidad | Conservado | No cambiar perfiles de entrada por perfiles latentes sin declarar un nuevo mecanismo |
| P1-04. Perfiles marginales y relaciones predictivas | Conservado | El preentrenamiento no elimina los contraejemplos de identificabilidad |
| P1-05. Separación y dependencia | Conservado y ampliado | Incluir corpus de preentrenamiento y linaje de representaciones |
| P1-06. Ablaciones y controles | Conservado y ampliado | Separar efecto estructural del régimen R0/R1/R2 |
| P1-07. Antecedentes y novedad | Conservado y ampliado | Añadir antecedentes de autoencoders y aprendizaje autosupervisado |
| P1-08. Banco, métrica y presupuesto | Conservado y ampliado | Contar decoder auxiliar, preentrenamiento y reajustes |

La auditoría histórica describe el PDF original y no certifica el nuevo método. Sus ejemplos de resumen, hipótesis y alcance deben adaptarse con el documento 01 antes de copiarse. En particular, las frases que presuponen entrenamiento desde cero o una sola forma de aprendizaje no deben prevalecer sobre la distinción de regímenes actualizada.

## 3. Lo que debe devolver Musashi

1. PDF, fuente LaTeX y bibliografía actualizados, con título conservado.
2. Matriz U01–U12: aplicado, reformulado con justificación o pendiente, indicando sección afectada.
3. Respuestas a las doce preguntas del documento 01 y al cierre de los ocho P1 originales.
4. Contrato conceptual y decisiones pendientes sincronizados con el texto.
5. Diagramas descritos mediante nodos y enlaces etiquetados, coincidentes con la arquitectura y el entrenamiento definitivos.
6. Evidencia del ejemplo canónico de integración del encoder: commit, función, configuración y formas de entrada/salida. Si existe una ejecución pertinente, aportar su referencia; no afirmar ejecución por inspección estática.
7. Identificación de qué variantes son desarrollo exploratorio y cuáles tienen contraste confirmatorio.
8. Verificación editorial y hash del PDF efectivamente devuelto. No declarar resolución de hallazgos por haber corregido solo esta carta de respuesta.

## 4. Criterio de proporcionalidad

No se solicita ejecutar una tesis antes de la admisión ni fijar todos los hiperparámetros. Sí se solicita que el lector pueda identificar un procedimiento coherente, las decisiones que se investigan, sus comparadores y qué conclusiones admitiría la evidencia. Un parámetro pendiente es aceptable si su mecanismo de elección está delimitado; un cambio silencioso de objeto científico no lo es.

## 5. Entrega de Takeshi

La revisión integra las aclaraciones posteriores de Harvey y las fuentes ya inspeccionadas. No se modificó el repositorio, no se enviaron mensajes a Musashi ni se entrenaron modelos. La auditoría original se conserva en antecedentes; el documento 01 y este registro determinan la vigencia de las instrucciones para la siguiente versión de la propuesta.
