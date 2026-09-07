# Instrucciones de entrega a Musashi — revisión 2

**Autor:** Takeshi. **Fecha:** 7 de septiembre de 2026.  
**Propósito:** actualizar la propuesta doctoral con las aclaraciones de Harvey posteriores a la primera auditoría y a la entrega de figuras.  
**Propuesta de referencia:** `predictor@249f800dd850ec0899428cac3db7bfd7c3e922a0`. No se ha recibido ni auditado aquí una propuesta posterior.

## Orden de lectura y vigencia

1. **01_ACTUALIZACION_PROPUESTA_PARA_MUSASHI.md**: instrucciones vigentes, matriz de cambios, ejemplos de redacción, evidencia y preguntas.
2. **02_DIAGRAMAS_NODOS_Y_ENLACES.md**: descripciones actualizadas de cuatro diagramas, mediante nodos y enlaces etiquetados. Dos son principales; los otros son detalles opcionales.
3. **antecedentes/AUDITORIA_INICIAL_2026_09_07.md**: auditoría completa de la propuesta de referencia. Conserva los ocho grupos P1 y la discusión bibliográfica; sus recomendaciones de arquitectura y entrenamiento se interpretan con las actualizaciones del documento 01.
4. **03_CONTROL_DE_CAMBIOS_Y_ENTREGA.md**: precedencia de instrucciones y condiciones de la siguiente entrega.

## Decisiones principales

- Mantener el título «Diseño y aprendizaje de representaciones temporales modulares».
- Reconocer el preentrenamiento y exportación de encoders como antecedente existente. El ajuste posterior integrado fue descrito por Harvey; las configuraciones inspeccionadas lo contemplan, pero falta localizar su conexión ejecutable exacta.
- Explicitar el detector temporal como submódulo del extractor. Separación modular no equivale a congelación obligatoria ni a independencia estadística.
- Tratar el autoencoder como un régimen de preentrenamiento disponible. No declararlo ganador ni convertir su uso en novedad doctoral por sí mismo.
- Mantener la contribución centrada en diseño y aprendizaje de representaciones; DOIN y la búsqueda L2 son infraestructura y mecanismos de soporte.
- Mantener generación sintética y validación adversarial fuera del núcleo de esta propuesta. Preservar la pareja encoder–decoder preentrenada y diferenciar sus versiones de los encoders ajustados al consumidor.
- Conservar los hallazgos estadísticos y de evaluación de la auditoría. Este paquete no los declara resueltos en un PDF que aún no hemos recibido.

## Formato de diagramas

La versión vigente describe exclusivamente nodos y enlaces con etiquetas. Los PDF gráficos y fuentes TikZ del primer paquete no se incluyen en esta revisión para evitar que sus esquemas anteriores se incorporen como especificación actual. Los archivos anteriores permanecen como entregas previas; no se han borrado.

Las etiquetas, pies y convenciones editoriales están especificados en el documento 02. No se solicita generar imágenes en esta entrega.

## Alcance de autorización

Este paquete está preparado para que Harvey lo entregue a Musashi. No se publicaron cambios en GitHub, no se enviaron mensajes y no se ejecutaron entrenamientos. Las instrucciones proponen la actualización documental y las decisiones necesarias para que el autor pueda revisar una propuesta coherente.
