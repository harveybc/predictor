# Revision bibliografica de la presentacion

Fecha: 2026-09-27. Alcance: fuentes, citas y bibliografia; no nuevas mediciones.

## Comprobaciones

- Trece referencias ordenadas por primera aparicion, con enlaces a las fuentes.
- Literatura contrastada con editoriales, arXiv, actas y documentacion oficial.
- `references.json` declara, referencia por referencia, la afirmacion respaldada
  y los limites de esa atribucion. Las hipotesis son propuestas, no resultados
  demostrados por las referencias.
- `VERIFICATION.json` registra hashes, partes modificadas y comprobaciones XML.
- Todas las imagenes conservan exactamente sus bytes originales. Los objetos
  existentes ajenos a las citas y bibliografia conservan su contenido y formato.
- La bibliografia ocupa dos diapositivas para mantener legibilidad. Se reviso
  el PDF renderizado, incluyendo las nuevas fuentes y ambas paginas finales.

## Fuentes de las figuras

La figura 1 cita la figura 43 de la tesis de H. Bastidas, A. Caicedo y
C. Sarmiento (2025). El DOCX se publico sin modificar en
`docs/tesis_maestria_ds/tesis_maestria_ciencia_datos_2025.docx`, commit
`80a1513a` de predictor.

La figura 2 que el usuario conservo es el barrido historico de un oraculo
direccional. Su pie se corrigio para no atribuirla al barrido posterior de
predicciones de precios H6/H144. La imagen no se sustituyo. CSV, datos de
entrada, codigo retenido y figura exacta estan en heuristic-strategy,
`docs/presentation_noise_20260927/historical`, commit `f2d3922`.
Los ultimos resultados nativos se preservan por separado en `latest_native`.
El archivo explica sus limites: no se afirma replay exacto del servicio
historico, ni se identifica su cruce de beneficio cero con el MAE naive.

## Preservacion

El respaldo local `antes_referencias.pptx` conserva el archivo recibido.
`update_references.py` opera sobre ese respaldo y genera otro archivo; nunca
reescribe las imagenes. El reemplazo del documento principal se hace solo
tras comprobar que el usuario no lo modifico desde el respaldo.

No se retocaron las hipotesis, las figuras, los textos de contenido ni la
numeracion de pagina preexistente fuera de la bibliografia.
