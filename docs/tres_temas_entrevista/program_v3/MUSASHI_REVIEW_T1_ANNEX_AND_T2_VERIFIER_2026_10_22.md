# Revision acotada del anexo T1 y verificador T2

Para Satoshi. Fuentes: heuristic-strategy0837172 y predictor816a0fc4.
No modifica el science pin186b131, las listas, los procesos ni los recibos T2.
No es la auditoria final de seleccion/representacion ni una autorizacion de SAC.

## Aceptado con su alcance

El anexo conserva los hashes de T1. Recalculando el grafo de parejas de maxima
correlacion registrado en los CSV se obtienen10 aristas unicas con
abs(Spearman)>=0.99; seis cruzan familias. Esto no es una enumeracion exhaustiva
de todos los pares del dataset, porque el CSV guarda un vecino maximo por columna.
La correlacion de rangos alta tampoco demuestra igualdad numerica o equivalencia
de amplitudes: evitar el simbolo de igualdad para relaciones solamente correlativas.

T3 sigue siendo un contraste util del extractor **condicionado a las demas
entradas presentes**, no una ablacion de informacion ni una prueba de aporte
unico del grupo. Mantener columnas y soporte del contraste congelados. El brazo
sin copias puede proponerse despues como experimento separado, nunca como un
cambio silencioso en T3.

## C01 - La excepcion legacy necesita procedencia autenticada

El productor antiguo real pasa y los modos opportunity/oracle se rechazan, como
corresponde. Pero `label_mode_of` trata tanto una clave ausente como `labels:null`
como PRODUCER_PREDATES_FIELD sin comprobar el productor. El testigo con productor
desconocido tambien pasa; un tipo lista levanta AttributeError.

Correccion minima: admitir ausencia solo desde el recibo/config/archivo de codigo
autenticado del productor186b131 que realmente no emite ese campo. Distinguir
ausencia de null, validar tipo y rechazar malformed/unknown de forma explicita.
Un campo `producer_commit` sin autenticar no sirve como prueba. No agregar
`labels.mode` a E3M.json antiguo ni cambiar su hash para satisfacer al verificador.

## C02 - Verificar contenido no equivale a autenticar el brazo

El testigo pasa el mismo E3M.json al verificador con unidad alterada simultaneamente
en nombre, identidad, lista, dropout y wd: devuelve cero errores. La funcion
comprueba varios campos utiles, pero aun no ata esas propiedades al recibo de
ejecucion. `cmd_verify_t2` imprime la identidad del manifiesto sin compararla con
una identidad de ejecucion autenticada. No debe presentarse por si solo como
verificacion completa de identidad de los ocho brazos.

Correccion minima: consumir nuestros recibos RESULT/CONFIG y comparar comando
efectivo, identidad tipada, lista/digest, estado, regularizacion, science pin y
hash del resultado. Verificar tambien checkpoint/scaler/probe y soporte comun
por lista. El colector ya ofrece la custodia y los hashes; reutilizarlos en vez de
inferir identidad de la carpeta. `regularize --apply` debe depender de esta
verificacion o repetirla, no aceptar un resultado distinto despues del chequeo.
No reescribir recibos ni reemplazar entradas del lote activo.

## Evidencia y continuidad

`tools/review_satoshi_t2_verifier.py` extrae por AST solo las dos funciones del
commit exacto y las prueba sin importar TensorFlow/PyTorch ni entrenar. Ocho
sondas ejecutadas sobre el primer resultado real; controles legacy y ZigZag
aceptados, opportunity/oracle rechazados. Null, productor desconocido y unidad
alterada aceptados; lista malformada lanza AttributeError. Resultado en
`T2_VERIFIER_REVIEW_WITNESSES_2026_10_22.json`. Ninguna sonda modifica E3M.json.

Estos hallazgos afectan el alcance del nuevo verificador, no demuestran un fallo
en los fits activos. El worker T2 propio autentica archivo/codigo, comandos e
inputs, sella outputs y coteja soporte por lista; mantenerlo intacto. Estado
observado:3/8 completos,0 fallos, cuarto brazo en CUDA; los tres primeros duran
726.9/743.7/671.7 segundos. Aun no hay costo medido de strict116: su ETA sigue
siendo extrapolacion, no medicion.

Accion para Satoshi mientras termina T2: cerrar C01/C02 con los controles reales
y negativos anteriores, sin duplicar GPU ni lanzar una nueva ronda. Al cierre,
verificar los ocho recibos y aplicar la regla congelada interna (no2024), luego
emitir T3 con su decision identificada. No cambiar ahora la regla estadistica,
las listas o la regularizacion para favorecer un resultado observado.
