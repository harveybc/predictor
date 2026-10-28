# Resultado del grid fino de ventana SAC

Cierre verificado2026-10-21: ALL_TRIALS_CLOSED,4/4 ensayos al primer intento,
spec9e13d797, runtime6916e49a. Ventanas9/11/15/17, stride2, seed7,
100000 transiciones fijas. TRAIN2020-2023, VALIDATION2024, mismas15 entradas,
negocio/costes/arquitectura; sin reajuste ni early stopping frecuente.

Retornos netos de los periodos completos; no comparar directamente el retorno
acumulado de cuatro anos TRAIN con el de un ano VALIDATION para medir brecha.

| Observaciones | TRAIN retorno% | VALIDATION retorno% | V DD% | V ordenes | V ganadoras% | V PF | V Sharpe diario no anualizado |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | -48.7107 | -31.7603 | 32.8962 | 199 | 30.1508 | 0.6572 | -0.1541 |
| 11 | -42.5564 | -34.4700 | 34.4700 | 204 | 28.4314 | 0.3799 | -0.2917 |
| 13 control | -14.5477 | -7.2783 | 10.7319 | 203 | 43.3498 | 0.8326 | -0.0610 |
| 15 | -26.7079 | -26.5253 | 27.8740 | 201 | 32.3383 | 0.5775 | -0.1766 |
| 17 | -30.2852 | -19.7006 | 21.0953 | 197 | 29.4416 | 0.6775 | -0.1504 |

La comparacion de seleccion usa el sufijo temporal comun congelado35, con
24827 filas TRAIN y6186 VALIDATION e igualdad de hashes de fechas. En este
lote las equities de ancla son10000, por lo que los retornos coinciden con los
nativos. Historias/posiciones anteriores no se reinician. Brecha de crecimiento
diario TRAIN-VALIDATION, en puntos porcentuales por fecha NY observada:
9=0.069074;11=0.091233;13=0.011630;15=0.074071;17=0.041479.
Es una descripcion comparable, no atribucion unica a sobreajuste.

Decision: conservar13 como base de esta receta. No se activaron vecinos ni
confirmaciones porque ninguna candidata mejoro el control; esto sigue la regla
predeclarada y no requiere otra autorizacion. No es un veredicto sobre otros
subconjuntos, modelos ni sobre el programa de adaptacion posterior.

OLAP nativo sin pendientes;10/10 informes adicionales de soporte comun con
readback,0 discrepancias. Coleccion independiente completa:5 corridas,control
incluido. Resumenes/ledger/metricas/pesos/configuraciones retenidos por los
scripts del lote. La cola GPU esta cerrada, no hay fit activo de este estudio.
TEST no se leyo. ETA de este lote: finalizado.

Siguiente etapa planificada: seleccion incremental de entradas segun
[su diseno](RL_INCREMENTAL_FEATURE_SUBSETS_2026_10_21.md). Su driver/configs
todavia no estan implementados ni desplegados; no se afirma un relevo a ese
lote. Extractores/componentes siguen despues; reajuste periodico al final.
