# Barrido de ventana SAC con muestreo cada2 barras

Orden vigente: ejecutar secuencialmente ventanas7 y25, conservando stride2 y
reutilizando el control autenticado de13. Mismas15 variables, normalizador,
arquitectura minima, costes, negocio y100000 transiciones. TRAIN2020-2023,
VALIDATION2024, TEST cerrado. Sin early stopping ni restauracion por metrica;
cuatro controles congelados a25k/50k/75k/100k. Una GPU pesada, preferentemente5090.

Contextos12/24/48 intervalos observados. No son horas transcurridas a traves de
un cierre. Se incluye el tick actual completado; no hay interpolacion, relleno
de fin de semana ni historia inventada. El entorno nativo calienta13/25/49
barras para ventanas7/13/25. El normalizador usa hasta256 previas, pero no
impone256 de calentamiento: se registra esta diferencia de soporte inicial,
sin afirmar origenes exactamente pareados ni cambiar el simulador a escondidas.

Filtros2 por caracteristica, nucleo4, kernel3, profundidad1 y cabezal8. La
entrada aplanada del cabezal varia con la ventana: actor/criticos870/1254,
1062/1638 y1446/2406 parametros, comprobados con el SAC real y recarga exacta.
No es una comparacion con igual numero total de parametros. Se conservan como
contrastes descriptivos de igual extension7x2 frente a13x1 y25x2 frente a13x4.

Dos ensayos semilla7. Solo si mejora VALIDATION sin empeorar drawdown ni brecha
diaria TRAIN-VALIDATION y con actividad, se confirman control/finalista con
semillas17/29. Maximo6 ensayos logicos; hasta2 intentos de raiz nueva por ensayo.
Publicacion y lectura de vuelta OLAP, pesos/configs/digests y copia independiente
automaticas. No DOIN, no cambios al carril de Satoshi y no cuenta real.

Revision CPU paralela de las trazas previas: costes registrados, acciones
propuestas frente a ejecutadas y recompensa/ledger si existen los campos.
Ausencias tipificadas, no ceros ficticios ni conclusiones causales a partir
solo de resumenes. Estado: pruebas de integracion; aun sin afirmar despliegue.

## Despliegue Verificado

Runtime0286775c, spec6b407f7e, contratoe031cac4. Ventana7/stride2 en5090:
1000 pasos en12.8648s al arranque; recibo posterior35000/100000. Actor870 y
criticos1254, admision GPU y recarga exacta verificadas.128 pruebas unitarias,
77 seleccionadas con integracion de ambas ventanas y51 en worker. Timers de
worker/publicacion/copia habilitados; sucesor25 y semillas condicionales automaticos.
Primer intento SIN FITS retirado por vencimiento del recibo GPU; recibo nuevo,
identidad nueva, ninguna medicion reclasificada ni cambio cientifico.

Revision CPU6 resumenes de los tres muestreos cerrados: sumas de PnL, recompensa
y coste de las trazas coinciden con los resumenes; maximo1 entrada diaria y
sin posicion al final. Control VALIDATION: coste registrado730.80USD, neto
-727.83USD; neto+coste=2.97USD es solo un proxy, no un contrafactual sin friccion
(slippage y trayectoria de tamanos permanecen). La columna de comision es
acumulada, no se suma por barra; financiacion separada no esta en esas trazas.
No es replay independiente de fills ni demostracion de una causa del fracaso.
