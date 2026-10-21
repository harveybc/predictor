# Afinamiento local y optimizacion por componente

Orden del propietario2026-10-21: continuar con un grid fino de ventana; despues
optimizar cada componente por separado y comprobar interacciones. El reajuste
semanal queda al final por coste, no desaparece del programa.

Precision del propietario: esta etapa elige la mejor BASE ESTATICA para probar
despues el reajuste periodico, por desempeno, riesgo y generalizacion. No exige
profit positivo para continuar. Se documentan todos los resultados, con
independencia del signo. El beneficio de la adaptacion se medira en su contraste;
no se usa como promesa ni como condicion que bloquee el afinamiento actual.

## Orden operativo

1. Ventanas9/11/15/17, stride2, frente al control13 autenticado. Si hay mejora
   bajo la regla congelada, probar los dos vecinos enteros del ganador. Luego
   confirmar finalista/control con semillas comunes17/29. Hasta10 fits nuevos,
   secuenciales en5090, dos intentos de raiz nueva como maximo por fit.
2. Contraste Dense por caracteristica frente a Conv1D por caracteristica,
   igualando o acotando capacidad y manteniendo datos, fechas, presupuesto,
   costes y negocio. El Dense historico con otra capacidad/monitor no prueba
   superioridad de familia. Usar lo que mejore experimentalmente, no por teoria.
3. Afinar ramas, fusion/bottleneck, nucleo y cabezales de actor/criticos por
   separado. Registrar el alcance exacto de cada cambio. Una mejora en nucleo
   no implica que aumentar ramas ayude, ni viceversa.
4. Comprobar interacciones entre las mejores ventanas y recetas; los mejores
   valores por coordenada no equivalen al optimo conjunto. No prometer optimo
   exacto a partir de una sola semilla o una rejilla local.
5. Diagnosticos CPU de costes, acciones, recompensa y aprendizaje en paralelo.
   Si aparece defecto real, corregir antes de interpretar su efecto economico.
6. Cadencia de reajuste al final: sin reajuste, trimestral, mensual y semanal,
   con prediccion anterior a incorporar las observaciones del periodo puntuado.

## Protocolo del primer lote

Mismas15 entradas, normalizacion causal, negocio SL/TP/volumen/maximo1 entrada
diaria/cierre y reapertura12h; costes asumidos intactos. TRAIN2020-2023 y
VALIDATION2024 completos, TEST cerrado. SAC nativo,100000 transiciones fijas,
cuatro controles a25k/50k/75k/100k; sin ES costoso ni seleccion de checkpoint.
Ramas2 filtros, nucleo4, kernel3, profundidad1 y heads8.

Se guardan metricas nativas completas y una comparacion adicional sobre el
mismo sufijo temporal despues de35 barras observadas de contexto. Se verifica
el hash de todas las fechas, filas y soporte, sin rellenar huecos. Se recalculan
crecimiento diario y DD con la equity anterior al tramo como ancla. No reinicia
cuentas ni posiciones: las historias anteriores pueden diferir. No se presenta
como un contrafactual con cuentas inicializadas identicamente en ese punto.
La capacidad del cabezal aplanado cambia con la ventana y se mide por modelo.

Seleccion: mayor crecimiento neto diario en VALIDATION,delta>1e-8, sin empeorar
DD ni brecha positiva TRAIN-VALIDATION, con actividad. Informar tambien retorno
nativo, Sharpe diario no anualizado, ordenes, ganadoras y PF de ambos periodos.
Mejorar una perdida no equivale a ganar dinero. El control no operar sigue0.

Publicacion OLAP/readback de metricas nativas y de sufijo; pesos, configuracion,
semillas y digests preservados, mas copia fisica independiente. No cambiar DOIN
ni el carril de Satoshi. Los sucesores posteriores al lote de ventana siguen
planificados, no se declaran ejecutados sin sus configuraciones y pruebas.

Implementacion: agent-multi `docs/work_plan/56_COMPONENT_OPTIMIZATION.md` y
`tools/temporal_capacity_sweep.py`, protocolo temporal_fine_window_sweep.v1.
Despliegue verificado: runtime6916e49a, spec9e13d797, contratoa7931e4e;
ventana9/stride2 en5090,1000/100000 en12.7458s.170 regresiones,4 integraciones
del motor real y48 pruebas CPU en worker pasan. Admisiones GPU16 updates y
recarga exacta, actor934/criticos1382. Timers de worker/publicacion/copia activos;
control con2/2 informes derivados readback y copia independiente verificada.
ETA provisional del lote inicial4h40, basada en~70min por corrida cerrada; si
se activan refinamiento y confirmaciones, hasta~11h40, sin contar reintentos.
