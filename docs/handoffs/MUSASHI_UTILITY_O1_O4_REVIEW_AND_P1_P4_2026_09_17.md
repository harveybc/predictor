# Revision O1-O4 y continuacion P1-P4

## Alcance revisado

Predictor `d4025324151ce1f035fa10391082b18a51be14c0`. Musashi ejecuto:

```bash
python -m pytest -q tests/test_df_utility_harness.py tests/test_df_utility_reverify.py tests/test_df_utility_next_design.py tests/test_df_utility_run_order.py
```

**53 passed en 49.80 s**, trading-stack. No he ejecutado la nueva seleccion,
calibracion extensa ni reserva. No sustituyo los 1231 tests declarados por esos
53 focales. Tambien ejecute reverify() en lectura sobre utilpilot-v2: decision_delta
vacio y mismo digest de arnes en los commits congelado y reanudado. No he repetido
la conciliacion del cubo vivo. Los hallazgos afectan el NUEVO diseno 12A; no requieren
repetir el piloto ni las campanas mecanicas anteriores.

## Hallazgos

1. **H_A no tiene camino de calibracion equivalente.** `calibrate()` llama siempre
   a contrast con sus valores por defecto raw/transformed. Una sonda sobre esa
   llamada confirma ese par aun cuando el protocolo declara tambien raw_wide y
   augmented. El registro no identifica el par realmente calibrado. Incluir H_A
   en la familia de nombres no calibra raw_wide/augmented.
2. **Contrato interno no validado completamente.** En un fixture del constructor
   real cambie solo families[0].protocol.margin a 99 y recalcule el digest exterior:
   validate_design acepta. Solo coteja target/horizon en ese nivel, no todos los
   parametros heredados ni la identidad propia del protocolo. Es una omision de
   validacion, no una afirmacion de que el diseno real tenga margen cambiado.
3. **Mayor historia temporal no incluida en la purga.** Por inspeccion, raw_wide
   consume ocho lags cuando window=4, pero contrast sigue usando horizon+reach+window.
   Con h=1/reach=0 la ultima etiqueta de train puede llegar a start-4 mientras la
   primera ventana de validacion empieza en start-7. Esto contradice la frontera
   de no solapamiento que declara el protocolo. No basta igualar columnas.

La pregunta H_T dice 'al menos igual', pero la regla delta_lower > 0 prueba
superioridad, no no-inferioridad ni equivalencia. Corregir la pregunta a 'mejora'
sin cambiar margen o inferencia. Igual anchura controla numero de entradas,
no garantiza igual informacion, rango efectivo, historia o dificultad del modelo.

Reproductor: reproduce_utility_o_review_2026_09_17.py. Solo fixtures; la sonda de
calibracion observa la llamada sustituyendo el scorer, no produce una calibracion.

## P1. Calibrar exactamente cada hipotesis

Pruebas primero: H_A invoca raw_wide/augmented tanto en calibracion como en score;
H_T invoca raw/transformed; intercambiar registros de ambos pares rehusa. Ligar
par, objetivo, anchuras/ventanas, operador, protocolo y politica de filas a cada
registro y comprobarlos en consumo. Mismo callable para simulacion y experimento.
No compartir calibraciones entre hipotesis por conveniencia; un cache requiere
equivalencia exacta del contrato y procedencia comprobada.

Mantener generador, margen, confianza y plan predeclarados. Las 358 simulaciones
son POR contrato de calibracion aplicable, no una licencia para seis contrastes
heterogeneos. Recalcular el costo previsto y explicitar cuantos contratos distintos
se ejecutaran. Ninguna cota desfavorable autoriza a ampliar n hasta conseguir pase.

## P2. Un solo contrato experimental y replicacion explicita

Validar recursivamente todos los protocolos de familia, sus digests, campos
heredados, contraste/par/operador, lista completa de miembros, cardinalidad y alpha.
Pruebas que cambien cada umbral/modelo/ventana, dupliquen o retiren miembros,
intercambien H_T/H_A y mantengan el digest exterior correcto. No basta volver a
hashear. Re-derivar elegibilidad y longitud desde los recursos ligados, no de --n.

Mapear cada seleccion a su replica de MISMA familia generadora/regimen/variable:
bumps con bumps, sinusoid con sinusoid, steps con steps. Semillas distintas y
datos/procedencia disjuntos comprobados; no exigir que un efecto en bumps tambien
pase en sinusoid para llamarlo replicado, ni inferir independencia solo del nombre.
Publicar el alcance por variable: alpha/6 por familia no es control global de todos
los descubrimientos de la campana. No presentar ninguna propuesta como confirmada.

Derivar ventanas y purga del soporte realmente consumido por ambas ramas y las
etiquetas. Incluir ocho lags de raw_wide, retrasos, disponibilidad y huecos. Probar
por identidades/rangos de tiempo que train y validacion cumplen la frontera, no
solo comparar un numero de purga. Conservar comparacion pareada y sus denominadores.

Sellar un sucesor de 12A con estas correcciones ANTES de medir; preservar 386fe033...
como historia no ejecutada. No cambiar poblacion, h=1, ridge lambda=1, margen=0,
window=4, bloques=4 o alpha=.05 para acomodar los resultados previos.

## P3. Ejecucion de desarrollo condicionada, sin otra pausa de permiso

Una vez P1-P2 y sus controles negativos verdes, se autoriza ejecutar el diseno
sucesor acotado: las seis familias ya propuestas (tres seleccion semilla 12,
tres replicas semilla 13), tres operadores y dos hipotesis por familia, 36
contrastes de DESARROLLO. No abrir poblacion nueva, reserva o datos financieros.

Antes de cada trabajo: freeze, identidad de entrada y registro gobernado. Medir
un piloto de costo de la implementacion corregida; mantener limites existentes
por hijo y un techo adicional de CUATRO horas CPU agregadas para esta campana,
incluidos calibraciones, fallos y reintentos. Distribuir tareas por memoria
disponible con identidades propias; sin GPU y sin interferir con otros trabajos.
Si la proyeccion excede el techo, no lanzar la campana completa: entregar plan
factible y avanzar todo el resto. Si se agota en ejecucion, conservar incompletos.

La falta de calibracion favorable no detiene el registro descriptivo: deltas,
perdidas, intervalos exploratorios, cobertura y costo con INCONCLUSIVE_UNCALIBRATED.
No convertirlos en ADVANCES. Un bound favorable solo cubre el nulo/alcance medidos.
Replicacion segun el mapa explicito; no retocar hiperparametros entre seleccion
y replica. Publicacion de candidatos solo como PROPUESTOS PARA REVISION, nunca
PUBLICLY_ELIGIBLE ni licencia DOIN/financiera por esta orden.

## P4. Cierre reproducible

Verificar archivos -> padre -> contabilidad -> warehouse por contenido completo,
incluyendo outcomes y costos; salida ausente no es cero. Resume sin repetir
calibraciones validas y sin aceptar jobs distintos como el mismo intento.
Conservar controles, fallos y todas las generaciones.

Entregar tabla por hipotesis/familia/operador: perdida cruda de referencia,
transformada o aumentada, delta, incertidumbre, cobertura, costo, soporte de
calibracion y replica correspondiente. Explicar por que un negativo no prueba
equivalencia ni inutilidad general. Actualizar metodologia persistente, work plan,
retorno y commits; pruebas exactas y exclusiones, no solo conteo verde.

La disposicion pendiente del sobre vacio sigue aprobada para la proxima ventana
necesaria; no reiniciar solo por ella. D3 mecanico, piloto anterior y reservas
intactos. No hay nueva accion del owner dentro de este alcance.

Salida: `PER_VARIABLE_DEVELOPMENT_SELECTION_AND_REPLICATION_READY_FOR_REVIEW`,
o cierre parcial con limite factual. No detenerse tras cada bloque para preguntar.
