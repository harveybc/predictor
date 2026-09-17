# Revision M1-M6 y orden N1-N5

## Hechos comprobados por Musashi

Revision predictor `b6facb854e4714ddc2dd17ffc09ea7df07eeb4bf`.
Pruebas CPU en trading-stack:

```bash
python -m pytest -q tests/test_df_utility_harness.py tests/test_d3_matrix_verify.py
```

**54 passed en 8.32 s**. Los contraejemplos anteriores tienen regresiones con
la API nueva. No considero que romper el reproductor antiguo pruebe la correccion;
la evidencia pertinente son esas regresiones y el comportamiento actual.

### Hallazgo 1: el ensayo completo no reporto las perdidas

`run_isolated()` devuelve como `score` el contenido de result.json, un resumen
del proceso. El resultado cientifico esta en contrast.json. El reportero busca
delta_mean en el resumen y por eso no crea las metricas.

Comprobacion de solo lectura sobre los artefactos reales de utilreh-v4:

| operador | delta_mean en contrast.json | delta en result.json |
|---|---:|---|
| cusum_causal | -0.18881876086493116 | ausente |
| delta_run_length | 0.019637893298198927 | ausente |
| mad_extremes_trailing | -0.18871307727173697 | ausente |

Consulta independiente a governed_terminals.body_json de la contabilidad:
los tres COMPLETED de `utilreh-v4-utility-rehearsal` tienen `metrics: []` y
`artifacts: []`. El control RESOURCE_EXCEEDED tampoco lleva puntuacion, correcto
para ese caso. La afirmacion del retorno de que los deltas llegaron como metricas
es falsa al alcance de estos tres terminales. Los resultados locales existen;
no afirmo perdida de sus archivos ni he alterado contabilidad o cubo.

### Hallazgo 2: calibracion vacia o no finita concede decision

Reproductor sobre datos fabricados y API actual: un registro con `n_sims=0`,
generador inexistente y tasa 0.0 produce DOES_NOT_ADVANCE en vez de rechazar o
INCONCLUSIVE_UNCALIBRATED. Con tasa NaN sucede exactamente lo mismo. El limite
inferior del ejemplo es -0.2863933730683032; no afirmo un falso ADVANCES medido
en este ejemplo, sino que la compuerta acepta una calibracion inexistente.

El registro no liga operador, parametros, longitud, protocolo ni resultados de
simulacion. El entry point calibra solo delta_run_length a n=700 y aplica ese
registro a tres operadores sobre n=2400. Eso no demuestra cobertura del alcance.
`with_calibration()` ademas descarta scored/advances, haciendo imposible recontar
su tasa desde lo que el contraste consume. Una tasa observada de cero no es una
cota de error de cero. Estructura AR(1) por si sola tampoco demuestra que un
contraste concreto tenga efecto positivo: hay que definir el nulo del contraste.

### Hallazgo 3: gobernanza posterior al trabajo

Por inspeccion del entry point, submit_campaign y la conciliacion before_run
ocurren DESPUES del bucle de contrastes. El freeze local previo es util, pero no
equivale a registrar la campana antes de ejecutarla. Los started_at/finished_at
se generan al reportar, no en los intervalos reales de los hijos.

Reproductor de calibracion adjunto. No he lanzado reservas ni nuevos experimentos
del proyecto. La revision no exige repetir las campanas mecanicas D3.

## N1. Recuperacion trazable de los resultados ya medidos

Congelar primero el fallo result.json/contrast.json como regresion. Tras un hijo
COMPLETED, el padre debe leer el archivo de salida que el runner verifico, cotejar
sus bytes/digest y validar su esquema, identidad de contraste/protocolo y valores
finitos. El score devuelto debe ser el resultado, no el resumen del proceso.
Ausente, alterado o discordante: refusal tipada sin fabricar ceros ni puntuaciones.
Los intentos incompletos no publican resultados parciales.

Prueba integrada con hijo real: un delta no nulo conocido debe ser identico en
archivo, retorno del padre, terminal enviado, contabilidad y warehouse. Reintento
sin duplicado; prueba negativa borrando o cambiando la salida.

Recuperar las tres mediciones de utilreh-v4 desde los archivos conservados y
sus digests, sin entrenamiento nuevo. Publicar evidencia correctiva aditiva por
el mecanismo de sucesion existente, ligada a los terminales originales; no
reescribir historia, no fingir otra medicion y no promoverla cientificamente.
Preservar sus desenlaces como historicos no gobernantes mientras N2 esta abierto.
Corregir el retorno con fecha y referencia a la evidencia. La conciliacion final
debe comparar las metricas esperadas desde los archivos, no solo terminales vacios
contra terminales vacios. Revisar tambien el sobre que hoy usa 'outcome' y cero
como sustituto cuando el resumen carece de delta: dato ausente no es perdida cero.

## N2. Calibracion verificable y de alcance explicito

Tests rojos: cero simulaciones, NaN/inf/tasa negativa, generador desconocido,
denominador cero/incompleto, traslado a otro operador/protocolo/longitud/familia.
Todos deben impedir decision calibrada. Validar todos los dominios numericos del
protocolo, no solo la forma del diccionario.

Conservar resultados por simulacion y ligar el registro a codigo/generador,
semillas, modelo, operador, parametros, familia de contrastes, margen, bloques,
longitud y politica de faltantes/cobertura. Recontar intentos, validos, fallos y
avances; una simulacion fallida no desaparece silenciosamente del denominador.
El consumidor verifica ese alcance y la identidad del registro. No reutilizar
la calibracion de un operador para los otros sin una justificacion probada.

Predeclarar nulos de la diferencia de perdida que incluyan dependencia relevante,
controles positivos separados y precision de Monte Carlo. La regla debe usar
incertidumbre de la tasa (por ejemplo cota binomial predeclarada), no solo el
estimador puntual. Fijar el presupuesto y tamanos antes de correr; si no alcanza
precision, INCONCLUSIVE. No aumentar simulaciones o margen hasta conseguir pase.
No convertir automaticamente los avances AR(1) en verdaderos positivos solo por
ver estructura en el generador. Sin soporte: resultados descriptivos, ningun
ADVANCES ni DOES_NOT_ADVANCE presentado como decision calibrada.

## N3. Registro previo y costos reales de toda la cadena

Registrar la campana y su poblacion antes de los contrastes; aplicar before_run
antes de cada hijo. Separar calibracion preparatoria como campana gobernada con
su propio diseno cuando aun no existe el freeze final de utilidad. Las pruebas
unitarias pequenas quedan fuera; los resultados de calibracion persistidos, no.

Probar con el entry point real que una negativa de registro evita ejecutar hijos;
trazar el orden con callbacks observados, no comparando texto fuente. Registrar
inicio/fin reales, costos de calibracion, mecanica preparatoria y contrastes,
incluidos fallos. No sustituir desconocidos por cero. Aplicar limites al trabajo
preparatorio tambien y conservar recuperacion tras caida sin duplicar resultados.

## N4. Ensayo gobernado y piloto descriptivo acotado

Tras N1-N3 verdes, repetir el ensayo de aceptacion completo con la correccion:
operadores reales, contrastes, control lento, metricas verificadas hasta DuckDB.
Conservar utilreh-v4 y usar identidad nueva. Reconciliar contenido y costos.

Se autoriza despues un piloto CPU DESCRIPTIVO sobre hasta tres unidades sinteticas
de DESARROLLO ya expuestas, explicitamente separadas de toda reserva. Congelar
antes unidades, operadores elegibles, modelo, parametros, horizontes, ramas,
presupuestos y regla de comparabilidad. Mantener los techos existentes del runner;
si no caben, producir el desenlace limitado, no ampliar silenciosamente.
No depende de obtener una calibracion favorable: sin calibracion valida publicar
perdidas/deltas descriptivos y INCONCLUSIVE_UNCALIBRATED, nunca seleccion.

Ese piloto sirve para medir costo y comprobar flujo real, no para afirmar mejora
general ni elegir representaciones para trading. Sin datos financieros reales,
GPU, live, confirmacion publica o reserva. Distribuir CPU si hay tareas suficientes
y memoria disponible; no ocupar hosts solo para afirmar que se usaron tres.

## N5. Cierre y frontera

Ejecutar todos los bloques sin nuevas pausas de permiso. No reiniciar servicios
por estos cambios salvo necesidad demostrada de adopcion y procedimiento existente.
No repetir D3 mecanico: la correccion actual es de utilidad/reporte/calibracion.
Actualizar work plan, trazabilidad, retorno y cifras de metricas recuperadas.
Entregar PRE/POST, suites con entorno, digests leidos, commits publicados,
recibos correctivos, ensayo y piloto descriptivo con reconciliacion de contenido.

Salida: `UTILITY_RESULTS_RECOVERED_CALIBRATION_SCOPED_DESCRIPTIVE_PILOT_REVIEW`.
Metabase, indice y terminos siguen aparte. Ninguna accion nueva del owner es
necesaria para N1-N5; ningun bloqueo estadistico obliga a detener el trabajo
independiente de recuperacion o el piloto descriptivo autorizado.
