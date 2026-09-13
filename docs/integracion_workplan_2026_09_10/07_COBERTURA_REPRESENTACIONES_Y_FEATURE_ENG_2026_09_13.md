# Cobertura de representaciones e ingenieria de caracteristicas

**Fecha:** 2026-09-13  
**Estado:** `AUDIT_COMPLETE / FEATURE_ENGINEERING_LANE_REQUIRED`  
**Alcance:** `financial-data`, `preprocessor`, `feature-eng`,
`feature-extractor`, `predictor`, `agent-multi` y DOIN.

## 1. Conclusion

El programa cubre ampliamente el acondicionamiento de senal y las
representaciones aprendidas, pero **no cubre todavia la ingenieria de
caracteristicas como una etapa sistematica, causal y seleccionable**.
`feature-eng` contiene trabajo util, pero su salida actual no puede entrar al
nuevo banco de variables como evidencia revisada.

La correccion no consiste en agregar indicadores a una tabla. Se incorpora un
carril transversal `F0-F5`, en paralelo a `D3-D5` y antes de `I5`:

```text
D0 contratos -> D1 perfil crudo -> D2 muestreo/ruido/denoise
                                      |                 |
                                      v                 v
                              F0-F5 feature bank    D3-D5 senal
                                      \                 /
                                       -> I5 seleccion -> I6-I10
```

La union ocurre en I5. Una salida de procesamiento de senal puede ser padre de
una caracteristica solo despues de pasar su propia reja. El numero de columnas
no es una medida de progreso.

## 2. Cobertura por familia de representacion

| Familia | Cobertura actual | Disposicion |
|---|---|---|
| Calendario, sesion y eventos conocidos | Codigo disperso, sin contrato comun | Entra en F1 con tiempo de disponibilidad explicito |
| Desfases, diferencias y rendimientos | Uso historico, sin banco comun | Entra en F2; `lookahead=0` obligatorio |
| Ventanas moviles y estadisticos robustos | Uso historico y `pandas_ta` | Entra en F3; solo ventanas hacia atras y warm-up declarado |
| Indicadores tecnicos | Plugin grande con parametros por defecto | Se descompone en operadores parametrizados; ningun indicador tiene autoridad por popularidad |
| Relaciones entre variables | D4/STEP 09-10 planificado | Entra en F4 solo despues de disponibilidad temporal y particion train-only |
| Transformaciones de senal | D2-D5 y STEP 01-13 | Cubierta como protocolo; ejecucion y elegibilidad siguen por etapa |
| Codificacion posicional | Script aislado y modelos posteriores | Se prueba como opcion de modelo en I7, no como columna estatica universal |
| Representaciones secuenciales aprendidas | `feature-extractor`, propuesta doctoral e I7 | Cubierta en diseno; falta evidencia bajo el contrato nuevo |
| Agrupacion de canales y grafos dinamicos | Propuesta doctoral e I7 | Cubierta como representacion multivariada, no como feature escalar |
| Regimenes y detectores de eventos | Codigo historico y STEP 07 | Deben separar diagnostico, feature de entrada y target |
| Grafos de conocimiento temporales | No cubiertos y no requeridos por los bancos numericos actuales | Solo abrir con entidades y relaciones temporales reales, tarea y baseline de grafo propios |

Los grafos de conocimiento temporales no son una tercera forma obligatoria de
representar cualquier serie. Resuelven hechos relacionales con vigencia
temporal. Forzarlos sobre columnas numericas agregaria infraestructura sin una
pregunta identificada. Las relaciones entre canales si forman parte de I7.

## 3. Estado real de `feature-eng`

La inspeccion del repositorio encontro:

- `setup.py:11-18` registra `tech_indicator`, dos productores de etiquetas,
  `ssa` y `fft` en el mismo grupo de plugins;
- `README.md:201-212` reconoce que el pipeline general solo funciona con
  `tech_indicator` y que `ssa`/`fft` no implementan `process()`;
- `README.md:166-177` registra una suite historica sin una prueba verde;
- `app/data_processor.py:154` convierte errores numericos y los rellena con
  cero; `app/data_processor.py:186` alinea con `ffill` y luego `-1`;
- `app/plugins/tech_indicator.py:607-610` usa relleno hacia atras en medias y
  volatilidad moviles, por lo que una salida temprana puede depender de una
  observacion posterior;
- el plugin tecnico concentra mas de mil lineas y mezcla indicadores, fuentes
  auxiliares, alineacion y calendarios;
- `app/positional_encoding.py` existe como script aparte, sin contrato de
  consumidor ni comparacion experimental comun.

Esto no declara inutil el repositorio. Declara que sus resultados historicos
son candidatos, no variables elegibles. Se conserva la implementacion como
inventario y se construye una interfaz comun antes de volver a usarla.

## 4. Carril F0-F5

### F0. Contrato y separacion de roles

Cada operador declara:

- `operator_id`, version y digest de codigo;
- clase: calendario, desfase, ventana, cruce entre variables, indicador de
  dominio o codificacion para modelo;
- variables padre y sus identidades;
- formula, parametros, unidad de salida y tipo;
- `lookback`, warm-up, retardo efectivo y `lookahead=0`;
- regla de tiempo de evento y tiempo de disponibilidad;
- si tiene `fit`, particion y estado ajustado;
- politica de ausencias; cero, `-1` y rellenos no son valores neutros por
  defecto;
- costo batch e incremental, memoria y latencia;
- esquema y digest de salida.

Targets y etiquetas futuras usan otro namespace y otro proceso. Ningun modulo
que escanea el futuro para construir `y` puede registrarse como productor de
`x`.

### F1. Calendario, sesion y eventos

Banco minimo: hora, dia de semana, mes, ciclo seno/coseno, sesion de mercado y
festivos o eventos solo cuando su publicacion era conocida a la hora de
decision. Se compara codificacion ordinal, ciclica y ausencia del feature.

### F2. Memoria explicita

Banco minimo: desfases declarados, diferencias, rendimientos simples y
logaritmicos, cambios estacionales y tiempo desde ultimo evento. Los lags se
eligen dentro de desarrollo, no mirando confirmacion.

### F3. Ventanas causales

Banco minimo: media, mediana, dispersion robusta, cuantiles, extremos, conteos,
pendientes, autocorrelacion y medidas de forma sobre ventanas exclusivamente
hacia atras. Cada operador publica warm-up y soporte efectivo. No se permite
ventana centrada ni `bfill`.

### F4. Caracteristicas multivariadas

Correlaciones, spreads, razones, residuales y factores solo se ajustan con datos
disponibles y dentro de training. Una fuente mas lenta no se desplaza hacia el
pasado para aparentar simultaneidad. Las salidas de D3-D5 pueden entrar aqui
como padres solo con estado revisado.

### F5. Utilidad y licencia

Cada candidato se compara con:

1. sus padres crudos;
2. sus padres acondicionados, si aplica;
3. la nueva caracteristica;
4. un control de igual anchura o costo;
5. una version deliberadamente inutil que licencie la sensibilidad de la
   prueba.

Se mide utilidad incremental fuera de muestra, estabilidad entre origenes,
extremos, cambio de regimen, costo y tasa de fallo. El resultado puede ser
`ELIGIBLE`, `REGIME_LIMITED`, `DOES_NOT_ADVANCE` o `INCONCLUSIVE`. Solo los dos
primeros pueden llegar a I5, y el segundo conserva su restriccion de regimen.

## 5. Pruebas obligatorias

Toda implementacion nueva de feature debe pasar pruebas estructurales y de
comportamiento:

- **invariancia de prefijo:** para cada corte, transformar el prefijo produce
  exactamente las salidas del batch completo hasta ese corte;
- **perturbacion futura:** cambiar cualquier dato posterior no cambia una salida
  anterior;
- **fit train-only:** cambiar validacion o test deja estado y salida de training
  intactos;
- **paridad batch/incremental/restart:** misma secuencia, mismos bytes o una
  tolerancia numerica predeclarada cuando la igualdad exacta no sea posible;
- **frontera de disponibilidad:** una fuente no aparece antes de su
  `available_time`;
- **ausencias y warm-up:** no se fabrican ceros, sentinelas o rellenos sin
  mascara y semantica declaradas;
- **independencia del target:** mutar `y` no modifica ninguna feature;
- **identidad:** mutar bytes, formula, parametros, padres o particion cambia el
  digest;
- **controles de fuga:** versiones centradas, `bfill`, normalizacion global y
  desplazamientos futuros deben ser rechazadas o detectadas;
- **wavelet/tiempo-frecuencia:** se repiten las pruebas para cada modo de borde,
  nivel y longitud. Una coincidencia en una sola longitud no licencia el
  operador.

## 6. Relacion con la propuesta doctoral

La propuesta estudia representaciones aprendidas y agrupadas, no promete que la
ingenieria manual sea su contribucion. El carril F cumple tres funciones:

- da controles fuertes y comprensibles a H1-H3;
- evita atribuir a una red una mejora causada por una feature con fuga;
- permite medir si el extractor aprende algo adicional a lags, ventanas y
  relaciones causales bien construidas.

Los experimentos doctorales registraran que conjunto F se uso, su costo y su
estado. El selector de representaciones no podra modificarlo durante la prueba
confirmatoria.

## 7. Orden operativo

1. Terminar y revisar D2 vigente.
2. Ejecutar F0: inventario y contratos de `feature-eng`; reparar primero CLI y
   suite, sin puntuar features.
3. Implementar F1-F3 con la metodologia de diseno guiado por pruebas y datos.
4. Ejecutar F4 en paralelo con D3-D5 cuando existan padres elegibles.
5. Ejecutar F5 en banco sintetico, publico y financiero de desarrollo.
6. Unir manifiestos D y F en I5; conservar ramas crudas y controles.
7. Solo despues abrir predictor, representaciones, DOIN y RL bajo identidades
   nuevas.

El estado persistente de este trabajo vive en
`FEATURE_BANK_METHOD_STATE.json`. La metodologia vinculante esta en
`../metodologias/DISENO_GUIADO_POR_PRUEBAS_Y_DATOS.md`.
