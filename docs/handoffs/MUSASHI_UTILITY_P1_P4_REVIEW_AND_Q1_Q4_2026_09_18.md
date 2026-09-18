# Revision P1-P4 y orden Q1-Q4

## Resultado y alcance independiente

Retorno predictor `fb1833c4770bebac9a8fdfef0933fa0050a4a6c8`.
Musashi ejecuto en trading-stack:

```bash
python -m pytest -q tests/test_df_utility_dev_close.py tests/test_df_utility_dev_run.py tests/test_df_utility_next_design.py tests/test_df_utility_reverify.py
```

**17 passed en 21.08 s**. Reejecute close() sobre utildev-v1 y 12B en lectura:
cuatro comprobaciones true, cero propuestas, 18 DOES_NOT_ADVANCE y 18
INCONCLUSIVE_UNCALIBRATED. Esta relectura verifica archivos y consume los recibos
de contabilidad/contenido conservados; NO equivale a una consulta nueva al cubo
vivo. No he repetido la campana ni los 1244 tests declarados por Satoshi.

Conclusiones permitidas: ningun par seleccion/replica avanzo bajo este diseno.
No equivalencia, no inutilidad general, no diagnostico de que los datos esten mal.
Un delta aumentado cercano a cero tampoco demuestra por si solo que el control
de capacidad 'hizo su trabajo': iguala dimensiones, no toda la historia disponible
ni la informacion accesible al modelo. Conservar la lectura acotada.

## Dos hallazgos del cierre

1. close() acepta un diseno vacio/no validado y devuelve las cuatro comprobaciones
   true: familias=[], mapa={}, operadores=[], hipotesis={}. No consulta la poblacion
   registrada para establecer que faltan las seis familias reales.
2. En un fixture con pares ADVANCES, el reproductor fuerza all_verified=False en
   el verificador de familia: close() devuelve files_verified=False y AUN emite
   doce PROPOSED_FOR_REVIEW. La CLI puede salir con error, pero la API y el artefacto
   siguen portando propuestas que un consumidor puede usar.

El segundo caso usa un verificador sustituido para aislar la politica del cierre;
no afirma que el verificador real haya producido esos avances. Ninguno demuestra
alteracion de utildev-v1. Reproductor adjunto, sin escrituras en datos reales.

## Calibraciones repetidas

Reconteo independiente de los 36 calibration.json conservados: para CADA uno de
los seis pares operador/hipotesis, seis registros y un unico digest canonico de
per_sim. CPU total de esos registros: aproximadamente 5761.29 s, antes de sumar
contrastes/piloto/overheads. Se confirma trabajo repetido, no 36 calibraciones
independientes. Son seis lotes de 358 simulaciones, cada lote repetido seis veces.

No hay que volver a medir para arreglarlo. Un cache correctamente ligado puede
ahorrar trabajo futuro; CAMBIAR SEMILLAS NO AHORRA COMPUTO, solo cambia las
simulaciones y su dependencia. No debe confundirse replicacion de los datos de
seleccion con independencia de las calibraciones compartidas. El ahorro futuro
requiere medicion incluyendo verificacion/transferencia, no una promesa de 4900 s.

## Q1. Cierre ligado a la poblacion y propuestas condicionadas

Congelar primero ambos casos anteriores. El cierre debe validar 12B/sucesor
contra su identidad registrada y derivar poblacion exacta, miembros, pares y mapa
de replicas. Diseno vacio, familia omitida, duplicado, miembro no previsto o
identidad incoherente deben ser rechazo tipado, no all([])=true.

No emitir propuestas consumibles si falla una comprobacion obligatoria del
alcance correspondiente. Es valido mostrar candidatos diagnosticos separados
como NO_VERIFICADOS, nunca bajo PROPOSED_FOR_REVIEW. Elegir y documentar politica
de cierre total o parcial; para un par exigido, ambas familias y todo su contenido
deben estar verificados. No fabricar una aceptacion parcial por orden de ruta.

Los recibos de reconciliacion y contenido deben ligar la identidad de la campana,
los terminales y los archivos comparados; all_equal o missing_units=[] aislados
no prueban pertenencia. Exigir poblacion completa, no solo igualdad de lo presente.
Pruebas contra API y CLI, ademas de helpers. Reverificar utildev-v1 sin repetir
mediciones, preservar CLOSE anterior y publicar cierre sucesor/delta explicito.

## Q2. Cache de calibracion: primero equivalencia, despues ahorro

Disenar clave canonica de COMPUTACION cientifica: arnes/dependencias numericas,
generador y parametros, cinta de semillas, n, operador/spec/params, par y anchuras,
modelo/ventanas/bloques/filas, target/horizonte, margen, alpha efectivo, politica de
fallos, n_sims y confianza. Separar etiquetas administrativas de familia de esos
campos, sin permitir que una diferencia cientifica desaparezca de la clave.

Un consumidor nuevo conserva su propio binding de campana y referencia la
evidencia compartida; no hacer pasar una lectura de cache por simulaciones nuevas.
Recontar per_sim y cota con el verificador actual; registro incompleto, bytes
alterados o contrato diferente no concede hit. No compartir H_T con H_A, ni
operadores distintos. Datos de seleccion/replica permanecen disjuntos.

Pruebas antes de implementacion: equivalencia positiva entre dos familias que solo
cambian etiqueta; miss ante cada campo cientifico; cache ausente/corrupto; concurrencia
sin duplicacion; recuperacion e idempotencia; contabilidad con simulaciones unicas,
consumidores, lecturas y costos separados. Ensayo gobernado sintetico acotado con
el hijo real: miss y luego hit, cero simulaciones nuevas en hit, contenido igual.

No repetir los 36 contratos para demostrar ahorro. Usar un banco de ensayo pequeno
que NO conceda soporte estadistico y un replay de verificacion sobre los lotes
conservados. Mantener todos los registros historicos; no deduplicar el cubo a ciegas.

## Q3. Controles de utilidad conocida y mapa data-centric

Antes de otro barrido, comprobar que el instrumento distingue lo que pretende
medir. Disenar controles positivos con ventaja del par especifico justificada,
nulos del contraste, caso donde la transformacion pierde informacion, y control
de fuga futura que debe ser rechazado antes de scoring. No declarar positivo a
cualquier proceso autocorrelacionado sin demostrar ventaja diferencial de R.

Vincular requisitos, estimando, generador, perdida y criterio esperado; conservar
calibracion y control positivo como papeles distintos. Pruebas de potencia/error
con presupuesto predeclarado, sin afinar parametros hasta pasar. Por ahora solo
fixtures fabricados para validar el instrumento; no abrir nueva seleccion o reserva.

Actualizar el mapa del work plan: este ensayo cubre tres operadores (CUSUM, delta,
MAD), tres familias sinteticas, h=1 y ridge; NO toda la ingenieria de variables,
denoising por caracteristica, compresion, representaciones aprendidas o feature
selection del lago. Enumerar lo probado y lo pendiente enlazando evidencia vigente,
sin convertir los resultados negativos en licencia para mas busqueda arbitraria.
Preparar una propuesta acotada del siguiente experimento y sus controles, no
lanzarlo todavia. No modificar el documento doctoral sin pedido especifico.

## Q4. Cierre y limites

Autorizadas Q1-Q3 completas sin pedir permiso entre bloques. Limites existentes
por hijo; techo conjunto DOS horas CPU para ensayos de cache/controles de esta
orden, contabilizando fallos. Si no cabe una prueba de potencia amplia, entregar
su diseno y ejecutar las verificaciones acotadas; no ampliar presupuesto ni simular
evidencia. Distribuir solo si hay tareas independientes y memoria disponible.

Sin GPU, live, datos financieros, reservas, nueva campana D3 ni repeticion de
utildev-v1. Gobernanza y metricas por contenido en todos los resultados persistidos.
No reiniciar servicios ni tocar el indice/Metabase por esta orden. Disposicion del
sobre vacio sigue para la proxima ventana necesaria, sin reinicio exclusivo.

Entregar PRE/POST, cierre sucesor, evidencia de cache, controles, mapa de cobertura,
costos y siguiente diseno. Actualizar estado persistente y work plan, publicar
commits y sincronizar. Salida: `DEVELOPMENT_CLOSURE_VERIFIED_CALIBRATION_REUSE_TESTED`.
No hay nueva accion del owner dentro de este alcance.
