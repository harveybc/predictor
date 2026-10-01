# Selección progresiva de características y representaciones temporales

Propuesta de Harvey, revisión de Musashi del 30 de septiembre de 2026.
Versión 3: enfoque aprobado por el propietario e incorporado al plan maestro.
Plan de implementación, no declaración de pruebas realizadas. Estado metodológico:
S3_ACCEPTANCE_TEST_DESIGN; las especificaciones de componentes y pruebas unitarias
se completan por incremento antes de implementar cada componente nuevo.
Órdenes: [continuación paralela](../../handoffs/SATOSHI_PROGRESSIVE_SELECTION_AND_MODULAR_CONTINUATION_2026_09_30.md).

Original conservado en `seleccion_caracteristicas_musashi.original.md`.
SHA256 original: `a73e397aac9f72c6a32c8db575c099c26a2bcc8dca8efa0e979db1b26e0e605f`.

## 1. Objetivo y criterios

Seleccionar entradas y extractores que aporten información a los targets cortos
y largos de la estrategia heurística y a una representación temporal para RL,
incluido SAC. No seleccionar las series más fáciles de reconstruir o de
pronosticarse a sí mismas.

La evidencia causal es un factor importante de selección. La calidad de
extracción, estabilidad y coste son evidencias adicionales. La capacidad de
generación condicionada es secundaria: prepara datos sintéticos futuros, pero
no compensa una mala utilidad predictiva ni excluye automáticamente una entrada
útil cuya distribución aún no sabemos generar bien.

Orden de decisión:

1. Disponibilidad temporal y validez del dato: requisito de admisión.
2. Aporte a los targets y al sistema de negocio, enriquecido con evidencia causal.
3. Calidad, preservación, estabilidad y coste del extractor aprendido.
4. Capacidad generativa condicionada, evaluada por separado.

La hipótesis de trabajo es que aprender una buena representación puede favorecer
la predicción. No se da por demostrada. Una reconstrucción deficiente puede
indicar capacidad/objetivo/entrenamiento inadecuados, no que la entrada sea ruido.
Reconstrucción perfecta puede conservar ruido y no aportar al target. Por ello
se comparan señal cruda, representación aleatoria y representación entrenada.

## 2. Lugar en el plan maestro

Secuencia progresiva: disponibilidad y perfil básico para todas las entradas
admisibles -> priorización reversible -> extracción y estudios causales en
paralelo -> perfiles ampliados de variantes prometedoras -> selección conjunta,
fusión y preentrenamiento del núcleo -> confirmación del finalista.
La sección 10 define incrementos, responsables, dependencias y pruebas.

No esperar a calcular todas las métricas sobre todas las descomposiciones de
todas las series. Perfilar ampliamente lo barato y asignar progresivamente el
cómputo caro. Reutilizar perfiles verificados con sus ausencias explícitas.
Cada transformación tiene identidad, soporte temporal, parámetros y linaje;
la señal original permanece como control.

| Frente | Responsabilidad propuesta |
| --- | --- |
| M03 / feature-eng | Inventario, disponibilidad, transformaciones, perfiles, grupos y candidatos |
| causal-inference | Estudios causales sobre datos históricos, supuestos, estimación y sensibilidad |
| M02 / feature-extractor | Encoders y objetivos de preentrenamiento; decoders/prior solo cuando la familia los requiera |
| M01 / predictor | Ensamblaje temporal, cabezales, sondas, ablaciones y compatibilidad |
| M04 / DOIN | Búsqueda acotada de selección, transformaciones, extractores y parámetros |
| agent-multi / gym-fx / LTS | SAC con representación temporal, estrategia heurística y evaluación semanal |
| data-gov / data-lake / warehouse | Acceso autorizado, artefactos, métricas y linaje |

Reutilizar los subplanes existentes de denoising y transformaciones. Esta
asignación no declara motores ya implementados. El README de causal-inference
inspeccionado describe un proveedor ATE separado y código experimental heredado;
no se instala ciegamente su paquete antiguo ni se presume una cadena completa.

La prioridad de datos es alta, después de definir el negocio. El mantenimiento
y reentrenamiento semanal del candidato vigente en LTS avanza en paralelo.
La síntesis y los contrafactuales no son una barrera para usar datos reales ni
para continuar los experimentos independientes.

## 3. Targets y separación temporal

La receta financiera candidata considera decisiones horarias, cuatro años de
TRAIN, un año de validación y uno de test cuando haya cobertura. No impone esa
frecuencia o partición a Electricity/Weather/Traffic: la réplica de literatura
conserva exactamente la receta publicada.

| Target | Definición candidata | Uso |
| --- | --- | --- |
| Y_s(t,h), h=1..6 h | Retorno acumulado del activo entre t y t+h | Cabeza corta |
| Y_l(t,h), h=24,48,...,144 h | Retorno acumulado a cada horizonte largo | Cabeza larga |
| Y_b(t) | Primera barrera TP/SL/timeout bajo reglas explícitas | Diagnóstico de cierre anticipado |
| J_policy | Objetivo del entorno con costes y restricciones | Política RL |

Son horizontes candidatos del negocio, no recuperación de una corrida histórica.
La construcción distingue horas transcurridas de filas. Si la estrategia
consume precios/trayectorias, la conversión desde retornos usa el precio conocido
en t y se verifica contra el plugin real. No cambiar su interfaz silenciosamente.
La escala del target, MAE/MSE normalizados y naive de persistencia se declaran
por horizonte sobre idénticas filas. Volatilidad usada para normalizar: solo la
disponible en t, con parámetros ajustados en el fold.

Y_b usa TP/SL de reglas fijas o pronósticos out-of-fold, nunca la predicción
ideal para seleccionar entradas de producción. Versionar ambigüedad intrabar,
vencimiento y censura. Evaluar también cerrar frente a mantener, neto de costes.

Selección, generadores y ajuste de umbrales se entrenan en folds cronológicos
internos de TRAIN. La validación externa puede servir al ajuste final declarado,
no a seleccionar variables. En evaluación semanal, early stopping y actualización
solo usan información realizada antes de la semana puntuada. No usar el año
completo para elegir retrospectivamente los checkpoints de sus primeras semanas.
Declarar si el test es estático o adaptativo con incorporación de etiquetas de
semanas ya pasadas; no cambiar la política viendo resultados de test.

La purga se deriva de soportes de ventanas, transformaciones, targets y tenencia,
no de un embargo fijo de 168 h. Y_l(t-1,144) aún contiene futuro respecto a t:
las sondas solo condicionan en historia observable y etiquetas ya realizadas.

## 4. Preparación y candidatos

### 4.1 Disponibilidad y perfil básico

- As-of join por disponibilidad, manteniendo separados event_time, published_at,
  received_at y vintage. Un retraso sobre datos revisados no recupera first release.
- Macro/calendario: conservar consenso previo, dato inicial, revisiones y sus
  tiempos; no fundir sorpresa de publicación con sorpresa disponible al sistema.
- Propagación válida de una serie diaria no equivale a 23 horas de datos perdidos.
  Medir cobertura nativa, cobertura utilizable, antigüedad y caducidad por fuente.
- Perfiles TRAIN-only: constantes, faltantes, escala/colas, volatilidad, tendencia,
  ACF, espectro, estacionariedad y estacionalidad. ADF/KPSS proponen variantes,
  no dictan qué encoder gana.
- Retornos, diferencias, log1p cuando sea válido, normalización y winsorización
  son variantes registradas; no borrar automáticamente extremos de interés.
- Pruebas de perturbación futura, fronteras, latencia y canarios. Correlación
  cruzada es alarma, no certificado de ausencia de fuga. Medir tasas de falsas
  selecciones en controles repetidos, no exigir cero falsos positivos siempre.

Indisponibilidad, fuga demostrada o datos inválidos excluyen técnicamente.
Cobertura parcial se reporta por fold; no inventar historia ni usar un único
umbral de cobertura para todas las frecuencias.

### 4.2 Redundancia y transformaciones

Detectar duplicados por fuente/subyacente y calidad point-in-time. Dependencia
lineal/no lineal, rezagos y clustering generan grupos candidatos, no descartes
universales. Mantener controles de todas las entradas admisibles, representantes
y reincorporación. Un par altamente correlacionado puede diferir por cabeza,
por régimen o por información conjunta.

Perfilar más profundamente las transformaciones priorizadas. Comprobar su
causalidad de emisión: ajustar solo en TRAIN no vuelve operativo un filtro
centrado o una descomposición global que utiliza observaciones futuras.
Medir el coste de pares/rezagos y RDC; no asumir que la fase es trivial.

## 5. Los tres peldaños causales con nuestros datos históricos

Los datos fijos NO limitan necesariamente el análisis al primer peldaño.
Intervenciones pasadas, variación de tratamientos y mecanismos aprendidos pueden
permitir inferencia interventional y contrafactual sin actuar sobre el mercado
ni construir un simulador de trading. La identificación depende de la pregunta,
del soporte y de supuestos causales, no de que nosotros ejecutemos la acción [1,2].

Buscar episodios donde ocurrió una acción es parte del método. Comparar sus
resultados sin controlar por qué ocurrió esa acción sigue siendo asociación.
Y otra fecha histórica no es el resultado alternativo observado del mismo
episodio: puede servir de control, pero ese resultado alternativo se estima.

### 5.1 Datos y unidad de análisis

Primer estudio: eventos económicos y respuesta del activo financiero que
alimenta las cabezas corta/larga. Usar recursos ya contratados/disponibles:
FXMacroData para calendario/actual/consenso cuando esos campos existan en el
snapshot, precios FX del lago para EURUSD, y Alpaca/Yahoo Finance para activos
y covariables que realmente cubran. No sustituir FX por otro activo por conveniencia.
La suscripción no demuestra cobertura, vintages ni disponibilidad de cada campo:
construir un manifiesto concreto antes del estudio, sin inventar datos faltantes.

Una fila es un episodio/evento identificado, no cada hora rellenada con el mismo
dato. Campos requeridos: tipo/país/evento, tiempos de publicación y recepción,
consenso previo con vintage, actual inicial, revisiones separadas, covariables
pre-evento, vector de eventos vecinos y precios con reloj/resolución conocidos.
No inferir si una noticia fue una sorpresa positiva mirando cómo se movió el precio.

Definir A como sorpresa del dato publicado: (actual_inicial - consenso_previo)
dividida por una escala histórica ajustada dentro de TRAIN. A=0 significa
"publicación igual al consenso", NO "no hubo publicación". Un cambio de tipos
es otra variable de tratamiento; no confundir la acción económica con su sorpresa.
Si solo hay sorpresa disponible al recibir, estimar ese objeto y etiquetarlo,
sin atribuirle automáticamente el impacto inicial del mercado.

Resultados Y_h: retornos o barreras a los horizontes declarados; covariables W:
estado de precios/volatilidad/régimen y expectativas disponibles ANTES del evento.
Los 144 h de resultado solapados no son episodios independientes. Remuestrear
por bloques/episodios y respetar su soporte. En datos horarios, comenzar en la
primera barra completa posterior estima el movimiento posterior y omite el
impacto inicial; usar precios de mayor resolución solo si existen y son aptos.

### 5.2 Peldaño 1: asociación y relevancia predictiva

Pregunta: "Con esta información disponible, ¿qué cambia en la distribución de
Y_s, Y_l o Y_b?" Estimar P(Y_h | X_historia, calendario, W) en folds internos.

1. Probar cada candidata y grupos contra los targets del activo, no contra su
   propia continuación. Comparar modelos con y sin candidata bajo presupuesto
   y población pareados.
2. Medir dependencia condicional/transferencia de información, rezagos y estabilidad
   por año/régimen. Las covariables condicionantes deben estar observadas en t.
3. GCMI es candidato de cribado; confirmar casos relevantes y controles no lineales
   con métodos adecuados. PCMCI+/LPCMCI ayudan a proponer estructuras bajo sus
   supuestos, no convierten toda arista en causalidad probada [3].
4. Registrar estimador, tamaño efectivo, p/q y familia de multiplicidad; calibrar
   el nulo temporal. Con 200 permutaciones, (b+1)/(B+1) no baja de 1/201, lo que
   puede ser insuficiente para miles de contrastes. No usar p=0 ni corregir solo
   los resultados favorables. La información mutua no tiene signo; evaluar
   estabilidad de dirección con un efecto firmado aparte.

Salida para selección: evidencia predictiva por cabeza y régimen, coste y
prioridad para extracción. Una señal específica de un régimen no se descarta
por no ser significativa en todos los demás.

### 5.3 Peldaño 2: efecto de intervenciones observadas históricamente

Pregunta: "¿Cómo cambiaría la respuesta si la sorpresa publicada fuera a en
lugar de a0, para una población de episodios comparable?"

1. Fijar el estimando: E[Y_h | do(A=a)] - E[Y_h | do(A=a0)], o CATE por W.
   Elegir contraste dentro del soporte real de cada tipo de evento; no mezclar
   NFP, inflación y tipos como dosis intercambiables.
2. Construir un DAG temporal con conocimiento económico y contrastes empíricos.
   Especificar qué variables explican tanto A como Y; no controlar mediadores
   posteriores ni colisionadores para estimar el efecto total.
3. Buscar eventos históricos con valores a/a0 o dosis cercanas y contextos W
   comparables. Emparejar/ponderar por covariables PRE-evento, no por la reacción
   futura del precio. Comprobar balance, soporte común y tamaño efectivo.
4. Cuando W satisface el criterio de ajuste, estimar mediante g-computation:
   m_h(a,w)=E[Y_h | A=a,W=w]; efecto medio = promedio de
   m_h(a,W_e)-m_h(a0,W_e) sobre la población declarada. DoWhy identifica el
   estimando; estimadores apropiados de DoWhy/EconML permiten ajuste y efectos
   heterogéneos [2]. Matching/ponderación son alternativas, no pruebas de
   ausencia de confusión. Para dosis continuas usar un estimador de dosis o
   intervalos predeclarados, no fingir un tratamiento binario.
5. Si no puede justificarse ajuste por W, examinar un instrumento, discontinuidad
   o experimento natural concreto solo si existe y cumple sus supuestos. Una
   sorpresa impredecible o residualizada no establece exogeneidad por sí sola.
6. Validar con placebos pre-evento, controles negativos, balance/solapamiento,
   sensibilidad a confusión no medida, especificaciones y folds temporales.
   Estas pruebas pueden refutar supuestos, no demostrar todos ellos.

Para efectos compuestos, A es un vector de eventos en una ventana. Modelar
interacciones y orden; no sumar coeficientes aislados automáticamente. Si un
evento posterior depende de otro anterior, usar DAG secuencial y ajuste apropiado
al tratamiento variable en el tiempo, o limitar el estudio a episodios separados.
En el momento operativo t solo entran realizaciones ya disponibles; de eventos
futuros puede conocerse el calendario, no su sorpresa realizada.

Salida: curvas de respuesta por horizonte, heterogeneidad, incertidumbre,
soporte y sensibilidad. Sirve como factor de selección y candidata a feature
de impacto; cualquier feature derivada para entrenar se calcula out-of-fold.
No excluir toda entrada sin efecto identificado: conservar su evidencia
predictiva y el estado causal NOT_IDENTIFIED. Así el problema de una variable
no bloquea otras ni el negocio semanal.

### 5.4 Peldaño 3: contrafactual del mismo episodio

Pregunta: "En este episodio observado, ¿qué habría pasado si esa sorpresa
hubiera sido cero, manteniendo las circunstancias previas de ese episodio?"

Ruta concreta con históricos, sin exigir simulador de mercado:

1. Ajustar en TRAIN un modelo causal estructural temporal, por tipo de evento
   y contexto. Una primera especificación contrastable es
   Y_h = f_h(A,W) + U_h. Es un supuesto de ruido aditivo e invariancia del
   mecanismo, no una propiedad demostrada por el VAE. Para un sistema con
   mediadores, ajustar también sus mecanismos y relaciones temporales.
2. **Abducción:** para el episodio e ya observado, estimar
   u_e,h = y_e,h - f_h(a_e,w_e). En modelos no invertibles, inferir la
   distribución posterior de perturbaciones y reportar esa incertidumbre.
3. **Acción:** reemplazar el mecanismo de A por A=a0, por ejemplo sorpresa cero.
   Conservar W previo y las perturbaciones inferidas; recalcular los descendientes
   afectados. No mantener arbitrariamente fijos mediadores que debían cambiar.
4. **Predicción:** en el caso aditivo,
   y_cf,e,h = f_h(a0,w_e) + u_e,h. Para el sistema multivariado, propagar los
   mecanismos en su orden temporal. Comparar factual y contrafactual por
   horizonte y, si existe una trayectoria suficientemente detallada, por
   decisiones de la estrategia. Un único retorno terminal no determina TP/SL.

DoWhy-GCM ofrece una ruta de inferencia contrafactual sobre modelos estructurales
con evidencia observada [4]. La variación histórica ajusta los mecanismos;
no necesita que nosotros provoquemos el evento. El contrafactual individual no
se convierte por ello en observación verificable ni queda identificado solo
por acertar predicciones. Declarar DAG, ecuaciones, soporte, supuestos de ruido
y alternativas; si distintos modelos compatibles producen respuestas distintas,
mostrar sensibilidad o cotas, no un único número presentado como verdad.

Los episodios vecinos emparejados ayudan a estimar y contrastar el comportamiento
medio, pero no son la segunda historia real del episodio e. Tampoco muestrear
un ruido nuevo responde la misma pregunta: cambia las circunstancias individuales.

Este análisis retrospectivo puede usar y_e para abducción una vez sucedido;
eso NO autoriza introducir u_e inferido del futuro en una feature emitida antes
del evento. La versión operativa usa solo información disponible y distribuciones
predictivas de las perturbaciones, nunca el resultado futuro real.

Salida para selección: sensibilidad contrafactual por episodio/grupo y estabilidad
de atribuciones bajo supuestos. Es evidencia adicional, no puerta obligatoria.
Si se usa un CVAE para mecanismos/abducción, debe cumplir y evaluar el modelo
estructural declarado. Cambiar c en D(z,c) por sí solo produce escenarios, no
certifica que z sea ruido exógeno ni que el escenario sea causal.

### 5.5 Intervenir la red no es intervenir el mercado

Reemplazar/permutar latentes y medir cambios del predictor o política es un
experimento controlado sobre el modelo. Se conserva como prueba de dependencia
funcional y utilidad, separado de los efectos causales anteriores. Permutar una
rama puede crear combinaciones fuera de distribución; matching por calendario
no garantiza coherencia con las otras ramas. Contrastar con retirada y ajuste
del modelo bajo presupuestos pareados, por ramas y por grupos.

## 6. Extracción y generación: decisiones separadas

### 6.1 Familias candidatas, sin imponer VAE

Separar dos ejes: arquitectura del encoder (Conv1D/TCN, recurrente, Transformer)
y objetivo (reconstrucción/denoising, contrastivo, enmascarado, predicción latente
o supervisado). No llamar ganador a una combinación antes de compararla.
Reutilizar código oficial fijado y comprobar licencia, pesos y protocolo.

| Familia | Referencia inicial | Papel y comprobación necesaria |
| --- | --- | --- |
| AE/denoising temporal | Implementación modular existente; DAE [7] | Control reconstruible, no SOTA presumido |
| Contrastiva temporal | TS2Vec [9] | Referencia reproducible por instante; confirmar soporte temporal al integrarla |
| Parches/enmascarado | PatchTST auto-supervisado [10] | Reproducir receta nativa; adaptación a ramas es experimento distinto |
| Modelo fundacional | MOMENT [11] | Auditar corpus de preentrenamiento y contaminación antes de comparación |
| Predicción latente | CF-JEPA [12] | Exploratoria, preprint; comprobar reproducción y salida temporal, no asumir SOTA |
| Generativa condicionada | CVAE [5], variante VAE-GAN [6] | Secundaria, si aporta generación o utilidad bajo presupuesto |

Estas referencias son puntos de partida verificables, no ranking universal.
Antes de sellar el piloto, actualizar búsqueda bibliográfica y elegir la receta
reproducible más pertinente por tarea. Registrar candidatos considerados y
exclusiones por licencia, recursos o ausencia de implementación. La reproducción
exacta conserva métricas/targets/splits del autor; una adaptación modular conserva
su propia identidad y se compara en un contrato emparejado, no contra otra tabla.
No imponer reconstrucción, KL o decoder a familias que no los tengan: NO_APLICA
es distinto de un fallo. Un generador posterior puede consumir representaciones,
pero se evalúa por separado y no convierte al encoder en un modelo causal.

Solo para la variante CVAE, por característica/grupo: encoder q_phi(Z_i | X_i,c_available), prior
p_psi(Z_i | c_available) y decoder p_theta(X_i | Z_i,c_available). El objetivo
declarado combina reconstrucción/probabilidad, KL ponderada y, en la variante
GAN, términos adversariales condicionados. Beta, annealing, ventanas, capacidad
y corrupción son parámetros optimizables con early stopping y coste registrados.

Z_i conserva (batch, tiempo, canales); media/varianza solo cuando procedan.
La base mantiene la ventana completa hasta la fusión, no un vector d=8 colapsado.
Ventanas 48/168/720 son candidatos; cada ensamblaje debe tener rejilla de fusión
común y soportes explícitos, no igualar tamaños rellenando observaciones ficticias.
La receta por defecto preserva los 24 pasos en cada rama. Una representación por
parches necesita un adaptador temporal declarado y validado, o se compara en su
receta nativa fuera de ese ensamblaje; no ocultar pooling ni interpolación como
identidad temporal. Un modo de salida global no es una secuencia conservada.
Condiciones: hora, día, época del año, sesión/festivos; calendarios, huso y DST
versionados. Codificar calendario en Z no es fuga cuando está disponible.

Encoders -> fusión -> núcleo preentrenable -> cabezal de forecast o política.
Decoders/discriminadores son piezas de entrenamiento, diagnóstico y generación,
no necesariamente de inferencia. Mantener R0/R1/R2 en branches y núcleo,
identidades de donantes y upstream. Declarar media posterior o muestreo en
inferencia; comparar siempre el mismo modo, o identificar la variante.

### 6.2 Target como condición

| Contrato | Condición permitida | Uso |
| --- | --- | --- |
| OPERATIONAL | Solo información disponible en t | Encoder para pronósticos y SAC |
| SYNTHETIC_OFFLINE | Calendario y opcionalmente target/escenario de TRAIN | Generación o aumento offline |

El target futuro real no se entrega al encoder operacional, tampoco al evaluarlo.
Un teacher condicionado en etiquetas necesitaría un student desplegable sin
ellas y pruebas independientes; es una variante, no una capacidad ya existente.

Para p(X | c,Y) offline, usar solo TRAIN y pares coherentes con el target.
Si Y es retorno, tiene que concordar con la trayectoria de precio generada; no
pegar etiquetas arbitrarias a historias. Registrar distribución de escenarios
y ponderaciones. No condicionar generaciones de entrenamiento en test/validación.

### 6.3 Evaluaciones separadas

- Reconstrucción: recuperación del dato observado fuera del ajuste del extractor.
- Denoising: recuperación frente a corrupción declarada; calibrar con verdad
  limpia conocida cuando exista. En mercado no llamar señal limpia verdadera
  a la reconstrucción ni ruido puro al residuo [7].
- Síntesis: muestras nuevas del prior condicionado, no reconstrucciones de test.
  Medir fidelidad condicionada, dinámica, extremos y posible copia de TRAIN.

Generar cada característica independientemente, incluso con calendario común,
puede destruir dependencia entre variables/targets. Aprobar generación marginal
no aprueba un dataset conjunto. La extensión multivariada requiere dependencia
conjunta/latentes compartidos y pruebas de rezagos, eventos y coherencia X/Y.
TimeGAN sirve como referencia de evaluación temporal/predictiva, no como solución
automática a nuestra generación condicionada [8].

## 7. Métricas de selección y diagnóstico

Por feature/grupo, cabeza, horizonte, fold, semilla, transformación y extractor:

| Evidencia | Medida | Decisión |
| --- | --- | --- |
| Reconstrucción | MAE_z/MSE_z global y en eventos, relativos a constante TRAIN y calendario solo | Diagnóstico, no selección automática |
| Aprendizaje útil | Delta_probe = Loss(Y, sonda(Z_aleatorio,c)) - Loss(Y, sonda(Z_entrenado,c)) | Positivo: entrenamiento añadió información utilizable sobre el target del negocio |
| Preservación | Loss(Y, sonda(X_i,c)) frente a Loss(Y, sonda(Z_i,c)) | Detecta degradación introducida por el extractor |
| Aporte incremental | Loss(Y, modelo(S)) - Loss(Y, modelo(S+i)) | Utilidad dentro de un conjunto y presupuesto |
| Retirada | Loss(Y, modelo sin i) - Loss(Y, modelo completo) | Separar ablación fija y reentrenamiento |
| Evidencia causal | Asociación condicional, efecto identificado y sensibilidad contrafactual | Factor con supuestos y soporte explícitos |
| Latente | Dimensión efectiva, sensibilidad a X, estabilidad; KL/unidades activas cuando proceda | Colapso o ignorancia de la entrada, no umbral universal de utilidad |
| Generación | Wasserstein/ACF/PSD, colas, eventos, dependencia cruzada, TSTR/TRTR | Secundaria; validar lo conjunto aparte |
| Coste | Tiempo, updates, RAM/VRAM, latencia y actualización semanal | Comparabilidad y presupuesto |

Y en las sondas es Y_s/Y_l/Y_b, no el futuro de X_i. Usar sondas temporales
adecuadas y presupuesto comparable. Una sonda lineal es diagnóstico adicional,
no la única puerta. Encoder aleatorio y entrenado comparten arquitectura y
forma; crudo puede tener dimensión distinta, lo que se declara en vez de fingir
igual capacidad. Calendario solo se entrena por separado: fijar z a su media en
el decoder entrenado es una ablación, no el mejor baseline de calendario.

MAE/MSE/naive por horizonte para retornos/precios; log-loss/Brier/calibración y
baseline apropiado para barreras. Para RL, las sondas orientan, pero la utilidad
se verifica con política/episodios y su recompensa declarada, no forecasting MAE.

Conservar diferencias 1e-5/1e-6 sin redondearlas a cero. No descartarlas por ser
pequeñas; evaluar su resolución numérica y variación por semana/semilla. Estimar
incertidumbre temporal y coste de aumentar evidencia sin consultar el test.
No significación no equivale a equivalencia ni a inutilidad.

Si el AE falla: verificar datos/escala/máscaras, tiny-overfit de diagnóstico,
gradientes, updates, campo receptivo y parada; luego revisar posterior collapse,
KL, decoder que ignora Z, capacidad y ventana. Comparar AE determinista y CVAE
y otras familias bajo búsqueda acotada. Marcar NO_EXTRACTOR_FOUND_UNDER_BUDGET,
no "solo ruido". Si crudo aporta, probar otro extractor o ruta temporal de
identidad/skip declarada. Eventos dispersos se evalúan en eventos, no solo ceros.

No existe una métrica que certifique utilidad nula o indispensabilidad universal.
Reportar aporta en este contraste, sustituible en este conjunto, dependiente de
régimen o evidencia insuficiente. La ablación fija sola no establece necesidad:
el resto del modelo puede recuperar la función al reentrenar.

## 8. Experimentos y selección final

Comparar con búsqueda y semillas pareadas:

1. Todas las entradas admisibles.
2. Selección predictiva y redundancia.
3. La anterior enriquecida con evidencia causal.
4. La anterior más diagnóstico de extracción.
5. Preferencia generativa secundaria como variante explícita.

Mantener controles de grupos e interacciones y reincorporar candidatos excluidos
por el cribado individual. No imponer simultáneamente todas las puertas de
significancia, CKA y reconstrucción. Calibrar reglas en TRAIN, no pesos arbitrarios
como verdades. Estudiar si la calidad de extracción predice utilidad incremental.

Después: encoders seleccionados -> fusión -> preentrenamiento del núcleo -> R0/R1/R2.
AE del núcleo es el control inicial; otros objetivos requieren contraste separado.
Comparar estrategia heurística y SAC con nuestra arquitectura temporal modular,
no sustituirla por MLP plano ni DQN. Declarar si los pesos de representación son
compartidos/congelados o ajustados por tarea, y mantener información, periodos,
costes y riesgo comparables.

La ampliación con sintéticos queda PLANIFICADA: real solo, real+sintético y
sintético solo como control, con exposición/coste explicitados y evaluación
siempre sobre real retenido. Generador/filtros se ajustan en TRAIN. No declarar
rentabilidad por TSTR ni reutilizar datos de test para escoger el generador.

## 9. Registro, recursos y aceptación

Salidas por feature/grupo/cabeza: selection_status, reason_code, evidence_scope,
causal_evidence_level y supuestos, extractor_status y donante, fusion_eligible,
synthetic_eligible con alcance marginal/condicional/conjunto, y cf_eligible solo
con expediente estructural. NOT_EVALUATED y NOT_IDENTIFIED no son cero.

Acceso por las rutas DataGov existentes. Pesos, grafos, latentes y matrices en
el lago; métricas y referencias en warehouse. Grano: run/attempt, etapa, feature
o grupo, conjunto padre, cabeza/target/horizonte, lag, fold, semilla, transformación,
modelo/comparador, condición, métrica/versión, población y reducción. Guardar
incertidumbre, soporte y decisiones. Reintentos nuevos ligados al anterior;
no deduplicar solo por nombre. Respetar retención y presupuestar artefactos.

Piloto representativo de unas 20 entradas/grupos y episodios de distintos tipos;
no implica potencia estadística suficiente para todos los estudios causales.
Medir por etapa antes de ampliar. CPU: perfiles/estadística/preparación. GPU:
extractores y candidato modular. Preferir 5090 externa 32 GiB; alternativas
4090 Laptop 16 GiB, 5070 Ti Laptop 12 GiB y 4070 Laptop 8 GiB, con admisión
fresca de RAM/VRAM/temperatura y leases existentes. No crear otro lockfile.
Early stopping y checkpoint por etapa, latido y ETA medidos; ASHA/halving se
calibra para no eliminar sistemáticamente familias de aprendizaje inicial lento.

Las siguientes pruebas se diseñan antes de implementar; NO están ejecutadas:

| ID | Aceptación |
| --- | --- |
| FS01 | Perturbar futuro no modifica entradas emitidas en t ni selección del fold |
| FS02 | Sondas ligadas a Y_s/Y_l/Y_b, no self-forecast oculto |
| FS03 | Target futuro prohibido en interfaz operacional, permitido solo en offline declarado |
| FS04 | Latente temporal, rejilla común, reload y R0/R1/R2 comprobados |
| FS05 | Falla de AE conserva diagnóstico y control crudo; no etiqueta ruido automáticamente |
| FS06 | Reconstrucción buena sin utilidad no obtiene selección automática |
| FS07 | Grupos y mejoras pequeñas preservados, sin promoción por ruido numérico |
| FS08 | Muestras nuevas condicionadas y consistencia de pares X/Y |
| FS09 | Fidelidad marginal no certifica dependencia multivariada |
| FS10 | Filtrar eventos no certifica do(); identificar ajuste y comprobar soporte |
| FS11 | Contrafactual conserva perturbaciones inferidas, propaga descendientes y no usa futuro en vivo |
| FS12 | Inferencia causal con confusión/ausencia de soporte conserva estado no identificado |
| FS13 | Selección interna y purga por soportes reales, recompensa RL separada de MAE |
| FS14 | Heurística y SAC comparados con representación, costes y periodos declarados |

## 10. Subplan ejecutable por incrementos

Prioridad: negocio y operación semanal, después admisibilidad/calidad de datos.
Este subplan se inserta en perfiles/selección del subplan modular y reutiliza
P-PRE de denoising y P-TRN de transformaciones del plan maestro. No crea una
segunda cola de entrenamiento ni bloquea el candidato DOIN/LTS vigente.

### 10.1 Casos de uso y entregables

UC-FS1: incorporar una serie admisible, obtener perfil, prioridad por target y
trazabilidad. UC-FS2: comparar representaciones y conservar una entrada útil aunque
su AE falle. UC-FS3: estudiar episodios macro con asociación, efecto identificado
y contrafactual bajo supuestos. UC-FS4: exportar selección/grupos/donantes a DOIN,
forecast y SAC. UC-FS5: reanudar sin recalcular perfiles ni usar futuros vintages.
Vías alternativas: dato no disponible, efecto no identificado, encoder no
compatible o presupuesto agotado; cada una conserva razón y deja seguir al resto.

| Etapa | Responsable | Entregable verificable y dependencia |
| --- | --- | --- |
| PS0 Inventario y contratos | M03 + M06 | Reutilizar 3.455 filas perfiladas reportadas; conciliar denominador 15.256 filas/15.228 columnas distintas. Particiones TRAIN y disponibilidad antes de perfilar cada recurso. |
| PS1 Matriz básica | M03 / CPU | Todas las características admisibles: calidad, distribución, volatilidad, ACF seleccionada, estación/estacionalidad y coste. Estados por celda de métrica, cobertura y ajustes; no solo una marca perfilada. |
| PS2 Priorización reversible | M03 + causal | Redundancia, relevancia por target y contexto causal disponible. Lista de trabajo, razones, grupos y muestra exploratoria fuera del ranking; no descarte definitivo por p-valor o AE. |
| PS3-R Representaciones | M02 + M01 / GPU admitida | Piloto de familias de sección 6, controles crudo/aleatorio/entrenado y utilidad Y_s/Y_l/Y_b. Depende de PS0 y del ensamblaje compatible, no del cierre causal completo. |
| PS3-C Causalidad | causal-inference / CPU | Expedientes de sección 5 para episodios disponibles; DAG, ajuste, solapamiento, placebos y contrafactuales bajo SCM. Depende de PS0, no de PS3-R. |
| PS4 Perfil ampliado | M03 + M02 | Métricas costosas/transformaciones sobre candidatos priorizados y muestra de exploración. Identidad del soporte, fuente y coste; no producto cartesiano sin límite. |
| PS5 Selección conjunta | M01 + M04 | Adición/retirada con reajuste, sinergias, reincorporación, una rama por entrada frente a grupos; R0/R1/R2 de ramas y núcleo. Misma población/presupuesto. |
| PS6 Confirmación y uso | M04 + M05 + RL | Finalista congelado, evaluación reservada conforme a su contrato, adapter instalado y shadow/paper. SAC y heurística comparados por negocio; no promoción automática a dinero real. |
| PS7 Síntesis opcional | M02 + causal | Real frente a real+sintético; fidelidad y utilidad conjunta; no requisito para PS6. |

PS0/PS1 continúan ampliando cobertura mientras PS2/PS3/PS4 procesan lotes listos.
La unidad del lote es recurso/feature/grupo con contrato, no todo el inventario.
PS5 recibe resultados parciales elegibles sin esperar a todas las familias.

### 10.2 Economía de recursos sin exclusiones silenciosas

Primera tanda: unas 20 entradas/grupos estratificados por fuente, frecuencia,
calidad, régimen y relevancia, incluidos candidatos de baja prioridad y grupos
con posible sinergia. No equivale a potencia causal suficiente. Registrar la
probabilidad/regla de muestreo y reservar explícitamente exploración en cada tanda.
No limitar la matriz básica a esas 20 entradas: el inventario se sigue perfilando.

Comparar en el piloto tres rutas bajo el mismo techo medido de cómputo y datos:
priorización causal primero, representación primero e intercalación progresiva.
La causalidad profunda no se ejecuta para cada columna por obligación. Medir
utilidad predictiva por horizonte, coste, cobertura y candidatos recuperados.
La tasa de falsa exclusión solo es conocida en controles sintéticos con verdad
o en el subconjunto evaluado exhaustivamente; no afirmar conocerla en todo el
mercado. El coste de búsqueda se incluye, no solo el ajuste final ganador.

En un núcleo piloto pequeño se evalúan exhaustivamente las alternativas
predeclaradas; en el inventario completo se usa asignación progresiva auditada.
No prometer enumerar todos los subconjuntos. Early stopping en cada ajuste;
halving solo tras comprobar que no penaliza familias lentas sistemáticamente.
Congelar criterios, semillas, exposición y fracción exploratoria antes de medir.
Asignaciones nuevas se cotizan con pilotos; no inventar horas/GiB autorizados.

### 10.3 Contratos entre componentes

`FeatureProfile`: recurso/vintage, feature, transformación, fold TRAIN, soporte,
métrica/versión/parámetros/valor/estado. `SelectionCandidate`: targets, conjunto
padre, prioridad, razones y evidencia, elegibilidad y decisión reversible.
`CausalStudy`: unidad/episodio, tratamiento, outcome/horizonte, DAG/ajuste,
estimando, soporte y sensibilidad; los tres peldaños tienen estados separados.
`RepresentationTrial`: arquitectura, objetivo, corpus previo, upstream, rejilla,
checkpoint, control y utilidad. `SelectionRelease`: selección/grupos/donantes,
protocolo de prueba congelado y contrato del consumidor.
Son diseños de campos, no nuevos paquetes obligatorios: extender esquemas y
entry points existentes antes de introducir abstracciones. El almacén conserva
grano de sección 9; las matrices/pesos viven en lago por referencia verificada.

### 10.4 Pruebas adicionales y trazabilidad

Todas son PLANNED, no ejecutadas. FS01-FS14 mantienen sus criterios de sección 9.
Antes de implementar cada incremento, su responsable añade casos nominales,
frontera, rechazo, mutación y recuperación; congela prueba roja y después código.

| ID | Requisito y contraejemplo | Dueño / etapa |
| --- | --- | --- |
| FS15 | Cobertura por métrica: una columna con ACF pero sin estacionariedad no aparece completa; una falla es distinta de cero. | M03 / PS1 |
| FS16 | Prioridad reversible y exploración: un par sintético útil solo conjuntamente reaparece pese al ranking individual. | M03+M01 / PS2,PS5 |
| FS17 | Encoder sin decoder no falla reconstrucción; modo pooled no satisface contrato temporal; pesos de corpus ajeno no se etiquetan TRAIN-only. | M02 / PS3-R |
| FS18 | Presupuesto completo y tres órdenes de selección sobre mismos folds; supervivientes no reemplazan el denominador esperado. | M04 / PS2-PS5 |
| FS19 | Perfil reiniciado es reutilizable solo con mismos bytes/parámetros/fold; cambio de vintage no reutiliza una decisión vieja. | M03+M06 / PS0-PS4 |
| FS20 | Fin a fin: lago -> perfil/selección -> donante -> DOIN -> resultado en warehouse -> adapter shadow; cambio de rejilla o upstream rechaza antes de cargar. | M01+M04+M05 / PS5-PS6 |

Etapas S4-S8 se detallan por componente sin frenar código existente ni carriles
independientes. La aceptación científica exige mediciones reales además de los
tests de contrato. Integrar incrementos aprobados continuamente, no esperar a que
todos terminen. El reporte y su progreso separan implementación, cobertura de
datos, celdas terminadas y utilidad demostrada; nunca mezclarlos en un porcentaje.

## Referencias

Métodos de referencia, no afirmación de SOTA financiero ni beneficio demostrado.

[1] J. Pearl, "Causal inference in statistics: An overview," Statistics Surveys,
vol. 3, pp. 96-146, 2009. https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf

[2] PyWhy, "Effect Estimation Using specific Effect Estimators," DoWhy v0.12.
https://www.pywhy.org/dowhy/v0.12/user_guide/causal_tasks/estimating_causal_effects/effect_estimation_with_estimators.html

[3] J. Runge, "Discovering contemporaneous and lagged causal relations in
autocorrelated nonlinear time series datasets," Proc. UAI, PMLR 124, 2020.
https://proceedings.mlr.press/v124/runge20a.html

[4] PyWhy, "Computing Counterfactuals," DoWhy v0.9.1.
https://www.pywhy.org/dowhy/v0.9.1/user_guide/gcm_based_inference/answering_causal_questions/computing_counterfactuals.html

[5] K. Sohn, H. Lee and X. Yan, "Learning Structured Output Representation using
Deep Conditional Generative Models," Advances in Neural Information Processing
Systems 28, 2015.
https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html

[6] A. B. L. Larsen, S. K. Sønderby, H. Larochelle and O. Winther,
"Autoencoding beyond pixels using a learned similarity metric," Proc. ICML,
PMLR 48, pp. 1558-1566, 2016. https://proceedings.mlr.press/v48/larsen16.html

[7] P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio and P.-A. Manzagol,
"Stacked Denoising Autoencoders: Learning Useful Representations in a Deep
Network with a Local Denoising Criterion," JMLR, vol. 11, pp. 3371-3408, 2010.
https://www.jmlr.org/papers/v11/vincent10a.html

[8] J. Yoon, D. Jarrett and M. van der Schaar, "Time-series Generative Adversarial
Networks," Advances in Neural Information Processing Systems 32, 2019.
https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html

[9] Z. Yue et al., "TS2Vec: Towards Universal Representation of Time Series,"
AAAI, 2022. https://arxiv.org/abs/2106.10466
Código: https://github.com/zhihanyue/ts2vec

[10] Y. Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers," ICLR, 2023. Código oficial con ruta auto-supervisada:
https://github.com/yuqinie98/PatchTST

[11] M. A. Goswami et al., "MOMENT: A Family of Open Time-series Foundation
Models," ICML, 2024. https://arxiv.org/abs/2402.03885
Código: https://github.com/moment-timeseries-foundation-model/moment

[12] WDSLab, "CF-JEPA," preprint y código de autores, 2026; candidato exploratorio.
https://arxiv.org/abs/2606.07031 ; https://github.com/WDSLab/CF-JEPA
