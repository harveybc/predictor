# 09 · Contrato de benchmarks y aplicación RL

**Documento auxiliar de la propuesta** «Diseño y aprendizaje de representaciones
temporales modulares» (`docs/propuesta_doctoral_representaciones_temporales_modulares.tex`).
**Origen:** encargo `SOLICITUD_MUSASHI_FIGURA_ADICIONAL_RL_Y_COMPARABILIDAD_2026_09_08.md`
(solicitante: Harvey Demian Bastidas Caicedo; preparado por Takeshi).
**Fecha:** 8 de septiembre de 2026.
**Estado:** contrato vigente + inventario inicial, revisado tras
`AUDITORIA_D19A55E_FIGURA_Y_CAMBIOS_2026_09_08.md` (Takeshi). El inventario de §7 es
orientador: no declara datos descargados, reproducidos ni seleccionados
definitivamente.

La propuesta (§5.2 del PDF) contiene la versión condensada; este documento es la
referencia normativa extensa.

## 1. Contrato mínimo para E3 (integración y evaluación RL obligatorias)

- **Una tarea propia y un algoritmo principal:** identificar configuración y ruta
  ejecutable existentes. Usar la opción compatible ya operativa; no iniciar una
  competición entre todos los algoritmos RL.
- **Dos variantes controladas:** representación actual de referencia y representación
  propuesta, manteniendo agente y entorno comparables. Las ramas y sus anchos pueden
  diferir cuando constituyen la intervención; declarar capacidad y costo.
- **Integración real:** ejecutar la representación dentro del agente, verificar formas,
  disponibilidad de observaciones, gradientes/actualizaciones según el régimen,
  guardado, carga e inferencia reproducible. No aceptar una exportación de CSV como
  prueba de ajuste conjunto si este se afirma.
- **Evaluación histórica reservada:** períodos, activos, frecuencia, proveedor y reglas
  de actualización fijados antes de puntuar. Conservar las reservas y restricciones ya
  vigentes del proyecto; este contrato no autoriza abrirlas ni reutilizarlas para
  diseñar la tesis.
- **Aprendizaje y estado:** declarar qué partes se congelan o ajustan, si actor y
  crítico tienen representaciones compartidas o distintas, y cómo entran posición y
  estado de cuenta. No transportar pesos entrenados en otro consumidor como si su
  beneficio estuviera demostrado.
- **Medición:** fijar una medida principal de utilidad de la aplicación y métricas de
  retorno neto, riesgo, exposición, actividad y costo. No escoger después el indicador
  en el que la variante gana.
- **Incertidumbre:** no tratar barras, episodios sobre el mismo período y semillas como
  mercados independientes. Reportar efectos pareados y variación entre
  entrenamientos/períodos con la dependencia temporal correspondiente.
- **Entrega:** modelo integrado, configuración reproducible, datos identificados o
  instrucciones de acceso autorizadas, resultados de ambas variantes y explicación de
  condiciones de fallo.

El conjunto público financiero admisible puede servir también a E3 si se ejecuta
mediante nuestra integración y cumple los contratos; no se crearán dos campañas casi
idénticas por tener dos rótulos. Si no se recupera un benchmark publicado comparable,
E3 sigue siendo obligatoria sobre la tarea propia y la comparación bibliográfica
financiera se informa como no realizada; no se reemplaza por cifras incomparables.

**Relación con la inferencia aprobada:** H1–H3, sus estimandos y el ajuste `K = F + 4`
conservan su ámbito. E3 no se añade a esa garantía: su análisis es aplicado,
predeclarado y separado; no rescata una hipótesis pública ni demuestra generalización
universal a RL. No se introduce una H4.

## 2. Procedimiento de selección de datasets

Orden: **pregunta y mecanismo → resultado publicado pertinente → tarea y protocolo →
artefactos → elegibilidad del conjunto.** Nunca por popularidad aislada ni después de
ver dónde gana nuestro modelo.

1. Identificar trabajos que estudien representaciones temporales, agrupación, campos
   receptivos o agentes financieros pertinentes.
2. Localizar la tabla o figura con resultados y su protocolo ejecutable: artículo,
   suplemento, código y configuración.
3. Verificar pertinencia del conjunto: variables observadas conjuntamente, historia
   suficiente, grupos posibles, escala temporal útil, acceso y licencia.
4. Identificar exactamente datos y particiones. Registrar copia o manifiesto con
   hashes cuando sea posible.
5. Comprobar compatibilidad con desarrollo por familias y prueba reservada. Las
   variantes de un mismo origen no se reparten como familias independientes.
6. Reproducir al menos una referencia antes de atribuir diferencias a nuestro método.
   Fijar la tolerancia de reproducción con precisión numérica y variación entre
   ejecuciones, no después de observar nuestro resultado.
7. Congelar la selección antes de ejecutar la comparación reservada. La inspección de
   pertinencia de perfiles usa solo entrenamiento y metadatos permitidos.

No se fija ahora una cantidad de familias «suficiente» para la inferencia: la
demostrarán el censo y el piloto.

## 3. Tres afirmaciones que no deben mezclarse

| Tipo de comparación | Qué se ejecuta | Qué se puede afirmar |
|---|---|---|
| Reproducción del resultado publicado | Referencia del artículo con su protocolo y artefactos recuperados | Si nuestro entorno reproduce su resultado dentro de la variabilidad documentada. |
| Comparación controlada bajo protocolo común | Nuestro método y las referencias con los mismos datos, evaluación y restricciones declaradas | Qué método funciona mejor bajo ese protocolo; si modifica el original, no es automáticamente el mismo benchmark de la tabla publicada. |
| Aplicación a nuestro sistema | Variantes en nuestro simulador, activos, frecuencia y actualización definidos | Utilidad para la tarea propia. No permite comparar directamente contra cifras de otros mercados o simuladores. |

Son tipos de afirmación, no tres campañas obligatorias: las ejecuciones se reutilizan
cuando cumplen los mismos contratos y esa coincidencia se documenta. **Usar modelos
diferentes es válido: esa es la comparación.** Para aislar el efecto de nuestra
representación, el contraste interno mantiene el algoritmo RL y el resto del sistema;
para comparar sistemas completos pueden cambiar algoritmo y arquitectura, pero la
atribución es al sistema completo y se declaran diferencias de recursos e información.

## 4. Ficha de comparabilidad (campos obligatorios)

| Aspecto | Registro necesario | Ejemplo de discrepancia que impide equivalencia |
|---|---|---|
| Identidad de datos | Proveedor, archivo, versión, fecha de descarga, hash, columnas y unidades | Mismo ticker descargado años después, con revisiones diferentes. |
| Universo | Instrumentos concretos e inclusión/exclusión; composición histórica cuando corresponda | Integrantes actuales de un índice aplicados retrospectivamente al período publicado. |
| Tiempo | Fechas y horas exactas de inicio/fin, inclusividad, zona, calendario y frecuencia | Evaluar 2020–2022 frente a 2018–2020; convertir datos diarios a cuatro horas. |
| Barras y precios | Bid/ask/mid, precios ajustados o no, volumen, dividendos, splits y reglas de agregación | Rentabilidad sobre adjusted close comparada con un simulador sin ajustes. |
| Particiones | Train/validation/test, límites por timestamp/índice y períodos de calentamiento | Mismo archivo con un 70/10/20 distinto al corte temporal del artículo. |
| Disponibilidad | Momento de publicación y de uso de cada variable | Utilizar el cierre de una barra para actuar a su apertura. |
| Preprocesamiento | Imputación, escalado, filtros, descomposición y muestras usadas para ajustarlos | Escalado global frente a escalado solo en entrenamiento. |
| Información de entrada | Variables, ventanas y fuentes adicionales | Añadir calendario económico o fundamentales solo a nuestro modelo. |
| Entrenamiento | Inicialización, corpus previo, validación, parada, número de candidatos y recursos | Comparar un único entrenamiento publicado con el mejor entre cientos de búsquedas propias. |
| Actualización | Modelo estático o reentrenado; frecuencia, datos y actualización de normalizadores | Reentrenar semanalmente el propio y mantener fija la referencia. |
| Simulador RL | Estado, acciones, restricciones, recompensa, terminación y paso de tiempo | Comparar un portafolio long-only con operaciones apalancadas long/short. |
| Ejecución | Instante y precio de ejecución, latencia, fills parciales, liquidez y prioridad | Decidir al cierre y ejecutar al mismo cierre cuando no estaba disponible. |
| Contabilidad | Efectivo inicial, comisiones, spread, slippage, financiación, apalancamiento y cierre final | Retorno bruto del artículo frente a retorno neto propio, o posiciones finales valoradas distinto. |
| Métrica | Fórmula, escala, anualización, tasa libre de riesgo, agregación y reducción | Confundir retorno acumulado con anualizado o MSE normalizado con MSE en unidades originales. |
| Incertidumbre | Semillas, cantidad de corridas, emparejamiento y dispersión | Comparar nuestro mejor seed con la media de todas las corridas del artículo. |
| Código | Commit, configuración, dependencias y cambios a la referencia | Ejecutar la rama actual de un repositorio como si fuese la versión del paper. |

Un mismo nombre de dataset no demuestra identidad de datos. Un mismo Sharpe no
demuestra idéntica fórmula ni idéntica serie de retornos. El reward de entrenamiento y
las métricas financieras de evaluación se registran por separado. Si el método cambia
la ventana, las variables o la política de actualización, la comparación puede seguir
siendo válida como comparación de sistemas, pero no se atribuye solo a la arquitectura
de representación; cuando esa atribución sea central se incluye un control con
información y actualización equivalentes.

## 5. MASE y métricas publicadas

Se conserva MASE y su agregación para H1. Se añaden **las métricas exactas del
protocolo publicado** (por ejemplo MSE o MAE) con su escala y reducción especificadas;
no se sustituyen ni se mezclan sus márgenes. Si la política rodante propia difiere del
test estático publicado, las referencias se vuelven a ejecutar bajo la política propia
y el resultado se rotula como **adaptación**, no comparable directamente con la cifra
original. ETTh1, ETTh2, ETTm1 y ETTm2 no cuentan como cuatro familias independientes:
se agrupan por procedencia según el contrato. Los datos ya usados para desarrollar la
regla (incluida exposición por experimentos previos del proyecto) no se presentan como
completamente no vistos sin examinar esa exposición.

## 6. Discrepancias o artefactos faltantes

- Sin snapshot idéntico: consignar «identidad de datos no verificada», no «misma base».
- Si paper, suplemento y código discrepan: registrar la contradicción y la versión
  adoptada; no elegir la que perjudique más a la referencia.
- Si el protocolo original incorpora una fuga: no reproducirla como validación; se
  documenta y se ejecuta una comparación corregida para todos, limitando la relación
  con el número publicado.
- Sin código: una reimplementación es un comparador legítimo identificado como tal,
  nunca «reproducción exacta».
- Solo media publicada sin dispersión: no declarar significación contra ese escalar;
  usar repeticiones de una reproducción propia cuando existan.
- Sin benchmark financiero recuperable: E3 se mantiene; la comparación externa se
  informa como limitación.

## 7. Inventario inicial de candidatos

Fila mínima por candidato: paper y tabla → tarea → dataset y versión →
período/frecuencia → particiones → variables/objetivo o estado/acciones → métricas
exactas → código/configuración → acceso/licencia → exposición previa del proyecto →
diferencias pendientes → costo piloto → elegible o descartado y razón. «Por verificar»
es un valor legítimo; no se rellenan huecos con supuestos.

### 7.1. TFB (pronóstico público)

| Campo | Estado |
|---|---|
| Paper y tabla | TFB, PVLDB 2024 (<https://www.vldb.org/pvldb/vol17/p2363-hu.pdf>); Tabla 5 y tablas experimentales. |
| Tarea | Pronóstico multivariado de largo plazo sobre conjuntos ETT, Electricity, Weather, Traffic, entre otros. |
| Dataset y versión | Por verificar contra el repositorio oficial (<https://github.com/decisionintelligence/TFB>). |
| Período/frecuencia | Por verificar por conjunto. |
| Particiones | Declaradas en su código; por identificar exactamente y por cruzar con la política rodante propia (si difiere → adaptación). |
| Variables/objetivo | Multivariado; por conjunto. |
| Métricas exactas | MSE/MAE según protocolo TFB; escala y reducción por verificar en el código. |
| Código/configuración | Repositorio público; commit por congelar. |
| Acceso/licencia | Por verificar por conjunto. |
| Exposición previa | POR VERIFICAR (T02): la afirmación previa «ETT usado en experimentos previos del proyecto» carecía de registro identificado. Pregunta abierta a Musashi: aportar ruta, configuración o registro que la sustente, o declarar que no existe. Distinguir una prueba técnica de carga de datos de su uso para desarrollar decisiones del método; registrar qué se observó antes de decidir elegibilidad. No se declara independencia por defecto ni se excluyen familias por suposición. Las variantes ETT* se agrupan por procedencia. |
| Diferencias pendientes | Cruce con protocolos de DUET y comparadores ya elegidos; priorizar intersecciones con artefactos recuperables y familias distintas. |
| Costo piloto | Por medir. |
| Veredicto | CANDIDATO; nada se incorpora íntegramente por defecto. |

### 7.2. TradeMaster (RL financiero)

| Campo | Estado |
|---|---|
| Paper y tabla | NeurIPS 2023 D&B, §4.1 y Tabla 2: ocho algoritmos de RL para gestión de portafolio sobre DJ30, medias de cinco corridas (<https://proceedings.neurips.cc/paper_files/paper/2023/file/b8f6f7f2ba4137124ac976286eacb611-Paper-Datasets_and_Benchmarks.pdf>). |
| Tarea | Gestión de portafolio DJ30 (la Tabla 1 lista además FX y cripto, sin garantía de la misma tabla de resultados ni protocolo). |
| Dataset y versión | Por verificar; discrepancia CONOCIDA en el README: entrada «BTC» descrita como Foreign Exchange con enlace a FX; diferencias entre fichas del repo y la tabla del artículo (<https://github.com/TradeMaster-NTU/TradeMaster#dataset>). Verificar archivos y configuración, no copiar el README. |
| Período/frecuencia | Por verificar en suplemento/configuración antes de adoptar fechas. |
| Estado/acciones/contabilidad | Por verificar; no comparar un agente propio de un solo activo con la tabla DJ30 como si fuera el mismo problema. |
| Métricas exactas | Financieras de la Tabla 2; fórmulas por verificar. |
| Código/configuración | Repositorio público; commit por congelar. |
| Acceso/licencia | Por verificar. |
| Exposición previa | Ninguna conocida. |
| Diferencias pendientes | ¿Alguna tarea corresponde al entorno propio o admite un adaptador acotado? Además, T01 (registro obligatorio, abajo). |
| Costo piloto | Por medir. |
| Veredicto | PRIMER CANDIDATO DE INSPECCIÓN (resultados localizables + herramientas públicas). No es una orden de migrar la plataforma a TradeMaster; no se afirma acceso idéntico verificado a sus datos. |

**T01 — comprobación de causalidad registrada (auditoría 2026-09-08).** El artículo
describe en su p. 7 («Experimental Setup») normalización específica de cada
partición, incluida prueba:

> El artículo describe normalización específica de cada partición, incluida prueba.
> Verificar en la versión ejecutable si utiliza estadísticas del período completo. Si
> ocurre, la evaluación causal propia utilizará escalado ajustado con información
> disponible para todos los métodos y se identificará como adaptación del protocolo.
> Las cifras de esa adaptación no se equipararán directamente con la Tabla 2
> publicada. Registrar asimismo la configuración de imputación y su disponibilidad
> temporal.

No se descarta TradeMaster ni se afirma que todo su código tenga una fuga: la
observación procede del artículo y la implementación no fue ejecutada ni
inspeccionada en esta ronda. Tampoco se ejecutará un protocolo con información futura
para presentarlo como validación causal. [Fuente primaria, pp. 6–7](https://proceedings.neurips.cc/paper_files/paper/2023/file/b8f6f7f2ba4137124ac976286eacb611-Paper-Datasets_and_Benchmarks.pdf).

### 7.3. FinRL-Meta (RL financiero)

| Campo | Estado |
|---|---|
| Paper | NeurIPS 2022 D&B (<https://papers.neurips.cc/paper_files/paper/2022/file/0bf54b80686d2c4dc0808c2e98d430f7-Paper-Datasets_and_Benchmarks.pdf>); descripción ampliada de datos: <https://link.springer.com/article/10.1007/s10994-023-06511-w>. |
| Tarea | Por localizar UN experimento concreto con cortes, universo, contabilidad y algoritmo; la marca «FinRL» no define un benchmark. |
| Dataset y versión | Parte de la infraestructura obtiene datos dinámicamente de proveedores externos: el mismo script NO garantiza el mismo snapshot histórico. |
| Código/configuración | Repositorio público (<https://github.com/AI4Finance-Foundation/FinRL-Meta>). |
| Resto de campos | Por verificar. |
| Veredicto | SEGUNDO CANDIDATO. No incorporar simultáneamente todas las tareas de FinRL y TradeMaster: escoger por pertinencia y reproducibilidad dentro del presupuesto reservado. |

No se añaden dígitos manuscritos ni tareas sin estructura temporal relevante para
completar listas: la elegibilidad depende del mecanismo y de la comparación prevista.
TFB, Monash y GIFT-Eval siguen siendo fuentes candidatas sin obligación de ejecutar
todas sus tareas.

## 8. Tabla de cierre A01–A11 (encargo 2026-09-08)

| ID | Ubicación en el PDF | Cambio aplicado | Cierre |
|---|---|---|---|
| A01 | Resumen, párrafos 2–3 (p. 1) | Arquitectura común con consolidador/núcleo/módulos de salida, dos usos alternativos y entrega RL obligatoria; definición de campo receptivo junto a su primera mención. | Se entiende desde la primera página. |
| A02 | Figura 1 (p. 2), citada desde el resumen | Figura introductoria añadida y luego SIMPLIFICADA según la especificación cerrada de la auditoría de d19a55e: seis componentes de alto nivel (entradas, extractores 1/$B$ con puntos suspensivos sin caja, consolidador, núcleo temporal, cabezales de salida), cinco conexiones con etiquetas «Grupo 1»/«Grupo $B$» y una compartida «Secuencias de características»; sin D/I/A, distribución, nota de perfiles, decoder ni estado del agente (el estado sigue declarado en el protocolo RL); dibujada al ancho final SIN escalado, texto efectivo 9 pt; pie propuesto por la auditoría con remisión a la Figura 2. Figuras 2 y 3 conservadas byte-idénticas en el fuente. | Hay tres figuras; las previas intactas; la nueva explica los cinco tipos de componentes sin capas internas. |
| A03 | §1 (motivación) y párrafo tras la pregunta (p. 2) | Motivación aplicada = entradas de los agentes RL del sistema existente; párrafo de aplicación obligatoria separada; «una mejora de pronóstico no se interpretará por sí sola como mejora de decisión». | No se infiere rentabilidad desde MASE. |
| A04 | §3.1–3.2 (p. 4) | Objetivo general con integración y evaluación RL; objetivo específico 3 sin aplicación financiera genérica; objetivo 4 RL independiente. | Entrega verificable propia. |
| A05 | §2.1 (p. 3) y §4.2 (p. 6) | Frase de correspondencia de notación (consumidor de pronóstico H1–H3, interfaces conservadas para RL); consolidador + núcleo temporal = mezclador ya descrito, sin módulo duplicado; salidas política/valor sin obligar red compartida actor–crítico. | Sin contradicción R0/R1/R2 ni segundo núcleo. |
| A06 | Tabla 1 fila E3 y §4.3 (p. 8) | E3 obligatoria: fila reemplazada + párrafo metodológico completo (contraste con entorno/observaciones/recompensa/acciones/calendario/presupuesto constantes; verificación de ensamblaje y actualización; métricas sobre trayectoria contable común; comparación publicada cuando sea recuperable; análisis separado de H1–H3). | Algoritmo y tarea se elegirán con protocolo definido. |
| A07 | §5.2 nueva (pp. 9–10) + este documento | Regla de selección (pregunta→referencia→protocolo→artefactos→elegibilidad), tres tipos de afirmación, adaptación vs reproducción, identidad de datos; ficha extensa e inventario aquí. | Reproducción, comparación común y uso propio diferenciados. |
| A08 | §6.1 y Tabla 2 + recorte | Semestre 2: «verificación del flujo de gradientes del modelo compuesto y prueba inicial de su interfaz con el agente de RL» (F04: el código interno U08 se retiró del PDF y queda solo en el seguimiento interno); protocolo E3 fijado en semestre 3; inventario desde semestre 1; semestre 5 = integración final y evaluación E3 obligatorias; recorte sustituido: E3 conserva una tarea, un agente y el contraste. | E3 no se elimina en el recorte; presupuesto reservado desde el piloto; «U08» ausente del PDF. |
| A09 | §7 (p. 12) | Quinta contribución: integración evaluada en agente RL del sistema existente, resultado positivo/negativo/inconcluso con costos. | No promete resultado favorable ni reinventa el sistema. |
| A10 | Referencias [33]–[35] | Añadidas solo las discutidas: Stable-Baselines3 (JMLR 2021), TradeMaster (NeurIPS 2023, título verificado contra la primera página de la fuente: «A Holistic Quantitative Trading Platform Empowered by Reinforcement Learning» — F03 corregido), FinRL-Meta (NeurIPS 2022), citadas en §4.3; las notas editoriales se retiraron de las entradas y sus hechos viven en este contrato. | Cada referencia con función, título exacto y fuente verificable. |
| A11 | Todo el PDF | Numeración automática; compilación sin referencias indefinidas; PDF recompuesto de 14 páginas revisado visualmente en sus páginas afectadas y transiciones. Precisión exigida por F05: la fuente del CUERPO está intacta y, además, el texto EFECTIVO de la Figura 1 es de 9 pt (dibujada al ancho final, sin `resizebox`) — en d19a55e la figura rendía ~5,3–6,0 pt y partía la oración «Falta establecer si descriptores… / permiten restringir…» (F01/F02); ahora la p. 1 termina con el resumen completo, la p. 2 abre con Figura 1 + pie y después el encabezado «1. Problema y motivación», con la oración reunida en un solo párrafo (verificado en capa de texto y render). El párrafo de R0/R1/R2 sigue fluyendo continuo hacia su tabla. | Continuidad resumen→figura→introducción verificada en el PDF renderizado; ninguna oración atravesada por flotantes. |

**Desviaciones declaradas:**

1. El nombre auxiliar propuesto `07_CONTRATO_BENCHMARKS_Y_APLICACION_RL.md` estaba
   ocupado (`07_SOLICITUD_RELECTURA_TAKESHI_2026_09_07.md`); se usó el siguiente
   número libre: `09_…` (autorizado por el propio encargo, §2).
2. La especificación original de la Figura 1 (encargo 2026-09-08, §4) fue SUSTITUIDA
   por la especificación cerrada de `AUDITORIA_D19A55E_FIGURA_Y_CAMBIOS_2026_09_08.md`
   §3, que anula aquella para la vista introductoria: sin submódulos D/I/A, sin
   distribución, sin nota de perfiles, sin estado del agente (declarado en el
   protocolo RL, no en esta vista), pie nuevo. La colocación usa un flotante `[H]`
   local (la política general de figuras no cambió), con el salto de página entre
   bloques completos que la auditoría permite.
3. El PDF quedó en 14 páginas (12 en la base 7390fb3, 15 en d19a55e); no se redujo
   la tipografía en ninguna ronda.

## 9. Registro de la ronda F01–F05 / T01 / T02 (auditoría de d19a55e)

| Hallazgo | Acción | Estado |
|---|---|---|
| F01 (figura sobredetallada, ~5,3–6,0 pt) | Dibujo y pie sustituidos por la especificación cerrada §3 de la auditoría; sin `resizebox`; 9 pt efectivos; Figuras 2 y 3 intactas; ninguna cuarta figura. | CERRADO |
| F02 (oración partida por el flotante) | Flotante `[H]` tras el resumen: p. 1 = resumen completo; p. 2 = figura + pie + encabezado §1; oración «Falta establecer… permiten restringir…» reunida (verificado en render y capa de texto). | CERRADO |
| F03 (título TradeMaster) | «A Holistic Quantitative Trading Platform Empowered by Reinforcement Learning», cotejado con la primera página del PDF oficial de NeurIPS 2023. | CERRADO |
| F04 (código interno U08) | Celda del semestre 2 sustituida por la redacción de la auditoría; `grep U08` = 0 en fuente y PDF. | CERRADO |
| F05 (ruta literal + cierres amplios) | Enlace descriptivo «protocolo de comparabilidad y aplicación RL» con URL fijada al commit de este auxiliar; «búsqueda de §4»→«inventario de §7»; A02 y A11 reescritos tras revisar el PDF recompuesto. | CERRADO |
| T01 (causalidad TradeMaster) | Bloque registrado en §7.2 (Diferencias pendientes); sin descartar el candidato ni afirmar fuga no verificada. | REGISTRADO |
| T02 (exposición previa ETT) | §7.1 → «POR VERIFICAR», pregunta abierta a Musashi (ruta/configuración/registro o declaración de inexistencia); sin independencia por defecto ni exclusión por suposición. | ABIERTO — RESPUESTA DE MUSASHI PENDIENTE |
| Opcionales §5 | (2) notas editoriales retiradas de [33]–[35]; (1) §5.2 conservada tal cual; (3) sin describir opciones concretas de SB3, cita genérica mantenida. | APLICADO PARCIAL (permitido) |
