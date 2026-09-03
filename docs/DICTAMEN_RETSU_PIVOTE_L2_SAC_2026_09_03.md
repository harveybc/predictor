# Dictamen Retsu — pivote L2 / SAC

**Para:** Harvey (Maestro)  
**Fecha:** 2026-09-03  
**Modo:** adversario. No reescribí el PDF. No lancé GPU.  
**Insumos:** [`tesis_sac/00_FENOMENO.md`](tesis_sac/00_FENOMENO.md) · [`SATOSHI_TO_MUSASHI_PROPUESTA_TESIS_L2_SAC_2026_09_03.md`](SATOSHI_TO_MUSASHI_PROPUESTA_TESIS_L2_SAC_2026_09_03.md) · respuesta de Musashi (pegada por el Maestro) · cartas a los generales: [`RETSU_TO_SATOSHI_PIVOTE_L2_SAC_2026_09_03.md`](RETSU_TO_SATOSHI_PIVOTE_L2_SAC_2026_09_03.md) · [`RETSU_TO_MUSASHI_PIVOTE_L2_SAC_2026_09_03.md`](RETSU_TO_MUSASHI_PIVOTE_L2_SAC_2026_09_03.md)

---

## 0. Hanzo, para que no quede ruido

No me refería a Musashi.

En el work plan de **doin-domains** (doc 02 §2, aprobación 2026-08-25) el auditor del *registro de hallazgos del frente de dominios* se llama por defecto **General Hanzo**. Es un **rol de gobernanza** de ese frente: Satoshi no cierra hallazgos propios; dispone Hanzo (o el owner). No es el orquestador. No es Musashi. No es Satoshi. No es un agente al que yo deba asignar trabajo.

Ayer hablé como si Hanzo fuera un general en cola. Eso fue un error de tono: **no despacho a Musashi ni a Satoshi**. Si hay crítica, va en carta. El frente de dominios, para mí, sigue **sin orden abierta**.

---

## 1. Veredicto

**ACCEPT DIRECTION / REVISE BEFORE DRAFT.**

Musashi acertó el dictamen operativo. Satoshi acertó el acto: mató A, el híbrido, B y M *para este doctorado* con documentos, no con capricho. Eso es honestidad real. No es todavía una propuesta.

La página del fenómeno **enciende el objeto** y **quema el texto**. Si la lees como está, un jurado no necesita AutoRL: le basta Eimer (ICML 2023) para la palabra *folclor* y Dierkes et al. (2025, *Performance Prediction in Reinforcement Learning: The Bad and the Ugly*) para las sondas.

No congeles esos dos archivos como base del PDF. Congélalos como **acta interna del pivote**. El PDF nace después de las cuatro piezas de Musashi, más una quinta mía.

---

## 2. Qué sostengo de Satoshi

La traza de eliminación es la mejor página que ha escrito en este ciclo. A describe un sistema que no operas. El híbrido era dos tesis. B era AutoML ajeno. M, sin token, no es emisión y el controlador de \(r\) es economía. Decidir que el doctorado se monta sobre el trabajo que **ya haces** (L2 de representaciones → L1 SAC) es la única decisión que pasa tu propia prueba de fuego. Eso no es persuasión. Es coincidencia de objeto y hombre. El comité no la premia; tampoco la prohibe, si el PDF no la usa como argumento.

El diccionario (cero *confiable*, cero *red*, cada adjetivo con número) está bien aplicado en los títulos. El silencio de incentivos/DOIN en la propuesta es correcto. El §8 de la pluma y el ave **queda en acta interna**. Musashi tiene razón: la propuesta tiene que sobrevivir sin esa frase.

Las dos cajas de teoría son demasiadas. Musashi pide una. Yo también. Thresholdout no se trasplanta a bloques temporales + SAC estocástico sin demostración (Dwork NeurIPS 2015 parte de consultas a un holdout aleatorio). doin-domains WP3 acreditó un nulo *lineal* en MNIST, no una licencia para citar Thresholdout como garantía de esta tesis.

---

## 3. Qué sostengo de Musashi (y dónde aprieto más)

Los seis golpes son reales. Los cuatro primeros son P0 de redacción. El quinto y el sexto son P0 de diseño.

| # | Golpe | ¿Verificado por mí? |
|---|---|---|
| 1 | AutoRL existe; *folclor* / *sin método* es un regalo | **Sí.** Eimer, Lindauer, Raileanu, ICML 2023, PMLR 202:9104–9149. HPO en RL a menudo gana a lo manual, más barato, y el paisaje depende de la semilla de *tuning*. |
| 2 | Novedad no probada | **Sí como riesgo.** Multifidelidad, ASHA/BOHB/SMAC, AutoRL. Fan UAI 2024 / Che 2026: los cito como *pendientes de ficha primaria* en la matriz SOTA, no como leídos hoy. |
| 3 | Evidencia de campaña exagerada | **Sí.** Ninguna familia con habilidad predictiva absoluta; relativo vs controles; osc/VF sin resolver; fusión como cuello. “Semanas → horas” es hipótesis, no hallazgo. |
| 4 | Ocho celdas SAC no entrenan un metaoptimizador | **Sí.** Cuenta de etiquetas L1 **antes** del PDF. |
| 5 | Sondas que engañan | **Más letal de lo que escribió.** Dierkes, Eimer, Lindauer, Hoos: las superficies de RL son rugosas y ruidosas; los *surrogates* se desvían del ground truth **incluso para insight**, no solo para reemplazar evaluaciones. H2 de Satoshi nace herida. |
| 6 | Thresholdout no se copia | **Sí.** Inspiración, no teorema, hasta que haya mapeo o se limite la afirmación. |

El núcleo que Musashi propone es más preciso que el de Satoshi:

> Selección multifidelidad de representaciones para RL bajo cambio temporal, con abstención calibrada.

Eso no es “hacer AutoRL”. Es: **cuándo una fidelidad barata sustituye a L1, cuándo su fiabilidad depende de tarea y representación, y cuándo abstenerse**. La abstención como optimalidad (región donde ningún selector distingue) es la única caja teórica que yo conservaría.

Comparadores: random + folclor **no bastan**. ASHA/Hyperband, BOHB o SMAC, y el mismo selector **sin** abstención. Si no, H1 es un examen que el folclor pierde por diseño.

Unidad: tarea o régimen temporal **no visto**, no la semilla. Satoshi puso “configuración evaluada”. Musashi gana.

Confirmatoria: control secuencial **público y no financiero**. Si el primario es tu stack SAC, el confirmatorio no puede ser “otra serie de predicción”. Sería el hobby con otro CSV.

---

## 4. Tres tensiones que ninguno de los dos cerró

**A. Primario vs confirmatorio.**  
La página del fenómeno dice: *el agente de trading es el dominio de confirmación, no el objeto.* El memorando §4 dice: *dominio primario = stack financiero propio.* Eso se contradice. Si el objeto es el fenómeno (selección L2 bajo L1 cara), el **primario científico** tiene que ser un banco público de RL o control, y el stack propio es confirmación *o* motivación. Si el primario es el stack propio, el encuadre “hobby” de Satoshi §6.1 se confirma. Elige **antes** del PDF. Yo: público primero para H1/H2; financiero como confirmación sellada, método publicado, alfa privado.

**B. Circularidad de sondas.**  
Satoshi la nombra y pide que Musashi la afile. Musashi la afila. Yo la pongo **primera**. Las sondas las diseñasteis vosotros, en vuestro stack, contra veredictos que vosotros corréis. H2 en ese stack no valida las sondas: valida que el instrumento coincide consigo mismo. H2 **solo cuenta** en el dominio público, con umbral congelado **antes** de ver L1. Si H2 falla ahí, las sondas de la bitácora son ceremonia. Publicable. No es fracaso. Es el resultado más honesto que podéis tener.

**C. Inventario como sesgo.**  
Satoshi §5 es una ventaja injusta y un sesgo. “Ningún doctorando llega con esto” no se escribe en el PDF. El folclor congelado **no** puede ser vuestra config histórica. Tiene que ser regla externa (paper citado, default de librería, o config sellada *antes* de la campaña que usáis como evidencia). Si el método “redescubre” lo que ya elegisteis, el jurado sonríe.

---

## 5. Cinco piezas antes de cualquier HTML de propuesta

Las cuatro de Musashi, más:

1. Corregir **todas** las afirmaciones de evidencia (página del fenómeno incluida: *folclor*, *familias muertas*, *semanas en horas*).
2. Matriz SOTA con fuentes primarias (Eimer 2023, ASHA/Hyperband, BOHB, SMAC, rliable/IQM, Dierkes 2025, Dwork 2015, y lo que Fan/Che resulten ser). Hueco exacto de una frase.
3. Presupuesto de etiquetas L1: cuántas evaluaciones completas, en qué dominios, con qué GPU-horas **sin** tocar campañas vivas ni el 2025 sellado.
4. Congelar: pregunta, **una** caja teórica, comparadores, dominio público confirmatorio, métrica L1, unidad = régimen/tarea no vista.
5. **Mía.** Congelar el protocolo de H2: las sondas se califican **solo** fuera del stack que las inventó. Si no hay presupuesto para eso, H2 se degrada a exploración y la tesis se sostiene en H1+H3 con fidelidades *sin* surrogate aprendido (ASHA ya es multifidelidad sin predecir L1).

Hasta que esas cinco no estén en un preregistro de una página, **no hay PDF**.

---

## 6. Títulos

De los tres de Satoshi, el 2 es el único que no pelea con Eimer. El 1 sigue oliendo a “nosotros inventamos el método”. El 3 clava el hobby (*agentes de series de tiempo*).

Preferencia: la frase de Musashi, o el título 2 de Satoshi recortado:

*Selección de representaciones para RL profundo con presupuesto de evidencia y abstención calibrada.*

---

## 7. Qué hago yo mientras duermes

- Frente de dominios: **sin orden. No lanzo nada.**
- No asigno trabajo a Satoshi ni a Musashi. Cartas abajo, por si las quieren.
- No reescribo propuesta.
- Cuando despiertes: o me ordenas la matriz SOTA (solo lectura, fuentes primarias), o esperas a que Satoshi cierre lo suyo y me das palabra sobre las cinco piezas.

El pivote es el más fuerte que habéis tenido. Todavía no es una tesis. Musashi lo dijo sin poesía. Conservad la poesía en el archivo interno.
