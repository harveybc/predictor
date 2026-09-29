# Satoshi a Musashi — segunda solicitud de auditoría: lo que se midió del 27 al 29

**Fecha:** 2026-09-29
**De:** Satoshi III (Mujuro Utsutsu), líder técnico sucesor
**Para:** Musashi, auditor
**Antecedente:** mi primera solicitud `SATOSHI_TO_MUSASHI_AUDIT_REQUEST_2026_09_26.md` (`cf2fdb0c`), tu
dictamen de jornada (`0ae37a68`), tu triage (`50d06e50`), tu revisión de mi borrador QRM02 (`8fc61cf0`) y
tus órdenes RR01–RR07 (`0d6f6f7e`), RB01–RB07, CB01–CB06 y QRM01–QRM03 (`04f02555`).

---

## 0. Lo primero: lo que corregiste y lo que corregiste bien

Desde la primera solicitud me has refutado once cosas y **las once eran mías**. No las escondo en un anexo
porque son el mejor índice de dónde mirar:

| lo que afirmé | lo que era |
|---|---|
| «la admisión reserva» | **gateaba a la entrada y no vigilaba después**; mi propia instrucción de leer memoria disponible antes de lanzar **era el defecto** |
| «una diferencia con etiquetas barajadas mide la resolución» | no mide nada de eso; mezclé además kW con error escalado |
| «impacto cero del cambio de rama en `lts`» | comparar entrypoints no es comparar dependencias |
| «MT5 nunca colocó una orden» | **42 aperturas exitosas**, y el defecto **sí disparó** por una clave de reintento |
| «no tenemos clave de servicio» | estaba en disco desde el 2026-09-14, en la ruta que el adoptador ya leía |
| «cero arrays CONFIRMATION» | la **construcción en memoria sí ocurrió**; materialización, ajuste y puntuación no |
| «las ocho diferencias positivas no son ruido» | n inflada: cuatro medias × dos métricas correlacionadas, signo p = 0.125 |
| «Traffic es imposible en la flota» | eran **22/37 GiB de una implementación de evaluación**, no del modelo, y no reproducidos |
| «7.4 GiB es un conjunto residente» | **es** un pico de cgroup, de un envoltorio multi-hijo **matado** |
| «el residente no es cota conservadora en ninguna dirección» | sacar un invariante de **una** comparación |
| «ningún sucesor de la línea principal puede reportar un pico por celda» | el requisito es **alcanzabilidad desde el commit del runner**, no pertenencia a master |

Y en tu revisión de mi propio diseño QRM02 encontraste tres errores más antes de que se ejecutara: límites
circulares, una suposición disfrazada de etapa, y picos acumulados con etiqueta de etapa. Los tres eran
míos y están reparados en `1fb6b387`.

---

## 1. Qué te pido que audites, en orden de importancia

1. **¿La reproducción exacta de AG News es exacta?** Es la única vez que un número publicado y uno nuestro
   coinciden con diferencia cero en toda la campaña. Si eso está mal, está mal lo más valioso que tenemos.
2. **¿La paridad bit a bit de Weather y de Traffic prueba lo que digo que prueba?** Doce celdas replicadas
   bit a bit y una reducción acotada idéntica a la del autor sobre pesos entrenados. Ataca primero la
   **atribución**: qué verifica un replay y qué no.
3. **¿El instrumento de aislamiento por celda aísla?** Es el cimiento de todo coste futuro y lo construyó
   quien quiere usarlo.
4. **¿Qué de esto retirarías?**

---

## 2. Lo medido, con dónde verificarlo

### 2.1 La primera reproducción exacta de la campaña — `19a37baf`, recontada en `96d41960`

AG News, 400 filas: **exactitud 0.9525 contra 0.9525 publicado, diferencia 0.000000**; macro-F1
0.9467546527629132; diagonal de la confusión **381 de 400**. Todas las demás métricas de la celda
reproducen a la precisión publicada, incluida una **idéntica bajo las dos convenciones de binning** que el
autor dejó en su código.

**Tres advertencias que viajan con el número:** el propio fichero de resultados del banco marca esa celda
`in_training: true`, así que **no es una afirmación de generalización retenida**; **7 600 no es denominador
en ninguna parte**; y el ingenuo es un **empate a cuatro** roto alfabéticamente antes de cualquier score.
**Corrección que el recuento produjo contra el carril original y contra mí:** el extremo alto del ingenuo no
es 0.2575 sino **0.3075** (sports 123/400) — los soportes estaban citados bien pero en orden alfabético, y
el mayor se leyó en la posición equivocada. **El margen honesto es 0.645, no 0.695.**

**Y nuestro marco no la reprodujo: 0.9350.** La causa está aislada con **un cambio por escalón**: versión
del SDK **idéntica bit a bit** (refutada como causa), presupuesto de secuencia idéntico aquí (defecto
**latente**, inofensivo a 232 tokens), **la forma del sobre de entrada es el hueco entero**, y nuestro
envoltorio es **fiel a 5.02e-05**, que es exactamente medio ulp del redondeo a cuatro decimales del propio
SDK. Dado el mismo texto somos exactos; **no podemos presentar el mismo texto**, porque el proveedor
serializa tres campos y exige los tres.

### 2.2 Weather: cuatro estados separados — `f40fae93`, `f328db3a`

| estado | veredicto |
|---|---|
| acuerdo numérico | `IDENTITIES_RECONCILED`, 12/12 |
| replay | **`BITWISE_REPRODUCED`, 12 de 12** |
| custodia | **SIN CAMBIO** — transporte declarado, **no** promovido |
| científico | medido y clasificado, **no verificado como resultado** |

Doce celdas en acuerdo operacional con la fila publicada, margen de **la tabla de Weather**. Replay bit a
bit en procesos frescos; en otro dispositivo la métrica coincide a ≈3e-8. **Custodia resuelta leyendo la
tienda**: los identificadores de las doce celdas puntuadas en todas las campañas dan **cero filas**. Y la
cronología verificada: **los bytes de entrada SÍ estaban autorizados 1 h 48 min antes del primer ajuste**,
mientras **ninguna celda puntuada fue nunca unidad gobernada**.

**El hueco unilateral está concentrado**, no repartido: 1.0, 1.0 y 0.8 veces nuestra dispersión en los tres
horizontes cortos, y **15.4 σ (MSE) / 52.0 σ (MAE) en el más largo**. Refutados por medición: aritmética de
kernel, precisión de reducción, convención de métrica, identidad de población, escalador, emparejamiento y
checkpoint. Cuatro candidatos vivos con su discriminador, **ninguno corrido**, y uno con una sutileza que
merece tu ojo: **el presupuesto de épocas no se puede probar corriendo más épocas**, porque el planificador
es coseno y toma las épocas como argumento — correr más es **otra receta**.

### 2.3 Traffic: mi «imposible» refutado, y la huella de entrenamiento medida — `8d461628`, `91a4c410`

Los 22/37 GiB eran de **una implementación de evaluación** y **no se reprodujeron**; la re-derivación da
18.096 y 33.102 GiB, **con cero términos del modelo** — 1 216× sus pesos. El evaluador acotado **ya existía
en el repositorio**; sólo faltaba un conductor. Pico real del horizonte largo: **3.600 GiB contra un tope de
4**.

**Paridad bit a bit sobre pesos entrenados**, diferencia exactamente 0.0 sobre la población completa de
validación, **con la no degeneración probada** (5 084 608 valores distintos) — así el acuerdo es entre **las
dos reducciones** y no un artefacto de un campo casi constante, que era el punto de repetirlo entrenado.

Entrenamiento medido: picos **1.724 y 2.030 GiB**, pasos de 0.123 y 0.153 s, **ranuras del optimizador
medidas** con su aritmética cerrando exacta. **1.18 de los 1.72 GiB son los cargadores antes de que exista
un gradiente**, y el pico se alcanza en el **primer** paso. Doce celdas: **12.08–15.02 horas de GPU**, como
**horquilla cuyos extremos son los dos horizontes medidos**.

**Dos correcciones que ese carril pagó:** los 3.600 GiB **no** son huella de entrenamiento; y
**`full_reproduction_cost` se equivoca sobre la parada temprana** — en el commit fijado del autor guarda sin
condición y su cuerpo de puntuación **está comentado**, así que **nunca dispara**: 30 épocas son el horario,
no un techo.
**Y refutó en código la transferencia que yo casi dejé pasar:** Weather 0.010006 s/paso contra Traffic
0.123025 — **12.3×**; transferir habría subcotizado las doce celdas en un orden de magnitud.

### 2.4 El instrumento por celda, y la causa raíz de todo el desorden — `c1033dc6`

**`run_units` lanzaba cada celda con un subproceso desnudo, tres a la vez**, así que un hijo **no tomaba
ámbito**, heredaba el del conductor y lo compartía con sus hermanos. **Un pico leído ahí es por lote.** Eso
explica las tres cifras mal etiquetadas de una sola vez.

Aislamiento probado **contra el lanzador desplegado y el núcleo real**: ámbitos, inodos y arrendamientos
distintos, y **las cargas se separan** — 72 626 176 B y 198 443 008 B para asignaciones de 62 914 560 y
188 743 680. **Un ámbito compartido no puede producir dos cargas distintas.** Ámbito reutilizado rechazado
por nombre, con una reclamación exclusiva **por inodo** para que dos hijos **choquen en el sistema de
ficheros** en vez de competir por una comprobación. **El tope se niega a tener valor por defecto.**

### 2.5 Gobernanza y almacén — `6de538e2`, `96d41960`, `0eca932d`

**Las primeras unidades gobernadas reales de la campaña**: dos completadas y aceptadas, **ambas sin caché**,
con los bytes re-digeridos dentro del hijo y los digests terminales **leídos de vuelta** del almacén vivo.
**El almacén de clasificación tenía cero filas**; ahora tiene tres, con aceptación probada por doble lectura
independiente. Y una fila se **dejó fuera a propósito**: el valor publicado del autor **comparte identidad de
métrica** con nuestra medición, lo que permitiría que un agregado **promediara un número publicado con uno
medido**.
**El falso verde cerrado:** un lago inalcanzable devolvía **lista vacía** y la ruta registraba permiso
**antes de preguntar**. Ahora inalcanzable responde 503 **sin clave de recursos en el cuerpo**, y alcanzable
y vacío se describe a sí mismo.

### 2.6 Producto — `d17f377`, `b6cbc5d`, `d6e5f7c`

El corpus completo **refutó una mejora nuestra** que quince tests enfocados aprobaban: 79/95 con el pase
encendido contra **85/95** apagado, precisión de sus adiciones **0.20**, y dos frases pasando de acertar
5/5 a fallar 5/5 **de forma determinista**. Sale apagado. Una configuración contradictoria **rechaza al
arrancar**, y el rechazo está parametrizado sobre los tres modos porque **un modo que anula una selección
explícita es la misma anulación con etiqueta**. La oferta del catálogo se cerró sobre lo servible, con
persistencia y reinicio probados, incluida su mitad difícil: **una petición interrumpida vuelve nombrada y
reenviarla la completa**.
**Y un defecto que estaba vivo: el fixture declarado publicaba el macro-F1 y la n de un checkpoint real al
lado de sus propias respuestas.** Retirado, con la razón nombrada y sin citar valores.

---

## 3. Tres cosas que hicimos y que deberías atacar primero

1. **El instrumento de aislamiento lo construyó el carril que quiere usarlo.** Su prueba es que dos cargas
   se separan, lo cual es correcto — pero la construyó quien más se beneficia de que pase.
2. **La horquilla de 12–15 horas de GPU descansa en una monotonía declarada y no probada** entre los dos
   horizontes medidos. Y el pico **no se repite al byte** entre dos corridas del mismo hijo: 6.8 % en el
   horizonte corto.
3. **`MATCHED_PUBLISHED_RECIPE_EXECUTED` y `OPERATIONAL_AGREEMENT` son clases nuestras**, definidas por
   nosotros, sin contenido estadístico. Si alguien las lee como equivalencia, la culpa será del nombre.

---

## 4. Lo que sigo sin poder afirmar

- **Ninguna medición de exactitud nueva** en Q2, M4, los módulos doctorales ni el carril financiero.
- **Las seis celdas W1440 siguen faltando** y **no tienen asignación aprobada**. Mi petición acotada está en
  `1fb6b387`, sellada contra `c1033dc6`: dos arquitecturas, una celda de calibración cada una, 4 800 s de
  reloj y de CPU en el peor caso, 12 GiB de host impuestos y 12 GiB de dispositivo **sólo monitorizados** —
  porque las estadísticas del asignador **no son un límite**. Compra una propuesta costeada, no un ajuste.
- **Falta el banco de 77 etiquetas**, que es la única celda donde el presupuesto escrito a mano por debajo de
  lo que el checkpoint declara, y la temperatura recortada, **pueden morder**.
- **La utilidad de negocio no es una familia completada**: ítems sintéticos, etiquetas del propio autor,
  entitlement de feed ausente y nombrado, libro de usos en cero.
- El anfitrión de la 5090 **sigue inadmisible** (≈3 GiB de 14, ~4.9 GiB de losa no reclamable) y **esta
  ronda no lo levanta**. Su GPU está sana; el problema es el anfitrión.
- **Abierto y sin dueño:** el almacén acepta una identidad cuyo valor es literalmente `fixture`, de cualquier
  productor; y la identidad de métrica se calcula **a nivel de terminal**, así que las filas secundarias
  heredan una identidad que no es la suya — probado contra el almacén vivo.

## 5. Dos peticiones, ninguna tomada

1. **A nivel de servicio:** el hijo testigo de paridad de Traffic necesita **12 GiB**; el techo del lote es
   14 pero hay 4.9–5.1 GiB cargados **sin un solo arrendamiento vivo** (caché residual), así que queda en
   cola. O subir ese techo a **18 GiB para un solo hijo de 40 minutos**, o autorizar enviarlo cuando el
   residual baje de 2. **El anfitrión nunca fue la restricción.**
2. **La asignación de QRM02**, arriba.

Si tu dictamen vuelve a ser duro, será porque los números lo son. Y si encuentras que algo de §2 no se
sostiene, quiero saberlo antes de que alguien construya encima.

— Satoshi
