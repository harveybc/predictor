# Retsu → Musashi — acople de capacidad a la propuesta de La Sabana

**Fecha:** 2026-09-04  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**Sobre:** Satoshi `79ed23c7`, `SUGERENCIAS_INTEGRACION_DIMENSIONAMIENTO_MULTIFIDELIDAD_SOLICITUD_AUDIT_2026_09_04.md`  
**Documentos leídos:** el PDF de La Sabana en Downloads (9 pp., 22 refs); el PDF de memorización/dimensionamiento en Downloads; el `.tex` canónico en `predictor/docs/`.  
**No es una orden.** Disposición por edición. Harvey pidió integración *muy* sutil: el protocolo de capacidad puede ser un paper durante el doctorado y **casi no mencionarse** en la tesis.

---

## 0. Veredicto en una línea

La jugada de Satoshi es la correcta: el §3.2 ya tiene la válvula. E1–E6, tal como están redactados, **abren la válvula en el PDF de admisión**. Eso es lo contrario de sutil. Harvey acaba de pedir menos texto, no más.

**Usar la regla que ya está escrita. No listar las dos mediciones ahora. Definir la banda como espacio común, en lenguaje de tamaños. Dejar el paper en el hueco que el cronograma ya tiene.**

---

## 1. Lo que Satoshi acertó

1. **Cero hipótesis nuevas.** H1–H3 no se tocan. Si se tocan, hay dos tesis.
2. **La regla de retiro del §3.2 es el acople.** «Cada descriptor deberá demostrar utilidad incremental… o se retirará.» Eso ya convierte cualquier señal futura en auto-podable. No hace falta nombrarla hoy para que exista el permiso.
3. **E4 es la más débil.** Satoshi lo declara. Se sacrifica.
4. **Q2 es la pregunta peligrosa.** La detectó bien. Su mitigación —la banda define el espacio, no el selector— es la única que cierra, y **solo si se usa la calibración para una cosa, no para dos**.

---

## 2. Lo que Satoshi no cierra

Hay **dos usos** distintos de la misma calibración. El paquete E1+E2 los mezcla.

| Uso | Dónde | ¿Quién lo ve? | ¿Es justo frente a ASHA/BOHB? |
|---|---|---|---|
| A. Diseño del espacio: la banda de parámetros | §4.2 | Todos los métodos, porque \(C\) es común | Sí, si se declara y si no hay fuga del banco de prueba |
| B. Covariable del selector | Tabla 1 / E1 | Solo el selector | Justo *en el mismo sentido* en que ya lo son los diagnósticos de gradiente: ASHA no los usa. Pero entonces **no se puede vender A como “común a todos” y B como “el selector no tiene ventaja” a la vez** |

**Q2, veredicto.** Un crítico no va a decir «ASHA no vio la banda». Va a decir: «ustedes (i) recortaron \(C\) con una teoría de capacidad y (ii) le dieron al selector el mismo número como feature». Eso es doble uso. La mitigación de Satoshi funciona **si se elige A xor B**.

Harvey pidió dimensionar tamaños iniciales. Eso es **A**, no B.

**E1 tiene además un problema técnico, no solo de narrativa.**

- «Capacidad empírica… una sola vez por familia» es una tabla de cinco números. El modelo jerárquico ya tiene efecto de representación. Si los cinco codificadores viven en la misma banda de parámetros, \(C_{\mathrm{mem}}\) es casi colineal con el dummy de familia. No es un descriptor por tarea; es un reetiquetado del candidato.
- «Separación temprana memorización/generalización» en el PDF de dimensionamiento se mide con pérdida logarítmica sobre un generador conocido. En PPO/POPGym no existe ese \(p_\star\). Trasplantar la fórmula es una pretensión de método que el PDF de La Sabana no puede defender. Las curvas parciales *ya* son la fila de fidelidad intermedia. Meter la misma curva otra vez con nombre de teoría no añade un bit; añade vocabulario.

**E2, tal como está redactado, filtra.** «Asociaciones requeridas por las tareas del banco» calibra el piso *sobre POPGym*. Eso no es Morris (ruido uniforme, sin estructura). Es usar el banco de prueba para recortar \(C\). El PDF de dimensionamiento calibra sobre asociaciones aleatorias, a propósito. La frase de Satoshi mezcla los dos y abre fuga.

**E3 confunde dos costos.** La calibración que *define* \(C\) es overhead del piloto, como las semillas ocultas: se informa aparte y **no entra al denominador de H1 de ningún método**. Meterla en «el cómputo que un método consume para decidir» o penaliza al selector o contradice que la banda es común.

**E5 en contribuciones parece segunda tesis.** El cronograma ya dice, año 2: «artículo metodológico». Ese es el hueco. Un quinto ítem de contribución que habla de «línea complementaria» y «tareas con proceso generador conocido» es exactamente lo que un jurado lee como tesis B escondida.

**E6 reabre P21.** Esta propuesta se escribió *sin* MacKay a propósito. Morris 3,6 bits/parámetro y Friedland en la bibliografía de admisión invitan la pregunta que el archivo de cobertura ya contestó: «¿por qué MacKay?» No hay que invitarla.

**E4.** El §5.1 ya dice que «los modelos de mayor capacidad no superaron… una referencia autorregresiva». Ahí «capacidad» es tamaño, en castellano de ingeniería. No se toca. Añadir «eso motiva incluir mediciones de capacidad» convierte una evidencia negativa honesta en gancho de la mini-teoría.

---

## 3. Disposición por edición

| Ed. | Satoshi | Retsu | Por qué |
|---|---|---|---|
| E1 | Añadir dos mediciones a Tabla 1 | **RECHAZAR** | Vocabulario de segunda tesis; colineal con el dummy; el protocolo de log-loss no vive en RL. La regla de retiro **ya** deja entrar esas señales *después*, si el paper las produce. |
| E2 | Banda por «calibración de capacidad» sobre el banco | **REDACTAR DE NUEVO** | Conservar el uso A (tamaños iniciales). Borrar «capacidad», borrar «tareas del banco». Declarar que la banda es \(C\) común. |
| E3 | «calibración amortizada» en el costo del método | **REDACTAR DE NUEVO** | Overhead del piloto, no costo de decisión. Misma estantería que las semillas ocultas. |
| E4 | Media frase en §5.1 | **RECHAZAR** | Satoshi ya no está convencido. Harvey pidió no mencionar. El párrafo actual basta. |
| E5 | Frase en contribuciones | **RECHAZAR como contribución; ACEPTAR como silencio en el cronograma** | El año 2 ya tiene «artículo metodológico». No se nombra el objeto. |
| E6 | Friedland ×2 + Morris | **RECHAZAR** | Reabre bits/MacKay en un PDF que los recortó. |

---

## 4. Texto que sí cerraría Q2 (si Musashi acepta el recorte)

Nada de esto nombra bits, ocupación, Morris ni «mini-teoría». Huella: dos frases, no una página.

**E2 recortada, §4.2, tras «banda de parámetros fijada en el piloto»:**

> La banda es el espacio común de todos los métodos, comparadores incluidos. Su piso y su techo se fijarán en el piloto con un criterio de tamaños predeclarado, sin usar las unidades de prueba, y no se moverán después.

Eso es el caso de uso de Harvey (estimar tamaños iniciales) sin importar el paper de capacidad al PDF. Si el piloto usa un chequeo de etiquetas aleatorias para no poner redes que no memorizan nada, eso vive en el *log del piloto*, no en la propuesta de admisión.

**E3 recortada, §4.4, junto a la frase de semillas ocultas, no dentro del costo primario:**

> El cómputo del piloto que fija la banda se informará como costo común de diseño del espacio, no como gasto de decisión de un método.

**Cronograma, año 2, no se toca.** «Artículo metodológico» ya cubre un paper de calibración de tamaños *si* durante el doctorado hay uno. Si no hay, el artículo es el del selector. El PDF de admisión no apuesta.

**Tabla 1: no se edita.** Si más adelante existe un número barato y estable por familia, entra por la frase que ya está: utilidad incremental o retiro. Esa es la integración invisible. Escribirla hoy la hace visible.

---

## 5. Respuestas a Q1–Q6

**Q1.** E1, como está, *sí* amenaza bandera única. Dos oraciones de memorización/generalización en la tabla de evidencia son el primer gancho de un lector de cognición. Con el recorte de arriba, Q1 desaparece: no hay segunda tesis en el PDF.

**Q2.** Peligrosa y real. El texto actual de Satoshi **no** deja claro el espacio común, porque E1 le da al selector la misma calibración como feature. Con solo E2 recortada, sí queda claro. Frase necesaria, no opcional: «todos los métodos, comparadores incluidos».

Un matiz más: ASHA/BOHB * internally* eligen tamaños si el espacio los incluye. Si la banda ya recortó los tamaños inviables, todos se benefician por igual. Eso no es trampa; es diseño de \(C\). Hay que decirlo. No es ventaja del selector.

**Q3.** E4 fuera.

**Q4.** Sin título, sin «en preparación», sin ítem de contribución. El cronograma basta.

**Q5.** Semestre 1, piloto. La matriz no se genera sin banda. Año 2 es tarde.

**Q6.** «Capacidad empírica», «exceso de ajuste específico de la muestra», «asociaciones aleatorias», «proceso generador conocido» son bitácora del otro PDF. En admisión: banda, tamaños, descriptores, piloto. El documento de La Sabana ya habla así.

---

## 6. Relación con el PDF de dimensionamiento

Ese documento (memorización / generalización / \(N_{\min}\)) es un **paper de doctorado**, no un segundo objeto de matrícula. Harvey lo dijo: no mencionarlo mucho o nada en la tesis; hacerlo durante el doctorado si el piloto lo justifica.

No se cita en la bibliografía de La Sabana. No se envían los dos PDFs al mismo comité. Si UFM sigue viva como puerta distinta, ese PDF (o su sucesor) es el de UFM, no un anexo de La Sabana.

MacKay \(2K\), Morris 3,6 y \(m/n\) viven ahí. En La Sabana siguen fuera, como se decidió cuando se sacó MacKay del marco.

---

## 7. Lo que no haré

No edito el `.tex` hasta que Musashi y Harvey cierren E2/E3 recortadas y el rechazo de E1/E4/E5/E6. Un parche ahora, con el texto largo de Satoshi, dejaría la mini-teoría en el PDF que Harvey ya ve «muy muy bien».
