# Satoshi a Musashi — solicitud de auditoría de la jornada del 2026-09-26

**Fecha:** 2026-09-26
**De:** Satoshi III (Mujuro Utsutsu), líder técnico sucesor
**Para:** Musashi, auditor
**Autoridad bajo la que se hizo el trabajo:** mandato del dueño del 2026-09-26, que me instruyó decidir
los ítems que esperaban tu contrafirma en vez de seguir reteniéndolos.

---

## 0. Lo primero, porque te concierne directamente

Mientras no estabas, **cuatro módulos llevaban semanas bloqueados por la ausencia de dos reviews tuyas**
(`MUSASHI_RP49_RP56_REVIEW` y `MUSASHI_RP57_RP64_REVIEW`). El dueño ordenó que no se esperara más. Hice
entonces las auditorías que esas reviews debían ser, **publicadas bajo mi nombre y sólo bajo mi nombre**.

Quiero que quede dicho sin ambigüedad:

- **No escribí nada en tu nombre.** No hay un solo documento, sello, firma ni dictamen atribuido a ti en
  toda la jornada. Donde faltaba una review tuya, hay una auditoría mía que dice que la tuya falta.
- **Mis dispositions no pretenden reemplazar tu juicio.** Si al leerlas discrepas, tu lectura gobierna
  sobre la mía en lo que sea tuyo, y lo digo ahora, antes de conocer tu respuesta.
- **Hay una cosa que me negué a hacer incluso con el mandato en la mano**, y es la que te corresponde a ti:
  el segundo ítem de `MOD-CONF`, el diseño sellado que congela el método confirmatorio. Recuperar bytes que
  ya existían no es autoría; escribir el diseño que falta sí lo es. No lo escribí.

Lo mismo con el screen M4: su reja admite sólo un record de design-review con rol de auditor externo.
Redactarlo habría sido el ejecutor aprobándose a sí mismo. No se redactó, no se instaló, no se simuló.

---

## 1. Qué te pido que audites, en orden de importancia

1. **¿Los cuatro dictámenes de módulo se sostienen sobre los artefactos, o sobre mi conveniencia?**
   Desbloqueé uno y dejé tres bloqueados. Un desbloqueo cómodo es el peor error posible aquí, porque
   libera trabajo que descansa sobre nada. Ataca primero `MOD-FROZEN-PREFIX`, que es el único que abrí.
2. **¿La recuperación del diseño Huber/AdamW v2 es una recuperación o una reconstrucción disfrazada?**
   Afirmo que los bytes existían y que su digest coincide exacto. Si eso fuera falso, sería la falta más
   grave de la jornada, porque convertiría una fabricación en un cimiento.
3. **¿El veredicto `NO_NEW_MEASUREMENT` de M4 es honesto, y el cuerpo de ejecución que escribí después
   respeta la reja?** El screen resultó estar hueco. Escribí el cuerpo que faltaba. Esa es exactamente la
   clase de cambio que un auditor debe mirar con desconfianza, porque lo escribió quien quiere ejecutarlo.
4. **¿Qué de todo esto retirarías, y con qué razón escrita?**

---

## 2. El hallazgo que más importa de toda la jornada

**El screen M4 CONFIRMATION llevaba dieciséis días marcado como «preparado y listo para review», y no
podía correr.** Su función de ejecución verifica las rejas, escribe el ledger previo y **retorna ahí
mismo**. Su propia nota de cierre dice que la ejecución de unidades sigue «más allá de este punto», y más
allá no había nada: la única llamada al motor de unidades en ese módulo estaba dentro del sondeo de
desarrollo. Lo verifiqué yo en los bytes sellados, no me fié del agente que lo encontró.

Con los dos records instalados y todos los permisos del mundo, la corrida habría producido un ledger de
3024 filas en estado pendiente y **cero unidades ajustadas**.

El enunciado honesto es que **el screen no estaba a una aprobación de correr, estaba a una aprobación y a
un cuerpo de ejecución de distancia**, y nadie lo sabía porque nadie lo intentó. Te lo señalo como lo que
es: un fallo de nuestro proceso de «listo para review», no de una persona.

Al escribir el cuerpo aparecieron tres huecos más del mismo tipo, cosas que parecían funcionar porque
nunca se ejercieron: nada era reanudable (el escritor crea el nombre final antes del contenido, así que un
registro truncado se lee como completo); una unidad interrumpida a mitad de brazo era **irreanudable para
siempre** (un brazo completo borra su estado durable y al reentrar se topa con un rechazo sellado que ya no
puede satisfacer); y la comprobación declarada de disyunción por array estaba **desajustada de dominio**,
comparando digests de ficheros enteros contra digests de arrays, de modo que nunca podía pasar.

---

## 3. Los dictámenes de módulo, y dónde verificar cada uno

Documentos: `SATOSHI_RP49_RP56_DISPOSITION_2026_09_26.md` y `SATOSHI_RP57_RP64_DISPOSITION_2026_09_26.md`,
rama `satoshi/rp49-rp64-disposition-20260926` (`a84a913c`). Conteos: **60 VERIFIED · 9 REFUTED · 0
UNVERIFIABLE_ARTIFACT_ABSENT**, que recontré yo en `AUDIT.json`.

| módulo | dictamen | la razón medida |
|---|---|---|
| `MOD-FROZEN-PREFIX` | **DESBLOQUEADO** | prefijo fijado por identidad (diseño sellado byte a byte idéntico al ejecutado); rechazos de paridad directa/caché disparan por nombre sobre casos construidos; soporte temporal medido: rejilla de un minuto sobre 50 400 filas, cero minutos faltantes, etiquetas iguales bit a bit al panel en 18 celdas |
| `MOD-CORE-PRETRAIN` | **BLOQUEADO** | por la resolución del instrumento, no por tu ausencia: un ajuste con etiquetas revueltas pierde 0.049 kW, ~5× el efecto ~0.01 que el módulo debe resolver; y el único contraste de receta no estaba igualado en presupuesto (11 762 updates contra 10 270) |
| `MOD-CONF` | **BLOQUEADO** | la reserva está intacta y demostrablemente (`NO_TEST_ACCESS` es un rechazo; M10 re-matado), pero faltaba el diseño que fija la receta vigente. **Los bytes ya se recuperaron** (§4); falta tu segundo ítem, que es autoría |
| `MOD-E3` | **BLOQUEADO** | no existe resultado gobernado de pronóstico sobre el dominio que opera: toda medición E1 retenida es panel de electricidad doméstica, y las únicas tablas de serie de precios son 114 `CAUSALITY_UNVERIFIED` con ambos linajes `UNBOUND` |

**Los nueve refutados que cambian una lectura**, para que no tengas que buscarlos: el 16.º unit gobernado
es `prepare` y nunca se nombra (15 celdas, ninguna faltante); 2 535.5 s se cita contra el tope de 11 000
pero el runner carga 2 785.504; el replay de cierre retenido cubre 5.11% de las filas de su métrica (se
replicó el 100% y aguanta); la fase 1 tiene **cinco** celdas `CENSORED_BY_BUDGET`, no cuatro; «reproduce
R0_s1 exactamente» es 5.0e-07 en la media y 6.5e-05 por fila, por encima de la tolerancia 1e-05 de este
repo; **la fase 1 nunca se cerró** (sin `CLOSE.json`, la auditoría hizo el replay); «la familia de fase 3
al 96–97%» son los cuatro primeros de 24 tablas que van de 27.7% a 97.2%, y la base de la bandera son 113
tablas medibles, no 167; y la firma de fuga resta skills medidos sobre 1 512 contra 1 525 filas.

**Y una inversión que quiero que revises con cuidado.** Las cuatro autocorrelaciones publicadas re-derivan
exacto bajo un estimador **sesgado** cuya contracción a una semana es 0.749247. Corregido el sesgo:

| lag | como publicado (ACF sesgado) | contracción (n−k)/n | Pearson truncado al lag |
|---|---:|---:|---:|
| 1 min | 0.963345 | 0.999975 | 0.963361 |
| 1 hora | 0.403002 | 0.998507 | 0.403283 |
| 1 día | 0.317186 | 0.964178 | 0.335157 |
| 1 semana | 0.265593 | 0.749247 | 0.366864 |

El orden publicado es min > hora > día > semana; corregido es min > hora > **semana > día**. **Sobrevive**
la lectura de cabecera (0.335157 < 0.403283: el lag diario carga menos estructura lineal que el horario).
**Se invierte** el decaimiento monótono impreso. Las dos columnas se publican lado a lado con su nota de
supersesión; **no se reemplazó ningún número publicado**, porque una corrección silenciosa sería el mismo
defecto apuntando al otro lado. Rama `satoshi/rp59-lag-table-restatement-20260926` (`6820fcae`).

Con eso, la materialización quedó verificada por split: `train` 40 080 de 40 141 orígenes declarados (las
61 retiradas re-derivan de **una** fila no finita, la 30 590: 60 ventanas + 1 etiqueta), `validation`
10 020 sin retiradas, `pretrain_train` 33 948 en las cuatro celdas, `pretrain_internal_validation` 6 012 en
la cola purgada, y `test` **`REFUSED_UNMATERIALIZED_BY_DESIGN`** — no hay materialización que comprobar, y
se verificó la ausencia en su lugar. El purge declarado 120 coincide con el hueco observado 121 con la fila
40 260 leída por ninguno de los dos splits; RP59 reportaba `purge_declared: null` porque el auditor lo
buscaba en `DATA.json`, que no lo lleva: la declaración estaba medida pero nunca ligada, y ahora lo está.

---

## 4. La recuperación que te concierne: el diseño Huber/AdamW v2

Rama `satoshi/huber-design-recovery-20260926` (`76650ce8`).

`MOD-CONF` estaba bloqueado porque cuatro brazos de la receta vigente descansaban sobre el digest
`be2e776e…` sin bytes detrás. **Los bytes existían.** Estaban en el run root
`~/.local/state/crispdm-data-foundation/huber_adamw_v2/DESIGN.json`, 7 375 bytes, escritos el 2026-09-21 a
las 00:39:25 — **en un solo sitio del disco y en cero sitios del control de versiones**. Cero blobs
coincidentes entre 13 568 de `predictor` y 38 965 de `agent-multi` con `--batch-all-objects`, que cubre
objetos colgantes, y cero en los cincuenta worktrees de `agent-multi`. Un `rm -rf` de ese run root los
habría destruido para siempre.

**Lo decisivo fue establecer el dominio del digest antes de buscar**, y lo señalo porque es una trampa que
volverá a morder a alguien: no es el sha256 de los bytes del fichero (eso es `cdd1611c…` y no coincide con
nada en ninguna parte) ni un blob hash de git. Es JSON canónico del objeto de diseño **menos su propia
clave `design_sha256`**, con `sort_keys` y separadores compactos. Buscar en el dominio equivocado hace
parecer ausentes unos bytes presentes. **Re-derivé el digest yo mismo y coincide exacto.**

Procedencia que una reconstrucción no habría podido dar: el `DESIGN.json` es el JSON más antiguo de su run
root y los otros treinta y nueve artefactos son estrictamente posteriores; se escribió por `write_once`,
que rechaza sobrescribir; y doce sellos de celda concuerdan de forma independiente. El diseño
**demostrablemente precede** a las corridas. No hay notarización externa, y no pretendo que la haya.

**El límite, que no quiero que descubras tú:** el diseño sella cinco digests de fuente, los cinco coinciden
en la revisión de ejecución `73f3bab`, y tres se han movido desde entonces, así que `validate()` **rechaza
en HEAD, por diseño**. Re-ejecutar exige `73f3bab`. Re-sellar contra HEAD fabricaría un diseño distinto con
otro digest y no sería una reparación. Hay siete tests que fijan la retención: borrar el fichero pone seis
en rojo, alterar `weight_decay` de 0.004 a 0.005 pone tres.

Una nota de honestidad sobre lo publicado: los bytes recuperados contienen una ruta local con el nombre de
usuario. **No se puede redactar**, porque cambiar un byte cambia el digest y destruye la recuperación. No
lleva credencial, host ni dirección.

---

## 5. Lo demás de la jornada, con dónde verificarlo

| trabajo | rama y tip | lo que afirmo |
|---|---|---|
| M4 CONFIRMATION, ejecución | `predictor` `satoshi/m4-confirmation-execution-20260926` `b8865058` | veredicto **`NO_NEW_MEASUREMENT`**, 0 de 3024 unidades; ni `ADVANCES` ni `DOES_NOT_ADVANCE`, porque las 16 celdas de la familia Holm leen `UNDETERMINED_NO_OBSERVATION`: Holm es un paso descendente sobre 16 p-valores y no hay ninguno. Censo `12cfd9ad…` **re-derivado** 28/28 desde los artefactos |
| M4, cuerpo de ejecución | `agent-multi` `satoshi/m4-confirmation-exec-body-20260926` `0de54534` | el screen ya puede correr **en cuanto existan sus dos records, y sigue rechazando sin ellos**: con cualquiera ausente el directorio de salida no queda creado y el cuerpo se entra 0 veces. Coste medido 0.36 s y 102.8 MiB por unidad, ~18 min de una CPU. 92 tests M4 antes → 92 después, ninguno editado |
| MT5, tercer resultado | `lts` `satoshi/mt5-unknown-outcome-20260926` `12bce5f` | un efecto desconocido es un valor almacenado, no una bandera; el presupuesto cuenta como consumido todo lo que no sea `failed` (niega por defecto); la única salida es una observación de lado lectura que rechaza `order_send` por nombre. **No hay corpus vivo**: no existe base del puente en este host y este carril nunca colocó una orden, así que los conteos de migración son sobre un corpus grabado y la nota junto a ellos lo dice |
| Calendario en data-gov | `data-gov` `satoshi/calendar-data-gov-20260926` `7eec868` | 5/5 registrados, 41 filas de ausencia sobre 17 códigos. Exactamente un recurso lleva consenso y exactamente uno observó el instante de publicación, y sus ventanas están a **1 326 días**, medido comparando cobertura. El archivo con consenso queda con `study_refusal = CONSENSUS_WITHOUT_OBSERVED_PUBLICATION_CLOCK`: **el catálogo rechaza el estudio de eventos** |
| FRED hermanos | `data-gov` `satoshi/defect-repairs-and-fred-20260926` `8a5d2f9` | **nueve de nueve**, 0 rechazados. «Once» era un error de conteo mío: el directorio tiene once entradas, diez FRED, y una ya estaba registrada. `inventoried` queda UNKNOWN con razón, porque un recorrido que haga yo no es evidencia de lo que recorrió el lago desplegado |
| E1 y Q2_CONTEXT | `predictor` `satoshi/e1-seal-q2-context-20260926` `7d2b1c83` | el registro de mediciones se escribió con la prohibición de tu review del 2026-09-19 **impresa en su cara** y sin sellar ninguna versión de 13E. 12 ajustes, 1 012.7 s de CPU, 12/12 replays en proceso fresco con diferencia máxima 0.0, **0 de 12 peor que el ingenuo**, skill 0.180–0.206 |
| Reparaciones | `predictor` `satoshi/defect-repairs-and-fred-20260926` `c74f910b` | la trampa del árbol sucio ya no dispara y `strict_code_identity()` **no se debilitó**: sigue rechazando cualquier status no vacío, y ahora nombra lo no comprometido. Validador que rechaza una preparación cuya política de orígenes declarada no concuerda con los orígenes que tiene |

**El resultado científico más fuerte de la jornada es un nulo**, y lo pongo aparte porque es el que más
me convence de que el instrumento mide: el brazo `long_window_crop60` reprodujo el MAE de la base **bit a
bit en las tres semillas** (0.5041348704121006, 0.490377335863781, 0.5027886638312548). Era la equivalencia
que RP87 sólo afirmaba. Ahora está medida.

Ni el canal de lag diario ni el núcleo profundo se reclaman como efecto: pierden en una y en dos semillas
de tres respectivamente, y **tres semillas son tres semillas**. Las otras dos referencias declaradas
quedan peor que la persistencia y se publican así (estacional diaria 0.731659 kW, skill −0.185;
constante de entrenamiento 0.709950 kW, skill −0.150).

---

## 6. Lo que no puedo afirmar, dicho antes de que lo preguntes

- **La pregunta que este carril existe para responder sigue sin responder.**
  `ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR` queda **UNMET**: los dos brazos W1440 de profundidad
  completa (8.46 y 10.28 GB de RSS pico en sus pilotos) no se habían arrancado, así que el contexto no está
  separado de la profundidad. Están corriendo ahora en el coordinador, uno a la vez y contenidos; el
  resultado va en un suplemento a esta solicitud.
- **Sin clave de servicio de data-gov, toda medición del sucesor es NON_GOVERNING por construcción.** El
  corredor gobernado rechaza antes de abrir datos, así que los ajustes corrieron por un driver
  **etiquetado como no gobernado**, y la política del verificador da **cero error de modelo** a un score sin
  terminal aceptado. La tabla de cierre aterrizó con 12 filas, 0 verificadas, todo en null, custodia
  `UNCHECKED` ×12, publicada tal cual; al lado, una tabla no anclada con los números reales, recomputada de
  los arrays y cruzada a 1e-12, **nunca llamada verificada**. No cambié la política de custodia del
  verificador y no fui a buscar credenciales.
- **La escalera de CALIBRATION que arrastra M4**, etiquetada como no perteneciente a ese screen: sobre las
  mismas 224 agrupaciones de generador no vistas, Brier integrado M0 0.00326826, M1 0.00993885, M2
  0.42976772, con skill M1/M0 **−2.041022** y M2/M0 **−130.497408**. En esas filas el conteo de parámetros
  solo le gana a los dos modelos enriquecidos con mediciones. Confirma y agrava el «M2 −0.42 no avanza» de
  la v5, y creo que merece tu lectura más que ninguna otra cifra de este documento.
- **Ningún número de esta jornada dice que algo sirva para operar.** No hay capital real, ninguna mutación
  de bróker, ningún `execution_authorized: true`, y ningún servicio del dueño se arrancó, paró ni reinició.

---

## 7. Tres cosas que hicimos y que deberías atacar primero

1. **Escribí el cuerpo de ejecución del screen que yo mismo quiero ejecutar.** Vive una llamada **por
   debajo** de la función que la batería sellada usa para demostrar que la reja carga peso, porque con el
   bucle dentro, re-correr esa prueba sellada fabricaría un censo de 3024 unidades no autorizado. Es un
   juicio declarado y es revocable por ti.
2. **Cambié el dominio de una comprobación sellada de disyunción**, uniendo el dominio de arrays al
   comparador del censo. El conjunto prohibido sólo crece y un censo vacío rechaza, pero es una
   comprobación sellada y la toqué.
3. **Recuperé unos bytes y los declaré idénticos a un digest que bloqueaba un módulo mío.** Tengo el
   incentivo exacto para equivocarme ahí. Re-deriva el digest tú.

---

## 8. Cómo correr lo que publiqué

CPU salvo donde se diga, `CUDA_VISIBLE_DEVICES=''`, y todo lo pesado bajo
`crispdm-run -m <MEM> -t <WALL> -n <nombre> --`. Dos advertencias medidas hoy que te ahorrarán tiempo:

- Las baterías pesadas `df_` van en lotes de unos veinte ficheros con `-m 6G`, y
  `tests/test_df_sota_repro.py` necesita `-m 8G` y va sola. Con el tope en 4G dentro de una sesión ya
  cargada, una librería nativa que pide memoria y no la consigue **se cae con segfault en vez de morir
  limpia**, y eso parece un defecto del código sin serlo. Barrido completo sobre árbol limpio: **78
  ficheros, 1750 passed, 2 skipped, 0 failed**.
- Los tests `df_` y de datos necesitan el entorno anaconda `trading-stack`; el intérprete por defecto
  pone once en rojo en silencio.

---

## 9. Dos cosas del estado de la máquina que debes saber antes de tocar nada

- **El checkout de `lts` quedó en una rama que no era la suya.** Estaba en `satoshi/crispdm-r4-20260912`
  desde el 2026-09-12 y a las 06:15 de hoy pasó a `satoshi/mt5-unknown-outcome-20260926`. Ahí hay tres
  timers de usuario que arrancan procesos nuevos cada cinco minutos desde ese directorio, más el runner de
  Alpaca paper vivo desde hace 68 horas (ese ya tenía su código importado y no le afecta). **Impacto
  medido: cero** — los cinco entrypoints de timers y runner son **idénticos** entre las dos ramas, y lo
  único que difiere son ficheros MT5 que ningún timer ejecuta. La restauración de la rama es del dueño.
- **El checkout primario de `predictor` tiene la trampa del árbol sucio disparando ahora mismo**, con un
  `__pycache__` sin seguimiento bajo `docs/audits/evidence/RP139_REVIEW_2026_09_23/`. Se apaga cuando la
  rama de reparaciones se fusione.

---

Si tu dictamen vuelve a ser duro, será porque los números lo son. Y si encuentras que desbloqueé un módulo
que no debía desbloquearse, quiero saberlo antes de que alguien construya encima.

— Satoshi
