# Dictamen Satoshi — Mecanismo de participación, recompensas y autoselección de DOIN

**Fecha:** 2026-09-02
**Autor:** General Satoshi III, en modo arquitecto de sistemas de IA y diseñador de mecanismos.
**Orden:** aclaración autoritativa del dueño (14 puntos) + 12 trabajos solicitados.
**Modo:** solo lectura. No se modificó propuesta, código ni servicio. No se ejecutó nada.
**Insumos leídos:** A inmutable (`predictor@7ae8643`), B local (digests verificados
`b30e911b…`/`151418b9…`), dictamen de Retsu + anexos 01–08, `doin-core@a90bca2`
(PoO, commit-reveal, reputación, quorum, fee market, canales), `doin-node@8bfc64f`
(node, unified, gpu_scheduler), `doin-domains@d01708ae`.

---

## 0. Resumen entendible

El dueño describe una red donde el operador **enchufa un nodo, configura qué
acepta (inferencia/optimización, dominios, límites), y trabaja sin publicar
tarifa ni puja**. El pago llega solo si el trabajo sobrevive reglas
verificables. No habrá token propio.

Ese último punto decide la mitad del dictamen por sí solo: **sin token no
existe "emisión pura" en sentido monetario** — no hay nada que emitir. Toda
recompensa debe salir de fondos reales y conservados (pagos de inferencia,
patrocinio, tesorería, fondos de dominio). Por lo tanto la minería-como-modelo
(opción A) es imposible tal cual; lo que sí es posible — y es lo que el dueño
describe — es una **recompensa publicada y dinámica por dominio y tipo de
trabajo, pagada desde fondos conservados, con autoselección del nodo**
(opción B), donde el umbral del PoO actual sobrevive como **limitador de
tasa**, no como creador de dinero. La subasta (C) queda excluida por el punto
10 del dueño y porque el código de consenso no la tiene.

**Veredicto: HÍBRIDO DELIMITADO — B como régimen central de optimización
(recompensa publicada dinámica desde fondos conservados, autoselección tipo
bandit del nodo), pago directo por respuesta aceptada en inferencia (que no es
A, B ni C: es tarifa de servicio, ya soportada por los canales de pago), cero
subasta en todas partes, y "emisión" reinterpretada como tasa presupuestada.**

La columna doctoral que recomiendo es el **controlador de recompensas entre
dominios frente a una población de nodos que aprenden a autoseleccionarse** —
es la única de las tres candidatas que es a la vez la pregunta del dueño, un
objeto teórico con teorema alcanzable y un sistema que el código actual ya
insinúa (`domain.weight`). El metaoptimizador de valor marginal (Propuesta B
doctoral) queda como programa separado; el scheduler local es el modelo del
agente, no la tesis.

---

## 1. Tres verdades separadas

### 1.1 Qué hace DOIN hoy (hechos de código, no deseo)

| Pieza | `doin-core@a90bca2` / `doin-node@8bfc64f` | Naturaleza |
|---|---|---|
| `ProofOfOptimization` | bloque cuando Σ ponderada de `performance_increment × domain.weight` ≥ umbral; el umbral se ajusta (±12.5%-style, factor 0.25) para un tiempo de bloque objetivo (600 s) | **dificultad de minería con otro nombre**; no paga a nadie — dispara bloques |
| `FeeMarket` | base fee EIP-1559 (quema), tip al generador, stake 5× por optimae, quema 20% si se rechaza, rate limits | antispam + prioridad; **presupone una unidad interna que se quema** |
| `Task` | `OPTIMAE_VERIFICATION` \| `INFERENCE_REQUEST`; el nodo hace **pull** con `GET /tasks/pending?domains=…` | pull por dominios soportados; no hay `Bid` de adquisición |
| Commit-reveal | hash(params+nonce) primero, reveal después | sello anti–front-running; no es oferta |
| `Quorum` | K-de-N evaluadores seleccionados determinísticamente (semilla = tip + optimae id), 2/3, tolerancia 5%, divergentes penalizados | verificación multi-evaluador |
| `Reputation` | EMA por trabajo verificado, penalización asimétrica, computable desde la cadena | historial; no es pago |
| `PaymentChannel` | canales L2 para micropagos de inferencia, depósito "DOIN" bloqueado, disputa | **tarifa de servicio por inferencia ya modelada** |
| `gpu_scheduler` (doin-node) | matcher por recursos/latencia/reputación **y `price_per_hour` en "DOIN", `max_bid`, `price_weight=0.3`** | **vestigio de mercado de cómputo** que contradice la aclaración del dueño |

**Divergencia interna que hay que nombrar:** el código habla tres idiomas
económicos a la vez — consenso tipo minería (PoO), fees tipo Ethereum
(quema/stake **de una unidad interna** que el punto 9 del dueño prohíbe como
token), y un scheduler con precio por hora tipo Akash que el punto 10
prohíbe. No es un defecto fatal: es un prototipo. Pero el doctorado debe
declarar cuál idioma es el mecanismo propuesto y cuáles son vestigios.

### 1.2 Qué desea el dueño (los 14 puntos, reordenados por consecuencia)

1. **Sin token propio (p.9)** ⇒ toda recompensa nace de fondos reales; el
   stake/quema actual debe re-denominarse (activo externo o unidad no
   monetaria) — ver §7.
2. **Sin tarifa ni puja del operador (p.10)** ⇒ no hay subasta de proveedores;
   la variable de decisión del operador es *participar o no, y dónde* — no un
   precio.
3. **Recompensas publicadas por dominio y trabajo (p.11)** ⇒ el protocolo es
   un **fijador de precios de un solo lado** (posted prices), no un mercado
   bilateral.
4. **Configurabilidad total del nodo (p.1–5)** ⇒ allowlist/denylist, tipo de
   trabajo, límites de recursos y rentabilidad, manual o scheduler local.
5. **Sin garantía de trabajo ni pago (p.6)** ⇒ no hay IR garantizada; la
   participación es una apuesta informada por la recompensa publicada.
6. **Pago solo a trabajo/respuesta aceptados (p.7–8)** ⇒ el pago es a
   *output* verificado (Holmström), nunca a esfuerzo.
7. **Fondos de procedencia abierta (p.12–13)** ⇒ inferencia puede financiar
   optimización del mismo dominio — el único puente demanda→oferta del
   sistema.
8. **PoO existente ampliable (p.14)** ⇒ prototipo vs mecanismo propuesto,
   siempre distinguidos.

### 1.3 Qué falta construir para pasar de 1.1 a 1.2

| Brecha | Hoy | Debe existir |
|---|---|---|
| **Pagador de optimización** | PoO dispara bloques; nadie cobra por incremento | regla de pago publicada π(d, tipo de trabajo) contra fondo F_d, con conservación |
| **Fondos por dominio** | no existen | ledger F_d: entradas (inferencia %, patrocinio, tesorería) y salidas (pagos, reservas de verificación, disputas) |
| **Re-denominación de stake/quema** | unidad interna implícita | depósito en activo externo; "quema" → confiscación al fondo/reserva, no destrucción |
| **Política del operador** | pull por dominios | allowlist/denylist + límites + rentabilidad mínima, aplicados ANTES del pull |
| **Scheduler local** | matcher con precio (vestigio) | bandit sobre (dominio × tipo) permitidos, sin precio — §5 |
| **Controlador entre dominios** | `domain.weight` estático | actualización publicada de π_d / pesos con anti-yacimiento-fácil — §8 y la tesis |
| **Suavizado de varianza** | nada (récord o nada implícito) | pago por evaluación verificada + bono de campeón amortizado — §6 |
| **Simulador económico** | no existe | el experimento doctoral ES un simulador; DOIN es runtime opcional |

---

## 2. Modelo formal mínimo

**Dominios** d ∈ D, cada uno con fondo F_d(t) ≥ 0, campeón q_d*, costo de
verificación κ_d, y peso/recompensa publicada.

**Nodos** i con: costo privado c_i,d por unidad de trabajo en d (electricidad,
oportunidad, afinidad de hardware), capacidad r_i, política de participación
P_i ⊆ D × {inf, opt} (allowlist/denylist ∩ límites), y regla de decisión
(manual o scheduler local).

**Trabajos.** Optimización: producir un optimae (x, q̂) para d; una
*evaluación verificada* es la re-ejecución/lookup con sello válido.
Inferencia: responder una solicitud; *aceptada* = pasa la validación del
solicitante/protocolo.

**Recompensas publicadas** (decisión del protocolo, no del nodo):
- π_d^eval: pago por evaluación solicitada y reproducible en d;
- π_d^rec: bono por mejora verificada del campeón (Δq_d > margen de
  originalidad), amortizado (§6);
- π_d^inf: pago por respuesta de inferencia aceptada (financiado por el
  consumidor vía canal, no por F_d).

**Conservación (invariante dura):** para todo t,
`pagos_d(t) + reservas_d(t) ≤ F_d(0) + entradas_d(t)`. Ninguna recompensa
aparece de la nada. La "tasa de emisión" es un **límite de gasto por época**
γ_d(t) ≤ F_d(t)/horizonte, y el umbral PoO es el limitador de ritmo de
aceptación, no una fuente de dinero.

**Utilidad del nodo:** U_i = Σ_d E[π recibida en d] − c_i,d·(trabajo) −
riesgo de confiscación de depósito. Participa donde E[U] ≥ 0 dentro de sus
límites (p.6: sin garantía).

**Objetivo del protocolo (la tesis):** elegir {π_d(t)} bajo conservación para
maximizar el valor verificado total Σ_d V_d(Δq_d verificado), donde V_d
refleja la prioridad del dominio (p. ej. su flujo de inferencia), evitando
(i) colapso al dominio barato, (ii) rentabilidad de fabricar, y (iii) pagar
dos veces el mismo incremento.

**Aceptación (sello):** commit-reveal + quorum K-de-N + re-ejecución/lookup;
el pago es función del sello, jamás de q̂ reportado.

---

## 3. Comparación de los cuatro regímenes

### Tabla comparativa

| Dimensión | A · Emisión pura | B · Recompensa publicada dinámica | C · Subasta de proveedores | D · Híbrido delimitado (recomendado) |
|---|---|---|---|---|
| Actores | nodos, protocolo | nodos, protocolo (fija π), financiadores de F_d | consumidores, oferentes, subastador | nodos, protocolo, financiadores, consumidores de inferencia |
| Info pública | umbral, campeón | π_d por trabajo, F_d agregado, campeón, umbral | pujas (selladas o no), tarea | π_d, F_d, campeón, umbral; tarifas de inferencia |
| Info privada | costos del nodo | costos del nodo, elasticidad | costos (revelados por puja) | costos del nodo; demanda del consumidor |
| Origen del dinero | **inflación del token** | fondos conservados F_d | presupuesto del consumidor | F_d (optimización) + pago del consumidor (inferencia) |
| Selección | autoselección total | autoselección guiada por π_d | el subastador elige ganadores | autoselección (bandit local) guiada por π_d |
| Pago | por bloque/incremento aceptado | π_d publicado por trabajo aceptado | precio de la puja ganadora | §6: por evaluación verificada + bono récord amortizado; inferencia por respuesta aceptada |
| Verificación | sello (o nada: hash) | sello obligatorio (re-ejecución/lookup + quorum) | según contrato | sello obligatorio; commit-reveal; quorum |
| Ataques típicos | basura disfrazada, yacimiento fácil, Sybil por bloque | manipulación del controlador, congestión del dominio rico, Sybil, replay | colusión de pujas, shill bids, adverse selection | los de B + manipulación de demanda de inferencia (§8.4) |
| Ventajas | UX minera simple; sin negociación | respeta p.6/10/11; presupuestable; dirigible por dominio | revela costos; eficiencia asignativa clásica | una sola vocabulario; demanda real financia oferta (p.13) |
| Defectos | **imposible sin token (p.9)**; paga trabajo sin valor si el sello es débil; varianza brutal | requiere estimar elasticidades; controlador manipulable si mal diseñado | **prohibida por p.10**; no existe en el código; carga UX al operador | complejidad de dos regímenes coexistiendo (mitigada porque no comparten fondo) |
| Pertinencia doctoral en IA | baja (economía monetaria) | **alta si el objeto es el controlador frente a agentes que aprenden** | media-baja (teoría de subastas madura; Myerson ya está escrito) | la de B |

### Detalle decisivo por alternativa

**A — Emisión pura.** Modelo mental ETH-PoW: conectas GPU, el protocolo
emite. Requiere una moneda cuya inflación pague el trabajo. El punto 9 del
dueño la prohíbe. Sin token, "emitir" es gastar un fondo — y entonces ya
estás en B con peor nombre. Además, la emisión pura paga el *ritmo* de
incrementos, no su *valor relativo entre dominios*: el umbral global del PoO
actual suma incrementos ponderados de todos los dominios en una sola olla, y
por eso no puede, por sí solo, castigar el yacimiento fácil (§8.1). A muere
por p.9 y por el teorema que la tesis debe escribir.

**B — Recompensa publicada y dinámica.** El protocolo publica π_d por tipo
de trabajo; el nodo se autoselecciona. Es exactamente los puntos 10–11 del
dueño. Es también, con honestidad bibliográfica, el vecindario de los
*posted-price mechanisms* y del control basado en mercado; y en el mundo
cripto-IA, el vecino incómodo es el reparto dinámico de emisión entre
subredes de Bittensor (dTAO) — **a diferenciar con búsqueda bibliográfica
antes de reclamar novedad** (ellos dirigen inflación de un token por señales
de staking; aquí se dirige un presupuesto conservado por valor verificado;
la diferencia es real pero hay que citarla, no proclamarla). B es viable,
compatible con el código (π_d generaliza `domain.weight`) y deja un objeto
doctoral: el controlador.

**C — Subasta.** Prohibida por p.10 (el operador no puja), ausente del
consenso (`Bid`/`Lease` no existen en doin-core; el matcher con precio de
doin-node es un vestigio a retirar o re-etiquetar), y doctoralmente el
terreno más ocupado (Myerson; subastas de procurement; Akash como sistema).
Se descarta entera. Único resquicio admisible: si algún día un *consumidor*
externo quiere pagar por una tarea dirigida, eso es un contrato de servicio
aparte — no el mecanismo de la red.

**D — Híbrido delimitado.** B para optimización + tarifa-por-servicio para
inferencia (que no es subasta: el protocolo publica la tarifa o el consumidor
la acepta del menú; el operador sigue sin pujar) + puente p.13 (un % del
ingreso de inferencia del dominio d alimenta F_d). La delimitación es nítida
porque **los dos regímenes no comparten fondo ni regla de pago**: inferencia
la paga quien consume; optimización la paga F_d. El único acople es el
flujo de financiamiento, que es una tubería contable, no un mecanismo.

---

## 4. Flujo completo del nodo (diagrama textual)

```
ARRANQUE
  │ lee config del operador:
  │   modo ∈ {inferencia, optimización, ambas, ninguna}
  │   allowlist / denylist de dominios  →  D_i permitido
  │   dominio fijo  ó  autoselección
  │   límites: CPU/GPU/mem/energía/tiempo/rentabilidad_mínima
  │   participación: manual | scheduler local
  ▼
BUCLE (mientras límites lo permitan)
  │ 1. SINCRONIZA cabecera de cadena + tabla publicada:
  │      {π_d^eval, π_d^rec, π_d^inf, F_d visible, campeón q_d*, umbral}
  │ 2. FILTRA dominios: D_i ∩ (F_d > 0) ∩ (π esperada ≥ rentabilidad_mínima
  │      estimada con SU costo local privado)   ← el costo nunca se publica
  │ 3. SELECCIONA (dominio, tipo):
  │      manual → el operador fijó (d, tipo)
  │      scheduler → bandit sobre brazos permitidos (§5)
  │ 4a. INFERENCIA: pull de INFERENCE_REQUEST del dominio
  │      → ejecuta → respuesta → validación del protocolo/solicitante
  │      → ACEPTADA: cobra π_d^inf vía canal de pago   → 8
  │      → RECHAZADA: sin pago; reputación decae       → 8
  │ 4b. OPTIMIZACIÓN: trabaja el dominio d
  │      → produce optimae (x, q̂)
  │ 5. COMPROMISO: deposita garantía (activo externo, §7);
  │      OPTIMAE_COMMIT = hash(params+nonce) a la cadena
  │      espera confirmación → OPTIMAE_REVEAL
  │ 6. VERIFICACIÓN (la hacen OTROS nodos, seleccionados por quorum
  │      determinístico K-de-N; el proponente jamás se autoverifica):
  │      re-ejecución / lookup del dominio + tolerancia + mediana
  │      → sello VÁLIDO / INVÁLIDO
  │      (servir verificaciones ES trabajo pagado: π_d^eval)
  │ 7. LIQUIDACIÓN:
  │      sello válido y Δq_d > margen  → π_d^rec amortizado (§6) + devolución de garantía
  │      sello válido sin récord       → π_d^eval si fue evaluación solicitada; si fue
  │                                      intento de récord: solo devolución (o micro-pago
  │                                      por evidencia, si el fondo lo publica)
  │      sello inválido                → confiscación parcial de garantía AL FONDO/reserva
  │      todo movimiento: contra F_d, jamás creación   ← invariante §2
  │ 8. ACTUALIZA ESTADÍSTICAS:
  │      protocolo: reputación (EMA, asimétrica), contadores de aceptación,
  │        F_d, y el CONTROLADOR recalcula π_d(t+1) (§8)
  │      nodo: recompensa/costo observados por brazo → posterior del bandit
  ▼
  vuelve a 1  (o se detiene: límites de energía/tiempo/rentabilidad)
```

Notas de contrato: (p.6) el pull puede volver vacío — no hay garantía de
trabajo; (p.7–8) los únicos eventos que pagan son sello válido o respuesta
aceptada; el paso 2 usa el costo privado localmente — nunca se transmite.

---

## 5. Scheduler local mínimo

El problema del nodo es un **bandit**: brazos = (dominio permitido × tipo de
trabajo), recompensa = pago realizado − costo local, no estacionario (π_d y
congestión cambian). Comparación:

| Política | Qué hace | Ventaja | Defecto | Veredicto |
|---|---|---|---|---|
| Manual | el operador fija (d, tipo) | soberanía total; cero código | ignora cambios de π y congestión | debe existir siempre (p.5) |
| Codiciosa E[π]/costo | elige el brazo de mejor razón esperada con estimación puntual | simple, entendible | sin exploración: se clava en un dominio y no ve mejoras; frágil a no-estacionariedad | base de comparación |
| **UCB / Thompson (recomendada: Thompson con descuento)** | posterior por brazo sobre "pago neto por hora"; muestrea y elige; descuento exponencial para no-estacionariedad | explora sola; óptima en regret para este tamaño de problema; 30 líneas de código | necesita unas decenas de muestras por brazo | **mínimo suficiente** |
| Bandit contextual | añade contexto (tamaño del trabajo, hora, temperatura GPU, precio de energía) | captura heterogeneidad fina | más datos, más superficie de bugs; el contexto útil aquí es corto | solo si la ablación muestra que Thompson simple pierde ≥X% frente a un oráculo |

**No se propone RL.** El estado relevante del nodo (posterior por brazo) no
tiene dinámica que justifique un MDP; Thompson con descuento cubre la
no-estacionariedad observada. Esta jerarquía además da a la tesis su modelo
de agente: la población de nodos se modela como bandits heterogéneos en
costos — el controlador (§8) se evalúa **contra esa población**, no contra
agentes omniscientes.

---

## 6. Recompensas separadas y reducción de varianza

1. **Inferencia — π_d^inf por solicitud aceptada.** Tarifa de servicio
   publicada por el protocolo/dominio, liquidada por canal de pago
   (el módulo ya existe). Fuente: el consumidor. Un porcentaje ρ_d
   (parámetro de gobernanza, p.13) se desvía a F_d.
2. **Optimización — π_d^eval por evaluación solicitada y reproducible.**
   El grueso del ingreso estable del minero honesto: servir verificaciones
   de quorum y evaluaciones encargadas por el protocolo (p. ej. re-ejecutar
   candidatos, poblar curvas). Pagadero por pieza contra F_d. Esto convierte
   "aportar GPU" en un oficio de renta razonablemente predecible.
3. **Bono de campeón — π_d^rec por mejora verificada.** El premio del
   concurso: Δq_d verificado sobre el campeón con margen de originalidad
   (anti-copia: el margen + commit-reveal + linaje del campeón hacen que
   re-empaquetar al campeón no pague).
4. **Reducción de varianza del "solo pagan los récords".** Pagar únicamente
   nuevos óptimos es una lotería (varianza de minería en solitario). Tres
   mecanismos, componibles, análogos a esquemas de pool de minería (PPS /
   PPLNS) que hay que citar como antecedente, no redescubrir:
   - **Pieza base (PPS-like):** π_d^eval por evaluación verificada — ingreso
     por trabajo útil aunque no haya récord (la evaluación ES útil: puebla
     el mapa del dominio).
   - **Bono amortizado (PPLNS-like):** π_d^rec no se paga 100% al autor del
     récord: fracción λ al autor, fracción (1−λ) prorrateada entre las
     últimas W evaluaciones verificadas del dominio (ventana). Reduce
     varianza, recompensa la exploración que habilitó el récord, y de paso
     desincentiva el hoarding de evidencia.
   - **Hitos intermedios:** el protocolo puede publicar micro-bonos por
     "evidencia nueva verificada" (cubrir región no explorada del espacio,
     con sello), acotados por época para que no sean minables como spam.
   Parámetros (λ, W, tasas) los fija el controlador bajo conservación.

---

## 7. Conservación económica sin token propio

**Principio:** cada unidad pagada tiene una procedencia trazable; el sistema
es un contable, no un banco central.

- **Fondo por dominio F_d.** Entradas: (i) depósitos de consumidores de
  inferencia → % ρ_d al fondo (p.13); (ii) patrocinadores del dominio
  (quien quiere que exista un buen modelo de d, paga por su optimización);
  (iii) tesorería general (asignaciones explícitas, p.12). Salidas: π_d^eval,
  π_d^rec amortizado, reserva de verificación, reserva de disputas.
- **Verificadores.** π_d^eval a los miembros de quorum con voto no
  divergente; los divergentes pierden el pago y reputación. La reserva de
  verificación garantiza que aceptar un optimae SIEMPRE tiene presupuesto de
  quorum: si F_d no cubre κ_d·K, el dominio no publica π_d^rec (se declara
  inactivo para récords — mejor que aceptar sin verificar).
- **Disputas.** Ventana de disputa post-liquidación (el módulo de canales ya
  tiene el patrón): quien prueba un sello inválido cobra de la reserva de
  disputas + parte de la garantía confiscada del infractor.
- **Stake/garantía sin token.** El depósito del optimae se re-denomina en el
  activo externo de liquidación. La "quema" del prototipo (20%) se convierte
  en **confiscación**: mitad a la reserva de disputas, mitad al F_d (nunca
  destrucción — sin token no hay deflación que hacer, y destruir fondos
  reales es regalarle dinero al custodio).
- **Invariante publicable:** por época, `Σ pagos + Δreservas = Σ entradas −
  ΔF_d`, auditable desde el ledger. El PoO limita el RITMO de aceptación
  (los bloques), el controlador limita el GASTO (γ_d). Ninguna recompensa
  aparece de la nada; cuando F_d se seca, π_d → 0 y la autoselección drena
  el dominio — ese es el comportamiento correcto, no un fallo.
- **Nota de honestidad doctoral:** en la tesis todo esto corre en un
  **simulador con unidades contables simuladas**; ningún experimento ejecuta
  operaciones financieras reales (coherente con la ética de B).

---

## 8. El problema del dominio fácil

**8.1 Diagnóstico con el código actual.** El PoO suma incrementos ponderados
en una sola olla global. Si el dominio e es barato de mejorar (lookup, ruido
alto, campeón joven), los nodos racionales migran a e; el umbral global sube
para mantener el tiempo de bloque, y eso **encarece los bloques para todos
los dominios por igual** — castiga al dominio difícil por la fiesta del
fácil. `domain.weight` existe pero es estático: nadie lo gobierna. Este es el
hallazgo técnico central y la semilla del teorema doctoral.

**8.2 Concentración y dificultad desigual.** El controlador debe publicar
π_d (o pesos) de modo que el **valor marginal verificado por unidad de
presupuesto se iguale entre dominios activos** (condición tipo
water-filling). Dominios con campeón maduro (Δq escaso) deben pagar más por
unidad de mejora o redirigir su fondo a π^eval (mapear el espacio) en vez de
π^rec. La no-estacionariedad (cada récord vuelve más difícil el siguiente)
exige un controlador adaptativo, no pesos de génesis.

**8.3 Costos diferentes.** κ_d (verificación) y costo de intento difieren
por órdenes de magnitud entre lookup y re-entreno. Regla dura: π_d^rec ≥
κ_d·K + margen o el dominio no acepta récords (§7). El controlador ve los
costos de verificación (públicos: los paga F_d); NO ve los costos privados
de los nodos — solo su respuesta agregada (elasticidad). Esa asimetría es la
que hace el problema interesante y no trivial.

**8.4 Manipulación de demanda.** Si ρ_d (inferencia→fondo) dirige el
presupuesto, un atacante puede inflar demanda de inferencia sintética del
dominio donde él mina, para engordar F_d y cosecharlo. Mitigaciones: la
inferencia sintética CUESTA (paga tarifa completa; el atacante recupera a lo
sumo ρ_d < 1 de lo que gasta — pérdida neta garantizada si ρ_d < 1 y hay
otros mineros); techo de crecimiento de F_d por época; señal de demanda
suavizada. Esto debe quedar como proposición, no como fe.

**8.5 Sybil.** Sin token no hay airdrop que cazar; las vías Sybil restantes:
(i) multiplicar identidades para capturar cuota de quorum → mitigado porque
el quorum paga por trabajo verificado (una identidad extra sin GPU extra no
gana más) y la selección es determinística por cadena; (ii) ventana PPLNS
del bono amortizado → prorratear por *evaluación verificada*, no por
identidad; (iii) reputación inicial neutra sin privilegios (nada que farmear).
El costo de identidad = garantía mínima por participación activa. Declarar,
como en A, que el ledger no elimina Sybil: lo vuelve improductivo.

**8.6 Copia y replay.** Commit-reveal (ya existe) + margen de originalidad
sobre el campeón + linaje (un optimae que re-etiqueta parámetros del campeón
con perturbación ε no supera el margen) + registro del primer commit gana
empates. Replay entre dominios/épocas: el sello liga (dominio, campeón
vigente, época) — un incremento válido ayer no es liquidable hoy.

**8.7 Goodhart del proxy.** Si el sello es lookup/holdout fijo, los mineros
optimizan el holdout (la vulnerabilidad clase-3 de la maestría, que
doin-domains WP3 ya midió con Thresholdout/Ladder en la mira). Mitigación
doctoral: presupuesto de consultas por identidad + rotación/holdout oculto +
el resultado WP3-C (el nulo del régimen lineal) como línea base de qué
ataques NO aterrizan. El dominio confirmatorio de re-entreno evita que toda
la tesis viva en oráculos de tabla.

---

## 9. Topologías

| Topología | Pros | Contras | Veredicto |
|---|---|---|---|
| **Una blockchain común** (actual) | seguridad y liquidez de participación unificadas; simple; N real de nodos es pequeño; el controlador ve todos los dominios (lo necesita) | throughput compartido; gobernanza de dominios acoplada | **RECOMENDADA hoy** |
| Blockchain soberana por dominio | aislamiento total de fondos y reglas | fragmenta la seguridad (un dominio chico es atacable), multiplica infraestructura, mata al controlador global, Sybil más barato por cadena | rechazada como punto de partida |
| Liquidación común + módulos/rollups por dominio | fondos y reglas por dominio aislados, seguridad común, el controlador vive en la capa de liquidación | complejidad de puentes; prematuro | **destino de migración** |

**Condiciones concretas de migración (común → liquidación+módulos):**
1. Congestión sostenida: tiempo de inclusión de transacciones > objetivo
   durante ≥N épocas por competencia ENTRE dominios (no por un pico).
2. Un dominio requiere reglas de verificación con cadencia/privacidad
   incompatibles con el bloque común (p. ej. datos regulados).
3. F_d de un dominio supera el umbral donde su compromiso contable necesita
   aislamiento auditable propio (patrocinador institucional que lo exige).
4. Dos dominios necesitan tiempos de bloque objetivo distintos en >1 orden
   de magnitud.
Hasta que una de esas condiciones se dispare con números, quedarse en la
cadena común. Jamás pasar por la fase "una cadena soberana por dominio".

---

## 10. Las tres columnas doctorales candidatas

| Columna | Qué es | Fortaleza | Debilidad fatal como tesis |
|---|---|---|---|
| Metaoptimizador de valor marginal | estima qué trabajo produciría más valor (la Propuesta B doctoral + hipótesis de fusión) | mejor encaje AutoML puro | Retsu lo demostró: no es el flujo del dueño; reintroduce el asignador-proxy (P0-1) y el corpus envenenable; compite frontalmente con OptFormer/HyperBO |
| Scheduler local que aprende | bandit del §5 | necesario, medible | **demasiado pequeño**: Thompson sobre 10 brazos es un capítulo, no un doctorado; el jurado 1 lo despacha como ejercicio |
| **Controlador de recompensas entre dominios** | publica π_d(t) bajo conservación frente a una población de nodos-bandit estratégicos, con sello que hace la basura no rentable | ES la pregunta del dueño (p.11); teorema alcanzable (§11); el código ya tiene el mango (`domain.weight`, umbral); el scheduler local se vuelve el modelo del agente y el metaoptimizador ni se necesita | encaje IA exige el marco correcto: *dirigir poblaciones de agentes que aprenden con incentivos publicados* — si se escribe como tokenomics, es Ingeniería |

**Elección: el controlador de recompensas entre dominios.** El scheduler
local queda como infraestructura + modelo de agente (y ablación: población
manual vs codiciosa vs Thompson). El metaoptimizador queda FUERA de esta
tesis — programa aparte (la Propuesta B doctoral sobrevive como alternativa
independiente si el programa prefiere AutoML; no se fusionan).

---

## 11. Propuesta de núcleo doctoral (sin redactar propuesta todavía)

**Pregunta madre.** ¿Bajo qué reglas publicadas de recompensa por dominio y
de verificación —con presupuesto conservado, sin token propio y sin que el
operador fije precio ni puje— una población heterogénea de nodos que
aprenden a autoseleccionarse asigna su cómputo de modo que el valor de
optimización **verificado** por unidad de presupuesto se aproxime al máximo
alcanzable, sin que fabricar incrementos, copiar campeones o saturar el
dominio barato sea una mejor respuesta?

**Objetivos (3).**
1. Formalizar el juego controlador–población: fondos conservados F_d,
   recompensas publicadas π_d, nodos-bandit con costos privados, sello de
   re-ejecución/lookup con costo κ_d; y el mapa sello→no-rentabilidad de la
   fabricación.
2. Diseñar y analizar el controlador (regla tipo water-filling adaptativa
   sobre valor marginal verificado por unidad de gasto) y las reglas de pago
   del §6 (pieza + bono amortizado), con sus condiciones de conservación.
3. Evaluar en simulador con atribución: dominios oráculo-1 (NATS/HPO-B como
   lookup, reutilizando doin-domains) + UN dominio de re-entreno barato;
   población de bandits heterogéneos; adversarios de repertorio fijo + mejor
   respuesta contra las reglas publicadas.

**Hipótesis (3).**
- **H1 · Participación racional.** La participación por dominio responde a
  π_d con elasticidad medible y los nodos con E[π]−c<0 salen; bajo la regla
  de pieza+bono, la varianza del ingreso de un nodo honesto cae ≥X% vs
  pagar solo récords, sin cambiar el total gastado (X del piloto).
- **H2 · Anti-yacimiento-fácil.** Con costos de verificación y dificultades
  desiguales, la regla de umbral global (PoO actual) produce una brecha
  documentada de valor verificado por unidad de gasto; el controlador la
  reduce hasta un factor (1+ε) del oráculo de asignación con elasticidades
  conocidas — o se caracteriza la región donde ninguna regla publicada lo
  logra. *El PoO actual puede FALLAR H2: eso es un resultado, no un
  fracaso.*
- **H3 · Basura no rentable.** Bajo sello + garantía confiscable + quorum,
  la mejor respuesta encontrada (fabricar, copiar+ε, replay, inflar demanda
  de inferencia con ρ_d<1) tiene utilidad ≤0 dentro del presupuesto de
  ataque declarado; el atacante de mejor respuesta se ejecuta contra reglas
  congeladas y publicadas.

**Resultado teórico realista.** (i) Proposición de conservación + IR;
(ii) teorema water-filling aproximado: si la respuesta de la población a
π es monótona con elasticidades acotadas conocidas, el controlador iguala
valor marginal verificado por unidad de gasto hasta (1+ε), con tasa de
convergencia de su versión online; (iii) **región de imposibilidad**: con
verificación no identificable (sello degradado) o presupuesto de
verificación insuficiente (F_d < κ_d·K), ninguna regla publicada evita el
colapso al dominio barato o el pago a basura — la frontera exacta es el
aporte. No se promete equilibrio del lazo cerrado población-controlador:
análisis por ventanas estacionarias, declarado en la pregunta.

**Experimento mínimo con atribución.** Simulador CPU determinista por
semillas. Brazos de regla: {umbral global PoO (prototipo), pesos estáticos,
controlador dinámico} × {sin adversario, con adversarios}; controles:
asignación aleatoria de π y oráculo omnisciente (cota superior); población:
N nodos Thompson con costos muestreados, +braz manual/codicioso como
ablación del agente. ≥30 réplicas por celda, unidad = campaña simulada,
piloto congela potencia, Holm. Cada brazo difiere del vecino en UN
componente — la atribución es por construcción.

**Plan de recorte (3 años).** Año 1: formalización + simulador + réplica del
diagnóstico §8.1 sobre el prototipo (paper de marco). Año 2: controlador +
teoremas + adversarios (paper de mecanismo). Año 3: dominio de re-entreno +
mejor respuesta + integración opcional con doin-node como validación de
viabilidad (paper aplicado + tesis). **Fuera:** token, mainnet, LLM, jueces,
metaoptimizador, equilibrio del lazo, Byzantine completo (quorum
honesto-mayoritario se asume y se declara), economía de inferencia como
mercado (solo como fuente parametrizada ρ_d), comparación ejecutada contra
Bittensor/Gensyn/Akash (solo posicionamiento).

---

## 12. ¿Aplica Hayek?

**Veredicto: no en el sentido de 1945; sí existe UNA variable con función de
precio, y es del protocolo, no del mercado.** Sin retórica:

- **Variable candidata:** π_d(t), la recompensa publicada por unidad de
  trabajo verificado en el dominio d (y su gemela π_d^eval).
- **Qué información transmite:** escasez relativa de progreso verificado por
  unidad de presupuesto entre dominios — cuánto valora HOY el sistema una
  GPU-hora en d frente a d′. Del lado del nodo, agrega lo que el nodo NO
  puede ver (fondos, demanda de inferencia, madurez del campeón en otros
  dominios).
- **Cómo se forma:** no por puja bilateral (no la hay), sino por la regla
  del controlador: entradas de F_d (donde el % de inferencia SÍ acarrea una
  señal de demanda real de consumidores — el único precio genuino del
  sistema es la tarifa de inferencia que alguien paga) + respuesta observada
  de la oferta (elasticidad de la población). Es un **posted price de un
  solo lado**, familia de mecanismos de precios publicados — no el orden
  espontáneo hayekiano.
- **Qué observación refutaría la lectura "precio":** (a) elasticidad cruzada
  nula — si la asignación de la población NO responde a cambios publicados
  de π_d (los nodos siguen su afinidad de hardware y ya), entonces π no
  transmite información a nadie y es solo una tasa contable; el experimento
  de H1 la mide directamente; (b) si π_d se determina solo por la
  contabilidad de F_d sin correlación con el valor marginal verificado,
  entonces es un dividendo, no un precio. Ambas refutaciones son medibles en
  el simulador — así se cita a Hayek: como hipótesis operacionalizada y
  refutable, o no se cita.

**Instrucción de redacción derivada:** Hayek 1945 no va en el lead de ningún
documento; si sobrevive, va en una subsección que declare exactamente lo
anterior. En esto Retsu tiene razón y su anexo 08 §10 queda ratificado con
una precisión: el sistema no carece de precios — tiene DOS (tarifa de
inferencia, pagada por consumidores reales; y π_d, publicado por el
protocolo); lo que no tiene es formación de precios por los operadores.

---

## 13. Objeciones letales (contra mi propia recomendación)

1. **"Esto es Ingeniería, no IA."** Si el PDF futuro se escribe como
   tokenomics/blockchain, el jurado 1 tiene razón. La única defensa: el
   objeto es *dirigir poblaciones de agentes que aprenden mediante
   incentivos publicados* (mecanismos frente a no-regret learners, línea
   activa de aprendizaje multiagente), y ≥60% de las páginas deben ser
   agentes, elasticidades, regret y experimentos — no fees ni bloques.
   Si el programa no traga ni eso, el repliegue es KEEP B (Propuesta B
   doctoral, AutoML), y este mecanismo se publica como paper de sistemas.
2. **"Su vecino es dTAO de Bittensor."** El reparto dinámico de emisión
   entre subredes existe en producción. Diferencias reales (presupuesto
   conservado sin token; objetivo = valor VERIFICADO por re-ejecución, no
   señales de staking; teoremas de imposibilidad), pero **la novedad no se
   afirma sin la pasada bibliográfica formal** (dTAO, mecanismos de posted
   prices, market-based control, incentivized exploration, PPS/PPLNS). Este
   dictamen no la ejecutó: queda como tarea previa a cualquier propuesta.
3. **"Economía de simulador."** Elasticidades y costos son muestreados por
   el candidato; el adversario de mejor respuesta mitiga pero no elimina la
   circularidad. Defensa parcial: calibrar rangos de costos con datos
   públicos de energía/GPU y declarar la validez como *de mecanismo bajo
   supuestos*, no *de mercado real*. El jurado 2 aceptará la modestia; no
   aceptará la omisión.
4. **"El teorema fácil y el difícil."** Water-filling con elasticidades
   conocidas es estándar (el jurado 2 lo llamará ejercicio); el valor está
   en el lado online/imposibilidad. Si la región de imposibilidad no sale
   nítida, la teoría queda delgada — plan B: convertir H2 empírica en el
   resultado principal con el prototipo PoO como contraste (la brecha
   documentada del sistema real es publicable por sí misma).
5. **"Sus dominios lookup trivializan H3."** Fabricar una fila que no está
   en la tabla se detecta con probabilidad 1 (Retsu, anexo 04 §"puente").
   Por eso el dominio de re-entreno barato NO es opcional: es donde κ_d>0
   hace que p, garantías y reservas signifiquen algo. Si el año 3 lo
   recorta, H3 queda coja y hay que decirlo.

---

## 14. Veredicto (formato ordenado)

**HÍBRIDO DELIMITADO**, con esta delimitación exacta y ninguna otra:

- **Optimización: RECOMPENSA PUBLICADA Y DINÁMICA (régimen B)** — π_d por
  evaluación verificada + bono de campeón amortizado, financiada por fondos
  conservados por dominio, autoselección del nodo vía Thompson local,
  aceptación por commit-reveal + quorum + re-ejecución/lookup.
- **Inferencia: pago directo por respuesta aceptada** (tarifa de servicio
  vía canales — no es A, ni B, ni C), con % ρ_d al fondo del dominio.
- **EMISIÓN PURA: imposible** bajo la prohibición de token (p.9); el PoO
  sobrevive como limitador de ritmo y objeto de estudio (H2), no como
  fuente de dinero.
- **SUBASTA: excluida** en ambos lados (p.10; sin `Bid` en consenso; el
  matcher con precio de doin-node se declara vestigio a retirar).
- **Columna doctoral: el controlador de recompensas entre dominios** frente
  a población de nodos que aprenden; scheduler local = modelo de agente e
  infraestructura; metaoptimizador = programa separado (la Propuesta B
  doctoral no se fusiona; queda como alternativa íntegra si el programa
  exige AutoML puro).

Esto es consistente con el flujo que el dueño describió, con el código que
existe, con el dictamen M de Retsu (lo refina: "emisión" → "recompensa
publicada presupuestada", porque sin token la emisión no existe), y deja una
tesis con teorema, simulador y falsación.

---

## 15. Preguntas que requieren decisión del dueño

1. **Ratificación del reencuadre M→B:** ¿acepta que, sin token (p.9), la
   "emisión tipo minería" se formalice como recompensa publicada desde
   fondos conservados (la opción B con UX de minero)? Es la piedra angular
   de este dictamen.
2. **Columna doctoral:** ¿controlador de recompensas (este dictamen) o
   KEEP B/metaoptimización (programa AutoML)? Retsu exigió marcar UNA
   casilla; este dictamen vota controlador, pero la casilla es suya.
3. **Denominación de garantías y pagos** en el mundo real (el simulador usa
   unidades contables): ¿activo externo único? ¿cuál rail de liquidación se
   asume como supuesto declarado?
4. **Bono amortizado:** ¿acepta que el récord NO pague 100% al autor
   (fracción λ al autor, resto a la ventana de evaluadores verificados)?
   Reduce varianza a costa de diluir al ganador — es una decisión de
   filosofía de la red, no técnica.
5. **ρ_d (porcentaje de inferencia al fondo de optimización):** ¿parámetro
   de gobernanza por dominio o constante de protocolo?
6. **El matcher con `price_per_hour` de doin-node:** ¿se retira, o se
   re-etiqueta como enrutador de latencia/recursos sin precio? Hoy
   contradice el punto 10 en el propio repositorio.
7. **Límite de rentabilidad del operador (p.4):** ¿se aplica solo localmente
   (recomendado: el costo privado jamás se transmite) o se permite
   publicarlo opcionalmente? Publicarlo filtra información de costos y
   reabre la puerta de mercado que usted cerró.
8. **Tiempo de bloque objetivo y horizonte de gasto γ_d:** constantes que el
   piloto debe congelar — ¿las fija usted o el piloto?

---

*Dictamen entregado. No se redactó propuesta nueva; no se modificó archivo
preexistente alguno; este archivo es el único artefacto creado y queda sin
comprometer en git, a la espera de la palabra del dueño.*
