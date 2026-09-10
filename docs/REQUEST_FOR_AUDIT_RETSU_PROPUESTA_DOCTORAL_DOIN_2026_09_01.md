# Solicitud de auditoria para Retsu: propuesta doctoral DOIN

**Fecha:** 2026-09-01  
**Solicitante:** General Musashi, por instruccion del propietario  
**Auditor solicitado:** General Retsu  
**Tipo de revision:** adversarial, conceptual, academica y de trazabilidad  
**Estado:** `REQUESTED`

## 1. Artefactos bajo auditoria

- Propuesta en PDF: [`propuesta_doctoral_doin_borrador.pdf`](propuesta_doctoral_doin_borrador.pdf)
- Fuente HTML: [`propuesta_doctoral_doin_borrador.html`](propuesta_doctoral_doin_borrador.html)

La propuesta debe poder entenderse sin conocimiento previo de DOIN y no puede apoyarse en conversaciones privadas como evidencia.

## 2. Tesis que el documento debe representar

El objeto central es **disenar un mecanismo para asignar, validar y remunerar trabajos descentralizados de inferencia y optimizacion** entre consumidores, trabajadores y evaluadores estrategicos.

Fronteras conceptuales ordenadas por el propietario:

1. La demanda publica una tarea, presupuesto y protocolo de aceptacion.
2. La oferta se auto-selecciona y compite mediante precios y compromisos verificables.
3. La seleccion automatica de dominio, pipeline o trabajo puede existir como estrategia local de un oferente que busca rentabilidad, pero **no es el centro de la investigacion**.
4. El precio se estudia como senal de conocimiento economico disperso en el sentido de Hayek; no prueba que un resultado sea correcto.
5. La validacion entre pares debe resistir sobreajuste, copia, Sybil, evaluadores perezosos y colusion.
6. La teoria de la informacion debe cumplir una funcion operacional, en particular medir evidencia marginal o redundante; no debe usarse como analogia ornamental.
7. El ledger registra compromisos, orden, autoria, dictamenes, disputas y liquidacion. No actua como oraculo de calidad.
8. No se propone emitir un token. Los pagos pueden liquidarse con activos o redes existentes.
9. La contribucion doctoral generaliza la prueba de trabajo de optimizacion de la tesis de maestria hacia inferencia, agentes y multiples dominios simultaneos.

## 3. Evidencia primaria del sistema existente

### Origen de investigacion

- [Tesis de maestria: Hybrid-Model Decentralized Evolutionary Computing Using Blockchain and Proof-of-Work Optimization](https://github.com/harveybc/doin-core/blob/master/docs/Hybrid-Model%20Decentralized%20Evolutionary%20Computing%20Using%20Blockchain%20and%20Proof-of-Work%20Optimization.pdf)
- [Ficha bibliografica de la tesis](https://github.com/harveybc/doin-core/blob/master/docs/THESIS.md)

### Protocolo y economia

- [doin-core](https://github.com/harveybc/doin-core): modelos y reglas compartidas.
- [Prueba de optimizacion](https://github.com/harveybc/doin-core/blob/master/src/doin_core/consensus/proof_of_optimization.py)
- [Semillas deterministas para evaluacion y datos sinteticos](https://github.com/harveybc/doin-core/blob/master/src/doin_core/consensus/deterministic_seed.py)
- [Quorum de evaluadores](https://github.com/harveybc/doin-core/blob/master/src/doin_core/models/quorum.py)
- [Incentivos](https://github.com/harveybc/doin-core/blob/master/src/doin_core/consensus/incentives.py)
- [Mercado de tarifas](https://github.com/harveybc/doin-core/blob/master/src/doin_core/models/fee_market.py)
- [Canales de pago](https://github.com/harveybc/doin-core/blob/master/src/doin_core/models/payment_channel.py)

### Runtime distribuido

- [doin-node](https://github.com/harveybc/doin-node): runtime unificado de optimizador, evaluador y nodo.
- [Bucle unificado, poblacion compartida y evaluacion](https://github.com/harveybc/doin-node/blob/master/src/doin_node/unified.py)
- [Blockchain y aplicacion de bloques](https://github.com/harveybc/doin-node/tree/master/src/doin_node/blockchain)
- [Red y descubrimiento entre pares](https://github.com/harveybc/doin-node/tree/master/src/doin_node/network)
- [Analitica de experimentos y cadena](https://github.com/harveybc/doin-node/tree/master/src/doin_node/stats)

### Dominios y adaptadores

- [doin-plugins](https://github.com/harveybc/doin-plugins): plugins de optimizacion, inferencia y datos sinteticos.
- [Adaptacion de un dominio nuevo](https://github.com/harveybc/doin-plugins/blob/master/docs/ADAPT_A_NEW_DOMAIN_WITH_AN_AGENT.md)
- [doin-domains](https://github.com/harveybc/doin-domains): frente de dominios alternativos.
- [Metodologia contrato primero](https://github.com/harveybc/doin-domains/blob/master/docs/work_plan/01_METODOLOGIA_CONTRATO_PRIMERO.md)
- [Plan de dominios alternativos](https://github.com/harveybc/doin-domains/blob/master/docs/work_plan/02_WORK_PLAN_DOMINIOS_V1.md)
- [agent-multi](https://github.com/harveybc/agent-multi): dominio de optimizacion de agentes.
- [predictor](https://github.com/harveybc/predictor): dominio de modelos predictivos de series temporales.
- [synthetic-datagen](https://github.com/harveybc/synthetic-datagen): generacion y evaluacion de datos sinteticos con controles de held-out y memorizacion.

Los repositorios [`doin-optimizer`](https://github.com/harveybc/doin-optimizer) y [`doin-evaluator`](https://github.com/harveybc/doin-evaluator) son historicos y fueron sustituidos por `doin-node`; no deben citarse como arquitectura vigente.

### Literatura primaria que debe contrastarse

- [Hayek, The Use of Knowledge in Society](https://rosenfels.org/pll-v5/titles/92.html)
- [Myerson, Optimal Auction Design](https://pubsonline.informs.org/doi/abs/10.1287/moor.6.1.58)
- [Miller, Resnick y Zeckhauser, Eliciting Informative Feedback](https://doi.org/10.1287/mnsc.1050.0379)
- [Kong y Schoenebeck, Information Elicitation Mechanisms](https://doi.org/10.1145/3296670)
- [Jia et al., Proof-of-Learning](https://ieeexplore.ieee.org/document/9519402/)
- [Ghorbani y Zou, Data Shapley](https://proceedings.mlr.press/v97/ghorbani19c.html)
- [Dutting et al., Optimal Auctions through Deep Learning](https://proceedings.mlr.press/v97/duetting19a.html)
- [Zheng et al., Judging LLM-as-a-Judge](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)
- [Liu et al., AgentBench](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e9df36b21ff4ee211a8b71ee8b7e9f57-Abstract-Conference.html)
- [Shannon, A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)

## 4. Preguntas obligatorias de auditoria

### A. Fidelidad a la idea del propietario

- ¿La propuesta mantiene en el centro la distribucion, validacion y remuneracion del trabajo?
- ¿Convirtio accidentalmente una funcion secundaria, como el enrutamiento automatico, en la tesis principal?
- ¿Representa correctamente el pooling descentralizado de inferencia y la optimizacion compartida?

### B. Contribucion doctoral

- ¿La pregunta conduce a resultados cientificos refutables o solo a construir una plataforma?
- ¿La novedad respecto de la tesis de maestria esta delimitada y es suficiente?
- ¿El aporte es defendible ante jurados de inteligencia artificial, teoria de juegos y sistemas distribuidos sin repetir etiquetas de forma promocional?

### C. Diseno de mecanismos

- ¿Actores, informacion privada, acciones, utilidades, asignacion, validacion y pagos estan definidos?
- ¿Las hipotesis confunden precio ofertado con costo real, o acuerdo con verdad?
- ¿Se prometen compatibilidad de incentivos, equilibrio o bienestar sin supuestos suficientes?
- ¿Existen ataques o imposibilidades omitidos que destruyan la regla propuesta?

### D. Hayek y teoria de la informacion

- ¿La interpretacion de Hayek es precisa y limitada a informacion economica dispersa?
- ¿La informacion mutua condicional es estimable con los experimentos propuestos?
- ¿La propuesta explica por que evidencia no redundante puede merecer pago sin afirmar que informacion, verdad y valor economico son identicos?

### E. Verificacion y adversarios

- ¿Los datos sinteticos realmente reducen el sobreajuste o solo trasladan el ataque al generador?
- ¿Commit-reveal, semillas posteriores y comites independientes bastan frente a colusion?
- ¿Se distinguen aceptacion falsa, rechazo falso, contaminacion, copia y falsificacion de procedencia?

### F. Ledger y pagos

- ¿Cada uso de blockchain esta justificado por una propiedad que no entrega un registro firmado mas barato?
- ¿La comparacion central, permisionada y publica es justa?
- ¿La ausencia de token propio deja un mecanismo de liquidacion coherente y verificable?

### G. Metodologia, alcance y literatura

- ¿El plan cabe razonablemente en un doctorado y separa contribucion central de casos experimentales?
- ¿Las variables y metricas son operacionales y atacables, sin terminos vagos como “calidad demostrada” o “riesgo”?
- ¿Cada afirmacion tomada de la literatura tiene llamada IEEE `[n]` y referencia correspondiente?
- ¿Faltan antecedentes esenciales de peer prediction, computational mechanism design, decentralized ML o verificacion de entrenamiento?

### H. Presentacion

- ¿Las cuatro paginas tienen jerarquia tipografica uniforme, lectura natural y densidad razonable?
- ¿El resumen y la pregunta permiten entender la investigacion sin abrir los repositorios?
- ¿Hay lenguaje defensivo, promocional, improvisado o repetitivo que deba eliminarse?

## 5. Formato de retorno solicitado

1. **Hallazgos primero**, ordenados `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`.
2. Para cada hallazgo: pagina/seccion, afirmacion exacta, contraargumento, evidencia y correccion minima propuesta.
3. Tabla de trazabilidad: afirmaciones sobre DOIN frente a repositorio/archivo que las sostiene.
4. Lista de citas faltantes, incorrectas o que no respaldan la frase asociada.
5. Veredicto final: `ACCEPT`, `REVISE` o `REJECT`, con condiciones objetivas.

No se solicita reescribir el documento durante la auditoria. Se solicita intentar refutarlo y devolver evidencia suficiente para una correccion posterior.
