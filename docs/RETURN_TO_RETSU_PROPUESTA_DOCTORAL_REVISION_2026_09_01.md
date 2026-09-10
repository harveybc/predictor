# Retorno a Retsu: revisión de la propuesta doctoral DOIN

**Fecha:** 2026-09-01  
**Estado:** `READY_FOR_REAUDIT`  
**Auditor solicitado:** Retsu  
**Dictamen de origen:** [`AUDIT_RETSU_PROPUESTA_DOCTORAL_DOIN_2026_09_01.md`](AUDIT_RETSU_PROPUESTA_DOCTORAL_DOIN_2026_09_01.md)

## Artefactos revisados

- [`propuesta_doctoral_doin_borrador.pdf`](propuesta_doctoral_doin_borrador.pdf)
- [`propuesta_doctoral_doin_borrador.html`](propuesta_doctoral_doin_borrador.html)
- [Solicitud y fuentes primarias de la auditoría original](REQUEST_FOR_AUDIT_RETSU_PROPUESTA_DOCTORAL_DOIN_2026_09_01.md)

## Disposición de los hallazgos

| Condición del dictamen | Cambio verificable |
|---|---|
| Una pregunta madre | La pregunta de §1 se limita a validación y pago con verdad parcial; asignación, ledger y routing son infraestructura. |
| Máximo tres hipótesis | §9 contiene H1–H3, cada una con comparación, presupuesto y margen de refutación. |
| Protocolo con y sin oráculo | §5 define dos reglas de aceptación, fuentes de verdad y tratamiento del desacuerdo. |
| Retirar CMI ornamental | §6 adopta Correlated Agreement; fija fórmula, N mínimo, tres evaluadores y cross-fitting de cinco pliegues. |
| Reconocer el límite sin verdad | §§1, 5, 6 y 11 distinguen aceptación administrativa, auditoría externa y acuerdo informado. |
| Separar DOIN actual del mercado | §3 declara que DOIN hoy consensa incrementos de optimización y que el mercado propuesto cambia el mecanismo. |
| Incluir imposibilidades | §11 nombra Myerson–Satterthwaite, relabeling/colusión, Sybil y ataques a Proof-of-Learning, y declara qué no se reclamará. |
| Incluir sistemas vecinos | §12 contrasta Bittensor, Gensyn, Akash y Golem con la pregunta específica de la tesis. |
| Formalizar pooling | §4 incorpora K, G y w en la tarea; K=1 frente a K=3 queda como ablación acotada. |
| Operacionalizar datos sintéticos | §§5 y 10 fijan retos posteriores al compromiso, presupuesto de consultas, holdout no reutilizado y ataque de contaminación. |
| Corregir alcance y literatura | Salen Data Shapley, mecanismos aprendidos y la comparación de tres ledgers; quedan dos dominios y tres fases. |
| Presentación | Cuerpo y referencias usan 9.5 pt; autor y pie son uniformes; se conservan cuatro páginas Letter. |

## Centro científico sometido a reauditoría

La afirmación principal ya no es que DOIN construirá un mercado multidominio completo. Es que, bajo supuestos explícitos y auditoría parcial, un mecanismo de elicitación de información puede hacer menos rentable el reporte de bajo esfuerzo y reducir pagos por resultados inválidos. El control tiene verdad objetiva; el experimento principal usa tareas LLM/agente con verdad parcial. Un resultado negativo bajo colusión, señales poco correlacionadas o auditoría insuficiente refuta las hipótesis sin reinterpretar el protocolo.

## Solicitud

Repetir la auditoría adversarial y devolver `ACCEPT`, `REVISE` o `REJECT`. En particular:

1. ¿La pregunta, las tres hipótesis y los dos dominios forman ahora una tesis y no un programa de laboratorio?
2. ¿La regla CA es ejecutable y está citada sin sobreafirmar verdad o compatibilidad de incentivos?
3. ¿La aceptación con oráculo parcial permite medir falsos positivos y negativos sin circularidad?
4. ¿El pool está suficientemente formalizado sin convertirse en otra contribución principal?
5. ¿La diferencia frente a DOIN actual y frente a Bittensor, Gensyn, Akash y Golem es defendible?
6. ¿La tipografía, densidad y trazabilidad IEEE son adecuadas para remitir el documento?
