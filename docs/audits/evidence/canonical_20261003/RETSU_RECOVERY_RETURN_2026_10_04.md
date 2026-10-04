# Recuperacion posterior a Retsu, 2026-10-04

## Alcance

Se auditaron las entregas M1, M2 y PS4 dejadas por Retsu. Las dos colas PS3-R se reanudaron como servicios durables sin repetir terminales cuyo estado, fichero y SHA-256 ya fueran validos.

## Resultado

| Frente | Resultado | Evidencia |
|---|---|---|
| M1 negocio semanal | Reparado; revision final pendiente al publicar este corte | Scorer sin efectos laterales; contrato y WeekSpec completos; diferencias pareadas; MAE financiero; restauracion semantica; entrega `AT_LEAST_ONCE_IDEMPOTENT_SCORE_DIGEST` |
| M2 readiness | ACEPTADO por revision independiente | 38/38; tres terminales alternativos reales; producto pliegue x familia x target x horizonte x metrica; 279 `NOT_IDENTIFIED` + 87 `OUTSIDE_JOIN_PENDING`; 366 `NOT_READY` |
| PS4 incremental | ACEPTADO por revision independiente | 20/20; 5 `MEASURED`, 1 `PENDING`; 105 metricas; indice aceptado externo y 441 filas PS3-R retenidas |
| Integracion combinada | PASA | 130/130 pruebas antes de la ultima reparacion semantica M1; 72/72 del bloque M1 despues de ella |

La seleccion final sigue `NOT_ISSUED`. Aceptar el ledger y el perfilador no convierte evidencia parcial en una decision de features.

## Ejecucion GPU

- Gamma, RTX 5090 externa: `canonical-lanef-5090.service`, UUID fisico explicito, semilla 0. Cerro siete celdas desde la recuperacion y avanzo a `fx.audusd.logret_24h`.
- Dragon, RTX 4090 Laptop: `canonical-baseline-4090.service`, UUID fisico explicito, semilla 0. Cerro `fred.credit.aaa.logret_1d` y avanzo a `fred.fx_indices.dtwexemegs.level`.
- Gamma, RTX 5070 Ti interna: deliberadamente sin una segunda carga pesada porque comparte la RAM de 14 GiB con la 5090.
- Omega, RTX 4070 Laptop: reservada para escritorio y verificaciones pequenas.

Los drivers omiten solamente un terminal `COMPLETED` cuyo `results.jsonl` existe y coincide con el `results_sha256` del manifiesto. Un fallo genera evidencia y el servicio continua con la siguiente celda; no reduce semillas ni repite una celda valida.

## Limites

- No hay feature seleccionada todavia.
- `NOT_IDENTIFIED` es abstencion causal, no descarte.
- Una buena reconstruccion es evidencia de extractibilidad, no seleccion automatica.
- La autenticidad del indice PS4 depende de custodiar su SHA-256 fuera del conjunto que autentica.
- El piloto anual semanal de negocio aun no se ha ejecutado.
