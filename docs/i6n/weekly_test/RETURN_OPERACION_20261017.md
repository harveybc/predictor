# RETURN: candidata de robustez operativa (ORDEN_SATOSHI_OPERACION)

Base exacta: `9beb31b6`. Rama candidata: `satoshi/prequential-ops-20261017`.
Commit final: `6e96be14` (fast-forward de los carriles A/B `satoshi/prequential-ops-ab-20261017`
y C `retsu/prequential-ops-c-20261017`). NO desplegada, NO fusionada sobre el
worktree activo, campana cientifica en curso intacta. Decision de aplicacion: Musashi.

## Commits

| Commit | Contenido |
| --- | --- |
| `c2514a29` | Parte C: sondas PS06/PS10 con limites simulados; predictor falso dependiente del donante para PS03/PS04 |
| `2e30973a` | Parte A: `RECOVERY_DESIGN.md`, clasificacion disenada y confirmada ANTES del codigo |
| `7bb2ebcb` | Parte A codigo+pruebas (`prequential_recovery.py`), Parte B codigo+pruebas (`prequential_closure.py`), `CLOSURE_COVERAGE.md` |
| `47ac0aa3` | Fusion de la parte C |
| `6e96be14` | Defecto hallado por el revisor: cabeza con cero pares anuales reventaba el tick de cierre (`weekly_test.py:183`); ahora `n 0 / sin soporte` |

Archivos cientificos byte a byte identicos a la base (model.py, train.py,
weekly_test_data.py, continuation.py, continuation_data.py y seis mas), exigido
por `test_scientific_files_are_byte_identical_to_the_base` y verificado por el
coordinador con `git diff --quiet 9beb31b6 HEAD`.

## Requisito -> prueba -> resultado (todas PASS en `6e96be14`)

| Req. | Prueba | Resultado |
| --- | --- | --- |
| A1 clasificacion por tipo/codigo tipificado, nunca por texto; version != GPU indisponible | `test_prequential_recovery.py::test_classification_is_by_type_and_typed_code_never_by_message`, `::test_runtime_version_and_gpu_visibility_are_split_by_type` | PASS |
| A2 max 2 reintentos, cooldown 30/60 s con reloj inyectado, ledger durable, terminal con causa y accion | `::test_cooldown_is_enforced_by_the_injected_clock_without_sleeping`, `::test_transient_budget_exhaustion_is_an_explicit_terminal` | PASS |
| A3 transitorio tras el sello: prediccion byte a byte intacta, mismo donante/seed/datos/escalas/config; update completado se adopta sin segundo fit; FAILED historico nunca borrado | `::test_transient_after_seal_recovers_with_identical_prediction_and_same_donor`, `::test_completed_update_is_adopted_on_retry_without_a_second_fit`, `::test_historical_failed_terminal_is_never_deleted_or_retried` | PASS |
| A1/A3 no reintento de digest/causalidad/version/desconocido | `::test_permanent_and_unknown_causes_are_never_retried` (4 casos) | PASS |
| A4 ningun parametro cientifico ni cap tocado | asercion de ausencia de `MemoryMax`/`max_epochs` en el modulo + prueba de bytes identicos | PASS |
| A5 relevo a la familia independiente durante cooldown; nunca sin su checkpoint previo | `::test_other_family_runs_during_a_cooldown_and_never_skips_its_own_dependency` | PASS |
| A5 bloqueo de recursos no consume intento | `::test_resource_block_consumes_no_attempt_and_is_typed` | PASS |
| A6 status RUNNING/WAIT_RESOURCES/RETRY_PENDING/bloqueo; sin ETA finita bajo bloqueo | `::test_status_distinguishes_states_and_never_promises_an_eta_under_a_block` | PASS |
| A ledger tras reinicio de proceso | `::test_ledger_resumes_attempt_numbering_after_a_simulated_restart` | PASS |
| B fronteras anuales como sin pares, invariante suma = declaradas, nunca 100 % | `test_prequential_closure.py::test_coverage_counts_both_annual_boundaries_as_no_pair_and_keeps_the_invariant` | PASS |
| B cabeza sin soporte en semana util, solo por horizonte | `::test_a_head_without_support_in_a_useful_week_is_counted_per_horizon_only` | PASS |
| B fallo y cierre incompleto explicitos | `::test_a_failure_and_an_incomplete_closure_are_explicit_and_never_100_percent` | PASS |
| B metricas intactas al anadir cobertura | `::test_tick_seam_adds_coverage_without_changing_annual_metrics` | PASS |
| B cierre con cabeza de cero pares completa | `test_retsu_prequential_audit.py::test_ps10_closure_with_a_head_lacking_annual_pairs_must_still_close_simulated_boundary` (antes xfail estricto) | PASS |
| C1 PS06 guardia real de pesos iniciales | `::test_ps06_real_initial_weights_guard_..._simulated_boundary` | PASS |
| C2 PS10 readback pendiente + reintento idempotente; espejo fallido impide CLOSE; cierre detiene solo timers propios | tres `::test_ps10_*_simulated_boundary` | PASS |
| C3 predictor falso dependiente del donante con control positivo; NPZ vs JSON-sin-reloj | `::test_ps03_*[StatefulFake]`, `::test_ps04_*[StatefulFake,StatelessFake]` | PASS |
| C3 PRE contra el falso antiguo | `::test_ps03_*[StatelessFake]` | XFAIL estricto por diseno |

## PRE (`9beb31b6`) -> POST (`6e96be14`), por archivo, mismo comando

| Archivo | PRE | POST |
| --- | --- | --- |
| `test_prequential_test.py` | 10 passed | 10 passed |
| `test_retsu_prequential_audit.py` | 11 passed | 19 passed, 1 xfailed |
| `test_weekly_test.py` | 5 passed | 5 passed |
| `test_prequential_recovery.py` (nuevo) | - | 16 passed |
| `test_prequential_closure.py` (nuevo) | - | 4 passed |

Verificacion del coordinador sobre la rama integrada: 54 passed, 1 xfailed, 2.98 s.

## Comando reproducible

```
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 120s -n preq-ops -- \
  python -m pytest -q -p no:cacheprovider \
  tests/i6dmo/test_prequential_recovery.py tests/i6dmo/test_prequential_closure.py \
  tests/i6dmo/test_prequential_test.py tests/i6dmo/test_retsu_prequential_audit.py \
  tests/i6dmo/test_weekly_test.py
```

Ningun comando necesito mas de 2 GiB / 120 s. No se corrio suite abierta.

## Limites

- Todo es sintetico en CPU con limites simulados (backends falsos, reloj
  inyectado, almacen falso, systemctl grabado). Ninguna prueba es GPU, E2E ni
  cierre anual real; el cierre anual real sigue sin verificar.
- `PROGRESS.json` RUNNING es un marcador de ultima escritura, no prueba de
  vida; un proceso muerto a mitad de intento lo deja hasta la siguiente admision.
- Un `RESULT.json` FAILED escrito por el codigo desplegado (`fa47d2b2`) sigue
  terminal bajo la candidata: no hay recuperacion automatica de fallos ya
  existentes, por orden.
- `COVERAGE.json` es un artefacto nuevo bajo `artifacts/`; entra en el conjunto
  espejado al cierre.
- Sin fits, GPU, datos TEST nuevos, despliegues, servicios, reinicios, DOIN,
  re-seleccion ni monitoreo. Sin denegaciones de permisos. Sin secretos,
  identidades de maquina ni rutas privadas (grep limpio).
