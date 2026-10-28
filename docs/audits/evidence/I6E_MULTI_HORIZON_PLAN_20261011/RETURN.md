# I6-E-M: actualizacion de plan, no ejecucion cientifica

2026-10-11. Worktree del coordinador; sin editar el checkout primario en
conflicto, sin borrar resultados ni modificar motores de corridas selladas.

## Resultado

El plan maestro, subplan, estado, checklist y cola apuntan a la misma grilla:
10 horizontes fisicos, 28 brazos downsampled y 10 horarios, 38 identidades
unicas. Las tres solicitudes equivalentes al horario no generan otro fit.
Primero toda la etapa downsampled, luego horario ascendente. Contexto 2h;
modelo y naive para el horizonte de X horas en USD/EUR original. El nuevo
calendario incluye el fragmento final de 2024 (53 segmentos). TEST reservado.

`tools/i6e_horizon_plan.py` genera/verifica el diseno determinista y calcula
MAE/MSE de precio contra el naive del mismo horizonte. NO es un ejecutor de
entrenamiento ni un despliegue. El JSON dice execution_ready=false y enumera
adapter, inputs, workers/outbox/status y pilotos pendientes. No hay medicion
de numero de filas ni ETA nueva que pueda atribuirse a esta entrega.

## Pruebas y alcance

PRE: escribir `tests/test_i6e_horizon_plan.py` y ejecutar pytest antes del
modulo produjo error de coleccion por modulo ausente. Esto demuestra ausencia
del componente, no un fallo conductual contra una version anterior.

POST: 15 casos puros del plan pasan. La verificacion conjunta con
`tests/test_i6e_price_metrics.py` y `tests/test_i7_multi_horizon.py` da
**25 passed in 0.65s**. `git diff --check` limpio; cuatro JSON de estado/grilla
decodificados sin error; CLI `check` verifica el diseno generado.

| Requisito | Evidencia | Alcance |
| --- | --- | --- |
| MH01 | tests de grilla y h=k*delta | reloj entero y contexto, sin lectura de datos |
| MH02 | 38 identidades / 41 solicitudes / 3 alias | especificacion, no leases reales |
| MH03 | test de orden D antes de H | orden de grilla; transicion durable pendiente |
| MH04 | tests de naive por h, escala de precio y h ajeno | helper puro; filas y fuentes reales pendientes |
| MH06 | tests calendario 2024/2025 | fronteras generadas; disponibilidad de datos pendiente |
| MH05, MH07-MH10 | pending_implementation del JSON | no aceptados ni ejecutados |

El recibo retenido de la campana anterior registra 312 disposiciones
estadisticas y 312 informes / 2910 metricas con PUBLISHED_READBACK_VERIFIED.
ARIMA 156 exitosas; ARIMAX 121 exitosas / 35 fallos. Esto NO es evidencia
de que la nueva grilla multihorizonte se haya ejecutado.

## Continuacion

Implementar MH05-MH10 y autenticar inputs en modulos nuevos. Verificar modelo
real, filas por h, memoria y costo; despues desplegar el coordinador que
completa D y H sin visitas del usuario. No modificar el motor sellado a 72 h,
reentrenar sus celdas para hacerlas pasar ni abrir TEST durante la seleccion.
