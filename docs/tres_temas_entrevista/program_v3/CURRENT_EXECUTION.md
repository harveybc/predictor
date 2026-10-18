# Ejecucion vigente

Actualizado2026-10-18. Autoridad: master v3 y RL_FIRST_AFTER_DATA_REPAIR_2026_10_18.md.
Ninguna orden ni donante de las campanas predictivas retiradas puede despachar
trabajo. No hay entrenamiento financiero autorizado en esta reparacion.

## Ahora

- Datos: recuperacion nativa2020-2025 y candidatos alternativos. Forexite declara
  CET/CEST en su pagina oficial: reloj verificado, 86 archivos/206858 filas
  normalizados sin interpolacion. Faltan quote side, etiqueta temporal de minuto,
  calendario real y admision de todo el inventario requerido.
- Nueva calidad:4 informes,1122 metricas y lectura de vuelta; sin discrepancias
  ni pendientes en la lectura de cierre de esta sesion. Copias fisicas y espejo
  verificados. Esto no completa seleccion ni certifica los datos.
- Evidencia nueva de reloj: 1 informe adicional, 6 metricas, lectura de vuelta
  sin pendientes/discrepancias y copias fisicas verificadas en dos maquinas.
- Detalle y status vivo: procedimiento17 de financial-data y su CLI
  `_scripts/native_recovery_status.py`; las cifras de este texto son un corte,
  no sustituyen el estado generado.
- Retornos anteriores de Satoshi integrados y revisados en
  SATOSHI_RETURN_REVIEW_2026_10_18.md. El propietario informa trabajo adicional
  de Satoshi en datos; no se presume completado ni se sobrescribe.
- Musashi ejecuta RL_EXECUTION_REPAIR_PLAN_2026_10_18.md: corregir primero la
  transicion real accion/fill/recompensa, luego costos y reglas del negocio.
- Corte de ingenieria: transicion next_open_v2, comisiones y capital de cierre,
  financiacion analitica explicita y seleccion sin NaN->0 corregidos; 256 pruebas
  de gym-fx, 65 de agent-multi y 7 del verificador pasan. No son fits financieros.
  Corregido tambien el ciclo nativo Trade tras un cierre directo: ya no pierde
  el registro de la siguiente operacion. Nueve regresiones nuevas pasan.
  Pendientes: admision/seleccion, costos reales y aceptacion completa del entorno.

## Siguiente

Datos admitidos -> preprocesamiento verificado -> seleccion recalculada ->
contrato/entorno RL -> baselines de negocio -> SAC compacto/modular ->
comparaciones RL -> DEMO/PAPER. Predictores y NEAT al final.

El insight de posible utilidad del reajuste queda como hipotesis cualitativa,
no como un resultado numerico conservado. Aumentar la frecuencia no garantiza
mejora. Separar inferencia/decision, puntuacion y actualizacion posterior.

No usar el naive de precios como compuerta RL: evaluar PnL neto, drawdown,
Sharpe, trades, exposicion, costes y reglas de negocio frente a controles
emparejados. No se afirma que RL vaya a solucionar por si solo la rentabilidad.
