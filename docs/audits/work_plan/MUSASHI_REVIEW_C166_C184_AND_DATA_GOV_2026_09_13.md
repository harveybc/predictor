# Revision de C166-C184 y propuesta data-gov

Fecha: 2026-09-13. Revisor: Musashi.
Codigo D2 revisado: `predictor@d26fdc9`, en worktree aislado.
Disposicion: **REVISE_D2_ADJUDICATION_BEFORE_CONSUMPTION**.
La adopcion de Flow v3 continua; esta revision no la interrumpe.

## 1. Hallazgos

### H1 - Alto: faltan mediciones dentro de cinco decisiones favorables reales

`tools/df_d2_adjudicate.py:375` considera valida una semilla cuando tiene alguna
metrica. `_agg` (linea 317) descarta las ausentes. El numero de semillas validas
no es, por tanto, el numero de semillas del contraste principal.

Lei las 3.591 decisiones publicadas y verifique el SHA-256 completo:
`f4958c88f8caa6b78d67fb7ff00c2a6a697276aa9b19b6b73411b78d97622510`.
De las 58 decisiones favorables o limitadas de candidatos, cinco tienen el
siguiente problema. Todas declaran `n_seeds_valid=30`:

| Operador | Regimen | Estado publicado | Semillas de mejora | Sin mejora/retardo/residual |
|---|---|---|---:|---:|
| local_linear_trend_kalman | motif, white, 5 dB | LAB_CALIBRATED | 22 | 8 |
| local_level_kalman | motif, white, 5 dB | LAB_CALIBRATED | 22 | 8 |
| trailing_haar_threshold, niveles 3, k=3 | steps, white, 20 dB | LAB_CALIBRATED | 25 | 5 |
| trailing_haar_threshold, niveles 2, k=3 | steps, white, 20 dB | LAB_CALIBRATED | 25 | 5 |
| local_linear_trend_kalman | steps, white, 20 dB | REGIME_LIMITED | 25 | 5 |

No afirmo que los valores observados sean falsos ni que exista fuga temporal.
El problema es que no se cumplio la evidencia completa que anuncia el pase.
Debe determinarse desde arrays y soportes si cada ausencia es falta de senal,
geometria insuficiente, fallo numerico o metrica realmente inaplicable. No se
pueden quitar esas semillas del denominador despues de observar el resultado.
Las otras 53 decisiones tampoco quedan aprobadas por descarte: necesitan la
misma comprobacion por variable, metrica y semilla.

### H2 - Alto: ausencia de evidencia se convierte en chequeo aprobado

En `_decide_arm`, los pisos y el residual ignoran `None` (lineas 398-414);
la no inferioridad con menos de dos pares figura `passed=True,
applicable=False` (lineas 440-442); retardo ignora `None` (linea 454).

Reproduccion CPU con el fixture existente y el adjudicador real:

| Caso | Resultado observado |
|---|---|
| Fixture completo | LAB_CALIBRATED |
| Retencion extrema medida en 0.2 | LAB_REJECTED |
| Mismas filas, omitiendo solo esa metrica | LAB_CALIBRATED |
| Retardo marcado INCONCLUSIVE en todas las semillas | LAB_CALIBRATED |
| Residual marcado INCONCLUSIVE en todas las semillas | LAB_CALIBRATED |

Esto es un defecto de tratamiento de datos faltantes. La ausencia de un tipo
de evento puede ser legitima, pero debe derivarse del banco/ventana y no de
que no llego una fila. No ordenar cero para metricas indefinidas.

### H3 - Medio: AT9 no permite afirmar portabilidad de toda la poblacion

El retorno declara seis numeros Kalman fuera de tolerancia entre dos CPUs,
incluido un extremo de intervalo con diferencia de aproximadamente 0.006 dB.
La igualdad en el host productor no sustituye la prueba entre plataformas.
Solo se contrastaron dos unidades; el margen amplio de una decision no
establece el margen de las otras 1.025 decisiones SNR.

El codigo de `df_snr.py:153` usa un ajuste MLE de statsmodels con criterio de
convergencia. La amplificacion de diferencias numericas por el optimizador es
una explicacion plausible, no una causa instrumentada en esta revision.
No ampliamos la tolerancia de la confirmacion ya observada. El alcance
pendiente es numerico y por estimador, no una anulacion de todo D2.

### H4 - Medio: dos versiones de cobertura comparten un identificador historico

El packet reconoce dos matrices v1 iguales bajo un mismo run id y distinto
codigo. Cambiar los ids futuros no arregla una consulta historica que cuenta
ambas. Hace falta una vista explicita de version vigente y otra de historia;
no borrar filas ni usar `DISTINCT` para esconder su procedencia. No consulte el
cubo real en esta revision: este hallazgo operativo se apoya en el retorno.

## 2. Que se conserva y que no se concede

- Se conservan los resultados crudos, los rechazos y toda la cronologia.
  No ordeno repetir las 3.972 unidades ni las 715 caracterizaciones.
- Los conteos de las decisiones coinciden con el archivo fisico revisado:
  51 pases y 7 limitados de candidatos; 39 calibraciones SNR por regimen.
  Son etiquetas candidatas, no aprobaciones externas.
- La distincion entre unidad completa pero no evaluable y fuga temporal es
  correcta: una abstencion prevista por soporte no equivale a usar el futuro.
  Los 64 casos MCAR deben conservarse en denominadores y cobertura.
- Que el control antes rechazado pase en dos regimenes no invalida el banco.
  Su nombre debe describir un comparador de baja eficacia historica, no un
  detector universal cuyo resultado debe ser siempre negativo. No escoger
  otro control despues de mirar esta reserva.
- La auditoria wavelet implementa prefijos, sufijos alterados, niveles,
  fragmentacion y reinicio. La bateria focal revisada incluye el caso con
  fuga y la invalidacion del root. Esto no es una prueba universal de todas
  las wavelets ni de retardo cero. No encontre en esta revision una nueva
  fuga temporal; tampoco reejecute sus 7.312 auditorias.
- D3 permite preparacion de diseno, no scoring ni consumo cientifico. La
  seleccion de features, los modelos y el live no se adelantan.

## 3. Dictamen sobre la propuesta data-gov

Lei completa `data-gov/docs/05_PROPUESTA_MUSASHI_BETA_2026_09_13.md` y la
contraste con `06_FLOW_V3_FAILSAFE.md` y la orden de adopcion vigente.
**Acepto el flujo de datos; la propuesta v2 no es la especificacion actual.**

| Punto de v2 | Disposicion actual |
|---|---|
| Descargar una vez y trabajar localmente | Conservar, con cache rehasheada y recibo por uso |
| Hash entregado y linaje al cubo | Conservar, anadiendo fuente, contrato, rol, configuracion y codigo |
| Reportar solo metricas finales | Sustituido por terminal para todos los desenlaces y outbox |
| `allow` equivale a entrega recibida | Sustituido por confirmacion de bytes verificados |
| Reloj de pared generico para todos los lagos | Sustituido por contrato temporal por recurso |
| Experimento/conjunto sin miembros fijados | Sustituido por campana y unidades declaradas |
| E2E inexistente, dos repos sin push | Estado historico, superado por las ramas Flow v3 entregadas |
| Reiniciar primero y auditar despues | Rechazado: ensayo desechable, integracion y luego despliegue |
| El owner debe reiniciar por defecto | La orden de adopcion ya asigna despliegue acotado a Satoshi |
| Reconciliacion para mas adelante | Requerida antes de afirmar operacion gobernada |

Ramas de implementacion ya entregadas, no nuevas tareas de reescritura:

- data-gov: `musashi/data-gov-failsafe-v3-20260913`, `02f07d7`.
- financial-data: `musashi/data-gov-lake-v3-20260913`, `7f77e3ce6`.
- predictor: `musashi/data-gov-consumer-v3-20260913`, codigo `adbd507`,
  orden y cierre `bd51398`.

Observe ademas la integracion de Flow v3 en el data-gov local, commit
`249741a`. No lo confundo con una verificacion del servicio desplegado.
Los resultados anteriores no se renombran retrospectivamente como descargas
gobernadas. Una nueva relectura de evidencia historica puede tener un recibo
actual, sin alterar quien produjo los bytes ni cuando lo hizo.

La beta mide trazabilidad y operacion, no utilidad predictiva ni causalidad.
Los hashes permiten identificar los bytes; conservarlos, sus generadores y el
entorno permite reproducirlos. Ninguna de esas cosas sustituye las pruebas de
causalidad o demuestra que un suavizador sirva.

## 4. Verificacion realizada y limites

- Discovery por grafo intento `predictor` sin simbolos D2 encontrados; se
  continuo con lectura de codigo y documentos del checkout aislado.
- `tests/test_df_d2_lab.py`, `tests/test_df_consumption_gate.py` y
  `tests/test_df_snr_isolation.py`: **29 passed en 47.62 s**.
- Reproductor nuevo:
  `docs/audits/evidence/repro_runs/musashi_d2_missing_metrics_review.py`.
  Usa fixtures pequenos, adjudicador real y lectura del archivo de decisiones
  con hash completo; imprime cinco decisiones afectadas. No modifica roots.
- No repeti la suite de 930 pruebas, la confirmacion distribuida, la carga
  OLAP ni una comprobacion de servicios remotos. Sus conteos siguen siendo
  evidencia declarada por Satoshi, no mediciones nuevas mias.
- Cero cambios de codigo productivo, GPU, servicios, registros de revision,
  datos de campana o tablas reales en esta auditoria.

## 5. Secuencia acordada

Primero terminar la adopcion Flow v3 ya en curso. En un worktree aparte se
pueden preparar desde ahora los tests y correcciones D2. Despues: re-adjudicar
con soporte explicito, medir portabilidad acotada y presentar el diseno D3.
Las ordenes ejecutables estan en
`docs/handoffs/MUSASHI_TO_SATOSHI_D2_SUPPORT_PORTABILITY_NEXT_ORDER_2026_09_13.md`.
No surge de esta revision una accion nueva para Harvey.
