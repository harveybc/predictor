# Continuacion paralela, 2026-09-30

Se retomaron cinco carriles independientes de la orden anterior sin tocar los
checkouts vivos ni iniciar GPU, brokers, B0 o confirmacion M4. Cada rama esta
publicada; ninguna de estas entregas equivale a integracion en master.

| Carril | Rama / tip | Verificacion y alcance |
| --- | --- | --- |
| Barrido horario sintetico | heuristic-strategy `codex/continuous-synthetic-20260930` @ `bfda08a` | 216 origenes consecutivos, horizontes 1-6/24-144 h; 3 pruebas enfocadas; evidencia completa `docs/continuous_synthetic/evidence/pilot.json`. Ideal/ideal: 6 cierres, equity marcada 14882.01; corto ruidoso/largo ideal: 6 cierres tempranos, 9682.21; persistencia/persistencia: sin entradas, 10000. Todo SYNTHETIC, no utilidad de mercado. |
| H-CORE | predictor `retsu/hcore-row-consumer-20260930` @ `9f533155` | 12 pruebas: consumidor por fila de bytes materializados con digestos y rechazo de identidad. No demuestra mejora predictiva ni entrenamiento nuevo. |
| DOIN archivo externo | doin-node `satoshi/offchain-shadow-20260930` @ `4c23a6f` | 24 pruebas contra core y byte store fijados: escritura, referencia, lectura por hash y proyeccion tras verificacion. Store desechable; cadena real y lago desplegado no probados. |
| Noticias | news-signal `retsu/quality-checkpoint-identity-20260930` @ `aebca51` | 78 pruebas pasan en venv aislado con el paquete instalado, incluido discovery por entry point: calidad v2 ligada al manifiesto del modelo servido; identidades ausentes o distintas rehusan. Sin pesos ni F1 nuevo. |
| Calendario | data-gov `codex/calendar-study-admission-20260930` @ `1c80c4b` | 9 pruebas enfocadas, 45 en la suite del agente: admision offline por revisiones fijadas/as-of y cobertura. No autoriza entrega ni identifica efectos causales. |

Revision separada M4: `docs/audits/work_plan/MUSASHI_M4_DR04_SCOPED_REVIEW_2026_09_30.md`.
Las reparaciones de mecanismo pasan; el contraste 15 sigue sin estimando
confirmatorio cerrado. No hay firma externa ni campana confirmatoria.

Siguientes trabajos independientes: integrar/medir el consumidor H-CORE con un downstream
predeclarado; disenar el barrido factorial corto/largo sobre soporte continuo
sin usar prediccion ideal como evidencia de negocio; y llevar el archivo de
DOIN del store desechable al servicio gobernado bajo pruebas de contenido.
No usar la asignacion GPU diagnostica, las calibraciones, codigo remoto de
clasificacion, B0, M4 confirmatorio ni brokers sin mandato explicito.
