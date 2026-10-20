# Especialistas modulares: siguiente contraste autorizado

2026-10-20. El propietario aprueba automatizar secuencialmente en la GPU
preferente el plan de especialistas, capacidad comparable y abstencion.
Implementacion: heuristic-strategy, rama codex/modular-specialists-20261020,
docs/short_horizon/MODULAR_CONTROLLED_PLAN.md y MODULAR_TRACEABILITY.md.

Orden: verificar datos/negocio; JOINT16, JOINT_MATCHED, FULL para las semillas
20261019/20/21; repetir las mismas tres arquitecturas para 20261022/23.
15 celdas totales, sin seleccionar una semilla ganadora. Misma entrada de
15 variables, ventana13 y stride2. Aprendizaje por imitacion BC+3 DAgger;
no hay cambio del SAC ni del protocolo DOIN. Todas las variantes aplican
un maximo de una apertura diaria, leverage50, SL/TP, rel_volume0.1 y las
reglas12h antes/despues del cierre/apertura. Los costes del simulador de
Satoshi permanecen positivos y supuestos; no son los mismos del SAC.

Para aislar el hallazgo, esta linea conserva expresamente los cortes de
Satoshi: TRAIN2020-2022, seleccion de checkpoint/umbral2023H1, diagnostico
2023H2, evaluacion de desarrollo2024 completo. No se presenta como cuatro
anos de ajuste ni como TEST. La regla de SAC de cuatro anos sigue registrada
y no cambia. Modelos, arbitros y semillas se fijan antes de las mediciones.

Con los mismos pesos, evaluar argmax frente a abstencion en JOINT; en FULL,
salidas aprendidas frente a solo protecciones (umbral original y recalibrado
en2023H1), BUY solo y SELL solo. Guardar metricas de TRAIN y cada periodo,
profit,Sharpe diario anualizado sqrt365,drawdown,trades,ganadoras,PF,costes,
exposicion y desglose por direccion. Guardar todas las curvas y pesos.

La cola finita y sus timers realizan relevo, reintentos limitados, publicacion
al warehouse y copia verificada independiente. No requiere agente vigilante.
Status y ETA consultables sin afectar la secuencia. Ningun resultado negativo
detiene los otros brazos. Una contradiccion de identidad/causalidad/negocio si
detiene la ejecucion, dejando una causa concreta para reparar.

Trabajo posterior conservado: ventana/stride separando cantidad de slots de
contexto temporal; diagnosticos de variables; senal/objetivo/experto; budget,
regularizacion y generalizacion. No lanzar una barrida de estos factores hasta
analizar el contraste actual. La replica de nuevas semillas ya esta incluida,
no depende de una nueva autorizacion ni de resultados intermedios positivos.

Desplegado: runtime f6ddd53c18c34a12f809498d124874ba82ee908f,
spec fb808aa85830deac27cc838fad9cc326ac3d4bb57790b51e782f583250b946a6.
Primer relevo automatico comprobado,8/8 reportes leidos de vuelta y copia
independiente del primer terminal verificada. 43 tests locales y43worker,
smoke GPU real sintetico con recarga bitwise. El analisis derivado incorpora
medias/desviaciones por grupo de semillas, diferencias pareadas y brechas de
crecimiento diario. Ninguna modificacion del runtime activo por ese analisis.
