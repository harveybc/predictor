# Presupuesto de evaluaciones de fidelidad completa

**Fecha:** 2026-09-03  
**Estado:** presupuesto de planeación, no autorización de cómputo  
**Unidad:** una evaluación completa de una configuración, en una tarea y una semilla, es un equivalente de corrida completa (FRE).

## 1. Hechos disponibles

ARLBench reporta 937 GPU-horas para 32 configuraciones completas, 10 semillas y sus 14 tareas representativas de DQN, PPO y SAC. El promedio publicado es aproximadamente:

`937 / (32 × 10 × 14) = 0,209 GPU-horas por FRE`

Ese valor mezcla algoritmos, tareas y hardware del estudio. Sirve para comprobar orden de magnitud, no para prometer una duración en nuestra flota. La extensión de representaciones puede ser más costosa.

El preflight local del extractor fuerte existente ejecutó 2.000 pasos y 1.000 actualizaciones SAC en 393,5 s de CPU, con 796.359 parámetros totales de política. Fue una prueba mecánica de presupuesto, no una corrida económica ni una medida de GPU. No se usa para extrapolar horas CUDA.

## 2. Cuenta máxima de etiquetas públicas

El diseño usa un conjunto discreto de **12 representaciones** y **30 unidades tarea-contexto**: 12 de desarrollo, 6 de calibración y 12 de prueba. Cinco semillas de selección se separan de cinco semillas finales. La cuenta máxima es:

| Bloque | Cálculo | Máximo |
|---|---:|---:|
| Desarrollo y calibración | 12 representaciones × 18 unidades × 5 semillas | 1.080 FRE |
| Tabla ciega de prueba para ejecutar los métodos bajo el mismo presupuesto | 12 × 12 × 5 | 720 FRE |
| Evaluación final independiente de la recomendación de cada método | 6 métodos × 12 unidades × 5 semillas nuevas | 360 FRE |
| Confirmación financiera externa | protocolo ya congelado | 40 FRE |

**Tope público:** 2.160 FRE.  
**Tope total con confirmación:** 2.200 FRE.

Las fidelidades cortas son cortes predeclarados de la misma curva de entrenamiento completa; no generan otra población ni se cobran dos veces. Las sondas de fidelidad 0 y el cómputo del selector sí se suman al tiempo real, aunque no sean FRE.

Al promedio publicado de ARLBench, 2.160 FRE equivalen a unas 452 GPU-horas. Se reserva un factor máximo de cuatro por la extensión de representación y preentrenamiento: **envolvente de planeación de 1.810 GPU-horas públicas**. No es una ETA. El primer piloto GPU debe reemplazar esta envolvente por mediciones reales antes de autorizar el corpus.

## 3. Ejecución escalonada

| Etapa | Propósito | Se carga a | Salida obligatoria |
|---|---|---|---|
| P0 | Integración pública y costo por representación | primeras corridas de desarrollo | medición por hardware y decisión de reducir el espacio |
| P1 | Piloto de fidelidades, margen, cobertura y potencia | desarrollo | protocolo congelado o abandono temprano |
| P2 | Corpus y calibración | 1.080 FRE | selector y envolventes calibrados; prueba intacta |
| P3 | Comparación pública ciega | 720 + hasta 360 FRE | H1, H2 y H3 públicas |
| P4 | Confirmación financiera | hasta 40 FRE | transferencia externa sin rescatar hipótesis públicas |

## 4. Cómo se evita gastar el presupuesto dos veces

- Los datos públicos existentes de ARLBench pueden usarse para implementar interfaces y reproducir comparadores, pero no etiquetan las nuevas representaciones.
- Los FRE se registran por configuración, tarea, semilla, presupuesto, hardware y estado terminal.
- Una corrida fallida consume el costo que realmente usó; no desaparece del denominador.
- Desarrollo, calibración y prueba tienen listas de unidades separadas y selladas.
- Las cinco semillas de evaluación no se cuentan como cinco tareas.
- Ningún resultado financiero cambia umbrales del dominio público.

## 5. Disposición de recursos

El programa de tres años es viable si P0 confirma que la envolvente cabe en la infraestructura disponible. La campaña completa no se lanza como una sola grilla. Cada etapa requiere el artefacto terminal de la anterior y una decisión explícita de continuar.

Si P0 mide un costo superior al factor cuatro, se reduce primero la cantidad de familias de representación y después los niveles de fidelidad redundantes. No se reducen silenciosamente semillas, tareas de prueba ni comparadores para conservar una afirmación positiva.

## 6. Estado actual

No existe todavía un corpus L1 suficiente para entrenar el selector propuesto. Las ocho celdas SAC materializadas en el frente financiero son evidencia de ingeniería y motivación, no metatareas. Los resultados locales actuales permiten afirmar únicamente:

- tres familias de señales mostraron valor relativo frente a controles en diagnósticos internos;
- dos familias permanecen sin evidencia útil bajo el candidato probado;
- ninguna de las cinco ha demostrado habilidad predictiva absoluta;
- la fusión actual es sospechosa de cuello de información;
- un preflight fuerte ejecutó el camino mecánico y terminó sin checkpoint promovible.

Estas observaciones justifican la pregunta. No cuentan como prueba de H1, H2 o H3.
