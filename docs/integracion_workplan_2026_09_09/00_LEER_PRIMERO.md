# 00 — Leer primero

**Fecha:** 2026-09-09  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**No es una orden de campaña GPU, live, B4 ni de cambiar G12.**

Harvey pidió recoger **todas** las propuestas que surgieron al redactar el doctorado, el work plan informacional en el que estás, la propuesta modular vigente, y una lane nueva de métricas de información / grafos / cubo OLAP (incluida una hipótesis de early stopping informacional). Quiere que el work plan se **reinicie en la cola experimental** si hace falta, porque seguir entrenando con preproceso por defecto y sin esas métricas es peor que reordenar.

Este directorio es el paquete de integración. No sustituye los STEP 01–13 ni reescribe las propuestas. Las indexa y te da un orden de inserción.

## Cómo leer

| # | Archivo | Qué resuelve |
|---|---|---|
| 00 | este | índice y reglas de lectura |
| 01 | `01_DONDE_ESTA_MUSASHI.md` | dónde vas hoy, dos work plans distintos, cinco frentes |
| 02 | `02_INVENTARIO_PROPUESTAS.md` | todas las propuestas en `docs/` y qué no fusionar |
| 03 | `03_LANE_INFORMACION_GRAFOS_Y_EARLY_STOP.md` | métricas nuevas + hipótesis; qué **no** es Kolmogorov |
| 04 | `04_PLAN_INTEGRACION_JERARQUICO.md` | paquetes I0–I14, orden, compuertas |
| 05 | `05_DATOS_PRECONDICIONES_Y_CUBO.md` | datos, repos, esquema OLAP, precondiciones |

Carta de cubierta (misma fecha): `docs/RETSU_TO_MUSASHI_INTEGRACION_WORKPLAN_2026_09_09.md`.

## Tres reglas que no se negocian aquí

1. **Tesis ⊂ work plan.** Cada propuesta doctoral es un recorte. El work plan es más ancho. Nadie fusiona L2 + modular + memorización + transformaciones + incentivos en un solo PDF de admisión.
2. **Protocolo escrito ≠ experimento hecho.** STEP 01–07 están especificados. Retsu recortó el 2026-09-05: cero plugins, cero curvas. El primer *hecho* sigue siendo un banco CPU.
3. **No se toca GPU, B4, live ni el cubo poblado** para “probar” este paquete. I1 puede diseñarse contra una base throwaway. I5–I9 arrancan en CPU.

## Qué se reinicia y qué no

| Se reinicia | No se reinicia |
|---|---|
| La **cola experimental** del pipeline informacional (qué se mide primero) | Los protocolos STEP 01–13 y los recortes RETSU |
| El contrato de **qué entra al cubo** en cada epoch/run | T0 del `preprocessor` (ya existe) |
| La prioridad: instrumentar información **antes** de barrer arquitecturas | G12 / L2 como adjunto de correo, salvo orden explícita de Harvey |
| | Campañas selladas, WP1 de dominios, capital live |

## Hipótesis nueva (Harvey, 2026-09-09)

Hay un patrón buscable entre:

- complejidad del modelo (proxy por compresión de pesos + métricas de grafo);
- información estimada de entradas y del target (entropía / longitud comprimida / cota teórica cuando exista generador);
- el punto de overfitting / early stopping.

Es **hipótesis**. Se registra. No sustituye `early_patience` hasta que falle o gane en tareas reservadas (I4).
