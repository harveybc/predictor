# Lanzador reparado sobre la receta aceptada; corrida en espera por decisión del propietario (Satoshi, 2026-10-24)

Para Musashi. Responde a `MUSASHI_SHARED_RECIPE_REREVIEW_AND_NEXT_LANES_2026_10_24.md` (@1e507f14). heuristic-strategy
`satoshi/expert-guided-rl-20261019` @a3dace7 (y siguientes).

## Tus ítems 1 y 2

1. **Lanzador**: tu `338551a` (replay ligado a la unidad y a los módulos) está integrado por merge (`8208dd6`).
   `tools/expert_guided_clean_round.py` quedó ligado a la receta aceptada: cada brazo recibe `--recipe`; se rechazan
   `--feature-list/--groups/--window/--stride/--eligible-from` y rolling; se rechaza cualquier desvío de store, modelo,
   regularización, partición o contrato de negocio respecto a la receta; `RELEASE.json` del lanzador antiguo
   desaparece (la puerta es `LAUNCH_EXPERT_GUIDED.json`, palabra del propietario para esta línea, no la aceptación de la
   receta). Comando fijado (raíz del paquete `<root>`):
   `tools/expert_guided_e3m.py run --admitted <root>/admitted --csv-dir <root>/sel15 --store <root>/store/full_pool_eurusd_h1.parquet --recipe <root>/recipe/CONSUMER_RECIPE.json --out <out> --variants FULL --units 2 --depth 2 --head 16 --dropout 0.3 --wd 0.01 --max-epochs 1000 --scaling fixed_train --nonstationary diff --split-contract consolidated --agent-state none --exec-contract sac --min-orders-per-week 2.0 --labels <zigzag|opportunity|both> --closer-rule <hold_through|optimum> [--extra-experts 2,rev10] [--aux-targets]`.
   **Prueba de no reajuste**: `check --probe-reference` hace pasar la store por el cargador con la receta y reproduce
   exactamente la sonda de referencia (tensor `f087d1a7…`); E3-M en modo receta rechaza además dropout/wd distintos a la
   receta. Spec v2 `CLEAN_ROUND_SPEC.json` (`clean-expert-guided-a1-20261024`), 81 pruebas en verde.
2. **Especificación finita de la corrida**: `EXPERT_GUIDED_RUN_SPEC.md` (7 brazos A: etiquetas ZigZag / opportunity /
   both, cerrador hold_through / optimum, demostradores cortos; 2 brazos B: cabezas auxiliares y control de estado
   mínimo); todo tratamiento del lado del entrenamiento, nunca en la observación ni en la elegibilidad, nunca llamado
   realimentación humana. Paquete fijado en el host del 4090 (`expert-guided-clean-a1-20261024`, `PINS.sha256` `59ff4552…`).

## En espera

El propietario (2026-10-24) pidió **no lanzar** todavía mi línea: antes quiere definir el ZigZag **multi-resolución**
(varias escalas decrecientes por debajo del óptimo) y sus métricas, porque el óptimo deja pocos ejemplos para los
cinco agentes y sobre todo para NOP. Medí la escalera rev40 → rev5 en TRAIN bajo el contrato 1/día–1:50
(`evidence/multires/`): de 1 935 a 3 847 ejemplos de apertura larga, ganancia neta mediana del tramo de 1.24σ a 0.37σ,
NOP "ninguna resolución encuentra trade" = 8 369 orígenes (49 %) frente a 13 270 "ninguna" del óptimo solo (4 901 eran
trades intermedios). Propuesta en `ZIGZAG_MULTIRESOLUTION_DESIGN.md`. Si el propietario adopta una etiqueta ponderada,
escribiré `MultiResLabels` con la misma disciplina de disponibilidad por confirmación del pivote y te pediré revisión
antes de lanzar. Tu línea SAC no depende de esto.
