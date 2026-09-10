# Expediente: selección automática de transformaciones temporales

**Fecha:** 2026-09-05

**Estado:** PDF committeado (`5449bed`); Satoshi `ACCEPT_WITH_INTEGRATION_NOTES` (`3116c362`); Retsu re-verificó 2026-09-06.

**Objeto:** selección de grafos pequeños de transformación para tareas temporales nuevas, bajo presupuesto y con abstención

**Correo de admisión (G12):** sigue siendo la propuesta L2/RL hasta decisión explícita de Harvey. Satoshi recomienda *esta* como madre de La Sabana; Retsu no cambia el adjunto por omisión. Carta: `docs/RETSU_TO_HARVEY_G12_TRANSFORMACIONES_VS_L2_2026_09_06.md`.

## Documento principal

- Fuente LaTeX: `docs/propuesta_doctoral_transformaciones_series_temporales.tex`
- Bibliografía: `docs/propuesta_doctoral_transformaciones_series_temporales.bib`
- PDF: `docs/propuesta_doctoral_transformaciones_series_temporales.pdf`

## Anexos de trabajo

- `01_MATRIZ_HIPOTESIS_EXPERIMENTOS.md`: operacionalización completa de las hipótesis.
- `02_ARQUITECTURA_TRANSFERENCIA_DOIN.md`: separación entre validación científica, modelos de referencia e integración con DOIN.
- `03_WORKPLAN_PATCH_TRANSFORMACIONES_DOIN.md`: cambio propuesto al plan de trabajo; no modifica campañas activas.
- `04_AUDITORIA_INTERNA_TRES_JURADOS.md`: lectura hostil desde estadística, AutoML y sistemas.
- `05_DICTAMEN_SATOSHI_VERIFICACION_E_INSERCION_DOIN_2026_09_05.md`: verificación byte a byte e inserción T0–T5 en el plan DOIN.

## Decisiones que gobiernan el expediente

1. `predictor` no es el sustrato científico de la propuesta. Puede ser un adaptador de pronóstico, igual que cualquier modelo externo.
2. Las hipótesis se prueban primero con señales cuya perturbación es conocida y después con series públicas.
3. En datos naturales no se afirma conocer la señal limpia ni la SNR verdadera.
4. Una transformación se acepta por utilidad fuera de muestra y preservación de información, no por producir una serie más suave.
5. DOIN consume únicamente operadores que ya superaron compuertas independientes y vuelve a validarlos dentro del dominio aplicado.
6. El meta-selector propone una población inicial o una lista corta. No reemplaza la optimización de nivel 2 ni el periodo retenido.
7. Una conclusión nula es un resultado: identidad, búsqueda sin transferencia o abstención pueden ser la decisión correcta.

## Secuencia de evidencia

```text
señales sintéticas y semisintéticas
        -> calibración de diagnósticos y conservación
series públicas no financieras
        -> utilidad de pronóstico y transferencia entre tareas
operadores aprobados y versionados
        -> adaptación del dominio DOIN
validación aplicada retenida
        -> adopción, rechazo o abstención
```

No se inicia una campaña GPU por la existencia de este documento. El primer trabajo ejecutable es un banco CPU pequeño y determinista (T0+T1 = G3; un banco, no dos). No se toca B4.

## Divergencia deliberada: Tabla 1 vs STEP 11

La Tabla 1 del PDF deja la corrupción/máscara del autoencoder **fuera** del núcleo doctoral. El STEP 11 del work plan la mantiene como módulo (máscara train-only sobre el AE que ya existe; compuertas 11A–11D; un nulo cierra).

No es contradicción: **tesis ⊂ work plan**. Ningún operador debe “resolverlo” fusionándolos, ni promover la máscara a capítulo, ni borrar el experimento de laboratorio. Recorte: `docs/SUGERENCIAS_STEP_11_PARA_WORK_AGENT.md`.
