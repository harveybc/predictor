# Expediente: selección automática de transformaciones temporales

**Fecha:** 2026-09-05

**Estado:** borrador doctoral listo para revisión del autor

**Objeto:** selección de grafos pequeños de transformación para tareas temporales nuevas, bajo presupuesto y con abstención

## Documento principal

- Fuente LaTeX: `docs/propuesta_doctoral_transformaciones_series_temporales.tex`
- Bibliografía: `docs/propuesta_doctoral_transformaciones_series_temporales.bib`
- PDF: `docs/propuesta_doctoral_transformaciones_series_temporales.pdf`

## Anexos de trabajo

- `01_MATRIZ_HIPOTESIS_EXPERIMENTOS.md`: operacionalización completa de las hipótesis.
- `02_ARQUITECTURA_TRANSFERENCIA_DOIN.md`: separación entre validación científica, modelos de referencia e integración con DOIN.
- `03_WORKPLAN_PATCH_TRANSFORMACIONES_DOIN.md`: cambio propuesto al plan de trabajo; no modifica campañas activas.
- `04_AUDITORIA_INTERNA_TRES_JURADOS.md`: lectura hostil desde estadística, AutoML y sistemas.

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

No se inicia una campaña GPU por la existencia de este documento. El primer trabajo ejecutable es un banco CPU pequeño y determinista.
