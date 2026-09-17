# 07C — Enmienda del diagnóstico de gemelos: comparaciones sensibles (L2)

Estado: **sellada antes de medir**. Datos sellados en `tools/df_d3_design.py`
(`D3_TWIN_SENSITIVITY_AMENDMENT_V1`, `design_sha256` propio, cita 07B por digest;
`D3_DESIGN_CURRENT` apunta a ella). No cambia soporte 50, imputación, umbrales ni parámetros;
no relaja ninguna prueba causal del candidato. `NON_GOVERNING`.

## Hecho demostrado por unidad (no hipótesis)

`d3_mechanics_v2/L2_TWIN_SENSITIVITY_DIAGNOSTIC.json`: para las 15 variables MCAR de
`wavelet_trailing` se enumeraron cortes (51 de prefijo, 21 de perturbación), emisiones del gemelo
centrado, comparaciones, y para cada comparación si el soporte del gemelo (w=50, half=25, alcance
a la derecha **24**) cruza el corte. Las cinco variables `FAILED` en v2 tienen **0 comparaciones
sensibles** en ambas pruebas: la salida disponible más cercana a un corte está a 47–256 muestras.
Las `PASSED` tienen ≥1 sensible o una detección efectiva. Es decir, en v2 el gemelo no fue
refutado ni confirmado allí: no fue probado.

## §1 Alcance declarado del gemelo

Cada gemelo declara `reach_right(n)`: cuántas muestras después de `i` consume su salida `i`.
Ventana centrada de `w`: `w − w//2 − 1`; `filtfilt` de fase cero: la serie entera; `cusum_lookahead`:
su `ahead`. Un gemelo con alcance 0 se rechaza como control: no puede demostrar no causalidad.

## §2 Comparación sensible

* **Prefijo**: la salida `i` comparada en el corte `c` es sensible sii `i + reach_right > c`.
* **Perturbación futura**: igual (la perturbación reemplaza todo lo posterior a `c`).
* Cruzar geométricamente es **necesario, no suficiente**: un coeficiente cero, la saturación o la
  falta de datos pueden impedir el efecto. Por eso la política no acepta por geometría.

## §3 Política

| hechos | resultado |
|---|---|
| alguna comparación (sensible o no) mueve valor, máscara de disponibilidad o instante de emisión | `PASSED` — **detección**; jamás se descarta por una declaración de soporte del mismo operador |
| ninguna infracción y **0** comparaciones sensibles | `INSUFFICIENT_TEST` (indecidido ⇒ `INCONCLUSIVE`) |
| ninguna infracción y ≥1 comparación sensible | `FAILED` — el gemelo declarado no causal no muestra efecto donde su alcance cruza el corte |

Registrado por control: `twin_emissions`, `twin_comparisons`, `twin_sensitive_comparisons`,
`twin_detections`, `twin_nearest_output_to_cut`, `twin_reach_right`, `first_detection`.

Las pruebas causales del candidato (`prefix_all_available`, `future_perturbation`) **no** se
restringen a la máscara de sensibilidad del gemelo.

## §4 Controles probados

Centro conocido (detección sensible); impulso futuro justo tras el corte (alcance 1, detectado);
pesos futuros extremos cero (cruce sin efecto ⇒ declaración fallida, no aprobación por geometría);
missingness aislada/bloques/MCAR; bordes y restart; controles deliberadamente no causales que
fallan **por máscara** y **por instante de emisión** aparte (la emisión dependiente de datos futuros;
una emisión posterior al corte se excluye por la regla de salidas retardadas de 07A, y por eso el
control depende de valores, no de marcas de tiempo). Se conservan el control con datos completos
y los tres rechazos por warm‑up.

## §5 Alcance del replay

Los hechos de sensibilidad no están en las filas conservadas de v2 (requieren las máscaras de
disponibilidad del gemelo por corte): hace falta medir de nuevo **solo** `non_causal_twin` para los
siete operadores con gemelo sobre **toda** la población congelada (también los casos antes
aceptados). Las otras once pruebas se **heredan** de `d3mech-v2` por digest: el freeze del replay
declara, por prueba, la corrida y el freeze de origen, y el verificador exige que cada fila
heredada esté ligada a ese origen y cada fila medida a este diseño.
