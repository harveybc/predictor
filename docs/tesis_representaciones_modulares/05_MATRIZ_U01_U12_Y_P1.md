# Matriz U01-U12 y cierre de P1

**Paquete de entrada:** Takeshi, revisión 2 (`PAQUETE_TAKESHI_AUDITORIA_Y_FIGURAS.zip`, 2026-09-07)  
**Documento evaluado:** `docs/propuesta_doctoral_representaciones_temporales_modulares.tex`  
**Estado:** revisión 3 posterior a las correcciones de Retsu y a la auditoría de Musashi.

Esta matriz registra qué llegó al PDF. No concede validez científica por sí misma; los puntos que dependen del piloto o de la revisión sistemática permanecen abiertos de forma expresa.

## U01-U12

| Código | Disposición | Ubicación final | Comprobación |
|---|---|---|---|
| U01 | Aplicado con alcance limitado | §2.3 y §6.1 | Exportar codificador/decodificador y declarar reajuste se presenta como experiencia previa, no como ensamblaje diferenciable ya probado. |
| U02 | Aplicado | §2.1, §4.2, Fig. 1 | Detector, integrador y adaptación tienen responsabilidades distintas dentro de cada rama. |
| U03 | Aplicado | §4.2 | R0/R1/R2 quedan definidos; R1 exige ausencia de gradiente y R2 conexión efectiva con la pérdida final. |
| U04 | Reformulado | §4.2 | TCN es la realización principal por tener campo receptivo calculable; no se presenta como arquitectura universal. |
| U05 | Aplicado | H1, §5.1 | El contraste principal usa el mismo régimen; el preentrenamiento no puede explicar solo al método. |
| U06 | Aplicado | §6.3 | Todo corpus de preentrenamiento debe declarar solapamiento; una familia ya vista no cuenta como completamente nueva. |
| U07 | Aplicado | §5.2 | El costo incluye perfil, preentrenamiento, decodificador, ajuste, evaluación y reutilización amortizada. |
| U08 | Cerrado como limitación | §2.3 y §6.1 | Se localizaron opciones y configuraciones, pero no el ensamblaje diferenciable; la propuesta exige una prueba de flujo de gradientes antes de usarlo. |
| U09 | Aplicado y redibujado | Figs. 1 y 2 | Se conservaron arquitectura y protocolo confirmatorio; se eliminaron códigos internos y cruces visuales. |
| U10 | Aplicado | §2.3 y bibliografía | Bengio, Ti-MAE y Keras se usan solo para situar preentrenamiento y reajuste. |
| U11 | Aplicado | §4.2 | La pareja preentrenada se preserva y las copias reajustadas mantienen linaje separado. |
| U12 | Aplicado por recorte | Todo el documento | La generación adversaria de otros proyectos no aparece en la propuesta; E0 usa sintético únicamente para identificar mecanismos. |

## Observaciones P1 de la primera auditoría

| Observación | Disposición en revisión 3 | Pendiente legítimo |
|---|---|---|
| P1-01: decisión de H1-H3 ambigua | Cerrada: se define $\Delta=\mathrm{MASE}_{método}-\mathrm{MASE}_{control}$ y cuatro veredictos por intervalo y margen $\varepsilon$. | El valor de $\varepsilon$ y la precisión mínima se fijan en E1, antes de E2. |
| P1-02: campo local confundido con memoria total y cabezal impreciso | Cerrada: se separan detector, integrador y cabezal; H3 compara fusión de secuencias con resumen temprano y publica memoria de activaciones. | Anchos y tolerancia de capacidad se fijan en el piloto. |
| P1-03: regla incompleta y normalización destructiva | Cerrada en diseño: retardos/frecuencias usan coordenadas comparables y el escalado se aprende en E1, no por tarea; $A_\eta$ es jerárquica con cuadrícula finita. | Fórmula final del perfil, corte y cuadrícula se congelan en E1. |
| P1-04: perfiles marginales no identifican relaciones cruzadas | Cerrada: agrupación significa compatibilidad de escala; H3 usa un diagnóstico rezagado separado y su prueba principal es sintética. | Elegir el diagnóstico público; si es inestable, H3 pública queda solo descriptiva. |
| P1-05: niveles de evaluación y linaje | Cerrada: familias de desarrollo y confirmación se separan; dentro de tarea hay entrenamiento, validación y prueba reservada. | Censo definitivo del banco. |
| P1-06: ablaciones no aislaban efectos | Cerrada en protocolo: monolítico, modular aleatorio, perfiles permutados, campo común y resumen temprano tienen funciones distintas. | El conjunto mínimo exacto se congela antes de E2. |
| P1-07: vecinos directos ausentes | Fortalecida: DUET y MSGNet se añaden; DUET pasa a comparador directo y la brecha se declara provisional. | Revisión sistemática del primer semestre. |
| P1-08: banco, MASE y presupuesto incompletos | Cerrada en contrato: unidades multivariadas elegibles, denominador MASE cero tipado y ecuación de costo completo. | Tamaño final derivado del piloto de precisión. |

## Respuestas a las doce preguntas de Takeshi

1. La evidencia existente muestra opciones para cargar y reajustar un extractor, no el flujo de gradientes completo.
2. Activaciones precalculadas y submodelo diferenciable se consideran rutas distintas; solo la segunda permite R2.
3. La realización inicial será una TCN causal; el codificador CNN previo es antecedente de viabilidad, no plantilla obligatoria.
4. Cada rama declara $L_j$, $d_j$, campo receptivo y correspondencia temporal; $L_j=L$ no se supone.
5. El decodificador es auxiliar del preentrenamiento y no participa en el pronóstico.
6. El preentrenamiento será por tarea salvo que E1 apruebe un corpus compartido con solapamiento declarado.
7. R0/R1/R2 se estudian en E1 y un régimen se congela para E2.
8. R1 comprueba gradiente nulo en el detector; R2 comprueba gradiente finito y actualización efectiva.
9. El codificador y decodificador originales se preservan; cada copia reajustada tiene identidad propia.
10. H1 iguala régimen, entradas, semillas y presupuesto de ajuste; el costo previo se registra aparte y en total.
11. Perfiles, agrupación jerárquica y TCN forman una realización falsable; la tesis no reclama que sea la única posible.
12. Los P1 quedan dispuestos en la tabla anterior, con sus pendientes preexperimentales visibles.

## Límite factual U08

No se afirma que el codificador histórico ya haya corrido de extremo a extremo dentro del predictor actual. Antes de cualquier experimento R1/R2 deberá existir una prueba que muestre, sobre el modelo compuesto, qué pesos reciben gradiente y cuáles cambian después de un paso de optimización.
