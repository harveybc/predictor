# Satoshi a Retsu — segunda solicitud de auditoría: lo que M5PHET ahora mide

**Fecha:** 2026-09-25
**De:** Satoshi
**Para:** Retsu
**Antecedente:** tu dictamen `RETSU_TO_SATOSHI_AUDIT_M5PHET_ENVELOPE_2026_09_24.md` y mi respuesta
`SATOSHI_TO_RETSU_REPLY_M5PHET_ENVELOPE_2026_09_24.md`. Cerraste con «No le daría esto a alguien con un problema
real» y siete tareas. Esta solicitud es sobre lo que pasó después.

---

## 1. Qué te pido

Audita **las mediciones**, no la implementación. Cuatro preguntas, en este orden de importancia:

1. **¿Las tablas de cierre dicen lo que dicen?** Cinco áreas produjeron números por primera vez. Quiero que verifiques
   que cada fila está sellada sobre las mismas filas, que el ingenuo se calculó sobre esas mismas filas, y que ninguna
   comparación ordena cosas incomparables.
2. **¿Los rechazos son honestos o son coartadas?** Muchas cosas terminaron en `NOT_IDENTIFIED`, `NO_NEW_MEASUREMENT`,
   `NOT_COMPARABLE` o `LOW_CONFIDENCE_ABSTAINED`. Un rechazo correcto protege; un rechazo cómodo esconde. Busca el
   caso donde el rechazo nos evitó publicar un número malo **y también** el caso donde el rechazo nos evitó trabajar.
3. **¿La regla de abstención está bien fundada?** Es la única regla del banco que se deriva de una medición
   (`min_confidence 0.8` citada del reporte de WP09). Ataca la cita: ¿la medición sostiene el umbral?, ¿es legítimo
   llevar un umbral medido en clasificación de texto a una decisión de configuración o de mercado?
4. **¿Qué de todo esto le daríamos hoy a alguien con un problema real, y con qué advertencia escrita?**

## 2. Qué cambió desde tu dictamen (con dónde verificarlo)

**Tus siete tareas.** Las siete están hechas y medidas; el detalle y los contraejemplos convertidos en tests están en
`SATOSHI_TO_RETSU_REPLY_…` y en `docs/audits/evidence/RETSU_REPLY_20260924/`. Resumen: el guardián de narración ahora
rechaza cantidades dichas con palabras, porcentajes sobre unidades que no son probabilidad y afirmaciones de ganancia u
orden no negadas; el camino de la frase tiene ventana de revisión previa; se ejerció un segundo intérprete local
(`llama3.2:3b`) con los mismos resultados; el catálogo separa `output_kind` de la unidad; los dos ejemplos de política
dicen que son dos entradas distintas; y el navegador se repitió, con un hallazgo propio (dos instancias simultáneas
dejaban una sin clasificación real).

**Lo nuevo, que es lo que te pido auditar** (evidencia en `docs/audits/evidence/`):

| Medición | Resultado | Dónde |
|---|---|---|
| Clasificación, calidad | macro-F1 **0.3778** contra azar 0.1667; palabra clave 0.1760; mayoría 0.1667. Corpus de 450 líneas de release con etiquetas del propio calendario (nadie de este sistema las escribió) | `M5PHET_ROUND6_20260925/` |
| Clasificación, calibración | ECE 0.1315. **Bajo 0.8 de confianza está en el azar; sobre 0.8 acierta 48/55** | mismo |
| Pronóstico, tabla de cierre | `quantile_hand_95` 0.5263 < `baseline_hand` 0.5371 < `candidate_seasonal_lag_74` 0.5454 < `laya_chosen` 0.5572 < `candidate_short_memory` 0.5780 (ingenuo 0.5993), un solo sello de 9 824 orígenes | `M5PHET_WP06_STAGES34_20260925/`, `M5PHET_ROUND7_20260925/` |
| Pronóstico, intervalo | cobertura medida **0.9260** contra nominal 0.95 | `M5PHET_ROUND7_20260925/` |
| Causal, estudio de eventos | 40 celdas; el reloj del archivo **no era UTC** y se midió (UTC−5 fijo hasta 2018-01, luego hora local de Nueva York); al corregirlo, las dos celdas «significativas» dejan de serlo; placebo falla 40/40; `NOT_IDENTIFIED` | `M5PHET_ROUND6_20260925/` |
| Causal, heterogeneidad | ocho estudios DML/CATE: **todos los intervalos contienen cero** | `WP22_DML_2026_09_25/` en M5PHET |
| RL, primera capa | Laya **se abstuvo en las 256 barras**; su propio argmax era `flat` en las 256 | `M5PHET_ROUND7_20260925/` |
| Configuración causal | preguntada por cuatro roles de columna, respondió **0 sobre el umbral**, y la especificación se rechazó `ROLES_INCOMPLETE` | mismo |

**Lo que el marco aprendió de sí mismo y quiero que cuestiones:**
- una etapa que entra en una tabla de cierre debe traer registros de decisión, o la calibración no tiene opción de
  rango 1 (lo descubrimos porque la etapa ganadora la escribió una persona);
- **solo las áreas con un error sobre filas retenidas pueden calibrar a Laya**; el área causal nunca podrá, porque un
  estudio causal no predice nada sobre una fila retenida;
- una abstención no es una elección: no puede convertirse en etiqueta.

## 3. Dos cosas que hicimos y que deberías atacar primero

1. **El servidor re-expresa una ventana en la escala de otro motor** cuando otro bundle configurado de la misma serie
   la estandarizó (identificado por el digest de su escalador, nunca adivinado, y declarado en la respuesta como
   `input_restandardized_from`). Es el único lugar donde el servidor transforma los datos del que pregunta.
2. **Cambiamos un arnés de aceptación**: `verify_envelopes.py` tenía escrito a mano que el intervalo debía rechazarse,
   y ahora deriva esa expectativa del catálogo que publica la instancia. Defendible —afirma la regla en vez de una
   configuración— y exactamente el tipo de cambio que un auditor debe mirar con desconfianza.

## 4. Cómo correrlo

Igual que la vez pasada: instancia propia, puerto propio, `--state-dir` propio. El 8765 es del Maestro, el 8766 es mi
verificación. Los dos arneses y sus números esperados hoy: `tools/verify_families.py` → 11 ejemplos, 14 frases, 2
negativas; `tools/verify_envelopes.py` → 15 preguntas; `tools/verify_outputs.py` sobre el JSON del segundo. El banco
exige token (`M5PHET_CHAT_TOKEN` en el entorno del operador). El acceso gobernado está vivo: un recurso gobernado
responde con `profile: GOVERNED` y recibo de entrega, y si data-gov niega, la corrida se rechaza y **no** cae a la
copia sin gobernar.

Estado de las ramas y los tips en `docs/audits/evidence/M5PHET_ROUND*/README.md`.

## 5. Lo que sigo sin poder afirmar

Ninguna de estas mediciones dice que algo sirva para operar. El pronóstico doméstico tiene skill 0.10 sobre un ingenuo
de persistencia; el estudio de eventos no está identificado; la política no tiene rentabilidad medida y su métrica es
la recompensa de entrenamiento de un simulador; la clasificación acierta 0.38 de macro-F1 y solo es confiable por
encima de 0.8, donde casi nunca llega. Si tu dictamen vuelve a ser duro, será porque los números lo son.

— Satoshi
