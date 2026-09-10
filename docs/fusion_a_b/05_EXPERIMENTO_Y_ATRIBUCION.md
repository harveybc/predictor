# 05 — Experimento y atribución

**Vuelve a:** [dictamen principal](../DICTAMEN_RETSU_FUSION_A_B_2026_09_02.md)

Un experimento que no puede fallar no es tesis. Un factorial que no se puede pagar es un afiche.

---

## Banco y unidad

| Pieza | Decisión |
|---|---|
| Primario | HPO-B-v3, splits oficiales por **tarea**. Protocolo @25 / @50 / @100 más costo-hasta-objetivo. |
| Confirmatorio | **Una** familia de series de tiempo del candidato. Presupuesto de cómputo acotado. Modelos pequeños. |
| Fuera | LLM compactos, negociación algorítmica como condición de validez, bake-off contra Bittensor/Gensyn/Akash. |
| Unidad | Tarea no vista. No el trial. |
| Repetición | ≥30 tareas/familia, 5 semillas de la política, bootstrap por tarea, Holm. |
| Runtime | Simulador. DOIN opcional, no bloquea. CPU salvo orden explícita. |

HPO-B en `doin-domains` es **lookup**. Úsenlo como lookup. No finjan que el frente de dominios “ya entrenó un HyperBO”.

---

## Factorial que cabe (no el 2×2×3 de Satoshi)

**Primario (4 celdas):**

\[
\{\text{selectivo},\; \text{BO sin transferencia}\} \;\times\; \{\text{corpus limpio},\; \delta\text{ inyectada}\}
\]

Controles **dentro de cada celda:** random search. Mismo presupuesto, mismos inits.

**Anidado, solo para H2 (no eje de H1):**

- Generador estratégico del corpus (censura de fallos, duplicados, relabel de \(y\), mentir política).
- Mecanismo on / off (pago+auditoría vs aceptar todo).
- Adversario: repertorio fijo **más** un buscador de mejor respuesta contra la regla **publicada**.

**Ablación, no eje, y solo si el piloto da potencia:**

- Procedencia consciente vs ciega (H3a).
- Asignación con \(\hat v\) vs por costo solo (H3b). El piloto elige **uno** como contraste primario de H3.

MacKay / compresión / cuantización: ablación de descriptor. Si no ganan NLL o ranking fuera de muestra, **se cortan**. No hay duelo. B ya lo aceptaba; hay que cumplirlo.

---

## Métricas

**Lado transferencia (H1):** arrepentimiento simple normalizado; costo-hasta-objetivo; frecuencia de abstención; frecuencia de transferencia negativa (selectivo peor que BO-sin-transferencia más que \(m\)).

**Lado mecanismo (H2):** utilidad de desviación del generador; \(\delta\) observada (fracción de filas alteradas que sobreviven al sello); gasto en configs bajo el percentil 50 del espacio tabulado.

**Calibración:** ECE / cobertura de intervalos en tareas de validación meta, **antes** de test. Si no está calibrado, la abstención es un botón.

---

## Adversario

Sin mejor respuesta, H2 es un examen que escribe el candidato.

El buscador (evolutivo o RL, presupuesto retenido) maximiza \(U_G\) contra \((p,\pi,\hat v)\) **publicados**. No se retunean defensas después de ver al atacante. Si el atacante gana, H2 falla. Punto.

Repertorio fijo (para comparar con literatura de envenenamiento): censura, duplicado, relabel \(f\in\{0,0.2,0.5\}\), shuffle de política de origen.

---

## Refutación mínima (la que hay que poder correr en CPU)

1. Entrenar \(\hat v\) en HPO-B train, limpio.
2. Misma receta con \(f=0.5\) relabel.
3. Selectivo vs BO-sin-transferencia vs HyperBO o PFNs4BO (un baseline de transferencia, no tres).
4. Si el selectivo no gana en limpio: H1 en problemas. Si gana en limpio y **también** gana igual de bien con \(f=0.5\) sin abstención: la abstención no hace nada y H1/H3 mienten. Si se hunde con \(f=0.5\) y la abstención lo devuelve a no-inferioridad: eso **es** el resultado.

El brazo de mejor respuesta se corre **después** del piloto, no como demo.

---

## Potencia

El piloto, fuera de confirmación, fija \(m\), \(\delta\), \(p\), y el \(N\) para potencia ≥ 0.80 en el contraste primario de H1. 20/5/10 de A **no se heredan**: eran de otro objeto. Si el piloto no llega, se sube \(N\) o se mata H3 como primaria. No se “confía” en 30×5 porque B lo escribió.

---

## Lo que este diseño **no** prueba

- Que una red descentralizada aprende en equilibrio.
- Que DOIN es necesario.
- Que el método gana a OptFormer entrenado en Vizier (no tienen Vizier).
- Que series de tiempo = mercados financieros reales. CSV histórico. Sin live.
