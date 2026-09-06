# Arquitectura de validación y transferencia a DOIN

## 1. Decisión de arquitectura

La propuesta no se implementa dentro de `predictor` ni se valida únicamente con datos financieros.

```text
Banco sintético y público
    -> operadores puros y artefactos de ajuste
    -> evaluación con pronosticadores intercambiables
    -> registro tarea-grafo
    -> meta-selector y abstención
    -> lista de operadores aprobados
    -> adaptador de dominio DOIN
    -> optimización L2 y validación aplicada retenida
```

Esto separa cuatro autoridades:

1. el generador sintético conoce señal y perturbación;
2. el banco público decide utilidad de pronóstico y transferencia;
3. el registro revisado autoriza qué operadores pueden llegar al dominio;
4. DOIN y su periodo retenido deciden si la adaptación aplicada sirve.

Ninguna capa puede acuñar el veredicto de la siguiente.

## 2. Ubicación de código

| Responsabilidad | Ubicación propuesta | Motivo |
|---|---|---|
| Operadores puros, ajuste temporal y banco CPU | repositorio `preprocessor` | ya es la aplicación destinada a transformar CSV y puede permanecer independiente de TensorFlow |
| Pronosticadores de referencia | paquetes públicos y adaptadores pequeños | evita convertir un modelo local en definición del fenómeno |
| Adaptador supervisado opcional | `predictor` | reutiliza modelos existentes, pero no es obligatorio ni exclusivo |
| Registro de operadores aprobados y adaptación aplicada | `doin-plugins` / plugin del dominio | traduce una identidad de grafo a ejecución sin reimplementar la transformación |
| Genes o espacio de búsqueda del dominio | `doin-domains`, después de evidencia | DOIN solo explora parámetros y presencia de operadores ya licenciados |
| Custodia de plan, resultados y decisiones | `agent-multi` | conserva diseño, evidencia, fallos y transiciones del programa |

No se crea un cuarto repositorio. Tampoco se coinstala el ejecutable `preprocessor` dentro del entorno de DOIN: el adaptador consume una biblioteca mínima o artefactos versionados, con dependencias explícitas.

## 3. Contrato mínimo de operador

Cada operador aprobado publica:

```text
operator_id
operator_version
applicability_contract
fit_interval
availability_rule
input_schema
output_schema
learned_parameters_digest
latency_or_delay
compute_cost
failure_policy
```

La interfaz conceptual es:

```text
fit(development_data, availability_contract) -> artifact
transform(past_and_present_data, artifact) -> transformed_data
```

`transform` nunca ajusta parámetros. El mismo prefijo debe producir los mismos bytes sin importar qué datos posteriores existan.

## 4. Qué recibe DOIN

DOIN no recibe una afirmación como “esta variable tiene 8 dB de SNR”. Recibe:

- la identidad de un operador aprobado;
- su artefacto de ajuste y digest;
- variables a las que puede aplicarse;
- parámetros dentro del intervalo validado;
- costo y retraso;
- regla de fallo;
- evidencia pública que permitió incorporarlo.

El genoma del dominio puede incluir `operator_id`, máscara por variable y parámetros licenciados. El meta-selector puede sugerir una población inicial o `top-k`. La evaluación L2 vuelve a medir el desempeño completo; no hereda un campeón.

## 5. Compatibilidad con campañas actuales

- No se modifica una campaña activa ni se reabre una campaña cerrada.
- Un resultado negativo previo sobre un objetivo neuronal no se usa como rechazo universal de transformaciones.
- El banco CPU puede avanzar en paralelo porque no consume GPU ni datos retenidos del dominio.
- La primera integración ocurre en una nueva identidad de dominio y con un control que reproduce el contrato vigente.
- Los resultados financieros no cambian los umbrales de los experimentos públicos.

## 6. Gates de incorporación

Un operador llega a DOIN solo si:

1. supera el contrato temporal incremental;
2. no depende de señal limpia fuera del laboratorio;
3. muestra utilidad o una función diagnóstica explícita en tareas públicas retenidas;
4. conserva eventos y publica la utilidad del residuo;
5. tiene costo y retraso acotados;
6. su artefacto puede reproducirse y verificarse por identidad;
7. el adaptador produce la misma salida que el operador de referencia;
8. el control aplicado reproduce el comportamiento previo antes de añadir el nuevo gen.

Fallar un gate retira el operador o limita su régimen. No se compensa con una mejora en otra métrica.
