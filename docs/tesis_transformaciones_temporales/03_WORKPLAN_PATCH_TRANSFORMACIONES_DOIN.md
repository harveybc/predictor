# Work plan patch propuesto: transformaciones temporales y transferencia a DOIN

**Fecha:** 2026-09-05

**Estado:** `PROPOSED_FOR_REVIEW`

**No modifica:** campañas activas, resultados cerrados, servicios, GPU ni periodos retenidos

## 1. Motivo

El plan vigente ya contiene preguntas de preprocesamiento, selección de características y metaoptimización, pero no separa con suficiente nitidez:

- calibración con señal y perturbación conocidas;
- utilidad de pronóstico en datos públicos;
- selección entre tareas;
- adopción como genes o configuraciones de un dominio DOIN.

Sin esa separación, una mejora local puede entrar al optimizador antes de demostrar que es reproducible o generalizable.

## 2. Inserción propuesta

Este paquete se inserta como una línea transversal entre la recuperación de evidencia de datos del plan 17 y cualquier ampliación costosa de representaciones o selección del plan 38.

No sustituye esos documentos. Añade una escalera de evidencia previa para nuevos operadores de entrada.

## 3. Paquetes de trabajo

### T0. Contrato y censo

- definir tarea, variable, horizonte y disponibilidad;
- inventariar datasets públicos y licencias;
- congelar modelos de referencia, métricas y costo;
- medir si existen suficientes tareas independientes.

**Salida:** inventario y veredicto `SUFFICIENT_FOR_PUBLIC_MATRIX` o `INSUFFICIENT_SUPPORT`.

### T1. Banco CPU de verdad conocida

- implementar generadores, perturbaciones y métricas por variable;
- probar ejecución incremental sin futuro;
- comparar identidad, filtros hacia atrás y entrega con residuo;
- publicar los regímenes donde cada diagnóstico está calibrado.

**Salida:** operadores `LAB_CALIBRATED`, `REGIME_LIMITED` o `REJECTED`.

### T2. Utilidad pública

- ejecutar el espacio pequeño en modelos sencillos congelados;
- medir pronóstico, extremos, residuo, retraso y costo;
- dejar fuera familias completas;
- retirar operadores sin utilidad incremental.

**Salida:** registro tarea-grafo y lista `PUBLICLY_ELIGIBLE`.

### T3. Meta-selección

- vecino por tarea;
- modelo de árboles;
- selección bajo presupuesto;
- abstención calibrada;
- comparadores random, Hyperband y BOHB.

**Salida:** `TRANSFER_DEMONSTRATED`, `SIMPLE_BASELINE_SUFFICIENT` o `TRANSFER_NOT_IDENTIFIED`.

### T4. Adaptador DOIN

- portar únicamente operadores `PUBLICLY_ELIGIBLE`;
- verificar paridad byte a byte entre referencia y adaptador;
- añadir identidad, máscara por variable y parámetros licenciados al dominio;
- materializar una nueva identidad de campaña sin tocar las anteriores.

**Salida:** preflight CPU sin autoridad económica.

### T5. Validación aplicada

- comparar contrato vigente, mejor fijo público, `top-k` y DOIN sin warm start;
- usar presupuestos y periodos pareados;
- conservar la autoridad L2 y el periodo retenido;
- medir si el ahorro público sobrevive al dominio financiero.

**Salida:** adopción, rechazo o abstención por operador.

## 4. Orden y paralelismo

```text
T0 -> T1 -> T2 -> T3 -> T4 -> T5
```

T0 y la infraestructura mínima de T1 pueden avanzar en CPU mientras continúan trabajos independientes. T4 no comienza con resultados sintéticos solamente. T5 no comienza sin paridad del adaptador.

## 5. Cambios solicitados al plan vigente

### Plan 17: datos y preprocesamiento

Añadir a E2 una frontera de evidencia:

> Los transformadores nuevos se calibran primero en un banco con perturbación conocida y se juzgan después por utilidad pública fuera de muestra. La selección financiera no puede redefinir la señal limpia ni promover un operador que falló su contrato temporal.

### Plan 38: selección y optimización

Añadir antes de expandir el espacio de características:

> Un operador de transformación entra como gen de L2 solo con identidad versionada, intervalo de parámetros licenciado, paridad del adaptador y estado `PUBLICLY_ELIGIBLE`. Un meta-selector L3 puede proponer población inicial, pero no declarar el campeón ni consultar la validación retenida.

El archivo del plan 38 tiene cambios activos ajenos; estas cláusulas no deben insertarse hasta revisión y resolución de su estado actual.

## 6. Presupuesto de cómputo

- T0-T1: CPU solamente.
- T2: CPU primero; una única confirmación neuronal tras gates.
- T3: modelos tabulares simples antes de cualquier arquitectura compleja.
- T4: preflight CPU.
- T5: GPU solo con diseño sellado y autorización de campaña.

No se reserva GPU para producir una matriz que modelos lineales pueden censar.

## 7. Decisiones del propietario

No se necesita una decisión del propietario para redactar contratos, ejecutar tests unitarios o construir el banco CPU sin datos privados.

Sí se necesita decisión antes de:

- descargar o incorporar un dataset con licencia restrictiva;
- abrir periodos retenidos del dominio;
- lanzar una confirmación GPU;
- modificar el espacio del dominio DOIN usado por campañas económicas;
- promover un operador a configuración predeterminada.

## 8. Criterio terminal

La línea se considera científicamente cerrada cuando ocurre uno de estos resultados:

1. hay transferencia demostrada y operadores trasladables a DOIN;
2. un baseline sencillo basta y se adopta sin meta-selector complejo;
3. los diagnósticos no identifican la transformación adecuada y la abstención es el resultado;
4. ningún operador supera identidad con costo completo.

Los cuatro son resultados válidos. Solo el primero autoriza la integración completa.
