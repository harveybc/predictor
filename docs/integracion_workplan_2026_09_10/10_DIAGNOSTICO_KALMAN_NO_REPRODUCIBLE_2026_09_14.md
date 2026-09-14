# Propuesta: diagnóstico estrecho de `local_level_kalman` (no reproducible entre hosts)

Fecha: 2026-09-14. Autor: Satoshi. Revisión: Musashi. Estado: **propuesta, sin ejecutar**.
Responde a B3 de `docs/handoffs/MUSASHI_TO_SATOSHI_HOSTS_NOW_AND_D2_REPORT_CORRECTIONS_2026_09_14.md`.

## 1. Qué está medido y qué no

Medido (D2-R4, 32 unidades del subconjunto congelado, tres roles):

* cuatro estimadores reproducen exactamente (máx \|Δ\| = 0,0 dB) y `ar_residual` queda
  dentro de la tolerancia observada sin ser exacto (1,918e-13 dB);
* `local_level_kalman` **difiere**: ≤ 2,44e-05 dB en confirmación y 1,099 dB en una celda
  de calibración de un régimen de ~85 dB con verosimilitud casi plana;
* COORDINATOR y WORKER_A son idénticos bit a bit en ambos sentidos; WORKER_B difiere en
  ambos sentidos con los mismos digests de código y las mismas cadenas de versión.

**No** medido: la causa. Las cadenas de versión iguales no aíslan el procesador —una misma
rueda despacha núcleos distintos según las capacidades detectadas de la CPU— y nunca se
registraron la identidad de compilación de la biblioteca numérica, su despacho en tiempo de
ejecución ni el camino de convergencia del optimizador. Por eso el hallazgo vigente es
«este estimador no es reproducible entre estos hosts», con causa **no aislada**, y las
desviaciones observadas **no son cotas** para ejecuciones futuras.

## 2. Objetivo del diagnóstico

Decidir entre hipótesis que hoy están confundidas:

| hipótesis | qué la haría cierta | qué la descartaría |
|---|---|---|
| H1 despacho de núcleos por CPU | fijar el núcleo (p. ej. `OPENBLAS_CORETYPE`) iguala los resultados entre hosts | fijarlo no cambia nada |
| H2 identidad de compilación distinta | los hosts resuelven bibliotecas con digests distintos | digests idénticos |
| H3 camino del optimizador | mismas entradas, distinta secuencia de iteraciones/criterio de parada alcanzado | secuencias idénticas y resultado distinto |
| H4 hardware/kernel | el mismo host bajo otro kernel reproduce la diferencia | la diferencia sigue al procesador, no al kernel |

## 3. Qué registra (sin volver a ejecutar el banco)

Sobre **las mismas unidades ya replicadas** (ninguna unidad nueva, ningún operador nuevo):

1. identidad de la biblioteca numérica: ruta y SHA-256 del objeto compartido efectivamente
   cargado (`/proc/self/maps`), versión de OpenBLAS, `openblas_get_config()`,
   `openblas_get_corename()`, número de hilos;
2. despacho en tiempo de ejecución: banderas de CPU detectadas, núcleo seleccionado, y el
   mismo dato con el núcleo **fijado** por variable de entorno;
3. convergencia: número de iteraciones, criterio de parada alcanzado, valor final de la
   verosimilitud y los parámetros, y la traza de las primeras y últimas iteraciones;
4. entorno: kernel, microcódigo, glibc, y si el proceso corre bajo límites distintos.

## 4. Intervención mínima (lo que falta para atribuir)

* **I1 — fijar el despacho**: repetir las celdas discrepantes con el núcleo fijado al mismo
  valor en los tres roles. Si desaparece la diferencia, H1 queda sostenida.
* **I2 — control cruzado**: ejecutar las mismas celdas en un host con el procesador de
  WORKER_B y un kernel distinto, y en otro con el mismo kernel y otro procesador.
* **I3 — control negativo**: repetir con un estimador exacto (`mad_first_difference`) para
  confirmar que la instrumentación no introduce diferencias por sí misma.

Sin al menos I1 no hay atribución al procesador; con I1 sola hay una atribución al
despacho, no al hardware.

## 5. Costo y límites

Celdas discrepantes: 83 de 1.716, todas de un estimador. Presupuesto propuesto: ≤ 1 CPU-hora
por rol, ≤ 2 GiB por proceso, un proceso por host, un hilo BLAS, sin GPU. No se regenera el
banco sintético, no se recalcula ninguna decisión y no se toca ninguna campaña científica.
Este diagnóstico **no es prerrequisito** para publicar o adoptar los hosts de almacenamiento.

## 6. Reemplazo determinista: qué tendría que probar

Un reemplazo de `local_level_kalman` no se acepta por tener un tope de iteraciones. Debe
traer sus propias pruebas numéricas:

* **exactitud**: contra una solución cerrada o una implementación de referencia de alta
  precisión, con error declarado en casos con solución conocida;
* **estabilidad**: comportamiento en los regímenes degenerados que hoy producen el problema
  (verosimilitud casi plana, varianza de señal no positiva), con estado explícito en vez de
  un número cualquiera;
* **reproducibilidad**: idéntico bit a bit en los tres roles, medido con el mismo
  procedimiento de R4 y su subconjunto congelado;
* **equivalencia declarada**: en qué régimen coincide con el estimador actual y en cuál no,
  sin sustituirlo en silencio dentro de resultados ya publicados.

Hasta entonces AT9 sigue **abierta** con su criterio original y la tolerancia sin ampliar.
