# Retorno consolidado del despacho `aa44bd14`

**Fecha:** 2026-09-30
**De:** Satoshi III (Mujuro Utsutsu), líder técnico sucesor
**Para:** Musashi (auditor) y el dueño
**Alcance:** los tres carriles del despacho `aa44bd14`, integrados. **`NO_NEW_MEASUREMENT` de exactitud**
en ninguno de los tres; las pruebas de mecanismo y las mediciones de recurso van separadas, como exige la
orden.

## 1. Agentes efectivamente despachados, con acuse

| carril | rama / tip | acuse de despacho | estado |
|---|---|---|---|
| dueño único de integración | `satoshi/qrm-integration-20260929` — `c4a918ca` (commit integrado) + `4d548a82` (evidencia) | worktree, tips fuente, primer comando en rojo, prueba de aceptación, presupuesto y bloqueo entregados en su primera respuesta | **cerrado** |
| clasificación | `satoshi/banking77-native-path-20260929` — `c89a8ff4` | ídem | **cerrado**, sin score, con una dependencia exacta |
| diseño de experimento de negocio | `heuristic-strategy` `satoshi/strategy-experiment-design-20260929` — `4bb763d` | ídem | **cerrado**, nada ejecutado |

Ningún agente se anunció antes de su acuse. Ninguno queda vivo; 0 reservas en ambos anfitriones al cierre.

## 2. Resultados, con lo que verifiqué yo

**Integración.** Un solo commit integrado con ambas ramas de reparación como ancestros verificados y sin
conflicto. **Una petición de GPU ya no puede caer a CPU sin que se note**: tres hechos independientes
(UUID del driver, registro de TensorFlow, colocación de una operación real) y un déficit **lanza excepción**
en vez de devolver un valor. Medido en vivo en cuatro estados. **La ruta gobernada completa de extremo a
extremo con readback del almacén, dos veces**, 8/8 saltos, con el sha256 del hijo sobre los bytes que él
mismo abrió. 212 tests, 0 fallos. `import torch` ausente del camino de memoria de TensorFlow (verificado).

**Clasificación.** La frase demasiado amplia quedó **al lado** de la evidencia, con el registro original
byte a byte idéntico (verificado). Valor publicado 0.914578 contra un ingenuo de **exactamente 1/77** sobre
las mismas 3 080 filas gobernadas (verificado: 40 filas por clase). Tres identidades del artefacto
publicado **corroboran que su población es la nuestra**. Sin score: bloqueo nombrado
`BANKING77_NATIVE_PATH_BLOCKED_ON_MODEL_AND_ENVIRONMENT_ACQUISITION`.

**Diseño de negocio.** Recuperado del código real: la entrada lee **sólo la familia larga**; objetivo y
stop son **señales al cierre, no órdenes protectoras**; el variante de salida por defecto **se contradice
tres veces** en el fichero; el swap **se resta del beneficio reportado y nunca se cobra al bróker**; y **no
existe ingenuo por horizonte en `app/`** (los tres verificados). Dos diseños previos a ejecución, cuatro de
cinco requisitos satisfechos y el quinto parcial y dicho como parcial. Trabajo previo de 241 celdas
encontrado y **no repetido**.

## 3. La petición de cómputo, ligada al runner integrado `c4a918ca`

**Reconciliada: 4 800 s de CPU y 3 600 s de pared** — no 4 800 / 4 800. Los 1 200 s de pared que pedí de
más eran **holgura sin mecanismo detrás**: la única parada de pared del camino integrado es por hijo.

| ítem | valor |
|---|---|
| celdas | 2 de calibración (una por arquitectura faltante), **secuenciales** |
| tope de host por celda | 12 GiB, **impuesto** por `MemoryMax` — **cabe en la holgura viva del techo existente de 14 GiB** (12.54 GiB libres, verificado) |
| CPU por celda | 2 400 s por `RLIMIT_CPU` |
| pared por celda | 1 800 s por el `-t` del lanzador |
| envolvente de dispositivo | 12 GiB, **sólo observada**, medida con el asignador de TensorFlow |
| lo que compra | una propuesta costeada del sucesor; **ningún ajuste, ninguna exactitud** |

**La petición de 18 GiB de techo queda RETIRADA por mí.** Era innecesaria.

**Pero la asignación por sí sola no la desbloquea.** El piloto necesita un anfitrión con **un dispositivo
de ≥12 GiB Y un TensorFlow que lo registre**, y hoy ninguno tiene ambos: la tarjeta del coordinador es de
8.00 GiB (verificado); el TensorFlow del obrero **registra cero dispositivos de forma reproducible**; la
5090 sigue inadmisible. Y **el único estado que mostró un piloto GPU funcionando no es reproducible hoy**
(`STATE_C_NOT_REPRODUCIBLE`).

Así que la decisión que sí es del dueño tiene **dos mitades**, y la segunda es la que cuenta:
1. la asignación de arriba;
2. **o** reparar el entorno GPU de TensorFlow del obrero (**una instalación**), **o** un diseño sucesor con
   envolvente de dispositivo que quepa en 8 GiB (**un cambio científico**, que Musashi debe revisar).

## 4. Correcciones a mi propio registro en este despacho

- El dueño del residual del slice es **INDETERMINADO**: la VM de MT5 está **fuera** del slice por membresía
  de cgroup, así que sus 5.78 GB **no son cargo del slice**. Mi atribución anterior era por RSS del anfitrión
  y no valía.
- "No cabe a ningún presupuesto" → **no está soportado por el envoltorio distribuido**. Una jerarquía es
  otro método y debe incluir sus errores de enrutado sobre toda la población.
- La regla del muestreo **no se refutó**; lo mal planteado era el objetivo de comparación.
- El estado reutilizable evita descargar dos veces **sólo en la ruta de Laya**; la ruta nativa **no se ha
  descargado ni una vez** y necesita un **tercer** entorno.

## 5. Decisiones pendientes, todas del dueño, ninguna técnica

1. La asignación de §3 **y** la mitad 2 (instalación o sucesor).
2. Para clasificación: **1 369 721 378 B** de artefactos del modelo, un tercer entorno (16 228 123 B en
   nivel 1) y la decisión explícita de **`trust_remote_code`** — ejecutar código del repositorio del
   modelo. Licencia CC BY-NC 4.0, sólo investigación.
3. Para la estrategia, no bloqueantes: horizontes por **desplazamiento de filas o por horas transcurridas**,
   y ejecución con **órdenes protectoras o señales al cierre**.

Nada de esto elige imports, esquemas ni convenciones de contabilidad: eso quedó decidido dentro de los
carriles, como la orden exigía.

— Satoshi
