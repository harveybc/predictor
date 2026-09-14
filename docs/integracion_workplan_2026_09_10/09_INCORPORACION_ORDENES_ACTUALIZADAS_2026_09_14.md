# Incorporacion al work plan de las ordenes actualizadas (2026-09-14)

Instruccion del owner: incorporacion obligatoria al work plan; los cierres de
experimentos anteriores continuan **con los servicios actuales, sin
interrumpirlos**. Este documento es la parte vinculante; los documentos 06, 07 y
08 conservan su contenido y se leen con este encima.

## 1. Insumos incorporados (leidos del repositorio, no de un resumen)

| Insumo | Identidad | Que aporta |
|---|---|---|
| Acta de reinicio y micro-run productivo | `predictor` `24ca58e`, `docs/handoffs/MUSASHI_FLOW_V3_PRODUCTION_RESTART_COMPLETED_2026_09_14.md` | N3 ejecutado por Musashi; relevo de cinco puntos a Satoshi |
| Consola de operador y publicacion | `data-gov` `ff4503a`, `2afabf6` (`app/operator_config.py`, `docs/operator-console/*`, `docs/INTEGRATION_EXAMPLES.md`) | configuracion **pendiente** editable sin tocar el contexto activo |
| Diseno de hosts reutilizables | `data-gov` `docs/STORE_PACKAGES_DESIGN.md` | arquitectura propuesta `data-lake` / `data-warehouse` + proveedores; no implementada |
| Publicacion de README/contexto | `financial-data cc0f15e6e`, `agent-multi 49764986`, `feature-eng d3d003d`, `preprocessor f8dff3e`, `gym-fx`, `lts`, `prediction_provider`, `doin-plugins`, `synthetic-datagen` | contexto de investigacion y guias de integracion |

## 2. Regla de operacion vigente (sustituye la espera de N3)

1. **No reiniciar** 5055, 5056 ni 5057. El bloqueo operativo esta resuelto y el
   owner no debe ejecutar nada para ese paso.
2. Los cierres de experimentos anteriores (D2 y los que sigan) se ejecutan
   **contra los servicios activos**. Un cambio de configuracion (por ejemplo un
   lago nuevo de evidencia) se prepara como configuracion **pendiente** por la
   consola de operador y se activa en una ventana deliberada: *un archivo
   pendiente no es autorizacion ni despliegue*, la activacion sigue siendo la
   carga de configuracion del servicio.
3. Editar `master` no actualiza un servicio desprendido. Los worktrees de
   runtime en uso no se modifican ni se borran mientras sirvan trafico.
4. Ningun despliegue cambia contratos temporales ni licencias por declaracion:
   los recursos financieros con semantica `UNKNOWN` siguen cerrados.

## 3. Identidad de lo que sirve hoy (comprobado, no declarado)

| Servicio | Worktree de runtime | Commit | Contiene |
|---|---|---|---|
| data-gov 5055 | `.worktrees/musashi-n3-data-gov-20260914T063541Z` | `ff4503a` | N1 (lake/warehouse), N2 (alcance ejecutable), N4 (disposicion de outbox), consola de operador |
| lago financiero 5056 | `.worktrees/musashi-n3-financial-data-20260914T063541Z` | `cc0f15e6e` | contenido **byte a byte identico** a mi rama probada `f00bc6c15` (N2 + las dos correcciones del servidor real) |
| warehouse 5057 | `.worktrees/musashi-n3-predictor-20260914T063541Z` | `a7a86e9` | `olap/lake` con `gov_*` y `write_terminal` |

Deficit de procedencia declarado: `financial-data/master` incorpora ese contenido
por copia, no por ascendencia; mis commits `7f77e3ce6 -> 13e6b1f47 -> f00bc6c15`
no son ancestros de `master`. La correccion minima es una fusion por identidad de
`satoshi/c122-c145-20260912` en `master`, que no cambia ningun byte; la decide el
duenio de la rama de publicacion. Mientras tanto la igualdad se sostiene por
digest, no por historia.

## 4. Estado de la secuencia tras la incorporacion

| Bloque | Estado | Evidencia |
|---|---|---|
| GOV-N1, N2, N4 | `IMPLEMENTED` + `PROVEN_DISPOSABLE` + **desplegados** | acta de reinicio + identidad de los worktrees (seccion 3) |
| GOV-N3 | `PROVEN_PRODUCTION` | acta `MUSASHI_FLOW_V3_PRODUCTION_RESTART_COMPLETED_2026_09_14.md` |
| GOV-N5 feature-eng | `PROVEN_DISPOSABLE` | `p5_feature_eng_throwaway.out`; su despliegue no es parte de esta orden |
| GOV-N5 feature-extractor | bloqueado por deriva de API con `stl_preprocessor` | decide Musashi: fijar commit de predictor o portar |
| D2-R1/R2 | `IMPLEMENTED` | contrato de soporte + reparacion + universo |
| **D2-R3** | **`PROVEN_PRODUCTION`** | campana `d2-support-readjudication-r3`, terminal COMPLETED, conciliacion exacta, sucesoras cargadas de forma aditiva (seccion 5) |
| D2-R4 | preparado; replay pendiente | prerrequisito operativo ya existe; falta ejecutar el subconjunto congelado |
| D2-R6 | ensayado en base desechable, sin aplicar | vistas de cobertura vigente/historica |
| D2-R7 / N7 | diseno y plan | documentos 07 y 08 |

## 5. D2-R3 ejecutado sobre los servicios actuales

Campana `d2-support-readjudication-r3`
(`733736b795e075bffc8c1331a0ce7c910e0ecf721b91123f19990eb1e577ab66`), unidad
`readjudication`, clase GOVERNING, modo de entrada SYNTHETIC: la especificacion
liga generador, diseno `3577c154...`, cinta `230a0a44...`, reserva, las tres
tablas conservadas por digest y el codigo que las produjo y ahora las relee. El
recibo es actual; la produccion original conserva su fecha y procedencia
anteriores a Flow v3. Terminal COMPLETED con 25 metricas (conteos de transicion
y del universo), siete artefactos con hash y conciliacion
`missing_units=[] accounting_only=[] lake_only=[]`.

Resultado cientifico: 3.591 decisiones sucesoras, **138 cambian, ninguna gana un
pase**; siete pierden su pase (los cinco casos revisados por Musashi y dos
controles de identidad en ventanas sin ruido). Las filas sucesoras se cargaron
en el cubo bajo el run id nuevo `d2r3_733736b795e075bffc8c1331` (3.591 filas),
con las historicas intactas (`d2v2_fresh_1fd0e710bc0689b3bf9d7162`, 3.591);
solo cambiaron `df_fact_d2_decision`, `df_dim_run` y `df_fact_load_receipt`. El
loader conservo `ActiveState=active`, `NRestarts=0`.

Esto es una revision registrada: no concede elegibilidad, no abre D3 y no
sustituye la revision externa de Musashi.

## 6. Lo que sigue (sin campanas nuevas)

1. **D2-R4**: ejecutar el subconjunto diagnostico congelado (16 regimenes, 32
   unidades) con los limites de R5, comparando los tres roles solo donde AT9 lo
   exige; AT9 permanece abierto con su tolerancia original.
2. **N5 feature-extractor**: decision de Musashi sobre la deriva de API.
3. **R6**: aplicar las vistas de cobertura por la ruta de adopcion (configuracion
   pendiente + ventana deliberada), no por edicion directa del cubo.
4. **N7**: contrato `doin_governed_result.v1` y fixture antes de tocar
   publicadores reales; live solo por replay offline.
5. **Hosts reutilizables** (`STORE_PACKAGES_DESIGN`): diseno aceptado como
   siguiente etapa de arquitectura; no se migra nada mientras los servicios
   actuales sirvan los cierres en curso.
