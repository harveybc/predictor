# Orden a Satoshi: adopcion transversal de data-gov Flow v3

**Prioridad:** P0, antes de nuevas campanas que puedan cambiar una decision.

## Insumos revisados

- `data-gov@02f07d7`, rama `musashi/data-gov-failsafe-v3-20260913`
- `financial-data@7f77e3ce6`, rama `musashi/data-gov-lake-v3-20260913`
- `predictor@adbd507`, rama `musashi/data-gov-consumer-v3-20260913`

No reescribir estas ramas. Integrar sus commits por identidad y conservar la
cronologia. Flow v1/v2 queda disponible solo para historia o mecanica; ninguna
salida nueva obtenida por esas rutas puede promover datos, transformaciones,
features, modelos, checkpoints ni decisiones de live.

## P0.1 - Revision e integracion

1. Re-ejecutar las suites declaradas en los tres retornos y el E2E desechable
   `financial lake -> data-gov -> OLAP`.
2. Revisar que el terminal ligue la evidencia completa de cada entrega:
   recurso, rol, rango, bytes, hash de fuente, hash entregado, contrato de
   disponibilidad y estado de confirmacion.
3. Integrar en este orden: lago OLAP, lago financiero, data-gov y consumidores.
   No reiniciar servicios hasta que todos los commits y configuraciones que
   deben convivir esten instalados.
4. Probar PostgreSQL en una base desechable. Queda prohibido usar reset o
   truncado sobre el cubo real.

## P0.2 - Contratos factuales de datos

1. Construir el inventario exacto de recursos que consumiran las siguientes
   campanas. No intentar contratar las 1.965 variables de una vez.
2. Para cada recurso, producir una entrada exacta de `resource_contracts` con
   columna de evento, columna de disponibilidad, zona horaria, unidad temporal
   y frecuencia. El valor debe provenir de evidencia del productor o de una
   derivacion causal revisable; nunca del nombre aparente de una columna.
3. Si solo esta demostrado el momento en que la informacion queda completa,
   pero no su disponibilidad operacional, declararlo como uso historico
   offline. No usar ese recurso para una afirmacion live-equivalent.
4. Validar los contratos contra los bytes fisicos antes del despliegue: las
   columnas deben existir, el hash del recurso debe coincidir y los tiempos de
   disponibilidad no pueden preceder al evento que dicen representar.
5. Un recurso sin contrato permanece cerrado. Esto es un deficit factual, no
   una autorizacion que deba pedirle al owner.

## P0.3 - Despliegue controlado

1. Respaldar solo las bases y configuraciones afectadas, sin copiar datasets.
2. Reiniciar los servicios de lago financiero, lago OLAP y data-gov con el
   codigo integrado. Verificar version, salud y configuracion efectiva.
3. Ejecutar un micro-run gobernado de CPU con directorio de salida fresco y
   datos no reservados. Debe crear campana, entrega confirmada, terminal y
   reconciliacion exacta: `missing_units=[]`, `accounting_only=[]`,
   `lake_only=[]`.
4. Interrumpir deliberadamente el destino terminal en una base desechable,
   demostrar que el resultado queda en el outbox y que el reintento lo carga
   exactamente una vez.
5. Publicar conteos del cubo antes y despues. Solo las tablas `gov_*` pueden
   cambiar durante esta prueba.

## P1 - Adopcion por todos los ejecutores

Aplicar el mismo contrato, sin llamadas remotas dentro de `fit`, `transform`,
`step`, `learn` o un batch, a:

- `preprocessor`, `feature-eng` y `feature-extractor`;
- `predictor` (confirmar la ruta ya implementada);
- `agent-multi` y los runners de evaluacion de trading;
- los publicadores/consumidores DOIN que realmente existan;
- los runners de live cuando reporten evidencia, sin cambiar su bucle de
  ejecucion.

Cada integracion debe registrar antes de abrir datos: identidad limpia de
codigo, configuracion efectiva, unidades, roles de datos y destino terminal.
Cada desenlace debe terminar en uno de `COMPLETED`, `FAILED`, `INCONCLUSIVE`,
`REFUSED` o `QUARANTINED`. La red se usa antes y despues del computo, nunca en
el bucle de aprendizaje.

Los tests unitarios sinteticos y las sondas de mecanica pueden permanecer
fuera de Flow v3 si llevan `NON_GOVERNING`. Si su resultado se usa para escoger
algo, debe repetirse como unidad gobernada.

## P1 - Reja del work plan

1. Anadir una comprobacion ejecutable a los dispatchers: una campana que toma
   decisiones no arranca sin manifest de campana, entregas declaradas y
   destino terminal Flow v3.
2. Un directorio de salida con artefactos previos debe rehusar sin sobrescribir.
3. La caida de data-gov o del cubo despues del computo no borra el resultado:
   lo deja pendiente y no gobernante hasta reconciliar.
4. El cubo debe recibir exitos, fallos, inconclusos, rechazos y cuarentenas.
5. Crear una consulta de cobertura que liste por proyecto: unidades declaradas,
   terminales, datasets, contratos, faltantes y divergencias de reconciliacion.

## Aceptacion

- E2E desechable y micro-run desplegado pasan con reconciliacion exacta.
- Ninguna cache, corte o artefacto cientifico se reemplaza en sitio.
- Una mutacion de bytes, rol, rango, unidad, contrato, codigo o configuracion
  cambia la identidad o rehusa.
- Una entrega no confirmada no concede linaje.
- No existe un camino productivo que escriba metricas de decision directamente
  en PostgreSQL sin pasar por data-gov.
- Los consumidores citados arriba quedan implementados o inventariados con un
  punto de llamada real y un deficit concreto; no contar imports, documentos o
  `__pycache__` como integracion.

## Fronteras

No iniciar entrenamiento GPU, confirmacion, live, venue ni promocion como parte
de esta orden. Esta orden prepara y verifica la base de datos y trazabilidad;
la ciencia se reanuda mediante una orden posterior sobre el despliegue ya
aceptado.

## Retorno requerido

Un solo informe con commits por repositorio, contratos instalados y su
procedencia, pruebas/mutaciones, E2E, conteos del cubo, estado de cada servicio,
matriz de adopcion por ejecutor y cualquier deficit factual restante. Separar
con claridad `IMPLEMENTED`, `DEPLOYED`, `GOVERNED_RUN_PROVEN` y
`NON_GOVERNING_ONLY`.
