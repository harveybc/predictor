# Publicacion y adopcion productiva de los hosts

Fecha: 2026-09-14. Revisor y ejecutor del despliegue: Musashi.
Alcance: infraestructura y transporte sintetico no cientifico. No autoriza
entrenamiento, seleccion de operadores, confirmacion cientifica ni trading.

## Resultado

- [data-lake](https://github.com/harveybc/data-lake) y
  [data-warehouse](https://github.com/harveybc/data-warehouse) son PUBLICOS.
- [PR 44](https://github.com/harveybc/predictor/pull/44) integrado en master:
  `19d1edced54d2c564b479865cd67298fe4825cad`. El proveedor OLAP ya no exige
  instalar desde una rama sin integrar.
- Los hosts sustituyeron los adaptadores anteriores en sus endpoints habituales.
  Se conservaron los IDs de los almacenes, el directorio de datos y el cubo.
- Inventario financiero: 5.275 recursos, igualdad completa antes/despues del
  despliegue aceptado. Ningun contrato financiero fue inventado o habilitado.
- Una instancia separada del mismo host y proveedor sirve SOLO un fixture
  sintetico de dos filas. El registro `governance_smoke` no abre recursos
  financieros. Su incorporacion fue el unico cambio funcional del catalogo.

## Identidades instaladas y pruebas independientes

| Distribucion | Revision instalada |
|---|---|
| data-lake-service | `1ba23edf09af2d4b0f6b60e0a720643a03ed94a4` |
| data-warehouse-service | `6f16565d515426e489c9bf4f8e22f5961de53176` |
| financial-data-store | `d9be1b368a073aa66877208f6a003d8594394bf8` |
| predictor-olap-store | `6d5c9ed079b23b046167ceb8a510f7b2c36d9721` |

Musashi ejecuto de nuevo 33 pruebas del lago, 28 del warehouse y cuatro del
proveedor OLAP. Las suites de los hosts se ejecutaron desde fuera de sus
checkouts contra el entorno instalado. El E2E desechable completo atraveso
los dos hosts instalados y concilio exactamente. Las capturas de escritorio
y movil publicadas por Satoshi son evidencia previa, no una nueva captura
realizada en esta revision.

El [micro-run reproducible](../../tools/verify_deployed_store_flow.py) tiene
cinco pruebas focales adicionales: calculo real, identidad incorrecta,
entrada vacia y valores no finitos. Conserva codigo, manifiesto, configuracion,
datos descargados, mediciones y recibos. No entrena modelos.

## Prueba en produccion

Recorrido ejecutado: cliente -> data-gov -> data-lake/financial-data-store
(fixture separado) -> cliente -> data-gov -> data-warehouse/predictor-olap-store
-> PostgreSQL real.

| Hecho | Resultado |
|---|---|
| Clasificacion | NON_GOVERNING; mecanica de transporte sintetico |
| Filas; media; minimo; maximo | 2; 1.5; 1; 2, calculados de los bytes entregados |
| Confirmacion de entrega | VERIFIED_TRANSFER |
| Reenvio del terminal identico | 200, mismo digest, sin segunda fila |
| Conciliacion | Sin unidades faltantes, sin registros exclusivos de ninguna parte |
| Delta verificado en PostgreSQL | gov_terminal +1; gov_terminal_dataset +1; gov_terminal_artifact +1; gov_terminal_metric +4 |
| Otras tablas | Conteos antes/despues identicos; no se emitio operacion de borrado o actualizacion historica |

Identidades completas:

- Campana: `9f33e27a038ca64a985da1223dba15870c2f08acc5d2b182d004f1fab9eb0291`.
- Terminal: `ea67283a460150119aaaf634f32a50d62c5da11e466c72283fd5d95c5e94b8be`.
- Datos: `5b4cabe43696951ea62d6c67ad4fb0e00f3a3822acc33077eec9fbccd44965d4`.
- Contrato temporal: `139f3adcd7028fbd2bec90381f9ec34170d32528f5c40a6a63bc41f2a02c90a1`.

Los recibos completos y respaldos son privados. El micro-run preserva una
copia del codigo ejecutado y su manifiesto; no usa un commit ficticio.

## Continuidad operativa

Servicios de usuario persistentes y habilitados: `crispdm-data-gov`,
`crispdm-data-lake-financial`, `crispdm-data-warehouse-olap` y
`crispdm-data-lake-synthetic`. Todos activos, con cero reinicios automaticos
al cierre, limite de memoria de 2 GiB por servicio y reinicio ante fallo.
Linger esta habilitado. La configuracion queda preparada para el siguiente
inicio del host; no se reinicio la maquina para demostrarlo.

El cargador OLAP mantuvo su proceso y cero reinicios. No se reiniciaron
PostgreSQL ni Metabase. Se conservaron respaldos de contabilidad,
configuraciones y un pg_dump completo antes de la transicion.
Los archivos de configuracion activa y pendiente son distintos.

## Incidentes de esta revision

1. Mi primer chequeo aceptaba solo VERIFIED_TRANSFER; habia seis entregas
   VERIFIED_CACHE igualmente terminadas. Corregi el chequeo conforme al
   contrato ejecutado antes de detener servicios.
2. La primera transicion levanto los nuevos hosts, pero la comparacion de
   inventarios difirio en los mtime de dos archivos. Se revirtio la transicion.
   El inventario antiguo conservaba metadata previa a la restauracion de esos
   archivos reportada por Satoshi. Una comparacion independiente identifico
   exactamente esos dos campos; con el inventario actualizado, la segunda
   transicion coincidio completamente. No se ignoro el campo ni se cambio
   la regla de comparacion.
3. Durante la preparacion se corrigio la separacion entre archivo activo y
   pendiente de data-gov antes de dejar instalados los servicios persistentes.

## Lo que sigue sin demostrar

La prueba anterior NO demuestra elegibilidad financiera. El proveedor
financiero sigue con resource_contracts vacio. Tampoco demuestra adopcion
por todos los consumidores ni utilidad de una transformacion o estimador SNR.
Los resultados D2 siguen teniendo los alcances y restricciones cientificas
de su revision; desplegar estos hosts no los amplifica.

Continuacion: [ordenes a Satoshi](MUSASHI_TO_SATOSHI_CONTRACTS_AND_CONSUMER_ADOPTION_2026_09_14.md).
