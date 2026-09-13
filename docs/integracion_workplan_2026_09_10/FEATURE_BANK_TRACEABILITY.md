# Trazabilidad inicial del banco de caracteristicas

**Estado:** requisitos en revision; las pruebas aun no autorizan implementacion.

| Requisito | Criterio observable | Caso | Aceptacion | Sistema | Componente | Integracion | Unidad | Estado |
|---|---|---|---|---|---|---|---|---|
| `FBR-001` | Toda feature tiene padres, formula, parametros y digest | `UC-F01` | `AT-F01` | `ST-F01` | contrato de operador | `IT-F01` | `UT-F01` | DRAFT |
| `FBR-002` | Ninguna salida en t cambia al perturbar datos posteriores | `UC-F02` | `AT-F02` | `ST-F02` | ejecutor causal | `IT-F02` | `UT-F02` | DRAFT |
| `FBR-003` | Estado ajustado usa exclusivamente training | `UC-F03` | `AT-F03` | `ST-F03` | fit/transform | `IT-F03` | `UT-F03` | DRAFT |
| `FBR-004` | Batch, incremental y restart concuerdan | `UC-F04` | `AT-F04` | `ST-F04` | estado durable | `IT-F04` | `UT-F04` | DRAFT |
| `FBR-005` | Tiempo de disponibilidad gobierna cada join | `UC-F05` | `AT-F05` | `ST-F05` | alineador | `IT-F05` | `UT-F05` | DRAFT |
| `FBR-006` | Targets futuros no son dependencias de features | `UC-F06` | `AT-F06` | `ST-F06` | registro de roles | `IT-F06` | `UT-F06` | DRAFT |
| `FBR-007` | Ausencias y warm-up conservan mascara y semantica | `UC-F07` | `AT-F07` | `ST-F07` | politica missing | `IT-F07` | `UT-F07` | DRAFT |
| `FBR-008` | Cada feature publica costo y retardo | `UC-F08` | `AT-F08` | `ST-F08` | medidor | `IT-F08` | `UT-F08` | DRAFT |
| `FBR-009` | Versiones con fuga son detectadas por la bateria | `UC-F09` | `AT-F09` | `ST-F09` | controles adversariales | `IT-F09` | `UT-F09` | DRAFT |
| `FBR-010` | Elegibilidad exige utilidad incremental fuera de muestra | `UC-F10` | `AT-F10` | `ST-F10` | adjudicador | `IT-F10` | `UT-F10` | DRAFT |
| `FBR-011` | Raw y controles de igual anchura/costo se conservan | `UC-F11` | `AT-F11` | `ST-F11` | arnes comparativo | `IT-F11` | `UT-F11` | DRAFT |
| `FBR-012` | Exitos, fallos, rechazos e inconclusos llegan al OLAP | `UC-F12` | `AT-F12` | `ST-F12` | outbox/loader | `IT-F12` | `UT-F12` | DRAFT |
| `FBR-013` | Una feature elegible proviene de campana y entregas gobernadas | `UC-F13` | `AT-F13` | `ST-F13` | manifest/receipts | `IT-F13` | `UT-F13` | DRAFT |
| `FBR-014` | Un resultado `NON_GOVERNING` no abre F5 ni I5 | `UC-F14` | `AT-F14` | `ST-F14` | adjudicador | `IT-F14` | `UT-F14` | DRAFT |
| `FBR-015` | El terminal se recupera tras indisponibilidad del servicio o cubo | `UC-F15` | `AT-F15` | `ST-F15` | outbox durable | `IT-F15` | `UT-F15` | DRAFT |
