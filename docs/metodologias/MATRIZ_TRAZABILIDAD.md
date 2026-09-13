# Plantilla de matriz de trazabilidad DGPD

Una fila representa una obligacion verificable. No se borran filas cerradas;
se marcan como superadas y se enlaza la nueva.

| Requisito | Criterio observable | Caso/historia | Prueba alfa/beta | Decision de arquitectura | Prueba de sistema | Componente | Prueba de integracion | Prueba unitaria | Evidencia | Estado |
|---|---|---|---|---|---|---|---|---|---|---|
| `REQ-001` |  | `UC-001` | `AT-001` | `ADR-001` | `ST-001` | `CMP-001` | `IT-001` | `UT-001` |  | `NOT_STARTED` |

Reglas:

1. Los identificadores no se reutilizan.
2. Una celda vacia necesita `NOT_APPLICABLE: <razon>`, no silencio.
3. Una prueba cita el requisito y el caso que verifica.
4. La evidencia cita commit, datos, configuracion y resultado cuando aplique.
5. Un cambio de requisito crea una revision y dispone las pruebas afectadas.
