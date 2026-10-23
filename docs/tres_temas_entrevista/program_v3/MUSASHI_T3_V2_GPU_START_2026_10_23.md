# T3 v2: mitad de Musashi en ejecucion

Para Satoshi y el propietario. Fecha local2026-10-23.
Respuesta al paquete predictor8d6dbd5b / heuristic-strategy1aaf367.

## Alcance y estado observado

T2 termino8/8 sin fallos de ejecucion y con copias verificadas. La decision
congelada de Satoshi es base (dropout0.1, wd1e-4). T3 es una campana distinta:
su mitad no estaba iniciada y se inicia en esta entrega. No se declara que SAC
o la ronda limpia guiada hayan comenzado.

La unidad `t3_unit_01` esta entrenando en la5090: proceso cientifico observado
en CUDA, no solo servicio activo. Inventario de feedback a ceros, escala
fixed_train, consolidated y verify=true observados en el log nativo. Las otras
20 unidades asignadas quedan encadenadas sin aprobacion de resultados intermedios.
La5090 es la unica GPU pesada de ese host. El otro worker mantiene su mitad
independiente; no se duplica el control ni ninguna unidad par.

## Identidades verificadas antes del arranque

| Objeto | Identidad |
| --- | --- |
| Manifiesto logico | t3v2_8875abf84a63 |
| Codigo cientifico | 1aaf367923fae467988b1ca87d5aa3c74321141f |
| Archivo Git app/tools | e490aeb154a7d6acb80a8dc9d3a130f5cdb9b7eaef804690cfb466fa887bd80a |
| T3_MANIFEST.json | 2ca6648c3c3868d17e5c52719fc0a1c605e7162534b5384c141cd5883df59b32 |
| T3_UNITS_musashi.json | 822816dc692ed724be3a5b63733e580f7018218d8396071b29df80406c668657 |
| PINS.sha256 del paquete | d6fd6415c84317d6bd6f2a9ab4f457241d96e74ae6bd8eefb2ea4d6c80546a1a |
| Config operacional | e6de712c4bd283bb182ae6a9cf061572645d1d3297dc8181d9f2eb5c8dd3060b |
| Store | 32edb4248d659661764ee5abf982a7ee9a0e7659e08a88c03b5a28ce2466ae67 |
| Lista strict116 | 902622d7e47377034ac131d84b08fae9e497105f47aa29d1a1fec1b4099276b1 |

El hash real del store se comprueba en el instalador contra el contrato T2/v4;
el archivo Git se compara con todos los Python app/tools extraidos del paquete.
Los manifiestos coinciden con los blobs publicados; tambien se verifica cada
entrada del PINS del paquete, incluidos los43 archivos de grupos.

## Contrato que no cambia

21 ids impares1..41 del listado recibido, comandos exactos con sustitucion
unicamente del placeholder de raiz. La identidad de comando es distinta de la
identidad de unidad y ambas quedan registradas. No se ejecuta el shell de marcas
DONE; los comandos se pasan como argv al admisor existente.

- Strict116; mismas columnas, codificaciones, calendario y soporte de entrada.
- Ventana13 observaciones, stride2, incluye el ultimo tick observado conforme al
  consumidor sellado; sin nuevas transformaciones.
- Units2/depth2/head16, hasta1000 epocas, tres semillas20261019/20/21.
- Regularizacion base0.1/1e-4, fixed_train, nonstationary diff declarada.
- Consolidated, etiquetas ZigZag, sin feedback, ejecucion1/dia y leverage50.
- Minimo2 ordenes/semana como media del periodo, no obligacion de operar.
- Cap4GiB con admisor/reserva/presion originales, sin reducir un cap rechazado.

Se preserva la advertencia: T2 midio regularizacion a profundidad1 y T3 la
transporta declaradamente a profundidad2. Los perfiles expresan expectativas,
no decisiones de extractor. La decision por grupo y combinacion las realiza
Satoshi con la regla congelada despues de recibir las dos mitades completas.

## Automatizacion y pruebas

Supervisor/timer propios, GPU elegida por UUID de la5090, lock y verificacion de
hijo vivo, reintentos acotados y adopcion de resultados sellados sin segundo fit.
Cada resultado exige semillas, scaler,12 checkpoints y sondas; ademas coteja
ventana, etiquetas, profundidad y layout exacto de grupos/nombres/indices/kernels
contra el archivo de esa unidad. Un soporte diferente no obtiene recibo valido.
Una perdida financiera no se convierte en fallo de ejecucion.

Se corrigio el recuperador antes de desplegar: usaba la lista legacy de trabajos
para rescatar un JSON terminal interrumpido; ahora consulta el plan activo.
Regresion PRE fallida, POST aprobada. Las tres pruebas positivas de T3 fallaban
antes del soporte; POST completo98 pruebas focalizadas en CPU aprobadas.

Cada cinco minutos el colector copia al volumen externo y al segundo worker,
verifica hashes y cierre. Primera copia real comprobada:0 resultados completos,
closed=false, sin fingir una medicion. Timers se desactivan al cierre del lote.
Es custodia fisica; no se presenta como publicacion OLAP ni RELEASE.

## Handoff

Raiz logica recibida: `satoshi-t3-for-musashi-20261023`, con ops/PLAN.json,
ops/STATUS.json y recibos ops/t3_unit_XX.RESULT.json. Replica logica:
`expert-guided-musashi-t3-replica-20261023`; mirror externo:
`shared-selection-e3m-t3-20261023`.

Satoshi: no duplicar estas21 unidades, consumir sus recibos/config y resultados
autenticados, unir con sus22 unidades y ejecutar la decision y combinacion
congeladas. No modificar inputs/codigo del paquete activo. Conservar hardware,
semillas, soporte y parametros por unidad en la tabla de consumo. La estimacion
previa de5-6h sigue siendo extrapolacion hasta medir el primer brazo T3 completo.

No se escribe RELEASE global ni se cambia DOIN, TEST o ningun servicio de trading.
