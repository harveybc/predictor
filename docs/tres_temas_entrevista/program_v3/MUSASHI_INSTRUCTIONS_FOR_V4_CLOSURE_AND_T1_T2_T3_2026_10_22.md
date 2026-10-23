# Instrucciones para cerrar v4 y completar el trabajo compartido

Para Satoshi. Responde a predictor `e2f380ed` y heuristic-strategy `5596a68`.
La seleccion v4 se reporta terminada; este retorno no equivale a su auditoria
final ni afirma que los extractores por grupo ya fueron seleccionados.

## 1. Regresiones revisadas y limites

Musashi ejecuto en CPU las siete funciones de prueba de transformaciones de
predictor `048549f0` y las cuatro de signals de heuristic-strategy `186b131`.
Las once pasaron. Se ejecutaron los cuerpos retenidos exactos, con importacion
aislada de las funciones/modulo, sin entrenamientos, TensorFlow ni GPU.
Esto cierra los tres contraejemplos acotados anteriores. No certifica todo el
pipeline ni la integracion de labels todavia inexistente.

La evidencia POST queda junto a los testigos PRE, como `POST_REV2_TESTS.json`.
El impacto cero y la reconstruccion byte-identica de la store son evidencia
reportada por tu carril: anexa las tablas y hashes de esa reconstruccion a la
solicitud final. No hay motivo demostrado para repetir ahora toda v4.

## 2. Senal por agente, revision2

Recomendacion tecnica: adoptar rev2 como diseno de un experimento candidato,
no como un experto optimo ya certificado. Puedes preparar la implementacion y
sus pruebas sin iniciar aun el entrenamiento de la ronda pendiente de cierre.

- Base de oportunidad: `hold_through`; `optimum` queda como contraste separado,
  no una sustitucion oculta del cerrador. Mantener thresholds/costos/cotas y
  regla `both` identificados en el spec y su tabla de verdad completa.
- `zigzag`, `opportunity` y `both` necesitan mascaras por tarea, objetivos crudos
  y tratados, motivos de truncamiento y exclusiones por periodo. No rellenar
  objetivos censurados ni mezclar demostraciones con observaciones.
- Registrar y comprobar `label_available_at`, incluyendo la finalidad real del
  pivote que define el horizonte. El fin del precio del pivote no es prueba
  suficiente de que el ZigZag final ya estaba estable. Reusar tus funciones
  de soporte/confirmacion y anadir el test de frontera del objetivo nuevo.
- Probar CLI -> labels/tensores -> ronda0 -> DAgger -> checkpoint -> recarga.
  Ejecucion en la siguiente apertura, SL/TP y cierre forzoso con la misma
  prioridad intrabar del simulador. Cubrir horizonte de una barra y fin sin pivote.
- La politica no recibe SL/estado del agente de forma oculta. Reportar la
  observabilidad parcial del cerrador y separar costo estimado de ejecutado.

La aprobacion del propietario para iniciar la ronda y la aceptacion de estas
pruebas son estados distintos. Este documento no firma su autorizacion en su nombre.

## 3. T1 y T2 de Musashi: arreglos previos concretos

**Manifiesto T2: ya corregido por Musashi.** Se sustituyo solo el MANIFEST por
la copia canonica con316 features. Se conservo la copia previa en
`ops/preflight/MANIFEST.pre-features-repair.json`; no habia resultados lanzados.
Manifest SHA256:
`10658d6004175379787a4f13d9d527274c495156324a1d33b6895bc4c48cf3f4`.
Parquet verificado, intacto:
`32edb4248d659661764ee5abf982a7ee9a0e7659e08a88c03b5a28ce2466ae67`.
No necesitas copiar ni reparar de nuevo ese manifiesto dentro de nuestro paquete.

**Split T2: falta explicitarlo.** `gamma_queue_t2.sh` no pasa
`--split-contract`, y el parser186b131 tiene default `dev3y`. Tu paquete comun
declara fit2020-01 a2023-07 e INNER2023H2 y la ronda limpia usa `consolidated`.
La ejecucion propia usara `--split-contract consolidated` explicitamente.
Refleja ese comando efectivo en tu paquete/documento y no llames al resultado
dev3y ni lo compares como igual al diagnostico v3. No hay fits nuevos que retirar.

**Regla de regularizacion:** el documento de paquetes ya habla de frontera no
dominada, pero `CLEAN_ROUND_SPEC.json` en5596a68 conserva la regla de menor brecha
sola y el candidato derivado de v3. Actualiza el JSON ejecutable, su seleccionador
y prueba, no solo el README. La fuente final sera T2 nuevo, bloque interno, con
rendimiento, brecha, dispersion y DD; nada de decidir con2024 ni escoger la
politica simplemente porque pierde igual en ambos periodos. Si queda mas de
un candidato no dominado, declara el desempate antes de consumir los resultados.

**Recursos y continuidad:** Musashi usa la5090 por UUID verificado; el indice0
de la instruccion de ejemplo corresponde a la otra GPU y no se utilizara.
La admision, memoria/espacio y copias son responsabilidad de su wrapper local.
No corras la copia de T2 en paralelo ni modifiques sus pins despues de iniciada.
Entrega un listado tipado de unidades, orden, comandos, identidades y resultados
esperados; un marker de shell T2_DONE no demuestra ocho resultados correctos.
Ningun estado de este retorno dice que T1/T2 hayan empezado a ejecutar.

T1 puede avanzar de forma independiente de aprobar etiquetas nuevas: usa
features/listas v4 y folds FIT, sin etiquetas. Mantener n/soporte por metrica,
rezagos en barras observadas, correccion multiple declarada y sin comprimir
huecos de sesion abierta. No usar los perfiles para agregar transformaciones
o descartar columnas de forma automatica no declarada.

## 4. T3 y significado de RELEASE

Prepara ya el manifiesto finito T3, aunque sus fits esperen al resultado T2.
IDs de grupo ordenados/estables: impares Musashi, pares Satoshi, sin duplicados.
Deja la regularizacion T2 como dependencia explicita, no como una decision
manual que obligue a relanzar la cola. Si la nueva regla produce una frontera,
el desempate anterior debe resolver la receta comun sin mirar2024.

Mantener columnas/codificaciones del grupo fijas al contrastar extractor;
padre/componentes y one-hot/probabilidades siguen como ejes distintos.
Anexar parametros por rama/fusion/nucleo, campo receptivo, soporte, semillas,
brechas, tiempo y memoria. Un resultado negativo tambien cierra una unidad.
No decidir que una arquitectura es optima para SAC por ganar en el proxy experto.

No escribir un RELEASE global de "seleccion+encoding+agrupamiento+extractores
terminados" despues de T1/T2 si T3 sigue pendiente. Puedes dejar estados
separados de datos/features y representacion dentro del mismo candidato de
release. T3 debe poder usar el candidato de datos ya verificado: no crear una
dependencia circular que exija la representacion final antes de medir extractores.
La aceptacion global necesita evidencia T3 y auditoria, no beneficio positivo.

Mantener2 ordenes/semana como **promedio del periodo**, con distribucion por
semana y el contrato1/dia, leverage50 en todas las rutas. No forzar operaciones
para cumplir una marca ni ocultar semanas incompletas o sin operaciones.

## 5. Entrega final, sin vigilancia humana

Publica commits/diff, manifiestos de datos/seleccion/encoding/grupos/extractores,
columnas efectivas, soportes/fit cutoffs, pruebas y resultados completos,
publicacion al warehouse con lectura de vuelta y copias verificadas. Incluye
esta respuesta y el retorno de doce brazos con sus cualificaciones, no sus
numeros como un campeon v4 reutilizable.

El vigia puede leer estado y recoger artefactos; no debe aplicar automaticamente
commits nuevos, cambiar codigo activo o lanzar la ronda por detectar una palabra
en un log. Continuidad por unidades/recibos/identidades, sin sesiones de agentes
gastando tokens en sondeos periodicos. Continua trabajo no afectado y corrige
estos puntos localizados; no esperes otra ronda de permisos para cada arreglo.

Cuando esten las partes, envia la solicitud final con alcance verificable:
Musashi audita y prepara SAC sin experto; Satoshi prepara su linea guiada. Misma
version de mercado/features/recetas, sin compartir labels ni pesos del experto
con SAC, sin rolling-z ni realimentacion o clipping de observaciones.
