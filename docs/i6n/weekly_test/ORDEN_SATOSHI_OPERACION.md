# Orden Acotada: Robustez Operativa

## Alcance Y Aislamiento

Base exacta: 9beb31b6, que contiene la auditoria integrada y esta revision, sin
cambios en el codigo cientifico desplegado. Crear una rama y
worktree NUEVOS, propios, por ejemplo `retsu/prequential-ops-candidate-20261017`.
No editar el worktree del coordinador ni del worker activos. Leer primero
PROCEDIMIENTO_CANONICO.md y MUSASHI_REVIEW_5c04fa42.md.

Autorizado: codigo candidato, pruebas sinteticas CPU y documentacion. Una sola
lectura inicial de metadatos existentes si es necesaria. Sin fits, GPU, nuevos
datos TEST, despliegues, servicios, reinicios, cambios DOIN, re-seleccion de
features ni monitoreo periodico. No esperar al cierre de la campana para terminar
el encargo. No publicar secretos, identidades de maquinas ni rutas privadas.

## A. Recuperacion Transitoria, Sin Alterar Ciencia

1. Disenar y probar ANTES de implementar una clasificacion explicita: transitorio
   recuperable por proceso, bloqueo de recursos y fallo permanente de integridad
   o contrato. Un error de version no es lo mismo que una GPU temporalmente
   indisponible, aunque hoy compartan un mensaje. No reintentar excepciones
   desconocidas por defecto.
2. Para los transitorios admitidos: maximo dos reintentos tras el intento inicial,
   cooldowns fijos de 30 y 60 segundos, intentos conservados en ledger durable.
   Agotados, estado terminal explicito con causa y minima accion requerida.
3. Nunca borrar un FAILED historico para reanudar. Conservar cada intento y su
   evidencia. No reescribir un sello PREDICTED/PREDICTION ni recalcular una
   prediccion ya sellada. Retomar desde el MISMO donante, semilla, datos, escalas
   y configuracion. Una actualizacion completada se adopta, no se vuelve a ajustar.
4. No tocar max_epochs, paciencia, min_delta, loss, arquitectura, columnas,
   ventanas, mascaras o presupuestos de memoria. Un OOM no autoriza disminuir
   el cap ni simplificar el modelo. Ningun reset de GPU o reinicio automatico.
5. Durante cooldown se puede ejecutar la otra familia si su dependencia esta
   resuelta. Nunca avanzar una familia sin su checkpoint anterior valido.
6. En status distinguir RUNNING, WAIT_RESOURCES, RETRY_PENDING y bloqueo terminal;
   incluir intento, causa, proximo intento y dependencia. No ocultar un bloqueo
   como RUNNING ni prometer ETA finito si no puede estimarse.

Pruebas: fallo transitorio despues del sello y recuperacion; prediccion byte a
byte intacta; donante/seed/config intactos; cooldown sin espera real inyectando
reloj; agotamiento; no reintento de digest/causalidad/version/errores desconocidos;
adopcion sin segundo fit; relevo a la familia independiente; reanudacion del
ledger tras reinicio de proceso. Todo con backends falsos, sin dormir/pollear.

## B. Cierre Con Cobertura Explicita

Anadir al JSON y Markdown de cierre, sin cambiar las metricas:

- Semanas declaradas por familia; semanas puntuadas, sin pares, fallidas y
  pendientes. Invariante exacta: suma de disposiciones = semanas declaradas.
- Lista de IDs y limites de semanas sin pares; incluir frontera de ano si
  corresponde. No confundir semana sin pares con MAE cero ni con un fit exitoso.
- Por horizonte: semanas con pares, semanas sin soporte y N de pares realmente
  puntuados. Mantener agregacion por sumas/conteos y el naive propio pareado.
- Glosario: SIN_PARES es el nombre descriptivo de SKIPPED_NO_EVAL_ROWS;
  NO_VALID_TARGET_IN_PERIOD es la disposicion por cabeza. No renombrar recibos.

Pruebas con ambas fronteras anuales, una cabeza sin soporte en una semana util,
un fallo y un cierre incompleto. El informe no debe declarar cobertura 100% de
predicciones sobre semanas omitidas.

## C. Completar Las Sondas Sin Afirmaciones Excesivas

- PS06: usando mocks en los limites de runtime/carga/fit, ejercitar el guardia
  real de pesos iniciales. Un hash inicial incorrecto debe rechazarse; el correcto
  debe venir del checkpoint. No basta que el backend falso copie el valor esperado.
- PS10: ejercitar tick/outbox/cierre con IO externo sustituido, filesystem temporal
  y warehouse falso. Probar readback pendiente y reintento idempotente, espejo
  fallido que impide CLOSE, y cierre posterior que detiene SOLO timers propios.
  No usar ssh/systemctl/warehouse reales en estas pruebas.
- Mejorar el predictor falso de PS03/PS04 para que dependa del estado/pesos del
  donante: cambiar outcomes de semana 1 puede cambiar semana 2, pero no semana 1;
  cambiar outcomes de semana 2 no puede cambiar el checkpoint ni prediccion de
  semana 1. Distinguir igualdad de NPZ de igualdad JSON sin reloj.
- Mantener los archivos de modelo/trainer/datos cientificos byte-identicos a la
  base. Si una sonda es de limite simulado, llamarla asi, nunca GPU/E2E real.

## Entrega Unica

Ejecutar pruebas focalizadas bajo `crispdm-run`, maximo 2 GiB y 120 s por comando,
CUDA_VISIBLE_DEVICES vacio y sin cache de pytest. Si requiere mas, explicar el
objeto y costo, no lanzar una suite abierta ni otro entrenamiento.

Un RETURN con commit exacto, tabla requisito -> prueba -> resultado, PRE/POST,
limites y comandos reproducibles. Publicar la rama candidata. NO desplegar,
fusionar sobre el worktree activo ni observar logs despues. Musashi revisa la
candidata y decide cuando aplicarla sin cambiar una campana cientifica en curso.
