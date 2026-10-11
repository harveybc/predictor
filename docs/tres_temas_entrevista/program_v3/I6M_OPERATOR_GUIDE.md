# I6-E-M: operacion sin intervencion por celda

## 1. Desplegar una vez

Desde un checkout limpio, comprometido y publicado. Usar los aliases SSH y
el token locales, nunca poner sus valores en Git. No hay acceso SSH inverso
desde workers al coordinador ni nuevas authorized_keys.

```bash
ROOT="$HOME/.local/state/canonical_20261003/i6e/multihorizon_2024_v1"
python -m tools.i6m_deploy \
  --root "$ROOT" \
  --parent "$HOME/.local/state/canonical_20261003/i6e/statistical72_v1" \
  --preferred-host <preferred-ssh-alias> \
  --secondary-host <secondary-ssh-alias> \
  --token-file <local-warehouse-secret-env>
```

El servicio maestro hace work-stealing entre dos slots GPU y un slot CPU
ARIMA limitado a 1 GiB. Cada modelo completa downsampling antes de sus controles
horarios. La 5090 tiene prioridad; solo ella se usa en su anfitrion. El secundario
recibe ventanas <=9; las grandes van al preferido con piloto de su clase.
El primer fit TRAIN de cada longitud/host es el fit real de semana cero:
guardado, autenticado y reutilizado, no una repeticion adicional.

Cap inicial de medicion: 4 GiB. Cap siguiente: ceil(1.25*pico cgroup medido).
No recortar el margen si no cabe en 7936 MiB; retener fallo, no bajar ni refitar.
El lanzador verifica memoria viva y vigila presion durante cada hijo. Los
supervisores son pequenos y separados. No se paran servicios de trading.

## 2. Consultar cualquier nivel

```bash
python -m tools.i6m_coordinator status --root "$ROOT"
python -m json.tool "$ROOT/STATUS.json"
python -m json.tool "$ROOT/OLAP_STATUS.json"
python -m json.tool "$ROOT/cells/<model>_h<h>_d<delta>_w<week>/STATUS.json"
journalctl --user -u predictor-i6m-coordinator.service -n 20 --no-pager
```

QUEUE.db conserva estado, slot y lease exclusivo por celda. STATUS muestra
ambas etapas por modelo, disposiciones, trabajos activos y latido. La ETA
inicial queda nula hasta observar un terminal; una proyeccion de coste igual
NO es la ETA medida de modelos largos todavia no perfilados.

## 3. Reanudar y cerrar

Los supervisores systemd reinician ante fallo. Un supervisor que encuentra un
hijo superviviente espera su terminal; no crea otro ajuste. Una interrupcion
sin checkpoint validado produce fallo retenido, no entrenamiento nuevo.
Un problema de entrega al warehouse deja outbox pendiente, nunca otro fit.

Al terminar los 2014 terminales por modelo, el maestro genera RESULTS.json,
RETURN.md y PROGRESS.png. El cierre espera lectura exacta del OLAP para
declarar entrega completa; fallos cientificos quedan separados y ningun brazo
incompleto recibe MAE anual ni promocion. El TEST no se abre automaticamente.

Los binarios (NPZ, Keras, SQLite) quedan en disco privado. Git conserva codigo,
contratos y evidencia compacta. OLAP conserva MAE, MSE, naive por h, skill,
parametros, tiempo, pico de memoria, semilla, digests y disposiciones.

## 4. Identidad y limites

38 brazos unicos, 53 segmentos de 2024, una semilla 6102026. Ventana fisica
2*h, emisión horaria, sin suavizado ni imputacion de cotizaciones. Escala de
MAE: precio USD/EUR; `skill=(MAE_h-naive_h)/naive_h`, negativo mejora.
La procedencia retenida permanece LOCAL_RETAINED_UNVERIFIED.

No comparar MAE de log-return con naive de precio. No promediar medias de
semanas: ponderar por las filas retenidas. No usar un naive de otro horizonte.
