# Ejecutor multihorizonte: verificacion de software

El test se escribio primero: coleccion PRE rechazo la ausencia de i6m_data.
POST: 36 pruebas pasan (contratos y TensorFlow real), CPU, bajo crispdm-run,
3 GiB y 180 s. La suite completa heredada no se ejecuto.

Se probaron ventanas 3,5,7,9,13,49,241: 34921 parametros; mascara de mercado,
perturbacion causal del prefijo, guardado/carga exactos. El factory nuevo
reproduce pesos y forward bit a bit del sellado en 3,5,9. ARIMA dinamico no
condiciona en observaciones futuras. Cola exclusiva, identidad/lease, reinicio,
barrera de etapas, purga por h, precios float64 y signo de skill probados.

Poblacion de VALIDATION 2024 autenticada por horizonte (horas: origenes):
2:6130, 3:6076, 4:6023, 5:5969, 6:5915, 24:4953, 48:3685,
72:3712, 96:3712, 120:3683. Se conserva el mismo vector dentro de cada h.
No TEST, entrenamiento nuevo ni exactitud financiera medidos por estas pruebas.

MH01-MH06 cubiertos por pruebas de contrato/modelo; MH07 cola/restart unitario
y MH08 esquema/outbox necesitan observacion operativa; MH09 pilotos reales y
MH10 cierre total se registran en el directorio privado de campana. No se
declaran completos anticipadamente. No se editaron motores sellados.

## Incidente operativo y sucesor

El primer despacho b0e2d366 encontro dos TypeError por claves duplicadas al
montar diccionarios de terminal/naives. NO eran fallos de ajuste ni de GPU.
Las pruebas originales no recorrian ese escritor real, brecha corregida con
dos regresiones que lo ejecutan sobre un fit sintético declarado. POST: 38
pruebas pasan bajo el mismo cap CPU, 19.03 s.

Los dos ajustes reales dejaron FIT.json y MODEL.keras autenticos (1980
updates, 22 epocas para h2; early stop promedio TRAIN/inner, mejor epoca 17).
El sucesor adopta esos pesos solo si escala, poblacion, parametros, seed,
TRAIN y contrato de ajuste coinciden exactamente con la identidad anterior;
registra ambos digests de codigo. Se conserva el FAILED original, sin editarlo
y SIN otro fit. Tambien se recuperan pronosticos ARIMA completos cuando sus
precios/timestamps y el fichero de parametros retenido pasan la verificacion.
El nuevo diseno y los nuevos recibos tienen otra identidad; no se sustituyen
resultados del experimento anterior en el OLAP. El pico de coste adoptado es
el maximo de los picos retenidos de ajuste y carga/lectura del checkpoint.

## Despliegue verificado

Runtime `48c55ac0`; diseno sucesor
`f57b9b92d4b2a2188ab49b9acaf10df655bde18f10c7b92e8bb26b63258859c1`.
Denominador: **4028** disposiciones (2014 por modelo).
Instantanea de operacion: 24 NEURAL COMPLETED, 14 ARIMA COMPLETED, cuatro
ARIMA FAILED_CONVERGENCE; los 42 terminales publicados y verificados por
lectura exacta. Dos ajustes GPU y un CPU activos; la cola avanza sola.
No resultado anual ni TEST abiertos. Los snapshots live siguientes reemplazan
solo el estado, no esta observacion fechada ni los terminales inmutables.

Coordinador/outbox instalados como servicios persistentes enabled, memoria
observada 112/53 MB; monitor cada minuto con pico observado 19 MB. El CPU
ARIMA tiene cap 1 GiB y cada GPU usa admision y vigilancia. Default runtime
maximo por hijo 7936 MiB; nuevas clases miden bajo 6 GiB en el preferido y
4 GiB en secundario. Produccion: margen 1.25 del pico medido, sin reducirlo.
Solo GPU preferida en su host; secundario <=9 muestras; no brokers tocados.

Suite final: 40 passed, 19.47 s, CPU bajo 3 GiB/180 s. La suite heredada
completa no se uso. El observador excluye score-only adoptado de la ETA de
ajuste, separa clases no medidas y conserva scope PARTIAL_YEAR_NOT_ELIGIBLE.
