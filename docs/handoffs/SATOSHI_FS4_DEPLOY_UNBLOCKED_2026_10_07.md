# FS4: despliegue desbloqueado (2026-10-07)

Los dos pasos de la solicitud `satoshi/fs4-deploy-20261007@bc519486` estan
resueltos. No pedir autorizacion para volver a hacerlos ni duplicar los
servicios.

1. El coordinador tiene las dos claves de worker en `authorized_keys`, con
   `command=` al `controller_gate.py` y `restrict`. Desde ambos workers el
   controlador responde `status`; una orden SSH ajena devuelve 126.
2. El warehouse activo sirve las cinco relaciones FS4 y conserva 2.653
   terminales gobernados previos. Antes del cambio se tomo una copia byte a
   byte y se abrio con 2.653 terminales en
   `/var/tmp/fs4_predeploy_20261007/cube.duckdb`. Los paquetes en el venv
   productivo son `predictor-olap-store 0.1.6`, `predictor-duckdb-store 0.1.5`
   y `data-warehouse-service 0.1.2`. El smoke de un plan ajeno al real tiene
   escritura, lectura y recibo verificado; su evidencia local esta en
   `~/.local/state/canonical_20261003/fs4/LIVE_SMOKE_20261007.json`.
3. `fs4-closure.timer` esta activo cada dos minutos. `closure/STATUS.json`
   es la consulta de estado; al verificarse esto tenia 7/6495 tareas completas,
   0 fallidas, 7 recibos cotejados y `pending_submit=0`. No leer logs en bucle.
4. En worker_b estan activos los slots GPU 4090, CPU RAW y CPU RANDOM. Los
   topes CPU son 565M y 1496M, derivados de picos de los primeros terminales.
   En worker_a esta activo el timer GPU 5070 Ti, sujeto a admision fresca de
   memoria antes de reclamar. La 5090 continua devolviendo `Unknown Error`;
   no colocarle tareas ni bajar el cap para aparentar admision.
5. El instalador tolera que otra GPU falle despues de listar el UUID sano
   (`predictor@1a14432f`). El runner de worker_a usa el sucesor
   `feature-extractor@e5fa67121ec6009121870c893d58dd34e4640491`, que
   aplica la misma regla y mantiene la prueba TensorFlow en el dispositivo.
   Worker_b conserva el pin previo; la diferencia de codigo es solo la
   comprobacion de GPU parcial. Ninguna celda fallida fue aceptada al cambiarlo.

Continuar con el wrapper semanal independiente. El cierre cientifico de FS4
sigue pendiente: 7/6495 no es seleccion final. Dar ETA solo desde
`closure/STATUS.json` y nombrar su base observada; el primer ETA fluctuara
conforme entren los brazos CPU. Si el timer de un host rechaza por salud, el
otro sigue. No tocar la copia previa ni el plan de humo del warehouse.
