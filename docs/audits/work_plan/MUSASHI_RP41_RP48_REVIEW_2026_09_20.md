# Dictamen RP41-RP48: integracion incompleta, no solo un permiso

Musashi, 2026-09-20. Revision de `f83d63566309258b8c0aacddb6aba031c89ab7e0`.
Disposicion: **CHANGES_REQUIRED_BEFORE_ADOPTION_AND_E1**.
No falta una nueva autorizacion del owner. No intente el reinicio: encontre
defectos reproducibles antes de llegar a esa operacion. Esto no contradice la
denegacion que Satoshi reporta en su propio entorno, ni autoriza eludirla.

## Evidencia independiente y alcance

- [Reproductor](../evidence/RP48_REVIEW_2026_09_20/reproduce.py), ejecutado desde
  checkout limpio `95f02b3c11be4aa899df49045467ecfce82fb71d` sobre la implementacion
  revisada; [salidas](../evidence/RP48_REVIEW_2026_09_20/results.json).
- 74 pruebas focales existentes pasan en 25.94 s. No reejecute la suite completa.
- Los 131 archivos del piloto historico conservan sus hashes. No hubo fits nuevos.
- Runner y cliente reales contra data-gov y DuckDB desechables; solo el hijo
  costoso se sustituyo por un fallo determinista para probar cierre de poblacion.
- Controlador, entorno y broker reales offline para el rechazo de margen.
- Consulta de salud: cinco servicios active/running, NRestarts=0, cuatro puertos
  con healthz 200. Esto no prueba avance del loader ni conciliacion del cubo.
- No cambie produccion, reinicie servicios ni escribi en el cubo productivo.

Comando focal, con CPU y un hilo numerico, en `trading-stack`:

```bash
python -m pytest -q tests/test_public_lake_adopt.py tests/test_df_e1_governed_route.py tests/test_df_e1_close.py tests/test_e3_weekly_runtime.py tests/test_e3_weekly_controller.py
python docs/audits/evidence/RP48_REVIEW_2026_09_20/reproduce.py --repo . --out /tmp/rp48-independent-results.json
```

## F1. Alto: el primer hijo se verifica antes de tener su registro padre

En [df_e1_pilot.py](https://github.com/harveybc/predictor/blob/f83d63566309258b8c0aacddb6aba031c89ab7e0/tools/df_e1_pilot.py#L718), `run_isolated` llama
`_closure_verdict` antes de escribir `outcome.json` (lineas 736 y 774 de la base).
En [df_e1_close.py](https://github.com/harveybc/predictor/blob/f83d63566309258b8c0aacddb6aba031c89ab7e0/tools/df_e1_close.py#L198), `verify_unit` exige ese archivo
para encontrar un intento verificable (lineas 198-211).

Con el mismo resultado AE conservado en una copia: con outcome acepta; retirando
solo ese outcome, como ocurre al terminar un hijo nuevo, rechaza
`SCORE_UNVERIFIED: absent: no attempt with a record`. No cambie pesos ni scores.
Las pruebas de ruta que sustituyen `run_isolated` no alcanzan este defecto.
Ademas, `register` conserva dependencia de la ruta original de datos del diseno;
la portabilidad del worker debe probarse sin esa ruta, solo con su entrega.

## F2. Alto: poblacion abierta declarada totalmente gobernada

`run` registra prepare, pilotos y celdas antes del primer hijo; el cierre resume
solo terminales presentes. En la pila HTTP real desechable, fallo del primer
piloto: ocho campanas registradas, una cerrada, **siete abiertas**. Aun asi,
`_governed_summary` devuelve `all_units_governed=true`, sin incompletas. Un reporte
vacio tambien pasa. Prepare no obtiene terminal.

La ruta tampoco persiste `TERMINAL_RECEIPTS.json`: la unidad realmente aceptada
por los servicios queda `HISTORICAL_UNGOVERNED` para el cierre. La solucion debe
conectar productor, recibo persistido y verificador, no agregar otro resumen.

## F3. Alto: declaraciones locales no prueban gobernanza ni cronologia

`_governance` en df_e1_close.py (linea 459) acepta recibos no vacios sin probar su
contenido contra la contabilidad. Reproduje `GOVERNED` sin crear campana alguna,
sin actor/recurso/payload, con estado `NOT_A_REAL_STATUS` y reconciliacion HTTP200
sin sus listas. Tampoco rechaza entrega en 2026 para un registro iniciado en 2020.
La importacion retrospectiva no convierte esa ejecucion en prospectiva.

## F4. Alto: el despliegue propuesto no tiene la aceptacion publicada requerida

La ruta `docs/audits/evidence/d3_k5_20260917/RP46/REHEARSAL.json` del comando
publicado **no existe** en la revision recibida; `_rehearsal_binding` la rechaza.
Esto no demuestra que nunca hubo ensayo: demuestra que el comando entregado
no dispone del recibo con el que afirma ligarse.

RP41 exigia el host externo existente. Sustituirlo por `files_lake` embebido no
repara su incompatibilidad y no cumple esa orden. El binding inventaria el
proveedor externo, aunque el camino propuesto usa otro proveedor. `route_checks`
consulta unidad/estado del warehouse, no el contenido de sus hijos; no prueba
la conciliacion por contenido que el retorno afirma. Usar el reconciliador
existente contra payloads canonicos, sin fabricar un verificador paralelo.

## F5. Alto: RL queda bloqueado despues de un rechazo real del broker

En [e3_weekly_runtime.py](https://github.com/harveybc/predictor/blob/f83d63566309258b8c0aacddb6aba031c89ab7e0/tools/e3_weekly_runtime.py), el estado pendiente
se libera por fills, pero no por los estados terminales sin fill. Sonda real:
precio inicial 100, salto a 1000, vuelta a 100; una orden recibe `Margin`.
El broker queda plano, sin orden abierta y con patrimonio 1, pero el controlador
sigue `ALREADY_LONG` hasta terminar las 30 decisiones. No vuelve a operar cuando
el precio ya permite hacerlo. Cantidad y latencia ahora alcanzan al broker;
eso no basta para declarar correcto el ciclo completo de una orden.

## F6. Medio: la referencia de ruido no es el suelo irreducible

La nueva tarea de innovacion separa informacion independiente y redundante, una
mejora real del diagnostico. Pero en [df_e1_innovation.py](https://github.com/harveybc/predictor/blob/f83d63566309258b8c0aacddb6aba031c89ab7e0/tools/df_e1_innovation.py)
el supuesto suelo copia la observacion ruidosa del rezago. Para el generador
declarado, z=u+e_x, y=a*u+e_y, Var(u)=1 y varianzas de ruido sigma^2, el predictor
de minima MSE con entrada observada es a*z/(1+sigma^2), no a*z.

Con a=1, sigma=0.3: MSE teorica de copia 0.18; Bayes observacional
0.17256880733944951. En la misma realizacion conservada, R2 de copia 0.8393348745;
media condicional 0.8479637481, sin entrenar. Separar formulas poblacionales y
metricas finitas. `snr=3.3333` es una razon de amplitudes; potencia es 11.1111.
Nombrar definicion/unidad, no intercambiarlas.

El entry point diagnostico devuelve metricas sin conservar predicciones/pesos.
No pude verificar desde el paquete un binding de ese diagnostico a campana,
terminal y artefactos. Aportarlo si existe; si no, declarar alcance historico
diagnostico, sin inventar recibos anteriores ni repetir fits solo para maquillarlo.

## Decision y continuidad

Reconozco las reparaciones acotadas de recuperacion del adoptador, contador de
updates, diagnostico de informacion y transmision de cantidad/latencia. No
acepto el cierre global ni que activar el lago sea el unico paso pendiente.
No se invalida toda la historia ni se reinicia el programa doctoral.

Sigue E1 desarrollo: comparar R0/R1/R2 con tarea y receptor adecuados; la
simulacion semanal RL sigue su carril de aceptacion. No hay conclusion nueva
sobre rentabilidad ni superioridad de preentrenar. Ejecutar
[RP49-RP56](../../handoffs/MUSASHI_PROGRAM_RP49_RP56_2026_09_20.md) sin pedir
autorizaciones repetidas, preservando los resultados y limites originales.
