# Revision N1-N5 y orden O1-O4

## Resultado de la revision

Retorno revisado: predictor `9880f45`. Se conserva el alcance DESCRIPTIVO del
piloto; ninguna calibracion sobre un generador especifico demuestra validez
universal, ausencia de utilidad o elegibilidad financiera.

Musashi ejecuto en trading-stack:

```bash
python -m pytest -q tests/test_df_utility_harness.py tests/test_df_utility_run_order.py tests/test_d3_matrix_verify.py
```

**68 passed en 29.07 s**. Consulta independiente de solo lectura a la contabilidad:
los tres terminales gen-2 de utilreh-v4 contienen cuatro metricas cada uno. Deltas:
cusum -0.18881876086493116, delta_run_length 0.019637893298198927,
mad_extremes -0.18871307727173697. Coinciden con los archivos leidos en la ronda
anterior. Recuperacion aceptada a ese alcance; no he repetido la conciliacion
completa del warehouse vivo ni los 301 tests declarados por Satoshi.

### Hallazgo alto: la cota no se rederiva

Leido el registro REAL de calibracion MAD del piloto: 1 avance, 538 scored,
upper_bound 0.008786971227311154. En una COPIA EN MEMORIA, cambiar solo
upper_bound a 0 hace pasar calibration_supports de:

```text
(False, 'upper bound 0.0088 exceeds alpha_adjusted 0.0056 (1/538 at 95%)')
```

a `(True, None)`. Ninguna simulacion fue cambiada. Tambien son aceptados por
calibration_record_problems los totales advances=0/rate=0 con el avance todavia
presente en per_sim. Su digest sella los registros, pero el consumidor no cuenta
sus desenlaces ni recalcula la cota. No se afirma que el piloto real haya sido
alterado: sus cifras publicadas son coherentes al alcance inspeccionado.

### Hallazgo alto: reanudacion contradictoria

Fixture desechable con resumen historico ADVANCES y contrast.json ausente:
run_isolated devuelve simultaneamente `outcome=ADVANCES`, `score=null`,
`resumed=true` y `refusal.outcome=SCORE_UNVERIFIED`. No arranque ningun hijo ni
toque un run real para probarlo. El desenlace viejo sobrevive a la revalidacion
fallida; un consumidor que mire outcome puede publicar exito sin evidencia.

Reproductor adjunto: reproduce_d3_n_review_2026_09_17.py. Solo lecturas del run
indicado, copias en memoria y fixtures temporales.

## O1. Recontar calibracion desde simulaciones

Congelar ambos contraejemplos como pruebas rojas antes de editar. Validar el
esquema de cada simulacion, indices/semillas previstos y unicidad; derivar scored,
failed, advances, tasa y Clopper-Pearson desde esos registros. Comparar todos los
resumenes con los derivados, no usar la cota proporcionada para conceder apoyo.
Verificar plan, confianza, alpha, margen y codigo aplicable contra el protocolo;
un campo extra o ausente no se vuelve un valor por defecto autorizante.

La decision por simulacion debe ser coherente con su delta_lower y el margen.
Probar cota 0, cambio de conteos con per_sim intacto, etiqueta incompatible con
delta, indices duplicados, NaN, plan/confianza discordantes y denominador parcial.
Declarar la politica de fallos antes de usarla: contarlos aparte no demuestra
que estimar error solo sobre scored sea valido cuando los fallos son selectivos.

Reverificar las calibraciones de utilreh-v7 y utilpilot-v2 desde sus resultados
conservados, sin repetir 538 simulaciones ni cambiar margen/confianza. Publicar
recibo sucesor del verificador y delta de decisiones; no reemplazar originales.
Si falta evidencia para rederivar, INCONCLUSIVE, nunca reconstruccion inventada.

## O2. Reanudacion con un unico desenlace veraz

Ante fallo de verified_score, devolver SCORE_UNVERIFIED como desenlace principal,
score nulo y razon tipada; conservar el resumen anterior solo como historia.
El reportero y el sobre no deben emitir COMPLETE/ADVANCES ni metricas basadas en
ese resumen. No reentrenar automaticamente para esconder la perdida de evidencia.

Probar ausente, bytes distintos, id/protocolo discordante, archivo no parseable
y el caso positivo identico. Ensayar el entry point completo con un intento
preexistente; no basta probar el helper. Revalidar tambien las salidas preparatorias
en resume, y ligar el job actual al registrado: cambiar configuracion no permite
reutilizar silenciosamente una calibracion o score anterior.

El modo resume-under-new-code registra una diferencia, no demuestra neutralidad.
Documentar y comprobar el diff limitado del arreglo de ids que reanudo el piloto:
misma poblacion, datos, protocolo, operadores y logica de calibracion/score. Si
se modifico algo cientifico, nueva identidad y nueva disposicion del alcance.
No pedir permiso por esta comprobacion; usar los commits ya conservados.

## O3. Cerrar el piloto y explicar sus limites

Tras O1-O2, verificar las doce unidades y comparar metricas completas con el
contenido persistido, sin iniciar otra campana. Conservar los resultados negativos
e inconclusos. Redactar una tabla legible: por variable/unidad, perdida cruda,
transformada, delta e intervalo, filas efectivamente pareadas, cobertura, costo
y alcance del nulo calibrado. DOES_NOT_ADVANCE no significa equivalencia ni
demuestra que un operador no sirve en otros dominios/horizontes/modelos.

Preparar el siguiente diseno DE DESARROLLO por variable: cruda, transformada y
aumento raw+R donde responda una hipotesis, control de capacidad, target/horizonte
congelados y replicacion independiente. Separar diagnostico de flujo, seleccion
en desarrollo, confirmacion publica y revalidacion financiera. No escoger un
umbral nuevo por los resultados vistos ni tratar nueve contrastes correlacionados
como nueve experimentos independientes. Preparar codigo/tests y diseno; no abrir
reservas ni ampliar el piloto en esta orden.

## O4. Operacion, disposicion y cierre continuo

El sobre correctivo vacio 2e68209e... debe conservarse, NO_ADMISIBLE cientificamente,
ligado al sucesor correcto. Publicar el manifiesto pendiente en la proxima ventana
ya necesaria del warehouse; no provocar un reinicio solo por ello. Mientras tanto
probar que ninguna seleccion cientifica lo incluye. Disposicion aprobada en ese
alcance, sin otra pregunta al owner.

Ejecutar O1-O4 sin pausas entre commits. Limites CPU vigentes, gobernanza de toda
nueva evidencia persistida y pruebas de integracion aisladas. Sin GPU, live,
finanzas, reserva, borrado historico o nueva campana D3. No ocupar tres maquinas
si solo hay relectura barata. Mantener el servicio y cargas ajenas intactos.

Actualizar PRE/POST, trazabilidad, work plan y retorno con suites, omisiones y
commits exactos. Salida: `UTILITY_PILOT_REVALIDATED_AND_NEXT_DEVELOPMENT_DESIGN_READY`.
No bloquear recuperacion/verificacion por Metabase, indice o terminos; esos
frentes conservan su propia evidencia y limites. No hay nueva accion del owner.
