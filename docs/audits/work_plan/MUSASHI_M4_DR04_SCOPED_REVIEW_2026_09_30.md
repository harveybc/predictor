# M4: revision externa acotada de DR04

Fecha: 2026-09-30. Codigo inspeccionado: agent-multi `4b009c35` en el worktree
`am-dr04-m4-20260926`. No es aprobacion de ejecucion confirmatoria.

## Hechos comprobados

Reejecute `m4_dr04_verifiability_post_2026_09_26.py` bajo
`crispdm-run -m 2G -t 120`, CPU y sin CUDA. Salida 0. El caso positivo paso
24 unidades DEVELOPMENT por el entrypoint y verifico 24 registros mediante
logs y reconstruccion. Los contraejemplos de F1-F5 ya rehusan por nombre:
resumen fabricado, endpoints alterados coherentemente, log ausente/alterado,
censo ajeno, semilla repetida, poblacion parcial y ledger de otro codigo.
Sin los dos registros externos, el entrypoint confirmatorio rehusa antes de
crear un directorio. El generador con `allow_confirmation=True` SI construye
un arreglo en memoria para la prueba de disyuncion; no hay evidencia de ajuste
o puntuacion confirmatoria. La frase historica de cero arreglos queda retirada.

El codigo enlaza un `implementation_sha256` de seis modulos con el ledger y
su verificador. `REVIEWED_TIP` sigue nombrando el tip de calibracion
`5e7a8fd4`, mientras el codigo que correria tiene otro digest. Estos son dos
identificadores con funciones distintas y ambos deben aparecer sin ambiguedad
en el registro de revision. El digest observado del candidato es
`68e4c0ed22bb910cad1ba5722f5d5647821064dd1db3ff0f1b1474e4923fa83c`;
no debe copiarse sin volver a calcularlo en el checkout de la aprobacion.

## Decision de diseno pendiente

El contraste 15 (`checkpoint_effect::primary_pair`) no tiene estimando de
poblacion suficientemente cerrado por el diseno v5: nombra el contraste, pero
no fija como combinar familia, ruido y anchura. DR04 usa una observacion por
identidad de generador completo y luego `ttest_1samp` sobre todas las celdas
elegibles. Esa es una eleccion nueva. El diseno tambien condiciona el pooling
de efectos de familia a una regla de heterogeneidad. No acepto que la prueba
positiva de una poblacion de desarrollo sustituya una decision sobre la
poblacion confirmatoria. Escribir antes de ver el resultado la unidad de
inferencia, pesos por celda, regla de pooling y que pasa cuando la
heterogeneidad no permite pooling; luego verificar que el codigo implementa
exactamente esa regla. No escoger la variante por el p-value.

La re-verificacion completa cuesta trabajo adicional. Antes de una campana
confirmatoria se requiere una asignacion que nombre ejecucion y replay por
separado con limites de pared/CPU y comportamiento al agotarse. El piloto
DEVELOPMENT no establece el costo de 3024 unidades.

## Disposicion

Las seis reparaciones son evidencia positiva de mecanismo en el alcance
probado; la aprobacion del diseno M4 sigue PENDIENTE por el estimando del
contraste 15, la identidad ejecutable final y la asignacion de ejecucion y
replay. No firmar el registro `EXTERNAL_AUDITOR`, no instalar un registro de
ejecucion del propietario y no correr ninguna unidad CONFIRMATION. Estos tres
faltantes pertenecen a M4; no detienen calendario, DOIN, H-CORE, noticias o
el barrido sintetico.
