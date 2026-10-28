# Auditoria de comparabilidad del barrido de ventana

Solicitud del propietario: explicar la diferencia grande respecto al control13
y verificar que las otras ventanas no hayan sufrido cambios ocultos.
Estado actual: auditoria y repeticion fresca de13 TERMINADAS. Comparacion
REPRODUCED_BITWISE_POLICY_AND_BUSINESS_TRACES, cero diferencias,100000 pasos,
10/10 publicaciones y copia independiente completa. Politica y trazas de ambos
splits exactamente iguales al control. TRAIN-14.547669%,VALIDATION-7.278253%,
DD VALIDATION10.731872%,203 operaciones. Una semilla,no optimo demostrado.
La continuacion aprobada es el nuevo estudio local12/13/14 con semillas7/17/29.
Las secciones EN CURSO inferiores retienen el corte previo como historia.

## Verificado sobre los cinco cierres

El verificador reproducible de agent-multi `tools/audit_window_evidence.py`
revalida2550 referencias de artefactos/cierres y carga los cinco modelos.
Compara configuraciones efectivas de TRAIN y VALIDATION, datos por hash,
manifiestos, seleccion15, reloj/negocio/costes, seed7 y condiciones de calculo.
Los parametros efectivos coinciden salvo ventana y profundidad1 ahora explicita
(antes valor por defecto). Las diferencias de rutas/IDs/recibos de GPU se
clasifican como operativas; se conserva y compara la asignacion GPU, gym y datos.
No se eliminan del cotejo semillas, costes, optimizador ni cargas de TRAIN.

Todos completaron100000 transiciones,99000 updates,cuatro evaluaciones y
checkpoint final100000,sin early stopping. Los pesos evaluados en VALIDATION
son los del TRAIN correspondiente y no cambian durante la evaluacion. Se
recalculan retornos y soporte desde las trazas. Coinciden las fechas del sufijo
comun (24827 filas TRAIN/6186 VALIDATION). Ninguna diferencia cientifica
inesperada detectada por estas comprobaciones; TEST no se lee.

## Diferencias reales que impiden atribuir todo al contexto

Limitacion adicional de diseno: NO se probaron12 ni14. La regla programada
solo refinaba si9/11/15/17 mejoraban13, por lo que omitio los vecinos del
control cuando este resulto ganador. La cola cumplio su contrato, pero esa
regla no permite decir que afinamos el pico alrededor de13. Corregir el
siguiente diseno para incluir12/14 aunque gane el control, sin reescribir el
contrato cerrado ni inventar resultados faltantes.

| Ventana | Parametros actor | Parametros criticos | Calentamiento nativo |
| --- | --- | --- | --- |
| 9 | 934 | 1382 | 17 |
| 11 | 998 | 1510 | 21 |
| 13 | 1062 | 1638 | 25 |
| 15 | 1126 | 1766 | 29 |
| 17 | 1190 | 1894 | 33 |

La ventana cambia la entrada aplanada al cabezal, el numero de pesos y el
inicio de los episodios de entrenamiento. El sufijo comun no vuelve iguales
las experiencias anteriores. Esto estaba declarado en el diseno; no es una
comparacion de capacidad total ni experiencias identicas.

Sonda nueva, solo inicializacion CPU y0 updates: los extractores del actor
arrancan con el mismo hash en9/11/13/15/17, pero los extractores del critico
tienen hashes distintos. Cambiar la forma cambia el consumo de RNG durante
la construccion anterior, aun usando seed7. Observado, no prueba de que esa
sea la causa del resultado. Igual semilla no significa pesos compartidos ni
un barrido suave de una funcion determinista del contexto aislado.

El13 se reutilizo del runtime anterior. Diferencia de codigo cientifico:
soporte opcional de segunda capa del nucleo; aqui profundidad1. Con las clases
exactas anteriores y actuales,13 produce el mismo hash inicial de politica:
59dd80e8ee475751362a24c52045f4f88166744c60b38ff20212e54963e91299.
Esto verifica inicializacion/parametros, no sustituye reproducir el fit completo.

## Repeticion en curso y conclusion permitida

Se materializo UNA repeticion fresca de13,seed7,desde cero,sin donantes,
con runtime6916e49a del barrido y exactamente la receta anterior. Contrato
f2437f30;3000/100000 transiciones verificadas en62.1238s. Mismo negocio,
datos/costes/monitor,una5090; publicacion/readback y copia independientes.
El primer intento de admision fue rechazado ANTES de entrenar por precedencia
de EnvironmentFile sobre CUDA_VISIBLE_DEVICES. Se corrigio el comando con
binding explicito por UUID, sin cambiar el contrato ni empezar un fit previo.

Comparacion automatica al cierre: hash de politica, trazas de negocio excluyendo
solo run_id/episode_id, metricas y configuracion cientifica de ambos splits.
Si difieren,se registra REPRODUCTION_MISMATCH; no se fuerza una equivalencia.
ETA provisional~70min de fit+evaluaciones,basada en el control original.

Hasta este cierre:13 fue mejor en las corridas seed7 observadas; NO se demuestra
que13 sea optimo ni una superioridad estable entre semillas. Tras comprobar la
reproduccion, medir los vecinos12/14 faltantes y contrastar13 con vecinos en
semillas compartidas es el siguiente discriminante de tamano/estabilidad, no
otra ampliacion ciega del grid. Esos vecinos AUN no estan desplegados.
La seleccion incremental sigue planificada; no se declara iniciada mientras
esta comprobacion ocupa la GPU. No se eliminan ni recalifican los resultados.

## Reproduccion del verificador

```bash
CUDA_VISIBLE_DEVICES="" python -m tools.audit_window_evidence \
  --archive <closed-window-archive> --output <separate-audit-output.json>
CUDA_VISIBLE_DEVICES="" python -m tools.audit_window_evidence \
  --initialization-windows 9 11 13 15 17 --output <initialization-output.json>
```

El modo de inicializacion usa las clases del entorno importado. Para contrastar
versiones, fijar sus fuentes y conservar hashes; nunca se llama a learn.
Pruebas del verificador y rejilla:27 pasan, incluidas deteccion de cambio de
costes/seed/profundidad, rutas externas, pesos de evaluacion alterados y hash
de trazas que ignora identificadores pero no equity. Es evidencia de auditoria,
no una nueva medicion financiera ni un mecanismo alternativo de entrenamiento.
