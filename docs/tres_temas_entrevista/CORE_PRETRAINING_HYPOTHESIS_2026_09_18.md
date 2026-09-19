# H-CORE: preentrenamiento del nucleo sobre representaciones fijadas

Autor de la hipotesis: Harvey. Registro: Musashi, 2026-09-18.
Estado: HIPOTESIS INCORPORADA AL PLAN; DISENO NUMERICO Y EJECUCION PENDIENTES.
No se envia una orden de interrupcion a Satoshi ni se modifica una corrida sellada.

## 1. Punto de entrada y alcance

Extension de desarrollo de P-MOD, programada tras E1: primero comparar R0/R1/R2
del detector, comprobar el modelo modular con cabezales y seleccionar en desarrollo
un extractor completo que pueda fijarse. Despues verificar su materializacion
causal por particion; solo entonces ejecutar H-CORE. No es un requisito de E0.

Tareas persistentes: MOD-E1 -> MOD-FROZEN-PREFIX -> MOD-CORE-PRETRAIN.
El primer contraste es de pronostico; su contraparte RL se evalua en E3, con sus
propios datos, acciones, reward y presupuesto. No transferir el veredicto por MAE.
Si se incorpora al procedimiento principal, revisarlo y fijarlo ANTES de su
confirmacion. H-CORE no se agrega silenciosamente a H1-H3 ni a su multiplicidad.
Conservar la extension aunque R0 sea mejor: un extractor entrenado conjuntamente
tambien puede congelarse. El preentrenamiento del detector no tiene que ganar.

## 2. Comparacion de arquitecturas aprobada para el work plan

El owner aprobo comparar alternativas antes de elegir: ARCH-A es la referencia
inicial, no la ganadora. Esta comparacion pertenece al desarrollo E0 y precede
a la eleccion del procedimiento en E1. Se adopta al conciliar el retorno de
Satoshi, sin cambiar su diseno o corrida actual. Si ya hay evidencia compatible,
verificarla antes de planificar nuevas mediciones.

| ID | Rama antes de fusion | Pregunta |
|---|---|---|
| ARCH-A, referencia | Conv1D causal de una o dos capas; integrador identidad | Basta detectar localmente y dejar la integracion temporal al nucleo? |
| ARCH-B | Bloque convolucional causal dilatado tipo TCN | Aporta integrar mayor historia dentro del grupo, sin recurrencia? |
| ARCH-C | Conv1D causal + GRU o LSTM con secuencias | Aporta estado recurrente por grupo frente a las alternativas convolucionales? |
| ARCH-0, control | Sin extractor aprendido; adaptacion dimensional si procede | Las ramas aprendidas aportan frente al nucleo/cabezal que recibe las entradas? |

GRU frente a LSTM es una subeleccion de desarrollo documentada antes del contraste,
no dos resultados entre los cuales escoger el favorable. No se ejecuta un producto
cartesiano ilimitado de profundidades, anchos, periodos y preprocessors.

Interfaz comun, ilustrada con ARCH-A:

    X por grupo -> Conv1D causal (1 o 2 capas) -> I=identidad
      -> A=identidad o proyeccion puntual necesaria
      -> alineacion causal + concatenacion de canales
      -> nucleo temporal compacto -> cabezal de la tarea

- Detector de ARCH-A: una o dos Conv1D causales con activacion declarada; ReLU como candidato
  inicial. Filtros, kernels, dilataciones y ventana se derivan del mecanismo y del
  presupuesto antes de medir; una/dos capas no implica alcance temporal suficiente.
- Integrador de ARCH-A: identidad. ARCH-C incorpora recurrencia como contraste,
  no como requisito universal entre detector y fusion.
- Adaptador: identidad si no hace falta; si existe una proyeccion Conv1D kernel 1,
  declarar anchos/activacion/pesos y entrenarlos o fijarlos segun contrato. La
  concatenacion NO exige que todas las ramas tengan el mismo numero de canales.
- Fusion: alinear tiempos de emision y concatenar, conservando secuencias. No Add,
  pooling temporal ni Flatten antes de fusion como sustituto encubierto.
- Nucleo: inicialmente candidato convolucional temporal causal compacto, seguido
  de proyeccion de canales si se necesita un cuello de botella. Receptivo total
  de rama+nucleo calculado; no fingir que una proyeccion puntual integra retardos.
  No se fija un ancho arbitrario ni se exige recurrencia en esta primera version.
- Cabezal de pronostico: explicitar extraccion de la ultima posicion valida y
  salida por target/horizonte, con salida lineal para regresion de valores reales.
  Politica/valor para RL tienen sus propias salidas y objetivos en E3.

Los cabezales forman parte de la prueba: la reconstruccion del autoencoder no es
el desenlace principal. Baselines ingenuos/lineales y oraculos permanecen como
controles; resultados de otros receptores no sustituyen los del modular real.
ARCH-A/B/C se comparan en desarrollo E0; atencion y comparadores publicos permanecen
en E1 conforme al protocolo, sin presumir transferencia entre arquitecturas.

Mantener el tipo/configuracion del nucleo y el cabezal controlados al investigar
las ramas; fijar interfaz de salida comun cuando sea viable. Si cambia la interfaz
en ARCH-0, contabilizar la adaptacion y sus parametros, no llamarlo mismo modelo
exacto. Controlar informacion accesible, filas/objetivos, contexto, oportunidades
de ajuste, capacidad y recursos; igualar parametros por si solo no basta.

Calcular el alcance rama+nucleo y verificarlo. La concatenacion no integra por
si sola instantes; una proyeccion kernel 1 tampoco. La profundidad local y el
alcance temporal total son variables diferentes. No condenar ARCH-A por darle al
nucleo historia insuficiente ni compensarla con un nucleo mucho mayor sin declararlo.

Comparar H2/H3 DENTRO de cada arquitectura y despues sus interacciones con regimen
y arquitectura. En H3 cada par recibe exactamente las mismas activaciones de su
extractor fijo; no comparar dos extractores distintos y atribuirlo a fusion.
Aprendizaje/volumen, efectos pareados, incertidumbre, estabilidad entre regimenes y
costo total gobiernan la decision. ARCH-A se prefiere si satisface adecuacion y
equivalencia predeclarada a menor costo; B/C si aportan mejora relevante que justifique
su costo. Si varia por regimen, conservar la regla condicionada. Si es inconcluso,
no declarar un ganador por el promedio. Seleccion solo DEV, nunca reserva cientifica.

La simplificacion reduce componentes y costo; no supone que un autoencoder deba
tener un decodificador espejo ni que una LSTM sea imposible de preentrenar.

## 3. Prerrequisitos verificables

1. E0 y desarrollo E1 han comprobado que el receptor modular aprende los controles
   positivos pertinentes, sin exigir que venza a sus controles de hipotesis.
2. Comparacion del detector R0/R1/R2 registrada, con curvas, datos/soporte, costo y
   parametros elegidos solo en desarrollo. El resultado puede ser negativo.
3. Prefijo completo fijado: preproceso, grupos, D/I/A, alineacion, fusion y estados
   mutables. R1 congela D, no automaticamente un adaptador aprendido ni todo E.
4. Exportacion por particion/corte de entrenamiento, modo de inferencia, mascaras,
   calentamiento, limites de ventana y resets declarados. Nada se ajusta con test.
5. Paridad entre salida directa y materializada bajo tolerancia justificada,
   identidades de filas y tiempos iguales, incluido el comportamiento en bordes.
6. Datos derivados registrados por data-gov/lake con linaje a inputs, codigo,
   contrato y checkpoint exactos. Metricas/costos/terminales en el warehouse.
7. Diseno propio de H-CORE: particiones, pares, semillas, cuello de botella,
   objetivos, margenes/precision y presupuesto total fijados antes de resultados.

La materializacion no es una serie generica por timestamp si el extractor depende
del inicio de cada ventana: conservar esa identidad y el contexto realmente usado.
Un extractor de una semana no se reutiliza retroactivamente como si hubiera estado
disponible en semanas anteriores. No exportar pesos elegidos mirando la prueba.

## 4. Hipotesis y contraste

**H-CORE:** con un prefijo identico y fijado, preentrenar el nucleo como codificador
de las secuencias fusionadas y ajustarlo para la tarea mejora el desempeno fuera
de muestra, o reduce costo a un nivel de desempeno predeclarado, frente al mismo
nucleo inicializado desde cero. Separar eficacia y eficiencia: no elegir despues
el criterio que resulte favorable.

| Brazo | Nucleo | Cabezal | Prefijo |
|---|---|---|---|
| CORE-A, referencia principal | Inicializacion desde cero; aprende con la tarea | Entrenado con la tarea | Fijo e identico |
| CORE-B, contraste principal | Preentrenado y luego ajustado con la tarea | Misma arquitectura que CORE-A | Fijo e identico |
| CORE-C, reutilizacion | Mismo checkpoint preentrenado que CORE-B, congelado | Solo se entrena el cabezal | Fijo e identico |

CORE-A/B/C identifica regimenes de entrenamiento del MISMO nucleo, no las familias
ARCH-A/B/C de la comparacion anterior. Son ejes y experimentos separados.

El autoencoder recibe secuencias fusionadas de entrenamiento; su encoder es la
misma arquitectura de nucleo usada en CORE-A/B/C y su decoder es auxiliar. Reconstruir
caracteristicas completas o enmascaradas es una eleccion de desarrollo explicitada,
no ambas intercambiadas segun el resultado. Mantener el eje temporal; calcular
distorsion por canal con escalado train-only, para evitar dominio de gran amplitud.

Contraste principal de pronostico: MASE(CORE-B)-MASE(CORE-A), con denominador compartido;
negativo favorece preentrenar. Reconstruccion y longitud latente son diagnosticos.
Incluir train/val/evaluacion, MAE, dispersion, parametros y costo por fase. Fijar
metrica economica y signo propios para la replica RL; ninguna mejora es presupuesta.

Comparar oportunidad de ajuste supervisado y contabilizar TODO el preentrenamiento,
decoder, exportacion, almacenamiento, lectura, fallos y repeticiones. Anadir control
de presupuesto total comparable para separar inicializacion de computo adicional.
Si se amortiza sobre varios usos, publicar costo inicial, marginal y numero real
de reutilizaciones; no comparar solo el tiempo del cabezal contra un entrenamiento
completo. No seleccionar semillas ganadoras ni tratar ventanas como tareas nuevas.

Una fase posterior permite ajustar conjuntamente prefijo+nucleo+cabezal, comparando
inicializaciones controladas. Consume inputs originales y deja de usar la cache fija
como sustituto del prefijo: hacen falta sus gradientes. No mezclar esa fase con CORE-A/B.

## 5. Pruebas y cierre exigidos

Antes de entrenar: cambio de futuro no altera activaciones pasadas; cache/directo
equivalentes; un checkpoint o fit distinto invalida reutilizacion; filas y mascaras
exactas; la cache no recibe gradientes; CORE-A/B comparten arquitectura, informacion y
protocolo; CORE-B actualiza nucleo, CORE-C no; decoder nunca interviene en inferencia final.
Verificar tambien entrenamiento real, parada por validacion, recarga y errores
desde arrays. Medir cobertura de reconstruccion y costo sin atribuirle utilidad.

Retorno con tabla CORE-A/B/C, control de costo total, tareas/regimenes, intervalos y
limites. Resultado admisible: mejora, equivalencia, perjuicio o inconcluso segun
criterios fijados. Nada de "extractor funciona super bien" como unica condicion.
Versionar el dataset derivado, conservar originales y registrar todos los intentos.

## 6. Adopcion sin interrumpir a Satoshi

El owner aprobo esta comparacion para el work plan e indica que Satoshi aun trabaja.
La adicion queda registrada, con adopcion pendiente de conciliar su retorno; no
ordena declarar ganadora a Conv1D simple. Si ya hay diseno/corrida comprometidos,
preservar contrato/resultados y abrir un sucesor explicito. No cambiar capas ni
hipotesis a mitad de medicion. No exigir una nueva microaprobacion de lo ya aprobado.
No se emite una campana adicional ni una orden de reiniciar servicios en este acto.
