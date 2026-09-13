# Diseno guiado por pruebas y datos

**Sigla interna:** DGPD  
**Version:** 1.0  
**Uso:** software nuevo, cambios amplios de arquitectura y experimentos de ML
que produzcan artefactos reutilizables.

## 1. Que es y que no es

DGPD combina cuatro ideas compatibles:

1. el modelo en V: definir y descomponer de arriba hacia abajo, integrar y
   verificar de abajo hacia arriba;
2. pruebas de aceptacion y de comportamiento definidas desde requisitos y casos
   de uso;
3. TDD en el nivel de componentes: una prueba unitaria falla antes de escribir
   el codigo que la satisface;
4. desarrollo centrado en datos: contratos, calidad, procedencia, particiones,
   controles de fuga y seguimiento del sistema completo.

No se llama TDD a todo el proceso. El TDD clasico trabaja en ciclos pequenos y
no sustituye el diseno previo de requisitos, arquitectura, interfaces y pruebas
de sistema. DGPD tampoco es una norma nueva ni una excusa para escribir todas
las pruebas antes de entender los datos.

## 2. Principios

- Cada requisito tiene identificador estable y un criterio observable.
- Cada nivel tiene pruebas estructurales y de comportamiento.
- Una prueba de nivel superior se disena antes que la implementacion que debe
  satisfacerla.
- Una decision cientifica se toma con datos que no participaron en su diseno.
- Un spike exploratorio se marca `NON_GOVERNING`: puede aclarar una interfaz,
  pero su salida no concede aceptacion ni elegibilidad.
- La ausencia de efecto, el rechazo y lo inconcluso son resultados distintos.
- Ningun estado se deduce de prosa. El archivo de estado nombra etapa,
  artefactos, evidencia y siguiente accion permitida.
- Un agente que retoma el trabajo lee y valida el estado antes de editar codigo.

## 3. Descenso: definir antes de construir

| Etapa | Trabajo | Salida minima | Puerta |
|---|---|---|---|
| S0 Descubrimiento | Problema, actores, entorno, restricciones, datos y riesgos | acta de problema y limites | la necesidad es entendible y verificable |
| S1 Requisitos | Requisitos funcionales, de calidad, datos, operacion y ciencia | lista con IDs y criterios | sin contradicciones ni verbos no observables |
| S2 Casos de uso | Historias, actores, flujo normal, alternos y fallos | casos ligados a requisitos | cubren operacion y rechazo |
| S3 Aceptacion | Pruebas alfa y beta opcional | protocolo de usuario antes del codigo | cada requisito externo tiene observacion |
| S4 Arquitectura | alternativas, decisiones, interfaces, datos y dependencias | ADR y diagrama | cada eleccion responde a requisitos |
| S5 Pruebas de sistema | carga, recursos, recuperacion, seguridad, datos y comportamiento | diseno de pruebas | cubre propiedades emergentes |
| S6 Componentes | jerarquia, contratos, estados y errores | especificaciones de componentes | interacciones cerradas |
| S7 Integracion | pruebas entre componentes y repositorios | matriz de interfaces | contratos y fallos cruzados cubiertos |
| S8 Pruebas unitarias | casos nominales, bordes, propiedades y mutantes | pruebas rojas revisadas | implementacion permitida |

La beta puede quedar `DEFERRED` con razon y condicion de apertura. El alfa con
los propios responsables no se omite.

## 4. Ascenso: construir y verificar

| Etapa | Trabajo | Evidencia para pasar |
|---|---|---|
| S9 Implementacion unitaria | codigo minimo para satisfacer S8 | pruebas unitarias y mutantes relevantes verdes |
| S10 Integracion | conectar componentes en el orden de dependencia | pruebas S7 verdes y fallos tipados |
| S11 Sistema | ejecutar la arquitectura completa | pruebas S5, recursos y recuperacion verdes |
| S12 Alfa | casos reales con el equipo | pruebas S3 alfa y disposicion de defectos |
| S13 Beta | usuarios externos cuando sea posible | protocolo, consentimiento y evidencia; o aplazamiento explicito |
| S14 Entrega/operacion | versionar, observar, revertir y mantener | artefacto reproducible, runbook, telemetria y criterio de retiro |

Corregir un defecto puede bajar al nivel que lo origino. La trazabilidad se
actualiza y se repiten todas las pruebas afectadas; no se reinicia el proyecto
entero ni se salta directamente al codigo.

## 5. Extensiones para datos y ML

Antes de S1, o dentro de S0, se aplica comprension del negocio y de los datos al
estilo CRISP-DM. Ademas de requisitos de software, se definen:

- unidad estadistica, dominio, objetivo, costo de error y politica de
  abstencion;
- identidad de datos, licencia, semantica, unidad, `event_time` y
  `available_time`;
- particiones temporales, embargo, origenes y datos reservados;
- baselines y controles negativos;
- presupuesto completo, semilla y tratamiento de multiplicidad;
- cambio de distribucion, ausencias, extremos y criterios de no inferioridad;
- diferencias entre calibracion sintetica, decision publica y revalidacion de
  dominio.

Pruebas obligatorias de pipelines temporales:

1. invariancia de prefijo y perturbacion del futuro;
2. ajuste solo en training;
3. paridad batch, incremental y restart;
4. identidad byte-a-byte de datos, particion, codigo y configuracion;
5. separacion estructural entre entradas y targets;
6. comparacion raw/procesado y controles de igual capacidad;
7. costos, fallos e inconclusos en la misma memoria experimental que los
   exitos.

Un buen resultado en entrenamiento no reemplaza ninguna de estas pruebas.

## 6. Pruebas estructurales y de comportamiento

**Estructurales:** comprueban que existe la frontera: firma de API, esquema,
dependencias permitidas, direccion de llamadas, ausencia de bypass, identidad
de artefactos y orden de las rejas.

**De comportamiento:** comprueban resultados observables ante ejemplos,
propiedades, perturbaciones, fallos, reinicios y controles adversariales.

Ambas se exigen en cada nivel. Una busqueda de texto no prueba comportamiento y
un ejemplo verde no prueba que no exista otro camino que eluda la frontera.

## 7. Trazabilidad

La matriz minima es:

```text
requisito -> historia/caso -> aceptacion -> arquitectura
          -> prueba de sistema -> componente -> integracion
          -> prueba unitaria -> commit -> evidencia -> disposicion
```

No se acepta un requisito sin prueba, una prueba sin requisito o codigo nuevo
sin ubicacion en la jerarquia. La plantilla esta en
`MATRIZ_TRAZABILIDAD.md`.

## 8. Estado persistente y reanudacion

Cada proyecto conserva `PROJECT_METHOD_STATE.json`, validado contra
`PROJECT_METHOD_STATE.schema.json`. Protocolo al retomar trabajo:

1. leer las instrucciones del repositorio;
2. ejecutar `python tools/test_led_state.py validate <estado>`;
3. leer `current_stage`, `next_allowed_actions`, decisiones y ultima evidencia;
4. comprobar que los artefactos citados existen y que el arbol no contiene
   cambios ajenos que se pretendan reemplazar;
5. ejecutar solo una accion permitida o registrar por que una accion concreta
   requiere un insumo externo;
6. actualizar estado, matriz y evidencia antes de terminar la sesion.

Estados permitidos: `NOT_STARTED`, `IN_PROGRESS`, `PASSED`, `BLOCKED` y
`DEFERRED`. `BLOCKED` siempre nombra el objeto faltante, su poseedor y la minima
decision o evidencia que lo abre. `DEFERRED` solo aplica a etapas opcionales.

## 9. Referencias metodologicas

- NASA, *Systems Engineering Handbook*, rev. 2: descomposicion a la izquierda
  del modelo en V e integracion, verificacion y validacion a la derecha:
  <https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf>.
- Janzen y Saiedian, *Test-Driven Development: Concepts, Taxonomy, and Future
  Direction*, sobre el ciclo prueba-codigo-refactor y su alcance:
  <https://people.eecs.ku.edu/~hossein/Pub/Journal/2005-Saiedian-IEEE-Computer.pdf>.
- Breck et al., *The ML Test Score*, sobre pruebas de datos, modelos,
  infraestructura y monitoreo:
  <https://research.google.com/pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf>.
- Sambasivan et al., *Everyone Wants to Do the Model Work, Not the Data Work*,
  sobre cascadas producidas por problemas de datos:
  <https://www.shivanikapania.com/assets/chi2021paper.pdf>.
- Sculley et al., *Hidden Technical Debt in Machine Learning Systems*, sobre
  dependencias de datos y deuda a nivel de sistema:
  <https://papers.nips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf>.
- Ribeiro et al., *Beyond Accuracy: Behavioral Testing of NLP Models with
  CheckList*, como ejemplo de capacidades y pruebas de comportamiento mas alla
  de una metrica agregada: <https://aclanthology.org/2020.acl-main.442.pdf>.
- IBM, vision general de CRISP-DM:
  <https://www.ibm.com/docs/en/spss-modeler/18.6.0?topic=dm-crisp-help-overview>.

Estas fuentes apoyan partes del metodo; ninguna define por si sola DGPD.
