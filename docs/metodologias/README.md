# Metodologias de desarrollo e investigacion

La guia vinculante para software nuevo es
[`DISENO_GUIADO_POR_PRUEBAS_Y_DATOS.md`](DISENO_GUIADO_POR_PRUEBAS_Y_DATOS.md).
Combina especificacion y verificacion en forma de V, pruebas de aceptacion y de
comportamiento, TDD en componentes y controles propios de sistemas de ML
centrados en datos.

Cada iniciativa mantiene un archivo `PROJECT_METHOD_STATE.json` validable con:

```bash
python tools/test_led_state.py validate <ruta-al-estado>
python tools/test_led_state.py status <ruta-al-estado>
```

La plantilla de trazabilidad esta en `MATRIZ_TRAZABILIDAD.md`; el esquema y un
estado minimo valido se incluyen en este directorio.
