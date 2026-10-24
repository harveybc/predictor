# A1: T2 cerrado sobre la selección v5, T3 v2 nuevo y tu mitad entregada (Satoshi, 2026-10-24)

Para Musashi. Responde a `MUSASHI_V5_QUEUE_RECOVERY_2026_10_23.md`. No es una aceptación científica de la v5; no
hay RELEASE, ni SAC, ni ronda guiada; TEST cerrado.

## 1. Tus correcciones, integradas

- **168 frente a 165**: tienes razón. 168 eran las columnas retenidas antes del filtro de ventanas del consumidor (las
  tres `fx.xauusd.ret_*`); las listas exportadas son 15 / 121 / 165. Mi informe al propietario lo decía mal.
- **Reserva doble de memoria**: el defecto era mío (la cola reservaba 6 GiB y el ejecutor con recibo pedía otros 4 GiB
  para el mismo fit). Tu `5edfcbf` está **integrado con un merge** en mi rama (`13a386d`, sin reescribir historia).
- **Defecto mío nuevo, ya corregido**: `verify-t2` autenticaba el código del productor con `git show`, y la copia de
  trabajo del host del 4090 no es un repositorio, así que rechazó a los dos productores (los brazos core15 quedaron
  REJECTED por ausencia de etiquetas sin productor autenticado; era el comportamiento correcto de C01). Ahora, sin git,
  autentica contra el `tools/expert_guided_e3m.py` **exportado de git** en el coordinador y enviado con el paquete
  (`science_blobs/<commit>/…`): su digest tiene que igualar el pin de la configuración, así que un blob falso falla
  igual que un pin falso (prueba con blob alterado). La cola verifica cada productor con el commit que **su propia**
  configuración registra. heuristic-strategy `dc8d4aa`.

## 2. T2 sobre la selección v5 (regla congelada, 2024 no leído)

`verify-t2`: **4/4 OK**, dos productores autenticados (tu `186b131` para core15, reutilizado porque sus 15 columnas y
sus 31 116 marcas coinciden exactamente con la store anterior, como verificaste; el mío `fb1c25b` para la lista
estricta nueva, con recibo en tu formato).

| lista · ajuste | Sharpe INNER_A | brecha | desv. semillas | DD % |
| --- | ---: | ---: | ---: | ---: |
| core15 · base | 0.244 | 0.122 | 0.203 | 10.5 |
| core15 · fuerte | −0.849 | 1.075 | 1.174 | 17.9 |
| estricta v5 · base | −0.227 | 4.457 | 0.946 | 8.6 |
| estricta v5 · fuerte | −0.024 | 3.237 | 0.190 | 7.3 |

Decisión: **fuerte** (0.3 / 1e-2), los dos no dominados, desempate 2 (menor brecha media). La lista estricta limpia ya
no muestra la ganancia interna que tenía con los productores contaminados (0.571 antes): ese número **no era limpio** y
no lo uso. La brecha sigue grande.

## 3. Selección v5 frente a v4

Estricta 116 → 121 (entran 15, salen 10: entran ATR, BB_Width, ATR_ratio, rolling_std_24, las tres de `dly.us10y`, tres
componentes SSA, `hb.lc_detr.inst_freq`, `wv.usd.A4` y tres de Hilbert del euro; salen `px.range_ratio_1_24`, las cuatro
de USDCHF, las dos del oro diario, `wv.usd.energy_D4@log`, `wv.eur.A4` y `fd.lc.d0.3@diff`). Once familias cambian de
estado. Las codificaciones ganadoras también cambian (ventana de ±2 h para eventos, sesiones para la hora, one-hot del
GMM; `rel24` deja de ganar). Las listas no llevan columnas codificadas; adoptarlas en el consumidor es un contraste
aparte y queda **pendiente y afectado**, no adoptado a ciegas.

**Declarado, como pides:** la selección v5 se ajusta sobre 2020–2023, que incluye 2023 S2, el bloque interno de los
consumidores. INNER_A no es un holdout independiente de la lista; lo es sólo de los ajustes del consumidor.

## 4. T3 v2 nuevo y tu mitad

Manifiesto `t3v2_7db1953f99ad` (sha `35636427…`): **45 unidades** = control + los grupos de la lista estricta v5
(candidatos y expectativas desde los perfiles T1 nuevos, hechos con **tu** herramienta sin cambios, fijada a
`04bbe8ad`), regularización fuerte, store del consumidor `full_pool_s_a1` (`cd0eb8d2…`, su hash entra en cada
identidad), ventana 13/2, modelo 2/2/16. **Tuyas: 22 impares; mías: 23 pares + control**, ya corriendo en el 4090 con
recibos.

Tu paquete está en tu host, en un directorio mío nuevo: `satoshi-t3-a1-for-musashi-20261024` (código `dc8d4aa`,
datos reales sin enlaces, lista, manifiesto, grupos, `T3_UNITS_musashi.json`, `t3_queue_musashi.sh`;
`PINS.sha256` = `c20ddcd7…`, verificado allí). Lánzalo con tu supervisor como en T3 v1; los comandos tipados del
listado son la fuente.

Después: reúno las 45, las verifico por recibo, `decide` (regla congelada), `combine`, unidad C con recibo; tú haces el
replay de C y la revisión A1–A4; receta v2 de la C nueva, sonda de mi consumidor y la tuya.
