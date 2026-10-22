# Paquete de ejecución para el worker de Musashi (2026-10-22, orden del propietario)

El propietario pidió repartir trabajo al worker libre. Es ejecución sobre la
versión v3 compartida, sin decisiones nuevas: los brazos están declarados, la
regla es "reportar, no seleccionar". Todo está ya en el host en
`<musashi-root>` = `/var/tmp/expert-guided-musashi-20261022/`.

| Ruta | Contenido | Identidad |
| --- | --- | --- |
| `code/` | heuristic-strategy `satoshi/expert-guided-rl-20261019` @31d90f7 (archivo `hs_code_final.tar.gz`) | archivo sha256 `efa23475193865fc…` |
| `data/admitted`, `data/sel15` | EURUSD admitido (c80847bb…) y CSV sel15 (etiquetas/calendario) | pins verificados por el cargador |
| `data/full_pool/` | pool completo materializado 31 116 × 280 | `2f7b3b8c…` (MANIFEST con `features`) |
| `data/full_pool_enc/` | pool + 12 columnas de codificación (sesiones, GMM one-hot, rel24, rel480) | `58e94711…` |
| `in/FEATURE_LIST_*.json` | listas v3: core15 15 / estricta 125 / shortlist 173 / estricta+codificaciones 137 / core15+relativas | SHORTLIST.json v3 |
| `gamma_queue_musashi.sh` | cola secuencial reanudable (marcadores en `STATE/`) | — |

## Qué corre (12 brazos, E3-M FULL head 16, escala fija TRAIN + regla de estacionariedad, 3 semillas, mismas filas)

**C. Contrato principal** (`--split-contract consolidated`: TRAIN 2020-2023 con
bloque interno de selección 2023H2, VALIDATION 2024 una lectura, TEST cerrado):
`cons_core15`, `cons_shortlist_strict`, `cons_shortlist`, `cons_shortlist_strict_enc`.
Los quité de mi cola en dragon (marcadores puestos) para no duplicar.

**D. Diagnóstico de brecha de generalización** (brazo dev3y, para el rediseño de la
ronda limpia): listas {core15, estricta} × estado del agente {full, none} × regularización
{base: dropout 0.1 / wd 1e-4, fuerte: dropout 0.3 / wd 1e-2} = 8 brazos. Contexto: con
escala fija los brazos dev3y ya cerrados dan core15 −1.14, estricta −1.13 y shortlist −1.96
de Sharpe en INNER_A, y las listas grandes memorizan TRAIN (netos de millones). Se reporta
TRAIN / INNER_A / INNER_B / 2024 y la brecha por brazo; ninguna selección sale de aquí.

## Cómo lanzarlo

```bash
cd /var/tmp/expert-guided-musashi-20261022 && MEM=4G GPU=0 setsid nohup bash gamma_queue_musashi.sh > nohup.log 2>&1 < /dev/null & disown
```

`MEM` y `GPU` son tuyos (tu política de admisión; el E3-M usó ~2–3 GB RSS en dragon;
si la admisión rechaza 4G, espera o cambia el comando, nunca bajes la petición de un
comando idéntico). Cada brazo deja `out/<nombre>/E3M.json` (con `support_hash` por
periodo, `split_contract`, `agent_state`, `params`), `SCALER.json` (constantes +
tabla de estacionariedad) y `PROBE.json`. Marca final: `QUEUE_MUSASHI_DONE` en `LOG`.

Cuando termine, con avisar basta: yo recojo `out/*/E3M.json` por `scp` para la
tabla de consumo de la solicitud de auditoría (`CONSUMERS.md`), con la atribución
de ejecución a tu worker. Nada de esto toca DOIN ni tus colas SAC.
