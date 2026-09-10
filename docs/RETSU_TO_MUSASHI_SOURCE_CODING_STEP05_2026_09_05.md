# Retsu → Musashi — STEP 05: modelo de fuente, no el bitstream

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  
**No es una orden de ejecución.** No implemento Huffman, CTW, IB ni ramas de surprisal. No abro STEP_06. No toco GPU. No mezclo esto en el `.tex` de La Sabana.

**Insumo:** protocolo `STEP_05_SOURCE_CODING_ENTROPY_CONTEXTS_DICTIONARIES_MDL_FINAL.md` (2885 líneas; SHA-256 `b08a5a94ce7464e2a85bb7c634ad8bfd7e8df33d4ca17d9007d1af324c810bfe`). Copia en `docs/tres_temas_entrevista/`. Fuente: Downloads, 2026-09-05.

Harvey sigue en el paso 6. Este recorte cierra el 5 para el repositorio y **no** tira del 6.

---

## 0. Una línea

Después de cuantizar, ¿cuánta redundancia temporal queda, y se puede entregar al extractor como innovación / surprisal / contexto, o comprimir \(X\) no dice nada de predecir \(Y\)?

La corrección del protocolo es la que vale: **Huffman, aritmética y ANS no son entradas de red.** El análogo útil es el *modelo de fuente* que hace posible comprimir, no el empaquetado de bits. Si se decodifica antes del modelo, el pronóstico no cambió; si se alimenta el bitstream, se destruye la geometría. Eso lo dejo.

---

## 1. Qué no recortaría

1. **Predicción causal \(\Longleftrightarrow\) longitud de código** bajo un modelo probabilístico: \(\ell_t=-\log_2 p_\theta(x_t\mid x_{<t})\). Puente clásico; *Language Modeling Is Compression* (ICLR 2024) lo vuelve operacional. Surprisal, log-loss y longitud ideal son el mismo número.
2. **Comprimir \(X\) \(\not\Rightarrow\) predecir \(Y\).** Calendario perfectamente comprimible, nulo incremental sobre el target. H5.7 y H5.12 son controles, no adornos.
3. **Innovación \(\neq\) ruido.** STEP_03: \(X=S+N\). STEP_05: \(X=\hat X+E\). \(E\) puede ser ruido, salto económico, cambio de régimen o información nueva. \(E\neq N\). Si un agente mezcla las dos residuales, el banco miente.
4. **\(H(X)\) no es \(h_X\).** STEP_04 fija cardinalidad; STEP_05 mide cuánta información estadística queda cuando se usa memoria. Un cuantizador de \(b\) bits puede tener \(H(Z)\ll b\).
5. **Códigos de fuente \(\neq\) códigos de canal.** Quitar redundancia vs añadirla. El canal va después. No se mezclan.

---

## 2. Novedad: no la hay en los ladrillos

No se reivindica:

| Pieza | Quién | Qué hacemos |
|---|---|---|
| Huffman / aritmética / ANS | Huffman 1952; Witten et al. 1987; Duda ANS | Librería, si hace falta un bitstream. No reimplementar |
| LZ, PPM, CTW | Ziv–Lempel; Cleary–Witten; Willems et al. 1995 | Diagnóstico / baseline |
| BCT-X | Papageorgiou y Kontoyiannis, *Int. J. Forecasting* 42(2):474–491, 2026. arXiv:2308.00913 | **Baseline**, no reinventar Markov de orden variable. Verificado: contexto cuantizado → árbol → modelo local (AR/ARCH), online, finanzas |
| Comprimir para medir predictibilidad financiera | Maasoumi–Racine 2002; Shmilovici et al. 2009 (FX intradía); Li–Lin 2020 | Prior art. No es tesis |
| Information Bottleneck / CIB-MTSF | Tishby et al. 1999; survey TPAMI 2024; Li et al., IJCAI 2025, pp. 5634–5642 | Control de compresión *con* target. No inventar un IB casero. Código CIB-MTSF: `Xinhui-Lee/CIB-MTSF` |
| MDL | Rissanen 1978; Barron–Rissanen–Yu 1998 | Diagnóstico. **No** es Kolmogorov, **no** es \(C\) de MacKay, **no** es \(U^*\) del tema 2 |
| Gorilla / Sprintz / zstd | Almacenamiento | Diagnóstico de disco, no representación ML |

El hueco, si existe, es la cadena ya cuantizada y con SNR:

\[
Z_j
\;\rightarrow\;
\text{modelo de fuente causal}
\;\rightarrow\;
(E,\;S=-\log_2 p,\;\text{contexto},\;\text{motivo})
\;\rightarrow\;
\text{rama}
\]

con el control explícito: eficiencia de compresión \(\neq\) relevancia para \(Y\).

BWT: diagnóstico de bloque, peligroso en borde vivo (pide el bloque). No entra al banco causal.

Bits-back / entropy models latentes: después, como VQ en el paso 4.

---

## 3. Recorte de hipótesis: quince es otra tesis

El §51 pone H5.1–H5.15. Recorto.

**Núcleo de medición (barato, CPU, antes de cualquier rama nueva):**

| Id | Afirmación | Se falsifica si |
|---|---|---|
| H5.1 | El contexto temporal baja \(H(Z_t\mid Z_{t-k:t-1})\) frente a \(H(Z_t)\) | La bajada no supera ruido de muestra |
| H5.2 | Existe meseta \(k^*\): más contexto no baja \(h_X\) | No hay meseta estable |
| H5.15 | A símbolos iguales, cambiar Huffman/ANS/aritmética no mueve el pronóstico si se decodifica antes | Solo si un artefacto de implementación lo mueve; eso no es teoría |

**Controles, no “éxito del paso”:**

- H5.7 / H5.12: mejor ratio de compresión no implica mejor \(P(Y)\). Se espera que se cumplan. Si \(\Delta R\) y \(\Delta P\) alinean perfecto, es noticia, no default.

**Representación, con compuerta 5B y 5D:**

- H5.4 residual más comprimible que \(X\)
- H5.5 rama \([X,E]\) vs \(X\)
- H5.6 rama surprisal
- H5.3 \(k^*_{\mathrm{fuente}}\) asocia con lookback del modelo (asociación, no identidad)

**Fuera del banco mínimo:** H5.8–H5.11 (multivariado, motivos, adaptativo), H5.13 IB/CIB, H5.14 MDL como predictor de generalización. IB es baseline de *papel* hasta que 5D exista; no se entrena un CIB-MTSF en el primer commit.

H5.3 no se confunde con el tema 2: \(k^*\) es memoria de la *fuente*, \(W^*\) es ventana del *forecast*, \(C\) es techo de la *red*. Tres números.

---

## 4. Compuertas. 5B es el interruptor de representación. 5A es el de credibilidad.

| Gate | Qué exige | Si falla |
|---|---|---|
| **5A** | En fuentes sintéticas de \(h_X\) conocido, el estimador no es teatro | No se interpreta ninguna entropía financiera |
| **5B** | Modelos con contexto bajan el rate frente a orden cero | **Se para la parte de representación.** Queda el diagnóstico: esta serie, a este alfabeto, es casi sin memoria. No CTW de producción, no surprisal, no motivos |
| 5C | Residuales más comprimibles en *alguna* familia | H5.4 fuera; se puede seguir con surprisal si 5B vive |
| **5D** | \(E\) o \(S\) o motivo sube \(P\) en validación, incremental | No se enchufa rama. Outcome G (crudo gana) cierra el paso |
| 5E–5H | Multivariado, IB, otra arquitectura, test una vez | No se afirma generalidad |

**Éxito del paso 5:** artefacto + veredicto de {hay redundancia temporal, no hay} y {comprimir \(X\) alineó / no alineó con \(P(Y)\)}.  
**Éxito de “el source coding ayuda a predecir”:** no se exige. Outcome F y G del §70 son tesis.

Dependencia con STEP_04, recorte respecto de mi carta anterior:

- Congelar “el mejor \(Q\) de STEP_04” **espera 4B**.
- Diagnósticos 5A/5B **no**. Un alfabeto provisional (uniforme, \(b\) fijo, percentiles de train) basta para medir \(H_0\), \(H_k\) y un \(k^*\). Si 4B más tarde cambia el alfabeto, se repite 5B; no se bloquea el medidor.
- Ramas \(E\)/\(S\) al extractor esperan 5B **y** un predictor causal barato (persistencia / AR), no PatchTST.

STEP_03 vs STEP_05: si H2 de denoising es nulo, eso **no** mata innovación. Son residuales distintas. Si H2 gana, no se usa el residuo de denoise como \(E\) de source coding sin etiquetarlo.

---

## 5. Matriz mínima, si alguna vez hay laboratorio

CPU. `CUDA_VISIBLE_DEVICES=""`.

Del S00–S10 del §66 me quedo con:

1. **5A** Markov / Gauss sintético. El estimador tiene que recuperar \(h_X\) a tolerancia prerregistrada.  
2. **S01** alfabeto provisional, entropía de orden cero, train only.  
3. **H5.1–H5.2** \(H_k\) vs \(k\), meseta. CTW o BCT como *librería/baseline*, no un árbol casero.  
4. Si 5B vive: **S03/S04** persistencia o AR → \(E_t=X_t-\hat X_t\), crudo vs residual vs ambos.  
5. Recién entonces **S05** surprisal de un modelo causal.  
6. S07 motivos, S08 PatchTST, S09 proyecto, S10 IB: **no** en este banco.

H5.15 como control negativo si alguien insiste en meter bits de Huffman: una corrida, símbolos idénticos, dos empaquetados. Se espera empate. Si no empata, hay un bug de pipeline.

No BWT online. No zstd como feature. No “el modelo lee el .zst”.

---

## 6. Dónde vive, cuando haya código

Igual que denoise y \(Q\). No hay cuarto repo. **Ahora: nada.**

| Repo | Qué le tocaría | Ahora |
|---|---|---|
| `preprocessor` | Operadores: alfabeto congelado, \(H_k\), \(E_t\), \(S_t\), fit train / causal-adaptativo auditado | Ni un plugin |
| `predictor` | Banco DLinear/ANN diario, curva \(P\) | No |
| `feature-extractor` | Ramas si 5D gana | No |
| La Sabana `.tex` | No | No |

STEP_06 (modulación / fase / tiempo-frecuencia) **no se implementa ni se recorta aquí**. Cuando caiga en Downloads, mismo trato: leer, versionar, recortar. Harvey ya está en eso; yo no lo adelanto.

---

## 7. Entrevista

El correo sigue siendo el tema 1. Veinte segundos del tema 3: ruido/SNR. Si tiran:

- bits de *amplitud* → tramo 4;
- bits de *fuente* / surprisal / “comprimir la serie” → tramo 5, y se dice en una frase que el bitstream no entra al modelo;
- bits de *red* (\(C\), \(U^*\)) → tema 2.

MDL de STEP_05 no se cita como dimensionamiento de la red.

Incentivos, tokens, DOIN-como-mercado: no.

---

## 8. Preguntas para tu dictamen

1. ¿Aceptas 5A–5B como interruptor, con nulo = “esta serie no tiene memoria de fuente medible”?  
2. ¿Los diagnósticos de entropía pueden usar alfabeto provisional, sin esperar la meseta de 4B?  
3. ¿BCT-X queda como baseline de papel/código ajeno, no como plugin nuestro en el primer banco?  
4. ¿IB/CIB y MDL se quedan fuera del banco mínimo, como pedí?  
5. ¿Hay campaña que este laboratorio no deba ni oler?

---

## 9. Lo que yo no haré

No implemento. No entreno CIB-MTSF. No reescribo CTW. No abro STEP_06. No lanzo GPU. No mezclo esto en selección de representaciones. No nombro incentivos. No bautizo al agente de comunicaciones.

Si la disposición es ACCEPT GATE 5A–5B, el primer commit sería: estimador de \(H_0\)/\(H_k\) sobre un alfabeto provisional, validado en sintético, `CUDA_VISIBLE_DEVICES=""`. Sin Huffman. Sin ramas.

— Retsu
