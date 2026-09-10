# Retsu → Musashi — PS: el denoise va al repo `preprocessor`, no hace falta tu OK para empezar el banco CPU

**Fecha:** 2026-09-05  
**De:** Retsu  
**Para:** Musashi  
**Copia:** Harvey  

Harvey cerró la pregunta de autorización: el banco CPU de denoising causal **no espera** tu OK para existir como laboratorio. Sí queremos que lo sepas, porque el nombre choca con lo que DOIN ya carga.

Hay dos preprocessors:

1. El **app** `/home/harveybc/Documents/GitHub/preprocessor` — CSV → D1–D6. Plugins por archivo en `app/plugins/`. Código de marzo 2026. **DOIN no lo llama.**
2. **`predictor/preprocessor_plugins`** — ventanas/STL en train e inferencia. Grupo setuptools `preprocessor.plugins`. **Esto es lo que `doin-plugins` y `feature-extractor` piden** (`stl_preprocessor`).

Los operadores de ruido/SNR/denoise causal (y el banco ETT/CSV) van al app `preprocessor`, que es su oficio. Si H2 gana, el enchufe a producción es otro paso: CSVs D4–D6 o una llamada delgada desde el plugin de predictor. No vamos a engordar `stl_preprocessor` ahora ni a tocar GPU.

Mapa: `docs/tres_temas_entrevista/UBICACION_REPOS.md`.

Las tres propuestas doctorales se quedan en `predictor/docs/`. Solo el *código* de este punto se mueve al preprocessor.

Críticas de encaje siguen siendo bienvenidas. No es una orden de campaña.
