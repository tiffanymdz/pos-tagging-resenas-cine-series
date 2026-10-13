# Análisis Morfosintáctico de Reseñas de Cine y Series en Español mediante POS Tagging

Proyecto de análisis de datos y procesamiento de lenguaje natural orientado al estudio de la estructura gramatical de reseñas de cine en español, con NLTK y spaCy.

## Institución

Colegio Universitario de Cartago — Costa Rica

**Curso:** Proyecto 1 · **Profesor:** Osvaldo González Chaves

## Integrantes

 | Integrante | 
|---|
 | [Felipe Montenegro Artavia] | 
| [Tiffany Mendez Quirós] |
[Mariel Rodriguez Arguello ] | 

## Descripción

Las reseñas de cine son uno de los géneros textuales donde la opinión se expresa con mayor claridad: quien escribe evalúa, compara, recomienda o advierte, y cada una de esas acciones deja huellas gramaticales (adjetivos valorativos, adverbios de intensidad, negaciones, marcas de primera y segunda persona).

Este proyecto construye un corpus de reseñas en español, lo etiqueta con **spaCy** (`es_core_news_md`) y con un etiquetador de **NLTK** entrenado con el corpus `cess_esp`, y analiza su estructura morfosintáctica en tres ejes:

- **Eje A:** comparación entre géneros cinematográficos.
- **Eje B:** relación entre la estructura gramatical y la valoración de la reseña (emocionalidad gramatical).
- **Eje C:** evolución temporal de la forma en que se escriben las reseñas.

Además se compara el rendimiento de las dos herramientas de etiquetado y se analiza el efecto de la traducción automática sobre el etiquetado, que es un hallazgo más del proyecto.

## Fuente de datos

El corpus se obtiene exclusivamente de un dataset público de Kaggle: **IMDB Spoiler Dataset** (`rmisra/imdb-spoiler-dataset`), en inglés. Se usan sus dos archivos:

- `IMDB_reviews.json`: texto de la reseña, calificación de 1 a 10, fecha de la reseña y marca de spoiler.
- `IMDB_movie_details.json`: géneros y fecha de estreno de cada película, que se une a las reseñas por `movie_id`.

Como el dataset está en inglés, las reseñas muestreadas se traducen al español con un modelo local de traducción.

Los datos se almacenan en:

- `data/raw/`: dataset original de Kaggle (no se sube al repositorio, está en `.gitignore`; se descarga desde el notebook con `kagglehub`).
- `data/processed/`: muestra traducida y corpus limpio.
- `data/results/`: resultados del etiquetado POS.

> Antes de elegir este dataset se descartaron otros dos: `Movies_Reviews_modified_version1.csv` no trae ninguna fecha, y `movieReviews.csv` no trae género ni título y concentra el 86 % de sus reseñas en 2019. Ninguno permitía los ejes A y C.

## Flujo del proyecto

```
Dataset de Kaggle (inglés)
       ↓
Carga, filtro por longitud y muestreo estratificado
       ↓
Traducción automática al español (MarianMT)
       ↓
Limpieza del texto
       ↓
POS Tagging con spaCy y con NLTK
       ↓
Cálculo de métricas morfosintácticas
       ↓
Análisis por ejes (A, B y C) y comparación NLTK vs spaCy
       ↓
Visualización y dashboard interactivo
```

| Etapa | Qué se hace | Salida |
|---|---|---|
| 1. Recolección y traducción | Carga del dataset, muestreo y traducción | `resenas_es.csv` |
| 2. Limpieza | Quitar ruido, duplicados y traducciones fallidas | `resenas_limpias.csv` |
| 3. POS Tagging | Etiquetado con spaCy y con el etiquetador NLTK entrenado | `pos_analysis.csv` |
| 4. Análisis | Métricas, comparación NLTK vs spaCy y respuesta a las preguntas | Tablas de resultados |
| 5. Visualización | Gráficos comparativos y dashboard interactivo | `outputs/figures/`, `dashboard.html` |

## Corpus final

| Dato | Valor |
|---|---|
| Reseñas muestreadas y traducidas | 6.400 |
| Reseñas tras la limpieza (`resenas_limpias.csv`) | **6.283** |
| Películas distintas | 1.158 |
| Géneros | Drama, Comedia, Acción, Terror y Animación (~1.257 reseñas cada uno) |
| Periodos | Antes de 2010 (3.151) y 2010 en adelante (3.132) |
| Rango de fechas de las reseñas | 1998 a 2018 |
| Longitud de las reseñas originales | Entre 30 y 400 palabras |
| Valoración | Positiva 4.175 · Negativa 1.203 · Intermedia 905 |

Columnas de `resenas_limpias.csv`: `id`, `movie_id`, `genero`, `periodo`, `anio_review`, `fecha`, `calificacion`, `valoracion`, `is_spoiler`, `texto_en`, `texto_es`.

## Decisiones de muestreo

1. **Filtro de longitud:** se conservan las reseñas de 30 a 400 palabras (texto original en inglés), antes de traducir.
2. **Películas con varios géneros:** se usa el **primer género de la lista**. Cada reseña queda en un solo género y no se repite.
3. **Géneros elegidos:** Drama, Comedy, Action, Horror y Animation, que son los de las preguntas del eje A.
4. **Periodos:** antes de 2010 y de 2010 en adelante, según la **fecha de la reseña** (no la de estreno). Con este corte quedan cantidades parejas en los dos periodos.
5. **Estratificación:** 5 géneros × 2 periodos = 10 celdas de **640 reseñas** cada una (6.400 en total). La muestra inicial es mayor que el mínimo de 3.000 para tener margen en la limpieza.
6. **Tope por película:** dentro de cada celda, el número de reseñas que puede aportar una película se calcula de forma dinámica (el menor tope con el que todavía se alcanza la cuota). Así se evita que pocas películas dominen un género. Máximo de reseñas de una misma película por género: Acción 5, Comedia 5, Drama 7, Animación 20 y Terror 34 (estos dos últimos géneros tienen pocas películas: 84 y 46).
7. **Semilla fija:** 42, para que el muestreo sea reproducible.
8. **Paquetes:** la muestra se divide en 3 paquetes parejos (2.140, 2.130 y 2.130 reseñas), con la misma mezcla de géneros y periodos, para repartir la traducción entre los integrantes.

## Traducción automática

- **Modelo:** `Helsinki-NLP/opus-mt-en-es` (MarianMT), local y gratuito.
- **Entorno:** Google Colab con GPU T4 (precisión fp16). Cada paquete tardó unos 10 minutos.
- **Procedimiento:** cada reseña se divide en oraciones (`sent_tokenize`), las oraciones se traducen por lotes de 64 ordenadas por largo, y se vuelven a unir. Se hace así porque el modelo trunca las entradas largas.
- **Checkpoints:** el avance se guarda cada 100 reseñas, para poder reanudar si Colab se desconecta.
- **Se conserva el original:** `resenas_limpias.csv` trae `texto_en` junto a `texto_es`.
- **No se corrigen las traducciones.** Se analizan tal como las produce el modelo.

Ejemplos de efectos de la traducción observados (para la discusión del efecto sobre el etiquetado):

| Original | Traducción | Tipo de error |
|---|---|---|
| "But get a call to go..." | "Pero haz una llamada..." | "get" interpretado como imperativo |
| "all pushing 70" | "todos empujando 70" | Expresión idiomática traducida literalmente |
| "dredge up" | "dragar" | Sentido literal en lugar del figurado |
| "Up" (título de la película) | "Arriba" | Título traducido |
| "Now, I am a Lynch fan" | "Ahora, soy un fan de Lynch" | Calco del inglés (artículo y pronombre explícitos) |

## Limpieza de datos

Reglas aplicadas al texto en español (son decisiones del equipo; el PDF solo pide quitar HTML, vacías, duplicadas y traducciones fallidas):

- Se quitan etiquetas HTML, líneas decorativas (`*****`, `-----`), signos repetidos (`!!!!!!!` pasa a `!!!`), tiras de símbolos sueltos que genera el traductor (`* * * *`) y espacios repetidos. También `1⁄2` pasa a `1/2`.
- **No** se pasa a minúsculas, **no** se quitan tildes y **no** se elimina la puntuación normal, porque spaCy los necesita para etiquetar bien.

Reseñas descartadas: **117 de 6.400**, guardadas en `data/processed/descartadas.csv` con su motivo.

| Motivo | Reseñas |
|---|---|
| Traducción cortada (ver abajo) | 113 |
| Palabra repetida muchas veces (bucle del traductor) | 2 |
| Codificación rota (`actuaciÃ3n`) | 1 |
| Traducción incompleta (menos del 70 % de las palabras del original) | 1 |
| Vacías, sin traducir, en inglés o duplicadas | 0 |

**Traducciones cortadas.** El modelo de traducción lee como máximo 256 tokens por oración y descarta el resto. En inglés es frecuente que el punto vaya pegado a la oración siguiente (`film.It was...`); el divisor de oraciones no corta ahí y queda una oración larguísima cuya parte final se pierde. Para detectarlas se estima cuántos tokens ocupa la oración más larga del original (cada signo de puntuación cuenta 1 y cada palabra, uno por cada ~4,5 letras) y se elimina la reseña si pasa de 240 (`traduccion_cortada`). El criterio se probó con 21 reseñas cortadas revisadas a mano y las detectó todas. Es **aproximado y conservador**: puede eliminar algunas reseñas completas y, a la inversa, podría quedar un número pequeño de traducciones con pérdida en el final de una oración.

## POS Tagging con spaCy y NLTK

El etiquetado se aplica sobre la columna `texto_es` de `resenas_limpias.csv` y se guarda en `data/results/pos_analysis.csv`, con **un token por fila**.

- **spaCy:** modelo `es_core_news_md`, con procesamiento por lotes (`nlp.pipe`) y el reconocimiento de entidades desactivado para ganar velocidad (el parser se conserva, porque se necesita para segmentar oraciones). Entrega, por token, la etiqueta UPOS (`upos_spacy`), el lema (`lema`) y los rasgos morfológicos (`morf`).
- **NLTK:** el etiquetador preentrenado de NLTK solo funciona para inglés, por lo que se entrena uno para español con el corpus anotado `cess_esp`, combinando etiquetadores de n-gramas con respaldo (unigrama y bigrama). Sus etiquetas EAGLES se convierten a UPOS (`upos_nltk`) para poder compararlo con spaCy.
- **Mismos tokens:** el etiquetador de NLTK se aplica sobre los tokens que produjo spaCy, de modo que ambas herramientas etiquetan exactamente las mismas unidades.
- **Exactitud del etiquetador NLTK:** unigrama [COMPLETAR] · bigrama con respaldo [COMPLETAR].

Columnas de `pos_analysis.csv`: `id`, `oracion`, `token`, `lema`, `upos_spacy`, `morf`, `upos_nltk`.

No se pasa a minúsculas ni se quita la puntuación antes de etiquetar, porque spaCy los necesita para funcionar correctamente.

## Métricas morfosintácticas

Todas se calculan **excluyendo la puntuación** y, cuando son frecuencias, normalizadas **por cada 100 tokens**.

| Métrica | Cálculo | Qué indica |
|---|---|---|
| Densidad léxica | (NOUN + VERB + ADJ + ADV) / total de tokens | Palabras con contenido frente a palabras funcionales |
| Ratio verbos/sustantivos | VERB / NOUN | Estilo narrativo (alto) frente a descriptivo (bajo) |
| Índice de adjetivación | ADJ por cada 100 tokens | Carga valorativa y descriptiva |
| Índice de negación | Negaciones por cada 100 tokens | Tendencia a rechazar o contradecir |
| Uso de pronombres por persona | PRON con `Person=1`, `2` o `3` por cada 100 tokens | Experiencia propia, interpelación al lector o descripción |
| Diversidad léxica (TTR) | Lemas distintos / tokens, sobre bloques de tamaño fijo | Riqueza de vocabulario |
| Longitud media de oración | Tokens por oración (segmentación de spaCy) | Complejidad sintáctica aproximada |

## Resultados por pregunta de investigación

Cada pregunta se responde con una tabla, al menos un gráfico y un párrafo de interpretación (en los notebooks 03 y 04). Aquí se resumen los hallazgos.

### Eje A: comparación entre géneros

*[COMPLETAR — Persona 3: dos preguntas del eje A, con su hallazgo principal.]*

### Eje B: emocionalidad gramatical

*[COMPLETAR — Persona 2: dos preguntas del eje B, con su hallazgo principal.]*

### Eje C: evolución temporal

Se responden dos preguntas del eje (la 3 y la 4). Se compararon las reseñas escritas **antes de 2010** (3.151) y **de 2010 en adelante** (3.132). Las métricas se calcularon por reseña a partir de `pos_analysis.csv` (etiquetas de spaCy, `es_core_news_md`), excluyendo la puntuación y normalizando por cada 100 tokens, y los periodos se compararon con la prueba de Mann-Whitney. Tabla completa en `outputs/tabla_eje_c.csv`.

| Métrica | Antes de 2010 | 2010 en adelante | Diferencia | p |
|---|---|---|---|---|
| Pronombres de 1.ª persona por 100 tokens | 0,79 | 0,75 | -0,04 | 0,102 |
| Verbos en 1.ª persona por 100 tokens | 1,77 | 1,49 | -0,28 | < 0,001 |
| Palabras por reseña | 177,6 | 202,8 | +25,2 | < 0,001 |

**P3. ¿Las reseñas recientes usan más la primera persona que las antiguas?** (`outputs/figures/eje_c_p3_primera_persona.png`)
No. Con la definición del PDF (pronombres con `Person=1`), la diferencia no es significativa (0,79 frente a 0,75; p = 0,102), y su dirección cambia según el género (Animación sube 0,06; Acción, Comedia y Drama bajan; Terror queda igual). Como en español el sujeto suele omitirse, se midió además la primera persona en los **verbos** ("me encantó", "vi"), y ahí sí hay una baja significativa (1,77 frente a 1,49 por 100 tokens; p < 0,001), presente en los cinco géneros, aunque de tamaño pequeño. Los datos no apoyan que las reseñas recientes usen más la primera persona; si algo, la usan algo menos en los verbos. Limitación: los modelos de traducción tienden a escribir el pronombre explícito ("yo") porque el inglés lo exige, lo que puede inflar el conteo de pronombres.

**P4. ¿Cambia con los años la longitud media de las reseñas?** (`outputs/figures/eje_c_p4_longitud_resenas.png`)
Sí: las reseñas de 2010 en adelante son unas 25 palabras más largas (203 frente a 178; mediana 182,5 frente a 157), y la diferencia se repite en los cinco géneros (entre +18 palabras en Drama y +36 en Comedia). Mirando año por año, el aumento no es continuo: sube desde unas 115 palabras en 1998 hasta unas 219 en 2009, se mantiene entre 200 y 220 hasta 2016 y baja a unas 180 en 2017. Limitaciones: la longitud se mide sobre el texto traducido (suele ser algo más largo que el original); el filtro de 30 a 400 palabras excluye reseñas muy cortas y muy largas; los años 1998 y 2018 tienen muy pocas reseñas.

**Conclusión del eje.** Las reseñas recientes son más largas, pero no usan más la primera persona: la diferencia en pronombres no es significativa y en verbos es una baja pequeña. Los resultados describen diferencias entre dos periodos de **fecha de reseña**, con un corte (2010) decidido por el equipo, y no permiten atribuir causas.

## Comparación NLTK vs spaCy

*[COMPLETAR — Persona 3.]* Esta sección debe incluir:

- Porcentaje de concordancia global, excluyendo la puntuación.
- Concordancia por categoría UPOS.
- Matriz de confusión entre ambas herramientas (mapa de calor).
- Al menos diez ejemplos de discrepancia, clasificados según su causa probable.
- Conclusión sobre cuál herramienta resulta más confiable para este corpus, respaldada por la revisión manual de al menos 50 tokens en discrepancia.
- Discusión del efecto de la traducción automática sobre el etiquetado.

## Visualización

Las visualizaciones se guardan como PNG en `outputs/figures/`:

- Distribución de categorías POS por género.
- Comparación de métricas derivadas entre géneros.
- Relación entre calificación y métricas (índice de adjetivación, índice de negación).
- Evolución de métricas entre periodos (Eje C: `eje_c_p3_primera_persona.png` y `eje_c_p4_longitud_resenas.png`).
- Matriz de confusión NLTK vs spaCy.
- Adjetivos más característicos por valoración.

Además se construye un **dashboard interactivo con Plotly Dash**, exportado como `outputs/dashboard.html`, que permite filtrar al menos por género y por valoración. *[COMPLETAR — estado del dashboard.]*

## Estructura del proyecto

```
proyecto-pos-tagging-resenas/
│
├── data/
│   ├── raw/                              # Dataset original de Kaggle (no se sube)
│   ├── processed/
│   │   ├── resenas_es.csv                # Muestra traducida (6.400 reseñas)
│   │   ├── resenas_limpias.csv           # Corpus final (6.283 reseñas)
│   │   └── descartadas.csv               # Reseñas eliminadas y su motivo
│   └── results/
│       └── pos_analysis.csv              # Un token por fila, con etiquetas spaCy y NLTK
│
├── notebooks/
│   ├── 01_carga_traduccion_limpieza.ipynb
│   ├── 02_pos_tagging.ipynb
│   ├── 03_analisis_metricas.ipynb
│   ├── 04_visualizaciones.ipynb
│   └── eje_c_evolucion_temporal.ipynb    # Eje C (tablas y gráficos)
│
├── src/
│   ├── pos_tagging.py                    # Traducción, limpieza y etiquetado
│   ├── metrics.py                        # Cálculo de métricas
│   └── visualizations.py                 # Funciones de gráficos
│
├── outputs/
│   ├── figures/                          # Gráficos PNG
│   ├── tablas_eje_c/                     # Tablas del Eje C en CSV
│   └── dashboard.html
│
├── README.md
├── requirements.txt
├── USO_DE_IA.md
└── .gitignore
```

## Tecnologías utilizadas

- Python
- pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Plotly y Dash
- spaCy (`es_core_news_md`)
- NLTK (corpus `cess_esp`)
- Transformers y PyTorch (modelo de traducción MarianMT)
- kagglehub
- Jupyter Notebook y Google Colab
- GitHub

## Instalación

Se recomienda utilizar Python 3.10 o superior.

Clonar el repositorio desde GitHub y acceder a la carpeta del proyecto.

Crear un entorno virtual:

```
python -m venv .venv
```

Activar el entorno en PowerShell:

```
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```
python -m pip install -r requirements.txt
```

Descargar el modelo de spaCy y los datos de NLTK:

```
python -m spacy download es_core_news_md
python -c "import nltk; nltk.download('cess_esp'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Ejecución

Los notebooks se ejecutan en este orden:

1. **`01_carga_traduccion_limpieza.ipynb`:** se ejecuta en **Google Colab con GPU** (Entorno de ejecución → Cambiar tipo → T4 GPU). Descarga el dataset, muestrea, traduce, une y limpia. La traducción guarda su avance cada 100 reseñas y se reanuda si Colab se desconecta, así que no repite lo ya traducido. Los CSV se guardan en Google Drive y deben copiarse a `data/processed/`.
2. **`02_pos_tagging.ipynb`:** etiquetado con spaCy y NLTK sobre `resenas_limpias.csv`; genera `data/results/pos_analysis.csv`.
3. **`03_analisis_metricas.ipynb`:** métricas y respuestas a las preguntas de investigación.
4. **`04_visualizaciones.ipynb`:** gráficos y dashboard.

El Eje C se encuentra en `notebooks/eje_c_evolucion_temporal.ipynb`.

Para revisar el proyecto no hace falta repetir la traducción: la muestra traducida y los resultados están en el repositorio.

Los CSV se guardan en UTF-8 con BOM para que Excel muestre bien las tildes. Para trabajar con ellos conviene usar `pandas`, no Excel.

## Estado actual

**Completado**

- Corpus de 6.283 reseñas en español, traducido y limpio.
- Etiquetado POS con spaCy y con NLTK (`pos_analysis.csv`).
- Eje C: preguntas 3 y 4, con tablas, gráficos e interpretación.

**En desarrollo** *[COMPLETAR según el avance de cada integrante]*

- Eje A (comparación entre géneros).
- Eje B (emocionalidad gramatical).
- Comparación NLTK vs spaCy.
- Dashboard interactivo.

## Limitaciones

- La traducción automática puede introducir calcos del inglés (pronombres de sujeto explícitos, construcciones literales) que alteran las distribuciones POS. No se corrigieron, y se discute su efecto en el análisis.
- La valoración está inclinada hacia lo positivo (alrededor de dos tercios de las reseñas), porque en el dataset original la calificación 10 es la más frecuente. Para el eje B conviene comparar proporciones y no conteos.
- Animación y Terror provienen de pocas películas (84 y 46), por lo que algunas películas aportan muchas reseñas.
- Los periodos usan la fecha de la reseña y un corte propio (2010).
- La detección de traducciones cortadas es aproximada: puede haber eliminado algunas reseñas completas y podría quedar un número pequeño de traducciones con pérdida en el final de una oración.

## Mejoras futuras

- Volver a traducir con oraciones más cortas las reseñas eliminadas por traducción cortada, para recuperarlas.
- Comparar los resultados con reseñas escritas originalmente en español.
- Incorporar un dataset de reseñas de series y comparar películas frente a series.
- Analizar más de dos periodos de tiempo.
- Ampliar la revisión manual del etiquetado para estimar la exactitud real de cada herramienta.

## Uso de inteligencia artificial

El uso de herramientas de IA se documenta en `USO_DE_IA.md`. El modelo de traducción automática forma parte del pipeline del proyecto y se documenta en este README.

## Uso responsable

Los resultados describen patrones en un corpus de reseñas traducidas automáticamente y no permiten atribuir causas. Deben interpretarse como un análisis académico y no como conclusiones generales sobre cómo se escriben las reseñas de cine en español.

## Uso académico

Proyecto desarrollado con fines académicos para aplicar conceptos de procesamiento de lenguaje natural, etiquetado morfosintáctico (POS Tagging) y análisis de datos.
