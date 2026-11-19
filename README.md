
<div align="center">

# 🎬 ANÁLISIS DE RESEÑAS DE PELÍCULAS
## Procesamiento de lenguaje natural con BoW, TF-IDF y Word2Vec

**Proyecto académico · Diplomado en Big Data**  
**Integrantes:**  
**Mariel Rodríguez** · **Tiffany Méndez** · **Felipe Artavia**  
**Noviembre de 2026**

---

**12.368 reseñas** · **CBOW y Skip-Gram** · **Python / Jupyter Notebook**

</div>

## 📖 Presentación

Este proyecto analiza reseñas cinematográficas mediante técnicas de procesamiento de lenguaje natural (PLN). Se estudia la distribución de los datos y se comparan representaciones de texto basadas en **Bag of Words (BoW)**, **TF-IDF** y **Word2Vec**, utilizando sus variantes **CBOW** y **Skip-Gram**.

El notebook `07_word2vec_analisis(1).ipynb` contiene el análisis y las visualizaciones. El módulo `embeddings_w2v.py` permite generar nuevamente los archivos procesados y entrenar modelos cuando sea necesario.

## 🎯 Objetivos

- Explorar las reseñas por género, polaridad, calificación y longitud del texto.
- Representar los textos mediante BoW y TF-IDF.
- Obtener vectores de reseñas con Word2Vec, mediante CBOW y Skip-Gram.
- Visualizar las representaciones y discutir sus diferencias y limitaciones.

## 🗂️ Estructura del proyecto

```text
Proyecto_Word2Vec/
├── README.md
├── requirements.txt
├── 07_word2vec_analisis(1).ipynb
├── embeddings_w2v.py
├── data/
│   └── resenas_mongodb.csv
└── resultados/
    ├── ids_resenas.csv
    ├── bow.npz
    ├── tfidf.npz
    ├── vocabulario.csv
    ├── vectores_cbow.csv.gz
    ├── vectores_skip_gram.csv.gz
    ├── modelo_cbow.model
    └── modelo_skip_gram.model
```

> **Nota:** los archivos de `resultados/` y `data/` deben estar presentes para ejecutar el notebook directamente. Si no se incluyen, se pueden regenerar con el módulo Python y el CSV original. Los nombres indicados corresponden a los archivos que espera el notebook o genera el módulo.

## 💻 Requisitos

- Python **3.10 o superior**.
- Jupyter Notebook o JupyterLab (también se puede usar PyCharm o VS Code con soporte para notebooks).
- Bibliotecas: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `gensim`, `notebook`.

Archivo sugerido `requirements.txt`:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
gensim
notebook
```

## 🚀 Instalación y ejecución

**1.** Descomprimir el proyecto y abrir una terminal **dentro de la carpeta `Proyecto_Word2Vec`**.

**2.** Crear y activar un entorno virtual:

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m notebook
```

**macOS / Linux:**

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m notebook
```

**3.** En Jupyter, abrir `07_word2vec_analisis(1).ipynb` y ejecutar las celdas en orden.

> **Importante:** el notebook utiliza `Path.cwd()` para localizar `data/` y `resultados/`. Por eso, Jupyter debe iniciarse desde la carpeta principal del proyecto. Si aparece el mensaje «Abra Jupyter desde la carpeta del proyecto», verificar la ubicación de la terminal y del directorio de trabajo del notebook.

## 🔄 Regeneración de resultados (opcional)

Si se dispone de `data/resenas_mongodb.csv` pero faltan los resultados, ejecutar **desde la carpeta principal**:

```bash
python embeddings_w2v.py
```

En Windows también se puede usar:

```powershell
.\.venv\Scripts\python.exe embeddings_w2v.py
```

El módulo valida la columna `id_resena`, procesa `texto_es`, genera las matrices BoW y TF-IDF, entrena los modelos Word2Vec y exporta los vectores por reseña. El entrenamiento puede tardar según el equipo y los valores pueden diferir ligeramente si se vuelven a entrenar los modelos. **No es necesario reentrenar** si los resultados ya están disponibles.

## 📊 Contenido del análisis

| Componente | Descripción |
|---|---|
| Exploración inicial | Géneros, polaridad y calificaciones de las reseñas |
| Longitud de los textos | Distribución de palabras y caracteres |
| BoW | Frecuencia de términos sin información de orden |
| TF-IDF | Importancia relativa de los términos en el corpus |
| CBOW | Representaciones aprendidas a partir del contexto |
| Skip-Gram | Representaciones aprendidas al predecir palabras del contexto |
| PCA | Proyección bidimensional para visualizar vectores de reseñas |

## 🔍 Interpretación de resultados

Los gráficos permiten observar características generales del corpus y visualizar las representaciones obtenidas por cada modelo. Una proyección PCA muestra una reducción a dos dimensiones: **la cercanía visual de los puntos no demuestra por sí sola calidad semántica superior**. Para evaluar las diferencias entre CBOW y Skip-Gram deben considerarse también las similitudes y los resultados de las consultas realizadas.

## 🧩 Solución de problemas frecuentes

- **`FileNotFoundError`:** revisar que `data/resenas_mongodb.csv` y la carpeta `resultados/` estén en las rutas indicadas y que Jupyter se haya abierto desde la raíz.
- **`ModuleNotFoundError`:** instalar los paquetes en el mismo entorno de Python seleccionado como kernel del notebook.
- **Error al leer modelos o vectores:** verificar que los archivos estén completos y que no se haya cambiado su nombre.
- **Resultados distintos tras entrenar:** es posible obtener pequeñas variaciones entre ejecuciones o entornos; conservar los archivos originales si se necesita reproducir el análisis presentado.

## ✅ Conclusión

El proyecto integra técnicas clásicas de vectorización y modelos de embeddings para explorar reseñas de películas. La comparación entre BoW, TF-IDF, CBOW y Skip-Gram permite reconocer las diferencias entre representar frecuencias de palabras y aprender relaciones a partir del contexto. Su organización separa el análisis en Jupyter del procesamiento reutilizable en Python.

---

<div align="center">

**Proyecto Word2Vec · Big Data · 2026**  
*Documentación para instalación, ejecución y consulta académica.*

</div>
