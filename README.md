
<div align="center">

# 🎬 WORD2VEC
### Análisis de reseñas de películas mediante procesamiento de lenguaje natural

**Proyecto de Big Data · Procesamiento de Lenguaje Natural (NLP)**

**Mariel Rodríguez Arguello**  
Colegio Universitario de Cartago (CUC)

---

**12.368 reseñas** · **BoW y TF-IDF** · **CBOW y Skip-Gram** · **Python y Jupyter Notebook**

*Un proyecto reproducible para convertir opiniones escritas en representaciones numéricas y explorar relaciones semánticas.*

</div>

---

## 📖 Presentación

Este proyecto procesa **12.368 reseñas de películas** procedentes del archivo `resenas_mongodb.csv`. El análisis utiliza el texto en español (`texto_es`) para construir representaciones numéricas con **Bag of Words (BoW)**, **TF-IDF** y **Word2Vec**, mediante sus dos arquitecturas: **CBOW** y **Skip-Gram**.

Además de entrenar modelos de palabras, se calcula un **vector promedio por reseña**, de manera que cada comentario tenga su propia representación vectorial. Los resultados se entregan listos para consultar, sin necesidad de entrenar nuevamente.

## 🎯 Objetivos

- Preparar y representar las reseñas en español para su análisis computacional.
- Construir matrices dispersas BoW y TF-IDF.
- Entrenar modelos Word2Vec con CBOW y Skip-Gram.
- Obtener vectores de 100 dimensiones por reseña para cada arquitectura.
- Mantener la correspondencia entre los vectores, las matrices y `id_resena`.
- Facilitar la ejecución del proyecto en diferentes computadoras.

## 🧠 Técnicas utilizadas

| Técnica | ¿Qué representa? | Resultado |
|:--|:--|:--|
| **BoW** | Frecuencia de términos en las reseñas | `bow.npz` |
| **TF-IDF** | Importancia relativa de los términos | `tfidf.npz` |
| **Word2Vec CBOW** | Aprende representaciones de palabras a partir de su contexto | `modelo_cbow.model` y `vectores_cbow.csv.gz` |
| **Word2Vec Skip-Gram** | Aprende representaciones prediciendo palabras del contexto | `modelo_skip_gram.model` y `vectores_skip_gram.csv.gz` |

> **Nota metodológica:** los archivos `vectores_*.csv.gz` contienen un vector **promedio de palabras por reseña**, no una lista de vectores individuales para cada palabra. Los modelos `.model` conservan las representaciones aprendidas de las palabras.

## 🗂️ Organización del proyecto

```text
Proyecto_Word2Vec/
│
├── README.md
├── requirements.txt
├── embeddings_w2v.py
├── 03_word2vec_analisis.ipynb
│
├── data/
│   └── resenas_mongodb.csv
│
└── resultados/
    ├── bow.npz
    ├── tfidf.npz
    ├── ids_resenas.csv
    ├── vocabulario.csv
    ├── vectores_cbow.csv.gz
    ├── vectores_skip_gram.csv.gz
    ├── modelo_cbow.model
    └── modelo_skip_gram.model
```

## 💻 Requisitos del sistema

| Componente | Recomendación |
|:--|:--|
| Sistema operativo | Windows, macOS o Linux |
| Python | Versión 3.10, 3.11 o 3.12, de 64 bits |
| Internet | Necesario para instalar dependencias; no para procesar las reseñas |
| Software opcional | JupyterLab para abrir el notebook |

El proyecto usa **rutas relativas** y no depende de carpetas personales ni de letras de unidad específicas.

## 🚀 Instalación y ejecución

### 1. Preparar la carpeta

Descargar y **extraer completamente** el archivo ZIP. No ejecutar el notebook directamente desde el archivo comprimido.

Abrir una terminal en la carpeta principal, donde se encuentra `requirements.txt`.

### 2. Crear y activar un entorno virtual

**Windows (PowerShell o Símbolo del sistema):**

```powershell
py -3 -m venv .venv
.venv\Scripts\activate
```

**macOS o Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar las dependencias

```bash
python -m pip install -r requirements.txt
```

### 4. Ejecutar el procesamiento completo (opcional)

```bash
python embeddings_w2v.py
```

Este paso vuelve a generar los resultados y puede tardar varios minutos según el equipo. **Si solo se desea revisar el análisis, no hace falta ejecutarlo:** los resultados están incluidos en `resultados/`.

### 5. Abrir el análisis en Jupyter

```bash
python -m jupyter lab
```

Abrir el archivo `03_word2vec_analisis.ipynb` y ejecutar las celdas en orden. Para visualizar resultados existentes, conservar la estructura de carpetas original.

## 📦 Archivos de resultados

| Archivo | Descripción |
|:--|:--|
| `ids_resenas.csv` | Identificadores de las 12.368 reseñas, en el orden usado por las matrices |
| `bow.npz` | Matriz dispersa de frecuencias de palabras |
| `tfidf.npz` | Matriz dispersa de ponderaciones TF-IDF |
| `vocabulario.csv` | Términos correspondientes a las columnas de las matrices |
| `vectores_cbow.csv.gz` | Identificador y 100 dimensiones por reseña, obtenidas con CBOW |
| `vectores_skip_gram.csv.gz` | Identificador y 100 dimensiones por reseña, obtenidas con Skip-Gram |
| `modelo_cbow.model` | Modelo Word2Vec CBOW entrenado |
| `modelo_skip_gram.model` | Modelo Word2Vec Skip-Gram entrenado |

**Importante:** `.csv.gz` es un archivo CSV comprimido con **gzip**; no es un ZIP. Pandas puede abrirlo directamente sin descomprimirlo manualmente.

## ✅ Comprobación de integridad

Ejecutar el siguiente código desde la carpeta principal del proyecto:

```python
import pandas as pd
from scipy.sparse import load_npz

ids = pd.read_csv("resultados/ids_resenas.csv")
cbow = pd.read_csv("resultados/vectores_cbow.csv.gz")
skip_gram = pd.read_csv("resultados/vectores_skip_gram.csv.gz")

assert len(ids) == len(cbow) == len(skip_gram) == 12368
assert ids["id_resena"].equals(cbow["id_resena"])
assert ids["id_resena"].equals(skip_gram["id_resena"])
assert load_npz("resultados/bow.npz").shape[0] == len(ids)
assert load_npz("resultados/tfidf.npz").shape[0] == len(ids)

print("✓ Las 12.368 reseñas están correctamente alineadas.")
```

Esta verificación comprueba que el número y orden de las reseñas coinciden en las matrices y en los vectores.

## 🔍 Consideraciones técnicas

- **Idioma de análisis:** español, a partir de `texto_es`.
- **Unidad de análisis:** una reseña identificada mediante `id_resena`.
- **Dimensionalidad Word2Vec:** 100 componentes por vector promedio.
- **Reproducibilidad:** se emplean semilla fija y un trabajador durante el entrenamiento; pueden existir pequeñas diferencias numéricas entre plataformas o versiones de bibliotecas.
- **Portabilidad:** para trabajar en otra computadora, trasladar la carpeta completa e instalar las dependencias del archivo `requirements.txt`.

## 🏁 Conclusión

El proyecto permite comparar distintas formas de representar reseñas de películas: desde enfoques basados en frecuencia (**BoW** y **TF-IDF**) hasta representaciones distribuidas que incorporan información del contexto lingüístico (**Word2Vec**). La entrega conserva los identificadores y ofrece resultados reutilizables para análisis y visualización posteriores.

---

<div align="center">

**Proyecto académico · Big Data**  
**Mariel Rodríguez Arguello**

*Procesamiento de Lenguaje Natural aplicado a reseñas cinematográficas*

</div>
