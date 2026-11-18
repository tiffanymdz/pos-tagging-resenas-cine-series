
# Word2Vec en reseñas de películas

Proyecto reproducible con **12.368 reseñas** del archivo suministrado, en español (`texto_es`). Incluye matrices BoW y TF-IDF, modelos CBOW y Skip-Gram, y un vector promedio de palabras por reseña para cada modelo.

## Requisitos

- Windows, macOS o Linux
- Python 3.10 a 3.12 recomendado (64 bits)
- Conexión a internet solo para instalar dependencias (no para procesar los datos)

## Ejecución en cualquier computadora

1. Descomprimir **todo** el ZIP en una carpeta (no abrir el notebook desde dentro del ZIP).
2. Abrir una terminal en esa carpeta.
3. Crear un entorno virtual:
   - Windows: `py -3 -m venv .venv` y `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` y `source .venv/bin/activate`
4. Instalar: `python -m pip install -r requirements.txt`
5. Ejecutar: `python embeddings_w2v.py`
6. Opcional: `python -m jupyter lab` y abrir `03_word2vec_analisis.ipynb`.

La generación completa puede tardar varios minutos, según el equipo. Los resultados ya están incluidos en `resultados/`, así que se pueden analizar sin volver a entrenar.

## Contenido

- `data/resenas_mongodb.csv`: fuente original.
- `embeddings_w2v.py`: pipeline completo reproducible, rutas relativas a su propia ubicación.
- `resultados/bow.npz`, `resultados/tfidf.npz`: matrices dispersas; **filas en el orden de `ids_resenas.csv`**.
- `resultados/ids_resenas.csv`: identificador de cada una de las 12.368 filas.
- `resultados/vocabulario.csv`: términos en el orden de las columnas de BoW y TF-IDF.
- `resultados/vectores_cbow.csv.gz`, `resultados/vectores_skip_gram.csv.gz`: una fila por reseña, con `id_resena` y 100 dimensiones numéricas.
- `resultados/modelo_cbow.model`, `resultados/modelo_skip_gram.model`: modelos entrenados de palabras.
- `03_word2vec_analisis.ipynb`: análisis y visualizaciones.

**Importante:** los archivos `.csv.gz` son CSV comprimidos con gzip, no ZIP. Leer con `pd.read_csv(ruta)`; no es necesario descomprimirlos manualmente. Los modelos entrenados son deterministas con semilla fija y un trabajador, aunque pequeñas diferencias numéricas pueden surgir entre versiones de bibliotecas o plataformas.

## Verificación rápida

```python
import pandas as pd
from scipy.sparse import load_npz
ids = pd.read_csv('resultados/ids_resenas.csv')
cbow = pd.read_csv('resultados/vectores_cbow.csv.gz')
sg = pd.read_csv('resultados/vectores_skip_gram.csv.gz')
assert len(ids) == len(cbow) == len(sg) == 12368
assert ids.id_resena.equals(cbow.id_resena)
assert ids.id_resena.equals(sg.id_resena)
assert load_npz('resultados/bow.npz').shape[0] == len(ids)
assert load_npz('resultados/tfidf.npz').shape[0] == len(ids)
print('Archivos alineados correctamente')
```
