"""Utilidades reproducibles para 07_word2vec_analisis(1).ipynb.

No modifica el notebook ni los resultados existentes. Python 3.10+.
"""
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

RAIZ = Path(__file__).resolve().parent


def preparar_corpus(df, columna='texto_es', usar_token=False):
    """Devuelve textos normalizados o listas de tokens, preservando el orden."""
    if columna not in df.columns:
        raise KeyError(f'No existe la columna {columna!r}')
    tokens = [re.findall(r'(?u)\b\w\w+\b', texto.lower())
              for texto in df[columna].fillna('').astype(str)]
    return tokens if usar_token else [' '.join(t) for t in tokens]


def ejecutar(archivo=None, salida=None, dimension=100, epochs=4,
             ventana=5, min_count=3, seed=42):
    """Genera los archivos que consume el notebook, sin alterar sus celdas.

    Advertencia: al entrenar otra vez los valores pueden diferir de los
    resultados originales, especialmente entre plataformas.
    """
    archivo = Path(archivo) if archivo else RAIZ / 'data' / 'resenas_mongodb.csv'
    salida = Path(salida) if salida else RAIZ / 'resultados'
    if not archivo.is_file():
        raise FileNotFoundError(f'No se encontró el CSV: {archivo}')
    df = pd.read_csv(archivo, dtype={'id_resena': str}, low_memory=False)
    if not {'id_resena', 'texto_es'}.issubset(df.columns):
        raise ValueError('El CSV necesita las columnas id_resena y texto_es')
    if df['id_resena'].isna().any() or df['id_resena'].duplicated().any():
        raise ValueError('id_resena debe ser único y no nulo')
    tokens = preparar_corpus(df, usar_token=True)
    corpus = [' '.join(t) for t in tokens]
    if not any(tokens):
        raise ValueError('No hay texto utilizable para entrenar')
    salida.mkdir(parents=True, exist_ok=True)
    bow_vectorizador = CountVectorizer(max_features=10000, token_pattern=r'(?u)\b\w+\b')
    bow = bow_vectorizador.fit_transform(corpus)
    tfidf_vectorizador = TfidfVectorizer(vocabulary=bow_vectorizador.vocabulary_,
                                        token_pattern=r'(?u)\b\w+\b')
    tfidf = tfidf_vectorizador.fit_transform(corpus)
    save_npz(salida / 'bow.npz', bow)
    save_npz(salida / 'tfidf.npz', tfidf)
    pd.DataFrame({'id_resena': df['id_resena']}).to_csv(salida / 'ids_resenas.csv', index=False)
    pd.DataFrame({'termino': bow_vectorizador.get_feature_names_out()}).to_csv(
        salida / 'vocabulario.csv', index=False)
    modelos = {}
    for nombre, sg in [('cbow', 0), ('skip_gram', 1)]:
        modelo = Word2Vec(sentences=tokens, vector_size=dimension, window=ventana,
                          min_count=min_count, workers=1, sg=sg, seed=seed, epochs=epochs)
        modelo.save(str(salida / f'modelo_{nombre}.model'))
        vectores = np.zeros((len(tokens), dimension), dtype=np.float32)
        for i, palabras in enumerate(tokens):
            disponibles = [modelo.wv[w] for w in palabras if w in modelo.wv]
            if disponibles:
                vectores[i] = np.mean(disponibles, axis=0)
        tabla = pd.DataFrame(vectores, columns=[f'dim_{j+1}' for j in range(dimension)])
        tabla.insert(0, 'id_resena', df['id_resena'].to_numpy())
        tabla.to_csv(salida / f'vectores_{nombre}.csv.gz', index=False,
                     compression='gzip', float_format='%.7g')
        modelos[nombre] = modelo
    print(f'Listo: {len(df)} reseñas | BoW {bow.shape} | TF-IDF {tfidf.shape}')
    return df, bow, tfidf, modelos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generar archivos para el notebook Word2Vec')
    parser.add_argument('--archivo', type=Path, default=None)
    parser.add_argument('--salida', type=Path, default=None)
    parser.add_argument('--dimension', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=4)
    args = parser.parse_args()
    ejecutar(args.archivo, args.salida, args.dimension, args.epochs)

