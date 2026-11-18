"""Pipeline reproducible de reseñas: BoW, TF-IDF y Word2Vec CBOW/Skip-Gram."""
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
    textos = df[columna].fillna('').astype(str).tolist()
    tokens = [re.findall(r'(?u)\b\w\w+\b', t.lower()) for t in textos]
    return tokens if usar_token else [' '.join(t) for t in tokens]

def ejecutar(archivo=None, salida=None, dimension=100, epochs=4):
    archivo = Path(archivo) if archivo else RAIZ / 'data' / 'resenas_mongodb.csv'
    salida = Path(salida) if salida else RAIZ / 'resultados'
    salida.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(archivo, dtype={'id_resena':str}, low_memory=False)
    if not {'id_resena','texto_es'}.issubset(df.columns):
        raise ValueError('El CSV debe contener id_resena y texto_es')
    if df.id_resena.isna().any() or df.id_resena.duplicated().any():
        raise ValueError('Los id_resena deben ser únicos y no nulos')
    corpus = preparar_corpus(df)
    tokens = [t.split() for t in corpus]
    bow_vec = CountVectorizer(max_features=10000, token_pattern=r'(?u)\b\w+\b')
    bow = bow_vec.fit_transform(corpus)
    tfidf_vec = TfidfVectorizer(vocabulary=bow_vec.vocabulary_, token_pattern=r'(?u)\b\w+\b')
    tfidf = tfidf_vec.fit_transform(corpus)
    save_npz(salida/'bow.npz', bow)
    save_npz(salida/'tfidf.npz', tfidf)
    pd.DataFrame({'id_resena':df.id_resena}).to_csv(salida/'ids_resenas.csv',index=False)
    pd.DataFrame({'termino':bow_vec.get_feature_names_out()}).to_csv(salida/'vocabulario.csv',index=False)
    modelos = {}
    for nombre, sg in [('cbow',0),('skip_gram',1)]:
        modelo = Word2Vec(sentences=tokens, vector_size=dimension, window=5, min_count=2, workers=2, sg=sg, seed=42, epochs=epochs)
        modelos[nombre]=modelo
        modelo.save(str(salida/f'modelo_{nombre}.model'))
        vectores = np.stack([np.mean([modelo.wv[w] for w in doc if w in modelo.wv],axis=0) if any(w in modelo.wv for w in doc) else np.zeros(dimension) for doc in tokens])
        tabla = pd.DataFrame(vectores,columns=[f'dim_{i+1}' for i in range(dimension)])
        tabla.insert(0,'id_resena',df.id_resena.to_numpy())
        tabla.to_csv(salida/f'vectores_{nombre}.csv.gz',index=False,compression='gzip',float_format='%.7g')
    print(f'Procesadas {len(df)} reseñas. BoW: {bow.shape}; TF-IDF: {tfidf.shape}; vectores: {dimension} dimensiones.')
    return df, bow, tfidf, modelos

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Reconstruir artefactos de Word2Vec')
    p.add_argument('--archivo',type=Path,default=None)
    p.add_argument('--salida',type=Path,default=None)
    p.add_argument('--dimension',type=int,default=100)
    p.add_argument('--epochs',type=int,default=4)
    a=p.parse_args()
    ejecutar(a.archivo,a.salida,a.dimension,a.epochs)
