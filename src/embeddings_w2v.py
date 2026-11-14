"""
embeddings_w2v.py
=================
Funciones para el notebook 03 (BoW / TF-IDF y Word2Vec) del proyecto de
reseñas de cine en español.

Qué contiene
------------
1. Preparación del corpus a partir de ``pos_analysis.csv`` (lemas de spaCy).
2. Bolsa de palabras (BoW) y TF-IDF con scikit-learn.
3. Entrenamiento de Word2Vec en sus dos arquitecturas: CBOW (sg=0) y
   Skip-Gram (sg=1), con gensim.
4. Análisis semántico sobre los embeddings:
   - vecinos más cercanos y comparación CBOW vs Skip-Gram,
   - campos semánticos (expansión a partir de palabras semilla),
   - analogías (a : b :: c : ?),
   - similitud entre categorías (género, valoración, periodo),
   - vocabulario exclusivo y distintivo por categoría.

Uso básico (desde ``notebooks/``)::

    import sys; sys.path.append("../src")
    import embeddings_w2v as ew

    docs, oraciones = ew.preparar_corpus()
    cbow = ew.entrenar_w2v(oraciones, sg=0)
    sg = ew.entrenar_w2v(oraciones, sg=1)

gensim se importa solo al entrenar, de modo que el resto de funciones
(BoW, TF-IDF, vocabulario exclusivo) funcionan sin tenerlo instalado.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Rutas y constantes
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
RUTA_POS = RAIZ / "data" / "results" / "pos_analysis.csv"
if not RUTA_POS.exists():  # en el repo el archivo está comprimido
    RUTA_POS = RAIZ / "data" / "results" / "pos_analysis.csv.gz"
RUTA_RESENAS = RAIZ / "data" / "processed" / "resenas_limpias.csv"
DIR_TABLAS = RAIZ / "outputs" / "tablas_w2v"
DIR_FIGURAS = RAIZ / "outputs" / "figures"
DIR_MODELOS = RAIZ / "outputs" / "modelos_w2v"

SEMILLA = 42

# Categorías gramaticales (UPOS de spaCy) que se conservan: palabras con contenido.
UPOS_CONTENIDO = ("NOUN", "VERB", "ADJ", "ADV")

# Lemas muy frecuentes que casi no aportan significado (verbos y adverbios
# "comodín"). Se quitan para que el BoW y los vecinos no se llenen de ellos.
# La negación ("no", "nunca", "ni") se deja a propósito: es parte de la
# emocionalidad gramatical que estudia el proyecto.
STOP_LEMAS = frozenset(
    {
        "tener", "hacer", "ir", "poder", "decir", "dar", "ver", "venir",
        "poner", "salir", "deber", "querer", "saber", "llegar", "pasar",
        "quedar", "seguir", "llevar", "parecer", "haber", "ser", "estar",
        "muy", "más", "mas", "menos", "tan", "tanto", "mucho", "poco",
        "ya", "también", "tampoco", "así", "aquí", "allí", "ahí", "ahora",
        "entonces", "luego", "después", "antes", "siempre", "solo", "sólo",
        "bien", "como", "cuando", "donde", "mientras", "incluso", "además",
        "sino", "quizá", "quizás", "casi", "todavía", "aún", "aun",
    }
)

# Campos semánticos de partida (lemas en español). Las palabras que no estén
# en el vocabulario del modelo se ignoran y se avisa.
CAMPOS_SEMILLA = {
    "miedo_terror": ["miedo", "terror", "susto", "sangre", "monstruo", "asesino"],
    "humor": ["risa", "gracioso", "chiste", "humor", "divertido", "reír"],
    "emocion": ["emocionante", "conmovedor", "lágrima", "triste", "emoción", "llorar"],
    "elogio": ["excelente", "genial", "maravilloso", "brillante", "magnífico", "perfecto"],
    "critica": ["aburrido", "terrible", "horrible", "malo", "pésimo", "decepcionante"],
    "aspectos_tecnicos": ["guion", "actuación", "dirección", "fotografía", "música", "efecto"],
}

# Analogías de prueba: (positivas, negativas, palabra esperada o None).
# Se lee como "positivas - negativas": actor + mujer - hombre ≈ actriz.
# Con un corpus de ~1 M de tokens son pruebas exigentes: se reporta cuántas
# aciertan, no se asume que todas funcionen.
ANALOGIAS_PRUEBA = [
    (["actor", "mujer"], ["hombre"], "actriz"),
    (["director", "mujer"], ["hombre"], "directora"),
    (["director", "actriz"], ["actor"], "directora"),
    (["rey", "mujer"], ["hombre"], "reina"),
    (["terror", "risa"], ["miedo"], "comedia"),
    (["comedia", "miedo"], ["risa"], "terror"),
    (["película", "televisión"], ["cine"], "serie"),
    (["libro", "película"], ["novela"], None),
]


# --------------------------------------------------------------------------
# 1. Preparación del corpus
# --------------------------------------------------------------------------
def preparar_corpus(
    ruta_pos: Path | str = RUTA_POS,
    ruta_resenas: Path | str = RUTA_RESENAS,
    upos=UPOS_CONTENIDO,
    stop_lemas=STOP_LEMAS,
    min_lemas_oracion: int = 2,
):
    """Convierte ``pos_analysis.csv`` (un token por fila) en listas de lemas.

    Se usan los **lemas** de spaCy (columna ``lema``) en minúscula, solo de las
    categorías en ``upos``, sin la lista ``stop_lemas`` y sin tokens que no
    sean palabras (números, símbolos, puntuación).

    Devuelve
    --------
    docs : DataFrame
        Una fila por reseña: ``id``, ``tokens`` (lista de lemas), ``genero``,
        ``valoracion``, ``periodo``.
    oraciones : list[list[str]]
        Una lista de lemas por oración. Es lo que recibe Word2Vec, para que la
        ventana de contexto no cruce el límite de una oración.
    """
    necesarias = ["id", "oracion", "lema", "upos_spacy"]
    columnas = pd.read_csv(ruta_pos, encoding="utf-8-sig", nrows=0).columns.tolist()
    faltan = [c for c in necesarias if c not in columnas]
    if faltan:
        raise KeyError(f"Faltan columnas {faltan} en {ruta_pos}. Columnas encontradas: {columnas}")
    pos = pd.read_csv(ruta_pos, encoding="utf-8-sig", usecols=necesarias)
    meta = pd.read_csv(
        ruta_resenas,
        encoding="utf-8-sig",
        usecols=["id", "genero", "valoracion", "periodo"],
    )

    pos = pos[pos["upos_spacy"].isin(upos)].copy()
    pos["lema"] = pos["lema"].astype(str).str.lower().str.strip()
    es_palabra = pos["lema"].str.fullmatch(r"[a-záéíóúüñ]{2,}")
    pos = pos[es_palabra & ~pos["lema"].isin(stop_lemas)]

    por_oracion = pos.groupby(["id", "oracion"], sort=False)["lema"].apply(list)
    oraciones = [o for o in por_oracion.tolist() if len(o) >= min_lemas_oracion]

    docs = (
        pos.groupby("id", sort=False)["lema"]
        .apply(list)
        .rename("tokens")
        .reset_index()
        .merge(meta, on="id", how="left")
    )
    return docs, oraciones


def resumen_corpus(docs: pd.DataFrame, oraciones: list) -> pd.DataFrame:
    """Tabla corta con el tamaño del corpus ya preparado."""
    todos = Counter(t for toks in docs["tokens"] for t in toks)
    filas = {
        "reseñas": len(docs),
        "oraciones (para Word2Vec)": len(oraciones),
        "tokens (lemas)": sum(todos.values()),
        "vocabulario (lemas distintos)": len(todos),
        "lemas que aparecen solo una vez": sum(1 for f in todos.values() if f == 1),
    }
    return pd.DataFrame({"valor": filas})


# --------------------------------------------------------------------------
# 2. BoW y TF-IDF
# --------------------------------------------------------------------------
def _identidad(x):
    return x


def construir_bow(docs: pd.DataFrame, min_df: int = 5, max_df: float = 0.5, ngramas=(1, 1)):
    """Bolsa de palabras (conteos). Devuelve ``(vectorizador, matriz_dispersa)``.

    ``min_df``: se descartan lemas que aparecen en menos de ese número de reseñas.
    ``max_df``: se descartan lemas presentes en más de esa fracción de reseñas.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(
        tokenizer=_identidad,
        preprocessor=_identidad,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngramas,
    )
    return vec, vec.fit_transform(docs["tokens"])


def construir_tfidf(docs: pd.DataFrame, min_df: int = 5, max_df: float = 0.5, ngramas=(1, 1)):
    """TF-IDF con frecuencia sublineal (1 + log tf). Devuelve ``(vectorizador, matriz)``."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        tokenizer=_identidad,
        preprocessor=_identidad,
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngramas,
        sublinear_tf=True,
    )
    return vec, vec.fit_transform(docs["tokens"])


def terminos_globales(X, vec, n: int = 20, columna: str = "peso") -> pd.DataFrame:
    """Los ``n`` términos con mayor suma en la matriz (conteos o TF-IDF)."""
    terminos = np.array(vec.get_feature_names_out())
    suma = np.asarray(X.sum(axis=0)).ravel()
    orden = suma.argsort()[::-1][:n]
    return pd.DataFrame({"termino": terminos[orden], columna: suma[orden]})


def terminos_por_categoria(X, vec, etiquetas, n: int = 15) -> pd.DataFrame:
    """Términos con mayor peso medio dentro de cada categoría.

    Con TF-IDF resalta lo que caracteriza a la categoría; con BoW, lo más
    frecuente (y por eso se parece entre categorías).
    Devuelve una tabla larga: ``categoria``, ``rank``, ``termino``, ``peso``.
    """
    terminos = np.array(vec.get_feature_names_out())
    etiquetas = np.asarray(etiquetas)
    filas = []
    for cat in sorted(pd.unique(etiquetas[~pd.isna(etiquetas)])):
        idx = np.where(etiquetas == cat)[0]
        centroide = np.asarray(X[idx].mean(axis=0)).ravel()
        for r, j in enumerate(centroide.argsort()[::-1][:n], start=1):
            filas.append((cat, r, terminos[j], float(centroide[j])))
    return pd.DataFrame(filas, columns=["categoria", "rank", "termino", "peso"])


def comparar_bow_tfidf(X_bow, X_tfidf, vec_bow, vec_tfidf, n: int = 20) -> pd.DataFrame:
    """Compara el ranking global de BoW (conteos) con el de TF-IDF.

    Sirve para mostrar qué términos "bajan" al ponderar por TF-IDF (muy
    comunes en todas las reseñas) y cuáles "suben".
    """
    t_bow = terminos_globales(X_bow, vec_bow, n=len(vec_bow.get_feature_names_out()))
    t_tf = terminos_globales(X_tfidf, vec_tfidf, n=len(vec_tfidf.get_feature_names_out()))
    t_bow["rank_bow"] = np.arange(1, len(t_bow) + 1)
    t_tf["rank_tfidf"] = np.arange(1, len(t_tf) + 1)
    union = t_bow[["termino", "rank_bow"]].merge(
        t_tf[["termino", "rank_tfidf"]], on="termino", how="inner"
    )
    union["cambio_de_puesto"] = union["rank_bow"] - union["rank_tfidf"]
    top = union[(union["rank_bow"] <= n) | (union["rank_tfidf"] <= n)]
    return top.sort_values("rank_tfidf").reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Word2Vec: CBOW y Skip-Gram
# --------------------------------------------------------------------------
def entrenar_w2v(
    oraciones: list,
    sg: int,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 20,
    negative: int = 10,
    workers: int = 1,
    seed: int = SEMILLA,
):
    """Entrena Word2Vec. ``sg=0`` es CBOW y ``sg=1`` es Skip-Gram.

    Se usa ``workers=1`` y semilla fija para que el resultado sea reproducible
    (con varios hilos gensim no garantiza los mismos vectores).
    """
    from gensim.models import Word2Vec

    return Word2Vec(
        sentences=oraciones,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        epochs=epochs,
        workers=workers,
        seed=seed,
    )


def resumen_modelos(modelos: dict) -> pd.DataFrame:
    """Tabla con los hiperparámetros y el tamaño de cada modelo ``{nombre: Word2Vec}``."""
    filas = {}
    for nombre, m in modelos.items():
        filas[nombre] = {
            "arquitectura": "Skip-Gram" if m.sg == 1 else "CBOW",
            "dimensión": m.vector_size,
            "ventana": m.window,
            "min_count": m.min_count,
            "épocas": m.epochs,
            "negative sampling": m.negative,
            "vocabulario": len(m.wv.key_to_index),
        }
    return pd.DataFrame(filas).T


def guardar_modelos(modelos: dict, carpeta: Path | str = DIR_MODELOS) -> None:
    """Guarda cada modelo en ``carpeta/<nombre>.model`` (para no reentrenar)."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, m in modelos.items():
        m.save(str(carpeta / f"{nombre}.model"))


# --------------------------------------------------------------------------
# 4. Vecinos, campos semánticos y analogías
# --------------------------------------------------------------------------
def en_vocab(wv, palabra: str) -> bool:
    return palabra in wv.key_to_index


def vecinos(wv, palabra: str, topn: int = 10) -> list:
    """``[(palabra, similitud), ...]``; lista vacía si la palabra no está en el vocabulario."""
    if not en_vocab(wv, palabra):
        return []
    return wv.most_similar(palabra, topn=topn)


def comparar_vecinos(wvs: dict, palabras: list, topn: int = 8) -> pd.DataFrame:
    """Vecinos de cada palabra en cada modelo, uno junto al otro.

    ``wvs`` es ``{"CBOW": modelo.wv, "Skip-Gram": modelo.wv}``.
    """
    filas = []
    for p in palabras:
        fila = {"palabra": p}
        for nombre, wv in wvs.items():
            v = vecinos(wv, p, topn)
            fila[nombre] = ", ".join(w for w, _ in v) if v else "(fuera del vocabulario)"
        filas.append(fila)
    return pd.DataFrame(filas)


def solapamiento_vecinos(wv_a, wv_b, palabras: list, topn: int = 10) -> pd.DataFrame:
    """Cuántos de los ``topn`` vecinos coinciden entre dos modelos (y Jaccard)."""
    filas = []
    for p in palabras:
        a = {w for w, _ in vecinos(wv_a, p, topn)}
        b = {w for w, _ in vecinos(wv_b, p, topn)}
        if not a or not b:
            continue
        comunes = a & b
        filas.append(
            {
                "palabra": p,
                "vecinos_comunes": len(comunes),
                "jaccard": len(comunes) / len(a | b),
                "comunes": ", ".join(sorted(comunes)),
            }
        )
    return pd.DataFrame(filas)


def coherencia_campo(wv, palabras: list) -> float:
    """Similitud coseno media entre todos los pares de palabras (más alta = campo más compacto)."""
    palabras = [p for p in palabras if en_vocab(wv, p)]
    if len(palabras) < 2:
        return float("nan")
    M = np.array([wv[p] for p in palabras])
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    S = M @ M.T
    iu = np.triu_indices(len(palabras), k=1)
    return float(S[iu].mean())


def campos_semanticos(wvs: dict, campos: dict | None = None, topn: int = 12):
    """Expande cada campo semántico a partir de sus palabras semilla.

    Para cada modelo y campo, busca las palabras más cercanas al **promedio** de
    las semillas que sí están en el vocabulario.

    Devuelve ``(expansion, resumen)``:
    - ``expansion``: tabla larga ``modelo``, ``campo``, ``rank``, ``palabra``, ``similitud``.
    - ``resumen``: por modelo y campo, semillas usadas, semillas ausentes y coherencia
      (de las semillas y de las semillas + palabras halladas).
    """
    campos = campos or CAMPOS_SEMILLA
    filas, resumen = [], []
    for nombre, wv in wvs.items():
        for campo, semillas in campos.items():
            usadas = [s for s in semillas if en_vocab(wv, s)]
            ausentes = [s for s in semillas if not en_vocab(wv, s)]
            if not usadas:
                resumen.append(
                    {
                        "modelo": nombre, "campo": campo, "semillas_usadas": "",
                        "semillas_ausentes": ", ".join(ausentes),
                        "coherencia_semillas": np.nan, "coherencia_expandido": np.nan,
                    }
                )
                continue
            cercanas = wv.most_similar(positive=usadas, topn=topn)
            for r, (p, s) in enumerate(cercanas, start=1):
                filas.append((nombre, campo, r, p, float(s)))
            resumen.append(
                {
                    "modelo": nombre, "campo": campo,
                    "semillas_usadas": ", ".join(usadas),
                    "semillas_ausentes": ", ".join(ausentes),
                    "coherencia_semillas": coherencia_campo(wv, usadas),
                    "coherencia_expandido": coherencia_campo(wv, usadas + [p for p, _ in cercanas]),
                }
            )
    expansion = pd.DataFrame(filas, columns=["modelo", "campo", "rank", "palabra", "similitud"])
    return expansion, pd.DataFrame(resumen)


def analogia(wv, positivas: list, negativas: list, topn: int = 5):
    """Resuelve ``positivas - negativas`` (p. ej. actor + mujer - hombre).

    Devuelve ``(resultados, ausentes)``. Si falta alguna palabra, ``resultados`` es ``None``.
    """
    ausentes = [p for p in positivas + negativas if not en_vocab(wv, p)]
    if ausentes:
        return None, ausentes
    return wv.most_similar(positive=positivas, negative=negativas, topn=topn), []


def evaluar_analogias(wvs: dict, pruebas: list | None = None, topn: int = 5) -> pd.DataFrame:
    """Corre las analogías en cada modelo. ``pruebas``: lista de (positivas, negativas, esperada)."""
    pruebas = pruebas or ANALOGIAS_PRUEBA
    filas = []
    for nombre, wv in wvs.items():
        for pos, neg, esperada in pruebas:
            texto = " + ".join(pos) + " - " + " - ".join(neg)
            res, ausentes = analogia(wv, pos, neg, topn)
            if res is None:
                filas.append(
                    {"modelo": nombre, "analogía": texto, "esperada": esperada or "",
                     "top1": "", f"top{topn}": "", "acierto_top1": None,
                     f"acierto_top{topn}": None, "nota": "fuera del vocabulario: " + ", ".join(ausentes)}
                )
                continue
            palabras = [w for w, _ in res]
            filas.append(
                {
                    "modelo": nombre, "analogía": texto, "esperada": esperada or "",
                    "top1": palabras[0], f"top{topn}": ", ".join(palabras),
                    "acierto_top1": (palabras[0] == esperada) if esperada else None,
                    f"acierto_top{topn}": (esperada in palabras) if esperada else None,
                    "nota": "",
                }
            )
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# 5. Similitud entre categorías
# --------------------------------------------------------------------------
def vector_documento(wv, tokens: list):
    """Promedio de los vectores de las palabras del documento que están en el vocabulario."""
    vs = [wv[t] for t in tokens if t in wv.key_to_index]
    return np.mean(vs, axis=0) if vs else None


def vectores_categoria(wv, docs: pd.DataFrame, columna: str) -> dict:
    """Vector de cada categoría = promedio de los vectores de sus reseñas."""
    out = {}
    for cat, g in docs.dropna(subset=[columna]).groupby(columna):
        vs = [v for v in (vector_documento(wv, t) for t in g["tokens"]) if v is not None]
        if vs:
            out[cat] = np.mean(vs, axis=0)
    return out


def matriz_similitud(vectores: dict, centrar: bool = True) -> pd.DataFrame:
    """Similitud coseno entre categorías.

    ``centrar=True`` resta el vector medio de todas las categorías antes de
    comparar. Sin eso, todos los promedios apuntan casi en la misma dirección
    (las palabras comunes dominan) y las similitudes salen todas cercanas a 1,
    lo que oculta las diferencias reales.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    cats = list(vectores)
    M = np.array([vectores[c] for c in cats])
    if centrar:
        M = M - M.mean(axis=0, keepdims=True)
    return pd.DataFrame(cosine_similarity(M), index=cats, columns=cats)


def similitud_tfidf_categorias(X_tfidf, etiquetas) -> pd.DataFrame:
    """Similitud coseno entre los centroides TF-IDF de cada categoría (línea base sin embeddings)."""
    from sklearn.metrics.pairwise import cosine_similarity

    etiquetas = np.asarray(etiquetas)
    cats = sorted(pd.unique(etiquetas[~pd.isna(etiquetas)]))
    C = np.vstack([np.asarray(X_tfidf[np.where(etiquetas == c)[0]].mean(axis=0)).ravel() for c in cats])
    C = C - C.mean(axis=0, keepdims=True)
    return pd.DataFrame(cosine_similarity(C), index=cats, columns=cats)


def pares_mas_y_menos_similares(S: pd.DataFrame) -> pd.DataFrame:
    """Lista los pares de categorías ordenados de más a menos similares."""
    cats = list(S.index)
    filas = [
        (cats[i], cats[j], float(S.iloc[i, j]))
        for i in range(len(cats))
        for j in range(i + 1, len(cats))
    ]
    return pd.DataFrame(filas, columns=["categoria_a", "categoria_b", "similitud"]).sort_values(
        "similitud", ascending=False
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# 6. Vocabulario exclusivo y distintivo
# --------------------------------------------------------------------------
def _frecuencias_por_categoria(docs: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Matriz lema × categoría con las frecuencias absolutas."""
    cont = {
        cat: Counter(t for toks in g["tokens"] for t in toks)
        for cat, g in docs.dropna(subset=[columna]).groupby(columna)
    }
    return pd.DataFrame(cont).fillna(0).astype(int)


def vocabulario_exclusivo(docs: pd.DataFrame, columna: str, min_frec: int = 5) -> dict:
    """Lemas que aparecen **solo** en una categoría (y al menos ``min_frec`` veces en ella).

    Devuelve ``{categoria: DataFrame(lema, frecuencia, por_10k)}`` ordenado por frecuencia.
    ``por_10k`` es la frecuencia por cada 10.000 lemas de la categoría, para comparar
    categorías de distinto tamaño.
    """
    F = _frecuencias_por_categoria(docs, columna)
    totales = F.sum(axis=0)
    out = {}
    for cat in F.columns:
        otras = [c for c in F.columns if c != cat]
        solo_aqui = (F[cat] >= min_frec) & (F[otras].sum(axis=1) == 0)
        df = F.loc[solo_aqui, cat].sort_values(ascending=False).rename("frecuencia").to_frame()
        df["por_10k"] = 1e4 * df["frecuencia"] / totales[cat]
        out[cat] = df.rename_axis("lema").reset_index()
    return out


def resumen_exclusivo(exclusivo: dict) -> pd.DataFrame:
    """Cantidad de lemas exclusivos y ejemplos más frecuentes por categoría."""
    return pd.DataFrame(
        {
            cat: {
                "lemas_exclusivos": len(df),
                "más_frecuentes": ", ".join(df["lema"].head(8)),
            }
            for cat, df in exclusivo.items()
        }
    ).T


def vocabulario_distintivo(docs: pd.DataFrame, columna: str, min_frec: int = 10, top: int = 15) -> pd.DataFrame:
    """Lemas más característicos de cada categoría frente al resto (log-odds con a priori informativo).

    Es la versión "suave" del vocabulario exclusivo: un lema puede usarse en
    todas las categorías y aun así ser mucho más típico de una. Se usa el
    z-score del log-odds de Monroe et al. (2008) con suavizado aditivo.
    Devuelve ``categoria``, ``rank``, ``lema``, ``z``, ``frec_categoria``, ``frec_resto``.
    """
    F = _frecuencias_por_categoria(docs, columna)
    F = F[F.sum(axis=1) >= min_frec]
    alpha = 0.5
    a0 = alpha * len(F)
    filas = []
    for cat in F.columns:
        f_i = F[cat].astype(float)
        f_j = F.drop(columns=cat).sum(axis=1).astype(float)
        n_i, n_j = f_i.sum(), f_j.sum()
        delta = np.log((f_i + alpha) / (n_i + a0 - f_i - alpha)) - np.log(
            (f_j + alpha) / (n_j + a0 - f_j - alpha)
        )
        var = 1.0 / (f_i + alpha) + 1.0 / (f_j + alpha)
        z = (delta / np.sqrt(var)).sort_values(ascending=False).head(top)
        for r, (lema, valor) in enumerate(z.items(), start=1):
            filas.append((cat, r, lema, float(valor), int(f_i[lema]), int(f_j[lema])))
    return pd.DataFrame(
        filas, columns=["categoria", "rank", "lema", "z", "frec_categoria", "frec_resto"]
    )


# --------------------------------------------------------------------------
# 7. Gráficos y guardado
# --------------------------------------------------------------------------
def guardar_tabla(df: pd.DataFrame, nombre: str, carpeta: Path | str = DIR_TABLAS, indice: bool = False) -> Path:
    """Guarda un CSV en UTF-8 con BOM (para que Excel muestre bien las tildes)."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}.csv"
    df.to_csv(ruta, index=indice, encoding="utf-8-sig")
    return ruta


def guardar_figura(fig, nombre: str, carpeta: Path | str = DIR_FIGURAS) -> Path:
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    return ruta


def grafico_mapa_calor(S: pd.DataFrame, titulo: str, ax=None, vmin=None, vmax=None):
    """Mapa de calor de una matriz de similitud."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(S, annot=True, fmt=".2f", cmap="viridis", vmin=vmin, vmax=vmax,
                square=True, cbar=False, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return ax


def grafico_campos_pca(wv, expansion: pd.DataFrame, modelo: str, ax=None, semillas: dict | None = None):
    """Proyecta a 2D (PCA) las palabras de los campos semánticos, coloreadas por campo."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    semillas = semillas or CAMPOS_SEMILLA
    sub = expansion[expansion["modelo"] == modelo]
    puntos = []  # (palabra, campo)
    for campo, g in sub.groupby("campo"):
        usadas = [s for s in semillas.get(campo, []) if en_vocab(wv, s)]
        palabras = list(dict.fromkeys(usadas + g["palabra"].tolist()))
        puntos += [(p, campo) for p in palabras]
    # una palabra solo se dibuja una vez (con el primer campo en que apareció)
    vistos, unicos = set(), []
    for p, c in puntos:
        if p not in vistos:
            vistos.add(p)
            unicos.append((p, c))
    M = np.array([wv[p] for p, _ in unicos])
    xy = PCA(n_components=2, random_state=SEMILLA).fit_transform(M)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    campos = sorted({c for _, c in unicos})
    colores = dict(zip(campos, plt.cm.tab10.colors))
    for (p, c), (x, y) in zip(unicos, xy):
        ax.scatter(x, y, color=colores[c], s=18)
        ax.annotate(p, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
    handles = [plt.Line2D([], [], marker="o", ls="", color=colores[c], label=c) for c in campos]
    ax.legend(handles=handles, title="Campo", fontsize=8, loc="best")
    ax.set_title(f"Campos semánticos — {modelo} (PCA)")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
