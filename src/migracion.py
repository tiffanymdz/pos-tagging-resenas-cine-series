"""migracion.py - Convierte el corpus del Proyecto 1 en documentos para MongoDB.

Entradas (Proyecto 1):
  resenas_limpias.csv     id, movie_id, genero, periodo, anio_review, fecha,
                          calificacion, valoracion, is_spoiler, texto_en, texto_es
  pos_analysis.csv(.gz)   id, oracion, token, lema, upos_spacy, morf, upos_nltk
                          (un token por fila)

Salida: una lista de diccionarios (un documento por reseña) con el esquema de
docs/esquema_mongodb.md. Los embeddings NO van aquí: se guardan después en la
colección "embeddings" (ver db_manager.guardar_embeddings).
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# El Proyecto 1 guardó los géneros en inglés; el esquema los pide en español.
GENEROS_ES = {
    "Drama": "Drama",
    "Comedy": "Comedia",
    "Action": "Acción",
    "Horror": "Terror",
    "Animation": "Animación",
}

# Mismo conjunto que NEGACIONES en src/metrics.py del Proyecto 1.
NEGACIONES = {"no", "nunca", "jamás", "nada", "ni"}
CONTENIDO = ["NOUN", "VERB", "ADJ", "ADV"]


def leer_resenas(ruta):
    """resenas_limpias.csv se guardó en UTF-8 con BOM, por eso utf-8-sig."""
    return pd.read_csv(ruta, encoding="utf-8-sig")


def leer_tokens(ruta):
    """Lee pos_analysis. keep_default_na=False evita que tokens como "NA",
    "null" o "nan" se conviertan en valores vacíos."""
    cols = ["id", "token", "upos_spacy", "upos_nltk"]
    tokens = pd.read_csv(ruta, usecols=cols, dtype=str, keep_default_na=False)
    tokens["id"] = tokens["id"].astype(int)
    return tokens


def calcular_metricas(tokens):
    """Métricas por reseña (sin puntuación, igual que el Proyecto 1).

    Devuelve un DataFrame indexado por id con densidad_lexica,
    ratio_verbos_sustantivos, indice_adjetivacion e indice_negacion.
    """
    t = tokens[tokens["upos_spacy"] != "PUNCT"]
    total = t.groupby("id").size()
    cont = (
        t.groupby(["id", "upos_spacy"]).size().unstack(fill_value=0)
        .reindex(columns=CONTENIDO, fill_value=0)
    )
    negaciones = t["token"].str.lower().isin(NEGACIONES).groupby(t["id"]).sum()
    return pd.DataFrame({
        "densidad_lexica": cont.sum(axis=1) / total,
        "ratio_verbos_sustantivos": cont["VERB"] / cont["NOUN"].replace(0, np.nan),
        "indice_adjetivacion": cont["ADJ"] / total * 100,
        "indice_negacion": negaciones / total * 100,
    })


def _num(x):
    """Número de Python (pymongo no acepta tipos de numpy); NaN pasa a None."""
    return None if pd.isna(x) else round(float(x), 4)


def construir_documentos(resenas, tokens, detalles=None):
    """Un documento por reseña.

    detalles (opcional): {movie_id: {"generos": [...], "anio_estreno": 2006,
    "titulo_obra": "..."}} para completar los campos que el CSV no trae.
    """
    detalles = detalles or {}
    ahora = datetime.now(timezone.utc)
    tags = tokens.groupby("id", sort=False).agg(
        spacy=("upos_spacy", list), nltk=("upos_nltk", list)
    )
    metricas = calcular_metricas(tokens)

    docs = []
    for r in resenas.itertuples(index=False):
        idr = int(r.id)
        extra = detalles.get(r.movie_id, {})
        genero = GENEROS_ES.get(r.genero, r.genero)
        m = metricas.loc[idr] if idr in metricas.index else None
        doc = {
            "id_resena": f"p1_{idr}",
            "texto_es": r.texto_es,
            "texto_en": r.texto_en,
            "origen_texto": "traducido",
            "titulo_obra": extra.get("titulo_obra"),
            "tipo_obra": "pelicula",
            "generos": extra.get("generos") or [genero],
            "genero_principal": genero,
            "anio_estreno": extra.get("anio_estreno"),
            "calificacion": int(r.calificacion),
            "polaridad": r.valoracion,
            "periodo": r.periodo,
            "fuente": "imdb_kaggle",
            "fecha_resena": datetime.strptime(r.fecha, "%Y-%m-%d"),
            "fecha_recopilacion": ahora,
            "movie_id": r.movie_id,
            "anio_review": int(r.anio_review),
            "is_spoiler": bool(r.is_spoiler),
            "pos_tags": (
                {"nltk": tags.at[idr, "nltk"], "spacy": tags.at[idr, "spacy"]}
                if idr in tags.index else None
            ),
            "metricas": None if m is None else {
                "num_palabras": len(r.texto_es.split()),
                "densidad_lexica": _num(m["densidad_lexica"]),
                "ratio_verbos_sustantivos": _num(m["ratio_verbos_sustantivos"]),
                "indice_adjetivacion": _num(m["indice_adjetivacion"]),
                "indice_negacion": _num(m["indice_negacion"]),
            },
        }
        # Esquema flexible: los campos sin valor simplemente no se guardan.
        docs.append({k: v for k, v in doc.items() if v is not None})
    return docs


def construir_actualizaciones(textos_por_id, tokens):
    """Para reseñas ya guardadas en MongoDB que aún no tienen etiquetas (las del
    scraping): arma, por id_resena, los campos pos_tags y metricas.

    textos_por_id: {id_resena: texto_es}
    tokens: DataFrame con id (= id_resena), token, upos_spacy y upos_nltk
    Devuelve {id_resena: {"pos_tags": {...}, "metricas": {...}}}.
    """
    tags = tokens.groupby("id", sort=False).agg(
        spacy=("upos_spacy", list), nltk=("upos_nltk", list)
    )
    metricas = calcular_metricas(tokens)
    salida = {}
    for idr, texto in textos_por_id.items():
        if idr not in tags.index or idr not in metricas.index:
            continue  # reseña sin tokens (por ejemplo, texto vacío)
        m = metricas.loc[idr]
        salida[idr] = {
            "pos_tags": {"nltk": tags.at[idr, "nltk"], "spacy": tags.at[idr, "spacy"]},
            "metricas": {
                "num_palabras": len(texto.split()),
                "densidad_lexica": _num(m["densidad_lexica"]),
                "ratio_verbos_sustantivos": _num(m["ratio_verbos_sustantivos"]),
                "indice_adjetivacion": _num(m["indice_adjetivacion"]),
                "indice_negacion": _num(m["indice_negacion"]),
            },
        }
    return salida
