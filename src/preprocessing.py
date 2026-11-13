"""preprocessing.py - Limpieza, tokenización y POS tagging (del Proyecto 1).

El etiquetado es el mismo del notebook 02_pos_tagging del Proyecto 1:
  - spaCy es_core_news_md (sin ner, el parser sigue activo para las oraciones).
  - NLTK: unigrama + bigrama entrenados con cess_esp (EAGLES -> UPOS), aplicados
    sobre los MISMOS tokens de spaCy y oración por oración.
Así las reseñas nuevas (scraping) quedan con las mismas columnas que
pos_analysis: id, oracion, token, lema, upos_spacy, morf, upos_nltk.

Uso típico:
    nlp = cargar_spacy()
    bigrama, precision = entrenar_etiquetador_nltk()
    tokens = procesar_resenas(df["id_resena"], df["texto_es"], nlp, bigrama)
"""
import re

import pandas as pd


# ---------------------------------------------------------------- limpieza
def limpiar_texto(texto):
    """Limpieza ligera. NO pasa a minúsculas, NO quita tildes ni puntuación
    normal (spaCy los necesita). Reglas basadas en el README del Proyecto 1;
    si tu src/pos_tagging.py ya tiene una función de limpieza, úsala en su lugar
    para que todo el corpus quede tratado igual."""
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"<[^>]+>", " ", texto)                  # etiquetas HTML
    texto = texto.replace("\u2044", "/")                    # 1⁄2 -> 1/2
    texto = re.sub(r"[*\-_=~#]{3,}", " ", texto)            # líneas decorativas
    texto = re.sub(r"(?:\s[*#]){3,}", " ", texto)           # * * * * sueltos
    texto = re.sub(r"([!?¡¿.])\1{3,}", r"\1\1\1", texto)    # !!!!!! -> !!!
    return re.sub(r"\s+", " ", texto).strip()               # espacios repetidos


# ---------------------------------------------------------------- spaCy
def cargar_spacy():
    import spacy
    return spacy.load("es_core_news_md", disable=["ner"])


def etiquetar_spacy(ids, textos, nlp, batch_size=64):
    """Un token por fila, sin espacios sueltos (igual que el Proyecto 1)."""
    filas = []
    for id_resena, doc in zip(ids, nlp.pipe(textos, batch_size=batch_size)):
        for n_oracion, oracion in enumerate(doc.sents):
            for token in oracion:
                if token.is_space:
                    continue
                filas.append({
                    "id": id_resena,
                    "oracion": n_oracion,
                    "token": token.text,
                    "lema": token.lemma_,
                    "upos_spacy": token.pos_,
                    "morf": str(token.morph),
                })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- NLTK
def eagles_a_upos(etiqueta):
    t = etiqueta.lower()
    if t.startswith("np"):
        return "PROPN"
    if t.startswith("cs"):
        return "SCONJ"
    if t.startswith(("va", "vs")):
        return "AUX"
    mapa = {"a": "ADJ", "c": "CCONJ", "d": "DET", "f": "PUNCT", "i": "INTJ",
            "n": "NOUN", "p": "PRON", "r": "ADV", "s": "ADP", "v": "VERB",
            "z": "NUM", "w": "NUM"}
    return mapa.get(t[:1], "X")


def entrenar_etiquetador_nltk():
    """Devuelve (bigrama, precision). El entrenamiento es determinista (corte
    90/10 sin azar), así que da el mismo etiquetador que el Proyecto 1
    (unigrama 86,2 % y bigrama 87,6 % sobre cess_esp)."""
    import nltk
    from nltk.corpus import cess_esp
    from nltk.tag import BigramTagger, DefaultTagger, UnigramTagger

    nltk.download("cess_esp", quiet=True)
    oraciones = [[(w.lower(), eagles_a_upos(t)) for w, t in s if w != "*0*"]
                 for s in cess_esp.tagged_sents()]
    corte = int(len(oraciones) * 0.9)
    entreno, prueba = oraciones[:corte], oraciones[corte:]
    unigrama = UnigramTagger(entreno, backoff=DefaultTagger("NOUN"))
    bigrama = BigramTagger(entreno, backoff=unigrama)
    precision = {"unigrama": unigrama.accuracy(prueba),
                 "bigrama": bigrama.accuracy(prueba)}
    return bigrama, precision


def etiquetar_nltk(tokens, bigrama):
    """Agrega upos_nltk etiquetando los mismos tokens de spaCy, oración por
    oración (el contexto del bigrama se reinicia en cada oración)."""
    tokens = tokens.copy()
    por_oracion = tokens.groupby(["id", "oracion"], sort=False)["token"].apply(
        lambda s: s.str.lower().tolist()
    )
    etiquetadas = bigrama.tag_sents(por_oracion.tolist())
    tokens["upos_nltk"] = [e for oracion in etiquetadas for _, e in oracion]
    return tokens


# ---------------------------------------------------------------- todo junto
def procesar_resenas(ids, textos, nlp, bigrama, batch_size=64):
    """Etiqueta reseñas ya limpias y devuelve el DataFrame de tokens con las
    7 columnas de pos_analysis."""
    tokens = etiquetar_spacy(ids, textos, nlp, batch_size)
    return etiquetar_nltk(tokens, bigrama)
