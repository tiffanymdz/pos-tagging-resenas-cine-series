NEGACIONES = {"no", "nunca", "jamás", "nada", "ni"}
INTENSIDAD = {"muy", "demasiado", "totalmente", "tan", "bastante", "sumamente",
              "absolutamente", "completamente", "extremadamente", "realmente"}


def sin_puntuacion(tokens):
    return tokens[tokens["upos_spacy"] != "PUNCT"]


def indice_por_cien(tokens, condicion, nombre):
    palabras = sin_puntuacion(tokens)
    total = palabras.groupby("id").size()
    conteo = palabras[condicion(palabras)].groupby("id").size()
    indice = conteo.reindex(total.index, fill_value=0) / total * 100
    return indice.rename(nombre).reset_index()


def indice_negacion(tokens):
    return indice_por_cien(tokens, lambda p: p["token"].str.lower().isin(NEGACIONES), "neg_100")


def indice_intensidad(tokens):
    return indice_por_cien(
        tokens,
        lambda p: (p["upos_spacy"] == "ADV") & p["token"].str.lower().isin(INTENSIDAD),
        "int_100",
    )

#Métricas del Eje C (evolución temporal)

import pandas as pd


def quitar_puntuacion(tokens):
    "Deja solo los tokens que no son puntuación ni espacios (las métricas se calculan sin ellos)."
    tokens = tokens[tokens["upos_spacy"] != "PUNCT"]
    tokens = tokens[tokens["upos_spacy"] != "SPACE"]
    return tokens


def primera_persona_por_100(tokens):
    """Por reseña: pronombres y verbos en 1.ª persona por cada 100 tokens (P3).

    -Pronombres: PRON con Person=1 (definición del PDF): yo, me, mí...
    -Verbos: VERB o AUX con Person=1 (en español el sujeto suele omitirse: "me encantó", "vi")."""

    tokens = quitar_puntuacion(tokens).copy()
    tokens["es_pron_1p"] = (tokens["upos_spacy"] == "PRON") & tokens["morf"].str.contains("Person=1", na=False)
    tokens["es_verbo_1p"] = tokens["upos_spacy"].isin(["VERB", "AUX"]) & tokens["morf"].str.contains("Person=1", na=False)

    por_resena = tokens.groupby("id")
    resultado = pd.DataFrame({
        "tokens": por_resena.size(),
        "pron_1p": por_resena["es_pron_1p"].sum(),
        "verbo_1p": por_resena["es_verbo_1p"].sum(),
    }).reset_index()
    resultado["pron_1p_100"] = resultado["pron_1p"] / resultado["tokens"] * 100
    resultado["verbo_1p_100"] = resultado["verbo_1p"] / resultado["tokens"] * 100
    return resultado[["id", "pron_1p_100", "verbo_1p_100"]]


def palabras_por_resena(resenas):
    "Por reseña: número de palabras del texto en español (P4)."
    resultado = resenas[["id"]].copy()
    resultado["palabras"] = resenas["texto_es"].str.split().str.len()
    return resultado