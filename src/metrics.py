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