{
  "$jsonSchema": {
    "bsonType": "object",
    "required": [
      "_id",
      "calificacion",
      "fecha_recopilacion",
      "fecha_resena",
      "fuente",
      "genero_principal",
      "generos",
      "id_resena",
      "metricas",
      "origen_texto",
      "periodo",
      "polaridad",
      "pos_tags",
      "texto_es",
      "tipo_obra"
    ],
    "properties": {
      "_id": {
        "bsonType": "objectId"
      },
      "anio_estreno": {
        "bsonType": "int"
      },
      "anio_review": {
        "bsonType": "int"
      },
      "calificacion": {
        "bsonType": "int"
      },
      "fecha_recopilacion": {
        "bsonType": "date"
      },
      "fecha_resena": {
        "bsonType": "date"
      },
      "fuente": {
        "bsonType": "string"
      },
      "genero_principal": {
        "bsonType": "string"
      },
      "generos": {
        "bsonType": "array",
        "items": {
          "bsonType": "string"
        }
      },
      "id_resena": {
        "bsonType": "string"
      },
      "is_spoiler": {
        "bsonType": "bool"
      },
      "metricas": {
        "bsonType": "object",
        "properties": {
          "densidad_lexica": {
            "bsonType": "double"
          },
          "indice_adjetivacion": {
            "bsonType": "double"
          },
          "indice_negacion": {
            "bsonType": "double"
          },
          "num_palabras": {
            "bsonType": "int"
          },
          "ratio_verbos_sustantivos": {
            "bsonType": [
              "double",
              "null"
            ]
          }
        },
        "required": [
          "densidad_lexica",
          "indice_adjetivacion",
          "indice_negacion",
          "num_palabras",
          "ratio_verbos_sustantivos"
        ]
      },
      "movie_id": {
        "bsonType": "string"
      },
      "origen_texto": {
        "bsonType": "string"
      },
      "periodo": {
        "bsonType": "string"
      },
      "polaridad": {
        "bsonType": "string"
      },
      "pos_tags": {
        "bsonType": "object",
        "properties": {
          "nltk": {
            "bsonType": "array",
            "items": {
              "bsonType": "string"
            }
          },
          "spacy": {
            "bsonType": "array",
            "items": {
              "bsonType": "string"
            }
          }
        },
        "required": [
          "nltk",
          "spacy"
        ]
      },
      "texto_en": {
        "bsonType": "string"
      },
      "texto_es": {
        "bsonType": "string"
      },
      "tipo_obra": {
        "bsonType": "string"
      },
      "titulo_obra": {
        "bsonType": "string"
      },
      "url_fuente": {
        "bsonType": "string"
      }
    }
  }
}