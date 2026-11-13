"""db_manager.py - Conexión y operaciones MongoDB (Proyecto 2).

La cadena de conexión se lee de la variable de entorno MONGODB_URI (por ejemplo
desde un archivo .env que esté en .gitignore). Si no existe, se usa MongoDB
local (mongodb://localhost:27017). Con Atlas, la cadena con contraseña NUNCA
debe escribirse en el código ni subirse a GitHub.

    # .env
    MONGODB_URI=mongodb+srv://usuario:clave@cluster.mongodb.net/   (Atlas)
    MONGODB_URI=mongodb://localhost:27017                           (local)
    MONGODB_DB=resenas_cine                                         (opcional)
"""
import os

from pymongo import ASCENDING, MongoClient, UpdateOne

try:  # opcional: si hay un archivo .env, se lee
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DB_NOMBRE = os.getenv("MONGODB_DB", "resenas_cine")
COL_RESENAS = "resenas"
COL_EMBEDDINGS = "embeddings"  # vectores aparte: ahorra espacio en Atlas free


# ---------------------------------------------------------------- conexión
def get_client():
    # Sin .env se usa MongoDB local, que no lleva contraseña.
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")  # falla pronto si la conexión está mal
    return client


def get_db(client=None):
    return (client or get_client())[DB_NOMBRE]


def crear_indices(db):
    """Índices para filtros y agregaciones frecuentes."""
    res = db[COL_RESENAS]
    res.create_index("id_resena", unique=True)  # evita duplicados al reanudar
    res.create_index("genero_principal")
    res.create_index("polaridad")
    res.create_index([("origen_texto", ASCENDING), ("fuente", ASCENDING)])
    db[COL_EMBEDDINGS].create_index(
        [("id_resena", ASCENDING), ("tipo", ASCENDING)], unique=True
    )


# ---------------------------------------------------------------- escritura
def guardar_resenas(db, docs, lote=500):
    """Inserta o actualiza reseñas (upsert por id_resena). Devuelve cuántas se
    escribieron. Se puede llamar varias veces con los mismos datos sin duplicar."""
    col = db[COL_RESENAS]
    total = 0
    for i in range(0, len(docs), lote):
        ops = [
            UpdateOne({"id_resena": d["id_resena"]}, {"$set": d}, upsert=True)
            for d in docs[i : i + lote]
        ]
        r = col.bulk_write(ops, ordered=False)
        total += r.upserted_count + r.modified_count
    return total


def ids_existentes(db, fuente):
    """id_resena ya guardados de una fuente: sirve para que el scraper
    reanude sin repetir solicitudes."""
    cursor = db[COL_RESENAS].find({"fuente": fuente}, {"id_resena": 1, "_id": 0})
    return {d["id_resena"] for d in cursor}


def guardar_embeddings(db, id_resena, tipo, vector, decimales=4):
    """tipo: 'word2vec_avg' o 'beto_cls'. Redondea para ahorrar espacio."""
    db[COL_EMBEDDINGS].update_one(
        {"id_resena": id_resena, "tipo": tipo},
        {"$set": {"vector": [round(float(x), decimales) for x in vector]}},
        upsert=True,
    )


# ---------------------------------------------------------------- consultas
def conteo_por(db, campo):
    """Cantidad de reseñas por valor de un campo (género, polaridad, fuente...)."""
    pipeline = [
        {"$group": {"_id": f"${campo}", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}},
    ]
    return list(db[COL_RESENAS].aggregate(pipeline))


def genero_por_origen(db):
    """Reseñas por género y origen (traducido / nativo)."""
    pipeline = [
        {
            "$group": {
                "_id": {"genero": "$genero_principal", "origen": "$origen_texto"},
                "n": {"$sum": 1},
            }
        },
        {"$sort": {"_id.genero": 1, "_id.origen": 1}},
    ]
    return list(db[COL_RESENAS].aggregate(pipeline))


def calificacion_media_por_genero(db):
    pipeline = [
        {
            "$group": {
                "_id": "$genero_principal",
                "calificacion_media": {"$avg": "$calificacion"},
                "n": {"$sum": 1},
            }
        },
        {"$sort": {"calificacion_media": -1}},
    ]
    return list(db[COL_RESENAS].aggregate(pipeline))


def buscar(db, filtro=None, campos=None, limite=0):
    """Filtro libre. Ej: buscar(db, {"origen_texto": "nativo", "polaridad": "negativa"})."""
    return list(db[COL_RESENAS].find(filtro or {}, campos, limit=limite))
