"""scraper.py - Web scraping de reseñas en español (Proyecto 2).

Este archivo tiene la parte que NO depende del sitio:
  - ScraperEducado: respeta robots.txt, pausa entre solicitudes, User-Agent
    identificatorio, reintentos con espera y se detiene si el sitio bloquea.
  - crear_documento: arma el documento MongoDB (sin nombres de usuario).
  - recolectar: recorre obras y páginas guardando el progreso en MongoDB.

Lo que SÍ depende del sitio son las dos funciones de parseo (parsear_resenas y
parsear_obra): se escriben viendo el HTML real del sitio elegido.
"""
import hashlib
import random
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    from src.preprocessing import limpiar_texto
    from src.db_manager import guardar_resenas
except ImportError:  # al probar fuera del repo
    from preprocessing import limpiar_texto
    from db_manager import guardar_resenas

# Poner aquí el correo de contacto del equipo.
USER_AGENT = "CUC-MineriaDeTextos-Proyecto2 (uso educativo; contacto: CORREO@ejemplo.com)"


class BloqueadoError(Exception):
    """El sitio nos bloquea (401/403). No se intenta evadir: se cambia de fuente."""


class ReglasRobots:
    """Lector de robots.txt que entiende los comodines * y $. Se escribe aquí
    porque urllib.robotparser de Python los trata como texto literal y dejaría
    pasar rutas como /*/sharerating, que sí están prohibidas."""

    def __init__(self, texto, user_agent):
        ua = user_agent.split()[0].split("/")[0].lower()
        grupos, actual = [], None
        for linea in texto.splitlines():
            linea = linea.split("#", 1)[0].strip()
            if ":" not in linea:
                continue
            campo, valor = (x.strip() for x in linea.split(":", 1))
            campo = campo.lower()
            if campo == "user-agent":
                if actual is None or actual["cerrado"]:
                    actual = {"agentes": [], "reglas": [], "retraso": None, "cerrado": False}
                    grupos.append(actual)
                actual["agentes"].append(valor.lower())
            elif actual is not None and campo in ("allow", "disallow", "crawl-delay"):
                actual["cerrado"] = True
                if campo == "crawl-delay":
                    try:
                        actual["retraso"] = float(valor)
                    except ValueError:
                        pass
                elif valor:
                    actual["reglas"].append((campo == "allow", self._patron(valor), len(valor)))
        elegido = (
            next((g for g in grupos if any(a != "*" and a in ua for a in g["agentes"])), None)
            or next((g for g in grupos if "*" in g["agentes"]), None)
            or {"reglas": [], "retraso": None}
        )
        self.reglas, self.retraso = elegido["reglas"], elegido["retraso"]

    @staticmethod
    def _patron(valor):
        fin = valor.endswith("$")
        partes = valor[:-1] if fin else valor
        regex = ".*".join(re.escape(p) for p in partes.split("*"))
        return re.compile(regex + ("$" if fin else ""))

    def permitido(self, url):
        p = urlparse(url)
        ruta = (p.path or "/") + (f"?{p.query}" if p.query else "")
        mejor = None  # gana la regla más larga; si empatan, gana Allow
        for permitir, patron, largo in self.reglas:
            if patron.match(ruta) and (
                mejor is None or largo > mejor[1] or (largo == mejor[1] and permitir)
            ):
                mejor = (permitir, largo)
        return True if mejor is None else mejor[0]


class ScraperEducado:
    def __init__(self, base_url, pausa=(2.0, 3.5), reintentos=3, user_agent=USER_AGENT):
        self.base = base_url.rstrip("/")
        self.user_agent = user_agent
        self.pausa = pausa
        self.reintentos = reintentos
        self.sesion = requests.Session()
        self.sesion.headers.update(
            {"User-Agent": user_agent, "Accept-Language": "es-ES,es;q=0.9"}
        )
        self._ultima = 0.0
        self.fallos_seguidos = 0  # URLs omitidas una tras otra
        self.max_fallos_seguidos = 5
        self._leer_robots()

    # ---- robots.txt
    def _leer_robots(self):
        """Lo descargamos nosotros: si el sitio responde 403, nos detenemos y
        avisamos en lugar de suponer que todo está permitido."""
        r = self.sesion.get(self.base + "/robots.txt", timeout=20)
        if r.status_code in (401, 403):
            raise BloqueadoError("El sitio no deja leer robots.txt (403).")
        self.robots = ReglasRobots(r.text if r.status_code == 200 else "", self.user_agent)
        if self.robots.retraso:  # si el sitio pide más espera, se respeta
            self.pausa = (max(self.pausa[0], self.robots.retraso),
                          max(self.pausa[1], self.robots.retraso + 1))

    def permitido(self, url):
        return self.robots.permitido(url)

    # ---- solicitudes
    def _esperar(self):
        falta = self._ultima + random.uniform(*self.pausa) - time.time()
        if falta > 0:
            time.sleep(falta)
        self._ultima = time.time()

    def get(self, url):
        """HTML de la URL, o None si no existe o falla tras los reintentos.

        Si el sitio responde 429 (demasiadas solicitudes) o falla, el scraper
        espera cada vez más y además se vuelve más lento para el resto de la
        sesión. Si se omiten varias URLs seguidas, se detiene (BloqueadoError)
        en vez de seguir insistiendo."""
        if not self.permitido(url):
            raise PermissionError(f"robots.txt no permite esta URL: {url}")
        ultimo = "sin respuesta"
        for intento in range(1, self.reintentos + 1):
            self._esperar()
            try:
                r = self.sesion.get(url, timeout=20)
            except requests.RequestException as e:
                ultimo = f"error de red ({type(e).__name__})"
                print(f"  {ultimo}; reintento {intento}")
                time.sleep(5 * 2 ** intento)
                continue
            if r.status_code == 200:
                self.fallos_seguidos = 0
                return r.text
            if r.status_code in (401, 403):
                raise BloqueadoError(f"Bloqueo ({r.status_code}) en {url}")
            if r.status_code == 404:
                return None
            ultimo = f"HTTP {r.status_code}"
            if r.status_code == 429 or r.status_code >= 500:
                if r.status_code == 429:  # el sitio pide ir más despacio: se duplica la pausa
                    self.pausa = (min(self.pausa[0] * 2, 30), min(self.pausa[1] * 2, 40))
                    print(f"  HTTP 429: la pausa sube a {self.pausa[0]:.0f}-{self.pausa[1]:.0f} s")
                espera = r.headers.get("Retry-After", "")
                time.sleep(int(espera) if espera.isdigit() else 10 * 2 ** intento)
        self.fallos_seguidos += 1
        print(f"  se omite {url} tras {self.reintentos} intentos (último: {ultimo})")
        if self.fallos_seguidos >= self.max_fallos_seguidos:
            raise BloqueadoError(
                f"{self.fallos_seguidos} URLs seguidas fallaron (último: {ultimo}). "
                "El sitio está rechazando las solicitudes: espera un rato antes de reintentar."
            )
        return None


# ---------------------------------------------------------------- normalización
def normalizar_calificacion(valor, escala_max):
    """Lleva la calificación a la escala 1-10 (5 estrellas: se multiplica por 2)."""
    return int(max(1, min(10, round(float(valor) * 10 / escala_max))))


def polaridad_de(calificacion):
    if calificacion <= 4:
        return "negativa"
    return "intermedia" if calificacion <= 6 else "positiva"


def periodo_de(fecha):
    return None if fecha is None else ("antes_2010" if fecha.year < 2010 else "2010_en_adelante")


def crear_documento(fuente, url_fuente, texto, calificacion, escala_max, generos,
                    titulo_obra=None, tipo_obra="pelicula", anio_estreno=None,
                    fecha_resena=None, id_sitio=None, is_spoiler=None, genero_principal=None):
    """Documento con el esquema de docs/esquema_mongodb.md. No recibe ni guarda
    el nombre del autor. fecha_resena: datetime o None."""
    texto = limpiar_texto(texto)
    cal = normalizar_calificacion(calificacion, escala_max)
    clave = id_sitio or hashlib.sha1(f"{url_fuente}|{texto[:200]}".encode()).hexdigest()[:16]
    doc = {
        "id_resena": f"{fuente}_{clave}",
        "texto_es": texto,
        "origen_texto": "nativo",
        "titulo_obra": titulo_obra,
        "tipo_obra": tipo_obra,
        "generos": generos,
        "genero_principal": genero_principal or (generos[0] if generos else None),
        "anio_estreno": anio_estreno,
        "calificacion": cal,
        "polaridad": polaridad_de(cal),
        "periodo": periodo_de(fecha_resena),
        "fuente": fuente,
        "url_fuente": url_fuente,
        "fecha_resena": fecha_resena,
        "fecha_recopilacion": datetime.now(timezone.utc),
        "is_spoiler": is_spoiler,
    }
    return {k: v for k, v in doc.items() if v is not None}


# ---------------------------------------------------------------- FilmAffinity
MESES = {m: i for i, m in enumerate(
    ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
     "septiembre", "octubre", "noviembre", "diciembre"], start=1)}


def parsear_fecha(texto):
    """'4 de noviembre de 2026' -> datetime, o None si no se entiende."""
    m = re.search(r"(\d{1,2}) de (\w+) de (\d{4})", texto or "", re.I)
    if not m or m.group(2).lower() not in MESES:
        return None
    return datetime(int(m.group(3)), MESES[m.group(2).lower()], int(m.group(1)))


def _texto(elemento):
    return re.sub(r"\s+", " ", elemento.get_text(" ", strip=True)).strip()


def parsear_resenas(html):
    """Reseñas de una página de críticas de FilmAffinity (5 por página).

    Devuelve [{"texto", "calificacion", "fecha", "id_sitio", "is_spoiler"}].
    NUNCA se lee el autor (.rev-author, .usercard-mini): no se guardan usuarios.
    Si la reseña tiene parte oculta por spoiler, se une al texto.
    """
    sopa = BeautifulSoup(html, "html.parser")
    filas = []
    for w in sopa.select("div.review-wrapper"):
        nota, cuerpo = w.select_one(".fa-user-rat-box"), w.select_one(".rev-text1")
        if nota is None or cuerpo is None:
            continue
        try:
            calificacion = float(nota.get_text(strip=True).replace(",", "."))
        except ValueError:
            continue
        texto = _texto(cuerpo)
        oculto = w.select_one(".spoiler-wrapper")
        if oculto is not None:
            texto += " " + re.sub(r"^\s*spoiler\s*:\s*", "", _texto(oculto), flags=re.I)
        fecha = w.select_one("small.rev-date")
        filas.append({
            "texto": texto,
            "calificacion": calificacion,
            "fecha": parsear_fecha(fecha.get_text() if fecha else ""),
            "id_sitio": w.get("data-review-id"),
            "is_spoiler": oculto is not None,
        })
    return filas


def parsear_obra(html):
    """Título, año de estreno y tipo de la obra, desde su página de críticas.
    El género no está en esa página: se toma del listado por género (ver recolectar)."""
    sopa = BeautifulSoup(html, "html.parser")
    titulo = sopa.select_one("#main-title a")
    anio = sopa.select_one('[itemprop="itemReviewed"] meta[itemprop="datePublished"]')
    tipo = sopa.select_one("span.movie-type")
    return {
        "titulo_obra": _texto(titulo) if titulo else None,
        "generos": [],
        "anio_estreno": int(anio["content"]) if anio and anio.get("content", "").isdigit() else None,
        "tipo_obra": "serie" if tipo and "serie" in tipo.get_text().lower() else "pelicula",
    }


def url_resenas_filmaffinity(url_obra, n):
    """De https://www.filmaffinity.com/cr/film624236.html a la página n de críticas."""
    m = re.search(r"film(\d+)\.html", url_obra)
    if not m:
        raise ValueError(f"No reconozco la URL de la película: {url_obra}")
    base = url_obra.split("/film")[0]
    return f"{base}/reviews/{n}/{m.group(1)}.html"


def id_de_obra(url):
    """Id de la película en la URL (FilmAffinity: film123.html; SensaCine: pelicula-123)."""
    m = re.search(r"film(\d+)\.html", url) or re.search(r"pelicula-(\d+)", url)
    return m.group(1) if m else None


def url_listado_filmaffinity(codigo, pagina, base="https://www.filmaffinity.com/cr"):
    """Listado de películas de un género, de la más votada a la menos votada
    (30 por página). Ejemplo de código: DR (Drama)."""
    return f"{base}/moviegenre.php?genre={codigo}&orderby=ratcount&chv=grid&nodoc&p={pagina}"


def parsear_listado(html):
    """(título de la página, URLs de las películas) de un listado por género."""
    sopa = BeautifulSoup(html, "html.parser")
    urls = []
    for a in sopa.select("div.card a.card-body[href]"):
        if id_de_obra(a["href"]) and a["href"] not in urls:
            urls.append(a["href"])
    return (sopa.title.get_text(strip=True) if sopa.title else ""), urls


def obras_por_genero(scraper, codigo, nombre, max_obras=150, max_paginas=10):
    """[(URL de la película, nombre del género)] recorriendo el listado del género.

    Comprueba que el código corresponde al género: la página debe llamarse
    'Género - <nombre> - Filmaffinity'; si no, se detiene en vez de guardar
    películas con el género equivocado."""
    obras, vistas = [], set()
    for pagina in range(1, max_paginas + 1):
        html = scraper.get(url_listado_filmaffinity(codigo, pagina))
        if html is None:
            break
        titulo, urls = parsear_listado(html)
        if pagina == 1 and nombre.lower() not in titulo.lower():
            raise ValueError(f"El código {codigo} no es '{nombre}': la página dice '{titulo}'.")
        nuevas = [u for u in urls if u not in vistas]
        if not nuevas:
            break
        vistas.update(nuevas)
        obras += [(u, nombre) for u in nuevas]
        if len(obras) >= max_obras:
            break
    return obras[:max_obras]


def ids_obras_guardadas(db, fuente):
    """Ids de las películas que ya tienen reseñas guardadas, para no volver a pedirlas."""
    urls = db["resenas"].distinct("url_fuente", {"fuente": fuente})
    ids = set()
    for u in urls:
        m = re.search(r"/reviews/\d+/(\d+)\.html", u) or re.search(r"pelicula-(\d+)/criticas", u)
        if m:
            ids.add(m.group(1))
    return ids


# ---------------------------------------------------------------- SensaCine
BASE_SENSACINE = "https://www.sensacine.com"


def url_listado_sensacine(codigo, pagina):
    """Películas de un género por popularidad (15 por página). Códigos que muestra
    el propio menú del sitio: Drama 13008, Comedia 13005, Acción 13025,
    Terror 13009, Animación 13026."""
    base = f"{BASE_SENSACINE}/peliculas/todas-peliculas/genero-{codigo}/"
    return base if pagina == 1 else f"{base}?page={pagina}"


def parsear_listado_sensacine(html):
    """(título de la página, [películas]) de un listado por género.

    Cada película: url, titulo, generos, anio_estreno y con_nota_usuarios (si ya
    tiene nota de usuarios; sin ella casi seguro no tiene críticas)."""
    sopa = BeautifulSoup(html, "html.parser")
    peliculas = []
    for c in sopa.select("div.entity-card-list"):
        a = c.select_one("a.meta-title-link")
        if a is None or not a.get("href") or not id_de_obra(a["href"]):
            continue
        con_nota = False
        for it in c.select("div.rating-item"):
            nombre, nota = it.select_one(".rating-title"), it.select_one(".stareval-note")
            if nombre is not None and nombre.get_text(strip=True) == "Usuarios" and nota is not None:
                con_nota = True
        info = c.select_one(".meta-body-info")
        fecha_tag = info.select_one(".date") if info else None
        fecha = parsear_fecha(fecha_tag.get_text()) if fecha_tag else None
        peliculas.append({
            "url": BASE_SENSACINE + a["href"],
            "titulo": _texto(a),
            "generos": [_texto(g) for g in info.select(".dark-grey-link")] if info else [],
            "anio_estreno": fecha.year if fecha else None,
            "con_nota_usuarios": con_nota,
        })
    return (sopa.title.get_text(strip=True) if sopa.title else ""), peliculas


def obras_por_genero_sensacine(scraper, codigo, nombre, max_obras=150, max_paginas=20):
    """[(URL, género, metadatos)] recorriendo el listado del género. Solo películas
    con nota de usuarios. El género de las reseñas es el del listado."""
    obras, vistas = [], set()
    for pagina in range(1, max_paginas + 1):
        html = scraper.get(url_listado_sensacine(codigo, pagina))
        if html is None:
            break
        titulo, peliculas = parsear_listado_sensacine(html)
        if pagina == 1 and nombre.lower() not in titulo.lower():
            print(f"  Aviso: la página dice '{titulo}'; comprueba que el código {codigo} sea {nombre}.")
        nuevas = [p for p in peliculas if p["url"] not in vistas]
        if not nuevas:
            break
        vistas.update(p["url"] for p in nuevas)
        obras += [(p["url"], nombre, {"titulo_obra": p["titulo"], "generos": p["generos"],
                                      "anio_estreno": p["anio_estreno"]})
                  for p in nuevas if p["con_nota_usuarios"]]
        if len(obras) >= max_obras:
            break
    return obras[:max_obras]


def url_resenas_sensacine(url_obra, n):
    """Página n de críticas de espectadores (10 por página) de una película."""
    m = re.search(r"pelicula-(\d+)", url_obra)
    if not m:
        raise ValueError(f"No reconozco la URL de la película: {url_obra}")
    base = f"{url_obra.split('/peliculas/')[0]}/peliculas/pelicula-{m.group(1)}/criticas-espectadores/"
    return base if n == 1 else f"{base}?page={n}"


def parsear_resenas_sensacine(html):
    """Críticas de espectadores de SensaCine. Nota en estrellas de 0,5 a 5 (usar
    escala_max=5). NUNCA se lee el bloque del autor (.review-card-aside)."""
    sopa = BeautifulSoup(html, "html.parser")
    filas = []
    for c in sopa.select("div.review-card"):
        nota = c.select_one(".review-card-meta .stareval-note")
        cuerpo = c.select_one(".review-card-content")
        if nota is None or cuerpo is None:
            continue
        try:
            calificacion = float(nota.get_text(strip=True).replace(",", "."))
        except ValueError:
            continue
        fecha = c.select_one(".review-card-meta-date")
        filas.append({
            "texto": _texto(cuerpo),
            "calificacion": calificacion,
            "fecha": parsear_fecha(fecha.get_text() if fecha else ""),
            "id_sitio": re.sub(r"\D", "", c.get("id", "")) or None,
        })
    return filas


def parsear_obra_sensacine(html):
    """Título de la película (género y año vienen del listado)."""
    sopa = BeautifulSoup(html, "html.parser")
    titulo = sopa.select_one("div.titlebar-title")
    return {"titulo_obra": _texto(titulo) if titulo else None, "generos": [],
            "anio_estreno": None, "tipo_obra": "pelicula"}


# ---------------------------------------------------------------- recolección
def recolectar(db, scraper, fuente, obras, url_pagina, parsear_resenas, parsear_obra,
               escala_max, max_paginas=5, max_nuevas=None, omitir=None):
    """Recorre obras y sus páginas de reseñas; guarda cada página apenas la
    obtiene, así un corte no pierde el progreso.

    obras: URLs, o tuplas (URL, género) o (URL, género, metadatos) cuando el
    género y los datos de la obra salen del listado.
    url_pagina(url_obra, n)  -> URL de la página n de reseñas de la obra.
    parsear_resenas(html)    -> [{"texto", "calificacion", "fecha", "id_sitio",
                                  "is_spoiler"}]  (sin usuarios)
    parsear_obra(html)       -> {"titulo_obra", "generos", "anio_estreno", "tipo_obra"}
    Para de recorrer una obra cuando una página no trae reseñas nuevas.
    max_nuevas: se detiene al llegar a esa cantidad de reseñas nuevas.
    omitir: ids de películas ya guardadas (ver ids_obras_guardadas).
    """
    nuevas = 0
    for item in obras:
        if max_nuevas is not None and nuevas >= max_nuevas:
            break
        if isinstance(item, tuple):
            url_obra, genero, *resto = item
            meta = resto[0] if resto else {}
        else:
            url_obra, genero, meta = item, None, {}
        if omitir and id_de_obra(url_obra) in omitir:
            continue
        html = scraper.get(url_pagina(url_obra, 1))
        if html is None:
            continue
        obra = parsear_obra(html)
        obra.update({k: v for k, v in meta.items() if v})  # datos del listado (título, géneros, año)
        if genero:
            obra["genero_principal"] = genero
            if not obra.get("generos"):
                obra["generos"] = [genero]
        vistos = set()
        for n in range(1, max_paginas + 1):
            url = url_pagina(url_obra, n)
            if n > 1:
                html = scraper.get(url)
                if html is None:
                    break
            filas = [f for f in parsear_resenas(html) if f.get("id_sitio") not in vistos]
            if not filas:
                break
            vistos.update(f.get("id_sitio") for f in filas)
            docs = [
                crear_documento(fuente, url, f["texto"], f["calificacion"], escala_max,
                                fecha_resena=f.get("fecha"), id_sitio=f.get("id_sitio"),
                                is_spoiler=f.get("is_spoiler"), **obra)
                for f in filas if f.get("texto") and f.get("calificacion") is not None
            ]
            nuevas += guardar_resenas(db, docs)
        print(f"{obra.get('titulo_obra')}: {nuevas} reseñas nuevas en total")
    return nuevas
