"""Epic: каталог films_big.json — 1000+ фильмов, русскоязычный датасет.

Ловушка языка (README): открытые датасеты обычно на английском, а запросы на русском
и казахском. Путь проекта — русскоязычный датасет, собранный из открытых источников:
  факты     — Wikidata (CC0): год P577, минуты P2047, страны P495, языки P364,
              жанры P136, Rotten Tomatoes P444/P447 — тем же кодом, что build_catalog.py;
  описания  — русская Википедия (CC BY-SA 4.0): викитекст статей пакетами по 50,
              1–2 первых предложения раздела «Сюжет», нет раздела — первое предложение статьи.
Жанры Wikidata приводятся к словам каталога таблицей GENRE_KEYWORDS в коде
(«научно-фантастический фильм» → «фантастика»), список слов — canon.GENRES.
Наши 56 фильмов из films.json входят в каталог с теми же id; новые — id «wd-Q…».
У всех фильмов есть title_en — английское название из Wikidata (labels en).

uv run build_big.py   скачать (с кешем в .cache/big) и записать films_big.json
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import tiktoken

import build_catalog as bc
from canon import GENRES

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "films_big.json"
CACHE = ROOT / ".cache" / "big"
SPARQL = "https://query.wikidata.org/sparql"
RUWIKI = "https://ru.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "kinomanyak-catalog/0.1 (educational project)"}
LIST_LIMIT = 1500   # с запасом: часть отсеется (нет минут, жанра или описания)
MIN_SITELINKS = 40  # популярность: статьи о фильме минимум в 40 языковых разделах
ANIMATED = {"Q202866", "Q29168811"}  # P31 «анимационный фильм» → мультфильм

# Таблица соответствий: основа слова в русском названии жанра Wikidata → слово каталога.
# Одно название может дать несколько слов («комедийная драма» → комедия, драма).
GENRE_KEYWORDS = [
    ("мульт", "мультфильм"), ("анимац", "мультфильм"), ("аниме", "мультфильм"),
    ("документ", "документальный"), ("ужас", "ужасы"), ("хоррор", "ужасы"),
    ("фэнтези", "фэнтези"), ("научн", "фантастика"), ("фантаст", "фантастика"),
    ("комеди", "комедия"), ("мелодрам", "мелодрама"), ("романтич", "мелодрама"),
    ("мюзикл", "мюзикл"), ("музыкальн", "мюзикл"), ("боевик", "боевик"),
    ("триллер", "триллер"), ("криминал", "криминал"), ("гангстер", "криминал"),
    ("нуар", "криминал"), ("детектив", "детектив"), ("военн", "военный"),
    ("истори", "исторический"), ("биограф", "биография"), ("приключен", "приключения"),
    ("семейн", "семейный"), ("детск", "семейный"), ("драм", "драма"),
]

LIST_QUERY = """
SELECT ?film ?links WHERE {
  ?film wdt:P31 wd:Q11424; wikibase:sitelinks ?links.
  FILTER(?links >= %d)
  ?ru schema:about ?film; schema:isPartOf <https://ru.wikipedia.org/>.
} ORDER BY DESC(?links) LIMIT %d
"""


def cached(name: str, make):
    path = CACHE / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = make()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


def popular_films() -> list[str]:
    """Один SPARQL-запрос. Во время аварии WDQS лимит 1 запрос в минуту — ждём."""
    url = SPARQL + "?" + urllib.parse.urlencode(
        {"query": LIST_QUERY % (MIN_SITELINKS, LIST_LIMIT)})
    request = urllib.request.Request(
        url, headers={**HEADERS, "Accept": "application/sparql-results+json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                rows = json.load(response)["results"]["bindings"]
            return [r["film"]["value"].rsplit("/", 1)[-1] for r in rows]
        except OSError as error:
            print(f"SPARQL: {error}; жду 65 с (попытка {attempt + 1})")
            time.sleep(65)
    raise SystemExit("Wikidata SPARQL недоступен")


def genre_words(labels: list[str]) -> list[str]:
    words = []
    for label in labels:
        low = label.lower()
        for stem, word in GENRE_KEYWORDS:
            if stem in low and word in GENRES and word not in words:
                words.append(word)
    return words


def ruwiki_texts(titles: list[str]) -> dict[str, str]:
    """Викитекст статей пакетами по 50 (revisions): 30 запросов вместо 1500,
    лимит ruwiki (429) при таком темпе не срабатывает; на 429 — ждём Retry-After."""
    texts: dict[str, str] = {}
    for start in range(0, len(titles), 50):
        batch = titles[start:start + 50]
        params = {"action": "query", "prop": "revisions", "rvprop": "content",
                  "rvslots": "main", "redirects": 1, "titles": "|".join(batch),
                  "format": "json", "formatversion": 2}
        url = RUWIKI + "?" + urllib.parse.urlencode(params)
        for attempt in range(6):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS),
                                            timeout=120) as response:
                    data = json.load(response)["query"]
                break
            except urllib.error.HTTPError as error:
                wait = int(error.headers.get("Retry-After") or 30 * (attempt + 1))
                print(f"ruwiki: {error}; жду {wait} с")
                time.sleep(wait)
        else:
            raise SystemExit("ruwiki недоступна")
        # редиректы и нормализация: исходное название → итоговая страница
        alias = {x["from"]: x["to"] for x in data.get("normalized", []) + data.get("redirects", [])}
        pages = {p["title"]: p["revisions"][0]["slots"]["main"]["content"]
                 for p in data.get("pages", []) if p.get("revisions")}
        for title in batch:
            final = alias.get(alias.get(title, title), alias.get(title, title))
            texts[title] = pages.get(final, "")
        print(f"  статей: {min(start + 50, len(titles))}/{len(titles)}")
        time.sleep(1)
    return texts


def plain_wikitext(text: str) -> str:
    """Викиразметка → обычный текст: без сносок, шаблонов, ссылок и оформления."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    while True:  # вложенные шаблоны {{...}} — изнутри наружу
        cleaned = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if cleaned == text:
            break
        text = cleaned
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.S)                 # таблицы
    text = re.sub(r"\[\[(?:Файл|File|Изображение|Image):[^\]]*\]\]", "", text)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)             # [[a|b]] → b
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)                      # [[a]] → a
    text = re.sub(r"\[https?://\S+\s([^\]]*)\]", r"\1", text)             # [url текст]
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text


SENTENCE = re.compile(r"(?<=[.!?…])\s+(?=[«\"А-ЯЁA-Z0-9])")


def sentences(text: str, count: int, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    out = ""
    for part in SENTENCE.split(text)[:count]:
        if out and len(out) + len(part) > limit:
            break
        out = f"{out} {part}".strip()
    return out[:limit].rstrip(" ,;:") if out else ""


def description(article: str) -> str:
    """1–2 предложения сюжета; нет раздела «Сюжет» — первое предложение статьи."""
    plot = re.search(r"^==\s*Сюжет\s*==\s*$(.*?)(?=^==[^=]|\Z)", article, re.S | re.M)
    if plot:
        body = re.sub(r"^=+.*?=+\s*$", " ", plot.group(1), flags=re.M)
        text = sentences(plain_wikitext(body), 2)
        if len(text) > 40:
            return text
    intro = plain_wikitext(re.split(r"^==", article, maxsplit=1, flags=re.M)[0])
    intro = re.sub(r"\s*\([^()]*\)", "", intro.strip(), count=1)  # «(англ. Inception)»
    return sentences(intro, 1)


def main() -> None:
    small = json.loads((ROOT / "films.json").read_text(encoding="utf-8"))
    own = {bc.QID.search(s).group(1) for f in small for s in bc.sources(f)
           if bc.QID.search(s)}

    qids = [q for q in cached("list.json", popular_films) if q not in own]
    print(f"Популярных фильмов со статьёй в ruwiki: {len(qids) + len(own)}, новых {len(qids)}")

    found = cached("entities.json",
                   lambda: bc.entities(qids, "claims|sitelinks|labels", "ru"))
    refs = sorted({bc.value(c)["id"] for e in found.values()
                   for prop in ("P495", "P364", "P136")
                   for c in bc.best(e.get("claims", {}).get(prop, []))})
    label_data = cached("labels.json", lambda: bc.entities(refs, "labels", "ru|en"))
    labels = {q: (e.get("labels", {}).get("ru") or e.get("labels", {}).get("en")
                  or {"value": q})["value"] for q, e in label_data.items()}

    titles = {q: e["sitelinks"]["ruwiki"]["title"] for q, e in found.items()
              if "ruwiki" in e.get("sitelinks", {})}
    texts = cached("ruwiki_wikitext.json", lambda: ruwiki_texts(sorted(set(titles.values()))))

    # английские названия (title_en) — для запросов, где фильм помнят по-английски
    # («что-то вроде Inception»): ищет по ним отбор, видит модель при выборе
    own_qids = {f["id"]: bc.QID.search(next(x for x in bc.sources(f) if bc.QID.search(x))).group(1)
                for f in small}
    english = cached("labels_en_mul.json",
                     lambda: bc.entities(sorted(set(qids) | set(own_qids.values())), "labels", "en|mul"))

    def title_en(qid: str) -> str | None:
        # у многих фильмов английское название лежит в общей метке «mul» (для всех языков)
        labels = english.get(qid, {}).get("labels", {})
        return (labels.get("en") or labels.get("mul") or {}).get("value")

    films = [dict(f, title_en=title_en(own_qids[f["id"]])) for f in small]
    dropped = {"нет минут или года": 0, "нет жанра каталога": 0, "нет описания": 0}
    for qid in qids:
        entity = found.get(qid, {})
        fact = bc.facts_of(entity, labels)
        claims = entity.get("claims", {})
        genre_labels = [labels.get(bc.value(c)["id"], "") for c in bc.best(claims.get("P136", []))]
        genres = genre_words(genre_labels)
        if {bc.value(c)["id"] for c in bc.best(claims.get("P31", []))} & ANIMATED:
            genres = ["мультфильм"] + [g for g in genres if g != "мультфильм"]
        text = description(texts.get(titles.get(qid, ""), ""))
        if not fact["year"] or not fact["minutes"]:
            dropped["нет минут или года"] += 1
            continue
        if not genres:
            dropped["нет жанра каталога"] += 1
            continue
        if not text:
            dropped["нет описания"] += 1
            continue
        title = (entity.get("labels", {}).get("ru", {}).get("value")
                 or re.sub(r"\s*\(.*\)$", "", titles[qid]))
        films.append({
            "id": f"wd-{qid}",
            "title": title,
            "title_en": title_en(qid),
            "year": fact["year"],
            "minutes": fact["minutes"],
            "genres": genres[:4],
            "countries": fact["countries"] or [],
            "languages": fact["languages"] or [],
            "description": text,
            "rotten_tomatoes": fact["rotten_tomatoes"],
            "source": [f"https://www.wikidata.org/wiki/{qid}",
                       # читаемая ссылка: кодировка %D0%… раздувала каталог в разы
                       "https://ru.wikipedia.org/wiki/" + titles[qid].replace(" ", "_")],
        })

    OUT.write_text(json.dumps(films, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    encoding = tiktoken.get_encoding("o200k_base")
    catalog = json.dumps(films, ensure_ascii=False)
    tokens = len(encoding.encode(catalog))
    # в промпт Epic уходит каталог без source (kino.build_request(compact=True))
    prompt = json.dumps([{k: v for k, v in f.items() if k != "source"} for f in films],
                        ensure_ascii=False)
    prompt_tokens = len(encoding.encode(prompt))
    print(f"Отсеяно: {dropped}")
    print(f"Записано: {OUT.name}, {len(films)} фильмов "
          f"({sum('Казахстан' in f['countries'] for f in films)} с «Казахстан»)")
    print(f"Каталог целиком: {len(catalog)} символов ≈ {tokens} токенов (o200k_base)")
    print(f"Весь каталог в промпте (без source): {len(prompt)} символов ≈ {prompt_tokens} токенов; "
          f"при $0.10 / 1M входа ≈ ${prompt_tokens * 0.10 / 1e6:.3f} за запрос")


if __name__ == "__main__":
    main()
