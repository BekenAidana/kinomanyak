"""Сборка фактов каталога из Wikidata (без ключа, без модели).

Год, длительность, страны, языки и рейтинг в films.json берутся только отсюда,
не из памяти модели. Ручными остаются id, title, genres, description.

У каждого фильма поле source = ссылка на запись Wikidata (wikidata.org/wiki/Q…).
Скрипт читает QID из source и через API сущностей (wbgetentities) забирает
  P577 дата выхода      → year             (самый ранний год: премьера)
  P2047 длительность    → minutes          (в минутах; значение лучшего ранга,
                                            если их несколько — меньшая, театральная)
  P495 страна           → countries
  P364 язык оригинала   → languages
  P444 оценка + P447 Rotten Tomatoes → rotten_tomatoes (Tomatometer, % положительных
                                            рецензий критиков; нет оценки → null)
и приводит русские названия Wikidata к словам каталога таблицей в коде
(«Соединённые Штаты Америки» → «США», «английский язык» → «английский»).
Ранги Wikidata соблюдаются: deprecated не берём, preferred важнее normal.

Если в Wikidata поля нет, факт берётся из OVERRIDES: там же ссылка на
другой источник, и она дописывается в source.

uv run build_catalog.py          показать расхождения, файл не трогать
uv run build_catalog.py --write  записать films.json
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
FILMS = ROOT / "films.json"
API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "kinomanyak-catalog/0.1 (educational project)"}
QID = re.compile(r"wikidata\.org/wiki/(Q\d+)")
BATCH = 50  # wbgetentities отдаёт до 50 сущностей за запрос
ROTTEN_TOMATOES = "Q105584"
# У Rotten Tomatoes два процента с одним P447: оценка критиков и зрителей.
# Различает их только квалификатор P459 (метод оценки). Берём критиков.
TOMATOMETER = "Q108403393"
UNIT_FACTOR = {  # единица P2047 → множитель в минуты
    "http://www.wikidata.org/entity/Q7727": 1,       # минута
    "http://www.wikidata.org/entity/Q25235": 60,     # час
    "http://www.wikidata.org/entity/Q11574": 1 / 60,  # секунда
}

# Русское название Wikidata → слово каталога. Остальные страны идут как есть.
COUNTRY = {
    "Соединённые Штаты Америки": "США",
    "Соединённое Королевство": "Великобритания",
    "Великобритания": "Великобритания",
    "Республика Корея": "Южная Корея",
    "Союз Советских Социалистических Республик": "СССР",
    "Объединённые Арабские Эмираты": "ОАЭ",
}

# Язык после снятия слова «язык» → слово каталога (остальные идут как есть)
LANGUAGE = {
    "севернокитайский": "китайский",
    "кантонский диалект": "кантонский",
    "амслен": "американский жестовый",
}

# Поле, которого нет в Wikidata: значение + ссылка на источник.
# Ключ — id фильма в films.json.
OVERRIDES: dict[str, dict] = {
    # В Wikidata нет P364: фильм снят без речи («No speech was used in the movie»)
    "kz-003": {"languages": ["без слов"],
               "source": "https://en.wikipedia.org/wiki/Kelin_(film)"},
}


def language_word(label: str) -> str:
    """«английский язык» → «английский»; «Казахский язык» → «казахский»."""
    word = re.sub(r"\s+язык$", "", label.strip()).lower()
    return LANGUAGE.get(word, word)


def country_word(label: str) -> str:
    return COUNTRY.get(label, label)


def get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except OSError as error:  # 429 / таймаут: подождать и повторить
            wait = 20 * (attempt + 1)
            print(f"Wikidata: {error}, повтор через {wait} с")
            time.sleep(wait)
    raise SystemExit("Wikidata недоступна")


def entities(ids: list[str], props: str, languages: str = "") -> dict:
    found: dict = {}
    for start in range(0, len(ids), BATCH):
        params = {"action": "wbgetentities", "ids": "|".join(ids[start:start + BATCH]),
                  "props": props}
        if languages:
            params["languages"] = languages
        found.update(get(params)["entities"])
        time.sleep(1)
    return found


def best(claims: list[dict]) -> list[dict]:
    """Лучший ранг Wikidata: preferred, если есть, иначе normal; deprecated — никогда."""
    alive = [c for c in claims if c.get("rank") != "deprecated"
             and c["mainsnak"].get("snaktype") == "value"]
    preferred = [c for c in alive if c.get("rank") == "preferred"]
    return preferred or alive


def value(claim: dict):
    return claim["mainsnak"]["datavalue"]["value"]


def facts_of(entity: dict, labels: dict[str, str]) -> dict:
    claims = entity.get("claims", {})
    years = [int(value(c)["time"][1:5]) for c in best(claims.get("P577", []))]
    minutes = []
    for c in best(claims.get("P2047", [])):
        factor = UNIT_FACTOR.get(value(c).get("unit"))
        if factor:
            minutes.append(round(float(value(c)["amount"]) * factor))
    countries = list(dict.fromkeys(
        country_word(labels[value(c)["id"]]) for c in best(claims.get("P495", []))))
    languages = list(dict.fromkeys(
        language_word(labels[value(c)["id"]]) for c in best(claims.get("P364", []))))
    ratings = set()
    for c in best(claims.get("P444", [])):
        score = str(value(c))
        if (ROTTEN_TOMATOES in qualifier_ids(c, "P447")
                and TOMATOMETER in qualifier_ids(c, "P459") and score.endswith("%")):
            ratings.add(int(score.rstrip("%")))
    # Несколько разных Tomatometer — не угадываем, а просим решить через OVERRIDES.
    rating = ratings.pop() if len(ratings) == 1 else None
    return {
        "year": min(years) if years else None,
        "minutes": min(minutes) if minutes else None,
        "countries": countries or None,
        "languages": languages or None,
        "rotten_tomatoes": rating,
        "_many_minutes": sorted(set(minutes)) if len(set(minutes)) > 1 else None,
        "_many_ratings": sorted(ratings) if len(ratings) > 1 else None,
    }


def qualifier_ids(claim: dict, prop: str) -> list[str]:
    return [q["datavalue"]["value"]["id"] for q in claim.get("qualifiers", {}).get(prop, [])
            if q.get("snaktype") == "value"]


def sources(film: dict) -> list[str]:
    value_ = film.get("source") or []
    return [value_] if isinstance(value_, str) else list(value_)


def main() -> None:
    films = json.loads(FILMS.read_text(encoding="utf-8"))
    qids = {}
    for film in films:
        match = next((QID.search(s) for s in sources(film) if QID.search(s)), None)
        if not match:
            raise SystemExit(f"{film['id']}: нет ссылки на Wikidata в source")
        qids[film["id"]] = match.group(1)

    found = entities(sorted(set(qids.values())), "claims")
    # подписи стран и языков одним заходом (ru, при отсутствии — en)
    refs = sorted({value(c)["id"] for e in found.values()
                   for prop in ("P495", "P364")
                   for c in best(e.get("claims", {}).get(prop, []))})
    label_data = entities(refs, "labels", "ru|en")
    labels = {q: (e.get("labels", {}).get("ru") or e.get("labels", {}).get("en")
                  or {"value": q})["value"] for q, e in label_data.items()}

    problems = 0
    for film in films:
        fact = facts_of(found[qids[film["id"]]], labels)
        override = OVERRIDES.get(film["id"], {})
        for field in ("year", "minutes", "countries", "languages", "rotten_tomatoes"):
            if field in override:
                fact[field] = override[field]
        extra = override.get("source")
        if extra and extra not in sources(film):
            film["source"] = sources(film) + [extra]

        notes = []
        if fact["_many_minutes"] and "minutes" not in override:
            notes.append(f"несколько длительностей {fact['_many_minutes']}, взята меньшая")
        if fact["_many_ratings"] and "rotten_tomatoes" not in override:
            notes.append(f"несколько Tomatometer {fact['_many_ratings']} — нужен OVERRIDES")
            problems += 1
        for field in ("year", "minutes", "countries", "languages", "rotten_tomatoes"):
            new = fact[field]
            if new is None and field != "rotten_tomatoes":
                notes.append(f"нет {field} в Wikidata — нужен OVERRIDES")
                problems += 1
                continue
            if film.get(field, "отсутствует") != new:
                notes.append(f"{field}: {film.get(field, '—')} → {new}")
                film[field] = new
        if notes:
            print(f"{film['id']} {film['title']} ({qids[film['id']]}): " + "; ".join(notes))

    rated = sum(f.get("rotten_tomatoes") is not None for f in films)
    print(f"\nRotten Tomatoes: есть у {rated} из {len(films)} фильмов")
    if "--write" in sys.argv:
        if problems:
            raise SystemExit(f"Не записано: {problems} пустых полей, допишите OVERRIDES")
        FILMS.write_text(json.dumps(films, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
        print(f"Записано: {FILMS.name}, {len(films)} фильмов")
    else:
        print("Файл не изменён. Записать: uv run build_catalog.py --write")


if __name__ == "__main__":
    main()
