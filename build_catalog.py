"""Сборка фактов каталога из Wikidata (без ключа, без модели).

Год, длительность, страны и языки в films.json берутся только отсюда,
не из памяти модели. Ручными остаются id, title, genres, description.

У каждого фильма поле source = ссылка на запись Wikidata (wikidata.org/wiki/Q…).
Скрипт читает QID из source, одним SPARQL-запросом забирает
  P577 дата выхода      → year     (самый ранний год: премьера)
  P2047 длительность    → minutes  (в минутах; значение лучшего ранга,
                                    если их несколько — театральная, меньшая)
  P495 страна           → countries
  P364 язык оригинала   → languages
и приводит русские названия Wikidata к словам каталога таблицей в коде
(«Соединённые Штаты Америки» → «США», «английский язык» → «английский»).

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
SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "kinomanyak-catalog/0.1 (educational project)",
           "Accept": "application/sparql-results+json"}
QID = re.compile(r"wikidata\.org/wiki/(Q\d+)")
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


def language_word(label: str) -> str:
    """«английский язык» → «английский»; «Казахский язык» → «казахский»."""
    word = re.sub(r"\s+язык$", "", label.strip()).lower()
    return LANGUAGE.get(word, word)


def country_word(label: str) -> str:
    return COUNTRY.get(label, label)


# Поле, которого нет в Wikidata: значение + ссылка на источник.
# Ключ — id фильма в films.json.
OVERRIDES: dict[str, dict] = {
    # В Wikidata нет P364: фильм снят без речи («No speech was used in the movie»)
    "kz-003": {"languages": ["без слов"],
               "source": "https://en.wikipedia.org/wiki/Kelin_(film)"},
}


QUERY = """
SELECT ?film ?date ?amount ?unit ?countryLabel ?langLabel WHERE {
  VALUES ?film { %s }
  OPTIONAL { ?film wdt:P577 ?date. }
  OPTIONAL { ?film p:P2047 ?st. ?st a wikibase:BestRank;
                 psv:P2047 [ wikibase:quantityAmount ?amount;
                             wikibase:quantityUnit ?unit ]. }
  OPTIONAL { ?film wdt:P495 ?country. }
  OPTIONAL { ?film wdt:P364 ?lang. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
}
"""


def fetch(qids: list[str]) -> dict[str, dict]:
    """Один SPARQL-запрос на все фильмы → {QID: {years, minutes, countries, languages}}."""
    values = " ".join(f"wd:{q}" for q in qids)
    url = SPARQL + "?" + urllib.parse.urlencode({"query": QUERY % values})
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.load(response)
            break
        except OSError as error:  # 429 / таймаут: подождать и повторить
            print(f"Wikidata: {error}, повтор через 65 с (лимит 1 запрос в минуту)")
            time.sleep(65)
    else:
        raise SystemExit("Wikidata недоступна")

    facts: dict[str, dict] = {q: {"years": set(), "minutes": set(),
                                  "countries": [], "languages": []} for q in qids}
    for row in data["results"]["bindings"]:
        qid = row["film"]["value"].rsplit("/", 1)[-1]
        fact = facts[qid]
        if "date" in row:
            fact["years"].add(int(row["date"]["value"][:4]))
        if "amount" in row:
            factor = UNIT_FACTOR.get(row["unit"]["value"])
            if factor:
                fact["minutes"].add(round(float(row["amount"]["value"]) * factor))
        if "countryLabel" in row:
            word = country_word(row["countryLabel"]["value"])
            if word not in fact["countries"]:
                fact["countries"].append(word)
        if "langLabel" in row:
            word = language_word(row["langLabel"]["value"])
            if word not in fact["languages"]:
                fact["languages"].append(word)
    return facts


def sources(film: dict) -> list[str]:
    value = film.get("source") or []
    return [value] if isinstance(value, str) else list(value)


def main() -> None:
    films = json.loads(FILMS.read_text(encoding="utf-8"))
    qids = {}
    for film in films:
        match = next((QID.search(s) for s in sources(film) if QID.search(s)), None)
        if not match:
            raise SystemExit(f"{film['id']}: нет ссылки на Wikidata в source")
        qids[film["id"]] = match.group(1)

    facts = fetch(sorted(set(qids.values())))
    problems = 0
    for film in films:
        fact = facts[qids[film["id"]]]
        override = OVERRIDES.get(film["id"], {})
        new = {
            "year": min(fact["years"]) if fact["years"] else None,
            "minutes": min(fact["minutes"]) if fact["minutes"] else None,
            "countries": fact["countries"] or None,
            "languages": fact["languages"] or None,
        }
        for field in ("year", "minutes", "countries", "languages"):
            if field in override:
                new[field] = override[field]
        extra = override.get("source")
        if extra and extra not in sources(film):
            film["source"] = sources(film) + [extra]

        notes = []
        if len(fact["minutes"]) > 1 and "minutes" not in override:
            notes.append(f"несколько длительностей {sorted(fact['minutes'])}, взята меньшая")
        for field, value in new.items():
            if value is None:
                notes.append(f"нет {field} в Wikidata — нужен OVERRIDES")
                problems += 1
                continue
            if film.get(field) != value:
                notes.append(f"{field}: {film.get(field)} → {value}")
                film[field] = value
        if notes:
            print(f"{film['id']} {film['title']} ({qids[film['id']]}): " + "; ".join(notes))

    if "--write" in sys.argv:
        if problems:
            raise SystemExit(f"Не записано: {problems} пустых полей, допишите OVERRIDES")
        FILMS.write_text(json.dumps(films, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
        print(f"\nЗаписано: {FILMS.name}, {len(films)} фильмов")
    else:
        print("\nФайл не изменён. Записать: uv run build_catalog.py --write")


if __name__ == "__main__":
    main()
