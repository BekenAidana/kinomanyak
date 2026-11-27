"""Опыт с памятью модели: год и длительность 10 фильмов БЕЗ каталога в контексте.

Модель отвечает «по весам», код сравнивает ответ с films.json (факты из
Wikidata, см. build_catalog.py). Число расхождений идёт в README, вопрос 1.

uv run memory_probe.py          10 первых фильмов каталога, gpt-6-luna
"""

from __future__ import annotations

import json
import sys

from openai import OpenAI
from pydantic import BaseModel

from kino import MODEL, ROOT, load_films

sys.stdout.reconfigure(encoding="utf-8")

N = 10
OUT = ROOT / "memory_probe.json"


class Fact(BaseModel):
    id: str
    year: int
    minutes: int


class Facts(BaseModel):
    films: list[Fact]


def main() -> None:
    films = load_films()[:N]
    # только id и название: ни года, ни длительности модель не видит
    asked = [{"id": str(f["id"]), "title": f["title"]} for f in films]
    completion = OpenAI().chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Ответь по памяти: год выхода и "
             "длительность в минутах для каждого фильма. Верни все id."},
            {"role": "user", "content": json.dumps(asked, ensure_ascii=False)},
        ],
        response_format=Facts,
        reasoning_effort="none",
    )
    memory = {f.id: f for f in completion.choices[0].message.parsed.films}

    rows = []
    for film in films:
        got = memory.get(str(film["id"]))
        guess = {"year": got.year, "minutes": got.minutes} if got else None
        truth = {"year": film["year"], "minutes": film["minutes"]}
        rows.append({"id": film["id"], "title": film["title"], "catalog": truth,
                     "memory": guess, "match": guess == truth})
        mark = "ok  " if guess == truth else "DIFF"
        print(f"{mark} {film['title']:40} каталог {truth}  память {guess}")

    mismatches = [r["id"] for r in rows if not r["match"]]
    result = {"model": MODEL, "n": len(rows), "mismatches": len(mismatches),
              "mismatch_ids": mismatches, "rows": rows,
              "usage": completion.usage.model_dump() if completion.usage else None}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\nРасхождений: {len(mismatches)}/{len(rows)} → {OUT.name}")


if __name__ == "__main__":
    main()
