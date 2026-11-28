"""Замер только шага Filters против эталона conditions — без выбора и без журнала.

Нужен, чтобы быстро сравнивать правки разбора (промпт, проверку языка, цитату):
33–44 вызова вместо полного прогона. В calls.jsonl ничего не пишет.

uv run filters_eval.py                 облако, 1 проход
uv run filters_eval.py --local --passes 2
uv run filters_eval.py --local --no-quote   без цитаты: язык проверяется только по запросу
uv run filters_eval.py --local --no-guard   без проверки языка кодом вообще
uv run filters_eval.py --big --only direction   набор Epic, только запросы с меткой direction (или en, typo)
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from collections import Counter

from openai import OpenAI

import kino
from check import FILMS_BIG, QUERIES, QUERIES_BIG, parse_report

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    big = "--big" in sys.argv
    films = kino.load_films(FILMS_BIG) if big else kino.load_films()
    items = json.loads((QUERIES_BIG if big else QUERIES).read_text(encoding="utf-8"))
    if "--only" in sys.argv:
        label = sys.argv[sys.argv.index("--only") + 1]
        items = [x for x in items if x.get(label)]
    local = "--local" in sys.argv
    passes = int(sys.argv[sys.argv.index("--passes") + 1]) if "--passes" in sys.argv else 1
    guard = "--no-guard" not in sys.argv
    quote = "--no-quote" not in sys.argv
    if local:
        client = OpenAI(base_url=kino.LOCAL_URL, api_key="ollama", max_retries=0)
        model, effort = kino.LOCAL_MODEL, "none"
    else:
        client, model, effort = OpenAI(), kino.MODEL, kino.CLOUD_EFFORT

    same = total = typo_same = typo_total = 0
    missed, extra, wrong = Counter(), Counter(), Counter()
    for _ in range(passes):
        for item in items:
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, parsed, _ = kino.call_model(
                    client, model, kino.FILTER_INSTRUCTIONS,
                    kino.build_filters_prompt(item["query"], films), kino.Filters, effort)
            if parsed is not None and not quote:
                parsed = parsed.model_copy(update={"language_quote": None})
            filters = (kino.normalize_filters(parsed, films, item["query"] if guard else None)
                       if parsed else kino.Filters())
            report = parse_report(item, filters.model_dump(exclude_none=True), films)
            total += 1
            same += report["same"]
            missed.update(report["missed"])
            extra.update(report["extra"])
            if not report["same"]:
                wrong[item["query"]] += 1
            if item.get("typo"):
                typo_total += 1
                typo_same += report["same"]

    print(f"{model} (проверка языка: {guard}, цитата: {quote}), проходов {passes}")
    print(f"Разбор = эталон: {same}/{total} ({same / total:.0%}); "
          f"с опечатками {typo_same}/{typo_total}")
    print(f"Пропущено условий: {dict(missed)}; лишних: {dict(extra)}")
    for query, count in wrong.most_common(10):
        print(f"  мимо {count}×  {query}")


if __name__ == "__main__":
    main()
