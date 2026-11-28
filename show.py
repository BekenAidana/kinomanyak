"""Журнал вызовов из calls.jsonl

Зачем: видеть исходы, токены и $ без повторного вызова модели.
Сдача: после check.py журнал показывает весь прогон queries.json.

uv run show.py        последние 10 вызовов и итог
uv run show.py all    все вызовы
uv run show.py compare  таблица моделей для README (Rare)
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

JOURNAL = Path(__file__).resolve().parent / "calls.jsonl"
LAST = 10


def reasoning(usage):
    details = (usage.get("completion_tokens_details")
               or usage.get("output_tokens_details") or {})
    return details.get("reasoning_tokens") or 0


def input_tokens(usage):
    return usage.get("prompt_tokens", usage.get("input_tokens", 0))


def output_tokens(usage):
    return usage.get("completion_tokens", usage.get("output_tokens", 0))


def table(rows):
    print(f"{'время':19}  {'модель':14}  {'шаг':7}  {'исход':14}  {'вход':>5}  {'выход':>5}  "
          f"{'рассуждения':>11}  {'сек':>5}  {'$':>8}")
    for row in rows:
        usage = row.get("usage") or {}
        print(f"{row.get('time', ''):19}  {str(row.get('model', '')):14}  "
              f"{str(row.get('step', '')):7}  "
              f"{str(row.get('outcome', '')):14}  "
              f"{input_tokens(usage):>5}  {output_tokens(usage):>5}  "
              f"{reasoning(usage):>11}  {row.get('seconds', 0):>5}  "
              f"{row.get('usd', 0) or 0:>8.5f}")


def summary(rows):
    choice = [r for r in rows if r.get("step", "choice") != "filters"]
    outcomes = Counter(row.get("outcome") for row in choice)
    usages = [row.get("usage") or {} for row in rows]
    total_in = sum(input_tokens(u) for u in usages)
    total_out = sum(output_tokens(u) for u in usages)
    total_reasoning = sum(reasoning(u) for u in usages)
    usd = sum(row.get("usd") or 0 for row in rows)
    rejected = sum(len(r.get("rejected") or []) for r in rows
                   if isinstance(r.get("rejected"), list))
    violations = sum(len(r.get("violations") or []) for r in rows)
    print()
    print(f"Строк журнала: {len(rows)} (фильтры {len(rows) - len(choice)}, "
          f"выбор {len(choice)})")
    print(f"Запросов: {len(choice)} ("
          + ", ".join(f"{name} {count}" for name, count in outcomes.most_common())
          + ")")
    print(f"Токенов: вход {total_in}, выход {total_out}, "
          f"из них рассуждения {total_reasoning}")
    print(f"Отклонённых id: {rejected}; нарушений условий отсёк код: {violations}")
    print(f"Стоимость всего журнала: ${usd:.4f}")


def compare(rows):
    """Таблица моделей из журнала + набора запросов.

    Группа = модель + вариант: без variant — Rare (films.json, queries.json);
    big — Epic pick_big, whole — весь каталог в промпте (films_big.json, queries_big.json).
    Запрос = строка step="choice"; прошёл ли он — expect_ok из check.py по
    outcome и shown. Во всех прогонах — доля запросов, пройденных каждый раз.
    Разбор = эталон — фильтры модели (строка filters) дали тех же кандидатов,
    что conditions. Нарушения (эталон) — показанные фильмы, которые не проходят
    настоящие условия запроса. Секунды, токены и $ — сумма строк filters + choice на запрос.
    """
    from check import FILMS_BIG, QUERIES, QUERIES_BIG
    from kino import load_films

    small = (load_films(), json.loads(QUERIES.read_text(encoding="utf-8")))
    big = ((load_films(FILMS_BIG), json.loads(QUERIES_BIG.read_text(encoding="utf-8")))
           if FILMS_BIG.exists() and QUERIES_BIG.exists() else None)

    print(f"{'модель':14}  {'вариант':7}  {'прогонов':>8}  {'пройдено':>13}  {'опечатки':>9}  "
          f"{'во всех':>8}  {'разбор=эталон':>13}  {'нарушения':>9}  {'отсёк код':>9}  "
          f"{'чужие id':>8}  {'вход ток.':>9}  {'сек/запрос':>10}  {'сек медиана':>11}  {'$/запрос':>9}  {'$ всего':>8}")
    groups = sorted({(str(r.get("model")), r.get("variant") or "rare") for r in rows})
    for model, variant in groups:
        data = small if variant == "rare" else big
        if data is None:
            continue
        mine = [r for r in rows if str(r.get("model")) == model
                and (r.get("variant") or "rare") == variant]
        line(model, variant, mine, *data)


def line(model, variant, mine, films, queries):
    from check import expect_ok, parse_report, real_violations
    from kino import by_id

    known = by_id(films)
    items = {f"<запрос>{x['query']}</запрос>": x for x in queries}
    by_query = {x["query"]: x for x in queries}

    def item_of(row):
        return next((x for tag, x in items.items()
                     if str(row.get("request", "")).startswith(tag)), None)

    mine = [r for r in mine if item_of(r)]
    choice = [r for r in mine if r.get("step", "choice") != "filters"]
    filters = [r for r in mine if r.get("step") == "filters"]
    if not choice:
        return
    verdicts = [(item_of(r)["query"],
                 expect_ok(item_of(r), r.get("outcome"), r.get("shown") or []))
                for r in choice]
    passed = sum(ok for _, ok in verdicts)
    typo = [ok for q, ok in verdicts if by_query[q].get("typo")]
    per_query = Counter(q for q, _ in verdicts)
    runs = min(per_query.get(q["query"], 0) for q in items.values())
    always = sum(all(ok for q2, ok in verdicts if q2 == q) for q in per_query)
    reports = [parse_report(item_of(r), r.get("filters"), films) for r in filters]
    reports = [x for x in reports if x]
    same = sum(x["same"] for x in reports)
    real = sum(len(real_violations(item_of(r), r.get("shown") or [], known))
               for r in choice)
    foreign = sum(len(r.get("rejected") or []) for r in mine)
    cut = sum(len(r.get("violations") or []) for r in mine)
    tokens = sum(input_tokens(r.get("usage") or {}) for r in mine) / len(choice)
    seconds = sum(r.get("seconds") or 0 for r in mine) / len(choice)
    # медиана секунд на запрос (filters + choice подряд): среднее у локальной модели
    # ломают выбросы — сон ноутбука или зависание Ollama дают вызовы по 15 минут
    query_seconds, acc = [], 0.0
    for r in mine:
        acc += r.get("seconds") or 0
        if r.get("step", "choice") != "filters":
            query_seconds.append(acc)
            acc = 0.0
    median = sorted(query_seconds)[len(query_seconds) // 2]
    usd = sum(r.get("usd") or 0 for r in mine)
    parse = f"{same}/{len(reports)} {same / len(reports):.0%}" if reports else "—"
    typo_cell = f"{sum(typo)}/{len(typo)}" if typo else "—"
    print(f"{model:14}  {variant:7}  {runs:>8}  {passed:>4}/{len(choice):<3} "
          f"{passed / len(choice):>4.0%}  {typo_cell:>9}  "
          f"{always:>3}/{len(per_query):<3} {parse:>14}  {real:>9}  {cut:>9}  {foreign:>8}  "
          f"{tokens:>9.0f}  {seconds:>10.1f}  {median:>11.1f}  {usd / len(choice):>9.5f}  {usd:>8.4f}")


def main():
    if not JOURNAL.exists():
        print("Журнал пуст: файла calls.jsonl ещё нет")
        return
    lines = JOURNAL.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        print("Журнал пуст")
        return
    if "compare" in sys.argv[1:]:
        compare(rows)
        return
    table(rows if "all" in sys.argv[1:] else rows[-LAST:])
    summary(rows)

if __name__ == "__main__":
    main()
