"""Прогон queries.json через pick — набор проверок.

Один удачный запуск ничего не доказывает: гоняем весь набор и смотрим
исходы + expect (must / forbid / no_match). Проверяется всё приложение:
промпт, схема, allow-list, журнал — не «только модель».

uv run check.py
uv run check.py --local   локальная qwen3.5:4b
uv run check.py --runs 3  три прогона подряд (Rare: сравнение моделей)

Epic (films_big.json + queries_big.json):
uv run check.py --big     pick_big: модели уходят только 30 кандидатов
uv run check.py --whole   весь каталог в промпте, один вызов (только облако)
uv run check.py --recall  без модели: попал ли фильм из must в 30 кандидатов
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import openai
from openai import OpenAI

from kino import (CLOUD_EFFORT, JOURNAL, LOCAL_MODEL, LOCAL_URL, MODEL, ROOT, K,
                  Filters, apply_filters, by_id, candidates, load_films, passes, pick,
                  pick_big, pick_whole)

sys.stdout.reconfigure(encoding="utf-8")

QUERIES = ROOT / "queries.json"
QUERIES_BIG = ROOT / "queries_big.json"
FILMS_BIG = ROOT / "films_big.json"
JOURNAL_NOTE = JOURNAL  # тот же файл, куда пишет pick


def expect_ok(item, outcome, ids) -> bool:
    """Совпал ли ответ с ожиданием в queries.json.

    must   — среди picks должен быть хотя бы один id из списка
    forbid — ни одного id из списка
    no_match — исход ровно no_match
    без expect — на check не влияет (считаем только outcome)
    """
    expect = item.get("expect") or {}
    if not expect:
        return True
    if expect.get("no_match"):
        return outcome == "no_match"
    if outcome != "ok":
        return False
    must = {str(x) for x in expect.get("must") or []}
    forbid = {str(x) for x in expect.get("forbid") or []}
    got = {str(x) for x in ids}
    if must and not (must & got):
        return False
    if forbid & got:
        return False
    return True


# --- Эталонные условия (conditions в queries.json) ---
# Нарушения, которые отсекает pick, считаются против фильтров САМОЙ модели:
# если модель потеряла «до двух часов», такой счётчик покажет 0. Эталон —
# условия, которые в запросе есть на самом деле (по правилам промпта фильтров).

def real_violations(item, ids, known) -> list[str]:
    """Показанные фильмы, которые нарушают настоящие условия запроса."""
    if "conditions" not in item:
        return []
    truth = Filters(**item["conditions"])
    return [str(i) for i in ids if not passes(known[str(i)], truth)]


def parse_report(item, model_filters: dict | None, films) -> dict | None:
    """Разбор условий моделью против эталона.

    same   — фильтры модели оставили тех же кандидатов, что и эталон
             (2031 как year_from или как year_from+year_to — одно и то же);
    missed — условия эталона, которых модель не поставила;
    extra  — условия модели, которых в запросе нет.
    """
    if "conditions" not in item or model_filters is None:
        return None
    truth = item["conditions"]
    # language_quote — доказательство для languages, а не условие: с эталоном не сравниваем
    mine = {k: v for k, v in model_filters.items() if k != "language_quote"}

    def chosen(conditions):
        return [f["id"] for f in apply_filters(films, Filters(**conditions))]

    return {"same": chosen(mine) == chosen(truth),
            "missed": sorted(set(truth) - set(mine)),
            "extra": sorted(set(mine) - set(truth))}


def recall(items, films) -> None:
    """Epic, замер 1: отбор без модели. Фильм из must попал в K кандидатов?

    Три варианта отбора: только поиск по описаниям; поиск + фильтры модели
    (из строк step="filters" прогона --big в журнале); поиск + эталонные условия.
    """
    model_filters: dict[str, list[dict]] = {}
    if JOURNAL_NOTE.exists():
        for line in JOURNAL_NOTE.read_text(encoding="utf-8").splitlines():
            row = json.loads(line) if line.strip() else {}
            if row.get("step") == "filters" and row.get("variant") == "big":
                query = row["request"].removeprefix("<запрос>").removesuffix("</запрос>")
                model_filters.setdefault(query, []).append(row.get("filters", {}))

    def hit(item, conditions):
        must = {str(x) for x in (item.get("expect") or {}).get("must") or []}
        found = {str(x) for x in candidates(item["query"], films, K,
                                            Filters(**conditions) if conditions is not None else None)}
        return bool(must & found)

    with_must = [x for x in items if (x.get("expect") or {}).get("must")]
    text = sum(hit(x, None) for x in with_must)
    gold = sum(hit(x, x.get("conditions", {})) for x in with_must)
    model_rows = [(x, f) for x in with_must for f in model_filters.get(x["query"], [])]
    clean = [(x, {k: v for k, v in f.items() if k != "language_quote"}) for x, f in model_rows]
    by_model = sum(hit(x, f) for x, f in clean)
    print(f"Отбор без модели, K={K}, каталог {len(films)} фильмов, запросов с must: {len(with_must)}")
    print(f"  только поиск по описаниям:      {text}/{len(with_must)} ({text / len(with_must):.0%})")
    if clean:
        print(f"  поиск + фильтры модели (журнал): {by_model}/{len(clean)} ({by_model / len(clean):.0%})")
    print(f"  поиск + эталонные условия:      {gold}/{len(with_must)} ({gold / len(with_must):.0%})")
    misses = [x["query"] for x in with_must if not hit(x, x.get("conditions", {}))]
    for query in misses:
        print(f"  мимо (с эталоном): {query}")


def main() -> None:
    big = "--big" in sys.argv or "--whole" in sys.argv or "--recall" in sys.argv
    items = json.loads((QUERIES_BIG if big else QUERIES).read_text(encoding="utf-8"))
    films = load_films(FILMS_BIG) if big else load_films()
    if "--recall" in sys.argv:
        recall(items, films)
        return
    flow = pick_whole if "--whole" in sys.argv else pick_big if big else pick
    if "--local" in sys.argv:
        client = OpenAI(base_url=LOCAL_URL, api_key="ollama", max_retries=0)
        model = LOCAL_MODEL
        effort = "none"
    else:
        try:
            client = OpenAI()
        except openai.OpenAIError:
            print("Нет ключа: скопируйте .env.example в .env и впишите OPENAI_API_KEY")
            return
        model = MODEL
        effort = CLOUD_EFFORT

    runs = 1
    if "--runs" in sys.argv:
        runs = int(sys.argv[sys.argv.index("--runs") + 1])
    for run in range(1, runs + 1):
        if runs > 1:
            print(f"\n=== прогон {run}/{runs} ===")
        run_set(client, model, effort, items, films, flow)


def run_set(client, model, effort, items, films, flow=pick) -> None:
    """Один прогон набора: исходы, доля пройденных, нарушения условий, чужие id,
    разбор условий против эталона."""
    print(f"Модель {model} (reasoning_effort={effort}), {flow.__name__}, каталог {len(films)}: "
          f"{len(items)} запросов…\n")
    known = by_id(films)
    outcomes: list[str] = []
    rejected_total = cut_total = real_total = 0
    expect_pass = expect_total = 0
    parsed = parsed_same = missed = extra = 0
    typo_pass = typo_total = 0
    started = time.time()

    for item in items:
        query = item["query"]
        # чужие id, отсечения и фильтры модели — из новых строк журнала этой итерации
        before = JOURNAL_NOTE.read_text(encoding="utf-8") if JOURNAL_NOTE.exists() else ""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            outcome, ids = flow(client, model, query, films, effort=effort)
        outcomes.append(outcome)
        after = JOURNAL_NOTE.read_text(encoding="utf-8") if JOURNAL_NOTE.exists() else ""
        cut = 0
        model_filters = None
        for line in after[len(before):].splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rejected_total += len(row.get("rejected") or [])
            cut += len(row.get("violations") or [])
            if row.get("step") == "filters" and "filters" in row:
                model_filters = row["filters"]
        cut_total += cut
        real = real_violations(item, ids, known)
        real_total += len(real)
        report = parse_report(item, model_filters, films)
        if report:
            parsed += 1
            parsed_same += report["same"]
            missed += len(report["missed"])
            extra += len(report["extra"])
        has_expect = bool(item.get("expect"))
        if has_expect:
            expect_total += 1
            ok = expect_ok(item, outcome, ids)
            expect_pass += int(ok)
            if item.get("typo"):
                typo_total += 1
                typo_pass += int(ok)
            mark = "✅" if ok else "❌"
        else:
            mark = "—"
        notes = []
        if report and not report["same"]:
            notes.append("разбор: " + ", ".join(
                [f"-{x}" for x in report["missed"]] + [f"+{x}" for x in report["extra"]]
                or ["не те значения"]))
        if cut:
            notes.append(f"код отсёк {cut}")
        if real:
            notes.append(f"нарушает условия: {', '.join(real)}")
        note = f"  ({'; '.join(notes)})" if notes else ""
        print(f"{mark} {outcome:14}  {query[:60]}{note}")

    counts = Counter(outcomes)
    seconds = time.time() - started
    share = expect_pass / expect_total if expect_total else 0
    print()
    print(f"Исходы: ok {counts.get('ok', 0)}, no_match {counts.get('no_match', 0)}, "
          f"invalid {counts.get('invalid', 0)}"
          + "".join(f", {k} {v}" for k, v in counts.items()
                    if k not in {"ok", "no_match", "invalid"}))
    print(f"Пройдено: {expect_pass}/{expect_total} ({share:.0%})"
          + (f", из них с опечатками {typo_pass}/{typo_total}" if typo_total else ""))
    print(f"Разбор условий совпал с эталоном: {parsed_same}/{parsed} "
          f"(пропущено условий {missed}, лишних {extra})")
    print(f"Нарушений условий запроса в выдаче (эталон): {real_total}")
    print(f"Нарушений отсёк код (против фильтров модели): {cut_total}")
    print(f"Чужих id поймала проверка: {rejected_total}")
    print(f"{seconds:.0f} с. Журнал: uv run show.py compare")


if __name__ == "__main__":
    main()
