"""Прогон queries.json через pick — набор проверок.

Один удачный запуск ничего не доказывает: гоняем весь набор и смотрим
исходы + expect (must / forbid / no_match). Проверяется всё приложение:
промпт, схема, allow-list, журнал — не «только модель».

uv run check.py
uv run check.py --local   локальная qwen3.5:4b
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

from kino import MODEL, load_films, pick

sys.stdout.reconfigure(encoding="utf-8")

QUERIES = Path("queries.json")
JOURNAL_NOTE = Path("calls.jsonl")


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


def main() -> None:
    items = json.loads(QUERIES.read_text(encoding="utf-8"))
    films = load_films()
    if "--local" in sys.argv:
        client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama",
                        max_retries=0)
        model = "qwen3.5:4b"
    else:
        try:
            client = OpenAI()
        except openai.OpenAIError:
            print("Нет ключа: скопируйте .env.example в .env и впишите OPENAI_API_KEY")
            return
        model = MODEL

    print(f"Модель {model}: {len(items)} запросов…\n")
    outcomes: list[str] = []
    rejected_total = 0
    expect_pass = expect_total = 0
    started = time.time()

    for item in items:
        query = item["query"]
        # сколько чужих id поймал allow-list в этой итерации (из новой строки журнала)
        before = JOURNAL_NOTE.read_text(encoding="utf-8") if JOURNAL_NOTE.exists() else ""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            outcome, ids = pick(client, model, query, films)
        outcomes.append(outcome)
        after = JOURNAL_NOTE.read_text(encoding="utf-8") if JOURNAL_NOTE.exists() else ""
        new_lines = after[len(before):].splitlines()
        for line in new_lines:
            if not line.strip():
                continue
            row = json.loads(line)
            rejected_total += len(row.get("rejected") or [])
        has_expect = bool(item.get("expect"))
        if has_expect:
            expect_total += 1
            ok = expect_ok(item, outcome, ids)
            expect_pass += int(ok)
            mark = "✅" if ok else "❌"
        else:
            mark = "—"
        print(f"{mark} {outcome:14}  {query[:60]}")

    counts = Counter(outcomes)
    seconds = time.time() - started
    print()
    print(f"Исходы: ok {counts.get('ok', 0)}, no_match {counts.get('no_match', 0)}, "
          f"invalid {counts.get('invalid', 0)}, "
          + ", ".join(f"{k} {v}" for k, v in counts.items()
                      if k not in {"ok", "no_match", "invalid"}))
    print(f"Чужих id поймала проверка: {rejected_total}")
    print(f"Запросов с ожиданием пройдено: {expect_pass}/{expect_total}")
    print(f"{seconds:.0f} с. Журнал: uv run show.py all")


if __name__ == "__main__":
    main()
