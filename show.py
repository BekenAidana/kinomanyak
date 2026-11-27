"""Журнал вызовов из calls.jsonl

Зачем: видеть исходы, токены и $ без повторного вызова модели.
Сдача: после check.py журнал показывает весь прогон queries.json.

uv run show.py        последние 10 вызовов и итог
uv run show.py all    все вызовы
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
    print(f"{'время':19}  {'модель':14}  {'исход':14}  {'вход':>5}  {'выход':>5}  "
          f"{'рассуждения':>11}  {'сек':>5}  {'$':>8}")
    for row in rows:
        usage = row.get("usage") or {}
        print(f"{row.get('time', ''):19}  {str(row.get('model', '')):14}  "
              f"{str(row.get('outcome', '')):14}  "
              f"{input_tokens(usage):>5}  {output_tokens(usage):>5}  "
              f"{reasoning(usage):>11}  {row.get('seconds', 0):>5}  "
              f"{row.get('usd', 0) or 0:>8.5f}")


def summary(rows):
    outcomes = Counter(row.get("outcome") for row in rows)
    usages = [row.get("usage") or {} for row in rows]
    total_in = sum(input_tokens(u) for u in usages)
    total_out = sum(output_tokens(u) for u in usages)
    total_reasoning = sum(reasoning(u) for u in usages)
    usd = sum(row.get("usd") or 0 for row in rows)
    rejected = sum(len(r.get("rejected") or []) for r in rows
                   if isinstance(r.get("rejected"), list))
    print()
    print(f"Вызовов: {len(rows)} ("
          + ", ".join(f"{name} {count}" for name, count in outcomes.most_common())
          + ")")
    print(f"Токенов: вход {total_in}, выход {total_out}, "
          f"из них рассуждения {total_reasoning}")
    print(f"Отклонённых id: {rejected}")
    print(f"Стоимость всего журнала: ${usd:.4f}")


def main():
    if not JOURNAL.exists():
        print("Журнал пуст: файла calls.jsonl ещё нет")
        return
    lines = JOURNAL.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        print("Журнал пуст")
        return
    table(rows if "all" in sys.argv[1:] else rows[-LAST:])
    summary(rows)


if __name__ == "__main__":
    main()
