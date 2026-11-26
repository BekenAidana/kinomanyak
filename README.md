# Киноманьяк

**Достигнутые уровни:** Common

Подбор 1–3 фильмов под сложный запрос **только** из локального `films.json`.  
Модель возвращает `film_id` + `why`; title / year / minutes / description печатает код из каталога.

Каркас: лекция 3 (fridge-chef) + семинар 3 (guardrails, cases, журнал).

## Запуск

Нужны [uv](https://docs.astral.sh/uv/) и ключ OpenAI.

```bash
uv sync
cp .env.example .env   # впишите OPENAI_API_KEY
uv run kino.py "что посмотреть с мамой на казахском, до двух часов"
uv run show.py
uv run check.py
uv run pytest -q
```

Модель по умолчанию: `gpt-6-luna`. Опционально локальная: `uv run kino.py "…" --local` (Ollama `qwen3.5:4b`, `reasoning_effort=none`).

## Как устроен проект

| Лекция / семинар | У нас | Зачем |
|---|---|---|
| `cook` | `pick` в `kino.py` | вызов модели + ветки ошибок |
| `guardrails.py` | `sanitize_picks` | allow-list id ∈ каталог |
| `calls.jsonl` + `show.py` | то же | журнал исходов, токенов, $ |
| `cases.py` | `queries.json` | запросы + `expect` |
| `check.py` | `check.py` | прогон всего приложения |
| `test_guardrails.py` | `tests/test_pick.py` | pytest без сети |
| каталог | `films.json` + `canon.py` | факты не из памяти LLM; словарь под Epic |

### Guardrails

Промпт не гарантирует защиту. Код:

1. system ≠ данные; `<каталог>` / `<запрос>` — только данные.
2. Схема `Choice`: `status` ok \| no_match, `picks[{film_id, why}]`. Схема ≠ правда.
3. Allow-list: только id из переданного `films`, без повторов, ≤3.
4. Чужой id → весь ответ `invalid`, пользователь **ничего** не видит; id в `rejected`.
5. Печать карточки из каталога; `why` — от модели.

Порядок в `pick`: API → length → refusal → schema → status → allow-list → печать → журнал.

### Журнал

В `calls.jsonl`: модель, компактный request (запрос + список id), outcome, секунды, usage/$.

```bash
uv run show.py
uv run show.py all
```

### `expect` в `queries.json`

- `must` — среди picks должен быть хотя бы один id из списка
- `forbid` — ни одного из списка
- `no_match` — исход ровно `no_match`
- без `expect` — на счётчик pass/total не влияет

## Каталог

- `films.json`: 56 фильмов, 10 с «Казахстан» в `countries`.
- Год и minutes сверены с TMDB / Кинопоиск / Wiki (снимок ноябрь 2026).
- `canon.py` — словарь жанров/языков/стран (задел под Epic; `pick` пока не импортирует).
- Обновление = новый снимок в git; live TMDB в `pick` нет.

## Пять вопросов лекции 1

1. **Цель.** Безопасный ответ на сложный запрос о кино без выдуманных фильмов.
2. **Без ML.** Фильтры по полям каталога. Не хватает семантики вроде «как Интерстеллар, но не про космос».
3. **Данные.** `films.json`: id, title, year, minutes, genres, countries, languages, description.
4. **Выдаёт.** 1–3 карточки или `no_match` / ошибка; журнал `calls.jsonl`.
5. **Цена ошибки.** Показать несуществующий фильм — потерять доверие; id сверяет код.

## Итог `check.py` (живой прогон)

Модель `gpt-6-luna`, 17 запросов:

- исходы: **ok 13**, **no_match 4**, invalid 0
- чужих id: **0**
- ожидания: **13/14**
- ~40 с, ~$0.01

## Итог `show.py`

- 17 вызовов (ok 13, no_match 4)
- токены: вход ~93408, выход ~1425, рассуждения 0
- **~$0.0101**; дороже всего **вход** (каталог в промпте)

## Опыт с памятью модели

Без каталога `gpt-6-luna`: **4/10** расхождений по году/minutes (`memory_probe.json`).  
Поэтому на экран идут поля из каталога.

## Ответы на вопросы задания

1. **Почему id?** Название неоднозначно; id бьётся с каталогом. Память 4/10 — `memory_probe.json`. Печать: `print_films` в `kino.py`.
2. **Чужой film_id.** Пользователь ничего не видит; `invalid` + `rejected`. Ловят `sanitize_picks` и `if rejected` в `pick`.
3. **«Без ошибки API» ≠ верный фильм.** Ловит `expect_ok` в `check.py` (`must` / `forbid` / `no_match`).
4. **Без ML.** Жёсткие фильтры; на `queries.json` не хватает семантики.
5. **Цена.** ~$0.0006 на вызов; доминирует `prompt_tokens`. См. `PRICE` в `kino.py`, `show.py`.

## Автопроверка

```bash
python3 "../[1] Киноманьяк/check/kino_check.py" .
```

## Структура

| Файл | Роль |
|---|---|
| `kino.py` | `pick`, промпт, схема, allow-list, журнал |
| `films.json` | каталог |
| `canon.py` | словарь (под Epic) |
| `queries.json` | набор + `expect` |
| `check.py` | прогон набора |
| `show.py` | сводка журнала |
| `calls.jsonl` | журнал |
| `memory_probe.json` | опыт памяти |
| `tests/` | pytest без API |
| `SPEC.md` | контракт Common |
