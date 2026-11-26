# Киноманьяк

**Достигнутые уровни:** Common

Подбор 1–3 фильмов под сложный запрос **только** из локального `films.json`.  
Модель возвращает `film_id` + `why`; title / year / minutes / description печатает код из каталога.

Каркас: лекция 3 (fridge-chef) + семинар 3 (guardrails, cases, журнал).

## Запуск

Нужны [uv](https://docs.astral.sh/uv/) и ключ OpenAI. Локальный qwen — через [Ollama](https://ollama.com) (`ollama pull qwen3.5:4b`).

```bash
uv sync
cp .env.example .env   # впишите OPENAI_API_KEY
uv run kino.py "что посмотреть с мамой на казахском, до двух часов"
uv run show.py
uv run check.py          # весь queries.json на gpt-6-luna
uv run check.py --local  # тот же набор на qwen3.5:4b
uv run pytest -q
```

`kino.py` как `chef.py` на лекции: **без флагов** сначала `gpt-6-luna`, потом `qwen3.5:4b` (`reasoning_effort=none`).  
Облако — `parse` (Structured Outputs). Если Ollama даёт 501 — повтор через обычный `create`, JSON проверяем схемой `Choice` сами.  
Без Ollama вторая модель — «Нет связи с сервером».  
`check.py --local` — прогон набора только на qwen.

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
4. Чужой id → весь ответ `invalid`, пользователь **ничего** не видит; id в `rejected` (как в инструкции).
5. Печать карточки из каталога; `why` — от модели.

Порядок в `pick`: API → length → refusal → schema → status → allow-list → печать → журнал.

### Журнал

В `calls.jsonl` на каждый вызов: модель, **компактный** request (текст запроса + список id, не весь каталог), outcome, секунды, usage/$.

```bash
uv run show.py        # последние 10
uv run show.py all
```

### `expect` в `queries.json`

- `must` — среди picks должен быть **хотя бы один** id из списка
- `forbid` — ни одного из списка
- `no_match` — исход ровно `no_match`
- без `expect` — на счётчик pass/total не влияет

## Каталог

- `films.json`: 56 фильмов, 10 с «Казахстан» в `countries`.
- Год и minutes сверены с TMDB / Кинопоиск / Wiki (снимок ноябрь 2026); не из памяти модели.
- `canon.py` — фиксированный русскоязычный словарь жанров/языков/стран (задел под Epic). Сейчас `pick` его не импортирует: слова уже записаны в `films.json`.
- Обновление = новый снимок в git; `pick` в TMDB на каждый запрос не ходит.

## Пять вопросов лекции 1

1. **Цель.** Безопасный ответ на сложный запрос о кино без выдуманных фильмов.
2. **Без ML.** Фильтры по полям каталога. Не хватает семантики вроде «как Интерстеллар, но не про космос».
3. **Данные.** `films.json`: id, title, year, minutes, genres, countries, languages, description.
4. **Выдаёт.** 1–3 карточки или `no_match` / ошибка; журнал `calls.jsonl`.
5. **Цена ошибки.** Показать несуществующий фильм — потерять доверие; поэтому id сверяет код.

## Итог `check.py` (живой прогон)

Модель `gpt-6-luna`, 17 запросов (`queries.json`):

- исходы: **ok 13**, **no_match 4**, invalid 0
- чужих id: **0**
- ожидания: **13/14**
- ~40 с, ~$0.01

## Итог `show.py`

После этого прогона в журнале:

- 17 вызовов (ok 13, no_match 4)
- токены: вход ~93408, выход ~1425, рассуждения 0
- **~$0.0101**; дороже всего **вход** (весь каталог в промпте модели)

## Опыт с памятью модели

Без каталога `gpt-6-luna` угадала год и minutes для 10 фильмов.  
Сверка кодом: **4/10** расхождений (`memory_probe.json`).  
Поэтому на экран идут поля из каталога, не из ответа модели.

## Ответы на вопросы задания

1. **Почему id?** Название неоднозначно и галлюцинируется; id бьётся с каталогом. Память: 4/10 ошибок — `memory_probe.json`. Печать: `print_films` в `kino.py`.
2. **Чужой film_id.** Пользователь ничего не видит; `invalid` + `rejected`. Ловят `sanitize_picks` и `if rejected` в `pick`.
3. **«Без ошибки API» ≠ верный фильм.** HTTP/схема могут быть ок, а id чужой или мимо `must`. Это ловит `expect_ok` в `check.py`.
4. **Без ML.** Жёсткие фильтры по полям; на `queries.json` не хватает семантики и казахских формулировок.
5. **Цена.** ~$0.0006 на вызов при ~56 фильмах; доминирует `prompt_tokens`. См. `PRICE` в `kino.py`, `show.py`.

## Автопроверка

```bash
python3 "../[1] Киноманьяк/check/kino_check.py" .
```

## Структура

| Файл | Роль |
|---|---|
| `kino.py` | `pick`, промпт, схема, allow-list, журнал; `main` = облако + qwen |
| `films.json` | каталог |
| `canon.py` | словарь (под Epic; runtime Common не использует) |
| `queries.json` | набор + `expect` |
| `check.py` | прогон набора (облако или `--local`) |
| `show.py` | сводка журнала |
| `calls.jsonl` | журнал вызовов |
| `memory_probe.json` | опыт «память без каталога» |
| `tests/` | pytest без API |
| `SPEC.md` | контракт Common |
