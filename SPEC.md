# SPEC.md — Киноманьяк (Common)

Версия: 0.3.1  
Статус: контракт перед реализацией (код не начат)  
Сверено с: инструкция (6 стр.) + семинар 3 mentor solutions + решения по каталогу (TMDB → Epic)  
Объём сдачи сейчас: **Common = 100%**  
Дальше по желанию: Rare / **Epic** / Legendary (независимы)  
Дедлайн LMS: до 29 ноября включительно  
Репозиторий: https://github.com/BekenAidana/kinomanyak  

При расхождении с кодом верны: инструкция + `check/kino_check.py` + `kino_probe.py`.

**Основа:** `chef.py` лекции 3 (+ ДЗ «Киноманьяк v0»).  
**Стек:** Python 3.12, `uv`, OpenAI SDK, Pydantic; облако (напр. `gpt-6-luna`) или `qwen3.5:4b` (Ollama).  
**Сдача:** git-ссылка; ментор без защиты. В начале README — достигнутые уровни.

---

## 1. Цель

Подбор 1–3 фильмов под сложный запрос **только из локального каталога**.  
Модель возвращает `film_id` + `why`; код сверяет id и печатает title / year / minutes / description из каталога.  
Нет подходящих → `no_match`.

---

## 2. Каталог и данные (решение проекта)

### 2.1 Источники

| Роль | Источник |
|---|---|
| Основной пайплайн / Epic later | **TMDB** (`with_origin_country=KZ`, discover, детали фильма) |
| Сверка фактов глазами (особенно KZ) | Кинопоиск и/или Википедия |
| Что сдаём в git | **снимок** `films.json` (не live-API в `pick`) |

- Кинопоиск **не** используем как скрапер/основной API (ToS, нет чистой лицензии для Epic).
- В README: источники фактов + дата сверки (TMDB ± Кинопоиск/Wiki).

### 2.2 Канон словаря (сразу под Epic)

Жанры, языки, страны в `films.json` — **фиксированный русскоязычный канон**  
(напр. `«драма»`, `«казахский»`, `«Казахстан»`).  

Позже `films_big.json` маппит EN-поля датасета **в этот же канон**  
(`Drama` → `драма`), иначе Rare-фильтры и Epic-поиск разъедутся.

### 2.3 Id и обновления

- Стабильные id в Common (напр. `tmdb-19495` или свой префикс) — **не менять** при рефреше метаданных.
- Обновление данных = скрипт → новый снимок → коммит.  
  **Не** ходить в TMDB на каждый `uv run kino.py` (ломает `expect` и воспроизводимость).
- Epic отдельно: `films_big.json` (1000+), свой namespace id (`tmdb-…`), тот же набор полей.

### 2.4 Объём Common

- [ ] 40–60 настоящих фильмов
- [ ] ≥5 с `countries`, содержащим **«Казахстан»** (TMDB KZ + добор глазами при дырках)
- [ ] Поля: `id`, `title`, `year`, `minutes`, `genres`, `countries`, `languages`, `description`
- [ ] Год/длительность не из памяти LLM

---

## 3. Стек и поставка

| Решение | Выбор |
|---|---|
| Python | 3.12 (`.python-version`) |
| Пакеты | `uv` + `uv.lock` |
| LLM | OpenAI SDK + Pydantic, `chat.completions.parse` |
| Каталог (сбор) | TMDB API-ключ локально (не в git); снимок в репо |
| Секреты | `.env` / `.env.example` |

Файлы HEAD: `pyproject.toml` (+ `pythonpath = ["."]`), `uv.lock`, `.python-version`, `.env.example`, `README.md`, `kino.py`, `check.py`, `show.py`, `films.json`, `queries.json`, `calls.jsonl`, `tests/`.

- [ ] нет `.env` и ключей в git/истории

---

## 4. Лекция 3 + семинар 3 (правила)

Эталон ментора (ответы к семинару 3):  
`Lessons/[3] AI API calls/Seminar/seminar3_mentor/seminar3_mentor/fridge-chef-solutions/`  
(+ презентация: `Seminar/seminar_03_presentation.html` — слайд «Домашка Киноманьяк: тот же каркас»).

| Файл ментора | Зачем нам в Киноманьяке |
|---|---|
| `chef.py` | порядок веток: API → length → refusal → status → output check → allow-list → печать; `log` возвращает строку |
| `guardrails.py` | идея до/после модели; у нас главный allow-list = id ∈ `films` |
| `cases.py` | образец набора с несколькими допустимыми `expect` |
| `check.py` | прогон набора, сводка pass/токены/$ |
| `attacks.py` | какие атаки ловить кодом (injection, подмена тегов) |
| `prompts.py` | one-change эксперименты (для улучшения промпта по цифрам) |
| `show.py` / `test_guardrails.py` | журнал и тесты без сети |

- [ ] system ≠ user-данные; теги = данные
- [ ] Structured Outputs; схема ≠ правда → код
- [ ] порядок проверок как у ментора (непроверенное на экран не попадает)
- [ ] `status` ≠ `refusal`; API errors не роняют процесс
- [ ] `ValidationError` → `invalid`
- [ ] `calls.jsonl` + `show.py`
- [ ] allow-list id (не stop-list) — сильнее `FORBIDDEN`
- [ ] `qwen`: `reasoning_effort="none"`
- [ ] `Literal` на `film_id` — опция, не замена коду

---

## 5. Common — продукт (инструкция п.1–11)

### Запуск и `pick`

```text
pick(client, model, query, films) -> (outcome, ids)
```

- [ ] `uv run kino.py "запрос"`
- [ ] импорт без side effects
- [ ] печать **внутри pick**: title, year, **minutes**, description; `why` от модели
- [ ] кроме `ok` — `ids` пустой
- [ ] не читает `films.json` сама; работает с переданным `films`
- [ ] промпт: `<каталог>` = `json.dumps(films, ensure_ascii=False)`, `<запрос>`, правило «теги = данные», **один пример**
- [ ] схема: `status` ok|no_match, `reason`, `picks[{film_id, why}]`
- [ ] код: только id из каталога, без повторов, ≤3; чужой → `invalid` + `rejected`
- [ ] исходы: `ok` | `no_match` | `invalid` | `refusal` | `length` | `no_connection` | `error_<код>`

### Журнал

- [ ] строка: `model`, `request`, `outcome` (+ `rejected` при invalid; желательно `usage`)
- [ ] `request` компактно: текст запроса + список id каталога (не раздувать мегабайтами)
- [ ] сданный журнал = прогон `check.py` по всему `queries.json`

### `queries.json` (15+)

- [ ] ≥3 `no_match`
- [ ] ≥3 `must`
- [ ] 3 на казахском
- [ ] 3 с «не/без/кроме»
- [ ] 1 провокация (игнор каталога / «Аватар 7»)

### `check.py` / тесты / README

- [ ] check печатает: ok / no_match / invalid, чужие id, сколько expect пройдено
- [ ] ≥3 pytest: чужой id, повтор, no_match (заглушка, без API)
- [ ] опыт **без каталога**: 10 фильмов, год+minutes vs каталог кодом → число в README Q1
- [ ] README: уровни; 5 вопросов L1; запуск; check/show; память модели; источник каталога; 5 вопросов задания со ссылками на код

---

## 6. Автопроверка S01–S21

- [ ] S01–S21 зелёные (`kino_check` без `--live`)
- [ ] ментор: `--live`, 3 фильма vs источник, README

Возврат: фильм не из каталога; проверка только в промпте; ключ в git; фейковые цифры; не стартует по README; сломан контракт.

---

## 7. Поток

```text
запрос → system+пример + <каталог>+<запрос>
→ parse(schema) → errors/refusal/length?
→ no_match? → allow-list / dedupe / ≤3
→ печать из каталога → calls.jsonl
```

---

## 8. Дальше: Epic (заложено, не делаем сейчас)

- [ ] `films_big.json` 1000+ из TMDB (или иного open dataset), лицензия в README
- [ ] маппинг жанров/языков/стран → **канон §2.2**
- [ ] `candidates(query, films, k=30)` без LLM; путь для RU/KK vs EN descriptions (эмбеддинги или перевод в файл)
- [ ] `pick_big` только на кандидатах; два замера vs «весь каталог в промпте»

Rare / Legendary — отдельные бонусы (Filters; Telegram+L5 process).

---

## 9. Источники

- Инструкция для студентов.pdf  
- `check/kino_check.py`, `kino_probe.py`  
- Лекция 3: `fridge-chef/`  
- Семинар 3 (студенты): `Seminar/fridge-chef-seminar/`  
- **Семинар 3 (mentor answers):** `Seminar/seminar3_mentor/.../fridge-chef-solutions/` + `seminar_03_presentation.html`  
- Решения проекта: TMDB primary, Кинопоиск только сверка, снимки в git, канон под Epic  
