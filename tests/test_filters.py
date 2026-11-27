"""Rare: условия запроса проверяет код. Без сети, заглушка из test_pick.

Первый вызов (схема Filters) разбирает запрос, код отбирает каталог,
второй вызов выбирает среди кандидатов, код ещё раз сверяет ответ.
"""

from __future__ import annotations

import json

from test_pick import FILMS, Fake, picks

import kino


def journal(tmp_path):
    path = tmp_path / "calls.jsonl"
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ids(films):
    return [f["id"] for f in films]


def test_apply_filters_each_field():
    apply = kino.apply_filters
    assert ids(apply(FILMS, kino.Filters())) == ["a1", "b2", "c3"]
    assert ids(apply(FILMS, kino.Filters(max_minutes=95))) == ["b2"]
    assert ids(apply(FILMS, kino.Filters(year_from=2002, year_to=2002))) == ["b2"]
    assert ids(apply(FILMS, kino.Filters(languages=["казахский"]))) == ["a1"]
    assert ids(apply(FILMS, kino.Filters(exclude_genres=["комедия"]))) == ["a1", "c3"]
    assert ids(apply(FILMS, kino.Filters(must_genres=["триллер"]))) == ["c3"]


def test_words_outside_catalog_are_dropped():
    # qwen без списка слов писал kazakh / family — такой фильтр не должен
    # обнулить каталог: слово не из каталога = условия нет
    clean = kino.normalize_filters(
        kino.Filters(languages=["kazakh", "Казахский"], must_genres=["family"]),
        FILMS)
    assert clean.languages == ["казахский"]
    assert clean.must_genres is None


def test_long_film_cut_by_code_and_not_in_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    fake = Fake("parsed", picks("c3", "a1"), filters={"max_minutes": 100})
    outcome, shown = kino.pick(fake, "m", "до 100 минут", FILMS)
    assert outcome == "ok"
    assert shown == ["a1"]                      # c3 = 110 мин, код отсёк
    prompt = fake.choice_prompts()[-1]
    assert json.dumps("Гамма", ensure_ascii=False) not in prompt
    choice = journal(tmp_path)[-1]
    assert choice["step"] == "choice"
    assert choice["violations"] == ["c3"]


def test_journal_marks_filters_and_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    fake = Fake("parsed", picks("a1"), filters={"languages": ["казахский"]})
    kino.pick(fake, "m", "на казахском", FILMS)
    rows = journal(tmp_path)
    assert [r["step"] for r in rows] == ["filters", "choice"]
    assert rows[0]["filters"] == {"languages": ["казахский"]}
    assert "на казахском" in rows[0]["request"]
    assert "на казахском" in rows[1]["request"]
    assert rows[1]["candidates"] == 1
    assert rows[1]["shown"] == ["a1"]


def test_no_candidates_means_no_match_without_choice_call(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    fake = Fake("parsed", picks("a1"), filters={"max_minutes": 10})
    outcome, shown = kino.pick(fake, "m", "до 10 минут", FILMS)
    assert (outcome, shown) == ("no_match", [])
    assert fake.choice_prompts() == []
    choice = journal(tmp_path)[-1]
    assert choice["step"] == "choice" and choice["candidates"] == 0


def test_broken_filters_fall_back_to_full_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    fake = Fake("parsed", picks("c3"), filters_action="invalid")
    outcome, shown = kino.pick(fake, "m", "любой", FILMS)
    assert (outcome, shown) == ("ok", ["c3"])
    rows = journal(tmp_path)
    assert rows[0]["step"] == "filters" and rows[0]["outcome"] == "invalid"
    assert rows[1]["candidates"] == 3


def test_foreign_id_is_invalid_not_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    fake = Fake("parsed", picks("a1", "alien-7"), filters={"max_minutes": 200})
    outcome, shown = kino.pick(fake, "m", "любой", FILMS)
    assert (outcome, shown) == ("invalid", [])
    choice = journal(tmp_path)[-1]
    assert choice["rejected"] == ["alien-7"]


def test_genre_in_must_and_exclude_is_excluded():
    clean = kino.normalize_filters(
        kino.Filters(must_genres=["комедия", "драма"], exclude_genres=["комедия"]),
        FILMS)
    assert clean.must_genres == ["драма"]
    assert clean.exclude_genres == ["комедия"]


def test_tagged_lines_parsed_when_model_skips_json():
    class Message:
        parsed = None
        content = '<max_minutes>: 120\n<languages>: ["казахский"]\n<year_from>: null'

    filters = kino.parsed_from(Message(), kino.Filters)
    assert filters.max_minutes == 120
    assert filters.languages == ["казахский"]
    assert filters.year_from is None


def test_whole_vocabulary_in_must_genres_is_ignored():
    films = FILMS + [{"id": "d4", "title": "Дельта", "year": 2004, "minutes": 95,
                      "genres": ["ужасы", "мюзикл"], "countries": ["США"],
                      "languages": ["английский"], "description": "Описание дельты."}]
    clean = kino.normalize_filters(
        kino.Filters(must_genres=["драма", "комедия", "триллер", "ужасы"]), films)
    assert clean.must_genres is None


def test_min_rating_needs_known_rating():
    films = [dict(FILMS[0], rotten_tomatoes=90), dict(FILMS[1], rotten_tomatoes=60),
             dict(FILMS[2], rotten_tomatoes=None)]
    # нет рейтинга в источнике → условие проверить нечем → фильм не проходит
    assert ids(kino.apply_filters(films, kino.Filters(min_rating=80))) == ["a1"]


def test_min_rating_normalized():
    assert kino.normalize_filters(kino.Filters(min_rating=8), FILMS).min_rating == 80
    assert kino.normalize_filters(kino.Filters(min_rating=150), FILMS).min_rating is None


def test_journal_records_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    kino.pick(Fake("parsed", picks("a1")), "m", "любой", FILMS, effort="low")
    for row in journal(tmp_path):
        assert row["effort"] == "low"
        assert row["limit"] == kino.LIMIT
        assert row["prompts"] == kino.PROMPTS


def test_language_needs_language_phrase_not_country_adjective():
    named = kino.language_named
    assert named("казахский", "что посмотреть на казахском")
    assert named("казахский", "қазақша драма")
    assert named("казахский", "қазақ тіліндегі кино")
    assert named("английский", "комедия по-английски")
    assert not named("казахский", "казахстанское кино")
    assert not named("французский", "лёгкая французская комедия")
    assert not named("японский", "японский мультфильм")


def test_country_adjective_does_not_become_language_filter():
    films = FILMS + [dict(FILMS[0], id="d4", languages=["японский"])]
    clean = kino.normalize_filters(
        kino.Filters(languages=["японский"], must_genres=["драма"]), films,
        "японская драма")
    assert clean.languages is None
    assert clean.must_genres == ["драма"]


def test_quote_with_typo_keeps_language():
    assert kino.language_evidence("казахский", "казакша комедия", "казакша")
    assert kino.language_evidence("английский", "фильм на англиском 90х", "на англиском")
    assert kino.language_evidence("казахский", "kazaksha drama", "kazaksha")


def test_quote_must_be_in_query_and_look_like_language_phrase():
    # выдуманная цитата: в запросе её нет
    assert not kino.language_evidence("казахский", "драма про степь", "на казахском")
    # прилагательное страны — не языковой оборот
    assert not kino.language_evidence("французский", "французская комедия", "французская")
    # оборот есть, но про другое
    assert not kino.language_evidence("английский", "фильм на Луне", "на Луне")


def test_quote_is_not_compared_with_gold():
    item = {"query": "казакша драма", "conditions": {"languages": ["казахский"]}}
    report = check_parse(item, {"languages": ["казахский"], "language_quote": "казакша"})
    assert report["same"] and report["extra"] == []


def check_parse(item, filters):
    import check
    return check.parse_report(item, filters, FILMS + [dict(FILMS[0], id="k1",
                                                           languages=["казахский"])])


def test_kazakh_language_form_with_one_typo():
    assert kino.one_edit_apart("казашка", "казакша")          # перестановка
    assert kino.language_evidence("казахский", "казашка жапондых аниме", "казашка")
    assert kino.language_evidence("казахский", "казахша драма", "казахша")
    # «про казашку» — о женщине, не о языке: две правки, не проходит
    assert not kino.language_evidence("казахский", "фильм про казашку", "казашку")
