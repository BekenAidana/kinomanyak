"""Epic: candidates без модели и pick_big, который отдаёт модели только кандидатов."""

from __future__ import annotations

import json

from test_pick import Fake, picks

import kino


def film(i, title, description, genres=("драма",), minutes=100, languages=("русский",)):
    return {"id": f"b{i}", "title": title, "year": 2000 + i % 20, "minutes": minutes,
            "genres": list(genres), "countries": ["Россия"], "languages": list(languages),
            "description": description}


BIG = [film(i, f"Фильм {i}", f"Обычная история номер {i} о жизни людей в городе.")
       for i in range(40)]
BIG += [
    film(40, "Звёздный путь домой", "Экипаж корабля возвращается из далёкого космоса на Землю.",
         ("фантастика",)),
    film(41, "Окопы", "Солдаты держат оборону во время долгой войны.", ("военный", "драма")),
    film(42, "Тарих", "Историческая драма о казахских ханах и степи.", ("исторический",),
         languages=("казахский",)),
    film(43, "Длинная сага", "Эпическая сага о трёх поколениях семьи.", minutes=200),
]


def test_candidates_are_ids_from_catalog_and_at_most_k():
    found = kino.candidates("любой фильм", BIG, 30)
    assert 0 < len(found) <= 30
    assert set(found) <= {f["id"] for f in BIG}


def test_candidates_find_by_description_and_are_stable():
    found = kino.candidates("фильм про космос и корабль", BIG, 5)
    assert found[0] == "b40"
    assert kino.candidates("фильм про космос и корабль", BIG, 5) == found


def test_negated_part_does_not_pull_films_up():
    assert kino.search_text("драма, но не про космос") .strip().startswith("драма")
    assert "космос" not in kino.search_text("драма, но не про космос")


def test_kazakh_words_are_expanded_for_search():
    assert "исторический" in kino.search_text("тарихи кино")
    assert kino.candidates("тарихи кино", BIG, 3)[0] == "b42"


def test_candidates_apply_field_filters():
    found = kino.candidates("сага", BIG, 30, kino.Filters(max_minutes=120))
    assert "b43" not in found
    assert all(next(f for f in BIG if f["id"] == i)["minutes"] <= 120 for i in found)


def test_no_filters_same_as_empty_filters():
    # автопроверка зовёт candidates без фильтров, а pick_big — с Filters() от заглушки
    assert kino.candidates("война", BIG, 30) == kino.candidates("война", BIG, 30, kino.Filters())


def test_pick_big_sends_only_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    chosen = kino.candidates("про космос", BIG, kino.K)[:2]
    fake = Fake("parsed", picks(*chosen))
    outcome, ids = kino.pick_big(fake, "m", "про космос", BIG)
    assert (outcome, ids) == ("ok", chosen)
    prompt = fake.choice_prompts()[-1]
    sent = [f for f in BIG if json.dumps(f["title"], ensure_ascii=False) in prompt]
    assert len(sent) == min(kino.K, len(BIG))
    rows = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert rows[-1]["variant"] == "big" and rows[-1]["candidates"] == len(sent)


def test_pick_big_alien_id_is_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    first = kino.candidates("про космос", BIG, kino.K)[0]
    outcome, ids = kino.pick_big(Fake("parsed", picks(first, "alien-1")), "m", "про космос", BIG)
    assert (outcome, ids) == ("invalid", [])


def test_pick_whole_sends_whole_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    fake = Fake("parsed", picks("b1"))
    kino.pick_whole(fake, "m", "любой", BIG)
    prompt = fake.choice_prompts()[-1]
    assert all(json.dumps(f["title"], ensure_ascii=False) in prompt for f in BIG)
    assert not [c for c in fake.calls if c["kind"] == "filters"]


def test_country_adjective_expands_to_catalog_country():
    assert "СССР" in kino.search_text("советская комедия")
    assert "Казахстан" in kino.search_text("казахстанская драма")


def test_language_variants_map_to_catalog_word():
    import build_catalog as bc
    assert bc.language_word("американский вариант английского языка") == "английский"
    assert bc.language_word("австралийский английский") == "английский"
    assert bc.language_word("бразильский вариант португальского языка") == "португальский"
    assert bc.language_word("путунхуа") == "китайский"
    assert bc.language_word("американский жестовый") == "американский жестовый"


def test_english_title_and_words_are_searchable():
    films = BIG + [dict(film(44, "Начало", "Вор крадёт секреты из снов."), title_en="Inception")]
    assert kino.candidates("something like Inception", films, 3)[0] == "b44"
    assert "мультфильм" in kino.search_text("funny animated movie")
    assert "ужасы" not in kino.search_text("comedy, but not horror")


def test_main_big_flag_uses_big_catalog_and_pick_big(monkeypatch):
    import sys
    import kino

    calls = []
    monkeypatch.setattr(sys, "argv", ["kino.py", "--big", "фильм про акулу"])
    monkeypatch.setattr(kino, "OpenAI", lambda **kw: object())
    monkeypatch.setattr(kino, "load_films", lambda path=kino.FILMS_PATH: [{"path": str(path)}])
    monkeypatch.setattr(kino, "pick", lambda *a, **kw: calls.append(("pick", a[2], a[3])))
    monkeypatch.setattr(kino, "pick_big", lambda *a, **kw: calls.append(("pick_big", a[2], a[3])))
    kino.main()
    assert [c[0] for c in calls] == ["pick_big", "pick_big"]
    assert calls[0][1] == "фильм про акулу"
    assert calls[0][2][0]["path"].endswith("films_big.json")


def test_main_without_flag_stays_rare(monkeypatch):
    import sys
    import kino

    calls = []
    monkeypatch.setattr(sys, "argv", ["kino.py", "что посмотреть"])
    monkeypatch.setattr(kino, "OpenAI", lambda **kw: object())
    monkeypatch.setattr(kino, "load_films", lambda path=kino.FILMS_PATH: [{"path": str(path)}])
    monkeypatch.setattr(kino, "pick", lambda *a, **kw: calls.append(("pick", a[3])))
    monkeypatch.setattr(kino, "pick_big", lambda *a, **kw: calls.append(("pick_big", a[3])))
    kino.main()
    assert [c[0] for c in calls] == ["pick", "pick"]
    assert calls[0][1][0]["path"].endswith("films.json")
