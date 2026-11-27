"""Метрики check.py против эталонных условий (conditions в queries.json)."""

from __future__ import annotations

from test_pick import FILMS

import check
import kino

KNOWN = kino.by_id(FILMS)


def test_real_violation_uses_true_conditions_not_model_filters():
    # модель потеряла «до 95 минут» → код ничего не отсёк, но эталон видит нарушение
    item = {"query": "до 95 минут", "conditions": {"max_minutes": 95}}
    assert check.real_violations(item, ["b2", "c3"], KNOWN) == ["c3"]


def test_no_conditions_means_no_real_violations():
    assert check.real_violations({"query": "любой"}, ["c3"], KNOWN) == []


def test_parse_report_compares_candidates_not_spelling():
    item = {"query": "2002 год", "conditions": {"year_from": 2002, "year_to": 2002}}
    # только year_from=2002 даёт b2 и c3, эталон — только b2: разбор не совпал
    report = check.parse_report(item, {"year_from": 2002}, FILMS)
    assert report == {"same": False, "missed": ["year_to"], "extra": []}
    # year_from=2002 + year_to=2002.. то же самое, что эталон
    same = check.parse_report(item, {"year_from": 2002, "year_to": 2002}, FILMS)
    assert same["same"] is True


def test_parse_report_counts_extra_condition():
    item = {"query": "драма", "conditions": {}}
    report = check.parse_report(item, {"languages": ["казахский"]}, FILMS)
    assert report["same"] is False
    assert report["extra"] == ["languages"]
