"""Rotten Tomatoes из Wikidata: берётся оценка критиков, а не зрителей."""
import build_catalog as bc

AUDIENCE = "Q108403540"


def rt_claim(score: str, method: str) -> dict:
    qualifier = lambda qid: {"snaktype": "value", "datavalue": {"value": {"id": qid}}}
    return {"rank": "normal",
            "mainsnak": {"snaktype": "value", "datavalue": {"value": score}},
            "qualifiers": {"P447": [qualifier(bc.ROTTEN_TOMATOES)], "P459": [qualifier(method)]}}


def rating(*claims: dict) -> dict:
    return bc.facts_of({"claims": {"P444": list(claims)}}, labels={})


def test_tomatometer_wins_in_any_claim_order():
    critics, audience = rt_claim("91%", bc.TOMATOMETER), rt_claim("78%", AUDIENCE)
    assert rating(critics, audience)["rotten_tomatoes"] == 91
    assert rating(audience, critics)["rotten_tomatoes"] == 91


def test_only_audience_score_gives_no_rating():
    assert rating(rt_claim("78%", AUDIENCE))["rotten_tomatoes"] is None


def test_conflicting_tomatometers_are_reported_not_guessed():
    fact = rating(rt_claim("91%", bc.TOMATOMETER), rt_claim("89%", bc.TOMATOMETER))
    assert fact["rotten_tomatoes"] is None
    assert fact["_many_ratings"] == [89, 91]
