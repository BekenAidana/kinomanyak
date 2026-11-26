"""Тесты pick с заглушкой клиента — без сети и без ключа.

Аналог test_guardrails.py на семинаре: проверяем код вокруг модели
(чужой id → invalid, повтор, no_match, ≤3, теги в промпте), не сам LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import openai
from openai.types.chat import ChatCompletion

import kino

try:
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

try:
    from openai.lib._parsing._completions import parse_chat_completion
except ImportError:  # pragma: no cover
    parse_chat_completion = None

OMIT = getattr(openai, "omit", openai.NOT_GIVEN)
REQUEST = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions")

FILMS = [
    {"id": "a1", "title": "Альфа", "year": 2001, "minutes": 100,
     "genres": ["драма"], "countries": ["Казахстан"], "languages": ["казахский"],
     "description": "Описание альфы."},
    {"id": "b2", "title": "Бета", "year": 2002, "minutes": 90,
     "genres": ["комедия"], "countries": ["США"], "languages": ["английский"],
     "description": "Описание беты."},
    {"id": "c3", "title": "Гамма", "year": 2003, "minutes": 110,
     "genres": ["триллер"], "countries": ["Россия"], "languages": ["русский"],
     "description": "Описание гаммы."},
]


class Completions:
    def __init__(self, owner):
        self.owner = owner

    def parse(self, **kwargs):
        return self.owner.parse(**kwargs)


class Chat:
    def __init__(self, owner):
        self.completions = Completions(owner)


class Fake:
    def __init__(self, action, payload=None):
        self.action = action
        self.payload = payload
        self.chat = Chat(self)
        self.calls = []

    def parse(self, *, messages=None, response_format=None, model=None, **_):
        prompt = "\n".join(str(m.get("content", "")) for m in messages or [])
        self.calls.append({"prompt": prompt, "model": model, "schema": response_format})
        if self.action == "connection":
            raise openai.APIConnectionError(request=REQUEST)
        if self.action == "status":
            response = httpx.Response(self.payload, request=REQUEST)
            raise openai.APIStatusError(
                f"Error code: {self.payload}", response=response, body=None)
        if self.action == "refusal":
            return self._complete(response_format, None, prompt, model,
                                  refusal="Не могу.")
        if self.action == "length":
            return self._complete(response_format, self.payload, prompt, model,
                                  finish="length")
        if self.action == "parsed":
            return self._complete(response_format, self.payload, prompt, model)
        raise AssertionError(f"unknown action {self.action}")

    def _complete(self, schema, payload, prompt, model, refusal=None, finish="stop"):
        content = None if payload is None else json.dumps(payload, ensure_ascii=False)
        if finish == "length":
            content = (content or "")[: max(len(content or "") // 2, 1)]
        raw = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": model or "model",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content,
                                     "refusal": refusal}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        completion = ChatCompletion.model_validate(raw)
        if parse_chat_completion is None or schema is None:
            return completion
        if finish == "length":
            raise openai.LengthFinishReasonError(completion=completion)
        return parse_chat_completion(
            response_format=schema, chat_completion=completion, input_tools=OMIT)


def picks(*ids, status="ok"):
    return {"status": status, "reason": "проверка",
            "picks": [{"film_id": i, "why": "проверка"} for i in ids]}


def test_ok_prints_catalog_fields(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    outcome, ids = kino.pick(Fake("parsed", picks("a1", "b2")), "m", "драма", FILMS)
    assert outcome == "ok"
    assert ids == ["a1", "b2"]
    out = capsys.readouterr().out
    assert "Альфа" in out and "2001" in out and "Описание альфы." in out
    assert "100 мин" in out


def test_alien_id_invalid_and_rejected(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    outcome, ids = kino.pick(
        Fake("parsed", picks("a1", "alien-99")), "m", "любой", FILMS)
    assert outcome == "invalid"
    assert ids == []
    rows = [json.loads(line) for line in
            (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert rows[-1]["outcome"] == "invalid"
    assert "alien-99" in [str(x) for x in rows[-1]["rejected"]]


def test_duplicate_not_shown_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    outcome, ids = kino.pick(Fake("parsed", picks("a1", "a1")), "m", "повтор", FILMS)
    assert outcome == "ok"
    assert ids == ["a1"]


def test_no_match(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    outcome, ids = kino.pick(
        Fake("parsed", picks(status="no_match")), "m", "мульт 2031", FILMS)
    assert outcome == "no_match"
    assert ids == []


def test_max_three(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    outcome, ids = kino.pick(
        Fake("parsed", picks("a1", "b2", "c3", "a1")), "m", "много", FILMS)
    assert outcome == "ok"
    assert ids == ["a1", "b2", "c3"]


def test_tags_and_full_catalog_in_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(kino, "JOURNAL", tmp_path / "calls.jsonl")
    fake = Fake("parsed", picks("a1"))
    kino.pick(fake, "m", "особо-уникальный-запрос", FILMS)
    prompt = fake.calls[-1]["prompt"]
    assert "<каталог>" in prompt and "</каталог>" in prompt
    assert "<запрос>" in prompt and "особо-уникальный-запрос" in prompt
    assert json.dumps("Альфа", ensure_ascii=False) in prompt
    assert json.dumps("Бета", ensure_ascii=False) in prompt


def test_import_has_pick():
    assert callable(kino.pick)
