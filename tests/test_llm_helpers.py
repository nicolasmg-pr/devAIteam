"""Unit tests for the LLM JSON-extraction helpers."""

import json

import pytest

from tools.llm_helpers import clean_llm_response, parse_llm_json


def test_plain_object():
    assert parse_llm_json('{"a": 1}') == {"a": 1}


def test_strips_markdown_fences():
    raw = '```json\n{"a": 1}\n```'
    assert parse_llm_json(raw) == {"a": 1}


def test_strips_think_block():
    raw = '<think>reasoning here</think>\n{"a": 1}'
    assert parse_llm_json(raw) == {"a": 1}


def test_unclosed_think_block_in_preamble():
    raw = '<think>cut off reasoning\n{"a": 1}'
    assert parse_llm_json(raw) == {"a": 1}


def test_braces_inside_string_values():
    # The naive first-{/last-} heuristic broke on this; the balanced scan must not.
    raw = 'prose {"regex": "[a-z]+", "code": "if (x) {y}"} trailing }'
    assert parse_llm_json(raw) == {"regex": "[a-z]+", "code": "if (x) {y}"}


def test_trailing_prose_with_braces():
    raw = '{"a": 1} Note: use {curly}'
    assert parse_llm_json(raw) == {"a": 1}


def test_top_level_array():
    raw = 'Here is the result: [1, 2, 3]'
    assert parse_llm_json(raw) == [1, 2, 3]


def test_nested_objects():
    obj = {"a": {"b": {"c": [1, {"d": "}"}]}}}
    raw = f"preamble {json.dumps(obj)} postscript"
    assert parse_llm_json(raw) == obj


def test_escaped_quote_inside_string():
    raw = '{"msg": "she said \\"hi\\" }"}'
    assert parse_llm_json(raw) == {"msg": 'she said "hi" }'}


def test_clean_returns_balanced_substring():
    raw = 'blah {"x": 1} blah'
    assert clean_llm_response(raw) == '{"x": 1}'


def test_repair_fallback_trailing_comma():
    # json-repair should rescue a trailing comma.
    raw = '{"a": 1, "b": 2,}'
    assert parse_llm_json(raw) == {"a": 1, "b": 2}


def test_unparseable_raises():
    with pytest.raises(Exception):
        parse_llm_json("this is not json at all")
