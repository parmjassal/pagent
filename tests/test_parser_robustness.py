import pytest
from agent_platform.runtime.core.parser import robust_json_parser

def test_parse_clean_json():
    text = '{"thought_process": "thinking", "strategy": "finish"}'
    res = robust_json_parser(text)
    assert res["strategy"] == "finish"

def test_parse_with_think_tags():
    text = '<think>I should do this</think>{"thought_process": "thinking", "strategy": "finish"}'
    res = robust_json_parser(text)
    assert res["strategy"] == "finish"
    assert "thought_process" in res

def test_parse_with_thought_tags():
    text = '<thought>I am thinking</thought>\n```json\n{"thought_process": "thinking", "strategy": "finish"}\n```'
    res = robust_json_parser(text)
    assert res["strategy"] == "finish"

def test_parse_with_markdown_fence():
    text = 'Here is my plan:\n```json\n{"thought_process": "thinking", "strategy": "finish"}\n```\nHope it helps.'
    res = robust_json_parser(text)
    assert res["strategy"] == "finish"

def test_parse_with_trailing_comma():
    text = '{"thought_process": "thinking", "strategy": "finish",}'
    res = robust_json_parser(text)
    assert res["strategy"] == "finish"

def test_parse_chatty_fallback():
    text = 'My thought_process is "exploring" and my strategy is "finish".'
    res = robust_json_parser(text)
    assert res["thought_process"] == "exploring"
    assert res["strategy"] == "finish"

def test_parse_python_dict_style():
    text = "{'thought_process': 'thinking', 'strategy': 'finish', 'flag': True}"
    res = robust_json_parser(text)
    assert res["strategy"] == "finish"
    assert res["flag"] is True

def test_parse_invalid_raises():
    text = 'Just some text with no fields.'
    with pytest.raises(Exception):
        robust_json_parser(text)
