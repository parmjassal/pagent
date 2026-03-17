import json
import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def robust_json_parser(text: str) -> Dict[str, Any]:
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_json_candidate(candidate: str) -> Optional[Dict[str, Any]]:
        """Try to load a JSON candidate safely. Return dict or first list item if dict."""
        if not candidate:
            return None

        candidate = candidate.strip()
        if not candidate:
            return None

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except Exception:
            return None

        return None

    def _extract_fenced_blocks(s: str) -> list[str]:
        """Extract full markdown fenced blocks, not regex-truncated pseudo-JSON."""
        return re.findall(r"```(?:json)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)

    def _extract_balanced_json(s: str) -> Optional[str]:
        """
        Extract the first balanced top-level JSON object/array.
        String-aware, so braces inside quoted strings do not break parsing.
        """
        start = None
        stack = []
        in_string = False
        escape = False

        for i, ch in enumerate(s):
            if escape:
                escape = False
                continue

            if in_string:
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch in "{[":
                if start is None:
                    start = i
                stack.append(ch)
                continue

            if ch in "}]":
                if not stack:
                    continue

                opening = stack.pop()
                if (opening == "{" and ch != "}") or (opening == "[" and ch != "]"):
                    # Mismatched bracket sequence
                    return None

                if not stack and start is not None:
                    return s[start:i + 1]

        return None

    # ------------------------------------------------------------------
    # 0. Normalize input
    # ------------------------------------------------------------------
    if text is None:
        logger.warning("Parser received None input")
        return {}

    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\ufeff", "").strip()

    if not text:
        logger.warning("Parser received empty string")
        return {}

    # ------------------------------------------------------------------
    # 0.5 Handle GLM / XML style <tool_call> outputs
    # ------------------------------------------------------------------
    if "<tool_call>" in text:
        try:
            tool_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL | re.IGNORECASE)
            if tool_match:
                block = tool_match.group(1).strip()

                tool_name_match = re.match(r"\s*([a-zA-Z0-9_\-]+)", block)
                tool_name = tool_name_match.group(1) if tool_name_match else "unknown_tool"

                args: Dict[str, Any] = {}
                arg_matches = re.findall(
                    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                    block,
                    re.DOTALL | re.IGNORECASE,
                )

                for key, value in arg_matches:
                    args[key.strip()] = value.strip()

                thought = re.sub(
                    r"<tool_call>.*?</tool_call>",
                    "",
                    text,
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()

                return {
                    "thought_process": thought,
                    "strategy": "tool_use",
                    "tool_call": {
                        "name": tool_name,
                        "args": args,
                    },
                }
        except Exception as e:
            logger.debug("Failed to parse XML tool_call: %s", e)

    # ------------------------------------------------------------------
    # 1. Strip reasoning / noise
    # ------------------------------------------------------------------
    text = re.sub(r"<(think|thought)>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\[Tool Result:.*?\](?:\n|\r\n)?.*?(?=\n\[|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"\[System\].*?(?=\n\[|$)", "", text, flags=re.DOTALL)
    text = text.strip()

    # ------------------------------------------------------------------
    # 2. Try fenced markdown blocks first
    # ------------------------------------------------------------------
    for block in _extract_fenced_blocks(text):
        parsed = _load_json_candidate(block)
        if parsed is not None:
            return parsed

        # Sometimes the fenced block contains extra prose + JSON
        candidate = _extract_balanced_json(block)
        if candidate:
            parsed = _load_json_candidate(candidate)
            if parsed is not None:
                return parsed

    # ------------------------------------------------------------------
    # 2.5 Handle unclosed markdown fence
    # ------------------------------------------------------------------
    open_fence = re.search(r"```(?:json)?\s*(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
    if open_fence:
        fenced_tail = open_fence.group(1).strip()

        parsed = _load_json_candidate(fenced_tail)
        if parsed is not None:
            return parsed

        candidate = _extract_balanced_json(fenced_tail)
        if candidate:
            parsed = _load_json_candidate(candidate)
            if parsed is not None:
                return parsed

    # ------------------------------------------------------------------
    # 3. Balanced JSON extraction from full text
    # ------------------------------------------------------------------
    candidate = _extract_balanced_json(text)
    if candidate:
        parsed = _load_json_candidate(candidate)
        if parsed is not None:
            return parsed

    # ------------------------------------------------------------------
    # 4. Escaped JSON string payloads
    # ------------------------------------------------------------------
    if text.startswith('"') and text.endswith('"'):
        try:
            unescaped = json.loads(text)
            if isinstance(unescaped, str):
                parsed = _load_json_candidate(unescaped)
                if parsed is not None:
                    return parsed
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 5. Last chance: direct full parse
    # ------------------------------------------------------------------
    parsed = _load_json_candidate(text)
    if parsed is not None:
        return parsed

    logger.error("Failed to parse JSON. Raw output:\n%s", text)
    return {}