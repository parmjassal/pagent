import json
import re
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def robust_json_parser(text: str) -> Dict[str, Any]:
    
    # ------------------------------------------------------------------
    # 0.5 Handle GLM / XML style <tool_call> outputs
    # ------------------------------------------------------------------
    if "<tool_call>" in text:
        try:
            tool_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)

            if tool_match:
                block = tool_match.group(1).strip()

                # Extract tool name (first token before arg tags)
                tool_name_match = re.match(r"\s*([a-zA-Z0-9_\-]+)", block)
                tool_name = tool_name_match.group(1) if tool_name_match else "unknown_tool"

                # Extract args
                args = {}

                arg_matches = re.findall(
                    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                    block,
                    re.DOTALL,
                )

                for key, value in arg_matches:
                    args[key.strip()] = value.strip()

                # Everything outside tool_call becomes thought_process
                thought = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()

                return {
                    "thought_process": thought,
                    "strategy": "tool_use",
                    "tool_call": {
                        "name": tool_name,
                        "args": args
                    }
                }

        except Exception as e:
            logger.debug("Failed to parse XML tool_call: %s", e)

    # ------------------------------------------------------------------
    # 0. Normalize text
    # ------------------------------------------------------------------
    if text is None:
        logger.warning("Parser received None input")
        return {}

    if not isinstance(text, str):
        text = str(text)   

    text = text.replace("\ufeff", "")  # Remove BOM
    text = text.strip()

    if not text:
        logger.warning("Parser received empty string")
        return {}

    # ------------------------------------------------------------------
    # 1. Strip reasoning tags
    # ------------------------------------------------------------------
    text = re.sub(r'<(think|thought)>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # ------------------------------------------------------------------
    # 1.5 Remove hallucinated tool outputs
    # ------------------------------------------------------------------
    text = re.sub(r'\[Tool Result:.*?\](?:\n|\r\n)?.*?(?=\n\[|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'\[System\].*?(?=\n\[|$)', '', text, flags=re.DOTALL)

    text = text.strip()

    # ------------------------------------------------------------------
    # 2. Markdown JSON blocks
    # ------------------------------------------------------------------
    json_block_matches = re.findall(r'```(?:json)?\s*([\[{].*?[\]}])\s*```', text, re.DOTALL)

    for block in json_block_matches:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed:
                return parsed[0]
        except json.JSONDecodeError:
            continue

    # ------------------------------------------------------------------
    # 2.5 Handle unclosed markdown fence
    # ------------------------------------------------------------------
    open_fence = re.search(r'```(?:json)?\s*([\[{].*)', text, re.DOTALL)
    if open_fence:
        candidate = open_fence.group(1)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed:
                return parsed[0]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 3. Balanced JSON extraction (handles {} and [])
    # ------------------------------------------------------------------
    def extract_balanced_json(s: str):
        stack = []
        start = None

        for i, ch in enumerate(s):
            if ch in "{[":
                if start is None:
                    start = i
                stack.append(ch)

            elif ch in "}]":
                if stack:
                    stack.pop()

                    if not stack and start is not None:
                        return s[start:i+1]

        return None

    candidate = extract_balanced_json(text)

    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed:
                return parsed[0]
        except json.JSONDecodeError:
            pass

    # ------------------------------------------------------------------
    # 4. Heuristic: first { to last }
    # ------------------------------------------------------------------
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx+1]

            if "'" in json_str and '"' not in json_str:
                json_str = json_str.replace("'", '"')
                json_str = json_str.replace("True", "true")
                json_str = json_str.replace("False", "false")
                json_str = json_str.replace("None", "null")

            json_str = re.sub(r',\s*\}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)

            parsed = json.loads(json_str)

            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed:
                return parsed[0]

    except json.JSONDecodeError:
        pass

    # ------------------------------------------------------------------
    # 4.5 Handle Python dict outputs
    # ------------------------------------------------------------------
    python_dict_match = re.search(r'\{.*\}', text, re.DOTALL)

    if python_dict_match:
        candidate = python_dict_match.group(0)

        try:
            fixed = candidate.replace("'", '"')
            fixed = fixed.replace("True", "true")
            fixed = fixed.replace("False", "false")
            fixed = fixed.replace("None", "null")

            fixed = re.sub(r',\s*\}', '}', fixed)
            fixed = re.sub(r',\s*\]', ']', fixed)

            return json.loads(fixed)

        except Exception:
            pass

    # ------------------------------------------------------------------
    # 5. Handle escaped JSON strings
    # Example: "{\"strategy\": \"finish\"}"
    # ------------------------------------------------------------------
    if text.startswith('"') and text.endswith('"'):
        try:
            unescaped = json.loads(text)
            return json.loads(unescaped)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 6. Manual heuristic extraction
    # ------------------------------------------------------------------
    extracted = {}

    thought_match = re.search(
        r'"?thought_process"?\s*(?::|is)\s*"?(.*?)"?(?:\s*and|,|\.|$)',
        text,
        re.DOTALL | re.IGNORECASE
    )

    if thought_match:
        extracted["thought_process"] = thought_match.group(1).strip()

    strategy_match = re.search(
        r'"?strategy"?\s*(?::|is)\s*"?(decompose|tool_use|finish)"?',
        text,
        re.IGNORECASE
    )

    if strategy_match:
        extracted["strategy"] = strategy_match.group(1).lower()

    if "thought_process" in extracted and "strategy" in extracted:
        return extracted

    # ------------------------------------------------------------------
    # 7. Last chance: try full parse safely
    # ------------------------------------------------------------------
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

        if isinstance(parsed, list) and parsed:
            return parsed[0]

    except Exception:
        logger.error("Failed to parse JSON. Raw output:\n%s", text)

    return {}